#!/usr/bin/env python
"""
main_two_phase.py

在原有基础上，改为先训练 HeadB (HGP) 再训练 HeadA (LN) 的两阶段策略：
 - Phase 1：只更新 head_b 参数，监控 F1_B 早停
 - Phase 2：加载 Phase1 最优权重，冻结 head_b，只更新 head_a，监控 F1_A 早停

不确定性加权已去掉，只保留单任务损失。
其他如 FocalLoss、EfficientNet 部分冻结等逻辑依然保留。
"""

import os
os.environ["MONAI_USE_META_TENSORS"] = "0"

import gc
import argparse
import numpy as np
import logging
import warnings
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_optimizer import Ranger

# =============================== 权重初始化函数 ==============================
def init_weights(m):
    """
    对模型的每个层进行初始化:
    - 对卷积层使用 He Initialization
    - 对全连接层使用 Xavier Initialization
    - 对偏置使用常数初始化为0
    """
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
    
    if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)) and m.bias is not None:
        nn.init.zeros_(m.bias)

###################################################################
# 1) 定义 FocalLoss，可调 alpha, gamma
###################################################################
class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    FL = -alpha * (1-p)^gamma * log(p)
    """
    def __init__(self, gamma=2.0, alpha=0.4, smoothing=0.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Label Smoothing：对于二分类，1->1-smoothing, 0->smoothing/2
        if self.smoothing > 0:
            targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing

        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = (1 - pt) ** self.gamma * BCE_loss

        if self.alpha is not None:
            alpha_tensor = torch.tensor([1 - self.alpha, self.alpha], device=inputs.device)
            alpha_factor = targets * alpha_tensor[1] + (1 - targets) * alpha_tensor[0]
            focal_loss = alpha_factor * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

###################################################################
# 2) DataLoader 的 collate_fn
###################################################################
def my_collate_fn(batch):
    filtered = []
    for i, sample in enumerate(batch):
        if sample is None:
            logging.warning(f"[collate_fn] Sample index={i} is None, skipping.")
            continue
        if not ("image_cropped" in sample and "head_a" in sample and "head_b" in sample):
            logging.warning(f"[collate_fn] Sample index={i} missing some keys, skipping.")
            continue
        filtered.append(sample)
    if len(filtered) == 0:
        logging.warning("[collate_fn] All samples in this batch are invalid. Returning None.")
        return None
    ref_shape = filtered[0]["image_cropped"].shape
    for i, s in enumerate(filtered):
        if s["image_cropped"].shape != ref_shape:
            logging.error(f"[collate_fn] Shape mismatch: {s['image_cropped'].shape} vs {ref_shape}, skip entire batch.")
            return None
    collated = {}
    for key in filtered[0].keys():
        collated[key] = torch.stack([s[key] for s in filtered], dim=0)
    return collated

###################################################################
# 3) 分阶段冻结函数：针对 EfficientNet 分支 (可选)
###################################################################
def freeze_efficientnet_layers(model, freeze_bn=True, freeze_conv_stages=2):
    """
    针对 FusionModel 中的 EfficientNet3D 分支进行部分冻结:
      - freeze_conv_stages 表示冻结 EfficientNet.features 中前几个 block
      - freeze_bn=True 时冻结所有 BN 层
    """
    effnet = model.efficientnet_branch.efficientnet
    for i in range(freeze_conv_stages):
        if i < len(effnet.features):
            for param in effnet.features[i].parameters():
                param.requires_grad = False

    if freeze_bn:
        for m in effnet.features.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

###################################################################
# 4) 注册激活值钩子，便于调试
###################################################################
activation_stats = {}
def register_activation_hooks(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            module.register_forward_hook(
                lambda m, i, o, name=name: activation_stats.update({
                    name: (o.mean().item(), o.std().item(), o.max().item(), o.min().item())
                })
            )

###################################################################
# 5) 导入数据集和模型
###################################################################
from datasets import LNClassificationDataset
from modelEfficientNet import FusionModel

from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, roc_curve, auc
from tqdm import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(level=logging.WARNING)
torch.multiprocessing.set_sharing_strategy('file_system')

def accuracy_score_metric(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def f1_recall_score_metric(y_true, y_pred, average='binary'):
    return f1_score(y_true, y_pred, average=average), recall_score(y_true, y_pred, average=average)

def compute_class_pos_weight(dataset, head='head_a'):
    from collections import Counter
    counter = Counter()
    for i in range(len(dataset)):
        sample = dataset[i]
        if sample is None:
            continue
        label = sample.get(head, -1)
        if isinstance(label, torch.Tensor):
            label = label.item()
        if label != -1:
            counter[int(label)] += 1
    neg_count = counter.get(0, 1)
    pos_count = counter.get(1, 1)
    pos_weight = neg_count / pos_count
    print(f"[INFO] Computed pos_weight for {head}: {pos_weight}")
    return pos_weight

def smooth_labels(targets, smoothing=0.0):
    if smoothing > 0:
        return targets * (1 - smoothing) + (1 - targets) * (smoothing / 2)
    return targets    

###################################################################
# ============ Phase 1：训练 Head B (HGP) ============ 
###################################################################
def train_phaseB(args, model, train_loader, val_loader, device, hgp_criterion):
    """
    只训练 head_b，用 HGP (head_b) 的 F1 来 early stop。
    """
    # 1) 冻结 LN 分支
    for param in model.head_a.parameters():
        param.requires_grad = False

    # 2) 依然可选择是否冻结/微调 backbone
    #   (看需求，如果想让backbone也被HGP更新，就不用freeze，否则可以freeze)
    #   此处示例：不冻结backbone，让其也学习HGP特征
    #   具体可以自行调整

    # 3) 构建优化器
    all_params = filter(lambda p: p.requires_grad, model.parameters())
    effective_lr = args.lr * (args.batch_size / args.reference_batch_size)
    optimizer = optim.AdamW(all_params, lr=effective_lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, min_lr=1e-5)

    # 混合精度
    scaler = GradScaler(enabled=args.use_amp)

    # Early stop 相关
    best_f1B = 0.0
    trigger_times = 0
    patience = args.patience
    save_path = os.path.join(args.model_dir, "best_model_phaseB.pth")

    history_b = {'train_loss': [], 'val_loss': [],
                 'f1B': [], 'accB': [], 'recallB': [], 'rocB': []}

    def get_warmup_lr(epoch, base_lr, warmup_epochs):
        return base_lr * float(epoch + 1) / warmup_epochs

    for epoch in range(args.epochs):
        # Warmup
        if epoch < args.warmup_epochs:
            warmup_lr = get_warmup_lr(epoch, effective_lr, args.warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            print(f"[PHASE B] Warmup Epoch {epoch+1}/{args.warmup_epochs}: lr={warmup_lr:.2e}")

        model.train()
        train_losses = []
        loop = tqdm(train_loader, desc=f"[Train HeadB] Epoch {epoch+1}/{args.epochs}")
        for batch_data in loop:
            if batch_data is None:
                continue
            image_cropped = batch_data["image_cropped"].to(device)
            head_b = batch_data["head_b"].to(device)

            with autocast(enabled=args.use_amp):
                out_a, out_b, _ = model(image_cropped)
                # 只计算 B 的 loss
                # (可选择 label smoothing)
                smooth_b = smooth_labels(head_b.view(-1), smoothing=0.1)
                loss_b = hgp_criterion(out_b.view(-1), smooth_b)
            
            scaler.scale(loss_b).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            train_losses.append(loss_b.item())

        mean_train_loss = float(np.mean(train_losses)) if train_losses else float('inf')

        # ---------- Validation ----------
        model.eval()
        val_losses = []
        all_probs_b, all_true_b = [], []
        with torch.no_grad():
            for batch_data in val_loader:
                if batch_data is None:
                    continue
                image_cropped = batch_data["image_cropped"].to(device)
                head_b = batch_data["head_b"].to(device)
                with autocast(enabled=args.use_amp):
                    _, out_b, _ = model(image_cropped)
                    vb = hgp_criterion(out_b.view(-1), head_b.view(-1))
                val_losses.append(vb.item())

                probs_b = torch.sigmoid(out_b).cpu().numpy().flatten()
                labels_b = head_b.cpu().numpy().flatten()
                all_probs_b.append(probs_b)
                all_true_b.append(labels_b)

        val_loss = float(np.mean(val_losses)) if val_losses else float('inf')
        all_probs_b = np.concatenate(all_probs_b, axis=0).flatten()
        all_true_b  = np.concatenate(all_true_b, axis=0).flatten()

        # 计算 F1_B
        best_thresh_b = 0.0
        best_f1_b = -1
        for t in np.arange(0.0, 1.01, 0.02):
            preds_b = (all_probs_b >= t).astype(int)
            f1_b, _rec_b = f1_recall_score_metric(all_true_b, preds_b)
            if f1_b > best_f1_b:
                best_f1_b = f1_b
                best_thresh_b = t
        final_preds_b = (all_probs_b >= best_thresh_b).astype(int)
        accB = accuracy_score_metric(all_true_b, final_preds_b)
        f1B, recallB = f1_recall_score_metric(all_true_b, final_preds_b)
        try:
            rocB = roc_auc_score(all_true_b, all_probs_b)
        except:
            rocB = float('nan')

        history_b['train_loss'].append(mean_train_loss)
        history_b['val_loss'].append(val_loss)
        history_b['f1B'].append(f1B)
        history_b['accB'].append(accB)
        history_b['recallB'].append(recallB)
        history_b['rocB'].append(rocB)

        print(f"[PHASE B] Ep={epoch+1}, TrainLoss={mean_train_loss:.4f}, ValLoss={val_loss:.4f}")
        print(f"          HGP: Acc={accB:.3f}, F1={f1B:.3f}, Rec={recallB:.3f}, ROC={rocB:.3f}")

        scheduler.step(val_loss)

        # Early Stop 监控 F1B
        if f1B > best_f1B:
            best_f1B = f1B
            save_dict = {
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "val_loss": val_loss, "f1B": f1B
            }
            torch.save(save_dict, save_path)
            print(f"[PHASE B] Saved best model at epoch {epoch+1}, f1B={f1B:.4f}")
            trigger_times = 0
        else:
            trigger_times += 1
            print(f"[PHASE B] No improvement in f1B for {trigger_times} epoch(s).")
            if trigger_times >= patience:
                print("[PHASE B] Early stopping triggered.")
                break

    return history_b, save_path


###################################################################
# ============ Phase 2：训练 Head A (LN) ============ 
###################################################################
def train_phaseA(args, model, phaseB_ckpt, train_loader, val_loader, device, ln_criterion):
    """
    加载 Phase B 的最优权重，冻结 head_b，只训练 head_a。
    监控 LN 的 F1 (f1A) 早停。
    """
    # 1) 加载最佳的 B
    ckpt = torch.load(phaseB_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[PHASE A] Loaded phaseB checkpoint, f1B={ckpt.get('f1B',0):.4f}")

    # 2) 冻结 B 分支
    for param in model.head_b.parameters():
        param.requires_grad = False

    # 也可考虑是否部分冻结 backbone，因为已在 Phase B 学到一些特征
    # 如果想让 LN 任务也微调backbone，就不freeze backbone
    # 如果想完全保留 B 期学到的特征，则可以把 backbone也冻结或者给极小 LR

    # 构建优化器 (只包含可训练参数)
    all_params = filter(lambda p: p.requires_grad, model.parameters())
    effective_lr = args.lr * (args.batch_size / args.reference_batch_size)
    optimizer = optim.AdamW(all_params, lr=effective_lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, min_lr=1e-5)
    scaler = GradScaler(enabled=args.use_amp)

    best_f1A = 0.0
    trigger_times = 0
    patience = args.patience
    save_path = os.path.join(args.model_dir, "best_model_phaseA.pth")

    history_a = {'train_loss': [], 'val_loss': [],
                 'f1A': [], 'accA': [], 'recallA': [], 'rocA': []}

    def get_warmup_lr(epoch, base_lr, warmup_epochs):
        return base_lr * float(epoch + 1) / warmup_epochs

    for epoch in range(args.epochs):
        # Warmup
        if epoch < args.warmup_epochs:
            warmup_lr = get_warmup_lr(epoch, effective_lr, args.warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            print(f"[PHASE A] Warmup Epoch {epoch+1}/{args.warmup_epochs}: lr={warmup_lr:.2e}")

        model.train()
        train_losses = []
        loop = tqdm(train_loader, desc=f"[Train HeadA] Epoch {epoch+1}/{args.epochs}")
        for batch_data in loop:
            if batch_data is None:
                continue
            image_cropped = batch_data["image_cropped"].to(device)
            head_a = batch_data["head_a"].to(device)

            with autocast(enabled=args.use_amp):
                out_a, out_b, _ = model(image_cropped)
                # 只计算 A 的 loss
                loss_a = ln_criterion(out_a.view(-1), head_a.view(-1))
            
            scaler.scale(loss_a).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            train_losses.append(loss_a.item())

        mean_train_loss = float(np.mean(train_losses)) if train_losses else float('inf')

        # ---------- Validation ----------
        model.eval()
        val_losses = []
        all_probs_a, all_true_a = [], []
        with torch.no_grad():
            for batch_data in val_loader:
                if batch_data is None:
                    continue
                image_cropped = batch_data["image_cropped"].to(device)
                head_a = batch_data["head_a"].to(device)
                with autocast(enabled=args.use_amp):
                    out_a, out_b, _ = model(image_cropped)
                    va = ln_criterion(out_a.view(-1), head_a.view(-1))
                val_losses.append(va.item())

                probs_a = torch.sigmoid(out_a).cpu().numpy().flatten()
                labels_a = head_a.cpu().numpy().flatten()
                all_probs_a.append(probs_a)
                all_true_a.append(labels_a)

        val_loss = float(np.mean(val_losses)) if val_losses else float('inf')
        all_probs_a = np.concatenate(all_probs_a, axis=0).flatten()
        all_true_a  = np.concatenate(all_true_a, axis=0).flatten()

        # 计算 F1_A
        best_thresh_a = 0.0
        best_f1_a = -1
        for t in np.arange(0.0, 1.01, 0.02):
            preds_a = (all_probs_a >= t).astype(int)
            f1_a, _rec_a = f1_recall_score_metric(all_true_a, preds_a)
            if f1_a > best_f1_a:
                best_f1_a = f1_a
                best_thresh_a = t
        final_preds_a = (all_probs_a >= best_thresh_a).astype(int)
        accA = accuracy_score_metric(all_true_a, final_preds_a)
        f1A, recallA = f1_recall_score_metric(all_true_a, final_preds_a)
        try:
            rocA = roc_auc_score(all_true_a, all_probs_a)
        except:
            rocA = float('nan')

        history_a['train_loss'].append(mean_train_loss)
        history_a['val_loss'].append(val_loss)
        history_a['f1A'].append(f1A)
        history_a['accA'].append(accA)
        history_a['recallA'].append(recallA)
        history_a['rocA'].append(rocA)

        print(f"[PHASE A] Ep={epoch+1}, TrainLoss={mean_train_loss:.4f}, ValLoss={val_loss:.4f}")
        print(f"          LN: Acc={accA:.3f}, F1={f1A:.3f}, Rec={recallA:.3f}, ROC={rocA:.3f}")

        scheduler.step(val_loss)

        # Early Stop 监控 F1A
        if f1A > best_f1A:
            best_f1A = f1A
            save_dict = {
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "val_loss": val_loss, "f1A": f1A
            }
            torch.save(save_dict, save_path)
            print(f"[PHASE A] Saved best model at epoch {epoch+1}, f1A={f1A:.4f}")
            trigger_times = 0
        else:
            trigger_times += 1
            print(f"[PHASE A] No improvement in f1A for {trigger_times} epoch(s).")
            if trigger_times >= patience:
                print("[PHASE A] Early stopping triggered.")
                break

    return history_a, save_path


###################################################################
# ============== 主函数：先 Phase B 再 Phase A ==================
###################################################################
def main():
    parser = argparse.ArgumentParser(description="Two-Phase Training: first HeadB(HGP), then HeadA(LN).")
    # 通用参数
    parser.add_argument('--root_path', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--reference_batch_size', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0])
    parser.add_argument('--grad_clip', type=float, default=0.2, help="Clip gradients to avoid exploding gradients")

    parser.add_argument('--loss_type', type=str, default="focal", choices=["focal", "bce"])
    parser.add_argument('--focal_alpha', type=float, default=0.75)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--warmup_epochs', type=int, default=5)

    # 冻结 EfficientNet
    parser.add_argument('--freeze_eff_stages', type=int, default=0)
    parser.add_argument('--freeze_bn_eff', action='store_true')
    parser.add_argument('--efficientnet_pretrain_path', type=str, default=None)

    args = parser.parse_args()

    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids[0])
        print(f"[INFO] Using GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.model_dir, exist_ok=True)

    # ------------------ 数据集加载 ------------------
    train_dataset = LNClassificationDataset(root_path=args.root_path, split='train', transforms=None, preprocessed=True)
    val_dataset   = LNClassificationDataset(root_path=args.root_path, split='test', transforms=None, preprocessed=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=my_collate_fn,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=my_collate_fn,
                              pin_memory=True, drop_last=False)

    # 不需要pos_weight的情况下，可直接用 BCEWithLogitsLoss()；或者使用FocalLoss
    if args.loss_type == "focal":
        hgp_criterion = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha).to(device)
        ln_criterion  = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha).to(device)
        print(f"[INFO] Using FocalLoss for both LN & HGP")
    else:
        hgp_criterion = nn.BCEWithLogitsLoss().to(device)
        ln_criterion  = nn.BCEWithLogitsLoss().to(device)

    # ------------------ 构造模型 + 部分冻结 ------------------
    model = FusionModel(
        num_classes_a=1,
        num_classes_b=1,
        efficientnet_pretrain_path=args.efficientnet_pretrain_path
    ).to(device)

    model.apply(init_weights)
    if args.freeze_eff_stages>0 or args.freeze_bn_eff:
        freeze_efficientnet_layers(model, freeze_bn=args.freeze_bn_eff, freeze_conv_stages=args.freeze_eff_stages)

    # 注册 ReLU 激活钩子(可选)
    register_activation_hooks(model)

    # ========== 先训练 HGP (HeadB) ==========
    print("======== PHASE B: Train head_b (HGP) ========")
    history_b, ckpt_phaseB = train_phaseB(args, model, train_loader, val_loader, device, hgp_criterion)

    # ========== 再训练 LN (HeadA) ==========
    print("======== PHASE A: Train head_a (LN) ========")
    history_a, ckpt_phaseA = train_phaseA(args, model, ckpt_phaseB, train_loader, val_loader, device, ln_criterion)

    print("[INFO] Two-phase training finished. Final LN model is at:", ckpt_phaseA)

if __name__ == "__main__":
    main()
