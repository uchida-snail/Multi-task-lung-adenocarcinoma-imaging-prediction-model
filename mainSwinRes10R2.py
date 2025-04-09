#!/usr/bin/env python
"""
main_two_phase_resnet_swin.py

两阶段训练策略：
  Phase 1：只训练 head_b (HGP)，冻结 LN 分支（head_a），监控 HGP 的 F1 实现 early stop
  Phase 2：加载 Phase1 最优权重，冻结 HGP 分支（head_b），只训练 LN 分支（head_a），监控 LN 的 F1 实现 early stop

不使用多任务不确定性加权，仅使用单任务损失，同时保留 FocalLoss 的可调参数以及 Swin/ResNet 部分冻结策略。
"""

import os
os.environ["MONAI_USE_META_TENSORS"] = "0"

import gc
import argparse
import numpy as np
import logging
import warnings
import csv
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch_optimizer import Ranger  # 如不使用可注释

# ------------------ 设置随机种子 ------------------
def set_random_seed(seed=1337):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed(1337)

# =============================== 权重初始化函数 ==============================
def init_weights(m):
    """
    对模型的每个层进行初始化:
     - 卷积层采用 He Initialization
     - 全连接层采用 Xavier Initialization
     - 偏置初始化为0
    """
    if isinstance(m, (nn.Conv3d, nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.zeros_(m.bias)

# 仅对非预训练部分进行初始化（避免覆盖 swin/resnet 预训练权重）
def selective_init_weights(model):
    """
    初始化模型中除预训练部分（swin_branch.encoder 和 resnet_branch.resnet2d）之外的模块
    """
    for name, module in model.named_modules():
        if name.startswith("swin_branch.encoder") or name.startswith("resnet_branch.resnet2d"):
            continue
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.Linear)):
            init_weights(module)

###################################################################
# 1) 定义 FocalLoss，可调 alpha, gamma
###################################################################
class FocalLoss(nn.Module):
    """
    Focal Loss 用于二分类：
      FL = -alpha * (1-p)^gamma * log(p)
    """
    def __init__(self, gamma=2.0, alpha=0.4, smoothing=0.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # 若为 None，则不使用 alpha 调整
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
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
            logging.warning(f"[collate_fn] Sample index={i} missing keys, skipping.")
            continue
        filtered.append(sample)
    if len(filtered) == 0:
        logging.warning("[collate_fn] All samples invalid, returning None.")
        return None
    ref_shape = filtered[0]["image_cropped"].shape
    for s in filtered:
        if s["image_cropped"].shape != ref_shape:
            logging.error("[collate_fn] Shape mismatch, skip entire batch.")
            return None
    collated = { key: torch.stack([s[key] for s in filtered], dim=0) for key in filtered[0].keys() }
    return collated

###################################################################
# 3) 分阶段冻结相关函数
###################################################################
def partial_freeze_swin(swin_branch, freeze_stages=2):
    if hasattr(swin_branch, 'encoder') and hasattr(swin_branch.encoder, 'freeze_encoder'):
        swin_branch.encoder.freeze_encoder(freeze_stages=freeze_stages)
    else:
        raise AttributeError("swin_branch does not have an encoder.freeze_encoder method")

def freeze_resnet_layers(model, freeze_bn=True, freeze_conv_stages=2):
    resnet2d = model.resnet_branch.resnet2d
    if freeze_conv_stages > 0:
        for param in resnet2d.conv1.parameters():
            param.requires_grad = False
        for param in resnet2d.bn1.parameters():
            param.requires_grad = False
    if freeze_conv_stages > 1:
        for param in resnet2d.layer1.parameters():
            param.requires_grad = False
    if freeze_conv_stages > 2:
        for param in resnet2d.layer2.parameters():
            param.requires_grad = False
    if freeze_bn:
        for m in resnet2d.modules():
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
# 5) 导入数据集与 FusionModel
###################################################################
from datasets import LNClassificationDataset
from model import FusionModel  # 请确保 FusionModel 支持 swin_branch 和 resnet_branch 的融合

from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score, roc_curve, auc
from tqdm import tqdm
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.basicConfig(level=logging.WARNING)
torch.multiprocessing.set_sharing_strategy('file_system')

# ------------------ 评估指标 ------------------
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
# ============ Phase 1：训练 HeadB (HGP) ==================
###################################################################
def train_phaseB(args, model, train_loader, val_loader, device, hgp_criterion):
    """
    仅训练 head_b (HGP)，冻结 head_a，监控 HGP 的 F1 实现 early stop
    """
    # 冻结 LN 分支
    for param in model.head_a.parameters():
        param.requires_grad = False

    effective_lr = args.lr * (args.batch_size / args.reference_batch_size)
    # 只优化需要训练的参数
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=effective_lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, min_lr=1e-4)
    scaler = GradScaler(enabled=args.use_amp)

    best_f1B = 0.0
    trigger_times = 0
    patience = args.patience
    save_path = os.path.join(args.model_dir, "best_model_phaseB.pth")
    history_b = {'train_loss': [], 'val_loss': [], 'accB': [], 'f1B': [], 'recallB': [], 'rocB': []}

    def get_warmup_lr(epoch, base_lr, warmup_epochs):
        return base_lr * float(epoch + 1) / warmup_epochs

    for epoch in range(args.epochs):
        # warmup 阶段
        if epoch < args.warmup_epochs:
            warmup_lr = get_warmup_lr(epoch, effective_lr, args.warmup_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr
            print(f"[PHASE B] Warmup Epoch {epoch+1}/{args.warmup_epochs}: lr={warmup_lr:.2e}")

        model.train()
        train_losses = []
        loop = tqdm(train_loader, desc=f"[Train HeadB] Epoch {epoch+1}/{args.epochs}")
        for batch in loop:
            if batch is None:
                continue
            image_cropped = batch["image_cropped"].to(device)
            head_b = batch["head_b"].to(device)
            with autocast(enabled=args.use_amp):
                # 前向传播，关注 head_b
                _, out_b, _ = model(image_cropped)
                smooth_b = smooth_labels(head_b.view(-1), smoothing=0.1)
                loss_b = hgp_criterion(out_b.view(-1), smooth_b)
            scaler.scale(loss_b).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            train_losses.append(loss_b.item())
        
        mean_train_loss = np.mean(train_losses) if train_losses else float('inf')

        # 验证阶段
        model.eval()
        val_losses = []
        all_probs_b, all_true_b = [], []
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                image_cropped = batch["image_cropped"].to(device)
                head_b = batch["head_b"].to(device)
                with autocast(enabled=args.use_amp):
                    _, out_b, _ = model(image_cropped)
                    vb = hgp_criterion(out_b.view(-1), head_b.view(-1))
                val_losses.append(vb.item())
                probs = torch.sigmoid(out_b).view(-1).cpu().numpy()
                labels = head_b.view(-1).cpu().numpy()
                all_probs_b.extend(probs)
                all_true_b.extend(labels)
        val_loss = np.mean(val_losses) if val_losses else float('inf')

        # 计算 HGP (head_b) 的 F1
        best_thresh_b = 0.0
        best_f1_b = -1
        for t in np.arange(0.0, 1.01, 0.02):
            preds = (np.array(all_probs_b) >= t).astype(int)
            f1, _ = f1_recall_score_metric(np.array(all_true_b), preds)
            if f1 > best_f1_b:
                best_f1_b = f1
                best_thresh_b = t
        final_preds = (np.array(all_probs_b) >= best_thresh_b).astype(int)
        accB = accuracy_score_metric(np.array(all_true_b), final_preds)
        f1B, recallB = f1_recall_score_metric(np.array(all_true_b), final_preds)
        try:
            rocB = roc_auc_score(np.array(all_true_b), np.array(all_probs_b))
        except Exception:
            rocB = float('nan')

        history_b['train_loss'].append(mean_train_loss)
        history_b['val_loss'].append(val_loss)
        history_b['accB'].append(accB)
        history_b['f1B'].append(f1B)
        history_b['recallB'].append(recallB)
        history_b['rocB'].append(rocB)

        print(f"[PHASE B] Epoch {epoch+1}: TrainLoss={mean_train_loss:.4f}, ValLoss={val_loss:.4f}")
        print(f"           HGP: Acc={accB:.3f}, F1={f1B:.3f}, Recall={recallB:.3f}, ROC={rocB:.3f}")

        scheduler.step(val_loss)

        # Early stopping：监控 HGP 的 F1
        if f1B > best_f1B:
            best_f1B = f1B
            save_dict = {"epoch": epoch, "model_state_dict": model.state_dict(),
                         "val_loss": val_loss, "f1B": f1B}
            torch.save(save_dict, save_path)
            print(f"[PHASE B] Saved best model at epoch {epoch+1} with f1B={f1B:.4f}")
            trigger_times = 0
        else:
            trigger_times += 1
            print(f"[PHASE B] No improvement in f1B for {trigger_times} epoch(s).")
            if trigger_times >= patience:
                print("[PHASE B] Early stopping triggered.")
                break

    return history_b, save_path

###################################################################
# ============ Phase 2：训练 HeadA (LN) ==================
###################################################################
def train_phaseA(args, model, phaseB_ckpt, train_loader, val_loader, device, ln_criterion):
    """
    加载 Phase 1 的最佳权重，冻结 head_b，只训练 head_a，监控 LN 的 F1 实现 early stop
    """
    ckpt = torch.load(phaseB_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[PHASE A] Loaded Phase B checkpoint with f1B={ckpt.get('f1B',0):.4f}")

    # 冻结 HGP 分支
    for param in model.head_b.parameters():
        param.requires_grad = False

    effective_lr = args.lr * (args.batch_size / args.reference_batch_size)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=effective_lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, min_lr=1e-4)
    scaler = GradScaler(enabled=args.use_amp)

    best_f1A = 0.0
    trigger_times = 0
    patience = args.patience
    save_path = os.path.join(args.model_dir, "best_model_phaseA.pth")
    history_a = {'train_loss': [], 'val_loss': [], 'accA': [], 'f1A': [], 'recallA': [], 'rocA': []}

    def get_warmup_lr(epoch, base_lr, warmup_epochs):
        return base_lr * float(epoch + 1) / warmup_epochs

    for epoch in range(args.epochs):
        # warmup 阶段
        if epoch < args.warmup_epochs:
            warmup_lr = get_warmup_lr(epoch, effective_lr, args.warmup_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr
            print(f"[PHASE A] Warmup Epoch {epoch+1}/{args.warmup_epochs}: lr={warmup_lr:.2e}")

        model.train()
        train_losses = []
        loop = tqdm(train_loader, desc=f"[Train HeadA] Epoch {epoch+1}/{args.epochs}")
        for batch in loop:
            if batch is None:
                continue
            image_cropped = batch["image_cropped"].to(device)
            head_a = batch["head_a"].to(device)
            with autocast(enabled=args.use_amp):
                out_a, _ , _ = model(image_cropped)  # 只关注 LN 分支输出
                loss_a = ln_criterion(out_a.view(-1), head_a.view(-1))
            scaler.scale(loss_a).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            train_losses.append(loss_a.item())
        mean_train_loss = np.mean(train_losses) if train_losses else float('inf')

        # 验证阶段
        model.eval()
        val_losses = []
        all_probs_a, all_true_a = [], []
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                image_cropped = batch["image_cropped"].to(device)
                head_a = batch["head_a"].to(device)
                with autocast(enabled=args.use_amp):
                    out_a, _ , _ = model(image_cropped)
                    va = ln_criterion(out_a.view(-1), head_a.view(-1))
                val_losses.append(va.item())
                probs = torch.sigmoid(out_a).view(-1).cpu().numpy()
                labels = head_a.view(-1).cpu().numpy()
                all_probs_a.extend(probs)
                all_true_a.extend(labels)
        val_loss = np.mean(val_losses) if val_losses else float('inf')

        # 计算 LN (head_a) 的 F1
        best_thresh_a = 0.0
        best_f1_a = -1
        for t in np.arange(0.0, 1.01, 0.02):
            preds = (np.array(all_probs_a) >= t).astype(int)
            f1, _ = f1_recall_score_metric(np.array(all_true_a), preds)
            if f1 > best_f1_a:
                best_f1_a = f1
                best_thresh_a = t
        final_preds = (np.array(all_probs_a) >= best_thresh_a).astype(int)
        accA = accuracy_score_metric(np.array(all_true_a), final_preds)
        f1A, recallA = f1_recall_score_metric(np.array(all_true_a), final_preds)
        try:
            rocA = roc_auc_score(np.array(all_true_a), np.array(all_probs_a))
        except Exception:
            rocA = float('nan')

        history_a['train_loss'].append(mean_train_loss)
        history_a['val_loss'].append(val_loss)
        history_a['accA'].append(accA)
        history_a['f1A'].append(f1A)
        history_a['recallA'].append(recallA)
        history_a['rocA'].append(rocA)

        print(f"[PHASE A] Epoch {epoch+1}: TrainLoss={mean_train_loss:.4f}, ValLoss={val_loss:.4f}")
        print(f"           LN: Acc={accA:.3f}, F1={f1A:.3f}, Recall={recallA:.3f}, ROC={rocA:.3f}")

        scheduler.step(val_loss)

        # Early stopping：监控 LN 的 F1
        if f1A > best_f1A:
            best_f1A = f1A
            save_dict = {"epoch": epoch, "model_state_dict": model.state_dict(),
                         "val_loss": val_loss, "f1A": f1A}
            torch.save(save_dict, save_path)
            print(f"[PHASE A] Saved best model at epoch {epoch+1} with f1A={f1A:.4f}")
            trigger_times = 0
        else:
            trigger_times += 1
            print(f"[PHASE A] No improvement in f1A for {trigger_times} epoch(s).")
            if trigger_times >= patience:
                print("[PHASE A] Early stopping triggered.")
                break

    return history_a, save_path

###################################################################
# ============== 主函数：先 Phase 1 再 Phase 2 ==================
###################################################################
def main():
    parser = argparse.ArgumentParser(description="Two-Phase Training for FusionModel (Swin+ResNet)")
    # 通用参数
    parser.add_argument('--root_path', type=str, required=True, help="数据集根目录")
    parser.add_argument('--model_dir', type=str, required=True, help="模型保存目录")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--reference_batch_size', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0])
    parser.add_argument('--grad_clip', type=float, default=0.2, help="梯度裁剪阈值")
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--loss_type', type=str, default="focal", choices=["focal", "bce"])
    parser.add_argument('--focal_alpha', type=float, default=0.75)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--warmup_epochs', type=int, default=10, help="warmup epoch 数")
    # 多任务损失相关（两阶段训练中不启用不确定性加权）
    parser.add_argument('--lambda_a', type=float, default=1.0, help="LN 任务 loss 权重")
    parser.add_argument('--lambda_b', type=float, default=1.0, help="HGP 任务 loss 权重")
    # 部分冻结参数
    parser.add_argument('--freeze_swin_stages', type=int, default=0, help="冻结 Swin 前多少个 block")
    parser.add_argument('--freeze_resnet_stages', type=int, default=0, help="冻结 ResNet 前多少个 stage")
    parser.add_argument('--freeze_bn_resnet', action='store_true', help="是否冻结 ResNet 的 BN 层")
    parser.add_argument('--fusion_dropout', type=float, default=0.5, help="融合层 dropout 比例")
    # 不确定性加权默认关闭，两阶段训练不使用
    parser.add_argument('--use_uncertainty', action='store_true', help="是否启用 Kendall 不确定性加权（两阶段训练中不启用）")
    # 新增缺失的参数
    parser.add_argument('--override_pos_weight', type=float, default=0, help="若 > 0 则覆盖 LN 的 pos_weight")
    parser.add_argument('--use_DropKey', action='store_true', help="是否在 FusionModel 中使用 DropKey")
    parser.add_argument('--mask_ratio', type=float, default=0.0, help="FusionModel 的 mask ratio")
    args = parser.parse_args()

    effective_lr = args.lr * (args.batch_size / args.reference_batch_size)
    print(f"[INFO] Effective LR (scaled by batch size): {effective_lr:.2e}")

    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids[0])
        print(f"[INFO] Using GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    os.makedirs(args.model_dir, exist_ok=True)

    # ------------------ 数据集加载 ------------------
    train_dataset = LNClassificationDataset(root_path=args.root_path, split='train', transforms=None, preprocessed=True)
    val_dataset   = LNClassificationDataset(root_path=args.root_path, split='test', transforms=None, preprocessed=True)
    test_dataset  = LNClassificationDataset(root_path=args.root_path, split='internal', transforms=None, preprocessed=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=my_collate_fn,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=my_collate_fn,
                              pin_memory=True, drop_last=False)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=my_collate_fn,
                              pin_memory=True, drop_last=False)

    pos_w_ln = compute_class_pos_weight(train_dataset, head='head_a')
    if args.override_pos_weight > 0:
        pos_w_ln = args.override_pos_weight
        print(f"[INFO] Overriding pos_weight for LN to: {pos_w_ln}")
    pos_weight_ln_t = torch.tensor([pos_w_ln], dtype=torch.float, device=device)

    # ------------------ 构造融合模型 ------------------
    model = FusionModel(
        num_classes_a=1,
        num_classes_b=1,
        swin_pretrain_path="/root/autodl-tmp/model_swinvit.pt",
        resnet_pretrain_path="resnet_10.pth",
        swin_embed_dim=126,
        fusion_dropout=args.fusion_dropout,
        use_DropKey=args.use_DropKey,
        mask_ratio=args.mask_ratio
    ).to(device)

    # 分阶段冻结：冻结 swin/resnet 部分权重
    partial_freeze_swin(model.swin_branch, freeze_stages=args.freeze_swin_stages)
    freeze_resnet_layers(model, freeze_bn=args.freeze_bn_resnet, freeze_conv_stages=args.freeze_resnet_stages)
    register_activation_hooks(model)
    # 仅对预训练之外的模块初始化，避免覆盖预训练权重
    selective_init_weights(model)

    # ------------------ 定义损失函数 ------------------
    if args.loss_type == "focal":
        ln_criterion = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha, reduction='mean').to(device)
        print(f"[INFO] Using FocalLoss (gamma={args.focal_gamma}, alpha={args.focal_alpha}) for LN.")
    else:
        ln_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_ln_t).to(device)
    hgp_criterion = nn.BCEWithLogitsLoss().to(device)
    # 两阶段训练中不启用多任务不确定性加权

    # ========== Phase 1：训练 HGP (head_b) ==========
    print("======== PHASE 1: Train head_b (HGP) =========")
    history_b, ckpt_phaseB = train_phaseB(args, model, train_loader, val_loader, device, hgp_criterion)

    # ========== Phase 2：训练 LN (head_a) ==========
    print("======== PHASE 2: Train head_a (LN) =========")
    history_a, ckpt_phaseA = train_phaseA(args, model, ckpt_phaseB, train_loader, val_loader, device, ln_criterion)

    print("[INFO] Two-phase training finished. Final LN model saved at:", ckpt_phaseA)

if __name__ == "__main__":
    main()
