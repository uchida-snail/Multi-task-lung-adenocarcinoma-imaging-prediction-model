#!/usr/bin/env python
"""
main.py

双任务 (LN, HGP) 训练示例（两阶段训练策略）。
要求 model.py 中 FusionModel_ViT3D_Swin.forward 返回 (out_a, out_b, fused) 三个值。

Phase 1：冻结 head_a，仅训练 head_b；
Phase 2：加载 Phase 1 最优权重后冻结 head_b，仅训练 head_a。
"""

import os
os.environ["MONAI_USE_META_TENSORS"] = "0"

import gc
import argparse
import numpy as np
import logging
import warnings
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch_optimizer import Ranger

# ------------------------ 设置随机种子 ------------------------
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ------------------------ 权重初始化 ------------------------
def init_weights(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
    if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.Linear)) and m.bias is not None:
        nn.init.zeros_(m.bias)

# ------------------------ FocalLoss 定义 ------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.4, smoothing=0.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
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

# ------------------------ 多任务不确定性加权（可选） ------------------------
class MultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_var_a = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.log_var_b = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, loss_a, loss_b):
        precision_a = torch.exp(-self.log_var_a)
        precision_b = torch.exp(-self.log_var_b)
        loss = precision_a * loss_a + self.log_var_a + precision_b * loss_b + self.log_var_b
        return loss

# ------------------------ DataLoader 的 collate_fn ------------------------
def my_collate_fn(batch):
    filtered = []
    for i, sample in enumerate(batch):
        if sample is None:
            continue
        if not ("image_cropped" in sample and "head_a" in sample and "head_b" in sample):
            continue
        filtered.append(sample)
    if len(filtered) == 0:
        return None
    ref_shape = filtered[0]["image_cropped"].shape
    for s in filtered:
        if s["image_cropped"].shape != ref_shape:
            return None
    collated = { key: torch.stack([s[key] for s in filtered], dim=0) for key in filtered[0].keys() }
    return collated

# ------------------------ 激活值钩子，用于调试 ------------------------
activation_stats = {}
def register_activation_hooks(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            def hook_fn(m, i, o, layer_name=name):
                activation_stats[layer_name] = (o.mean().item(), o.std().item(), o.max().item(), o.min().item())
            module.register_forward_hook(hook_fn)

# ------------------------ 导入数据集与模型 ------------------------
from datasets import LNClassificationDataset
from model import FusionModel_ViT3D_Swin  # 要求 forward 返回 (out_a, out_b, fused)

from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from tqdm import tqdm

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

#####################################
# Phase 1：训练 head_b (HGP)
#####################################
def train_phaseB(args, model, train_loader, val_loader, device, hgp_criterion):
    # 冻结 LN 分支 (head_a)
    for param in model.head_a.parameters():
        param.requires_grad = False

    effective_lr = args.lr * (args.batch_size / args.reference_batch_size)
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
        if epoch < args.warmup_epochs:
            warmup_lr = get_warmup_lr(epoch, effective_lr, args.warmup_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr
            print(f"[Phase B] Warmup Epoch {epoch+1}/{args.warmup_epochs}: lr={warmup_lr:.2e}")

        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"[Phase B Train] Epoch {epoch+1}/{args.epochs}"):
            if batch is None:
                continue
            images = batch["image_cropped"].to(device)
            head_b = batch["head_b"].to(device)
            with autocast(enabled=args.use_amp):
                _, out_b, _ = model(images)
                loss_b = hgp_criterion(out_b.view(-1), head_b.view(-1))
            scaler.scale(loss_b).backward()
            # ---------------- 添加梯度裁剪 ----------------
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            train_losses.append(loss_b.item())
        mean_train_loss = np.mean(train_losses) if train_losses else float('inf')

        model.eval()
        val_losses = []
        all_probs_b, all_true_b = [], []
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                images = batch["image_cropped"].to(device)
                head_b = batch["head_b"].to(device)
                with autocast(enabled=args.use_amp):
                    _, out_b, _ = model(images)
                    loss_val = hgp_criterion(out_b.view(-1), head_b.view(-1))
                val_losses.append(loss_val.item())
                probs = torch.sigmoid(out_b).view(-1).cpu().numpy()
                labels = head_b.view(-1).cpu().numpy()
                all_probs_b.extend(probs)
                all_true_b.extend(labels)
        val_loss = np.mean(val_losses) if val_losses else float('inf')

        # 计算 HGP 的 F1
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

        print(f"[Phase B] Epoch {epoch+1}: TrainLoss={mean_train_loss:.4f}, ValLoss={val_loss:.4f}")
        print(f"           HGP: Acc={accB:.3f}, F1={f1B:.3f}, Recall={recallB:.3f}, ROC={rocB:.3f}")

        scheduler.step(val_loss)

        if f1B > best_f1B:
            best_f1B = f1B
            save_dict = {"epoch": epoch, "model_state_dict": model.state_dict(),
                         "val_loss": val_loss, "f1B": f1B}
            torch.save(save_dict, save_path)
            print(f"[Phase B] Saved best model at epoch {epoch+1} with f1B={f1B:.4f}")
            trigger_times = 0
        else:
            trigger_times += 1
            print(f"[Phase B] No improvement in f1B for {trigger_times} epoch(s).")
            if trigger_times >= patience:
                print("[Phase B] Early stopping triggered.")
                break

    return history_b, save_path

#####################################
# Phase 2：训练 head_a (LN)
#####################################
def train_phaseA(args, model, phaseB_ckpt, train_loader, val_loader, device, ln_criterion):
    # 加载 Phase 1 最优权重
    ckpt = torch.load(phaseB_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[Phase A] Loaded Phase B checkpoint with f1B={ckpt.get('f1B',0):.4f}")

    # 冻结 HGP 分支 (head_b)
    for param in model.head_b.parameters():
        param.requires_grad = False

    effective_lr = args.lr * (args.batch_size / args.reference_batch_size)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=effective_lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, min_lr=1e-4)
    scaler = GradScaler(enabled=args.use_amp)

    best_f1_a = 0.0
    trigger_times = 0
    patience = args.patience
    save_path = os.path.join(args.model_dir, "best_model_phaseA.pth")
    history_a = {'train_loss': [], 'val_loss': [], 'accA': [], 'f1A': [], 'recallA': [], 'rocA': []}

    def get_warmup_lr(epoch, base_lr, warmup_epochs):
        return base_lr * float(epoch + 1) / warmup_epochs

    for epoch in range(args.epochs):
        if epoch < args.warmup_epochs:
            warmup_lr = get_warmup_lr(epoch, effective_lr, args.warmup_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr
            print(f"[Phase A] Warmup Epoch {epoch+1}/{args.warmup_epochs}: lr={warmup_lr:.2e}")

        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"[Phase A Train] Epoch {epoch+1}/{args.epochs}"):
            if batch is None:
                continue
            images = batch["image_cropped"].to(device)
            head_a = batch["head_a"].to(device)
            with autocast(enabled=args.use_amp):
                out_a, _, _ = model(images)
                loss_a = ln_criterion(out_a.view(-1), head_a.view(-1))
            scaler.scale(loss_a).backward()
            # ---------------- 添加梯度裁剪 ----------------
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            train_losses.append(loss_a.item())
        mean_train_loss = np.mean(train_losses) if train_losses else float('inf')

        model.eval()
        val_losses = []
        all_probs_a, all_true_a = [], []
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                images = batch["image_cropped"].to(device)
                head_a = batch["head_a"].to(device)
                with autocast(enabled=args.use_amp):
                    out_a, _, _ = model(images)
                    loss_val = ln_criterion(out_a.view(-1), head_a.view(-1))
                val_losses.append(loss_val.item())
                probs = torch.sigmoid(out_a).view(-1).cpu().numpy()
                labels = head_a.view(-1).cpu().numpy()
                all_probs_a.extend(probs)
                all_true_a.extend(labels)
        val_loss = np.mean(val_losses) if val_losses else float('inf')

        best_thresh_a = 0.0
        best_f1_a_epoch = -1
        for t in np.arange(0.0, 1.01, 0.02):
            preds = (np.array(all_probs_a) >= t).astype(int)
            f1, _ = f1_recall_score_metric(np.array(all_true_a), preds)
            if f1 > best_f1_a_epoch:
                best_f1_a_epoch = f1
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

        print(f"[Phase A] Epoch {epoch+1}: TrainLoss={mean_train_loss:.4f}, ValLoss={val_loss:.4f}")
        print(f"           LN: Acc={accA:.3f}, F1={f1A:.3f}, Recall={recallA:.3f}, ROC={rocA:.3f}")

        scheduler.step(val_loss)

        if f1A > best_f1_a:
            best_f1_a = f1A
            save_dict = {"epoch": epoch, "model_state_dict": model.state_dict(),
                         "val_loss": val_loss, "f1A": f1A, "f1B": ckpt.get("f1B", 0)}
            torch.save(save_dict, save_path)
            print(f"[Phase A] Saved best model at epoch {epoch+1} with LN F1={f1A:.4f}")
            trigger_times = 0
        else:
            trigger_times += 1
            print(f"[Phase A] No improvement in LN F1 for {trigger_times} epoch(s).")
            if trigger_times >= patience:
                print("[Phase A] Early stopping triggered.")
                break

    return history_a, save_path

# ------------------------ 主函数 ------------------------
def main():
    # 设置随机种子为 1337
    set_random_seed(1337)
    
    parser = argparse.ArgumentParser(description="Multi-task Training (LN & HGP) with FusionModel_ViT3D_Swin - Two Phase Training")
    parser.add_argument('--use_DropKey', action='store_true')
    parser.add_argument('--mask_ratio', type=float, default=0.2)
    parser.add_argument('--root_path', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--reference_batch_size', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--optimizer', type=str, default="adamw")
    parser.add_argument('--override_pos_weight', type=float, default=-1)
    parser.add_argument('--grad_clip', type=float, default=0.2)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0])
    parser.add_argument('--loss_type', type=str, default="focal", choices=["focal", "bce"])
    parser.add_argument('--focal_alpha', type=float, default=0.75)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--use_uncertainty', action='store_true')
    parser.add_argument('--lambda_a', type=float, default=1.0)
    parser.add_argument('--lambda_b', type=float, default=1.0)
    parser.add_argument('--freeze_swin_stages', type=int, default=0)
#    parser.add_argument('--fusion_dropout', type=float, default=0.5)
    args = parser.parse_args()

    effective_lr = args.lr * (args.batch_size / args.reference_batch_size)
    print(f"[INFO] Effective LR (scaled by batch size): {effective_lr:.2e}")

    if args.gpu_ids is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_ids[0])
        print(f"[INFO] Using GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    torch.autograd.set_detect_anomaly(True)
    os.makedirs(args.model_dir, exist_ok=True)

    # ------------------------ 数据集加载 ------------------------
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

    # ------------------------ 构造模型 ------------------------
    model = FusionModel_ViT3D_Swin(
        in_channels=2,
        num_classes=1,
        dim_swin=256,
        dim_vit=96,
        fused_dim=512,
        use_DropKey=args.use_DropKey,
        mask_ratio=args.mask_ratio,
  #      fusion_dropout=args.fusion_dropout
    ).to(device)

    register_activation_hooks(model)
    model.apply(init_weights)

    if args.loss_type == "focal":
        ln_criterion = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha, reduction='mean').to(device)
        print(f"[INFO] Using FocalLoss (gamma={args.focal_gamma}, alpha={args.focal_alpha}) for LN.")
    else:
        ln_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_ln_t).to(device)
    hgp_criterion = nn.BCEWithLogitsLoss().to(device)

    # ------------------------ 两阶段训练 ------------------------
    print("======== Phase 1: Train head_b (HGP) =========")
    history_b, ckpt_phaseB = train_phaseB(args, model, train_loader, val_loader, device, hgp_criterion)

    print("======== Phase 2: Train head_a (LN) =========")
    history_a, ckpt_phaseA = train_phaseA(args, model, ckpt_phaseB, train_loader, val_loader, device, ln_criterion)

    print("[INFO] Two-phase training finished. Final LN model saved at:", ckpt_phaseA)

if __name__ == "__main__":
    main()
