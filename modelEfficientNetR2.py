#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
model.py

基于原始代码，将 ResNet50_3D 替换为 EfficientNet3D 分支进行特征提取。
去除了所有关于 Swin 的模块，只保留 EfficientNet3D 分支以及相应的任务头。
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0

##################################
# 1. EfficientNet3D 分支相关模块
##################################
class AttentionPooling(nn.Module):
    """
    对特征做注意力池化
    输入 shape: [B, D, F], 输出 [B, F]
    """
    def __init__(self, in_features, hidden_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features // 2
        self.attention = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, 1)
        )
    def forward(self, x):
        attn_scores = self.attention(x)  # [B, D, 1]
        attn_scores = torch.softmax(attn_scores, dim=1)
        x_weighted = (x * attn_scores).sum(dim=1)
        return x_weighted

class EfficientNet3D(nn.Module):
    """
    3D -> 2D 切片再用 EfficientNetB0 提取特征
    """
    def __init__(self, in_channels=2, pretrained_path=None):
        super(EfficientNet3D, self).__init__()
        self.efficientnet = efficientnet_b0(weights=None)
        # 修改第一层卷积，适配 in_channels=2
        self.efficientnet.features[0][0] = nn.Conv2d(
            in_channels,
            32,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )
        # 移除分类头，仅保留特征提取部分
        self.efficientnet.classifier = nn.Identity()
        # B0 输出维度为1280
        self.attn_pool = AttentionPooling(in_features=1280)
        
        # 可加载预训练权重
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"[INFO] Loading pre-trained EfficientNet from {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location="cpu")
            # 若权重是3通道，简单截取前2通道
            if "features.0.0.weight" in state_dict and state_dict["features.0.0.weight"].shape[1] == 3 and in_channels == 2:
                state_dict["features.0.0.weight"] = state_dict["features.0.0.weight"][:, :2, :, :]
            self.efficientnet.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        # x shape: [B, 2, D, H, W]
        B, C, D, H, W = x.shape
        # 在 D 维度切片: [B*D, C, H, W]
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, H, W)
        # 送进 EfficientNetB0 提取特征
        features = self.efficientnet(x)  # [B*D, 1280]
        # 再将 [B*D, 1280] reshape 回 [B, D, 1280]
        features = features.view(B, D, -1)
        # 利用注意力池化得到 [B, 1280]
        pooled = self.attn_pool(features)
        return pooled

##################################
# 2. 最终融合模型 FusionModel (仅使用 EfficientNet3D 分支)
##################################
class FusionModel(nn.Module):
    def __init__(self, num_classes_a=1, num_classes_b=1,
                 efficientnet_pretrain_path="/root/autodl-tmp/efficientnet_b0.pth"):
        super().__init__()
        # EfficientNet3D 分支
        self.efficientnet_branch = EfficientNet3D(
            in_channels=2,
            pretrained_path=efficientnet_pretrain_path
        )
        # 线性聚合层，将 EfficientNet 输出维度 1280 映射到 512
        self.fc = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        # 最终的任务头
        self.head_a = nn.Linear(512, num_classes_a)
        self.head_b = nn.Linear(512, num_classes_b)

    def forward(self, x):
        # EfficientNet 分支输出 1280 维特征
        feat_efficientnet = self.efficientnet_branch(x)  # [B, 1280]
        # 线性聚合
        feat = self.fc(feat_efficientnet)  # [B, 512]
        out_a = self.head_a(feat)
        out_b = self.head_b(feat)
        return out_a, out_b, feat

    def compute_loss(self, outputs, targets):
        """
        简单示例：二分类任务的 BCE Loss
        """
        out_a, out_b, _ = outputs
        target_a, target_b = targets
        loss_a = F.binary_cross_entropy_with_logits(out_a, target_a)
        loss_b = F.binary_cross_entropy_with_logits(out_b, target_b)
        return loss_a + loss_b

##################################
# 3. 测试代码 (仅用于验证网络的尺寸与运行)
##################################
if __name__ == "__main__":
    model = FusionModel()
    x = torch.randn(2, 2, 48, 48, 48)
    out_a, out_b, feat = model(x)
    print("out_a shape:", out_a.shape)    # e.g. [2, 1]
    print("out_b shape:", out_b.shape)    # e.g. [2, 1]
    print("feat shape:", feat.shape)      # e.g. [2, 512]
