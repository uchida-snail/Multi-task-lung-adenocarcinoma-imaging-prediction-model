#!/usr/bin/env python
"""
model.py

该模型移除了 Swin 分支，仅保留 ResNet10_3D 分支，
用于 3D 数据的多任务预测。其它部分保持不变，保留了 resnet10 的代码。
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, Bottleneck
from torch_optimizer import Ranger

##################################
# AttentionPooling 模块
##################################
class AttentionPooling(nn.Module):
    """
    对 ResNet10_3D 分支中 D 维度的特征进行注意力加权池化
    输入 x shape: [B, D, F]，输出 [B, F]
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

##################################
# SEBlock 模块（Squeeze-and-Excitation）
##################################
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        scale = self.fc(x)
        return x * scale

##################################
# ChannelAttention 通道注意力（可选）
##################################
class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
    def forward(self, x):
        avg = self.fc(x.mean(dim=0))
        mx = self.fc(x.max(dim=0)[0])
        return x * (avg + mx)

##################################
# ResNet10 模型构造函数
##################################
def resnet10(weights=None):
    """
    构造 ResNet10 模型，使用 Bottleneck 模块，层数为 [1, 1, 1, 1]，
    最终输出通道为 512 * Bottleneck.expansion = 2048
    """
    return ResNet(Bottleneck, [1, 1, 1, 1])

##################################
# ResNet10_3D 分支
##################################
class ResNet50_3D(nn.Module):
    """
    在 D 维度做切片，再用 2D ResNet10 提取特征，
    最后在 D 维度利用注意力池化得到整张 volume 的特征
    """
    def __init__(self, in_channels=2, pretrained_path=None):
        super(ResNet50_3D, self).__init__()
        self.resnet2d = resnet10(weights=None)
        # 修改第一层卷积以适应 in_channels（例如 2 通道）
        self.resnet2d.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet2d.bn1 = nn.BatchNorm2d(64)
        self.resnet2d.fc = nn.Identity()
        self.attn_pool = AttentionPooling(in_features=2048)
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"[INFO] Loading pre-trained ResNet10 from {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            if "conv1.weight" in state_dict and state_dict["conv1.weight"].shape[1] == 3 and in_channels == 2:
                state_dict["conv1.weight"] = state_dict["conv1.weight"][:, :2, :, :]
            self.resnet2d.load_state_dict(state_dict, strict=False)
    def forward(self, x):
        B, C, D, H, W = x.shape
        # 将 3D volume 在 D 维度切片后传入 2D ResNet10 提取特征
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, H, W)
        features = self.resnet2d(x)
        features = features.view(B, D, -1)
        pooled = self.attn_pool(features)
        return pooled

##################################
# 最终模型：仅使用 ResNet10_3D 分支
##################################
class FusionModel(nn.Module):
    def __init__(self, num_classes_a=1, num_classes_b=1,
                 resnet_pretrain_path="resnet_10.pth"):
        super(FusionModel, self).__init__()
        # ResNet 分支（使用 ResNet10）
        self.resnet_branch = ResNet50_3D(
            in_channels=2,
            pretrained_path=resnet_pretrain_path
        )
        # 聚合层：将 ResNet 分支输出的 2048 维特征映射到 512 维
        self.agg = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.head_a = nn.Linear(512, num_classes_a)
        self.head_b = nn.Linear(512, num_classes_b)

    def forward(self, x):
        # ResNet 分支输出 2048 维特征
        feat_resnet = self.resnet_branch(x)       # [B, 2048]
        aggregated = self.agg(feat_resnet)          # [B, 512]
        out_a = self.head_a(aggregated)
        out_b = self.head_b(aggregated)
        return out_a, out_b, aggregated

##################################
# 测试代码
##################################
if __name__ == "__main__":
    model = FusionModel()
    x = torch.randn(2, 2, 48, 48, 48)  # 输入形状 [B, C, D, H, W]
    out_a, out_b, aggregated = model(x)
    print("Output shapes:", out_a.shape, out_b.shape, aggregated.shape)
