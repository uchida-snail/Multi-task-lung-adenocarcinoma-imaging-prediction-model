#!/usr/bin/env python
"""
ViT 3D UNETR Advanced Model with Skip Connections and Multi-scale Decoder
---------------------------------------------------------------------------
本模块实现了一个改进版的基于传统 ViT 的 3D 分割模型，
在原始简化模型基础上增加了跳跃连接和多尺度解码器，
使模型具有更强的表达能力。
主要流程：
  1. 使用 3D 卷积进行 patch embedding，并添加 learnable 位置编码（支持动态插值）。
  2. 使用多层 Transformer block 进行全局特征提取，并在部分 block 后保存 skip token。
  3. 重构 Transformer token 为 3D 特征图，通过多尺度解码器逐级上采样融合 skip 特征，
     最终恢复到原始分辨率，并通过 1x1 卷积输出分割结果。
     
可根据需要修改 in_channels、out_channels、img_size、patch_size、embed_dim、depth、num_heads、mlp_dim、dropout_rate 等参数。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# MLP 模块
# ---------------------------
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, dropout_rate=0.0):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

# ---------------------------
# Transformer Block（标准 ViT block）
# ---------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, dropout_rate=0.0):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                          dropout=dropout_rate, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_dim, dropout_rate)
    
    def forward(self, x):
        # x: (B, N, dim)
        attn_out = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

# ---------------------------
# Decoder Block：上采样融合跳跃连接
# ---------------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, skip):
        x = self.upsample(x)
        # 若尺寸不匹配，则补零
        if x.shape[2:] != skip.shape[2:]:
            diffD = skip.shape[2] - x.shape[2]
            diffH = skip.shape[3] - x.shape[3]
            diffW = skip.shape[4] - x.shape[4]
            x = F.pad(x, [diffW//2, diffW - diffW//2,
                          diffH//2, diffH - diffH//2,
                          diffD//2, diffD - diffD//2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

# ---------------------------
# ViT 3D UNETR Advanced Model
# ---------------------------
class ViT3DUNETR(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, img_size=(80,80,80), patch_size=(8,8,8),
                 embed_dim=96, depth=12, num_heads=8, mlp_dim=768, dropout_rate=0.1):
        """
        参数：
          - in_channels: 输入通道数
          - out_channels: 分割类别数
          - img_size: 用于初始化位置编码的参考图像尺寸 (D, H, W)
          - patch_size: 每个 patch 的尺寸（3D）
          - embed_dim: patch embedding 的通道数
          - depth: Transformer block 数目
          - num_heads: 注意力头数
          - mlp_dim: MLP 隐藏层维度
          - dropout_rate: dropout 概率
        """
        super(ViT3DUNETR, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # 计算参考 patch grid 大小
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1],
                          img_size[2] // patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        # Patch embedding
        self.patch_embed = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # 初始化位置编码（参考尺寸）
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer 编码器
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # 为了获取跳跃连接，我们取部分 Transformer block 的输出
        # 例如：在 block 4 和 block 8 后保存输出
        self.skip_indices = [depth//3, 2*depth//3]

        # Decoder部分：将最后输出和跳跃连接进行融合，逐级上采样恢复原始分辨率
        self.decoder1 = DecoderBlock(embed_dim, embed_dim, embed_dim)
        self.decoder2 = DecoderBlock(embed_dim, embed_dim, embed_dim)
        self.final_conv = nn.Conv3d(embed_dim, out_channels, kernel_size=1)

    def forward(self, x):
        """
        输入：
          x: (B, in_channels, D, H, W)
        输出：
          out: (B, out_channels, D, H, W)
        """
        B = x.shape[0]
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, D', H', W')
        Dp, Hp, Wp = x.shape[2], x.shape[3], x.shape[4]
        # Flatten并转置
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        N = x.shape[1]
        # 动态位置编码调整
        if N != self.pos_embed.shape[1]:
            pos_embed = F.interpolate(self.pos_embed.transpose(1,2), size=N, mode='linear', align_corners=False).transpose(1,2)
        else:
            pos_embed = self.pos_embed
        x = x + pos_embed

        # 记录跳跃连接输出
        skips = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.skip_indices:
                skips.append(x)
        x = self.norm(x)  # (B, N, embed_dim)

        # 将 transformer token 重构为 3D 特征图
        x = x.transpose(1,2).view(B, self.embed_dim, Dp, Hp, Wp)

        # 将跳跃连接也重构为 3D 特征图
        skip_features = []
        for s in skips:
            s = s.transpose(1,2).view(B, self.embed_dim, Dp, Hp, Wp)
            skip_features.append(s)

        # Decoder：这里简单采用两级 decoder，将最后的特征与较低层的跳跃连接融合
        d = self.decoder1(x, skip_features[-1])  # 融合最后一个跳跃连接
        d = self.decoder2(d, skip_features[0])     # 融合较早的跳跃连接
        # 最后上采样恢复到原始分辨率，注意patch_embed下采样因子为 patch_size
        d = F.interpolate(d, scale_factor=self.patch_size, mode='trilinear', align_corners=False)
        out = self.final_conv(d)
        return out

if __name__ == "__main__":
    # 简单测试
    model = ViT3DUNETR(in_channels=1, out_channels=3,
                        img_size=(80, 80, 80), patch_size=(8, 8, 8),
                        embed_dim=96, depth=12, num_heads=8, mlp_dim=768, dropout_rate=0.1)
    x = torch.randn(1, 1, 80, 80, 80)
    out = model(x)
    print("输出形状:", out.shape)  # 期望输出形状为 (1, 3, 80, 80, 80)
