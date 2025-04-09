#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
model.py

基于原始代码，将 ResNet50_3D 替换为 EfficientNet3D 分支进行特征提取。
其他部分逻辑（Swin-UNETR3D分支、多任务头、Cross-Attention融合等）保持不变。
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0

######################################
# 1. 基础工具模块：DropPath, Mlp, 窗口分割等
######################################
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def drop_path(x, drop_prob=0., training=False):
    """DropPath 实现"""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor

# -----------------------------
# 使用 resmlp 构造 Mlp 模块
# -----------------------------
try:
    from resmlp import ResMLP  # 需要安装 resmlp-pytorch 库
except ImportError:
    raise ImportError("请先安装 resmlp-pytorch 库: pip install resmlp-pytorch")

class Mlp(nn.Module):
    """
    使用 ResMLP 进行构造
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.resmlp = ResMLP(
            in_channels=in_features,
            image_size=1,
            patch_size=1,
            num_classes=in_features,
            dim=in_features,
            depth=1,
            mlp_dim=hidden_features
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        orig_shape = x.shape
        x = x.view(-1, orig_shape[-1])
        x = x.unsqueeze(-1).unsqueeze(-1)  # [N, in_features, 1, 1]
        x = self.resmlp(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.drop(x)
        return x.view(orig_shape)

def window_partition(x, window_size):
    """
    将 3D 特征图在 (D, H, W) 维度上切分指定大小的窗口
    x shape: [B, D, H, W, C]
    """
    B, D, H, W, C = x.shape
    wD, wH, wW = window_size
    wD = min(wD, D)
    wH = min(wH, H)
    wW = min(wW, W)
    pad_d = (wD - D % wD) % wD
    pad_h = (wH - H % wH) % wH
    pad_w = (wW - W % wW) % wW
    x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
    D_padded = D + pad_d
    H_padded = H + pad_h
    W_padded = W + pad_w
    x = x.view(B,
               D_padded // wD, wD,
               H_padded // wH, wH,
               W_padded // wW, wW,
               C)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return x.view(-1, wD, wH, wW, C)

def window_reverse(windows, window_size, B, D, H, W):
    """
    将窗口还原为原始特征图形状
    """
    wD, wH, wW = window_size
    x = windows.view(B, D // wD, H // wH, W // wW, wD, wH, wW, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return x.view(B, D, H, W, -1)

###############################################
# 2. Swin Transformer模块（3D版）
###############################################
class WindowAttention3D(nn.Module):
    """
    窗口多头自注意力
    """
    def __init__(self, dim, window_size, num_heads,
                 qkv_bias=True, attn_drop=0., proj_drop=0.,
                 use_DropKey=False, mask_ratio=0.2):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_DropKey = use_DropKey
        self.mask_ratio = mask_ratio

    def forward(self, x):
        B, orig_D, orig_H, orig_W, C = x.shape
        wD, wH, wW = self.window_size
        pad_d = (wD - orig_D % wD) % wD
        pad_h = (wH - orig_H % wH) % wH
        pad_w = (wW - orig_W % wW) % wW
        if pad_d or pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        _, D, H, W, _ = x.shape

        windows = window_partition(x, (wD, wH, wW))              # [n_windows*B, wD, wH, wW, C]
        windows = windows.view(-1, wD * wH * wW, C)              # [n_windows*B, wD*wH*wW, C]

        qkv = self.qkv(windows).reshape(-1, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]                         # [batch*n_windows, num_heads, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # 使用 DropKey
        if self.use_DropKey:
            # 这里通过将某些 attention 分数设为 -1e12 来模拟被 mask
            m_r = torch.ones_like(attn) * self.mask_ratio
            attn = attn + torch.bernoulli(m_r) * -1e12

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(-1, wD, wH, wW, C)
        x = window_reverse(x, (wD, wH, wW), B, D, H, W)
        return x[:, :orig_D, :orig_H, :orig_W, :]

class SwinTransformerBlock3D(nn.Module):
    def __init__(self, dim, input_resolution, num_heads,
                 window_size=(7,7,7), shift_size=(0,0,0),
                 mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0.,
                 ls_init_value=1e-5,
                 use_DropKey=False, mask_ratio=0.2):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(
            dim, window_size, num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            use_DropKey=use_DropKey,  # 传递
            mask_ratio=mask_ratio
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
        # LayerScale
        self.ls_gamma1 = nn.Parameter(ls_init_value * torch.ones((dim)), requires_grad=True)
        self.ls_gamma2 = nn.Parameter(ls_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1)  # -> [B, D, H, W, C]

        # 自注意力
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(self.ls_gamma1 * x)

        # MLP
        shortcut2 = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut2 + self.drop_path(self.ls_gamma2 * x)

        x = x.permute(0, 4, 1, 2, 3)  # -> [B, C, D, H, W]
        return x

class SwinUNETR3DEncoderPlus(nn.Module):
    """
    SwinUNETR3D 的编码器部分
    """
    def __init__(self, in_channels=2, embed_dim=126,
                 depths=(2,2,4,4,2), num_heads=(3,6,9,9,7),
                 window_size=(7,7,7), mlp_ratio=4.,
                 drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.2, input_size=(48,48,48),
                 use_DropKey=False, mask_ratio=0.2):
        super().__init__()
        self.patch_embed = nn.Conv3d(in_channels, embed_dim, kernel_size=4, stride=4, padding=0)
        self.blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        idx = 0
        for i in range(len(depths)):
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(
                    SwinTransformerBlock3D(
                        dim=embed_dim,
                        input_resolution=(input_size[0] // 4, input_size[1] // 4, input_size[2] // 4),
                        num_heads=num_heads[i],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[idx],
                        use_DropKey=use_DropKey,
                        mask_ratio=mask_ratio
                    )
                )
                idx += 1
            self.blocks.append(nn.Sequential(*stage_blocks))

    def forward(self, x):
        # x: [B, 2, D, H, W]
        x = self.patch_embed(x)  # [B, embed_dim, D/4, H/4, W/4]
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features

def load_swin_unetr_weights(model, pretrain_path, weights_only=True):
    """
    加载 SwinUNETR3D 预训练权重
    """
    if (not pretrain_path) or (not os.path.exists(pretrain_path)):
        print(f"[WARN] Swin UNETR 预训练权重 '{pretrain_path}' 未找到，将使用随机初始化。")
        return
    checkpoint = torch.load(pretrain_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    # 过滤掉一些不需要的key
    filtered_dict = {k: v for k, v in state_dict.items() if not k.startswith("head_a")}
    model.load_state_dict(filtered_dict, strict=False)
    print("[INFO] 已加载 Swin UNETR 预训练权重。")

##################################
# 3. SwinUNETR3DPlus 编码器整体
##################################
class SwinUNETR3DPlus(nn.Module):
    """
    包含编码器 + 特征聚合 + 双任务头
    """
    def __init__(self, in_channels=2, num_classes_a=1, num_classes_b=1,
                 embed_dim=126, depths=(2,2,4,4,2), num_heads=(3,6,9,9,7),
                 window_size=(6,6,6), mlp_ratio=4., drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, input_size=(48,48,48),
                 use_DropKey=False, mask_ratio=0.2):
        super().__init__()
        self.encoder = SwinUNETR3DEncoderPlus(
            in_channels=in_channels,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            input_size=input_size,
            use_DropKey=use_DropKey,
            mask_ratio=mask_ratio
        )
        # 线性聚合
        self.agg = nn.Sequential(
            nn.Linear(embed_dim * len(depths), 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        self.head_a = nn.Linear(256, num_classes_a)
        self.head_b = nn.Linear(256, num_classes_b)

    def forward(self, x):
        """
        x: [B, 2, D, H, W]
        """
        features = self.encoder(x)
        pooled_list = []
        for feat in features:
            # feat: [B, C, D/4, H/4, W/4]
            pooled = feat.mean(dim=(2,3,4))  # [B, C]
            pooled_list.append(pooled)
        aggregated = torch.cat(pooled_list, dim=1)  # [B, C1 + C2 + ...]
        x = self.agg(aggregated)                    # [B, 256]
        return self.head_a(x), self.head_b(x), x

##################################
# 4. EfficientNet3D 分支
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
        # 在D维度切片: [B*D, C, H, W]
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * D, C, H, W)
        # 送进 EfficientNetB0 提取特征
        features = self.efficientnet(x)  # [B*D, 1280]
        # 再将 [B*D, 1280] reshape 回 [B, D, 1280]
        features = features.view(B, D, -1)
        # 利用注意力池化得到 [B, 1280]
        pooled = self.attn_pool(features)
        return pooled

##################################
# 5. Cross-Attention Fusion 模块
##################################
class CrossAttentionFusion(nn.Module):
    """
    双向交叉注意力 + 残差融合
    """
    def __init__(self, dim_swin, dim_resnet, fused_dim=512, num_heads=8,
                 use_DropKey=False, mask_ratio=0.2):
        super().__init__()
        self.use_DropKey = use_DropKey
        self.mask_ratio = mask_ratio

        self.query_proj1 = nn.Linear(dim_swin, fused_dim)
        self.key_proj1   = nn.Linear(dim_resnet, fused_dim)
        self.value_proj1 = nn.Linear(dim_resnet, fused_dim)

        self.query_proj2 = nn.Linear(dim_resnet, fused_dim)
        self.key_proj2   = nn.Linear(dim_swin, fused_dim)
        self.value_proj2 = nn.Linear(dim_swin, fused_dim)

        self.cross_attn1 = nn.MultiheadAttention(embed_dim=fused_dim, num_heads=num_heads, batch_first=True)
        self.cross_attn2 = nn.MultiheadAttention(embed_dim=fused_dim, num_heads=num_heads, batch_first=True)

        # 合并两个方向的注意力结果
        self.fusion_fc = nn.Linear(fused_dim * 2, fused_dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(fused_dim * 2, fused_dim),
            nn.Sigmoid()
        )
        # 残差合并
        self.final_fc = nn.Linear(dim_swin + dim_resnet, fused_dim)
        self.final_gate = nn.Sequential(
            nn.Linear(dim_swin + dim_resnet, fused_dim),
            nn.Sigmoid()
        )

    def forward(self, swin_feat, resnet_feat):
        # 1) Swin -> EfficientNet
        Q1 = self.query_proj1(swin_feat).unsqueeze(1)       # [B, 1, fused_dim]
        K1 = self.key_proj1(resnet_feat).unsqueeze(1)       # [B, 1, fused_dim]
        V1 = self.value_proj1(resnet_feat).unsqueeze(1)     # [B, 1, fused_dim]
        attn_out1, _ = self.cross_attn1(Q1, K1, V1)
        attn_out1 = attn_out1.squeeze(1)                    # [B, fused_dim]

        # 2) EfficientNet -> Swin
        Q2 = self.query_proj2(resnet_feat).unsqueeze(1)     # [B, 1, fused_dim]
        K2 = self.key_proj2(swin_feat).unsqueeze(1)         # [B, 1, fused_dim]
        V2 = self.value_proj2(swin_feat).unsqueeze(1)       # [B, 1, fused_dim]
        attn_out2, _ = self.cross_attn2(Q2, K2, V2)
        attn_out2 = attn_out2.squeeze(1)                    # [B, fused_dim]

        # 3) concat & 门控融合
        attn_concat = torch.cat([attn_out1, attn_out2], dim=1)   # [B, 2*fused_dim]
        gate_val = self.fusion_gate(attn_concat)                 # [B, fused_dim]
        fused_attn = self.fusion_fc(attn_concat)                 # [B, fused_dim]
        fused_attn = gate_val * fused_attn

        # 4) 残差合并 (原始特征 + 注意力特征)
        original_concat = torch.cat([swin_feat, resnet_feat], dim=1)  # [B, dim_swin + dim_resnet]
        gate_orig = self.final_gate(original_concat)                  # [B, fused_dim]
        orig_proj = self.final_fc(original_concat)                    # [B, fused_dim]
        final = gate_orig * fused_attn + (1 - gate_orig) * orig_proj
        return final

##################################
# 6. 最终融合模型 FusionModel
##################################
class FusionModel(nn.Module):
    def __init__(self, num_classes_a=1, num_classes_b=1,
                 swin_pretrain_path="/root/autodl-tmp/model_swinvit.pt",
                 efficientnet_pretrain_path="/root/autodl-tmp/efficientnet_b0.pth",
                 swin_embed_dim=126,
                 fusion_dropout=0.3,
                 use_DropKey=False, mask_ratio=0.2):
        super().__init__()
        # 6.1 Swin UNETR 3D 分支
        self.swin_branch = SwinUNETR3DPlus(
            in_channels=2,
            embed_dim=swin_embed_dim,
            depths=(3,3,6,6,2),
            num_heads=(3,6,9,9,7),
            window_size=(6,6,6),
            input_size=(48,48,48),
            use_DropKey=use_DropKey,
            mask_ratio=mask_ratio
        )
        load_swin_unetr_weights(self.swin_branch.encoder, swin_pretrain_path, weights_only=True)

        # 6.2 EfficientNet3D 分支
        self.efficientnet_branch = EfficientNet3D(
            in_channels=2,
            pretrained_path=efficientnet_pretrain_path
        )

        # 6.3 Cross-Attention Fusion
        #    dim_swin=256 (SwinUNETR3DPlus 的聚合层输出)
        #    dim_resnet=1280 (EfficientNet-B0 输出)
        self.cross_fusion = CrossAttentionFusion(
            dim_swin=256, dim_resnet=1280, fused_dim=512, num_heads=8,
            use_DropKey=use_DropKey, mask_ratio=mask_ratio
        )
        
        # 6.4 最终的任务头
        self.head_a = nn.Linear(512, num_classes_a)
        self.head_b = nn.Linear(512, num_classes_b)

    def forward(self, x):
        # Swin 分支输出 (out_a, out_b, feat_swin)
        _, _, feat_swin = self.swin_branch(x)            # [B, 256]
        # EfficientNet 分支输出 1280维特征
        feat_efficientnet = self.efficientnet_branch(x)  # [B, 1280]

        # 融合
        fused = self.cross_fusion(feat_swin, feat_efficientnet)  # [B, 512]
        out_a = self.head_a(fused)
        out_b = self.head_b(fused)
        return out_a, out_b, fused

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
# 7. 测试代码 (仅用于验证网络的尺寸与运行)
##################################
if __name__ == "__main__":
    model = FusionModel()
    x = torch.randn(2, 2, 48, 48, 48)
    out_a, out_b, fused = model(x)
    print("out_a shape:", out_a.shape)    # e.g. [2, 1]
    print("out_b shape:", out_b.shape)    # e.g. [2, 1]
    print("fused shape:", fused.shape)    # e.g. [2, 512]
