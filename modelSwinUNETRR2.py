#!/usr/bin/env python
"""
swinUNETR3DPlus 模型

包含 SwinUNETR3DPlus 模型及相关组件，用于 3D 数据的多任务预测。
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from resmlp import ResMLP  # 请确保已安装 resmlp-pytorch 库，例如：pip install resmlp-pytorch
except ImportError:
    raise ImportError("请安装 resmlp-pytorch 库，例如：pip install resmlp-pytorch")

#######################################
# 基础工具模块：DropPath, Mlp, 窗口分割等
#######################################

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor

class Mlp(nn.Module):
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
    wD, wH, wW = window_size
    x = windows.view(B, D // wD, H // wH, W // wW, wD, wH, wW, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return x.view(B, D, H, W, -1)

#######################################
# Swin 模块
#######################################

class WindowAttention3D(nn.Module):
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
        windows = window_partition(x, (wD, wH, wW))
        windows = windows.view(-1, wD * wH * wW, C)
        qkv = self.qkv(windows).reshape(-1, 3, self.num_heads, C // self.num_heads).permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        if self.use_DropKey:
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
            use_DropKey=use_DropKey,
            mask_ratio=mask_ratio
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

        self.ls_gamma1 = nn.Parameter(ls_init_value * torch.ones((dim)), requires_grad=True)
        self.ls_gamma2 = nn.Parameter(ls_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1)  # -> [B, D, H, W, C]

        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(self.ls_gamma1 * x)

        shortcut2 = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut2 + self.drop_path(self.ls_gamma2 * x)

        x = x.permute(0, 4, 1, 2, 3)  # -> [B, C, D, H, W]
        return x

class SwinUNETR3DEncoderPlus(nn.Module):
    """
    SwinUNETR3D 编码器
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
        for i in range(len(depths)):
            block = nn.Sequential(*[
                SwinTransformerBlock3D(
                    dim=embed_dim,
                    input_resolution=(input_size[0]//4, input_size[1]//4, input_size[2]//4),
                    num_heads=num_heads[i],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i]) + j],
                    use_DropKey=use_DropKey,
                    mask_ratio=mask_ratio
                ) for j in range(depths[i])
            ])
            self.blocks.append(block)

    def forward(self, x):
        x = self.patch_embed(x)
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features

    def freeze_encoder(self, freeze_stages=2):
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        for i, block in enumerate(self.blocks):
            if i < freeze_stages:
                for param in block.parameters():
                    param.requires_grad = False

def load_swin_unetr_weights(model, pretrain_path, weights_only=True):
    if (not pretrain_path) or (not os.path.exists(pretrain_path)):
        print(f"[WARN] Pretrain path '{pretrain_path}' not found. Use random init.")
        return
    checkpoint = torch.load(pretrain_path, map_location='cpu', weights_only=weights_only)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    filtered_dict = {k: v for k, v in state_dict.items() if not k.startswith("head_a")}
    model.load_state_dict(filtered_dict, strict=False)
    print("[INFO] Pre-trained weights loaded.")

class SwinUNETR3DPlus(nn.Module):
    """
    SwinUNETR3DPlus 主体模型：包含编码器、特征聚合层及两个任务 head
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
        load_swin_unetr_weights(self.encoder, pretrain_path="/root/autodl-tmp/model_swinvit.pt", weights_only=True)
        self.agg = nn.Sequential(
            nn.Linear(embed_dim * len(depths), 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        self.head_a = nn.Linear(256, num_classes_a)
        self.head_b = nn.Linear(256, num_classes_b)

    def forward(self, x):
        features = self.encoder(x)
        pooled_list = []
        for feat in features:
            pooled = feat.mean(dim=(2,3,4))
            pooled_list.append(pooled)
        aggregated = torch.cat(pooled_list, dim=1)
        x = self.agg(aggregated)
        return self.head_a(x), self.head_b(x), x

#######################################
# 测试代码
#######################################
if __name__ == "__main__":
    model = SwinUNETR3DPlus()
    x = torch.randn(2, 2, 48, 48, 48)
    out_a, out_b, features = model(x)
    print("Output shapes:", out_a.shape, out_b.shape, features.shape)
