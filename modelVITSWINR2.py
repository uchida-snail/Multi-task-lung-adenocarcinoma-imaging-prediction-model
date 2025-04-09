#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

##############################################
# 1. 简化版 ViT3D UNETR
##############################################
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

class ViT3DUNETR(nn.Module):
    """
    一个简化版的 ViT3D UNETR，仅保留编码器部分，并提供 extract_features 接口，
    在需要融合特征时可返回全局向量。
    """
    def __init__(self, 
                 in_channels=1, 
                 out_channels=3,
                 img_size=(80,80,80), 
                 patch_size=(8,8,8),
                 embed_dim=96, 
                 depth=12, 
                 num_heads=8, 
                 mlp_dim=768, 
                 dropout_rate=0.1):
        super(ViT3DUNETR, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # 计算总的patch数
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1],
                          img_size[2] // patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        # Patch embedding
        self.patch_embed = nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer 编码器
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # 这里只做一个简化的 segmentation head
        self.final_conv = nn.Conv3d(embed_dim, out_channels, kernel_size=1)

    def forward(self, x):
        """
        标准 forward：输出分割结果
        """
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, embed_dim, D', H', W')
        Dp, Hp, Wp = x.shape[2], x.shape[3], x.shape[4]
        # Flatten并转置
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        x = x + self.pos_embed[:, :x.size(1), :]
        # Transformer编码
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)  # (B, N, embed_dim)
        # 重构回3D特征图
        x = x.transpose(1,2).view(B, self.embed_dim, Dp, Hp, Wp)
        # 最终输出分割
        x = F.interpolate(x, scale_factor=self.patch_size, mode='trilinear', align_corners=False)
        out = self.final_conv(x)
        return out

    def extract_features(self, x):
        """
        用于融合模型时，提取全局特征向量
        """
        B = x.shape[0]
        # Patch embed
        x = self.patch_embed(x)  # (B, embed_dim, D', H', W')
        Dp, Hp, Wp = x.shape[2], x.shape[3], x.shape[4]
        # Flatten并转置
        x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        # 加上位置编码
        if x.shape[1] != self.num_patches:
            # 若与原始预设形状不同，可根据需要插值
            pos_embed = F.interpolate(self.pos_embed.transpose(1,2), 
                                      size=x.shape[1], 
                                      mode='linear', 
                                      align_corners=False).transpose(1,2)
        else:
            pos_embed = self.pos_embed
        x = x + pos_embed[:, :x.size(1), :]
        # Transformer编码
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)  # (B, N, embed_dim)
        # 用 mean pool 获取全局表征
        x = x.mean(dim=1)  # [B, embed_dim]
        return x

##############################################
# 2. SwinUNETR3DPlus (只保留必要代码)
##############################################

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

def window_partition(x, window_size):
    # x: [B, D, H, W, C]
    B, D, H, W, C = x.shape
    wD, wH, wW = window_size
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
    # shape: [B, D//wD, H//wH, W//wW, wD, wH, wW, C] => flatten => [B*(#windows), wD, wH, wW, C]
    return x.view(-1, wD, wH, wW, C)

def window_reverse(windows, window_size, B, D, H, W):
    wD, wH, wW = window_size
    x = windows.view(B,
                     D // wD, H // wH, W // wW,
                     wD, wH, wW, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return x.view(B, D, H, W, -1)

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
        # x: [B, D, H, W, C]
        B, orig_D, orig_H, orig_W, C = x.shape
        wD, wH, wW = self.window_size
        pad_d = (wD - orig_D % wD) % wD
        pad_h = (wH - orig_H % wH) % wH
        pad_w = (wW - orig_W % wW) % wW
        if pad_d or pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        _, D, H, W, _ = x.shape

        windows = window_partition(x, (wD, wH, wW))
        # windows shape: [num_windows, wD, wH, wW, C]
        num_windows = windows.shape[0]
        N_tokens = wD * wH * wW

        # flatten => [num_windows, N_tokens, C]
        windows = windows.view(num_windows, N_tokens, C)
        qkv = self.qkv(windows)  # => [num_windows, N_tokens, 3*C]
        head_dim = C // self.num_heads
        # => reshape => [num_windows, N_tokens, 3, num_heads, head_dim]
        qkv = qkv.view(num_windows, N_tokens, 3, self.num_heads, head_dim)
        # => permute => [3, num_windows, num_heads, N_tokens, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each => [num_windows, num_heads, N_tokens, head_dim]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        if self.use_DropKey:
            m_r = torch.ones_like(attn) * self.mask_ratio
            attn = attn + torch.bernoulli(m_r) * -1e12
        attn = self.attn_drop(attn)

        x = (attn @ v)  # => [num_windows, num_heads, N_tokens, head_dim]
        x = x.transpose(1, 2).reshape(num_windows, N_tokens, C)
        # => [num_windows, wD*wH*wW, C]
        # 还原回 [B, D, H, W, C]
        x = x.view(num_windows, wD, wH, wW, C)
        x = window_reverse(x, (wD, wH, wW), B, D, H, W)
        # 截断多余 padding
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
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop)

        self.ls_gamma1 = nn.Parameter(ls_init_value * torch.ones((dim)), requires_grad=True)
        self.ls_gamma2 = nn.Parameter(ls_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        # x: [B, C, D, H, W]
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1)  # => [B, D, H, W, C]

        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        x = shortcut + self.drop_path(self.ls_gamma1 * x)

        shortcut2 = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut2 + self.drop_path(self.ls_gamma2 * x)

        x = x.permute(0, 4, 1, 2, 3)  # => [B, C, D, H, W]
        return x

class SwinUNETR3DEncoderPlus(nn.Module):
    def __init__(self, 
                 in_channels=2, 
                 embed_dim=96,
                 depths=(2,2,4,4),
                 num_heads=(3,6,12,12),
                 window_size=(7,7,7), 
                 mlp_ratio=4.,
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0.2, 
                 input_size=(80,80,80),
                 use_DropKey=False, 
                 mask_ratio=0.2):
        super().__init__()
        self.patch_embed = nn.Conv3d(in_channels, embed_dim, kernel_size=4, stride=4, padding=0)

        self.blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        idx = 0
        for i in range(len(depths)):
            block_list = []
            for _ in range(depths[i]):
                block = SwinTransformerBlock3D(
                    dim=embed_dim,
                    input_resolution=(input_size[0]//4, input_size[1]//4, input_size[2]//4),
                    num_heads=num_heads[i],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[idx],
                    use_DropKey=use_DropKey,
                    mask_ratio=mask_ratio
                )
                block_list.append(block)
                idx += 1
            self.blocks.append(nn.Sequential(*block_list))

    def forward(self, x):
        x = self.patch_embed(x)
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features

    def freeze_encoder(self, freeze_stages=2):
        # 在 partial_freeze_swin 中会调用
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
    checkpoint = torch.load(pretrain_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    filtered_dict = {k: v for k, v in state_dict.items() if not k.startswith("head_a")}
    model.load_state_dict(filtered_dict, strict=False)
    print("[INFO] Pre-trained weights loaded.")

class SwinUNETR3DPlus(nn.Module):
    def __init__(self, 
                 in_channels=2, 
                 num_classes_a=1, 
                 num_classes_b=1,
                 embed_dim=96, 
                 depths=(2,2,4,4), 
                 num_heads=(3,6,12,12),
                 window_size=(7,7,7), 
                 mlp_ratio=4., 
                 drop_rate=0.,
                 attn_drop_rate=0., 
                 drop_path_rate=0.2, 
                 input_size=(80,80,80),
                 use_DropKey=False, 
                 mask_ratio=0.2):
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
        返回 (head_a_logits, head_b_logits, 全局特征向量)
        """
        features = self.encoder(x)
        pooled_list = []
        for feat in features:
            B, C, D, H, W = feat.shape
            pooled = feat.mean(dim=(2,3,4))  # [B, C]
            pooled_list.append(pooled)
        aggregated = torch.cat(pooled_list, dim=1)  # [B, C * num_stage]
        out_vec = self.agg(aggregated)             # [B, 256]
        out_a = self.head_a(out_vec)               # LN 任务
        out_b = self.head_b(out_vec)               # HGP 任务
        return out_a, out_b, out_vec

    def extract_features(self, x):
        """
        提供与 ViT 分支对齐的全局特征接口
        """
        with torch.no_grad():
            _, _, feat = self.forward(x)
        return feat

##############################################
# 3. CrossAttentionFusion (可选，如果做分支融合)
##############################################
# 这里省略了 CrossAttentionFusion，如果你在 FusionModel_ViT3D_Swin
# 用不到可不写，或者按你需求增删。
##############################################

##############################################
# 4. FusionModel：Swin + ViT3D 双分支
##############################################
class FusionModel_ViT3D_Swin(nn.Module):
    """
    利用 SwinUNETR3DPlus 和 ViT3DUNETR 做双分支，
    通过(可选) CrossAttentionFusion 进行最终特征融合，
    输出可做分类/多任务预测。
    """
    def __init__(self, 
                 in_channels=2,
                 num_classes=1,
                 dim_swin=256,  # 与 SwinUNETR3DPlus agg 输出相匹配 => 256
                 dim_vit=96,    # 与 ViT3DUNETR embed_dim 相匹配 => 96
                 fused_dim=512, 
                 use_DropKey=False, 
                 mask_ratio=0.2):
        super().__init__()
        # 1) Swin 分支
        self.swin_branch = SwinUNETR3DPlus(
            in_channels=in_channels,
            num_classes_a=1,
            num_classes_b=1,
            embed_dim=96, 
            depths=(2,2,4,4),
            num_heads=(3,6,12,12),
            window_size=(7,7,7),
            input_size=(80,80,80),
            use_DropKey=use_DropKey,
            mask_ratio=mask_ratio
        )
        # 2) ViT3D 分支
        self.vit_branch = ViT3DUNETR(
            in_channels=in_channels,
            out_channels=1,  # 只是给一个值，这里不影响 extract_features
            img_size=(80,80,80),
            patch_size=(8,8,8),
            embed_dim=dim_vit,
            depth=6,
            num_heads=8,
            mlp_dim=dim_vit*4,
            dropout_rate=0.1
        )

        # 3) 最终融合：这里演示最简单的 concat + 2个 head
        #    你也可以选择 cross-attention 或别的注意力方式
        self.fuse_fc = nn.Linear(dim_swin + dim_vit, fused_dim)
        self.head_a = nn.Linear(fused_dim, 1)  # LN 任务
        self.head_b = nn.Linear(fused_dim, 1)  # HGP 任务

    def forward(self, x):
        """
        返回 (out_a, out_b, fused)
        """
        # 从 Swin 分支得到 256 维特征
        outA_swin, outB_swin, feat_swin = self.swin_branch(x)  # 这里 outA_swin/outB_swin 不用 => 只是 skip
        # 从 ViT3D 分支得到 96 维特征
        feat_vit = self.vit_branch.extract_features(x)         # shape [B, 96]

        # 简单拼接 => [B, 256 + 96]
        fused = torch.cat([feat_swin, feat_vit], dim=1)
        fused = self.fuse_fc(fused)  # [B, fused_dim]

        # 两个任务头
        out_a = self.head_a(fused)   # LN
        out_b = self.head_b(fused)   # HGP
        # 返回3个值
        return out_a, out_b, fused

