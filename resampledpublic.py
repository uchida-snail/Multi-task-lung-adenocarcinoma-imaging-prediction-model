#!/usr/bin/env python
"""
预处理脚本（针对 resampled 数据）

处理 resampled 文件中的 "image" (CT原图) 和 "mask" (二值 mask)，  
首先利用 mask 计算 ROI 的中心，并根据设定大小裁剪，  
随后对裁剪后的图像进行数据增强，  
最后拼接得到 2 通道的 "image_cropped"。

标签：  
  根据 CSV 文件中的信息，生成双输出头：  
    - head_a: LN 分类（若 LN==0 则为 0，否则为 1）  
    - head_b: HGP 标签（直接采用 CSV 中的 HGP 数值）

要求：  
  - 只处理文件夹中带 "resampled_" 前缀的文件  
  - 文件命名规则：  
      image 文件： resampled_{PatientID}.nii.gz  
      mask 文件： resampled_{PatientID}label.nii.gz  
  - 标签 CSV 文件路径： /root/autodl-tmp/LNMetastasis250402V2.csv  
  - 数据文件夹路径： /root/autodl-tmp/ConvertedV2-250404
"""

import os
import torch
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from monai.transforms import (
    Compose, LoadImaged, Spacingd,
    ScaleIntensityRanged, CropForegroundd, EnsureTyped,
    RandFlipd, RandRotate90d, RandShiftIntensityd,
    ToTensord, Lambda
)

logging.basicConfig(level=logging.INFO)

# -------------------- 1) 自定义拼接函数 --------------------
def cat_image_mask(sample):
    """
    将 sample["image"] 和 sample["mask"]（均为 [1, D, H, W] 的 tensor）  
    拼接得到 [2, D, H, W]，同时保留双输出标签 head_a 和 head_b。
    """
    img = sample["image"]   # shape: [1, D, H, W]
    msk = sample["mask"]    # shape: [1, D, H, W]
    image_cropped = torch.cat([img, msk], dim=0)  # => [2, D, H, W]
    sample["image_cropped"] = image_cropped
    return sample

# -------------------- 2) 自定义基于 mask 计算 ROI 中心并裁剪的 transform --------------------
def CropROICenterd(roi_size=(80, 80, 80)):
    """
    根据 mask 中的非零值计算 ROI 的中心，并以该中心裁剪出指定大小的区域。  
    如果 mask 为空，则使用 image 的中心进行裁剪。  
    返回的 image 和 mask 都被裁剪为 roi_size 大小。
    """
    def _crop_roi(sample):
        image = sample["image"]  # 期望形状 [C, D, H, W]
        mask = sample["mask"]    # 期望形状 [C, D, H, W]
        mask_np = mask.cpu().numpy()[0]  # shape: [D, H, W]
        if np.count_nonzero(mask_np) == 0:
            center = [s // 2 for s in image.shape[1:]]
        else:
            nonzero = np.nonzero(mask_np)
            center = [int(np.mean(nonzero[i])) for i in range(3)]
        crop_slices = []
        for i, c in enumerate(center):
            size = roi_size[i]
            start = max(c - size // 2, 0)
            end = start + size
            if end > image.shape[i+1]:
                end = image.shape[i+1]
                start = end - size
            crop_slices.append(slice(start, end))
        image_cropped = image[:, crop_slices[0], crop_slices[1], crop_slices[2]]
        mask_cropped = mask[:, crop_slices[0], crop_slices[1], crop_slices[2]]
        sample["image"] = image_cropped
        sample["mask"] = mask_cropped
        return sample
    return Lambda(_crop_roi)

# -------------------- 3) 定义 transforms --------------------
WW, WL = 1500, -750
a_min = WL - WW / 2  # -1600
a_max = WL + WW / 2  # 400

preprocess_transforms = Compose([
    LoadImaged(keys=["image", "mask"], ensure_channel_first=True),
    Spacingd(
        keys=["image", "mask"],
        pixdim=(1.0, 1.0, 1.0),
        mode=("bilinear", "nearest")
    ),
    ScaleIntensityRanged(
        keys=["image"],
        a_min=a_min, a_max=a_max,
        b_min=0.0, b_max=1.0,
        clip=True
    ),
    CropForegroundd(keys=["image", "mask"], source_key="image", select_fn=lambda x: x > 0.5, margin=5),
    EnsureTyped(keys=["image", "mask"]),
    CropROICenterd(roi_size=(96, 96, 96)),
    RandFlipd(keys=["image", "mask"], spatial_axis=[0, 1, 2], prob=0.2),
    RandRotate90d(keys=["image", "mask"], prob=0.2, max_k=3),
    RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.4),
    ToTensord(keys=["image", "mask"]),
    # 根据 CSV 中的 LN 与 HGP 生成标签，其中 head_a: 0 if LN==0 else 1；head_b: HGP
    Lambda(lambda d: {**d, 
                        "head_a": torch.tensor(0.0 if float(d["LN"])==0 else 1.0), 
                        "head_b": torch.tensor(float(d["HGP"]))}),
    Lambda(cat_image_mask),
])

# -------------------- 4) 自定义数据集类 --------------------


from monai.data import Dataset  # monai.data.Dataset 是对 list 的简单封装
import os
import torch
import pandas as pd
import logging

class ResampledDataset(Dataset):
    """
    自定义数据集，用于加载 resampled 文件对或预处理后的数据。
    
    参数:
      - root_path: 如果 processed=False，则表示存放原始 resampled 文件的目录；
                   如果 processed=True，则表示存放预处理后 .pt 文件的目录。
      - label_csv: 当 processed=False 时使用，CSV 文件需包含 PatientID, LN, HGP 列；processed=True 时可以忽略。
      - transforms: 原始数据的预处理流程，只有在 processed=False 时使用。
      - processed: 是否使用预处理后的数据，默认为 False。
    """
    def __init__(self, root_path, label_csv=None, transforms=None, processed=False):
        self.processed = processed
        self.transforms = transforms
        if self.processed:
            # 直接加载预处理后的 .pt 文件
            self.data = sorted([os.path.join(root_path, f) for f in os.listdir(root_path) if f.endswith(".pt")])
            logging.info(f"Loaded {len(self.data)} processed samples from {root_path}.")
        else:
            # 使用 CSV 文件和原始文件路径构建数据列表
            self.data = []
            df = pd.read_csv(label_csv, encoding="utf-8-sig", sep=",")
            df.columns = df.columns.str.strip()
            print("CSV Columns:", df.columns.tolist())
            for _, row in df.iterrows():
                patient_id = str(row["PatientID"]).zfill(3)
                image_path = os.path.join(root_path, f"resampled_{patient_id}.nii.gz")
                mask_path = os.path.join(root_path, f"resampled_{patient_id}label.nii.gz")
                if os.path.exists(image_path) and os.path.exists(mask_path):
                    item = {
                        "image": image_path,
                        "mask": mask_path,
                        "LN": row["LN"],
                        "HGP": row["HGP"],
                        "PatientID": patient_id,
                    }
                    self.data.append(item)
                else:
                    logging.warning(f"缺失文件，患者 {patient_id} 的 image 或 mask 不存在。")
            logging.info(f"共找到 {len(self.data)} 个完整的患者数据。")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if self.processed:
            # 加载预处理好的样本
            sample = torch.load(self.data[index])
            return sample
        else:
            # 读取原始数据并执行 transforms
            item = self.data[index]
            if self.transforms:
                item = self.transforms(item)
            return item



# -------------------- 5) 主函数 --------------------
def main():
    # 数据文件夹路径（存放 resampled 文件）
    root_path = "/root/autodl-tmp/ConvertedV2-250404"
    output_root = "/root/autodl-tmp/preprocessed_data_resampled"
    os.makedirs(output_root, exist_ok=True)

    # 标签 CSV 文件路径
    label_csv = "/root/autodl-tmp/LNMetastasis250402V2.csv"
    
    dataset = ResampledDataset(
        root_path=root_path,
        label_csv=label_csv,
        transforms=preprocess_transforms
    )
    
    logging.info(f"数据集中共有 {len(dataset)} 个样本。")
    
    for idx in tqdm(range(len(dataset)), desc="Processing samples"):
        sample = dataset[idx]
        if sample is None:
            logging.warning(f"样本 {idx} 为空，跳过。")
            continue
        out_path = os.path.join(output_root, f"sample_{sample['PatientID']}.pt")
        torch.save(sample, out_path)
        logging.info(f"保存预处理样本 {sample['PatientID']} 到 {out_path}")

if __name__ == "__main__":
    main()
