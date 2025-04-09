import os
import torch
import torchio as tio
import nibabel as nib
import numpy as np
from tqdm import tqdm
import gc

# 配置路径和参数
root_path = '/root/autodl-tmp/ConvertedV2-250404'
target_depth = 477  # z 轴目标深度
# 重采样目标体素间距为 1mm x 1mm x 1mm
resample_transform = tio.Resample(target=(1, 1, 1))

def resample_image(input_path, output_path, target_depth=477, is_mask=False):
    """
    对输入的 nii.gz 文件进行重采样，并调整 z 方向的深度为 target_depth，
    最后保存为新的文件。
    
    参数:
        input_path: 输入文件路径
        output_path: 输出文件路径
        target_depth: 调整后的 z 轴深度
        is_mask: 如果为 True，表示处理 mask 文件（使用最近邻插值）
    """
    try:
        # 加载图像数据
        nii = nib.load(input_path)
        data = nii.get_fdata(dtype=np.float32)
        # 增加通道维度，变为 4D 张量 [C, X, Y, Z]，此处 C=1
        tensor = torch.from_numpy(data).unsqueeze(0)
        
        # 使用 torchio 创建 ScalarImage 对象
        image = tio.ScalarImage(tensor=tensor, affine=nii.affine)
        
        # 针对 mask 使用最近邻插值，其他图像使用默认线性插值
        if is_mask:
            resample = tio.Resample(target=(1, 1, 1), image_interpolation='nearest')
        else:
            resample = resample_transform
        
        # 重采样图像
        resampled = resample(image)
        
        # 调整 z 轴深度（x、y 保持不变）
        new_shape = (resampled.shape[1], resampled.shape[2], target_depth)
        crop_or_pad = tio.CropOrPad(new_shape)
        final_image = crop_or_pad(resampled)
        
        # 保存处理后的图像
        final_image.save(output_path)
        print(f"Saved: {output_path}")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
    finally:
        # 手动释放内存
        for var in ['nii', 'data', 'tensor', 'image', 'resampled', 'crop_or_pad', 'final_image']:
            if var in locals():
                del locals()[var]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    # 获取目录下所有文件列表
    files = os.listdir(root_path)

    # 筛选图像文件和 mask 文件
    # 图像文件：文件名为纯数字 + ".nii.gz"
    image_files = [f for f in files if f.endswith('.nii.gz') and ('label' not in f) and (not f.startswith('resampled_'))]
    # mask 文件：文件名包含 "label" 且以 "label.nii.gz" 结尾
    mask_files = [f for f in files if f.endswith('label.nii.gz') and (not f.startswith('resampled_'))]

    # 构建字典：患者ID -> 文件名  
    # 注意：对于图像文件，患者ID为文件名去掉后缀 ".nii.gz"（例如 "123.nii.gz" -> "123"）  
    #       对于 mask 文件，患者ID为去掉 "label.nii.gz"（例如 "123label.nii.gz" -> "123"）
    image_dict = {}
    for f in image_files:
        patient_id = f[:-7]  # 去掉 ".nii.gz"
        image_dict[patient_id] = f

    mask_dict = {}
    for f in mask_files:
        patient_id = f[:-12]  # 去掉 "label.nii.gz"
        mask_dict[patient_id] = f

    # 遍历每个患者，处理图像和对应的 mask
    for patient_id in tqdm(image_dict, desc="Processing patients"):
        # 处理图像文件
        image_file = os.path.join(root_path, image_dict[patient_id])
        output_image_file = os.path.join(root_path, "resampled_" + image_dict[patient_id])
        resample_image(image_file, output_image_file, target_depth=target_depth, is_mask=False)
        
        # 如果存在对应的 mask 文件，则处理 mask
        if patient_id in mask_dict:
            mask_file = os.path.join(root_path, mask_dict[patient_id])
            output_mask_file = os.path.join(root_path, "resampled_" + mask_dict[patient_id])
            resample_image(mask_file, output_mask_file, target_depth=target_depth, is_mask=True)
        else:
            print(f"未找到患者 {patient_id} 对应的 mask 文件。")

if __name__ == '__main__':
    main()
