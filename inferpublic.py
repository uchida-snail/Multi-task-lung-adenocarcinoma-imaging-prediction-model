import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import logging
import importlib

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

###################################################
# 一、模型配置管理
###################################################
# 确保每个模型对应的模块和类名称正确，
# 此处的 module_name 表示模型定义所在的 .py 文件（不含后缀），
# class_name 表示文件内的模型类名称。
model_config_dict = {
    # 示例：EfficientNetSwin
    "EfficientNetSwin": {
        "module_name": "modelEfficientNetSwin",
        "class_name": "FusionModel",
        "ckpt_path": "/root/autodl-tmp/EfficientNetSwin2/best_model_phaseA.pth",
        "init_kwargs": {}
    },
}

###################################################
# 二、定义 collate_fn（假设所有模型使用相同数据预处理）
###################################################
def my_collate_fn(batch):
    filtered = []
    for i, sample in enumerate(batch):
        if sample is None:
            logging.warning(f"[collate_fn] Sample index={i} is None, skipping.")
            continue
        # 双任务头要求包含 "image_cropped", "head_a" 和 "head_b"
        if not ("image_cropped" in sample and "head_a" in sample and "head_b" in sample):
            logging.warning(f"[collate_fn] Sample index={i} missing required keys, skipping.")
            continue
        filtered.append(sample)
    if len(filtered) == 0:
        logging.warning("[collate_fn] All samples in this batch are invalid. Returning None.")
        return None
    ref_shape = filtered[0]["image_cropped"].shape
    for i, s in enumerate(filtered):
        if s["image_cropped"].shape != ref_shape:
            logging.error(f"[collate_fn] Shape mismatch: {s['image_cropped'].shape} vs {ref_shape}. Skipping entire batch.")
            return None

    collated = {}
    for key in filtered[0].keys():
        values = [s[key] for s in filtered]
        # 如果第一个元素是 Tensor，则对所有值做 stack
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values, dim=0)
        # 如果是 numpy array，则转换为 Tensor 后 stack
        elif isinstance(values[0], np.ndarray):
            collated[key] = torch.stack([torch.tensor(v) for v in values], dim=0)
        else:
            # 对于其他类型（比如 int, str 等），直接收集成 list
            collated[key] = values
    return collated

###################################################
# 三、推理函数 (双任务头)
###################################################
def get_probs_labels(loader, device, model, task_head='a', use_amp=False):
    """
    对单个 DataLoader 进行推理，并返回预测概率和标签。
    task_head: 'a' 或 'b'
    """
    all_probs = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            images = batch["image_cropped"].to(device)
            labels = batch[f"head_{task_head}"].to(device)
            with autocast(enabled=use_amp):
                # 假设模型的 forward 返回 (out_a, out_b, extra)
                out_a, out_b, _ = model(images)
                out_sel = out_a if task_head == 'a' else out_b
            probs = torch.sigmoid(out_sel).view(-1).cpu().numpy()
            labels = labels.view(-1).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels)
    return np.array(all_probs), np.array(all_labels)

###################################################
# 四、主程序
###################################################
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # 【修改点1】更新了默认数据集路径，使用预处理后的 resampled 数据
    parser.add_argument("--root_path", type=str, default="/root/autodl-tmp/preprocessed_data_resampled",
                        help="数据集根目录")
    # 修改默认模型列表，仅包含配置表中存在的模型名称
    parser.add_argument("--model_list", type=str, nargs="+",
                        default=["EfficientNetSwin"],
                        help="待推理模型列表，对应 model_config_dict 的 key")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "internal", "test"],
                        help="待推理数据集分割")
    parser.add_argument("--task_heads", type=str, nargs="+", default=["LN:a", "HGP:b"],
                        help="任务名称:头标识，例如 LN:a 表示用 head_a 推理，HGP:b 表示用 head_b 推理")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader 工作进程数")
    parser.add_argument("--use_amp", action="store_true", help="是否使用混合精度推理")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 【修改点2】导入自定义的 ResampledDataset 数据集，而不是 LNClassificationDataset
    from resampledpublic import ResampledDataset

    # 遍历每个待推理的模型
    for model_name in args.model_list:
        if model_name not in model_config_dict:
            continue

        cfg = model_config_dict[model_name]
        module_name = cfg["module_name"]
        class_name = cfg["class_name"]
        ckpt_path = cfg["ckpt_path"]
        init_kwargs = cfg["init_kwargs"]

        if not os.path.exists(ckpt_path):
            logging.info(f"权重文件 {ckpt_path} 不存在，跳过模型 {model_name}.")
            continue

        try:
            model_module = importlib.import_module(module_name)
            ModelClass = getattr(model_module, class_name)
        except Exception as e:
            logging.info(f"导入模块 {module_name} 或类 {class_name} 失败: {e}")
            continue

        # 实例化模型并加载权重
        model = ModelClass(**init_kwargs).to(device)
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logging.info(f"已加载模型 {model_name} 的权重: {ckpt_path}")

        # 针对每个数据集 split 进行推理
        for split in args.splits:
            # 【修改点3】使用 ResampledDataset 数据集，并传入 CSV 文件路径（请根据实际情况修改 CSV 路径）
            dataset = ResampledDataset(root_path=args.root_path,
                                       label_csv="/root/autodl-tmp/LNMetastasis250402V2.csv",
                                       transforms=None,
                                       processed=True)
            loader = DataLoader(dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=args.num_workers,
                                collate_fn=my_collate_fn,
                                pin_memory=True)

            # 针对每个任务头进行推理
            for task_item in args.task_heads:
                task_name, head_label = task_item.split(":")
                probs, labels = get_probs_labels(loader=loader,
                                                 device=device,
                                                 model=model,
                                                 task_head=head_label,
                                                 use_amp=args.use_amp)
                save_filename = f"{split}_predictions_{model_name}_{task_name}.csv"
                df = pd.DataFrame({"probability": probs, "label": labels})
                df.to_csv(save_filename, index=False)
                logging.info(f"模型 {model_name} 在 {split} 集上任务 {task_name} 的推理结果已保存至 {save_filename}")
