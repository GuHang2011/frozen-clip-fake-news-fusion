# datasets/kicp_collator.py
# -*- coding: utf-8 -*-

"""
作用：
    将 KICPDataset 返回的一批样本整理成 CLIP 可以直接使用的 batch。

Dataset 单条样本格式：
    {
        "id": str,
        "text": str,
        "image": PIL.Image.Image,
        "label": int
    }

Collator 输出 batch 格式：
    {
        "input_ids": Tensor,
        "attention_mask": Tensor,
        "pixel_values": Tensor,
        "labels": Tensor,
        "ids": list[str],
        "texts": list[str]
    }

说明：
    这里使用 transformers 的 CLIPProcessor。
    它会自动完成：
        1. 文本分词；
        2. 文本 padding；
        3. 文本 truncation；
        4. 图片 resize；
        5. 图片 normalize；
        6. 转成 PyTorch Tensor。
"""

from typing import List, Dict, Any

import torch
from transformers import CLIPProcessor


class KICPCollator:
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-large-patch14-336",
        max_length: int = 77,
    ):
        """
        参数：
            clip_model_name:
                CLIP 模型名称。
                KICP 论文中使用的是 CLIP ViT-L/14@336px，
                对应 Hugging Face 名称：
                openai/clip-vit-large-patch14-336

            max_length:
                CLIP 文本最大长度。
                CLIP 默认通常是 77。
        """
        self.clip_model_name = clip_model_name
        self.max_length = max_length

        print("正在加载 CLIPProcessor...")
        print(f"模型名称：{self.clip_model_name}")

        self.processor = CLIPProcessor.from_pretrained(self.clip_model_name)

        print("CLIPProcessor 加载完成。")

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        DataLoader 每次取出一个 batch 后，会自动调用这个函数。
        """
        ids = [item["id"] for item in batch]
        texts = [item["text"] for item in batch]
        images = [item["image"] for item in batch]
        labels = [item["label"] for item in batch]

        encoded = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        labels = torch.tensor(labels, dtype=torch.long)

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "pixel_values": encoded["pixel_values"],
            "labels": labels,
            "ids": ids,
            "texts": texts,
        }