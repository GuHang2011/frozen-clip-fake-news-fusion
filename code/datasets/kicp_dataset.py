# datasets/kicp_dataset.py
# -*- coding: utf-8 -*-

"""
作用：
    定义 KICP 项目的 PyTorch Dataset 类。

读取的数据格式：
    id,text,image_path,label

例如：
    data/processed/politifact.csv
    data/processed/gossipcop.csv

返回内容：
    {
        "id": 样本ID,
        "text": 新闻标题或正文,
        "image": PIL.Image 图片对象,
        "label": 0 或 1
    }

标签规则：
    0 = real
    1 = fake / rumor

说明：
    你当前还没有真实图片，所以 image_path 统一是 data/assets/blank_336.png。
    如果图片不存在，本 Dataset 会自动生成一张 336x336 白色图片。
"""

from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class KICPDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        image_size: int = 336,
        max_samples: Optional[int] = None,
    ):
        self.csv_path = Path(csv_path)
        self.image_size = image_size

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"找不到数据文件：{self.csv_path}\n"
                f"请确认文件是否存在，例如：data/processed/politifact.csv"
            )

        self.df = pd.read_csv(self.csv_path, encoding="utf-8-sig")

        required_columns = ["id", "text", "image_path", "label"]
        missing_columns = [col for col in required_columns if col not in self.df.columns]

        if missing_columns:
            raise ValueError(
                f"数据文件缺少必要列：{missing_columns}\n"
                f"当前列名是：{list(self.df.columns)}\n"
                f"正确格式必须是：id,text,image_path,label"
            )

        self.df = self.df[required_columns].copy()

        before_count = len(self.df)

        self.df = self.df.dropna(subset=["text"])
        self.df = self.df[self.df["text"].astype(str).str.strip() != ""]

        after_count = len(self.df)

        if before_count != after_count:
            print(f"提示：删除空文本样本 {before_count - after_count} 条。")

        self.df["label"] = self.df["label"].astype(int)

        unique_labels = sorted(self.df["label"].unique().tolist())
        if not set(unique_labels).issubset({0, 1}):
            raise ValueError(
                f"标签只能是 0 或 1，但当前发现：{unique_labels}\n"
                f"请检查 label 列。"
            )

        if max_samples is not None:
            self.df = self.df.head(max_samples).copy()

        self.samples = self.df.to_dict("records")

        print("=============== Dataset 初始化完成 ===============")
        print(f"数据文件：{self.csv_path}")
        print(f"样本数：{len(self.samples)}")
        print("标签统计：")
        print(self.df["label"].value_counts().sort_index())
        print("=================================================")

    def __len__(self) -> int:
        return len(self.samples)

    def _create_blank_image(self) -> Image.Image:
        return Image.new(
            mode="RGB",
            size=(self.image_size, self.image_size),
            color=(255, 255, 255),
        )

    def _load_image(self, image_path: str) -> Image.Image:
        if image_path is None:
            return self._create_blank_image()

        image_path = str(image_path).strip()

        if image_path == "":
            return self._create_blank_image()

        path = Path(image_path)

        if not path.exists():
            return self._create_blank_image()

        try:
            image = Image.open(path).convert("RGB")
            image = image.resize((self.image_size, self.image_size))
            return image
        except Exception:
            return self._create_blank_image()

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]

        sample_id = str(sample["id"])
        text = str(sample["text"])
        image_path = str(sample["image_path"])
        label = int(sample["label"])

        image = self._load_image(image_path)

        return {
            "id": sample_id,
            "text": text,
            "image": image,
            "label": label,
        }