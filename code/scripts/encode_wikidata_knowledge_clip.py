# scripts/encode_wikidata_knowledge_clip.py
# -*- coding: utf-8 -*-

"""
使用 CLIP 文本编码器对 Wikidata 知识文本进行编码。

输入：
    knowledge/wikidata_knowledge_politifact_filtered.csv

输出：
    knowledge/wikidata_knowledge_politifact_embeddings.pt

输出 pt 内容：
    {
        "embeddings": Tensor[N, D],
        "metadata": List[Dict],
        "clip_model_name": str,
        "text_column": str,
        "normalize": bool
    }

说明：
    1. 默认使用 openai/clip-vit-large-patch14-336；
    2. 输出的 embeddings 已经 L2 normalize，方便后续余弦相似度 Top-K 检索；
    3. 当前只编码知识文本，不训练模型。
"""

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


def clean_text(value) -> str:
    if value is None:
        return ""

    value = str(value)

    if value.lower() == "nan":
        return ""

    value = " ".join(value.split())
    return value.strip()


def load_knowledge_csv(csv_path: str, text_column: str) -> pd.DataFrame:
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"找不到知识 CSV：{csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    if text_column not in df.columns:
        raise ValueError(f"缺少文本列：{text_column}，当前列名：{list(df.columns)}")

    df[text_column] = df[text_column].apply(clean_text)
    df = df[df[text_column].str.len() > 0].copy()
    df = df.reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("知识 CSV 中没有可编码的 knowledge_text。")

    return df


def batch_iter(items: List[str], batch_size: int):
    for start in range(0, len(items), batch_size):
        end = start + batch_size
        yield start, end, items[start:end]


def encode_knowledge(
    input_csv: str,
    output_pt: str,
    clip_model_name: str,
    text_column: str,
    batch_size: int,
    device: str,
    normalize: bool,
) -> None:
    device_obj = torch.device(device)

    df = load_knowledge_csv(input_csv, text_column=text_column)

    print()
    print("========================================")
    print("开始使用 CLIP 编码 Wikidata 知识文本")
    print("========================================")
    print(f"输入 CSV：{input_csv}")
    print(f"输出 PT：{output_pt}")
    print(f"CLIP 模型：{clip_model_name}")
    print(f"知识数量：{len(df)}")
    print(f"batch_size：{batch_size}")
    print(f"device：{device}")
    print()

    processor = CLIPProcessor.from_pretrained(clip_model_name)
    model = CLIPModel.from_pretrained(clip_model_name)
    model.to(device_obj)
    model.eval()

    texts = df[text_column].tolist()
    all_embeddings = []

    with torch.no_grad():
        for _, _, batch_texts in tqdm(
            batch_iter(texts, batch_size=batch_size),
            total=(len(texts) + batch_size - 1) // batch_size,
            desc="Encoding",
        ):
            inputs = processor(
                text=batch_texts,
                images=None,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            )

            input_ids = inputs["input_ids"].to(device_obj)
            attention_mask = inputs["attention_mask"].to(device_obj)

            text_features = model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            if normalize:
                text_features = F.normalize(text_features, dim=-1)

            all_embeddings.append(text_features.cpu())

    embeddings = torch.cat(all_embeddings, dim=0)

    metadata = []

    for _, row in df.iterrows():
        item: Dict = {}

        for col in df.columns:
            item[col] = clean_text(row[col])

        metadata.append(item)

    output_pt = Path(output_pt)
    output_pt.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "embeddings": embeddings,
            "metadata": metadata,
            "clip_model_name": clip_model_name,
            "text_column": text_column,
            "normalize": normalize,
        },
        output_pt,
    )

    print()
    print("========================================")
    print("CLIP 知识编码完成")
    print("========================================")
    print(f"输出文件：{output_pt}")
    print(f"embeddings shape：{tuple(embeddings.shape)}")
    print(f"metadata 数量：{len(metadata)}")
    print("========================================")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="过滤后的 Wikidata 知识 CSV。",
    )

    parser.add_argument(
        "--output_pt",
        type=str,
        required=True,
        help="输出知识向量文件 .pt。",
    )

    parser.add_argument(
        "--clip_model_name",
        type=str,
        default="openai/clip-vit-large-patch14-336",
        help="CLIP 模型名称。",
    )

    parser.add_argument(
        "--text_column",
        type=str,
        default="knowledge_text",
        help="需要编码的文本列。",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="编码 batch size。CPU 建议 4 或 8。",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="运行设备。",
    )

    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="不做 L2 normalize。默认会 normalize。",
    )

    args = parser.parse_args()

    encode_knowledge(
        input_csv=args.input_csv,
        output_pt=args.output_pt,
        clip_model_name=args.clip_model_name,
        text_column=args.text_column,
        batch_size=args.batch_size,
        device=args.device,
        normalize=not args.no_normalize,
    )


if __name__ == "__main__":
    main()
