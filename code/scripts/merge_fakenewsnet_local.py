# scripts/merge_fakenewsnet_local.py
# -*- coding: utf-8 -*-

"""
作用：
    读取你已经下载好的 FakeNewsNet 四个 CSV 文件：

        data/raw/politifact/politifact_fake.csv
        data/raw/politifact/politifact_real.csv
        data/raw/gossipcop/gossipcop_fake.csv
        data/raw/gossipcop/gossipcop_real.csv

    合并并整理成 KICP 训练需要的格式：

        id,text,image_path,label

标签规则：
    0 = real
    1 = fake / rumor

说明：
    当前阶段先不下载真实图片。
    image_path 统一写成 data/assets/blank_336.png。
    这个文件暂时可以不存在，后面写 Dataset 类时会自动处理缺失图片。
"""

from pathlib import Path
import sys

import pandas as pd


RAW_FILES = {
    "politifact": {
        "fake": "data/raw/politifact/politifact_fake.csv",
        "real": "data/raw/politifact/politifact_real.csv",
    },
    "gossipcop": {
        "fake": "data/raw/gossipcop/gossipcop_fake.csv",
        "real": "data/raw/gossipcop/gossipcop_real.csv",
    },
}


LABEL_MAP = {
    "real": 0,
    "fake": 1,
}


def get_blank_image_path():
    """
    当前阶段只返回占位图路径，不真正创建图片文件。
    后面 Dataset 类会统一处理图片不存在的情况。
    """
    assets_dir = Path("data/assets")
    assets_dir.mkdir(parents=True, exist_ok=True)

    blank_path = assets_dir / "blank_336.png"
    return str(blank_path)


def read_csv_with_columns(path):
    """
    读取 CSV，并打印列名，方便检查原始数据结构。
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"找不到文件：{path}\n"
            f"请确认你当前运行目录是 D:\\KICP_project，并且 CSV 文件已经放到 data/raw 对应目录。"
        )

    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1")

    print("\n======================================")
    print(f"正在读取：{path}")
    print(f"样本数：{len(df)}")
    print("列名如下：")
    for i, col in enumerate(df.columns):
        print(f"  {i}: {col}")
    print("前 2 行预览：")
    print(df.head(2))
    print("======================================\n")

    return df


def find_text_column(df):
    """
    FakeNewsNet minimal CSV 一般有 title。
    如果没有 title，就尝试其他常见文本列。
    """
    candidates = [
        "title",
        "text",
        "content",
        "news",
        "body",
        "statement",
    ]

    lower_map = {str(col).lower(): col for col in df.columns}

    for name in candidates:
        if name in lower_map:
            return lower_map[name]

    raise ValueError(
        "没有找到文本列。\n"
        "请检查 CSV 列名。常见文本列应该是 title 或 text。\n"
        f"当前列名是：{list(df.columns)}"
    )


def find_id_column(df):
    """
    FakeNewsNet minimal CSV 一般有 id。
    如果没有，就自动生成。
    """
    candidates = [
        "id",
        "news_id",
        "post_id",
        "tweet_id",
    ]

    lower_map = {str(col).lower(): col for col in df.columns}

    for name in candidates:
        if name in lower_map:
            return lower_map[name]

    return None


def merge_one_dataset(dataset_name, blank_image_path):
    """
    合并一个数据集的 fake 和 real 文件。
    """
    all_rows = []

    for label_name in ["fake", "real"]:
        csv_path = RAW_FILES[dataset_name][label_name]
        df = read_csv_with_columns(csv_path)

        text_col = find_text_column(df)
        id_col = find_id_column(df)

        label = LABEL_MAP[label_name]

        print(f"当前处理：{dataset_name}_{label_name}")
        print(f"使用文本列：{text_col}")
        print(f"使用 ID 列：{id_col if id_col is not None else '自动生成'}")
        print(f"标签：{label}")
        print()

        skipped_empty_text = 0

        for idx, row in df.iterrows():
            if id_col is None:
                sample_id = f"{dataset_name}_{label_name}_{idx}"
            else:
                sample_id = str(row[id_col])

            text = row[text_col]

            if pd.isna(text) or str(text).strip() == "":
                skipped_empty_text += 1
                continue

            all_rows.append(
                {
                    "id": sample_id,
                    "text": str(text).replace("\n", " ").replace("\r", " ").strip(),
                    "image_path": blank_image_path,
                    "label": label,
                }
            )

        if skipped_empty_text > 0:
            print(f"{dataset_name}_{label_name} 跳过空文本样本：{skipped_empty_text}")

    out_df = pd.DataFrame(
        all_rows,
        columns=["id", "text", "image_path", "label"]
    )

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{dataset_name}.csv"
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n=============== 合并完成 ===============")
    print(f"数据集：{dataset_name}")
    print(f"输出文件：{output_path}")
    print(f"有效样本数：{len(out_df)}")
    print()
    print("标签统计：")
    print(out_df["label"].value_counts().sort_index())
    print()
    print("输出列名：")
    print(list(out_df.columns))
    print()
    print("前 5 行：")
    print(out_df.head())
    print("======================================\n")


def main():
    print("当前工作目录：")
    print(Path(".").resolve())
    print()

    blank_image_path = get_blank_image_path()
    print(f"当前阶段统一使用占位图片路径：{blank_image_path}")
    print("注意：这个图片文件现在可以不存在，后面 Dataset 类会处理。")
    print()

    merge_one_dataset("politifact", blank_image_path)
    merge_one_dataset("gossipcop", blank_image_path)

    print("全部处理完成。")
    print("你现在应该看到：")
    print("  data/processed/politifact.csv")
    print("  data/processed/gossipcop.csv")
    print()
    print("这两个文件就是下一步 Dataset 类要读取的数据。")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n程序运行失败：")
        print(e)
        sys.exit(1)