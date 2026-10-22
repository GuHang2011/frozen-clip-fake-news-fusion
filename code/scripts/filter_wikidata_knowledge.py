# scripts/filter_wikidata_knowledge.py
# -*- coding: utf-8 -*-

"""
过滤 Wikidata 原始知识库，去掉明显噪声实体，并按 QID 去重。

输入：
    knowledge/wikidata_knowledge_politifact.csv

输出：
    knowledge/wikidata_knowledge_politifact_filtered.csv

输出字段：
    knowledge_id,qid,label,description,knowledge_text,entity_texts,source_sample_ids,n_sources

说明：
    1. 只保留 qid 以 Q 开头的有效 Wikidata 实体；
    2. 去掉 natural number / given name / disambiguation page 等明显噪声；
    3. 按 qid 去重，保留唯一知识项；
    4. 后续 CLIP 编码脚本会读取 filtered.csv。
"""

import argparse
import re
from pathlib import Path

import pandas as pd


BAD_DESCRIPTION_KEYWORDS = [
    "natural number",
    "integer",
    "male given name",
    "female given name",
    "given name",
    "family name",
    "surname",
    "Wikimedia disambiguation page",
    "Wikimedia category",
    "category page",
    "template page",
    "list article",
    "scientific article",
]


BAD_LABELS = {
    "III",
    "II",
    "IV",
    "V",
    "VI",
    "VII",
    "VIII",
    "IX",
    "X",
    "NOW",
    "RUNE",
}


def clean_text(value) -> str:
    if value is None:
        return ""

    value = str(value)

    if value.lower() == "nan":
        return ""

    value = value.replace("\u00a0", " ")
    value = re.sub(r"\s+", " ", value)
    value = value.strip()

    return value


def valid_qid(value) -> bool:
    value = clean_text(value)
    return bool(re.fullmatch(r"Q\d+", value))


def is_bad_knowledge(row, min_knowledge_len: int) -> bool:
    qid = clean_text(row.get("qid", ""))
    label = clean_text(row.get("label", ""))
    description = clean_text(row.get("description", ""))
    knowledge_text = clean_text(row.get("knowledge_text", ""))
    entity_text = clean_text(row.get("entity_text", ""))

    if not valid_qid(qid):
        return True

    if label == "":
        return True

    if len(knowledge_text) < min_knowledge_len:
        return True

    if label in BAD_LABELS or entity_text in BAD_LABELS:
        return True

    desc_lower = description.lower()

    for keyword in BAD_DESCRIPTION_KEYWORDS:
        if keyword.lower() in desc_lower:
            return True

    # 纯罗马数字、纯数字一般是噪声
    if re.fullmatch(r"[IVX]+", label):
        return True

    if re.fullmatch(r"\d+", label):
        return True

    return False


def join_unique(values) -> str:
    output = []

    for value in values:
        value = clean_text(value)

        if value and value not in output:
            output.append(value)

    return " || ".join(output)


def filter_knowledge(input_path: str, output_path: str, min_knowledge_len: int) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_path}")

    df = pd.read_csv(input_path, encoding="utf-8-sig")

    required_columns = [
        "source_sample_id",
        "entity_text",
        "qid",
        "label",
        "description",
        "knowledge_text",
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"缺少必要列：{col}，当前列名：{list(df.columns)}")

    before_count = len(df)

    df["_is_bad"] = df.apply(
        lambda row: is_bad_knowledge(row, min_knowledge_len=min_knowledge_len),
        axis=1,
    )

    filtered = df[~df["_is_bad"]].copy()

    # 按 qid 聚合去重
    grouped_rows = []

    for index, (qid, group) in enumerate(filtered.groupby("qid"), start=1):
        first = group.iloc[0]

        grouped_rows.append(
            {
                "knowledge_id": f"K{index:06d}",
                "qid": clean_text(qid),
                "label": clean_text(first["label"]),
                "description": clean_text(first["description"]),
                "knowledge_text": clean_text(first["knowledge_text"]),
                "entity_texts": join_unique(group["entity_text"].tolist()),
                "source_sample_ids": join_unique(group["source_sample_id"].tolist()),
                "n_sources": group["source_sample_id"].nunique(),
            }
        )

    out_df = pd.DataFrame(
        grouped_rows,
        columns=[
            "knowledge_id",
            "qid",
            "label",
            "description",
            "knowledge_text",
            "entity_texts",
            "source_sample_ids",
            "n_sources",
        ],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print()
    print("========================================")
    print("Wikidata 知识过滤完成")
    print("========================================")
    print(f"输入文件：{input_path}")
    print(f"输出文件：{output_path}")
    print(f"原始记录数：{before_count}")
    print(f"过滤后有效记录数：{len(filtered)}")
    print(f"按 QID 去重后知识数：{len(out_df)}")
    print()
    print("前 10 条过滤后知识：")
    if len(out_df) > 0:
        print(out_df[["knowledge_id", "qid", "label", "description"]].head(10))
    else:
        print("暂无有效知识")
    print("========================================")
    print()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入 Wikidata 原始知识 CSV。",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出过滤后的知识 CSV。",
    )

    parser.add_argument(
        "--min_knowledge_len",
        type=int,
        default=10,
        help="knowledge_text 最小长度。",
    )

    args = parser.parse_args()

    filter_knowledge(
        input_path=args.input,
        output_path=args.output,
        min_knowledge_len=args.min_knowledge_len,
    )


if __name__ == "__main__":
    main()
