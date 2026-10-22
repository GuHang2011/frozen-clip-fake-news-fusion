# scripts/prepare_splits.py
# Build real (image-available) multimodal splits for faithful KICP reproduction.
#
# Decisions (see report / paper revision plan):
#   * Keep only samples that have a REAL image on disk (exclude the shared
#     data/assets/blank_336.png placeholder). This fixes the "blank image"
#     artefact of the original V2 run and makes the multimodal claim honest.
#   * PolitiFact: use all image-available samples (small).
#   * GossipCop: stratified-subsample to GOSSIP_TARGET (=6805) to match the
#     scale reported in the paper and keep the cached CLIP feature bank small
#     enough to fit in RAM. Documented as a random stratified subsample.
#   * 70/15/15 stratified train/val/test split, fixed seed. We report TEST
#     metrics (best epoch chosen on VAL) -> removes the val==test optimism of
#     the original run.
#
# Output: data/splits_real/<dataset>/{train,val,test}.csv  +  stats.json
# Run from the project root (KICP_project). ASCII only.

import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
GOSSIP_TARGET = 6805
VAL_RATIO = 0.15
TEST_RATIO = 0.15
OUT_ROOT = os.path.join("data", "splits_real")

DATASETS = [
    {"name": "politifact", "csv": os.path.join("data", "processed", "politifact_full.csv"), "target": None},
    {"name": "gossipcop", "csv": os.path.join("data", "processed", "gossipcop_full.csv"), "target": GOSSIP_TARGET},
]


def is_real_image(x):
    if not isinstance(x, str):
        return False
    x = x.strip()
    if not x:
        return False
    low = x.lower().replace("\\", "/")
    if "assets" in low or "blank" in os.path.basename(low):
        return False
    return os.path.exists(x)


def stratified_subsample(df, target, seed):
    if target is None or len(df) <= target:
        return df
    frac = target / len(df)
    parts = []
    for label, g in df.groupby("label"):
        n = int(round(len(g) * frac))
        n = max(1, min(n, len(g)))
        parts.append(g.sample(n=n, random_state=seed))
    out = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def main():
    summary = {}
    for ds in DATASETS:
        name, csv_path, target = ds["name"], ds["csv"], ds["target"]
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        df = df.dropna(subset=["text"]).copy()
        df["text"] = df["text"].astype(str)
        df = df[df["text"].str.strip() != ""]
        df = df[df["label"].isin([0, 1])].copy()
        df["label"] = df["label"].astype(int)

        df = df[df["image_path"].apply(is_real_image)].copy()
        n_avail = len(df)

        df = stratified_subsample(df, target, SEED)
        df = df[["id", "text", "image_path", "label"]].reset_index(drop=True)
        n_used = len(df)

        # 70/15/15 stratified
        train_df, tmp_df = train_test_split(
            df, test_size=VAL_RATIO + TEST_RATIO, random_state=SEED, stratify=df["label"]
        )
        rel = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
        val_df, test_df = train_test_split(
            tmp_df, test_size=rel, random_state=SEED, stratify=tmp_df["label"]
        )

        out_dir = os.path.join(OUT_ROOT, name)
        os.makedirs(out_dir, exist_ok=True)
        for split_name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
            sdf.to_csv(os.path.join(out_dir, f"{split_name}.csv"), index=False, encoding="utf-8-sig")

        avg_words = df["text"].str.split().apply(len).mean()
        stats = {
            "dataset": name,
            "image_available_total": int(n_avail),
            "used_total": int(n_used),
            "fake": int((df["label"] == 1).sum()),
            "real": int((df["label"] == 0).sum()),
            "avg_words": float(round(avg_words, 1)),
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
            "train_fake": int((train_df["label"] == 1).sum()),
            "val_fake": int((val_df["label"] == 1).sum()),
            "test_fake": int((test_df["label"] == 1).sum()),
            "seed": SEED,
            "subsampled_to": target,
        }
        with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        summary[name] = stats
        print("=" * 60)
        for k, v in stats.items():
            print(f"  {k}: {v}")

    with open(os.path.join(OUT_ROOT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\nsaved splits under", OUT_ROOT)


if __name__ == "__main__":
    main()
