# scripts/precompute_clip.py
# Precompute frozen CLIP ViT-L/14@336 features once and cache to disk, so every
# frozen-backbone head-training run can reuse them instead of paying the CLIP
# forward cost again.
#
# We cache the RAW frozen outputs and leave the (trainable) image patch
# projection 1024->768 to the head, exactly mirroring models/kicp_model.py:
#   * text_seq    : text_model.last_hidden_state            fp16[N, 77, 768]
#   * image_seq   : vision_model.last_hidden_state (RAW)    fp16[N, 577, 1024]
#   * text_pooled : normalize(text_projection(pooler))      fp16[N, 768]
#   * image_pooled: normalize(visual_projection(pooler))    fp16[N, 768]
#   * attn_mask   : 1 = real token, 0 = pad                 long[N, 77]
#   * labels, ids
#
# Run from the project root (KICP_project). ASCII only.
#   python scripts/precompute_clip.py --dataset all --device cuda

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

CLIP_NAME = "openai/clip-vit-large-patch14-336"
MAXLEN = 77
SPLIT_ROOT = os.path.join("data", "splits_real")
CACHE_DIR = os.path.join("cache", "clip")


def load_image(path, size=336):
    try:
        return Image.open(path).convert("RGB").resize((size, size))
    except Exception:
        return Image.new("RGB", (size, size), (255, 255, 255))


@torch.no_grad()
def encode_split(clip, processor, df, device, batch_size):
    n = len(df)
    texts = df["text"].astype(str).tolist()
    paths = df["image_path"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()
    ids = df["id"].astype(str).tolist()

    text_seq = np.zeros((n, MAXLEN, 768), dtype=np.float16)
    image_seq = np.zeros((n, 577, 1024), dtype=np.float16)
    text_pooled = np.zeros((n, 768), dtype=np.float16)
    image_pooled = np.zeros((n, 768), dtype=np.float16)
    attn = np.zeros((n, MAXLEN), dtype=np.int64)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        imgs = [load_image(p) for p in paths[start:end]]
        enc = processor(
            text=texts[start:end], images=imgs, return_tensors="pt",
            padding="max_length", truncation=True, max_length=MAXLEN,
        )
        ids_t = enc["input_ids"].to(device)
        attn_t = enc["attention_mask"].to(device)
        pix_t = enc["pixel_values"].to(device)

        tout = clip.text_model(input_ids=ids_t, attention_mask=attn_t)
        vout = clip.vision_model(pixel_values=pix_t)
        tp = F.normalize(clip.text_projection(tout.pooler_output), dim=-1)
        ip = F.normalize(clip.visual_projection(vout.pooler_output), dim=-1)

        text_seq[start:end] = tout.last_hidden_state.float().cpu().numpy().astype(np.float16)
        image_seq[start:end] = vout.last_hidden_state.float().cpu().numpy().astype(np.float16)
        text_pooled[start:end] = tp.float().cpu().numpy().astype(np.float16)
        image_pooled[start:end] = ip.float().cpu().numpy().astype(np.float16)
        attn[start:end] = attn_t.cpu().numpy()
        if (start // batch_size) % 25 == 0:
            print(f"    {end}/{n}", flush=True)

    return {
        "ids": ids,
        "labels": torch.tensor(labels, dtype=torch.long),
        "text_seq": torch.from_numpy(text_seq),
        "image_seq": torch.from_numpy(image_seq),
        "text_pooled": torch.from_numpy(text_pooled),
        "image_pooled": torch.from_numpy(image_pooled),
        "attn_mask": torch.from_numpy(attn),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=["all", "politifact", "gossipcop"])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print("device:", device)

    os.makedirs(CACHE_DIR, exist_ok=True)
    processor = CLIPProcessor.from_pretrained(CLIP_NAME)
    clip = CLIPModel.from_pretrained(CLIP_NAME).eval().to(device)
    for p in clip.parameters():
        p.requires_grad_(False)

    datasets = ["politifact", "gossipcop"] if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        for split in ["train", "val", "test"]:
            df = pd.read_csv(os.path.join(SPLIT_ROOT, ds, f"{split}.csv"), encoding="utf-8-sig")
            print(f"[{ds}/{split}] N={len(df)} -> encoding", flush=True)
            cache = encode_split(clip, processor, df, device, args.batch_size)
            out_path = os.path.join(CACHE_DIR, f"{ds}_{split}.pt")
            torch.save(cache, out_path)
            print(f"  saved {out_path}  ({os.path.getsize(out_path)/1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
