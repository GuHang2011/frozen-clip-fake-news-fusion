# scripts/train_head.py
# -*- coding: utf-8 -*-
#
# Train the KICP head on cached frozen-CLIP features. Selects the best epoch on
# the VALIDATION macro-F1 and reports TEST metrics (removes the val==test
# optimism of the original V2 run). Supports k-shot sampling and ablations.
#
# Usable as a module (run_one) by scripts/run_matrix.py, or standalone:
#   python scripts/train_head.py --dataset politifact --variant kicp --shot full
# Run from the project root (KICP_project). ASCII only.

import argparse
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW

from models.kicp_head import KICPHead

CACHE_DIR = os.path.join("cache", "clip")
KNOW = {
    "politifact": os.path.join("knowledge", "wikidata_knowledge_politifact_embeddings.pt"),
    "gossipcop": os.path.join("knowledge", "wikidata_knowledge_gossipcop_embeddings.pt"),
}
VARIANT_FLAGS = {
    "kicp": {},
    "-IM": {"use_image": False},
    "-TX": {"use_text": False},
    "-CoAttention": {"use_coattention": False},
    "-CPT": {"use_prompt": False},
    "-KB": {"use_knowledge": False},
}
FEAT_KEYS = ["text_seq", "image_seq", "text_pooled", "image_pooled", "attn_mask"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_cache(ds):
    out = {}
    for split in ["train", "val", "test"]:
        out[split] = torch.load(os.path.join(CACHE_DIR, f"{ds}_{split}.pt"), weights_only=False)
    return out


def kshot_indices(labels, shot, seed):
    # Paper protocol: Z instances PER CLASS (capped by availability when a class
    # has fewer than Z training samples). "full" uses all training samples.
    n = len(labels)
    if shot == "full" or shot is None:
        return np.arange(n)
    k = int(shot)
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    parts = []
    for cls in (0, 1):
        idx = np.where(labels == cls)[0]
        m = min(k, len(idx))
        parts.append(rng.choice(idx, size=m, replace=False))
    sel = np.concatenate(parts)
    rng.shuffle(sel)
    return sel


def batches(cache, indices, batch_size, shuffle, device, seed=0):
    idx = np.asarray(indices)
    if shuffle:
        np.random.RandomState(seed).shuffle(idx)
    for s in range(0, len(idx), batch_size):
        b = idx[s:s + batch_size]
        bt = torch.as_tensor(b, dtype=torch.long)
        feats = {k: cache[k][bt].float().to(device) for k in ["text_seq", "image_seq", "text_pooled", "image_pooled"]}
        feats["attn_mask"] = cache["attn_mask"][bt].to(device)
        labels = cache["labels"][bt].to(device)
        yield feats, labels


@torch.no_grad()
def evaluate(model, cache, indices, device, batch_size=128):
    model.eval()
    yt, yp = [], []
    for feats, labels in batches(cache, indices, batch_size, False, device):
        out = model(feats["text_seq"], feats["image_seq"], feats["text_pooled"],
                    feats["image_pooled"], feats["attn_mask"])
        yp.extend(torch.argmax(out["probs"], dim=-1).cpu().tolist())
        yt.extend(labels.cpu().tolist())
    return {
        "acc": accuracy_score(yt, yp),
        "f1_macro": f1_score(yt, yp, average="macro", zero_division=0),
        "f1_binary": f1_score(yt, yp, average="binary", zero_division=0),
    }


def run_one(ds, variant, shot, seed, caches=None, epochs=20, lr=3e-5, weight_decay=1e-3,
            dropout=0.6, batch_size=32, device="cuda"):
    set_seed(seed)
    dev = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    cache = caches if caches is not None else load_cache(ds)
    flags = VARIANT_FLAGS[variant]

    train_idx = kshot_indices(cache["train"]["labels"].numpy(), shot, seed)
    val_idx = np.arange(len(cache["val"]["labels"]))
    test_idx = np.arange(len(cache["test"]["labels"]))

    use_knowledge = flags.get("use_knowledge", True)
    model = KICPHead(
        knowledge_embedding_path=KNOW[ds] if use_knowledge else None,
        dropout=dropout, coattention_heads=8, prompt_length=16, **flags,
    ).to(dev)

    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)

    best_val, best_test, best_epoch = -1.0, None, -1
    bs = min(batch_size, max(1, len(train_idx)))
    for epoch in range(1, epochs + 1):
        model.train()
        for feats, labels in batches(cache["train"], train_idx, bs, True, dev, seed + epoch):
            opt.zero_grad()
            out = model(feats["text_seq"], feats["image_seq"], feats["text_pooled"],
                        feats["image_pooled"], feats["attn_mask"], labels=labels)
            out["loss"].backward()
            opt.step()
        val_m = evaluate(model, cache["val"], val_idx, dev)
        if val_m["f1_macro"] > best_val:
            best_val = val_m["f1_macro"]
            best_epoch = epoch
            best_test = evaluate(model, cache["test"], test_idx, dev)

    return {
        "dataset": ds, "variant": variant, "shot": str(shot), "seed": seed,
        "n_train": int(len(train_idx)), "best_epoch": best_epoch,
        "val_f1_macro": round(best_val, 4),
        "test_f1_macro": round(best_test["f1_macro"], 4),
        "test_f1_binary": round(best_test["f1_binary"], 4),
        "test_acc": round(best_test["acc"], 4),
        "trainable_params": model.count_trainable(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="politifact", choices=["politifact", "gossipcop"])
    ap.add_argument("--variant", default="kicp", choices=list(VARIANT_FLAGS.keys()))
    ap.add_argument("--shot", default="full")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    shot = "full" if args.shot == "full" else int(args.shot)
    res = run_one(args.dataset, args.variant, shot, args.seed, epochs=args.epochs, device=args.device)
    print(res)


if __name__ == "__main__":
    main()
