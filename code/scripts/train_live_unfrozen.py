# scripts/train_live_unfrozen.py
# -*- coding: utf-8 -*-
#
# The single ablation row that cannot use cached features: "w/o Frozen Backbone"
# fine-tunes the whole CLIP ViT-L/14@336 together with the KICP head. Runs live
# (raw images/text) with AMP + gradient checkpointing to fit a 10GB GPU.
# Appends rows (variant="w/o-frozen") to the same results CSV as run_matrix.
#
# Run from the project root (KICP_project). ASCII only.
#   python scripts/train_live_unfrozen.py --dataset politifact --shots 2,4,8,16,100,full

import argparse
import os
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader

from datasets.kicp_dataset import KICPDataset
from datasets.kicp_collator import KICPCollator
from models.kicp_model import KICPModel

SPLIT_ROOT = os.path.join("data", "splits_real")
KNOW = {
    "politifact": os.path.join("knowledge", "wikidata_knowledge_politifact_embeddings.pt"),
    "gossipcop": os.path.join("knowledge", "wikidata_knowledge_gossipcop_embeddings.pt"),
}
CLIP_NAME = "openai/clip-vit-large-patch14-336"


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def kshot_df(df, shot, seed):
    if shot == "full":
        return df
    k = int(shot)
    n = len(df)
    fake = df[df["label"] == 1]; real = df[df["label"] == 0]
    ratio = len(fake) / max(n, 1)
    n_fake = min(len(fake), max(1, int(round(k * ratio))))
    n_real = min(len(real), max(1, k - n_fake))
    out = pd.concat([fake.sample(n_fake, random_state=seed), real.sample(n_real, random_state=seed)])
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def make_loader(df, batch_size, shuffle):
    tmp = os.path.join("cache", "_live_tmp.csv")
    os.makedirs("cache", exist_ok=True)
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    ds = KICPDataset(tmp)
    coll = KICPCollator(clip_model_name=CLIP_NAME, max_length=77)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=coll, num_workers=0)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); yt, yp = [], []
    for b in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device),
                        pixel_values=b["pixel_values"].to(device))
        yp.extend(torch.argmax(out["probs"], -1).cpu().tolist()); yt.extend(b["labels"].tolist())
    return {"acc": accuracy_score(yt, yp), "f1_macro": f1_score(yt, yp, average="macro", zero_division=0),
            "f1_binary": f1_score(yt, yp, average="binary", zero_division=0)}


def run(ds, shot, seed, epochs, batch_size, device):
    set_seed(seed)
    dev = torch.device(device)
    tr = pd.read_csv(os.path.join(SPLIT_ROOT, ds, "train.csv"), encoding="utf-8-sig")
    va = pd.read_csv(os.path.join(SPLIT_ROOT, ds, "val.csv"), encoding="utf-8-sig")
    te = pd.read_csv(os.path.join(SPLIT_ROOT, ds, "test.csv"), encoding="utf-8-sig")
    tr = kshot_df(tr, shot, seed)

    train_loader = make_loader(tr, batch_size, True)
    val_loader = make_loader(va, batch_size, False)
    test_loader = make_loader(te, batch_size, False)

    model = KICPModel(clip_model_name=CLIP_NAME, class_names=["real", "fake"], hidden_dim=768,
                      prompt_length=16, coattention_heads=8, coattention_dropout=0.6,
                      classifier_temperature=0.07, freeze_clip=False, use_knowledge=True,
                      top_k_text=5, top_k_image=5, knowledge_embedding_path=KNOW[ds]).to(dev)
    try:
        model.clip_encoder.clip.gradient_checkpointing_enable()
    except Exception:
        pass
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-5, weight_decay=1e-3)
    scaler = torch.cuda.amp.GradScaler()

    best_val, best_test, best_ep = -1.0, None, -1
    for ep in range(1, epochs + 1):
        model.train()
        for b in train_loader:
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(input_ids=b["input_ids"].to(dev), attention_mask=b["attention_mask"].to(dev),
                            pixel_values=b["pixel_values"].to(dev), labels=b["labels"].to(dev))
            scaler.scale(out["loss"]).backward(); scaler.step(opt); scaler.update()
        vm = evaluate(model, val_loader, dev)
        if vm["f1_macro"] > best_val:
            best_val = vm["f1_macro"]; best_ep = ep; best_test = evaluate(model, test_loader, dev)
    return {"dataset": ds, "variant": "w/o-frozen", "shot": str(shot), "seed": seed,
            "n_train": len(tr), "best_epoch": best_ep, "val_f1_macro": round(best_val, 4),
            "test_f1_macro": round(best_test["f1_macro"], 4), "test_f1_binary": round(best_test["f1_binary"], 4),
            "test_acc": round(best_test["acc"], 4),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["politifact", "gossipcop"])
    ap.add_argument("--shots", default="2,4,8,16,100,full")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join("results", "real_results.csv"))
    args = ap.parse_args()

    shots = ["full" if s == "full" else int(s) for s in args.shots.split(",") if s]
    rows = pd.read_csv(args.out).to_dict("records") if os.path.exists(args.out) else []
    done = {(r["dataset"], r["variant"], str(r["shot"]), int(r["seed"])) for r in rows}
    for shot in shots:
        if (args.dataset, "w/o-frozen", str(shot), args.seed) in done:
            continue
        t0 = time.time()
        res = run(args.dataset, shot, args.seed, args.epochs, args.batch_size, args.device)
        res["seconds"] = round(time.time() - t0, 1)
        rows.append(res)
        pd.DataFrame(rows).to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"[{args.dataset}/w-o-frozen/{shot}] testF1m={res['test_f1_macro']} "
              f"valF1m={res['val_f1_macro']} n_tr={res['n_train']} ({res['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
