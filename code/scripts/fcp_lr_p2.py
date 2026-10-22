# scripts/fcp_lr_p2.py
# -*- coding: utf-8 -*-
#
# FCP (the KICP head) with the learning rate selected on the validation set, as
# the baselines in baselines_p2.py have their hyperparameters selected there.
# Same splits, few-shot sampler, seeds, epochs and batch order as train_head.py.
# Saves test predictions of every run and per-item duplicate flags of the test
# split (image or text near duplicate of a training item).
#
# Run from the project root (KICP_project):
#   python -u scripts/fcp_lr_p2.py
# ASCII only.

import gc
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW

from models.kicp_head import KICPHead
from train_head import KNOW, kshot_indices, load_cache, set_seed

LRS = [3e-5, 1e-4, 3e-4]
SEEDS = [42, 1, 2]
SHOTS = [2, 4, 8, 16, 100, "full"]
DATASETS = ["politifact", "gossipcop"]
EPOCHS = 20
IMG_DUP_COS = 0.98
TXT_DUP_COS = 0.99
OUT_CSV = os.path.join("results", "fcp_lr_p2.csv")
PRED_DIR = os.path.join("results", "fcp_preds")
FLAG_DIR = os.path.join("results", "test_flags")
FIELDS = ["dataset", "shot", "seed", "lr", "n_train", "best_epoch", "val_f1_macro",
          "test_f1_macro", "test_acc", "seconds"]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def write_row(row):
    new = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", encoding="utf-8") as f:
        if new:
            f.write(",".join(FIELDS) + "\n")
        f.write(",".join(str(row[k]) for k in FIELDS) + "\n")


def to_dev(split, idx):
    bt = torch.as_tensor(np.asarray(idx), dtype=torch.long)
    feats = {k: split[k][bt].float().to(DEV) for k in ("text_seq", "image_seq", "text_pooled", "image_pooled")}
    feats["attn_mask"] = split["attn_mask"][bt].to(DEV)
    return feats, split["labels"][bt].to(DEV)


@torch.no_grad()
def predict(model, split, batch_size=256):
    model.eval()
    n = len(split["labels"])
    out_all = []
    for s in range(0, n, batch_size):
        feats, _ = to_dev(split, np.arange(s, min(n, s + batch_size)))
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEV.type == "cuda")):
            out = model(feats["text_seq"], feats["image_seq"], feats["text_pooled"],
                        feats["image_pooled"], feats["attn_mask"])
        out_all.append(out["logits"].float().argmax(dim=-1).cpu())
    return torch.cat(out_all).numpy()


def train_one(cache, ds, shot, seed, lr):
    set_seed(seed)
    train_idx = kshot_indices(cache["train"]["labels"].numpy(), shot, seed)
    model = KICPHead(knowledge_embedding_path=KNOW[ds], dropout=0.6, coattention_heads=8,
                     prompt_length=16).to(DEV)
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-3)
    bs = min(32, max(1, len(train_idx)))
    y_val = cache["val"]["labels"].numpy()
    best_val, best_state, best_epoch = -1.0, None, -1
    for epoch in range(1, EPOCHS + 1):
        model.train()
        idx = np.asarray(train_idx).copy()
        np.random.RandomState(seed + epoch).shuffle(idx)
        for s in range(0, len(idx), bs):
            feats, labels = to_dev(cache["train"], idx[s:s + bs])
            opt.zero_grad()
            out = model(feats["text_seq"], feats["image_seq"], feats["text_pooled"],
                        feats["image_pooled"], feats["attn_mask"], labels=labels)
            out["loss"].backward()
            opt.step()
        v = f1_score(y_val, predict(model, cache["val"]), average="macro", zero_division=0)
        if v > best_val:
            best_val, best_epoch = v, epoch
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    pred = predict(model, cache["test"])
    y_test = cache["test"]["labels"].numpy()
    row = {"dataset": ds, "shot": str(shot), "seed": seed, "lr": "%g" % lr, "n_train": int(len(train_idx)),
           "best_epoch": best_epoch, "val_f1_macro": "%.4f" % best_val,
           "test_f1_macro": "%.4f" % f1_score(y_test, pred, average="macro", zero_division=0),
           "test_acc": "%.4f" % accuracy_score(y_test, pred)}
    del model, opt, best_state
    torch.cuda.empty_cache()
    return row, pred


def test_flags(cache, ds):
    def norm(x):
        return F.normalize(x.float(), dim=-1).to(DEV)
    img = (norm(cache["test"]["image_pooled"]) @ norm(cache["train"]["image_pooled"]).T).max(dim=1).values
    txt = (norm(cache["test"]["text_pooled"]) @ norm(cache["train"]["text_pooled"]).T).max(dim=1).values
    img_dup = (img >= IMG_DUP_COS).cpu().numpy()
    txt_dup = (txt >= TXT_DUP_COS).cpu().numpy()
    np.savez(os.path.join(FLAG_DIR, "%s.npz" % ds), img_dup_train=img_dup, txt_dup_train=txt_dup,
             labels=cache["test"]["labels"].numpy())
    log(ds, "test items with train near-duplicate image: %d, text: %d, either: %d of %d"
        % (img_dup.sum(), txt_dup.sum(), (img_dup | txt_dup).sum(), len(img_dup)))


def main():
    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(FLAG_DIR, exist_ok=True)
    if os.path.exists(OUT_CSV):
        os.replace(OUT_CSV, OUT_CSV + ".prev")
    for ds in DATASETS:
        t0 = time.time()
        cache = load_cache(ds)
        log("loaded", ds, "%.0fs" % (time.time() - t0))
        test_flags(cache, ds)
        for shot in SHOTS:
            for seed in SEEDS:
                for lr in LRS:
                    t1 = time.time()
                    row, pred = train_one(cache, ds, shot, seed, lr)
                    row["seconds"] = "%.1f" % (time.time() - t1)
                    write_row(row)
                    np.save(os.path.join(PRED_DIR, "%s_%s_%s_%g.npy" % (ds, shot, seed, lr)), pred.astype(np.int8))
                    log(ds, shot, seed, "lr=%g" % lr, "val", row["val_f1_macro"], "test", row["test_f1_macro"],
                        row["seconds"] + "s")
        del cache
        gc.collect()
        torch.cuda.empty_cache()
    log("ALL DONE")


if __name__ == "__main__":
    main()
