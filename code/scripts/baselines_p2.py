# scripts/baselines_p2.py
# -*- coding: utf-8 -*-
#
# Same-split baselines on the cached frozen-CLIP features, plus data-quality
# checks. Uses exactly the splits, few-shot sampler (kshot_indices) and seeds of
# scripts/train_head.py, and the same knowledge retriever as the KICP head.
#
# Run from the project root (KICP_project):
#   python -u scripts/baselines_p2.py
# ASCII only.

import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from train_head import CACHE_DIR, KNOW, kshot_indices, set_seed
from models.real_knowledge_retriever import RealKnowledgeRetriever

SEEDS = [42, 1, 2]
SHOTS = [2, 4, 8, 16, 100, "full"]
DATASETS = ["politifact", "gossipcop"]
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
TIP_ALPHA = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
TIP_BETA = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
NEAR_DUP_COS = 0.98
TEXT_DUP_COS = 0.99
OUT_CSV = os.path.join("results", "baselines_p2.csv")
OUT_JSON = os.path.join("results", "data_quality.json")
PRED_DIR = os.path.join("results", "baseline_preds")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIP_NAME = "openai/clip-vit-large-patch14-336"
FIELDS = ["dataset", "method", "features", "shot", "seed", "n_train", "val_f1_macro",
          "test_f1_macro", "test_acc", "hparams", "seconds"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def macro_f1(y, p):
    return f1_score(y, p, average="macro", zero_division=0)


def write_row(row):
    new = not os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", encoding="utf-8") as f:
        if new:
            f.write(",".join(FIELDS) + "\n")
        vals = []
        for k in FIELDS:
            v = row[k]
            if isinstance(v, float):
                v = "%.4f" % v
            v = str(v)
            if "," in v or '"' in v:
                v = '"' + v.replace('"', '""') + '"'
            vals.append(v)
        f.write(",".join(vals) + "\n")


def save_pred(ds, method, feats, shot, seed, pred):
    os.makedirs(PRED_DIR, exist_ok=True)
    name = "%s_%s_%s_%s_%s.npy" % (ds, method, feats.replace("+", ""), shot, seed)
    np.save(os.path.join(PRED_DIR, name), np.asarray(pred, dtype=np.int8))


def load_pooled(ds):
    out = {}
    for split in ["train", "val", "test"]:
        path = os.path.join(CACHE_DIR, "%s_%s.pt" % (ds, split))
        t0 = time.time()
        d = torch.load(path, map_location="cpu", weights_only=False)
        out[split] = {
            "T": F.normalize(d["text_pooled"].float(), dim=-1),
            "I": F.normalize(d["image_pooled"].float(), dim=-1),
            "y": d["labels"].long().numpy(),
            "ids": [str(x) for x in d["ids"]] if "ids" in d else None,
        }
        del d
        gc.collect()
        log("loaded", path, "n=%d" % len(out[split]["y"]), "%.1fs" % (time.time() - t0))
    return out


def kb_size(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    if torch.is_tensor(d):
        return int(d.shape[0])
    for key in ("embeddings", "knowledge_embeddings", "features"):
        if isinstance(d, dict) and key in d and torch.is_tensor(d[key]):
            return int(d[key].shape[0])
    return None


def add_knowledge(ds, P):
    retr = RealKnowledgeRetriever(knowledge_embedding_path=KNOW[ds], hidden_dim=768, text_top_k=5,
                                  image_top_k=5, temperature=0.07, normalize=True)
    retr.eval()
    with torch.no_grad():
        for split in P:
            ks = []
            T, I = P[split]["T"], P[split]["I"]
            for s in range(0, len(T), 512):
                ko = retr(text_query_features=T[s:s + 512], image_query_features=I[s:s + 512])
                ks.append(ko["knowledge_features"].float().cpu())
            P[split]["K"] = F.normalize(torch.cat(ks), dim=-1)


def clip_assets():
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(CLIP_NAME, local_files_only=True).to(DEV).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_NAME, local_files_only=True)
    templates = [("real news", "fake news"),
                 ("This is a real news article.", "This is a fake news article."),
                 ("a true news story", "a false news story"),
                 ("reliable news", "misinformation")]
    with torch.no_grad():
        W = []
        for cls in (0, 1):  # label 0 = real, 1 = fake
            tok = proc(text=[t[cls] for t in templates], return_tensors="pt", padding=True).to(DEV)
            e = F.normalize(model.get_text_features(**tok).float(), dim=-1).mean(0)
            W.append(F.normalize(e, dim=-1))
        W = torch.stack(W).cpu()
        white = Image.new("RGB", (336, 336), (255, 255, 255))
        px = proc(images=white, return_tensors="pt").to(DEV)
        w_emb = F.normalize(model.get_image_features(**px).float(), dim=-1).cpu()[0]
    del model
    torch.cuda.empty_cache()
    return W, w_emb, [list(t) for t in templates]


def feat(Ps, combo):
    return torch.cat([Ps[c] for c in combo.split("+")], dim=1)


# ---------------------------------------------------------------- baselines
def run_lr(P, combo, tr_idx):
    Xtr = feat(P["train"], combo)[tr_idx].numpy()
    ytr = P["train"]["y"][tr_idx]
    Xva, yva = feat(P["val"], combo).numpy(), P["val"]["y"]
    Xte = feat(P["test"], combo).numpy()
    best = None
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced")
        clf.fit(Xtr, ytr)
        v = macro_f1(yva, clf.predict(Xva))
        if best is None or v > best[0]:
            best = (v, C, clf)
    v, C, clf = best
    return v, {"C": C}, clf.predict(Xte)


class MLP(nn.Module):
    def __init__(self, d_in, hidden=512, p=0.5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(p), nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x)


def run_mlp(P, combo, tr_idx, seed, epochs=100, lr=1e-3, wd=1e-2, batch_size=32):
    set_seed(seed)
    Xtr = feat(P["train"], combo)[tr_idx].to(DEV)
    ytr = torch.as_tensor(P["train"]["y"][tr_idx], dtype=torch.long, device=DEV)
    Xva, yva = feat(P["val"], combo).to(DEV), P["val"]["y"]
    Xte = feat(P["test"], combo).to(DEV)
    model = MLP(Xtr.shape[1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n = len(tr_idx)
    bs = min(batch_size, max(1, n))
    rng = np.random.RandomState(seed)
    best_v, best_pred, best_ep = -1.0, None, -1
    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(n)
        for s in range(0, n, bs):
            b = torch.as_tensor(perm[s:s + bs], device=DEV)
            opt.zero_grad()
            F.cross_entropy(model(Xtr[b]), ytr[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = macro_f1(yva, model(Xva).argmax(1).cpu().numpy())
            if v > best_v:
                best_v, best_ep = v, ep
                best_pred = model(Xte).argmax(1).cpu().numpy()
    n_params = sum(p.numel() for p in model.parameters())
    return best_v, {"epoch": best_ep, "params": n_params}, best_pred


def fused(Ps):
    return F.normalize(Ps["T"] + Ps["I"], dim=-1)


def run_zero_shot(P, W, combo):
    if combo == "T+I":
        Fva, Fte = fused(P["val"]), fused(P["test"])
    else:
        Fva, Fte = P["val"][combo], P["test"][combo]
    v = macro_f1(P["val"]["y"], (Fva @ W.T).argmax(1).numpy())
    return v, {}, (Fte @ W.T).argmax(1).numpy()


def run_tip(P, W, tr_idx):
    Ftr = fused(P["train"])[tr_idx].to(DEV)
    Ltr = F.one_hot(torch.as_tensor(P["train"]["y"][tr_idx]), 2).float().to(DEV)
    Wd = W.to(DEV)
    out = {}
    for split in ("val", "test"):
        Fq = fused(P[split]).to(DEV)
        out[split] = (100.0 * Fq @ Wd.T, Fq @ Ftr.T)
    best = None
    for a in TIP_ALPHA:
        for b in TIP_BETA:
            zs, sim = out["val"]
            logits = zs + a * (torch.exp(-b * (1.0 - sim)) @ Ltr)
            v = macro_f1(P["val"]["y"], logits.argmax(1).cpu().numpy())
            if best is None or v > best[0]:
                best = (v, a, b)
    v, a, b = best
    zs, sim = out["test"]
    pred = (zs + a * (torch.exp(-b * (1.0 - sim)) @ Ltr)).argmax(1).cpu().numpy()
    return v, {"alpha": a, "beta": b}, pred


# ---------------------------------------------------------------- data quality
def near_dup_counts(X, thr):
    Xd = X.to(DEV)
    n = Xd.shape[0]
    counts = torch.zeros(n, dtype=torch.long)
    for s in range(0, n, 1024):
        sim = Xd[s:s + 1024] @ Xd.T
        counts[s:s + 1024] = ((sim >= thr).sum(1) - 1).cpu()
    return counts.numpy()


def quality(ds, P, w_emb):
    splits = ["train", "val", "test"]
    I = torch.cat([P[s]["I"] for s in splits])
    T = torch.cat([P[s]["T"] for s in splits])
    y = np.concatenate([P[s]["y"] for s in splits])
    split_of = np.concatenate([[s] * len(P[s]["y"]) for s in splits])
    n = len(y)
    hashes = [hashlib.md5(np.round(v.numpy(), 3).astype(np.float32).tobytes()).hexdigest() for v in I]
    groups = {}
    for i, h in enumerate(hashes):
        groups.setdefault(h, []).append(i)
    dup_groups = sorted([g for g in groups.values() if len(g) > 1], key=len, reverse=True)
    exact_dup_samples = int(sum(len(g) for g in dup_groups))
    top_groups = [{"size": len(g), "fake": int(y[g].sum()), "real": int(len(g) - y[g].sum())} for g in dup_groups[:8]]
    img_near = near_dup_counts(I, NEAR_DUP_COS)
    txt_near = near_dup_counts(T, TEXT_DUP_COS)
    white_cos = (I @ w_emb).numpy()
    n_white = int((white_cos >= 0.99).sum())

    def cross_split(X, thr):
        Xtr = P["train"][X].to(DEV)
        res = {}
        for s in ("val", "test"):
            sim = P[s][X].to(DEV) @ Xtr.T
            res[s] = int(((sim >= thr).sum(1) > 0).sum().item())
        return res

    q = {
        "n_samples": int(n),
        "image_exact_duplicate_samples": exact_dup_samples,
        "image_exact_duplicate_groups": len(dup_groups),
        "image_largest_duplicate_groups": top_groups,
        "image_near_duplicate_samples_cos%.2f" % NEAR_DUP_COS: int((img_near > 0).sum()),
        "image_white_like_samples_cos0.99": n_white,
        "text_near_duplicate_samples_cos%.2f" % TEXT_DUP_COS: int((txt_near > 0).sum()),
        "val_test_with_train_image_near_dup": cross_split("I", NEAR_DUP_COS),
        "val_test_with_train_text_near_dup": cross_split("T", TEXT_DUP_COS),
        "knowledge_bank_size": kb_size(KNOW[ds]),
    }
    offset = 0
    for s in splits:
        m = len(P[s]["y"])
        P[s]["unique_img"] = (img_near[offset:offset + m] == 0) & (white_cos[offset:offset + m] < 0.99)
        offset += m
    return q


def unique_image_probe(ds, P):
    rows = []
    tr = np.where(P["train"]["unique_img"])[0]
    va = np.where(P["val"]["unique_img"])[0]
    te = np.where(P["test"]["unique_img"])[0]
    sub = {
        "train": {k: (P["train"][k][tr] if k != "ids" else None) for k in ("T", "I", "K", "y")},
        "val": {k: P["val"][k][va] for k in ("T", "I", "K", "y")},
        "test": {k: P["test"][k][te] for k in ("T", "I", "K", "y")},
    }
    for combo in ("T", "I", "T+I", "T+I+K"):
        t0 = time.time()
        v, hp, pred = run_lr(sub, combo, np.arange(len(tr)))
        yte = sub["test"]["y"]
        hp.update({"n_val": int(len(va)), "n_test": int(len(te))})
        rows.append({"dataset": ds, "method": "LR-uniqueimg", "features": combo, "shot": "full", "seed": "all",
                     "n_train": int(len(tr)), "val_f1_macro": v, "test_f1_macro": macro_f1(yte, pred),
                     "test_acc": accuracy_score(yte, pred), "hparams": json.dumps(hp), "seconds": time.time() - t0})
    return rows


# ---------------------------------------------------------------- main
def main():
    os.makedirs("results", exist_ok=True)
    if os.path.exists(OUT_CSV):
        os.replace(OUT_CSV, OUT_CSV + ".prev")
    W, w_emb, templates = clip_assets()
    report = {"env": {"torch": torch.__version__, "device": str(DEV)}, "zero_shot_templates": templates}
    try:
        import sklearn, transformers
        report["env"].update({"sklearn": sklearn.__version__, "transformers": transformers.__version__})
    except Exception:
        pass

    for ds in DATASETS:
        log("=== dataset", ds)
        P = load_pooled(ds)
        add_knowledge(ds, P)
        yte = P["test"]["y"]

        for combo in ("T", "I", "T+I"):
            t0 = time.time()
            v, hp, pred = run_zero_shot(P, W, combo)
            write_row({"dataset": ds, "method": "ZeroShotCLIP", "features": combo, "shot": "0", "seed": "all",
                       "n_train": 0, "val_f1_macro": v, "test_f1_macro": macro_f1(yte, pred),
                       "test_acc": accuracy_score(yte, pred), "hparams": json.dumps(hp), "seconds": time.time() - t0})
            save_pred(ds, "ZeroShotCLIP", combo, "0", "all", pred)

        for shot in SHOTS:
            seeds = SEEDS
            for seed in seeds:
                tr_idx = kshot_indices(P["train"]["y"], shot, seed)
                n_tr = int(len(tr_idx))
                lr_seeds_done = (shot == "full" and seed != SEEDS[0])
                for combo in ("T", "I", "T+I", "T+I+K"):
                    if lr_seeds_done:
                        continue  # logistic regression on the full training set is deterministic
                    t0 = time.time()
                    v, hp, pred = run_lr(P, combo, tr_idx)
                    write_row({"dataset": ds, "method": "LinearProbe", "features": combo, "shot": shot,
                               "seed": "all" if shot == "full" else seed, "n_train": n_tr, "val_f1_macro": v,
                               "test_f1_macro": macro_f1(yte, pred), "test_acc": accuracy_score(yte, pred),
                               "hparams": json.dumps(hp), "seconds": time.time() - t0})
                    save_pred(ds, "LinearProbe", combo, shot, seed, pred)
                for combo in ("T+I", "T+I+K"):
                    t0 = time.time()
                    v, hp, pred = run_mlp(P, combo, tr_idx, seed)
                    write_row({"dataset": ds, "method": "MLPFusion", "features": combo, "shot": shot, "seed": seed,
                               "n_train": n_tr, "val_f1_macro": v, "test_f1_macro": macro_f1(yte, pred),
                               "test_acc": accuracy_score(yte, pred), "hparams": json.dumps(hp),
                               "seconds": time.time() - t0})
                    save_pred(ds, "MLPFusion", combo, shot, seed, pred)
                t0 = time.time()
                v, hp, pred = run_tip(P, W, tr_idx)
                write_row({"dataset": ds, "method": "TipAdapter", "features": "T+I", "shot": shot, "seed": seed,
                           "n_train": n_tr, "val_f1_macro": v, "test_f1_macro": macro_f1(yte, pred),
                           "test_acc": accuracy_score(yte, pred), "hparams": json.dumps(hp),
                           "seconds": time.time() - t0})
                save_pred(ds, "TipAdapter", "T+I", shot, seed, pred)
                log(ds, "shot", shot, "seed", seed, "done")

        log("quality checks", ds)
        report[ds] = quality(ds, P, w_emb)
        for row in unique_image_probe(ds, P):
            write_row(row)
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        log(ds, "quality", json.dumps(report[ds]))
        del P
        gc.collect()

    log("ALL DONE")


if __name__ == "__main__":
    main()
