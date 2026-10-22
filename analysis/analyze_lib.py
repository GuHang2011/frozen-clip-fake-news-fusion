# CPU-only re-analysis of saved test predictions (no training, no new data).
# Usage: python3 -I analyze.py <p2_results_dir> <out_dir>
import sys, os, csv, json, glob
import numpy as np
R = os.environ["P2R"]
DS = ["politifact", "gossipcop"]; SHOTS = ["2", "4", "8", "16", "100", "full"]; SEEDS = [42, 1, 2]
LRS = ["3e-05", "0.0001", "0.0003"]

def f1_macro(y, p):
    out = []
    for c in (0, 1):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        d = 2 * tp + fp + fn
        out.append(0.0 if d == 0 else 2 * tp / d)
    return float(np.mean(out))

def stats(y, p):
    return dict(f1=100 * f1_macro(y, p), acc=100 * float(np.mean(y == p)),
                rec_fake=100 * float(np.mean(p[y == 1] == 1)), rec_real=100 * float(np.mean(p[y == 0] == 0)),
                pred_fake=100 * float(np.mean(p == 1)))

flags = {d: np.load(os.path.join(R, "test_flags", d + ".npz")) for d in DS}
Y = {d: flags[d]["labels"].astype(int) for d in DS}
CLEAN = {d: ~(flags[d]["img_dup_train"] | flags[d]["txt_dup_train"]) for d in DS}
IMGDUP = {d: flags[d]["img_dup_train"].astype(bool) for d in DS}

# ---- FCP lr selection exactly as in make_tables_figs.py (max val F1, tie -> smaller lr)
rows = list(csv.DictReader(open(os.path.join(R, "fcp_lr_p2.csv"))))
by = {}
for r in rows: by.setdefault((r["dataset"], r["shot"], int(r["seed"])), []).append(r)
sel = {k: sorted(v, key=lambda r: (-float(r["val_f1_macro"]), float(r["lr"])))[0]["lr"] for k, v in by.items()}
lr_counts = {d: {lr: sum(1 for (dd, s, se), l in sel.items() if dd == d and l == lr) for lr in LRS} for d in DS}

def fcp_preds(d, shot, tuned):
    out = []
    for seed in SEEDS:
        lr = sel[(d, shot, seed)] if tuned else "3e-05"
        f = os.path.join(R, "fcp_preds", "%s_%s_%s_%s.npy" % (d, shot, seed, ("%g" % float(lr))))
        out.append(np.load(f).astype(int))
    return out

def base_preds(d, name, shot):
    pat = os.path.join(R, "baseline_preds", "%s_%s_%s_*.npy" % (d, name, shot))
    fs = sorted(glob.glob(pat))
    order = {"42": 0, "1": 1, "2": 2}
    fs = sorted(fs, key=lambda f: order.get(f.rsplit("_", 1)[1][:-4], 9))
    return [np.load(f).astype(int) for f in fs]

METHODS = [("LP(T)", "LinearProbe_T"), ("LP(I)", "LinearProbe_I"), ("LP(T+I)", "LinearProbe_TI"),
           ("LP(T+I+K)", "LinearProbe_TIK"), ("MLP(T+I)", "MLPFusion_TI"), ("MLP(T+I+K)", "MLPFusion_TIK"),
           ("Tip(T+I)", "TipAdapter_TI")]

def get(d, m, shot):
    if m == "FCP-fixed": return fcp_preds(d, shot, False)
    if m == "FCP-tuned": return fcp_preds(d, shot, True)
    return base_preds(d, dict(METHODS)[m], shot)

