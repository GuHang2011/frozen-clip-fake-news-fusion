# CPU-only re-analysis of saved test predictions (no training, no new data).
# Usage: python3 -I analyze.py <p2_results_dir> <out_dir>
import sys, os, csv, json, glob
import numpy as np
R, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
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

ALL = [m for m, _ in METHODS] + ["FCP-fixed", "FCP-tuned"]
res = {}
for d in DS:
    for shot in SHOTS:
        for m in ALL:
            P = get(d, m, shot)
            if not P: continue
            def agg(mask):
                S = [stats(Y[d][mask], p[mask]) for p in P]
                return {k: (float(np.mean([s[k] for s in S])), float(np.std([s[k] for s in S], ddof=1)) if len(S) > 1 else None) for k in S[0]}
            res[(d, shot, m)] = dict(n_runs=len(P), all=agg(np.ones_like(Y[d], bool)), clean=agg(CLEAN[d]),
                                     imgdup=agg(IMGDUP[d]), noimgdup=agg(~IMGDUP[d]))

# ---- check against reported CSV / manuscript (recompute baselines CSV means)
brow = list(csv.DictReader(open(os.path.join(R, "baselines_p2.csv"))))
mism = []
for r in brow:
    if r["method"] in ("ZeroShotCLIP", "LR-uniqueimg"): continue
    key = {"LinearProbe": "LP", "MLPFusion": "MLP", "TipAdapter": "Tip"}[r["method"]] + "(" + r["features"] + ")"
    shot = r["shot"]
    P = get(r["dataset"], key, shot)
    idx = {"42": 0, "1": 1, "2": 2}.get(r["seed"], 0)
    if idx >= len(P): mism.append(("missing", r)); continue
    f = f1_macro(Y[r["dataset"]], P[idx])
    if abs(f - float(r["test_f1_macro"])) > 6e-5: mism.append((key, shot, r["seed"], f, r["test_f1_macro"]))
for r in rows:
    f = f1_macro(Y[r["dataset"]], np.load(os.path.join(R, "fcp_preds", "%s_%s_%s_%g.npy" % (r["dataset"], r["shot"], r["seed"], float(r["lr"])))).astype(int))
    if abs(f - float(r["test_f1_macro"])) > 6e-5: mism.append(("FCP", r["dataset"], r["shot"], r["seed"], r["lr"], f, r["test_f1_macro"]))

# ---- paired bootstrap with Holm correction (FCP tuned vs MLP(T+I) and LP(T+I))
rng = np.random.default_rng(0)
B = 2000
boot = []
for d in DS:
    n = len(Y[d]); idxs = rng.integers(0, n, size=(B, n))
    for shot in SHOTS:
        A = get(d, "FCP-tuned", shot)
        for comp in ["MLP(T+I)", "LP(T+I)", "LP(T)"]:
            Bp = get(d, comp, shot)
            diffs = np.zeros(B)
            pairs = [(A[i], Bp[i if len(Bp) > 1 else 0]) for i in range(len(A))]
            for b in range(B):
                ii = idxs[b]; y = Y[d][ii]
                diffs[b] = np.mean([f1_macro(y, a[ii]) - f1_macro(y, c[ii]) for a, c in pairs])
            obs = np.mean([f1_macro(Y[d], a) - f1_macro(Y[d], c) for a, c in pairs])
            p_two = min(1.0, 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)))
            boot.append(dict(dataset=d, shot=shot, comp=comp, diff=100 * obs, lo=100 * np.percentile(diffs, 2.5),
                             hi=100 * np.percentile(diffs, 97.5), p_boot=float(p_two)))
# Holm within each comparison family (12 settings)
for comp in ["MLP(T+I)", "LP(T+I)", "LP(T)"]:
    fam = sorted([b for b in boot if b["comp"] == comp], key=lambda b: b["p_boot"])
    m = len(fam); run = 0.0
    for i, b in enumerate(fam):
        run = max(run, min(1.0, (m - i) * b["p_boot"])); b["p_holm"] = run

# ---- variance across seeds vs between methods
var = []
for d in DS:
    for shot in ["2", "4", "8", "16", "100"]:
        means = [res[(d, shot, m)]["all"]["f1"][0] for m in ALL if res[(d, shot, m)]["n_runs"] == 3]
        sds = [res[(d, shot, m)]["all"]["f1"][1] for m in ALL if res[(d, shot, m)]["n_runs"] == 3]
        var.append(dict(dataset=d, shot=shot, sd_between_method_means=float(np.std(means, ddof=1)),
                        median_seed_sd=float(np.median(sds)), range_means=float(max(means) - min(means))))

def ser(o):
    return {"|".join(k): v for k, v in o.items()}
out = dict(lr_counts=lr_counts, prediction_csv_mismatches=[str(x) for x in mism], n_checked_baseline_rows=len(brow),
           n_checked_fcp_rows=len(rows), test_prior={d: dict(n=int(len(Y[d])), fake=int(Y[d].sum()),
           clean=int(CLEAN[d].sum()), clean_fake=int(Y[d][CLEAN[d]].sum()), imgdup=int(IMGDUP[d].sum()),
           imgdup_fake=int(Y[d][IMGDUP[d]].sum())) for d in DS},
           results=ser(res), bootstrap=boot, variance=var)
json.dump(out, open(os.path.join(OUT, "reanalysis.json"), "w"), indent=1)
print("mismatches:", len(mism)); print(mism[:5]); print("lr_counts", lr_counts); print(out["test_prior"])
