# Per-run operating points (predicted-fake share), macro-F1 and balanced accuracy for the
# few-shot runs, from the saved test predictions. Usage: P2R=<p2_results_dir> python3 -I analyze3.py <out_dir>
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_lib import *
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
rows = []
for d in DS:
    for s in ["2", "4", "8", "16"]:
        for m in [m for m, _ in METHODS] + ["FCP-fixed", "FCP-tuned"]:
            for seed, p in zip(SEEDS, get(d, m, s)):
                y = Y[d]
                rows.append(dict(dataset=d, shot=s, method=m, seed=seed, pred_fake=100 * float(p.mean()),
                                 f1=100 * f1_macro(y, p), bacc=50 * float(np.mean(p[y == 1] == 1) + np.mean(p[y == 0] == 0))))
json.dump(rows, open(os.path.join(OUT, "per_run_operating_points.json"), "w"), indent=1)
for r in rows:
    if r["dataset"] == "gossipcop" and r["method"] == "MLP(T+I)" and r["shot"] in ("2", "4", "8"):
        print("%s %s seed %d: fake %.1f%%  MF1 %.1f  BA %.1f" % (r["method"], r["shot"], r["seed"], r["pred_fake"], r["f1"], r["bacc"]))
