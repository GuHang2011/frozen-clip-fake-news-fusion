# Balanced accuracy (prior-insensitive) re-analysis with paired bootstrap; dup-stratified and clean-subset views.
# Usage: P2R=<p2_results_dir> python3 -I analyze2.py <out_dir>
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_lib import *
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)

def bacc(y, p):
    return 0.5 * (np.mean(p[y == 1] == 1) + np.mean(p[y == 0] == 0))

ALLM = [m for m, _ in METHODS] + ["FCP-fixed", "FCP-tuned"]
tab = {}
for d in DS:
    for s in SHOTS:
        for m in ALLM:
            P = get(d, m, s)
            for sub, mask in [("all", np.ones_like(Y[d], bool)), ("clean", CLEAN[d]), ("imgdup", IMGDUP[d]), ("noimgdup", ~IMGDUP[d])]:
                v = [100 * bacc(Y[d][mask], p[mask]) for p in P]
                f = [100 * f1_macro(Y[d][mask], p[mask]) for p in P]
                a = [100 * np.mean(Y[d][mask] == p[mask]) for p in P]
                tab["|".join([d, s, m, sub])] = dict(bacc=float(np.mean(v)), bacc_sd=float(np.std(v, ddof=1)) if len(v) > 1 else None,
                    f1=float(np.mean(f)), f1_sd=float(np.std(f, ddof=1)) if len(f) > 1 else None, acc=float(np.mean(a)), n=int(mask.sum()), runs=len(P))

rng = np.random.default_rng(0); B = 2000
boot = []
for d in DS:
    n = len(Y[d]); idxs = rng.integers(0, n, size=(B, n))
    for s in SHOTS:
        A = get(d, "FCP-tuned", s)
        for comp in ["MLP(T+I)", "LP(T+I)", "LP(T)"]:
            Bp = get(d, comp, s)
            pairs = [(A[i], Bp[i if len(Bp) > 1 else 0]) for i in range(len(A))]
            for metric, fn in [("bacc", bacc), ("f1", f1_macro)]:
                diffs = np.array([np.mean([fn(Y[d][ii], a[ii]) - fn(Y[d][ii], c[ii]) for a, c in pairs]) for ii in idxs])
                obs = np.mean([fn(Y[d], a) - fn(Y[d], c) for a, c in pairs])
                p2 = min(1.0, 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)))
                boot.append(dict(dataset=d, shot=s, comp=comp, metric=metric, diff=100 * float(obs),
                                 lo=100 * float(np.percentile(diffs, 2.5)), hi=100 * float(np.percentile(diffs, 97.5)), p=float(p2)))
for comp in ["MLP(T+I)", "LP(T+I)", "LP(T)"]:
    for metric in ["bacc", "f1"]:
        fam = sorted([b for b in boot if b["comp"] == comp and b["metric"] == metric], key=lambda b: b["p"])
        run = 0.0
        for i, b in enumerate(fam):
            run = max(run, min(1.0, (len(fam) - i) * b["p"])); b["p_holm"] = run
json.dump(dict(table=tab, bootstrap=boot), open(os.path.join(OUT, "reanalysis_bacc.json"), "w"), indent=1)

ms = ["LP(T)", "LP(I)", "LP(T+I)", "MLP(T+I)", "MLP(T+I+K)", "Tip(T+I)", "FCP-fixed", "FCP-tuned"]
for d in DS:
    for sub in ["all", "clean", "imgdup", "noimgdup"]:
        print("== bacc", d, sub, "n=", tab["|".join([d, "full", "LP(T)", sub])]["n"])
        for m in ms:
            print("%-11s" % m, " ".join("%5.1f" % tab["|".join([d, s, m, sub])]["bacc"] for s in SHOTS))
print("\n== bootstrap FCP-tuned minus comp")
for b in boot:
    print("%-10s %4s %-9s %-4s %+6.1f [%+6.1f,%+6.1f] p=%.3f holm=%.3f" % (b["dataset"], b["shot"], b["comp"], b["metric"], b["diff"], b["lo"], b["hi"], b["p"], b["p_holm"]))
