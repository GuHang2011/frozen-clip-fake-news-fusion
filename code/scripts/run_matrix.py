# scripts/run_matrix.py
# -*- coding: utf-8 -*-
#
# Run the full KICP experiment matrix on cached frozen-CLIP features and write a
# tidy results CSV. Main KICP runs use several seeds (for mean +/- std on the
# headline table); ablations use one seed. The expensive "w/o Frozen Backbone"
# ablation lives in a separate live-CLIP script.
#
# Run from the project root (KICP_project). ASCII only.
#   python scripts/run_matrix.py --datasets politifact gossipcop

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from scripts.train_head import run_one, load_cache, VARIANT_FLAGS

ALL_SHOTS = [2, 4, 8, 16, 100, "full"]
ABLATIONS = ["-IM", "-TX", "-CoAttention", "-CPT", "-KB"]
OUT_DIR = "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["politifact", "gossipcop"])
    ap.add_argument("--shots", default=",".join(str(s) for s in ALL_SHOTS),
                    help="comma-separated, e.g. 2,4,full")
    ap.add_argument("--kicp_seeds", nargs="+", type=int, default=[42, 1, 2])
    ap.add_argument("--ablation_seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--variants", default=",".join(["kicp"] + ABLATIONS),
                    help="comma-separated, e.g. kicp,-IM,-KB")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "real_results.csv"))
    args = ap.parse_args()

    args.variants = [v for v in args.variants.split(",") if v]
    shots = ["full" if s == "full" else int(s) for s in args.shots.split(",") if s]
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = []
    if os.path.exists(args.out):
        rows = pd.read_csv(args.out).to_dict("records")
    done = {(r["dataset"], r["variant"], str(r["shot"]), int(r["seed"])) for r in rows}

    for ds in args.datasets:
        print(f"\n==== loading cache: {ds} ====", flush=True)
        cache = load_cache(ds)
        for variant in args.variants:
            seeds = args.kicp_seeds if variant == "kicp" else args.ablation_seeds
            for shot in shots:
                for seed in seeds:
                    key = (ds, variant, str(shot), int(seed))
                    if key in done:
                        continue
                    t0 = time.time()
                    res = run_one(ds, variant, shot, seed, caches=cache,
                                  epochs=args.epochs, device=args.device)
                    res["seconds"] = round(time.time() - t0, 1)
                    rows.append(res)
                    done.add(key)
                    pd.DataFrame(rows).to_csv(args.out, index=False, encoding="utf-8-sig")
                    print(f"[{ds}/{variant}/{shot}/s{seed}] "
                          f"testF1m={res['test_f1_macro']} valF1m={res['val_f1_macro']} "
                          f"n_tr={res['n_train']} ({res['seconds']}s)", flush=True)
        del cache

    print(f"\nDONE. {len(rows)} runs -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
