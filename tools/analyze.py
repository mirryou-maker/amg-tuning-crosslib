#!/usr/bin/env python3
"""Analyse a sweep result file: oracle speedup distribution, failure rates,
and whether the best configuration varies across matrices.

The oracle speedup for a matrix is

    T(default config) / T(best config)

computed only over matrices where BOTH the default and at least one other
configuration succeeded. Matrices tagged graphlike are excluded from the
speedup statistics (they never solve) but are reported in the failure table,
which is where they carry information.

No threshold is hardcoded: the literature survey found no credible oracle
speedup magnitude to compare against, so the distribution itself is the
result. Read the quartiles, not a single number.

Usage:
    python analyze.py --results ../data/local_scan.jsonl
"""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# AMGCL's own defaults: smoothed aggregation with eps_strong 0.08 and spai0.
# The baseline must be what a user gets without tuning -- comparing against
# the best of some other hand-picked config would inflate every speedup.
DEFAULT_CONFIG = {
    "precond.coarsening.type": "smoothed_aggregation",
    "precond.coarsening.aggr.eps_strong": "0.08",
    "precond.relax.type": "spai0",
}


def is_default(rec):
    cid = rec.get("config_id", "")
    return all(f"{k}={v}" in cid for k, v in DEFAULT_CONFIG.items())


def load(path):
    recs = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path,
                    default=ROOT / "data" / "local_scan.jsonl")
    args = ap.parse_args()

    recs = load(args.results)
    print(f"{len(recs)} runs loaded from {args.results}\n")

    print("=== status counts ===")
    for s, c in Counter(r["status"] for r in recs).most_common():
        print(f"  {s:<10} {c:5d}")

    # Per-matrix best/default times over successful runs.
    by_matrix = defaultdict(list)
    for r in recs:
        by_matrix[(r["group"], r["name"])].append(r)

    print("\n=== per-matrix summary ===")
    print(f"{'matrix':<34} {'ok':>3} {'fail':>4} {'default(s)':>11} "
          f"{'best(s)':>9} {'oracle':>7} {'best config':<32}")
    print("-" * 108)

    speedups, rows = [], []
    for (g, n), rs in sorted(by_matrix.items()):
        ok = [r for r in rs if r["status"] == "ok"]
        # A 0-iteration "success" means the RHS was degenerate (b = 0 for
        # zero-row-sum matrices under the old b = A*1 scheme): nothing was
        # solved, so such runs must not enter timing comparisons.
        ok = [r for r in ok if r.get("iters", 1) > 0]
        fail = len(rs) - len(ok)
        graphlike = rs[0].get("graphlike", 0)

        default = next((r for r in ok if is_default(r)), None)
        best = min(ok, key=lambda r: r["t_total"]) if ok else None

        if default and best and best["t_total"] > 0:
            sp = default["t_total"] / best["t_total"]
            if not graphlike:
                speedups.append(sp)
            spstr, dstr = f"{sp:7.2f}", f"{default['t_total']:11.4f}"
        else:
            spstr = "      -"
            dstr = "          -"

        bstr = (f"{best['t_total']:9.4f}" if best else "        -")
        bcfg = (best["config_id"].replace("precond.", "") if best else "-")
        tag = " [graph]" if graphlike else ""
        print(f"{g + '/' + n:<34} {len(ok):3d} {fail:4d} {dstr} {bstr} "
              f"{spstr} {bcfg:<32}{tag}")
        if best and not graphlike:
            rows.append(best["config_id"].replace("precond.", ""))

    if speedups:
        speedups.sort()
        q = statistics.quantiles(speedups, n=4) if len(speedups) >= 4 else [0, 0, 0]
        print(f"\n=== oracle speedup (n={len(speedups)}, graphlike excluded) ===")
        print(f"  min    {min(speedups):6.2f}")
        print(f"  Q1     {q[0]:6.2f}")
        print(f"  median {statistics.median(speedups):6.2f}")
        print(f"  Q3     {q[2]:6.2f}")
        print(f"  max    {max(speedups):6.2f}")
        top = sum(1 for s in speedups if s >= 2.0)
        print(f"  >= 2x  {top}/{len(speedups)} ({top / len(speedups) * 100:.0f}%)")

    if rows:
        print("\n=== winning configuration frequency ===")
        for cfg, c in Counter(rows).most_common():
            print(f"  {cfg:<34} {c:3d}")
        print("\n  (a single dominant winner would mean the default should just "
              "be changed;\n   a spread means per-matrix selection has room)")


if __name__ == "__main__":
    main()
