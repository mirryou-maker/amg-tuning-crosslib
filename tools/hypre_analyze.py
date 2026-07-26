#!/usr/bin/env python3
"""Analyse a hypre sweep and (optionally) compare to AMGCL head-to-head.

For H1 the question is narrow: does hypre BoomerAMG show a per-matrix oracle
speedup at all? hypre's default is Falgout coarsening (6), L1-symmetric-GS
relaxation (8), strong threshold 0.25 -- so the baseline here is that config.

If an AMGCL sweep is passed too, prints a side-by-side oracle table and the
overlap of "hard" (all-fail) matrices, the seeds of the H3 cross-library story.

Usage:
    python hypre_analyze.py --hypre ../results/hypre_pilot.jsonl \
        [--amgcl ../results/p2_sweep.jsonl]
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# hypre default config = Falgout(6) + L1-sym-GS(8) + strong 0.25.
HYPRE_DEFAULT = "coarsen=6 relax=8 strong=0.25 interp=6"
# AMGCL default = smoothed_aggregation + eps 0.08 + spai0.
AMGCL_DEFAULT = {"coarsening.type=smoothed_aggregation",
                 "coarsening.aggr.eps_strong=0.08", "relax.type=spai0"}


def load(p):
    return [json.loads(l) for l in Path(p).open() if l.strip()]


def oracle_hypre(recs):
    by = defaultdict(list)
    for r in recs:
        by[(r["group"], r["name"])].append(r)
    out = {}
    for key, rs in by.items():
        ok = [r for r in rs if r["status"] == "ok" and r.get("iters", 1) > 0]
        default = next((r for r in ok if r.get("config_id") == HYPRE_DEFAULT), None)
        if ok and default:
            best = min(ok, key=lambda r: r["t_total"])
            out[key] = {"oracle": default["t_total"] / best["t_total"],
                        "best_cfg": best["config_id"], "nok": len(ok),
                        "default_t": default["t_total"], "best_t": best["t_total"]}
    return out, by


def oracle_amgcl(recs):
    by = defaultdict(list)
    for r in recs:
        by[(r["group"], r["name"])].append(r)
    out = {}
    for key, rs in by.items():
        ok = [r for r in rs if r["status"] == "ok" and r.get("iters", 1) > 0]
        d = next((r for r in ok
                  if all(s in r.get("config_id", "") for s in AMGCL_DEFAULT)), None)
        if ok and d:
            best = min(ok, key=lambda r: r["t_total"])
            out[key] = d["t_total"] / best["t_total"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hypre", type=Path, default=ROOT / "results" / "hypre_pilot.jsonl")
    ap.add_argument("--amgcl", type=Path)
    args = ap.parse_args()

    hrecs = load(args.hypre)
    hor, hby = oracle_hypre(hrecs)

    print(f"hypre runs: {len(hrecs)}")
    from collections import Counter
    print("status:", dict(Counter(r["status"] for r in hrecs)))
    print(f"\n=== hypre per-matrix oracle (baseline = hypre default "
          f"Falgout+L1GS+0.25) ===")
    print(f"{'matrix':<32} {'nok':>3} {'default':>8} {'best':>8} {'oracle':>7}  best config")
    print("-" * 92)
    for key in sorted(hor, key=lambda k: -hor[k]["oracle"]):
        d = hor[key]
        print(f"{key[0]+'/'+key[1]:<32} {d['nok']:3d} {d['default_t']:8.4f} "
              f"{d['best_t']:8.4f} {d['oracle']:7.2f}  {d['best_cfg']}")

    if hor:
        sp = sorted(d["oracle"] for d in hor.values())
        q = statistics.quantiles(sp, n=4) if len(sp) >= 4 else [sp[0]] * 3
        print(f"\nhypre oracle (n={len(sp)}): min {min(sp):.2f}  Q1 {q[0]:.2f}  "
              f"median {statistics.median(sp):.2f}  Q3 {q[2]:.2f}  max {max(sp):.2f}")
        print(f"  >= 1.5x: {sum(1 for s in sp if s >= 1.5)}/{len(sp)}  "
              f">= 2x: {sum(1 for s in sp if s >= 2)}/{len(sp)}")
        # H1 gate verdict
        ok_gate = q[2] >= 1.5
        print(f"\n  H1 GATE (Q3 >= 1.5x): {'PASS' if ok_gate else 'FAIL'} "
              f"-- hypre {'shows' if ok_gate else 'lacks'} tunable oracle speedup")

    if args.amgcl and args.amgcl.exists():
        aor = oracle_amgcl(load(args.amgcl))
        common = sorted(set(hor) & set(aor))
        print(f"\n=== head-to-head oracle (matrices in both, n={len(common)}) ===")
        print(f"{'matrix':<32} {'AMGCL':>8} {'hypre':>8}")
        for key in common:
            print(f"{key[0]+'/'+key[1]:<32} {aor[key]:8.2f} {hor[key]['oracle']:8.2f}")


if __name__ == "__main__":
    main()
