#!/usr/bin/env python3
"""H2/H3 combined analysis: hypre replication + default-quality + cross-library.

Three questions, one script, run on the full 150-matrix hypre sweep alongside
the AMGCL Phase 2 sweep:

  A. REPLICATION (H1-H3 at scale): hypre oracle distribution, winner spread,
     default-fails-tuned-succeeds. Does "internal params matter" hold at n=40+?

  B. DEFAULT QUALITY (the H1 nuance, quantified at scale): for every matrix
     solved by both libraries' defaults, compare default times and count wins.
     Tests "AMGCL's large oracle is partly a weak-default artifact." Caveat:
     absolute times cross implementations, so this is a trend not a benchmark.

  C. CROSS-LIBRARY CONSISTENCY (H4 seeds):
     - per-matrix oracle correlation (Spearman) between the two libraries.
     - overlap of "hard" (all-fail) matrix sets (Jaccard).
     - whether high-AMGCL-oracle matrices are the ones where hypre's default
       already wins (i.e. the artifact hypothesis, matrix by matrix).

Usage:
    python hypre_h3_analyze.py --hypre ../results/hypre_h2.jsonl \
        --amgcl ../results/p2_sweep.jsonl
"""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

HYPRE_DEFAULT = "coarsen=6 relax=8 strong=0.25 interp=6"
AMGCL_DEFAULT = {"coarsening.type=smoothed_aggregation",
                 "coarsening.aggr.eps_strong=0.08", "relax.type=spai0"}


def load(p):
    return [json.loads(l) for l in Path(p).open() if l.strip()]


def by_matrix(recs):
    d = defaultdict(list)
    for r in recs:
        d[(r["group"], r["name"])].append(r)
    return d


def ok_runs(rs):
    return [r for r in rs if r["status"] == "ok" and r.get("iters", 1) > 0]


def hypre_default(rs):
    return next((r for r in ok_runs(rs) if r.get("config_id") == HYPRE_DEFAULT), None)


def amgcl_default(rs):
    return next((r for r in ok_runs(rs)
                 if all(s in r.get("config_id", "") for s in AMGCL_DEFAULT)), None)


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1)) if n > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hypre", type=Path, default=ROOT / "results" / "hypre_h2.jsonl")
    ap.add_argument("--amgcl", type=Path, default=ROOT / "results" / "p2_sweep.jsonl")
    args = ap.parse_args()

    H = by_matrix(load(args.hypre))
    A = by_matrix(load(args.amgcl))

    # ---- A. replication ------------------------------------------------
    print("=== A. hypre replication (150-matrix full sweep) ===")
    hor, winners, dfail = {}, Counter(), 0
    for key, rs in H.items():
        ok = ok_runs(rs)
        if not ok:
            continue
        d = hypre_default(rs)
        best = min(ok, key=lambda r: r["t_total"])
        winners[best["config_id"]] += 1
        if d:
            hor[key] = d["t_total"] / best["t_total"]
        else:
            dfail += 1
    if hor:
        sp = sorted(hor.values())
        q = statistics.quantiles(sp, n=4)
        print(f"  oracle n={len(sp)}: median {statistics.median(sp):.2f} "
              f"Q1 {q[0]:.2f} Q3 {q[2]:.2f} max {max(sp):.2f}  "
              f">=1.5x {sum(1 for s in sp if s>=1.5)}/{len(sp)}")
    print(f"  default-fails / tuned-succeeds: {dfail}")
    print(f"  distinct winning configs: {len(winners)}")
    print(f"  top winners: {winners.most_common(5)}")

    # ---- B. default quality --------------------------------------------
    print("\n=== B. default quality: AMGCL-default vs hypre-default ===")
    hw = aw = both = 0
    ratios = []
    for key in set(H) & set(A):
        hd, ad = hypre_default(H[key]), amgcl_default(A[key])
        if hd and ad:
            both += 1
            ratios.append(hd["t_total"] / ad["t_total"])
            if hd["t_total"] < ad["t_total"]:
                hw += 1
            else:
                aw += 1
    if both:
        print(f"  matrices both defaults solve: {both}")
        print(f"  hypre-default faster: {hw}   AMGCL-default faster: {aw}")
        print(f"  median hypre/AMGCL default-time ratio: {statistics.median(ratios):.2f} "
              f"(<1 => hypre default faster)")
    # matrices only one default solves
    h_only = sum(1 for key in set(H) & set(A)
                 if hypre_default(H[key]) and not amgcl_default(A[key]))
    a_only = sum(1 for key in set(H) & set(A)
                 if amgcl_default(A[key]) and not hypre_default(H[key]))
    print(f"  default solves for ONLY hypre: {h_only}   ONLY AMGCL: {a_only}")

    # ---- C. cross-library consistency ----------------------------------
    print("\n=== C. cross-library consistency ===")
    aor = {}
    for key, rs in A.items():
        ok = ok_runs(rs)
        d = amgcl_default(rs)
        if ok and d:
            aor[key] = d["t_total"] / min(r["t_total"] for r in ok)
    common = sorted(set(hor) & set(aor))
    if len(common) > 2:
        xs = [aor[k] for k in common]
        ys = [hor[k] for k in common]
        print(f"  oracle Spearman corr (n={len(common)}): {spearman(xs, ys):.2f}")
    # hard-matrix overlap
    h_hard = {k for k, rs in H.items() if not ok_runs(rs)}
    a_hard = {k for k, rs in A.items() if not ok_runs(rs)}
    inter, union = h_hard & a_hard, h_hard | a_hard
    if union:
        print(f"  all-fail matrices: hypre {len(h_hard)}, AMGCL {len(a_hard)}, "
              f"both {len(inter)}, Jaccard {len(inter)/len(union):.2f}")
    # artifact test: among top-AMGCL-oracle matrices, does hypre-default win?
    top = sorted(common, key=lambda k: -aor[k])[:10]
    print("  top-10 AMGCL-oracle matrices: AMGCL-oracle vs hypre-oracle vs "
          "hypre-default-faster?")
    for k in top:
        hd, ad = hypre_default(H[k]), amgcl_default(A[k])
        faster = "hypre-def faster" if hd and ad and hd["t_total"] < ad["t_total"] else "-"
        print(f"    {k[0]+'/'+k[1]:<28} AMGCL {aor[k]:6.1f}x  hypre {hor[k]:5.2f}x  {faster}")


if __name__ == "__main__":
    main()
