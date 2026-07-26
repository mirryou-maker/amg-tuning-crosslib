#!/usr/bin/env python3
"""Select the Phase 1 pilot matrix set from the SuiteSparse index.

Stratification unit is the SuiteSparse *group*. Two reasons:
  1. Phase 3 evaluates predictors with group-wise train/test splits (the gap
     the literature survey found), so the pilot should already span groups.
  2. ssstats.csv carries no 'kind' field, and group is the best available
     proxy for problem domain.

Three strata, because the solver/parameter space differs qualitatively:
  spd        -> CG is applicable
  sym_indef  -> symmetric but not positive definite
  nonsym     -> BiCGSTAB territory

Within each stratum we cap matrices per group and spread over size deciles,
so no single group or size band dominates.

Usage:
    python select_matrices.py --out data/pilot_matrices.csv
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from fetch_suitesparse import load_index

ROOT = Path(__file__).resolve().parent.parent

# Stratum targets. Nonsymmetric gets the largest share: parameter sensitivity
# is expected to be higher there, and it is the less-studied half.
TARGETS = {"spd": 18, "sym_indef": 12, "nonsym": 20}
MAX_PER_GROUP = 2


def stratum(m):
    if m["spd"]:
        return "spd"
    if m["psym"] >= 0.99:
        return "sym_indef"
    return "nonsym"


def eligible(index, nmin, nmax, nnz_max):
    out = []
    for m in index:
        if m["nrows"] != m["ncols"]:
            continue
        if not m["real"] or m["binary"]:
            continue
        if not (nmin <= m["nrows"] <= nmax):
            continue
        if m["nnz"] > nnz_max:
            continue
        # Skip near-diagonal trivia: AMG has nothing to do there.
        if m["nnz"] < 3 * m["nrows"]:
            continue
        out.append(m)
    return out


def pick(pool, target, max_per_group):
    """Round-robin over groups, walking size order, until target is met.

    Round-robin gives every group a first pick before any group gets a second,
    which maximises group coverage for a fixed count.
    """
    by_group = defaultdict(list)
    for m in sorted(pool, key=lambda m: m["nrows"]):
        by_group[m["group"]].append(m)

    # Deterministic group order, largest groups last so rare groups get in first.
    groups = sorted(by_group, key=lambda g: (len(by_group[g]), g))

    chosen, taken = [], defaultdict(int)
    for round_i in range(max_per_group):
        for g in groups:
            if len(chosen) >= target:
                break
            cands = by_group[g]
            if taken[g] >= min(max_per_group, len(cands)):
                continue
            # Spread within a group: first pick smallest, then largest.
            m = cands[0] if round_i == 0 else cands[-1]
            if m in chosen:
                continue
            chosen.append(m)
            taken[g] += 1
    return chosen[:target]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmin", type=int, default=10_000)
    ap.add_argument("--nmax", type=int, default=500_000)
    ap.add_argument("--nnz-max", type=int, default=5_000_000,
                    help="cap nnz so a pilot run stays cheap")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "pilot_matrices.csv")
    args = ap.parse_args()

    index = load_index()
    pool = eligible(index, args.nmin, args.nmax, args.nnz_max)

    strata = defaultdict(list)
    for m in pool:
        strata[stratum(m)].append(m)

    selected = []
    for s, target in TARGETS.items():
        got = pick(strata[s], target, MAX_PER_GROUP)
        for m in got:
            m["stratum"] = s
        selected.extend(got)
        ngroups = len({m["group"] for m in got})
        print(f"{s:10s} pool={len(strata[s]):4d}  target={target:3d}  "
              f"selected={len(got):3d}  groups={ngroups}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "name", "stratum", "nrows", "nnz", "spd", "psym"])
        for m in sorted(selected, key=lambda m: (m["stratum"], m["group"], m["name"])):
            w.writerow([m["group"], m["name"], m["stratum"], m["nrows"],
                        m["nnz"], int(m["spd"]), m["psym"]])

    print(f"\ntotal={len(selected)}  distinct groups={len({m['group'] for m in selected})}")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
