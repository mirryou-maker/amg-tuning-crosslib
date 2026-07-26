#!/usr/bin/env python3
"""Select ~150 SuiteSparse matrices for Phase 2, with MULTIPLE per group.

The Phase 1 set took ~1 matrix per group to maximise diversity. That is
exactly wrong for a predictor evaluated leave-one-group-out: with every matrix
its own group there is no within-group structure to learn, and the PoC showed
no signal as a result. Phase 2 deliberately picks several matrices from each
chosen group so LOGO has something to generalise across.

Selection rules:
  * Only groups with >= MIN_GROUP eligible matrices (so each contributes a
    small cluster, not a singleton).
  * Cap CAP per group so no collection dominates.
  * Exclude pure graph/network collections (DIMACS10, SNAP, ...) -- Phase 1
    showed AMG fails on all of them, so they burn compute for no oracle signal.
  * Keep an SPD share and spread over size deciles within each group.

Emits data/phase2_matrices.csv in the same schema as the pilot set, with a
`solver` column (cg for spd else bicgstab) so the existing sweep runs unchanged.

Usage:
    python select_phase2.py --target 150
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from fetch_suitesparse import load_index

ROOT = Path(__file__).resolve().parent.parent

MIN_GROUP = 3      # a chosen group must have at least this many eligible
CAP = 6            # at most this many from one group

# Pure graph/network collections: AMG has no reason to work; Phase 1 confirmed
# every such matrix failed all configs. Excluded to spend compute where a
# linear solve is meaningful.
GRAPH_GROUPS = {"DIMACS10", "SNAP", "Gset", "Arenas", "Newman",
                "Pajek", "vanHeukelum", "Barabasi"}


def eligible(index, nmin, nmax, nnz_max):
    out = []
    for m in index:
        if m["nrows"] != m["ncols"] or not m["real"] or m["binary"]:
            continue
        if not (nmin <= m["nrows"] <= nmax) or m["nnz"] > nnz_max:
            continue
        if m["nnz"] < 3 * m["nrows"]:
            continue
        if m["group"] in GRAPH_GROUPS:
            continue
        out.append(m)
    return out


def pick_from_group(ms, cap):
    """Spread over size: sort by n, take evenly across the range."""
    ms = sorted(ms, key=lambda m: m["nrows"])
    if len(ms) <= cap:
        return ms
    if cap == 1:
        return [ms[len(ms) // 2]]        # single pick: median size
    idx = [round(i * (len(ms) - 1) / (cap - 1)) for i in range(cap)]
    return [ms[i] for i in sorted(set(idx))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=150)
    ap.add_argument("--nmin", type=int, default=10_000)
    ap.add_argument("--nmax", type=int, default=500_000)
    ap.add_argument("--nnz-max", type=int, default=5_000_000)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "phase2_matrices.csv")
    args = ap.parse_args()

    index = load_index()
    pool = eligible(index, args.nmin, args.nmax, args.nnz_max)
    by_group = defaultdict(list)
    for m in pool:
        by_group[m["group"]].append(m)

    # Candidate groups: enough members, ordered so smaller (rarer) groups are
    # served first, giving broader group coverage before deepening any one.
    groups = sorted((g for g, v in by_group.items() if len(v) >= MIN_GROUP),
                    key=lambda g: (len(by_group[g]), g))

    selected, per_group = [], {}
    # Round-robin deepening: give each group its share up to CAP until target.
    depth = 1
    while len(selected) < args.target and depth <= CAP:
        for g in groups:
            if len(selected) >= args.target:
                break
            take = pick_from_group(by_group[g], depth)
            have = per_group.get(g, [])
            for m in take:
                if m not in have and len(selected) < args.target:
                    have.append(m)
                    selected.append(m)
            per_group[g] = have
        depth += 1

    def stratum(m):
        return "spd" if m["spd"] else "nonsym"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "name", "stratum", "solver", "nrows", "nnz",
                    "spd", "psym", "graphlike"])
        for m in sorted(selected, key=lambda m: (m["group"], m["nrows"])):
            s = stratum(m)
            w.writerow([m["group"], m["name"], s,
                        "cg" if s == "spd" else "bicgstab",
                        m["nrows"], m["nnz"], int(m["spd"]), m["psym"], 0])

    gs = defaultdict(int)
    for m in selected:
        gs[m["group"]] += 1
    multi = sum(1 for g, c in gs.items() if c >= 2)
    spd = sum(1 for m in selected if m["spd"])
    print(f"selected {len(selected)} matrices across {len(gs)} groups")
    print(f"  groups with >=2 matrices: {multi} (LOGO needs these)")
    print(f"  spd={spd} nonsym={len(selected) - spd}")
    print(f"  per-group counts: {dict(sorted(gs.items(), key=lambda x: -x[1]))}")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
