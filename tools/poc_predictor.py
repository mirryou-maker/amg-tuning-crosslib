#!/usr/bin/env python3
"""Phase 2 proof-of-concept predictor on the Phase 1 (50-matrix) data.

Gate question: is there ANY signal from cheap features to the per-matrix
winning coarse label (coarsening family + relaxation type)? If a leave-one-
group-out evaluation beats the majority-class baseline using only Tier 0/1
features, the full Phase 2 collection is worth the compute. If not, more
matrices of the same kind will not help and the design needs rethinking.

Deliberately honest choices:
  * Target = coarse label (family + relax), the 78%-tol-stable target from the
    Phase 1b tol probe -- not the exact eps_strong/k, which is tol-sensitive.
  * Evaluation = leave-one-GROUP-out. SuiteSparse sibling matrices share a
    group; a random split would leak them across train/test and inflate the
    score. This is the split the literature survey found nobody uses.
  * Baseline = always predict the training-set majority label. A predictor
    that cannot beat this has learned nothing.
  * Tiers reported separately, so the accuracy Tier 2 buys is explicit against
    its 100-1000x cost.

No sklearn dependency assumed: uses a compact k-NN in standardized feature
space (works with tiny data and needs no training loop).

Usage:
    python poc_predictor.py --features ../results/features.jsonl \
                            --sweep ../results/p1b_ramp.jsonl
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TIER0 = ["density", "avg_row_nnz", "std_row_nnz", "max_row_nnz",
         "bandwidth_rel", "pattern_sym", "diag_present_frac"]
TIER1 = ["diag_dom_frac", "value_asym", "diag_abs_spread",
         "gershgorin_radius", "power_eig_est"]
TIER2 = ["cond_est", "eig_max", "eig_min_abs"]


def load_jsonl(p):
    return [json.loads(l) for l in Path(p).open() if l.strip()]


def coarse_label(rec):
    return f"{rec['coarsening']}|{rec['relaxation']}"


def winners(sweep):
    by = defaultdict(list)
    for r in sweep:
        by[(r["group"], r["name"])].append(r)
    out = {}
    for key, rs in by.items():
        ok = [r for r in rs if r["status"] == "ok" and r.get("iters", 1) > 0]
        if ok:
            out[key] = coarse_label(min(ok, key=lambda r: r["t_total"]))
    return out


def build(features, wins, cols):
    X, y, groups, names = [], [], [], []
    for f in features:
        key = (f["group"], f["name"])
        if key not in wins:
            continue
        row = [f.get(c) for c in cols]
        if any(v is None for v in row):
            continue                       # skip matrices missing a feature
        X.append([float(v) for v in row])
        y.append(wins[key])
        groups.append(f["group"])
        names.append(f"{f['group']}/{f['name']}")
    return X, y, groups, names


def standardize(X):
    m = len(X[0])
    mean = [sum(r[j] for r in X) / len(X) for j in range(m)]
    var = [sum((r[j] - mean[j]) ** 2 for r in X) / len(X) for j in range(m)]
    std = [math.sqrt(v) or 1.0 for v in var]
    return [[(r[j] - mean[j]) / std[j] for j in range(m)] for r in X]


def knn_logo(X, y, groups, k=3):
    """Leave-one-group-out k-NN accuracy."""
    Xs = standardize(X)
    correct = total = 0
    preds = []
    for gi in sorted(set(groups)):
        tr = [i for i in range(len(y)) if groups[i] != gi]
        te = [i for i in range(len(y)) if groups[i] == gi]
        for t in te:
            d = sorted(((sum((Xs[t][j] - Xs[i][j]) ** 2
                             for j in range(len(Xs[t]))), i) for i in tr))
            near = [y[i] for _, i in d[:k]]
            pred = Counter(near).most_common(1)[0][0]
            preds.append((y[t], pred))
            correct += (pred == y[t])
            total += 1
    return correct / total if total else 0.0, preds


def majority_logo(y, groups):
    correct = total = 0
    for gi in sorted(set(groups)):
        tr = [y[i] for i in range(len(y)) if groups[i] != gi]
        te = [y[i] for i in range(len(y)) if groups[i] == gi]
        maj = Counter(tr).most_common(1)[0][0]
        correct += sum(1 for v in te if v == maj)
        total += len(te)
    return correct / total if total else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=ROOT / "results" / "features.jsonl")
    ap.add_argument("--sweep", type=Path, default=ROOT / "results" / "p1b_ramp.jsonl")
    ap.add_argument("-k", type=int, default=3)
    args = ap.parse_args()

    features = load_jsonl(args.features)
    wins = winners(load_jsonl(args.sweep))
    print(f"matrices with a winning coarse label: {len(wins)}")
    print(f"distinct coarse labels: {len(set(wins.values()))}")
    print("label distribution:")
    for lab, c in Counter(wins.values()).most_common():
        print(f"   {c:2d}  {lab}")

    for name, cols in [("Tier0", TIER0),
                       ("Tier0+1", TIER0 + TIER1),
                       ("Tier0+1+2", TIER0 + TIER1 + TIER2)]:
        X, y, groups, names = build(features, wins, cols)
        if len(X) < 5:
            print(f"\n{name}: too few usable matrices ({len(X)})")
            continue
        base = majority_logo(y, groups)
        acc, _ = knn_logo(X, y, groups, k=args.k)
        n_groups = len(set(groups))
        print(f"\n{name:10s} n={len(X)} groups={n_groups} "
              f"majority-baseline={base:.2f}  kNN-LOGO={acc:.2f}  "
              f"lift={acc - base:+.2f}")


if __name__ == "__main__":
    main()
