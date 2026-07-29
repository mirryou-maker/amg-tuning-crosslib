#!/usr/bin/env python3
"""Recommend AMG parameters for YOUR matrix.

This is the practical deployment of the paper's predictor. Given a Matrix
Market file, it extracts cheap (Tier 0/1) features, trains the gradient-boosted
predictor on ALL released sweep data (no held-out split -- you want every drop
of signal for a real prediction), scores every configuration for your matrix,
and prints a ranked short-list of configurations with their ready-to-paste
parameter strings.

Honest expectations (from the paper, leave-one-group-out): the top pick solves
about 83-89% of unseen matrices and captures a median ~43-49% of the achievable
speedup over the default. It is a helpful heuristic, not a guarantee -- try the
top few and keep the fastest.

Usage:
    python tools/recommend.py path/to/your.mtx                # AMGCL (default)
    python tools/recommend.py path/to/your.mtx --library hypre
    python tools/recommend.py path/to/your.mtx --spd          # force CG/SPD
    python tools/recommend.py path/to/your.mtx --top 8
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from compute_features import tier0, tier1   # reuse the paper's extractors

RES = ROOT / "results"
MCOLS = ["density", "avg_row_nnz", "std_row_nnz", "bandwidth_rel",
         "pattern_sym", "diag_present_frac", "diag_dom_frac", "value_asym",
         "power_eig_est", "gershgorin_radius"]

TRAIN = {"amgcl": "phase_e_sweep.jsonl", "hypre": "hypre_h2.jsonl"}
# fall back to the 150-matrix files if the expanded ones are absent
FALLBACK = {"amgcl": "p2_sweep.jsonl", "hypre": "hypre_h2.jsonl"}


def load_train(library):
    p = RES / TRAIN[library]
    if not p.exists():
        p = RES / FALLBACK[library]
    return [json.loads(l) for l in p.open() if l.strip()], p.name


def my_features(mtx_path):
    A = sp.csr_matrix(scipy.io.mmread(str(mtx_path)))
    A.eliminate_zeros()
    f = {}
    f.update(tier0(A))
    f.update(tier1(A))
    n = A.shape[0]
    sym = "symmetric" if (abs((A - A.T)).nnz == 0) else "general"
    return f, n, sym


def amgcl_config_axes(recs):
    coars, relax, eps, k = set(), set(), set(), set()
    for r in recs:
        cid = r.get("config_id", "")
        coars.add(re.search(r"coarsening\.type=(\w+)", cid).group(1))
        relax.add(re.search(r"relax\.type=(\w+)", cid).group(1))
        m = re.search(r"eps_strong=([\d.]+)", cid)
        eps.add(m.group(1) if m else "")
        mk = re.search(r"relax\.k=(\d+)", cid)
        k.add(mk.group(1) if mk else "")
    return sorted(coars), sorted(relax), sorted(eps), sorted(k)


def amgcl_vec(cid, coars, relax, eps, k):
    c = re.search(r"coarsening\.type=(\w+)", cid).group(1)
    x = re.search(r"relax\.type=(\w+)", cid).group(1)
    e = (re.search(r"eps_strong=([\d.]+)", cid) or [None, ""])[1] if "eps_strong" in cid else ""
    kk = (re.search(r"relax\.k=(\d+)", cid) or [None, ""])[1] if "relax.k" in cid else ""
    return ([1.0 if c == v else 0.0 for v in coars] +
            [1.0 if x == v else 0.0 for v in relax] +
            [1.0 if e == v else 0.0 for v in eps] +
            [1.0 if kk == v else 0.0 for v in k])


def hypre_vec(r, coars, relax, strong):
    def g(key):
        if key in r:
            return r[key]
        return re.search(rf"{key}=([\d.]+)", r["config_id"]).group(1)
    c, x, s = int(g("coarsen")), int(g("relax")), float(g("strong"))
    return ([1.0 if c == v else 0.0 for v in coars] +
            [1.0 if x == v else 0.0 for v in relax] +
            [1.0 if abs(s - v) < 1e-6 else 0.0 for v in strong])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("matrix", type=Path, help="Matrix Market (.mtx) file")
    ap.add_argument("--library", choices=["amgcl", "hypre"], default="amgcl")
    ap.add_argument("--spd", action="store_true", help="treat as SPD (CG)")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    from sklearn.ensemble import (GradientBoostingClassifier,
                                  GradientBoostingRegressor)

    print(f"reading {args.matrix} ...")
    f, n, sym = my_features(args.matrix)
    phi = [float(f[c]) for c in MCOLS]
    print(f"  n={n}, symmetry={sym}")

    recs, src = load_train(args.library)
    print(f"training on {len(recs)} solves from {src} ...")

    def ok(r):
        return r["status"] == "ok" and r.get("iters", 1) > 0

    if args.library == "amgcl":
        coars, relax, eps, k = amgcl_config_axes(recs)
        configs = sorted({r["config_id"] for r in recs if "config_id" in r})
        enc = lambda r: amgcl_vec(r["config_id"], coars, relax, eps, k)
        cand_enc = lambda cid: amgcl_vec(cid, coars, relax, eps, k)
        label = lambda cid: cid.replace("precond.", "")
    else:
        cs = sorted({int(re.search(r"coarsen=(\d+)", r["config_id"]).group(1)) for r in recs})
        rs = sorted({int(re.search(r"relax=(\d+)", r["config_id"]).group(1)) for r in recs})
        ss = sorted({float(re.search(r"strong=([\d.]+)", r["config_id"]).group(1)) for r in recs})
        configs = sorted({r["config_id"] for r in recs})
        enc = lambda r: hypre_vec(r, cs, rs, ss)
        cand_enc = lambda cid: hypre_vec({"config_id": cid}, cs, rs, ss)
        label = lambda cid: cid

    # training matrix (features are per-matrix; join by (group,name))
    feats_by_mat = {}
    # we don't have features for training matrices here, so read the released ones
    fp = RES / ("features_e.jsonl" if (RES / "features_e.jsonl").exists()
                else "features_p2.jsonl")
    F = {(x["group"], x["name"]): x for x in (json.loads(l) for l in fp.open())}

    X, ysolve, ylogt = [], [], []
    for r in recs:
        ff = F.get((r["group"], r["name"]))
        if not ff:
            continue
        mv = [ff.get(c) for c in MCOLS]
        if any(v is None for v in mv):
            continue
        X.append([float(v) for v in mv] + enc(r))
        ysolve.append(1 if ok(r) else 0)
        ylogt.append(np.log(r["t_total"]) if ok(r) else np.nan)
    X = np.array(X); ysolve = np.array(ysolve); ylogt = np.array(ylogt)

    clf = GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=0).fit(X, ysolve)
    mask = ~np.isnan(ylogt)
    reg = GradientBoostingRegressor(n_estimators=120, max_depth=3, random_state=0).fit(X[mask], ylogt[mask])

    # score every configuration for the user's matrix
    scored = []
    for cid in configs:
        v = np.array(phi + cand_enc(cid)).reshape(1, -1)
        ps = clf.predict_proba(v)[0, 1]
        pt = float(np.exp(reg.predict(v)[0]))
        scored.append((ps, pt, cid))
    # rank: among likely-to-solve, fastest predicted
    scored.sort(key=lambda t: (t[0] < 0.5, t[1]))

    solver = "cg" if (args.spd or sym == "symmetric") else \
             ("cg" if args.library == "amgcl" else "gmres")
    print(f"\nTop {args.top} recommended configurations "
          f"({args.library}, solver={solver}):")
    print(f"{'rank':>4}  {'P(solve)':>8}  {'pred.time':>9}  configuration")
    for i, (ps, pt, cid) in enumerate(scored[:args.top], 1):
        print(f"{i:>4}  {ps:>8.2f}  {pt:>9.3f}  {label(cid)}")
    print("\nNote: predictions are a heuristic (top pick solves ~83-89% of "
          "unseen matrices,\ncaptures ~43-49% of the oracle). Try the top few "
          "and keep the fastest.")


if __name__ == "__main__":
    main()
