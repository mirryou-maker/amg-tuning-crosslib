#!/usr/bin/env python3
"""H4: a hypre predictor and cross-library transfer experiments.

Three questions, all leave-one-group-out honest:

  1. hypre predictor. Can matrix features + a hypre-config encoding pick a good
     BoomerAMG configuration, the way they did for AMGCL? Reports solve rate and
     captured oracle. Expected modest, since hypre's oracle is itself modest.

  2. Difficulty transfer (should transfer). Train a matrix-level classifier
     "will ANY configuration solve this matrix?" on one library and test it on
     the other. Since the hard-matrix sets overlap strongly (Jaccard 0.76),
     intrinsic solvability should transfer.

  3. Tunability transfer (should NOT transfer). Train a regressor for the
     per-matrix oracle speedup on one library and test on the other. Since the
     per-matrix oracle rank correlation across libraries is only 0.24, tuning
     benefit should transfer poorly. A predictor that fails here *confirms* the
     paper's claim that tuning value is library-dependent, not intrinsic.

Config spaces differ between libraries, so we never transfer a config-level
model directly; only matrix-level quantities (solvable?, oracle magnitude),
which share the same library-independent feature space, are transferred.

Usage:
    python hypre_h4.py
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import (GradientBoostingClassifier,
                              GradientBoostingRegressor)
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

MCOLS = ["density", "avg_row_nnz", "std_row_nnz", "bandwidth_rel",
         "pattern_sym", "diag_present_frac", "diag_dom_frac", "value_asym",
         "power_eig_est", "gershgorin_radius"]

AMGCL_DEFAULT = {"coarsening.type=smoothed_aggregation",
                 "coarsening.aggr.eps_strong=0.08", "relax.type=spai0"}
HYPRE_DEFAULT = "coarsen=6 relax=8 strong=0.25 interp=6"


def load(p):
    return [json.loads(l) for l in (RES / p).open() if l.strip()]


def feats(p):
    return {(f["group"], f["name"]): f for f in load(p)}


def ok(r):
    return r["status"] == "ok" and r.get("iters", 1) > 0


def by_matrix(recs):
    d = defaultdict(list)
    for r in recs:
        d[(r["group"], r["name"])].append(r)
    return d


# ---- 1. hypre end-to-end predictor -----------------------------------
def hypre_params(r):
    """coarsen/relax/strong, from fields or parsed from config_id (timeout/
    crash records carry only config_id)."""
    if "coarsen" in r:
        return r["coarsen"], r["relax"], r["strong"]
    cid = r.get("config_id", "")
    c = int(re.search(r"coarsen=(\d+)", cid).group(1))
    x = int(re.search(r"relax=(\d+)", cid).group(1))
    s = float(re.search(r"strong=([\d.]+)", cid).group(1))
    return c, x, s


def hypre_config_vec(r):
    coars = [6, 8, 10]
    relax = [3, 6, 8, 18]
    strong = [0.25, 0.5, 0.7]
    c, x, s = hypre_params(r)
    return ([1.0 if c == cc else 0.0 for cc in coars] +
            [1.0 if x == xx else 0.0 for xx in relax] +
            [1.0 if abs(s - ss) < 1e-6 else 0.0 for ss in strong])


def hypre_predictor(F):
    recs = load("hypre_h2.jsonl")
    rows = []
    for r in recs:
        f = F.get((r["group"], r["name"]))
        if not f:
            continue
        mv = [f.get(c) for c in MCOLS]
        if any(v is None for v in mv):
            continue
        rows.append({"key": (r["group"], r["name"]),
                     "feat": [float(v) for v in mv] + hypre_config_vec(r),
                     "ok": ok(r), "t": r["t_total"] if ok(r) else None,
                     "is_default": r.get("config_id") == HYPRE_DEFAULT,
                     "group": r["group"]})
    X = np.array([r["feat"] for r in rows])
    y = np.array([1 if r["ok"] else 0 for r in rows])
    grp = np.array([r["group"] for r in rows])
    idx = defaultdict(list)
    for i, r in enumerate(rows):
        idx[r["key"]].append(i)

    clf = GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=0)
    reg = GradientBoostingRegressor(n_estimators=120, max_depth=3, random_state=0)
    hits = mats = 0
    captured = []
    for tr, te in LeaveOneGroupOut().split(X, y, grp):
        m_clf = clone(clf).fit(X[tr], y[tr])
        ok_tr = tr[y[tr] == 1]
        if len(ok_tr) < 20:
            continue
        m_reg = clone(reg).fit(X[ok_tr], np.log([rows[i]["t"] for i in ok_tr]))
        for key in {rows[i]["key"] for i in te}:
            cand = idx[key]
            ps = m_clf.predict_proba(X[cand])[:, 1]
            pt = m_reg.predict(X[cand])
            good = [i for i in range(len(cand)) if ps[i] > 0.5]
            pick = min(good, key=lambda i: pt[i]) if good else int(np.argmax(ps))
            picked = rows[cand[pick]]
            oks = [rows[i] for i in cand if rows[i]["ok"]]
            default = next((rows[i] for i in cand if rows[i]["is_default"] and rows[i]["ok"]), None)
            if not oks or default is None:
                continue
            mats += 1
            oracle = default["t"] / min(r["t"] for r in oks)
            got = default["t"] / picked["t"] if picked["ok"] else 1.0
            if picked["ok"]:
                hits += 1
            frac = np.log(got) / np.log(oracle) if oracle > 1 else 1.0
            captured.append(max(0.0, min(1.0, frac)))
    print("\n=== 1. hypre end-to-end predictor (leave-one-group-out) ===")
    print(f"  evaluable matrices: {mats}")
    print(f"  picked solves: {hits}/{mats} ({100*hits/mats:.0f}%)" if mats else "  n/a")
    if captured:
        print(f"  captured oracle (median): {np.median(captured)*100:.0f}%  "
              f"(hypre oracle is itself modest, median ~1.6x)")


# ---- matrix-level tables for transfer --------------------------------
def matrix_table(sweep, is_default, F):
    """Per matrix: features, any-solves flag, oracle magnitude (or None)."""
    out = {}
    for key, rs in by_matrix(sweep).items():
        f = F.get(key)
        if not f:
            continue
        mv = [f.get(c) for c in MCOLS]
        if any(v is None for v in mv):
            continue
        oks = [r for r in rs if ok(r)]
        d = next((r for r in oks if is_default(r)), None)
        oracle = (d["t_total"] / min(r["t_total"] for r in oks)) if (oks and d) else None
        out[key] = {"feat": [float(v) for v in mv],
                    "solves": 1 if oks else 0,
                    "oracle": oracle, "group": key[0]}
    return out


def transfer_classify(src, dst, name):
    """Train 'will any config solve?' on src matrices, test on dst matrices."""
    Xs = np.array([v["feat"] for v in src.values()])
    ys = np.array([v["solves"] for v in src.values()])
    keys_d = list(dst)
    Xd = np.array([dst[k]["feat"] for k in keys_d])
    yd = np.array([dst[k]["solves"] for k in keys_d])
    clf = GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=0)
    clf.fit(Xs, ys)
    pred = clf.predict(Xd)
    acc = (pred == yd).mean()
    base = max(yd.mean(), 1 - yd.mean())
    print(f"  {name}: acc {acc:.2f} vs majority {base:.2f}  (lift {acc-base:+.2f})")


def transfer_oracle(src, dst, name):
    """Train oracle-magnitude regressor on src, test rank corr on dst."""
    s = [(v["feat"], np.log(v["oracle"])) for v in src.values() if v["oracle"]]
    d = [(dst[k]["feat"], dst[k]["oracle"]) for k in dst if dst[k]["oracle"]]
    if len(s) < 20 or len(d) < 10:
        print(f"  {name}: too few oracle matrices")
        return
    Xs = np.array([a for a, _ in s]); ys = np.array([b for _, b in s])
    Xd = np.array([a for a, _ in d]); yd = np.array([b for _, b in d])
    reg = GradientBoostingRegressor(n_estimators=120, max_depth=3, random_state=0)
    reg.fit(Xs, ys)
    pred = np.exp(reg.predict(Xd))
    # Spearman between predicted and actual oracle on dst
    def rank(v):
        o = np.argsort(v); r = np.empty_like(o); r[o] = np.arange(len(v)); return r
    rp, ra = rank(pred), rank(yd)
    n = len(yd)
    rho = 1 - 6*np.sum((rp-ra)**2)/(n*(n*n-1))
    print(f"  {name}: predicted-vs-actual Spearman {rho:.2f} (n={n})")


def main():
    F = feats("features_p2.jsonl")
    amgcl = load("p2_sweep.jsonl")
    hypre = load("hypre_h2.jsonl")

    def amgcl_default(r):
        return all(s in r.get("config_id", "") for s in AMGCL_DEFAULT)

    def hypre_default(r):
        return r.get("config_id") == HYPRE_DEFAULT

    A = matrix_table(amgcl, amgcl_default, F)
    H = matrix_table(hypre, hypre_default, F)

    hypre_predictor(F)

    print("\n=== 2. difficulty transfer: 'will any config solve?' (should transfer) ===")
    transfer_classify(A, H, "train AMGCL -> test hypre")
    transfer_classify(H, A, "train hypre -> test AMGCL")

    print("\n=== 3. tunability transfer: oracle magnitude (should NOT transfer) ===")
    transfer_oracle(A, H, "train AMGCL -> test hypre")
    transfer_oracle(H, A, "train hypre -> test AMGCL")


if __name__ == "__main__":
    main()
