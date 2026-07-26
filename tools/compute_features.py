#!/usr/bin/env python3
"""Extract matrix features in cost tiers for the Phase 2 predictor.

The tiers exist so the predictor can be honest about feature-extraction
overhead -- the gap the literature survey found nobody accounts for. Each
tier's marginal cost is reported, and the predictor is evaluated tier by tier
so "how much accuracy does Tier 2 actually buy?" has an answer.

  Tier 0  O(nnz), one pass, no arithmetic on values beyond abs/compare:
          size, density, row-nnz distribution, pattern symmetry, structural
          diagonal presence, bandwidth. These are essentially free.
  Tier 1  O(nnz), a few passes with real arithmetic: value symmetry proxy,
          diagonal-dominance fraction, diagonal magnitude spread, Gershgorin
          radius, a handful of power-iteration steps for a dominant-eigenvalue
          estimate.
  Tier 2  expensive: sparse condition-number estimate via a few Lanczos-like
          steps (largest/smallest eigenvalue ratio of the symmetric part).

Reads Matrix Market via scipy. Writes one JSON object per matrix with a
`t_tier{0,1,2}` timing so overhead is measurable.

Usage:
    python compute_features.py --matrices ../data/pilot_matrices_final.csv \
                               --out ../results/features.jsonl
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "matrices"


def find_mtx(group, name):
    d = CACHE / group / name
    cands = sorted(d.glob("*.mtx"))
    exact = [p for p in cands if p.stem == name]
    return (exact or cands or [None])[0]


def tier0(A):
    """Structure only. A is CSR."""
    n = A.shape[0]
    nnz = A.nnz
    row_nnz = np.diff(A.indptr)
    # bandwidth: max |i - j| over stored entries, sampled cheaply via coo
    coo = A.tocoo()
    band = int(np.max(np.abs(coo.row - coo.col))) if nnz else 0
    # structural pattern symmetry: fraction of (i,j) whose (j,i) is also stored
    P = (A != 0).astype(np.int8)
    inter = P.multiply(P.T)
    psym = inter.nnz / nnz if nnz else 1.0
    # structural diagonal presence
    diag_present = np.count_nonzero(A.diagonal() != 0) / n
    return {
        "n": int(n),
        "nnz": int(nnz),
        "density": nnz / (n * n),
        "avg_row_nnz": float(row_nnz.mean()),
        "std_row_nnz": float(row_nnz.std()),
        "max_row_nnz": int(row_nnz.max()),
        "min_row_nnz": int(row_nnz.min()),
        "bandwidth_rel": band / n,
        "pattern_sym": float(psym),
        "diag_present_frac": float(diag_present),
    }


def tier1(A):
    """Cheap arithmetic on values. A is CSR."""
    n = A.shape[0]
    d = A.diagonal()
    absd = np.abs(d)
    # diagonal dominance fraction: |a_ii| >= sum_{j!=i} |a_ij|
    absrowsum = np.abs(A).sum(axis=1).A1
    offdiag = absrowsum - absd
    dom_frac = float(np.mean(absd >= offdiag))
    # value symmetry proxy: ||A - A^T||_F / ||A||_F (0 = symmetric)
    fro = spla.norm(A)
    asym = spla.norm(A - A.T) / fro if fro > 0 else 0.0
    # Gershgorin spectral-radius bound
    gersh = float(np.max(absd + offdiag)) if n else 0.0
    # dominant eigenvalue estimate: few power iterations on |A|
    absA = abs(A)
    v = np.ones(n) / np.sqrt(n)
    ev = 0.0
    for _ in range(8):
        w = absA @ v
        nw = np.linalg.norm(w)
        if nw == 0:
            break
        v = w / nw
        ev = nw
    return {
        "diag_dom_frac": dom_frac,
        "value_asym": float(asym),
        "diag_abs_min": float(absd.min()) if n else 0.0,
        "diag_abs_max": float(absd.max()) if n else 0.0,
        "diag_abs_spread": float(absd.max() / absd.min()) if n and absd.min() > 0 else np.inf,
        "gershgorin_radius": gersh,
        "power_eig_est": float(ev),
    }


def tier2(A):
    """Expensive: crude condition-number estimate on the symmetric part."""
    n = A.shape[0]
    S = (A + A.T) * 0.5
    out = {"cond_est": np.inf, "eig_max": np.nan, "eig_min_abs": np.nan}
    try:
        emax = spla.eigsh(S, k=1, which="LM", maxiter=300,
                          return_eigenvectors=False, tol=1e-3)
        emin = spla.eigsh(S, k=1, which="SM", maxiter=300,
                          return_eigenvectors=False, tol=1e-3)
        emax = float(abs(emax[0]))
        emin = float(abs(emin[0]))
        out["eig_max"] = emax
        out["eig_min_abs"] = emin
        out["cond_est"] = emax / emin if emin > 0 else np.inf
    except Exception as e:
        out["cond_error"] = str(e)[:80]
    return out


def clean(d):
    """JSON-safe: inf/nan -> null."""
    return {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
            for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrices", type=Path,
                    default=ROOT / "data" / "pilot_matrices_final.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "features.jsonl")
    ap.add_argument("--max-tier", type=int, default=2)
    args = ap.parse_args()

    rows = list(csv.DictReader(args.matrices.open()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for i, r in enumerate(rows, 1):
            p = find_mtx(r["group"], r["name"])
            if not p:
                print(f"[{i}/{len(rows)}] MISSING {r['group']}/{r['name']}")
                continue
            A = scipy.io.mmread(str(p))
            A = sp.csr_matrix(A)
            A.eliminate_zeros()

            rec = {"group": r["group"], "name": r["name"],
                   "stratum": r["stratum"], "graphlike": int(r["graphlike"])}
            t = time.monotonic()
            rec.update(clean(tier0(A))); rec["t_tier0"] = time.monotonic() - t
            if args.max_tier >= 1:
                t = time.monotonic()
                rec.update(clean(tier1(A))); rec["t_tier1"] = time.monotonic() - t
            if args.max_tier >= 2:
                t = time.monotonic()
                rec.update(clean(tier2(A))); rec["t_tier2"] = time.monotonic() - t

            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"[{i}/{len(rows)}] {r['group']}/{r['name']} "
                  f"n={rec['n']} t0={rec['t_tier0']:.3f} "
                  f"t1={rec.get('t_tier1', 0):.3f} t2={rec.get('t_tier2', 0):.3f}")


if __name__ == "__main__":
    main()
