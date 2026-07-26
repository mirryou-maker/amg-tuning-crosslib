#!/usr/bin/env python3
"""Re-stratify the pilot set using the Matrix Market header, not ssstats.csv.

ssstats.csv's psym field is *pattern* symmetry (nonzero structure), which the
first selection pass wrongly treated as value symmetry. The .mtx banner
carries the real answer:

    %%MatrixMarket matrix coordinate real {general|symmetric|skew-symmetric|hermitian}

'symmetric' means only the lower triangle is stored, so the matrix is
numerically symmetric. 'general' means it is not (or is not declared so).

Emits data/pilot_matrices_v2.csv with corrected strata plus a `graphlike`
tag for non-PDE matrices, which are kept but excluded from oracle statistics.

Usage:
    python reclassify.py
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "matrices"
SRC = ROOT / "data" / "pilot_matrices.csv"
DST = ROOT / "data" / "pilot_matrices_v2.csv"

# Non-PDE matrices: combinatorial graphs, economic models, NLP.
# AMG has no reason to work on these. Kept for failure-prediction analysis,
# tagged so they can be excluded from oracle-speedup statistics.
GRAPHLIKE = {
    ("Sorensen", "Linux_call_graph"),   # source call graph
    ("Tromble", "language"),            # natural language processing
    ("Arenas", "PGPgiantcompo"),        # social/trust network
    ("Gset", "G67"),                    # random graph (max-cut benchmark)
    ("Williams", "mac_econ_fwd500"),    # macroeconomic model
}


def banner(mtx_path):
    """Return the symmetry word from the Matrix Market header line."""
    with mtx_path.open("r", errors="replace") as f:
        first = f.readline().strip().lower()
    if not first.startswith("%%matrixmarket"):
        return "unknown"
    parts = first.split()
    return parts[-1] if len(parts) >= 5 else "unknown"


def find_mtx(group, name):
    d = CACHE / group / name
    cands = sorted(d.glob("*.mtx"))
    if not cands:
        return None
    # Some archives ship auxiliary _b.mtx / _coord.mtx files; prefer the
    # one named exactly after the matrix.
    exact = [p for p in cands if p.stem == name]
    return exact[0] if exact else cands[0]


def main():
    rows = list(csv.DictReader(SRC.open()))
    out = []
    changed = 0

    for r in rows:
        g, n = r["group"], r["name"]
        p = find_mtx(g, n)
        sym = banner(p) if p else "missing"
        spd = r["spd"] == "1"

        if spd:
            stratum = "spd"
        elif sym == "symmetric":
            stratum = "sym_indef"
        elif sym in ("general", "skew-symmetric", "hermitian"):
            stratum = "nonsym"
        else:
            stratum = "unknown"

        if stratum != r["stratum"]:
            changed += 1

        out.append({
            "group": g, "name": n,
            "stratum": stratum,
            "stratum_old": r["stratum"],
            "mm_symmetry": sym,
            "nrows": r["nrows"], "nnz": r["nnz"],
            "spd": r["spd"], "psym": r["psym"],
            "graphlike": int((g, n) in GRAPHLIKE),
        })

    with DST.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    counts = {}
    for r in out:
        counts[r["stratum"]] = counts.get(r["stratum"], 0) + 1

    print(f"reclassified {len(out)} matrices; {changed} changed stratum\n")
    for s in sorted(counts):
        ng = len({r["group"] for r in out if r["stratum"] == s})
        print(f"  {s:10s} {counts[s]:3d}  (groups={ng})")

    print("\nchanged:")
    for r in out:
        if r["stratum"] != r["stratum_old"]:
            print(f"  {r['group']}/{r['name']:22s} "
                  f"{r['stratum_old']} -> {r['stratum']}  (mm={r['mm_symmetry']})")

    ngl = sum(r["graphlike"] for r in out)
    print(f"\ngraphlike tagged: {ngl}")
    print(f"written: {DST}")


if __name__ == "__main__":
    main()
