#!/usr/bin/env python3
"""Fold the pilot set to two strata.

AMGCL ships no MINRES (solvers: cg, bicgstab, bicgstabl, gmres, lgmres,
fgmres, idrs, richardson, preonly), so a symmetric-indefinite stratum would
be run with exactly the same solver configuration as the nonsymmetric one.
A stratum that shares its experimental condition with another is not a
stratum, so sym_indef is folded into nonsym.

Nothing is lost: mm_symmetry is retained per row, so the symmetric-indefinite
subset stays recoverable for post-hoc analysis.

Writes data/pilot_matrices_final.csv with a `solver` column.
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "pilot_matrices_v2.csv"
DST = ROOT / "data" / "pilot_matrices_final.csv"


def main():
    rows = list(csv.DictReader(SRC.open()))
    folded = 0
    for r in rows:
        r["stratum_v2"] = r["stratum"]
        if r["stratum"] == "sym_indef":
            r["stratum"] = "nonsym"
            folded += 1
        # Krylov solver follows directly from the stratum.
        r["solver"] = "cg" if r["stratum"] == "spd" else "bicgstab"

    fields = ["group", "name", "stratum", "solver", "mm_symmetry",
              "nrows", "nnz", "spd", "psym", "graphlike",
              "stratum_v2", "stratum_old"]
    with DST.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    counts, gl = {}, 0
    for r in rows:
        counts[r["stratum"]] = counts.get(r["stratum"], 0) + 1
        gl += int(r["graphlike"])
    print(f"folded {folded} sym_indef rows into nonsym")
    for s in sorted(counts):
        print(f"  {s:8s} {counts[s]:3d}  solver={'cg' if s == 'spd' else 'bicgstab'}")
    print(f"graphlike (kept, excluded from oracle stats): {gl}")
    print(f"written: {DST}")


if __name__ == "__main__":
    main()
