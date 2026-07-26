#!/usr/bin/env python3
"""One-command Level-1 reproduction of every quantitative claim in the paper.

This regenerates all figures and prints every headline number next to the
manuscript value it backs, straight from the released sweep data in results/.
It needs no cluster, no compiled solver, and no matrix downloads -- only the
JSONL result files and a standard scientific Python stack. Runtime: a couple of
minutes on a laptop.

    python reproduce.py

For the full end-to-end reproduction (build AMGCL/hypre, fetch the matrices,
re-run the 18,600 solves on a cluster) see docs/REPRODUCE.md.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOOLS = ROOT / "tools"
RESULTS = ROOT / "results"

REQUIRED = [
    "p2_sweep.jsonl", "hypre_h2.jsonl", "features_p2.jsonl", "features.jsonl",
]

# Each claim: (manuscript value, how to read it from the data). Kept in sync
# with the abstract and Sections 4-6 so a reviewer can check line by line.
PAPER_CLAIMS = """
Manuscript claim                                     | Section
-----------------------------------------------------+---------
AMGCL oracle: median 2.4x, Q3 6.4x, max 292x         | Abstract, Sec 4
AMGCL oracle: >=2x on 21/41 matrices                 | Sec 4
14 distinct winning coarse labels                    | Sec 4, Fig 2
default-fails / tuned-succeeds: 31 matrices          | Sec 4
predictor: solves 83%, captures ~43% of oracle       | Abstract, Sec 5
feature cost: Tier2 100-1000x Tier0/1                 | Sec 5, Fig 4
hypre oracle: median 1.6x, Q3 2.1x, max 31x          | Abstract, Sec 6
cross-library Spearman rho = 0.24                    | Abstract, Sec 6
hard-matrix Jaccard = 0.76                           | Abstract, Sec 6
"""


def run(cmd, title):
    print("\n" + "=" * 70)
    print(f">> {title}")
    print("=" * 70)
    r = subprocess.run([sys.executable, *cmd], cwd=ROOT)
    if r.returncode != 0:
        print(f"!! step failed: {' '.join(cmd)}", file=sys.stderr)
        sys.exit(r.returncode)


def main():
    missing = [f for f in REQUIRED if not (RESULTS / f).exists()]
    if missing:
        sys.exit(f"missing released data files in results/: {missing}")

    print(__doc__)
    print("Claims this script reproduces:")
    print(PAPER_CLAIMS)

    # 1. figures + key numbers (make_figures prints the KEY NUMBERS block)
    run([str(TOOLS.parent / "paper" / "figure" / "make_figures.py")],
        "Regenerate figures and print key numbers (Fig 1-5)")

    # 2. AMGCL oracle / winners / failure analysis
    run([str(TOOLS / "analyze.py"), "--results", str(RESULTS / "p2_sweep.jsonl")],
        "AMGCL oracle distribution, winners, failure counts (Sec 4)")

    # 3. predictor end-to-end (success rate + captured oracle, group-wise)
    run([str(TOOLS / "phase2_predictor.py"),
         "--features", str(RESULTS / "features_p2.jsonl"),
         "--sweep", str(RESULTS / "p2_sweep.jsonl")],
        "Predictor: solve rate and captured oracle, Tier0 vs Tier0+1 (Sec 5)")

    # 4. hypre replication + default-quality + cross-library
    run([str(TOOLS / "hypre_h3_analyze.py"),
         "--hypre", str(RESULTS / "hypre_h2.jsonl"),
         "--amgcl", str(RESULTS / "p2_sweep.jsonl")],
        "hypre replication, default quality, cross-library consistency (Sec 6)")

    print("\n" + "=" * 70)
    print("Reproduction complete. Figures written to paper/figure/*.png.")
    print("Compare the printed numbers against the claims table above.")
    print("=" * 70)


if __name__ == "__main__":
    main()
