#!/usr/bin/env python3
"""Concept diagram of the study: what we did, end to end.
Produces results/concept_diagram.png. English labels for font portability.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path(__file__).resolve().parent / "concept_diagram.png"
plt.rcParams.update({"font.size": 9})

fig, ax = plt.subplots(figsize=(11, 6.2))
ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")

def box(x, y, w, h, title, lines, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.5",
                 fc=fc, ec="#333", lw=1.2))
    ax.text(x+w/2, y+h-2.6, title, ha="center", va="top", fontsize=9.5, fontweight="bold")
    ax.text(x+w/2, y+h-6.2, "\n".join(lines), ha="center", va="top", fontsize=7.6, color="#222")

def arrow(x1, y1, x2, y2, label=None, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=14, lw=1.4, color="#444"))
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2+1.3, label, ha="center", fontsize=7, color="#555", style="italic")

BLUE="#dbe8f5"; GREEN="#dcecd9"; ORANGE="#f6e6cf"; PINK="#f3dbe0"; GREY="#e8e8e8"

# Row 1: input -> sweep -> oracle
box(2, 44, 20, 15, "1. Matrix set",
    ["150 (→300) SuiteSparse", "matrices, many per group",
     "graph collections excluded"], BLUE)
box(28, 44, 22, 15, "2. Parameter sweep",
    ["88 AMGCL configs / matrix", "coarsening x smoother x", "strength threshold",
     "= 13,200 timed solves"], GREEN)
box(56, 44, 22, 15, "3. Oracle speedup",
    ["T(default) / T(best)", "median 2.4x, max 292x", "winner varies (14 labels)",
     "31 matrices: default fails"], ORANGE)
arrow(22, 51.5, 28, 51.5)
arrow(50, 51.5, 56, 51.5)

# down to features/predictor
box(56, 24, 22, 15, "4. Cheap features",
    ["Tier 0 structure (free)", "Tier 1 values", "Tier 2 spectral (100-1000x)",
     "-> Tier 0 alone suffices"], BLUE)
box(28, 24, 22, 15, "5. Predictor (GBDT)",
    ["features + config", "-> solve? + log time", "leave-one-group-out",
     "pick best per matrix"], GREEN)
box(2, 24, 20, 15, "6. Result",
    ["solves 83% (89% @300)", "captures 43% (49%)", "of oracle speedup",
     "from near-free features"], ORANGE)
arrow(67, 44, 67, 39)
arrow(56, 31.5, 50, 31.5)
arrow(28, 31.5, 22, 31.5)

# Row 3: cross-library validation
box(20, 3, 60, 15, "7. Cross-library validation (hypre BoomerAMG, same 150 matrices)",
    ["tunability replicates but MODEST (median 1.6x) -- magnitude is DEFAULT-DRIVEN, not intrinsic",
     "cross-library oracle rank correlation only 0.24  ->  tuning benefit does NOT transfer",
     "hard-matrix overlap Jaccard 0.76  ->  intrinsic difficulty IS library-independent",
     "transfer test: 'will it solve?' transfers (+0.25..0.33);  oracle magnitude does not (rho~0.2)"], PINK)
arrow(12, 24, 12, 18.2)
arrow(40, 24, 40, 18.2, "does it\ngeneralise?")

ax.text(50, 60.6, "Tuning internal AMG parameters: how much to gain, can it be predicted, does it generalise?",
        ha="center", fontsize=11, fontweight="bold")

fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)
