#!/usr/bin/env python3
"""Concept diagram of the study: what we did, end to end.
Writes results/concept_diagram.png and a copy into paper/figure/ for the
manuscript. English labels for font portability.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "concept_diagram.png"
OUT2 = HERE.parent / "paper" / "figure" / "concept_diagram.png"

# Fonts enlarged ~1.5x vs the first version; box sizes unchanged.
FS_TITLE, FS_BODY, FS_SUP, FS_LAB = 14, 11.5, 16, 12

fig, ax = plt.subplots(figsize=(12.5, 7.0))
ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")

def box(x, y, w, h, title, lines, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.5",
                 fc=fc, ec="#333", lw=1.2))
    ax.text(x+w/2, y+h-2.4, title, ha="center", va="top",
            fontsize=FS_TITLE, fontweight="bold")
    ax.text(x+w/2, y+h-6.6, "\n".join(lines), ha="center", va="top",
            fontsize=FS_BODY, color="#222", linespacing=1.35)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=18, lw=1.7, color="#444"))

BLUE="#dbe8f5"; GREEN="#dcecd9"; ORANGE="#f6e6cf"; PINK="#f3dbe0"

# Row 1
box(2, 44, 20, 15, "1. Matrix set",
    ["150 (→300)", "SuiteSparse matrices,", "many per group"], BLUE)
box(28, 44, 22, 15, "2. Parameter sweep",
    ["88 configs / matrix", "(coarsen×smooth", "×threshold)", "= 13,200 solves"], GREEN)
box(56, 44, 22, 15, "3. Oracle speedup",
    ["T(default)/T(best)", "median 2.4×, max 292×", "winner varies (14)"], ORANGE)
arrow(22, 51.5, 28, 51.5)
arrow(50, 51.5, 56, 51.5)

# Row 2
box(56, 24, 22, 15, "4. Cheap features",
    ["Tier 0 structure (free)", "Tier 1, Tier 2 (100-1000×)", "→ Tier 0 suffices"], BLUE)
box(28, 24, 22, 15, "5. Predictor (GBDT)",
    ["features + config", "→ solve? + time", "leave-one-group-out"], GREEN)
box(2, 24, 20, 15, "6. Result",
    ["solves 83% (89%)", "captures 43% (49%)", "of oracle"], ORANGE)
arrow(67, 44, 67, 39)          # oracle -> features
arrow(56, 31.5, 50, 31.5)      # features -> predictor
arrow(28, 31.5, 22, 31.5)      # predictor -> result

# Row 3: cross-library, moved LEFT so arrows from boxes 5 and 6 point into it
box(4, 2, 65, 15, "7. Cross-library validation (hypre BoomerAMG)",
    ["tunability replicates but MODEST (median 1.6×):",
     "magnitude is default-driven, not intrinsic (ρ=0.24)",
     "difficulty IS library-independent (Jaccard 0.76)",
     "‘will it solve?’ transfers; oracle magnitude does not"], PINK)
arrow(12, 24, 12, 17.2)        # from box 6
arrow(40, 24, 40, 17.2)        # from box 5
ax.text(26, 20.5, "does it generalise?", ha="center", va="center",
        fontsize=FS_LAB, fontstyle="italic", color="#7a2a3a", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#c98", alpha=0.95))

ax.text(50, 60.8, "Tuning internal AMG parameters: how much to gain, "
        "can it be predicted, does it generalise?",
        ha="center", fontsize=FS_SUP, fontweight="bold")

fig.savefig(OUT, dpi=300, bbox_inches="tight")
fig.savefig(OUT2, dpi=300, bbox_inches="tight")
print("wrote", OUT, "and", OUT2)
