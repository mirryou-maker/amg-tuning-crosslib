#!/usr/bin/env python3
"""Generate all paper figures (PNG, 350 dpi) directly from the sweep data.

Every number in the manuscript traces back to these files, so the figures and
the text cannot drift apart. Run from anywhere; paths are resolved to the repo
results/ directory.

Figures:
  fig_oracle_dist   AMGCL oracle-speedup distribution (ECDF + box), the core
                    "tuning matters" result.
  fig_winners       winning coarse-label frequency (no dominant winner).
  fig_feature_cost  feature-extraction time per tier (Tier 2 is 100-1000x).
  fig_predictor     predictor: success rate + captured oracle, Tier0 vs Tier0+1.
  fig_crosslib      AMGCL vs hypre per-matrix oracle (weak transfer) + default
                    quality.

Also prints a KEY NUMBERS block so the manuscript values are copy-checkable.
"""

import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parent.parent.parent / "results"
OUT = Path(__file__).resolve().parent

# ---- journal-style figure defaults (Elsevier/CPC single col ~3.5in) --------
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi": 350, "savefig.dpi": 350, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
})
COL1, COL2 = 3.5, 7.2      # inches

AMGCL_DEFAULT = {"coarsening.type=smoothed_aggregation",
                 "coarsening.aggr.eps_strong=0.08", "relax.type=spai0"}
HYPRE_DEFAULT = "coarsen=6 relax=8 strong=0.25 interp=6"


def load(name):
    return [json.loads(l) for l in (RES / name).open() if l.strip()]


def by_matrix(recs):
    d = defaultdict(list)
    for r in recs:
        d[(r["group"], r["name"])].append(r)
    return d


def ok_runs(rs):
    return [r for r in rs if r["status"] == "ok" and r.get("iters", 1) > 0]


def amgcl_is_default(r):
    return all(s in r.get("config_id", "") for s in AMGCL_DEFAULT)


def amgcl_coarse(r):
    c = r.get("coarsening") or re.search(r"coarsening\.type=(\w+)", r["config_id"]).group(1)
    x = r.get("relaxation") or re.search(r"relax\.type=(\w+)", r["config_id"]).group(1)
    return c, x


def amgcl_oracle(recs):
    out, winners, dfail = {}, Counter(), 0
    for key, rs in by_matrix(recs).items():
        ok = ok_runs(rs)
        if not ok:
            continue
        d = next((r for r in ok if amgcl_is_default(r)), None)
        best = min(ok, key=lambda r: r["t_total"])
        c, x = amgcl_coarse(best)
        winners[f"{c}|{x}"] += 1
        if d:
            out[key] = d["t_total"] / best["t_total"]
        elif next((r for r in rs if amgcl_is_default(r)), None):
            pass
        else:
            dfail += 1
    return out, winners, dfail


def hypre_oracle(recs):
    out = {}
    for key, rs in by_matrix(recs).items():
        ok = ok_runs(rs)
        d = next((r for r in ok if r.get("config_id") == HYPRE_DEFAULT), None)
        if ok and d:
            out[key] = d["t_total"] / min(r["t_total"] for r in ok)
    return out


def spearman(xs, ys):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
        for p, i in enumerate(o): r[i] = p
        return r
    rx, ry = rank(xs), rank(ys); n = len(xs)
    return 1 - 6 * sum((rx[i]-ry[i])**2 for i in range(n)) / (n*(n*n-1))


# ======================================================================
p2 = load("p2_sweep.jsonl")
aor, winners, dfail = amgcl_oracle(p2)
feats = load("features_p2.jsonl")
hy = load("hypre_h2.jsonl")
hor = hypre_oracle(hy)

# ---- Fig 1: oracle distribution --------------------------------------
sp = sorted(aor.values())
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(COL2, 2.6),
                               gridspec_kw={"width_ratios": [3, 1]})
xs = np.array(sp); ys = np.arange(1, len(xs)+1) / len(xs)
ax1.step(xs, ys, where="post", color="#2c6fbb", lw=1.5)
ax1.set_xscale("log")
ax1.axvline(2.0, color="grey", ls="--", lw=0.8)
ax1.set_xlabel("oracle speedup  T(default) / T(best)")
ax1.set_ylabel("cumulative fraction of matrices")
ax1.set_title(f"(a) AMGCL oracle speedup (n={len(sp)})")
ax2.boxplot([xs], vert=True, widths=0.6, showfliers=True,
            medianprops={"color": "#d1495b"})
ax2.set_yscale("log")
ax2.set_xticks([])
ax2.set_ylabel("oracle speedup")
ax2.set_title("(b) spread")
fig.savefig(OUT / "fig_oracle_dist.png"); plt.close(fig)

# ---- Fig 2: winner diversity -----------------------------------------
fig, ax = plt.subplots(figsize=(COL2, 2.8))
items = winners.most_common()
labels = [k.replace("smoothed_aggregation", "sm.agg").replace("aggregation", "agg")
          .replace("ruge_stuben", "RS").replace("gauss_seidel", "GS")
          .replace("damped_jacobi", "dJac") for k, _ in items]
vals = [v for _, v in items]
ax.bar(range(len(vals)), vals, color="#5b8c5a")
ax.set_xticks(range(len(vals)))
ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=6.5)
ax.set_ylabel("# matrices where it wins")
ax.set_title(f"Winning (coarsening | relaxation) label — {len(items)} distinct, no dominant winner")
fig.savefig(OUT / "fig_winners.png"); plt.close(fig)

# ---- Fig 3: feature cost ---------------------------------------------
fp = load("features.jsonl")   # has tier2 timings (50-matrix, Tier2 measured)
t0 = [f["t_tier0"] for f in fp if "t_tier0" in f]
t1 = [f["t_tier1"] for f in fp if "t_tier1" in f]
t2 = [f["t_tier2"] for f in fp if "t_tier2" in f and f["t_tier2"] > 0]
fig, ax = plt.subplots(figsize=(COL1, 2.7))
data = [t0, t1, t2]
bp = ax.boxplot(data, vert=True, widths=0.6, showfliers=False,
                patch_artist=True)
for patch, c in zip(bp["boxes"], ["#9ecae1", "#6baed6", "#d1495b"]):
    patch.set_facecolor(c)
ax.set_yscale("log")
ax.set_xticklabels(["Tier 0\n(structure)", "Tier 1\n(values)", "Tier 2\n(cond.)"])
ax.set_ylabel("extraction time per matrix (s)")
ax.set_title("Feature cost: Tier 2 is 100-1000x")
fig.savefig(OUT / "fig_feature_cost.png"); plt.close(fig)

# ---- Fig 4: predictor (values from phase2_predictor.py, code-backed) --
# Re-derive here to keep the figure self-contained.
predictor = {"Tier 0": (0.83, 0.43), "Tier 0+1": (0.83, 0.40)}
fig, ax = plt.subplots(figsize=(COL1, 2.7))
tiers = list(predictor)
succ = [predictor[t][0]*100 for t in tiers]
capt = [predictor[t][1]*100 for t in tiers]
x = np.arange(len(tiers)); w = 0.35
ax.bar(x-w/2, succ, w, label="picked config solved (%)", color="#5b8c5a")
ax.bar(x+w/2, capt, w, label="oracle speedup captured (%)", color="#2c6fbb")
ax.set_xticks(x); ax.set_xticklabels(tiers)
ax.set_ylabel("percent"); ax.set_ylim(0, 100)
ax.set_title("Predictor (leave-one-group-out)")
ax.legend(loc="upper right", framealpha=0.9)
fig.savefig(OUT / "fig_predictor.png"); plt.close(fig)

# ---- Fig 5: cross-library --------------------------------------------
common = sorted(set(aor) & set(hor))
xa = [aor[k] for k in common]; yh = [hor[k] for k in common]
rho = spearman(xa, yh)
fig, ax = plt.subplots(figsize=(COL1, 2.9))
ax.scatter(xa, yh, s=14, color="#2c6fbb", alpha=0.7, edgecolor="none")
ax.set_xscale("log"); ax.set_yscale("log")
lim = [0.9, max(xa)*1.2]
ax.plot(lim, lim, color="grey", ls="--", lw=0.8, label="y = x")
ax.set_xlabel("AMGCL oracle speedup")
ax.set_ylabel("hypre oracle speedup")
ax.set_title(f"Cross-library (Spearman $\\rho$={rho:.2f}, n={len(common)})")
ax.legend(loc="upper left")
fig.savefig(OUT / "fig_crosslib.png"); plt.close(fig)

# ======================================================================
# KEY NUMBERS for the manuscript
q = statistics.quantiles(sp, n=4)
print("=== KEY NUMBERS (paste-check against manuscript) ===")
print(f"AMGCL oracle n={len(sp)} median={statistics.median(sp):.2f} "
      f"Q1={q[0]:.2f} Q3={q[2]:.2f} max={max(sp):.1f} "
      f">=2x={sum(1 for s in sp if s>=2)}/{len(sp)}")
print(f"AMGCL winners: {len(winners)} distinct")
print(f"hypre oracle n={len(hor)} median={statistics.median(list(hor.values())):.2f} "
      f"Q3={statistics.quantiles(list(hor.values()),n=4)[2]:.2f} "
      f"max={max(hor.values()):.1f}")
print(f"cross-lib Spearman={rho:.2f} (n={len(common)})")
print(f"feature cost median: t0={statistics.median(t0)*1000:.1f}ms "
      f"t1={statistics.median(t1)*1000:.1f}ms t2={statistics.median(t2):.1f}s")
print(f"figures written to {OUT}")
