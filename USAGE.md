# Quick start / user guide

This guide covers the two things a user typically wants:

1. **Reproduce** the paper's figures and numbers from the released data.
2. **Get a parameter recommendation** for *your own* matrix.

Neither needs the compute cluster or a built solver — only a standard
scientific Python stack.

```bash
pip install -r requirements.txt      # numpy, scipy, scikit-learn, matplotlib
```

---

## 1. Reproduce the paper (minutes)

```bash
python reproduce.py
```

Regenerates every figure into `paper/figure/*.png` and prints each headline
number next to the manuscript claim it backs, straight from the released JSONL
sweep data in `results/`. See [`docs/REPRODUCE.md`](docs/REPRODUCE.md) for the
full end-to-end pipeline (build AMGCL/hypre, fetch matrices, re-run the sweep).

---

## 2. Recommend parameters for your matrix

Given a matrix in Matrix Market format, `recommend.py` extracts cheap features,
trains the predictor on all released sweep data, and prints a ranked short-list
of configurations with ready-to-paste parameter strings.

```bash
# AMGCL (default). Auto-detects SPD from the .mtx symmetry.
python tools/recommend.py your_matrix.mtx

# hypre BoomerAMG
python tools/recommend.py your_matrix.mtx --library hypre

# force SPD (use CG), and show more candidates
python tools/recommend.py your_matrix.mtx --spd --top 8
```

Example output (AMGCL):

```
Top 5 recommended configurations (amgcl, solver=cg):
rank  P(solve)  pred.time  configuration
   1      0.98      0.391  coarsening.aggr.eps_strong=0.01 coarsening.type=aggregation relax.type=ilu0
   2      0.99      0.403  coarsening.aggr.eps_strong=0.01 coarsening.type=aggregation relax.type=gauss_seidel
   ...
```

### Using a recommendation

The `configuration` column is the AMGCL runtime parameter string. With the
`boost::property_tree` interface, set for example:

```cpp
prm.put("precond.coarsening.type", "aggregation");
prm.put("precond.coarsening.aggr.eps_strong", 0.01);
prm.put("precond.relax.type", "ilu0");
prm.put("solver.type", "cg");   // "bicgstab" if not SPD
```

For hypre, the columns map to the BoomerAMG setters:

```c
HYPRE_BoomerAMGSetCoarsenType(amg, 8);      /* coarsen=8  (PMIS) */
HYPRE_BoomerAMGSetRelaxType(amg, 18);       /* relax=18   (L1-Jacobi) */
HYPRE_BoomerAMGSetStrongThreshold(amg, 0.5);/* strong=0.5 */
```

### What to expect (be realistic)

The predictor is a **heuristic**, evaluated honestly in the paper with a
leave-one-group-out protocol:

- the top pick actually solves about **83-89%** of unseen matrices;
- it captures a median **~43-49%** of the achievable oracle speedup over the
  library default.

So it is a good starting point, not a guarantee. **Try the top few candidates
and keep the fastest** — that is exactly how the paper's "captured oracle"
metric is defined. If none of the top picks converges, your matrix may fall in
the hard set that no configuration solves (see the paper's failure analysis).

### Notes

- Features are near-free (Tier 0/1); the expensive Tier 2 features are not used
  by `recommend.py` because the paper found they add no accuracy.
- The predictor is trained on 150-300 SuiteSparse matrices in the size band
  `1e4 <= n <= 5e5`. Very different matrices (much larger, or from domains far
  outside the training set) are extrapolation — treat recommendations with more
  caution there.
