# Tuning algebraic multigrid: a cross-library empirical study

Reproducibility artifact for the paper *"Tuning Internal Parameters of
Algebraic Multigrid: A Cross-Library Study of Achievable Speedup and
Cheap-Feature Prediction"* (Chun-Yeol You, DGIST).

**New here?** See [`USAGE.md`](USAGE.md) for a two-step quick start:
reproduce the paper, or get a parameter recommendation for your own matrix
(`python tools/recommend.py your_matrix.mtx`).

We sweep the internal parameters of two standard algebraic multigrid (AMG)
libraries — [AMGCL](https://github.com/ddemidov/amgcl) and
[hypre BoomerAMG](https://github.com/hypre-space/hypre) — over 150 SuiteSparse
matrices, quantify the achievable *oracle* speedup, build a cheap-feature
predictor evaluated with a strict leave-one-group-out protocol, and
cross-validate the findings across the two libraries.

## Two levels of reproduction

**Level 1 — figures and numbers from released data (minutes, any laptop).**
All 18,600 solve results are released as JSONL in [`results/`](results/), so
every figure, table, and headline number reproduces without a cluster:

```bash
pip install -r requirements.txt
python reproduce.py
```

This regenerates `paper/figure/*.png` and prints every quantitative claim in
the paper next to the manuscript value it backs.

**Level 2 — full end-to-end (build solvers, fetch matrices, re-run the sweep).**
See [`docs/REPRODUCE.md`](docs/REPRODUCE.md). Requires a C/C++ toolchain, Boost
headers, a sequential hypre build, and a compute cluster for the ~18,600 solves.

## Headline results

| | AMGCL | hypre BoomerAMG |
|---|---|---|
| oracle speedup (median / Q3 / max) | 2.4× / 6.4× / 292× | 1.6× / 2.1× / 31× |
| matrices with a distinct winner | 14 labels | 22 labels |
| default fails but tuning succeeds | 31 | 16 |

- A gradient-boosted predictor picks a configuration that **solves 83 %** of
  held-out matrices and **captures ~43 %** of the oracle speedup under
  leave-one-group-out evaluation.
- **Near-free structural (Tier 0) features suffice**; spectral (Tier 2)
  features cost 100–1000× more and add nothing.
- Tuning benefit **does not transfer between libraries** (Spearman ρ = 0.24)
  and is largely a **weak-default artifact**; intrinsic solvability *is*
  library-independent (hard-matrix Jaccard 0.76).

## Repository layout

```
reproduce.py           Level-1 driver: released data -> figures + numbers
requirements.txt       Python dependencies for Level 1
LICENSE                MIT
data/                  matrix manifests (CSV); matrices themselves are fetched
results/               RELEASED DATA: sweep JSONL + per-phase analysis notes
tools/                 all code (see below)
paper/                 manuscript (article + ACM TOMS versions) and figures
docs/REPRODUCE.md      Level-2 full pipeline
```

### `tools/` (code)

| purpose | files |
|---|---|
| AMGCL runner + sweep | `runner.cpp`, `sweep.py`, `sweep_generic.pbs`, `shard_worker.sh` |
| hypre runner + sweep | `hypre_runner.c`, `hypre_sweep.py`, `hypre_sweep.pbs`, `hypre_shard_worker.sh` |
| matrix selection / fetch | `select_phase2.py`, `fetch_suitesparse.py` |
| features + predictor | `compute_features.py`, `phase2_predictor.py`, `poc_predictor.py` |
| **recommend for your matrix** | `recommend.py` (see [`USAGE.md`](USAGE.md)) |
| analysis | `analyze.py`, `hypre_h3_analyze.py`, `hypre_analyze.py` |
| cluster build / probes | `build_iremb.sh`, `build_hypre` (see docs), `cv_probe.py` |

### `results/` (released data)

| file | contents |
|---|---|
| `p2_sweep.jsonl` | AMGCL sweep: 150 matrices × 88 configs = 13,200 solves |
| `hypre_h2.jsonl` | hypre sweep: 150 matrices × 36 configs = 5,400 solves |
| `features_p2.jsonl` | Tier 0/1 features for the 150 matrices |
| `features.jsonl` | Tier 0/1/2 features (Tier 2 timings, 50-matrix subset) |
| `PHASE*_*.md`, `HYPRE_*.md` | per-phase analysis notes |

Each JSONL record is one solve: matrix, configuration, status
(`ok`/`diverged`/`maxiter`/`timeout`/`error`), iterations, residual, and
`t_setup`/`t_solve`/`t_total` (seconds, minimum over in-process repeats).

## Citing

If you use this artifact, please cite the paper (see `paper/`).

## License

MIT (see `LICENSE`). Third-party solvers (AMGCL, hypre) and the SuiteSparse
matrices are used but not redistributed; see `LICENSE` and
`tools/fetch_suitesparse.py`.
