# Full end-to-end reproduction (Level 2)

Level 1 (`python reproduce.py`) regenerates every figure and number from the
released JSONL data and needs no solvers or cluster. This document covers the
full pipeline: building both solvers, fetching the matrices, and re-running the
~18,600 solves that produced that data. The heavy step (the sweeps) is designed
for a cluster; everything else runs on a workstation.

## 0. Prerequisites

- C++14 compiler (GCC 11+ used) and [Boost](https://www.boost.org) headers
  (header-only; AMGCL needs only `boost/property_tree`).
- C compiler for hypre.
- Python stack from `requirements.txt`.
- For the cluster sweep: a scheduler (we used PBS Pro) and an MPI launcher for
  cross-node work distribution (the solvers themselves run single-process).

## 1. Fetch dependencies

```bash
# AMGCL (header-only)
git clone --depth 1 https://github.com/ddemidov/amgcl ext/amgcl
# Boost headers (property_tree); any recent Boost works
#   place headers so that ext/boost/boost/property_tree/ptree.hpp exists

# hypre, built SEQUENTIALLY (no MPI in the solver -> avoids the main build risk)
git clone --depth 1 https://github.com/hypre-space/hypre ext/hypre
cd ext/hypre/src
cmake -B build -DHYPRE_ENABLE_MPI=OFF -DHYPRE_ENABLE_SHARED=OFF \
      -DCMAKE_INSTALL_PREFIX=$PWD/../install-seq
cmake --build build -j && cmake --install build
# note: the library lands in install-seq/lib64/libHYPRE.a
```

## 2. Build the runners

```bash
mkdir -p build
# AMGCL runner
g++ -O2 -std=c++14 -fopenmp -I ext/amgcl -I ext/boost \
    tools/runner.cpp -o build/runner
# hypre runner (needs the POSIX clock macro; links the static hypre lib)
H=ext/hypre/src/install-seq
gcc -O2 -std=c11 tools/hypre_runner.c \
    -I $H/include -L $H/lib64 -lHYPRE -lm -o build/hypre_runner
```

Smoke-test a single solve:

```bash
OMP_NUM_THREADS=1 ./build/runner <matrix.mtx> \
    precond.coarsening.type=smoothed_aggregation precond.relax.type=spai0 \
    solver.type=cg repeat=3
OMP_NUM_THREADS=1 ./build/hypre_runner <matrix.mtx> \
    coarsen=6 relax=8 strong=0.25 solver=cg repeat=3
```

## 3. Fetch the matrices

The SuiteSparse matrices are not redistributed; download them from the manifest:

```bash
python tools/fetch_suitesparse.py --manifest data/phase2_matrices.csv
```

The matrix selection itself (multiple matrices per group, graph collections
excluded) is reproduced by `tools/select_phase2.py`.

## 4. Run the sweeps

Measurement discipline (critical for valid timings): **one single-threaded
solver process per node**, matrices sharded across nodes. Co-running solves on
one node inflates timings via memory-bandwidth contention (a 36 ms solve
becomes 116 ms at 4 concurrent — verify on your hardware with
`tools/cv_probe.py`).

On a single workstation (small subsets):

```bash
OMP_NUM_THREADS=1 python tools/sweep.py \
    --matrices data/phase2_matrices.csv --out results/p2_sweep.jsonl \
    --reps 1 --repeat 3 --timeout 30 --jobs 1
OMP_NUM_THREADS=1 python tools/hypre_sweep.py \
    --matrices data/phase2_matrices.csv --out results/hypre_h2.jsonl \
    --reps 1 --repeat 3 --timeout 30
```

On a PBS cluster (full sweep, sharded across nodes) the provided job scripts
distribute the matrices; see `tools/sweep_generic.pbs` (AMGCL) and
`tools/hypre_sweep.pbs` (hypre), which call the per-node workers
`tools/shard_worker.sh` and `tools/hypre_shard_worker.sh`. Both are resumable:
re-running skips completed (matrix, config, rep) triples.

## 5. Features and analysis

```bash
python tools/compute_features.py --matrices data/phase2_matrices.csv \
    --out results/features_p2.jsonl --max-tier 1
# Tier-2 timings (expensive; 50-matrix subset in the paper):
python tools/compute_features.py --matrices data/pilot_matrices_final.csv \
    --out results/features.jsonl --max-tier 2
```

Then run `python reproduce.py` (Level 1) to regenerate figures and numbers from
the freshly produced JSONL.

## Notes

- Absolute timings are hardware-dependent; the paper's *relative* quantities
  (oracle speedups, rank correlations, overlaps) are what reproduce across
  machines. Cross-library absolute-time comparisons are read as trends only.
- The right-hand side is `b = A x*` with a ramp `x*_i = 1 + i/n` (not the
  all-ones vector, which is degenerate for zero-row-sum matrices).
