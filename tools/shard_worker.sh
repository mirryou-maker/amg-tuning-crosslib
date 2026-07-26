#!/bin/sh
# One MPI rank per node. Each rank runs sweep.py over its own matrix shard and
# writes to a per-rank output file, so nodes never append to the same file
# (a shared-filesystem append race would corrupt lines). The parent PBS script
# merges the rank files after mpirun returns.
#
# Parameterized via environment so every phase-1b job reuses it:
#   SWEEP_OUT     output prefix (default sweep_iremb) -> data/<prefix>.rankN.jsonl
#   SWEEP_TOL     solver tolerance          (default 1e-8)
#   SWEEP_TIMEOUT max per-run seconds       (default 120)
#   SWEEP_POLICY  fixed | adaptive          (default adaptive)
#   SWEEP_EXTRA   extra key=value list for every config (e.g. scale=1)
#   SWEEP_ONLY    substring filter for matrices (optional)
#
# Rank index comes from OpenMPI's environment; total is argv[1].

set -e
N="$1"
RANK="${OMPI_COMM_WORLD_RANK:-0}"

. /etc/profile.d/modules.sh
module load ANACONDA/1-2024.10

cd "$(dirname "$0")/.."          # -> $WORK

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores

PREFIX="${SWEEP_OUT:-sweep_iremb}"
OUT="data/${PREFIX}.rank${RANK}.jsonl"
echo "rank $RANK/$N on $(hostname) -> $OUT (tol=${SWEEP_TOL:-1e-8} policy=${SWEEP_POLICY:-adaptive} extra='${SWEEP_EXTRA:-}')"

set -- \
    --matrices "${SWEEP_MATRICES:-data/pilot_matrices_final.csv}" \
    --out "$OUT" \
    --shard "${RANK}/${N}" \
    --reps 1 --repeat 3 --jobs 1 \
    --tol "${SWEEP_TOL:-1e-8}" \
    --timeout "${SWEEP_TIMEOUT:-120}" \
    --timeout-policy "${SWEEP_POLICY:-adaptive}"
[ -n "${SWEEP_EXTRA:-}" ] && set -- "$@" --extra "$SWEEP_EXTRA"
[ -n "${SWEEP_ONLY:-}" ] && set -- "$@" --only "$SWEEP_ONLY"

python tools/sweep.py "$@"
echo "rank $RANK done on $(hostname)"
