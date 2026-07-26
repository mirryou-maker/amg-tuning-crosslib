#!/bin/sh
# One MPI rank per node -> runs hypre_sweep.py over its matrix shard, writing a
# per-rank file (no shared-file append race). Parent merges after mpirun joins.
# Rank index from OpenMPI env; total is argv[1].

set -e
N="$1"
RANK="${OMPI_COMM_WORLD_RANK:-0}"

. /etc/profile.d/modules.sh
module load ANACONDA/1-2024.10

cd "$(dirname "$0")/.."          # -> $WORK
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores

OUT="data/${SWEEP_OUT:-hypre_sweep}.rank${RANK}.jsonl"
echo "rank $RANK/$N on $(hostname) -> $OUT"

python tools/hypre_sweep.py \
    --matrices "${SWEEP_MATRICES:-data/phase2_matrices.csv}" \
    --out "$OUT" \
    --shard "${RANK}/${N}" \
    --reps 1 --repeat 3 --timeout "${SWEEP_TIMEOUT:-30}"
echo "rank $RANK done on $(hostname)"
