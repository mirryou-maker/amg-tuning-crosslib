#!/bin/sh
# Build the sweep runner on iREMB.
#
# Run this on the LOGIN node: compute nodes have no network, and this script
# does not need one either -- but the sources it compiles must already be in
# place, which is the login node's job.
#
# AMGCL is header-only and needs Boost headers only (property_tree). No
# compiled Boost libraries, no MPI, no CUDA.

set -e

. /etc/profile.d/modules.sh
module purge
module load DEVTOOLSET/11

SRC="${SRC:-$HOME/repos/Sparse-Matrix}"
AMGCL="$SRC/ext/amgcl"
BOOST="$SRC/ext/boost"

for d in "$AMGCL/amgcl" "$BOOST/boost"; do
    [ -d "$d" ] || { echo "missing: $d" >&2; exit 1; }
done

mkdir -p "$SRC/build"
g++ -O2 -std=c++14 -fopenmp \
    -I "$AMGCL" -I "$BOOST" \
    "$SRC/tools/runner.cpp" -o "$SRC/build/runner"

echo "built: $SRC/build/runner"
"$SRC/build/runner" 2>&1 | head -3 || true
