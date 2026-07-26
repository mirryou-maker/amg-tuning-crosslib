#!/bin/sh
# Stage everything the sweep needs onto iREMB.
#
# Run this on the LOCAL machine (Git-bash). It pushes sources, matrices, and
# Boost headers to the LOGIN node, which is the only node with a filesystem
# the compute nodes also see. Compute nodes have no network, so nothing here
# may run there -- this is a pure login-node transfer.
#
# Transfers go as tarballs, not `scp -r`: the Boost header tree alone is
# ~15,900 files, and per-file scp round-trips would take far longer than
# streaming one compressed stream. Matrices (~1.2 GB) are the reverse case --
# already few, large, poorly compressible files -- so they go uncompressed and
# are skipped when already present.
#
# Idempotent: safe to re-run. Re-running re-sends code (cheap) but skips the
# matrix corpus if the remote copy already exists.
#
# Usage:
#   SRC and DEST default as below; override via environment if needed.
#     REMOTE=iremb  tools/stage_iremb.sh
#     FORCE_MATRICES=1 tools/stage_iremb.sh   # re-send matrices anyway

set -e

REMOTE="${REMOTE:-iremb}"
SRC="${SRC:-$(cd "$(dirname "$0")/.." && pwd)}"
DEST="${DEST:-\$HOME/repos/Sparse-Matrix}"      # expanded on the remote side
BOOST_LOCAL="$SRC/ext/boost_extract/boost_1_87_0/boost"
AMGCL_LOCAL="$SRC/ext/amgcl"

for d in "$BOOST_LOCAL" "$AMGCL_LOCAL" "$SRC/data/matrices"; do
    [ -d "$d" ] || { echo "missing local dir: $d" >&2; exit 1; }
done

echo ">> remote=$REMOTE  dest=$DEST"

# 1. Directory skeleton on the remote.
ssh "$REMOTE" "mkdir -p $DEST/ext/boost $DEST/ext/amgcl $DEST/tools $DEST/data/matrices"

# 2. Code: tools/ + runner source. Small, always re-sent.
echo ">> code (tools/)"
tar -C "$SRC" -czf - tools | ssh "$REMOTE" "tar -C $DEST -xzf -"

# 3. AMGCL headers.
echo ">> amgcl headers"
tar -C "$SRC/ext/amgcl" -czf - amgcl | ssh "$REMOTE" "tar -C $DEST/ext/amgcl -xzf -"

# 4. Boost headers -> land at $DEST/ext/boost/boost (matches build_iremb.sh).
echo ">> boost headers (~15.9k files, compressed stream)"
tar -C "$SRC/ext/boost_extract/boost_1_87_0" -czf - boost \
    | ssh "$REMOTE" "tar -C $DEST/ext/boost -xzf -"

# 5. Matrix selection CSV.
echo ">> matrix selection csv"
tar -C "$SRC/data" -czf - pilot_matrices_final.csv \
    | ssh "$REMOTE" "tar -C $DEST/data -xzf -"

# 6. Matrix corpus (~1.2 GB). Skip if already present unless forced.
if [ "${FORCE_MATRICES:-0}" = "1" ]; then
    have=""
else
    have=$(ssh "$REMOTE" "ls $DEST/data/matrices 2>/dev/null | head -1")
fi
if [ -n "$have" ]; then
    echo ">> matrices already staged (found '$have'); skipping. FORCE_MATRICES=1 to resend."
else
    echo ">> matrices (~1.2 GB, uncompressed stream)"
    # -z omitted on purpose: .mtx text compresses little and would just burn CPU.
    tar -C "$SRC/data" -cf - matrices | ssh "$REMOTE" "tar -C $DEST/data -xf -"
fi

echo ">> verifying remote layout"
ssh "$REMOTE" "
    echo -n 'boost ptree: '; test -f $DEST/ext/boost/boost/property_tree/ptree.hpp && echo OK || echo MISSING
    echo -n 'amgcl hdr  : '; test -f $DEST/ext/amgcl/amgcl/make_solver.hpp && echo OK || echo MISSING
    echo -n 'runner src : '; test -f $DEST/tools/runner.cpp && echo OK || echo MISSING
    echo -n 'pbs script : '; test -f $DEST/tools/sweep.pbs && echo OK || echo MISSING
    echo -n 'matrices   : '; ls $DEST/data/matrices | wc -l | tr -d ' ';
"

echo ">> done. Next, on the LOGIN node ($REMOTE):"
echo "     ssh $REMOTE"
echo "     sh ~/repos/Sparse-Matrix/tools/build_iremb.sh        # build runner"
echo "     # validate the timing noise at the intended --jobs BEFORE the real run:"
echo "     cd ~/repos/Sparse-Matrix && OMP_NUM_THREADS=1 python tools/check_cv.py"
echo "     qsub tools/sweep.pbs                                 # submit the sweep"
