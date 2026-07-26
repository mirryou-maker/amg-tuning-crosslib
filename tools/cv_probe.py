#!/usr/bin/env python3
"""Measure how timing reproducibility degrades under concurrency.

The sweep runs JOBS single-threaded runners at once to fill a node. Sparse
solves are memory-bandwidth bound, so concurrent runners contend for bandwidth
and perturb each other's timings -- exactly the noise that would masquerade as
a parameter effect. This probe quantifies that before the real run commits.

For each concurrency level it launches that many runners of the SAME matrix
simultaneously and reports the coefficient of variation of their reported
t_total across a batch. A level is acceptable if CV stays under ~5%.

Must run on a compute node under node-exclusive allocation; login-node or
shared timings are meaningless here.

Usage:
    OMP_NUM_THREADS=1 python cv_probe.py --levels 1,4,8,16 --batches 5
"""

import argparse
import concurrent.futures as cf
import json
import os
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "build" / ("runner.exe" if os.name == "nt" else "runner")
CACHE = ROOT / "data" / "matrices"

# A mid-size solve: big enough that bandwidth contention shows, small enough
# to probe quickly. Uses a config that converges fast on this matrix.
CASE = ("Cunningham", "qa8fm",
        ["precond.coarsening.type=ruge_stuben",
         "precond.coarsening.eps_strong=0.25",
         "precond.relax.type=damped_jacobi",
         "solver.type=cg"])


def one_run():
    group, name, cfg = CASE
    mtx = CACHE / group / name / f"{name}.mtx"
    cmd = [str(RUNNER), str(mtx), *cfg, "repeat=5"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    for line in reversed(p.stdout.splitlines()):
        if line.startswith("{"):
            return json.loads(line)["t_total"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="1,4,8,16")
    ap.add_argument("--batches", type=int, default=5,
                    help="how many concurrent batches to sample per level")
    args = ap.parse_args()

    levels = [int(x) for x in args.levels.split(",")]
    print(f"probe matrix: {CASE[0]}/{CASE[1]}")
    print(f"{'jobs':>5} {'runs':>5} {'min(ms)':>9} {'mean(ms)':>9} {'CV%':>7} {'verdict':>9}")
    print("-" * 50)

    for lvl in levels:
        times = []
        for _ in range(args.batches):
            with cf.ThreadPoolExecutor(max_workers=lvl) as pool:
                batch = [f.result() for f in
                         [pool.submit(one_run) for _ in range(lvl)]]
            times.extend(t for t in batch if t is not None)
        if not times:
            print(f"{lvl:5d}  no successful runs")
            continue
        mean = statistics.mean(times)
        cv = statistics.pstdev(times) / mean * 100 if mean else 0
        verdict = "PASS" if cv < 5.0 else "FAIL"
        print(f"{lvl:5d} {len(times):5d} {min(times) * 1e3:9.2f} "
              f"{mean * 1e3:9.2f} {cv:7.2f} {verdict:>9}")


if __name__ == "__main__":
    main()
