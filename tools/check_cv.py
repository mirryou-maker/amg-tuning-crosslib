#!/usr/bin/env python3
"""Measure timing reproducibility of the runner.

Runs each matrix in several separate processes and reports the coefficient of
variation of the reported t_total across processes. The gate for proceeding to
a full sweep is CV < 5%.

Usage:
    python check_cv.py --repeat 10 --procs 10
"""

import argparse
import json
import os
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "build" / ("runner.exe" if os.name == "nt" else "runner")
CACHE = ROOT / "data" / "matrices"

CASES = [
    ("Schmid", "thermal1", "cg"),
    ("Engwirda", "airfoil_2d", "bicgstab"),
    ("Bindel", "ted_B", "cg"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=10, help="in-process repeats")
    ap.add_argument("--procs", type=int, default=10, help="separate processes")
    ap.add_argument("--coarsening", default="smoothed_aggregation")
    ap.add_argument("--relaxation", default="spai0")
    args = ap.parse_args()

    print(f"{'matrix':<24} {'min(ms)':>9} {'CV%':>7} {'inproc CV%':>11} {'verdict':>9}")
    print("-" * 66)
    for group, name, solver in CASES:
        mtx = CACHE / group / name / f"{name}.mtx"
        totals, incv = [], []
        for _ in range(args.procs):
            p = subprocess.run(
                [str(RUNNER), str(mtx), args.coarsening, args.relaxation,
                 solver, "1e-8", "1000", str(args.repeat)],
                capture_output=True, text=True)
            line = [l for l in p.stdout.splitlines() if l.startswith("{")]
            if not line:
                print(f"{name:<24} FAILED: {p.stderr[:60]}")
                break
            rec = json.loads(line[-1])
            totals.append(rec["t_total"])
            incv.append(rec["t_total_cv"])
        if len(totals) < args.procs:
            continue

        mean = statistics.mean(totals)
        cv = statistics.pstdev(totals) / mean * 100 if mean else 0
        verdict = "PASS" if cv < 5.0 else "FAIL"
        print(f"{group + '/' + name:<24} {min(totals) * 1e3:9.2f} {cv:7.2f} "
              f"{statistics.mean(incv) * 100:11.2f} {verdict:>9}")


if __name__ == "__main__":
    main()
