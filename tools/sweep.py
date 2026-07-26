#!/usr/bin/env python3
"""Phase 1 sweep driver: run every (matrix, parameter combination) and append
one JSON line per run.

Timeout is enforced here rather than inside runner.cpp because AMGCL exposes
no time-based abort hook, and an external kill also covers hangs during setup
and hard crashes -- neither of which an in-process guard would catch.

Results are appended immediately, so a killed job leaves usable partial data.
Runs already present in the output file are skipped, making the sweep
resumable.

Usage:
    python sweep.py --matrices ../data/pilot_matrices_final.csv \
                    --out ../data/sweep_results.jsonl --reps 3
"""

import argparse
import concurrent.futures as cf
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "build" / ("runner.exe" if os.name == "nt" else "runner")
CACHE = ROOT / "data" / "matrices"

# --- parameter grid -------------------------------------------------------
#
# The intra-solver knobs are the point of the study, so the grid crosses
# coarsening *strength* and relaxation *fill*, not just the catalog choice
# that prior selection work stops at.
#
# Strength-threshold paths differ by family and cannot be unified:
#   aggregation / smoothed_aggregation -> precond.coarsening.aggr.eps_strong
#   ruge_stuben                        -> precond.coarsening.eps_strong
# AMGCL defaults are 0.08 and 0.25 respectively; both are in the sweep so the
# default is measured on the same footing as everything else.

AGGR_EPS = [0.01, 0.03, 0.08, 0.15]      # 0.08 is the AMGCL default
RS_EPS = [0.10, 0.25, 0.50]              # 0.25 is the AMGCL default

RELAXATIONS = [
    {"precond.relax.type": "spai0"},
    {"precond.relax.type": "spai1"},
    {"precond.relax.type": "damped_jacobi"},
    {"precond.relax.type": "gauss_seidel"},
    {"precond.relax.type": "chebyshev"},
    {"precond.relax.type": "ilu0"},
    {"precond.relax.type": "iluk", "precond.relax.k": 1},
    {"precond.relax.type": "iluk", "precond.relax.k": 2},
]


def coarsening_configs():
    cfgs = []
    for c in ("smoothed_aggregation", "aggregation"):
        for e in AGGR_EPS:
            cfgs.append({"precond.coarsening.type": c,
                         "precond.coarsening.aggr.eps_strong": e})
    for e in RS_EPS:
        cfgs.append({"precond.coarsening.type": "ruge_stuben",
                     "precond.coarsening.eps_strong": e})
    return cfgs


def grid():
    """Every (coarsening, relaxation) parameter combination."""
    return [{**c, **r} for c in coarsening_configs() for r in RELAXATIONS]


def config_id(cfg):
    """Stable identity for resume/dedup."""
    return " ".join(f"{k}={v}" for k, v in sorted(cfg.items()))


def env_stamp():
    """Recorded with every run: mixing timings across machines is a real risk."""
    return {
        "host": platform.node(),
        "system": f"{platform.system()} {platform.release()}",
        "omp_threads": os.environ.get("OMP_NUM_THREADS", ""),
    }


def key(rec):
    return (rec["group"], rec["name"], rec.get("config_id", ""), rec["rep"])


def load_done(out_path):
    done = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue          # tolerate a truncated final line
                if all(k in r for k in ("group", "name", "config_id", "rep")):
                    done.add(key(r))
    return done


def run_one(mtx, cfg, solver, tol, maxiter, timeout, repeat):
    cmd = [str(RUNNER), str(mtx),
           f"solver.type={solver}", f"solver.tol={tol}",
           f"solver.maxiter={maxiter}", f"repeat={repeat}"]
    cmd += [f"{k}={v}" for k, v in sorted(cfg.items())]
    t0 = time.monotonic()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "t_wall": time.monotonic() - t0,
                "message": f"killed after {timeout}s"}
    wall = time.monotonic() - t0

    out = p.stdout.strip().splitlines()
    for line in reversed(out):
        if line.startswith("{"):
            try:
                rec = json.loads(line)
                rec["t_wall"] = wall
                return rec
            except json.JSONDecodeError:
                pass
    return {"status": "crash", "t_wall": wall,
            "message": (p.stderr or "no JSON on stdout")[:300],
            "returncode": p.returncode}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrices", type=Path,
                    default=ROOT / "data" / "pilot_matrices_final.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "sweep_results.jsonl")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--maxiter", type=int, default=1000)
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="wall-clock seconds per run before the process is killed")
    ap.add_argument("--timeout-policy", choices=["fixed", "adaptive"],
                    default="fixed",
                    help="adaptive: per-matrix budget 30 + 40*(nnz/1e6)s, "
                         "clamped to [30, --timeout]. Phase 1 found a flat 30s "
                         "cut misclassifies big-but-solvable matrices "
                         "(pdb1HYS pattern) while a flat 120s wastes hours on "
                         "small hopeless ones.")
    ap.add_argument("--extra", default="",
                    help="comma list of key=value appended to EVERY config "
                         "(e.g. 'scale=1'); recorded in config_id")
    ap.add_argument("--repeat", type=int, default=5,
                    help="in-process repeats; runner reports the minimum")
    ap.add_argument("--only", help="restrict to group/name substrings (comma list)")
    ap.add_argument("--shard", default="0/1",
                    help="i/n: process only matrices whose index %% n == i. "
                         "Distributes matrices across nodes so each node runs "
                         "jobs=1 (no intra-node bandwidth contention) while "
                         "still using the whole allocation. A matrix's configs "
                         "all stay on one node, so its oracle ratio is "
                         "measured under identical conditions.")
    ap.add_argument("--jobs", type=int, default=1,
                    help="concurrent single-threaded runners. >1 fills a node "
                         "but sparse solves are memory-bandwidth bound, so "
                         "concurrent runs perturb each other's timings; "
                         "validate CV at the chosen value before trusting times")
    args = ap.parse_args()

    if not RUNNER.exists():
        sys.exit(f"runner not built: {RUNNER}")

    rows = list(csv.DictReader(args.matrices.open()))
    if args.only:
        pats = [s.strip() for s in args.only.split(",") if s.strip()]

        def match(r, p):
            # "group/name" -> exact; bare word -> substring. Exact form exists
            # because substrings bite: "Lin" also matches Linux_call_graph.
            if "/" in p:
                return p == f"{r['group']}/{r['name']}"
            return p in r["group"] or p in r["name"]

        rows = [r for r in rows if any(match(r, p) for p in pats)]

    si, sn = (int(x) for x in args.shard.split("/"))
    if sn > 1:
        rows = [r for idx, r in enumerate(rows) if idx % sn == si]
        print(f"shard {si}/{sn}: {len(rows)} matrices on this node")

    done = load_done(args.out)
    stamp = env_stamp()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    configs = grid()
    if args.extra:
        extra = dict(kv.split("=", 1) for kv in args.extra.split(",") if kv)
        configs = [{**c, **extra} for c in configs]

    def budget(row):
        if args.timeout_policy == "fixed":
            return args.timeout
        nnz = int(row.get("nnz", 0) or 0)
        return max(30.0, min(args.timeout, 30.0 + 40.0 * nnz / 1e6))

    todo = []
    for r in rows:
        mtx = CACHE / r["group"] / r["name"] / f"{r['name']}.mtx"
        if not mtx.exists():
            cands = sorted((CACHE / r["group"] / r["name"]).glob("*.mtx"))
            if not cands:
                print(f"  MISSING {r['group']}/{r['name']}", file=sys.stderr)
                continue
            mtx = cands[0]
        for cfg in configs:
            cid = config_id(cfg)
            for rep in range(args.reps):
                if (r["group"], r["name"], cid, rep) in done:
                    continue
                todo.append((r, mtx, cfg, cid, rep))

    print(f"{len(rows)} matrices x {len(configs)} configs x {args.reps} reps; "
          f"{len(todo)} runs to do ({len(done)} already done), "
          f"jobs={args.jobs}, timeout={args.timeout}s")

    def execute(item):
        r, mtx, cfg, cid, rep = item
        rec = run_one(mtx, cfg, r["solver"], args.tol, args.maxiter,
                      budget(r), args.repeat)
        rec.update({
            "group": r["group"], "name": r["name"],
            "stratum": r["stratum"], "graphlike": int(r["graphlike"]),
            "mm_symmetry": r.get("mm_symmetry", ""),
            "config_id": cid, "rep": rep, "jobs": args.jobs, **stamp,
        })
        return rec

    done_n = 0
    with args.out.open("a") as f:
        if args.jobs > 1:
            with cf.ThreadPoolExecutor(max_workers=args.jobs) as pool:
                # Threads only wait on subprocesses, so the GIL is not a factor.
                for rec in pool.map(execute, todo):
                    done_n += 1
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    print(f"[{done_n}/{len(todo)}] {rec['group']}/{rec['name']} "
                          f"{rec.get('coarsening')}/{rec.get('relaxation')} "
                          f"-> {rec['status']} "
                          f"{rec.get('t_total', rec.get('t_wall', 0)):.3f}s")
        else:
            for item in todo:
                rec = execute(item)
                done_n += 1
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(f"[{done_n}/{len(todo)}] {rec['group']}/{rec['name']} "
                      f"{rec.get('coarsening')}/{rec.get('relaxation')} "
                      f"-> {rec['status']} "
                      f"{rec.get('t_total', rec.get('t_wall', 0)):.3f}s")


if __name__ == "__main__":
    main()
