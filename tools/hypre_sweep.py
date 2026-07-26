#!/usr/bin/env python3
"""hypre BoomerAMG sweep driver -- the hypre counterpart of sweep.py.

Loops a grid of BoomerAMG parameters over a matrix list, calls build/hypre_runner
once per (matrix, config), and appends one JSON line each. Timeout is enforced
by killing the process (hypre exposes no time abort). Resumable: runs already
in the output file are skipped.

Grid (kept fixed across H1 pilot and H2 full run so they compare):
  coarsen  {6 Falgout, 8 PMIS, 10 HMIS}
  relax    {3 hybrid-GS, 6 sym-GS, 8 L1-sym-GS, 18 L1-Jacobi}
  strong   {0.25, 0.5, 0.7}       (interp fixed at 6 = ext+i)
= 36 configs per matrix.

solver follows the matrix stratum (cg for spd else gmres), same rule as AMGCL.

Usage:
    python hypre_sweep.py --matrices ../data/phase2_matrices.csv \
        --out ../results/hypre_pilot.jsonl --only <list> --reps 1 --repeat 3
"""

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "build" / ("hypre_runner.exe" if os.name == "nt" else "hypre_runner")
CACHE = ROOT / "data" / "matrices"

COARSEN = [6, 8, 10]
RELAX = [3, 6, 8, 18]
STRONG = [0.25, 0.5, 0.7]
INTERP = 6


def grid():
    return [{"coarsen": c, "relax": r, "strong": s, "interp": INTERP}
            for c in COARSEN for r in RELAX for s in STRONG]


def config_id(cfg):
    return " ".join(f"{k}={cfg[k]}" for k in ("coarsen", "relax", "strong", "interp"))


def env_stamp():
    return {"host": platform.node(),
            "omp_threads": os.environ.get("OMP_NUM_THREADS", "")}


def load_done(out_path):
    done = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if all(k in r for k in ("group", "name", "config_id", "rep")):
                done.add((r["group"], r["name"], r["config_id"], r["rep"]))
    return done


def run_one(mtx, cfg, solver, tol, maxiter, timeout, repeat):
    cmd = [str(RUNNER), str(mtx),
           f"coarsen={cfg['coarsen']}", f"relax={cfg['relax']}",
           f"strong={cfg['strong']}", f"interp={cfg['interp']}",
           f"solver={solver}", f"tol={tol}", f"maxiter={maxiter}",
           f"repeat={repeat}"]
    t0 = time.monotonic()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "t_wall": time.monotonic() - t0,
                "message": f"killed after {timeout}s"}
    wall = time.monotonic() - t0
    for line in reversed(p.stdout.splitlines()):
        if line.startswith("{"):
            try:
                rec = json.loads(line)
                rec["t_wall"] = wall
                return rec
            except json.JSONDecodeError:
                pass
    return {"status": "crash", "t_wall": wall,
            "message": (p.stderr or "no JSON")[:200], "returncode": p.returncode}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrices", type=Path, default=ROOT / "data" / "phase2_matrices.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "hypre_sweep.jsonl")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--maxiter", type=int, default=1000)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--only", help="comma list of group/name or substrings")
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()

    if not RUNNER.exists():
        sys.exit(f"hypre_runner not built: {RUNNER}")

    rows = list(csv.DictReader(args.matrices.open()))
    if args.only:
        pats = [s.strip() for s in args.only.split(",") if s.strip()]

        def match(r, p):
            if "/" in p:
                return p == f"{r['group']}/{r['name']}"
            return p in r["group"] or p in r["name"]
        rows = [r for r in rows if any(match(r, p) for p in pats)]

    si, sn = (int(x) for x in args.shard.split("/"))
    if sn > 1:
        rows = [r for i, r in enumerate(rows) if i % sn == si]

    done = load_done(args.out)
    stamp = env_stamp()
    configs = grid()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    todo = []
    for r in rows:
        mtx = CACHE / r["group"] / r["name"] / f"{r['name']}.mtx"
        if not mtx.exists():
            cands = sorted((CACHE / r["group"] / r["name"]).glob("*.mtx"))
            if not cands:
                print(f"  MISSING {r['group']}/{r['name']}", file=sys.stderr); continue
            mtx = cands[0]
        for cfg in configs:
            cid = config_id(cfg)
            for rep in range(args.reps):
                if (r["group"], r["name"], cid, rep) not in done:
                    todo.append((r, mtx, cfg, cid, rep))

    print(f"{len(rows)} matrices x {len(configs)} configs; {len(todo)} runs to do")
    with args.out.open("a") as f:
        for i, (r, mtx, cfg, cid, rep) in enumerate(todo, 1):
            solver = r.get("solver") or ("cg" if r.get("spd") == "1" else "gmres")
            rec = run_one(mtx, cfg, solver, args.tol, args.maxiter,
                          args.timeout, args.repeat)
            rec.update({"group": r["group"], "name": r["name"],
                        "stratum": r.get("stratum", ""), "graphlike": int(r.get("graphlike", 0)),
                        "config_id": cid, "rep": rep, **stamp})
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(f"[{i}/{len(todo)}] {r['group']}/{r['name']} {cid} -> "
                  f"{rec['status']} {rec.get('t_total', rec.get('t_wall', 0)):.3f}s")


if __name__ == "__main__":
    main()
