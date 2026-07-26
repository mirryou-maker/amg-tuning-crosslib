#!/usr/bin/env python3
"""Phase 1 enriched analysis + validity audit.

Beyond analyze.py's oracle table, this answers the questions that decide
whether Phase 1 is solid enough to build Phase 2 on:

  A. default-fails / tuned-succeeds: on how many matrices does the AMGCL
     default config fail outright while some grid config solves? (The
     predictor's value beyond speed.)
  B. host mixing: the sweep spanned two PBS jobs on different physical nodes.
     For each oracle matrix, did default and best run on the same host? If
     not, the oracle ratio mixes machines and must be flagged.
  C. timeout casualties: matrices where nothing succeeded but many runs died
     at the timeout wall -- these may be solvable with a longer budget
     (pdb1HYS pattern), unlike genuinely diverging matrices.
  D. setup/solve split of winning configs: does the winner win in setup, in
     solve, or both?
  E. eps_strong sensitivity: within (matrix, coarsening, relaxation), how much
     does the strength threshold alone move total time?

Usage:
    python enrich_analysis.py --results ../results/sweep_iremb.jsonl
"""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_KEYS = {
    "precond.coarsening.type": "smoothed_aggregation",
    "precond.coarsening.aggr.eps_strong": "0.08",
    "precond.relax.type": "spai0",
}


def is_default(rec):
    cid = rec.get("config_id", "")
    return all(f"{k}={v}" in cid for k, v in DEFAULT_KEYS.items())


def load(path):
    with path.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=ROOT / "results" / "sweep_iremb.jsonl")
    args = ap.parse_args()

    recs = load(args.results)
    by_matrix = defaultdict(list)
    for r in recs:
        by_matrix[(r["group"], r["name"])].append(r)

    # ---- A. default fails, tuned succeeds -------------------------------
    print("=== A. default-fails / tuned-succeeds (graphlike excluded) ===")
    a_rows = []
    for (g, n), rs in sorted(by_matrix.items()):
        if rs[0].get("graphlike"):
            continue
        ok = [r for r in rs if r["status"] == "ok"]
        default = [r for r in rs if is_default(r)]
        d_status = default[0]["status"] if default else "absent"
        if ok and d_status != "ok":
            best = min(ok, key=lambda r: r["t_total"])
            a_rows.append((g, n, d_status, len(ok), best["t_total"]))
    for g, n, ds, nok, bt in a_rows:
        print(f"  {g + '/' + n:<34} default={ds:<8} ok_configs={nok:3d} best={bt:8.3f}s")
    print(f"  -> {len(a_rows)} matrices where tuning turns failure into success\n")

    # ---- B. host mixing in oracle pairs ---------------------------------
    print("=== B. host consistency of oracle (default, best) pairs ===")
    mixed = same = 0
    for (g, n), rs in sorted(by_matrix.items()):
        if rs[0].get("graphlike"):
            continue
        ok = [r for r in rs if r["status"] == "ok"]
        default = next((r for r in ok if is_default(r)), None)
        if not (default and ok):
            continue
        best = min(ok, key=lambda r: r["t_total"])
        dh, bh = default.get("host", "?"), best.get("host", "?")
        tag = "SAME" if dh == bh else "MIXED"
        if dh == bh:
            same += 1
        else:
            mixed += 1
        print(f"  {g + '/' + n:<34} default@{dh:<12} best@{bh:<12} {tag}")
    print(f"  -> same-host {same}, mixed-host {mixed}\n")

    # per-matrix host spread overall
    multi = sum(1 for rs in by_matrix.values()
                if len({r.get('host') for r in rs}) > 1)
    print(f"  matrices whose runs span >1 host: {multi}/{len(by_matrix)}\n")

    # ---- C. timeout casualties ------------------------------------------
    print("=== C. all-fail matrices: genuinely hopeless vs timeout-limited ===")
    for (g, n), rs in sorted(by_matrix.items()):
        ok = [r for r in rs if r["status"] == "ok"]
        if ok:
            continue
        st = Counter(r["status"] for r in rs)
        n_to = st.get("timeout", 0)
        verdict = "TIMEOUT-LIMITED (retry w/ bigger budget)" if n_to >= len(rs) * 0.5 \
            else "hopeless (diverge/error dominated)"
        gtag = " [graph]" if rs[0].get("graphlike") else ""
        print(f"  {g + '/' + n:<34} {dict(st)}  -> {verdict}{gtag}")
    print()

    # ---- D. setup vs solve of winners -----------------------------------
    print("=== D. winner setup/solve split ===")
    for (g, n), rs in sorted(by_matrix.items()):
        ok = [r for r in rs if r["status"] == "ok"]
        default = next((r for r in ok if is_default(r)), None)
        if not (default and ok) or rs[0].get("graphlike"):
            continue
        best = min(ok, key=lambda r: r["t_total"])
        print(f"  {g + '/' + n:<30} default setup/solve "
              f"{default['t_setup']:7.3f}/{default['t_solve']:7.3f}  "
              f"best {best['t_setup']:7.3f}/{best['t_solve']:7.3f}")
    print()

    # ---- E. eps_strong sensitivity --------------------------------------
    print("=== E. eps_strong-only sensitivity (max/min total time within "
          "(matrix, coarsening, relax), ok runs, >=2 eps values) ===")
    ratios = []
    for (g, n), rs in by_matrix.items():
        if rs[0].get("graphlike"):
            continue
        groups = defaultdict(list)
        for r in rs:
            if r["status"] != "ok":
                continue
            groups[(r["coarsening"], r["relaxation"],
                    "k=" + str(r["config_id"].count("relax.k=2")))].append(r)
        for key, g_rs in groups.items():
            if len(g_rs) >= 2:
                ts = [r["t_total"] for r in g_rs]
                ratios.append(max(ts) / min(ts))
    if ratios:
        ratios.sort()
        print(f"  n={len(ratios)} groups; median {statistics.median(ratios):.2f}x, "
              f"Q3 {statistics.quantiles(ratios, n=4)[2]:.2f}x, "
              f"max {max(ratios):.1f}x")
        big = sum(1 for x in ratios if x >= 2)
        print(f"  eps_strong alone changes total time >=2x in "
              f"{big}/{len(ratios)} ({big / len(ratios) * 100:.0f}%) of groups")


if __name__ == "__main__":
    main()
