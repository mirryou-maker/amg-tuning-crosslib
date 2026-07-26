#!/usr/bin/env python3
"""Phase 2 end-to-end predictor evaluation.

The money metric: if we USE the predictor to pick one config per matrix, how
much of the oracle speedup do we actually capture -- evaluated honestly with
leave-one-group-out so no sibling matrix leaks into training?

Pipeline per held-out group:
  1. Train a success classifier P(solve | features, config) on other groups.
  2. Train a solve-time regressor E[log t_total | features, config] on the
     OK pairs of other groups.
  3. For each test matrix, score all 88 configs; among those predicted to
     succeed, pick the one with the smallest predicted time. Fall back to the
     highest success probability if none clears the threshold.
  4. Look up what that picked config ACTUALLY did on the test matrix.

Reported:
  * hit-rate: fraction of matrices where the picked config actually solved.
  * captured speedup: T(default)/T(picked) vs the oracle T(default)/T(best),
    as a fraction -- "how much of the achievable speedup did we get".
  * baseline: always using the AMGCL default config.

Tiers compared so the value of Tier 1 features is explicit.
"""

import argparse
import json
import re
from collections import defaultdict

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import LeaveOneGroupOut

from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

MCOLS_T0 = ["density", "avg_row_nnz", "std_row_nnz", "bandwidth_rel",
            "pattern_sym", "diag_present_frac"]
MCOLS_T1 = ["diag_dom_frac", "value_asym", "power_eig_est", "gershgorin_radius"]

DEFAULT = {"coarsening.type=smoothed_aggregation",
           "coarsening.aggr.eps_strong=0.08", "relax.type=spai0"}


def cfg_of(r):
    c = r.get("coarsening") or re.search(r"coarsening\.type=(\w+)", r["config_id"]).group(1)
    x = r.get("relaxation") or re.search(r"relax\.type=(\w+)", r["config_id"]).group(1)
    return c, x, r.get("config_id", "")


def is_default(r):
    return all(s in r.get("config_id", "") for s in DEFAULT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=ROOT / "results" / "features_p2.jsonl")
    ap.add_argument("--sweep", type=Path, default=ROOT / "results" / "p2_sweep.jsonl")
    args = ap.parse_args()

    feats = {(f["group"], f["name"]): f
             for f in (json.loads(l) for l in args.features.open())}
    recs = [json.loads(l) for l in args.sweep.open() if l.strip()]

    coars = sorted({cfg_of(r)[0] for r in recs})
    relax = sorted({cfg_of(r)[1] for r in recs})
    cfg_ids = sorted({cfg_of(r)[2] for r in recs})

    def cfg_vec(c, x):
        return [1.0 if c == cc else 0.0 for cc in coars] + \
               [1.0 if x == xx else 0.0 for xx in relax]

    for tier_name, mcols in [("Tier0", MCOLS_T0),
                             ("Tier0+1", MCOLS_T0 + MCOLS_T1)]:
        rows = []
        for r in recs:
            f = feats.get((r["group"], r["name"]))
            if not f:
                continue
            mv = [f.get(c) for c in mcols]
            if any(v is None for v in mv):
                continue
            c, x, cid = cfg_of(r)
            ok = r["status"] == "ok" and r.get("iters", 1) > 0
            rows.append({
                "group": r["group"], "name": r["name"], "cid": cid,
                "feat": [float(v) for v in mv] + cfg_vec(c, x),
                "ok": ok,
                "t": r["t_total"] if ok else None,
                "is_default": is_default(r),
            })

        X = np.array([r["feat"] for r in rows])
        yok = np.array([1 if r["ok"] else 0 for r in rows])
        grp = np.array([r["group"] for r in rows])
        idx_by_mat = defaultdict(list)
        for i, r in enumerate(rows):
            idx_by_mat[(r["group"], r["name"])].append(i)

        logo = LeaveOneGroupOut()
        clf = GradientBoostingClassifier(n_estimators=120, max_depth=3, random_state=0)
        reg = GradientBoostingRegressor(n_estimators=120, max_depth=3, random_state=0)

        captured, hits, mats = [], 0, 0
        for tr, te in logo.split(X, yok, grp):
            m_clf = clone(clf).fit(X[tr], yok[tr])
            ok_tr = tr[yok[tr] == 1]
            if len(ok_tr) < 20:
                continue
            logt = np.log(np.array([rows[i]["t"] for i in ok_tr]))
            m_reg = clone(reg).fit(X[ok_tr], logt)

            te_mats = {(rows[i]["group"], rows[i]["name"]) for i in te}
            for key in te_mats:
                cand = idx_by_mat[key]
                psucc = m_clf.predict_proba(X[cand])[:, 1]
                ptime = m_reg.predict(X[cand])
                # pick: among predicted-success (p>0.5), smallest predicted time;
                # else highest success prob
                order = [i for i in range(len(cand)) if psucc[i] > 0.5]
                if order:
                    pick = min(order, key=lambda i: ptime[i])
                else:
                    pick = int(np.argmax(psucc))
                picked = rows[cand[pick]]

                # actual outcomes on this matrix
                oks = [rows[i] for i in cand if rows[i]["ok"]]
                default = next((rows[i] for i in cand if rows[i]["is_default"] and rows[i]["ok"]), None)
                if not oks or default is None:
                    continue
                mats += 1
                best_t = min(r["t"] for r in oks)
                oracle = default["t"] / best_t
                if picked["ok"]:
                    hits += 1
                    got = default["t"] / picked["t"]
                else:
                    got = 1.0            # picked a failing config -> stuck with default
                # fraction of oracle speedup captured (log scale, robust)
                frac = np.log(got) / np.log(oracle) if oracle > 1 else 1.0
                captured.append(max(0.0, min(1.0, frac)))

        print(f"\n=== {tier_name} ===")
        print(f"  evaluable matrices (default+best ok): {mats}")
        print(f"  picked-config actually solved: {hits}/{mats} "
              f"({100 * hits / mats:.0f}%)" if mats else "  n/a")
        if captured:
            print(f"  oracle speedup captured (median): {np.median(captured) * 100:.0f}%")
            print(f"  oracle speedup captured (mean):   {np.mean(captured) * 100:.0f}%")


if __name__ == "__main__":
    main()
