#!/usr/bin/env python3
"""Apply the frozen G1-01 plateau rule (pseudocode §1.1) to the seed-42 convergence pilot.

For every pilot run and loop (clustering, supervised), with L(N) the best-observed
train_loss after N objective evaluations, L_x0 = L(1) and L_end = L(nfev):

* N*  = smallest N with L(N) - L_end <= 0.02 * (L_x0 - L_end)
* not plateaued  <=> stopped by budget AND L(nfev - 30) - L_end > 0.02 * (L_x0 - L_end)
* final max_nfev = max N* over all runs, rounded up to a multiple of 10, capped at the
  pilot cap; if any run is not plateaued the final budget is the cap (reported as a
  limitation: "at equal objective-evaluation budget", not "at convergence");
  in every case final > n_params + 1 for all models (rule 5).
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from core import constants as C  # noqa: E402
from core.artifact_io import load_json, read_jsonl, save_json_atomic  # noqa: E402

TOL_FRACTION = 0.02
TAIL = 30


def curve_stats(losses, stopped_by_budget):
    best = np.minimum.accumulate(np.asarray(losses, dtype=float))
    l_x0, l_end, nfev = float(best[0]), float(best[-1]), len(best)
    total = l_x0 - l_end
    thr = TOL_FRACTION * total
    n_star = int(np.flatnonzero(best - l_end <= thr)[0]) + 1 if total > 0 else 1
    tail_impr = float(best[nfev - TAIL - 1] - l_end) if nfev > TAIL else float("inf")
    plateaued = not (stopped_by_budget and tail_impr > thr)
    return {"nfev": nfev, "L_x0": l_x0, "L_end": l_end, "total_improvement": total, "n_star": n_star,
            "tail_improvement_last30": tail_impr, "threshold_2pct": thr, "stopped_by_budget": stopped_by_budget,
            "plateaued": plateaued}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", default=str(ROOT))
    p.add_argument("--tag", default="convergence")
    p.add_argument("--cap", type=int, default=C.PILOT_MAX_NFEV_CAP)
    a = p.parse_args()
    root = Path(a.project_root)
    rows = []
    max_params = 0
    for cfg_path in sorted((root / C.RUNS_DIR).rglob("config.json")):
        m = load_json(cfg_path)
        if m.get("run_mode") != "PILOT" or m.get("pilot_tag") != a.tag or m.get("track") == "JALUR_B":
            continue
        d = cfg_path.parent
        for loop in ("clustering", "supervised"):
            log = read_jsonl(d / "logs" / f"{loop}_log.jsonl")
            opt = load_json(d / "logs" / f"{loop}_optimizer_result.json")
            st = curve_stats([r["train_loss"] for r in log], bool(opt["stopped_by_budget"]))
            max_params = max(max_params, opt["n_params"])
            rows.append({"run_uid": m["run_uid"], "architecture": m["architecture"], "feature_method": m["feature_method"],
                         "K": m["n_classes"], "loop": loop, "n_params": opt["n_params"], "max_nfev": opt["max_nfev"],
                         "status": opt["status"], "message": opt["message"], **st})
    if not rows:
        raise SystemExit("no convergence pilot runs found")
    decision = {"rule": "pseudocode §1.1 (frozen 2026-09-27; cap revised to %d on 2026-10-04 before any pilot run)" % a.cap,
                "tol_fraction": TOL_FRACTION, "tail": TAIL, "cap": a.cap, "n_runs": len({r['run_uid'] for r in rows})}
    for loop in ("clustering", "supervised"):
        lr = [r for r in rows if r["loop"] == loop]
        n_star_max = max(r["n_star"] for r in lr)
        not_plateaued = [r["run_uid"] for r in lr if not r["plateaued"]]
        final = min(a.cap, int(math.ceil(n_star_max / 10.0) * 10))
        if not_plateaued:
            final = a.cap
        floor = int(math.ceil((max_params + 2) / 10.0) * 10)
        rule5_applied = final < floor
        final = max(final, floor)
        decision[loop] = {"n_star_max": n_star_max, "n_star_by_run": {r["run_uid"]: r["n_star"] for r in lr},
                          "not_plateaued_runs": not_plateaued, "final_max_nfev": final,
                          "rule5_floor_applied": rule5_applied,
                          "limitation": bool(not_plateaued)}
    out = root / "research_data" / "pilot"
    out.mkdir(parents=True, exist_ok=True)
    save_json_atomic(decision, out / "convergence_budget_decision.json")
    with open(out / "convergence_curves_summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "n_star_by_run"})
                      for k, v in decision.items()}, indent=2))


if __name__ == "__main__":
    main()
