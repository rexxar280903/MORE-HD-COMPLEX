#!/usr/bin/env python3
"""Check the smoke-test acceptance criteria of pseudocode §1.1 on PILOT runs with a given tag."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.artifact_io import load_json, read_jsonl, save_json_atomic  # noqa: E402

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--tag", default="smoke")
p.add_argument("--out", default=None, help="optional JSON evidence file (runs + verdict)")
a = p.parse_args()
rows = []
ok_all = True
for cfg in sorted((ROOT / "runs").rglob("config.json")):
    m = load_json(cfg)
    if m.get("pilot_tag") != a.tag or m.get("track") == "JALUR_B":
        continue
    d = cfg.parent
    res = {"run_uid": m["run_uid"], "architecture": m["architecture"], "K": m["n_classes"]}
    for loop in ("clustering", "supervised"):
        log = read_jsonl(d / "logs" / f"{loop}_log.jsonl")
        opt = load_json(d / "logs" / f"{loop}_optimizer_result.json")
        n = opt["n_params"]
        first_opt = next((r["eval_id"] for r in log if r["phase"] == "optimization"), None)
        checks = {
            "len_log_eq_nfev": len(log) == opt["nfev"],
            "phase_switch_at_n_params_plus_1": first_opt == n + 1,
            "final_point_eval_id_set": opt["final_point_eval_id"] is not None,
            "budget_respected": opt["nfev"] == opt["max_nfev"] or not opt["stopped_by_budget"],
        }
        res[loop] = {"nfev": opt["nfev"], "first_optimization_eval_id": first_opt, **checks}
        ok_all &= all(checks.values())
    st = load_json(d / "logs" / "run_status.json")["status"]
    res["status"] = st
    res["backend_max_abs_diff"] = m["backend_self_check"]["max_abs_diff"]
    res["accuracy"] = m["final_metrics"]["accuracy"]
    ok_all &= st == "COMPLETED"
    rows.append(res)
ok_all = bool(ok_all and rows)
if a.out:
    save_json_atomic({"tag": a.tag, "n_runs": len(rows), "all_criteria_ok": ok_all, "runs": rows}, Path(a.out))
print(json.dumps(rows, indent=1))
print("ALL SMOKE CRITERIA OK" if ok_all else "SMOKE CRITERIA FAILED")
