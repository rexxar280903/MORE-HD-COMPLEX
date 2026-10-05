#!/usr/bin/env python3
"""After the seed-42 pilots: exercise Jalur B, classical baselines, consolidation and the
analysis on PILOT data (G0-02, G1-04, G1-10, G3-03, G3-06, G3-07 verification).
Writes evidence to research_data/pilot/. Nothing here is a confirmatory result."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from core import constants as C  # noqa: E402
from core.artifact_io import load_json, save_json_atomic  # noqa: E402
from core.config import run_name_for  # noqa: E402
from core.workbook import consolidate, discover_runs  # noqa: E402

OUT = ROOT / "research_data" / "pilot"
OUT.mkdir(parents=True, exist_ok=True)
evidence = {}

# 1. Jalur B on convergence pilot runs (A and D, K=3 and K=10, PCA)
jb_dirs = []
for arch in (C.ARCH_MORE_HD, C.ARCH_MORE_HD_C):
    for k in (3, 10):
        src = ROOT / "runs" / run_name_for(arch, "PCA", k, 42, run_mode="PILOT", pilot_tag="convergence")
        r = subprocess.run([sys.executable, str(ROOT / "main_selected_clustering.py"), "--source-run-dir", str(src),
                            "--n-parallel-declared", "1", "--project-root", str(ROOT)], capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(r.stderr[-3000:])
        info = json.loads(r.stdout)
        jb_dirs.append(ROOT / "runs" / info["run_name"])
        evidence.setdefault("jalur_b", []).append(info)

# 2. Classical baselines on the convergence pilot cells
r = subprocess.run([sys.executable, str(ROOT / "main_classical_baseline.py"), "--run-mode", "PILOT", "--pilot-tag",
                    "convergence", "--ks", "3", "10", "--features", "PCA", "ZERNIKE", "--project-root", str(ROOT)],
                   capture_output=True, text=True)
if r.returncode != 0:
    raise SystemExit(r.stderr[-3000:])
evidence["baselines"] = r.stdout.strip()

# 3. Consolidation of the convergence pilot per track (+ Jalur B, benchmark, baselines)
bench = sorted((ROOT / "benchmarks").glob("circuit_microbenchmark_block*.json"))
baseline_csv = ROOT / "runs" / "baselines_pilot" / "baseline_results.csv"
evidence["consolidation"] = {}
for track, tpl, name in (("PRIMARY", C.MASTER_SPREADSHEET_PATH, "pilot_primary_consolidated.xlsx"),
                         ("ABLATION", C.ABLATION_SPREADSHEET_PATH, "pilot_ablation_consolidated.xlsx"),
                         ("MORE_REFERENCE", C.MORE_REFERENCE_SPREADSHEET_PATH, "pilot_more_reference_consolidated.xlsx")):
    dirs = [d for d in discover_runs(ROOT / "runs", track, "PILOT") if load_json(d / "config.json").get("pilot_tag") == "convergence"]
    kw = {"jalur_b_dirs": jb_dirs, "benchmark_files": bench, "baseline_csv": baseline_csv} if track == "PRIMARY" else {}
    res = consolidate(ROOT / tpl, dirs, OUT / name, track, **kw)
    res2 = consolidate(ROOT / tpl, dirs, OUT / ("_check_" + name), track, **kw)       # idempotency
    (OUT / ("_check_" + name)).unlink()
    evidence["consolidation"][track] = {"n_runs": res["n_runs"], "run_uids": res["run_uids"],
                                        "idempotent_same_count": res["n_runs"] == res2["n_runs"]}

# 4. Representative A/B/C/D/M cell (smoke K=3 PCA) consolidated into the ablation workbook
smoke_abl = [d for d in discover_runs(ROOT / "runs", "ABLATION", "PILOT") if load_json(d / "config.json").get("pilot_tag") == "smoke"]
res = consolidate(ROOT / C.ABLATION_SPREADSHEET_PATH, smoke_abl, OUT / "pilot_smoke_ablation_cell.xlsx", "ABLATION")
evidence["ablation_representative_cell"] = res["run_uids"]

# 5. Analysis machinery on the pilot batch
r = subprocess.run([sys.executable, str(ROOT / "analysis" / "run_analysis.py"), "--run-mode", "PILOT", "--pilot-tag",
                    "convergence", "--out-dir", str(ROOT / "research_data" / "analysis_pilot")], capture_output=True, text=True)
if r.returncode != 0:
    raise SystemExit(r.stderr[-3000:])
evidence["analysis"] = r.stdout.strip()
r = subprocess.run([sys.executable, str(ROOT / "analysis" / "run_analysis.py"), "--run-mode", "PILOT", "--pilot-tag",
                    "smoke", "--out-dir", str(ROOT / "research_data" / "analysis_pilot" / "smoke_cell")], capture_output=True, text=True)
evidence["analysis_smoke_cell"] = r.stdout.strip() if r.returncode == 0 else r.stderr[-2000:]
save_json_atomic(evidence, OUT / "pilot_bundle_evidence.json")
print(json.dumps(evidence, indent=2)[:4000])
