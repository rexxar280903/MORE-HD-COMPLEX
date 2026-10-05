#!/usr/bin/env python3
"""Consolidate run artifacts into the three master workbooks (G3-03, G3-06, G3-07).

PRIMARY also receives sheet 15 (classical baselines), sheet 17 (Jalur B, COMPLETED runs only)
and sheet 18 (circuit microbenchmark). Every workbook is rebuilt from its template and the run
artifacts (idempotent); duplicate run_uid + attempt is rejected by the consolidator.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import constants as C  # noqa: E402
from core.artifact_io import load_json  # noqa: E402
from core.workbook import consolidate, discover_runs  # noqa: E402

TRACKS = (("PRIMARY", C.MASTER_SPREADSHEET_PATH, "primary_240runs.xlsx"),
          ("ABLATION", C.ABLATION_SPREADSHEET_PATH, "ablation_90runs.xlsx"),
          ("MORE_REFERENCE", C.MORE_REFERENCE_SPREADSHEET_PATH, "more_reference_120runs.xlsx"))


def completed(d: Path) -> bool:
    p = d / "logs" / "run_status.json"
    return p.exists() and load_json(p)["status"] == C.STATUS_COMPLETED


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-mode", choices=C.RUN_MODES, default="CONFIRMATORY")
    p.add_argument("--pilot-tag", default=None, help="PILOT only: consolidate one pilot stage (e.g. convergence)")
    p.add_argument("--project-root", default=str(ROOT))
    p.add_argument("--out-dir", default=None, help="default: research_data/consolidated[_pilot]")
    p.add_argument("--benchmark-dir", default=None,
                   help="microbenchmark JSON folder (default: research_data/confirmatory_machine/benchmarks, "
                        "or research_data/pilot/benchmarks for PILOT)")
    a = p.parse_args()
    root = Path(a.project_root)
    runs = root / C.RUNS_DIR
    pilot = a.run_mode == "PILOT"
    out = Path(a.out_dir) if a.out_dir else root / "research_data" / ("consolidated_pilot" if pilot else "consolidated")
    out.mkdir(parents=True, exist_ok=True)

    def tag_ok(cfg: dict) -> bool:
        return not pilot or a.pilot_tag is None or cfg.get("pilot_tag") == a.pilot_tag

    jalur_b = []
    for d in discover_runs(runs, "JALUR_B", a.run_mode):
        m = load_json(d / "config.json")
        if completed(d) and tag_ok(load_json(root / m["source_run_dir"] / "config.json")):
            jalur_b.append(d)
    bench_dir = Path(a.benchmark_dir) if a.benchmark_dir else \
        root / "research_data" / ("pilot" if pilot else "confirmatory_machine") / "benchmarks"
    bench = sorted(bench_dir.glob("circuit_microbenchmark_block*.json"))
    baseline_csv = runs / ("baselines_pilot" if pilot else "baselines") / "baseline_results.csv"

    for track, template, name in TRACKS:
        dirs = [d for d in discover_runs(runs, track, a.run_mode) if tag_ok(load_json(d / "config.json"))]
        extra = {}
        if track == "PRIMARY":
            extra = {"jalur_b_dirs": jalur_b, "benchmark_files": bench,
                     "baseline_csv": baseline_csv if baseline_csv.exists() else None}
        res = consolidate(root / template, dirs, out / name, track, **extra)
        print(f"{track}: {res['n_runs']} runs -> {out / name}")
    print(f"Jalur B: {len(jalur_b)} COMPLETED; benchmark files: {len(bench)}; "
          f"baselines: {baseline_csv if baseline_csv.exists() else 'not found'}")


if __name__ == "__main__":
    main()
