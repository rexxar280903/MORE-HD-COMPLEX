#!/usr/bin/env python3
"""Produce an auditable MADBase pilot/confirmatory readiness report."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import openpyxl  # noqa: E402

from core import constants as C  # noqa: E402
from core.artifact_io import load_json, save_json_atomic, sha256_file  # noqa: E402
from core.datasets import load_dataset  # noqa: E402
from core.splits import validate_madbase_runtime_manifest  # noqa: E402
from core.workbook import validate_schema_map  # noqa: E402


EXPECTED_RUNTIME = {
    "numpy": "2.5.2",
    "scipy": "1.18.1",
    "scikit-learn": "1.9.0",
    "pennylane": "0.45.1",
    "joblib": "1.6.0",
    "openpyxl": "3.1.5",
    "threadpoolctl": "3.6.0",
    "Pillow": "12.3.0",
}


def package_versions():
    import importlib.metadata as metadata

    out = {}
    for name in EXPECTED_RUNTIME:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", default=str(ROOT))
    p.add_argument("--out", default=None)
    p.add_argument("--require-confirmatory", action="store_true")
    args = p.parse_args()
    root = Path(args.project_root).resolve()
    checks = {}

    data = load_dataset(root, C.DATASET_MADBASE)
    checks["dataset_training_view"] = {
        "ok": data.train_images.shape == (60_000, 28, 28) and not hasattr(data, "test_images"),
        "dataset_id": data.dataset_id,
        "snapshot_sha256": data.snapshot_sha256,
        "official_test_loaded": hasattr(data, "test_images"),
    }

    registry_path = root / C.MADBASE_SPLITS_DIR / "split_registry.json"
    registry = load_json(registry_path)
    split_errors = []
    for item in registry["manifests"]:
        path = root / item["path"]
        try:
            if sha256_file(path) != item["sha256"]:
                raise ValueError("file hash mismatch")
            manifest = load_json(path)
            validate_madbase_runtime_manifest(manifest, item["seed"], data, item["run_mode"])
        except Exception as exc:
            split_errors.append({"path": item["path"], "error": repr(exc)})
    checks["split_manifests"] = {
        "ok": not split_errors and len(registry["manifests"]) == 6,
        "count": len(registry["manifests"]),
        "errors": split_errors,
        "fixed_confirmatory_test_hash": registry["confirmatory_test_sample_ids_sha256"],
    }

    workbook_paths = (
        C.MADBASE_MASTER_SPREADSHEET_PATH,
        C.MADBASE_ABLATION_SPREADSHEET_PATH,
        C.MADBASE_MORE_REFERENCE_SPREADSHEET_PATH,
    )
    workbook_errors = []
    for rel in workbook_paths:
        try:
            wb = openpyxl.load_workbook(root / rel, read_only=False, data_only=False)
            problems = validate_schema_map(wb)
            if problems:
                raise ValueError("; ".join(problems[:5]))
        except Exception as exc:
            workbook_errors.append({"path": rel, "error": repr(exc)})
    checks["workbook_templates"] = {"ok": not workbook_errors, "count": 3, "errors": workbook_errors}

    smoke_path = root / "research_arabic_digits/research_data/pilot/smoke_check.json"
    smoke = load_json(smoke_path) if smoke_path.exists() else {}
    checks["smoke_matrix"] = {
        "ok": bool(smoke.get("all_criteria_ok") and smoke.get("n_runs") == 5),
        "n_runs": smoke.get("n_runs", 0),
        "architectures": sorted(r.get("architecture") for r in smoke.get("runs", [])),
    }
    test_path = root / "research_arabic_digits/research_data/pilot/test_summary.json"
    tests = load_json(test_path) if test_path.exists() else {}
    checks["tests"] = {"ok": tests.get("verdict") == "PASS" and tests.get("failed") == 0, **tests}

    actual_versions = package_versions()
    py = tuple(int(x) for x in platform.python_version().split(".")[:2])
    checks["environment"] = {
        "ok": (3, 11) <= py < (3, 14) and actual_versions == EXPECTED_RUNTIME,
        "python": platform.python_version(),
        "expected": EXPECTED_RUNTIME,
        "actual": actual_versions,
    }
    freeze_path = root / "research_arabic_digits/PROTOCOL_FREEZE.json"
    freeze = load_json(freeze_path)
    checks["protocol_freeze"] = {
        "ok": freeze.get("dataset_id") == C.DATASET_IDS[C.DATASET_MADBASE],
        "status": freeze.get("status"),
    }
    free_gb = round(shutil.disk_usage(root).free / (1024 ** 3), 2)
    checks["disk"] = {"ok": free_gb >= 5.0, "free_gb": free_gb, "minimum_gb": 5.0}

    pilot_keys = ("dataset_training_view", "split_manifests", "workbook_templates", "smoke_matrix",
                  "tests", "environment", "protocol_freeze", "disk")
    pilot_ready = all(checks[k]["ok"] for k in pilot_keys)
    decision_path = root / "research_arabic_digits/research_data/pilot/convergence_budget_decision.json"
    decision = load_json(decision_path) if decision_path.exists() else {}
    convergence_frozen = (decision.get("status") == "FROZEN" and
                          decision.get("dataset") == C.DATASET_MADBASE)
    checks["convergence_budget"] = {
        "ok": convergence_frozen,
        "path": str(decision_path.relative_to(root)),
        "status": decision.get("status", "MISSING"),
    }
    report = {
        "schema_version": "1.0",
        "dataset": C.DATASET_MADBASE,
        "dataset_id": C.DATASET_IDS[C.DATASET_MADBASE],
        "pilot_ready": pilot_ready,
        "confirmatory_ready": bool(pilot_ready and convergence_frozen),
        "checks": checks,
        "next_required_action": (None if convergence_frozen else
                                 "run MADBase convergence pilot, then analyze_pilot_budget.py --dataset MADBASE"),
    }
    out = Path(args.out) if args.out else root / "research_arabic_digits/research_data/manifests/readiness_report.json"
    save_json_atomic(report, out)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    ready = report["confirmatory_ready"] if args.require_confirmatory else report["pilot_ready"]
    raise SystemExit(0 if ready else 1)


if __name__ == "__main__":
    main()
