"""Audit MADBase pilot provenance and the full planned confirmatory matrix without training."""
import json
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.environment import lock_threads
lock_threads()
import numpy as np
import openpyxl
from core import constants as C
from core.artifact_io import load_json, save_json_atomic, sha256_array, sha256_file
from core.cli import budget_from_args
from core.config import RunConfig, validate_config
from core.datasets import load_dataset
from core.pipeline import pair_identity_hash
from core.pilot_audit import completed_pilots
from core.run_status import existing_attempts
from core.splits import validate_madbase_runtime_manifest
from core.workbook import rows_for_run, validate_schema_map
from scripts.print_run_plan import entries


def main():
    checks = {}
    report = {"timestamp": datetime.now().astimezone().isoformat(), "dataset": "MADBASE", "checks": checks}
    data = load_dataset(ROOT, "MADBASE")
    admitted, excluded = completed_pilots(ROOT, "MADBASE", "convergence")
    errors, group_inputs, group_samples, group_pairs, sample_sizes = [], defaultdict(list), defaultdict(list), defaultdict(list), {}
    for directory, cfg in admitted:
        try:
            assert cfg["dataset_snapshot_sha256"] == data.snapshot_sha256
            assert cfg["test_origin"] == "CONSTRUCTED_PILOT_TRAIN_POOL"
            assert sha256_file(ROOT / cfg["split_manifest"]) == cfg["split_manifest_sha256"]
            for name, digest in cfg["input_array_sha256"].items():
                arr = np.load(directory / "artifacts" / f"{name}.npy", allow_pickle=False)
                assert sha256_array(arr) == digest, f"array hash: {name}"
                if np.issubdtype(arr.dtype, np.number):
                    assert np.isfinite(arr).all(), f"nonfinite array: {name}"
            pair = load_json(directory / "artifacts/pair_manifest.json")
            assert pair_identity_hash(pair) == cfg["paired_pair_manifest_sha256"]
            for name in ("train_sample_id", "val_sample_id", "test_sample_id"):
                sample = np.load(directory / "artifacts" / f"{name}.npy", allow_pickle=False).astype(str)
                assert np.all(np.char.startswith(sample, "train/"))
                group_samples[cfg["n_classes"]].append((name, sha256_array(sample)))
            for name in ("train", "val", "test"):
                arr = np.load(directory / "artifacts" / f"X_{name}_scaled.npy", allow_pickle=False)
                size = cfg[{"train":"n_train_per_class", "val":"n_val_per_class", "test":"n_test_per_class"}[name]]
                assert arr.shape == (size * cfg["n_classes"], 8)
                assert np.isfinite(arr).all()
            group_inputs[(cfg["feature_method"], cfg["n_classes"])].append(cfg["input_array_sha256"])
            group_pairs[cfg["n_classes"]].append(cfg["paired_pair_manifest_sha256"])
            assert cfg["backend_self_check"]["max_abs_diff"] < 1e-12
            assert all(n == 1 for n in cfg["environment"]["blas_threads"].values())
            pred = np.load(directory / "logs/test_predictions.npy", allow_pickle=False)
            labels = np.load(directory / "artifacts/y_test.npy", allow_pickle=False)
            cm = np.load(directory / "logs/confusion_matrix.npy", allow_pickle=False)
            assert pred.shape == labels.shape and int(cm.sum()) == labels.size
            assert np.isin(pred, cfg["classes"]).all()
            assert abs(float(np.mean(pred == labels)) - cfg["final_metrics"]["accuracy"]) < 1e-12
            rows = rows_for_run(directory)
            assert len(rows["11_Test_Predictions"]) == labels.size
            sample_sizes[(cfg["architecture"], cfg["feature_method"], cfg["n_classes"])] = sum(
                p.stat().st_size for p in directory.rglob("*") if p.is_file())
        except Exception as exc:
            errors.append({"run_uid": cfg["run_uid"], "error": str(exc) or type(exc).__name__})
    for key, records in group_inputs.items():
        if any(r != records[0] for r in records[1:]):
            errors.append({"paired_inputs": str(key), "error": "mismatch"})
    for k, records in group_samples.items():
        for name in ("train_sample_id", "val_sample_id", "test_sample_id"):
            if len({digest for field, digest in records if field == name}) != 1:
                errors.append({"paired_samples": [k, name], "error": "mismatch"})
    for k, records in group_pairs.items():
        if len(set(records)) != 1:
            errors.append({"paired_pairs": k, "error": "mismatch"})
    checks["pilot_provenance_and_export"] = dict(ok=not errors, admitted=len(admitted), excluded=excluded, errors=errors)

    split_errors = []
    registry = load_json(ROOT / C.MADBASE_SPLITS_DIR / "split_registry.json")
    for item in registry["manifests"]:
        path = ROOT / item["path"]
        try:
            assert sha256_file(path) == item["sha256"]
            validate_madbase_runtime_manifest(load_json(path), item["seed"], data, item["run_mode"])
        except Exception as exc:
            split_errors.append(str(exc))
    checks["splits"] = dict(ok=not split_errors, count=len(registry["manifests"]), errors=split_errors,
                            official_test_loaded=hasattr(data, "test_images"))

    decision = load_json(ROOT / "research_arabic_digits/research_data/pilot/convergence_budget_decision.json")
    from argparse import Namespace
    clustering, supervised = budget_from_args(Namespace(dataset="MADBASE", run_mode="CONFIRMATORY",
        project_root=str(ROOT), max_nfev_clustering=None, max_nfev_supervised=None))
    plan, plan_errors, status_counts = [], [], Counter()
    for track, script, flag, arch, feature, k, seed in entries():
        cfg = RunConfig(arch, feature, k, seed, dataset="MADBASE", run_mode="CONFIRMATORY",
                        max_nfev_clustering=clustering, max_nfev_supervised=supervised,
                        n_parallel_declared=6, project_root=str(ROOT))
        try:
            assert not validate_config(cfg)
        except Exception as exc:
            plan_errors.append({"run_uid": cfg.run_uid, "error": str(exc)})
        attempts = existing_attempts(cfg.run_dir)
        state = "NOT_STARTED" if not attempts else attempts[-1][2].get("status", "MISSING")
        status_counts[state] += 1
        plan.append(dict(run_uid=cfg.run_uid, track=track, architecture=arch, feature=feature, k=k, seed=seed,
                         run_dir=str(cfg.run_dir.relative_to(ROOT)), status=state,
                         clustering_budget=clustering, supervised_budget=supervised, parallel_limit=6,
                         command=f"python {script} {flag} --dataset MADBASE --feature {feature} --k {k} --seed {seed} --run-mode CONFIRMATORY --n-parallel-declared 6"))
    counts = Counter(r["track"] for r in plan)
    checks["confirmatory_plan"] = dict(ok=not plan_errors and len({r["run_uid"] for r in plan}) == 450 and
                                       counts == {"PRIMARY":240,"ABLATION":90,"MORE_REFERENCE":120},
                                       tracks=dict(counts), status_counts=dict(status_counts), errors=plan_errors)
    checks["budget"] = dict(ok=decision["status"] == "FROZEN" and decision["n_runs"] == 14,
                            clustering=clustering, supervised=supervised,
                            plateau_limitation=decision["clustering"]["limitation"] or decision["supervised"]["limitation"])
    schema_errors = []
    for rel in (C.MADBASE_MASTER_SPREADSHEET_PATH, C.MADBASE_ABLATION_SPREADSHEET_PATH, C.MADBASE_MORE_REFERENCE_SPREADSHEET_PATH):
        wb = openpyxl.load_workbook(ROOT / rel)
        schema_errors.extend(validate_schema_map(wb))
        wb.close()
    checks["workbooks"] = dict(ok=not schema_errors, templates=3, errors=schema_errors,
                               max_primary_clustering_rows=240 * clustering, max_primary_supervised_rows=240 * supervised)
    projected = 0
    for row in plan:
        arch = row["architecture"]
        if arch == C.ARCH_MORE_HD_C_FIXED_RZ:
            arch = C.ARCH_MORE_HD_C
        feature = "ZERNIKE" if row["feature"] == "HU" else row["feature"]
        sizes = [(k, size) for (a, f, k), size in sample_sizes.items() if a == arch and f == feature]
        projected += max(size for _, size in sizes) + 1024 ** 2  # conservative per-run workbook allowance
    free = shutil.disk_usage(ROOT).free
    required = 2 * projected + 1024 ** 3  # reserve for secondary runs, consolidation and operational variance
    checks["storage"] = dict(ok=free >= required, free_gb=round(free/1024**3,2),
                             estimated_primary_artifacts_and_workbooks_gb=round(projected/1024**3,2),
                             required_with_reserve_gb=round(required/1024**3,2), estimate=True)
    report["ready"] = all(c["ok"] for c in checks.values())
    report["training_started_by_audit"] = False
    out = ROOT / "research_arabic_digits/research_data/reports"
    save_json_atomic(plan, out / "confirmatory_run_plan.json")
    save_json_atomic(report, out / "full_readiness_audit.json")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["ready"] else 1)


if __name__ == "__main__":
    main()
