"""Admit the complete frozen MADBase study before final Excel/statistical exports."""
import json
import sys
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from core import constants as C
from core.artifact_io import load_json, read_jsonl, save_json_atomic, sha256_array, sha256_file
from core.config import RunConfig
from core.jalur_b import select_clustering_checkpoint
from core.pipeline import pair_identity_hash
from core.run_status import existing_attempts, read_status
from scripts.confirmatory_worker import baseline_directory
from scripts.print_run_plan import entries


def main():
    errors, admitted = [], {}
    inputs, samples, pairs = defaultdict(list), defaultdict(list), defaultdict(list)
    decision = load_json(ROOT / "research_arabic_digits/research_data/pilot/convergence_budget_decision.json")
    registry = load_json(ROOT / C.MADBASE_SPLITS_DIR / "split_registry.json")
    snapshot = load_json(ROOT / "research_arabic_digits/research_data/manifests/dataset_inventory.json")["snapshot_sha256"]
    split_hashes = {item["seed"]: item["sha256"] for item in registry["manifests"] if item["run_mode"] == "CONFIRMATORY"}
    for track, _, _, arch, feature, k, seed in entries():
        cfg = RunConfig(arch, feature, k, seed, dataset="MADBASE", run_mode="CONFIRMATORY", project_root=str(ROOT))
        try:
            completed = [p for _, p, status in existing_attempts(cfg.run_dir)
                         if status and status.get("status") == "COMPLETED"]
            assert len(completed) == 1, "completed attempt count"
            directory = completed[0]
            m = load_json(directory / "config.json")
            assert m["run_uid"] == cfg.run_uid and m["dataset"] == "MADBASE"
            assert m["run_mode"] == "CONFIRMATORY" and m["seed"] == seed and m["architecture"] == arch
            assert m["feature_method"] == feature and m["n_classes"] == k
            assert m["dataset_snapshot_sha256"] == snapshot
            assert m["test_origin"] == "OFFICIAL_SOURCE_FIXED_WRITER_SUBSET"
            assert m["split_manifest_sha256"] == split_hashes[seed]
            assert sha256_file(ROOT / m["split_manifest"]) == split_hashes[seed]
            assert m["max_nfev_clustering"] == decision["clustering"]["final_max_nfev"]
            assert m["max_nfev_supervised"] == decision["supervised"]["final_max_nfev"]
            for name, digest in m["input_array_sha256"].items():
                arr = np.load(directory / "artifacts" / (name + ".npy"), allow_pickle=False)
                assert sha256_array(arr) == digest, name + " hash"
                if np.issubdtype(arr.dtype, np.number):
                    assert np.isfinite(arr).all(), name + " finite"
            ids = np.load(directory / "artifacts/test_sample_id.npy", allow_pickle=False).astype(str)
            assert ids.size == 200 * k and np.all(np.char.startswith(ids, "test/"))
            samples[k].append(sha256_array(ids))
            inputs[(seed, feature, k)].append(m["input_array_sha256"])
            pairs[(seed, k)].append(m["paired_pair_manifest_sha256"])
            assert pair_identity_hash(load_json(directory / "artifacts/pair_manifest.json")) == m["paired_pair_manifest_sha256"]
            for loop in ("clustering", "supervised"):
                opt = load_json(directory / "logs" / (loop + "_optimizer_result.json"))
                rows = read_jsonl(directory / "logs" / (loop + "_log.jsonl"))
                assert len(rows) == opt["nfev"] <= opt["max_nfev"]
                assert opt["final_point_eval_id"] is not None
                assert all(np.isfinite(r["train_loss"]) for r in rows)
            labels = np.load(directory / "artifacts/y_test.npy", allow_pickle=False)
            pred = np.load(directory / "logs/test_predictions.npy", allow_pickle=False)
            confusion = np.load(directory / "logs/confusion_matrix.npy", allow_pickle=False)
            assert pred.shape == labels.shape and confusion.sum() == labels.size
            assert abs(float(np.mean(pred == labels)) - m["final_metrics"]["accuracy"]) < 1e-12
            assert zipfile.is_zipfile(directory / C.LOCAL_RUN_SPREADSHEET_NAME)
            assert (directory / "workbook_export_complete.json").exists()
            admitted[cfg.run_uid] = (directory, m)
        except Exception as exc:
            errors.append(dict(run_uid=cfg.run_uid, error=str(exc) or type(exc).__name__))
    for key, rows in inputs.items():
        if any(r != rows[0] for r in rows):
            errors.append(dict(group=str(key), error="paired input mismatch"))
    for label, groups in (("test instances", samples), ("clustering pairs", pairs)):
        for key, rows in groups.items():
            if len(set(rows)) != 1:
                errors.append(dict(group=str(key), error=label + " mismatch"))
    secondary_count, baseline_fits = 0, 0
    for directory, source in admitted.values():
        if source["track"] != "PRIMARY" or source["n_classes"] not in C.JALUR_B_K_VALUES:
            continue
        try:
            opt = load_json(directory / "logs/clustering_optimizer_result.json")
            selected, _, _ = select_clustering_checkpoint(read_jsonl(directory / "logs/clustering_log.jsonl"), opt["n_params"])
            secondary = directory.with_name(directory.name + f"_jalurB_auto_eval{selected:04d}")
            status = read_status(secondary)
            m = load_json(secondary / "config.json")
            assert status["status"] == "COMPLETED"
            assert m["source_run_uid"] == source["run_uid"] and m["dataset"] == "MADBASE"
            assert m["selected_eval_id"] == selected and m["test_used_for_selection"] is False
            assert m["max_nfev_supervised"] == decision["supervised"]["final_max_nfev"]
            assert m["dataset_snapshot_sha256"] == snapshot
            pred = np.load(secondary / "logs/test_predictions.npy", allow_pickle=False)
            labels = np.load(directory / "artifacts/y_test.npy", allow_pickle=False)
            assert pred.shape == labels.shape
            assert abs(float(np.mean(pred == labels)) - m["final_metrics"]["accuracy"]) < 1e-12
            secondary_count += 1
        except Exception as exc:
            errors.append(dict(run_uid=source["run_uid"] + "-JB", error=str(exc) or type(exc).__name__))
    for seed in C.CONFIRMATORY_SEEDS:
        for k in C.K_VALUES:
            for feature in C.FEATURE_METHODS:
                cell = baseline_directory(seed, feature, k)
                try:
                    rows = load_json(cell / "baseline_rows.json")
                    m = load_json(cell / "baseline_cell_manifest.json")
                    assert m["dataset"] == "MADBASE" and m["input_hash_match_A_D"] is True
                    source_directory, _ = admitted[m["source_run_uid_A"]]
                    assert m["split_manifest_hash"] == split_hashes[seed]
                    for name, digest in m["input_array_sha256"].items():
                        assert sha256_file(source_directory / "artifacts" / (name + ".npy")) == digest
                    assert len(rows) == 3 and {r["baseline"] for r in rows} == {"NC", "LR", "CHANCE"}
                    labels = np.load(source_directory / "artifacts/y_test.npy", allow_pickle=False)
                    for row in rows:
                        if row["baseline"] == "CHANCE":
                            assert row["accuracy"] == 1 / k
                            continue
                        pred = np.load(cell / (row["baseline"] + "_test_predictions.npy"), allow_pickle=False)
                        assert pred.shape == labels.shape
                        assert abs(float(np.mean(pred == labels)) - row["accuracy"]) < 1e-12
                        baseline_fits += 1
                except Exception as exc:
                    errors.append(dict(cell=str(cell), error=str(exc) or type(exc).__name__))
    report = dict(timestamp=datetime.now().astimezone().isoformat(), dataset="MADBASE",
                  main_runs=len(admitted), secondary_runs=secondary_count, classical_fits=baseline_fits,
                  errors=errors, accepted=not errors and len(admitted) == 450 and secondary_count == 90 and baseline_fits == 240)
    save_json_atomic(report, ROOT / "research_arabic_digits/research_data/reports/final_admission.json")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["accepted"] else 1)


if __name__ == "__main__":
    main()
