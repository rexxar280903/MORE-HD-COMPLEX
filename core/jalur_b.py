"""Jalur B: deterministic validation-selected clustering checkpoint (pseudocode §10; G0-02, G1-06, G2-04).

Secondary/sensitivity path. It starts from a COMPLETED Jalur A run, never
reruns DATA_PIPELINE or CLUSTERING_LOOP, selects one clustering evaluation with
the frozen lexicographic rule on train/validation metrics only, and then runs
quantum-label extraction, supervised training (budget inherited from the source
run) and the final evaluation. Official-test arrays are opened only inside the
``final_evaluation`` stage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import constants as C
from .artifact_io import ParamLog, load_json, load_npy, read_jsonl, save_json_atomic
from .circuits import BatchedStatevector, build_initialization_bundle, build_model
from .config import RunConfig
from .environment import environment_record
from .evaluation import OfficialTestVault, final_evaluation, structural_diagnostics
from .optimizer import objective_phase
from .pairing import cluster_positions_from_manifest
from .pipeline import select_val_monitor_set
from .quantum_labels import extract_quantum_labels
from .run_status import read_status, write_status
from .supervised import supervised_loop
from .timing import RunClock

CRITERIA_ORDER = [
    "pseudo_accuracy_val DESC",
    "min_separation_val DESC",
    "mean_true_distance_val ASC",
    "train_loss ASC",
    "eval_id ASC",
]
EXCLUDED_DIAGNOSTICS = [
    "active_dimensions", "correlation_consistency", "avg_margin_val",
    "min_separation_val_ratio", "yodd_norm_fraction",
]


def eligible_clustering_eval_ids(log_rows: list, n_params: int) -> list:
    """ELIGIBLE_CLUSTERING_EVAL_IDS: phase == optimization and no degenerate centroid."""
    eligible = []
    for row in log_rows:
        if row["phase"] != objective_phase(row["eval_id"], n_params):
            raise AssertionError(f"phase label mismatch at eval_id {row['eval_id']}")
        if (row["phase"] == "optimization" and row["n_degenerate_centroids_train"] == 0
                and row["n_degenerate_centroids_val"] == 0):
            eligible.append(row["eval_id"])
    return eligible


def selection_key(row: dict) -> tuple:
    return (-row["pseudo_accuracy_val"], -row["min_separation_val"], row["mean_true_distance_val"],
            row["train_loss"], row["eval_id"])


def select_clustering_checkpoint(log_rows: list, n_params: int) -> tuple[int, dict, list]:
    """SELECT_CLUSTERING_CHECKPOINT_DETERMINISTIC on in-memory rows (pure function)."""
    eligible = eligible_clustering_eval_ids(log_rows, n_params)
    if not eligible:
        raise ValueError("Tidak ada objective evaluation eligible untuk Jalur B")
    elig = set(eligible)
    candidates = sorted((r for r in log_rows if r["eval_id"] in elig), key=selection_key)
    selected = candidates[0]
    record = {
        "selection_method": "deterministic_lexicographic_validation",
        "criteria_order": CRITERIA_ORDER,
        "selected_eval_id": selected["eval_id"],
        "selected_metrics": selected,
        "eligible_eval_ids": eligible,
        "n_eligible_eval_ids": len(eligible),
        "test_used_for_selection": False,
        "diagnostics_excluded_from_selection": EXCLUDED_DIAGNOSTICS,
    }
    return selected["eval_id"], record, candidates


def select_from_source(source_run_dir) -> tuple[int, dict]:
    src = Path(source_run_dir)
    opt = load_json(src / "logs" / "clustering_optimizer_result.json")
    rows = read_jsonl(src / "logs" / "clustering_log.jsonl")
    if len(rows) != opt["nfev"]:
        raise AssertionError("clustering log length != nfev")
    eval_id, record, candidates = select_clustering_checkpoint(rows, opt["n_params"])
    sel_path = src / "logs" / "selected_checkpoint.json"
    log_path = src / "logs" / "checkpoint_selection_log.json"
    payload_log = {"candidates": candidates, "selection_method": record["selection_method"],
                   "criteria_order": CRITERIA_ORDER}
    if sel_path.exists():                       # determinism check instead of overwrite
        if load_json(sel_path)["selected_eval_id"] != eval_id:
            raise RuntimeError("selector is not reproducible for this source run")
    else:
        save_json_atomic(record, sel_path)
        save_json_atomic(payload_log, log_path)
    return eval_id, record


def run_jalur_b(source_run_dir, n_parallel_declared=None, project_root=".", write_workbook: bool = False) -> dict:
    root = Path(project_root).resolve()
    src = Path(source_run_dir).resolve()
    clock = RunClock(n_parallel_declared)
    sm = load_json(src / "config.json")
    if sm["path_type"] != "jalur_a_automatic" or sm["track"] != C.TRACK_PRIMARY:
        raise ValueError("Jalur B starts only from a primary Jalur A run")
    st = read_status(src)
    if st is None or st["status"] != C.STATUS_COMPLETED:
        raise ValueError("source run is not COMPLETED")
    for rel in ("artifacts/clustering_params.bin", "logs/clustering_log.jsonl", "logs/clustering_optimizer_result.json"):
        if not (src / rel).exists():
            raise FileNotFoundError(rel)
    cfg = RunConfig(
        architecture=sm["architecture"], feature_method=sm["feature_method"], k=sm["n_classes"], seed=sm["seed"],
        run_mode=sm["run_mode"], max_nfev_clustering=sm["max_nfev_clustering"],
        max_nfev_supervised=sm["max_nfev_supervised"], n_train_per_class=sm["n_train_per_class"],
        n_val_per_class=sm["n_val_per_class"], n_test_per_class=sm["n_test_per_class"],
        n_parallel_declared=n_parallel_declared, project_root=str(root),
        val_monitor_policy=sm["val_monitor_policy"], n_val_monitor_per_class=sm["n_val_monitor_per_class"],
    )
    selected_eval_id, record = select_from_source(src)
    run_name = f"{sm['run_name']}_jalurB_auto_eval{selected_eval_id:04d}"
    run_dir = root / C.RUNS_DIR / run_name
    run_dir.mkdir(parents=False, exist_ok=False)
    (run_dir / "artifacts").mkdir()
    (run_dir / "logs").mkdir()
    uid = f"{sm['run_uid']}-JB"
    write_status(run_dir, uid, 1, C.STATUS_RUNNING, clock.started_at)
    env = environment_record(root)
    clock.write_run_started(run_dir, run_name, "jalur_b_deterministic_validation_selection",
                            extra={"source_run_uid": sm["run_uid"], "git_commit": env["git_commit"]})
    try:
        manifest = _execute(cfg, sm, src, run_dir, run_name, clock, env, selected_eval_id, record, root)
    except BaseException as exc:
        write_status(run_dir, uid, 1, C.STATUS_FAILED, clock.started_at, error=exc)
        raise
    write_status(run_dir, uid, 1, C.STATUS_COMPLETED, clock.started_at)
    return manifest


def _execute(cfg, sm, src, run_dir, run_name, clock, env, selected_eval_id, record, root) -> dict:
    sa = src / "artifacts"
    clock.begin_stage("load_artifacts")
    x_train = load_npy(sa / "X_train_scaled.npy")
    y_train = load_npy(sa / "y_train.npy")
    x_val = load_npy(sa / "X_val_scaled.npy")
    y_val = load_npy(sa / "y_val.npy")
    mon = select_val_monitor_set(y_val, cfg)
    src_mon = load_json(sa / "val_monitor_manifest.json")
    if mon.tolist() != src_mon["val_positions"]:
        raise RuntimeError("set monitoring validation Jalur B berbeda dari run sumber (G3-05b)")
    save_json_atomic(src_mon, run_dir / "artifacts" / "val_monitor_manifest.json")
    init = build_initialization_bundle(cfg.seed)
    model = build_model(cfg.architecture, init)
    engine = BatchedStatevector(model)
    plog = ParamLog(sa / "clustering_params.bin", model.n_trainable, create=False)
    selected_theta = plog.read(selected_eval_id)
    pm = load_json(sa / "pair_manifest.json")
    pos, cls = cluster_positions_from_manifest(pm, cfg.classes)
    class_index = {c: i for i, c in enumerate(cfg.classes)}
    cls_idx = np.array([class_index[int(c)] for c in cls])
    y_train_idx = np.array([class_index[int(c)] for c in y_train])
    y_val_idx = np.array([class_index[int(c)] for c in y_val])
    phi_train = engine.encode(x_train)
    phi_val = engine.encode(x_val)
    clock.end_stage("load_artifacts")

    clock.begin_stage("quantum_label_extraction")
    ql = extract_quantum_labels(engine, selected_theta, phi_train[pos], cls_idx, cfg, run_dir)
    clock.end_stage("quantum_label_extraction")

    clock.begin_stage("supervised_loop")
    theta_f, summary_s = supervised_loop(engine, selected_theta, ql["labels"], phi_train, y_train_idx,
                                         phi_val[mon], y_val_idx[mon], cfg, run_dir)
    clock.end_stage("supervised_loop")

    clock.begin_stage("structural_diagnostics")
    structural_diagnostics(engine, {"clustering_selected": selected_theta, "supervised_final": theta_f}, phi_val, cfg, run_dir)
    clock.end_stage("structural_diagnostics")

    clock.begin_stage("final_evaluation")
    test_hashes = {k: sm["input_array_sha256"][k] for k in ("X_test_scaled", "y_test", "test_mnist_index")}
    vault = OfficialTestVault(sa, test_hashes, clock)
    metrics = final_evaluation(engine, theta_f, ql["labels"], vault, cfg, run_dir)
    clock.end_stage("final_evaluation")

    timing = clock.summary()
    manifest = {
        "run_name": run_name,
        "run_uid": f"{sm['run_uid']}-JB",
        "source_run_uid": sm["run_uid"],
        "source_condition_id": sm["condition_id"],
        "track": "JALUR_B",
        "path_type": "jalur_b_deterministic_validation_selection",
        "analysis_role": "secondary_sensitivity",
        "timestamp": timing["finished_at"],
        "source_run_dir": str(src.relative_to(root)) if src.is_relative_to(root) else str(src),
        "source_primary_theta": "result.x",
        "selected_eval_id": selected_eval_id,
        "n_eligible_eval_ids": record["n_eligible_eval_ids"],
        "selection_method": record["selection_method"],
        "selection_criteria_order": CRITERIA_ORDER,
        "selected_metrics": record["selected_metrics"],
        "test_used_for_selection": False,
        "inherits_supervised_budget_from_source": True,
        "max_nfev_supervised": cfg.max_nfev_supervised,
        "architecture": cfg.architecture, "feature_method": cfg.feature_method,
        "classes": cfg.classes, "n_classes": cfg.k, "seed": cfg.seed, "run_mode": cfg.run_mode,
        "n_parallel_declared": cfg.n_parallel_declared,
        "device_name": cfg.device_name, "sim_dtype": cfg.sim_dtype, "environment": env,
        "degenerate_quantum_label": ql["diagnostics"]["degenerate_quantum_label"],
        "supervised_optimizer_result": summary_s,
        "final_metrics": {k: v for k, v in metrics.items() if k != "per_class"},
        "total_runtime_sec": timing["total_runtime_sec"],
        "timing": timing,
        "artifact_paths": {
            "quantum_labels": "artifacts/quantum_labels.json",
            "quantum_label_diagnostics": "artifacts/quantum_label_diagnostics.json",
            "structural_diagnostics": "artifacts/structural_diagnostics.json",
            "supervised_params_bin": "artifacts/supervised_params.bin",
            "supervised_params_final": "artifacts/supervised_params_final.npy",
            "supervised_log": "logs/supervised_log.jsonl",
            "supervised_optimizer_result": "logs/supervised_optimizer_result.json",
            "metrics_final": "logs/metrics_final.json",
            "confusion_matrix": "logs/confusion_matrix.npy",
            "source_selected_checkpoint": "logs/selected_checkpoint.json (source run)",
        },
    }
    save_json_atomic(manifest, run_dir / "config.json")
    return manifest


def jalur_b_row(run_dir) -> dict:
    """One row of sheet 17_JalurB_Selection."""
    m = load_json(Path(run_dir) / "config.json")
    started = load_json(Path(run_dir) / "logs" / "run_started.json")
    sel = m["selected_metrics"]
    s = m["supervised_optimizer_result"]
    f = m["final_metrics"]
    return {
        "jalurb_run_name": m["run_name"], "source_run_uid": m["source_run_uid"],
        "condition_id": m["source_condition_id"], "seed": m["seed"], "K": m["n_classes"],
        "feature_method": m["feature_method"], "architecture": m["architecture"],
        "selection_method": m["selection_method"], "n_eligible_eval_ids": m["n_eligible_eval_ids"],
        "selected_eval_id": m["selected_eval_id"], "selected_phase": sel["phase"],
        "selected_pseudo_accuracy_val": sel["pseudo_accuracy_val"],
        "selected_min_separation_val": sel["min_separation_val"],
        "selected_mean_true_distance_val": sel["mean_true_distance_val"],
        "selected_train_loss": sel["train_loss"], "max_nfev_supervised": m["max_nfev_supervised"],
        "supervised_nfev": s["nfev"], "supervised_final_point_eval_id": s["final_point_eval_id"],
        "accuracy": f["accuracy"], "precision_macro": f["precision"], "recall_macro": f["recall"],
        "f1_macro": f["f1_score"], "test_used_for_selection": m["test_used_for_selection"],
        "total_runtime_sec": m["total_runtime_sec"], "started_at": started["started_at"],
        "finished_at": m["timing"]["finished_at"], "total_cpu_sec": m["timing"]["total_cpu_sec"],
        "n_parallel_declared": m["n_parallel_declared"],
    }
