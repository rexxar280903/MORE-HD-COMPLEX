"""One run = one condition (pseudocode §9.1 MAIN, §11.11.7 ABLATION_MAIN; MORE reference track).

The same function executes the primary track (MORE-HD, MORE-HD-C), the
targeted ablation (MORE-HD-60P, MORE-HD-C-FixedRZ) and the MORE reproduction
(MORE-REPRO). Only the circuit and the identifiers differ; data, pairing,
correlation matrix, losses, budgets and evaluation are shared code.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from . import constants as C
from .artifact_io import load_json, save_json_atomic, save_npy_atomic, sha256_array, sha256_file, sha256_json
from .circuits import BatchedStatevector, backend_self_check, build_initialization_bundle, build_model, initial_params
from .clustering import ClusteringInputs, clustering_loop
from .config import RunConfig, primary_condition_id, run_name_for, validate_config
from .correlation import correlation_matrix
from .data_pipeline import run_data_pipeline
from .environment import environment_record
from .evaluation import OfficialTestVault, final_evaluation, structural_diagnostics
from .mnist import load_mnist
from .pairing import build_pairing, cluster_positions_from_manifest
from .quantum_labels import extract_quantum_labels
from .run_status import check_attempt_allowed, write_status
from .seeds import manifest_seeds
from .splits import create_or_load_split_manifest
from .supervised import supervised_loop
from .timing import RunClock

PATH_TYPE = {
    C.TRACK_PRIMARY: "jalur_a_automatic",
    C.TRACK_ABLATION: "ablation",
    C.TRACK_MORE_REFERENCE: "more_reference",
}
TEMPLATE = {
    C.TRACK_PRIMARY: C.MASTER_SPREADSHEET_PATH,
    C.TRACK_ABLATION: C.ABLATION_SPREADSHEET_PATH,
    C.TRACK_MORE_REFERENCE: C.MORE_REFERENCE_SPREADSHEET_PATH,
}


def select_val_monitor_set(y_val: np.ndarray, cfg: RunConfig) -> np.ndarray:
    """SELECT_VAL_MONITOR_SET (G3-05b): FULL_VAL or first n per class in manifest order."""
    if cfg.val_monitor_policy == "FULL_VAL":
        return np.arange(y_val.size)
    keep = []
    for c in cfg.classes:
        pos = np.flatnonzero(y_val == c)
        keep.extend(pos[: cfg.n_val_monitor_per_class].tolist())
    return np.asarray(keep, dtype=np.int64)


def pair_identity_hash(pair_manifest: dict) -> str:
    ident = {c: {"train_positions": v["train_positions"], "mnist_indices": v["mnist_indices"]}
             for c, v in pair_manifest["selected_by_class"].items()}
    return sha256_json(ident)


def primary_run_dir(root: Path, cfg: RunConfig, architecture: str) -> Path:
    name = run_name_for(architecture, cfg.feature_method, cfg.k, cfg.seed,
                        (cfg.n_train_per_class, cfg.n_val_per_class, cfg.n_test_per_class),
                        cfg.run_mode, cfg.pilot_tag)
    return root / C.RUNS_DIR / name


def paired_input_check(root: Path, cfg: RunConfig, input_hashes: dict) -> dict:
    """Compare this run's arrays with the matching primary MORE-HD (A) run when it exists."""
    a_dir = primary_run_dir(root, cfg, C.ARCH_MORE_HD)
    meta_path = a_dir / "artifacts" / "feature_metadata.json"
    if cfg.architecture == C.ARCH_MORE_HD or not meta_path.exists():
        return {"reference_run_dir": str(a_dir), "status": "SELF" if cfg.architecture == C.ARCH_MORE_HD else "A_NOT_AVAILABLE"}
    ref = load_json(meta_path)["array_sha256"]
    mismatched = [k for k, v in input_hashes.items() if ref.get(k) != v]
    if mismatched:
        raise RuntimeError(f"input arrays differ from primary A run {a_dir}: {mismatched}")
    return {"reference_run_dir": str(a_dir), "status": "MATCH"}


def run_condition(cfg: RunConfig, mnist=None, write_workbook: bool = True) -> dict:
    root = Path(cfg.project_root).resolve()
    warnings = validate_config(cfg)
    clock = RunClock(cfg.n_parallel_declared)
    base_cfg = RunConfig(**{**cfg.__dict__, "attempt": 1})
    check_attempt_allowed(root / C.RUNS_DIR / base_cfg.run_name, cfg.attempt)
    run_dir = root / C.RUNS_DIR / cfg.run_name
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(exist_ok=False)                       # CREATE_RUN_DIRECTORY_EXCLUSIVE
    (run_dir / "artifacts").mkdir()
    (run_dir / "logs").mkdir()
    write_status(run_dir, cfg.run_uid, cfg.attempt, C.STATUS_RUNNING, clock.started_at)
    env = environment_record(root)
    clock.write_run_started(run_dir, cfg.run_name, PATH_TYPE[cfg.track], extra={
        "run_uid": cfg.run_uid, "condition_id": cfg.condition_id, "attempt": cfg.attempt,
        "track": cfg.track, "architecture": cfg.architecture, "git_commit": env["git_commit"],
        "git_dirty": env["git_dirty"], "config_warnings": warnings,
    })
    template = root / TEMPLATE[cfg.track]
    if write_workbook and template.exists():
        shutil.copyfile(template, run_dir / C.LOCAL_RUN_SPREADSHEET_NAME)
    try:
        result = _execute(cfg, root, run_dir, clock, env, warnings, mnist)
    except BaseException as exc:
        write_status(run_dir, cfg.run_uid, cfg.attempt, C.STATUS_FAILED, clock.started_at, error=exc)
        raise
    write_status(run_dir, cfg.run_uid, cfg.attempt, C.STATUS_COMPLETED, clock.started_at)
    if write_workbook and (run_dir / C.LOCAL_RUN_SPREADSHEET_NAME).exists():
        from .workbook import update_local_run_spreadsheet

        update_local_run_spreadsheet(run_dir)
    return result


def _execute(cfg: RunConfig, root: Path, run_dir: Path, clock: RunClock, env: dict, warnings: list, mnist) -> dict:
    art = run_dir / "artifacts"
    classes = cfg.classes
    k = cfg.k
    executions = {}

    # ---- data_pipeline -------------------------------------------------------------
    clock.begin_stage("data_pipeline")
    if mnist is None:
        mnist = load_mnist(root / C.MNIST_ROOT, download=cfg.mnist_download)
    split_manifest, split_path = create_or_load_split_manifest(
        cfg.seed, mnist.train_labels, mnist.test_labels, root / C.SPLITS_DIR
    )
    data = run_data_pipeline(cfg, mnist, split_manifest, run_dir)
    clock.end_stage("data_pipeline")

    # ---- setup ------------------------------------------------------------------------
    clock.begin_stage("setup")
    corr = correlation_matrix(data.x_train, data.y_train, classes)
    s_matrix = corr["S"]
    save_npy_atomic(s_matrix, art / "correlation_matrix.npy")
    save_npy_atomic(corr["raw_mse"], art / "correlation_raw_mse.npy")
    save_json_atomic({
        "rule": corr["rule"], "classes": classes, "class_to_index": corr["class_to_index"],
        "n_mean_samples_per_class": corr["n_mean_samples"],
        "normalizer_max_offdiag_mse": corr["normalizer_max_offdiag_mse"],
        "fallback_all_offdiag_one": corr["fallback_all_offdiag_one"],
    }, art / "correlation_info.json")
    paired_check = paired_input_check(root, cfg, data.input_hashes)

    init = build_initialization_bundle(cfg.seed)
    model = build_model(cfg.architecture, init)
    x0 = initial_params(cfg.architecture, init)
    if x0.shape != (model.n_trainable,):
        raise AssertionError("initial parameter length mismatch")
    save_npy_atomic(x0, art / "initial_params.npy")
    init_manifest = {
        "ry_core_seed": init["ry_core_seed"], "rz_phase_seed": init["rz_phase_seed"],
        "ry_extra_seed": init["ry_extra_seed"], "more_init_seed": init["more_init_seed"],
        "ry_core_sha256": sha256_array(init["ry_core"]), "rz_phase_sha256": sha256_array(init["rz_phase"]),
        "ry_extra_sha256": sha256_array(init["ry_extra"]), "more_init_sha256": sha256_array(init["more_init"]),
        "initial_params_sha256": sha256_array(x0),
        "packing_rule": model.meta.get("packing_rule"),
        "init_distribution": "U[0,1) (qiskit-machine-learning default, MORE)" if cfg.architecture == C.ARCH_MORE_REPRO else "U[0, 2pi)",
    }
    if model.fixed_values is not None:
        save_npy_atomic(model.fixed_values, art / "fixed_rz.npy")
        init_manifest["fixed_rz_sha256"] = sha256_array(model.fixed_values)
    save_json_atomic(init_manifest, art / "initialization_manifest.json")
    engine = BatchedStatevector(model)
    self_check = backend_self_check(model, x0, data.x_train[: min(4, data.x_train.shape[0])])
    if self_check["max_abs_diff"] > C.BACKEND_SELF_CHECK_TOL:
        raise RuntimeError(f"engine differs from PennyLane default.qubit: {self_check}")
    mon = select_val_monitor_set(data.y_val, cfg)
    save_json_atomic({
        "val_monitor_policy": cfg.val_monitor_policy,
        "n_val_monitor_per_class": cfg.n_val_monitor_per_class,
        "n_val_monitor_total": int(mon.size),
        "val_positions": mon.tolist(),
        "val_mnist_index": data.val_mnist_index[mon].tolist(),
    }, art / "val_monitor_manifest.json")
    class_index = {c: i for i, c in enumerate(classes)}
    y_train_idx = np.array([class_index[int(c)] for c in data.y_train])
    y_val_idx = np.array([class_index[int(c)] for c in data.y_val])
    phi_train = engine.encode(data.x_train)
    phi_val_full = engine.encode(data.x_val)
    phi_val_mon = phi_val_full[mon]
    clock.end_stage("setup")
    executions["setup"] = engine.n_executions

    # ---- clustering_loop ----------------------------------------------------------------
    clock.begin_stage("clustering_loop")
    pairing = build_pairing(data.y_train, data.train_mnist_index, classes, cfg.seed)
    save_json_atomic(pairing["manifest"], art / "pair_manifest.json")
    save_json_atomic(pairing["stats"], art / "pair_stats.json")
    sel = pairing["selected_positions"]
    cluster_class_idx = np.array([class_index[int(c)] for c in pairing["selected_classes"]])
    pair_s = s_matrix[cluster_class_idx[pairing["pair_a"]], cluster_class_idx[pairing["pair_b"]]]
    phi_cluster = phi_train[sel]
    yodd = tuple(i for i, lab in enumerate(model.observable_labels) if lab.count("Y") % 2 == 1)
    inp = ClusteringInputs(phi_cluster, cluster_class_idx, pairing["pair_a"], pairing["pair_b"], pair_s,
                           phi_val_mon, y_val_idx[mon], k, s_matrix, yodd)
    n_before = engine.n_executions
    theta_c, summary_c = clustering_loop(engine, x0, inp, cfg, run_dir)
    executions["clustering_per_eval"] = (engine.n_executions - n_before) / max(1, summary_c["nfev"])
    clock.end_stage("clustering_loop")

    # ---- quantum_label_extraction ---------------------------------------------------------
    clock.begin_stage("quantum_label_extraction")
    pm = load_json(art / "pair_manifest.json")
    pos_from_disk, _ = cluster_positions_from_manifest(pm, classes)
    if not np.array_equal(pos_from_disk, sel):
        raise AssertionError("pair_manifest.json does not reproduce the clustering samples")
    ql = extract_quantum_labels(engine, theta_c, phi_train[pos_from_disk], cluster_class_idx, cfg, run_dir)
    labels = ql["labels"]
    clock.end_stage("quantum_label_extraction")

    # ---- supervised_loop -----------------------------------------------------------------
    clock.begin_stage("supervised_loop")
    n_before = engine.n_executions
    theta_f, summary_s = supervised_loop(engine, theta_c, labels, phi_train, y_train_idx,
                                         phi_val_mon, y_val_idx[mon], cfg, run_dir)
    executions["supervised_per_eval"] = (engine.n_executions - n_before) / max(1, summary_s["nfev"])
    clock.end_stage("supervised_loop")

    # ---- structural_diagnostics ---------------------------------------------------------
    clock.begin_stage("structural_diagnostics")
    structural_diagnostics(engine, {"clustering_final": theta_c, "supervised_final": theta_f}, phi_val_full, cfg, run_dir)
    clock.end_stage("structural_diagnostics")

    # ---- final_evaluation -----------------------------------------------------------------
    clock.begin_stage("final_evaluation")
    vault = OfficialTestVault(art, data.test_hashes, clock)
    metrics = final_evaluation(engine, theta_f, labels, vault, cfg, run_dir)
    clock.end_stage("final_evaluation")

    manifest = build_manifest(cfg, root, run_dir, clock, env, warnings, split_path, data, corr, model,
                              summary_c, summary_s, metrics, ql, pm, paired_check, self_check, executions)
    save_json_atomic(manifest, run_dir / "config.json")
    return manifest


def build_manifest(cfg, root, run_dir, clock, env, warnings, split_path, data, corr, model,
                   summary_c, summary_s, metrics, ql, pair_manifest, paired_check, self_check, executions) -> dict:
    fm = cfg.feature_method
    seeds = manifest_seeds(cfg.seed)
    matching = {}
    if cfg.track != C.TRACK_PRIMARY:
        matching = {
            "matching_run_uid_A": f"{primary_condition_id(cfg.k, fm, C.ARCH_MORE_HD)}-S{cfg.seed}",
            "matching_run_uid_D": f"{primary_condition_id(cfg.k, fm, C.ARCH_MORE_HD_C)}-S{cfg.seed}",
        }
    timing = clock.summary()
    resources = model.resource_summary()
    counts = resources["gate_counts"]
    body_counts: dict = {}
    for op in model.body:
        body_counts[op.gate] = body_counts.get(op.gate, 0) + 1
    manifest = {
        "run_id": cfg.condition_id,
        "condition_id": cfg.condition_id,
        "run_uid": cfg.run_uid,
        "run_name": cfg.run_name,
        "attempt": cfg.attempt,
        "track": cfg.track,
        "path_type": PATH_TYPE[cfg.track],
        "model_code": cfg.model_code,
        "timestamp": timing["finished_at"],
        "classes": cfg.classes,
        "n_classes": cfg.k,
        "n_train_per_class": cfg.n_train_per_class,
        "n_val_per_class": cfg.n_val_per_class,
        "n_test_per_class": cfg.n_test_per_class,
        "max_nfev_clustering": cfg.max_nfev_clustering,
        "max_nfev_supervised": cfg.max_nfev_supervised,
        "architecture": cfg.architecture,
        "feature_method": fm,
        "n_data_qubits": cfg.n_data_qubits if cfg.architecture != C.ARCH_MORE_REPRO else 8,
        "n_readout_qubits": cfg.n_readout_qubits if cfg.architecture != C.ARCH_MORE_REPRO else 1,
        "n_wires": model.n_wires,
        "n_input_channels": cfg.n_input_channels,
        "observables": list(model.observable_labels),
        "pca_n_components": C.PCA_N_COMPONENTS if fm == "PCA" else None,
        "pca_svd_solver": cfg.pca_svd_solver if fm == "PCA" else None,
        "pca_random_state": cfg.seed if fm == "PCA" else None,
        "hu_raw_dim": C.HU_RAW_DIM if fm == "HU" else None,
        "hu_input_mode": C.HU_INPUT_MODE if fm == "HU" else None,
        "hu_padding_value": C.HU_PADDING_VALUE if fm == "HU" else None,
        "zernike_terms": [list(t) for t in C.ZERNIKE_TERMS] if fm == "ZERNIKE" else None,
        "cobyla_tol": cfg.cobyla_tol,
        "cobyla_rhobeg": cfg.cobyla_rhobeg,
        "scipy_version": env["versions"]["scipy"],
        "numpy_version": env["versions"]["numpy"],
        "pennylane_version": env["versions"]["pennylane"],
        "sklearn_version": env["versions"]["sklearn"],
        "python_version": env["versions"]["python"],
        "environment": env,
        "simulation_mode": cfg.simulation_mode,
        "shots": cfg.shots,
        "sim_dtype": cfg.sim_dtype,
        "device_name": cfg.device_name,
        "backend_self_check": self_check,
        "seed": cfg.seed,
        "run_mode": cfg.run_mode,
        "pilot_tag": cfg.pilot_tag,
        **seeds,
        "subseed_rule": C.SUBSEED_RULE,
        "split_manifest": str(Path(split_path).relative_to(root)) if Path(split_path).is_absolute() else str(split_path),
        "split_manifest_sha256": sha256_file(split_path),
        "paired_split_key": f"seed{cfg.seed}_K{cfg.k}",
        "paired_pair_manifest_sha256": pair_identity_hash(pair_manifest),
        "input_array_sha256": data.input_hashes,
        "paired_input_check": paired_check,
        "n_cluster_pair_samples": cfg.n_cluster_pair_samples,
        "pair_balance_policy": cfg.pair_balance_policy,
        "pair_weighting": cfg.pair_weighting,
        "correlation_rule": corr["rule"],
        "correlation_fallback_all_offdiag_one": corr["fallback_all_offdiag_one"],
        "active_dim_threshold": cfg.active_dim_threshold,
        "eps_norm": cfg.eps_norm,
        "centroid_rule": cfg.centroid_rule,
        "centroid_source": cfg.centroid_source,
        "cluster_output_cache": cfg.cluster_output_cache,
        "val_monitor_policy": cfg.val_monitor_policy,
        "n_val_monitor_per_class": cfg.n_val_monitor_per_class,
        "loss_adjuster_policy": cfg.loss_adjuster_policy,
        "n_parallel_declared": cfg.n_parallel_declared,
        "degenerate_quantum_label": ql["diagnostics"]["degenerate_quantum_label"],
        "mnist_root": C.MNIST_ROOT,
        "mnist_download": cfg.mnist_download,
        "local_spreadsheet_path": str((run_dir / C.LOCAL_RUN_SPREADSHEET_NAME).relative_to(root)),
        "config_warnings": warnings,
        "circuit_resources": resources,
        "n_trainable_params": model.n_trainable,
        "n_fixed_rz_params": model.n_fixed,
        "n_variational_layers": model.meta.get("n_variational_layers"),
        "variational_gate_count": sum(v for g, v in body_counts.items() if g in ("RY", "RZ", "RX")),
        "entangling_block_count": model.meta.get("entangling_block_count"),
        "cnot_count": counts.get("CNOT", 0),
        "circuit_executions": executions,
        **matching,
        "optimizer_results": {"clustering": summary_c, "supervised": summary_s},
        "final_metrics": {k: v for k, v in metrics.items() if k != "per_class"},
        "artifact_paths": artifact_paths(cfg, model),
    }
    manifest["total_runtime_sec"] = timing["total_runtime_sec"]
    manifest["timing"] = timing
    return manifest


def artifact_paths(cfg, model) -> dict:
    paths = {
        "feature_metadata": "artifacts/feature_metadata.json",
        "scaler_params": "artifacts/scaler_params.json",
        "X_train_scaled": "artifacts/X_train_scaled.npy", "y_train": "artifacts/y_train.npy",
        "train_mnist_index": "artifacts/train_mnist_index.npy",
        "X_val_scaled": "artifacts/X_val_scaled.npy", "y_val": "artifacts/y_val.npy",
        "val_mnist_index": "artifacts/val_mnist_index.npy",
        "X_test_scaled": "artifacts/X_test_scaled.npy", "y_test": "artifacts/y_test.npy",
        "test_mnist_index": "artifacts/test_mnist_index.npy",
        "val_monitor_manifest": "artifacts/val_monitor_manifest.json",
        "correlation_matrix": "artifacts/correlation_matrix.npy",
        "correlation_raw_mse": "artifacts/correlation_raw_mse.npy",
        "correlation_info": "artifacts/correlation_info.json",
        "initial_params": "artifacts/initial_params.npy",
        "initialization_manifest": "artifacts/initialization_manifest.json",
        "pair_manifest": "artifacts/pair_manifest.json", "pair_stats": "artifacts/pair_stats.json",
        "clustering_params_bin": "artifacts/clustering_params.bin",
        "clustering_params_final": "artifacts/clustering_params_final.npy",
        "clustering_params_best_observed": "artifacts/clustering_params_best_observed.npy",
        "clustering_log": "logs/clustering_log.jsonl",
        "clustering_callback_log": "logs/clustering_callback_log.jsonl",
        "clustering_optimizer_result": "logs/clustering_optimizer_result.json",
        "quantum_labels": "artifacts/quantum_labels.json",
        "quantum_label_diagnostics": "artifacts/quantum_label_diagnostics.json",
        "supervised_params_bin": "artifacts/supervised_params.bin",
        "supervised_params_final": "artifacts/supervised_params_final.npy",
        "supervised_params_best_observed": "artifacts/supervised_params_best_observed.npy",
        "supervised_log": "logs/supervised_log.jsonl",
        "supervised_callback_log": "logs/supervised_callback_log.jsonl",
        "supervised_optimizer_result": "logs/supervised_optimizer_result.json",
        "structural_diagnostics": "artifacts/structural_diagnostics.json",
        "metrics_final": "logs/metrics_final.json",
        "confusion_matrix": "logs/confusion_matrix.npy",
        "test_predictions": "logs/test_predictions.npy",
        "test_label_distances": "logs/test_label_distances.npy",
        "test_margins": "logs/test_margins.npy",
        "run_started": "logs/run_started.json",
        "run_status": "logs/run_status.json",
        "stage_timing": "logs/stage_timing.jsonl",
        "run_spreadsheet": C.LOCAL_RUN_SPREADSHEET_NAME,
    }
    if cfg.feature_method == "PCA":
        paths["pca_model"] = "artifacts/pca_model.joblib"
    if model.fixed_values is not None:
        paths["fixed_rz"] = "artifacts/fixed_rz.npy"
    return paths
