"""Classical reference baselines (pseudocode §12; G1-10).

Chance 1/K, Nearest Centroid and multinomial Logistic Regression on exactly the
8-channel arrays given to the circuit (``X_*_scaled.npy`` of the matching
primary MORE-HD run). No preprocessing is repeated. Official-test arrays are
loaded only after both models are frozen.
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

import numpy as np
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neighbors import NearestCentroid

from . import constants as C
from .artifact_io import load_json, load_npy, save_json_atomic, save_npy_atomic, sha256_file
from .config import primary_condition_id, run_name_for
from .environment import git_info
from .run_status import existing_attempts

LR_C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
LR_MAX_ITER = 5000
ARRAYS = ("X_train_scaled", "y_train", "X_val_scaled", "y_val", "X_test_scaled", "y_test")


def _attempt_with_arrays(base: Path) -> Path | None:
    candidates = existing_attempts(base)
    completed = [p for _, p, st in candidates if st and st.get("status") == C.STATUS_COMPLETED]
    if completed:
        return completed[0]
    with_arrays = [p for _, p, _ in candidates if (p / "artifacts" / "feature_metadata.json").exists()]
    return with_arrays[-1] if with_arrays else None


def resolve_baseline_source(root: Path, seed: int, feature: str, k: int, sizes=(1000, 100, 200),
                            run_mode: str = "CONFIRMATORY", pilot_tag: str = "") -> dict:
    a_base = root / C.RUNS_DIR / run_name_for(C.ARCH_MORE_HD, feature, k, seed, sizes, run_mode, pilot_tag)
    d_base = root / C.RUNS_DIR / run_name_for(C.ARCH_MORE_HD_C, feature, k, seed, sizes, run_mode, pilot_tag)
    dir_a = _attempt_with_arrays(a_base)
    if dir_a is None:
        raise FileNotFoundError(f"primary MORE-HD data arrays not found for {a_base.name}")
    hashes_a = {n: sha256_file(dir_a / "artifacts" / f"{n}.npy") for n in ARRAYS}
    dir_d = _attempt_with_arrays(d_base)
    hash_match = None
    if dir_d is not None:
        hashes_d = {n: sha256_file(dir_d / "artifacts" / f"{n}.npy") for n in ARRAYS}
        if hashes_d != hashes_a:
            raise RuntimeError(f"Array input A dan D berbeda untuk cell ini: {a_base.name}")
        hash_match = True
    split_path = root / C.SPLITS_DIR / f"seed{seed}.json"
    return {
        "dir_A": dir_a,
        "run_uid_A": f"{primary_condition_id(k, feature, C.ARCH_MORE_HD)}-S{seed}",
        "check_run_uid_D": f"{primary_condition_id(k, feature, C.ARCH_MORE_HD_C)}-S{seed}" if dir_d else None,
        "input_array_sha256": hashes_a,
        "input_hash_match_A_D": hash_match,
        "split_manifest_hash": sha256_file(split_path) if split_path.exists() else None,
    }


def _fit_lr(x, y, c):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model = LogisticRegression(C=c, l1_ratio=0.0, solver="lbfgs", max_iter=LR_MAX_ITER,
                                   fit_intercept=True, class_weight=None).fit(x, y)
    n_iter = int(np.max(model.n_iter_))
    converged = n_iter < LR_MAX_ITER and not any(isinstance(w.message, ConvergenceWarning) for w in caught)
    return model, n_iter, converged


def run_baseline_cell(root: Path, seed: int, feature: str, k: int, out_root: Path, sizes=(1000, 100, 200),
                      access_log: list | None = None, run_mode: str = "CONFIRMATORY", pilot_tag: str = "") -> list:
    src = resolve_baseline_source(root, seed, feature, k, sizes, run_mode, pilot_tag)
    cell_dir = out_root / f"S{seed}_cls-{'-'.join(str(c) for c in range(k))}_{feature}"
    cell_dir.mkdir(parents=True, exist_ok=False)
    art = src["dir_A"] / "artifacts"

    def load(name):
        if access_log is not None:
            access_log.append(name)
        return load_npy(art / f"{name}.npy")

    x_tr, y_tr, x_va, y_va = load("X_train_scaled"), load("y_train"), load("X_val_scaled"), load("y_val")
    if x_tr.shape[1] != 8 or x_va.shape[1] != 8:
        raise AssertionError("baseline input must have 8 channels")
    nc = NearestCentroid(metric="euclidean", shrink_threshold=None).fit(x_tr, y_tr)
    grid = []
    for c in LR_C_GRID:
        model, n_iter, conv = _fit_lr(x_tr, y_tr, c)
        grid.append({"C": c, "val_accuracy": float(accuracy_score(y_va, model.predict(x_va))),
                     "n_iter": n_iter, "converged": conv})
    best = max(g["val_accuracy"] for g in grid)
    c_sel = min(g["C"] for g in grid if g["val_accuracy"] == best)      # tie -> smallest C
    lr, n_iter_sel, conv_sel = _fit_lr(x_tr, y_tr, c_sel)               # refit on TRAIN only
    save_json_atomic({"grid": grid, "C_selected": c_sel, "selected_n_iter": n_iter_sel,
                      "selected_converged": conv_sel}, cell_dir / "LR_cv_log.json")
    if access_log is not None:
        access_log.append("MODELS_FROZEN")
    x_te, y_te = load("X_test_scaled"), load("y_test")
    rows = []
    labels = list(range(k))
    for name, model in (("NC", nc), ("LR", lr)):
        y_pred = model.predict(x_te).astype(np.int64)
        save_npy_atomic(y_pred, cell_dir / f"{name}_test_predictions.npy")
        save_npy_atomic(confusion_matrix(y_te, y_pred, labels=labels), cell_dir / f"{name}_confusion_matrix.npy")
        rows.append({
            "seed": seed, "feature_method": feature, "K": k, "baseline": name,
            "accuracy": float(accuracy_score(y_te, y_pred)),
            "macro_f1": float(f1_score(y_te, y_pred, labels=labels, average="macro", zero_division=0)),
            "per_class_f1": f1_score(y_te, y_pred, labels=labels, average=None, zero_division=0).tolist(),
            "C_selected": c_sel if name == "LR" else None,
            "lr_converged": conv_sel if name == "LR" else None,
            "n_test": int(y_te.size),
            "source_run_uid_A": src["run_uid_A"], "check_run_uid_D": src["check_run_uid_D"],
            "input_hash_match_A_D": src["input_hash_match_A_D"],
        })
    rows.append({"seed": seed, "feature_method": feature, "K": k, "baseline": "CHANCE",
                 "accuracy": 1.0 / k, "macro_f1": 1.0 / k, "per_class_f1": None, "C_selected": None,
                 "lr_converged": None, "n_test": int(y_te.size), "source_run_uid_A": src["run_uid_A"],
                 "check_run_uid_D": src["check_run_uid_D"], "input_hash_match_A_D": src["input_hash_match_A_D"]})
    save_json_atomic({
        "seed": seed, "feature_method": feature, "K": k,
        "source_run_uid_A": src["run_uid_A"], "source_run_dir_A": str(src["dir_A"]),
        "check_run_uid_D": src["check_run_uid_D"], "input_array_sha256": src["input_array_sha256"],
        "input_hash_match_A_D": src["input_hash_match_A_D"], "split_manifest_hash": src["split_manifest_hash"],
        "loss_adjuster_policy": C.LOSS_ADJUSTER_POLICY, "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__, **git_info(root),
    }, cell_dir / "baseline_cell_manifest.json")
    return rows


def write_results_csv(rows: list, path: Path) -> None:
    fields = ["seed", "feature_method", "K", "baseline", "accuracy", "macro_f1", "per_class_f1", "C_selected",
              "lr_converged", "n_test", "source_run_uid_A", "check_run_uid_D", "input_hash_match_A_D"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def baseline_config() -> dict:
    return {
        "models": ["NC", "LR"], "lr_c_grid": list(LR_C_GRID), "lr_penalty": "l2 (l1_ratio=0.0)",
        "lr_solver": "lbfgs", "lr_max_iter": LR_MAX_ITER, "lr_tie_rule": "SMALLEST_C",
        "nc_metric": "euclidean", "loss_adjuster_policy": C.LOSS_ADJUSTER_POLICY,
    }
