"""STRUCTURAL_DIAGNOSTICS and FINAL_EVALUATION (pseudocode §4.3.2, §8; G0-01, G2-05, G3-02, G4-02)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, precision_score, recall_score

from .artifact_io import load_npy, save_json_atomic, save_npy_atomic, sha256_array
from .constants import ACTIVITY_THRESHOLDS
from .numerics import argmin_class, cosine_distance_matrix, is_degenerate, observable_activity


class OfficialTestAccessError(RuntimeError):
    """Raised when official-test arrays are requested outside FINAL_EVALUATION (G0-01)."""


class OfficialTestVault:
    """Holds only the *paths* and hashes of the official-test arrays.

    ``open()`` succeeds only while the run clock is inside the
    ``final_evaluation`` stage, so no training/monitoring stage can read test data.
    """

    def __init__(self, artifacts_dir: str | Path, hashes: dict, clock):
        self.dir = Path(artifacts_dir)
        self.hashes = dict(hashes)
        self.clock = clock
        self.opened = False

    def open(self):
        if self.clock.current_stage != "final_evaluation":
            raise OfficialTestAccessError(
                f"official test requested during stage {self.clock.current_stage!r}"
            )
        x = load_npy(self.dir / "X_test_scaled.npy")
        y = load_npy(self.dir / "y_test.npy")
        idx = load_npy(self.dir / "test_mnist_index.npy")
        for name, arr in (("X_test_scaled", x), ("y_test", y), ("test_mnist_index", idx)):
            if sha256_array(arr) != self.hashes[name]:
                raise OfficialTestAccessError(f"{name} hash mismatch")
        self.opened = True
        return x, y, idx


def structural_diagnostics(engine, checkpoints: dict, phi_val_full, cfg, run_dir) -> dict:
    """One pass on the FULL validation split per checkpoint (never the test split)."""
    labels = list(engine.model.observable_labels)
    yodd = tuple(i for i, lab in enumerate(labels) if lab.count("Y") % 2 == 1)
    result = {
        "source": "validation_full",
        "eps_norm": cfg.eps_norm,
        "primary_threshold": cfg.active_dim_threshold,
        "thresholds": list(ACTIVITY_THRESHOLDS),
        "observables": labels,
        "yodd_labels": [labels[i] for i in yodd],
        "checkpoints": {},
    }
    for name, theta in checkpoints.items():
        out = engine.expectations(phi_val_full, engine.unitary_columns(theta))
        result["checkpoints"][name] = observable_activity(out, labels, yodd, cfg.eps_norm, cfg.active_dim_threshold)
    save_json_atomic(result, Path(run_dir) / "artifacts" / "structural_diagnostics.json")
    return result


def final_evaluation(engine, theta_final, labels, vault: OfficialTestVault, cfg, run_dir) -> dict:
    """FINAL_EVALUATION: the only place where official-test data are read."""
    x_test, y_test, test_idx = vault.open()
    classes = list(cfg.classes)
    y_idx = np.searchsorted(classes, y_test)
    out = engine.outputs(x_test, theta_final)
    dmat = cosine_distance_matrix(out, labels, cfg.eps_norm)
    pred_idx = argmin_class(dmat)
    y_pred = np.asarray(classes)[pred_idx]
    rows = np.arange(y_test.size)
    d_true = dmat[rows, y_idx]
    d_pred = dmat[rows, pred_idx]
    masked = dmat.copy()
    masked[rows, y_idx] = np.inf
    margin = masked.min(axis=1) - d_true
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    p_c, r_c, f_c, s_c = precision_recall_fscore_support(y_test, y_pred, labels=classes, zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, labels=classes, average="macro", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, labels=classes, average="macro", zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, labels=classes, average="macro", zero_division=0)),
        "n_test": int(y_test.size),
        "n_correct": int(np.trace(cm)),
        "n_wrong": int(cm.sum() - np.trace(cm)),
        "n_degenerate_test_outputs": int(np.sum(is_degenerate(out, cfg.eps_norm))),
        "per_class": [
            {"class": int(c), "support": int(s_c[i]), "precision": float(p_c[i]), "recall": float(r_c[i]),
             "f1": float(f_c[i]), "correct": int(cm[i, i]), "wrong": int(cm[i].sum() - cm[i, i])}
            for i, c in enumerate(classes)
        ],
    }
    run_dir = Path(run_dir)
    save_json_atomic(metrics, run_dir / "logs" / "metrics_final.json")
    save_npy_atomic(cm, run_dir / "logs" / "confusion_matrix.npy")
    save_npy_atomic(y_pred.astype(np.int64), run_dir / "logs" / "test_predictions.npy")
    save_npy_atomic(dmat, run_dir / "logs" / "test_label_distances.npy")
    save_npy_atomic(margin, run_dir / "logs" / "test_margins.npy")
    save_npy_atomic(test_idx, run_dir / "logs" / "test_mnist_index.npy")
    save_npy_atomic(d_pred, run_dir / "logs" / "test_distance_pred.npy")
    return metrics
