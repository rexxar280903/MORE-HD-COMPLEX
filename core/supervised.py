"""SUPERVISED_LOOP (pseudocode §7; G0-01, G1-02, G1-06, G2-04).

Objective = mean cosine distance between each TRAIN output and the quantum label
of its class (MORE ``myClassifyLoss``; no loss adjuster R, G1-10). Validation is
monitored passively with the same nearest-label rule as FINAL_EVALUATION.
"""

from __future__ import annotations

import numpy as np

from .numerics import argmin_class, cosine_distance_matrix, cosine_distance_rows, is_degenerate
from .optimizer import CobylaRecorder


def supervised_loop(engine, theta_clustering, labels, phi_train, y_train_idx, phi_val, y_val_idx, cfg, run_dir):
    eps = cfg.eps_norm
    rows_v = np.arange(y_val_idx.size)
    target = labels[y_train_idx]

    def core(theta: np.ndarray):
        u = engine.unitary_columns(theta)
        out_t = engine.expectations(phi_train, u)
        loss = float(np.mean(cosine_distance_rows(out_t, target, eps)))
        out_v = engine.expectations(phi_val, u)
        dmat = cosine_distance_matrix(out_v, labels, eps)
        d_true = dmat[rows_v, y_val_idx]
        masked = dmat.copy()
        masked[rows_v, y_val_idx] = np.inf
        return loss, {
            "val_loss": float(np.mean(d_true)),
            "pseudo_accuracy_val": float(np.mean(argmin_class(dmat) == y_val_idx)),
            "avg_margin_val": float(np.mean(masked.min(axis=1) - d_true)),
            "n_degenerate_train_outputs": int(np.sum(is_degenerate(out_t, eps))),
            "n_degenerate_val_outputs": int(np.sum(is_degenerate(out_v, eps))),
        }

    recorder = CobylaRecorder(
        "supervised", run_dir, engine.model.n_trainable, cfg.max_nfev_supervised,
        cfg.cobyla_tol, cfg.cobyla_rhobeg, cfg.run_mode, core,
        extra_summary={"device_name": cfg.device_name, "sim_dtype": cfg.sim_dtype,
                       "loss_adjuster_policy": cfg.loss_adjuster_policy},
    )
    return recorder.run(theta_clustering)
