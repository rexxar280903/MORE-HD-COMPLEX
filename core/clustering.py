"""CLUSTERING_LOOP (pseudocode §5; G0-01, G1-02, G1-03, G1-06, G2-04..G2-07, G3-05).

The function receives only TRAIN and the VALIDATION monitoring set; it never
receives official-test data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .circuits import BatchedStatevector
from .constants import simplex_bound
from .numerics import (
    active_dimensions_centroid,
    argmin_class,
    centroids_by_class,
    correlation_consistency,
    cosine_distance_matrix,
    cosine_distance_rows,
    is_degenerate,
    min_separation,
    yodd_norm_fraction,
)
from .optimizer import CobylaRecorder


@dataclass
class ClusteringInputs:
    phi_cluster: np.ndarray        # encoded 5K clustering samples, (class, position) order
    cluster_class_idx: np.ndarray  # class index (0..K-1) of each clustering sample
    pair_a: np.ndarray             # pair endpoints as rows of phi_cluster
    pair_b: np.ndarray
    pair_s: np.ndarray             # S[class_i, class_j] per pair
    phi_val: np.ndarray            # encoded validation monitoring set
    y_val_idx: np.ndarray          # class index of each validation sample
    k: int
    s_matrix: np.ndarray
    yodd_idx: tuple


def class_groups(class_idx: np.ndarray, k: int) -> list:
    return [np.flatnonzero(class_idx == c) for c in range(k)]


def pair_loss(out_cluster: np.ndarray, pair_a, pair_b, pair_s, eps_norm: float) -> tuple[float, np.ndarray]:
    """train_loss = mean_pairs(-S_ij * cosine_distance(v_i, v_j)) (MORE myClusterLoss minus its +2)."""
    dist = cosine_distance_rows(out_cluster[pair_a], out_cluster[pair_b], eps_norm)
    losses = -pair_s * dist
    return float(np.mean(losses)), losses


def clustering_eval_metrics(out_c: np.ndarray, out_v: np.ndarray, inp: ClusteringInputs, cfg) -> dict:
    eps = cfg.eps_norm
    k = inp.k
    groups_c = class_groups(inp.cluster_class_idx, k)
    tc = centroids_by_class(out_c, groups_c, eps)
    temp = tc["centroids"]
    dmat = cosine_distance_matrix(out_v, temp, eps)
    pred = argmin_class(dmat)
    yv = inp.y_val_idx
    rows = np.arange(yv.size)
    d_true = dmat[rows, yv]
    masked = dmat.copy()
    masked[rows, yv] = np.inf
    d_other = masked.min(axis=1)
    groups_v = class_groups(yv, k)
    vc = centroids_by_class(out_v, groups_v, eps)
    msv, _, _ = min_separation(vc["centroids"], eps)
    ms, ci, cj = min_separation(temp, eps)
    bound = simplex_bound(k)
    yfrac, _ = yodd_norm_fraction(out_c, inp.yodd_idx, eps)
    return {
        "pseudo_accuracy_val": float(np.mean(pred == yv)),
        "avg_margin_val": float(np.mean(d_other - d_true)),
        "mean_true_distance_val": float(np.mean(d_true)),
        "min_separation_val": float(msv),
        "min_separation_val_ratio": float(msv / bound),
        "min_separation": float(ms),
        "min_separation_ratio": float(ms / bound),
        "closest_class_i": int(cfg.classes[ci]),
        "closest_class_j": int(cfg.classes[cj]),
        "correlation_consistency": correlation_consistency(inp.s_matrix, temp, eps),
        "active_dimensions": active_dimensions_centroid(temp, cfg.active_dim_threshold),
        "yodd_norm_fraction": yfrac,
        "n_degenerate_cluster_outputs": int(np.sum(tc["n_excluded"])),
        "n_degenerate_val_outputs": int(np.sum(is_degenerate(out_v, eps))),
        "n_degenerate_centroids_train": int(np.sum(tc["degenerate"])),
        "n_degenerate_centroids_val": int(np.sum(vc["degenerate"])),
        "min_centroid_prenorm_train": float(np.min(tc["prenorm"])),
    }


def clustering_loop(engine: BatchedStatevector, initial_params: np.ndarray, inp: ClusteringInputs, cfg, run_dir):
    """Run COBYLA on the clustering objective; returns (result.x, optimizer summary)."""

    def core(theta: np.ndarray):
        u = engine.unitary_columns(theta)
        out_c = engine.expectations(inp.phi_cluster, u)        # G3-05a: once per unique sample
        loss, _ = pair_loss(out_c, inp.pair_a, inp.pair_b, inp.pair_s, cfg.eps_norm)
        out_v = engine.expectations(inp.phi_val, u)            # passive validation monitoring
        return loss, clustering_eval_metrics(out_c, out_v, inp, cfg)

    recorder = CobylaRecorder(
        "clustering", run_dir, engine.model.n_trainable, cfg.max_nfev_clustering,
        cfg.cobyla_tol, cfg.cobyla_rhobeg, cfg.run_mode, core,
        extra_summary={"device_name": cfg.device_name, "sim_dtype": cfg.sim_dtype},
    )
    return recorder.run(initial_params)
