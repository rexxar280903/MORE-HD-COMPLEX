"""QUANTUM_LABEL_EXTRACTION (pseudocode §6; G2-07)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .artifact_io import save_json_atomic
from .clustering import class_groups
from .numerics import centroids_by_class


def extract_quantum_labels(engine, theta_clustering, phi_cluster, cluster_class_idx, cfg, run_dir) -> dict:
    """Labels = CLASS_CENTROID_MORE over the 5 clustering samples per class (pair_manifest)."""
    out = engine.expectations(phi_cluster, engine.unitary_columns(theta_clustering))
    res = centroids_by_class(out, class_groups(cluster_class_idx, len(cfg.classes)), cfg.eps_norm)
    labels = res["centroids"]
    obs = list(engine.model.observable_labels)
    per_class = {}
    label_json = {}
    for i, c in enumerate(cfg.classes):
        label_json[str(c)] = {o: float(labels[i, j]) for j, o in enumerate(obs)}
        per_class[str(c)] = {
            "n_samples": int(np.sum(cluster_class_idx == i)),
            "n_degenerate_samples_excluded": int(res["n_excluded"][i]),
            "prenorm": float(res["prenorm"][i]),
            "degenerate": bool(res["degenerate"][i]),
        }
    run_dir = Path(run_dir)
    save_json_atomic({"observables": obs, "labels": label_json}, run_dir / "artifacts" / "quantum_labels.json")
    diag = {
        "centroid_rule": cfg.centroid_rule,
        "centroid_source": cfg.centroid_source,
        "eps_norm": cfg.eps_norm,
        "per_class": per_class,
        "degenerate_quantum_label": any(d["degenerate"] for d in per_class.values()),
    }
    save_json_atomic(diag, run_dir / "artifacts" / "quantum_label_diagnostics.json")
    return {"labels": labels, "diagnostics": diag}


def load_quantum_labels(path, classes) -> np.ndarray:
    from .artifact_io import load_json

    data = load_json(path)
    obs = data["observables"]
    return np.array([[data["labels"][str(c)][o] for o in obs] for c in classes], dtype=np.float64)
