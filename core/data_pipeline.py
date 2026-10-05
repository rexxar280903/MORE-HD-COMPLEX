"""DATA_PIPELINE (pseudocode §2; G0-01, G0-03, G2-01..G2-03, G4-02).

Returns TRAIN and VALIDATION arrays only. Official-test arrays are written to
disk (for FINAL_EVALUATION and the classical baselines) and represented by
their hashes; they are never returned to the training stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from .artifact_io import save_json_atomic, save_npy_atomic, sha256_array
from .constants import HU_PADDING_VALUE, HU_SIGNED_LOG_EPSILON, PCA_N_COMPONENTS, ZERNIKE_TERMS
from .features import MinMaxAngleScaler, extract_features, fit_pca


@dataclass
class DataBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    train_mnist_index: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    val_mnist_index: np.ndarray
    test_hashes: dict
    input_hashes: dict
    feature_metadata: dict


def _take(manifest: dict, key: str, classes, n_per_class: int) -> np.ndarray:
    out = []
    for c in classes:
        lst = manifest[key][str(c)]
        if n_per_class > len(lst):
            raise ValueError(f"requested {n_per_class} > manifest size {len(lst)}")
        out.extend(lst[:n_per_class])            # prefix of the seeded order: nested and paired
    return np.asarray(out, dtype=np.int64)


def run_data_pipeline(cfg, mnist, split_manifest: dict, run_dir) -> DataBundle:
    run_dir = Path(run_dir)
    art = run_dir / "artifacts"
    classes = cfg.classes
    tr_idx = _take(split_manifest, "train_idx_by_class", classes, cfg.n_train_per_class)
    va_idx = _take(split_manifest, "val_idx_by_class", classes, cfg.n_val_per_class)
    te_idx = _take(split_manifest, "test_idx_by_class", classes, cfg.n_test_per_class)
    if set(tr_idx.tolist()) & set(va_idx.tolist()):
        raise AssertionError("train/validation overlap")

    x_tr_img = mnist.train_images[tr_idx].astype(np.float64) / 255.0
    x_va_img = mnist.train_images[va_idx].astype(np.float64) / 255.0
    x_te_img = mnist.test_images[te_idx].astype(np.float64) / 255.0
    y_tr = mnist.train_labels[tr_idx].astype(np.int64)
    y_va = mnist.train_labels[va_idx].astype(np.int64)
    y_te = mnist.test_labels[te_idx].astype(np.int64)

    counters: dict = {}
    pca = None
    if cfg.feature_method == "PCA":
        pca = fit_pca(x_tr_img.reshape(x_tr_img.shape[0], -1), cfg.seed)
        joblib.dump(pca, art / "pca_model.joblib")
        save_npy_atomic(pca.components_, art / "pca_components.npy")
    f_tr, c_tr = extract_features(cfg.feature_method, x_tr_img, pca)
    f_va, c_va = extract_features(cfg.feature_method, x_va_img, pca)
    f_te, c_te = extract_features(cfg.feature_method, x_te_img, pca)
    for name, cnt in (("train", c_tr), ("val", c_va), ("test", c_te)):
        for key, value in cnt.items():
            counters[f"{key}_{name}"] = value

    if cfg.feature_method == "HU":
        scaler = MinMaxAngleScaler().fit(f_tr[:, :7])             # padding channel excluded

        def scale(f):
            return np.concatenate([scaler.transform(f[:, :7]), np.full((f.shape[0], 1), 0.0)], axis=1)
    else:
        scaler = MinMaxAngleScaler().fit(f_tr)

        def scale(f):
            return scaler.transform(f)

    x_tr, x_va, x_te = scale(f_tr), scale(f_va), scale(f_te)
    for arr in (x_tr, x_va, x_te):
        if arr.shape[1] != 8 or not np.all(np.isfinite(arr)):
            raise AssertionError("scaled features must be finite with 8 channels")

    save_json_atomic(scaler.to_dict(), art / "scaler_params.json")
    joblib.dump(scaler, art / "scaler_params.joblib")
    arrays = {
        "X_train_scaled": x_tr, "y_train": y_tr, "train_mnist_index": tr_idx,
        "X_val_scaled": x_va, "y_val": y_va, "val_mnist_index": va_idx,
        "X_test_scaled": x_te, "y_test": y_te, "test_mnist_index": te_idx,
    }
    for name, arr in arrays.items():
        save_npy_atomic(arr, art / f"{name}.npy")
    hashes = {name: sha256_array(arr) for name, arr in arrays.items()}
    raw_dim = {"PCA": PCA_N_COMPONENTS, "HU": 7, "ZERNIKE": len(ZERNIKE_TERMS)}[cfg.feature_method]
    meta = {
        "feature_method": cfg.feature_method,
        "raw_descriptor_dim": raw_dim,
        "quantum_input_dim": 8,
        "pixel_normalisation": "uint8 / 255.0 -> [0, 1] (float64)",
        "pca_svd_solver": cfg.pca_svd_solver if cfg.feature_method == "PCA" else None,
        "pca_random_state": cfg.seed if cfg.feature_method == "PCA" else None,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist() if pca is not None else None,
        "hu_input_mode": "grayscale" if cfg.feature_method == "HU" else None,
        "hu_padding_value": HU_PADDING_VALUE if cfg.feature_method == "HU" else None,
        "hu_signed_log_epsilon": HU_SIGNED_LOG_EPSILON if cfg.feature_method == "HU" else None,
        "n_non_finite_hu": sum(v for k, v in counters.items() if k.startswith("n_non_finite_hu")) if cfg.feature_method == "HU" else None,
        "zernike_terms": [list(t) for t in ZERNIKE_TERMS] if cfg.feature_method == "ZERNIKE" else None,
        "zernike_disk": "centre ((W-1)/2,(H-1)/2), radius min(H,W)/2, mass-normalised (mahotas convention)" if cfg.feature_method == "ZERNIKE" else None,
        "n_zero_mass_zernike": sum(v for k, v in counters.items() if k.startswith("n_zero_mass_zernike")) if cfg.feature_method == "ZERNIKE" else None,
        "scaler": "MinMax(0, pi) fit on train only" + (" (7 Hu columns; channel 8 fixed 0.0)" if cfg.feature_method == "HU" else ""),
        "clip_after_scaling": False,
        "n_val_outside_range": int(np.sum((x_va < -1e-12) | (x_va > np.pi + 1e-12))),
        "n_test_outside_range": int(np.sum((x_te < -1e-12) | (x_te > np.pi + 1e-12))),
        "counters": counters,
        "array_sha256": hashes,
    }
    save_json_atomic(meta, art / "feature_metadata.json")
    test_hashes = {k: hashes[k] for k in ("X_test_scaled", "y_test", "test_mnist_index")}
    input_hashes = {k: v for k, v in hashes.items()}
    return DataBundle(x_tr, y_tr, tr_idx, x_va, y_va, va_idx, test_hashes, input_hashes, meta)
