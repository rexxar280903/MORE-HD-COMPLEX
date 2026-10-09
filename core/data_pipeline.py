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
from . import constants as C
from .constants import HU_PADDING_VALUE, HU_SIGNED_LOG_EPSILON, PCA_N_COMPONENTS, ZERNIKE_TERMS
from .datasets import load_official_test, sample_ids
from .features import MinMaxAngleScaler, extract_features, fit_pca


@dataclass
class DataBundle:
    x_train: np.ndarray
    y_train: np.ndarray
    train_source_index: np.ndarray
    train_sample_id: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    val_source_index: np.ndarray
    val_sample_id: np.ndarray
    test_hashes: dict
    input_hashes: dict
    feature_metadata: dict

    # Compatibility aliases for the frozen MNIST study.  New artifacts and all
    # MADBase code use source_index/sample_id.
    @property
    def train_mnist_index(self) -> np.ndarray:
        return self.train_source_index

    @property
    def val_mnist_index(self) -> np.ndarray:
        return self.val_source_index


def _take(manifest: dict, key: str, classes, n_per_class: int) -> np.ndarray:
    out = []
    for c in classes:
        lst = manifest[key][str(c)]
        if n_per_class > len(lst):
            raise ValueError(f"requested {n_per_class} > manifest size {len(lst)}")
        out.extend(lst[:n_per_class])            # prefix of the seeded order: nested and paired
    return np.asarray(out, dtype=np.int64)


def _scale_function(feature_method: str, scaler: MinMaxAngleScaler):
    if feature_method == "HU":
        def scale(f):
            return np.concatenate([scaler.transform(f[:, :7]), np.full((f.shape[0], 1), 0.0)], axis=1)
        return scale
    return scaler.transform


def _save_identity_arrays(art: Path, prefix: str, source_index: np.ndarray, sample_id: np.ndarray) -> dict:
    arrays = {
        f"{prefix}_source_index": np.asarray(source_index, dtype=np.int64),
        f"{prefix}_sample_id": np.asarray(sample_id).astype(str),
    }
    for name, arr in arrays.items():
        save_npy_atomic(arr, art / f"{name}.npy")
    return arrays


def run_data_pipeline(cfg, dataset, split_manifest: dict, run_dir) -> DataBundle:
    run_dir = Path(run_dir)
    art = run_dir / "artifacts"
    classes = cfg.classes
    tr_idx = _take(split_manifest, "train_idx_by_class", classes, cfg.n_train_per_class)
    va_idx = _take(split_manifest, "val_idx_by_class", classes, cfg.n_val_per_class)
    if set(tr_idx.tolist()) & set(va_idx.tolist()):
        raise AssertionError("train/validation overlap")

    x_tr_img = dataset.train_images[tr_idx].astype(np.float64) / 255.0
    x_va_img = dataset.train_images[va_idx].astype(np.float64) / 255.0
    y_tr = dataset.train_labels[tr_idx].astype(np.int64)
    y_va = dataset.train_labels[va_idx].astype(np.int64)
    train_ids = sample_ids(dataset, "train")
    tr_sid, va_sid = train_ids[tr_idx], train_ids[va_idx]

    counters: dict = {}
    pca = None
    if cfg.feature_method == "PCA":
        pca = fit_pca(x_tr_img.reshape(x_tr_img.shape[0], -1), cfg.seed)
        joblib.dump(pca, art / "pca_model.joblib")
        save_npy_atomic(pca.components_, art / "pca_components.npy")
    f_tr, c_tr = extract_features(cfg.feature_method, x_tr_img, pca)
    f_va, c_va = extract_features(cfg.feature_method, x_va_img, pca)
    for name, cnt in (("train", c_tr), ("val", c_va)):
        for key, value in cnt.items():
            counters[f"{key}_{name}"] = value

    if cfg.feature_method == "HU":
        scaler = MinMaxAngleScaler().fit(f_tr[:, :7])             # padding channel excluded

    else:
        scaler = MinMaxAngleScaler().fit(f_tr)
    scale = _scale_function(cfg.feature_method, scaler)
    x_tr, x_va = scale(f_tr), scale(f_va)
    for arr in (x_tr, x_va):
        if arr.shape[1] != 8 or not np.all(np.isfinite(arr)):
            raise AssertionError("scaled features must be finite with 8 channels")

    save_json_atomic(scaler.to_dict(), art / "scaler_params.json")
    joblib.dump(scaler, art / "scaler_params.joblib")
    arrays = {
        "X_train_scaled": x_tr, "y_train": y_tr,
        "X_val_scaled": x_va, "y_val": y_va,
    }
    arrays.update(_save_identity_arrays(art, "train", tr_idx, tr_sid))
    arrays.update(_save_identity_arrays(art, "val", va_idx, va_sid))
    for name, arr in arrays.items():
        save_npy_atomic(arr, art / f"{name}.npy")

    # Keep legacy aliases only for MNIST so old analysis remains byte-for-byte
    # readable while the new schema stays dataset-neutral.
    if cfg.dataset == C.DATASET_MNIST:
        for legacy, arr in (("train_mnist_index", tr_idx), ("val_mnist_index", va_idx)):
            save_npy_atomic(arr, art / f"{legacy}.npy")
            arrays[legacy] = arr
    hashes = {name: sha256_array(arr) for name, arr in arrays.items()}
    raw_dim = {"PCA": PCA_N_COMPONENTS, "HU": 7, "ZERNIKE": len(ZERNIKE_TERMS)}[cfg.feature_method]
    meta = {
        "dataset": cfg.dataset,
        "dataset_id": split_manifest.get("dataset_id", C.DATASET_IDS[cfg.dataset]),
        "dataset_snapshot_sha256": split_manifest.get("dataset_snapshot_sha256"),
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
        "n_test_outside_range": None,
        "test_preprocessing_stage": "final_evaluation",
        "counters": counters,
        "array_sha256": hashes,
    }
    save_json_atomic(meta, art / "feature_metadata.json")
    input_hashes = {k: v for k, v in hashes.items()}
    bundle = DataBundle(x_tr, y_tr, tr_idx, tr_sid, x_va, y_va, va_idx, va_sid, {}, input_hashes, meta)

    # Preserve the legacy MNIST data-pipeline contract.  MADBase deliberately
    # postpones all held-out preprocessing until FINAL_EVALUATION.
    if cfg.dataset == C.DATASET_MNIST:
        prepare_test_artifacts(cfg, dataset, split_manifest, run_dir, bundle)
    return bundle


def prepare_test_artifacts(cfg, dataset, split_manifest: dict, run_dir, bundle: DataBundle) -> dict:
    """Create held-out features after model selection, immediately before evaluation."""
    run_dir = Path(run_dir)
    art = run_dir / "artifacts"
    te_idx = _take(split_manifest, "test_idx_by_class", cfg.classes, cfg.n_test_per_class)
    source = split_manifest.get("test_index_space", "OFFICIAL_TEST")
    if source == "TRAIN_POOL":
        images, labels, ids = dataset.train_images, dataset.train_labels, sample_ids(dataset, "train")
    elif source in ("OFFICIAL_TEST", "original MNIST test order (torchvision test=10000)"):
        test_data = load_official_test(dataset)
        images, labels, ids = test_data.test_images, test_data.test_labels, sample_ids(test_data, "test")
    else:
        raise ValueError(f"test_index_space tidak dikenal: {source}")
    x_img = images[te_idx].astype(np.float64) / 255.0
    y_test = labels[te_idx].astype(np.int64)
    sid = ids[te_idx]
    pca = joblib.load(art / "pca_model.joblib") if cfg.feature_method == "PCA" else None
    scaler = joblib.load(art / "scaler_params.joblib")
    f_test, counters = extract_features(cfg.feature_method, x_img, pca)
    x_test = _scale_function(cfg.feature_method, scaler)(f_test)
    if x_test.shape[1] != 8 or not np.all(np.isfinite(x_test)):
        raise AssertionError("test features must be finite with 8 channels")

    arrays = {"X_test_scaled": x_test, "y_test": y_test}
    arrays.update(_save_identity_arrays(art, "test", te_idx, sid))
    for name, arr in arrays.items():
        save_npy_atomic(arr, art / f"{name}.npy")
    hashes = {name: sha256_array(arr) for name, arr in arrays.items()}
    test_hashes = {k: hashes[k] for k in ("X_test_scaled", "y_test", "test_source_index", "test_sample_id")}
    if cfg.dataset == C.DATASET_MNIST:
        save_npy_atomic(te_idx, art / "test_mnist_index.npy")
        hashes["test_mnist_index"] = sha256_array(te_idx)
        # Existing MNIST tests and completed-study tooling use these three keys.
        test_hashes = {k: hashes[k] for k in ("X_test_scaled", "y_test", "test_mnist_index")}

    meta = dict(bundle.feature_metadata)
    meta["n_test_outside_range"] = int(np.sum((x_test < -1e-12) | (x_test > np.pi + 1e-12)))
    for key, value in counters.items():
        meta.setdefault("counters", {})[f"{key}_test"] = value
    meta["array_sha256"] = {**meta["array_sha256"], **hashes}
    meta["test_origin"] = split_manifest.get("test_origin", "OFFICIAL_SOURCE")
    meta["test_index_space"] = source
    save_json_atomic(meta, art / "feature_metadata.json")
    bundle.feature_metadata = meta
    bundle.input_hashes.update(hashes)
    bundle.test_hashes = test_hashes
    return test_hashes
