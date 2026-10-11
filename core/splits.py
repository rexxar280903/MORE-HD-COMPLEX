"""Paired, nested split manifests (pseudocode §0.3; G0-01, G0-03, G0-04, G4-02)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import constants as C
from .artifact_io import load_json, save_json_atomic, sha256_file
from .constants import N_TEST_PER_CLASS, N_TRAIN_PER_CLASS, N_VAL_PER_CLASS
from .datasets import dataset_id, sample_ids, snapshot_sha256, writer_ids
from .seeds import derive_subseed

SPLIT_RULE = (
    "per digit: train pool shuffled with RNG(DERIVE_SUBSEED(seed, 'data:train:<digit>')); "
    "train = first 1000, validation = next 100 (disjoint, same train pool); official test = "
    "first 200 of the test pool shuffled with RNG(DERIVE_SUBSEED(seed, 'data:test:<digit>'))"
)
NESTED_K_RULE = "reuse same class-specific indices for every K containing that class"


def split_manifest_path(seed: int, splits_dir: str | os.PathLike = "splits") -> Path:
    return Path(splits_dir) / f"seed{int(seed)}.json"


def build_split_manifest(seed: int, train_labels: np.ndarray, test_labels: np.ndarray) -> dict:
    """CREATE_OR_LOAD_SPLIT_MANIFEST body: per-digit seeded permutation of MNIST indices."""
    train_idx, val_idx, test_idx = {}, {}, {}
    for digit in range(10):
        train_pool = np.flatnonzero(train_labels == digit)
        test_pool = np.flatnonzero(test_labels == digit)
        rng_train = np.random.default_rng(derive_subseed(seed, f"data:train:{digit}"))
        rng_test = np.random.default_rng(derive_subseed(seed, f"data:test:{digit}"))
        shuffled_train = rng_train.permutation(train_pool)
        shuffled_test = rng_test.permutation(test_pool)
        need = N_TRAIN_PER_CLASS + N_VAL_PER_CLASS
        if shuffled_train.size < need or shuffled_test.size < N_TEST_PER_CLASS:
            raise ValueError(f"digit {digit}: MNIST pool too small for the locked protocol")
        tr = shuffled_train[:N_TRAIN_PER_CLASS]
        va = shuffled_train[N_TRAIN_PER_CLASS:need]
        te = shuffled_test[:N_TEST_PER_CLASS]
        assert not set(tr.tolist()) & set(va.tolist())
        train_idx[str(digit)] = [int(i) for i in tr]
        val_idx[str(digit)] = [int(i) for i in va]
        test_idx[str(digit)] = [int(i) for i in te]
    return {
        "master_seed": int(seed),
        "n_train_per_class": N_TRAIN_PER_CLASS,
        "n_val_per_class": N_VAL_PER_CLASS,
        "n_test_per_class": N_TEST_PER_CLASS,
        "index_space": "original MNIST order (torchvision train=60000, test=10000)",
        "split_rule": SPLIT_RULE,
        "train_idx_by_class": train_idx,
        "val_idx_by_class": val_idx,
        "test_idx_by_class": test_idx,
        "nested_k_rule": NESTED_K_RULE,
    }


def validate_split_manifest(manifest: dict, seed: int, train_labels: np.ndarray, test_labels: np.ndarray) -> None:
    """VALIDATE_SPLIT_MANIFEST: re-derivable, disjoint, correct sizes and labels."""
    if int(manifest["master_seed"]) != int(seed):
        raise ValueError("split manifest seed mismatch")
    rebuilt = build_split_manifest(seed, train_labels, test_labels)
    for key in ("train_idx_by_class", "val_idx_by_class", "test_idx_by_class"):
        if manifest[key] != rebuilt[key]:
            raise ValueError(f"split manifest {key} is not reproducible from seed {seed}")
    all_train = [i for d in range(10) for i in manifest["train_idx_by_class"][str(d)]]
    all_val = [i for d in range(10) for i in manifest["val_idx_by_class"][str(d)]]
    if set(all_train) & set(all_val):
        raise ValueError("train/validation overlap")
    for d in range(10):
        if not np.all(train_labels[manifest["train_idx_by_class"][str(d)]] == d):
            raise ValueError("train label mismatch")
        if not np.all(train_labels[manifest["val_idx_by_class"][str(d)]] == d):
            raise ValueError("validation label mismatch")
        if not np.all(test_labels[manifest["test_idx_by_class"][str(d)]] == d):
            raise ValueError("test label mismatch")


def create_or_load_split_manifest(
    seed: int, train_labels: np.ndarray, test_labels: np.ndarray, splits_dir: str | os.PathLike = "splits"
) -> tuple[dict, Path]:
    path = split_manifest_path(seed, splits_dir)
    if path.exists():
        manifest = load_json(path)
        validate_split_manifest(manifest, seed, train_labels, test_labels)
        return manifest, path
    manifest = build_split_manifest(seed, train_labels, test_labels)
    validate_split_manifest(manifest, seed, train_labels, test_labels)
    save_json_atomic(manifest, path)
    return manifest, path


def split_manifest_hash(path: str | os.PathLike) -> str:
    return sha256_file(path)


def indices_for_classes(manifest: dict, classes, which: str) -> np.ndarray:
    """Concatenate class-specific indices in class order (nested-K by construction)."""
    key = {"train": "train_idx_by_class", "val": "val_idx_by_class", "test": "test_idx_by_class"}[which]
    return np.asarray([i for c in classes for i in manifest[key][str(c)]], dtype=np.int64)


def madbase_split_manifest_path(seed: int, run_mode: str, splits_dir: str | os.PathLike) -> Path:
    mode = str(run_mode).lower()
    return Path(splits_dir) / f"{mode}_seed{int(seed)}.json"


def _indices_for_writers(labels, writers, selected_writers, digit: int, seed: int, namespace: str) -> np.ndarray:
    mask = (np.asarray(labels) == digit) & np.isin(np.asarray(writers), np.asarray(selected_writers))
    pool = np.flatnonzero(mask)
    rng = np.random.default_rng(derive_subseed(seed, f"madbase:{namespace}:{digit}"))
    return rng.permutation(pool).astype(np.int64)


def build_madbase_split_manifest(seed: int, data, run_mode: str) -> dict:
    """Build a writer-disjoint MADBase split with stable source sample IDs.

    PILOT never consumes the official test set: its held-out set is a third
    writer-disjoint slice of the official training pool.  CONFIRMATORY uses one
    fixed official-test writer subset shared by all experiment seeds.
    """
    if dataset_id(data) != C.DATASET_IDS[C.DATASET_MADBASE]:
        raise ValueError("build_madbase_split_manifest requires MADBASE-AUC-V1")
    if run_mode not in C.RUN_MODES:
        raise ValueError(f"run_mode tidak valid: {run_mode}")
    train_writers = writer_ids(data, "train")
    test_writers = writer_ids(data, "test")
    if train_writers is None:
        raise ValueError("MADBase train writer IDs are required for group-aware splits")

    unique_train = np.unique(train_writers)
    rng = np.random.default_rng(derive_subseed(seed, "madbase:writer_split"))
    shuffled = rng.permutation(unique_train)
    # Each MADBase writer contributes exactly 10 samples per digit.
    n_train_writers = N_TRAIN_PER_CLASS // 10
    n_val_writers = N_VAL_PER_CLASS // 10
    n_holdout_writers = N_TEST_PER_CLASS // 10
    tr_w = np.sort(shuffled[:n_train_writers])
    va_w = np.sort(shuffled[n_train_writers:n_train_writers + n_val_writers])
    pilot_w = np.sort(shuffled[n_train_writers + n_val_writers:
                               n_train_writers + n_val_writers + n_holdout_writers])
    if set(tr_w) & set(va_w) or set(tr_w) & set(pilot_w) or set(va_w) & set(pilot_w):
        raise AssertionError("MADBase writer groups overlap")

    if run_mode == "CONFIRMATORY":
        if test_writers is None or not hasattr(data, "test_labels"):
            raise ValueError("full MADBase data is required to create a confirmatory split manifest")
        unique_test = np.unique(test_writers)
        final_rng = np.random.default_rng(
            derive_subseed(C.MADBASE_FINAL_TEST_WRITER_SEED, "madbase:final_test_writers")
        )
        final_w = np.sort(final_rng.permutation(unique_test)[:n_holdout_writers])
        test_pool_labels = data.test_labels
        test_pool_writers = test_writers
        test_pool_ids = sample_ids(data, "test")
        test_source = "OFFICIAL_TEST"
        test_origin = "OFFICIAL_SOURCE_FIXED_WRITER_SUBSET"
        test_seed = C.MADBASE_FINAL_TEST_WRITER_SEED
    else:
        final_w = pilot_w
        test_pool_labels = data.train_labels
        test_pool_writers = train_writers
        test_pool_ids = sample_ids(data, "train")
        test_source = "TRAIN_POOL"
        test_origin = "CONSTRUCTED_PILOT_TRAIN_POOL"
        test_seed = seed

    train_ids = sample_ids(data, "train")
    train_idx, val_idx, test_idx = {}, {}, {}
    train_sid, val_sid, test_sid = {}, {}, {}
    for digit in range(10):
        tr = _indices_for_writers(data.train_labels, train_writers, tr_w, digit, seed, "train")
        va = _indices_for_writers(data.train_labels, train_writers, va_w, digit, seed, "validation")
        te = _indices_for_writers(test_pool_labels, test_pool_writers, final_w, digit, test_seed, "test")
        if tr.size != N_TRAIN_PER_CLASS or va.size != N_VAL_PER_CLASS or te.size != N_TEST_PER_CLASS:
            raise ValueError(
                f"digit {digit}: unexpected writer-group sizes train={tr.size}, val={va.size}, test={te.size}"
            )
        train_idx[str(digit)] = tr.tolist()
        val_idx[str(digit)] = va.tolist()
        test_idx[str(digit)] = te.tolist()
        train_sid[str(digit)] = train_ids[tr].tolist()
        val_sid[str(digit)] = train_ids[va].tolist()
        test_sid[str(digit)] = test_pool_ids[te].tolist()

    return {
        "schema_version": "2.0",
        "dataset_key": C.DATASET_MADBASE,
        "dataset_id": dataset_id(data),
        "dataset_snapshot_sha256": snapshot_sha256(data),
        "master_seed": int(seed),
        "run_mode": run_mode,
        "classes": list(range(10)),
        "n_train_per_class": N_TRAIN_PER_CLASS,
        "n_val_per_class": N_VAL_PER_CLASS,
        "n_test_per_class": N_TEST_PER_CLASS,
        "train_idx_by_class": train_idx,
        "val_idx_by_class": val_idx,
        "test_idx_by_class": test_idx,
        "train_sample_ids_by_class": train_sid,
        "val_sample_ids_by_class": val_sid,
        "test_sample_ids_by_class": test_sid,
        "train_writer_ids": tr_w.tolist(),
        "validation_writer_ids": va_w.tolist(),
        "test_writer_ids": final_w.tolist(),
        "test_index_space": test_source,
        "test_origin": test_origin,
        "official_test_access_during_pilot": False,
        "final_test_writer_seed": C.MADBASE_FINAL_TEST_WRITER_SEED if run_mode == "CONFIRMATORY" else None,
        "split_rule": (
            "writer-group split; 100 train writers, 10 validation writers, and 20 held-out writers; "
            "indices shuffled per class; train/validation/pilot-holdout are disjoint"
        ),
        "nested_k_rule": NESTED_K_RULE,
        "duplicate_policy": (
            "retain the official snapshot unchanged; exact and near-duplicate flags are reported, never silently removed"
        ),
    }


def validate_madbase_split_manifest(manifest: dict, seed: int, data, run_mode: str) -> None:
    rebuilt = build_madbase_split_manifest(seed, data, run_mode)
    keys = (
        "schema_version", "dataset_id", "dataset_snapshot_sha256", "master_seed", "run_mode",
        "train_idx_by_class", "val_idx_by_class", "test_idx_by_class",
        "train_sample_ids_by_class", "val_sample_ids_by_class", "test_sample_ids_by_class",
        "train_writer_ids", "validation_writer_ids", "test_writer_ids", "test_index_space", "test_origin",
    )
    for key in keys:
        if manifest.get(key) != rebuilt.get(key):
            raise ValueError(f"MADBase split manifest mismatch pada {key}")
    if set(manifest["train_writer_ids"]) & set(manifest["validation_writer_ids"]):
        raise ValueError("MADBase train/validation writer leakage")
    if run_mode == "PILOT" and set(manifest["test_writer_ids"]) & (
        set(manifest["train_writer_ids"]) | set(manifest["validation_writer_ids"])
    ):
        raise ValueError("MADBase pilot holdout writer leakage")


def validate_madbase_runtime_manifest(manifest: dict, seed: int, data, run_mode: str) -> None:
    """Validate a frozen manifest without loading official-test arrays."""
    if manifest.get("dataset_id") != dataset_id(data):
        raise ValueError("MADBase dataset ID mismatch")
    if manifest.get("dataset_snapshot_sha256") != snapshot_sha256(data):
        raise ValueError("MADBase snapshot hash mismatch")
    if int(manifest.get("master_seed", -1)) != int(seed) or manifest.get("run_mode") != run_mode:
        raise ValueError("MADBase split seed/run_mode mismatch")
    train_w = writer_ids(data, "train")
    ids = sample_ids(data, "train")
    tr_group = np.asarray(manifest["train_writer_ids"], dtype=np.int64)
    va_group = np.asarray(manifest["validation_writer_ids"], dtype=np.int64)
    group_rng = np.random.default_rng(derive_subseed(seed, "madbase:writer_split"))
    shuffled = group_rng.permutation(np.unique(train_w))
    expected_tr_group = np.sort(shuffled[:N_TRAIN_PER_CLASS // 10])
    expected_va_group = np.sort(shuffled[N_TRAIN_PER_CLASS // 10:
                                        N_TRAIN_PER_CLASS // 10 + N_VAL_PER_CLASS // 10])
    if not np.array_equal(tr_group, expected_tr_group) or not np.array_equal(va_group, expected_va_group):
        raise ValueError("MADBase train/validation writer selection is not reproducible")
    if set(tr_group.tolist()) & set(va_group.tolist()):
        raise ValueError("MADBase train/validation writer leakage")
    for digit in range(10):
        tr = _indices_for_writers(data.train_labels, train_w, tr_group, digit, seed, "train")
        va = _indices_for_writers(data.train_labels, train_w, va_group, digit, seed, "validation")
        key = str(digit)
        if manifest["train_idx_by_class"][key] != tr.tolist():
            raise ValueError(f"MADBase train split mismatch for digit {digit}")
        if manifest["val_idx_by_class"][key] != va.tolist():
            raise ValueError(f"MADBase validation split mismatch for digit {digit}")
        if manifest["train_sample_ids_by_class"][key] != ids[tr].tolist():
            raise ValueError(f"MADBase train sample IDs mismatch for digit {digit}")
        if manifest["val_sample_ids_by_class"][key] != ids[va].tolist():
            raise ValueError(f"MADBase validation sample IDs mismatch for digit {digit}")
        test_ids = manifest["test_sample_ids_by_class"][key]
        if len(test_ids) != N_TEST_PER_CLASS or len(set(test_ids)) != N_TEST_PER_CLASS:
            raise ValueError(f"MADBase test sample IDs invalid for digit {digit}")
    if run_mode == "PILOT":
        expected = build_madbase_split_manifest(seed, data, run_mode)
        for key in ("test_idx_by_class", "test_sample_ids_by_class", "test_writer_ids", "test_origin"):
            if manifest.get(key) != expected.get(key):
                raise ValueError(f"MADBase pilot manifest mismatch pada {key}")
    else:
        if manifest.get("test_index_space") != "OFFICIAL_TEST":
            raise ValueError("confirmatory MADBase test must use OFFICIAL_TEST")
        if manifest.get("final_test_writer_seed") != C.MADBASE_FINAL_TEST_WRITER_SEED:
            raise ValueError("MADBase final-test writer seed mismatch")
        if any(not str(sid).startswith("test/writer")
               for rows in manifest["test_sample_ids_by_class"].values() for sid in rows):
            raise ValueError("confirmatory MADBase test IDs must come from official test")


def create_or_load_dataset_split_manifest(cfg, data, project_root: str | os.PathLike) -> tuple[dict, Path]:
    """Dataset-aware split entry point used by the training pipeline."""
    root = Path(project_root)
    if cfg.dataset == C.DATASET_MNIST:
        return create_or_load_split_manifest(
            cfg.seed, data.train_labels, data.test_labels, root / C.SPLITS_DIR
        )
    split_dir = root / C.MADBASE_SPLITS_DIR
    path = madbase_split_manifest_path(cfg.seed, cfg.run_mode, split_dir)
    if path.exists():
        manifest = load_json(path)
    else:
        if cfg.run_mode == "CONFIRMATORY":
            raise FileNotFoundError(
                f"frozen confirmatory split manifest missing: {path}; run scripts/build_madbase_splits.py first"
            )
        manifest = build_madbase_split_manifest(cfg.seed, data, cfg.run_mode)
        save_json_atomic(manifest, path)
    validate_madbase_runtime_manifest(manifest, cfg.seed, data, cfg.run_mode)
    return manifest, path
