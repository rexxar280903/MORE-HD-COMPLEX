"""Paired, nested split manifests (pseudocode §0.3; G0-01, G0-03, G0-04, G4-02)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .artifact_io import load_json, save_json_atomic, sha256_file
from .constants import N_TEST_PER_CLASS, N_TRAIN_PER_CLASS, N_VAL_PER_CLASS
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
