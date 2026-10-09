"""Dataset registry and provenance helpers.

The training pipeline accepts an explicit dataset key and rejects injected data
from a different dataset.  This prevents an Arabic-digit run from silently
falling back to MNIST (or vice versa).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from . import constants as C
from .madbase import DATASET_ID as MADBASE_DATASET_ID, load_madbase, load_madbase_training
from .mnist import load_mnist


def normalise_dataset(value: str) -> str:
    aliases = {
        "MNIST": C.DATASET_MNIST,
        "MADBASE": C.DATASET_MADBASE,
        "MADBASE-AUC-V1": C.DATASET_MADBASE,
        "ARABIC": C.DATASET_MADBASE,
        "ARABIC_DIGITS": C.DATASET_MADBASE,
    }
    key = str(value).strip().upper().replace("-", "_")
    if key == "MADBASE_AUC_V1":
        key = "MADBASE-AUC-V1"
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError(f"dataset tidak dikenali: {value!r}; pilih {C.DATASETS}") from exc


def dataset_id(data) -> str:
    return str(getattr(data, "dataset_id", C.DATASET_IDS[C.DATASET_MNIST]))


def dataset_key(data) -> str:
    return C.DATASET_MADBASE if dataset_id(data) == MADBASE_DATASET_ID else C.DATASET_MNIST


def snapshot_sha256(data) -> str:
    value = getattr(data, "snapshot_sha256", None)
    if value:
        return str(value)
    h = hashlib.sha256()
    for name, digest in sorted(getattr(data, "file_sha256", {}).items()):
        h.update(str(name).encode("utf-8"))
        h.update(str(digest).encode("ascii"))
    return h.hexdigest()


def sample_ids(data, split: str) -> np.ndarray:
    attr = f"{split}_sample_ids"
    if hasattr(data, attr):
        return np.asarray(getattr(data, attr)).astype(str)
    labels = np.asarray(getattr(data, f"{split}_labels"))
    return np.asarray([f"{split}/{i:05d}" for i in range(labels.size)])


def writer_ids(data, split: str) -> np.ndarray | None:
    attr = f"{split}_writer_ids"
    return np.asarray(getattr(data, attr), dtype=np.int64) if hasattr(data, attr) else None


def load_dataset(project_root: str | Path, key: str, *, mnist_download: bool = False):
    root = Path(project_root)
    key = normalise_dataset(key)
    if key == C.DATASET_MNIST:
        return load_mnist(root / C.MNIST_ROOT, download=mnist_download)
    if mnist_download:
        raise ValueError("--mnist-download tidak boleh dipakai pada MADBase")
    return load_madbase_training(root / C.MADBASE_ROOT)


def load_official_test(data):
    """Return an object with official-test arrays, loading it only when requested."""
    if hasattr(data, "test_images") and hasattr(data, "test_labels"):
        return data
    root = getattr(data, "root", None)
    if root is None:
        raise ValueError("dataset object cannot lazily load its official test split")
    return load_madbase(root)


def assert_dataset_matches(key: str, data) -> None:
    expected = normalise_dataset(key)
    actual = dataset_key(data)
    if actual != expected:
        raise ValueError(f"dataset mismatch: config={expected}, data={actual} ({dataset_id(data)})")
