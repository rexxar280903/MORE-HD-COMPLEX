"""MNIST loading identical to ``torchvision.datasets.MNIST`` without a torch dependency.

The four IDX files are the ones torchvision downloads (same mirror, same MD5
checksums) and are stored in torchvision's on-disk layout ``<root>/MNIST/raw``.
Sample order is therefore the original MNIST order used by torchvision, so the
indices stored in split manifests are the canonical MNIST indices (G4-02).
"""

from __future__ import annotations

import gzip
import hashlib
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MIRRORS = (
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
)
# file name -> MD5 published in torchvision.datasets.MNIST.resources
RESOURCES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


@dataclass(frozen=True)
class MNIST:
    train_images: np.ndarray  # (60000, 28, 28) uint8
    train_labels: np.ndarray  # (60000,) int64
    test_images: np.ndarray   # (10000, 28, 28) uint8
    test_labels: np.ndarray   # (10000,) int64
    file_sha256: dict


def _raw_dir(root: str | os.PathLike) -> Path:
    return Path(root) / "MNIST" / "raw"


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def mnist_exists(root: str | os.PathLike) -> bool:
    raw = _raw_dir(root)
    return all((raw / name).exists() for name in RESOURCES)


def download_mnist(root: str | os.PathLike) -> None:
    raw = _raw_dir(root)
    raw.mkdir(parents=True, exist_ok=True)
    for name, md5 in RESOURCES.items():
        target = raw / name
        if target.exists() and _md5(target) == md5:
            continue
        last_error = None
        for mirror in MIRRORS:
            tmp = target.with_suffix(".part")
            try:
                urllib.request.urlretrieve(mirror + name, tmp)
                if _md5(tmp) != md5:
                    raise IOError(f"MD5 mismatch for {name} from {mirror}")
                os.replace(tmp, target)
                last_error = None
                break
            except Exception as exc:  # try next mirror
                last_error = exc
                if tmp.exists():
                    tmp.unlink()
        if last_error is not None:
            raise RuntimeError(f"could not download {name}: {last_error}")


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as fh:
        data = fh.read()
    magic = int.from_bytes(data[0:4], "big")
    ndim = magic & 0xFF
    dtype_code = (magic >> 8) & 0xFF
    if dtype_code != 0x08:
        raise ValueError(f"{path}: unexpected IDX dtype code {dtype_code:#x}")
    dims = [int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    offset = 4 + 4 * ndim
    arr = np.frombuffer(data, dtype=np.uint8, offset=offset)
    return arr.reshape(dims).copy()


def load_mnist(root: str | os.PathLike, download: bool = False) -> MNIST:
    """LOAD_MNIST_FROM_TORCHVISION(root, download) equivalent with MD5 verification."""
    if not mnist_exists(root):
        if not download:
            raise FileNotFoundError(
                f"MNIST lokal belum tersedia di {root}. Jalankan satu pilot run dengan "
                "--mnist-download terlebih dahulu."
            )
        download_mnist(root)
    raw = _raw_dir(root)
    sha = {}
    for name, md5 in RESOURCES.items():
        path = raw / name
        if _md5(path) != md5:
            raise IOError(f"{path}: MD5 mismatch; file is not the torchvision MNIST resource")
        sha[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    train_images = _read_idx(raw / "train-images-idx3-ubyte.gz")
    train_labels = _read_idx(raw / "train-labels-idx1-ubyte.gz").astype(np.int64)
    test_images = _read_idx(raw / "t10k-images-idx3-ubyte.gz")
    test_labels = _read_idx(raw / "t10k-labels-idx1-ubyte.gz").astype(np.int64)
    assert train_images.shape == (60000, 28, 28) and train_labels.shape == (60000,)
    assert test_images.shape == (10000, 28, 28) and test_labels.shape == (10000,)
    return MNIST(train_images, train_labels, test_images, test_labels, sha)
