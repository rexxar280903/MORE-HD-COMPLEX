"""Validated loader for the official AUC MADBase handwritten digit archives.

MADBase uses the same 28x28 geometry and 60k/10k split sizes as MNIST, but
the downloaded BMP files have a white background and dark foreground.  The
loader therefore returns canonical uint8 images with black background and
bright foreground so the existing feature definitions receive the same
polarity convention as the MNIST study.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

DATASET_ID = "MADBASE-AUC-V1"
OFFICIAL_PAGE = "https://datacenter.aucegypt.edu/shazeem/"
TRAIN_ARCHIVE = "MAHDBase_TrainingSet.rar"
TEST_ARCHIVE = "MAHDBase_TestingSet.rar"
DOWNLOADS = {
    TRAIN_ARCHIVE: "https://datacenter.aucegypt.edu/shazeem/Files/MAHDBase_TrainingSet.rar",
    TEST_ARCHIVE: "https://datacenter.aucegypt.edu/shazeem/Files/MAHDBase_TestingSet.rar",
}
EXPECTED_SHA256 = {
    TRAIN_ARCHIVE: "ae2d5bcd658320eaf054de1ba46aac187a0470830d685258a0a2b0659c8d2eb8",
    TEST_ARCHIVE: "a8294a266f1843b4c659291dfda6e71620abbfcaec47e4313024c691eb494b1d",
}
TRAIN_DIR = "MAHDBase_TrainingSet"
TEST_DIR = "MAHDBase_TestingSet"
FILENAME_RE = re.compile(
    r"^writer(?P<writer>\d{3})_pass(?P<pass>\d{2})_digit(?P<label>\d)\.bmp$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MADBase:
    train_images: np.ndarray
    train_labels: np.ndarray
    train_sample_ids: np.ndarray
    train_writer_ids: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray
    test_sample_ids: np.ndarray
    test_writer_ids: np.ndarray
    file_sha256: dict[str, str]
    dataset_id: str = DATASET_ID
    test_origin: str = "OFFICIAL_SOURCE"
    raw_polarity: str = "white_background_dark_foreground"
    canonical_polarity: str = "black_background_bright_foreground"

    @property
    def snapshot_sha256(self) -> str:
        h = hashlib.sha256()
        for name, digest in sorted(self.file_sha256.items()):
            h.update(name.encode("utf-8"))
            h.update(digest.encode("ascii"))
        return h.hexdigest()


@dataclass(frozen=True)
class MADBaseTrainingView:
    """Training-only view used before FINAL_EVALUATION.

    It intentionally has no ``test_images`` or ``test_labels`` attributes, so
    accidental official-test access during preprocessing or optimisation fails
    immediately instead of silently leaking held-out data.
    """

    train_images: np.ndarray
    train_labels: np.ndarray
    train_sample_ids: np.ndarray
    train_writer_ids: np.ndarray
    file_sha256: dict[str, str]
    root: Path
    dataset_id: str = DATASET_ID
    test_origin: str = "OFFICIAL_SOURCE"
    raw_polarity: str = "white_background_dark_foreground"
    canonical_polarity: str = "black_background_bright_foreground"

    @property
    def snapshot_sha256(self) -> str:
        h = hashlib.sha256()
        for name, digest in sorted(self.file_sha256.items()):
            h.update(name.encode("utf-8"))
            h.update(digest.encode("ascii"))
        return h.hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _roots(root: str | Path) -> tuple[Path, Path, Path]:
    root = Path(root)
    extracted = root / "extracted" if (root / "extracted").is_dir() else root
    return root, extracted / TRAIN_DIR, extracted / TEST_DIR


def madbase_exists(root: str | Path) -> bool:
    _, train_dir, test_dir = _roots(root)
    return train_dir.is_dir() and test_dir.is_dir()


def _catalog(split_dir: Path, split: str, expected_count: int):
    records = []
    for path in split_dir.rglob("*.bmp"):
        match = FILENAME_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"MADBase filename tidak dikenali: {path}")
        writer = int(match.group("writer"))
        pass_id = int(match.group("pass"))
        label = int(match.group("label"))
        records.append((writer, pass_id, label, path))
    records.sort(key=lambda row: (row[0], row[1], row[2], str(row[3])))
    if len(records) != expected_count:
        raise ValueError(f"{split}: expected {expected_count} BMP, found {len(records)}")
    ids = [f"{split}/{path.stem}" for _, _, _, path in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{split}: duplicate source sample ID")
    return records, np.asarray(ids)


def _decode(records, split: str):
    images = np.empty((len(records), 28, 28), dtype=np.uint8)
    labels = np.empty(len(records), dtype=np.int64)
    writers = np.empty(len(records), dtype=np.int64)
    passes = np.empty(len(records), dtype=np.int64)
    def read_one(record):
        writer, pass_id, label, path = record
        with Image.open(path) as image:
            raw = np.asarray(image.convert("L"), dtype=np.uint8)
        if raw.shape != (28, 28):
            raise ValueError(f"{path}: expected 28x28, found {raw.shape}")
        return 255 - raw, label, writer, pass_id

    workers = min(16, max(2, os.cpu_count() or 2))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        decoded = executor.map(read_one, records)
        for index, (image, label, writer, pass_id) in enumerate(decoded):
            images[index] = image
            labels[index] = label
            writers[index] = writer
            passes[index] = pass_id

    expected_per_class = len(records) // 10
    counts = np.bincount(labels, minlength=10)
    if counts.tolist() != [expected_per_class] * 10:
        raise ValueError(f"{split}: unexpected class counts {counts.tolist()}")
    unique_writers, writer_counts = np.unique(writers, return_counts=True)
    if not np.all(writer_counts == 100):
        raise ValueError(f"{split}: every writer must contribute 100 images")
    if not np.all((passes >= 1) & (passes <= 10)):
        raise ValueError(f"{split}: pass IDs must be 01..10")
    return images, labels, writers, unique_writers


def load_madbase(root: str | Path) -> MADBase:
    """Load and validate the official MADBase train/test archives after extraction."""
    root, train_dir, test_dir = _roots(root)
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            f"MADBase belum diekstrak di {root}. Expected {train_dir} and {test_dir}."
        )

    raw_dir = root / "raw"
    hashes = {}
    for name in (TRAIN_ARCHIVE, TEST_ARCHIVE):
        path = raw_dir / name
        if path.is_file():
            hashes[name] = _sha256_file(path)
    if len(hashes) != 2:
        raise FileNotFoundError(f"official MADBase archives are required in {raw_dir}")
    if hashes != EXPECTED_SHA256:
        raise IOError(f"MADBase archive SHA-256 mismatch: {hashes}")

    processed = root / "processed"
    cache_path = processed / "madbase_auc_v1.npz"
    cache_meta_path = processed / "madbase_auc_v1.json"
    cache_ok = False
    if cache_path.is_file() and cache_meta_path.is_file():
        try:
            cache_meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
            cache_ok = cache_meta.get("archive_sha256") == hashes
        except (OSError, ValueError):
            cache_ok = False

    if cache_ok:
        with np.load(cache_path, allow_pickle=False) as cache:
            train_images = cache["train_images"]
            train_labels = cache["train_labels"]
            train_ids = cache["train_sample_ids"]
            train_writers = cache["train_writer_ids"]
            test_images = cache["test_images"]
            test_labels = cache["test_labels"]
            test_ids = cache["test_sample_ids"]
            test_writers = cache["test_writer_ids"]
        unique_train_writers = np.unique(train_writers)
        unique_test_writers = np.unique(test_writers)
    else:
        train_records, train_ids = _catalog(train_dir, "train", 60_000)
        test_records, test_ids = _catalog(test_dir, "test", 10_000)
        train_images, train_labels, train_writers, unique_train_writers = _decode(train_records, "train")
        test_images, test_labels, test_writers, unique_test_writers = _decode(test_records, "test")
        processed.mkdir(parents=True, exist_ok=True)
        tmp_path = processed / "madbase_auc_v1.tmp.npz"
        np.savez_compressed(
            tmp_path,
            train_images=train_images,
            train_labels=train_labels,
            train_sample_ids=train_ids,
            train_writer_ids=train_writers,
            test_images=test_images,
            test_labels=test_labels,
            test_sample_ids=test_ids,
            test_writer_ids=test_writers,
        )
        os.replace(tmp_path, cache_path)
        cache_meta_path.write_text(
            json.dumps({"dataset_id": DATASET_ID, "archive_sha256": hashes}, indent=2) + "\n",
            encoding="utf-8",
        )
    if set(unique_train_writers.tolist()) & set(unique_test_writers.tolist()):
        raise ValueError("MADBase official train/test writer sets overlap")
    if unique_train_writers.tolist() != list(range(1, 601)):
        raise ValueError("MADBase train writers must be writer001..writer600")
    if unique_test_writers.tolist() != list(range(601, 701)):
        raise ValueError("MADBase test writers must be writer601..writer700")

    return MADBase(
        train_images=train_images,
        train_labels=train_labels,
        train_sample_ids=train_ids,
        train_writer_ids=train_writers,
        test_images=test_images,
        test_labels=test_labels,
        test_sample_ids=test_ids,
        test_writer_ids=test_writers,
        file_sha256=hashes,
    )


def load_madbase_training(root: str | Path) -> MADBaseTrainingView:
    """Load only MADBase official-train arrays for leakage-safe model fitting."""
    root, train_dir, test_dir = _roots(root)
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(f"MADBase belum diekstrak di {root}")
    raw_dir = root / "raw"
    hashes = {name: _sha256_file(raw_dir / name) for name in (TRAIN_ARCHIVE, TEST_ARCHIVE)
              if (raw_dir / name).is_file()}
    if hashes != EXPECTED_SHA256:
        raise IOError(f"MADBase archive SHA-256 mismatch: {hashes}")
    cache_path = root / "processed" / "madbase_auc_v1.npz"
    cache_meta_path = root / "processed" / "madbase_auc_v1.json"
    if not cache_path.is_file() or not cache_meta_path.is_file():
        raise FileNotFoundError(
            "MADBase verified cache belum tersedia; jalankan scripts/download_madbase.py dan audit sebelum training"
        )
    meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
    if meta.get("archive_sha256") != hashes:
        raise IOError("MADBase cache provenance does not match the pinned archives")
    with np.load(cache_path, allow_pickle=False) as cache:
        train_images = cache["train_images"]
        train_labels = cache["train_labels"]
        train_ids = cache["train_sample_ids"]
        train_writers = cache["train_writer_ids"]
    if train_images.shape != (60_000, 28, 28) or train_labels.shape != (60_000,):
        raise ValueError("MADBase training cache shape mismatch")
    if np.unique(train_writers).tolist() != list(range(1, 601)):
        raise ValueError("MADBase train writers must be writer001..writer600")
    return MADBaseTrainingView(
        train_images=train_images,
        train_labels=train_labels,
        train_sample_ids=train_ids,
        train_writer_ids=train_writers,
        file_sha256=hashes,
        root=root,
    )
