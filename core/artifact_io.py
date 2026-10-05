"""Crash-safe artifact I/O (pseudocode "crash-safe append-only log", G1-02, G3-04)."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _to_jsonable(obj: Any) -> Any:
    """Convert numpy scalars/arrays and non-finite floats to strict JSON values."""
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def dumps(obj: Any) -> str:
    return json.dumps(_to_jsonable(obj), ensure_ascii=False, allow_nan=False)


def save_json_atomic(obj: Any, path: str | os.PathLike) -> None:
    """SAVE_ATOMIC_JSON: write to a temp file in the same directory, fsync, then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(_to_jsonable(obj), fh, ensure_ascii=False, indent=2, allow_nan=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load_json(path: str | os.PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def create_empty_file(path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8"):
        pass


def append_jsonl(path: str | os.PathLike, record: Any) -> None:
    """APPEND_LINE(TO_JSON(record)) with flush + fsync so a crash loses at most one line."""
    line = dumps(record)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_npy_atomic(array: np.ndarray, path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            np.save(fh, np.asarray(array), allow_pickle=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def load_npy(path: str | os.PathLike) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    """Hash of dtype + shape + C-ordered bytes (SHA256_ARRAY in §4)."""
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(json.dumps(_to_jsonable(obj), sort_keys=True).encode()).hexdigest()


class ParamLog:
    """Append-only fixed-record float64 parameter log (``*_params.bin``).

    Record ``eval_id`` sits at byte offset ``eval_id * n_params * 8``. A partially
    written tail record (crash) is detectable because the file size is not a
    multiple of the record size.
    """

    def __init__(self, path: str | os.PathLike, n_params: int, create: bool = True):
        self.path = Path(path)
        self.n_params = int(n_params)
        self.record_size = self.n_params * 8
        if create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "xb"):
                pass

    def append(self, theta: np.ndarray) -> None:
        arr = np.ascontiguousarray(np.asarray(theta, dtype="<f8"))
        if arr.shape != (self.n_params,):
            raise ValueError(f"param record shape {arr.shape} != ({self.n_params},)")
        with open(self.path, "ab") as fh:
            fh.write(arr.tobytes())
            fh.flush()
            os.fsync(fh.fileno())

    def n_records(self) -> int:
        size = self.path.stat().st_size
        if size % self.record_size:
            raise IOError(f"{self.path}: truncated record (size {size} not multiple of {self.record_size})")
        return size // self.record_size

    def read(self, eval_id: int) -> np.ndarray:
        n = self.n_records()
        if not 0 <= eval_id < n:
            raise IndexError(f"eval_id {eval_id} outside 0..{n - 1}")
        with open(self.path, "rb") as fh:
            fh.seek(eval_id * self.record_size)
            data = fh.read(self.record_size)
        return np.frombuffer(data, dtype="<f8").astype(np.float64)

    def read_all(self) -> np.ndarray:
        n = self.n_records()
        return np.fromfile(self.path, dtype="<f8").reshape(n, self.n_params)

    def find_eval_id(self, theta: np.ndarray) -> int:
        """FIND_EVAL_ID_OF_THETA: first record bit-identical to ``theta``."""
        target = np.asarray(theta, dtype=np.float64)
        records = self.read_all()
        hits = np.flatnonzero(np.all(records == target[None, :], axis=1))
        if hits.size == 0:
            raise RuntimeError("result.x not found in parameter log; check SciPy version")
        return int(hits[0])


def iter_files(root: str | os.PathLike, pattern: str) -> Iterable[Path]:
    return sorted(Path(root).glob(pattern))
