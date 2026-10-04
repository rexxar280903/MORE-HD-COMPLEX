"""Environment metadata recorded in every manifest (G4-01)."""

from __future__ import annotations

import importlib
import os
import platform
import subprocess
from pathlib import Path

from .constants import LOCKED_THREADS, THREAD_ENV_VARS

PACKAGES = ("numpy", "scipy", "sklearn", "pennylane", "joblib", "openpyxl")


def lock_threads() -> None:
    """Set the G4-01 thread limits; must run before numpy is imported to take effect."""
    for var in THREAD_ENV_VARS:
        os.environ.setdefault(var, LOCKED_THREADS)


def package_versions() -> dict:
    out = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception:  # pragma: no cover - optional dependency missing
            out[name] = None
    return out


def blas_threads() -> dict:
    try:
        from threadpoolctl import threadpool_info

        return {f"{d.get('internal_api')}:{d.get('filepath', '').split('/')[-1]}": d.get("num_threads")
                for d in threadpool_info()}
    except Exception:  # pragma: no cover
        return {}


def git_info(root: str | Path = ".") -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False).stdout.strip()

    commit = run("rev-parse", "HEAD") or None
    dirty_files = [line for line in run("status", "--porcelain").splitlines() if line.strip()]
    return {"git_commit": commit, "git_dirty": bool(dirty_files), "git_dirty_files": dirty_files[:50]}


def environment_record(root: str | Path = ".") -> dict:
    rec = {
        "versions": package_versions(),
        "platform": platform.platform(),
        "blas_threads": blas_threads(),
        "thread_env": {k: os.environ.get(k, "") for k in THREAD_ENV_VARS},
    }
    rec.update(git_info(root))
    return rec
