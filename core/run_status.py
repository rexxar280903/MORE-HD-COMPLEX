"""Run status and attempt policy (G3-04).

* ``logs/run_status.json`` is written as RUNNING right after the run folder is
  created and rewritten as COMPLETED or FAILED (with the error and traceback).
* A hard crash (power loss, kill -9) leaves RUNNING; such a run must be marked
  FAILED explicitly with ``scripts/mark_run_failed.py`` before a new attempt.
* A new attempt of the same ``run_uid`` uses the next attempt number, a new
  folder (``..._attempt<N>``) and is only allowed when every previous attempt is
  FAILED. Old folders are never modified, so failures cannot damage completed
  runs and reruns never overwrite earlier artifacts.
"""

from __future__ import annotations

import re
import traceback
from pathlib import Path

from .artifact_io import load_json, save_json_atomic
from .constants import STATUS_COMPLETED, STATUS_FAILED, STATUS_RUNNING
from .timing import now_iso

ATTEMPT_RE = re.compile(r"^(?P<base>.+?)(?:_attempt(?P<n>\d+))?$")


def status_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "logs" / "run_status.json"


def write_status(run_dir, run_uid: str, attempt: int, status: str, started_at: str, error: BaseException | None = None) -> None:
    payload = {
        "run_uid": run_uid,
        "attempt": attempt,
        "status": status,
        "started_at": started_at,
        "updated_at": now_iso(),
    }
    if error is not None:
        payload["error_type"] = type(error).__name__
        payload["error_message"] = str(error)
        payload["traceback"] = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    save_json_atomic(payload, status_path(run_dir))


def read_status(run_dir) -> dict | None:
    p = status_path(run_dir)
    return load_json(p) if p.exists() else None


def existing_attempts(base_dir: Path) -> list[tuple[int, Path, dict | None]]:
    """All folders of the same run (base name and _attempt<N> suffixes)."""
    parent = base_dir.parent
    if not parent.exists():
        return []
    out = []
    for p in parent.iterdir():
        m = ATTEMPT_RE.match(p.name)
        if p.is_dir() and m and m.group("base") == base_dir.name:
            n = int(m.group("n") or 1)
            out.append((n, p, read_status(p)))
    return sorted(out, key=lambda t: t[0])


def check_attempt_allowed(base_dir: Path, attempt: int) -> None:
    attempts = existing_attempts(base_dir)
    numbers = [n for n, _, _ in attempts]
    if attempt in numbers:
        raise FileExistsError(f"attempt {attempt} already exists for {base_dir.name}")
    expected = (max(numbers) + 1) if numbers else 1
    if attempt != expected:
        raise ValueError(f"next attempt for {base_dir.name} must be {expected}, got {attempt}")
    for n, path, st in attempts:
        state = None if st is None else st.get("status")
        if state != STATUS_FAILED:
            raise RuntimeError(
                f"attempt {n} ({path}) has status {state!r}; a new attempt is allowed only when all "
                "previous attempts are FAILED (mark a crashed run with scripts/mark_run_failed.py)"
            )


def mark_failed(run_dir, reason: str) -> None:
    st = read_status(run_dir)
    if st is None:
        raise FileNotFoundError("no run_status.json")
    if st["status"] == STATUS_COMPLETED:
        raise RuntimeError("refusing to mark a COMPLETED run as FAILED")
    st["status"] = STATUS_FAILED
    st["updated_at"] = now_iso()
    st["error_type"] = st.get("error_type") or "ManualMark"
    st["error_message"] = reason
    save_json_atomic(st, status_path(run_dir))


__all__ = [
    "STATUS_RUNNING", "STATUS_COMPLETED", "STATUS_FAILED", "write_status", "read_status",
    "existing_attempts", "check_attempt_allowed", "mark_failed",
]
