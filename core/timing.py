"""Run clock, stage timing and run-start metadata (pseudocode §9.4)."""

from __future__ import annotations

import datetime as _dt
import os
import platform
import socket
import time
from pathlib import Path

from .artifact_io import append_jsonl, create_empty_file, read_jsonl, save_json_atomic
from .constants import THREAD_ENV_VARS


def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def load_avg_1m():
    try:
        return float(os.getloadavg()[0])
    except (AttributeError, OSError):
        return None


def thread_env() -> dict:
    return {k: os.environ.get(k, "") for k in THREAD_ENV_VARS}


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


class RunClock:
    """START_RUN_CLOCK / BEGIN_STAGE / END_STAGE / RUN_TIMING_SUMMARY."""

    def __init__(self, n_parallel_declared=None):
        self.t0 = time.perf_counter()
        self.c0 = time.process_time()
        self.started_at = now_iso()
        self.load_avg_1m_start = load_avg_1m()
        self.n_parallel_declared = n_parallel_declared
        self.open_stage: dict | None = None
        self.run_dir: Path | None = None
        self.completed_stages: list[str] = []

    @property
    def current_stage(self):
        return None if self.open_stage is None else self.open_stage["stage"]

    def write_run_started(self, run_dir: str | Path, run_name: str, path_type: str, extra: dict | None = None) -> None:
        self.run_dir = Path(run_dir)
        payload = {
            "run_name": run_name,
            "path_type": path_type,
            "started_at": self.started_at,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "cpu_count": os.cpu_count(),
            "cpu_model": cpu_model(),
            "load_avg_1m_start": self.load_avg_1m_start,
            "n_parallel_declared": self.n_parallel_declared,
            "thread_env": thread_env(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        }
        if extra:
            payload.update(extra)
        save_json_atomic(payload, self.run_dir / "logs" / "run_started.json")
        create_empty_file(self.run_dir / "logs" / "stage_timing.jsonl")

    def begin_stage(self, stage: str) -> None:
        if self.open_stage is not None:
            raise RuntimeError(f"stage {self.open_stage['stage']} still open")
        self.open_stage = {"stage": stage, "t0": time.perf_counter(), "c0": time.process_time()}

    def end_stage(self, stage: str) -> None:
        if self.open_stage is None or self.open_stage["stage"] != stage:
            raise RuntimeError(f"end_stage({stage}) without matching begin_stage")
        append_jsonl(self.run_dir / "logs" / "stage_timing.jsonl", {
            "stage": stage,
            "wall_sec": time.perf_counter() - self.open_stage["t0"],
            "cpu_sec": time.process_time() - self.open_stage["c0"],
            "finished_at": now_iso(),
        })
        self.completed_stages.append(stage)
        self.open_stage = None

    def total_runtime(self) -> float:
        return time.perf_counter() - self.t0

    def summary(self) -> dict:
        stages = read_jsonl(self.run_dir / "logs" / "stage_timing.jsonl")
        from .artifact_io import load_json

        started = load_json(self.run_dir / "logs" / "run_started.json")
        return {
            "started_at": self.started_at,
            "finished_at": now_iso(),
            "total_runtime_sec": self.total_runtime(),
            "total_cpu_sec": time.process_time() - self.c0,
            "stage_wall_sec": {s["stage"]: s["wall_sec"] for s in stages},
            "stage_cpu_sec": {s["stage"]: s["cpu_sec"] for s in stages},
            "hostname": started["hostname"],
            "cpu_count": started["cpu_count"],
            "thread_env": started["thread_env"],
            "n_parallel_declared": self.n_parallel_declared,
            "load_avg_1m_start": self.load_avg_1m_start,
            "load_avg_1m_end": load_avg_1m(),
        }
