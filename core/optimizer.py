"""COBYLA wrapper with objective-evaluation provenance (pseudocode §1.1, §5, §7; G1-02, G1-06)."""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import scipy
from scipy.optimize import minimize

from .artifact_io import ParamLog, append_jsonl, create_empty_file, save_json_atomic, save_npy_atomic


def objective_phase(eval_id: int, n_params: int) -> str:
    """OBJECTIVE_PHASE (G1-06): eval_id 0 = x0, 1..n_params = simplex probes."""
    return "initial_simplex" if eval_id <= n_params else "optimization"


def validate_optimizer_budget(max_nfev: int, n_params: int, run_mode: str) -> None:
    """VALIDATE_OPTIMIZER_BUDGET: SciPy silently raises budgets below n_params + 2."""
    simplex_size = n_params + 1
    if max_nfev < n_params + 2:
        raise ValueError(
            f"max_nfev={max_nfev} < n_params + 2 = {n_params + 2}; SciPy COBYLA would raise it silently"
        )
    if max_nfev <= simplex_size and run_mode == "CONFIRMATORY":
        raise ValueError("max_nfev must exceed n_params + 1 for confirmatory runs (G1-06)")


class CobylaRecorder:
    """Runs COBYLA on ``core_fn(theta) -> (train_loss, metrics)`` and records every evaluation.

    Files (prefix = 'clustering' | 'supervised'):
      artifacts/<prefix>_params.bin            theta of each eval_id (fixed-size records)
      logs/<prefix>_log.jsonl                  one line per objective evaluation
      logs/<prefix>_callback_log.jsonl         one line per optimizer callback (callback_id)
      logs/<prefix>_optimizer_result.json      COBYLA termination + provenance summary
      artifacts/<prefix>_params_final.npy      result.x (official final point)
      artifacts/<prefix>_params_best_observed.npy
    """

    def __init__(
        self,
        prefix: str,
        run_dir: str | Path,
        n_params: int,
        max_nfev: int,
        tol: float,
        rhobeg: float,
        run_mode: str,
        core_fn: Callable[[np.ndarray], tuple[float, dict]],
        extra_summary: dict | None = None,
    ):
        self.prefix = prefix
        self.run_dir = Path(run_dir)
        self.n_params = int(n_params)
        self.max_nfev = int(max_nfev)
        self.tol = float(tol)
        self.rhobeg = float(rhobeg)
        self.run_mode = run_mode
        self.core_fn = core_fn
        self.extra_summary = extra_summary or {}
        validate_optimizer_budget(self.max_nfev, self.n_params, run_mode)
        self.param_log = ParamLog(self.run_dir / "artifacts" / f"{prefix}_params.bin", self.n_params)
        self.log_path = self.run_dir / "logs" / f"{prefix}_log.jsonl"
        self.cb_path = self.run_dir / "logs" / f"{prefix}_callback_log.jsonl"
        create_empty_file(self.log_path)
        create_empty_file(self.cb_path)
        self.n_evals = 0
        self.callback_id = 0
        self.best_fun = np.inf
        self.best_eval_id = None
        self.best_theta = None
        self.log_rows: list[dict] = []

    def _objective(self, theta: np.ndarray) -> float:
        eval_id = self.n_evals
        t0 = time.perf_counter()
        c0 = time.process_time()
        theta = np.array(theta, dtype=np.float64, copy=True)
        loss, metrics = self.core_fn(theta)
        loss = float(loss)
        if not np.isfinite(loss):
            raise FloatingPointError(f"non-finite {self.prefix} train_loss at eval_id {eval_id}")
        self.param_log.append(theta)
        if loss < self.best_fun:
            self.best_fun = loss
            self.best_eval_id = eval_id
            self.best_theta = theta.copy()
        row = {"eval_id": eval_id, "phase": objective_phase(eval_id, self.n_params), "train_loss": loss}
        row.update(metrics)
        row["eval_runtime_sec"] = time.perf_counter() - t0
        row["eval_cpu_sec"] = time.process_time() - c0
        append_jsonl(self.log_path, row)
        self.log_rows.append(row)
        self.n_evals += 1
        return loss

    def _callback(self, intermediate_result) -> None:
        entry = {
            "callback_id": self.callback_id,
            "objective_evals_seen": self.n_evals,
            "theta_source": "optimizer_callback",
            "fun": float(intermediate_result.fun),
            "nit": int(getattr(intermediate_result, "nit", -1)),
        }
        append_jsonl(self.cb_path, entry)
        self.callback_id += 1

    def run(self, x0: np.ndarray) -> tuple[np.ndarray, dict]:
        x0 = np.asarray(x0, dtype=np.float64)
        if x0.shape != (self.n_params,):
            raise ValueError("x0 length must equal n_params")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = minimize(
                self._objective,
                x0,
                method="COBYLA",
                callback=self._callback,
                options={"maxiter": self.max_nfev, "tol": self.tol, "rhobeg": self.rhobeg},
            )
        budget_warnings = [str(w.message) for w in caught if "MAXFUN" in str(w.message)]
        if budget_warnings:
            raise RuntimeError(f"SciPy changed the evaluation budget: {budget_warnings}")
        if int(result.nfev) != self.n_evals:
            raise RuntimeError(f"result.nfev={result.nfev} != logged evaluations {self.n_evals}")
        final_x = np.array(result.x, dtype=np.float64)
        final_eval_id = self.param_log.find_eval_id(final_x)
        save_npy_atomic(final_x, self.run_dir / "artifacts" / f"{self.prefix}_params_final.npy")
        save_npy_atomic(self.best_theta, self.run_dir / "artifacts" / f"{self.prefix}_params_best_observed.npy")
        summary = {
            "optimizer": "COBYLA",
            "implementation": "scipy.optimize.minimize(method='COBYLA')",
            "max_nfev": self.max_nfev,
            "tol": self.tol,
            "rhobeg": self.rhobeg,
            "scipy_version": scipy.__version__,
            "n_params": self.n_params,
            "simplex_size": self.n_params + 1,
            "n_initial_simplex_evals": min(int(result.nfev), self.n_params + 1),
            "n_optimization_evals": max(0, int(result.nfev) - (self.n_params + 1)),
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "stopped_by_budget": int(result.nfev) >= self.max_nfev,
            "fun": float(result.fun),
            "nfev": int(result.nfev),
            "callback_count": self.callback_id,
            "final_point_eval_id": final_eval_id,
            "final_point_path": f"artifacts/{self.prefix}_params_final.npy",
            "best_observed_eval_id": self.best_eval_id,
            "best_observed_fun": float(self.best_fun),
            "best_observed_point_path": f"artifacts/{self.prefix}_params_best_observed.npy",
        }
        summary.update(self.extra_summary)
        save_json_atomic(summary, self.run_dir / "logs" / f"{self.prefix}_optimizer_result.json")
        return final_x, summary
