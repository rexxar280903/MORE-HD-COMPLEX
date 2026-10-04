"""G1-02 (objective-evaluation provenance) and G1-06 (simplex phase, budget rules) on real COBYLA."""

import numpy as np
import pytest

from core.artifact_io import ParamLog, load_json, read_jsonl
from core.optimizer import CobylaRecorder, objective_phase, validate_optimizer_budget


def _quadratic(n, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    q = a @ a.T / n + np.eye(n)

    def core(theta):
        return float(theta @ q @ theta + np.sin(theta).sum()), {"aux": float(theta.sum())}

    return core


def _recorder(tmp_path, n, max_nfev, run_mode="PILOT", prefix="clustering"):
    (tmp_path / "artifacts").mkdir(exist_ok=True)
    (tmp_path / "logs").mkdir(exist_ok=True)
    return CobylaRecorder(prefix, tmp_path, n, max_nfev, 1e-4, 1.0, run_mode, _quadratic(n))


def test_phase_labels():
    assert objective_phase(0, 30) == "initial_simplex"
    assert objective_phase(30, 30) == "initial_simplex"
    assert objective_phase(31, 30) == "optimization"
    assert objective_phase(61, 60) == "optimization"


def test_budget_validation():
    with pytest.raises(ValueError):
        validate_optimizer_budget(31, 30, "PILOT")
    validate_optimizer_budget(32, 30, "PILOT")
    validate_optimizer_budget(32, 30, "CONFIRMATORY")
    with pytest.raises(ValueError):
        validate_optimizer_budget(10, 30, "PILOT")


def test_logging_provenance_and_simplex_probes(tmp_path):
    n = 30
    rec = _recorder(tmp_path, n, 70)
    x0 = np.random.default_rng(1).uniform(0, 2 * np.pi, n)
    x, summary = rec.run(x0)
    rows = read_jsonl(tmp_path / "logs" / "clustering_log.jsonl")
    assert len(rows) == summary["nfev"] == 70
    assert [r["eval_id"] for r in rows] == list(range(70))
    assert [r["phase"] for r in rows] == ["initial_simplex"] * 31 + ["optimization"] * 39
    assert summary["n_initial_simplex_evals"] == 31 and summary["n_optimization_evals"] == 39
    plog = ParamLog(tmp_path / "artifacts" / "clustering_params.bin", n, create=False)
    params = plog.read_all()
    assert params.shape == (70, n) and np.array_equal(params[0], x0)
    assert np.array_equal(params[summary["final_point_eval_id"]], x)
    assert rows[summary["best_observed_eval_id"]]["train_loss"] == min(r["train_loss"] for r in rows)
    # every simplex probe moves exactly one coordinate by +-rhobeg from the best vertex so far
    best = 0
    for k in range(1, n + 1):
        diff = params[k] - params[best]
        nz = np.flatnonzero(diff)
        assert nz.size == 1 and abs(abs(diff[nz[0]]) - 1.0) < 1e-12
        if rows[k]["train_loss"] < rows[best]["train_loss"]:
            best = k
    cb = read_jsonl(tmp_path / "logs" / "clustering_callback_log.jsonl")
    assert [c["callback_id"] for c in cb] == list(range(len(cb)))
    saved = load_json(tmp_path / "logs" / "clustering_optimizer_result.json")
    for key in ("success", "status", "message", "fun", "nfev", "final_point_eval_id", "best_observed_eval_id",
                "scipy_version", "n_params", "simplex_size", "rhobeg", "tol", "max_nfev"):
        assert key in saved
    assert np.array_equal(np.load(tmp_path / "artifacts" / "clustering_params_final.npy"), x)


def test_param_log_detects_truncated_tail(tmp_path):
    plog = ParamLog(tmp_path / "p.bin", 4)
    plog.append(np.arange(4.0))
    with open(tmp_path / "p.bin", "ab") as fh:
        fh.write(b"\x00" * 5)
    with pytest.raises(IOError):
        plog.n_records()
