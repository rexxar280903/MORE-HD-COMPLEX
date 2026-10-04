"""G1-01 plateau rule, G1-11 6<->9 sensitivity, and run-identity rules."""

import importlib.util

import numpy as np
import pytest

from core import constants as C
from core.config import ablation_condition_id, auto_generate_run_name, more_reference_condition_id, primary_condition_id
from tests.conftest import ROOT


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_plateau_rule():
    budget = _load("scripts/analyze_pilot_budget.py", "budget")
    losses = [10.0] + [10.0 - 9.0 * (1 - np.exp(-i / 20)) for i in range(1, 300)]
    st = budget.curve_stats(losses, stopped_by_budget=True)
    best = np.minimum.accumulate(losses)
    thr = 0.02 * (best[0] - best[-1])
    assert st["n_star"] == int(np.flatnonzero(best - best[-1] <= thr)[0]) + 1
    assert st["plateaued"]
    still = [10.0 - 0.03 * i for i in range(300)]
    assert not budget.curve_stats(still, stopped_by_budget=True)["plateaued"]
    assert budget.curve_stats(still, stopped_by_budget=False)["plateaued"]          # stopped by tol
    assert budget.curve_stats([1.0] * 50, stopped_by_budget=True)["n_star"] == 1


def test_six_nine_sensitivity_row():
    analysis = _load("analysis/run_analysis.py", "analysis")
    cm = np.diag([20] * 10).astype(int)
    cm[6, 9], cm[6, 6] = 5, 15
    cm[9, 6], cm[9, 9] = 3, 17
    row = analysis._six_nine_row(cm, "A", "MORE-HD", "HU", 101)
    assert row["six_nine_confusions"] == 8
    assert row["accuracy"] == pytest.approx(192 / 200)
    assert row["accuracy_merged_6_9"] == pytest.approx(200 / 200)
    assert row["six_nine_confusion_rate"] == pytest.approx(8 / 40)
    assert row["share_of_errors_6_9"] == pytest.approx(1.0)


def test_identifiers():
    assert primary_condition_id(3, "PCA", "MORE-HD") == "R001"
    assert primary_condition_id(10, "ZERNIKE", "MORE-HD-C") == "R048"
    assert ablation_condition_id(3, "PCA", "MORE-HD-60P") == "ABL001"
    assert ablation_condition_id(10, "ZERNIKE", "MORE-HD-C-FixedRZ") == "ABL018"
    assert more_reference_condition_id(3, "PCA") == "MREF001"
    assert more_reference_condition_id(10, "ZERNIKE") == "MREF024"
    assert auto_generate_run_name([0, 1, 2], 1000, 100, 200, "MORE-HD", "PCA", 101) == \
        "cls-0-1-2_ntrain1000_nval100_ntest200_PCA_MORE-HD_seed101"
    n_primary = len(C.K_VALUES) * 3 * 2 * 5
    n_ablation = len(C.ABLATION_K_VALUES) * 3 * 2 * 5
    n_ref = len(C.K_VALUES) * 3 * 5
    assert (n_primary, n_ablation, n_ref) == (240, 90, 120)


def test_confirmatory_config_validation(tmp_path, monkeypatch):
    from core.config import RunConfig, validate_config

    for v in C.THREAD_ENV_VARS:
        monkeypatch.setenv(v, "1")
    ok = RunConfig("MORE-HD", "PCA", 3, 101, run_mode="CONFIRMATORY", n_parallel_declared=1,
                   max_nfev_clustering=100, max_nfev_supervised=100)
    assert validate_config(ok) == []
    with pytest.raises(ValueError):
        validate_config(RunConfig("MORE-HD", "PCA", 3, 42, run_mode="CONFIRMATORY", n_parallel_declared=1))
    with pytest.raises(ValueError):
        validate_config(RunConfig("MORE-HD", "PCA", 3, 101, run_mode="CONFIRMATORY"))          # n_parallel missing
    with pytest.raises(ValueError):
        validate_config(RunConfig("MORE-HD", "PCA", 3, 101, run_mode="CONFIRMATORY", n_parallel_declared=1,
                                  n_train_per_class=50))
    with pytest.raises(ValueError):
        validate_config(RunConfig("MORE-HD-60P", "PCA", 4, 42))                                 # ablation K
    with pytest.raises(ValueError):
        validate_config(RunConfig("MORE-HD", "PCA", 3, 42, shots=1000))
    with pytest.raises(ValueError):
        validate_config(RunConfig("MORE-HD", "PCA", 3, 42, eps_norm=1e-8))
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    with pytest.raises(ValueError):
        validate_config(ok)
