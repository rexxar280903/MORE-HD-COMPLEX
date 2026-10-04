"""G1-04: statistical reporting functions against independent references."""

import itertools

import numpy as np
import pytest
from scipy import stats as st

from core.stats import describe, hedges_g_paired, holm, mcnemar_exact, paired_summary, sign_flip_exact, t_confidence_interval


def test_t_interval_matches_scipy():
    d = np.array([0.02, 0.05, -0.01, 0.04, 0.03])
    lo, hi = t_confidence_interval(d)
    ref = st.ttest_1samp(d, 0.0).confidence_interval(0.95)
    assert lo == pytest.approx(ref.low) and hi == pytest.approx(ref.high)


def test_hedges_g_paired_formula():
    d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    dz = d.mean() / d.std(ddof=1)
    assert hedges_g_paired(d) == pytest.approx(dz * (1 - 3 / (4 * 4 - 1)))
    assert hedges_g_paired([1.0, 1.0, 1.0]) is None


def test_sign_flip_exact_enumeration():
    d = np.array([0.3, -0.1, 0.2, 0.4, 0.25])
    obs = abs(d.mean())
    flips = [abs((d * np.array(s)).mean()) for s in itertools.product([1, -1], repeat=5)]
    assert sign_flip_exact(d) == pytest.approx(np.mean([f >= obs - 1e-12 for f in flips]))
    assert sign_flip_exact([1, 1, 1, 1, 1]) == pytest.approx(2 / 32)


def test_mcnemar_exact_against_binomial():
    a = np.array([1, 1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=bool)
    b = np.array([0, 0, 1, 0, 1, 0, 0, 0, 1, 1], dtype=bool)
    res = mcnemar_exact(a, b)
    assert (res["a_right_b_wrong"], res["a_wrong_b_right"]) == (4, 1)
    assert res["p_value"] == pytest.approx(st.binomtest(1, 5, 0.5).pvalue)
    assert mcnemar_exact(a, a)["p_value"] == 1.0


def test_holm_known_example():
    p = [0.01, 0.04, 0.03, 0.005]
    assert holm(p) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert holm([0.2, None]) == [0.2, None]


def test_paired_summary_and_describe():
    a = [0.70, 0.72, 0.69, 0.71, 0.70]
    d = [0.74, 0.75, 0.70, 0.76, 0.73]
    s = paired_summary(a, d)
    assert s["n_pairs"] == 5 and s["mean_delta"] == pytest.approx(np.mean(np.subtract(d, a)))
    assert s["ci95_low"] < s["mean_delta"] < s["ci95_high"]
    desc = describe([1, 2, 3, 4, 5])
    assert desc["median"] == 3 and desc["iqr"] == 2 and desc["sd"] == pytest.approx(np.std([1, 2, 3, 4, 5], ddof=1))
