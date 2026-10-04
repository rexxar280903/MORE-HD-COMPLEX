"""Statistical reporting frozen in MORE_HD_STATISTICAL_ANALYSIS_PLAN.md §7-§9 (G1-04).

Descriptive five-seed summaries, paired differences with 95% Student-t CI,
paired Hedges g, exact two-sided sign-flip permutation test, exact McNemar,
and Holm adjustment.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import stats as st


def describe(values) -> dict:
    v = np.asarray([x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=float)
    n = int(v.size)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "median": None, "q1": None, "q3": None, "iqr": None,
                "min": None, "max": None}
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {
        "n": n, "mean": float(v.mean()), "sd": float(v.std(ddof=1)) if n > 1 else None,
        "median": float(med), "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1),
        "min": float(v.min()), "max": float(v.max()),
    }


def hedges_g_paired(deltas) -> float | None:
    """d_z = mean(delta) / sd(delta), times the small-sample correction J = 1 - 3 / (4 df - 1)."""
    d = np.asarray(deltas, dtype=float)
    n = d.size
    if n < 2:
        return None
    sd = d.std(ddof=1)
    if sd == 0:
        return None
    df = n - 1
    j = 1.0 - 3.0 / (4.0 * df - 1.0)
    return float(j * d.mean() / sd)


def t_confidence_interval(deltas, level: float = 0.95):
    d = np.asarray(deltas, dtype=float)
    n = d.size
    if n < 2:
        return None, None
    se = d.std(ddof=1) / math.sqrt(n)
    h = st.t.ppf(0.5 + level / 2.0, n - 1) * se
    return float(d.mean() - h), float(d.mean() + h)


def sign_flip_exact(deltas) -> float | None:
    """Exact two-sided sign-flip permutation p-value for the mean of paired differences."""
    d = np.asarray(deltas, dtype=float)
    n = d.size
    if n == 0:
        return None
    obs = abs(d.mean())
    count = 0
    total = 0
    for signs in itertools.product((1.0, -1.0), repeat=n):
        total += 1
        if abs((d * np.asarray(signs)).mean()) >= obs - 1e-12:
            count += 1
    return count / total


def mcnemar_exact(correct_a, correct_b) -> dict:
    """Exact (binomial) McNemar test on paired per-sample correctness."""
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    n01 = int(np.sum(a & ~b))     # A right, B wrong
    n10 = int(np.sum(~a & b))     # A wrong, B right
    n = n01 + n10
    p = 1.0 if n == 0 else float(st.binomtest(min(n01, n10), n, 0.5, alternative="two-sided").pvalue)
    return {"a_right_b_wrong": n01, "a_wrong_b_right": n10, "p_value": p}


def holm(pvalues) -> list:
    """Holm step-down adjusted p-values (None entries are passed through)."""
    idx = [i for i, p in enumerate(pvalues) if p is not None]
    m = len(idx)
    order = sorted(idx, key=lambda i: pvalues[i])
    adjusted = [None] * len(pvalues)
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * pvalues[i])
        running = max(running, val)
        adjusted[i] = running
    return adjusted


def paired_summary(a_values, b_values) -> dict:
    """Delta = B - A over seed-matched pairs (SAP §6: Delta_AD = D - A)."""
    pairs = [(a, b) for a, b in zip(a_values, b_values) if a is not None and b is not None]
    d = np.asarray([b - a for a, b in pairs], dtype=float)
    desc = describe(d)
    lo, hi = t_confidence_interval(d)
    return {
        "n_pairs": int(d.size), "deltas": d.tolist(), "mean_delta": desc["mean"], "sd_delta": desc["sd"],
        "median_delta": desc["median"], "iqr_delta": desc["iqr"], "ci95_low": lo, "ci95_high": hi,
        "hedges_g": hedges_g_paired(d), "sign_flip_p": sign_flip_exact(d) if d.size else None,
    }
