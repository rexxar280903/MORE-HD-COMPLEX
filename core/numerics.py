"""Shared numerical functions (pseudocode §4.3.1, §4.3.2; G2-04, G2-05, G2-07).

Every cosine distance, normalisation, nearest-label decision and centroid in
the pipeline goes through this module. All dot products and norms are
computed as elementwise products reduced over the last (observable) axis, so a
given pair of vectors yields bit-identical results whichever function or batch
shape is used (required by the G3-05 cache-equivalence test).
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import spearmanr

from .constants import ACTIVITY_THRESHOLDS, EPS_NORM


class NumericalError(FloatingPointError):
    """Non-finite circuit output: a bug, the run is marked FAILED (G2-04, G3-04)."""


def assert_finite(v: np.ndarray) -> None:
    if not np.all(np.isfinite(v)):
        raise NumericalError("output sirkuit non-finite")


def l2_norms(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return np.sqrt((v * v).sum(axis=-1))


def is_degenerate(v: np.ndarray, eps_norm: float = EPS_NORM) -> np.ndarray:
    return l2_norms(v) < eps_norm


def normalize_rows(v: np.ndarray, eps_norm: float = EPS_NORM) -> np.ndarray:
    """NORMALIZE_VECTOR row-wise; rows with norm < eps_norm become the zero vector."""
    v = np.asarray(v, dtype=np.float64)
    assert_finite(v)
    n = l2_norms(v)
    deg = n < eps_norm
    safe = np.where(deg, 1.0, n)
    out = v / safe[..., None]
    out[deg] = 0.0
    return out


def _cos_from(dot: np.ndarray, nu: np.ndarray, nv: np.ndarray, eps_norm: float) -> np.ndarray:
    deg = (nu < eps_norm) | (nv < eps_norm)
    denom = np.where(deg, 1.0, nu * nv)
    c = np.clip(dot / denom, -1.0, 1.0)
    d = 1.0 - c
    return np.where(deg, 1.0, d)


def cosine_distance_rows(u: np.ndarray, v: np.ndarray, eps_norm: float = EPS_NORM) -> np.ndarray:
    """COSINE_DISTANCE for aligned rows: d_i = 1 - cos(u_i, v_i); degenerate -> 1.0."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    assert_finite(u)
    assert_finite(v)
    return _cos_from((u * v).sum(axis=-1), l2_norms(u), l2_norms(v), eps_norm)


def cosine_distance_matrix(v: np.ndarray, labels: np.ndarray, eps_norm: float = EPS_NORM) -> np.ndarray:
    """(N, d) outputs x (K, d) labels -> (N, K) cosine distances with the G2-04 fallback."""
    v = np.asarray(v, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    assert_finite(v)
    assert_finite(labels)
    dot = (v[:, None, :] * labels[None, :, :]).sum(axis=-1)
    return _cos_from(dot, l2_norms(v)[:, None], l2_norms(labels)[None, :], eps_norm)


def argmin_class(distances: np.ndarray) -> np.ndarray:
    """ARGMIN_CLASS: column index of the minimum; ties -> first class (np.argmin rule)."""
    return np.argmin(distances, axis=-1)


def class_centroid_more(outputs_c: np.ndarray, eps_norm: float = EPS_NORM) -> tuple[np.ndarray, int, float, bool]:
    """CLASS_CENTROID_MORE (MORE find_center): normalise -> component median -> normalise.

    Degenerate samples are excluded from the median and counted.
    Returns (centroid, n_excluded, prenorm, is_degenerate).
    """
    outputs_c = np.asarray(outputs_c, dtype=np.float64)
    assert_finite(outputs_c)
    norms = l2_norms(outputs_c)
    keep = norms >= eps_norm
    n_excluded = int((~keep).sum())
    d = outputs_c.shape[-1]
    if not np.any(keep):
        return np.zeros(d), n_excluded, 0.0, True
    units = outputs_c[keep] / norms[keep][:, None]
    med = np.median(units, axis=0)
    prenorm = float(l2_norms(med))
    if prenorm < eps_norm:
        return np.zeros(d), n_excluded, prenorm, True
    return med / prenorm, n_excluded, prenorm, False


def centroids_by_class(outputs: np.ndarray, groups: list, eps_norm: float = EPS_NORM) -> dict:
    """Centroids for row-index groups (one group per class, in class order)."""
    cents, excl, pren, deg = [], [], [], []
    for rows in groups:
        c, e, p, g = class_centroid_more(outputs[rows], eps_norm)
        cents.append(c)
        excl.append(e)
        pren.append(p)
        deg.append(g)
    return {"centroids": np.stack(cents), "n_excluded": excl, "prenorm": pren, "degenerate": deg}


def pairwise_label_distances(labels: np.ndarray, eps_norm: float = EPS_NORM) -> np.ndarray:
    k = labels.shape[0]
    dmat = np.zeros((k, k))
    for i in range(k):
        dmat[i] = cosine_distance_rows(np.repeat(labels[i:i + 1], k, axis=0), labels, eps_norm)
    return dmat


def min_separation(labels: np.ndarray, eps_norm: float = EPS_NORM) -> tuple[float, int, int]:
    """Minimum pairwise cosine distance; ties -> lexicographically smallest (i, j)."""
    k = labels.shape[0]
    best, bi, bj = np.inf, -1, -1
    dmat = pairwise_label_distances(labels, eps_norm)
    for i in range(k):
        for j in range(i + 1, k):
            if dmat[i, j] < best:
                best, bi, bj = float(dmat[i, j]), i, j
    return best, bi, bj


def correlation_consistency(s_matrix: np.ndarray, labels: np.ndarray, eps_norm: float = EPS_NORM):
    """Spearman between off-diagonal S values and label distances (diagnostic only)."""
    k = labels.shape[0]
    iu = np.triu_indices(k, 1)
    dmat = pairwise_label_distances(labels, eps_norm)
    s_vals = s_matrix[iu]
    d_vals = dmat[iu]
    if np.ptp(s_vals) == 0 or np.ptp(d_vals) == 0:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = spearmanr(s_vals, d_vals).statistic
    return None if not np.isfinite(rho) else float(rho)


def active_dimensions_centroid(centroids: np.ndarray, threshold: float) -> int:
    return int(np.sum(np.any(np.abs(centroids) > threshold, axis=0)))


def yodd_norm_fraction(outputs: np.ndarray, yodd_idx, eps_norm: float = EPS_NORM):
    """YODD_NORM_FRACTION: mean over non-degenerate samples of ||v_Yodd||^2 / ||v||^2."""
    outputs = np.asarray(outputs, dtype=np.float64)
    n2 = (outputs * outputs).sum(axis=-1)
    keep = np.sqrt(n2) >= eps_norm
    n_excl = int((~keep).sum())
    if not np.any(keep) or len(yodd_idx) == 0:
        return (None if not np.any(keep) else 0.0), n_excl
    y = outputs[keep][:, list(yodd_idx)]
    frac = (y * y).sum(axis=-1) / n2[keep]
    return float(frac.mean()), n_excl


def observable_activity(outputs: np.ndarray, labels, yodd_idx, eps_norm: float, primary_threshold: float) -> dict:
    """OBSERVABLE_ACTIVITY (§4.3.2): per-observable statistics on per-sample outputs."""
    outputs = np.asarray(outputs, dtype=np.float64)
    a = np.abs(outputs)
    norms2 = (outputs * outputs).sum(axis=-1)
    nondeg = np.sqrt(norms2) >= eps_norm
    per_obs = []
    for k, lab in enumerate(labels):
        frac = None
        if np.any(nondeg):
            frac = float((outputs[nondeg, k] ** 2 / norms2[nondeg]).mean())
        per_obs.append({
            "observable": lab,
            "is_yodd": k in yodd_idx,
            "mean_abs": float(a[:, k].mean()),
            "max_abs": float(a[:, k].max()),
            "std_abs": float(a[:, k].std()),
            "norm_fraction": frac,
            "is_active": bool(a[:, k].max() > primary_threshold),
        })
    sens = []
    for tau in ACTIVITY_THRESHOLDS:
        sens.append({
            "threshold": tau,
            "n_active": int(sum(p["max_abs"] > tau for p in per_obs)),
            "n_active_yodd": int(sum(p["is_yodd"] and p["max_abs"] > tau for p in per_obs)),
        })
    yfrac, n_excl = yodd_norm_fraction(outputs, yodd_idx, eps_norm)
    yodd_max = max((p["max_abs"] for p in per_obs if p["is_yodd"]), default=None)
    return {
        "n_samples": int(outputs.shape[0]),
        "n_degenerate_excluded": n_excl,
        "yodd_norm_fraction": yfrac,
        "yodd_max_abs": yodd_max,
        "n_active_primary": int(sum(p["is_active"] for p in per_obs)),
        "per_observable": per_obs,
        "threshold_sensitivity": sens,
    }
