"""Inter-class correlation matrix S (pseudocode §3; G4-04).

Decision 2026-10-04 (comparability with MORE): S follows MORE's public code
``data_helper.py::calc_class_rela`` instead of the [0.5, 1] min-max rule of the
superseded thesis protocol:

* class means are taken over the first ``CORR_N_MEAN_SAMPLES_PER_CLASS`` (=100)
  training samples of each class, in split-manifest order (seeded random order);
* ``raw_mse[i, j] = mean_k (mu_i[k] - mu_j[k])**2`` over the 8 input channels;
* off-diagonal ``S[i, j] = raw_mse[i, j] / max_{a != b} raw_mse[a, b]``
  (MORE: ``(mse - min) / max - min`` with ``min = 0`` on the diagonal);
* diagonal ``S[i, i] = -1`` (same class: pulled together).

MORE rounds intermediate values to 3 decimals; this implementation keeps full
float64 precision (documented deviation). If every off-diagonal MSE is zero the
fallback sets all off-diagonal entries to 1.0 and records it.
"""

from __future__ import annotations

import numpy as np

from .constants import CORR_N_MEAN_SAMPLES_PER_CLASS, CORRELATION_RULE


def correlation_matrix(
    x_train: np.ndarray,
    y_train: np.ndarray,
    classes,
    n_mean_samples: int = CORR_N_MEAN_SAMPLES_PER_CLASS,
) -> dict:
    classes = [int(c) for c in classes]
    k = len(classes)
    class_to_index = {c: i for i, c in enumerate(classes)}
    means = np.zeros((k, x_train.shape[1]), dtype=np.float64)
    n_used = {}
    for c in classes:
        rows = np.flatnonzero(y_train == c)[:n_mean_samples]
        if rows.size == 0:
            raise ValueError(f"class {c} has no training samples")
        means[class_to_index[c]] = x_train[rows].mean(axis=0)
        n_used[c] = int(rows.size)
    diff = means[:, None, :] - means[None, :, :]
    raw_mse = (diff ** 2).mean(axis=2)
    off = ~np.eye(k, dtype=bool)
    max_off = float(raw_mse[off].max()) if k > 1 else 0.0
    fallback = False
    s = np.empty((k, k), dtype=np.float64)
    if max_off > 0.0:
        s[off] = raw_mse[off] / max_off
    else:
        s[off] = 1.0
        fallback = True
    np.fill_diagonal(s, -1.0)
    # symmetry is exact: raw_mse[i, j] and raw_mse[j, i] are computed from the same squares
    assert np.array_equal(s, s.T)
    assert np.all(np.diag(s) == -1.0)
    assert np.all((s[off] >= 0.0) & (s[off] <= 1.0))
    return {
        "S": s,
        "raw_mse": raw_mse,
        "class_to_index": class_to_index,
        "class_means": means,
        "n_mean_samples": n_used,
        "normalizer_max_offdiag_mse": max_off,
        "fallback_all_offdiag_one": fallback,
        "rule": CORRELATION_RULE,
    }
