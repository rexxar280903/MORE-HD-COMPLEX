"""G2-04 (zero-norm safety), G2-05 (Y-odd diagnostics), G2-06 (separation), G2-07 (MORE centroid)."""

import numpy as np
import pytest

from core import constants as C
from core.numerics import (
    NumericalError,
    argmin_class,
    class_centroid_more,
    cosine_distance_matrix,
    cosine_distance_rows,
    min_separation,
    normalize_rows,
    observable_activity,
    yodd_norm_fraction,
)


def more_find_center(points):
    """MORE MORE_clustering.py::find_center for one class (resize -> median -> resize)."""
    v = [p / np.linalg.norm(p) for p in points]
    c = np.median(np.array(v), axis=0)
    return c / np.linalg.norm(c)


def more_cos_dist(a, b):
    """MORE evaluation distance: 1 - dot(a/|a|, b/|b|)."""
    a = np.asarray(a) / np.linalg.norm(a)
    b = np.asarray(b) / np.linalg.norm(b)
    return 1 - np.dot(a, b)


def test_zero_and_near_zero_norm_fallbacks():
    z = np.zeros((1, 15))
    tiny = np.full((1, 15), 1e-12)
    v = np.ones((1, 15))
    assert cosine_distance_rows(z, v)[0] == 1.0
    assert cosine_distance_rows(tiny, v)[0] == 1.0
    assert cosine_distance_rows(v, z)[0] == 1.0
    assert np.all(normalize_rows(np.vstack([z, tiny])) == 0.0)
    d = cosine_distance_matrix(np.vstack([z, v]), np.vstack([v, -v]))
    assert np.all(np.isfinite(d)) and d[0, 0] == 1.0 and d[0, 1] == 1.0
    assert d[1, 0] == pytest.approx(0.0) and d[1, 1] == pytest.approx(2.0)
    with pytest.raises(NumericalError):
        cosine_distance_rows(np.full((1, 15), np.nan), v)


def test_identical_to_more_formulas_for_normal_vectors():
    rng = np.random.default_rng(0)
    u = rng.normal(size=(200, 15))
    v = rng.normal(size=(200, 15))
    ours = cosine_distance_rows(u, v)
    ref = np.array([more_cos_dist(a, b) for a, b in zip(u, v)])
    assert np.max(np.abs(ours - ref)) < 1e-14
    mat = cosine_distance_matrix(u[:10], v[:4])
    for i in range(10):
        for j in range(4):
            assert mat[i, j] == cosine_distance_rows(u[i:i + 1], v[j:j + 1])[0]   # bit-identical paths
    for n in (5, 4, 7):                        # odd and even counts (NumPy median convention)
        pts = rng.normal(size=(n, 15))
        c, n_excl, prenorm, deg = class_centroid_more(pts)
        assert np.max(np.abs(c - more_find_center(pts))) < 1e-15
        assert n_excl == 0 and not deg and prenorm > 0


def test_centroid_excludes_and_counts_degenerate_samples():
    rng = np.random.default_rng(1)
    pts = rng.normal(size=(5, 15))
    pts[2] = 0.0
    pts[4] = 1e-13
    c, n_excl, _, deg = class_centroid_more(pts)
    assert n_excl == 2 and not deg
    assert np.max(np.abs(c - more_find_center(pts[[0, 1, 3]]))) < 1e-15
    c, n_excl, prenorm, deg = class_centroid_more(np.zeros((5, 15)))
    assert deg and n_excl == 5 and prenorm == 0.0 and np.all(c == 0.0)
    a = np.zeros(15)
    a[0] = 1.0
    c, _, prenorm, deg = class_centroid_more(np.vstack([a, -a]))    # median exactly zero
    assert deg and prenorm == 0.0


def test_argmin_tie_breaks_to_first_class():
    d = np.array([[0.3, 0.1, 0.1], [0.2, 0.2, 0.5]])
    assert argmin_class(d).tolist() == [1, 0]


def test_min_separation_and_simplex_bound():
    labels = np.eye(3)[:, :3]
    ms, i, j = min_separation(np.hstack([labels, np.zeros((3, 12))]))
    assert ms == pytest.approx(1.0) and (i, j) == (0, 1)
    assert C.simplex_bound(3) == pytest.approx(1.5) and C.simplex_bound(10) == pytest.approx(1 + 1 / 9)


def test_yodd_fraction_and_activity():
    rng = np.random.default_rng(2)
    out = rng.normal(size=(30, 15))
    out[:, list(C.YODD_INDICES_15)] = 0.0
    frac, n_excl = yodd_norm_fraction(out, C.YODD_INDICES_15)
    assert frac == 0.0 and n_excl == 0
    out2 = rng.normal(size=(30, 15))
    out2[0] = 0.0
    frac, n_excl = yodd_norm_fraction(out2, C.YODD_INDICES_15)
    assert 0.0 < frac < 1.0 and n_excl == 1
    act = observable_activity(out, list(C.OBSERVABLE_LABELS_15), C.YODD_INDICES_15, C.EPS_NORM, 1e-6)
    assert act["yodd_norm_fraction"] == 0.0 and act["n_active_primary"] == 9
    assert [s["n_active_yodd"] for s in act["threshold_sensitivity"]] == [0, 0, 0, 0, 0]
    assert [s["threshold"] for s in act["threshold_sensitivity"]] == list(C.ACTIVITY_THRESHOLDS)
    yodd = [p["observable"] for p in act["per_observable"] if p["is_yodd"]]
    assert yodd == ["IY", "XY", "YI", "YX", "YZ", "ZY"]
    for p in act["per_observable"]:
        assert 0.0 <= p["norm_fraction"] <= 1.0
