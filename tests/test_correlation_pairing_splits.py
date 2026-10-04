"""G4-04 (correlation matrix), G1-03/G1-08 (pairs), G0-03/G0-04/G4-02 (seeds and splits)."""

from math import comb

import numpy as np
import pytest

from core import constants as C
from core.correlation import correlation_matrix
from core.pairing import build_pairing
from core.seeds import derive_subseed, manifest_seeds
from tests.conftest import requires_mnist


def more_calc_class_rela(x, y, classes, n=100):
    """MORE data_helper.py::calc_class_rela without the 3-decimal rounding."""
    imgs = {c: x[y == c][:n].mean(axis=0) for c in classes}
    k = len(classes)
    mse = np.zeros((k, k))
    for i, a in enumerate(classes):
        for j, b in enumerate(classes):
            mse[i, j] = np.sum((imgs[a] - imgs[b]) ** 2) / float(imgs[a].shape[0])
    s = (mse - mse.min()) / mse.max() - mse.min()
    np.fill_diagonal(s, -1.0)
    return s, mse


def test_correlation_matrix_follows_more():
    rng = np.random.default_rng(0)
    classes = [0, 1, 2, 3, 4]
    y = np.repeat(classes, 150)
    x = rng.normal(size=(750, 8)) + y[:, None] * rng.uniform(0.1, 0.5, 8)
    res = correlation_matrix(x, y, classes)
    s_ref, mse_ref = more_calc_class_rela(x, y, classes)
    assert np.max(np.abs(res["S"] - s_ref)) < 1e-12
    assert np.max(np.abs(res["raw_mse"] - mse_ref)) < 1e-12
    s = res["S"]
    off = ~np.eye(5, dtype=bool)
    assert np.array_equal(s, s.T) and np.all(np.diag(s) == -1.0)
    assert np.all((s[off] > 0) & (s[off] <= 1.0)) and np.isclose(s[off].max(), 1.0)
    assert res["n_mean_samples"] == {c: 100 for c in classes}


def test_correlation_noncontiguous_labels_and_constant_fallback():
    rng = np.random.default_rng(1)
    classes = [2, 5, 7]
    y = np.repeat(classes, 20)
    x = rng.normal(size=(60, 8))
    res = correlation_matrix(x, y, classes)
    assert res["class_to_index"] == {2: 0, 5: 1, 7: 2}
    same = np.ones((60, 8))
    res = correlation_matrix(same, y, classes)
    assert res["fallback_all_offdiag_one"]
    assert np.all(res["S"][~np.eye(3, dtype=bool)] == 1.0) and np.all(np.isfinite(res["S"]))


@pytest.mark.parametrize("k", range(3, 11))
def test_pair_counts_and_no_self_or_duplicates(k):
    y = np.repeat(np.arange(k), 30)
    idx = np.arange(y.size) + 1000
    p = build_pairing(y, idx, list(range(k)), 101)
    st = p["stats"]
    assert st["n_pairs_total"] == comb(5 * k, 2)
    assert st["n_pairs_same_class"] == 10 * k
    assert st["n_pairs_different_class"] == 25 * comb(k, 2)
    assert st["self_pairs"] == 0 and st["duplicate_pairs"] == 0
    assert np.all(p["pair_a"] < p["pair_b"])
    keys = set(zip(p["pair_a"].tolist(), p["pair_b"].tolist()))
    assert len(keys) == st["n_pairs_total"]
    assert st["same_class_ratio"] == pytest.approx(10 * k / comb(5 * k, 2))


def test_pairing_deterministic_nested_and_seed_dependent():
    y = np.repeat(np.arange(10), 30)
    idx = np.arange(300)
    p3 = build_pairing(y[:90], idx[:90], [0, 1, 2], 101)
    p10 = build_pairing(y, idx, list(range(10)), 101)
    again = build_pairing(y[:90], idx[:90], [0, 1, 2], 101)
    other = build_pairing(y[:90], idx[:90], [0, 1, 2], 202)
    for c in ("0", "1", "2"):
        sel = p3["manifest"]["selected_by_class"][c]
        assert sel == p10["manifest"]["selected_by_class"][c]           # nested-K
        assert sel == again["manifest"]["selected_by_class"][c]         # deterministic
    assert p3["manifest"]["selected_by_class"] != other["manifest"]["selected_by_class"]


def test_subseeds_match_workbook_prefill():
    s = manifest_seeds(101)
    assert (s["data_seed"], s["ry_core_seed"], s["rz_phase_seed"], s["pair_seed"], s["label_seed"]) == (
        2188677521, 339717531, 2683203510, 2679144630, 2661278565)
    assert derive_subseed(42, "data") == derive_subseed(42, "data")


@requires_mnist
def test_split_manifest_rules(mnist, tmp_path):
    from core.splits import build_split_manifest, create_or_load_split_manifest, indices_for_classes, validate_split_manifest

    m1 = build_split_manifest(101, mnist.train_labels, mnist.test_labels)
    m2, path = create_or_load_split_manifest(101, mnist.train_labels, mnist.test_labels, tmp_path)
    assert m1["train_idx_by_class"] == m2["train_idx_by_class"]
    validate_split_manifest(m2, 101, mnist.train_labels, mnist.test_labels)
    for d in range(10):
        tr, va, te = (m1[k][str(d)] for k in ("train_idx_by_class", "val_idx_by_class", "test_idx_by_class"))
        assert (len(tr), len(va), len(te)) == (C.N_TRAIN_PER_CLASS, C.N_VAL_PER_CLASS, C.N_TEST_PER_CLASS)
        assert not set(tr) & set(va)
        assert np.all(mnist.train_labels[tr] == d) and np.all(mnist.test_labels[te] == d)
    k3 = indices_for_classes(m1, [0, 1, 2], "train")
    k10 = indices_for_classes(m1, list(range(10)), "train")
    assert np.array_equal(k10[: k3.size], k3)                              # nested-K
    other = build_split_manifest(202, mnist.train_labels, mnist.test_labels)
    assert other["train_idx_by_class"] != m1["train_idx_by_class"]
    m2["train_idx_by_class"]["0"][0] = m2["val_idx_by_class"]["0"][0]
    with pytest.raises(ValueError):
        validate_split_manifest(m2, 101, mnist.train_labels, mnist.test_labels)
