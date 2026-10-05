"""G2-01 (Hu), G2-02 (Zernike), G2-03 (scaling without clipping, train-only fit)."""

import numpy as np
import pytest

from core.features import (
    MinMaxAngleScaler,
    extract_features,
    hu_moments,
    map_image_to_unit_disk,
    signed_log_transform,
    unit_disk,
    zernike_magnitudes,
)
from tests.conftest import requires_mnist


def _images(n=40, seed=0):
    rng = np.random.default_rng(seed)
    imgs = np.zeros((n, 28, 28))
    for i in range(n):          # random strokes inside the frame, digit-like support
        r0, c0 = rng.integers(4, 12, 2)
        imgs[i, r0:r0 + rng.integers(6, 14), c0:c0 + rng.integers(2, 10)] = rng.uniform(0.2, 1.0)
        imgs[i] += rng.uniform(0, 0.05, (28, 28)) * (rng.uniform(size=(28, 28)) > 0.9)
    return np.clip(imgs, 0, 1)


def test_hu_matches_opencv():
    cv2 = pytest.importorskip("cv2")
    imgs = _images()
    ours = hu_moments(imgs)
    ref = np.array([cv2.HuMoments(cv2.moments(im)).ravel() for im in imgs])
    assert ours.shape == (40, 7)
    assert np.allclose(ours, ref, rtol=1e-8, atol=1e-12)


def test_signed_log_and_padding_channel():
    hu = np.array([[1e-3, -2e-5, 0.0, 1e-20, -1e-25, 3e-10, -4e-8]])
    out, n_bad = signed_log_transform(hu)
    assert n_bad == 0 and np.all(np.isfinite(out))
    assert out[0, 0] == pytest.approx(-np.log10(1e-3 + 1e-30))
    assert out[0, 1] == pytest.approx(np.log10(2e-5 + 1e-30))
    assert out[0, 2] == 0.0
    bad, n_bad = signed_log_transform(np.array([[np.nan, np.inf, 1.0]]))
    assert n_bad == 2 and bad[0, 0] == 0.0 and bad[0, 1] == 0.0
    feats, counters = extract_features("HU", _images(5))
    assert feats.shape == (5, 8) and np.all(feats[:, 7] == 0.0) and np.all(np.isfinite(feats))
    assert counters["n_non_finite_hu"] == 0


def test_zernike_matches_mahotas_and_is_deterministic():
    mahotas = pytest.importorskip("mahotas")
    imgs = _images()
    ours, n_zero = zernike_magnitudes(imgs)
    assert ours.shape == (40, 8) and n_zero == 0
    order = [(n, l) for n in range(6) for l in range(n + 1) if (n - l) % 2 == 0]
    idx = [order.index(t) for t in [(2, 0), (2, 2), (3, 1), (3, 3), (4, 0), (4, 2), (5, 1), (5, 5)]]
    ref = np.array([mahotas.features.zernike_moments(im, 14, degree=5, cm=(13.5, 13.5))[idx] for im in imgs])
    assert np.max(np.abs(ours - ref) / np.maximum(ref, 1e-12)) < 1e-10
    again, _ = zernike_magnitudes(imgs)
    assert np.array_equal(ours, again)


def test_zernike_reference_images():
    disk = unit_disk()
    assert disk.radius == 14.0 and disk.center_row == 13.5 and disk.center_col == 13.5
    # rotation by 90 degrees about the geometric centre maps the pixel grid onto itself
    img = _images(1)[0]
    z0, _ = zernike_magnitudes(img)
    z90, _ = zernike_magnitudes(np.rot90(img))
    assert np.max(np.abs(z0 - z90)) < 1e-12
    # a radially symmetric image has (numerically) zero magnitude for m != 0
    yy, xx = np.mgrid[:28, :28]
    ring = (np.hypot(yy - 13.5, xx - 13.5) < 9).astype(float)
    z, _ = zernike_magnitudes(ring)
    m_nonzero = [1, 2, 3, 5, 6, 7]                 # (2,2),(3,1),(3,3),(4,2),(5,1),(5,5)
    assert np.max(z[0, m_nonzero]) < 1e-12 and z[0, 0] > 0.1
    masked = map_image_to_unit_disk(np.ones((28, 28)))
    assert masked[0, 0] == 0.0 and masked[13, 13] == 1.0


def test_scaler_is_fit_on_train_only_without_clipping():
    rng = np.random.default_rng(1)
    train = rng.normal(size=(50, 8))
    val = rng.normal(scale=3.0, size=(20, 8))
    sc = MinMaxAngleScaler().fit(train)
    assert np.allclose(sc.data_min, train.min(axis=0)) and np.allclose(sc.data_max, train.max(axis=0))
    t = sc.transform(train)
    assert np.isclose(t.min(), 0.0) and np.isclose(t.max(), np.pi)
    v = sc.transform(val)
    assert v.min() < 0.0 or v.max() > np.pi            # out-of-range values preserved (no clipping)


@requires_mnist
@pytest.mark.parametrize("feature", ["PCA", "HU", "ZERNIKE"])
def test_data_pipeline_scaling_and_identity(feature, tmp_path, mnist):
    from core.data_pipeline import run_data_pipeline
    from core.splits import build_split_manifest
    from tests.conftest import tiny_config

    (tmp_path / "artifacts").mkdir()
    cfg = tiny_config(tmp_path, feature=feature, k=4)
    manifest = build_split_manifest(42, mnist.train_labels, mnist.test_labels)
    data = run_data_pipeline(cfg, mnist, manifest, tmp_path)
    x = data.x_train
    assert x.shape == (80, 8) and np.all(np.isfinite(x))
    if feature == "HU":
        assert np.all(x[:, 7] == 0.0) and np.all(data.x_val[:, 7] == 0.0)
        assert np.allclose(x[:, :7].min(axis=0), 0.0) and np.allclose(x[:, :7].max(axis=0), np.pi)
    else:
        assert np.allclose(x.min(axis=0), 0.0) and np.allclose(x.max(axis=0), np.pi)
    expected = [i for c in range(4) for i in manifest["train_idx_by_class"][str(c)][:20]]
    assert data.train_mnist_index.tolist() == expected
    assert not set(data.train_mnist_index.tolist()) & set(data.val_mnist_index.tolist())
    assert set(data.test_hashes) == {"X_test_scaled", "y_test", "test_mnist_index"}
