"""Feature extraction and scaling (pseudocode §2, §2.8; G2-01, G2-02, G2-03).

Hu and Zernike are implemented directly in NumPy (float64) so the formulas are
visible in the code; ``tests/test_features.py`` verifies them against the
reference libraries named in the pseudocode (OpenCV ``cv2.HuMoments`` and
``mahotas.features.zernike_moments``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

from .constants import (
    HU_PADDING_VALUE,
    HU_SIGNED_LOG_EPSILON,
    PCA_N_COMPONENTS,
    PCA_SVD_SOLVER,
    SCALE_RANGE,
    ZERNIKE_TERMS,
)

# ---------------------------------------------------------------------------
# Hu moments (G2-01)
# ---------------------------------------------------------------------------


def hu_moments(images: np.ndarray) -> np.ndarray:
    """HU_MOMENTS for a batch of grayscale images in [0, 1] -> (N, 7) float64.

    Same definitions as OpenCV ``cv2.moments`` + ``cv2.HuMoments``: x = column
    index, y = row index, pixel values used directly as mass (no binarisation),
    normalised central moments ``nu_pq = mu_pq / m00^((p+q)/2 + 1)``.
    """
    imgs = np.asarray(images, dtype=np.float64)
    if imgs.ndim == 2:
        imgs = imgs[None]
    n, h, w = imgs.shape
    y = np.arange(h, dtype=np.float64)
    x = np.arange(w, dtype=np.float64)
    m00 = imgs.sum(axis=(1, 2))
    m10 = np.einsum("nhw,w->n", imgs, x)
    m01 = np.einsum("nhw,h->n", imgs, y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xbar = m10 / m00
        ybar = m01 / m00
    dx = x[None, :] - xbar[:, None]          # (n, w)
    dy = y[None, :] - ybar[:, None]          # (n, h)

    def mu(p: int, q: int) -> np.ndarray:
        return np.einsum("nhw,nw,nh->n", imgs, dx ** p, dy ** q)

    with np.errstate(divide="ignore", invalid="ignore"):
        def nu(p: int, q: int) -> np.ndarray:
            return mu(p, q) / m00 ** ((p + q) / 2.0 + 1.0)

        n20, n02, n11 = nu(2, 0), nu(0, 2), nu(1, 1)
        n30, n03, n21, n12 = nu(3, 0), nu(0, 3), nu(2, 1), nu(1, 2)
    t0 = n30 + n12
    t1 = n21 + n03
    q0 = t0 * t0
    q1 = t1 * t1
    n4 = 4.0 * n11
    s = n20 + n02
    d = n20 - n02
    h1 = s
    h2 = d * d + n4 * n11
    h4 = q0 + q1
    h6 = d * (q0 - q1) + n4 * t0 * t1
    a = n30 - 3.0 * n12
    b = 3.0 * n21 - n03
    h3 = a * a + b * b
    h5 = a * t0 * (q0 - 3.0 * q1) + b * t1 * (3.0 * q0 - q1)
    h7 = b * t0 * (q0 - 3.0 * q1) - a * t1 * (3.0 * q0 - q1)
    return np.stack([h1, h2, h3, h4, h5, h6, h7], axis=1)


def signed_log_transform(hu: np.ndarray, epsilon: float = HU_SIGNED_LOG_EPSILON) -> tuple[np.ndarray, int]:
    """SIGNED_LOG_TRANSFORM: h' = -sign(h) * log10(|h| + epsilon); non-finite -> 0.0 (counted)."""
    hu = np.asarray(hu, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out = -np.sign(hu) * np.log10(np.abs(hu) + epsilon)
    bad = ~np.isfinite(out)
    n_bad = int(bad.sum())
    out = np.where(bad, 0.0, out)
    return out, n_bad


# ---------------------------------------------------------------------------
# Zernike moments (G2-02)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UnitDisk:
    """MAP_IMAGE_TO_UNIT_DISK geometry for an H x W image (identical for all MNIST images)."""

    height: int
    width: int
    center_row: float
    center_col: float
    radius: float
    inside: np.ndarray       # (H*W,) bool, rho <= 1
    rho: np.ndarray          # (H*W,) distance / radius, clipped below at 1e-9 (mahotas)
    cos_t: np.ndarray        # (H*W,) x / D
    sin_t: np.ndarray        # (H*W,) y / D


def unit_disk(height: int = 28, width: int = 28) -> UnitDisk:
    """Geometric centre ((W-1)/2, (H-1)/2), radius min(H, W)/2, no interpolation."""
    cr = (height - 1) / 2.0
    cc = (width - 1) / 2.0
    radius = min(height, width) / 2.0
    yy, xx = np.mgrid[:height, :width]
    yn = ((yy.astype(np.float64) - cr) / radius).ravel()
    xn = ((xx.astype(np.float64) - cc) / radius).ravel()
    rho = np.sqrt(xn * xn + yn * yn)
    rho = np.maximum(rho, 1e-9)
    inside = rho <= 1.0
    return UnitDisk(height, width, cr, cc, radius, inside, rho, xn / rho, yn / rho)


def map_image_to_unit_disk(img: np.ndarray, disk: UnitDisk | None = None) -> np.ndarray:
    """Return the image with pixels outside the unit disk set to 0 (pseudocode §2.8.4)."""
    disk = disk or unit_disk(*img.shape)
    flat = np.asarray(img, dtype=np.float64).ravel().copy()
    flat[~disk.inside] = 0.0
    return flat.reshape(img.shape)


def radial_polynomial(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """R_n^m(rho) for n - |m| even."""
    m = abs(m)
    out = np.zeros_like(rho)
    for s in range((n - m) // 2 + 1):
        coef = ((-1) ** s) * math.factorial(n - s) / (
            math.factorial(s) * math.factorial((n + m) // 2 - s) * math.factorial((n - m) // 2 - s)
        )
        out = out + coef * rho ** (n - 2 * s)
    return out


def zernike_basis(terms=ZERNIKE_TERMS, disk: UnitDisk | None = None) -> np.ndarray:
    """(n_inside, n_terms) complex matrix of (n+1)/pi * conj(V_nm) on in-disk pixels."""
    disk = disk or unit_disk()
    rho = disk.rho[disk.inside]
    ang = np.arctan2(disk.sin_t[disk.inside], disk.cos_t[disk.inside])
    cols = []
    for n, m in terms:
        if (n - abs(m)) % 2:
            raise ValueError(f"invalid Zernike term {(n, m)}")
        v = radial_polynomial(n, m, rho) * np.exp(-1j * m * ang)
        cols.append((n + 1) / math.pi * v)
    return np.stack(cols, axis=1)


def zernike_magnitudes(images: np.ndarray, terms=ZERNIKE_TERMS) -> tuple[np.ndarray, int]:
    """EXTRACT_ZERNIKE_TERMS for a batch -> (N, len(terms)) |Z_nm| and #images with zero disk mass.

    Mass normalisation follows mahotas.features.zernike_moments: the in-disk
    pixel values are divided by their sum before projection.
    """
    imgs = np.asarray(images, dtype=np.float64)
    if imgs.ndim == 2:
        imgs = imgs[None]
    disk = unit_disk(imgs.shape[1], imgs.shape[2])
    basis = zernike_basis(terms, disk)
    flat = imgs.reshape(imgs.shape[0], -1)[:, disk.inside]
    flat = np.where(flat > 0, flat, 0.0)
    mass = flat.sum(axis=1)
    zero_mass = mass <= 0
    safe = np.where(zero_mass, 1.0, mass)
    p = flat / safe[:, None]
    z = p @ basis
    mags = np.abs(z)
    mags[zero_mass] = 0.0
    return mags, int(zero_mass.sum())


# ---------------------------------------------------------------------------
# PCA and scaling (pseudocode §2.4-§2.5; G2-03)
# ---------------------------------------------------------------------------


def fit_pca(x_train_flat: np.ndarray, seed: int) -> PCA:
    """PCA fitted on train only; exact LAPACK SVD (svd_solver='full', decision 2026-10-04)."""
    pca = PCA(n_components=PCA_N_COMPONENTS, svd_solver=PCA_SVD_SOLVER, random_state=seed)
    pca.fit(x_train_flat)
    return pca


@dataclass
class MinMaxAngleScaler:
    """MinMaxScaler(feature_range=(0, pi)) fitted on train; no clipping (G2-03)."""

    data_min: np.ndarray | None = None
    data_max: np.ndarray | None = None
    feature_range: tuple = SCALE_RANGE

    def fit(self, x: np.ndarray) -> "MinMaxAngleScaler":
        x = np.asarray(x, dtype=np.float64)
        self.data_min = x.min(axis=0)
        self.data_max = x.max(axis=0)
        return self

    @property
    def scale(self) -> np.ndarray:
        span = self.data_max - self.data_min
        span = np.where(span == 0.0, 1.0, span)   # sklearn convention for constant features
        lo, hi = self.feature_range
        return (hi - lo) / span

    def transform(self, x: np.ndarray) -> np.ndarray:
        lo, _ = self.feature_range
        x = np.asarray(x, dtype=np.float64)
        return (x - self.data_min) * self.scale + lo

    def to_dict(self) -> dict:
        return {
            "feature_range": list(self.feature_range),
            "data_min": self.data_min.tolist(),
            "data_max": self.data_max.tolist(),
            "scale": self.scale.tolist(),
            "clip": False,
        }


def extract_features(method: str, images01: np.ndarray, pca: PCA | None = None) -> tuple[np.ndarray, dict]:
    """Raw (unscaled) 8-channel features for one split; PCA must already be fitted."""
    counters: dict = {}
    if method == "PCA":
        if pca is None:
            raise ValueError("PCA must be fitted on train first")
        feats = pca.transform(images01.reshape(images01.shape[0], -1))
    elif method == "HU":
        hu, n_bad = signed_log_transform(hu_moments(images01))
        counters["n_non_finite_hu"] = n_bad
        feats = np.concatenate([hu, np.full((hu.shape[0], 1), HU_PADDING_VALUE)], axis=1)
    elif method == "ZERNIKE":
        feats, n_zero = zernike_magnitudes(images01)
        counters["n_zero_mass_zernike"] = n_zero
    else:
        raise ValueError(f"unknown feature_method {method!r}")
    if feats.shape[1] != 8:
        raise AssertionError("feature interface must have exactly 8 channels")
    return np.asarray(feats, dtype=np.float64), counters
