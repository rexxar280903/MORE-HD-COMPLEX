import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from core.mnist import mnist_exists  # noqa: E402

HAS_MNIST = mnist_exists(ROOT / "data")
requires_mnist = pytest.mark.skipif(not HAS_MNIST, reason="MNIST not downloaded (python scripts/download_mnist.py)")


@pytest.fixture(scope="session")
def mnist():
    if not HAS_MNIST:
        pytest.skip("MNIST not downloaded")
    from core.mnist import load_mnist

    return load_mnist(ROOT / "data")


def make_project(tmp_path: Path) -> Path:
    """Temporary project root with links to the shared MNIST data and workbook templates."""
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "data").symlink_to(ROOT / "data")
    (proj / "research_data").symlink_to(ROOT / "research_data")
    return proj


def tiny_config(proj, architecture="MORE-HD", feature="PCA", k=3, nfev=None, **kw):
    from core.config import RunConfig

    n_params = {"MORE-HD": 30, "MORE-HD-C": 60, "MORE-HD-60P": 60, "MORE-HD-C-FixedRZ": 30, "MORE-REPRO": 91}[architecture]
    nfev = nfev or n_params + 12
    base = dict(architecture=architecture, feature_method=feature, k=k, seed=42, run_mode="PILOT",
                max_nfev_clustering=nfev, max_nfev_supervised=nfev, n_train_per_class=20, n_val_per_class=6,
                n_test_per_class=8, project_root=str(proj))
    base.update(kw)
    return RunConfig(**base)
