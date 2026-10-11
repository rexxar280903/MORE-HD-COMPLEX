"""MADBase dataset-aware split, feature, vault, and identity regression tests."""

from pathlib import Path

import numpy as np
import pytest

from core import constants as C
from core.artifact_io import load_json
from core.config import RunConfig, validate_config
from core.data_pipeline import prepare_test_artifacts, run_data_pipeline
from core.evaluation import OfficialTestAccessError, OfficialTestVault
from core.madbase import load_madbase, madbase_exists
from core.splits import build_madbase_split_manifest, validate_madbase_split_manifest


ROOT = Path(__file__).resolve().parents[1]
MADBASE_ROOT = ROOT / C.MADBASE_ROOT
requires_madbase = pytest.mark.skipif(not madbase_exists(MADBASE_ROOT), reason="MADBase not extracted")


@pytest.fixture(scope="module")
def madbase():
    return load_madbase(MADBASE_ROOT)


def _cfg(tmp_path, feature="PCA", run_mode="PILOT"):
    return RunConfig(
        architecture=C.ARCH_MORE_HD,
        feature_method=feature,
        k=3,
        seed=42 if run_mode == "PILOT" else 101,
        dataset=C.DATASET_MADBASE,
        run_mode=run_mode,
        max_nfev_clustering=42,
        max_nfev_supervised=42,
        n_train_per_class=20,
        n_val_per_class=6,
        n_test_per_class=8,
        n_parallel_declared=1,
        project_root=str(tmp_path),
    )


@requires_madbase
def test_writer_disjoint_splits_and_fixed_confirmatory_test(madbase):
    pilot = build_madbase_split_manifest(42, madbase, "PILOT")
    validate_madbase_split_manifest(pilot, 42, madbase, "PILOT")
    assert pilot["test_index_space"] == "TRAIN_POOL"
    assert not (set(pilot["train_writer_ids"]) & set(pilot["validation_writer_ids"]))
    assert not (set(pilot["test_writer_ids"]) &
                (set(pilot["train_writer_ids"]) | set(pilot["validation_writer_ids"])))
    a = build_madbase_split_manifest(101, madbase, "CONFIRMATORY")
    b = build_madbase_split_manifest(202, madbase, "CONFIRMATORY")
    assert a["test_sample_ids_by_class"] == b["test_sample_ids_by_class"]
    assert a["test_origin"] == "OFFICIAL_SOURCE_FIXED_WRITER_SUBSET"


@requires_madbase
@pytest.mark.parametrize("feature", ["PCA", "HU", "ZERNIKE"])
def test_madbase_features_are_train_only_and_finite(tmp_path, madbase, feature):
    cfg = _cfg(tmp_path, feature)
    validate_config(cfg)
    manifest = build_madbase_split_manifest(42, madbase, "PILOT")
    run_dir = tmp_path / feature
    (run_dir / "artifacts").mkdir(parents=True)
    bundle = run_data_pipeline(cfg, madbase, manifest, run_dir)
    assert bundle.x_train.shape == (60, 8)
    assert bundle.x_val.shape == (18, 8)
    assert np.all(np.isfinite(bundle.x_train)) and np.all(np.isfinite(bundle.x_val))
    assert bundle.test_hashes == {}
    assert not (run_dir / "artifacts" / "X_test_scaled.npy").exists()
    hashes = prepare_test_artifacts(cfg, madbase, manifest, run_dir, bundle)
    assert set(hashes) == {"X_test_scaled", "y_test", "test_source_index", "test_sample_id"}
    meta = load_json(run_dir / "artifacts" / "feature_metadata.json")
    assert meta["test_preprocessing_stage"] == "final_evaluation"
    assert meta["test_origin"] == "CONSTRUCTED_PILOT_TRAIN_POOL"


class _Clock:
    current_stage = "setup"


@requires_madbase
def test_madbase_vault_blocks_early_access(tmp_path, madbase):
    cfg = _cfg(tmp_path)
    manifest = build_madbase_split_manifest(42, madbase, "PILOT")
    run_dir = tmp_path / "vault"
    (run_dir / "artifacts").mkdir(parents=True)
    bundle = run_data_pipeline(cfg, madbase, manifest, run_dir)
    prepare_test_artifacts(cfg, madbase, manifest, run_dir, bundle)
    clock = _Clock()
    vault = OfficialTestVault(run_dir / "artifacts", bundle.test_hashes, clock)
    with pytest.raises(OfficialTestAccessError):
        vault.open()
    clock.current_stage = "final_evaluation"
    x, y, idx, sid = vault.open()
    assert x.shape == (24, 8) and y.shape == idx.shape == sid.shape == (24,)


def test_dataset_mismatch_is_rejected():
    from core.datasets import assert_dataset_matches, load_dataset

    class FakeMNIST:
        pass

    with pytest.raises(ValueError, match="dataset mismatch"):
        assert_dataset_matches(C.DATASET_MADBASE, FakeMNIST())
    view = load_dataset(ROOT, C.DATASET_MADBASE)
    assert view.train_images.shape == (60_000, 28, 28)
    assert not hasattr(view, "test_images") and not hasattr(view, "test_labels")
