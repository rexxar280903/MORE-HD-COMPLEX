from pathlib import Path

import numpy as np
import pytest

from core.madbase import load_madbase, madbase_exists


ROOT = Path(__file__).resolve().parents[1]
MADBASE_ROOT = ROOT / "research_arabic_digits" / "research_data"


@pytest.mark.skipif(not madbase_exists(MADBASE_ROOT), reason="MADBase not extracted")
def test_official_madbase_snapshot_and_mapping():
    data = load_madbase(MADBASE_ROOT)
    assert data.train_images.shape == (60_000, 28, 28)
    assert data.test_images.shape == (10_000, 28, 28)
    assert data.train_images.dtype == np.uint8
    assert data.test_images.dtype == np.uint8
    assert np.bincount(data.train_labels, minlength=10).tolist() == [6_000] * 10
    assert np.bincount(data.test_labels, minlength=10).tolist() == [1_000] * 10
    assert set(data.train_writer_ids).isdisjoint(set(data.test_writer_ids))
    assert data.train_sample_ids[0] == "train/writer001_pass01_digit0"
    assert data.test_sample_ids[0] == "test/writer601_pass01_digit0"
    # The canonical array is inverted from the white-background source.
    assert int(data.train_images[0, 0, 0]) == 0
    assert int(data.train_images.max()) == 255
    assert data.test_origin == "OFFICIAL_SOURCE"
    assert len(data.snapshot_sha256) == 64
