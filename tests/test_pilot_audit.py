import json

import pytest

from core.pilot_audit import completed_pilots


def write_attempt(root, folder, state, attempt=1, uid="R001-S42"):
    directory = root / "runs" / folder
    (directory / "logs").mkdir(parents=True)
    cfg = dict(run_mode="PILOT", pilot_tag="convergence", track="PRIMARY",
               dataset="MNIST", run_uid=uid, attempt=attempt)
    (directory / "config.json").write_text(json.dumps(cfg))
    (directory / "logs/run_status.json").write_text(json.dumps(
        dict(status=state, run_uid=uid, attempt=attempt)))
    return directory


def test_failed_attempt_is_excluded_and_completed_retry_admitted(tmp_path):
    write_attempt(tmp_path, "failed", "FAILED")
    retry = write_attempt(tmp_path, "retry", "COMPLETED", attempt=2)
    admitted, excluded = completed_pilots(tmp_path, "MNIST", "convergence")
    assert [d for d, _ in admitted] == [retry]
    assert excluded[0]["attempt"] == 1


def test_duplicate_completed_attempt_is_rejected(tmp_path):
    write_attempt(tmp_path, "one", "COMPLETED")
    write_attempt(tmp_path, "two", "COMPLETED", attempt=2)
    with pytest.raises(ValueError, match="Duplicate completed"):
        completed_pilots(tmp_path, "MNIST", "convergence")


def test_running_pilot_prevents_budget_freeze(tmp_path):
    write_attempt(tmp_path, "one", "RUNNING")
    with pytest.raises(ValueError, match="not completed"):
        completed_pilots(tmp_path, "MNIST", "convergence")


def test_madbase_requires_complete_convergence_matrix(tmp_path):
    directory = write_attempt(tmp_path, "one", "COMPLETED")
    path = directory / "config.json"
    cfg = json.loads(path.read_text())
    cfg.update(dataset="MADBASE", architecture="MORE-HD", feature_method="PCA", n_classes=3)
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="Incomplete convergence matrix"):
        completed_pilots(tmp_path, "MADBASE", "convergence")


def test_failed_attempt_before_config_is_kept_in_audit(tmp_path):
    failed = write_attempt(tmp_path, "failed", "FAILED")
    (failed / "config.json").unlink()
    write_attempt(tmp_path, "retry", "COMPLETED", attempt=2)
    admitted, excluded = completed_pilots(tmp_path, "MNIST", "convergence")
    assert len(admitted) == 1
    assert len(excluded) == 1
    assert excluded[0]["path"] == str(failed)
