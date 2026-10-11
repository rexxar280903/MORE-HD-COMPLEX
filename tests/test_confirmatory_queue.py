import pytest

from core.artifact_io import save_json_atomic
from core.config import RunConfig
from scripts.run_confirmatory_queue import main_command


def task(root):
    cfg = RunConfig("MORE-HD", "PCA", 3, 101, dataset="MADBASE", run_mode="CONFIRMATORY",
                    n_parallel_declared=8, project_root=str(root))
    return dict(cfg=cfg, key=cfg.run_uid, script="main_train.py", flag="--architecture MORE-HD")


def status(directory, uid, attempt, state):
    save_json_atomic(dict(run_uid=uid, attempt=attempt, status=state), directory / "logs/run_status.json")


def test_completed_model_repairs_excel_without_retraining(tmp_path):
    t = task(tmp_path)
    status(t["cfg"].run_dir, t["key"], 1, "COMPLETED")
    cmd, directory = main_command(t)
    assert "export" in cmd and "main_train.py" not in " ".join(cmd)
    save_json_atomic(dict(run_uid=t["key"]), directory / "workbook_export_complete.json")
    assert main_command(t)[0] is None


def test_live_attempt_cannot_be_started_twice(tmp_path):
    t = task(tmp_path)
    status(t["cfg"].run_dir, t["key"], 1, "RUNNING")
    with pytest.raises(RuntimeError, match="unfinished attempt"):
        main_command(t)


def test_failed_attempt_gets_next_attempt_and_preserves_folder(tmp_path):
    t = task(tmp_path)
    old = t["cfg"].run_dir
    status(old, t["key"], 1, "FAILED")
    cmd, directory = main_command(t)
    assert cmd[cmd.index("--attempt") + 1] == "2"
    assert directory.name.endswith("_attempt2")
    assert old.exists()


def test_duplicate_completed_attempts_stop_queue(tmp_path):
    t = task(tmp_path)
    base = t["cfg"].run_dir
    status(base, t["key"], 1, "COMPLETED")
    status(base.with_name(base.name + "_attempt2"), t["key"], 2, "COMPLETED")
    with pytest.raises(RuntimeError, match="Duplicate completed"):
        main_command(t)


def test_live_parallel_control_can_reduce_limit(tmp_path, monkeypatch):
    import scripts.run_confirmatory_queue as queue
    monkeypatch.setattr(queue, "QUEUE", tmp_path)
    assert queue.parallel_limit(8) == 8
    save_json_atomic({"parallel": 6}, tmp_path / "control.json")
    assert queue.parallel_limit(8) == 6
    save_json_atomic({"parallel": 0}, tmp_path / "control.json")
    with pytest.raises(ValueError, match="parallel limit"):
        queue.parallel_limit(8)
