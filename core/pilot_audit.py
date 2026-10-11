"""Admit completed pilot attempts before checks or budget freezing."""
from pathlib import Path

from . import constants as C
from .artifact_io import load_json


def completed_pilots(root, dataset, tag):
    admitted, excluded, identities = [], [], set()
    for path in sorted((Path(root) / C.RUNS_DIR).rglob("config.json")):
        cfg = load_json(path)
        if (cfg.get("run_mode") != "PILOT" or cfg.get("pilot_tag") != tag or
                cfg.get("track") == "JALUR_B" or cfg.get("dataset", C.DATASET_MNIST) != dataset):
            continue
        status_path = path.parent / "logs" / "run_status.json"
        status = load_json(status_path) if status_path.exists() else {}
        state = status.get("status", "MISSING")
        if state == "FAILED":
            excluded.append({"path": str(path.parent), "status": state,
                             "run_uid": cfg["run_uid"], "attempt": cfg.get("attempt", 1)})
            continue
        if state != "COMPLETED":
            raise ValueError(f"Pilot is not completed: {path.parent} ({state})")
        if status.get("run_uid") != cfg["run_uid"] or status.get("attempt") != cfg.get("attempt", 1):
            raise ValueError(f"Pilot status/config identity mismatch: {path.parent}")
        if cfg["run_uid"] in identities:
            raise ValueError(f"Duplicate completed pilot: {cfg['run_uid']}")
        identities.add(cfg["run_uid"])
        admitted.append((path.parent, cfg))

    if not admitted:
        raise ValueError("No completed pilot runs found")
    # Early failures can precede the final config.json; retain their provenance.
    for path in sorted((Path(root) / C.RUNS_DIR).rglob("run_status.json")):
        directory = path.parent.parent
        if (directory / "config.json").exists():
            continue
        status = load_json(path)
        if status.get("run_uid") in identities:
            if status.get("status") != "FAILED":
                raise ValueError(f"Unfinished attempt without config: {directory}")
            excluded.append({"path": str(directory), "status": "FAILED",
                             "run_uid": status["run_uid"], "attempt": status["attempt"]})
    if dataset == C.DATASET_MADBASE and tag == "convergence":
        from scripts.run_pilot import plan

        expected = {(arch, feature, k) for arch, feature, k, _ in plan(tag)}
        actual = {(cfg["architecture"], cfg["feature_method"], cfg["n_classes"]) for _, cfg in admitted}
        if actual != expected or len(admitted) != len(expected):
            raise ValueError(f"Incomplete convergence matrix: missing={expected - actual}, extra={actual - expected}")
        for directory, cfg in admitted:
            if (cfg["seed"] != C.PILOT_SEED or
                    (cfg["n_train_per_class"], cfg["n_val_per_class"], cfg["n_test_per_class"]) !=
                    (C.N_TRAIN_PER_CLASS, C.N_VAL_PER_CLASS, C.N_TEST_PER_CLASS) or
                    cfg["max_nfev_clustering"] != C.PILOT_MAX_NFEV_CAP or
                    cfg["max_nfev_supervised"] != C.PILOT_MAX_NFEV_CAP):
                raise ValueError(f"Convergence protocol mismatch: {directory}")
    return admitted, excluded
