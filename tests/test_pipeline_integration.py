"""Integration tests on real MNIST with tiny sizes.

G0-01 no official-test access before FINAL_EVALUATION; G3-05 cache equivalence and
execution counts; §9.4 stage timing; G3-04 status/attempts; G1-07 ablation
controls; G2-05/G2-06 fields; G2-07 label = centroid at final_point_eval_id.
"""

import hashlib
import json

import numpy as np
import pytest

from core import constants as C
from core.artifact_io import load_json, read_jsonl
from core.circuits import BatchedStatevector
from core.clustering import pair_loss
from core.numerics import cosine_distance_rows
from tests.conftest import make_project, requires_mnist, tiny_config

pytestmark = requires_mnist


def _row_hashes(x):
    return {hashlib.sha1(np.ascontiguousarray(r).tobytes()).hexdigest() for r in np.atleast_2d(x)}


@pytest.fixture()
def proj(tmp_path):
    return make_project(tmp_path)


def test_no_official_test_access_before_final_evaluation(proj, mnist, monkeypatch):
    from core import pipeline
    from core.evaluation import OfficialTestAccessError, OfficialTestVault

    seen = []                                         # (stage, row hashes) for every encoded batch
    clock_holder = {}
    orig_encode = BatchedStatevector.encode
    orig_open = OfficialTestVault.open

    def spy_encode(self, x):
        seen.append((clock_holder["clock"].current_stage if "clock" in clock_holder else None, _row_hashes(x)))
        return orig_encode(self, x)

    def spy_open(self):
        clock_holder["clock"] = self.clock
        return orig_open(self)

    orig_init = pipeline.RunClock.__init__

    def spy_clock(self, *a, **k):
        orig_init(self, *a, **k)
        clock_holder["clock"] = self

    monkeypatch.setattr(BatchedStatevector, "encode", spy_encode)
    monkeypatch.setattr(OfficialTestVault, "open", spy_open)
    monkeypatch.setattr(pipeline.RunClock, "__init__", spy_clock)
    cfg = tiny_config(proj, "MORE-HD-C", "PCA", 3)
    pipeline.run_condition(cfg, mnist=mnist, write_workbook=False)
    run_dir = cfg.run_dir
    test_rows = _row_hashes(np.load(run_dir / "artifacts" / "X_test_scaled.npy"))
    before_final = [h for stage, h in seen if stage != "final_evaluation"]
    assert before_final and all(not (h & test_rows) for h in before_final)
    assert any(stage == "final_evaluation" and (h & test_rows) for stage, h in seen)
    vault = OfficialTestVault(run_dir / "artifacts", {}, clock_holder["clock"])
    with pytest.raises(OfficialTestAccessError):
        vault.open()


def test_cache_equivalence_and_execution_counts(proj, mnist):
    from core.pipeline import run_condition

    for arch, k in (("MORE-HD", 3), ("MORE-HD-C", 10), ("MORE-HD-60P", 3), ("MORE-HD-C-FixedRZ", 10), ("MORE-REPRO", 3)):
        cfg = tiny_config(proj, arch, "ZERNIKE", k)
        m = run_condition(cfg, mnist=mnist, write_workbook=False)
        run_dir = cfg.run_dir
        n_val = cfg.n_val_per_class * k
        assert m["circuit_executions"]["clustering_per_eval"] == 5 * k + n_val
        assert m["circuit_executions"]["supervised_per_eval"] == cfg.n_train_per_class * k + n_val
        # recompute train_loss at the final point: cached outputs vs one engine call per pair
        from core.circuits import build_initialization_bundle, build_model

        model = build_model(arch, build_initialization_bundle(42))
        eng = BatchedStatevector(model)
        theta = np.load(run_dir / "artifacts" / "clustering_params_final.npy")
        x = np.load(run_dir / "artifacts" / "X_train_scaled.npy")
        pm = load_json(run_dir / "artifacts" / "pair_manifest.json")
        s = np.load(run_dir / "artifacts" / "correlation_matrix.npy")
        pos = [p for c in range(k) for p in pm["selected_by_class"][str(c)]["train_positions"]]
        cls = [c for c in range(k) for _ in range(5)]
        a, b = np.triu_indices(len(pos), 1)
        pair_s = s[np.array(cls)[a], np.array(cls)[b]]
        u = eng.unitary_columns(theta)
        cached = eng.expectations(eng.encode(x[pos]), u)
        loss_cached, _ = pair_loss(cached, a, b, pair_s, C.EPS_NORM)
        if k == 3:                         # literal no-cache version: both endpoints re-simulated per pair
            per_pair = []
            for i, j in zip(a, b):
                vi = eng.expectations(eng.encode(x[[pos[i]]]), u)
                vj = eng.expectations(eng.encode(x[[pos[j]]]), u)
                per_pair.append(-pair_s[len(per_pair)] * cosine_distance_rows(vi, vj)[0])
        else:                              # K=10: one isolated simulation per sample, then per-pair loss
            single = np.vstack([eng.expectations(eng.encode(x[[p]]), u) for p in pos])
            assert np.array_equal(single, cached)
            per_pair = [-pair_s[n] * cosine_distance_rows(single[[i]], single[[j]])[0] for n, (i, j) in enumerate(zip(a, b))]
        assert loss_cached == float(np.mean(np.array(per_pair)))       # exact equality (G3-05)
        logs = read_jsonl(run_dir / "logs" / "clustering_log.jsonl")
        opt = load_json(run_dir / "logs" / "clustering_optimizer_result.json")
        assert logs[opt["final_point_eval_id"]]["train_loss"] == loss_cached
        # G2-07: quantum labels equal the centroids logged at final_point_eval_id
        ql = load_json(run_dir / "artifacts" / "quantum_labels.json")
        from core.numerics import min_separation

        labels = np.array([[ql["labels"][str(c)][o] for o in ql["observables"]] for c in range(k)])
        ms, _, _ = min_separation(labels)
        assert ms == logs[opt["final_point_eval_id"]]["min_separation"]
        assert m["optimizer_results"]["clustering"]["final_point_eval_id"] == opt["final_point_eval_id"]


def test_stage_timing_status_and_attempt_policy(proj, mnist, monkeypatch):
    from core import clustering, pipeline
    from core.run_status import read_status

    cfg = tiny_config(proj, "MORE-HD", "HU", 3)
    m = pipeline.run_condition(cfg, mnist=mnist, write_workbook=False)
    stages = read_jsonl(cfg.run_dir / "logs" / "stage_timing.jsonl")
    assert [s["stage"] for s in stages] == list(C.STAGES_JALUR_A)
    assert sum(s["wall_sec"] for s in stages) <= m["total_runtime_sec"]
    for loop in ("clustering", "supervised"):
        rows = read_jsonl(cfg.run_dir / "logs" / f"{loop}_log.jsonl")
        wall = next(s["wall_sec"] for s in stages if s["stage"] == f"{loop}_loop")
        assert sum(r["eval_runtime_sec"] for r in rows) <= wall
    assert read_status(cfg.run_dir)["status"] == C.STATUS_COMPLETED
    # a COMPLETED run cannot get a new attempt
    with pytest.raises(RuntimeError):
        pipeline.run_condition(tiny_config(proj, "MORE-HD", "HU", 3, attempt=2), mnist=mnist, write_workbook=False)

    # forced failure in the middle of clustering_loop
    def boom(*a, **k):
        raise RuntimeError("forced failure")

    monkeypatch.setattr(pipeline, "clustering_loop", boom)
    bad = tiny_config(proj, "MORE-HD-C", "HU", 3)
    with pytest.raises(RuntimeError):
        pipeline.run_condition(bad, mnist=mnist, write_workbook=False)
    st = read_status(bad.run_dir)
    assert st["status"] == C.STATUS_FAILED and "forced failure" in st["traceback"]
    assert (bad.run_dir / "logs" / "run_started.json").exists()
    assert [s["stage"] for s in read_jsonl(bad.run_dir / "logs" / "stage_timing.jsonl")] == ["data_pipeline", "setup"]
    monkeypatch.setattr(pipeline, "clustering_loop", clustering.clustering_loop)
    with pytest.raises(ValueError):                                 # attempt numbers are consecutive
        pipeline.run_condition(tiny_config(proj, "MORE-HD-C", "HU", 3, attempt=3), mnist=mnist, write_workbook=False)
    retry = tiny_config(proj, "MORE-HD-C", "HU", 3, attempt=2)
    pipeline.run_condition(retry, mnist=mnist, write_workbook=False)
    assert retry.run_dir.name.endswith("_attempt2") and read_status(retry.run_dir)["status"] == C.STATUS_COMPLETED
    assert read_status(bad.run_dir)["status"] == C.STATUS_FAILED                # old attempt untouched


def test_ablation_and_reference_manifests(proj, mnist):
    from core.pipeline import run_condition

    a = run_condition(tiny_config(proj, "MORE-HD", "PCA", 3), mnist=mnist, write_workbook=False)
    c = run_condition(tiny_config(proj, "MORE-HD-C-FixedRZ", "PCA", 3), mnist=mnist, write_workbook=False)
    b = run_condition(tiny_config(proj, "MORE-HD-60P", "PCA", 3), mnist=mnist, write_workbook=False)
    mref = run_condition(tiny_config(proj, "MORE-REPRO", "PCA", 3), mnist=mnist, write_workbook=False)
    for m in (b, c, mref):
        assert m["paired_input_check"]["status"] == "MATCH"
        assert m["paired_pair_manifest_sha256"] == a["paired_pair_manifest_sha256"]
        assert m["split_manifest_sha256"] == a["split_manifest_sha256"]
        assert m["matching_run_uid_A"] == "R001-S42" and m["matching_run_uid_D"] == "R002-S42"
    assert c["run_uid"] == "ABL002-S42" and b["run_uid"] == "ABL001-S42" and mref["run_uid"] == "MREF001-S42"
    assert c["n_trainable_params"] == 30 and c["n_fixed_rz_params"] == 30
    assert b["n_trainable_params"] == 60 and b["n_variational_layers"] == 6 and b["cnot_count"] == 66
    init = load_json(proj / "runs" / c["run_name"] / "artifacts" / "initialization_manifest.json")
    fixed = np.load(proj / "runs" / c["run_name"] / "artifacts" / "fixed_rz.npy")
    assert np.all(fixed != 0) and init["fixed_rz_sha256"] == init["rz_phase_sha256"]
    final = np.load(proj / "runs" / c["run_name"] / "artifacts" / "supervised_params_final.npy")
    assert final.shape == (30,)                                    # RZ never enters the optimizer
    sd = load_json(proj / "runs" / a["run_name"] / "artifacts" / "structural_diagnostics.json")
    for ck in sd["checkpoints"].values():
        assert ck["yodd_norm_fraction"] < 1e-20 and all(s["n_active_yodd"] == 0 for s in ck["threshold_sensitivity"])
    sd_c = load_json(proj / "runs" / c["run_name"] / "artifacts" / "structural_diagnostics.json")
    assert sd_c["checkpoints"]["clustering_final"]["yodd_norm_fraction"] > 1e-6
    row = read_jsonl(proj / "runs" / a["run_name"] / "logs" / "clustering_log.jsonl")[-1]
    for key in ("min_separation_ratio", "min_separation_val_ratio", "yodd_norm_fraction", "n_degenerate_centroids_train",
                "closest_class_i", "eval_cpu_sec"):
        assert key in row
    assert json.dumps(a)                                           # manifest is strict JSON
