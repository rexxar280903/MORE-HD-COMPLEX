"""Run configuration, identities and validation (pseudocode §0.2, §0.3, §1, §11.11.8)."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import constants as C


@dataclass
class RunConfig:
    architecture: str
    feature_method: str
    k: int
    seed: int
    run_mode: str = "PILOT"
    max_nfev_clustering: int = C.SMOKE_MAX_NFEV
    max_nfev_supervised: int = C.SMOKE_MAX_NFEV
    n_train_per_class: int = C.N_TRAIN_PER_CLASS
    n_val_per_class: int = C.N_VAL_PER_CLASS
    n_test_per_class: int = C.N_TEST_PER_CLASS
    attempt: int = 1
    n_parallel_declared: int | None = None
    project_root: str = "."
    mnist_download: bool = False
    pilot_tag: str = ""                       # free text for PILOT runs (e.g. "smoke", "convergence")
    # locked protocol values (validated, never set by users)
    n_data_qubits: int = C.N_DATA_QUBITS
    n_readout_qubits: int = C.N_READOUT_QUBITS
    n_input_channels: int = C.N_INPUT_CHANNELS
    cobyla_tol: float = C.COBYLA_TOL
    cobyla_rhobeg: float = C.COBYLA_RHOBEG
    n_cluster_pair_samples: int = C.N_CLUSTER_PAIR_SAMPLES
    pair_balance_policy: str = C.PAIR_BALANCE_POLICY
    pair_weighting: str = C.PAIR_WEIGHTING
    active_dim_threshold: float = C.ACTIVE_DIM_THRESHOLD
    eps_norm: float = C.EPS_NORM
    centroid_rule: str = C.CENTROID_RULE
    centroid_source: str = C.CENTROID_SOURCE
    simulation_mode: str = C.SIMULATION_MODE
    shots: object = C.SHOTS
    sim_dtype: str = C.SIM_DTYPE
    device_name: str = C.DEVICE_NAME
    cluster_output_cache: str = C.CLUSTER_OUTPUT_CACHE
    val_monitor_policy: str = C.VAL_MONITOR_POLICY
    n_val_monitor_per_class: int | None = None
    loss_adjuster_policy: str = C.LOSS_ADJUSTER_POLICY
    correlation_rule: str = C.CORRELATION_RULE
    corr_n_mean_samples_per_class: int = C.CORR_N_MEAN_SAMPLES_PER_CLASS
    pca_svd_solver: str = C.PCA_SVD_SOLVER
    extra: dict = field(default_factory=dict)

    # -- derived -------------------------------------------------------------
    @property
    def classes(self) -> list:
        return list(range(self.k))

    @property
    def track(self) -> str:
        return C.TRACK_OF_ARCHITECTURE[self.architecture]

    @property
    def model_code(self) -> str:
        return C.MODEL_CODE[self.architecture]

    @property
    def condition_id(self) -> str:
        if self.track == C.TRACK_PRIMARY:
            return primary_condition_id(self.k, self.feature_method, self.architecture)
        if self.track == C.TRACK_ABLATION:
            return ablation_condition_id(self.k, self.feature_method, self.architecture)
        return more_reference_condition_id(self.k, self.feature_method)

    @property
    def run_uid(self) -> str:
        return f"{self.condition_id}-S{self.seed}"

    @property
    def run_name(self) -> str:
        if self.track == C.TRACK_PRIMARY:
            base = auto_generate_run_name(self.classes, self.n_train_per_class, self.n_val_per_class,
                                          self.n_test_per_class, self.architecture, self.feature_method, self.seed)
        else:
            sub = "ablation" if self.track == C.TRACK_ABLATION else "more_reference"
            cls = "-".join(str(c) for c in self.classes)
            base = f"{sub}/{self.run_uid}_cls-{cls}_{self.feature_method}_{self.architecture}"
        return base if self.attempt == 1 else f"{base}_attempt{self.attempt}"

    @property
    def run_dir(self) -> Path:
        return Path(self.project_root) / C.RUNS_DIR / self.run_name

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update({
            "classes": self.classes,
            "n_classes": self.k,
            "track": self.track,
            "model_code": self.model_code,
            "condition_id": self.condition_id,
            "run_uid": self.run_uid,
            "run_name": self.run_name,
        })
        return d


def auto_generate_run_name(classes, n_train, n_val, n_test, architecture, feature_method, seed) -> str:
    """AUTO_GENERATE (§1): cls-0-1-2_ntrain1000_nval100_ntest200_PCA_MORE-HD_seed42."""
    return (
        "cls-" + "-".join(str(c) for c in classes)
        + f"_ntrain{n_train}_nval{n_val}_ntest{n_test}_{feature_method}_{architecture}_seed{seed}"
    )


def primary_condition_id(k: int, feature_method: str, architecture: str) -> str:
    """GENERATE_RUN_ID (§0.2): R001-R048."""
    if not 3 <= k <= 10:
        raise ValueError("K must be 3..10")
    feature_offset = {"PCA": 0, "HU": 2, "ZERNIKE": 4}[feature_method]
    arch_offset = {C.ARCH_MORE_HD: 0, C.ARCH_MORE_HD_C: 1}[architecture]
    return "R" + str((k - 3) * 6 + feature_offset + arch_offset + 1).zfill(3)


def ablation_condition_id(k: int, feature_method: str, model: str) -> str:
    """ABL001-ABL018 (§11.11.8): K 3->6->10, PCA->HU->ZERNIKE, 60P->FixedRZ."""
    k_pos = {3: 0, 6: 1, 10: 2}[k]
    f_pos = {"PCA": 0, "HU": 1, "ZERNIKE": 2}[feature_method]
    m_pos = {C.ARCH_MORE_HD_60P: 0, C.ARCH_MORE_HD_C_FIXED_RZ: 1}[model]
    return "ABL" + str(k_pos * 6 + f_pos * 2 + m_pos + 1).zfill(3)


def more_reference_condition_id(k: int, feature_method: str) -> str:
    """MREF001-MREF024 (decision 2026-10-04): K 3->10, PCA->HU->ZERNIKE."""
    if not 3 <= k <= 10:
        raise ValueError("K must be 3..10")
    f_pos = {"PCA": 0, "HU": 1, "ZERNIKE": 2}[feature_method]
    return "MREF" + str((k - 3) * 3 + f_pos + 1).zfill(3)


def validate_config(cfg: RunConfig) -> list:
    """VALIDATE_CONFIG; returns warnings, raises on protocol violations."""
    warnings = []
    if cfg.architecture not in C.ALL_ARCHITECTURES:
        raise ValueError(f"architecture tidak dikenali: {cfg.architecture}")
    if cfg.feature_method not in C.FEATURE_METHODS:
        raise ValueError("feature_method harus PCA, HU, atau ZERNIKE")
    if cfg.k not in C.K_VALUES:
        raise ValueError("K harus 3..10")
    if cfg.track == C.TRACK_ABLATION and cfg.k not in C.ABLATION_K_VALUES:
        raise ValueError("ablation hanya pada K={3,6,10} (G1-05)")
    if cfg.run_mode not in C.RUN_MODES:
        raise ValueError("run_mode harus PILOT atau CONFIRMATORY")
    if cfg.run_mode == "PILOT" and cfg.seed != C.PILOT_SEED:
        raise ValueError("PILOT hanya seed 42")
    if cfg.run_mode == "CONFIRMATORY" and cfg.seed not in C.CONFIRMATORY_SEEDS:
        raise ValueError("seed konfirmatori harus 101/202/303/404/505")
    locked = {
        "n_data_qubits": C.N_DATA_QUBITS, "n_readout_qubits": C.N_READOUT_QUBITS,
        "n_input_channels": C.N_INPUT_CHANNELS, "cobyla_tol": C.COBYLA_TOL, "cobyla_rhobeg": C.COBYLA_RHOBEG,
        "n_cluster_pair_samples": C.N_CLUSTER_PAIR_SAMPLES, "pair_balance_policy": C.PAIR_BALANCE_POLICY,
        "pair_weighting": C.PAIR_WEIGHTING, "active_dim_threshold": C.ACTIVE_DIM_THRESHOLD,
        "eps_norm": C.EPS_NORM, "centroid_rule": C.CENTROID_RULE, "centroid_source": C.CENTROID_SOURCE,
        "simulation_mode": C.SIMULATION_MODE, "shots": C.SHOTS, "sim_dtype": C.SIM_DTYPE,
        "device_name": C.DEVICE_NAME, "cluster_output_cache": C.CLUSTER_OUTPUT_CACHE,
        "loss_adjuster_policy": C.LOSS_ADJUSTER_POLICY, "correlation_rule": C.CORRELATION_RULE,
        "corr_n_mean_samples_per_class": C.CORR_N_MEAN_SAMPLES_PER_CLASS, "pca_svd_solver": C.PCA_SVD_SOLVER,
    }
    for key, value in locked.items():
        if getattr(cfg, key) != value:
            raise ValueError(f"{key}={getattr(cfg, key)!r} melanggar protokol terkunci ({value!r})")
    if cfg.run_mode == "CONFIRMATORY" and (
        cfg.n_train_per_class, cfg.n_val_per_class, cfg.n_test_per_class
    ) != (C.N_TRAIN_PER_CLASS, C.N_VAL_PER_CLASS, C.N_TEST_PER_CLASS):
        raise ValueError("ukuran data konfirmatori dikunci 1000/100/200 (G0-03)")
    if cfg.n_val_per_class <= 0:
        raise ValueError("validation wajib ada (G0-01)")
    if cfg.val_monitor_policy == "FULL_VAL":
        if cfg.n_val_monitor_per_class is not None:
            raise ValueError("n_val_monitor_per_class harus None pada FULL_VAL")
    elif cfg.val_monitor_policy == "STRATIFIED_FIXED_SUBSET":
        n = cfg.n_val_monitor_per_class
        if n is None or not 0 < n < cfg.n_val_per_class:
            raise ValueError("subset monitoring harus 0 < n < n_val_per_class (G3-05b)")
    else:
        raise ValueError("val_monitor_policy harus FULL_VAL atau STRATIFIED_FIXED_SUBSET")
    if cfg.n_parallel_declared is None or cfg.n_parallel_declared < 1:
        if cfg.run_mode == "CONFIRMATORY":
            raise ValueError("n_parallel_declared wajib diisi (>= 1) untuk run konfirmatori (§9.4)")
        warnings.append("n_parallel_declared kosong; hanya boleh untuk pilot")
    if cfg.attempt < 1:
        raise ValueError("attempt >= 1")
    for var in C.THREAD_ENV_VARS:
        if os.environ.get(var) != C.LOCKED_THREADS:
            msg = f"{var}={os.environ.get(var)!r}; G4-01 mengunci 1 thread per proses"
            if cfg.run_mode == "CONFIRMATORY":
                raise ValueError(msg)
            warnings.append(msg)
    return warnings
