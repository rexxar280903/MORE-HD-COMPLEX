"""Frozen protocol constants (pseudocode §0.3, §1, §4.3.2, §11, §12)."""

from __future__ import annotations

import math

# --- seeds (G0-04) ---------------------------------------------------------
PILOT_SEED = 42
CONFIRMATORY_SEEDS = (101, 202, 303, 404, 505)
RUN_MODES = ("PILOT", "CONFIRMATORY")
SUBSEED_RULE = (
    'SHA256(str(master_seed) + ":" + namespace), first 8 hex -> uint32; '
    "numpy.random.default_rng (PCG64)"
)

# --- data protocol (G0-03) -------------------------------------------------
N_TRAIN_PER_CLASS = 1000
N_VAL_PER_CLASS = 100
N_TEST_PER_CLASS = 200
K_VALUES = (3, 4, 5, 6, 7, 8, 9, 10)
ABLATION_K_VALUES = (3, 6, 10)

# --- tracks and models ------------------------------------------------------
TRACK_PRIMARY = "PRIMARY"
TRACK_ABLATION = "ABLATION"
TRACK_MORE_REFERENCE = "MORE_REFERENCE"
TRACKS = (TRACK_PRIMARY, TRACK_ABLATION, TRACK_MORE_REFERENCE)

ARCH_MORE_HD = "MORE-HD"                 # A
ARCH_MORE_HD_60P = "MORE-HD-60P"         # B (ablation)
ARCH_MORE_HD_C_FIXED_RZ = "MORE-HD-C-FixedRZ"  # C (ablation)
ARCH_MORE_HD_C = "MORE-HD-C"             # D
ARCH_MORE_REPRO = "MORE-REPRO"           # M (MORE without loss adjuster R, reproduced)

PRIMARY_ARCHITECTURES = (ARCH_MORE_HD, ARCH_MORE_HD_C)
ABLATION_MODELS = (ARCH_MORE_HD_60P, ARCH_MORE_HD_C_FIXED_RZ)
REFERENCE_MODELS = (ARCH_MORE_REPRO,)
ALL_ARCHITECTURES = PRIMARY_ARCHITECTURES + ABLATION_MODELS + REFERENCE_MODELS
MODEL_CODE = {
    ARCH_MORE_HD: "A",
    ARCH_MORE_HD_60P: "B",
    ARCH_MORE_HD_C_FIXED_RZ: "C",
    ARCH_MORE_HD_C: "D",
    ARCH_MORE_REPRO: "M",
}
TRACK_OF_ARCHITECTURE = {
    ARCH_MORE_HD: TRACK_PRIMARY,
    ARCH_MORE_HD_C: TRACK_PRIMARY,
    ARCH_MORE_HD_60P: TRACK_ABLATION,
    ARCH_MORE_HD_C_FIXED_RZ: TRACK_ABLATION,
    ARCH_MORE_REPRO: TRACK_MORE_REFERENCE,
}

FEATURE_METHODS = ("PCA", "HU", "ZERNIKE")

# --- features (G2-01, G2-02, G2-03) -----------------------------------------
N_DATA_QUBITS = 8
N_READOUT_QUBITS = 2
N_INPUT_CHANNELS = 8
PCA_N_COMPONENTS = 8
PCA_SVD_SOLVER = "full"          # decision 2026-10-04 (exact SVD, see gate log)
HU_RAW_DIM = 7
HU_INPUT_MODE = "grayscale"
HU_PADDING_VALUE = 0.0
HU_SIGNED_LOG_EPSILON = 1e-30
ZERNIKE_TERMS = ((2, 0), (2, 2), (3, 1), (3, 3), (4, 0), (4, 2), (5, 1), (5, 5))
ZERNIKE_USE_MAGNITUDE = True
SCALE_RANGE = (0.0, math.pi)
CLIP_AFTER_SCALING = False

# --- correlation matrix (G4-04, decision 2026-10-04: follow MORE calc_class_rela)
CORR_N_MEAN_SAMPLES_PER_CLASS = 100
CORRELATION_RULE = "MORE_CALC_CLASS_RELA_MSE_OVER_MAX"

# --- optimizer (G1-01, G1-06) ------------------------------------------------
COBYLA_TOL = 1e-4
COBYLA_RHOBEG = 1.0
SMOKE_MAX_NFEV = 100             # revised 2026-10-04 (MORE-REPRO has 91 params)
PILOT_MAX_NFEV_CAP = 1000        # revised 2026-10-04 before any pilot run (see gate log)
# Final confirmatory budget (G1-01): set ONLY from the frozen plateau rule applied to the
# seed-42 convergence pilot (scripts/analyze_pilot_budget.py). None = confirmatory runs blocked.
FINAL_MAX_NFEV_CLUSTERING = None
FINAL_MAX_NFEV_SUPERVISED = None

# --- clustering pairs (G1-03, G1-08) ----------------------------------------
N_CLUSTER_PAIR_SAMPLES = 5
PAIR_BALANCE_POLICY = "NATURAL_FULL_PAIRING"
PAIR_WEIGHTING = "NONE"

# --- numerics (G2-04, G2-05, G2-07) ------------------------------------------
EPS_NORM = 1e-10
ACTIVE_DIM_THRESHOLD = 1e-6
ACTIVITY_THRESHOLDS = (1e-10, 1e-8, 1e-6, 1e-4, 1e-2)
CENTROID_RULE = "MORE_NORMALIZE_MEDIAN_NORMALIZE"
CENTROID_SOURCE = "CLUSTER_SAMPLES"

# --- simulation backend (G4-01) ------------------------------------------------
SIMULATION_MODE = "ANALYTIC_STATEVECTOR"
SHOTS = None
SIM_DTYPE = "complex128"
DEVICE_NAME = "core.engine.BatchedStatevector"   # locked 2026-10-04 (G4-01)
REFERENCE_DEVICE_NAME = "default.qubit"          # PennyLane cross-check
ENGINE_CHUNK_ROWS = 512
BACKEND_SELF_CHECK_TOL = 1e-10
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
LOCKED_THREADS = "1"

# --- G3-05 -------------------------------------------------------------------
CLUSTER_OUTPUT_CACHE = "PER_UNIQUE_SAMPLE_PER_EVAL"
VAL_MONITOR_POLICY = "FULL_VAL"

# --- loss adjuster (G1-10) ----------------------------------------------------
LOSS_ADJUSTER_POLICY = "NONE"

# --- observables (§4.3, §4.3.2) -----------------------------------------------
PAULI = ("I", "X", "Y", "Z")
OBSERVABLE_LABELS_15 = tuple(a + b for a in PAULI for b in PAULI if not (a == "I" and b == "I"))
OBSERVABLE_LABELS_MORE = ("X", "Y", "Z")


def yodd_indices(labels):
    """Indices of observables with an odd number of Y factors."""
    return tuple(i for i, lab in enumerate(labels) if lab.count("Y") % 2 == 1)


YODD_INDICES_15 = yodd_indices(OBSERVABLE_LABELS_15)
assert OBSERVABLE_LABELS_15 == (
    "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ"
)
assert YODD_INDICES_15 == (1, 5, 7, 8, 10, 13)

# --- stage names (§9.4) ------------------------------------------------------
STAGES_JALUR_A = (
    "data_pipeline",
    "setup",
    "clustering_loop",
    "quantum_label_extraction",
    "supervised_loop",
    "structural_diagnostics",
    "final_evaluation",
)
STAGES_JALUR_B = (
    "load_artifacts",
    "quantum_label_extraction",
    "supervised_loop",
    "structural_diagnostics",
    "final_evaluation",
)

# --- run status (G3-04) ------------------------------------------------------
STATUS_RUNNING = "RUNNING"
STATUS_COMPLETED = "COMPLETED"
STATUS_FAILED = "FAILED"

# --- paths -------------------------------------------------------------------
MASTER_SPREADSHEET_PATH = "research_data/MORE_HD_master_confirmatory_240runs.xlsx"
ABLATION_SPREADSHEET_PATH = "research_data/MORE_HD_master_ablation_90runs.xlsx"
MORE_REFERENCE_SPREADSHEET_PATH = "research_data/MORE_HD_master_more_reference_120runs.xlsx"
LOCAL_RUN_SPREADSHEET_NAME = "run_result.xlsx"
MNIST_ROOT = "data/"
SPLITS_DIR = "splits/"
RUNS_DIR = "runs/"

# --- Jalur B scope (SAP §12.1; frozen 2026-10-04) ---------------------------------
JALUR_B_K_VALUES = (3, 6, 10)

# --- MORE literature reference (SAP §10.2.6) ---------------------------------------
MORE_TABLE_I = {
    3: (84.13, 88.7),
    4: (63.45, 70.1),
    5: (53.5, 63.3),
    6: (40.48, 50.2),
    7: (29.06, 37.2),
    8: (44.8, 48.6),
    9: (29.4, 33.0),
    10: (22.6, 27.8),
}


def simplex_bound(k: int) -> float:
    """Regular-simplex bound on the minimum pairwise cosine distance (G2-06)."""
    return 1.0 + 1.0 / (k - 1)
