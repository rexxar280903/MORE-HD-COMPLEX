# MORE-HD-C Targeted Ablation Protocol

Status: DESIGN LOCKED — implementation and runtime verification pending  
Decision date: 2026-09-27  
Prepared against GitHub `main` after commit `fa8cc3f95a1398c59798707012f773f625a58b40`  
Scope: G1-05 / G1-07 parameter-count versus complex-phase control

## 1. Purpose

The primary 2 × 3 × 8 experiment remains unchanged and compares MORE-HD against MORE-HD-C across K=3..10, PCA/HU/ZERNIKE, and five confirmatory seeds. The targeted ablation defined here is an additional confirmatory experiment designed to separate four effects that are confounded in the direct MORE-HD versus MORE-HD-C comparison: (1) additional real-valued trainable capacity, (2) access to a complex-valued readout state, (3) optimization of the complex phase, and (4) the combined full MORE-HD-C modification.

The ablation is intentionally restricted to representative class-count regimes K ∈ {3, 6, 10}. It uses all three feature representations and the same five confirmatory seeds used by the primary experiment: 101, 202, 303, 404, and 505. Seed 42 remains PILOT ONLY.

## 2. Four-model ablation family

| Code | Model | State constraint | Variational structure | Trainable parameters | Fixed phase parameters | Role |
|---|---|---|---|---:|---:|---|
| A | MORE-HD | real | 3 × RY-only layers | 30 | 0 | primary baseline; reused from the 240-run main experiment |
| B | MORE-HD-60P | real | 6 × RY-only layers | 60 | 0 | parameter-budget / real-capacity control |
| C | MORE-HD-C-FixedRZ | complex allowed | 3 × (RY + fixed RZ) layers | 30 RY | 30 RZ | complex-state access at the same trainable-parameter count as A |
| D | MORE-HD-C | complex allowed | 3 × (RY + trainable RZ) layers | 60 | 0 | primary proposed model; reused from the 240-run main experiment |

Models A and D are not rerun for the ablation. Their matching main-experiment runs are referenced by `run_uid` and reused. Only B and C generate new executions.

## 3. Ablation matrix and run count

The targeted conditions are:

```text
K = {3, 6, 10}
feature_method = {PCA, HU, ZERNIKE}
new ablation variants = {B, C}
confirmatory seeds = {101, 202, 303, 404, 505}
```

Therefore:

```text
3 K levels × 3 feature methods × 2 new variants = 18 ablation conditions per seed
18 conditions × 5 confirmatory seeds = 90 additional confirmatory runs
```

The complete confirmatory workload becomes:

```text
240 primary runs + 90 additional ablation runs = 330 unique confirmatory runs
```

The four-model analysis still contains A/B/C/D for each ablation cell, but A and D are linked to existing primary runs and are not counted again.

## 4. Pairing and data-control rules

For a fixed `(seed, K, feature_method)` cell, A, B, C, and D must use the same raw train/validation/official-test split manifest. The same preprocessing fit policy is retained: all learned transformations and scalers are fit only on the training partition.

The clustering-pair protocol is also paired. The selected five training instances per class and the resulting unordered unique pair manifest must be identical across A/B/C/D for the same `(seed, K)` cell. B and C therefore reuse the same deterministic pair-selection namespace and sample identities used by A/D.

No official-test information may be used to choose the architecture, initialization, optimizer checkpoint, ablation subset, or any other design decision.

## 5. Deterministic paired initialization

Paired initialization is shared with the primary A/D runs and derived from the confirmatory master seed using independent deterministic namespaces. This is required so that A and D can be reused from the primary matrix without breaking the ablation pairing.

```text
ry_core_seed  = DERIVE_SUBSEED(master_seed, "init:ry_core")
ry_extra_seed = DERIVE_SUBSEED(master_seed, "ablation:init:ry_extra")
rz_phase_seed = DERIVE_SUBSEED(master_seed, "init:rz_phase")
```

The common 30-element `theta_RY_core` is used to initialize the corresponding RY parameters in A, C, and D. Primary `MODEL_SETUP` must therefore build A and D from this same initialization bundle before any confirmatory run is executed. Model B uses the same 30-element core initialization for its first three RY layers and receives 30 additional RY parameters from `ry_extra_seed` for layers four through six.

Models C and D share the same 30-element initial RZ vector produced by `rz_phase_seed`. Every RZ angle must be non-zero. If deterministic generation produces an exact zero due to numerical representation, that element is deterministically redrawn from the same RNG stream until it is non-zero.

For model C, the RZ vector is frozen for both clustering and supervised optimization. Only the 30 RY parameters are optimized. For model D, the same initial RZ vector is trainable. This gives C-versus-D a common phase starting point.

## 6. Circuit builders

### 6.1 Model B — MORE-HD-60P

```text
FUNCTION BUILD_CIRCUIT_MORE_HD_60P(
    n_data_qubits=8,
    n_readout_qubits=2,
    n_layers=6
):
    ASSERT n_data_qubits == 8
    ASSERT n_readout_qubits == 2

    n_qubits = 10
    n_trainable = 6 * 10 = 60

    FUNCTION circuit_fn(x, theta):
        theta_layers = RESHAPE(theta, (6, 10))

        FOR i IN range(8):
            APPLY_GATE(RY(x[i]), wire=i)

        FOR layer_idx IN range(6):
            RY_ONLY_LAYER(theta_layers[layer_idx], n_qubits)
            ENTANGLING_BLOCK_CNOT(8, 2)

        RETURN MEASURE_15_OBSERVABLES(readout_wires=[8, 9])

    RETURN circuit_fn, n_trainable
```

Model B remains in the real-state family because it uses only RY rotations and CNOT gates. It is a parameter-budget control, but it is not a perfect depth-matched control: increasing from three to six RY layers also increases variational depth and the number of repeated entangling blocks. Therefore B-versus-D must be described as an equal-trainable-parameter comparison, not as a claim that all other circuit-resource dimensions are identical. Gate counts, variational depth, and runtime are recorded as secondary diagnostics.

### 6.2 Model C — MORE-HD-C-FixedRZ

```text
FUNCTION BUILD_CIRCUIT_MORE_HD_C_FIXED_RZ(
    fixed_rz,
    n_data_qubits=8,
    n_readout_qubits=2,
    n_layers=3
):
    ASSERT LENGTH(fixed_rz) == 30
    ASSERT ALL(angle != 0.0 FOR angle IN fixed_rz)

    n_qubits = 10
    n_trainable = 3 * 10 = 30

    FUNCTION circuit_fn(x, theta_ry):
        ry_layers = RESHAPE(theta_ry, (3, 10))
        rz_layers = RESHAPE(fixed_rz, (3, 10))

        FOR i IN range(8):
            APPLY_GATE(RY(x[i]), wire=i)

        FOR layer_idx IN range(3):
            FOR q IN range(10):
                APPLY_GATE(RY(ry_layers[layer_idx][q]), wire=q)
                APPLY_GATE(RZ(rz_layers[layer_idx][q]), wire=q)
            ENTANGLING_BLOCK_CNOT(8, 2)

        RETURN MEASURE_15_OBSERVABLES(readout_wires=[8, 9])

    RETURN circuit_fn, n_trainable
```

The fixed RZ vector must be saved as an artifact and hashed in the run manifest. It must not be updated by COBYLA during either training stage.

## 7. Ablation configuration and execution

Ablation runs use a dedicated entry point so that the 48 primary conditions and their R001-R048 identity remain unchanged.

```text
main_ablation.py

STRUCT AblationConfig EXTENDS Config:
    experiment_track = "ABLATION"
    ablation_model    = "MORE-HD-60P" | "MORE-HD-C-FixedRZ"
    K                 = 3 | 6 | 10
    feature_method    = "PCA" | "HU" | "ZERNIKE"
    seed              = one of [101,202,303,404,505]

FUNCTION ABLATION_MAIN(config):
    VALIDATE_ABLATION_CONFIG(config)
    CREATE_RUN_DIRECTORY_EXCLUSIVE(config)
    LOAD_OR_COPY_ABLATION_RUN_SPREADSHEET(config)

    split_manifest = CREATE_OR_LOAD_SPLIT_MANIFEST(config.seed)
    X_train, y_train, X_val, y_val, X_test, y_test = DATA_PIPELINE(config)
    S = CORRELATION_MATRIX(X_train, y_train, config.classes)

    init_bundle = BUILD_ABLATION_INITIALIZATION(config.seed)

    IF config.ablation_model == "MORE-HD-60P":
        circuit_fn, n_trainable = BUILD_CIRCUIT_MORE_HD_60P()
        initial_params = CONCAT(init_bundle.ry_core, init_bundle.ry_extra)
        fixed_rz = NULL

    ELSE IF config.ablation_model == "MORE-HD-C-FixedRZ":
        fixed_rz = init_bundle.rz_phase
        circuit_fn, n_trainable = BUILD_CIRCUIT_MORE_HD_C_FIXED_RZ(fixed_rz)
        initial_params = init_bundle.ry_core
        SAVE(fixed_rz, run_dir + "/artifacts/fixed_rz.npy")

    RUN_STANDARD_TWO_STAGE_PIPELINE_WITH_EXISTING_SPLIT_AND_PAIRS(...)
    FINAL_EVALUATION(...)
    SAVE_ABLATION_MANIFEST(...)
```

The standard clustering and supervised losses remain unchanged. The ablation does not introduce a new optimizer, new feature representation, new label definition, or new test-selection rule.

## 8. Ablation IDs and folder names

Ablation condition IDs are separate from R001-R048. Use `ABL001` through `ABL018` with the following deterministic order:

```text
K: 3 -> 6 -> 10
within K: PCA -> HU -> ZERNIKE
within feature: MORE-HD-60P -> MORE-HD-C-FixedRZ
```

A unique ablation execution uses:

```text
ablation_run_uid = ablation_condition_id + "-S" + seed
```

Recommended folder examples:

```text
runs/ablation/ABL001-S101_cls-0-1-2_PCA_MORE-HD-60P/
runs/ablation/ABL002-S101_cls-0-1-2_PCA_MORE-HD-C-FixedRZ/
```

Every ablation manifest must also record the matching primary run IDs for A and D so that the four-model cell can be assembled without rerunning them.

## 9. Required ablation artifacts

Each B/C run must save the same core artifacts as a primary run plus:

```text
ablation_condition_id
ablation_run_uid
ablation_model
matching_run_uid_A
matching_run_uid_D
n_trainable_params
n_fixed_rz_params
n_variational_layers
variational_gate_count
entangling_block_count
cnot_count
ry_core_seed
ry_extra_seed            # B only
rz_phase_seed            # C only
fixed_rz.npy             # C only
fixed_rz_sha256          # C only
paired_pair_manifest_sha256
paired_split_manifest_sha256
```

Structural diagnostics must explicitly record the six Y-odd observables `[IY, XY, YI, YX, YZ, ZY]`, their mean/max absolute magnitudes, and the Y-odd norm fraction.

## 10. Planned comparisons

The ablation is interpreted through paired, seed-matched contrasts within every `(feature_method, K)` cell:

```text
A -> B : additional real-valued trainable capacity / depth effect
A -> C : access to a complex-valued state with the same number of trainable parameters (30)
B -> D : equal trainable-parameter budget (60) comparison between deeper real RY-only and trainable complex RY+RZ
C -> D : effect of optimizing RZ instead of keeping the same initial RZ phase frozen
```

The two strongest mechanistic checks are A-versus-C and B-versus-D, interpreted jointly with the Y-odd activation diagnostics. Because B has greater variational depth than D, B-versus-D is not described as a perfectly architecture-matched causal contrast.

The primary publication claim remains architecture-level unless the combined ablation evidence supports a narrower phase-specific interpretation.

## 11. Statistical treatment

The same five confirmatory seeds form paired replications for the ablation. For each planned contrast and metric, report raw five paired differences, mean difference, standard deviation, median [IQR], 95% Student-t confidence interval for the mean difference, and paired Hedges g. The exact two-sided sign-flip permutation test is retained as a sensitivity test.

If inferential p-values for all four ablation contrasts are reported within the same `(feature_method, K, metric)` family, Holm correction is applied across the four planned contrasts. Effect magnitude and the raw five paired deltas remain more important than a binary significance threshold because n=5.

## 12. Claim boundary

Without this ablation, the direct A-versus-D result supports only a whole-architecture statement because trainable parameter count changes from 30 to 60 together with the addition of RZ. After the ablation, a phase-specific statement is permitted only when the relevant paired comparisons and structural diagnostics are mutually consistent.

In particular, activation of the previously structural-zero Y-odd observables demonstrates removal of the real-state restriction; it does not by itself prove that any accuracy increase is caused by complex phase. Classification metrics, separation metrics, and structural diagnostics must be interpreted together.