# MORE-HD-C Statistical Analysis Plan

Version: 1.1  
Primary design frozen: 2026-09-26  
Targeted ablation extension frozen: 2026-09-27  
Scope: confirmatory primary experiment + targeted G1-05/G1-07 ablation

## 1. Purpose and freeze rule

This document freezes the statistical analysis for the MORE-HD versus MORE-HD-C confirmatory experiment before final results are opened and extends the frozen plan with the pre-specified targeted ablation required to separate parameter-count and complex-phase effects. Analyses not specified here are exploratory and must be labeled as such. Pilot seed 42 is excluded from confirmatory aggregation and inferential reporting.

## 2. Confirmatory design

### 2.1 Primary experiment

The primary design contains 48 experimental conditions: two architectures (A = MORE-HD, D = MORE-HD-C), three feature representations (PCA, HU, ZERNIKE), and eight cumulative MNIST class scenarios (K=3..10). Each condition is replicated with five pre-specified confirmatory seeds: 101, 202, 303, 404, and 505, producing 240 primary confirmatory runs.

The primary experimental condition identifier remains R001-R048. A unique primary execution is identified by `run_uid = condition_id-S<seed>`, for example `R001-S101`.

### 2.2 Targeted ablation experiment

The G1-05/G1-07 ablation is restricted to K in {3, 6, 10}, all three feature representations, and the same five confirmatory seeds. Four models are compared in each ablation cell:

| Code | Model | State family | Trainable parameters | Fixed parameters | Role |
|---|---|---|---:|---:|---|
| A | MORE-HD | real | 30 RY | 0 | primary baseline; reused from main experiment |
| B | MORE-HD-60P | real | 60 RY | 0 | parameter-budget / real-capacity control |
| C | MORE-HD-C-FixedRZ | complex allowed | 30 RY | 30 non-zero RZ | complex-state access at the same trainable count as A |
| D | MORE-HD-C | complex allowed | 30 RY + 30 RZ = 60 | 0 | proposed primary model; reused from main experiment |

Only B and C create new executions. There are `3 K × 3 features × 2 new variants = 18` new ablation conditions per seed and `18 × 5 = 90` additional confirmatory runs. A and D are linked to their matching primary `run_uid` and are not rerun. Therefore the project contains 330 unique confirmatory executions: 240 primary + 90 additional ablation runs.

Ablation conditions use separate identifiers `ABL001-ABL018` in deterministic order `K: 3 -> 6 -> 10`, then `feature: PCA -> HU -> ZERNIKE`, then `model: B -> C`. A unique ablation execution is `ablation_run_uid = ablation_condition_id-S<seed>`.

## 3. Paired split and nested-K rule

For a given confirmatory seed, the raw MNIST train, validation, and official-test sample identities are fixed in `splits/seed<seed>.json` and shared by PCA, HU, ZERNIKE, MORE-HD, MORE-HD-C, and the targeted ablation controls. Therefore all planned architecture and ablation comparisons are paired on the same raw samples.

The class scenarios are nested. For a given seed, samples assigned to an existing digit remain unchanged as K increases. Moving from K to K+1 only adds the newly introduced digit. Preprocessing transformations and scalers remain fit on the training partition only.

## 4. Randomness and replication unit

Each master seed deterministically derives independent RNG streams for data selection, parameter initialization, clustering-pair construction, and quantum-label sampling using `SHA256(master_seed:namespace)`, with the first eight hexadecimal digits interpreted as uint32.

The primary replication unit for optimizer/sampling variability is the seed-specific run. The five confirmatory seeds are treated as five paired replications. No seed may be removed because its result is unfavorable, and no replacement seed may be introduced after confirmatory results are inspected.

Primary A/D and targeted ablation initialization use the following pre-specified deterministic namespaces so the reused primary runs remain paired with B/C:

```text
ry_core_seed  = DERIVE_SUBSEED(master_seed, "init:ry_core")
ry_extra_seed = DERIVE_SUBSEED(master_seed, "ablation:init:ry_extra")
rz_phase_seed = DERIVE_SUBSEED(master_seed, "init:rz_phase")
```

The common 30-element RY core initialization is shared by the corresponding parameters in A, C, and D. Primary A and D runs are generated with this structured initialization before confirmatory execution; D packs the RY core and RZ phase by variational layer to match its RY+RZ parameter ordering. Model B uses the same 30-element RY core for its first three RY layers and obtains its additional 30 RY parameters from `ry_extra_seed`. Models C and D share the same initial 30-element RZ vector from `rz_phase_seed`. Every RZ element must be non-zero; model C freezes this vector during both training stages whereas model D optimizes it.

## 4.1 Frozen clustering-pair protocol

The clustering dataset follows the original MORE sampling concept as closely as possible while making the implementation deterministic and auditable. Exactly five training instances per class are selected without replacement. Selection uses class-specific deterministic substreams derived from the master seed (`SHA256(master_seed:"cluster_pairs:" + class_id)`), so sample identities for an existing digit remain unchanged as K increases and are shared across PCA, HU, ZERNIKE, MORE-HD, MORE-HD-C, and the ablation controls for the same seed and class scenario.

All unordered unique pairs among the selected 5K instances are included once (`i < j`). Self-pairs, duplicate pairs, and simultaneous inclusion of both `(i,j)` and `(j,i)` are prohibited. The pair set is constructed once before COBYLA begins and remains frozen for all objective-function evaluations in that run. No validation or official-test observation can enter pair construction.

No same-class/different-class balancing or reweighting is applied. The policy is frozen as `pair_balance_policy="NATURAL_FULL_PAIRING"` and `pair_weighting="NONE"` to preserve the original MORE-style loss composition. With five selected samples per class, `N_same = 10K`, `N_different = 25*C(K,2)`, and `N_total = C(5K,2)`. Consequently, pair composition changes with K; this is treated as a protocol characteristic and interpretive limitation rather than silently corrected after results are observed.

Every run must save `pair_manifest.json` and `pair_stats.json`, including the master/pair seeds, selected training identities by class, total/same/different pair counts and ratios, and zero-valued duplicate/self-pair counts. Confirmatory interpretation of K-related trends must acknowledge that the natural same-versus-different pair proportion changes with K.

## 4.2 Ablation pairing and circuit-resource diagnostics

For each `(seed, K, feature_method)` ablation cell, A/B/C/D must resolve to the same split manifest and the same pair manifest. Their manifest hashes are stored and checked before a four-model cell is admitted to analysis.

Model B has six RY-only variational layers and therefore matches D in trainable parameter count (60), but not in circuit depth or repeated entangling-block count. B-versus-D is therefore an equal-trainable-parameter comparison rather than a claim that all circuit resources are matched. Trainable parameter count, fixed parameter count, variational depth, RY/RZ gate counts, CNOT count, entangling-block count, and runtime are recorded for every A/B/C/D model.

## 5. Primary outcomes

The primary classification outcomes are official-test accuracy and macro-F1. The primary representation outcome is `min_separation_ratio`. Structural-zero diagnostics, active-dimension measures, Y-odd norm fraction, per-class metrics, confusion patterns, optimizer behavior, gate/depth diagnostics, and runtime are secondary outcomes.

Official-test metrics are computed only after optimization, checkpoint selection, and all protocol decisions for that run are frozen.

## 6. Primary estimand

For each feature_method × K combination and each confirmatory seed s:

`Delta_AD,s = Metric(D = MORE-HD-C, s) - Metric(A = MORE-HD, s)`.

The primary estimand is the mean paired difference across the five confirmatory seeds. The same paired differences are also summarized by their median and interquartile range.

Positive Delta indicates a larger metric for MORE-HD-C. For runtime, interpretation is reversed: positive Delta indicates greater computational cost.

## 6.1 Pre-specified ablation estimands

Within each targeted `(feature_method, K)` cell, four paired contrasts are pre-specified:

```text
Delta_AB,s = Metric(B,s) - Metric(A,s)
Delta_AC,s = Metric(C,s) - Metric(A,s)
Delta_BD,s = Metric(D,s) - Metric(B,s)
Delta_CD,s = Metric(D,s) - Metric(C,s)
```

Their interpretations are constrained as follows. A-versus-B measures the combined effect of additional real-valued trainable capacity and added RY-only depth. A-versus-C tests access to a complex-valued circuit family while holding the number of trainable parameters at 30. B-versus-D compares two models with 60 trainable parameters but does not perfectly match circuit depth. C-versus-D tests whether optimizing RZ provides benefit beyond using the same non-zero RZ phase vector as a fixed circuit component.

The strongest phase-related interpretation requires the A-versus-C and B-versus-D evidence to be considered jointly with direct structural diagnostics of the six Y-odd observables `[IY, XY, YI, YX, YZ, ZY]`. No single contrast is treated as a complete causal decomposition.

## 7. Descriptive aggregation

For every primary architecture × feature_method × K condition, report the five seed values and summarize them using mean ± standard deviation. Median [IQR] is reported as a robustness summary. Minimum and maximum may be retained in the workbook for diagnostics but are not the primary paper summary.

The same five-seed summaries are produced for B and C in targeted ablation cells. No best-seed, best-run, or best-of-five result is used as the main reported performance.

## 8. Confidence interval and effect size

For each planned paired architecture or ablation comparison, report a 95% Student-t confidence interval for the mean of the five paired seed differences. Because n=5 is small, the confidence interval is interpreted together with the raw five deltas and not as a stand-alone stability claim.

The standardized paired effect size is paired Hedges g, obtained by applying the small-sample correction to the paired standardized mean difference. The unstandardized paired difference remains the primary effect measure.

## 9. Statistical tests

The seed-level sensitivity test is an exact two-sided sign-flip permutation test on the five paired differences. With only five replications, exact p-values are coarse; therefore statistical conclusions emphasize effect magnitude, confidence intervals, and consistency of the five paired deltas rather than a binary significance threshold.

As a secondary instance-level check, MORE-HD and MORE-HD-C predictions may be compared with McNemar's exact test within each seed because the two architectures use identical official-test sample identities for that seed. If these secondary tests are reported across the 24 feature_method × K architecture comparisons within one seed, Holm correction is applied within that seed. McNemar results do not replace the seed-level analysis because they do not quantify optimizer variability across independent runs.

For the targeted ablation, if inferential p-values for all four planned contrasts are reported within the same `(feature_method, K, metric)` family, Holm correction is applied across A-B, A-C, B-D, and C-D. This multiplicity rule is fixed before ablation outcomes are inspected.

## 10. Factorial and mechanistic interpretation

K and feature_method are pre-specified stratification factors. Primary results are reported for every K=3..10 and every feature method; conditions may not be selectively omitted based on performance. Primary architecture effects are examined within each feature_method × K cell. Cross-K trends are interpreted cautiously because the cumulative class design changes both class count and the identity of the newly added digit.

The targeted ablation is not a replacement for the 2 × 3 × 8 primary design and is not extended post hoc to whichever K values look favorable. It is frozen at K={3,6,10} to cover low, intermediate, and full ten-class regimes across all three feature methods.

No claim that MORE-HD-C improves performance purely because of complex phase is permitted from A-versus-D alone. Activation of previously structural-zero Y-odd observables supports removal of the real-state restriction, but does not by itself prove an accuracy benefit. A narrower phase-specific interpretation is permitted only if the pre-specified ablation contrasts, performance/separation outcomes, and Y-odd structural diagnostics are mutually consistent, while explicitly acknowledging the depth mismatch in B-versus-D.

## 11. Missing, failed, and repeated attempts

A technical failure before a valid final evaluation does not create a new seed. The same `run_uid` or `ablation_run_uid` is repeated as a new attempt using the same split manifest, pair manifest, initialization protocol, and seed. The corresponding workbook records attempt number and retains provenance. Exactly one valid completed attempt per execution identifier is used in confirmatory aggregation; failed attempts remain auditable.

A run with a protocol deviation is not silently repaired after official-test inspection. The deviation is logged and handled according to the readiness-gate decision made before aggregate analysis.

## 12. Workbook mapping

`MORE_HD_master_confirmatory_240runs.xlsx` remains the canonical primary-experiment aggregation template. Sheet `01_Run_Summary` contains one row per primary `run_uid`; `02_Run_Config` records seed propagation and split provenance; `12_Seed_Aggregation` stores the five-seed summaries; `13_Paired_Comparison` stores seed-matched A-versus-D differences. Detailed history and official-test prediction sheets are append-only during consolidation.

The targeted ablation uses a separate planned template `MORE_HD_master_ablation_90runs.xlsx` with one row per new B/C `ablation_run_uid`. It records `matching_run_uid_A` and `matching_run_uid_D` rather than duplicating A/D executions. The ablation template additionally stores model code, trainable/fixed parameter counts, depth/gate counts, initialization namespaces/hashes, fixed-RZ hash for C, paired split/pair-manifest hashes, Y-odd diagnostics, and the four-model contrast linkage.

Publication analysis joins the two workbooks by seed, K, feature method, and matching execution identifiers. The primary 240-run workbook is not replaced or expanded merely to duplicate A/D rows.

## 13. Confirmatory boundary

Seed 42 and any other development run are PILOT ONLY. They may be used for debugging, convergence inspection, timing, and protocol locking, but they are excluded from confirmatory means, standard deviations, confidence intervals, effect sizes, tests, and publication tables presenting final performance.

No official-test outcome from either the primary or ablation track may be inspected to alter the frozen K subset, feature set, model definitions, initialization pairing, fixed-RZ vector policy, planned contrasts, multiplicity rule, or optimizer-budget policy.