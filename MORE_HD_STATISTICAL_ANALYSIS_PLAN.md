# MORE-HD-C Statistical Analysis Plan

Version: 1.7  
Primary design frozen: 2026-09-26  
Targeted ablation extension frozen: 2026-09-27  
Cumulative class-sequence scope (G1-09) frozen: 2026-09-27  
Classical reference baselines and MORE reference policy (G1-10) frozen: 2026-09-27  
Consistency audit (smoke/pilot budget, workbook schema map, Jalur B sheet): 2026-09-27  
Quantum-label centroid rule (G2-07) and zero-norm safety (G2-04) frozen: 2026-10-01  
Validation-monitoring set and per-evaluation caching (G3-05) frozen: 2026-10-01  
Runtime reporting policy and circuit microbenchmark frozen: 2026-10-01  
Scope: confirmatory primary experiment + targeted G1-05/G1-07 ablation + secondary classical reference baselines

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

All unordered unique pairs among the selected 5K instances are included once (`i < j`), following the paper's clustering-dataset size `C(5K, 2)` (decision 2026-10-01). The public MORE code instead caps pairs at a default `pairs_num = 1000` drawn by an unseeded shuffle (active only at K=10, where 1,225 pairs exist) and takes the first five samples per class rather than a random draw; both code behaviours are deliberately not reproduced and are reported as differences from the public implementation. Self-pairs, duplicate pairs, and simultaneous inclusion of both `(i,j)` and `(j,i)` are prohibited. The pair set is constructed once before COBYLA begins and remains frozen for all objective-function evaluations in that run. No validation or official-test observation can enter pair construction.

No same-class/different-class balancing or reweighting is applied. The policy is frozen as `pair_balance_policy="NATURAL_FULL_PAIRING"` and `pair_weighting="NONE"` to preserve the original MORE-style loss composition. With five selected samples per class, `N_same = 10K`, `N_different = 25*C(K,2)`, and `N_total = C(5K,2)`. Consequently, pair composition changes with K; this is treated as a protocol characteristic and interpretive limitation rather than silently corrected after results are observed.

Every run must save `pair_manifest.json` and `pair_stats.json`, including the master/pair seeds, selected training identities by class, total/same/different pair counts and ratios, and zero-valued duplicate/self-pair counts. Confirmatory interpretation of K-related trends must acknowledge that the natural same-versus-different pair proportion changes with K.

## 4.2 Ablation pairing and circuit-resource diagnostics

For each `(seed, K, feature_method)` ablation cell, A/B/C/D must resolve to the same split manifest and the same pair manifest. Their manifest hashes are stored and checked before a four-model cell is admitted to analysis.

Model B has six RY-only variational layers and therefore matches D in trainable parameter count (60), but not in circuit depth or repeated entangling-block count. B-versus-D is therefore an equal-trainable-parameter comparison rather than a claim that all circuit resources are matched. Trainable parameter count, fixed parameter count, variational depth, RY/RZ gate counts, CNOT count, entangling-block count, and runtime are recorded for every A/B/C/D model; runtime comparisons between models follow §4.5.

## 4.3 Optimizer-budget policy (G1-01/G1-06, frozen 2026-09-27)

All architectures in a given track receive the same **total** COBYLA objective-evaluation budget (`max_nfev_clustering`, `max_nfev_supervised`). COBYLA spends its first `n_params + 1` evaluations building the initial simplex (A/C: 31; B/D: 61), so under equal total budget MORE-HD-C receives 30 fewer post-simplex evaluations than MORE-HD. This is treated as conservative toward MORE-HD-C and reported descriptively through `n_optimization_evals`; the B-versus-D and A-versus-C contrasts have identical simplex lengths and are free of this imbalance.

Every objective evaluation is labeled `phase = initial_simplex` (0-based `eval_id <= n_params`, including `x0`) or `phase = optimization`. The final budget is set from the seed-42 pilot (cap 300; K in {3,10} x {PCA, ZERNIKE} x {MORE-HD, MORE-HD-C}, plus MORE-HD-60P at K=10 x {PCA, ZERNIKE}, for 10 pilot runs; B is required because the final budget also governs the B ablation runs) using the plateau rule frozen in `Pseudocode_2x3_manual_runs.md` section 1.1 before the pilot is run. If any pilot run has not plateaued at 300, the final budget remains 300 and conclusions are stated as performance at equal objective-evaluation budget rather than at convergence. Pilot results are not used for any confirmatory estimate.

The smoke test uses `max_nfev = 70` for every architecture. SciPy COBYLA (verified on 1.17.1) silently raises any budget below `n_params + 2`, so smaller smoke budgets would not be the budget actually applied; the pipeline rejects them in every run mode.

## 4.4 Validation-monitoring set and per-evaluation caching (G3-05, frozen 2026-10-01)

Within each clustering objective evaluation, the circuit output of each of the 5K clustering samples is computed once and reused by the pair loss and the per-evaluation class centroids. This is a computational change only: the cached loss must equal the per-pair loss exactly.

All passive validation metrics in the clustering and supervised loops, including the Jalur B selector metrics (§12.1), are computed on the full validation split (`val_monitor_policy = FULL_VAL`). Computing passive metrics only at new `train_loss` records is not permitted, because it would silently shrink the Jalur B candidate domain. A fixed stratified subset (`STRATIFIED_FIXED_SUBSET`: the first `n_val_monitor_per_class` samples of each class in split-manifest order, hence paired across architectures and nested across K) may replace the full split only if the timing pilot shows that `FULL_VAL` is infeasible under a compute criterion written into Gate G3-05 before the timing pilot runs. That switch is global (all conditions, seeds, ablation runs, and Jalur B), and `n_val_monitor_per_class` is fixed before the convergence pilot. The subset never affects official-test evaluation, and classical baselines (§10.2) keep selecting `C` on the full validation split. The policy in force is recorded per run (`val_monitor_policy`, `n_val_monitor_per_class`, `val_monitor_manifest.json`) and reported in the methods.

## 4.5 Runtime reporting and computational cost (frozen 2026-10-01)

Production-run runtime (Jalur A, targeted ablation, and Jalur B) is recorded at four levels: per objective evaluation (wall-clock and process CPU time), per pipeline stage, at run start (start time, host, declared number of concurrent runs, load average, thread environment), and at run end. These values are reported descriptively only (mean, median, and IQR per condition) and are used for auditing and workload projection. No paired runtime Delta, confidence interval, statistical test, or claim that one architecture is faster than another is derived from production runs, because their wall-clock times depend on factors outside the architecture: the number of concurrent runs (intentionally not fixed), machine load, throttling, and execution order.

Claims about the computational cost of A, B, C, and D rely only on a controlled microbenchmark (`Pseudocode_2x3_manual_runs.md` §1.2.1): one process, one thread, no other runs active, the confirmatory simulator settings, three blocks of 1,000 timed single-circuit executions per model after warm-up, with model order rotated across blocks. Because the objective-evaluation budget and data sizes are identical across architectures, the number of circuit executions per run is the same for A and D, so the per-execution cost measured by the benchmark accounts for the architectural difference in compute. The benchmark reports median, Q1, Q3, and 95th-percentile time per execution together with gate count, depth, and parameter count, and is repeated if the device, library versions, or machine change. It uses no test data and does not influence any protocol decision other than workload projection.

## 5. Primary outcomes

The primary classification outcomes are official-test accuracy and macro-F1. The primary representation outcome is `min_separation_ratio`. Structural-zero diagnostics, active-dimension measures, Y-odd norm fraction, per-class metrics, confusion patterns, optimizer behavior, and gate/depth diagnostics are secondary outcomes. Runtime is a descriptive measure only (§4.5).

Official-test metrics are computed only after optimization, checkpoint selection, and all protocol decisions for that run are frozen.

**Centroid and label definition (G2-07, frozen 2026-10-01).** Quantum labels, the per-evaluation class centroids behind `pseudo_accuracy_val`, `min_separation`, and `min_separation_ratio`, and the validation centroids behind `min_separation_val` all follow the original MORE implementation (`MORE_clustering.py::find_center`): each sample output is normalized to a unit vector, the component-wise median is taken, and the result is normalized again. As in MORE, class centroids and quantum labels are built from the five clustering samples per class recorded in `pair_manifest.json` (validation centroids use all validation samples of the class, because MORE has no validation-centroid metric). This deviates from FRD-09 (mean of raw outputs over all training samples, then normalization); results for architecture A are therefore not protocol-identical to FRD-09 and the paper must state this.

**Zero-norm safety (G2-04, frozen 2026-10-01).** Cosine distance and normalization use `eps_norm = 1e-10`: a vector with norm below `eps_norm` has no direction, its cosine distance is 1.0, its normalization is the zero vector, and it is excluded from the median centroid. Non-finite outputs fail the run. Nearest-label ties resolve to the first class in ascending class order. For vectors with norm at least `eps_norm` these definitions are identical to MORE. Per-evaluation degeneracy counts and the run-level flag `degenerate_quantum_label` are recorded; runs with a degenerate quantum label remain in the confirmatory aggregation (the protocol is deterministic) but are listed in the Gate D data-quality report and discussed if present.

## 6. Primary estimand

For each feature_method × K combination and each confirmatory seed s:

`Delta_AD,s = Metric(D = MORE-HD-C, s) - Metric(A = MORE-HD, s)`.

The primary estimand is the mean paired difference across the five confirmatory seeds. The same paired differences are also summarized by their median and interquartile range.

Positive Delta indicates a larger metric for MORE-HD-C. Runtime is not analyzed as a paired Delta (§4.5).

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

The same five-seed summaries are produced for B and C in targeted ablation cells. No best-seed, best-run, or best-of-five result is used as the main reported performance. Runtime is summarized in the same descriptive way (mean, median, IQR) but is excluded from §8 and §9 (see §4.5).

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

## 10.1 Cumulative class-sequence scope (G1-09, frozen 2026-09-27)

The K scenarios use the cumulative MNIST class sequence `[0..K-1]`. Each transition K→K+1 therefore adds one specific digit, so the effect of the number of classes cannot be separated from the identity of the added digit (for example, a change at K=8 may reflect the similarity of digit 7 to digit 1 rather than the class count itself). The cumulative sequence is retained without random class subsets, to preserve direct comparability with FRD-09 and to keep the frozen paired/nested split protocol (G0-04) unchanged.

Scope rules:

1. The primary estimand (seed-matched A-versus-D difference within each `feature_method × K` cell) is not affected, because both architectures are evaluated on identical classes, split manifests, pair manifests, and seeds.
2. Any trend across K, and any architecture × K interaction (for example, that MORE-HD-C degrades less as K increases), is reported as specific to the cumulative sequence `[0..K-1]` and is not generalized to arbitrary class sets of size K.
3. Changes in performance between adjacent K values are not attributed solely to the number of classes. The class-count effect is additionally coupled with the change in natural pair composition (G1-08) and, for HU/ZERNIKE at K=10, with rotation-invariance ambiguity between digits 6 and 9 (G1-11).
4. No random class subsets are run, so no subset list exists to be frozen.

Class-identity analyses (proximity of the newly added class mean to existing class means in the run's feature space, classical baselines on non-cumulative class sets, and per-class confusion) are not part of the confirmatory plan. If performed, they are labeled exploratory. To keep them possible without re-execution, every run must retain `logs/confusion_matrix.npy`, the fitted feature transformers, and its split manifest.

## 10.2 Classical reference baselines and MORE reference policy (G1-10, frozen 2026-09-27)

### 10.2.1 Purpose and status

Classical baselines are added to contextualize the quantum results: they show how much class information a simple classical model can use from exactly the same 8-channel input that is given to the circuit. They are **reference points, not competitors and not part of the confirmatory estimand**. The primary estimand (Section 6) and the ablation estimands (Section 6.1) are unchanged. All quantum-versus-classical comparisons are secondary and descriptive.

### 10.2.2 Baseline set

| Baseline | Specification | Hyperparameters |
|---|---|---|
| Chance | Accuracy `1/K`; expected macro-F1 of a uniform random predictor, also `1/K`. The official test set is class-balanced (200 per class), so the majority-class rate equals `1/K`. | none (computed, not fitted) |
| Nearest Centroid (NC) | Class means on train; Euclidean distance; `shrink_threshold=None`. Classical analogue of assigning a sample to the nearest class representative. | none |
| Logistic Regression (LR) | Multinomial logistic regression, L2 penalty, `lbfgs`, `fit_intercept=True`, `class_weight=None`, `max_iter=5000`. `9K` trainable parameters (27 at K=3, 90 at K=10), the same order of magnitude as the 30/60 quantum trainable parameters. | `C` selected from `{0.01, 0.1, 1, 10, 100}` by validation accuracy; ties resolved toward the smallest `C` |

The grid, tie rule, solver, and iteration cap are frozen before any baseline or quantum official-test outcome is inspected. LR is refit on train only with the selected `C`; validation is never merged into train, matching the quantum pipeline. Convergence warnings are recorded per fit.

### 10.2.3 Input and pairing

For each `(seed, feature_method, K)` cell, the baselines read `X_train_scaled.npy`, `y_train.npy`, `X_val_scaled.npy`, `y_val.npy`, `X_test_scaled.npy`, and `y_test.npy` saved by `DATA_PIPELINE` (pseudocode §2.7) in the matching primary MORE-HD (A) run. No preprocessing is repeated, so baseline input is identical to circuit input, including the `[0, π]` scaling and, for HU, the constant eighth channel. Before fitting, the array hashes are compared with the matching MORE-HD-C (D) run when it exists; any mismatch aborts the cell. The split-manifest hash of the source run is stored with the result.

Arrays are cell-specific: PCA and the MinMax scaler are fitted on the K-class train set of that cell, so K=3..9 cells are not derived by filtering the K=10 arrays.

Scope: 3 feature methods × 8 K × 5 confirmatory seeds = 120 cells; NC and LR per cell = 240 fits. Seed 42 is excluded, as for all confirmatory analyses.

### 10.2.4 Outcomes and reporting

Outcomes are official-test accuracy and macro-F1, plus per-class F1 and the confusion matrix. For each `feature_method × K`, report the five seed values as mean ± SD for NC and LR, and chance as a fixed reference line.

Because baselines and quantum runs share the same official-test samples within a seed, seed-matched differences `Metric(quantum, s) - Metric(baseline, s)` for A and D against NC and LR are reported descriptively (mean ± SD of the five differences). If McNemar's exact test between a quantum model and a baseline is reported, it is labeled secondary, and Holm correction is applied within each seed across the reported `feature_method × K` cells for that model pair. These comparisons cannot change the conclusion of the primary A-versus-D analysis.

Per-class results and 6↔9 confusion of the baselines may be used as supporting evidence for G1-09 (difficulty of the digit added at each K) and G1-11 (feature-level ambiguity of rotation-invariant HU/ZERNIKE), labeled as supporting descriptive evidence.

### 10.2.5 Claim boundary

Baseline accuracy is interpreted as the performance of a simple classical model on the same 8 features, **not** as a measure of the total information contained in a feature representation. LR is a linear lower bound. Statements such as "HU contains less information than PCA" are not permitted from these baselines. Non-linear classical baselines (kNN, SVM-RBF) and a raw-pixel baseline are deferred to future work; if run later, they are exploratory.

### 10.2.6 Loss adjuster R and MORE literature reference

The MORE loss adjuster R (Wu et al., 2023, Eq. 14) is **not** implemented in any track (`loss_adjuster_policy = "NONE"`). Rationale: (1) R targets the same symptom as the V1 modification (crowded quantum labels), so adding it would confound attribution of the A-versus-D effect; (2) R requires two per-task hyperparameters (`r` from inter-label cosine distance, `w` tuned over 0.1-1.0) that would have to be set separately for architectures with different label geometry; (3) tuning them would add an unplanned search budget.

Wu et al. (2023) Table I reports both MORE\R (without R) and MORE (with R) accuracy on the same cumulative tasks 0-2 ... 0-9 (K=3..10). The MORE\R column is the primary literature reference; the MORE (+R) column is shown as context only. These values are reported numbers, not reproduced executions:

| K | MORE\R accuracy (%) | MORE +R accuracy (%) |
|---:|---:|---:|
| 3 | 84.13 | 88.7 |
| 4 | 63.45 | 70.1 |
| 5 | 53.5 | 63.3 |
| 6 | 40.48 | 50.2 |
| 7 | 29.06 | 37.2 |
| 8 | 44.8 | 48.6 |
| 9 | 29.4 | 33.0 |
| 10 | 22.6 | 27.8 |

Comparison with MORE uses accuracy only (MORE does not report macro-F1), is descriptive, and carries no statistical test, because each MORE value is a single reported number from a different protocol (data sizes, feature extraction, number of runs/seeds, and, for the +R column, the loss adjuster). Every table or sentence comparing with MORE must state that the protocols differ.

## 11. Missing, failed, and repeated attempts

A technical failure before a valid final evaluation does not create a new seed. The same `run_uid` or `ablation_run_uid` is repeated as a new attempt using the same split manifest, pair manifest, initialization protocol, and seed. The corresponding workbook records attempt number and retains provenance. Exactly one valid completed attempt per execution identifier is used in confirmatory aggregation; failed attempts remain auditable.

A run with a protocol deviation is not silently repaired after official-test inspection. The deviation is logged and handled according to the readiness-gate decision made before aggregate analysis.

## 12. Workbook mapping

`MORE_HD_master_confirmatory_240runs.xlsx` remains the canonical primary-experiment aggregation template. Sheet `01_Run_Summary` contains one row per primary `run_uid`; `02_Run_Config` records seed propagation and split provenance; `12_Seed_Aggregation` stores the five-seed summaries; `13_Paired_Comparison` stores seed-matched A-versus-D differences. Detailed history and official-test prediction sheets are append-only during consolidation.

The targeted ablation uses a separate planned template `MORE_HD_master_ablation_90runs.xlsx` with one row per new B/C `ablation_run_uid`. It records `matching_run_uid_A` and `matching_run_uid_D` rather than duplicating A/D executions. Stage-level timing and run-start metadata (§4.5) are recorded in `01_Run_Summary` and, for Jalur B, in `17_JalurB_Selection`; per-evaluation CPU time is recorded in `03`/`04`; microbenchmark results are stored in sheet `18_Circuit_Benchmark`. The ablation template additionally stores the same timing fields, model code, trainable/fixed parameter counts, depth/gate counts, initialization namespaces/hashes, fixed-RZ hash for C, paired split/pair-manifest hashes, Y-odd diagnostics, and the four-model contrast linkage.

Sheet `00_Schema_Map` is the single source of truth for which artifact file and field fills every workbook column, at which granularity. Run-level clustering metrics in `01_Run_Summary` are read from the `clustering_log.jsonl` row whose `eval_id` equals `final_point_eval_id` (the evaluation identical to COBYLA's `result.x`, i.e. the parameters that define the quantum labels), not from the last logged row. Columns without a defined source are marked `GAP` or `PENDING` in the map and must be resolved before Gate C.

Publication analysis joins the two workbooks by seed, K, feature method, and matching execution identifiers. Classical reference baselines (Section 10.2) are recorded in sheet `15_Classical_Baselines` of the primary workbook (one row per `(seed, feature_method, K, baseline)`, linked to `source_run_uid_A` and `check_run_uid_D`), and the reported MORE values in sheet `16_MORE_Reference`. The primary 240-run workbook is not replaced or expanded merely to duplicate A/D rows.

## 12.1 Secondary Jalur B sensitivity analysis

The frozen count of 330 unique confirmatory executions refers only to the 240 primary A/D runs plus the 90 new B/C targeted-ablation runs. Jalur B is a separate secondary sensitivity path and is not included in that count.

The primary result for every A/D run remains the Jalur A result obtained by passing the COBYLA final point `result.x` from clustering into quantum-label extraction and supervised training. Jalur B must not replace that primary result or be used to redefine the primary estimand after official-test outcomes are known.

If Jalur B is executed, it starts from an already completed Jalur A source run and does not rerun data preprocessing or clustering. The clustering checkpoint is chosen automatically from eligible objective evaluations using the pre-specified lexicographic rule (validation metrics are computed on the validation-monitoring set of §4.4, inherited unchanged from the source run):

```text
1. pseudo_accuracy_val      DESC
2. min_separation_val       DESC
3. mean_true_distance_val   ASC
4. train_loss               ASC
5. eval_id                  ASC
```

No weighted score is used. `active_dimensions`, `correlation_consistency`, and `avg_margin_val` are diagnostics and cannot affect checkpoint selection. The supervised objective-evaluation budget is inherited unchanged from the source Jalur A run. Official-test arrays are not read by the selector or supervised optimization and are used only by the final evaluation after the selected checkpoint and downstream training are frozen.

Only objective evaluations with `phase = optimization` are eligible candidates (Gate G1-06, frozen 2026-09-27); all initial-simplex evaluations, including `x0`, are excluded. Evaluations with any degenerate training or validation centroid (`n_degenerate_centroids_train > 0` or `n_degenerate_centroids_val > 0`) are also excluded (Gate G2-04, frozen 2026-10-01), because the zero-norm fallback can inflate `min_separation_val` for collapsed centroids. The number of eligible candidates per run is recorded. The scope of Jalur B executions (all primary runs or a pre-specified subset) must be frozen before those secondary results are analyzed; no subset may be selected because of official-test performance. Jalur B results are reported as secondary/sensitivity evidence and are kept separate from the primary 240-run A/D aggregation and the 90-run B/C ablation aggregation. Each Jalur B execution is recorded as one row of sheet `17_JalurB_Selection` (source `run_uid`, `selected_eval_id`, number of eligible candidates, the selection metrics of the chosen evaluation, supervised budget, and final official-test metrics); Jalur B never writes to sheets `01`–`04`.

## 13. Confirmatory boundary

Seed 42 and any other development run are PILOT ONLY. They may be used for debugging, convergence inspection, timing, and protocol locking, but they are excluded from confirmatory means, standard deviations, confidence intervals, effect sizes, tests, and publication tables presenting final performance.

No official-test outcome from either the primary or ablation track may be inspected to alter the frozen K subset, feature set, model definitions, initialization pairing, fixed-RZ vector policy, planned contrasts, multiplicity rule, optimizer-budget policy, classical baseline set and hyperparameter grid, or loss-adjuster policy.