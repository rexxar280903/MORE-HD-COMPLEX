# MORE-HD-C Statistical Analysis Plan

Version: 1.0  
Frozen: 2026-09-26  
Scope: confirmatory experiment only

## 1. Purpose and freeze rule

This document freezes the statistical analysis for the MORE-HD versus MORE-HD-C confirmatory experiment before final results are opened. Analyses not specified here are exploratory and must be labeled as such. Pilot seed 42 is excluded from confirmatory aggregation and inferential reporting.

## 2. Confirmatory design

The primary design contains 48 experimental conditions: two architectures (MORE-HD, MORE-HD-C), three feature representations (PCA, HU, ZERNIKE), and eight cumulative MNIST class scenarios (K=3..10). Each condition is replicated with five pre-specified confirmatory seeds: 101, 202, 303, 404, and 505, producing 240 confirmatory runs.

The experimental condition identifier remains R001-R048. A unique execution is identified by `run_uid = condition_id-S<seed>`, for example `R001-S101`.

## 3. Paired split and nested-K rule

For a given confirmatory seed, the raw MNIST train, validation, and official-test sample identities are fixed in `splits/seed<seed>.json` and shared by PCA, HU, ZERNIKE, MORE-HD, and MORE-HD-C. Therefore architecture comparisons are paired on the same raw samples.

The class scenarios are nested. For a given seed, samples assigned to an existing digit remain unchanged as K increases. Moving from K to K+1 only adds the newly introduced digit. Preprocessing transformations and scalers remain fit on the training partition only.

## 4. Randomness and replication unit

Each master seed deterministically derives independent RNG streams for data selection, parameter initialization, clustering-pair construction, and quantum-label sampling using `SHA256(master_seed:namespace)`, with the first eight hexadecimal digits interpreted as uint32.

The primary replication unit for optimizer/sampling variability is the seed-specific run. The five confirmatory seeds are treated as five paired replications. No seed may be removed because its result is unfavorable, and no replacement seed may be introduced after confirmatory results are inspected.

## 5. Primary outcomes

The primary classification outcomes are official-test accuracy and macro-F1. The primary representation outcome is `min_separation_ratio`. Structural-zero diagnostics, active-dimension measures, Y-odd norm fraction, per-class metrics, confusion patterns, and runtime are secondary outcomes.

Official-test metrics are computed only after optimization, checkpoint selection, and all protocol decisions for that run are frozen.

## 6. Primary estimand

For each feature_method × K combination and each confirmatory seed s:

`Delta_s = Metric(MORE-HD-C, s) - Metric(MORE-HD, s)`.

The primary estimand is the mean paired difference across the five confirmatory seeds. The same paired differences are also summarized by their median and interquartile range.

Positive Delta indicates a larger metric for MORE-HD-C. For runtime, interpretation is reversed: positive Delta indicates greater computational cost.

## 7. Descriptive aggregation

For every architecture × feature_method × K condition, report the five seed values and summarize them using mean ± standard deviation. Median [IQR] is reported as a robustness summary. Minimum and maximum may be retained in the workbook for diagnostics but are not the primary paper summary.

No best-seed, best-run, or best-of-five result is used as the main reported performance.

## 8. Confidence interval and effect size

For each planned paired architecture comparison, report a 95% Student-t confidence interval for the mean of the five paired seed differences. Because n=5 is small, the confidence interval is interpreted together with the raw five deltas and not as a stand-alone stability claim.

The standardized paired effect size is paired Hedges g, obtained by applying the small-sample correction to the paired standardized mean difference. The unstandardized paired difference remains the primary effect measure.

## 9. Statistical tests

The seed-level sensitivity test is an exact two-sided sign-flip permutation test on the five paired differences. With only five replications, exact p-values are coarse; therefore statistical conclusions emphasize effect magnitude, confidence intervals, and consistency of the five paired deltas rather than a binary significance threshold.

As a secondary instance-level check, MORE-HD and MORE-HD-C predictions may be compared with McNemar's exact test within each seed because the two architectures use identical official-test sample identities for that seed. If these secondary tests are reported across the 24 feature_method × K architecture comparisons within one seed, Holm correction is applied within that seed. McNemar results do not replace the seed-level analysis because they do not quantify optimizer variability across independent runs.

## 10. Factorial interpretation

K and feature_method are pre-specified stratification factors. Results are reported for every K=3..10 and every feature method; conditions may not be selectively omitted based on performance. Architecture effects are examined within each feature_method × K cell. Cross-K trends are interpreted cautiously because the cumulative class design changes both class count and the identity of the newly added digit.

No claim that MORE-HD-C improves performance purely because of complex phase is permitted unless the separate parameter-count/phase ablation required by G1-05/G1-07 supports that claim.

## 11. Missing, failed, and repeated attempts

A technical failure before a valid final evaluation does not create a new seed. The same `run_uid` is repeated as a new attempt using the same split manifest and seed protocol. The master workbook records attempt number and retains provenance. Exactly one valid completed attempt per `run_uid` is used in confirmatory aggregation; failed attempts remain auditable.

A run with a protocol deviation is not silently repaired after official-test inspection. The deviation is logged and handled according to the readiness-gate decision made before aggregate analysis.

## 12. Workbook mapping

`MORE_HD_master_confirmatory_240runs.xlsx` is the canonical aggregation template. Sheet `01_Run_Summary` contains one row per `run_uid`; `02_Run_Config` records seed propagation and split provenance; `12_Seed_Aggregation` stores the five-seed summaries; `13_Paired_Comparison` stores seed-matched MORE-HD versus MORE-HD-C differences. Detailed history and official-test prediction sheets are append-only during consolidation.

## 13. Confirmatory boundary

Seed 42 and any other development run are PILOT ONLY. They may be used for debugging, convergence inspection, timing, and protocol locking, but they are excluded from confirmatory means, standard deviations, confidence intervals, effect sizes, tests, and publication tables presenting final performance.
