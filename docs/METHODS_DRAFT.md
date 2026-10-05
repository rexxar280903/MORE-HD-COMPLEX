# MORE-HD-C — Methods, Comparison with MORE, and Limitations (paper draft)

Status: draft text for the IEEE paper, written before any confirmatory result
was produced (2026-10-04/05). It is the authoritative description of the
protocol for publication purposes and **supersedes the undergraduate thesis
documents (BAB 1, FRD-09)**, which are kept only as historical records. Every
statement below is implemented in the repository (`core/`, `main_*.py`) and
fixed by `MORE_HD_STATISTICAL_ANALYSIS_PLAN.md` (v2.0) and
`Pseudocode_2x3_manual_runs.md`. Numbers that will come from the confirmatory
experiment are marked `[RESULT]`.

---

## 1. Research questions

1. Does adding a trainable RZ rotation after each RY rotation in the variational
   layers of MORE-HD (MORE-HD-C) change classification accuracy, macro-F1 and the
   separation of quantum labels on cumulative MNIST tasks with K = 3..10 classes?
2. Is any change attributable to access to complex-valued states rather than to the
   larger number of trainable parameters (targeted ablation)?
3. How do MORE-HD and MORE-HD-C compare with the original MORE classifier
   (Wu et al., 2023) when all three are trained and evaluated under one identical
   protocol (in-protocol reproduction), and how do they relate to the accuracies
   reported by Wu et al. (descriptive)?
4. How do the three classical feature representations (PCA, Hu moments, Zernike
   moments) interact with the circuits?

## 2. Models

All circuits read eight input channels per sample. MORE-HD and its variants use
**10 qubits: 8 data qubits and 2 readout qubits; there is no additional ancilla
qubit** (readout qubits are wires 8 and 9). The readout is the vector of the 15
non-identity two-qubit Pauli expectation values on the readout pair, in the fixed
order `IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ`.

| Code | Model | Variational layer | Trainable | Fixed | Role |
|---|---|---|---:|---:|---|
| A | MORE-HD | 3 × (RY on 10 qubits + CNOT block) | 30 | 0 | primary baseline |
| D | MORE-HD-C | 3 × (RY then RZ on 10 qubits + CNOT block) | 60 | 0 | proposed model |
| B | MORE-HD-60P | 6 × (RY + CNOT block) | 60 | 0 | ablation: real-valued capacity |
| C | MORE-HD-C-FixedRZ | 3 × (RY + frozen non-zero RZ + CNOT block) | 30 | 30 | ablation: complex access at 30 trainable |
| M | MORE-REPRO | MORE QCNN (Wu et al., 2023) | 91 | 0 | in-protocol reproduction of MORE |

Encoding (A–D): `RY(x_i)` on data qubit `i`. CNOT block (A–D): ring on the data
qubits (`i → i+1`, `7 → 0`), last data qubit to both readout qubits, and readout
qubit 8 → 9.

**Claim wording for the RZ modification.** With RY and CNOT gates only, every
amplitude of the state is real, so the expectation of every Pauli operator with
an odd number of `Y` factors is identically zero: six of the 15 readout
observables (`IY, XY, YI, YX, YZ, ZY`) are structural zeros of MORE-HD. Adding RZ
removes this real-state restriction, so these six observables **can** take
non-zero values; whether they do, and by how much, is measured empirically
(Section 6.3). RZ does not guarantee that all 15 dimensions are active, and an
activation of the Y-odd observables is not by itself evidence of an accuracy
benefit.

**MORE reproduction (M).** The MORE circuit is rebuilt gate by gate from the
public code (`github.com/Jindi0/MORE`, commit 867d194, `model.py::build_qcnn`):
8 qubits, a Qiskit `ZFeatureMap` with two repetitions, a QCNN ansatz of three
convolution/pooling stages with 91 parameters in Qiskit's parameter order, and a
single readout qubit (qubit 7) measured with X, Y and Z. MORE scales PCA
features to `[0, 2π]`; our features are scaled to `[0, π]`, so M encodes
`2·x`, which reproduces MORE's encoding range exactly. Parameters are
initialised from `U[0, 1)` as in qiskit-machine-learning. The loss adjuster R
of MORE is not used in any model (Section 8). A unit test shows that our
implementation of M equals MORE's Qiskit circuit to `< 1e-12` for random inputs
and parameters.

**Parameter counts.** MORE-HD uses 30 and MORE-HD-C 60 trainable parameters,
both fewer than the 91 of MORE. Comparisons between A and D change the
parameter count together with the gate set; the ablation (B, C) separates the
two.

## 3. Data

MNIST (the torchvision files, MD5-verified; indices refer to the original order).
Tasks use the cumulative classes `[0..K-1]`, K = 3..10. For each master seed a
split manifest fixes, per digit, 1,000 training and 100 validation samples drawn
without overlap from the official training pool and 200 official-test samples
from the official test pool. Splits are **paired** (identical sample identities
for all models and feature methods of a seed) and **nested** (the samples of an
existing digit do not change when K grows). Confirmatory seeds are
101, 202, 303, 404 and 505; seed 42 is used only for smoke tests and pilots.
All random choices derive from `SHA256("<seed>:<namespace>")` sub-seeds.

## 4. Feature representations (all produce exactly 8 channels)

* **PCA**: 8 components of the flattened `[0, 1]` images, exact SVD, fitted on
  the training split only.
* **Hu moments**: the 7 Hu invariants of the grayscale image (OpenCV
  definitions), stabilised with `h' = −sign(h)·log10(|h| + 1e-30)`; the eighth
  channel is a constant 0 (so the eighth data qubit receives `RY(0)`). **Only
  the Hu representation uses a padding channel.**
* **Zernike moments**: magnitudes `|Z_nm|` of eight orders
  `(2,0), (2,2), (3,1), (3,3), (4,0), (4,2), (5,1), (5,5)` on the unit disk
  centred at the geometric image centre with radius 14 px (pixels outside the
  disk ignored, in-disk mass normalised as in `mahotas`). **Zernike provides
  eight genuine coefficients; it uses no padding channel.**

Each channel is MinMax-scaled to `[0, π]` with statistics fitted on the training
split only; validation and test values are transformed with the same scaler and
are not clipped (identical treatment for all three methods). Hu and Zernike
implementations agree with OpenCV and mahotas to within `1e-8` relative error.

## 5. Training procedure (two phases, following MORE)

**Inter-class correlation matrix.** Following MORE (`calc_class_rela`), the mean
feature vector of the first 100 training samples of each class is computed, the
mean squared difference between class means gives `MSE_ij`, off-diagonal entries
are `S_ij = MSE_ij / max_{a≠b} MSE_ab`, and `S_ii = −1`.

**Phase 1 — variational quantum clustering.** As in MORE, five training samples
per class are drawn at random (seeded, without replacement) and all
`C(5K, 2)` unordered pairs are used, without balancing or reweighting. The
loss is `mean_pairs(−S_ij · d_cos(v_i, v_j))`, where `v` is the readout vector
and `d_cos` the cosine distance (MORE's loss adds the constant 2, which does not
affect optimisation). Quantum labels are the per-class centroids of the readout
vectors of the five clustering samples, computed as in MORE: normalise each
vector, take the component-wise median, normalise again.

**Phase 2 — supervised learning.** Starting from the clustering parameters, the
loss is the mean cosine distance between the readout vector of each training
sample (all 1,000 per class) and the quantum label of its class. A sample is
classified to the class whose label has the smallest cosine distance (ties go to
the lowest class index, as in MORE).

**Optimiser.** COBYLA (SciPy 1.17.1), `rhobeg = 1.0`, `tol = 1e-4`, with the
same total number of objective evaluations for every model in each phase:
`max_nfev_clustering = 840` and `max_nfev_supervised = 1000`, fixed by a pre-registered
plateau rule on a seed-42 pilot (Section 9). The optimiser's returned point
`result.x` is used; validation data are monitored passively and never enter the
objective. The official test split is read only once per run, after training.

**Numerical safeguards.** Vectors with norm below `1e-10` have no direction:
their cosine distance is 1, their normalisation is the zero vector, and they are
excluded from centroid medians. For vectors above that norm all formulas are
identical to MORE.

**Simulation.** Exact statevector simulation in complex128 without shot noise.
The production engine (`core/circuits.py`) builds the columns of `U(θ)` once per
objective evaluation and evaluates all samples with fixed-shape matrix products;
it is verified against PennyLane `default.qubit` (unit tests and a per-run
self-check, maximum difference `< 1e-10`, observed `~1e-15`) and, for M, against
MORE's Qiskit circuit. Each process uses one BLAS thread.

## 6. Outcomes

1. **Primary classification outcomes:** official-test accuracy and macro-F1.
2. **Primary representation outcome:** `min_separation_ratio`, the minimum
   pairwise cosine distance between quantum labels divided by the regular-simplex
   bound `1 + 1/(K−1)`. The unnormalised value (`min_separation`) corresponds to
   the "Min. label distance" reported by Wu et al. (2023, Table I). **The simplex
   bound is a normalisation reference, not an attainable optimum**: expectation
   vectors of two readout qubits cannot point in arbitrary directions, so a ratio
   below 1 must not be described as a shortfall from an achievable optimum.
3. **Y-odd activation** (secondary): `yodd_norm_fraction`, the mean over samples of
   `‖v_Yodd‖² / ‖v‖²`, plus per-observable statistics and active counts at
   thresholds `1e-10 … 1e-2` (primary `1e-6`), computed on the full validation
   split at the final clustering and supervised parameters.

**Geometric claim (replaces the thesis claim that "9 active dimensions cannot
hold 8–10 classes").** K unit vectors can form a regular simplex in K−1
dimensions, so nine active dimensions are not geometrically insufficient for
ten classes. The relevant limitation of the real-valued MORE-HD is the region of
readout space that the circuit can reach; we therefore report separation as
`min_separation_ratio` and do not argue from the count of active dimensions.
Crowded quantum labels at large K were identified by Wu et al. (2023) as the
failure mechanism of MORE ("curse of density" in our earlier wording is our
adaptation of their observation).

## 7. Statistical analysis (summary of the frozen plan)

Five paired seeds per condition. Primary estimand: `Δ = Metric(D) − Metric(A)`
per feature method and K, summarised by mean, SD, median [IQR], 95% Student-t
CI, paired Hedges g, and an exact two-sided sign-flip test; McNemar's exact test
per seed is secondary (Holm within seed). Ablation contrasts A–B, A–C, B–D, C–D
use the same summaries with Holm correction over the four contrasts. Comparisons
with the MORE reproduction (A−M, D−M) use the same paired summaries and are
secondary. Runtime is descriptive only; computational cost claims come from a
controlled microbenchmark.

## 8. Comparison with MORE (Wu et al., 2023)

Two levels are reported and never mixed:

1. **In-protocol comparison (MORE-REPRO).** MORE's circuit is trained and
   evaluated with exactly our data, splits, features, pair set, correlation
   matrix, label rule, optimiser budget and official test set, five seeds per
   condition. Differences A−M and D−M therefore isolate the circuit/readout
   design. Neither MORE nor our models use MORE's loss adjuster R.
2. **Reported values (descriptive).** The MORE\R column of Table I in Wu et al.
   (2023) is the primary literature reference (MORE with R is context only).
   No statistical test is applied, because MORE's numbers come from a different
   protocol.

Differences between our protocol and MORE's published code, stated in the paper:
(a) we use all `C(5K, 2)` clustering pairs as described in the paper, whereas the
code caps pairs at 1,000 by an unseeded shuffle (active only at K = 10);
(b) the five clustering samples per class are drawn at random with a recorded
seed, as described in the paper, whereas the code takes the first five;
(c) the code fits a separate PCA and scaler on the test set and removes
outliers beyond two standard deviations from both splits; we fit on training
data only and remove nothing; (d) the code evaluates on the first 500 test
samples of the task (class-imbalanced); we use 200 per class; (e) our supervised
phase uses 1,000 training samples per class and a pilot-calibrated COBYLA budget,
which may differ from the settings behind Table I; (f) the correlation matrix is
computed in full precision (the code rounds to three decimals); (g) the loss
adjuster R is not used.

## 9. Pilot and budget determination (seed 42, excluded from all confirmatory analyses)

Smoke test: K = 3, PCA, all five models, budget 100. Timing pilot: K = 10, PCA,
all five models, budget 100; projected single-thread compute for the 450
confirmatory runs plus 90 Jalur B runs at the pilot cap was ≈32 CPU-hours
with validation monitoring at ≈9% of objective time, so full validation
monitoring was retained. Convergence pilot (cap 1,000): K ∈ {3, 10} ×
{PCA, Zernike} × {A, D, M} plus B at K = 10 × {PCA, Zernike}. For each run and
phase, N* is the first evaluation at which the best loss so far is within 2% of
the run's total improvement; the final budget per phase is the largest N* over
the pilot runs, rounded up to a multiple of 10 and capped at 1,000. A run counts
as not plateaued if it was stopped by the budget and its best loss still improved
by more than 2% of its total improvement over its last 30 evaluations; then the
cap is used and results are described as performance at equal
objective-evaluation budget. In the clustering phase every pilot run plateaued (largest N* = 834), giving a final budget of 840 evaluations. In the supervised phase run R048-S42 had not plateaued at the cap (its best loss still improved by slightly more than 2% of its total improvement over the last 30 evaluations), so the supervised budget is the cap, 1,000 evaluations.

## 10. Classical reference baselines

Chance (1/K), Nearest Centroid (Euclidean) and multinomial logistic regression
(L2, lbfgs, `C ∈ {0.01, 0.1, 1, 10, 100}` selected on validation accuracy, ties to
the smallest C) are trained on exactly the 8-channel arrays given to the
circuits (`X_*_scaled.npy`) and evaluated on the same official test samples.
They are reference points, not competitors; logistic regression is a linear
lower bound and does not measure the information content of a representation.

## 11. Limitations (to be kept in the paper)

* **Cumulative class sequence.** Each step K → K+1 adds one specific digit, so the
  effect of the number of classes cannot be separated from the identity of the
  added digit (for example, a change at K = 8 may reflect the similarity of digit
  7 to digit 1). Comparisons between models at the same K are unaffected because
  classes, splits, pairs and seeds are identical; trends across K and model × K
  interactions are reported as specific to the sequence `[0..K−1]` and are not
  generalised to arbitrary class sets.
* **Pair composition.** With five samples per class the share of same-class pairs
  falls from 28.6% (K = 3) to 8.2% (K = 10); following MORE we do not rebalance,
  so changes across K also reflect this composition.
* **Rotation invariance (digits 6 and 9).** Hu moments and Zernike magnitudes are
  invariant to rotation, so digits 6 and 9 (approximately a 180° rotation of each
  other) are hard to separate at the feature level. At K = 10 we report the 6↔9
  confusion rate and the accuracy with 6 and 9 merged as a sensitivity analysis,
  and we do not attribute 6↔9 errors of Hu/Zernike runs to crowded quantum labels.
* **Ablation depth.** B matches D in trainable parameters (60) but has six
  entangling blocks instead of three; B versus D is an equal-trainable-parameter
  comparison, not a depth-matched one.
* **Budget.** All models receive the same total number of objective evaluations;
  models with more parameters spend more of it building COBYLA's initial simplex
  (A/C 31, B/D 61, M 92 evaluations). Every pilot run reached its plateau in the clustering phase. In the supervised phase one pilot run had not plateaued at the 1,000-evaluation cap, so supervised results are described as performance at equal objective-evaluation budget rather than at convergence.
* **Simulation.** Results are exact noiseless simulations; no hardware noise or
  shot noise is modelled.
* **Runtime.** Production runtimes are descriptive; they depend on machine load
  and concurrency.

## 12. Deviations from the earlier thesis protocol (FRD-09)

The confirmatory study is a new experiment, trained from scratch. Relative to the
thesis protocol: (i) a three-way split with 1,000/100/200 samples per class
replaces the earlier train/test protocol; (ii) quantum labels follow MORE
(normalise–median–normalise over five clustering samples per class) instead of
the mean of raw outputs over all training samples; (iii) the correlation matrix
follows MORE (`MSE / max`) instead of a min–max rescaling to `[0.5, 1]`;
(iv) COBYLA budgets are calibrated by a pilot and equal across models;
(v) Zernike uses eight coefficients without padding; (vi) the comparison with MORE
uses an in-protocol reproduction and MORE\R values, not MORE with R;
(vii) software: Python 3.11, NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.9.1,
PennyLane 0.45.1 (reference simulator). Results of MORE-HD in this study are
therefore not protocol-identical to the thesis.
