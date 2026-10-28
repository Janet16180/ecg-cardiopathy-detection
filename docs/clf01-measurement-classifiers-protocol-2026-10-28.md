# CLF-01: RBF SVM, MLP and logistic regression against the frozen HGB at SPH

Committed 28 October 2026 before any score. Follow-up of
[HEADS-01](heads01-token-superclass-results-2026-10-28.md), same branch. Reused development data only;
no Challenge test group, EchoNext test, PTB fold 10 or final frozen test is opened.

## Question

PR 74's application and notebook `notebooks/Avance4.8.ipynb` rank gradient boosting first and the RBF SVM
close behind, using internal PTB-XL validation only (111 selected features; CV AUROC GB 0.955, SVM 0.951,
MLP 0.949, LR 0.935; fold 9 GB 0.943, SVM 0.939, MLP 0.940, LR 0.928). On SPH, the external hospital, at
the project's operating point, do the tuned RBF SVM, the tuned MLP or the logistic baseline beat the frozen
HGB recipe on the same measurement input?

## Training and models

All models are fitted once on the all-age 25,000-ECG cohort (7,393 negatives, 17,607 positives) and its
130 frozen raw measurements (`cpu_screening041.RAW_FEATURES`), with no feature selection.

- Reference HGB: the frozen `HGB_CONFIG` recipe with native missing values (the all-age blend's component).
- Challengers share one preprocessing pipeline fitted on the training cohort only:
  `SimpleImputer(strategy="median", keep_empty_features=True)` then `StandardScaler()`. No training column
  is entirely missing; 85 of 130 have some missing values. Avance 4.8 has no scaler inside `SVM_BASE`; its
  features were standardized upstream in Avance 2 with a training-only `StandardScaler`, which this
  pipeline reproduces on the new input.
- SVM: `SVC(kernel="rbf", C=3, gamma=0.0056, probability=True, random_state=42)`, the Avance 4.8 tuned
  values, not retuned (gamma was tuned for 111 standardized features and is kept as is). Its standalone
  score is `decision_function`, as in Avance 4.8; `probability=True` only adds libsvm's internal
  cross-validated Platt scaling, whose `predict_proba` is used for the blend.
- MLP: `MLPClassifier(hidden_layer_sizes=(128,), activation="relu", alpha=3.6929e-05,
  learning_rate_init=3.6825e-04, batch_size=64, early_stopping=True, max_iter=500, random_state=42)`.
- LR: `LogisticRegression(C=0.1, max_iter=5000)`, the Avance 4.8 baseline.

Fit-time profile on training rows only, before this commit (no score): the SVM pipeline took 0.3, 0.3 and
1.3 seconds on 3,000, 6,000 and 12,000 rows. A full 25,000-row fit is far below the 30-minute limit, so no
subsample is used.

Blends: each challenger replaces HGB in the fixed 0.5/0.5 blend with the frozen all-age xECG logistic
(`full_development.fit_logistic` on the saved 1,024 mean-pooled features). The blend uses each model's
positive-class probability (SVM: `predict_proba`). The reference blend is the all-age blend.

## Operating point, comparisons and decision rule

SPH roles are unchanged: 6,968 calibration normals set each model's threshold (descending sort, index
`floor(0.05 * 6968)`, refer on strict `score > threshold`); 10,434 evaluation ECGs (3,584 positive, 10,116
patients) are scored. PTB fold 8 normals and fold 9 are a descriptive secondary site.

Primary: SVM minus HGB SPH sensitivity, each model alone. The MLP minus HGB and LR minus HGB contrasts use
the same rule. Paired whole-patient bootstrap of the frozen referral decisions
(`intervals.two_class_resamples`, 2,000 draws, seed 20261028, skipped draws reported). Bonferroni across
the three challengers: two-sided 98.33% percentile intervals (1 − 0.05/3); 95% also reported.

Decision rule per challenger, fixed now: it beats HGB if the lower 98.33% bound is above zero; HGB beats it
if the upper bound is below zero; otherwise no difference is established. The specificity change
(challenger minus HGB, evaluation normals) and paired AUROC differences (`intervals.paired_auroc_difference`,
same draws and seed, 95%) are reported descriptively and do not enter the rule. The same rule is applied
descriptively to each challenger blend minus the all-age blend. No result promotes a model.

## Integrity check before anything new

Refit the all-age HGB and xECG logistic from the saved features; their SPH and PTB scores must equal the
saved PR 74 predictions to 1e-12, and the recomputed all-age HGB and blend SPH points (threshold, confusion
matrix, AUROC) must equal the saved evaluation exactly. A failure stops the run before any challenger fit.

## Resources and outputs

CPU only, no GPU lock, two numerical threads (`threadpool_limits(2)`, OMP/OpenBLAS/MKL at 2).
Local outputs: `outputs/clf01_measurement_classifiers_20261028/` (models, per-record predictions, receipt).
Committed: this protocol, source, test, `reports/clf01-measurement-classifiers-2026-10-28/aggregate.json`,
the results report, queue files and backlog. SPH never enters fitting, imputation or scaling.
