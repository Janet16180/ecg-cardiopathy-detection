# Experiment 041: a calibrated CPU screening baseline

Prospective protocol, 18 October 2026. The user authorized autonomous multi-agent research with
existing data and no cardiologist. The local GPU driver is unavailable. This study uses the local
Avance feature caches and makes no downloads. Freeze this document in Git before fitting or scoring.

## Question and existing evidence

Does a fixed nonlinear model improve ECG annotation screening over a matched logistic regression
when raw signal measurements, fitting patients and a separate normal-only calibration set are fixed?

The locally saved Avance 3 logistic model reports fold-9 AUROC 0.9280305990508397 and average
precision 0.9511171977377277. Its notebook is absent on this branch but available in the local
`avance3/baseline` Git tree. Historical notebook outputs already favored gradient boosting; this is
an explicitly development-only successor, not a new discovery that nonlinear models can help.
Earlier preprocessing and supervised selection used all training folds before cross-validation;
fold 9 also informed earlier feature choices. Neither fold 9 nor PTB-XL fold 10 is untouched.

## Inputs, targets and patient roles

- Raw measurements: `outputs/avance2_fusion/caracteristicas_primarias_ee39c8f6f7.csv`.
- Existing identity and labels: `outputs/avance2_fusion/ptbxl_caracteristicas_seleccionadas.csv`.
- Integrity comparator: `outputs/avance3_baseline/baseline_logistica.joblib` and
  `baseline_metricas.json`.
- PTB-XL metadata and statement dictionary are read locally through the existing file symlinks.
  Verify their labels against the existing `superclass_label` helper.

Preserve Avance's existing superclass target exactly: positive if any MI, STTC, CD or HYP superclass
is present; negative if NORM is the only diagnostic superclass; undefined without a diagnostic
superclass. Ignore likelihood when determining membership, following the existing helper. This is
also the standard-label target described by pipeline v2. It differs from the older strict project
rule that accepts only NORM/SR as negative. Do not relabel rhythm findings, add clinical definitions,
or describe the target as confirmed heart disease.

Fit on official folds 1–7, calibrate on fold 8, evaluate on fold 9. Assert unique ECG IDs, exact
identity/label alignment and no patient overlap among these roles. Fold 8's positive labels are
used only for role accounting; thresholds use its target-negative rows only. Do not fit or select
anything on fold 9. No fold-10 scoring or new waveform reads; whole input CSV hashes necessarily
cover their historical fold-10 rows. Filter out fold 10 before model inputs are constructed.
Challenge test and EchoNext test remain closed; neither source is opened.

Use the 130 numeric raw waveform measurements, in cache order, excluding `ecg_id`, `status`,
`implausible` and `flat_leads`. Check the feature count and numeric schema explicitly. No selected
Avance transformations, PCA, supervised feature filtering, age or sex enters either new model.
Convert nonfinite values to missing. Preserve missing values and existing cached plausibility
handling; no new clipping or record exclusion follows from development performance. This tests
cached measurement modeling, not independent validation of the measurement algorithm.

## Integrity gate before new fits

On the selected-feature cache, reproduce the saved baseline's fold-9 AUROC and average precision
within absolute 1e-12, using its ordered saved features. Refit the historical logistic model on
folds 1–8 with its exact settings (`C=0.1`, `max_iter=5000`, other original defaults), and reproduce
the saved fold-9 probabilities within absolute 1e-12. Stop and record any failure. This integrity
calculation is separate from the primary matched comparison and does not open fold 10.

## Fixed models and referral rule

One fit per arm, no search, tuning, early stopping or additional seed:

1. Logistic regression: training-only median imputation with missingness indicators and empty-column
   preservation, then training-only standardization, then L2 logistic regression, `C=0.1`,
   `max_iter=5000`, `tol=1e-8`.
2. Histogram gradient boosting: native missing values, `max_iter=200`, `learning_rate=0.05`,
   `max_leaf_nodes=15`, `min_samples_leaf=30`, `l2_regularization=10`, `early_stopping=False`,
   `random_state=20261018`; other installed scikit-learn defaults.

For each fitted arm, score the fold-8 normals. Let m be their count and k=floor(0.05 m).
The threshold is the (k+1)-th highest normal score. Refer strictly above it, so ties are conservative.
Record the threshold, normal count and achieved calibration referral rate. Calibration determines
workload, not a sensitivity target. It is a PTB-XL simulated-site rule and cannot be transferred to
a university population without evidence.

## Primary comparison and decision

Primary endpoint: fold-9 AUROC difference, boosting minus matched logistic regression.
Use the shared `paired_auroc_difference`, 2,000 whole-patient draws, seed 20261018, reporting
2.5/97.5 percentiles and skipped single-class draws.

Return `advance_development` only if all four conditions hold:

- primary point difference is at least 0.01;
- its paired 95% lower bound is strictly positive;
- boosting's sensitivity at its frozen 5% referral threshold is no more than 0.02 below logistic's;
- boosting refers no more than 8% of fold-9 target-negative ECGs.

Otherwise return `do_not_advance`. The 8% guard allows calibration sampling variability but does
not claim a guarantee on a new site. A pass justifies further development and external evaluation;
it does not replace pipeline v2 or authorize a final test.

Secondary summaries: each arm's AUROC, average precision, Brier score, fixed-threshold sensitivity
and specificity with 2,000 whole-patient intervals, and paired sensitivity difference at the frozen
thresholds with the same draw seed/convention. Use shared bootstrap helpers. Count fold-9 positive
misses by existing MI, STTC, CD and HYP flags, allowing overlap; do not interpret them as exclusive
diagnoses or fit extra heads. No age analysis or anatomical/temporal localization claim.

## Resources, artifacts and closure

CPU only; at most four numerical threads for this run and eight across active agents. Before
launch, inspect model defaults and test scientific invariants on artificial data. No full neural
training is authorized by this CPU protocol. Operational budget: two wall hours for this fixed
comparison, including intervals, with no rerun to improve a score. Preserve failed attempts.

New reusable code, a thin runner and scientific invariant tests live in new files. Do not change
any pinned sources, dependency files or original outputs. Write an exclusive new directory
`outputs/experiment041_cpu_screening_v1/` with saved models, feature order, thresholds, aligned local
predictions, counts, timing, decisions, exact source/input/protocol SHA-256 bindings, versions and
Git revision. Stored paths use `ecg_experiment.paths.to_stored`.

The coordinator running the experiment writes `docs/experiment-041-cpu-screening-results.md`
from these outputs, updates both queues and the ranked backlog, and obtains an independent
implementation/arithmetic review. Only source, protocol and aggregate reports may enter Git.
Clinical localization remains unsupported without native regional annotations; failure to establish
that endpoint is not evidence that disease-presence screening is impossible.
