# Corrective PR 74 evaluation: label-first age cohorts

Committed before new feature extraction, model fitting or scores. This study evaluates two newly
assembled supervised cohorts on reused development data. It does not open the original Challenge
test groups or EchoNext test, establish clinical diagnostic accuracy, or select a clinical pipeline.

## Question and primary comparison

Does the strict 18–25 training selection improve sensitivity over an equally sized general-age
selection matched exactly within source, label-evidence rank and binary target, when both use the
same fixed model and a separately fitted 5% normal-referral calibration rule at SPH? There are 770
admitted 18–25 records (506 negatives, 264 positives). These two primary training selections use the
same frozen admission/cleaning and both classes. Overlap between training selections is allowed and
reported. Matching reduces these measured composition differences but cannot establish a causal
effect of age. No age-subgroup evaluation or target redefinition is introduced.

The completed 25,000-record age-preferred and all-age selections are secondary. Their broad age
distributions and differing source/class composition prevent calling them student cohorts. This
amendment precedes all new extraction/fitting/comparative scores. The user was offered the primary
scope choice; after more than two minutes without an answer, the parent stated the recommended
strict-age comparison as the working assumption. Any later steering retains completed arms.

Primary score: the fixed arithmetic mean of HGB and frozen-xECG logistic scores (weight 0.50 each).
Primary contrast: strict 18–25 minus matched-general sensitivity on the exact existing SPH evaluation patients.
Practical development advantage requires a point gain of at least 0.02 and a positive lower endpoint
of a two-sided 95% paired whole-patient bootstrap interval (2,000 draws, seed 20261028). Otherwise the
prespecified advantage is not established. No automatic winner or pipeline promotion follows.
This is conditional on the fitted models and fixed calibration sample, not total retraining or
threshold-calibration uncertainty. Both cohorts and all results are reported regardless of direction.

HGB and xECG components are secondary. Matched original PTB-only HGB/xECG/50:50 scores are secondary
anchors. AUROC, average precision, sensitivity, specificity, precision, referral fraction and the
four confusion counts are descriptive secondary metrics; no multiple-comparison winner is selected.

## Admission and predecessor integrity

Require completed cohort receipts and all unchanged input/policy/source/artifact hashes. Cohort
training records, known patient IDs and canonical/full-signal hashes must be disjoint from the
covered PTB folds 8–10, original Challenge held-out assignments and existing SPH references. Unknown
patient IDs limit patient-independence claims. References do not audit the still-closed EchoNext
test or every possible alternative waveform window. No original Challenge held-out model score or
feature is computed. Released xECG pretraining included Chapman/Ningbo; that historical exposure
cannot be removed by this downstream split.

Before any new model fit, reproduce existing SCREEN-01/041/042 saved development score metrics,
normal-budget thresholds, integer confusion counts and the known fixed 50:50 mixture. Verify
artifact hashes against completed predecessor receipts and exact row/role alignment. Reproduce
32 admitted PTB training measurement-cache rows through the frozen extractor/serialization and
32 admitted PTB xECG cached representations under the official release (atol=rtol=1e-5). A failed
integrity gate stops the study. Predecessor data are trusted local artifacts; PR 74's joblibs,
new randomized split, provisional label taxonomy and 111-feature selection are not used.

## Fixed recipes

- HGB: the existing 130 raw ECG measurements, unchanged `measurement_extractor042.extract_array`,
  float64 arithmetic and the historical %.6g serialization. Reuse verified cached PTB/SPH measures;
  extract only admitted Challenge training windows. Native missing-value handling; no feature
  selection, imputer, rescaling or label-informed transform. Fixed HistGradientBoostingClassifier:
  max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30,
  l2_regularization=10, early_stopping=False, random_state=20261018. No class weighting or tuning.
- xECG: the released frozen `riccardolunelli/xECG_base_model_v1` checkpoint SHA-256
  `812dec69ac0fbf13f39e50bde4f35f85435a66966d8785baec65c2b5c70e722c`, vanilla backend,
  eval mode, FP32, no gradients or encoder updates. Preserve the official float64-to-100-Hz input
  conversion and 1,024 pooled features. PTB and SPH use verified predecessor features; only admitted
  Challenge training windows need extraction. Fit StandardScaler and L2 logistic regression solely
  on each training cohort, using the existing `full_development.fit_logistic`: C=0.01, lbfgs,
  max_iter=5000, tol=1e-8, intercept, no class weights, random_state=42. Fail on non-convergence.
- Fusion: 0.5 times each positive-class model score, fixed before evaluation. These are uncalibrated
  model scores, not population disease probabilities. No fusion-weight selection occurs here.

Training and cached feature alignment use record IDs and manifest hashes, never file row position
alone. PTB follows its original raw float64 path; Challenge uses checksum-verified canonical strict
10-second, 12-lead, 500-Hz mV windows and their admitted hashes. Longer-record labels are excluded
upstream. Quality failures at inference produce abstention before prediction.

## Resource gate

GPU0 is shared and has two other VLLM allocations across the host's GPUs; never modify those jobs.
Use the project GPU lock, no concurrent project GPU stage, at most two CPU threads and an initial
xECG batch size of 16. Profile admitted training records through actual decoding, measurement
extraction and GPU inference. Require at least 4 GiB initial free GPU memory, less than 3 GiB added
peak allocation, a conservative two-times measured full extraction estimate below two hours, and
new feature output below 1 GiB. A profile failure stops full extraction and is reported before any
protocol revision or alternative recipe. No raw-data/checkpoint downloads or backbone fitting.

## Calibration and evaluation roles

Use exact existing SCREEN-01 SPH row identities: 6,968 normal calibration ECGs and 10,434 evaluation
ECGs (3,584 positives, 6,850 negatives; 10,116 patients). No SPH record fits a model, scaler or feature
selection. Only normal calibration scores set the threshold: sort descending, index
floor(0.05*n_normal), and refer on strict `score > threshold`, so ties cannot exceed the budget.
Use existing PTB fold 8 normal calibration and fold 9 development evaluation as a secondary site.
PTB fold 10, Challenge test and EchoNext test remain closed for scoring.

The 5% budget concerns local normal ECGs, not total population referral fraction. Evaluation normal
referral may differ from 5%. Matrix counts include every admitted evaluation record; zero-denominator
precision is null. Preserve fixed operating thresholds independently of descriptive slider sweeps.
Each sweep is an ascending list of score thresholds, with monotone counts and exact integer totals.
Showing a sweep cannot authorize selecting the deployed threshold on evaluation data.

## Outputs and decision reporting

Standalone stage runner: qualify, profile, extract, fit/evaluate and aggregate export, with explicit
local input/output roots and completed-stage hash gates. Source, protocol, checkpoint, cohort,
predecessor and output hashes are captured before execution and rechecked before completion.
Training artifacts, features, models and patient-level predictions remain local. Git holds source,
tests, protocols/results and an aggregate UI artifact only. No patient-level data leave the machine.

`screening_evaluation.json` schema 1 contains model/cohort IDs, development scope, uncalibrated-score
interpretation, per-site fixed calibration rule, integer TP/TN/FP/FN, denominator counts, descriptive
threshold sweeps and bootstrap intervals. Its completion receipt binds its SHA-256. The deployed
model is never switched because a slider is moved or because one development cohort scored higher.
The results report states the prespecified decision, all failed gates, provenance limitations and
the absence of new untouched clinical validation. Follow-up ideas enter the backlog, not an
unapproved expensive run.
