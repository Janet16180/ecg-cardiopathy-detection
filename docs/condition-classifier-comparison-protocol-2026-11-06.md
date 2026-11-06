# KEY-04: matched multiclass and independent-label classifiers

Prospective 6 November 2026. The user requested JR naming, all prior results, a multiclass
comparison, and a deeper validation than the earlier rapid analysis. Freeze before any new fit
or performance score. Metadata feasibility found all 16 four-bit annotation combinations in
training (smallest 32 records); this accounting is repeated during qualification.

## Task and fixed roles

Use exactly KEY-01/02's 25,000 training records and 10,434 SPH evaluation records from the same
verified HEADS-01 caches. Four existing binary memberships are MI, STTC, CD and HYP. Their union
must reproduce the frozen proxy target. No label definition, age analysis, source admission,
clinical referral threshold or closed-test permission changes. SPH remains reused development.
Only pooled features are needed; no new encoder, GPU, raw waveform or external inference.

A true multiclass target is the existing membership combination, encoded as
MI + 2*STTC + 4*CD + 8*HYP. Zero is the existing normal proxy, not clinical certification.
This preserves co-occurrence instead of choosing a dominant diagnosis or excluding mixed cases.
The sixteen states are combinations of four annotation groups, not sixteen diseases.

## Models and fair comparison

1. Frozen whole-record condition prototypes from KEY-01, the primary reference.
2. `joint_logistic` (primary candidate): multinomial L2 logistic regression over the 16 states.
   Return four marginal scores by summing probabilities of states containing each group.
3. `independent_logistic` (secondary): four L2 binary logistic regressions, each group against
   every other training record. Return independent group probabilities, permitting overlap.
4. Training-majority joint class (descriptive dummy baseline for exact-state classification only).

All fitted classifiers use the same whole-record features as the prototype reference:
training-fitted StandardScaler, followed by per-row L2 normalization, in float64. No new features,
label-guided dimensionality reduction or clinical measurements are added. The primary comparison
is a learned readout against averaging in the same normalized representation, with the same labels.
Every final fit uses all 25,000 references. Probabilities are uncalibrated at the external site.

## Training-only selection

Three folds from `GroupKFold(3, shuffle=True, random_state=20261106)`. Known patients are grouped
by source plus patient ID; unknown-identity records use their unique record IDs. Keep repeated
known patients entirely within one fold. Require no group overlap. This does not establish
patient independence for the unknown-identity Challenge sources. Save fold assignments and counts.

Select C separately for each model family from (0.1, 1, 10), using mean fold macro one-versus-rest
AUROC across the four annotations. Ties choose the smaller C. Refit the scaler on each fold's
training rows; never reuse the full-cohort scaler in cross-validation. Use the same folds for
both families. Save every fold's scores and counts; do not select a model family from SPH results.
Logistic recipe: lbfgs, L2 (installed default l1_ratio=0), tol=1e-8, max_iter=5000, no class weights,
random_state=42. Treat convergence warnings or iteration-cap hits as failures, not usable results.
No further hyperparameter search after any SPH score.

## Primary comparison and uncertainty

Primary: joint_logistic minus frozen whole-record keys in SPH macro one-versus-rest AUROC.
Paired whole-patient bootstrap, 2,000 draws, seed 20261103, shared KEY-01 interval helper.
Declare a ranking improvement only if gain >=0.01 and the 95% lower bound >0. No clinical pipeline
promotion. The independent classifier's paired contrast is exploratory and carries no separate
promotion decision. Report per-group AUROC/AP, macro AP, and the earlier condition-versus-normal
metrics without replacing the primary harder task. No model or analysis is selected after scoring.

Secondary genuine-multiclass endpoints for joint_logistic: 16x16 confusion matrix, per-state support,
precision/recall/F1, exact-state accuracy, macro recall across states with evaluation support,
log loss, top-two state accuracy, and the train-majority dummy. The hard class is probability argmax,
not a fitted screening/referral threshold. Record absent evaluation classes and do not treat their
recall as zero in the supported-state macro. Prototype margins are not joint probabilities, so
there is no fabricated exact-state accuracy or log loss for the keys.

Error analysis is descriptive from these saved outputs: confusion concentration, incorrect-state
patterns, and per-group AUROC/AP. Compare exclusive versus co-occurring positive annotations
against the same frozen normal negative pool; disclose changed denominators and do not imply
independent clinical subtypes. No new patient-level ECG inspection or diagnosis.

## Reference-sampling sensitivity

After C is selected, freeze it. Run ten prespecified bootstrap refits, seeds 20261206–20261215.
Within each source family, resample the known-patient/unknown-record groups with replacement,
keeping every row of each sampled group. Each draw fits its own scaler, whole-record prototypes,
joint classifier and independent classifiers using the same drawn rows. All original records
remain from permitted training roles. Save sampled positions, models and every score.

Report all ten paired macro-AUROC/AP differences versus the same-draw prototypes, min/median/max,
and signs. Do not call ten-draw ranges confidence intervals or use them to choose a seed, model,
C or cohort. These probe sensitivity to reference sampling conditional on the fixed encoder and
SPH evaluation pool; patient uncertainty remains incomplete for unknown-identity sources.

## Integrity and execution

Verify direct predecessor input hashes and original saved artifact hashes. Recompute KEY-01/02
metric dictionaries and the KEY-01 primary patient interval exactly. Require the whole-record
prototype reconstruction to reproduce saved scores within 1e-12 and its metric dictionary exactly.
Check ordered training/query identities, label arrays, no train/SPH record overlap and known-patient
fold independence. Clinical reference-waveform audit results remain descriptive and do not filter data.

Two numerical threads, no GPU. Profile the first actual fold at C=1 for both model families, retain
these fits as their corresponding CV entries, and conservatively extrapolate remaining CV/full and
bootstrap fits. Gate full execution at a 7,200-second forecast; overall stage-boundary ceiling
10,800 seconds. Record actual fit/score times and convergence; preserve failures and refuse overwrite.
Commit scientific sources before fitting. Private output: `outputs/condition_classifiers_20261106/`.
Publish only aggregates, plots, code and reports; keep models/record scores and fold IDs local.

## Consolidated JR notebook and review

Create `notebooks/11-jr-ecg-condition-keys-and-classification.ipynb`, preserving the older artifacts.
Include all KEY-01/02 performance arms, cluster findings, KEY-03 reference audit and links to the
local waveform atlas, and the new classifier comparison, training-only selection, confusion/error
analysis and all ten reference-sampling draws. Execute sequentially. Label descriptive analyses,
clinical scope limits, original failures and the distinction between this work and actual JEPA/CroPA.

Before delivery independently recompute saved predictions from saved model parameters and features,
all performance metrics, the primary patient interval, CV selection, fold-group separation and
bootstrap group multiplicities. Verify notebook plots/tables against aggregates and source hashes.
Audit prior interpretations for unsupported causality, unique-patient counts, invented class labels,
clinical localization, and claims that style or arithmetic checks validate scientific performance.
Write the results report from executed outputs, update both queue documents and rank follow-ups.
