# MLC-02: models for the 14-condition classifier

Prospective, 10 November 2026. Frozen before any fit or score. It runs on the MLC-01 dataset
(`outputs/mlc01_condition_dataset_20261110_v4/`, the final build; protocol and Amendments 1 to 3 in
`docs/mlc01-condition-dataset-protocol-2026-11-10.md`). The roles, label states and evaluation
rules of that document bind this one and are not repeated. No Challenge test group and no
EchoNext record is opened.

## Question

Which model best names the 14 conditions at a hospital it never saw, and does it beat the
project's earlier approach (logistic regressions on pooled xECG embeddings)? The comparison of
frozen encoders, a network trained from scratch and a fine-tuned encoder is secondary.

## What the encoders saw before

From the published source lists, checked on 10 November 2026:

| Encoder | Pretraining data | Overlap with this dataset |
|---|---|---|
| xECG | CODE, Chapman and Ningbo, INCART | Chapman and Ningbo waveforms, in train and validation |
| ECG-JEPA | Chapman and Ningbo, CODE-15% | Chapman and Ningbo waveforms, in train and validation |
| ECG-CPC | HEEDB | None |

No encoder used labels, and none lists PTB-XL or SPH, so both test sets are outside every
encoder's pretraining data. The Chapman-Ningbo validation source and its leave-one-family-out
arm are not, for xECG and ECG-JEPA; reports mark them. Whether the PhysioNet release used for
pretraining holds exactly the Challenge records is unverified.

## Stage 1: inputs (no score)

1. Waveform cache on local disk (`/var/tmp/janetrivera/mlc02/`): first 10 s, twelve leads in the
   order I, II, III, aVR, aVL, aVF, V1 to V6, 500 Hz, millivolts, one array per role and source,
   aligned to the manifest and checked against its first-10-s hash.
2. Pooled features for every record, each encoder with its own documented input pipeline:
   xECG (1,024), ECG-JEPA (768), ECG-CPC (512), and the 130 measurements. A record whose
   measurements cannot be computed (flat lead) keeps missing values.
3. Integrity: new features equal the cached ones of HEADS-01, PR-74, SCREEN-01 and Avance 4 on
   the shared records within floating-point tolerance, and KEY-04's SPH macro AUROC (0.9608) is
   reproduced from its cached features.

## Stage 2: candidate models

Every model outputs 14 probabilities and is trained with undefined labels masked. Standardizers
and every other fitted transform use training records only.

| Id | Model | Recipe |
|---|---|---|
| P0 | Predecessor baseline | xECG pooled, standardized, rows L2-normalized; one L2 logistic regression per condition, no class weights; C from {0.01, 0.1, 1, 10, 100} per condition |
| L-J, L-C | Same on ECG-JEPA and on ECG-CPC | As P0 |
| G-M | Gradient boosting on the 130 measurements | One histogram gradient-boosting classifier per condition; learning rate {0.05, 0.1}, leaves {15, 31}, early stopping on validation |
| H-X, H-J, H-C | Small network on one encoder's pooled features | Standardized input, one hidden layer of 512 units, dropout 0.3, AdamW, learning rate {1e-3, 3e-4}, weight decay {1e-4, 1e-2}, batch 256, up to 100 epochs |
| H-All | Same on the three encoders and the measurements together | As H-X; missing measurements imputed with the training median plus a missing flag |
| S-R | xresnet1d101 trained from scratch | 100 Hz, random 2.5 s crops, AdamW with a one-cycle schedule, peak rate 1e-2, weight decay 1e-2, batch 128, 50 epochs (Strodthoff et al., 2021); evaluation is the mean over the four consecutive 2.5 s windows of the 10 s |
| F | Fine-tuned encoder | The encoder of the best of H-X, H-J, H-C on validation; 14-output head started from that head; layer-wise rate decay 0.75; top rate {1e-5, 3e-5}; up to 5 epochs; mixed precision |
| E | Average | Mean probability of the three best candidates above that are not variants of each other |

Rules for all:

- Each grid value and each epoch is chosen on the validation selection metric of MLC-01 (mean
  over validation sources of the within-source macro AUROC). Per-condition choices (C in P0, L-J,
  L-C; G-M settings) use that condition's within-source AUROC averaged over validation sources.
- H, S-R and F run with three seeds (20261110, 20261111, 20261112) at the chosen setting. The
  candidate's prediction is the mean probability of its three seeds; each seed is also reported.
  Grid search uses the first seed only.
- F runs only if a real profile forecasts at most 12 GPU hours for the whole grid and seeds; if
  not, it runs with the top rate 1e-5 alone, and the report says so.
- No augmentation beyond the random crop of S-R. No class re-weighting. These are fixed so that
  the comparison does not depend on tuning effort we cannot afford for every model.

## Stage 3: selection, then tests

1. Write and commit the selection record: the validation table of every candidate (per source
   and mean), the chosen settings, and the **selected model**, which is the candidate with the
   highest selection metric. Ties within 0.001 go to the simpler model, in the table's order.
2. Fix on validation, for every candidate: the per-condition threshold giving 90% specificity on
   the validation records where the condition is defined, and a per-condition Platt calibration.
3. Only then compute test predictions, once, for all candidates, on PTB-XL fold 10 and the SPH
   evaluation half. All candidates are reported; only the selected model against P0 on the SPH
   evaluation half is confirmatory.
4. Secondary analysis on the SPH calibration half: thresholds and calibration refitted there,
   then applied to the SPH evaluation half, to measure what local data adds.
5. Leave-one-family-out over train and validation for P0, L-C, H-All and S-R (one seed).
6. Audits listed in MLC-01: source identification from each feature set (logistic regression on
   validation records, five-fold), one condition against several, sex, flat leads, likelihood
   below 50 undefined, window-mean for longer records (SPH, where recordings exceed 10 s),
   space A beside KEY-04, derived top-1, top-2 and exact-set accuracy.

## Reading the result

- Confirmatory: selected model minus P0, macro AUROC over 14 conditions, SPH evaluation half,
  paired patient bootstrap (2,000). Better if the interval's lower end is above zero; if the
  interval includes zero, we report no demonstrated gain and keep P0 as the reference model.
- A model is not called usable for a condition whose test AUROC interval lower end is below 0.80
  or which has fewer than 100 positives in that test set; it is reported as it is.
- SPH and PTB-XL fold 10 were scored in earlier experiments for other targets. These results are
  the best evidence available without new data, not an untouched final test.

## Resources

GPU 0 has about 10 GB free and GPU 1 about 4.7 GB; both are shared with other servers. Jobs
check free memory, take the per-GPU lock, and profile before a full run. Fine-tuning runs on
GPU 0. Caches live on local disk; outputs go to `outputs/mlc02_condition_models_20261110/`.

## Deliverables

New modules and scripts only, with tests; features and receipts; the committed selection record;
predictions for every candidate and role; `docs/mlc02-condition-models-results-2026-11-10.md`
written from the outputs; queue and backlog updates.

## Addendum 1 (10 November 2026, before any fit or score)

The scoring code was written and tested on synthetic data and on the real labels with random
scores before any model existed (`docs/mlc02-evaluation-code-2026-11-10.md`). Its readings of
the rules are adopted as written there. Three choices it left open:

1. Calibration is reported as the slope of the joint fit and the calibration-in-the-large
   intercept (slope fixed at 1). The joint-fit intercept is listed beside them.
2. Test-set calibration, Brier score and the top-1 and top-2 views use the model's raw
   probabilities. The same quantities after the validation-fitted Platt calibration are listed
   beside them.
3. The top-1 and top-2 views use records with exactly one positive condition and allow undefined
   entries among the others (702 records at PTB-XL fold 10, 3,882 at the SPH evaluation half).
   The stricter reading, all 14 defined, is listed beside it (204 and 3,434); it can never
   include a left anterior fascicular block record.

Logistic fits run with the default OpenBLAS kernel, which reproduces KEY-04's coefficients
exactly on this machine; the receipt records the kernel and thread settings. The features and
waveform cache cover all 100,158 candidate records; models use the 99,167 included in the final
dataset build (`included_v4`).
