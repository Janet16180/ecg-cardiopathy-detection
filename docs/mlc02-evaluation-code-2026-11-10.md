# MLC-02 evaluation code

10 November 2026. The scoring code of MLC-02 was written and tested before any MLC-02 model exists, so that it is
fixed and verified on synthetic data and cannot be shaped by real results. It implements the evaluation rules
of the [MLC-01 protocol](mlc01-condition-dataset-protocol-2026-11-10.md) (section "Evaluation rules (binding for
MLC-02)", with its three amendments) and Stages 2 and 3 and "Reading the result" of the
[MLC-02 protocol](mlc02-condition-models-protocol-2026-11-10.md). No model was fitted on real data and no model
score was computed. The only run on real labels uses uniformly random scores, as a null check; every number in
that section comes from random scores and from label counts. No waveform, Challenge test-group record or
EchoNext record was opened: the code reads the structural and label-state columns of the final manifest,
`outputs/mlc01_condition_dataset_20261110_v4/manifest.csv`.

| File | Content |
|---|---|
| `ecg_experiment/mlc02_data.py` | Manifest arrays, roles, prediction files |
| `ecg_experiment/mlc02_evaluation.py` | Point metrics, selection metric, operating points, derived multiclass view, audits, selection record |
| `ecg_experiment/mlc02_bootstrap.py` | Shared patient-cluster resamples, intervals, paired comparisons, Holm adjustment |
| `tests/test_mlc02_evaluation.py` | 36 tests: hand cases, synthetic properties, real-label checks (skipped without the manifest) |
| `scripts/validation/check_mlc02_evaluation_null.py` | Null check on the real labels with random scores, and timings |

## What each function computes

Labels are `[records, conditions]` matrices of 1 (positive), 0 (negative) and NaN (undefined); probabilities
are aligned matrices with every value in [0, 1]. An undefined entry is left out of that condition's metrics and
is never a negative.

`mlc02_data`:

- `load_manifest(path)`: included records as arrays. These are `record_key`, `source`, `family`, `role`,
  `origin`, `group` (patient for PTB-XL and SPH, record for the Challenge sources), `sex` (`F`, `M`,
  `unknown`), `lead_flag` (a flat or missing lead in the first 10 s), `labels` (space B, `[n, 14]`),
  `labels_lik50` (space B with `lik50_` positives undefined) and `labels_a` (space A, `[n, 4]`). It raises if:
  - `mlc_labels.SPACE_B` is not the protocol's 14 conditions in order;
  - a key repeats, or a role is unknown;
  - a role holds other sources than the protocol's;
  - a validation or test record has another origin than fold 9 or calibration, fold 10, or its SPH half;
  - a group kind does not match its source, or a patient appears in two roles;
  - a state is not 1, 0 or NaN, or a likelihood flag marks anything but a PTB-XL positive.
- `role_data(data, role)` and `subset(data, mask)`: the records of one role, or of a mask, in manifest order.
- `save_predictions`, `load_predictions` and `prediction_path`: the prediction-file convention. A file is an
  `.npz` (no pickle) holding `format`, `record_key` (manifest order of one role), `condition`, `probability`
  (float64 `[n, 14]`), `model_id` and `seeds` (the seeds whose mean probability it holds). The path is
  `<dir>/<model_id>/<role>__seeds-<s1>-<s2>-<s3>.npz`. The loader raises if the format, the conditions, or the
  record keys (count or order) differ from what the caller expects, or if a probability is outside [0, 1].

`mlc02_evaluation`:

- `support` counts positives, negatives and undefined entries per condition. `enters_macro` marks conditions
  with at least 20 positives and 20 negatives defined in the set.
- `auroc_by_condition` and `macro_auroc`: AUROC (`sklearn.metrics.roc_auc_score`) on defined entries, and its
  mean over entering conditions. The macro is NaN if no condition enters.
- `condition_metrics(y, p, threshold)` and `metrics_table(labels, p, thresholds)` give, per condition:
  - positives, negatives, undefined, prevalence (positives over defined records), `enters_macro`, and
    `imprecise` (fewer than 100 positives);
  - AUROC, average precision, and Brier score;
  - `calibration_intercept` and `calibration_slope`, the `a` and `b` of the logistic recalibration
    `logit P(y=1) = a + b logit(p)`, and `calibration_in_the_large`, the `a` with `b` fixed at 1;
  - with a threshold, sensitivity and specificity at `p >= threshold`;
  - for the table, the entering conditions and the macro AUROC.
- `logistic_fit` and `offset_intercept`: unpenalized logistic fits by Newton's method, vectorized over many
  weightings at once. They give NaN when they do not converge (for example under perfect separation).
- `selection_metric(labels, p, sources)`: the mean, over validation sources, of the within-source macro AUROC.
  Besides the mean it returns:
  - the per-source macros and each source's entering conditions;
  - `excluded_sources`, the sources with no entering condition, which are left out of the mean;
  - `per_condition`, each condition's within-source AUROC averaged over the sources where it enters.

  It raises if any record is not from a validation source.
- `specificity_threshold(s)`: per condition, on validation records pooled over sources where the condition is
  defined, the lowest observed score whose specificity is at least 0.90. With `p >= t` positive, this is the
  highest-sensitivity point with false-positive rate at most 0.10, the same as the lowest
  `roc_curve(drop_intermediate=False)` threshold with `fpr <= 0.10`. The count of negatives is integer
  (`ceil(0.9 N)`), not a float product.
- `fit_platt` and `apply_platt`: per-condition `sigmoid(a + b logit(p))`, fitted by maximum likelihood on
  pooled validation records and applied to another set. `apply_thresholds` gives predicted sets.
- `single_condition_mask`, `top_k_accuracy`, `exact_set_accuracy` and `multiclass_view` give the derived
  multiclass view, under both readings (see below).
- `sex_masks`, `lead_masks`, `subgroup_tables`, `cooccurrence_labels` and `cooccurrence_tables` are the
  subgroup audits. The macro entry rule is applied within each subgroup.
- `source_identification(features, families, groups)`: cross-validated multinomial logistic regression
  predicting the source family. It uses five folds, stratified by family and grouped by patient or record,
  with the standardizer fitted on training folds only. It returns out-of-fold accuracy, balanced accuracy,
  macro one-vs-rest AUROC, log loss, the majority-class and 1/K chance references, and the confusion matrix.
  It was not run on real features.
- `select_candidate`, `selection_record`, `selection_markdown` and `write_selection_record` write the Stage 3
  record (`selection_record.json` and `.md`) with every candidate's per-source table, settings and the selected
  model. The record raises if candidates were scored on different validation labels.

`mlc02_bootstrap`:

- `cluster_resamples(groups, labels, draws=2000, seed=20261110)`: the resamples of one set, drawn through
  `intervals.patient_groups` and `intervals.patient_resample` and stored as uint16 record weights.
  - A draw is redrawn, from the same generator, when any condition that enters the set's macro has no positive
    or no negative.
  - The resamples depend only on groups, labels and seed, so they are computed once per set and reused for every
    model.
  - `fingerprint` is the SHA-256 of the weights, and `labels_sha256` that of the label matrix.
- `weighted_ranking` and `weighted_rates`: AUROC (ties count one half), average precision, sensitivity and
  specificity under integer weights. On every tested resample they equal the sklearn metrics on the
  concatenated resample.
- `bootstrap_metrics(labels, p, resamples, thresholds, calibration)` returns one value per resample and
  condition for every metric of `metrics_table`, and the per-resample macro AUROC. `bootstrap_models` runs
  several models on the same resamples in threads.
- `percentile_interval` and `interval_report` give percentile 95% intervals (2.5 and 97.5,
  `np.nanpercentile`, linear) of every metric and of the macro AUROC. The report also marks `usable`: the AUROC
  interval's lower end is at least 0.80, and there are at least 100 positives.
- `paired_comparison(first, second)` compares two models on the same resamples (it raises otherwise). It
  gives:
  - the macro AUROC difference with its interval, and `better` when the lower end is above zero;
  - the per-condition AUROC differences with intervals, two-sided bootstrap p-values, and their Holm
    adjustment (`holm_adjust`).

## Ambiguities and how they were read

Where a rule allowed more than one reading, the code takes the more conservative one, and the lead's ruling is
requested where marked.

1. **Calibration intercept.** The task says "calibration slope and intercept (logistic recalibration of the
   logit)"; the clinical literature often defines the intercept as calibration in the large, with slope fixed
   at 1. Both are computed: `calibration_intercept` is the joint fit's `a`, and `calibration_in_the_large` is
   reported beside it. Probabilities are clipped to [1e-7, 1 - 1e-7] before the logit. *Ruling requested:*
   which intercept the report names "the" intercept.
2. **Raw or Platt-calibrated probabilities for the calibration metrics and the multiclass view.** The protocol
   fixes a Platt calibration on validation but does not say whether test calibration, Brier score and top-k are
   reported on raw or calibrated probabilities. The functions take whatever matrix they are given. The
   proposal is raw as the primary report and validation-Platt beside it. AUROC is unchanged by Platt when its
   slope is positive. *Ruling requested.*
3. **"Threshold giving 90% specificity."** The code never goes below the target: it uses the lowest observed
   validation score with specificity at least 0.90, and `p >= t` is positive. Pooling over validation sources
   follows the MLC-02 protocol; it is a fitted operating point, not a reported pooled metric.
4. **Platt.** This is plain maximum-likelihood logistic regression on `logit(p)`, without Platt's smoothed
   targets. With thousands of validation records per condition the difference is negligible.
5. **Validation sources** are the manifest's `source` values, six in all, not families. CPSC 2018 and
   CPSC-Extra are separate, and so are Chapman and Ningbo. A source with no entering condition would be left out
   of the mean and listed. On v4 every source has one (14, 14, 9, 12, 5 and 7 entering conditions), so the case
   does not arise. The per-condition variant averages over the sources where that condition enters.
6. **Redraw rule.** "Redrawn when a condition loses all positives" is applied to all positives or all
   negatives, for the conditions that enter the full set's macro. Conditions that do not enter get no interval
   (NaN), and they do not trigger redraws. Both test sets have all 14 entering, so this only matters for
   subgroups and validation sources. A set needing more than 50 attempts per kept draw raises.
7. **Bootstrap seed.** The protocols name none. The code uses 20261110, the first MLC-02 seed, rather than
   the 2026 of `intervals.py`. The draw convention is that of `intervals.py`; its 5/95 PTB-XL benchmark
   percentiles are not used.
8. **Holm family and p-values.** The family is the conditions that enter the set, 14 in both test sets. The
   p-values are percentile-bootstrap p-values with the +1 correction,
   `min(1, 2 min((#{d <= 0} + 1)/(B + 1), (#{d >= 0} + 1)/(B + 1)))`. Their smallest value at 2,000 draws is
   about 0.001, or 0.014 after Holm over 14. The interval is not Holm-adjusted; only the p-values are.
9. **Calibration intervals.** Resamples whose fit does not converge are left out of that metric's interval and
   counted in `<metric>_nan_draws`.
10. **Top-1 and top-2.** The conservative reading counts a record only if exactly one condition is positive
    and all 14 are defined. The variant "exactly one positive, undefined allowed" is computed beside it. In both
    readings all 14 outputs are ranked, undefined ones included, because a deployed model names one of the 14.
    Ties go to the earlier condition.
11. **Exact-set accuracy** on "all records": a record matches when the predicted set equals the labelled set on
    every defined condition, with undefined entries ignored. The accuracy on records with all 14 defined is
    reported beside it.
12. **Flat-lead subgroup**: a flat or a missing lead in the first 10 s (`n_flat_leads > 0` or
    `n_missing_leads > 0`). **Sex**: female, male, and unknown.
13. **One condition against several**: each condition's positives are split by the record's number of positive
    conditions, 1 or at least 2, with undefined entries not counted. Both groups keep all of that condition's
    negatives.
14. **Tie rule**: every candidate within 0.001 of the highest (inclusive) is tied, and the earliest in the
    protocol table is selected. The order is P0, L-J, L-C, G-M, H-X, H-J, H-C, H-All, S-R, F, E. Ties are not
    chained: with 0.9000, 0.9009 and 0.9018, the middle one is selected.
15. **"Usable"**: a condition is usable only if both holds: the AUROC interval's lower end is at least 0.80,
    and there are at least 100 positives.
16. **Sensitivity analyses and space A** use the same functions with `labels_lik50` or `labels_a`. Resamples
    must be drawn for each label matrix; `bootstrap_metrics` checks `labels_sha256`. Space A has no entering
    group in validation CPSC 2018, because its defined STTC and CD entries are all positive.
17. **Source identification** needs finite features: the caller imputes missing measurements (training median
    plus a missing flag, as in H-All). It uses L2 regularization with C = 1, five folds, and seed 20261110.

## Null check: real labels, RANDOM scores

These are not model results. Each probability is drawn uniformly at random with
`np.random.default_rng([20261110, model])`, on the real v4 label matrix.

The run is `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_CORETYPE=Haswell PYTHONPATH=.
.venv/bin/python -m scripts.validation.check_mlc02_evaluation_null`. It ran at commit `47c82c0` with no
modified tracked file, and took 260.5 s. The output is
`outputs/mlc02_evaluation_null_check_20261110/null_check.json`, which also records the code and manifest
SHA-256 (manifest `25d75f8f...e8f3a0`).

### Macro AUROC of one random model

"Null SD" is the expected spread of a random model's macro AUROC, `sqrt(sum_c (P_c + N_c + 1) / (12 P_c N_c))`
divided by the number of entering conditions C; z is (macro - 0.5) / null SD [`sets`].

| set | records | entering (B) | macro B | null SD | z | macro B, likelihood variant | entering (A) | macro A |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| validation ptbxl | 2,183 | 14 | 0.5065 | 0.0076 | 0.86 | 0.5059 | 4 | 0.5000 |
| validation ningbo | 6,972 | 14 | 0.5039 | 0.0068 | 0.57 | 0.5039 | 4 | 0.5234 |
| validation chapman_shaoxing | 2,036 | 9 | 0.5042 | 0.0095 | 0.44 | 0.5042 | 2 | 0.5148 |
| validation georgia | 2,043 | 12 | 0.4982 | 0.0080 | -0.23 | 0.4982 | 3 | 0.4982 |
| validation cpsc_2018 | 1,251 | 5 | 0.4673 | 0.0110 | -2.97 | 0.4673 | 0 | - |
| validation cpsc_2018_extra | 653 | 7 | 0.4963 | 0.0193 | -0.19 | 0.4963 | 4 | 0.5016 |
| test ptbxl (fold 10) | 2,198 | 14 | 0.4991 | 0.0076 | -0.12 | 0.4990 | 4 | 0.5058 |
| test sph (evaluation half) | 12,698 | 14 | 0.4962 | 0.0056 | -0.68 | 0.4962 | 4 | 0.5142 |
| sph calibration half | 12,883 | 14 | 0.5083 | 0.0055 | 1.52 | 0.5083 | 4 | 0.4896 |

The entering counts reproduce the support table of the MLC-01 results report: 14, 14, 9, 12, 5 and 7 at
validation, and 14 in both test sets.

Validation CPSC 2018 is 3 null SDs below 0.5, so I checked it against 200 further random models (`1000 + r`)
[`null_distribution`]. Over those models z had a mean of -0.043 and an SD of 0.965 there. In every set the mean
of z lay between -0.066 and 0.063, the SD of z between 0.94 and 1.11, and the largest |z| between 2.28 and 3.50.
The -2.97 is therefore a tail draw of a correctly centred null, not a bias.

The selection metric of the random model is 0.4961. No source was excluded [`operating_points.selection`].

### Operating points of the random model

Thresholds were fitted at 90% specificity on all validation records [`operating_points`]:

- On validation, specificity is 0.9000-0.9001 across the 14 conditions.
- On the tests, specificity ranges over the conditions as follows, with sensitivity near 0.10 as chance
  predicts:

  | set | specificity | sensitivity |
  |---|---|---|
  | PTB-XL fold 10 | 0.882-0.912 | 0.037-0.167 |
  | SPH evaluation half | 0.890-0.906 | 0.052-0.116 |
  | SPH calibration half | 0.894-0.905 | 0.051-0.186 |

- Platt slopes are -0.032 to 0.036, as they should be for scores unrelated to the labels.

### Single-condition records (labels only) and the derived view of the random model

[`multiclass`]

| set | records | no positive | 2+ positives | single, all 14 defined | single, undefined allowed | top-1 (strict / loose) | top-2 (strict / loose) | exact set, all records | records all defined |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|
| validation ptbxl | 2,183 | 932 | 576 | 203 | 675 | 0.123 / 0.095 | 0.207 / 0.181 | 0.122 | 741 |
| validation ningbo | 6,972 | 1,175 | 1,940 | 2,930 | 3,857 | 0.069 / 0.070 | 0.140 / 0.139 | 0.051 | 5,299 |
| validation chapman_shaoxing | 2,036 | 404 | 517 | 0 | 1,115 | - / 0.087 | - / 0.151 | 0.095 | 0 |
| validation georgia | 2,043 | 609 | 613 | 0 | 821 | - / 0.067 | - / 0.135 | 0.099 | 0 |
| validation cpsc_2018 | 1,251 | 350 | 32 | 0 | 869 | - / 0.055 | - / 0.119 | 0.209 | 0 |
| validation cpsc_2018_extra | 653 | 270 | 120 | 0 | 263 | - / 0.084 | - / 0.152 | 0.162 | 0 |
| test ptbxl (fold 10) | 2,198 | 929 | 567 | 204 | 702 | 0.064 / 0.056 | 0.152 / 0.124 | 0.115 | 711 |
| test sph (evaluation half) | 12,698 | 7,424 | 1,392 | 3,434 | 3,882 | 0.073 / 0.075 | 0.139 / 0.141 | 0.141 | 11,678 |
| sph calibration half | 12,883 | 7,569 | 1,489 | 3,398 | 3,825 | 0.079 / 0.078 | 0.154 / 0.151 | 0.142 | 11,864 |

With random scores, top-1 and top-2 are near 1/14 = 0.071 and 2/14 = 0.143 where the counts are large. PTB-XL's
strict top-1 of 0.123 rests on only 203 records.

The single-condition records per named condition, all 14 defined against undefined allowed, are:

| condition | PTB-XL fold 10 | SPH evaluation half |
|---|---|---|
| sinus_tachycardia | 9 / 30 | 199 / 223 |
| sinus_bradycardia | 14 / 40 | 698 / 743 |
| sinus_arrhythmia | 19 / 44 | 463 / 473 |
| af_or_flutter | 8 / 36 | 139 / 154 |
| premature_ventricular_complex | 5 / 23 | 271 / 288 |
| first_degree_av_block | 8 / 15 | 14 / 15 |
| complete_rbbb | 3 / 11 | 207 / 208 |
| incomplete_rbbb | 15 / 48 | 444 / 446 |
| lafb | 0 / 23 | 0 / 31 |
| left_axis_deviation | 6 / 7 | 38 / 38 |
| lvh | 15 / 72 | 14 / 38 |
| st_depression | 0 / 15 | 286 / 286 |
| t_wave_abnormality | 43 / 131 | 628 / 904 |
| mi_any | 59 / 207 | 33 / 35 |

Under the strict reading LAFB never qualifies, in any source. Amendment 3, ruling 1 makes left axis deviation
undefined on every LAFB record without a left-axis code. At PTB-XL fold 10 the strict reading keeps 204 of 702
records, because left axis deviation is undefined on 1,360 records and ST depression or the T wave on about 300
each. *Ruling requested:* the strict reading changes which conditions can be named, so the proposal is to report
the "undefined allowed" reading as primary, with the strict one beside it.

### Bootstrap on the test sets

There were 12 random models, 2,000 resamples, and 8 threads with BLAS limited to one thread [`bootstrap`].

| | SPH evaluation half | PTB-XL fold 10 |
|---|---:|---:|
| records / patients | 12,698 / 12,246 | 2,198 / 1,904 |
| redraws | 0 | 0 |
| drawing the 2,000 resamples | 3.2 s | 0.5 s |
| one model, every metric including calibration, one thread | 15.7 s | 2.9 s |
| 12 models, AUROC, AP, Brier, sensitivity and specificity | 38.8 s | 6.4 s |
| 12 models, every metric including the three calibration fits | 120.7 s | 17.0 s |
| 11 paired comparisons with Holm | < 0.1 s | < 0.1 s |
| random model 0: macro AUROC and 95% interval | 0.496 (0.485-0.507) | 0.499 (0.484-0.513) |
| conditions whose AUROC interval contains 0.5 | 14 of 14 | 12 of 14 |
| calibration resamples that did not converge | 0 | 0 |
| conditions called usable | 0 | 0 |
| paired macro intervals (models 1-11 minus model 0) containing 0 | 11 of 11 | 11 of 11 |
| Holm-adjusted condition tests below 0.05 | 0 of 154 (smallest 0.238) | 1 of 154 (0.014) |

The one rejection at PTB-XL is consistent with chance: Holm holds the familywise error at 5% within each of
the 11 comparisons.

## Tests and checks

- `tests/test_mlc02_evaluation.py`: 36 passed in 16.6 s. Two tests read the real manifest, labels only, and
  skip without it. They cover:
  - hand cases with undefined entries for AUROC, AP, Brier, prevalence, sensitivity and specificity, the
    calibration slope and intercept (a saturated two-point design, `a = -ln 3`, `b = 2 ln 3`), and calibration
    in the large;
  - AUROC and AP equal to sklearn on defined entries, and the Newton fit equal to sklearn's unpenalized
    logistic regression within 1e-6;
  - a perfect scorer (AUROC 1, AP 1, calibration NaN under separation) and a random scorer (AUROC within 0.02
    of 0.5);
  - the macro entry rule at 19/20 positives and 19/20 negatives, and the imprecise flag at 99/100 positives;
  - the selection metric, including a source without an entering condition and a refusal of SPH records;
  - the 90% specificity threshold against `roc_curve`, and its integer counting;
  - Platt recovering `(a, b) = (-0.7, 1.6)` and `(0.4, 0.5)` within 0.03, with a slope of 1 after calibration;
  - top-k, exact-set and co-occurrence hand cases; subgroups; source identification on shifted families and on
    noise;
  - the tie rule, including the inclusive boundary and the unchained case; the selection record writer;
  - bootstrap determinism and identity with the `intervals.patient_resample` draws, whole-patient movement, the
    redraw rule, and every bootstrap metric equal to sklearn on concatenated resamples;
  - coverage: 200 clustered synthetic datasets with a known AUROC of 0.7734, 2,000 resamples each. The test
    requires 90-98%; rerunning its datasets with the same seeds gave 0.95 (190 of 200).
  - paired differences: the interval excludes 0 for a clearly better model and includes it for two noisy
    copies of one model;
  - Holm against a hand example, and bootstrap p-values;
  - prediction files rejecting shuffled or missing keys, other conditions, out-of-range probabilities and
    unsafe ids;
  - the manifest counts of the MLC-01 results report: roles, the T-wave cells at SPH, left axis deviation at
    fold 10, 391 MI and 177 LVH likelihood-variant positives, and the validation entering counts.
- `tests/test_intervals.py`, `tests/test_mlc_labels.py` and `tests/test_mlc01_dataset.py` still pass (101
  together with the new file).
- `ruff check ecg_experiment scripts tests conftest.py`: passed.

## Not done here

- No model was fitted and no model score was computed. Source identification was not run on real features.
- The full test suite was not run. Only the new file and the three suites above were.
- The rulings requested above are open: which intercept is primary (1); raw or Platt probabilities for
  calibration and top-k (2); and the primary top-k reading (multiclass section).
