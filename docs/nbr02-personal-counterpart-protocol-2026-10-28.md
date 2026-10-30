# NBR-02 protocol: a personalized normal counterpart fitted on a multi-source cohort

Frozen 28 October 2026, before any replacement score of a new arm is read. Branch
`research/nbr02-personal-counterpart-20261028`, started from `fb20de7` on the NBR-01 branch; it is stacked
on NBR-01 and its pull request waits until NBR-01 is on `main`. The user chose this study; the design and
decision rule come from the lead's brief and are fixed here.

## Question

NBR-01 ([results](nbr01-normal-beat-replacement-results-2026-10-28.md)) replaced regions with 070's
population-mean counterpart, fitted on 2,666 PTB-XL normals. That template changed normal ECGs about as
much as abnormal ones (mean absolute score change per whole-lead replacement 0.0221 on normals against
0.0207 on abnormals for `all_age_blend`). Does a personalized counterpart, fitted on the normals of the
cohort that trained the screening model, change normal ECGs less relative to abnormal ones, without losing
NBR-01's territory localization?

## Arms (2 x 2)

| Fit set | Rank 0 (population mean) | Rank 6 (personalized) |
| --- | --- | --- |
| P: NBR-01's 2,666 LocBench `fit` clean PTB-XL normals | P0 (= NBR-01) | P6 |
| C: normals of the all-age 25k cohort | C0 | C6 (primary arm) |

Fit sets (stage `fitsets`, already run; no SPH waveform or score involved):

- P: the 2,666 records of NBR-01, read from the LocBench cache; all usable. The refitted P0 model is
  identical, field by field, to NBR-01's saved `counterpart/rank0.npz`.
- C: the 7,393 `target == 0` rows of `outputs/cohort_age_label_priority_20261028/all_age.csv` (PTB-XL
  3,038, Ningbo 2,581, Chapman 819, Georgia 801, CPSC 2018 154; all `split=train`, quality passed, 5,000
  samples, window start 0), read with `pr74_screening_features.read_training_signal` through each row's
  own `waveform_path`, `backend`, `window_start` and verified raw and canonical hashes. Exclusions: missing
  age 54 (Ningbo 41, Georgia 13), unknown sex 0, fewer than two complete beats 0. Used: 7,339 (PTB-XL
  3,038, Ningbo 2,540, Chapman 819, Georgia 788, CPSC 2018 154). Sex is `sex_normalized` mapped to 062's
  coding (M 0, F 1).
- C closed-data checks, run before any waveform was read, against the Challenge split table
  `outputs/cohort25_recovery_20261028/challenge/full/rows.csv`, whose SHA-256 equals the
  `docs/challenge-splits-v1.md` value (`5ecd8215...`): all 4,355 Challenge rows match a `train` record by
  source and record name; the cohort's full-signal hash equals the split table's signal hash for all 4,355;
  no cohort hash (window or full signal) equals any test or calibration signal or window hash; no record
  matches a test or calibration record; no row is SPH by source, path or signal hash (against the 17,402
  SPH hashes of the frozen baseline table); PTB-XL rows come from folds 1-7 only; no EchoNext, MIMIC or
  CODE-15 row.

Models: for each fit set, 062's `fit_population(medians, covariates, 0)` and `fit_population(medians,
covariates, SUBSPACE)` with `SUBSPACE = 6`, on 062 `record_features` medians and the 062 `design` (heart
rate, age, sex). The rank-0 and rank-6 models of one fit set share the same regression coefficients.

Templates for an SPH record, on 062's grid: rank 0 is `covariates @ coefficients` (identical to 070's
`template_beat`); rank 6 is `covariates @ coefficients + scale * (basis @ coordinates)` with `coordinates =
subspace_coefficients(model, (median - covariates @ coefficients) / scale)`, the Huber-reweighted PPCA
posterior clipped at `BOX = 2.58` SD, where `median` is the record's own 062 median beat. This equals the
median minus 062's `population_residual`. Every function is 062's or 070's, called unchanged.

Counterpart and replacement: as NBR-01. The template is warped to every complete beat with 070's
`template_timeline`, `c = x - clean + T`, and replaced through NBR-01's component masks (reused from
NBR-01's plans), 22 ms Hann taper and 52 regions (36 grid cells, 12 leads, 3 components, full). The
self-swap control does not depend on the template, so every arm uses NBR-01's self-swap scores and energies.

## Sample, models and scores

The same 1,037 SPH ECGs as NBR-01, in the same order (inferior 49, anterior 55, lbbb 34, rbbb 299, stt 300,
normal 300). `all_age_blend` is primary; `ptb_baseline_blend` and both HGB and xECG components are
secondary, scored from shared measurements and xECG features as in NBR-01, with the frozen thresholds. P0
reuses NBR-01's saved scores; the unreplaced and self-swap scores of every arm are NBR-01's arrays. New
encoding is 52 variants for each of P6, C0 and C6 per ECG (156), in full batches of 16 with the last batch
padded.

Each arm's statistics are NBR-01's statistics (`nbr01_analysis.statistics`, unchanged) applied to an
NBR-01-layout array in which the 52 normal-counterpart variants and their energies come from the arm. So
each arm has its own energy map, recomputed from its own replacement energies.

## Primary decision

C6 replaces P0 as the page counterpart only if all three hold, on 2,000 whole-patient bootstrap draws
(`intervals.patient_groups` and `patient_resample`, `numpy.random.default_rng(20261028)`), every statistic
of every arm on the same draws, 95% percentile intervals:

- A. Selectivity improves. Selectivity of an arm is the mean over the 737 abnormal ECGs (inferior,
  anterior, lbbb, rbbb, stt) of each ECG's mean absolute `all_age_blend` score change over the 12 whole-lead
  replacements, divided by the same mean over the 300 SPH normals (NBR-01: 0.0207 / 0.0221). The paired
  difference C6 minus P0 must have a lower bound above 0.
- B. Territory localization is non-inferior. NBR-01's primary territory contrast for C6 minus P0 must have a
  lower bound above -0.05.
- C. Guardrail. C6's operator check passes: the mean abnormal full-replacement drop has a lower bound above 0.

Rationale (from the brief): a generic template changes normal people's own features; a better counterpart
should change normals less relative to abnormals, and shrinking every change does not move a ratio.

Review of the rule before any score, kept unchanged: A uses absolute changes, so a counterpart that created
artefacts raising abnormal scores would also raise the numerator. The operator check and the separate
numerator and denominator contrasts (secondary 2) expose that case: an improvement that comes from larger
abnormal changes rather than smaller normal changes will be visible and will be reported as such. The
non-inferiority margin of 0.05 is about one sixth of NBR-01's contrast (0.306) and close to its interval
half-width (0.058). No rule was changed.

## Secondary (intervals, no decision)

1. The 2 x 2 decomposition, paired on the same draws: cohort effect (C0 - P0, C6 - P6) and personalization
   effect (P6 - P0, C6 - C0), plus C6 - P0, for selectivity, territory contrast, territory minus energy,
   wave contrasts, BBB lead contrast, top-1 and top-3 hits, abnormal and normal mean absolute lead and grid
   changes, abnormal and normal full-replacement drops, abnormal full-replacement score and flip rates, for
   every score (both blends and their HGB and xECG components), and for each arm's energy map.
2. For each arm, every NBR-01 endpoint with its own energy map: territory contrast, model minus energy,
   self-swap contrast, wave contrasts, BBB lead contrast, top-1 and top-3 hits against energy, flips,
   full-replacement sanity, normals' full drop; selectivity numerator and denominator separately.
3. Template movement: per record, the mean absolute difference in mV over the 12 x 380 grid between
   templates P6 and P0, C6 and C0, C0 and P0, and C6 and P0; mean by group with intervals, with LBBB and
   RBBB reported explicitly. Clip frequency: per group, the share of records with any of the six P6 or C6
   coordinates at the clip bound, and the mean number of clipped coordinates.

No age-subgroup analysis.

## Integrity checks

1. Rebuilt P0 counterparts equal NBR-01's saved counterparts for all 1,037 records (largest difference
   0.0 mV required; 20 of 20 in the profile), and each record's 062 knots equal NBR-01's.
2. The new analysis code, applied to NBR-01's saved scores (arm P0), reproduces every one of NBR-01's 769
   point statistics exactly, including the primary +0.3056275517940945. The unreplaced and self-swap scores
   are NBR-01's arrays bit for bit, because they are not recomputed.
3. The cohort fit-set checks above (passed), and the refitted P0 model equals NBR-01's (passed).
4. Variant SHA-256 per arm between preparation and encoding, as in NBR-01.

Disclosure: new variants are encoded in batches of the record's own new variants, while NBR-01's
unreplaced variants were encoded with NBR-01's variants. NBR-01 measured such batch effects at up to
2.1e-8 on the blend.

## Profile (already run, no score)

On the first 20 analysed ECGs: preparation 3.43 CPU-seconds per record (16 single-thread workers, 5.9 s
for 20); encoding of the 156 new variants 1.43 s per record on GPU 0 (0.62 GB peak, 10.6 GB free); P0
rebuilt exactly (0.0 mV). Forecast for 1,037 ECGs: preparation about 4 minutes, encoding about 12 minutes
split over both GPUs, each under its own lock.

## Page export

`outputs/nbr02_personal_counterpart_20261028/examples.json`, NBR-01's final schema (v1 fields plus
`summary`): NBR-01's 16 example IDs in the same order; the C6 counterpart, components, scores and energies,
with the original and self-swap scores copied from NBR-01; `summary.statistics` with every arm under
prefixed keys (for example `C6/all_age_blend/territory`) and the arm contrasts (for example
`C6-P0/all_age_blend/selectivity`); `summary.decisions` with parts A, B and C and the overall decision;
top-level `arm: "C6"` and `fit_set` counts by source. Validated by `nbr02_export.validate` and its test. The
export is the C6 arm whatever the decision; the decision says whether the page should switch to it.

## Data rules

SPH never fits anything: both fit sets are training normals, checked above. No Challenge test or
calibration group, EchoNext, MIMIC, CODE-15, PTB-XL fold 10 or final frozen test is read.

## Outputs

`outputs/nbr02_personal_counterpart_20261028/` (stage receipts, models, `score/scores.npz`,
`analyze/results.json`, `examples.json`); work files in `/tmp/nbr02-work`. Code:
`ecg_experiment/localization/nbr02_*.py`, `scripts/experiments/run_nbr02.py`,
`tests/localization/test_nbr02_counterpart.py`. Report: `docs/nbr02-personal-counterpart-results-2026-10-28.md`.
