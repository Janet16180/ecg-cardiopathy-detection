# Experiment 062: normative phase-aligned population map

Frozen 12 October 2026 before any LocBench score. Part of the
[localization programme](localization-program-2026-10-12.md); scored with LocBench v1
([Experiment 061](experiment-061-locbench.md)).

## Question

Does a population model of delineated, phase-warped beats per lead (conditioned on heart rate, age and
sex), combined with the within-ECG beat residual, localize persistent and focal abnormalities better than
the best 061 baseline, without being drawn to noise?

## How the design answers the 041-060 diagnosis

- Cause 1 (two references, never combined): every lead-sample gets a persistent score (the record's median
  beat against a population of clean normals) and a focal score (each beat against the median of the
  record's other beats), on one calibrated scale, summed.
- Cause 2 (fixed windows): beats are delineated and each lead is warped piecewise-linearly onto a canonical
  grid, so QRS, ST and T line up across people whatever their heart rate and QRS width.
- Cause 5 (contaminated reference): the population model uses only the 2,666 `fit`-partition clean normals of
  `docs/locbench-partition-v1.json`; noise is handled explicitly (below), not absorbed into the reference.

## Method (one primary recipe, `normative_v1`)

Code: `ecg_experiment/normative_map062.py`, runner `scripts/experiments/run_normative062.py`. Input: the raw
500 Hz LocBench cache (mV), age and sex. CPU only, at most 8 processes.

1. Delineation (frozen, waveform only). R peaks and per-beat, per-lead P/QRS/T supports from
   `morphology_boundaries.adaptive_boundaries` (Experiments 055 and 059; R peaks from the frozen
   `fragment_localization.r_peaks`). Joint per-beat fiducials: QRS onset and offset are the 10th and 90th
   percentiles of the 12 lead supports (a robust form of 059's earliest-onset/latest-offset rule, which was
   validated on two channels); T onset and T end the median over leads with a T support (at least three);
   P onset the median over leads with a P support (at least three), else missing. P never sets a warp knot.
   Missing T falls back to the record's median QRS-onset-to-T-end and QRS-offset-to-T-onset durations, and
   with no T in the record to QT = 0.40 s times the square root of RR. Knots are forced increasing.
   Beats whose QRS or 24 ms baseline window is cut by the record edge are dropped (their RR still counts).
   No ECGdeli output is used, so the region set (B) is not circular; no ECGdeli variant is run.
2. Preprocessing. Zero-phase 40 Hz low-pass (second-order Butterworth). Baseline: per lead, the mean over
   24 ms ending 6 ms before each QRS onset (the PR segment), linearly interpolated between beats and
   subtracted. Amplitude is never normalized per lead or per beat.
3. Warp. Canonical grid of 380 bins per lead: 150 bins before QRS onset at 1 sample per bin (300 ms, so a
   long PR or a missing P shows as a deviation where P normally sits), QRS onset to offset in 50 bins, QRS
   offset to T onset in 40, T onset to T end in 90, and 50 bins after T end at 1 sample per bin. Each beat's
   knots are shifted by one whole-sample shift in [-4, +4] (shared by the leads) that best matches the
   record's initial median QRS (054's alignment idea; scale-normalized squared error, smallest shift on ties).
4. Record summaries. Dominant beats: RMS distance to the initial median at most twice the median distance
   (at least three kept). Median beat and robust beat-to-beat dispersion (1.4826 MAD) over dominant beats;
   leave-one-out focal deviation of every beat from the median of the other dominant beats. Intervals: heart
   rate, median QRS duration, median QRS-onset-to-T-end (QT), median PR (when P is found in at least two
   dominant beats). Per beat: RR prematurity (preceding RR over median RR), QRS-duration change from the
   record median, and the 20-40 Hz RMS outside QRS (beat-lead noise).
5. Population model, fitted on the 2,666 `fit` clean normals only. Design: intercept, heart rate and age
   (linear and squared), sex (age above 89 set to 90). Ridge regression of the flattened 12 x 380 median beat
   on the design; per-dimension robust scale of the residual; PCA of the standardized residual, rank 6.
   A record's subspace coordinates are a Huber-reweighted probabilistic-PCA posterior mean, each clipped to
   2.58 normal standard deviations, so that normal-sized correlated variation (axis, global amplitude) is
   explained and larger excursions stay in the residual. Persistent z per lead and bin = residual (mV) over
   the square root of (normal residual scale squared + the median's standard error squared), the standard
   error being 1.2533 x dispersion / sqrt(dominant beats): a noisy lead's median is trusted less.
6. Focal z per beat, lead and bin = focal deviation over the square root of (population focal scale squared
   + record dispersion squared + beat-lead 20-40 Hz noise squared). After smoothing, each lead's magnitude
   is replaced by its geometric mean with the mean over leads (physiological beat changes reach several
   leads; electrode artifacts often do not).
7. Smoothing: local RMS over 9 grid bins.
8. Calibration on a common scale: every component becomes -log10 of its tail probability among normal
   records. Persistent: per lead and segment (pre-QRS, QRS, ST, T, post-T) tables from patient-held-out
   residuals (5 patient folds of the fit normals, seed 62062; the residual scales come from the same
   out-of-fold residuals). Focal: per lead and segment tables from all beats of the fit normals. Interval
   evidence: QRS (longer only), QT and PR (either direction) z against the covariate regression, as
   -log10 tail probability, written into the QRS, T and pre-QRS segments of all leads. Beat evidence: RR
   prematurity into that beat's pre-QRS and QRS segments, QRS-duration change into its QRS segment.
   Tables extrapolate beyond their last level by one decade per last-decade spread.
9. Combination: the sum of the four calibrated components (Fisher-style accumulation of -log10 p), per
   beat, lead and bin; mapped back to the original timeline through each beat's warp, maximum where beats
   overlap; samples outside every beat get the map's minimum (the LocBench contract). Records with fewer
   than two complete beats get an all-zero map and are counted.

## Declared variants

`normative_v1` is the only primary. Ablations are descriptive, scored on the same LocBench run, and cannot
replace a failed primary:

| Variant | Change from primary |
| --- | --- |
| `max_combination` | maximum instead of sum of components |
| `persistent_only` | persistent + interval evidence only |
| `focal_only` | focal + beat evidence only |
| `no_subspace` | rank 0 (covariate regression only) |
| `no_noise_weight` | no standard-error, dispersion, beat-noise or coherence terms |

## Design data disclosure

Every choice above was made before any LocBench score, using only `fit`-partition records: the clean normals
for fitting, and three descriptive checks that are not LocBench metrics.

- Joint-fiducial quantiles: agreement with PTB-XL+ ECGdeli global fiducials on 120 fit clean normals
  (QRS onset 10th percentile: MAE 4.9 ms; QRS offset 90th percentile: 9.1 ms; T end median: 18.7 ms; the
  maximum over leads was 79 ms late for T end). This nudges the delineation toward ECGdeli on normals, so
  the B readings are not fully independent of it; no evaluation record was used.
- Subspace rank (0, 3, 6, 10, 20) and bound (Mahalanobis ball or per-coordinate box): lead AUROC of the
  persistent map for 100 fit-partition records each of IMI, ASMI, AMI, LMI, CRBBB, LAFB, ISCAL, INJAS. Every
  setting gave a mean of 0.555-0.573; rank 6 with a 2.58 box was taken as the middle setting.
- Combination and focal noise handling: on 132 fit-partition records with an extra-beat annotation and a beat
  with RR ratio at most 0.85, the share whose top-scoring beat was that beat (sum 0.758, max 0.462), and on
  300 fit-partition normals with a lead-named noise annotation, the noisy-lead AUROC (primary 0.611,
  `no_noise_weight` 0.719, `persistent_only` 0.497).

During smoke tests maps were built for 24 ECGAD development records; no label was read and nothing was scored.
The source hashes of `morphology_boundaries.py` and `fragment_localization.py` equal those in Experiment 059's
receipt.

## Primary comparison and decision rule

Primary: paired within-record localization AUROC (A2) on the 94 marked ECGAD development records, `normative_v1`
minus the bar, with LocBench's 2,000 whole-patient bootstrap draws (seed 61061). The bar is the 061 baseline with
the highest A2 point estimate as frozen in `docs/experiment-061-locbench-results.md`; if that report does not
exist when 062 is scored, the comparison is made against both `U_B` and `residual056` and the primary must beat
both.

`normative_v1` beats the bar if the paired A2 difference has a 95% lower bound above zero. Reported beside it,
with paired intervals against the bar: A1, A3-A6, B1 macro, B2, B3, strict-region sensitivity, C1, C2 and D1.
A noise concern is flagged if C1 exceeds the bar's C1 by more than 0.05. A detection concern is flagged if D1 is
below the bar's D1 by more than 0.05. Neither flag changes the primary verdict; both are reported.

## Integrity checks (before reading any 062 score)

- Rescore one saved 061 baseline (`residual056`) from `outputs/experiment061_locbench_v1/maps/` with
  `scripts/experiments/run_locbench_score.py` and reproduce its saved A2 exactly. A mismatch stops the scoring.
- Every scored ID is mapped; no ECGAD holdout record is mapped (the runner raises).
- Maps are finite; the counts of records without a map (fewer than two complete beats) are reported.
- Fit uses only `fit` clean normals: the runner reads that list from the frozen partition file, whose hash is
  recorded.

## Iteration

The ECGAD development half may be used to iterate. Every re-scored version gets a dated amendment in this file
with its changes committed before scoring, a new output folder (`experiment062_normative_v2`, ...), and all
versions are reported. The ECGAD holdout half, fold 9/10 outside ECGAD development, SPH, the Challenge test
groups and EchoNext test are never read.

## Outputs

`outputs/experiment062_normative_v1/`: `normative.pkl`, `fit_summary.json`, `maps_<variant>.npy` (float16, rows in
`ecg_ids.npy` order: exactly `ecg_experiment.locbench.scored_ecg_ids()`), `record_summary.csv`,
`maps_receipt.json`, the scoring outputs per variant (`locbench/<variant>/summary.json` and tables),
`result.json` and `run.log`. The running agent writes `docs/experiment-062-normative-map-results.md` from
`result.json`.

## Amendment 1 (12 October 2026, before any 062 score)

After the protocol commit, branch `exp/061-locbench` was merged (commits `ce42921` "LocBench ready" and `301aaca`).
Experiment 061 has completed (`outputs/experiment061_locbench_v1/result.json`: status `complete`, `adopted`
true, bar `qrs_prior`, A2 0.7419 [0.7082, 0.7743] on 92 records), but its results report is not yet committed.

1. Bar. The primary comparison is against `qrs_prior`, the bar named in 061's completed receipt, because the
   programme fixes the bar as the best 061 baseline and that receipt is final. This is stricter than the
   fallback in the frozen text. Paired differences against `U_B` and `residual056` (and the other baselines)
   are reported beside it. The decision rule is otherwise unchanged: `normative_v1` beats the bar if the paired
   A2 difference against `qrs_prior` has a 95% lower bound above zero.
2. Integrity check. 061 saved maps only for a display subset (1,578 records), so a full rescoring through the
   CLI is impossible. The check becomes: (a) recompute per-record A2, A3, A4 and A6 of every ECGAD development
   record from the saved `residual056` display maps with `locbench.ecgad_record_metrics` and match the saved
   `tables/residual056/ecgad.csv` (exactly, or to float16 rounding if 061 scored float32 maps, which is then
   stated); (b) `locbench.summarize(load_tables(...))` on the saved `residual056` tables reproduces the A2, B1
   macro, C1 and D1 values and intervals in 061's `result.json` exactly. A failure stops the scoring.
3. Scoring uses `scripts/experiments/run_locbench_score.py` once per variant, with `--compare` against the five
   061 baseline table folders.

## Amendment 2 (12 October 2026, before any 062 score)

Written after the coordinator's note that Experiment 061 is complete (`docs/experiment-061-locbench-results.md`;
bar `qrs_prior`, A2 0.7419 [0.7082, 0.7743]) and that the cardiologist marks concentrate on QRS complexes (16.1%
of samples, 35.9% of marks). Maps of the six declared variants had been built (`maps_receipt.json`), and no 062
score had been computed. Branch `experiment/fragment-localization041` (061 complete plus its Amendment 2) is merged.

1. QRS-stratified reading. After `run_locbench_score`, `scripts/experiments/run_locbench_strata.py` is run with
   the `qrs_prior` tables as reference and the `residual056` tables and every 062 variant's tables
   (061 Amendment 2: `A2_qrs` inside the 059 hybrid QRS mask, `A2_nonqrs` outside). Reported for the primary:
   paired A2 against `qrs_prior` (the primary comparison, unchanged), and paired `A2_qrs` and `A2_nonqrs`
   against `qrs_prior` and against `residual056`.
2. One added declared variant, secondary and descriptive, combining a physiological prior with the anomaly
   evidence: `qrs_lexicographic` = Q + u / (u + 5), where u is the `normative_v1` map and Q is 1 inside the
   QRS onset-to-offset interval of each of the method's own delineated beats (after the alignment shift),
   0 elsewhere. It ranks QRS above non-QRS samples, as `qrs_prior` does, and inside each stratum ranks by
   the anomaly evidence instead of `qrs_prior`'s amplitude tiebreak; the squashing keeps float16 resolution.
   `normative_v1` (anomaly evidence alone) stays the only primary; `qrs_lexicographic` is reported against
   the bar with the same intervals but cannot change the primary verdict.
3. Map storage: all maps are float16; the largest `normative_v1` value is about 93, far below the float16
   limit, so no further transform is applied.
