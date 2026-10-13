# Experiment 073 results: per-lead interval criteria and beat-level criteria

Run on 13 October 2026 (UTC) under the [protocol](experiment-073-per-lead-criteria.md), committed as 9f0c80f before
any 073 map existed. Amendment 1 (`qtc_short` stays record-level) and the code were committed together as 40378e1,
after the smoke run and before the full run or any score. Every number below comes from
`outputs/experiment073_per_lead_criteria_v1/result.json` (SHA-256 `d901f3bb…`) unless it is marked as read from the
saved per-record tables. Intervals are LocBench's 95% whole-patient bootstraps (2,000 draws, seed 61061, no skipped
draws). A2 uses the 92 marked ECGAD development records of 90 patients, and A3 uses the 64 records with 1-11 marked
leads. ECGAD development has been compared many times, so every ECGAD reading here is exploratory and none is
called confirmed.

## Decision

- Primary: `perlead` has A2 0.7334 [0.7048, 0.7598], against 0.7419 for `qrs_prior`. The paired difference is
  -0.0085 [-0.0411, +0.0227], so under the declared words `perlead` **matches** the bar. It has the higher A2 in 40
  of 92 records (read from the saved tables).
- Key secondary: `perlead` A3 is 0.531, against 0.495 for 069 `criteria`. The paired difference is +0.036 [-0.018,
  +0.090], so lead localization **does not recover** under the declared rule. Against `qrs_prior` the A3 gain is
  +0.074 [+0.007, +0.138]. For 069 `criteria` it was +0.039 [-0.021, +0.095].
- Reliability factor: `perlead_agree` minus `perlead` is -0.0041 [-0.0096, -0.0005] on C1_std and -0.00005
  [-0.0016, +0.0016] on A2. That meets the declared rule, so the factor is **kept**. The C1_std drop is as small as
  in 069 (-0.0032). It also raises A3 by +0.022 [+0.004, +0.050].

Painting the interval criteria only on leads that meet them cuts the marked ECGAD records with evidence on all 12
leads from 72% to 38%. The lead AUROC moves up, but not by enough to separate it from 069. A2 stays level with the
bar. The QRS-prior version `perlead_qrs` reaches A2 0.7661, above `qrs_prior` by +0.0242 [+0.0061, +0.0436]. It
is level with 069 `criteria_qrs` (+0.0030 [-0.0105, +0.0164]), so the gain still comes from the QRS time prior.

## Main table

| Map | A2 | vs `qrs_prior` | vs 069 `criteria` | A3 | A4 | A2_qrs | A2_nonqrs | B1 | B2 | C1_std | D1 |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `perlead` (primary) | 0.7334 | -0.0085 [-0.0411, +0.0227] | +0.0093 [-0.0141, +0.0328] | 0.531 | 0.831 | 0.556 | 0.693 | 0.756 | 0.615 | 0.536 | 0.863 |
| `perlead_median` | 0.7252 | -0.0167 [-0.0492, +0.0151] | +0.0011 [-0.0214, +0.0240] | 0.531 | 0.827 | 0.540 | 0.685 | 0.757 | 0.620 | 0.531 | 0.854 |
| `perlead_shift` | 0.7434 | +0.0015 [-0.0297, +0.0314] | +0.0193 [-0.0032, +0.0433] | 0.515 | 0.834 | 0.546 | 0.703 | 0.760 | 0.615 | 0.535 | 0.872 |
| `perlead_agree` | 0.7334 | -0.0086 [-0.0417, +0.0231] | +0.0093 [-0.0143, +0.0330] | 0.553 | 0.829 | 0.555 | 0.694 | 0.755 | 0.611 | 0.531 | 0.861 |
| `perlead_qrs` | 0.7661 | +0.0242 [+0.0061, +0.0436] | +0.0420 [+0.0096, +0.0751] | 0.502 | 0.863 | 0.554 | 0.693 | 0.774 | 0.624 | 0.546 | 0.822 |
| `perlead_own` (diagnostic) | 0.6993 | -0.0426 [-0.0793, -0.0053] | -0.0248 [-0.0553, +0.0077] | 0.578 | 0.747 | 0.595 | 0.651 | 0.730 | 0.608 | 0.562 | 0.836 |
| 069 `criteria` (reference) | 0.7241 | -0.0178 (069 report) | | 0.495 | 0.796 | 0.542 | 0.684 | 0.777 | 0.630 | 0.526 | 0.876 |
| 069 `criteria_qrs` (reference) | 0.7631 | +0.0212 (069 report) | | 0.492 | 0.840 | 0.551 | 0.684 | 0.780 | 0.630 | 0.546 | 0.809 |
| 061 `qrs_prior` (bar) | 0.7419 | | | 0.457 | 0.851 | 0.539 | 0.661 | 0.728 | 0.576 | 0.560 | 0.632 |

B1 and B2 are macro means. The reference strata and C1_std values were recomputed in this run from tables copied into
`ref/`. The `qrs_prior` A3, A4, B and D1 values are those of the 069 report. Records with the higher A2 than
`qrs_prior`, read from the saved tables: `perlead_median` 37, `perlead_shift` 42, `perlead_agree` 40, `perlead_qrs`
48 and `perlead_own` 34 of 92.

Other `perlead` readings: A1 0.770, A5 0.885, A6 0.442 (069 `criteria` 0.407), B3 0.805, C1 0.366, C2 0.491.

## Where the evidence lands on ECGAD

Lead specificity on the 94 marked development records (declared readings, median-beat evidence):

| Map | All 12 leads | No lead | Mean evidence leads |
| --- | ---: | ---: | ---: |
| 069 `criteria` | 72.3% | 3.2% | 9.70 |
| `perlead` | 38.3% | 3.2% | 8.13 |

27.7% of the marked records also carry beat-level evidence.

`perlead` against 069 `criteria`:

- A3: +0.036 [-0.018, +0.090].
- A4: +0.035 [+0.009, +0.063]. Less evidence on ST, T and P across all leads helps the time ranking.
- A2_qrs: +0.014 [-0.009, +0.038]. A2_nonqrs: +0.009 [-0.017, +0.037].

`perlead` against `qrs_prior`: A3 +0.074 [+0.007, +0.138], A4 -0.019 [-0.054, +0.016], A2_qrs +0.017 [-0.014,
+0.049] and A2_nonqrs +0.032 [+0.005, +0.061].

Only 38% of records still have evidence on every lead, yet the lead AUROC rose only from 0.495 to 0.531. Many leads
still carry evidence (8.1 on average, against about 7 marked leads per marked record in the 069 report). The
lead-specific criteria that fire often in these records (ST depression, T inversion, V1 R/S, wide P; 069 report)
were not changed.

## Beat-level criteria

`perlead` minus `perlead_median` (beat-level criteria added):

- A2: +0.0082 [+0.0037, +0.0135].
- A4: +0.0042 [+0.0015, +0.0076].
- A3: 0.000 exactly. The lead score is the 95th percentile of a lead's samples, and evidence on a single beat
  covers too few samples to move it.
- C1_std: +0.0047 [+0.0001, +0.0099]. The beat criteria are slightly drawn to noisy leads.
- D1: +0.0099 [+0.0073, +0.0127].

On the 2,666 `fit` clean normals, 6.2% of records and 0.75% of beats carry beat-level evidence. The fitted
thresholds are pi01 = 0.891 (prematurity) and delta99 = 1.24-1.50 (relative QRS deviation, by lead).

## Shifted thresholds

`perlead_shift` lowers each per-lead threshold by the normal offset between the lead's interval and the record's.
Against `perlead`, its A2 is +0.0100 [+0.0035, +0.0177], A3 -0.015 [-0.049, +0.020] and B1 +0.0041 [+0.0010,
+0.0068]. On `fit` clean normals it makes `ivcd` fire in 23.9% of records (069: 5.3%, `perlead` 2.6%) and
`qtc_long` in 5.7%. It gains A2 by painting more leads, not by finding the right ones.

## Region test (independent: no 073 map sees `REGION_TABLE` or a statement)

| Map | B1 vs `qrs_prior` | B2 vs `qrs_prior` | B1 vs 069 `criteria` | B2 vs 069 `criteria` |
| --- | --- | --- | --- | --- |
| `perlead` | +0.028 [+0.018, +0.037] | +0.039 [+0.020, +0.059] | -0.020 [-0.027, -0.013] | -0.015 [-0.029, +0.001] |
| `perlead_shift` | +0.032 [+0.022, +0.042] | +0.039 [+0.019, +0.060] | -0.016 [-0.023, -0.009] | -0.015 [-0.030, +0.001] |
| `perlead_qrs` | +0.046 [+0.039, +0.052] | +0.048 [+0.032, +0.064] | -0.003 [-0.011, +0.007] | -0.006 [-0.026, +0.014] |
| `perlead_own` | +0.002 [-0.009, +0.013] | +0.032 [+0.007, +0.058] | -0.046 [-0.056, -0.035] | -0.022 [-0.045, +0.001] |

The per-lead maps still land on the expected regions better than the disease-blind bar, but less well than 069
`criteria` on B1. The biggest B1 losses are in families whose expected region covers all leads, where per-lead
painting removes evidence: CLBBB 0.887 (069: 0.949), 1AVB 0.830 (0.869) and LNGQT 0.640 (0.680). Lead-resolving
the fascicular patterns did not help LAFB (B1 0.770 against 0.796, B2 0.434 against 0.439). B2 rose for IMI (0.531
against 0.505) and fell for CRBBB (0.639 against 0.707) and anterior MI (0.733 against 0.768).

Region-set specificity for `perlead` (852 records in families whose lead set is not all 12 leads): 5.9 evidence leads
per record (069: 7.1), 3.5 of them outside the textbook set (069: 4.3). 77% of records depart from the textbook set
(069: 79%), the mean Jaccard index is 0.32 (069: 0.33), and 6.8% of records have no evidence. CRBBB records still
average 11.4 evidence leads.

On ECGAD (34 marked records with a region statement whose lead set is not all 12 leads), marked shares by lead
group under `perlead` are: departure 77 / 130 (0.59), shared 113 / 155 (0.73), textbook only 31 / 52 (0.60) and
neither 28 / 71 (0.39). The departures are still marked more often than the leftover leads, but the gap is smaller
than under 069 (0.61 against 0.21).

## Noise

C1_std of `perlead` is 0.536 [0.510, 0.563]: -0.025 [-0.054, +0.005] against `qrs_prior` and +0.010 [-0.005,
+0.025] against 069 `criteria`. `perlead_qrs` is below the bar on C1_std (-0.014 [-0.029, -0.002]). `perlead_own`
has C1 0.472 and C1_std 0.562, the highest of the 073 maps. Own per-lead intervals are least reliable in the
low-amplitude leads, which are the leads most often annotated as noisy.

## Fit-partition checks (descriptive)

Share of the 2,666 `fit` clean normals meeting a criterion in some lead, 069 consensus against `perlead`:

| Criterion | 069 | `perlead` | `perlead_shift` |
| --- | ---: | ---: | ---: |
| `p_wide` (P duration >= 120 ms in II) | 25.5% | 24.6% | 24.6% |
| `rvh_v1` (V1 R >= 0.7 mV or R/S > 1; unchanged) | 5.7% | 5.7% | 5.7% |
| `ivcd` (QRS >= 110 ms) | 5.3% | 2.6% | 23.9% |
| `pr_long` (PR >= 200 ms) | 4.2% | 10.3% | 6.8% |
| `qtc_long` | 1.4% | 1.8% | 5.7% |
| `qtc_short` (record-level, amendment 1) | 9.2% | 9.2% | 9.2% |

The wide-P rule still fires in a quarter of clean normals when measured on lead II's own P wave. The V1 R/S rule
fires in 5.7%. Both remain questions for the cardiologist (backlog). Per-lead `pr_long` fires more often than the
record-level rule because Uni-G's per-lead PR runs from a common P onset to that lead's QRS onset. So it is never
shorter than the record PR, and its lead-to-lead variation reflects when the QRS appears in each lead, not
atrioventricular conduction.

Share of `fit` records with a statement that meet the matching criterion:

| Statement | Records | Criterion | 069 | `perlead` | `perlead_shift` |
| --- | ---: | --- | ---: | ---: | ---: |
| CLBBB | 294 | `lbbb_lat` | 0.97 | 0.91 | 0.96 |
| CRBBB | 297 | `rbbb_v1` | 0.89 | 0.85 | 0.88 |
| 1AVB | 417 | `pr_long` | 0.85 | 0.82 | 0.78 |
| LVH | 966 | voltage | 0.72 | 0.72 | 0.72 |
| IMI | 892 | `q_path` II or aVF | 0.63 | 0.63 | 0.63 |
| ASMI | 1,065 | `q_v2v3` | 0.56 | 0.56 | 0.56 |
| LAFB | 842 | `lafb` (073: either pattern) | 0.52 | 0.52 | 0.52 |
| LNGQT | 59 | `qtc_long` | 0.44 | 0.41 | 0.53 |

Per-lead thresholds keep most of the record-level sensitivity for the conduction findings.

## Integrity

| Check | Result |
| --- | --- |
| Bar: 061 `qrs_prior` ECGAD maps rescored | per-record A2 within 2.2e-16; mean 0.7419383358406556 against 0.7419383358406554; passed |
| 069 `criteria` lead x wave scores rebuilt from 069 measurements | identical to `evidence_scored.npz` on 4,871 records (largest difference 0.0); passed |
| 069 `criteria` ECGAD maps repainted | largest map difference 0.0; per-record A2 within 1.1e-16; mean 0.7240953915617057 against 0.7240953915617058; passed |
| 069 `criteria_qrs` ECGAD maps repainted | largest map difference 0.0; per-record A2 within 1.1e-16; mean 0.7631057507901342 against 0.7631057507901341; passed |
| Beat components against 069 | equal for all 11,154 measured records; passed |
| Separation | 2,666 normal-reference records, all `fit` clean normals; no ECGAD or `evaluation` record in the reference or sanity sets; no measured record outside folds 1-8 and ECGAD development; no ECGAD holdout record mapped; passed |

## Deviations and disclosures

- Amendment 1, made after the smoke run and before any score: per-lead `qtc_short` fired in 76.5% of the smoke
  run's 200 `fit` clean normals, so `qtc_short` keeps the 069 record-level condition. Only `fit` clean-normal
  prevalences had been read.
- Before the freeze, unit and lead-specificity checks of the per-lead interval columns were run on `fit` clean
  normals only, along with a 300-record own prototype (`fit` clean normals). These checks chose the sources: Uni-G
  QRS, PR and QT, and 12SL P duration. ECGdeli and own intervals were left out of the primary. No fold 9 or 10 row
  outside ECGAD was read at any point: the feature files were filtered to the measured records right after
  reading.
- Per-lead criteria that were not possible: a per-lead short QT (amendment 1), and a truly per-lead PR. Uni-G's P
  onset is common to all leads, and 12SL and ECGdeli give no usable per-lead PR. A per-lead P duration is available
  only from 12SL, because Uni-G's equals its global value.
- The `perlead` and 069 numbers use 12SL and Uni-G measurements, which go beyond the 061 map contract, as 069
  declared. `perlead_own` keeps to the contract and is below the bar on A2 (-0.0426 [-0.0793, -0.0053]).
- The strata and noise CLIs wrote into copies of the reference tables under `ref/`. No 061 or 069 file was
  written.

## Runtime

CPU only, at most 8 processes: measurements 128 s (11,154 records), evidence and maps 59 s, scoring with integrity,
six LocBench scorings, strata and noise 698 s. That is about 15 minutes in total. The maps (7 GB, float32) are in
`/tmp/ecg073-work/full`.

## Why lead localization did not recover

1. The interval criteria were only part of the problem. The lead-specific criteria that fire in many abnormal
   ECGs, and in some normals (ST depression, T inversion, V1 R/S, wide P), still spread evidence over many leads.
   So `perlead` still averages 8.1 evidence leads on marked records.
2. Per-lead intervals from Uni-G vary only moderately between leads (median within-record range 30 ms for QRS, in
   normals). In a wide-QRS ECG most leads still cross 110 or 120 ms.
3. Beat-level criteria add time-local evidence (A2 +0.008, A4 +0.004) but cannot change a lead's 95th-percentile
   score, so they cannot help A3 as defined.

## Follow-up ideas (for the backlog)

1. Rank evidence within a record. Score each lead by its share of the record's criteria evidence, or keep only the
   top-k leads by exceedance, so that A3 rewards the leads that exceed the most rather than every lead that crosses
   a threshold.
2. Pair the beat-level criteria with a lead score that sees focal beats (for example, the maximum over beats of the
   beat mean instead of the 95th percentile over samples), as a declared secondary LocBench reading for focal
   findings. Count the ECGAD records with PVC statements first.
3. Combine `perlead_qrs` (0.766, label-free) with `perlead_agree`'s reliability. The factor was kept here: it lowers
   C1_std and raises A3 by +0.022. A QRS-prior map with the factor is the cheapest label-free candidate for the next
   marked set.
4. Ask the cardiologist whether the wide-P rule (24.6% of clean normals even on lead II's own P wave) and the V1 R/S
   rule (5.7%) should be kept, raised or dropped.

## Reproduce

```bash
cd /tmp/ecg-wt-073
export PYTHONPATH=/tmp/ecg-wt-073 OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
.venv/bin/python -u -m scripts.experiments.run_criteria073 measure
.venv/bin/python -u -m scripts.experiments.run_criteria073 maps
.venv/bin/python -u -m scripts.experiments.run_criteria073 score
```

The measure and maps phases read 069's work folder `/tmp/ecg069-work/full` (`measurements.pkl`,
`evidence_scored.npz`).
