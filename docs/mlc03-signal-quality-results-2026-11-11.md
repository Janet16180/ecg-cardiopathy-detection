# MLC-03 Part A results: signal quality

11 November 2026. Executes Part A of the [MLC-03 protocol](mlc03-data-quality-protocol-2026-11-11.md). No model
was trained or scored. Measures and thresholds were developed on the train and validation records only. After
the flags were committed, the frozen code ran once on the three test roles (`test_same_source`, PTB-XL fold 10;
`test_external` and `sph_calibration`, the two SPH halves) to count flags and near-duplicates. Their labels and
stored predictions were not read (`labels_read: false` in `final_test_roles.receipt.json`) and none of their
waveforms was plotted. No Challenge test group, EchoNext or MIMIC record was opened.

Every number below comes from an output under `outputs/mlc03_data_quality_20261111/` (`signal/` and
`review/signal/`), named in brackets, or from the table printed by
`scripts/reports/mlc03_signal_tables.py --section <name>` from those outputs, cited as [section `<name>`]. The
visual-check counts are readings of the plots made for this report. They are not clinical readings; clinical
validation is pending (validación pendiente).

## Summary

- 72,190 train and validation candidates were measured: the 71,388 included records (train 56,250, validation
  15,138) and 802 duplicate copies that MLC-01 excluded [`measures_train_validation.receipt.json`, section
  `flags-role`]. The 63 train and validation records shorter than 10 s had already been excluded by MLC-01 and
  were not measured [section `short-records`].
- 6,015 included records (8.4%) carry at least one of the 18 flags. By source: Ningbo 3,753 of 27,891, PTB-XL
  972 of 19,601, Chapman 564 of 8,133, CPSC 2018 353 of 5,014, CPSC-Extra 190 of 2,589, Georgia 183 of 8,160
  [section `flags`].
- 15 flags pass the protocol's visual check (at least 20 of 24 flagged plots show the defect, or the same share
  when fewer than 24 records carry the flag). Two of the 15, `reversal_la_ra` and `reversal_neutral`, pass only
  in a second version whose check reused the first viewing, so they are not independently confirmed. Two flags
  fail: `reversal_ra_ll` (1 of 12 viewed) and `rhythm_unmeasurable` (4 of 12, then 3 of 12 after a revision).
  `duplicate_lead` flags no record [section `tallies`].
- None of the four MLC-02 input pipelines removes clipping, baseline steps, baseline wander, contact loss or
  flat leads: the median ratio of the defect measure after the pipeline to its 500 Hz value is 0.89 to 1.02
  [`pipeline_effects/pipeline_effects_summary.csv`]. Missing samples enter every pipeline as the rail value
  -32.768 mV. FFT resampling spreads that value over up to 24% of the lead's xECG input, when 0.1% to 1.2% of
  the lead was missing at 500 Hz [`pipeline_effects/missing_sample_spread.csv`]. FFT resampling also rings at the
  record edges in every source [`pipeline_effects/edge_ringing_by_source.csv`].
- No near-duplicate beyond exact hashes was found to be a copy. Of the 10 non-identical train and validation
  pairs scoring 0.95 or more, none shares noise above 60 Hz. Two are two recordings of one person or probably
  so, one of them across train and validation (Ningbo JS19048 and JS19053)
  [`review/signal/near_duplicates/tally.csv`]. No test record has the same first 10 s as a record of another
  role. The highest cross-role scores involving a test role are 0.9554 and 0.9532, and no such pair reaches 0.98
  [section `near-test-max`].
- Proposed rules for Part C, not adopted here: six record exclusions would remove 148 train and 47 validation
  records (0.27%), 37 SPH calibration and 39 SPH test records, and no fold-10 record. Four lead-level rules would
  mark flat, near-flat, contact-loss and inconsistent derived leads [section `rules`]. Part C allows only record
  exclusion and undefined labels, so the lead-level rules need a decision.
- On the test roles, fold 10 has 129 flagged records of 2,198 and SPH has 463 of 25,581 [section `flags-role`].
  SPH has 174 identity violations and 64 records with a value of 32.7 mV or more. Both definitions were made
  for 16-bit integer storage, and SPH stores float16 samples. The cause of these counts was not examined.

## Code, runs and receipts

Branch `research/mlc03-signal-quality-20261111`, from `research/mlc03-data-quality-20261111` (`1f6563d`, which
holds the protocol). New files only:

- `ecg_experiment/mlc03_signal_quality.py`: per-window and full-record measures.
- `ecg_experiment/mlc03_signal_runs.py`: role guards, cache reading with hash checks, parallel runs and receipts.
- `ecg_experiment/mlc03_signal_flags.py`: flag definitions, thresholds and reasons (`FLAGS`) and their
  computation.
- `ecg_experiment/mlc03_signal_review.py`: contact sheets.
- `ecg_experiment/mlc03_near_duplicates.py`: near-duplicate searches.
- `ecg_experiment/mlc03_pipeline_effects.py`: what the MLC-02 input pipelines do to defects.
- Scripts and tests: `scripts/experiments/run_mlc03_signal_quality.py`, `scripts/reports/mlc03_record_review.py`
  (records review verdicts), `scripts/reports/mlc03_signal_tables.py` (report tables) and
  `tests/test_mlc03_signal_quality.py`.

Commits, in order:

- `4ae164d` and `3063fda`: measures, runner, review sheets and tests. The measure and full-record runs used
  `3063fda`.
- `34f84fa`, `d9dde21`, `f616ec3`, `13908a2`, `e72be1e`: flags, near-duplicate search, review sheets and verdict
  recording. The near-duplicate search ran at `13908a2`, its review at `e72be1e`.
- `ee4280b`: second versions of five flags after the first visual check. The final flags were written at this
  commit.
- `062bd83`, `07db3c3`: pipeline effects and the frozen test-role step. The pipeline run used `07db3c3`.
- `cb6563e`, `60483e7`, `ab274e4`, `8727265`, `3215781`, `bb09e0b`, `6d38134`: report tables; at `bb09e0b` also a
  repository-relative table path in the measure receipt for future runs.

Command: `PYTHONPATH=. .venv/bin/python -m scripts.experiments.run_mlc03_signal_quality <step>`, with
`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` at 1 or 2. Scratch files are in
`/var/tmp/janetrivera/mlc03/` and were not copied.

| step | records | workers | wall time | commit in receipt | receipt |
|---|---|---|---|---|---|
| `measure` | 72,190 | 16 | 296 s | `3063fda` | `signal/measures_train_validation.receipt.json` |
| `full-records` | 5,173 | 16 | 14 s | `3063fda` | `signal/full_records_train_validation.receipt.json` |
| `near-duplicates` | 72,190 | 10 | 743 s | `13908a2` | `signal/near_duplicates/near_duplicates_train_validation.receipt.json` |
| `flags` | 72,190 | - | not recorded | `ee4280b` | `signal/flags_train_validation.receipt.json` |
| `review --flag <flag>` | 24 + 24 per flag | - | not recorded | per sheet | `review/signal/<flag>/receipt.json` |
| `near-review` | 10 + 24 pairs | - | not recorded | `e72be1e` | `review/signal/near_duplicates/receipt.json` |
| `pipelines` | 104 + 600 | - | not recorded | `07db3c3` | `signal/pipeline_effects/pipeline_effects.receipt.json` |
| `final` | 27,968 (27,779 included) | 14 | 514 s | `ab274e4` (see Departures) | `signal/final_test_roles.receipt.json` |

The receipts record the command, the commit, modified tracked files, thread variables, package versions,
SHA-256 of the input tables and code, and the uncorrected-memory-error counter (`edac_uncorrected`). The counter
was 1 before the work began and is 1 in all 39 receipts. Verdicts are recorded with
`python -m scripts.reports.mlc03_record_review <flag> --flagged <y/n/- per panel> --note "<panel>=<text>"`,
which writes `tally.csv` and `tally.json` next to the sheets.

## Records and inputs

Each record's first 10 s came from the MLC-02 waveform cache (500 Hz, mV, NaN for missing samples), and each
window was checked against the MLC-01 `sha256_10s` when read. Records longer than 10 s (CPSC 2018 and CPSC-Extra
only) were also measured over the whole raw recording [section `short-records`]:

| source | train and validation records | shorter than 10 s (excluded by MLC-01) | exactly 10 s | longer |
|---|---|---|---|---|
| chapman_shaoxing | 8,197 | 0 | 8,197 | 0 |
| cpsc_2018 | 5,496 | 10 | 1,923 | 3,563 |
| cpsc_2018_extra | 2,766 | 10 | 1,146 | 1,610 |
| georgia | 8,273 | 43 | 8,230 | 0 |
| ningbo | 27,920 | 0 | 27,920 | 0 |
| ptbxl | 19,601 | 0 | 19,601 | 0 |

Integrity check against the predecessor: the flat-lead and missing-sample flags reproduce MLC-01's lead flags
exactly on all 72,190 candidates [section `mlc01-agreement`]:

| check | flagged here | flagged by MLC-01 | disagreements |
|---|---|---|---|
| record with a flat lead | 1,238 | 1,238 | 0 |
| record with missing samples (MLC-01: whole recording) | 87 | 87 | 0 |
| flat leads, counted per record | 5,770 | 5,770 | 0 |

The measures table has one row per record and 304 columns [`measures_train_validation.parquet`]. Record-level
summaries are in `record_summaries_train_validation.parquet` and their quantiles per source in
`measure_quantiles_train_validation.csv`.

## Flags

Full definitions, thresholds and reasons are in `signal/flag_definitions.csv`, written from `FLAGS`. Flags were
computed on every candidate; counts here are of included records, as in `flag_counts_train_validation.csv`
(`flag_counts_train_validation_all.csv` counts all candidates).

| flag | item | flagged when | threshold |
|---|---|---|---|
| `missing_samples` | 1 | a sample of the first 10 s reads as missing (stored -32768) | more than 0 |
| `edge_hold` | 1 | every non-constant lead holds its first (or last) value together | at least 150 ms |
| `adc_rail` | 1 | a sample at the 16-bit storage limit | absolute value at least 32.7 mV |
| `flat_lead` | 2 | a lead is constant over the 10 s | range 0 |
| `near_flat_lead` | 2 | 1-99 percentile range of a non-constant lead's 0.5-40 Hz signal | below 0.05 mV |
| `duplicate_lead` | 2 | two non-constant leads identical or sign-flipped, unless forced by a constant I or II | any pair |
| `identity_violation` | 3 | largest residual of III = II - I, aVR = -(I + II)/2, aVL = I - II/2, aVF = II - I/2 | above 0.05 mV |
| `reversal_la_ra` | 3 | median beat: lead I inverted (P -0.3 to -0.05 mV, QRS and T negative), QRS upright in aVR and V6, QRS below 120 ms, beat similarity at least 0.8 | all eight |
| `reversal_ra_ll` | 3 | median beat: lead II inverted (P, QRS, T) and aVR upright (QRS, P above 0.05 mV) | all five |
| `reversal_neutral` | 3 | lead I, II or III nearly flat: 0.5-40 Hz range below 0.06 mV and below 10% of the largest of the three | both |
| `amplitude_outlier` | 4 | median lead range against the pooled median of 0.93 mV | ratio at least 4 or at most 1/4 |
| `baseline_wander` | 5 | peak-to-peak below 0.5 Hz, 0.3-9.7 s | at least 2.0 mV |
| `mains_interference` | 5 | 50 or 60 Hz sinusoid above the neighbouring spectrum, in leads whose largest second difference is below 2 mV | at least 0.05 mV |
| `hf_noise` | 5 | median 50 ms moving RMS above 40 Hz with mains lines removed, same leads | at least 0.05 mV |
| `baseline_step` | 5 | largest 60 ms change of the median-filtered baseline, 0.3-9.7 s | at least 1.0 mV |
| `isolated_spike` | 5 | second difference of I, II or V1-V6 at an instant where at most two independent leads exceed 0.5 mV | at least 2.0 mV |
| `contact_loss` | 5 | independent lead without QRS (range below 0.05 mV within 60 ms of the R peak) for consecutive beats, or a lead holding one value, 0.3-9.7 s | 3 beats or 500 ms |
| `rhythm_unmeasurable` | 6 | multi-lead detector finds 3 or more beats, but no two of three beat lists (multi-lead, XQRS on II, XQRS on V5) agree | every pairwise F1 below 0.8 |

Why each threshold, from the per-source quantiles of included train and validation records [section
`quantiles`, `measure_quantiles_train_validation.csv`]:

- `missing_samples`, `adc_rail`, `flat_lead`, `duplicate_lead`: by definition. -32768 is the WFDB code for an
  invalid sample, and every training source stores 1,000 units per mV in 16 bits, so 32.767 mV is the storage
  limit.
- `edge_hold`: PTB-XL holds every lead at the end of every record for 90 to 130 ms (`edge_hold_ms`: minimum 90,
  1st percentile 100, maximum 130 ms, over 19,601 records). The other sources' 99th percentile is 8 ms or less.
  150 ms lies above both.
- `near_flat_lead`: the 0.01st percentile of a non-constant lead's range is 0.070 mV (aVL) or more in every lead
  [section `lead-amplitude`].
- `identity_violation`: the 99th percentile of the largest residual is 0.0035 mV or less in every source.
- `amplitude_outlier`: source medians of the amplitude ratio are 0.947 to 1.014, so no source differs by a
  constant factor (0.88 to 0.94 mV). The 10th to 90th percentile spread of log10 of the ratio, divided by 2.56,
  is 0.13 to 0.15 in every source, so a factor of 4 (0.60 in log10) is 3.9 to 4.5 robust standard deviations.
- `baseline_wander`: 2.0 mV is twice the median lead amplitude. The 99th percentiles of the largest excursion are
  1.78 mV (Georgia), 2.01 mV (CPSC-Extra), 2.18 mV (CPSC 2018), 2.72 mV (Chapman), 3.64 mV (PTB-XL) and 5.01 mV
  (Ningbo).
- `mains_interference`: 0.05 mV (0.1 mV peak to peak) is one small box at standard gain. The 99th percentile of
  the largest 50 Hz amplitude is 0.014 to 0.082 mV by source.
- `hf_noise`: medians of the largest value are 0.007 to 0.013 mV and 99th percentiles 0.029 to 0.066 mV.
- `baseline_step`: the median filters remove P, QRS and T. The 24 flagged plots all show artefacts (below).
- `isolated_spike`: restricting to isolated instants cut the paced records among records with a 2 mV spike from
  649 to 47 (of 1,297 paced records) [section `paced-spikes`]. The record's largest second difference is 0.70 mV
  or less at the 90th percentile in every source (`max_spike`).
- `contact_loss`: PTB-XL holds exact values inside the record for up to 178 samples (356 ms) at the 99.9th
  percentile (`max_const_run_interior`), so the hold threshold is 500 ms. The 0.01st percentile robust amplitude of
  a precordial lead is 0.128 mV (V1) or more.
- `rhythm_unmeasurable`: two detections that agree to F1 0.8 (matching within 75 ms) give a usable rhythm.
- The reversal flags are stated rules from the electrode-reversal patterns; V6 separates left-right arm reversal
  from dextrocardia.

### Counts per source, included train and validation records

[section `flags`]

| flag | ptbxl | ningbo | chapman_shaoxing | georgia | cpsc_2018 | cpsc_2018_extra | total |
|---|---|---|---|---|---|---|---|
| missing_samples | 0 | 82 | 0 | 5 | 0 | 0 | 87 |
| edge_hold | 0 | 14 | 0 | 0 | 1 | 0 | 15 |
| adc_rail | 1 | 77 | 0 | 3 | 22 | 13 | 116 |
| flat_lead | 1 | 1202 | 13 | 6 | 9 | 5 | 1236 |
| near_flat_lead | 1 | 4 | 0 | 5 | 1 | 2 | 13 |
| duplicate_lead | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| identity_violation | 6 | 0 | 0 | 0 | 28 | 13 | 47 |
| reversal_la_ra | 2 | 0 | 0 | 5 | 0 | 1 | 8 |
| reversal_ra_ll | 12 | 44 | 17 | 1 | 3 | 1 | 78 |
| reversal_neutral | 0 | 4 | 0 | 4 | 0 | 0 | 8 |
| amplitude_outlier | 4 | 22 | 1 | 5 | 19 | 14 | 65 |
| baseline_wander | 748 | 990 | 180 | 61 | 66 | 27 | 2072 |
| mains_interference | 23 | 456 | 131 | 38 | 26 | 17 | 691 |
| hf_noise | 47 | 102 | 39 | 18 | 10 | 6 | 222 |
| baseline_step | 276 | 1303 | 228 | 52 | 263 | 138 | 2260 |
| isolated_spike | 60 | 709 | 75 | 20 | 47 | 16 | 927 |
| contact_loss | 3 | 275 | 1 | 14 | 36 | 28 | 357 |
| rhythm_unmeasurable | 26 | 146 | 3 | 13 | 18 | 18 | 224 |
| any_flag | 972 | 3753 | 564 | 183 | 353 | 190 | 6015 |
| records | 19601 | 27891 | 8133 | 8160 | 5014 | 2589 | 71388 |

### Counts per role and source, including the test roles

[section `flags-role`; test roles from `flag_counts_test_roles.csv`]

| flag | train chapman | train cpsc | train extra | train georgia | train ningbo | train ptbxl | val chapman | val cpsc | val extra | val georgia | val ningbo | val ptbxl | fold 10 ptbxl | SPH test | SPH calibration |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| missing_samples | 0 | 0 | 0 | 3 | 63 | 0 | 0 | 0 | 0 | 2 | 19 | 0 | 0 | 0 | 0 |
| edge_hold | 0 | 1 | 0 | 0 | 9 | 0 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 |
| adc_rail | 0 | 16 | 10 | 2 | 60 | 1 | 0 | 6 | 3 | 1 | 17 | 0 | 0 | 31 | 33 |
| flat_lead | 10 | 7 | 1 | 6 | 919 | 1 | 3 | 2 | 4 | 0 | 283 | 0 | 0 | 0 | 0 |
| near_flat_lead | 0 | 1 | 2 | 4 | 4 | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| duplicate_lead | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| identity_violation | 0 | 20 | 10 | 0 | 0 | 4 | 0 | 8 | 3 | 0 | 0 | 2 | 1 | 71 | 103 |
| reversal_la_ra | 0 | 0 | 0 | 4 | 0 | 2 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| reversal_ra_ll | 13 | 3 | 1 | 1 | 38 | 10 | 4 | 0 | 0 | 0 | 6 | 2 | 1 | 2 | 1 |
| reversal_neutral | 0 | 0 | 0 | 3 | 3 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 |
| amplitude_outlier | 0 | 13 | 11 | 4 | 21 | 3 | 1 | 6 | 3 | 1 | 1 | 1 | 0 | 18 | 19 |
| baseline_wander | 138 | 51 | 19 | 47 | 733 | 649 | 42 | 15 | 8 | 14 | 257 | 99 | 100 | 113 | 92 |
| mains_interference | 89 | 17 | 11 | 30 | 330 | 19 | 42 | 9 | 6 | 8 | 126 | 4 | 2 | 0 | 3 |
| hf_noise | 28 | 7 | 5 | 14 | 77 | 43 | 11 | 3 | 1 | 4 | 25 | 4 | 5 | 5 | 8 |
| baseline_step | 166 | 199 | 100 | 41 | 954 | 246 | 62 | 64 | 38 | 11 | 349 | 30 | 30 | 158 | 141 |
| isolated_spike | 56 | 34 | 11 | 15 | 529 | 50 | 19 | 13 | 5 | 5 | 180 | 10 | 7 | 15 | 19 |
| contact_loss | 1 | 29 | 18 | 11 | 204 | 3 | 0 | 7 | 10 | 3 | 71 | 0 | 0 | 2 | 2 |
| rhythm_unmeasurable | 3 | 13 | 13 | 10 | 103 | 22 | 0 | 5 | 5 | 3 | 43 | 4 | 10 | 0 | 3 |
| any_flag | 411 | 262 | 134 | 145 | 2794 | 843 | 153 | 91 | 56 | 38 | 959 | 129 | 129 | 227 | 236 |
| records | 6097 | 3763 | 1936 | 6117 | 20919 | 17418 | 2036 | 1251 | 653 | 2043 | 6972 | 2183 | 2198 | 12698 | 12883 |

## Visual check

For every flag, 24 flagged and 24 unflagged included train and validation records were drawn and plotted with
the flagged lead and time marked (`review/signal/<flag>/flagged_*.png`, `unflagged_*.png`, `sample.csv`). A flag
with fewer than 24 records had all of them drawn. Each flagged plot was read for whether it shows the defect the
flag describes [section `tallies`, `review/signal/<flag>/tally.csv`]:

| review | seed | flagged records | drawn | viewed | show the defect | controls viewed | controls with the defect | reliable |
|---|---|---|---|---|---|---|---|---|
| missing_samples | 20261111 | 87 | 24 | 24 | 24 | 8 | 0 | yes |
| edge_hold | 20261111 | 15 | 15 | 15 | 15 | 8 | 0 | yes |
| adc_rail | 20261111 | 116 | 24 | 24 | 24 | 8 | 0 | yes |
| flat_lead | 20261111 | 1236 | 24 | 24 | 24 | 0 | - | yes |
| near_flat_lead | 20261111 | 13 | 13 | 13 | 13 | 8 | 0 | yes |
| identity_violation | 20261111 | 47 | 24 | 24 | 24 | 8 | 0 | yes |
| reversal_la_ra, first version | 20261111 | 15 | 15 | 15 | 7 | 0 | - | no |
| reversal_la_ra, second version | 20261112 | 8 | 8 | 8 | 7 (carried over) | 0 | - | yes, not independent |
| reversal_ra_ll | 20261111 | 78 | 24 | 12 | 1 | 0 | - | no |
| reversal_neutral, first version | 20261111 | 11 | 11 | 11 | 8 | 0 | - | no |
| reversal_neutral, second version | 20261112 | 8 | 8 | 8 | 8 (carried over) | 0 | - | yes, not independent |
| amplitude_outlier | 20261111 | 65 | 24 | 24 | 20 | 0 | - | yes |
| baseline_wander | 20261111 | 2072 | 24 | 24 | 24 | 24 | 0 | yes |
| mains_interference, first version | 20261111 | 959 | 24 | 24 | 19 | 0 | - | no |
| mains_interference, second version | 20261112 | 691 | 24 | 24 | 20 | 0 | - | yes |
| hf_noise, first version | 20261111 | 629 | 24 | 12 | 5 | 0 | - | no |
| hf_noise, second version | 20261112 | 222 | 24 | 24 | 23 | 0 | - | yes |
| baseline_step | 20261111 | 2260 | 24 | 24 | 24 | 8 | 1 | yes |
| isolated_spike | 20261111 | 927 | 24 | 24 | 24 | 0 | - | yes |
| contact_loss | 20261111 | 357 | 24 | 24 | 21 | 0 | - | yes |
| rhythm_unmeasurable, first version | 20261111 | 335 | 24 | 12 | 4 | 0 | - | no |
| rhythm_unmeasurable, second version | 20261112 | 224 | 24 | 12 | 3 | 0 | - | no |

Pools are from the review receipts [section `pools`]. Where viewing stopped at 12, at most 13 to 17 of 24 could
have shown the defect, below 20. What the plots showed, from the notes in each `tally.csv`:

- `missing_samples`: in all 24, the lead swings to the rails around the gaps and carries no QRS afterwards.
- `edge_hold`: the hold is at the end of the record in the 14 Ningbo records and at the start in CPSC A3816.
- `adc_rail`: in Ningbo, one chest lead saturates, shows a gap or rail samples, then carries no QRS. In 7 of the
  9 CPSC 2018 and CPSC-Extra records drawn, all twelve leads reach the rail at the same instant.
- `flat_lead`: two Ningbo patterns, all six chest leads zero (the commonest), and V2, V4 and V6 zero with V1, V3
  and V5 recorded.
- `near_flat_lead`: 6 of 13 show one limb lead reduced to noise while the other two carry large QRS, the
  neutral-electrode reversal pattern; CPSC-Extra Q2436 has no QRS in any lead.
- `identity_violation`: 15 of 24 are a wrap-around in CPSC 2018 or CPSC-Extra, where the stored derived lead jumps
  about 65.5 mV from the recomputed value (a 16-bit overflow). 6 are a constant offset between stored and
  recomputed aVR (-0.08 mV in Q2790). 3 are PTB-XL single-sample spikes of about 0.1 mV in one derived lead. The
  99.9th percentile of the residual is 65.50 mV in CPSC 2018 and 65.52 mV in CPSC-Extra.
- `reversal_la_ra`: the first version's 8 misses were pacing spikes measured as P waves (for example -2.29 mV),
  wide complexes, noisy limb leads and a saturated V6. The second version adds the P bound, the QRS width and the
  beat similarity.
- `reversal_ra_ll`: most flagged plots show a left-axis or inferior-Q morphology, not a reversal.
- `reversal_neutral`: the first version's 3 misses (ranges 0.064 to 0.084 mV) kept small QRS deflections in step
  with the beats.
- `amplitude_outlier`: the 4 misses are 2 low-side records (ratio 0.23, a plausible low-voltage ECG and small noisy
  limb leads), one PTB-XL record with large but plausible complexes and one record whose amplitude is set by
  pacemaker spikes. Neither low-side record drawn shows a defect. Ningbo JS18956 and JS19039 to JS19049 are
  consecutive records with the same wide-complex morphology at 5 to 6.5 times the usual amplitude in every lead.
  That is probably a recording gain not reflected in the stored scale, which the waveform alone cannot confirm.
- `baseline_wander`: the flag catches any large slow excursion, not only wander. The notes name true wander in 6
  plots, start-of-record transients in 5, steps with exponential decay in 4, leads dead after saturation in 2 and
  a dip in one lead in 1.
- `mains_interference`: the first version's misses were paced records (pacing spikes put broadband power at 50
  and 60 Hz) and leads with 0.06 to 0.09 mV and no visible oscillation. The second version leaves out leads with a
  2 mV second difference. Its 4 misses are leads at 0.052 to 0.077 mV with no visible oscillation. One of the 20
  hits is a decaying oscillation burst rather than steady mains.
- `hf_noise`: in the first version, removing the mains bins of a lead with 10 mV pacing spikes spread their
  energy over the whole lead. The second version leaves such leads out; its one miss is at 0.051 mV.
- `baseline_step`: steps, dips with recovery, transients and jumps after saturation. One of the 8 controls viewed
  shows a step below 1 mV.
- `isolated_spike`: sharp spikes, oscillation bursts, rectangular pulses and saturation in one chest lead.
- `contact_loss`: hits include a lead without ECG for the first seconds (the commonest Ningbo pattern) and all
  leads holding one value for the first 1.3 s in CPSC records. The 3 misses (Georgia E05479 and E03189,
  CPSC-Extra Q1676) are real ECGs with an exact hold of 504 to 552 ms between slow beats.
- `rhythm_unmeasurable`: in most misses the beats are visible by eye. Detectors disagree because of artefacts in
  some leads, wide or premature complexes, or pacing, and one saturation glitch can set the multi-lead detector's
  relative threshold.

## Findings by source

- PTB-XL holds all twelve leads constant for the last 90 to 130 ms of every record (`min_tail_run` 45 to 65
  samples) [section `quantiles`]. It also holds exact values inside the record more often than the other sources.
  The median of the longest interior exact run is 66 samples (132 ms) against 9 to 30 elsewhere
  (`max_const_run_interior`).
- Chapman, Ningbo and Georgia store values on a grid of about 4.88 µV (median quantisation step 4.88, 4.88 and
  4.83 µV), while their headers give 1,000 units per mV. PTB-XL, CPSC 2018 and CPSC-Extra have a 1.0 µV step
  (`median_quant_step_uv`).
- No source has a gain or unit error at source level: source medians of the amplitude ratio are 0.947 to 1.014
  (`amplitude_ratio`).
- Ningbo carries most lead defects. 1,202 of 27,891 records (4.3%) have a constant lead, always a chest lead
  except one lead I [section `leads --flag flat_lead`]. Ningbo also has 82 of the 87 records with missing samples
  and 275 of the 357 with contact loss. Its contact loss is most often in V6 (124 records) and V5 (75) [section
  `leads --flag contact_loss`].
- CPSC 2018 and CPSC-Extra have derived-lead wrap-around (identity violation) and all-lead rail events, and, in
  their long recordings, defects that begin after the first 10 s (next section).
- Georgia has 5 of the 8 left-right arm reversals and 4 of the 8 neutral-electrode reversals.

The MLC-02 measurement extractor status [`outputs/mlc02_condition_models_20261110/features/measurements_status.csv`]
is `ok` on all 65,373 unflagged included records and on 77.8% of flagged ones [section `measurement-status`]. It
reports `derivacion_plana` on every flat-lead record. It still reports `ok` on 37.9% of records with missing
samples, 55.2% with rail samples, 97.9% with identity violations, 88.8% with contact loss and 85.5% with
isolated spikes. So the extractor's status does not catch most of these defects.

## Long recordings

Defects beyond the first 10 s of the 5,173 longer recordings, all CPSC 2018 or CPSC-Extra [section
`long-records`, `full_records_train_validation.parquet`]:

| source | long records | median s | max s | missing samples | rail only after 10 s | lead constant only after 10 s | constant run of 1 s or more anywhere | all-lead end hold of 150 ms or more |
|---|---|---|---|---|---|---|---|---|
| cpsc_2018 | 3,563 | 15.0 | 132.0 | 0 | 18 | 0 | 21 | 0 |
| cpsc_2018_extra | 1,610 | 15.9 | 88.0 | 0 | 12 | 0 | 5 | 0 |

The flags use the first 10 s, the window the MLC-02 models read. A model that reads another crop of these
recordings would see 30 rail events that the flags miss.

## What the input pipelines do to the defects

The four MLC-02 pipelines, reused by import (`ecg_experiment/mlc03_pipeline_effects.py`):

- `xecg`: FFT resampling of the 10 s to 100 Hz, raw mV; also the input of feature model F.
- `jepa`: leads I, II and V1 to V6 only, FFT resampling to 250 Hz, raw mV.
- `cpc`: four 2.5 s crops, each resampled to 240 Hz with `resampy`, raw mV.
- `scratch`: one polyphase pass to 100 Hz, then a single training mean and standard deviation (an affine map,
  applied after the measure).

All four first set missing samples to the stored rail value, -32.768 mV. For each flag, the first 8 records whose
plot showed the defect were taken in review order, and the same measure was computed on the flagged lead at
500 Hz and on each pipeline input. The table shows the median ratio (input to 500 Hz) and the median 500 Hz
value [`pipeline_effects/pipeline_effects_summary.csv`, 104 records]:

| flag | measure | xecg | jepa | cpc | scratch | median 500 Hz value (mV) |
|---|---|---|---|---|---|---|
| adc_rail | largest absolute value | 0.972 | 0.997 | 1.008 | 0.974 | 32.765 |
| amplitude_outlier | 1-99 percentile range | 0.999 | 1.003 | 1.003 | 0.998 | 4.485 |
| baseline_step | baseline range below 0.5 Hz | 1.021 | 1.000 | 0.996 | 1.000 | 1.482 |
| baseline_wander | baseline range below 0.5 Hz | 0.889 | 0.909 | 0.981 | 1.000 | 2.686 |
| contact_loss | 1-99 percentile range | 0.995 | 0.998 | 0.987 | 0.993 | 1.735 |
| edge_hold | 1-99 percentile range | 0.973 | 0.998 | 1.001 | 0.972 | 0.842 |
| hf_noise | RMS above 40 Hz | 0.403 | 0.906 | 0.854 | 0.359 | 0.057 |
| identity_violation | largest absolute value | 0.201 | not in input | 0.422 | 0.201 | 32.753 |
| isolated_spike | largest absolute value | 0.891 | 0.984 | 0.985 | 0.887 | 7.194 |
| mains_interference | 50 Hz amplitude | 1.220 | 1.000 | 1.000 | 0.590 | 0.076 |
| missing_samples | 1-99 percentile range | 0.945 | 0.980 | 0.997 | 0.946 | 36.773 |
| near_flat_lead | 1-99 percentile range | 0.974 | 0.987 | 0.981 | 0.974 | 0.060 |

Flat leads stay flat in every input (range 0).

- Clipping, amplitude outliers, steps, wander, contact loss, held edges and near-flat leads reach every model
  almost unchanged, since no pipeline filters the baseline or normalises per record.
- High-frequency noise is cut to about 0.4 of its value at 100 Hz (only 40 to 50 Hz remains) and kept at 0.85 to
  0.91 by JEPA and CPC.
- 50 Hz mains is kept at the Nyquist frequency by xECG's FFT resampling (ratio 1.22, measured in the 47-50 Hz
  band) and halved by the polyphase filter of the scratch pipeline.
- The single-sample wrap-around spikes of derived leads are smoothed to 0.20 (100 Hz) and 0.42 (CPC) of their
  height. JEPA does not read derived leads.
- Missing samples: 8 Ningbo records, one lead each, were run as stored and with the missing samples interpolated
  [`pipeline_effects/missing_sample_spread.csv`]. At 500 Hz, 0.1% to 1.2% of the lead is missing. In the xECG
  input, 0.2% to 24% of the lead's samples differ by more than 0.05 mV, with a largest difference of 0.8 to
  45.3 mV. The shares are 0.3% to 12.5% for JEPA, 0.3% to 3.8% for CPC and 0.2% to 4.6% for scratch.

FFT resampling treats the 10 s as periodic, so a difference between the first and last samples rings near both
edges. Against a polyphase pass at the same rate, on 100 included records per source (seed 20261111), the largest
difference over all leads is [`pipeline_effects/edge_ringing_by_source.csv`]:

| source | xecg edge, median | xecg interior, median | xecg edge, 90th percentile | jepa edge, median | jepa interior, median | jepa edge, 90th percentile |
|---|---|---|---|---|---|---|
| chapman_shaoxing | 0.089 | 0.017 | 0.485 | 0.057 | 0.006 | 0.317 |
| cpsc_2018 | 0.070 | 0.011 | 0.218 | 0.043 | 0.003 | 0.152 |
| cpsc_2018_extra | 0.070 | 0.010 | 0.430 | 0.046 | 0.003 | 0.237 |
| georgia | 0.070 | 0.010 | 0.222 | 0.043 | 0.004 | 0.140 |
| ningbo | 0.117 | 0.016 | 0.328 | 0.074 | 0.006 | 0.219 |
| ptbxl | 0.089 | 0.011 | 0.374 | 0.057 | 0.004 | 0.230 |

Values in mV; the edge is the first and last 0.2 s.

## Near-duplicates

Two searches on fingerprints (each lead high-passed at 0.5 Hz, resampled to 50 Hz and z-scored;
`ecg_experiment/mlc03_near_duplicates.py`):

1. Zero lag: every pair of records scored by the mean correlation of leads I and II over the first 10 s; pairs at
   0.8 or more kept (1,300 pairs among train and validation).
2. Shifted copies: records sharing keys of three consecutive RR intervals at a consistent lag (the whole raw
   recording for long records), verified by fingerprint correlation over at least 4 s of overlap; 1,821,929
   candidates verified [`near_duplicates_train_validation.receipt.json`].

Train and validation pairs by score band [section `near-train`]:

| search | score | same first-10-s hash | other pairs | other pairs across train and validation |
|---|---|---|---|---|
| zero lag | 0.80-0.90 | 0 | 531 | 187 |
| zero lag | 0.90-0.95 | 0 | 75 | 22 |
| zero lag | 0.95-0.98 | 0 | 8 | 2 |
| zero lag | 0.98-0.99 | 0 | 1 | 0 |
| zero lag | 0.99 or more | 685 | 0 | 0 |
| shifted | 0.80-0.90 | 0 | 6,786 | 2,100 |
| shifted | 0.90-0.95 | 0 | 149 | 49 |
| shifted | 0.95-0.98 | 0 | 1 | 0 |
| shifted | 0.98 or more | 658 | 0 | 0 |

The 685 identical pairs are copies MLC-01's exact rule already handles. The shifted search recovers 658 of them,
at lag 0. All 10 non-identical pairs at 0.95 or more were plotted as overlays, each with its noise above 60 Hz
[`review/signal/near_duplicates/`]. None is a copy, since no pair shares its noise above 60 Hz:

- Chapman JS02795 and Ningbo JS14384 (0.9835, the highest): different people (47 M, 82 F), beats aligned by
  chance.
- PTB-XL 21419 and 21517 (shifted 0.9715): two recordings of one patient (12516), 1.3 s apart, both in one fold.
- Ningbo JS19048 (train) and JS19053 (validation) (0.9663): both 74 F, probably two recordings of one person
  across roles. The Challenge sources have no patient identifier, so the split cannot prevent this.
- The other 7 are paced or regular-rhythm pairs with similar morphology but different chest leads or noise.

The 24 control pairs from the 224 pairs at 0.90 to 0.95 were drawn but not viewed.

Across roles involving a test role (final run, frozen code; pairs at 0.90 or more kept by the zero-lag search)
[sections `near`, `near-test-max`]:

| search | test role | other role | pairs at 0.90 or more | at 0.95 or more | at 0.98 or more | highest |
|---|---|---|---|---|---|---|
| zero lag | sph_calibration | train | 4 | 1 | 0 | 0.9554 |
| shifted | sph_calibration | train | 29 | 0 | 0 | 0.9407 |
| shifted | sph_calibration | validation | 5 | 0 | 0 | 0.9288 |
| shifted | sph_calibration | test_external | 5 | 0 | 0 | 0.9248 |
| shifted | test_external | train | 21 | 0 | 0 | 0.9335 |
| shifted | test_external | validation | 7 | 0 | 0 | 0.9181 |
| shifted | test_external | sph_calibration | 11 | 0 | 0 | 0.9369 |
| shifted | test_same_source | train | 4 | 1 | 0 | 0.9532 |
| shifted | test_same_source | validation | 5 | 0 | 0 | 0.9402 |
| shifted | test_same_source | test_external | 3 | 0 | 0 | 0.9085 |
| shifted | test_same_source | sph_calibration | 0 | 0 | 0 | 0.8956 |

No zero-lag pair of 0.90 or more exists for the other role pairs, and no test record has the same first-10-s
hash as a record of another role. The pairs are listed in `near_duplicates_test_roles_pairs.csv`. They were not
plotted, because the test-role waveforms stay unviewed.

## Proposed rules for Part C

These are proposals with their counts. Part C adopts or rejects them [section `rules`,
`proposed_rule_counts.csv`]:

| rule | action | flag | visual check | train | validation | fold 10 | SPH calibration | SPH test |
|---|---|---|---|---|---|---|---|---|
| R1 | exclude record | missing_samples | 24/24 | 66 | 21 | 0 | 0 | 0 |
| R2 | exclude record | adc_rail | 24/24 | 89 | 27 | 0 | 33 | 31 |
| R3 | exclude record | edge_hold | 15/15 | 10 | 5 | 0 | 0 | 0 |
| R8 | exclude record | reversal_la_ra | 7/8, not independent | 6 | 2 | 0 | 0 | 0 |
| R9 | exclude record | reversal_neutral | 8/8, not independent | 6 | 2 | 0 | 0 | 0 |
| R10 | exclude record | amplitude_outlier, ratio 4 or more only | 20/22 on the high side | 38 | 11 | 0 | 19 | 18 |
| R1-R3, R8-R10 | union | | | 148 | 47 | 0 | 37 | 39 |
| R4 | mark lead unusable | flat_lead | 24/24 | 944 (4,450 leads) | 292 (1,307 leads) | 0 | 0 | 0 |
| R5 | mark lead unusable | near_flat_lead | 13/13 | 12 (12 leads) | 1 (1 lead) | 0 | 0 | 0 |
| R6 | mark lead unusable | contact_loss | 21/24 | 266 (454 leads) | 91 (125 leads) | 0 | 2 (3 leads) | 2 (2 leads) |
| R7 | mark derived leads unusable | identity_violation | 24/24 | 34 | 13 | 1 | 103 | 71 |

Union by source: train CPSC 2018 23, CPSC-Extra 15, Georgia 10, Ningbo 94, PTB-XL 6, Chapman 0; validation
Chapman 1, CPSC 2018 9, CPSC-Extra 4, Georgia 5, Ningbo 28, PTB-XL 0.

- R1 to R3: the stored signal is not a measurement (invalid code, clipped, or held).
- R8 and R9: a limb electrode was misplaced, so the limb leads describe another vector. Their checks reuse the
  first viewing.
- R10 covers the high side only: 20 of the 22 high-side plots drawn show a defect (rail samples set the scale,
  or the probable Ningbo gain series), but the misses include plausible large complexes. The low side (16
  records) has no supporting plot and no rule.
- R4 to R7 act on leads, not records. Part C allows only record exclusion and undefined labels. Excluding the
  R4 records instead would remove 4.3% of Ningbo; MLC-01 kept them with lead flags. This needs a decision.
- No rule is proposed for the noise flags (baseline wander, mains, high-frequency noise, steps, isolated spikes).
  They describe real defects (all reliable), but a noisy lead in an otherwise readable record is a matter of
  degree, and the counts are large (2,072 to 2,260 for wander and steps). `reversal_ra_ll` and
  `rhythm_unmeasurable` are unreliable and support no rule.

## Test roles

The frozen code (flags at `ee4280b`, final step at `07db3c3`) ran once on 27,968 test-role candidates (27,779
included): 2,198 fold-10 records, 12,698 SPH test and 12,883 SPH calibration records
[`final_test_roles.receipt.json`]. Counts by flag are in the role table above and in
`flag_counts_test_roles.csv`.

- PTB-XL fold 10: 129 flagged of 2,198 (5.9%, against 5.0% in PTB-XL train and validation), mostly baseline
  wander (100) and steps (30), and 10 `rhythm_unmeasurable`, a flag that is unreliable.
- SPH: 463 flagged of 25,581 (1.8%). The largest counts are baseline steps (299), baseline wander (205) and
  identity violations (174: 103 in calibration, 71 in test), then 64 records at 32.7 mV or more and 37 amplitude
  outliers.
- SPH stores float16 samples (`docs/clean-ningbo-v1.md`), so the rail definition (16-bit integer limit) has no
  storage meaning there: a value of 32.7 mV or more is a large value, not necessarily a clipped one. The
  identity threshold was set on integer-stored sources. The cause of the SPH identity violations was not
  examined: only counts were produced on the test roles.

## Departures and doubts

- Five flags were revised after their first visual check (`ee4280b`): `reversal_la_ra`, `reversal_neutral`,
  `mains_interference`, `hf_noise` and `rhythm_unmeasurable`. The protocol sets thresholds from the
  distributions and checks them by eye; a revision after the check makes the second check partly a fit to the
  first. The second versions of the three noise and rhythm flags were checked on a new
  draw (seed 20261112, not the protocol's 20261111). The two reversal flags' second versions flag subsets of the
  records already viewed (all 8 and 8 were among the first 15 and 11), so their verdicts were carried over and
  those checks are not independent.
- Viewing stopped at 12 of 24 for `hf_noise` (first version), `reversal_ra_ll` and both `rhythm_unmeasurable`
  versions, once 20 of 24 was out of reach.
- 24 unflagged controls were drawn for every flag, but viewed only for some: 24 for `baseline_wander`, 8 for
  `missing_samples`, `edge_hold`, `adc_rail`, `near_flat_lead`, `identity_violation` and `baseline_step`, none
  for the others.
- Four flags have fewer than 24 records (`edge_hold`, `near_flat_lead` and both reversal flags). All their
  records were drawn and the same share (20 of 24) was applied.
- The measure run's receipt lists `tests/test_mlc03_signal_quality.py` as modified during the run. The test file
  is not measurement code, and the code hashes in the receipt equal the files at `3063fda`. The receipt's table
  path is absolute; the code writes a repository-relative path from `bb09e0b`, and the receipt was not rewritten.
- The final run started on a clean tree at `cb6563e`. Its receipt records `ab274e4`, the commit at the moment it
  was written. The commits in between (`60483e7`, `ab274e4`) changed only `scripts/reports/mlc03_signal_tables.py`,
  which the run does not import. The receipt holds the hashes of the 9 code files the run used.
- Three statements in the `reason` texts of `flag_definitions.csv` are inexact against the quantile tables;
  this report gives the corrected values above:
  - `baseline_wander` says 2.0 mV is above the 99th percentile in CPSC 2018, CPSC-Extra and Georgia. It is above
    only in Georgia (1.78 mV) and at or just below it in CPSC-Extra (2.01) and CPSC 2018 (2.18).
  - `edge_hold` says PTB-XL holds 100 to 130 ms. It holds 90 to 130 ms, with 1st percentile 100.
  - `contact_loss` says no precordial lead is below 0.13 mV at the 0.01st percentile. V1 is at 0.128 mV.
- Two reason texts rest on checks not kept in the outputs. `baseline_step` cites scratch calibration sheets at 1
  to 2, 2 to 4 and above 4 mV; `isolated_spike` says a QRS stays below 1 mV in second difference. The thresholds
  are unchanged. The file was not rewritten, because the flag module belongs to the frozen test-role run.
- No measure or threshold was developed on SPH, whose storage differs (float16). Defects specific to SPH can be
  missed or miscounted, as the protocol's limits anticipate.
- The full-record pass (defects after the first 10 s) ran on train and validation only. Test-role recordings
  were measured in their first 10 s. Test-role measures stay in scratch
  (`/var/tmp/janetrivera/mlc03/signal/measures_test_chunks/`). Only their flags and counts are published.
- Item 3: precordial electrode swaps were not attempted. Item 6: the beat lists come from a multi-lead detector
  and XQRS on II and V5. The resulting flag is unreliable, so rhythm measurability rests on the MLC-02 extractor
  status above.
- The pipeline check used 8 confirmed records per flag (the first 8 in review order), not every flagged record.
- The zero-lag search uses leads I and II only, so a copy with those leads changed is missed. The shifted search
  drops RR keys shared by more than 200 records, so shifted copies of very regular rhythms can be missed.

## Follow-up ideas

- Rhythm measurability: beat lists from more leads, with votes robust to one artefact lead, then a new visual
  check.
- Electrode reversals: a rule for right arm and left leg using the P axis and the precordial leads, and a check
  for precordial swaps from R-wave progression.
- SPH: aggregate checks of why identity residuals exceed 0.05 mV and why values reach 32.7 mV (float16 rounding,
  derived-lead storage or saturation). This needs the user's approval, since SPH holds test roles.
- Long CPSC recordings: per-window flags, if a model reads crops beyond the first 10 s (30 records reach the rail
  only after 10 s; 26 have a constant run of 1 s or more somewhere in the recording).
- The Ningbo amplitude series JS18956 and JS19039 to JS19049, and whether consecutive Ningbo identifiers are the
  same person across roles (as JS19048 and JS19053 appear to be).
- Input pipelines: interpolating or masking missing samples instead of the rail value, and resampling that does
  not ring at the edges, each as a new pipeline version.
- Part C: whether lead-level marking (R4 to R7) is allowed, or which records it should exclude instead.

## Checks

- `tests/test_mlc03_signal_quality.py`: 48 passed (planted defects for every flag, a clean synthetic ECG raising
  no flag, the role guards, synthetic near-duplicate copies, and the pacing exclusion of the spike and noise
  measures).
- Every cached window matched its MLC-01 `sha256_10s` when read (`read_checked` raises otherwise; every run
  completed).
- `ruff check` passes on every new file.

## Files

- `outputs/mlc03_data_quality_20261111/signal/`: `measures_train_validation.parquet`,
  `record_summaries_train_validation.parquet`, `measure_quantiles_train_validation.csv`,
  `full_records_train_validation.parquet`, `flags_train_validation.parquet`,
  `flag_counts_train_validation.csv`, `flag_counts_train_validation_all.csv`, `flag_definitions.csv`,
  `proposed_rule_counts.csv`, `flags_test_roles.parquet`, `flag_counts_test_roles.csv`,
  `near_duplicates_test_roles_pairs.csv`, `near_duplicates_test_roles_counts.csv`, `near_duplicates/`,
  `pipeline_effects/` and the receipts.
- `outputs/mlc03_data_quality_20261111/review/signal/<flag>/`: contact sheets, `sample.csv`, `tally.csv`,
  `tally.json` and `receipt.json` per review (`__v2` for second versions), and `near_duplicates/`.
- Tables of this report: `PYTHONPATH=. .venv/bin/python -m scripts.reports.mlc03_signal_tables --section <name>`.
