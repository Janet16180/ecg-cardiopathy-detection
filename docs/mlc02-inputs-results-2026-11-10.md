# MLC-02 Stage 1 results: inputs (no score)

10 November 2026. Executes Stage 1 of the frozen [MLC-02 protocol](mlc02-condition-models-protocol-2026-11-10.md).
This stage builds a local waveform cache and extracts features. No classifier was fitted and no performance
score was computed on the MLC-01 records. The only fit is the protocol's integrity check: KEY-04's own logistic
regressions refitted on KEY-04's own cached features. Every number below is read from the executed outputs
named in brackets.

The cache and features cover every MLC-01 candidate record: the 100,158 manifest records with no exclusion
reason (readable, twelve named leads, 500 Hz, at least 10 s), whether or not a build includes them. Candidate
keys, roles, paths and first-10-s hashes are the same in every MLC-01 build; the builds differ only in which
duplicate copies they include. A model selects its records with the `included_<build>` columns of `rows.csv`.
The protocol names v4 (`outputs/mlc01_condition_dataset_20261110_v4/`, commit `1a020b6`); its 99,167
included records are all covered.

- Code and tests, committed before any full run: `38ee420` (`ecg_experiment/mlc02_waveforms.py`,
  `mlc02_cache_build.py`, `mlc02_cpc.py`, `mlc02_features.py`, `mlc02_integrity.py`, `mlc02_stage1.py`,
  `scripts/experiments/run_mlc02_inputs.py`, `tests/test_mlc02_inputs.py`). Later commits:
  - `9d2f3e8`: BLAS kernel and saved-model replay in the KEY-04 record.
  - `846875e`: PTB-XL float64 attribution check (`ecg_experiment/mlc02_attribution.py`).
  - `f4b5400`: vendored package versions in the receipt.
  - `aad7c97`: comparison with v4.
  - `514046c`: every candidate record (`ecg_experiment/mlc02_candidates.py`, `tests/test_mlc02_candidates.py`),
    committed before the supplement ran.

  Every run records its commit and whether tracked files were modified (none were).
- Run: `PYTHONPATH=. OPENBLAS_CORETYPE=Haswell .venv/bin/python -m scripts.experiments.run_mlc02_inputs <step>`
  with steps `cache`, `profile`, `extract`, `supplement-cache`, `supplement-extract`, `publish`, `integrity`,
  `attribution` and `key04`. CPU threads were limited to 1 per reader or measurement process and 2 per GPU
  process.

## Waveform cache

`/var/tmp/janetrivera/mlc02/waveforms/` (local NVMe, not copied): float32 `[records, 12, 5000]` files, the first
10 s, leads I, II, III, aVR, aVL, aVF and V1 to V6, at 500 Hz, in mV.

- The first build cached the 99,167 included records of v3 (SHA-256 `17430a1f...`), one file per role and source
  (`<role>__<source>.npy`, 23.80 GB), with `index.csv` and `receipt.json`.
- After the brief was corrected, the other 991 candidates (654 dropped copies and 337 dropped conflicts) were
  added as `<role>__<source>__supplement.npy` files, at positions 99,167 to 100,157, with
  `receipt_supplement.json`.
- `index_all.csv` lists all 100,158 candidates: global position, file, row, manifest `sha256_10s`, the float32
  window hash of the earlier caches (`public_sources.signal_sha256`), missing samples and flat leads.

Candidates per role and source [`features/rows.csv`]:

| role | ptbxl | ningbo | chapman_shaoxing | georgia | cpsc_2018 | cpsc_2018_extra | sph |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 17,418 | 20,941 | 6,147 | 6,171 | 4,113 | 2,067 | 0 |
| validation | 2,183 | 6,979 | 2,050 | 2,059 | 1,373 | 689 | 0 |
| test_same_source | 2,198 | 0 | 0 | 0 | 0 | 0 | 0 |
| test_external | 0 | 0 | 0 | 0 | 0 | 0 | 12,793 |
| sph_calibration | 0 | 0 | 0 | 0 | 0 | 0 | 12,977 |

The v4 included set per role and source is the table of the MLC-01 results report with 81 more CPSC-Extra
records (1,936 train and 653 validation).

- Reading: the MLC-01 reader (`mlc01_dataset.read_signal`, `canonical_order`) with 16 processes. The first
  build took 448.3 s; peak resident memory was 806,508 KiB in the parent and 574,832 KiB in a reader
  [`build.json`]. The supplement took 7.0 s [`receipt_supplement.json`].
- Closed data: every path opened had to be a candidate path of the manifest. Before opening, every Challenge
  path was also checked against the closed (test) groups of the frozen split table. No Challenge test-group
  record or EchoNext record was opened, and no label column was read.
- Hash checks. All 100,158 records reproduce the manifest's `sha256_10s` (int32 microvolts, the MLC-01 rule):
  from the float64 read, after the cast to float32, and recomputed from the written files. The float32 window
  hashes of the two passes were identical [`receipt.json`, `receipt_supplement.json`]. Float32 storage
  therefore does not break hash equality: a value of k/1000 mV rounds back to k microvolts, and SPH's float16
  samples are exact in float32.
- Builds [`features/receipt.json` `coverage`]. v1, v2, v3 and v4 have the same 100,158 candidates in the same
  order, with the same key, source, role and `sha256_10s`. Every build's receipt records a clean tree. Their
  included sets are covered 100%: v1 99,086 of 99,086, and v2, v3 and v4 99,167 of 99,167. v1 lacks 81
  CPSC-Extra records (59 train, 22 validation) that were duplicate conflicts in v1 and became kept copies after
  the label amendments.
- Missing samples: 87 records hold 3,511 samples that WFDB reads as missing. They are 82 Ningbo and 5 Georgia
  records, all included and all train or validation. Each sample was checked to be the stored format-16 value
  -32768 at gain 1000 and baseline 0, the negative amplifier rail (-32.768 mV). They stay NaN in the cache.
- Flat leads: 1,238 candidates have at least one lead that is constant over the first 10 s, 1,236 of them
  included in v4. The included count equals MLC-01's (Ningbo 919 train and 283 validation, Chapman 10 and 3,
  Georgia 6, CPSC 2018 7 and 2, CPSC-Extra 1 and 4, PTB-XL 1).

## Features

`outputs/mlc02_condition_models_20261110/features/` (local copy in `/var/tmp/janetrivera/mlc02/features/final/`).
Every array has one row per candidate, in manifest order, aligned with `rows.csv`. `rows.csv` holds record key,
manifest row, source, role, `sha256_10s`, float32 window hash, missing samples, flat leads, cache file and
row, `included_v1` to `included_v4` and v4's `duplicate_status`.

| file | shape | dtype | input path |
|---|---|---|---|
| `xecg.npy` | 100,158 x 1,024 | float32 | window as float64, FFT to 100 Hz, time-major, vanilla xLSTM backend, microbatches of 16 |
| `jepa.npy` | 100,158 x 768 | float32 | leads I, II, V1-V6 of the float32 window, FFT to 250 Hz, batches of 4, mean of tokens |
| `cpc.npy` | 100,158 x 512 | float32 | four 2.5 s float32 crops (samples 0, 1250, 2500, 3750), `resampy` to 240 Hz, batches of 16 records, mean of tokens then of crops |
| `measurements.npy` | 100,158 x 130 | float64 | `measurement_extractor042.extract_array` on the window as float64 `(5000, 12)`, C order |
| `measurements_status.csv` | 100,158 rows | | record key and the extractor's `status`, `implausible`, `flat_leads` |

Before every extractor, the 3,511 missing samples were set to -32.768 mV (see Departures). The receipt
[`features/receipt.json`] records:

- the SHA-256 of every file, both cache receipts and the hash of `index_all.csv`;
- the SHA-256 of the checkpoints and code each extractor used;
- the git commit, and the run records of the first run and the supplement;
- the torch numerical flags (torch defaults: TF32 off for matmul and on for cuDNN, no deterministic algorithms);
- the environment variables and package versions.

This publication replaced the first one (99,167 v3-included rows, made before the brief was corrected). The
first run's shards are unchanged, so those rows hold the same values.

### Profiles and timing

Each extractor first ran on 504 records (the first 72 of each source) and wrote its forecast for the 99,167
records of the first run [`features/profile/<extractor>/profile.json`]. The full runs started only after the
profile integrity check below.

| extractor | device | profile (504 records) | forecast | first run (99,167) | supplement (991) | peak GPU memory (allocated / reserved) | free before |
|---|---|---:|---:|---:|---:|---:|---:|
| xECG | GPU 0 | 4.9 s | 965 s | 951 s | 9.0 s | 622 / 707 MB | 10.6 GB |
| ECG-JEPA | GPU 1 | 3.3 s | 655 s | 704 s | 8.9 s | 470 / 528 MB | 4.6 GB |
| ECG-CPC | GPU 1, supplement GPU 0 | 9.9 s | 1,954 s | 1,398 s | 19.0 s | 505 / 663 MB | 4.6 GB |
| measurements | CPU, 16 processes (8 for the supplement) | 7.8 ms per record and process | 48 s | 54.0 s | 3.9 s | | |

The ECG-CPC time is dominated by `resampy` (8.2 of 9.9 s in the profile, including its first compilation).

### Missing measurements

Records per extractor status and records with missing values, all candidates [`receipt.json`
`extractors.measurements_missing`; the same per role, and for v4's included set, in
`measurements_missing_included_v4`]:

| source | records | ok | flat lead | no beats | any missing | all 130 missing | missing values |
|---|---:|---:|---:|---:|---:|---:|---:|
| ptbxl | 21,799 | 21,790 | 1 | 8 | 4,022 | 1 | 8,841 |
| ningbo | 27,920 | 26,631 | 1,202 | 87 | 10,246 | 1,202 | 181,674 |
| chapman_shaoxing | 8,197 | 8,184 | 13 | 0 | 2,843 | 13 | 7,356 |
| georgia | 8,230 | 8,222 | 6 | 2 | 1,905 | 6 | 4,768 |
| cpsc_2018 | 5,486 | 5,468 | 10 | 8 | 1,671 | 10 | 5,285 |
| cpsc_2018_extra | 2,756 | 2,748 | 6 | 2 | 563 | 6 | 2,080 |
| sph | 25,770 | 25,748 | 0 | 22 | 3,531 | 0 | 8,878 |

- On v4's included set the totals are 97,803 `ok`, 1,236 flat lead, 128 no beats, 24,409 records with any
  missing value and 217,786 missing values.
- A flat lead stops the extractor before any measurement (`derivacion_plana`, all 130 missing). The 1,238
  such records are exactly the flat-lead candidates of the cache.
- With fewer than two RR intervals (`sin_latidos`, 129 candidates) the spectral and amplitude measurements are
  kept and the beat measurements are missing.
- Among records with status `ok`:
  - `pr_ms` and `p_amp_II` are missing on 23,139 records, `qrs_dur_ms` on 646 and `qt_ms` on 101 (no P onset
    found, or a value outside the extractor's plausibility range).
  - 797 records carry an `implausible` entry.
- Of the 87 records with filled rail samples, 53 end as `sin_latidos`, 33 as `ok` and 1 as `derivacion_plana`.
- No measurement is infinite.

## Integrity

### Earlier feature caches

New features against each cached set on the records it shares with the candidates
[`integrity/integrity.csv`]. For the encoders, the relative difference of a record is its largest absolute
difference divided by its largest absolute cached value; the table gives the largest over records. For the
measurements, it is element by element, relative to nonzero cached values.

| cache | extractor | cache rows | shared | max abs | max rel | records > 1e-4 rel | bit-identical records | same window hash | equal at six digits |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HEADS-01 training | xECG | 28,568 | 28,568 | 3.86e-05 | 7.45e-06 | 0 | 19,664 |  |  |
| HEADS-01 evaluation | xECG | 21,677 | 21,677 | 2.83e-05 | 4.68e-06 | 0 | 17,392 |  |  |
| PR-74 PTB-XL | xECG | 19,229 | 19,229 | 3.86e-05 | 8.59e-06 | 0 | 0 | 19,229 |  |
| PR-74 SPH | xECG | 17,402 | 17,402 | 9.54e-06 | 1.60e-06 | 0 | 17,392 | 17,402 |  |
| PR-74 PTB-XL | measurements | 19,229 | 19,229 | 4.93e-03 | 1.39e-02 | 68 |  | 19,229 | 8,739 |
| PR-74 SPH | measurements | 17,402 | 17,402 | 4.06e-02 | 4.99e-06 | 0 |  | 17,402 | 17,402 |
| SCREEN-01 PTB-XL | xECG | 19,229 | 19,229 | 3.86e-05 | 8.59e-06 | 0 | 0 |  |  |
| SCREEN-01 PTB-XL | ECG-JEPA | 19,229 | 19,229 | 0 | 0 | 0 | 19,229 |  |  |
| SCREEN-01 SPH | xECG | 17,402 | 17,402 | 9.54e-06 | 1.60e-06 | 0 | 17,392 | 17,402 |  |
| SCREEN-01 SPH | ECG-JEPA | 17,402 | 17,402 | 4.77e-07 | 1.56e-07 | 0 | 17,400 | 17,402 |  |
| Avance 4 PTB-XL | xECG | 21,387 | 21,387 | 3.86e-05 | 8.59e-06 | 0 | 0 |  |  |
| Avance 4 PTB-XL | ECG-JEPA | 21,387 | 21,387 | 3.58e-07 | 1.42e-07 | 0 | 21,384 |  |  |
| Avance 4 PTB-XL | ECG-CPC | 21,387 | 21,387 | 0 | 0 | 0 | 21,387 |  |  |

The measurement `max abs` of 4.9e-3 (PTB-XL) and 4.1e-2 (SPH) comes from the cache's six-digit rounding of
large values (for example `qrs_area_V3` of sph:A01896, -42,692.459 stored as -42,692.5).

- Every cached row is now shared. Before the supplement, 29 HEADS-01 training rows (Challenge records that
  MLC-01 removed as duplicates: 21 dropped copies, 8 conflicts) were not; all 29 are bit-identical.
- Where a cache stored the float32 window hash (PR-74 PTB-XL and SPH, SCREEN-01 SPH), every shared record has
  the same window (19,229 of 19,229 and 17,402 of 17,402).
- No encoder difference exceeds 1e-4 relative. ECG-JEPA matches the Avance 4 PTB-XL cache bit for bit on all but
  3 of 21,387 records (those differ by at most 3.6e-7). ECG-CPC matches it bit for bit on all 21,387.

Why the xECG features are not all bit-identical, from a per-source breakdown and the attribution run
[`integrity/attribution.json`]:

- PTB-XL (0 bit-identical, up to 8.6e-6 relative). The earlier caches resampled PTB-XL from WFDB's float64
  samples; MLC-02 resamples from the float32 cache.
  - Recomputing the first 2,048 records from the raw float64 samples, in the same microbatches as the full
    run, gives features bit-identical to PR-74 (1,768 shared records), SCREEN-01 (1,768) and Avance 4 (2,025).
  - On these 2,048 records the float32 storage alone moves the features by at most 1.9e-5 (4.0e-6 relative).
- Challenge (HEADS-01): bit-identical on all Chapman, CPSC 2018, CPSC-Extra and Georgia records, and on all
  but 2 Ningbo records (largest 8.0e-7 relative). The earlier Challenge path already used float32.
- SPH: 17,392 of 17,402 bit-identical (largest 1.6e-6 relative), with identical input windows. The
  remaining 10 differ inside the GPU computation. PR-74 documented the same batch-composition effect: 13
  vectors changed by up to 6.4e-6 when the batch shape changed
  (`outputs/pr74_fresh_baseline_exact_20261028/features/completion.json`). The 2 SPH and 3 PTB-XL ECG-JEPA
  records that differ (at most 4.8e-7) are of the same order.

Measurements:

- SPH: all 17,402 records equal PR-74 after PR-74's six-significant-digit serialization.
- PTB-XL: 8,739 of 19,229 records are equal after six digits. 68 records have an element above 1e-4 relative,
  in 26 columns, mostly `st_*` and `q_depth_*`; the largest relative difference is 1.4e-2. The cause is the
  float32 storage:
  - Recomputed from the raw float64 samples, 19,228 records equal PR-74 after six digits and none exceeds 1e-4
    relative [`attribution.json` `measurements.raw_float64_vs_cache`]. The one exception (ptbxl:15414,
    `q_depth_III` = 0.00095281449999...) lies on a rounding boundary.
  - Between the float32 run and the float64 recomputation, the largest absolute difference over all
    measurements is 7.4e-5.
  - The 70 elements above 1e-4 relative all have cached values of at most 5.9e-4 in absolute value and differ
    by at most 1.5e-7.
- No missing-value pattern differs.

### KEY-04

[`integrity/key04_reproduction_SkylakeX.json`, `key04_reproduction_Haswell.json`]

| check | value |
|---|---|
| KEY-04 result.json, independent logistic, SPH macro AUROC | 0.9608446027429329 |
| recomputed from `predictions.npz` (read-only) | 0.9608446027429329, equal |
| saved `final_models.joblib` applied to KEY-04's cached features | scores identical (max difference 0) |
| refit (`qualify`, `fit_package(..., "independent_logistic", 10.0)`), OpenBLAS default kernel (SkylakeX) | 0.9608446027429329, equal; coefficients and scores identical; iterations 278, 421, 360, 295 as saved |
| same refit with `OPENBLAS_CORETYPE=Haswell` | 0.9608441052951218 (difference 5.0e-7, 6 decimals); coefficients differ by up to 5.8e-4, scores by up to 3.1e-5; iterations 307, 426, 361, 316 |

KEY-04 reproduces exactly, but only with the machine's default OpenBLAS kernel: only then are the saved
iteration counts and coefficients reproduced. Both runs limited threads to 2, as KEY-04 did.

## Departures and doubts

All dated 10 November 2026. None changes a decision of the protocol.

1. Record set. The brief first named v1's included set and said that only label columns would differ in the
   rebuild. The included sets do differ, by 81 CPSC-Extra records. During the stage the protocol was pointed at
   v3 (`fddeb71`) and then v4 (`1a020b6`). Following the corrected brief, the outputs cover every candidate;
   every build is compared row by row and its included set is fully covered.
2. Missing samples. The protocol does not say how a model sees a missing sample. The earlier Challenge path
   rejected nonfinite windows, and none of the 87 records is in the HEADS-01 caches. Each of the 3,511 samples
   is the stored rail value, so every extractor received -32.768 mV there and the cache keeps NaN. 53 of the 87
   records then have no measured beats.
3. Float32 cache. The protocol specifies float32 storage. For PTB-XL, xECG and the measurements therefore no
   longer equal the earlier float64-based caches bit for bit (differences above). ECG-JEPA and ECG-CPC always
   used float32 and are unaffected.
4. BLAS kernel. The brief asked for `OPENBLAS_CORETYPE=Haswell`, and the feature runs used it. The GPU encoders
   do not use OpenBLAS, and the measurements equal PR-74 as shown. The KEY-04 refit reproduces only without it;
   its macro AUROC moves by 5.0e-7 between the two kernels. Stage 2's logistic fits (P0) need the kernel fixed
   and recorded.
5. Measurement precision. The measurements are stored at full float64 precision; PR-74 stored six significant
   digits.
6. ECG-CPC code. The pipeline of `scripts/features/extract_ecg_cpc.py` is repeated in
   `ecg_experiment/mlc02_cpc.py`, because scripts cannot be imported. A unit test checks that its input equals
   the script's `preprocess` exactly, and the features equal the Avance 4 cache.
7. Batches and scheduling.
   - The 991 supplement records were extracted in their own shards, so their batch neighbours differ from a
     single run's. The 29 of them that HEADS-01 holds are bit-identical to it.
   - ECG-CPC's first run used GPU 1, after ECG-JEPA.
   - Profile records were extracted again in the full runs rather than reused.

## Checks

`ruff check ecg_experiment scripts tests conftest.py` passed. `pytest tests/test_mlc02_inputs.py
tests/test_mlc02_candidates.py`: 19 passed. The full suite in the worktree: 1,636 passed, 2 skipped (commit `514046c`), with
`tests/test_xecg_encoder_motion_v11.py::test_graph_rejects_missing_and_conflicting_leaves` deselected because
it rejects the worktree's symlinked `outputs` (MLC-01 report).

## Files

- Cache (local only): `/var/tmp/janetrivera/mlc02/waveforms/`, holding:
  - `index.csv` (first build) and `index_all.csv` (all candidates);
  - `plan.json`, `plan_supplement.json`, `build.json`, `receipt.json`, `receipt_supplement.json`, `build.log`;
  - `chunks/` and `chunks_supplement/`.
- Feature shards, profiles and run records (local): `/var/tmp/janetrivera/mlc02/features/<extractor>/`,
  `<extractor>__supplement/` and `profile/<extractor>/`; logs in `/var/tmp/janetrivera/mlc02/logs/`.
- Outputs:
  - `outputs/mlc02_condition_models_20261110/features/`: arrays, `rows.csv`, `measurements_status.csv`,
    `receipt.json`.
  - `outputs/mlc02_condition_models_20261110/integrity/`: `integrity.csv`, `integrity.json`,
    `integrity_worst.csv`, `attribution.json` and both KEY-04 records.
