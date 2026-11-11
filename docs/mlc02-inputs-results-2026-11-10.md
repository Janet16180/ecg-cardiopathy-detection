# MLC-02 Stage 1 results: inputs (no score)

10 November 2026. Executes Stage 1 of the frozen [MLC-02 protocol](mlc02-condition-models-protocol-2026-11-10.md).
The cache was built from the v3 MLC-01 manifest (`outputs/mlc01_condition_dataset_20261110_v3/`). The protocol
now names v4 (commit `1a020b6`), which has the same 99,167 included records and the same identity columns, so
the cache and features apply to v4 unchanged (see Other builds). This stage builds a local waveform
cache and extracts features. No classifier was fitted and no performance score was computed on the MLC-01
records. The only fit is the protocol's integrity check: KEY-04's own logistic regressions refitted on
KEY-04's own cached features. Every number below is read from the executed outputs named in brackets.

- Code and tests, committed before any full run: `38ee420` (`ecg_experiment/mlc02_waveforms.py`,
  `mlc02_cache_build.py`, `mlc02_cpc.py`, `mlc02_features.py`, `mlc02_integrity.py`, `mlc02_stage1.py`,
  `scripts/experiments/run_mlc02_inputs.py`, `tests/test_mlc02_inputs.py`). Later commits added the BLAS
  kernel to the KEY-04 record (`9d2f3e8`) and the PTB-XL float64 attribution check (`846875e`,
  `ecg_experiment/mlc02_attribution.py`), vendored package versions to the receipt (`f4b5400`) and the v4
  comparison (`aad7c97`, after which the receipt was rewritten with identical file hashes). Every run records
  its commit and whether tracked files were modified (none were).
- Run: `PYTHONPATH=. OPENBLAS_CORETYPE=Haswell .venv/bin/python -m scripts.experiments.run_mlc02_inputs <step>`
  with steps `cache`, `profile`, `extract`, `publish`, `integrity`, `attribution` and `key04`. CPU threads were
  limited to 1 per reader or measurement process and 2 per GPU process.

## Waveform cache

`/var/tmp/janetrivera/mlc02/waveforms/` (local NVMe, not copied): one float32 `[records, 12, 5000]` file per
role and source, `<role>__<source>.npy`, 23.80 GB in total. Each file holds the first 10 s, leads I, II, III,
aVR, aVL, aVF and V1 to V6, at 500 Hz, in mV. `index.csv` lists every record with its global position (the
manifest order of included records), file, row, manifest `sha256_10s`, the float32 window hash of the earlier
caches (`public_sources.signal_sha256`), missing samples and flat leads [`receipt.json`].

| role | ptbxl | ningbo | chapman_shaoxing | georgia | cpsc_2018 | cpsc_2018_extra | sph |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 17,418 | 20,919 | 6,097 | 6,117 | 3,763 | 1,936 | 0 |
| validation | 2,183 | 6,972 | 2,036 | 2,043 | 1,251 | 653 | 0 |
| test_same_source | 2,198 | 0 | 0 | 0 | 0 | 0 | 0 |
| test_external | 0 | 0 | 0 | 0 | 0 | 0 | 12,698 |
| sph_calibration | 0 | 0 | 0 | 0 | 0 | 0 | 12,883 |

99,167 records in total, the included records of the v3 manifest (SHA-256 `17430a1f...`).

- Reading: the MLC-01 reader (`mlc01_dataset.read_signal`, `canonical_order`) with 16 processes. The build
  took 448.3 s; peak resident memory was 806,508 KiB in the parent and 574,832 KiB in a reader
  [`build.json`].
- Closed data: every path opened had to be an included path of the manifest, and every Challenge path was
  checked against the closed (test) groups of the frozen split table before opening. No Challenge test-group
  record or EchoNext record was opened.
- Hash checks. All 99,167 records reproduce the manifest's `sha256_10s` (int32 microvolts, the MLC-01 rule)
  from the float64 read and again after the cast to float32 [`build.json`]. A separate pass recomputed the hash
  of every record from the written files; all 99,167 matched again, and the float32 window hashes were
  identical [`receipt.json` `sha256_cache_ok`]. Float32 storage therefore does not break hash equality: a
  value of k/1000 mV rounds back to k microvolts, and SPH's float16 samples are exact in float32.
- Other builds. On all 100,221 manifest rows, v1 (`outputs/mlc01_condition_dataset_20261110/`), v2 and v4
  (SHA-256 `25d75f8f...`) agree with v3 on record key, source, role, raw path and `sha256_10s`. v2 and v4
  include the same 99,167 records as v3 [`features/receipt.json` `other_manifests`]. v1 includes 99,086: the 81 CPSC-Extra records it lacks (59 train, 22 validation) were duplicate conflicts
  in v1 and became kept copies after the label amendments [`receipt.json` `plan.other_manifests`].
- Missing samples: 87 records (82 Ningbo, 5 Georgia; all train or validation) hold 3,511 samples that WFDB
  reads as missing. Each one was checked to be the stored format-16 value -32768 at gain 1000 and baseline 0,
  which is the negative amplifier rail (-32.768 mV). They stay NaN in the cache.
- Flat leads: 1,236 records have at least one lead that is constant over the first 10 s (Ningbo 919 train
  and 283 validation, Chapman 10 and 3, Georgia 6, CPSC 2018 7 and 2, CPSC-Extra 1 and 4, PTB-XL 1). This
  equals the MLC-01 count.

## Features

`outputs/mlc02_condition_models_20261110/features/` (local copy in `/var/tmp/janetrivera/mlc02/features/final/`),
rows in the order of `rows.csv` (the manifest order of included records):

| file | shape | dtype | input path |
|---|---|---|---|
| `xecg.npy` | 99,167 x 1,024 | float32 | window as float64, FFT to 100 Hz, time-major, vanilla xLSTM backend, microbatches of 16 |
| `jepa.npy` | 99,167 x 768 | float32 | leads I, II, V1-V6 of the float32 window, FFT to 250 Hz, batches of 4, mean of tokens |
| `cpc.npy` | 99,167 x 512 | float32 | four 2.5 s float32 crops (samples 0, 1250, 2500, 3750), `resampy` to 240 Hz, batches of 16 records, mean of tokens then of crops |
| `measurements.npy` | 99,167 x 130 | float64 | `measurement_extractor042.extract_array` on the window as float64 `(5000, 12)`, C order |
| `measurements_status.csv` | 99,167 rows | | extractor `status`, `implausible`, `flat_leads` |

Before every extractor, the 3,511 missing samples were set to -32.768 mV (see Departures). The receipt
[`features/receipt.json`] holds the SHA-256 of every file, the cache receipt and index hashes, the SHA-256 of
the checkpoints and code each extractor used, the git commit, the run records, the torch numerical flags
(torch defaults: TF32 off for matmul and on for cuDNN, no deterministic algorithms) and the environment
variables.

### Profiles and timing

Each extractor first ran on 504 records (the first 72 of each source) and wrote its forecast for all
records [`features/profile/<extractor>/profile.json`]; the full runs started only after the profile
integrity check below.

| extractor | device | profile (504 records) | forecast | full run | peak GPU memory (allocated / reserved) | free before |
|---|---|---:|---:|---:|---:|---:|
| xECG | GPU 0 | 4.9 s | 965 s | 951 s | 622 / 707 MB | 10.6 GB |
| ECG-JEPA | GPU 1 | 3.3 s | 655 s | 704 s | 470 / 528 MB | 4.6 GB |
| ECG-CPC | GPU 1 | 9.9 s | 1,954 s | 1,398 s | 505 / 663 MB | 4.6 GB |
| measurements | CPU, 16 processes | 7.8 ms per record and process | 48 s | 54.0 s | | |

The ECG-CPC time is dominated by `resampy` (8.2 of 9.9 s in the profile, including its first compilation).

### Missing measurements

Records per extractor status and records with missing values [`receipt.json` `extractors.measurements_missing`]:

| source | records | ok | flat lead | no beats | any missing | all 130 missing | missing values |
|---|---:|---:|---:|---:|---:|---:|---:|
| ptbxl | 21,799 | 21,790 | 1 | 8 | 4,022 | 1 | 8,841 |
| ningbo | 27,891 | 26,602 | 1,202 | 87 | 10,223 | 1,202 | 181,625 |
| chapman_shaoxing | 8,133 | 8,120 | 13 | 0 | 2,807 | 13 | 7,281 |
| georgia | 8,160 | 8,152 | 6 | 2 | 1,898 | 6 | 4,754 |
| cpsc_2018 | 5,014 | 4,998 | 9 | 7 | 1,456 | 9 | 4,639 |
| cpsc_2018_extra | 2,589 | 2,582 | 5 | 2 | 504 | 5 | 1,832 |
| sph | 25,581 | 25,559 | 0 | 22 | 3,499 | 0 | 8,814 |

- A flat lead stops the extractor before any measurement (`derivacion_plana`, all 130 missing); the 1,236
  such records are exactly the flat-lead records of the cache.
- With fewer than two RR intervals (`sin_latidos`, 128 records) the spectral and amplitude measurements are
  kept and the beat measurements are missing.
- Among records with status `ok`, `pr_ms` and `p_amp_II` are missing on 22,784 records, `qrs_dur_ms` on 613
  and `qt_ms` on 93 (no P onset found, or a value outside the extractor's plausibility range; 764 records carry
  an `implausible` entry).
- Of the 87 records with filled rail samples, 53 end as `sin_latidos`, 33 as `ok` and 1 as `derivacion_plana`.

Per role, the same table is in the receipt. No measurement is infinite.

## Integrity

### Earlier feature caches

New features against each cached set on the records it shares with the v3 manifest
[`integrity/integrity.csv`]. For the encoders, the relative difference of a record is its largest absolute
difference divided by its largest absolute cached value; the table gives the largest over records. For the
measurements, it is element by element, relative to nonzero cached values.

| cache | extractor | cache rows | shared | max abs | max rel | records > 1e-4 rel | bit-identical records | same window hash | equal at six digits |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HEADS-01 training | xECG | 28,568 | 28,539 | 3.86e-05 | 7.45e-06 | 0 | 19,635 |  |  |
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

The measurement `max abs` of 4.9e-3 (PTB-XL) and 4.1e-2 (SPH) is the cache's six-digit rounding of large
values (for example `qrs_area_V3` of sph:A01896, -42,692.459 stored as -42,692.5).

- The 29 HEADS-01 training rows not shared are records MLC-01 removed as duplicates (21 dropped copies, 8
  conflicts); MLC-02 did not open them.
- Where a cache stored the float32 window hash (PR-74 PTB-XL and SPH, SCREEN-01 SPH), every shared record has
  the same window (19,229 and 17,402 of 17,402).
- No difference exceeds 1e-4 relative for any encoder. ECG-JEPA and ECG-CPC equal the Avance 4 PTB-XL caches
  bit for bit on almost every record (ECG-JEPA: all but 3 of 21,387, by at most 3.6e-7; ECG-CPC: all 21,387).

Why the xECG features are not all bit-identical, from the per-source breakdown and the attribution run
[`integrity/attribution.json`]:

- PTB-XL (0 of 19,229 to 21,387 bit-identical, up to 8.6e-6 relative): the earlier caches resampled PTB-XL
  from WFDB's float64 samples, MLC-02 from the float32 cache. Recomputing the first 2,048 records from the raw
  float64 samples, in the same microbatches as the full run, gives features bit-identical to PR-74 (1,768
  shared records), SCREEN-01 (1,768) and Avance 4 (2,025). On these 2,048 records the float32 storage alone
  moves the features by at most 1.9e-5 (4.0e-6 relative).
- Challenge (HEADS-01): bit-identical on all Chapman, CPSC 2018, CPSC-Extra and Georgia records and on 9,950
  of 9,952 Ningbo records (largest 7.96e-7 relative). The earlier Challenge path already used float32.
- SPH: 17,392 of 17,402 bit-identical (largest 1.6e-6 relative), with identical input windows. The
  remaining 10 differ inside the GPU computation. PR-74 documented the same effect of batch composition (13
  vectors changed by up to 6.4e-6 when the batch shape changed,
  `outputs/pr74_fresh_baseline_exact_20261028/features/completion.json`). The 2 SPH and 3 PTB-XL ECG-JEPA
  records that differ (at most 4.8e-7) are of the same order.

Measurements:

- SPH: all 17,402 records equal PR-74 after PR-74's six-significant-digit serialization.
- PTB-XL: 8,739 of 19,229 records are equal after six digits. 68 records have an element above 1e-4
  relative, in 26 columns, mostly `st_*` and `q_depth_*`; the largest relative difference is 1.4e-2. The cause is
  the float32 storage. Recomputed from the raw float64 samples, 19,228 records equal PR-74 after six digits
  and none exceeds 1e-4 relative [`attribution.json` `measurements.raw_float64_vs_cache`]. The one exception
  (ptbxl:15414, `q_depth_III` = 0.00095281449999...) lies on a rounding boundary. Between the float32 run and
  the float64 recomputation the largest absolute difference is 7.4e-5. The 70 elements above 1e-4 relative
  all have cached values of at most 5.9e-4 in absolute value and differ by at most 1.5e-7. No
  missing-value pattern differs.

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

1. Manifest. The brief named the first build and said that only label columns would differ in the rebuild.
   During the stage the protocol was pointed first at v3 (`fddeb71`), then at v4 (`1a020b6`). v3 and v4
   include 81 more records than v1 (above). The cache was built from v3; v1, v2 and v4 were compared row by
   row, agree on every identity column, and v4's included set equals v3's.
2. Missing samples. The protocol does not say how a model sees a missing sample. The earlier Challenge path
   rejected nonfinite windows, and none of the 87 records is in the HEADS-01 caches. Each of the 3,511 is the
   stored rail value, so every extractor received -32.768 mV there and the cache keeps NaN. 53 of the 87
   records then have no measured beats.
3. Float32 cache. The protocol specifies float32 storage. For PTB-XL, xECG and the measurements therefore no
   longer equal the earlier float64-based caches bit for bit (differences above). ECG-JEPA and ECG-CPC always used
   float32 and are unaffected.
4. BLAS kernel. The brief asked for `OPENBLAS_CORETYPE=Haswell`, and the feature runs used it (the GPU
   encoders do not use OpenBLAS, and the measurements equal PR-74 as shown). The KEY-04 refit reproduces only
   without it; its macro AUROC moves by 5.0e-7 between the two kernels. Stage 2's logistic fits (P0) need the
   kernel fixed and recorded.
5. Measurement precision. The measurements are stored at full float64 precision; PR-74 stored six
   significant digits.
6. ECG-CPC code. The pipeline of `scripts/features/extract_ecg_cpc.py` is repeated in
   `ecg_experiment/mlc02_cpc.py`, because scripts cannot be imported. A unit test checks that its input equals
   the script's `preprocess` exactly, and the features equal the Avance 4 cache.
7. Scheduling. ECG-CPC ran on GPU 1 after ECG-JEPA, not on GPU 0 after xECG; the results are unaffected. Profile
   records were extracted again in the full runs rather than reused.

## Checks

`ruff check ecg_experiment scripts tests conftest.py` passed. `pytest tests/test_mlc02_inputs.py`: 16 passed.
The full suite in the worktree: 1,633 passed, 2 skipped (on commit `f4b5400`; `aad7c97` only adds a path), with
`tests/test_xecg_encoder_motion_v11.py::test_graph_rejects_missing_and_conflicting_leaves` deselected because
it rejects the worktree's symlinked `outputs` (MLC-01 report).

## Files

- Cache (local only): `/var/tmp/janetrivera/mlc02/waveforms/` with `index.csv`, `plan.json`, `build.json`,
  `receipt.json`, `build.log` and `chunks/`.
- Feature shards, profiles and run records (local): `/var/tmp/janetrivera/mlc02/features/<extractor>/`,
  `profile/<extractor>/`; logs in `/var/tmp/janetrivera/mlc02/logs/`.
- Outputs: `outputs/mlc02_condition_models_20261110/features/` (arrays, `rows.csv`,
  `measurements_status.csv`, `receipt.json`) and `outputs/mlc02_condition_models_20261110/integrity/`
  (`integrity.csv`, `integrity.json`, `integrity_worst.csv`, `attribution.json`, both KEY-04 records).
