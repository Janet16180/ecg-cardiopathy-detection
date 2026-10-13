# ECGAD localization benchmark, version 1

12 October 2026. This document records where the benchmark comes from, how its records were identified, and
the rules for using it. The rules were agreed with the user before any localization score was computed on it.

## What it is

Jiang et al., "Multi-scale Cross-restoration Framework for Electrocardiogram Anomaly Detection", MICCAI 2023
([arXiv 2308.01639](https://arxiv.org/abs/2308.01639), [code](https://github.com/MediaBrain-SJTU/ECGAD), MIT
licence for the code). Two cardiologists marked the abnormal signal points of 400 PTB-XL ECGs, separately for
each lead. The files `benchmark_data.npy` and `benchmark_label.npy` were downloaded from the authors' Google
Drive folder on 12 October 2026. Both are float64 arrays of shape (400, 5000, 12). The data array is the PTB-XL
500 Hz record after the authors' 1 Hz high-pass, 35 Hz notch and 25 Hz low-pass filters; the labels are 0/1 per
sample and lead. The annotation files carry no licence statement, so they stay on this machine and are not
redistributed.

Local copy: `data/raw/ecgad/` (ignored by Git), with `SHA256SUMS.txt`:

| File | SHA-256 |
| --- | --- |
| `benchmark_data.npy` | `3d2a9c8cea7ab84121f047dedb5da08a0472bf03670fbc06e17b2a28f41e3060` |
| `benchmark_label.npy` | `8b73c64bacf0bbe812b7f92ce3338886c74914fc58d3d4f9297fcd42c8e23c9b` |
| `ecgad_ptbxl_match.csv` | `fa3e09c99d6101cebba785d9a5bc980583b0fca726973e23aae7fcf67b678a80` |

## How the records were identified

The benchmark ships no record IDs. Each benchmark ECG and each PTB-XL record was band-passed (1-25 Hz,
second-order Butterworth, zero phase), cut to samples 100-4900, reduced to leads II and V2 at 100 Hz,
z-scored per lead and compared by cosine similarity. The authors' code builds its test data from fold 10, so
fold 10 was searched first. Every benchmark ECG matched a fold-10 record with similarity at least 0.983 (median
1.000); the search stopped there. Benchmark indices 161 and 162 are the same record (ecg_id 2507), with
identical signals and labels, so the benchmark holds 399 distinct records from 380 patients. No benchmark
patient has a record in folds 1-9.

The marks agree with the PTB-XL statements: 208 NORM records carry no mark, 187 abnormal records carry marks,
3 NORM records carry marks and 2 abnormal records carry none. Most frequent statements among marked records:
ASMI 60, LVH 47, IMI 36, 1AVB 30, LAFB 25, CRBBB 23, IRBBB 21, AFIB 20, PVC 19. Marked runs have a median length
of 182 ms (10th-90th percentile 84-424 ms). In 47 marked records all 12 leads are marked; 143 have marks on
only some leads.

## Closed-data disclosure

Fold 10 is the PTB-XL test fold reserved for `final_frozen_test`. To identify the records, lead II and V2 of
all 2,198 fold-10 records were read as similarity fingerprints. No model was applied and no score computed.
The statements of the 399 matched records were printed for the agreement table above. No statement of any
other fold-10 record was read.

## Decision (user, 12 October 2026)

1. The 399 ECGAD records leave the PTB-XL test fold. `final_frozen_test` uses the remaining 1,799 fold-10
   records; they stay closed.
2. The 399 records are split by patient into a development half and a holdout half
   (`docs/ecgad-split-v1.json`, built by `scripts/data/build_ecgad_split.py`, seed 61061). Patients are
   stratified by whether any of their records is marked, shuffled and alternately assigned. The duplicate
   index 162 is marked `duplicate` and never scored.

| Partition | Records | Marked | Unmarked |
| --- | ---: | ---: | ---: |
| development | 198 | 94 | 104 |
| holdout | 201 | 96 | 105 |

3. Localization methods are iterated on the development half only. The holdout half is scored once, at the end
   of the localization programme, for the methods frozen by then, and the scores are reported whatever they
   are.
4. No ECGAD record may enter a normal reference, a training set or a pretraining cohort. Fold 10 is already
   outside cohorts v3 and v4.
