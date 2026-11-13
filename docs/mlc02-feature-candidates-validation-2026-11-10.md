# MLC-02 Stage 2: feature-based candidates on validation (draft)

10 November 2026. No test-set quantity was computed. No prediction and no metric exists for PTB-XL fold 10,
the SPH evaluation half or the SPH calibration half. Every number below comes from role `train` (fitting) or
role `validation` (scoring), or, for leave-one-family-out, from train and validation records together. The
test predictions are a later, separate step, after the selection record is committed.

This executes the feature-based rows of Stage 2 of the frozen
[MLC-02 protocol](mlc02-condition-models-protocol-2026-11-10.md): P0, L-J, L-C, G-M, H-X, H-J, H-C and H-All.
It also runs leave-one-family-out (Stage 3 item 5) for P0, L-C and H-All, and the source-identification
audit (Stage 3 item 6). S-R, F and E are run elsewhere. Every metric, the selection metric and the
prediction-file convention come from the tested scoring library (`ecg_experiment/mlc02_data.py`,
`mlc02_evaluation.py`; [evaluation code](mlc02-evaluation-code-2026-11-10.md)). Numbers are read from the
files named in brackets, under `outputs/mlc02_condition_models_20261110/`.

## Validation results

Within-source macro AUROC per validation source, and the selection metric (their mean)
[`candidates/<id>/candidate.json`, `validation`]:

| candidate | PTB-XL | Ningbo | Chapman | Georgia | CPSC 2018 | CPSC-Extra | selection metric |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 | 0.9654 | 0.9850 | 0.9834 | 0.9538 | 0.9548 | 0.9433 | 0.9643 |
| L-J | 0.9639 | 0.9812 | 0.9837 | 0.9475 | 0.9543 | 0.9378 | 0.9614 |
| L-C | 0.9391 | 0.9626 | 0.9705 | 0.9166 | 0.9407 | 0.9114 | 0.9401 |
| G-M | 0.9676 | 0.9781 | 0.9824 | 0.9587 | 0.9477 | 0.9444 | 0.9632 |
| H-X | 0.9682 | 0.9865 | 0.9864 | 0.9552 | 0.9613 | 0.9397 | 0.9662 |
| H-J | 0.9680 | 0.9848 | 0.9855 | 0.9525 | 0.9611 | 0.9360 | 0.9646 |
| H-C | 0.9547 | 0.9733 | 0.9773 | 0.9312 | 0.9560 | 0.9229 | 0.9526 |
| H-All | 0.9732 | 0.9881 | 0.9873 | 0.9611 | 0.9668 | 0.9469 | 0.9706 |

- The H rows are the mean probability of three seeds, as the protocol defines the candidate.
- 14, 14, 9, 12, 5 and 7 conditions enter the macro in the six sources, the same for every candidate. No source
  is excluded.
- Among H-X, H-J and H-C, H-X has the highest selection metric: 0.9662, against 0.9646 for H-J and 0.9526 for
  H-C. The difference between H-X and H-J, 0.0016, exceeds the protocol's 0.001 tie margin. By the protocol,
  F therefore fine-tunes the xECG encoder.
- Among these eight candidates, H-All has the highest selection metric (0.9706) and L-C the lowest (0.9401).
  This is not the selection record: S-R, F and E are still missing.
- Pretraining overlap: xECG and ECG-JEPA were pretrained on Chapman and Ningbo waveforms. Their Ningbo and
  Chapman validation numbers (P0, L-J, H-X, H-J, H-All) come from waveforms their encoder has seen, without
  labels. ECG-CPC (L-C, H-C) and the measurements (G-M) have no such overlap.
- Ningbo, Georgia and Chapman-Ningbo validation records share some patients with training records; the MLC-01
  audit estimates 6-13% of them. This holds for every candidate.

Records [executed `load_split(("train", "validation"), ())` crosstab]:

| role | PTB-XL | Ningbo | Chapman | Georgia | CPSC 2018 | CPSC-Extra | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 17,418 | 20,919 | 6,097 | 6,117 | 3,763 | 1,936 | 56,250 |
| validation | 2,183 | 6,972 | 2,036 | 2,043 | 1,251 | 653 | 15,138 |

### Per condition

Each condition's within-source validation AUROC, averaged over the validation sources where it enters (the
library's per-condition variant) [`candidate.json`, `validation.per_condition`]:

| condition | P0 | L-J | L-C | G-M | H-X | H-J | H-C | H-All |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sinus tachycardia | 0.992 | 0.992 | 0.989 | 0.992 | 0.993 | 0.992 | 0.991 | 0.993 |
| sinus bradycardia | 0.990 | 0.988 | 0.986 | 0.989 | 0.990 | 0.991 | 0.991 | 0.993 |
| sinus arrhythmia | 0.962 | 0.973 | 0.815 | 0.977 | 0.966 | 0.977 | 0.871 | 0.979 |
| AF or flutter | 0.984 | 0.983 | 0.979 | 0.980 | 0.983 | 0.983 | 0.978 | 0.985 |
| PVC | 0.935 | 0.940 | 0.881 | 0.935 | 0.941 | 0.945 | 0.915 | 0.948 |
| first-degree AV block | 0.976 | 0.975 | 0.970 | 0.965 | 0.976 | 0.975 | 0.970 | 0.979 |
| complete RBBB | 0.990 | 0.987 | 0.987 | 0.988 | 0.990 | 0.989 | 0.990 | 0.990 |
| incomplete RBBB | 0.958 | 0.946 | 0.921 | 0.949 | 0.961 | 0.958 | 0.943 | 0.968 |
| LAFB | 0.978 | 0.971 | 0.949 | 0.976 | 0.978 | 0.972 | 0.956 | 0.981 |
| left axis deviation | 0.986 | 0.978 | 0.949 | 0.988 | 0.987 | 0.978 | 0.960 | 0.987 |
| LVH | 0.952 | 0.941 | 0.933 | 0.960 | 0.954 | 0.945 | 0.936 | 0.959 |
| ST depression | 0.949 | 0.948 | 0.944 | 0.945 | 0.954 | 0.955 | 0.957 | 0.963 |
| T-wave abnormality | 0.932 | 0.931 | 0.913 | 0.934 | 0.936 | 0.938 | 0.931 | 0.943 |
| any MI | 0.923 | 0.902 | 0.881 | 0.933 | 0.929 | 0.911 | 0.908 | 0.934 |

ECG-CPC features are weakest for sinus arrhythmia (0.815 for L-C, 0.871 for H-C, against 0.962-0.979
elsewhere) and premature ventricular complexes (0.881 and 0.915). The per-source tables are in the appendix.

### Seed spread of the networks

Each seed's best epoch and validation selection metric, and the candidate (mean of the three seeds)
[`candidate.json`, `seed_runs` and `by_seed`]:

| candidate | seed 20261110 | seed 20261111 | seed 20261112 | range | mean of three seeds |
|---|---|---|---|---:|---:|
| H-X | 0.9657 (epoch 13) | 0.9654 (10) | 0.9652 (8) | 0.0005 | 0.9662 |
| H-J | 0.9633 (37) | 0.9633 (35) | 0.9635 (16) | 0.0002 | 0.9646 |
| H-C | 0.9508 (55) | 0.9512 (59) | 0.9508 (50) | 0.0005 | 0.9526 |
| H-All | 0.9696 (11) | 0.9693 (10) | 0.9691 (14) | 0.0005 | 0.9706 |

- In every network, the mean of the three seeds scores above its best single seed, by 0.0005 (H-X) to 0.0013
  (H-C).
- The seed spread (0.0002-0.0005) is smaller than every gap between the four network candidates.
- The first seed's run at the chosen setting is bit-identical to its grid run, for all four networks
  (`first_seed_rerun_identical`).

## Chosen settings

### P0, L-J, L-C: C per condition

Chosen C, from {0.01, 0.1, 1, 10, 100}, by each condition's per-condition validation metric
[`candidate.json`, `chosen_c` and `grid`]:

| condition | P0 | L-J | L-C |
|---|---:|---:|---:|
| sinus tachycardia | 10 | 10 | 100 |
| sinus bradycardia | 100 | 100 | 10 |
| sinus arrhythmia | 100 | 100 | 10 |
| AF or flutter | 10 | 10 | 1 |
| PVC | 0.1 | 1 | 10 |
| first-degree AV block | 10 | 100 | 1 |
| complete RBBB | 10 | 10 | 10 |
| incomplete RBBB | 100 | 100 | 100 |
| LAFB | 10 | 10 | 100 |
| left axis deviation | 10 | 100 | 100 |
| LVH | 10 | 10 | 100 |
| ST depression | 10 | 100 | 10 |
| T-wave abnormality | 10 | 100 | 10 |
| any MI | 10 | 100 | 100 |

- Grid edge: no condition chose C = 0.01. C = 100, the upper edge, was chosen for 3 conditions in P0, 8 in L-J
  and 6 in L-C [`at_grid_edge`].
- Where C = 100 was chosen, its gain over C = 10 is 0.0002-0.0016 in P0, 0.0005-0.0039 in L-J and 0.0003-0.0034
  in L-C (from the `grid` tables). No grid value tied.
- lbfgs converged in every fit. The most iterations were 1,363 (L-J, T-wave abnormality, C = 100), against the
  recipe's cap of 5,000 [`training_log.csv`].

### G-M: learning rate, leaves and iterations per condition

[`candidates/G-M/candidate.json`, `chosen` and `grid`]

| condition | learning rate | leaves | iterations | validation metric |
|---|---:|---:|---:|---:|
| sinus tachycardia | 0.05 | 31 | 127 | 0.9916 |
| sinus bradycardia | 0.05 | 31 | 836 | 0.9894 |
| sinus arrhythmia | 0.1 | 31 | 358 | 0.9765 |
| AF or flutter | 0.1 | 31 | 958 | 0.9796 |
| PVC | 0.05 | 31 | 477 | 0.9352 |
| first-degree AV block | 0.1 | 15 | 95 | 0.9648 |
| complete RBBB | 0.05 | 31 | 735 | 0.9882 |
| incomplete RBBB | 0.05 | 31 | 253 | 0.9493 |
| LAFB | 0.05 | 31 | 151 | 0.9759 |
| left axis deviation | 0.05 | 31 | 555 | 0.9877 |
| LVH | 0.05 | 15 | 798 | 0.9604 |
| ST depression | 0.05 | 31 | 664 | 0.9446 |
| T-wave abnormality | 0.05 | 31 | 630 | 0.9345 |
| any MI | 0.1 | 31 | 199 | 0.9333 |

- Twelve of the 14 conditions chose 31 leaves, and 10 chose the rate 0.05.
- No grid point reached the 1,000-iteration cap. The largest chosen count is 958 (AF or flutter), and the
  largest best count in any grid point is 993 [`at_iteration_cap` is empty].
- The refitted models reproduce the staged validation probabilities exactly (largest difference 0)
  [`refit_max_abs_difference`].

### Networks: learning rate, weight decay and epoch

Grid runs with seed 20261110, best epoch out of 100, and its validation selection metric
[`candidate.json`, `grid`]:

| candidate | inputs | lr 1e-3, wd 1e-4 | lr 1e-3, wd 1e-2 | lr 3e-4, wd 1e-4 | lr 3e-4, wd 1e-2 | chosen |
|---|---:|---|---|---|---|---|
| H-X | 1,024 | 0.9645 (13) | 0.9641 (4) | 0.9657 (13) | 0.9656 (13) | 3e-4, 1e-4 |
| H-J | 768 | 0.9622 (21) | 0.9620 (18) | 0.9632 (22) | 0.9633 (37) | 3e-4, 1e-2 |
| H-C | 512 | 0.9497 (17) | 0.9494 (17) | 0.9508 (55) | 0.9506 (38) | 3e-4, 1e-4 |
| H-All | 2,564 | 0.9676 (5) | 0.9672 (2) | 0.9696 (11) | 0.9695 (11) | 3e-4, 1e-4 |

- Every network chose the lower learning rate, 3e-4.
- At a fixed learning rate, weight decay changes the best-epoch metric by at most 0.00045.
- No run's best epoch is 1 or 100 [`at_epoch_edge` is empty]; the seed runs' best epochs are 8-59 (table
  above). The earliest best epochs, 2, 4 and 5, are grid runs at the rate 1e-3. At epoch 100 every grid run
  is 0.0026-0.0073 below its best epoch [`last_epoch_metric`]: training past the best epoch overfits.

## Leave-one-family-out

Fitted on the train and validation records of three families, scored on the train and validation records of
the held-out family, at the main run's settings: C per condition for P0 and L-C; for H-All the learning rate
3e-4, weight decay 1e-4 and 11 epochs (the main run's best epoch for seed 20261110), seed 20261110. Within-source
macro AUROC per held-out source [`lofo/<id>/lofo.csv`]:

| held-out family | source | records | entering | P0 | L-C | H-All |
|---|---|---:|---:|---:|---:|---:|
| PTB-XL | PTB-XL | 19,601 | 14 | 0.9427 | 0.9072 | 0.9399 |
| Chapman and Ningbo | Ningbo | 27,891 | 14 | 0.9666 | 0.9274 | 0.9324 |
| Chapman and Ningbo | Chapman | 8,133 | 10 | 0.9650 | 0.9387 | 0.9533 |
| Georgia | Georgia | 8,160 | 12 | 0.9452 | 0.9048 | 0.9465 |
| CPSC | CPSC 2018 | 5,014 | 5 | 0.9129 | 0.9015 | 0.9164 |
| CPSC | CPSC-Extra | 2,589 | 10 | 0.9106 | 0.8926 | 0.9048 |

- These are different records from the validation table (train and validation of the held-out family), so
  the two tables are not compared row by row.
- Unweighted over the six held-out sources, the means are 0.9405 (P0), 0.9120 (L-C) and 0.9322 (H-All),
  computed from this table.
- H-All, the highest of the eight on validation, is above P0 in two held-out sources (Georgia, CPSC 2018)
  and below it in four. On the Chapman-Ningbo arm it is 0.0342 (Ningbo) and 0.0117 (Chapman) below P0.
- In the Chapman-Ningbo arm, P0's xECG features and H-All's xECG and ECG-JEPA features come from encoders
  pretrained on those waveforms. L-C's ECG-CPC features do not.
- L-C is lowest in every held-out source.

## Source identification

Five-fold multinomial logistic regression predicting the source family (PTB-XL, Chapman-Ningbo, Georgia,
CPSC) from each feature set, on the 15,138 validation records, folds grouped by patient or record and stratified
by family (`mlc02_evaluation.source_identification`, C = 1, seed 20261110)
[`source_identification/source_identification.json`]:

| features | width | accuracy | balanced accuracy | macro one-vs-rest AUROC | log loss |
|---|---:|---:|---:|---:|---:|
| xECG | 1,024 | 0.8524 | 0.7958 | 0.9565 | 0.4973 |
| ECG-JEPA | 768 | 0.8532 | 0.7909 | 0.9543 | 0.4386 |
| ECG-CPC | 512 | 0.9131 | 0.8737 | 0.9802 | 0.2902 |
| measurements (imputed, with flags) | 260 | 0.7154 | 0.5402 | 0.8454 | 0.8033 |
| all together | 2,564 | 0.9309 | 0.8992 | 0.9876 | 0.3098 |

- Chance references: majority class 0.5951 (Chapman-Ningbo, 9,008 of 15,138), 1/K balanced accuracy 0.25.
- Every feature set identifies the source family far above chance. ECG-CPC identifies it best of the single
  sets (balanced accuracy 0.874), and the measurements least (0.540).
- No convergence warning was raised.

## Training times

Wall time from each receipt [`receipt.json`, `wall_seconds`]; logistic and boosting fits ran in worker
processes, networks on GPU 1 (RTX 3090).

| run | wall | workers x threads | note |
|---|---:|---|---|
| P0 | 231 s | 6 x 2 | 70 fits, 1,305 s summed over workers |
| L-J | 223 s | 6 x 2 | 1,259 s summed |
| L-C | 133 s | 4 x 2 | 506 s summed |
| G-M | 252 s | 4 x 3 | 56 grid fits to 1,000 iterations (871 s summed), then 14 refits |
| H-X | 302 s | GPU, 4 threads | 7 runs of 100 epochs; profile forecast 208 s |
| H-J | 250 s | GPU, 4 threads | forecast 252 s |
| H-C | 220 s | GPU, 4 threads | forecast 223 s |
| H-All | 284 s | GPU, 4 threads | forecast 236 s; peak 845 MiB allocated |
| LOFO P0 | 361 s | 4 x 2 | |
| LOFO L-C | 204 s | 4 x 2 | |
| LOFO H-All | 93 s | GPU, 4 threads | |
| source identification | 65 s | 12 threads | |

Each network's profile of 3 epochs ran first and wrote `profile.json`, and the full run read it
(0.30-0.36 s per epoch, peak GPU memory 271-845 MiB allocated). The forecasts were 208-252 s per candidate;
the full runs took 220-302 s.

## Integrity checks

- Inputs: the local feature copy (`/var/tmp/janetrivera/mlc02/features/final/`) has the SHA-256 of every
  array and of `rows.csv` recorded in the Stage 1 receipt; every run checks this when it writes its receipt,
  and stores the hashes there. `rows.csv` was checked against the manifest by record key, first-10-s hash,
  role, source and inclusion (`included_v4` equal to the manifest's `included`).
- Role guard: `mlc02_feature_data.load_split` raises `PermissionError` for `test_same_source`,
  `test_external` and `sph_calibration` before any file is opened (tested). After the runs, all 32 saved
  prediction files were checked against the manifest: they hold train and validation records only (executed
  audit, 10 November).
- `predict` reproduces every saved validation file. `tests/test_mlc02_feature_models.py::
  test_predict_reproduces_the_saved_validation_predictions` reloads the validation features and compares, for
  every candidate and every network seed. The tolerance is 0 for logistic and boosting models and 1e-6 for
  networks; it passed. During the runs, `predict` matched the in-fit probabilities within 7.0e-15 (logistic)
  and 0 (G-M).
- Networks: the epoch was chosen on GPU float32 probabilities, and the saved files come from `predict`
  (float64 on the CPU from the same weights). They differ by at most 1.2e-6 per probability
  [`by_seed.gpu_max_abs_difference`], and every seed's selection metric agrees to 4 decimals.
- Code: the runs used commit `8cba1d3` with no modified tracked file (`git.modified_tracked` empty in every
  receipt); the receipts hold the SHA-256 of the 12 code files used.
- Environment: logistic fits ran with `OPENBLAS_CORETYPE` unset (the run refuses it), on OpenBLAS's SkylakeX
  kernel, with 2 BLAS and 2 OpenMP threads per worker, as in KEY-04's exact reproduction. Networks ran with
  deterministic algorithms, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, TF32 off and float32 matmul precision
  "highest". The receipts list numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.1, torch 2.6.0+cu124 and
  threadpoolctl 3.7.0.

## Departures and readings

All dated 10 November 2026. Where the protocol was silent, the reading below was fixed in code before the run
and is the same for every candidate it touches.

1. G-M "early stopping on validation". sklearn's built-in early stopping holds out a random tenth of the
   training records, which is not the validation role, so it is off. Each condition and setting is boosted to
   1,000 iterations. The iteration count with the highest per-condition validation metric is kept (the
   earliest on ties), and the model is refitted to that count. The cap of 1,000 is not in the protocol; no
   chosen count reached it. Unstated parameters keep sklearn's defaults (no L2 penalty, 20 records per leaf,
   255 bins); `random_state` is 20261110.
2. Network details the protocol does not fix:
   - ReLU activation; dropout after the hidden layer; PyTorch's default initialization;
   - a constant learning rate; AdamW's default betas, with weight decay on all parameters;
   - the loss averages binary cross-entropy over the defined entries of each batch;
   - records are reshuffled every epoch by a CPU generator seeded with the run's seed, and the last partial
     batch is kept;
   - the kept epoch is the first with the highest validation selection metric, chosen separately for each
     seed.
3. Logistic models: one standardizer on all training records, then row L2 normalization (KEY-04). Each
   condition is fitted on the training records where it is defined. A tie between C values would go to the
   smaller C; none occurred.
4. H-All inputs: a missing flag for every one of the 130 measurements, including those never missing in
   training (2,564 inputs). The imputed values and the flags are standardized with the encoder features.
5. Leave-one-family-out:
   - H-All trains for a fixed 11 epochs, the main run's best epoch for seed 20261110, because the held-out
     records cannot be used for stopping.
   - Standardizers and imputation medians are refitted on the three fitting families.
6. File names: logistic prediction files carry seed 42, the recipe's `random_state`, which lbfgs does not use.
   G-M files carry 20261110.
7. Thread budget. P0 and L-J ran with 6 workers at 2 BLAS and 2 OpenMP threads each. Each worker used about
   2.9 cores (the main thread plus one helper of each library), about 18 cores for those 8 minutes, above the
   brief's 16. The other runs used 4 workers (about 12 cores) beside one GPU job at 4 threads.
8. Hardware fault. The first full H-All run was killed at 17:36:20 by SIGBUS after an uncorrectable ECC memory
   error (dmesg: `EDAC MC0: 1 UE on mc#0csrow#0channel#9`, DIMM locator P0ME0; EDAC counts 1 uncorrected,
   0 corrected). It had saved nothing but its profile. It was rerun from the start at the same commit. The
   error was detected and reported, not silent, and dmesg shows no other process signalled, so the finished
   runs are not affected.
9. Smoke tests ran first on reduced grids (two C values, 15 boosting iterations, 3 epochs) and wrote only to
   a scratch folder, since deleted. They found one code bug, the CSV writer's missing field names, fixed in
   `8cba1d3` before any real run. Their numbers are not results.

## Doubts

- At C = 100, the upper edge of the protocol's grid, 17 of the 42 logistic condition choices sit at the
  edge. The protocol fixes the grid, so it was not widened; a larger C might help those conditions slightly.
- Every chosen setting and epoch is picked on the same validation records that the table reports. The table
  is therefore optimistic for every candidate; the protocol accepts this and scores the tests once later.
- The validation gap between H-All and P0 (0.0063) does not carry over to leave-one-family-out: there P0 is
  above H-All in four of six held-out sources. Part of the validation advantage may come from source-specific
  signal. Source identification is high for all feature sets, highest when they are combined.
- PTB-XL left axis deviation reaches a validation AUROC of 0.999-1.000 with xECG, JEPA and the measurements.
  PTB-XL defines it from `heart_axis`, with ambiguous values undefined (MLC-01 Amendment 1, ruling 4), so the
  remaining labels may be easy to separate. The label rules were not changed.

## How the later step calls `predict`

```python
import numpy as np

from ecg_experiment.mlc02_data import load_manifest, role_data
from ecg_experiment.mlc02_feature_data import FEATURE_FILES, LOCAL, aligned_rows, feature_positions
from ecg_experiment.mlc02_feature_models import INPUTS, predict

records = role_data(load_manifest(), role)            # one test role, manifest order
rows = aligned_rows(LOCAL / "rows.csv")               # checks key, hash, role, source, inclusion
index = feature_positions(records["record_key"], records["role"], rows)
features = {name: np.load(LOCAL / FEATURE_FILES[name], mmap_mode="r")[index] for name in INPUTS[candidate]}
probability = predict(candidate, features)            # [n, 14], CONDITIONS order; seeds=(s,) for one seed
```

`predict` reads `candidates/<id>/model.json`, checks the SHA-256 of every artefact, and accepts raw features
(float32 or float64, measurements with NaN). For the H candidates it returns the mean of the three seeds, or
one seed with `seeds=(s,)`. `load_split` is deliberately unusable for test roles. Thresholds at 90%
specificity and Platt calibration (Stage 3 item 2) are fitted in that later step from the saved validation
files.

## Files

- Code (commit `b6a64da`, fix `8cba1d3`): `ecg_experiment/mlc02_feature_data.py` (role guard, alignment,
  imputation), `ecg_experiment/mlc02_feature_models.py` (fits, artefacts, `predict`),
  `ecg_experiment/mlc02_feature_runs.py` (grids, selection, leave-one-family-out, source identification,
  receipts), `scripts/experiments/run_mlc02_feature_candidates.py`, `tests/test_mlc02_feature_models.py`
  (19 tests).
- Commands: `PYTHONPATH=. .venv/bin/python -m scripts.experiments.run_mlc02_feature_candidates <step>` with
  steps `logistic --candidate P0|L-J|L-C`, `boosting`, `network --candidate H-X|H-J|H-C|H-All`
  (`--profile-epochs 3` first; `CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8`), `lofo --candidate
  P0|L-C|H-All` and `source-id`. Logs: `/var/tmp/janetrivera/mlc02/logs/feature_candidates/`.
- `outputs/mlc02_condition_models_20261110/candidates/<id>/`:
  - `model.json` and the artefacts it hashes: `scaler.npz`, `logistic.npz`, `boosting__<condition>.joblib`,
    `imputation.npz`, `weights__seed-<s>.npz`;
  - `validation__seeds-<...>.npz` prediction files (one per network seed and their mean);
  - `candidate.json` (settings, grid, validation metrics), `validation_by_source*.csv`, `training_log.csv`
    (per fit or per epoch), `receipt.json`;
  - also `grid.npz` (every logistic fit), `curves.npz` (G-M metric per iteration) and `profile.json`
    (networks).
- `outputs/mlc02_condition_models_20261110/lofo/<id>/`: `held_out__<family>.npz`, `lofo.csv`, `lofo.json`,
  `receipt.json`.
- `outputs/mlc02_condition_models_20261110/source_identification/`: `source_identification.json` (with
  confusion matrices), `.csv`, `receipt.json`.

## Appendix: validation AUROC by source and condition

AUROC within each validation source [`candidates/<id>/validation_by_source.csv`]. Values in parentheses do
not enter that source's macro (fewer than 20 positives or negatives); a dash means the condition is undefined
there or has one class only.

P0

| condition | PTB-XL | Ningbo | Chapman | Georgia | CPSC 2018 | CPSC-Extra |
|---|---:|---:|---:|---:|---:|---:|
| sinus tachycardia | 0.994 | 0.999 | 0.998 | 0.994 | - | 0.978 |
| sinus bradycardia | 0.970 | 0.997 | 0.999 | 0.995 | - | (0.967) |
| sinus arrhythmia | 0.929 | 0.984 | - | 0.975 | - | (0.545) |
| AF or flutter | 0.995 | 0.999 | 0.999 | 0.927 | 0.995 | 0.991 |
| PVC | 0.993 | 0.986 | 0.980 | 0.870 | 0.911 | 0.872 |
| first-degree AV block | 0.978 | 0.987 | 0.987 | 0.979 | 0.977 | 0.951 |
| complete RBBB | 0.997 | 0.997 | 0.993 | 0.992 | 0.977 | 0.986 |
| incomplete RBBB | 0.952 | 0.970 | - | 0.953 | - | (0.941) |
| LAFB | 0.984 | 0.986 | - | 0.963 | - | - |
| left axis deviation | 1.000 | 0.988 | 0.981 | 0.975 | - | - |
| LVH | 0.949 | 0.995 | (0.980) | 0.927 | - | 0.937 |
| ST depression | 0.950 | 0.970 | 0.963 | - | 0.913 | (0.910) |
| T-wave abnormality | 0.916 | 0.964 | 0.951 | 0.896 | - | (0.811) |
| any MI | 0.909 | 0.969 | (0.986) | - | - | 0.890 |

L-J

| condition | PTB-XL | Ningbo | Chapman | Georgia | CPSC 2018 | CPSC-Extra |
|---|---:|---:|---:|---:|---:|---:|
| sinus tachycardia | 0.994 | 0.998 | 0.998 | 0.994 | - | 0.977 |
| sinus bradycardia | 0.965 | 0.996 | 0.999 | 0.993 | - | (0.969) |
| sinus arrhythmia | 0.955 | 0.988 | - | 0.976 | - | (0.843) |
| AF or flutter | 0.994 | 0.998 | 0.998 | 0.924 | 0.992 | 0.993 |
| PVC | 0.996 | 0.987 | 0.992 | 0.881 | 0.922 | 0.864 |
| first-degree AV block | 0.973 | 0.983 | 0.987 | 0.977 | 0.978 | 0.949 |
| complete RBBB | 0.995 | 0.995 | 0.987 | 0.985 | 0.968 | 0.992 |
| incomplete RBBB | 0.951 | 0.944 | - | 0.942 | - | (0.895) |
| LAFB | 0.979 | 0.981 | - | 0.953 | - | - |
| left axis deviation | 0.999 | 0.983 | 0.972 | 0.958 | - | - |
| LVH | 0.941 | 0.987 | (0.978) | 0.895 | - | 0.941 |
| ST depression | 0.943 | 0.968 | 0.968 | - | 0.912 | (0.901) |
| T-wave abnormality | 0.916 | 0.962 | 0.953 | 0.893 | - | (0.806) |
| any MI | 0.893 | 0.966 | (0.971) | - | - | 0.849 |

L-C

| condition | PTB-XL | Ningbo | Chapman | Georgia | CPSC 2018 | CPSC-Extra |
|---|---:|---:|---:|---:|---:|---:|
| sinus tachycardia | 0.991 | 0.997 | 0.996 | 0.990 | - | 0.971 |
| sinus bradycardia | 0.962 | 0.994 | 0.998 | 0.989 | - | (0.962) |
| sinus arrhythmia | 0.785 | 0.882 | - | 0.779 | - | (0.504) |
| AF or flutter | 0.991 | 0.997 | 0.995 | 0.917 | 0.987 | 0.986 |
| PVC | 0.975 | 0.940 | 0.939 | 0.803 | 0.862 | 0.764 |
| first-degree AV block | 0.970 | 0.983 | 0.990 | 0.973 | 0.970 | 0.933 |
| complete RBBB | 0.996 | 0.994 | 0.987 | 0.986 | 0.973 | 0.987 |
| incomplete RBBB | 0.915 | 0.917 | - | 0.931 | - | (0.828) |
| LAFB | 0.953 | 0.967 | - | 0.926 | - | - |
| left axis deviation | 0.981 | 0.950 | 0.936 | 0.928 | - | - |
| LVH | 0.932 | 0.985 | (0.988) | 0.887 | - | 0.927 |
| ST depression | 0.937 | 0.963 | 0.966 | - | 0.911 | (0.913) |
| T-wave abnormality | 0.890 | 0.945 | 0.926 | 0.892 | - | (0.835) |
| any MI | 0.869 | 0.961 | (0.989) | - | - | 0.812 |

G-M

| condition | PTB-XL | Ningbo | Chapman | Georgia | CPSC 2018 | CPSC-Extra |
|---|---:|---:|---:|---:|---:|---:|
| sinus tachycardia | 0.992 | 0.993 | 0.995 | 0.994 | - | 0.984 |
| sinus bradycardia | 0.965 | 0.996 | 0.999 | 0.997 | - | (0.968) |
| sinus arrhythmia | 0.965 | 0.975 | - | 0.990 | - | (0.641) |
| AF or flutter | 0.989 | 0.991 | 0.993 | 0.929 | 0.991 | 0.984 |
| PVC | 0.993 | 0.977 | 0.986 | 0.886 | 0.909 | 0.860 |
| first-degree AV block | 0.960 | 0.961 | 0.974 | 0.963 | 0.971 | 0.960 |
| complete RBBB | 0.997 | 0.994 | 0.990 | 0.990 | 0.971 | 0.988 |
| incomplete RBBB | 0.946 | 0.945 | - | 0.957 | - | (0.932) |
| LAFB | 0.983 | 0.984 | - | 0.961 | - | - |
| left axis deviation | 1.000 | 0.988 | 0.983 | 0.980 | - | - |
| LVH | 0.959 | 0.991 | (0.991) | 0.953 | - | 0.939 |
| ST depression | 0.951 | 0.965 | 0.967 | - | 0.896 | (0.940) |
| T-wave abnormality | 0.918 | 0.960 | 0.954 | 0.905 | - | (0.677) |
| any MI | 0.929 | 0.975 | (0.980) | - | - | 0.896 |

H-X (mean of three seeds)

| condition | PTB-XL | Ningbo | Chapman | Georgia | CPSC 2018 | CPSC-Extra |
|---|---:|---:|---:|---:|---:|---:|
| sinus tachycardia | 0.994 | 0.999 | 0.999 | 0.995 | - | 0.977 |
| sinus bradycardia | 0.968 | 0.998 | 1.000 | 0.995 | - | (0.965) |
| sinus arrhythmia | 0.942 | 0.985 | - | 0.972 | - | (0.493) |
| AF or flutter | 0.995 | 0.998 | 0.999 | 0.928 | 0.996 | 0.983 |
| PVC | 0.996 | 0.994 | 0.994 | 0.870 | 0.926 | 0.865 |
| first-degree AV block | 0.976 | 0.986 | 0.990 | 0.981 | 0.980 | 0.942 |
| complete RBBB | 0.997 | 0.998 | 0.993 | 0.993 | 0.981 | 0.978 |
| incomplete RBBB | 0.958 | 0.971 | - | 0.953 | - | (0.942) |
| LAFB | 0.985 | 0.984 | - | 0.965 | - | - |
| left axis deviation | 0.999 | 0.990 | 0.982 | 0.976 | - | - |
| LVH | 0.955 | 0.994 | (0.980) | 0.934 | - | 0.935 |
| ST depression | 0.954 | 0.973 | 0.965 | - | 0.925 | (0.907) |
| T-wave abnormality | 0.918 | 0.970 | 0.958 | 0.899 | - | (0.772) |
| any MI | 0.918 | 0.970 | (0.991) | - | - | 0.898 |

H-J (mean of three seeds)

| condition | PTB-XL | Ningbo | Chapman | Georgia | CPSC 2018 | CPSC-Extra |
|---|---:|---:|---:|---:|---:|---:|
| sinus tachycardia | 0.990 | 0.999 | 0.998 | 0.995 | - | 0.978 |
| sinus bradycardia | 0.970 | 0.998 | 1.000 | 0.996 | - | (0.974) |
| sinus arrhythmia | 0.960 | 0.991 | - | 0.979 | - | (0.775) |
| AF or flutter | 0.995 | 0.999 | 0.998 | 0.928 | 0.995 | 0.985 |
| PVC | 0.996 | 0.993 | 0.996 | 0.893 | 0.926 | 0.865 |
| first-degree AV block | 0.976 | 0.983 | 0.989 | 0.979 | 0.981 | 0.944 |
| complete RBBB | 0.997 | 0.995 | 0.988 | 0.991 | 0.978 | 0.982 |
| incomplete RBBB | 0.965 | 0.962 | - | 0.947 | - | (0.928) |
| LAFB | 0.982 | 0.981 | - | 0.954 | - | - |
| left axis deviation | 0.999 | 0.984 | 0.970 | 0.960 | - | - |
| LVH | 0.949 | 0.986 | (0.986) | 0.907 | - | 0.937 |
| ST depression | 0.951 | 0.972 | 0.971 | - | 0.926 | (0.860) |
| T-wave abnormality | 0.922 | 0.970 | 0.958 | 0.901 | - | (0.811) |
| any MI | 0.899 | 0.974 | (0.977) | - | - | 0.860 |

H-C (mean of three seeds)

| condition | PTB-XL | Ningbo | Chapman | Georgia | CPSC 2018 | CPSC-Extra |
|---|---:|---:|---:|---:|---:|---:|
| sinus tachycardia | 0.992 | 0.999 | 0.999 | 0.992 | - | 0.974 |
| sinus bradycardia | 0.972 | 0.997 | 0.999 | 0.994 | - | (0.977) |
| sinus arrhythmia | 0.866 | 0.910 | - | 0.836 | - | (0.684) |
| AF or flutter | 0.992 | 0.998 | 0.997 | 0.919 | 0.990 | 0.973 |
| PVC | 0.986 | 0.963 | 0.961 | 0.849 | 0.901 | 0.832 |
| first-degree AV block | 0.975 | 0.986 | 0.991 | 0.975 | 0.974 | 0.917 |
| complete RBBB | 0.996 | 0.996 | 0.989 | 0.989 | 0.979 | 0.990 |
| incomplete RBBB | 0.935 | 0.951 | - | 0.944 | - | (0.916) |
| LAFB | 0.964 | 0.969 | - | 0.933 | - | - |
| left axis deviation | 0.990 | 0.959 | 0.946 | 0.945 | - | - |
| LVH | 0.935 | 0.986 | (0.989) | 0.897 | - | 0.924 |
| ST depression | 0.954 | 0.972 | 0.966 | - | 0.935 | (0.939) |
| T-wave abnormality | 0.917 | 0.959 | 0.948 | 0.901 | - | (0.810) |
| any MI | 0.891 | 0.981 | (0.996) | - | - | 0.851 |

H-All (mean of three seeds)

| condition | PTB-XL | Ningbo | Chapman | Georgia | CPSC 2018 | CPSC-Extra |
|---|---:|---:|---:|---:|---:|---:|
| sinus tachycardia | 0.992 | 0.999 | 0.999 | 0.995 | - | 0.979 |
| sinus bradycardia | 0.976 | 0.998 | 1.000 | 0.997 | - | (0.980) |
| sinus arrhythmia | 0.963 | 0.991 | - | 0.982 | - | (0.670) |
| AF or flutter | 0.996 | 0.999 | 0.999 | 0.936 | 0.996 | 0.982 |
| PVC | 0.997 | 0.994 | 0.995 | 0.892 | 0.926 | 0.886 |
| first-degree AV block | 0.977 | 0.986 | 0.990 | 0.980 | 0.982 | 0.959 |
| complete RBBB | 0.996 | 0.998 | 0.992 | 0.992 | 0.985 | 0.979 |
| incomplete RBBB | 0.967 | 0.972 | - | 0.964 | - | (0.930) |
| LAFB | 0.986 | 0.987 | - | 0.969 | - | - |
| left axis deviation | 0.999 | 0.990 | 0.982 | 0.976 | - | - |
| LVH | 0.955 | 0.993 | (0.988) | 0.941 | - | 0.946 |
| ST depression | 0.962 | 0.977 | 0.969 | - | 0.944 | (0.888) |
| T-wave abnormality | 0.929 | 0.973 | 0.960 | 0.910 | - | (0.807) |
| any MI | 0.929 | 0.975 | (0.996) | - | - | 0.898 |
