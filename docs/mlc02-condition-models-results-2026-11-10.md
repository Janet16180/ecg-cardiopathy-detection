# MLC-02 results: models for the 14-condition classifier

11 November 2026. This report executes Stage 3 of the frozen
[MLC-02 protocol](mlc02-condition-models-protocol-2026-11-10.md) (commit `2516880`; Addenda 1 to 3 in `b093b61`,
`788db81` and `b8c5719`) on the final MLC-01 dataset
([protocol](mlc01-condition-dataset-protocol-2026-11-10.md), [results](mlc01-condition-dataset-results-2026-11-10.md)).
It is an ECG-annotation benchmark. The labels are annotation statements, not diagnoses, and clinical validation
is pending.

Every number below is read from the outputs under `outputs/mlc02_condition_models_20261110/` by
`scripts/reports/mlc02_results_tables.py`. The script computes no metric. It reads what the Stage 2 runs and
`run_mlc02_stage3 evaluate` wrote, takes differences of stored values, counts, and checks the SHA-256 of every
file the receipts pin. To print every table from the repository root (under a second):

```
PYTHONPATH=. .venv/bin/python -m scripts.reports.mlc02_results_tables
```

Add `--section <name>` for one table; each table below names its section. Intervals are 95% percentile
intervals over 2,000 patient resamples, shared by every candidate within a test set.

Earlier reports, cited and not repeated: [Stage 1 inputs](mlc02-inputs-results-2026-11-10.md),
[evaluation code](mlc02-evaluation-code-2026-11-10.md),
[feature candidates](mlc02-feature-candidates-validation-2026-11-10.md),
[S-R](mlc02-scratch-network-validation-2026-11-10.md), [F](mlc02-finetune-validation-2026-11-10.md),
[Stage 3 code](mlc02-stage3-code-2026-11-10.md) and the [selection record](mlc02-selection-record-2026-11-10.md).

## 1. Result

- Selected on validation: E, the mean probability of H-All, S-R and F. Its validation selection metric is
  0.9728. The selection record was committed as `7f4271c` at 00:46 PST on 11 November; the test predictions
  started at 00:52.
- Confirmatory comparison: E minus P0, macro AUROC over the 14 conditions, SPH evaluation half (12,698 records,
  12,246 patients): **-0.00123 (-0.00259, +0.00006)**. The interval includes zero (`confirmatory`).
- Decision by the protocol's rule: **no demonstrated gain. P0, the independent logistic regressions on pooled
  xECG embeddings, stays the reference model.**
- Secondary, not confirmatory: at PTB-XL fold 10, E is above P0 (0.9738 against 0.9632; difference +0.0106
  (0.0087, 0.0127)). At SPH the highest macro AUROC is H-X's (0.9748), and H-X is the only candidate whose
  interval against P0 lies above zero there (+0.0011 (0.0004, 0.0018)). H-X was not selected, and this is one
  of nine secondary comparisons at SPH, not adjusted for multiplicity.

## 2. Macro AUROC of every candidate

Raw probabilities, all 14 conditions entering in both sets (`macro`):

| candidate | validation selection metric | PTB-XL fold 10 | SPH evaluation half |
|---|---:|---:|---:|
| P0 | 0.9643 | 0.9632 (0.9591, 0.9669) | 0.9737 (0.9720, 0.9754) |
| L-J | 0.9614 | 0.9611 (0.9570, 0.9648) | 0.9660 (0.9636, 0.9682) |
| L-C | 0.9401 | 0.9417 (0.9362, 0.9469) | 0.9479 (0.9448, 0.9509) |
| G-M | 0.9632 | 0.9670 (0.9633, 0.9703) | 0.9691 (0.9670, 0.9711) |
| H-X | 0.9662 | 0.9655 (0.9616, 0.9690) | 0.9748 (0.9730, 0.9764) |
| H-J | 0.9646 | 0.9640 (0.9601, 0.9675) | 0.9686 (0.9664, 0.9706) |
| H-C | 0.9526 | 0.9527 (0.9476, 0.9574) | 0.9538 (0.9509, 0.9566) |
| H-All | 0.9706 | 0.9711 (0.9674, 0.9744) | 0.9719 (0.9691, 0.9742) |
| S-R | 0.9664 | 0.9694 (0.9656, 0.9730) | 0.9626 (0.9597, 0.9651) |
| F | 0.9685 | 0.9682 (0.9645, 0.9715) | 0.9706 (0.9684, 0.9725) |
| E | 0.9728 | 0.9738 (0.9703, 0.9768) | 0.9725 (0.9701, 0.9746) |

- Order on validation: E, H-All, F, S-R, H-X, H-J, P0, G-M, L-J, H-C, L-C.
- Order at PTB-XL fold 10: E, H-All, S-R, F, G-M, H-X, H-J, P0, L-J, H-C, L-C.
- Order at SPH: H-X, P0, E, H-All, F, G-M, H-J, L-J, S-R, H-C, L-C.
- The four candidates with the highest validation metric (E, H-All, F, S-R) are also the top four at fold 10.
  At SPH they are third to fifth and ninth, below H-X and P0. S-R moves from third at fold 10 to ninth at SPH.
- The validation selection metric is a mean over six validation sources from the training families; fold 10 is
  a PTB-XL fold; SPH is a hospital no model was trained on.
- After the validation Platt calibration the macro AUROC changes by at most 1.86e-4 over all candidates and
  sets. All 154 validation Platt slopes are positive (`selection`), so the map keeps the order of scores and can
  only create ties: the scoring code takes the logit of probabilities clipped to [1e-7, 1 - 1e-7]
  (`mlc02_evaluation.logit`).

## 3. Paired differences against P0

Candidate minus P0, macro AUROC, on the same resamples (`macro-differences`). Only E at SPH is confirmatory;
the other 19 cells are secondary and not adjusted for multiplicity.

| candidate minus P0 | PTB-XL fold 10 | lower end above 0 | SPH evaluation half | lower end above 0 |
|---|---:|---:|---:|---:|
| L-J | -0.0021 (-0.0046, 0.0003) | False | -0.0078 (-0.0094, -0.0061) | False |
| L-C | -0.0215 (-0.0256, -0.0174) | False | -0.0258 (-0.0284, -0.0233) | False |
| G-M | 0.0038 (0.0007, 0.0071) | True | -0.0046 (-0.0060, -0.0030) | False |
| H-X | 0.0023 (0.0010, 0.0036) | True | 0.0011 (0.0004, 0.0018) | True |
| H-J | 0.0008 (-0.0014, 0.0030) | False | -0.0051 (-0.0066, -0.0037) | False |
| H-C | -0.0105 (-0.0143, -0.0068) | False | -0.0199 (-0.0222, -0.0176) | False |
| H-All | 0.0078 (0.0061, 0.0098) | True | -0.0018 (-0.0039, -0.0002) | False |
| S-R | 0.0062 (0.0031, 0.0094) | True | -0.0111 (-0.0130, -0.0093) | False |
| F | 0.0050 (0.0031, 0.0071) | True | -0.0031 (-0.0044, -0.0019) | False |
| E | 0.0106 (0.0087, 0.0127) | True | -0.0012 (-0.0026, 0.0001) | False |

- At fold 10, six candidates are above P0 with the interval above zero: G-M, H-X, H-All, S-R, F and E.
- At SPH, only H-X is. Eight candidates are below P0 with the whole interval below zero: L-J, L-C, G-M, H-J,
  H-C, H-All, S-R and F.

Per condition, E minus P0 at SPH, with bootstrap p-values Holm-adjusted over the 14 conditions
(`condition-differences-sph`):

| condition | E minus P0 (95% interval) | p | Holm p | Holm p < 0.05 |
|---|---:|---:|---:|---:|
| sinus tachycardia | 0.0003 (0.0001, 0.0005) | 0.003 | 0.033 | yes |
| sinus bradycardia | 0.0019 (0.0012, 0.0025) | 0.001 | 0.014 | yes |
| sinus arrhythmia | 0.0053 (0.0002, 0.0099) | 0.044 | 0.352 |  |
| AF or flutter | 0.0000 (0.0000, 0.0001) | 0.013 | 0.130 |  |
| PVC | -0.0006 (-0.0051, 0.0037) | 0.760 | 0.924 |  |
| first-degree AV block | -0.0012 (-0.0025, -0.0001) | 0.021 | 0.189 |  |
| complete RBBB | -0.0006 (-0.0019, 0.0005) | 0.308 | 0.924 |  |
| incomplete RBBB | -0.0090 (-0.0182, 0.0002) | 0.056 | 0.392 |  |
| LAFB | 0.0009 (-0.0002, 0.0021) | 0.117 | 0.702 |  |
| left axis deviation | -0.0063 (-0.0217, 0.0030) | 0.319 | 0.924 |  |
| LVH | 0.0017 (-0.0008, 0.0043) | 0.204 | 0.875 |  |
| ST depression | -0.0120 (-0.0180, -0.0064) | 0.001 | 0.014 | yes |
| T-wave abnormality | 0.0053 (0.0039, 0.0069) | 0.001 | 0.014 | yes |
| any MI | -0.0030 (-0.0080, 0.0010) | 0.175 | 0.875 |  |

- After Holm, E is higher than P0 for sinus tachycardia, sinus bradycardia and T-wave abnormality, and lower for
  ST depression. The largest point differences are ST depression (-0.0120) and incomplete RBBB (-0.0090).
- The smallest attainable p-value at 2,000 resamples is about 0.001, or 0.014 after Holm
  ([evaluation code](mlc02-evaluation-code-2026-11-10.md), reading 8).

The same at PTB-XL fold 10 (`condition-differences-ptbxl`):

| condition | E minus P0 (95% interval) | p | Holm p | Holm p < 0.05 |
|---|---:|---:|---:|---:|
| sinus tachycardia | -0.0005 (-0.0028, 0.0010) | 0.723 | 1.000 |  |
| sinus bradycardia | 0.0136 (0.0015, 0.0289) | 0.024 | 0.090 |  |
| sinus arrhythmia | 0.0181 (0.0042, 0.0340) | 0.014 | 0.084 |  |
| AF or flutter | 0.0035 (0.0011, 0.0066) | 0.001 | 0.014 | yes |
| PVC | 0.0047 (0.0015, 0.0087) | 0.002 | 0.014 | yes |
| first-degree AV block | 0.0032 (0.0005, 0.0060) | 0.022 | 0.090 |  |
| complete RBBB | 0.0014 (0.0002, 0.0029) | 0.018 | 0.090 |  |
| incomplete RBBB | 0.0144 (0.0085, 0.0203) | 0.001 | 0.014 | yes |
| LAFB | 0.0057 (0.0032, 0.0085) | 0.001 | 0.014 | yes |
| left axis deviation | 0.0002 (-0.0008, 0.0011) | 0.696 | 1.000 |  |
| LVH | 0.0155 (0.0097, 0.0216) | 0.001 | 0.014 | yes |
| ST depression | 0.0174 (0.0080, 0.0275) | 0.001 | 0.014 | yes |
| T-wave abnormality | 0.0211 (0.0137, 0.0300) | 0.001 | 0.014 | yes |
| any MI | 0.0297 (0.0227, 0.0371) | 0.001 | 0.014 | yes |

At fold 10, E is higher than P0 after Holm for eight conditions and lower for none. ST depression and incomplete
RBBB, E's two largest losses at SPH, are among E's gains at fold 10.

## 4. Per condition, E and P0

Raw probabilities. Sensitivity and specificity are at the threshold fixed on validation for 90% specificity,
pooled over validation sources. On validation, specificity at those thresholds is 0.9000-0.9001 for every
candidate and condition (`selection`). "Imprecise": fewer than 100 positives. "Not usable": the AUROC interval's
lower end is below 0.80 or there are fewer than 100 positives (protocol, "Reading the result").

SPH evaluation half (`conditions-sph`):

| condition | positives (prevalence) | AUROC E | AUROC P0 | AP E / P0 | sensitivity E / P0 | specificity E / P0 | flags |
|---|---:|---:|---:|---:|---:|---:|---:|
| sinus tachycardia | 360 (0.028) | 0.999 (0.999, 1.000) | 0.999 (0.999, 0.999) | 0.981 / 0.969 | 1.000 / 1.000 | 0.956 / 0.936 |  |
| sinus bradycardia | 1302 (0.103) | 0.995 (0.994, 0.996) | 0.993 (0.992, 0.994) | 0.953 / 0.936 | 1.000 / 0.998 | 0.851 / 0.888 |  |
| sinus arrhythmia | 781 (0.062) | 0.945 (0.937, 0.953) | 0.940 (0.933, 0.947) | 0.649 / 0.572 | 0.932 / 0.923 | 0.782 / 0.811 |  |
| AF or flutter | 379 (0.030) | 1.000 (1.000, 1.000) | 1.000 (1.000, 1.000) | 0.997 / 0.995 | 1.000 / 1.000 | 0.971 / 0.976 |  |
| PVC | 517 (0.041) | 0.990 (0.985, 0.995) | 0.991 (0.985, 0.995) | 0.959 / 0.941 | 0.967 / 0.959 | 0.939 / 0.980 |  |
| first-degree AV block | 112 (0.009) | 0.994 (0.992, 0.996) | 0.995 (0.994, 0.997) | 0.633 / 0.678 | 0.991 / 1.000 | 0.916 / 0.940 |  |
| complete RBBB | 350 (0.028) | 0.996 (0.990, 0.999) | 0.996 (0.992, 0.999) | 0.967 / 0.965 | 0.997 / 0.997 | 0.755 / 0.781 |  |
| incomplete RBBB | 619 (0.049) | 0.843 (0.824, 0.860) | 0.852 (0.835, 0.868) | 0.389 / 0.346 | 0.498 / 0.407 | 0.965 / 0.971 |  |
| LAFB | 77 (0.006) | 0.994 (0.992, 0.996) | 0.993 (0.991, 0.995) | 0.485 / 0.487 | 1.000 / 0.987 | 0.945 / 0.950 | imprecise, not usable |
| left axis deviation | 68 (0.005) | 0.976 (0.951, 0.992) | 0.983 (0.973, 0.990) | 0.335 / 0.283 | 0.971 / 0.971 | 0.910 / 0.919 | imprecise, not usable |
| LVH | 100 (0.008) | 0.991 (0.985, 0.995) | 0.989 (0.985, 0.993) | 0.550 / 0.548 | 0.970 / 0.900 | 0.959 / 0.965 |  |
| ST depression | 497 (0.042) | 0.928 (0.919, 0.937) | 0.940 (0.932, 0.947) | 0.415 / 0.464 | 0.901 / 0.767 | 0.796 / 0.913 |  |
| T-wave abnormality | 1633 (0.129) | 0.977 (0.975, 0.980) | 0.972 (0.969, 0.975) | 0.873 / 0.860 | 0.904 / 0.862 | 0.933 / 0.940 |  |
| any MI | 118 (0.009) | 0.985 (0.978, 0.991) | 0.988 (0.983, 0.992) | 0.648 / 0.697 | 1.000 / 0.992 | 0.721 / 0.886 |  |

PTB-XL fold 10 (`conditions-ptbxl`):

| condition | positives (prevalence) | AUROC E | AUROC P0 | AP E / P0 | sensitivity E / P0 | specificity E / P0 | flags |
|---|---:|---:|---:|---:|---:|---:|---:|
| sinus tachycardia | 82 (0.037) | 0.995 (0.991, 0.998) | 0.996 (0.993, 0.998) | 0.888 / 0.881 | 0.988 / 1.000 | 0.924 / 0.927 | imprecise, not usable |
| sinus bradycardia | 64 (0.029) | 0.959 (0.940, 0.974) | 0.945 (0.916, 0.966) | 0.566 / 0.425 | 0.938 / 0.953 | 0.837 / 0.846 | imprecise, not usable |
| sinus arrhythmia | 77 (0.035) | 0.962 (0.942, 0.977) | 0.944 (0.920, 0.964) | 0.569 / 0.495 | 0.935 / 0.857 | 0.863 / 0.874 | imprecise, not usable |
| AF or flutter | 157 (0.071) | 0.981 (0.966, 0.993) | 0.978 (0.960, 0.991) | 0.899 / 0.904 | 0.955 / 0.949 | 0.913 / 0.915 |  |
| PVC | 114 (0.052) | 0.994 (0.991, 0.996) | 0.989 (0.984, 0.993) | 0.829 / 0.799 | 1.000 / 0.974 | 0.901 / 0.913 |  |
| first-degree AV block | 96 (0.044) | 0.984 (0.978, 0.990) | 0.981 (0.974, 0.987) | 0.728 / 0.707 | 1.000 / 0.990 | 0.873 / 0.879 | imprecise, not usable |
| complete RBBB | 54 (0.025) | 0.998 (0.996, 0.999) | 0.996 (0.994, 0.998) | 0.895 / 0.823 | 1.000 / 1.000 | 0.941 / 0.957 | imprecise, not usable |
| incomplete RBBB | 112 (0.051) | 0.980 (0.974, 0.986) | 0.966 (0.956, 0.975) | 0.743 / 0.613 | 1.000 / 0.964 | 0.837 / 0.839 |  |
| LAFB | 162 (0.074) | 0.988 (0.984, 0.992) | 0.983 (0.977, 0.987) | 0.871 / 0.809 | 1.000 / 0.994 | 0.837 / 0.857 |  |
| left axis deviation | 134 (0.160) | 0.998 (0.995, 1.000) | 0.998 (0.995, 1.000) | 0.992 / 0.989 | 0.993 / 0.993 | 0.952 / 0.969 |  |
| LVH | 214 (0.098) | 0.958 (0.947, 0.969) | 0.943 (0.929, 0.956) | 0.775 / 0.721 | 0.935 / 0.916 | 0.825 / 0.830 |  |
| ST depression | 101 (0.053) | 0.951 (0.929, 0.969) | 0.934 (0.908, 0.955) | 0.649 / 0.486 | 0.891 / 0.812 | 0.876 / 0.887 |  |
| T-wave abnormality | 255 (0.135) | 0.944 (0.929, 0.958) | 0.923 (0.905, 0.940) | 0.783 / 0.699 | 0.839 / 0.753 | 0.904 / 0.913 |  |
| any MI | 532 (0.246) | 0.939 (0.928, 0.950) | 0.909 (0.896, 0.922) | 0.854 / 0.783 | 0.947 / 0.885 | 0.706 / 0.754 |  |

The positive counts equal those of the MLC-01 report's support table for both sets.

Conditions not called usable (`usability`, every candidate):

- PTB-XL fold 10: sinus tachycardia (82 positives), sinus bradycardia (64), sinus arrhythmia (77), first-degree AV
  block (96) and complete RBBB (54), for every candidate, because they have fewer than 100 positives. L-C's sinus
  arrhythmia also has an interval lower end of 0.796.
- SPH evaluation half: LAFB (77 positives) and left axis deviation (68), for every candidate, because they have
  fewer than 100 positives. Also incomplete RBBB for L-J and L-C (lower ends 0.774 and 0.777) and sinus
  arrhythmia for L-C (0.769).
- For E and P0 every condition that is not called usable fails on the count of positives alone.

Other readings of these tables:

- Incomplete RBBB is the weakest condition at SPH for both models (AUROC 0.843 and 0.852), against 0.980 and
  0.966 at fold 10. At the validation thresholds both catch about half of the SPH positives (0.498, 0.407).
- With the validation thresholds, specificity at SPH ranges over E and P0 from 0.721 (E, any MI) to 0.980 (P0,
  PVC) instead of 0.90. Section 6 shows what thresholds fitted at SPH change.

## 5. Calibration

Rule, as recorded in `evaluation.json` (`calibration_rule`), from Addendum 2: the reported calibration slope and
joint intercept are the damped-Newton fits, with intervals on the same resamples; the scoring library's values
are listed beside them, with the count of fits where they are NaN; calibration in the large (CITL, slope fixed at
1) is the library's. A slope above 1 means the probabilities are not spread enough; a positive CITL means they
are too low on average.

E at SPH (`calibration-E-sph`):

| condition | raw slope (damped) | raw intercept (damped) | raw CITL | raw slope (library) | Platt slope (damped) | Platt CITL | Brier raw | Brier Platt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sinus tachycardia | 1.48 (1.30, 1.79) | -0.02 (-0.35, 0.37) | -0.25 | NaN | 1.30 (1.14, 1.58) | -0.27 | 0.0031 | 0.0031 |
| sinus bradycardia | 1.43 (1.33, 1.55) | -0.13 (-0.26, 0.01) | -0.20 | 1.43 | 0.95 (0.89, 1.03) | -0.11 | 0.0164 | 0.0156 |
| sinus arrhythmia | 1.35 (1.29, 1.42) | 0.29 (0.16, 0.43) | -0.22 | 1.35 | 0.86 (0.82, 0.90) | -0.09 | 0.0331 | 0.0330 |
| AF or flutter | 1.58 (1.39, 2.13) | -0.36 (-1.14, 0.38) | -0.49 | NaN | 1.48 (1.30, 1.99) | -0.73 | 0.0010 | 0.0010 |
| PVC | 1.87 (1.74, 2.05) | 1.51 (1.22, 1.82) | -0.09 | NaN | 1.36 (1.26, 1.48) | -0.03 | 0.0063 | 0.0045 |
| first-degree AV block | 1.13 (1.01, 1.29) | -1.69 (-1.98, -1.43) | -1.68 | NaN | 1.06 (0.95, 1.22) | -1.90 | 0.0068 | 0.0075 |
| complete RBBB | 1.44 (1.26, 1.71) | -2.80 (-3.37, -2.43) | -2.20 | NaN | 1.45 (1.27, 1.73) | -2.23 | 0.0105 | 0.0108 |
| incomplete RBBB | 0.61 (0.58, 0.65) | 1.20 (0.97, 1.42) | 3.14 | 0.61 | 0.54 (0.51, 0.58) | 3.29 | 0.0432 | 0.0426 |
| LAFB | 1.19 (1.04, 1.42) | -0.24 (-0.58, 0.13) | -0.44 | NaN | 1.15 (1.01, 1.38) | -0.37 | 0.0041 | 0.0040 |
| left axis deviation | 0.85 (0.67, 1.12) | -3.02 (-3.54, -2.74) | -3.23 | NaN | 0.80 (0.63, 1.06) | -3.36 | 0.0170 | 0.0175 |
| LVH | 1.17 (1.04, 1.36) | -0.38 (-0.71, -0.04) | -0.58 | NaN | 1.08 (0.96, 1.24) | -0.96 | 0.0048 | 0.0052 |
| ST depression | 0.97 (0.91, 1.03) | -0.59 (-0.73, -0.45) | -0.55 | 0.97 | 0.85 (0.80, 0.91) | -0.72 | 0.0321 | 0.0340 |
| T-wave abnormality | 1.35 (1.30, 1.42) | 0.06 (-0.03, 0.16) | -0.13 | 1.35 | 1.21 (1.16, 1.27) | -0.22 | 0.0388 | 0.0385 |
| any MI | 1.46 (1.32, 1.65) | -3.95 (-4.35, -3.68) | -3.47 | 1.46 | 1.19 (1.08, 1.34) | -3.78 | 0.0342 | 0.0350 |

P0 at SPH (`calibration-P0-sph`):

| condition | raw slope (damped) | raw intercept (damped) | raw CITL | raw slope (library) | Platt slope (damped) | Platt CITL | Brier raw | Brier Platt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sinus tachycardia | 1.53 (1.32, 1.87) | -1.33 (-1.68, -1.05) | -1.02 | NaN | 1.38 (1.19, 1.69) | -1.05 | 0.0050 | 0.0048 |
| sinus bradycardia | 0.87 (0.82, 0.93) | -0.83 (-0.96, -0.70) | -0.95 | 0.87 | 0.85 (0.80, 0.91) | -1.13 | 0.0207 | 0.0213 |
| sinus arrhythmia | 0.77 (0.73, 0.81) | -0.71 (-0.81, -0.60) | -0.48 | 0.77 | 0.77 (0.73, 0.81) | -0.56 | 0.0399 | 0.0403 |
| AF or flutter | 1.34 (1.20, 1.64) | -0.70 (-1.35, 0.03) | -0.85 | NaN | 1.32 (1.18, 1.61) | -0.91 | 0.0012 | 0.0012 |
| PVC | 2.12 (1.94, 2.35) | 3.30 (2.76, 3.96) | 0.23 | 2.12 | 1.38 (1.27, 1.54) | 0.55 | 0.0118 | 0.0091 |
| first-degree AV block | 1.05 (0.94, 1.21) | -1.48 (-1.78, -1.19) | -1.48 | NaN | 1.06 (0.95, 1.23) | -1.52 | 0.0060 | 0.0061 |
| complete RBBB | 1.50 (1.35, 1.74) | -1.82 (-2.16, -1.56) | -1.67 | 1.50 | 1.46 (1.31, 1.68) | -1.91 | 0.0061 | 0.0072 |
| incomplete RBBB | 0.45 (0.42, 0.48) | 0.38 (0.19, 0.58) | 3.02 | 0.45 | 0.53 (0.49, 0.56) | 2.84 | 0.0426 | 0.0431 |
| LAFB | 1.07 (0.93, 1.26) | -0.68 (-0.99, -0.34) | -0.74 | 1.07 | 1.18 (1.03, 1.40) | -0.42 | 0.0045 | 0.0042 |
| left axis deviation | 0.69 (0.57, 0.84) | -2.93 (-3.28, -2.68) | -3.36 | 0.69 | 0.72 (0.59, 0.87) | -3.08 | 0.0167 | 0.0149 |
| LVH | 1.09 (0.97, 1.25) | -0.90 (-1.22, -0.60) | -0.99 | 1.09 | 1.10 (0.98, 1.26) | -0.95 | 0.0055 | 0.0054 |
| ST depression | 0.95 (0.90, 1.02) | 0.12 (-0.04, 0.29) | 0.21 | 0.95 | 0.95 (0.89, 1.01) | 0.24 | 0.0292 | 0.0293 |
| T-wave abnormality | 1.18 (1.13, 1.24) | 0.06 (-0.04, 0.14) | -0.07 | 1.18 | 1.14 (1.09, 1.20) | -0.12 | 0.0409 | 0.0409 |
| any MI | 1.32 (1.19, 1.51) | -2.74 (-3.05, -2.49) | -2.69 | 1.32 | 1.25 (1.13, 1.42) | -2.63 | 0.0141 | 0.0130 |

Every candidate (`calibration-summary`). The library columns count the 14 point slopes and the 28,000 resampled
raw slopes (14 conditions by 2,000) where the library's fit gave NaN; the damped columns count the same for the
damped fit.

| set | candidate | library point slopes NaN (of 14) | library NaN resamples (raw) | damped point slopes NaN | damped NaN resamples (raw) | damped raw slope range | damped Platt slope range | Brier lower after Platt |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| PTB-XL fold 10 | P0 | 1 | 1923 | 0 | 8 | 0.52-1.58 | 0.50-1.36 | 4 of 14 |
| PTB-XL fold 10 | L-J | 2 | 2978 | 0 | 0 | 0.49-1.33 | 0.46-1.44 | 3 of 14 |
| PTB-XL fold 10 | L-C | 0 | 0 | 0 | 0 | 0.55-1.34 | 0.49-1.61 | 6 of 14 |
| PTB-XL fold 10 | G-M | 5 | 8984 | 0 | 60 | 0.44-0.96 | 0.56-1.18 | 7 of 14 |
| PTB-XL fold 10 | H-X | 3 | 5279 | 0 | 6 | 0.58-1.20 | 0.52-1.27 | 5 of 14 |
| PTB-XL fold 10 | H-J | 2 | 2950 | 0 | 0 | 0.53-1.42 | 0.52-1.60 | 6 of 14 |
| PTB-XL fold 10 | H-C | 1 | 2000 | 0 | 0 | 0.52-1.04 | 0.56-1.37 | 7 of 14 |
| PTB-XL fold 10 | H-All | 2 | 3469 | 0 | 1 | 0.60-1.14 | 0.58-1.32 | 7 of 14 |
| PTB-XL fold 10 | S-R | 5 | 8956 | 0 | 23 | 0.93-2.52 | 0.63-1.22 | 9 of 14 |
| PTB-XL fold 10 | F | 2 | 3477 | 0 | 43 | 0.64-1.32 | 0.59-1.38 | 11 of 14 |
| PTB-XL fold 10 | E | 4 | 7235 | 0 | 36 | 0.88-1.56 | 0.62-1.29 | 8 of 14 |
| SPH evaluation half | P0 | 3 | 5996 | 0 | 0 | 0.45-2.12 | 0.53-1.46 | 7 of 14 |
| SPH evaluation half | L-J | 4 | 7953 | 0 | 0 | 0.52-1.50 | 0.55-1.36 | 8 of 14 |
| SPH evaluation half | L-C | 2 | 2989 | 0 | 0 | 0.46-1.56 | 0.59-1.46 | 9 of 14 |
| SPH evaluation half | G-M | 9 | 17992 | 0 | 0 | 0.46-1.21 | 0.66-1.53 | 9 of 14 |
| SPH evaluation half | H-X | 5 | 9110 | 0 | 0 | 0.52-1.41 | 0.51-1.46 | 6 of 14 |
| SPH evaluation half | H-J | 3 | 5989 | 0 | 0 | 0.48-1.36 | 0.49-1.49 | 6 of 14 |
| SPH evaluation half | H-C | 2 | 3976 | 0 | 0 | 0.40-1.29 | 0.51-1.56 | 7 of 14 |
| SPH evaluation half | H-All | 5 | 9888 | 0 | 0 | 0.45-1.33 | 0.50-1.47 | 7 of 14 |
| SPH evaluation half | S-R | 7 | 14000 | 0 | 0 | 0.73-3.31 | 0.69-1.61 | 8 of 14 |
| SPH evaluation half | F | 5 | 9964 | 0 | 0 | 0.45-1.34 | 0.46-1.44 | 6 of 14 |
| SPH evaluation half | E | 8 | 15996 | 0 | 0 | 0.61-1.87 | 0.54-1.48 | 7 of 14 |

- The library's fit gives a NaN point slope for 0 to 9 of the 14 conditions per candidate and set; the damped fit
  gives none. The damped fit fails in a few resamples at fold 10 only (at most 60 of 28,000, G-M).
- Calibration at SPH is poor for several conditions, in E and P0 alike: CITL +3.14 and +3.02 for incomplete RBBB
  (probabilities too low), and -3.47 and -2.69 for any MI, -3.23 and -3.36 for left axis deviation (too high).
- The validation Platt calibration does not repair this at SPH: E's Platt CITL is 3.29 for incomplete RBBB and
  -3.78 for any MI. It lowers the Brier score for 7 of the 14 conditions for E and for P0.
- At fold 10, E's raw damped slopes range from 0.88 to 1.56 and its raw CITL from -0.77 to +1.17
  (`calibration-E-ptbxl`; P0 in `calibration-P0-ptbxl`).

## 6. Secondary analysis: thresholds and calibration refitted at SPH

Thresholds (90% specificity) and the damped Platt fit were refitted on the SPH calibration half (12,883 records)
and applied to the SPH evaluation half. The differences are local fit minus validation fit, on the evaluation
half. By Addendum 3, item 4, their intervals resample the evaluation half with the local thresholds and
calibration held fixed. They leave out the variability of the fit on the calibration half, so they are too
narrow.

E (`secondary-E`):

| condition | sens. validation thr. | sens. local thr. | difference | spec. validation thr. | spec. local thr. | difference | Platt slope, validation fit | Platt slope, local fit | difference |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sinus tachycardia | 1.000 | 1.000 | 0.000 (0.000, 0.000) | 0.956 | 0.897 | -0.059 (-0.063, -0.055) | 1.30 | 1.05 | -0.25 (-0.30, -0.22) |
| sinus bradycardia | 1.000 | 0.998 | -0.002 (-0.004, 0.000) | 0.851 | 0.900 | 0.049 (0.045, 0.053) | 0.95 | 1.05 | 0.10 (0.09, 0.11) |
| sinus arrhythmia | 0.932 | 0.839 | -0.093 (-0.115, -0.073) | 0.782 | 0.905 | 0.123 (0.117, 0.129) | 0.86 | 1.03 | 0.17 (0.16, 0.18) |
| AF or flutter | 1.000 | 1.000 | 0.000 (0.000, 0.000) | 0.971 | 0.897 | -0.074 (-0.079, -0.070) | 1.48 | 1.01 | -0.47 (-0.63, -0.41) |
| PVC | 0.967 | 0.973 | 0.006 (0.000, 0.013) | 0.939 | 0.902 | -0.037 (-0.040, -0.034) | 1.36 | 0.94 | -0.42 (-0.46, -0.39) |
| first-degree AV block | 0.991 | 0.991 | 0.000 (0.000, 0.000) | 0.916 | 0.897 | -0.018 (-0.021, -0.016) | 1.06 | 1.01 | -0.05 (-0.06, -0.04) |
| complete RBBB | 0.997 | 0.991 | -0.006 (-0.015, 0.000) | 0.755 | 0.902 | 0.147 (0.141, 0.153) | 1.45 | 1.10 | -0.35 (-0.42, -0.31) |
| incomplete RBBB | 0.498 | 0.653 | 0.155 (0.127, 0.184) | 0.965 | 0.903 | -0.062 (-0.066, -0.058) | 0.54 | 1.03 | 0.49 (0.46, 0.52) |
| LAFB | 1.000 | 1.000 | 0.000 (0.000, 0.000) | 0.945 | 0.900 | -0.044 (-0.048, -0.041) | 1.15 | 0.93 | -0.22 (-0.26, -0.19) |
| left axis deviation | 0.971 | 0.971 | 0.000 (0.000, 0.000) | 0.910 | 0.895 | -0.016 (-0.018, -0.014) | 0.80 | 0.97 | 0.17 (0.14, 0.23) |
| LVH | 0.970 | 0.980 | 0.010 (0.000, 0.034) | 0.959 | 0.899 | -0.060 (-0.065, -0.056) | 1.08 | 0.95 | -0.13 (-0.15, -0.12) |
| ST depression | 0.901 | 0.748 | -0.153 (-0.186, -0.121) | 0.796 | 0.900 | 0.104 (0.098, 0.109) | 0.85 | 1.04 | 0.19 (0.18, 0.21) |
| T-wave abnormality | 0.904 | 0.945 | 0.041 (0.031, 0.051) | 0.933 | 0.899 | -0.034 (-0.038, -0.031) | 1.21 | 1.03 | -0.18 (-0.19, -0.17) |
| any MI | 1.000 | 0.975 | -0.025 (-0.057, 0.000) | 0.721 | 0.900 | 0.179 (0.172, 0.185) | 1.19 | 1.09 | -0.10 (-0.11, -0.09) |

P0 (`secondary-P0`):

| condition | sens. validation thr. | sens. local thr. | difference | spec. validation thr. | spec. local thr. | difference | Platt slope, validation fit | Platt slope, local fit | difference |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sinus tachycardia | 1.000 | 1.000 | 0.000 (0.000, 0.000) | 0.936 | 0.899 | -0.038 (-0.041, -0.035) | 1.38 | 1.06 | -0.32 (-0.39, -0.28) |
| sinus bradycardia | 0.998 | 0.998 | 0.000 (0.000, 0.000) | 0.888 | 0.904 | 0.015 (0.013, 0.017) | 0.85 | 1.07 | 0.22 (0.21, 0.24) |
| sinus arrhythmia | 0.923 | 0.809 | -0.114 (-0.137, -0.093) | 0.811 | 0.903 | 0.092 (0.087, 0.097) | 0.77 | 0.99 | 0.22 (0.21, 0.23) |
| AF or flutter | 1.000 | 1.000 | 0.000 (0.000, 0.000) | 0.976 | 0.900 | -0.076 (-0.081, -0.071) | 1.32 | 0.91 | -0.41 (-0.50, -0.37) |
| PVC | 0.959 | 0.979 | 0.019 (0.008, 0.032) | 0.980 | 0.902 | -0.078 (-0.083, -0.073) | 1.38 | 0.99 | -0.39 (-0.43, -0.36) |
| first-degree AV block | 1.000 | 1.000 | 0.000 (0.000, 0.000) | 0.940 | 0.896 | -0.044 (-0.048, -0.041) | 1.06 | 1.04 | -0.02 (-0.02, -0.02) |
| complete RBBB | 0.997 | 0.997 | 0.000 (0.000, 0.000) | 0.781 | 0.904 | 0.124 (0.118, 0.130) | 1.46 | 1.15 | -0.30 (-0.35, -0.27) |
| incomplete RBBB | 0.407 | 0.607 | 0.200 (0.170, 0.232) | 0.971 | 0.904 | -0.066 (-0.071, -0.062) | 0.53 | 1.03 | 0.50 (0.47, 0.53) |
| LAFB | 0.987 | 1.000 | 0.013 (0.000, 0.045) | 0.950 | 0.897 | -0.053 (-0.057, -0.049) | 1.18 | 1.09 | -0.10 (-0.12, -0.09) |
| left axis deviation | 0.971 | 0.971 | 0.000 (0.000, 0.000) | 0.919 | 0.895 | -0.025 (-0.028, -0.022) | 0.72 | 0.93 | 0.21 (0.17, 0.26) |
| LVH | 0.900 | 0.990 | 0.090 (0.038, 0.150) | 0.965 | 0.899 | -0.066 (-0.070, -0.062) | 1.10 | 0.91 | -0.19 (-0.21, -0.16) |
| ST depression | 0.767 | 0.787 | 0.020 (0.008, 0.033) | 0.913 | 0.901 | -0.012 (-0.014, -0.010) | 0.95 | 1.08 | 0.13 (0.12, 0.14) |
| T-wave abnormality | 0.862 | 0.921 | 0.059 (0.047, 0.071) | 0.940 | 0.899 | -0.041 (-0.045, -0.037) | 1.14 | 1.01 | -0.13 (-0.13, -0.12) |
| any MI | 0.992 | 0.992 | 0.000 (0.000, 0.000) | 0.886 | 0.901 | 0.015 (0.013, 0.017) | 1.25 | 1.09 | -0.16 (-0.18, -0.14) |

Every candidate, range over the 14 conditions on the evaluation half (`secondary-summary`):

| candidate | specificity, validation thresholds | specificity, local thresholds | sensitivity, validation thresholds | sensitivity, local thresholds | Platt slope, validation fit | Platt slope, local fit |
|---|---:|---:|---:|---:|---:|---:|
| P0 | 0.781-0.980 | 0.895-0.904 | 0.407-1.000 | 0.607-1.000 | 0.53-1.46 | 0.91-1.15 |
| L-J | 0.816-0.983 | 0.895-0.905 | 0.341-1.000 | 0.525-1.000 | 0.55-1.36 | 0.90-1.14 |
| L-C | 0.614-0.977 | 0.896-0.908 | 0.346-1.000 | 0.446-1.000 | 0.59-1.46 | 0.84-1.11 |
| G-M | 0.796-0.985 | 0.895-0.908 | 0.557-1.000 | 0.614-1.000 | 0.66-1.53 | 0.95-1.15 |
| H-X | 0.778-0.977 | 0.893-0.905 | 0.446-1.000 | 0.625-1.000 | 0.51-1.46 | 0.92-1.12 |
| H-J | 0.776-0.978 | 0.894-0.906 | 0.370-1.000 | 0.549-1.000 | 0.49-1.49 | 0.87-1.14 |
| H-C | 0.431-0.967 | 0.895-0.903 | 0.439-1.000 | 0.504-1.000 | 0.51-1.56 | 0.85-1.23 |
| H-All | 0.664-0.968 | 0.895-0.906 | 0.494-1.000 | 0.614-1.000 | 0.50-1.47 | 0.91-1.21 |
| S-R | 0.648-0.964 | 0.895-0.907 | 0.525-1.000 | 0.560-1.000 | 0.69-1.61 | 0.89-1.13 |
| F | 0.665-0.975 | 0.897-0.908 | 0.410-1.000 | 0.612-1.000 | 0.46-1.44 | 0.91-1.10 |
| E | 0.721-0.971 | 0.895-0.905 | 0.498-1.000 | 0.653-1.000 | 0.54-1.48 | 0.93-1.10 |

What local data adds, on these numbers:

- Thresholds fitted at the hospital hold the target. With them, specificity on the evaluation half is
  0.893-0.908 for every candidate and condition, against 0.431-0.985 with the validation thresholds.
- Sensitivity follows. For E it falls for ST depression (-0.153) and sinus arrhythmia (-0.093), where the
  validation threshold had bought sensitivity with low specificity, and rises for incomplete RBBB (+0.155).
- The local Platt fit brings the calibration slopes to 0.84-1.23 over all candidates, against 0.46-1.61 after
  the validation fit.

## 7. Leave-one-family-out and source identification

Both were run in Stage 2 on train and validation records only; no test record enters them.

Leave-one-family-out: fitted on three families, scored on every train and validation record of the held-out
family; one seed (`lofo`, from `lofo/<id>/lofo.json`):

| held-out source | records | entering (S-R / others) | P0 | L-C | H-All | S-R (seed 20261110) |
|---|---:|---:|---:|---:|---:|---:|
| PTB-XL | 19601 | 14 / 14 | 0.9427 | 0.9072 | 0.9399 | 0.9314 |
| Ningbo | 27891 | 14 / 14 | 0.9666 | 0.9274 | 0.9324 | 0.9522 |
| Chapman | 8133 | 10 / 10 | 0.9650 | 0.9387 | 0.9533 | 0.9585 |
| Georgia | 8160 | 12 / 12 | 0.9452 | 0.9048 | 0.9465 | 0.9323 |
| CPSC 2018 | 5014 | 5 / 5 | 0.9129 | 0.9015 | 0.9164 | 0.9014 |
| CPSC-Extra | 2589 | 10 / 10 | 0.9106 | 0.8926 | 0.9048 | 0.8996 |
| unweighted mean |  |  | 0.9405 | 0.9120 | 0.9322 | 0.9292 |

- For these four candidates the held-out mean orders them P0, H-All, S-R, L-C. SPH orders them the same way
  (0.9737, 0.9719, 0.9626, 0.9479). Validation orders them H-All, S-R, P0, L-C, and so does fold 10. This is an
  observation on four candidates, not a test.
- In the Chapman-Ningbo arm, P0's and H-All's xECG and ECG-JEPA features come from encoders pretrained on those
  waveforms, so that arm is not clean for them; S-R and L-C have no such overlap.
- F has no leave-one-family-out run (section 9).

Source identification: five-fold multinomial logistic regression predicting the source family from each feature
set on the 15,138 validation records (`source-identification`):

| features | width | accuracy | balanced accuracy | macro one-vs-rest AUROC | majority class | 1/K |
|---|---:|---:|---:|---:|---:|---:|
| xecg | 1024 | 0.8524 | 0.7958 | 0.9565 | 0.5951 | 0.25 |
| jepa | 768 | 0.8532 | 0.7909 | 0.9543 | 0.5951 | 0.25 |
| cpc | 512 | 0.9131 | 0.8737 | 0.9802 | 0.5951 | 0.25 |
| measurements | 260 | 0.7154 | 0.5402 | 0.8454 | 0.5951 | 0.25 |
| all | 2564 | 0.9309 | 0.8992 | 0.9876 | 0.5951 | 0.25 |

Every feature set names the source family far above chance. The combined input of H-All names it best (balanced
accuracy 0.899); the measurements least (0.540).

## 8. Audits

Point estimates, by Addendum 3, item 3. Raw probabilities and validation thresholds.

Macro AUROC within subgroups, E and P0 (`subgroups`; every candidate is printed there). The number of entering
conditions is in parentheses; groups with different entering sets are not directly comparable.

| set | candidate | female | male | one condition | several conditions |
|---|---|---:|---:|---:|---:|
| PTB-XL fold 10 | P0 | 0.9624 (13) | 0.9603 (14) | 0.9516 (10) | 0.9642 (14) |
| PTB-XL fold 10 | E | 0.9742 (13) | 0.9704 (14) | 0.9640 (10) | 0.9753 (14) |
| SPH evaluation half | P0 | 0.9720 (12) | 0.9731 (14) | 0.9719 (13) | 0.9755 (14) |
| SPH evaluation half | E | 0.9716 (12) | 0.9731 (14) | 0.9734 (13) | 0.9712 (14) |

- Sex: 1,066 female and 1,132 male records at fold 10, 5,601 and 7,097 at SPH; no record has unknown sex.
  Complete RBBB does not enter the female macro at fold 10 (18 positives); left axis deviation (17) and any MI
  (13) do not enter it at SPH.
- One condition against several: each group keeps all negatives of the condition. At fold 10 first-degree AV
  block, complete RBBB, left axis deviation and ST depression have too few single-condition positives to enter
  (15, 11, 7, 15); at SPH first-degree AV block (15).
- Flat leads: neither test set has a record with a flat or missing lead in the first 10 s; only the
  all-leads-present group exists (2,198 and 12,698 records). The MLC-01 report counts 0 flat-lead records at SPH.

Likelihood below 50 undefined, PTB-XL fold 10 only (`likelihood`). 178 label entries become undefined; 391 MI and
177 LVH positives remain, and all 14 conditions still enter.

| candidate | main labels | likelihood >= 50 (95% interval) | any MI main / lik | LVH main / lik |
|---|---:|---:|---:|---:|
| P0 | 0.9632 | 0.9655 (0.9613, 0.9692) | 0.909 / 0.935 | 0.943 / 0.949 |
| L-J | 0.9611 | 0.9643 (0.9602, 0.9681) | 0.868 / 0.902 | 0.940 / 0.952 |
| L-C | 0.9417 | 0.9455 (0.9398, 0.9506) | 0.853 / 0.894 | 0.925 / 0.937 |
| G-M | 0.9670 | 0.9686 (0.9649, 0.9719) | 0.935 / 0.958 | 0.954 / 0.954 |
| H-X | 0.9655 | 0.9674 (0.9636, 0.9709) | 0.916 / 0.939 | 0.947 / 0.951 |
| H-J | 0.9640 | 0.9669 (0.9630, 0.9704) | 0.878 / 0.908 | 0.945 / 0.955 |
| H-C | 0.9527 | 0.9566 (0.9517, 0.9612) | 0.875 / 0.915 | 0.935 / 0.949 |
| H-All | 0.9711 | 0.9733 (0.9697, 0.9765) | 0.926 / 0.950 | 0.954 / 0.961 |
| S-R | 0.9694 | 0.9713 (0.9675, 0.9749) | 0.942 / 0.962 | 0.956 / 0.963 |
| F | 0.9682 | 0.9704 (0.9667, 0.9737) | 0.921 / 0.947 | 0.948 / 0.953 |
| E | 0.9738 | 0.9757 (0.9722, 0.9787) | 0.939 / 0.960 | 0.958 / 0.964 |

Every candidate's macro AUROC rises when low-likelihood statements are undefined, mostly through any MI. The
order of the candidates is unchanged.

Derived multiclass view (`multiclass`). Top-k is on records with exactly one positive condition, under both
readings of Addendum 1: undefined entries allowed among the others (702 records at fold 10, 3,882 at SPH) and all
14 defined (204 and 3,434). Each cell is raw / after validation Platt. Exact set is on all records (2,198 and
12,698), and on those with all 14 defined (711 and 11,678), at the validation thresholds.

| set | candidate | top-1, undefined allowed | top-1, all defined | top-2, undefined allowed | top-2, all defined | exact set | exact set, all defined |
|---|---|---:|---:|---:|---:|---:|---:|
| PTB-XL fold 10 | P0 | 0.698 / 0.697 | 0.809 / 0.814 | 0.846 / 0.853 | 0.922 / 0.917 | 0.259 | 0.291 |
| PTB-XL fold 10 | L-J | 0.650 / 0.648 | 0.725 / 0.721 | 0.848 / 0.838 | 0.907 / 0.892 | 0.263 | 0.284 |
| PTB-XL fold 10 | L-C | 0.561 / 0.571 | 0.623 / 0.623 | 0.776 / 0.778 | 0.838 / 0.838 | 0.216 | 0.243 |
| PTB-XL fold 10 | G-M | 0.702 / 0.687 | 0.838 / 0.814 | 0.875 / 0.868 | 0.956 / 0.941 | 0.306 | 0.326 |
| PTB-XL fold 10 | H-X | 0.702 / 0.712 | 0.819 / 0.828 | 0.858 / 0.860 | 0.931 / 0.936 | 0.255 | 0.281 |
| PTB-XL fold 10 | H-J | 0.677 / 0.678 | 0.755 / 0.745 | 0.840 / 0.845 | 0.892 / 0.897 | 0.237 | 0.270 |
| PTB-XL fold 10 | H-C | 0.611 / 0.618 | 0.716 / 0.716 | 0.808 / 0.811 | 0.887 / 0.877 | 0.222 | 0.266 |
| PTB-XL fold 10 | H-All | 0.729 / 0.731 | 0.843 / 0.848 | 0.877 / 0.879 | 0.956 / 0.956 | 0.242 | 0.286 |
| PTB-XL fold 10 | S-R | 0.741 / 0.735 | 0.848 / 0.843 | 0.875 / 0.870 | 0.946 / 0.941 | 0.274 | 0.328 |
| PTB-XL fold 10 | F | 0.719 / 0.729 | 0.838 / 0.848 | 0.865 / 0.868 | 0.941 / 0.926 | 0.253 | 0.295 |
| PTB-XL fold 10 | E | 0.748 / 0.749 | 0.853 / 0.858 | 0.885 / 0.885 | 0.961 / 0.956 | 0.265 | 0.316 |
| SPH evaluation half | P0 | 0.766 / 0.772 | 0.774 / 0.780 | 0.878 / 0.882 | 0.872 / 0.877 | 0.346 | 0.353 |
| SPH evaluation half | L-J | 0.756 / 0.760 | 0.761 / 0.763 | 0.865 / 0.865 | 0.858 / 0.858 | 0.356 | 0.365 |
| SPH evaluation half | L-C | 0.651 / 0.652 | 0.643 / 0.644 | 0.790 / 0.787 | 0.781 / 0.778 | 0.278 | 0.288 |
| SPH evaluation half | G-M | 0.746 / 0.753 | 0.747 / 0.755 | 0.864 / 0.874 | 0.857 / 0.868 | 0.374 | 0.383 |
| SPH evaluation half | H-X | 0.781 / 0.783 | 0.789 / 0.791 | 0.891 / 0.889 | 0.884 / 0.883 | 0.332 | 0.337 |
| SPH evaluation half | H-J | 0.766 / 0.765 | 0.773 / 0.770 | 0.864 / 0.862 | 0.859 / 0.857 | 0.319 | 0.325 |
| SPH evaluation half | H-C | 0.630 / 0.631 | 0.625 / 0.624 | 0.793 / 0.787 | 0.785 / 0.779 | 0.169 | 0.170 |
| SPH evaluation half | H-All | 0.753 / 0.753 | 0.755 / 0.756 | 0.865 / 0.863 | 0.860 / 0.858 | 0.252 | 0.255 |
| SPH evaluation half | S-R | 0.733 / 0.757 | 0.732 / 0.758 | 0.851 / 0.868 | 0.844 / 0.861 | 0.190 | 0.182 |
| SPH evaluation half | F | 0.761 / 0.757 | 0.770 / 0.766 | 0.858 / 0.861 | 0.852 / 0.855 | 0.244 | 0.241 |
| SPH evaluation half | E | 0.771 / 0.773 | 0.775 / 0.780 | 0.871 / 0.875 | 0.864 / 0.869 | 0.209 | 0.205 |

- At SPH, E names the single positive condition first on 77.1% of single-condition records (undefined allowed)
  and P0 on 76.6%; H-X is highest (78.1%).
- Exact-set accuracy is low for every candidate (0.169-0.383). Each condition's threshold was set for 90%
  validation specificity on its own; the rule does not aim at the whole set.

Top-1 by true condition at SPH, undefined allowed (`multiclass-by-condition-sph`; fold 10 in
`multiclass-by-condition-ptbxl`):

| condition | single-condition records | top-1 P0 | top-1 E |
|---|---:|---:|---:|
| sinus tachycardia | 223 | 0.987 | 0.960 |
| sinus bradycardia | 743 | 0.985 | 0.969 |
| sinus arrhythmia | 473 | 0.877 | 0.827 |
| AF or flutter | 154 | 0.981 | 1.000 |
| PVC | 288 | 0.729 | 0.910 |
| first-degree AV block | 15 | 0.800 | 0.933 |
| complete RBBB | 208 | 0.952 | 0.976 |
| incomplete RBBB | 446 | 0.099 | 0.004 |
| LAFB | 31 | 0.000 | 0.000 |
| left axis deviation | 38 | 0.921 | 0.947 |
| LVH | 38 | 0.342 | 0.211 |
| ST depression | 286 | 0.552 | 0.675 |
| T-wave abnormality | 904 | 0.833 | 0.842 |
| any MI | 35 | 0.971 | 0.971 |

- Incomplete RBBB is almost never named first at SPH (E 0.004, P0 0.099 of 446 records), against 0.792 and 0.708
  of 48 records at fold 10. The outputs do not say which condition was named instead.
- LAFB is never named first at SPH (31 records) and on 1 of 23 at fold 10 (0.043). The view ranks all 14 outputs,
  undefined ones included ([evaluation code](mlc02-evaluation-code-2026-11-10.md), reading 10), and left axis
  deviation is undefined on LAFB records without a left-axis code (MLC-01 Amendment 3, ruling 1). Whether left
  axis deviation is the condition named instead is not in the outputs.

## 9. Not done, and departures

Not done:

- Space A beside KEY-04. Deferred to the backlog by Addendum 3, item 2: the models output the 14 space-B
  conditions only, and no rule mapping them to MI, STTC, CD and HYP was fixed before the test predictions.
- Window mean over SPH recordings longer than 10 s. Deferred by Addendum 3, item 2: it needs features for every
  10 s window from the raw SPH files.
- Leave-one-family-out for F. The protocol lists it for P0, L-C, H-All and S-R only; the F report states it was
  not run (departure 6 there).
- Per-seed test results. The protocol reports each seed of H, S-R and F. `test-predict` wrote one file per
  candidate and role, holding the mean of its seeds (`test_predictions_receipt.json`, `files`), so no per-seed
  test number exists. Per-seed validation numbers are in the Stage 2 reports.
- Intervals for the sex, flat-lead, one-against-several and top-k audits (point estimates by Addendum 3, item 3).

Departures from the protocols, with their dates:

1. 10 November, Addendum 2, after validation scores and before any test prediction: calibration fits by damped
   Newton, because the library's fit does not converge when a score separates the classes well.
2. 10 November, Addendum 3, same timing: E's families, the two deferred audits, point-estimate audits, the
   likelihood audit at fold 10 only, and the fixed-fit interval of the secondary analysis.
3. The per-seed test results above.
4. Earlier departures of Stages 1 and 2 are listed in their reports.

## 10. Limitations

- The SPH evaluation half and PTB-XL fold 10 were scored in earlier experiments for other targets, so this is the
  best evidence available without new data, not an untouched final test (protocol, "Reading the result").
- The labels are ECG-annotation proxies, not diagnoses, and clinical validation is pending (MLC-01, decision 6).
- 17 of the 42 per-condition C choices of P0, L-J and L-C sit at the upper edge of the grid, C = 100
  (`selection`; feature report, Doubts).
- S-R's epochs and F's rate and kept points were chosen on the validation records they are scored on, so their
  validation numbers are optimistic (S-R report, Validation results; F report, Doubts).
- E's validation metric is optimistic twice, since its members and E itself are chosen on the same validation
  records (Addendum 3, item 1).
- xECG and ECG-JEPA were pretrained on Chapman and Ningbo waveforms, which sit in training, validation and the
  Chapman-Ningbo leave-one-family-out arm (protocol, "What the encoders saw before").
- The Challenge train and validation groups share patients, an estimated 6-13% of validation records in Ningbo,
  Georgia and Chapman-Ningbo, which affects model choice, stopping and thresholds but not the test sets
  (MLC-01 Amendment 3).
- Complete RBBB in CPSC 2018 and Chapman may include incomplete blocks; this is unverified (MLC-01 erratum to
  Amendment 3).
- SPH code 146 counts as T-wave abnormality; 521 of the 1,633 SPH evaluation-half T-wave positives rest on it
  alone (MLC-01 Amendment 3).
- PTB-XL codes sinus bradycardia near 50 beats per minute: 13.4% of its bradycardia negatives are sinus-rhythm
  records below 60 (MLC-01 Amendment 3).
- S-R's held-out leave-one-family-out records for Georgia and CPSC were rescored from the same checkpoints after
  a publication failure, and their first scores were not printed (S-R report, departure 4).
- One uncorrectable memory error was logged at 17:36:20 PST on 10 November; S-R's file and `predict` checks ran
  after it, F's runs started after it, and `select` reproduced every stored validation metric exactly
  (S-R report, departure 9; F report; selection record). The counter still read 1 uncorrected and 0 corrected
  at 01:26 PST on 11 November, after `evaluate` finished.
- S-R and F are not bit-reproducible on the GPU (S-R report, departure 5; F report, Timings).
- Five conditions at fold 10 and two at SPH have fewer than 100 positives (section 4).

## 11. Provenance

Hashes checked by the script against the receipts (`provenance`):

| file | path | pinned SHA-256 | equal |
|---|---|---|---:|
| selection record | `outputs/mlc02_condition_models_20261110/stage3/selection_record.json` | `2dc8810ea94f` | True |
| operating points | `outputs/mlc02_condition_models_20261110/stage3/operating_points.json` | `a2896ab02e5f` | True |
| test-prediction receipt | `outputs/mlc02_condition_models_20261110/stage3/test_predictions_receipt.json` | `c576ad2d2851` | True |
| selection record document | `docs/mlc02-selection-record-2026-11-10.md` | `6473df3ffb80` | True |
| manifest | `outputs/mlc01_condition_dataset_20261110_v4/manifest.csv` | `25d75f8f7e34` | True |
| test prediction files | `stage3/predictions/<id>/<role>__seeds-<...>.npz` | receipt | 33 of 33 |

Stage 3 steps, from their receipts:

| step | commit | modified tracked files | command | started (local time) | wall seconds |
|---|---|---:|---|---|---:|
| select | `a0c0c53` | 0 | `select` | 2026-11-11 00:45:52 PST | 2 |
| fix | `7f4271c` | 0 | `fix` | 2026-11-11 00:46:02 PST | 1 |
| test-predict | `7f4271c` | 0 | `test-predict --device cuda --i-am-the-final-run` | 2026-11-11 00:52:28 PST | 684 |
| evaluate | `7f4271c` | 0 | `evaluate --workers 8` | 2026-11-11 01:04:09 PST | 976 |

- Each command is `PYTHONPATH=<worktree> .venv/bin/python -m scripts.experiments.run_mlc02_stage3 <step>`, with
  the thread settings of the [Stage 3 code report](mlc02-stage3-code-2026-11-10.md); `evaluate` ran with
  `OPENBLAS_NUM_THREADS=1` and numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.1, pandas 3.0.6.
- Resamples: fold 10, 2,198 records in 1,904 patients; SPH evaluation half, 12,698 records in 12,246 patients;
  2,000 resamples each, no redraw, seed 20261110 (fingerprints `66b16ed0b350` and `8536ebde48b0`).
- Full SHA-256: selection record `2dc8810ea94fb336efc7e5ac74bba4e0362f5a4554ba65db26a1f91b23db599a`, operating
  points `a2896ab02e5f40251d959b439226f531c8d69fb7405987af002d128dd2f67f82`, test-prediction receipt
  `c576ad2d2851720bff2e3a73918696c0f56bcf23eedb9803df612f1c3578491e`, manifest
  `25d75f8f7e3449a635c0914cd5cdd8c2134352efa1f9b0e9d723c8d73be8f3a0`.
- The `evaluate` CSVs agree with `evaluation.json` to the precision of the CSV text: 0 of 44 rows of
  `macro.csv`, 0 of 300 of `comparisons.csv` and 0 of 1,232 of `conditions.csv` differ (`csv-agreement`).
- Integrity before new work: KEY-04's SPH macro AUROC, 0.9608446027429329, was reproduced exactly in Stage 1
  (Stage 1 report, KEY-04); `select` reproduced every Stage 2 selection metric from its saved validation file with
  difference 0 (selection record).
- Protocols: MLC-01 `d4e25c7` with Amendments `8720f86`, `ab833a3`, `4a17ca1` and erratum `4b60fbc`; MLC-02
  `2516880` with Addenda `b093b61`, `788db81`, `b8c5719`.
- Outputs: `outputs/mlc02_condition_models_20261110/` (`candidates/`, `lofo/`, `source_identification/`,
  `stage3/`), local and not in Git.
