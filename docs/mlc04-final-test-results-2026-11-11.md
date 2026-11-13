# MLC-04 results: the one final test of the 14-condition classifier on the Challenge test groups

11 November 2026. This report executes the frozen [MLC-04 protocol](mlc04-final-test-protocol-2026-11-11.md)
(commit `fbf26bf`) with its Amendment 1 (`deaefa2`), after the [freeze record](mlc04-freeze-record-2026-11-11.md)
(commit `9faec02`, code `89e2f3a` and `e9df26e`). It was written from the outputs by the agent that ran the
experiment. Every number below is printed from `outputs/mlc04_final_test_20261111/` by

```
PYTHONPATH=. .venv/bin/python -m scripts.reports.mlc04_tables --section <name>
```

and each table names its section. The script computes no metric; it reads, selects and formats stored values.

What this is and is not:

- The test groups come from the same five hospitals as the training data. This is a test on new records of
  known hospitals, not on a new hospital.
- The split is per record and the sources have no patient identifier. Some test patients also appear in
  training (MLC-01 estimated 6-13% for validation), and the intervals resample records, so they are somewhat
  too narrow.
- The labels are ECG-annotation proxies, not diagnoses. No cardiologist has reviewed any label, rule, model
  or result. Clinical validation is pending (validación pendiente).
- Nothing here shows clinical performance or screening value for university students.

## 1. Result

[sections `primary`, `secondary`]

| set | records | metric, entry 100 (95% interval) | metric, entry 20 (95% interval) | sources |
|---|---|---|---|---|
| test groups | 12914 | 0.9602 (0.9542, 0.9657) | 0.9714 (0.9684, 0.9742) | 5 |
| validation (calibration groups) | 12916 | 0.9611 (0.9557, 0.9663) | 0.9701 (0.9662, 0.9736) | 5 |

Like-for-like validation value 0.9610687901468651 (five sources, entry 100, protocol Amendment 1); bound 0.9510687901468651; reading: **confirmed**. Six-source validation value 0.9697697420282113 (entry 20) and its bound 0.9597697420282113, printed without a reading.

- **Primary.** P0-v5 (the independent logistic regressions on pooled xECG, trained on dataset v5) reaches
  0.9602 (0.9542, 0.9657) on the 12,914 test records with v5-rule labels. This is the mean over the five
  sources of the within-source macro AUROC, over the conditions with at least 100 positives and 100 negatives
  in the source.
- **Reading.** Under Amendment 1 the test value is compared with the like-for-like validation value, 0.9611
  (0.9557, 0.9663): the same metric on the validation (calibration) groups. The bound is that value minus
  0.010, 0.95107. The test interval's lower bound, 0.9542, is above it, so **the validation estimate is
  confirmed**. The amendment writes the bound as 0.9511; the lower bound is above both.
- **Six-source value, no reading.** The protocol as first written compared the test value with 0.9698, the
  six-source validation metric with the 20-positive rule. Its bound, 0.9598, lies inside the test interval.
  Amendment 1 replaced that comparison before any test value existed (section 9).
- **Validation entry rule.** With the 20-positive rule of validation the metric is 0.9714 (0.9684, 0.9742) on
  the test groups and 0.9701 (0.9662, 0.9736) on the validation groups.

The three secondary comparisons, paired on the same resamples, Holm-adjusted over the three:

| comparison | difference (95% interval) | null | p | Holm p |
|---|---|---|---|---|
| P0-v5 minus P0-v4, v5-rule labels | 0.0013 (0.0007, 0.0019) | 0.0 | 0.0010 | 0.0030 |
| P0-v5 minus P0-v4, v4 labels | -0.0000 (-0.0009, 0.0008) | 0.0 | 0.9905 |  |
| H-All-v5 minus P0-v5 | 0.0038 (0.0010, 0.0065) | 0.0 | 0.0070 | 0.0070 |
| K6 minus P0-v5 | 0.0010 (-0.0021, 0.0041) | -0.01 | 0.0010 | 0.0030 |

Cleaning: cleaning helps (v5 labels: above; v4-label interval not below zero: True). Larger model: H-All above P0. Key: equivalent.

- **Cleaning.** By the protocol's rule, cleaning helps. P0-v5 minus P0-v4 is +0.0013 (+0.0007, +0.0019) on
  v5-rule labels (Holm p 0.0030). On the v4 labels it is -0.0000 (-0.0009, +0.0008), which is not below zero,
  as the rule requires. Plainly: on the original labels the two models rank the same; the gain is measured
  only on labels cleaned by the rules the v5 model was trained with.
- **Larger model.** H-All-v5 ranks above P0-v5, +0.0038 (+0.0010, +0.0065), Holm p 0.0070.
- **Key.** K6 is equivalent to P0-v5 for ranking within the KEY-05 margin: +0.0010 (-0.0021, +0.0041) against
  -0.010, Holm p 0.0030. The per-source differences below show that this rests on one source.
- **p-values.** 0.0010 is the smallest two-sided bootstrap p-value 2,000 draws can give, and 0.0030 the
  smallest after Holm over three.

Per source [section `comparisons-by-source`]:

| comparison | Ningbo | Chapman | Georgia | CPSC 2018 | CPSC-Extra |
|---|---|---|---|---|---|
| P0-v5 minus P0-v4, v5-rule labels | 0.0007 (0.0003, 0.0010) | 0.0002 (-0.0000, 0.0005) | 0.0010 (0.0000, 0.0018) | 0.0031 (0.0012, 0.0053) | 0.0014 (-0.0005, 0.0034) |
| P0-v5 minus P0-v4, v4 labels | -0.0001 (-0.0005, 0.0003) | 0.0002 (-0.0000, 0.0005) | 0.0008 (-0.0003, 0.0017) | -0.0023 (-0.0060, 0.0011) | 0.0014 (-0.0005, 0.0034) |
| H-All-v5 minus P0-v5 | 0.0016 (0.0008, 0.0025) | 0.0034 (0.0021, 0.0047) | 0.0042 (0.0021, 0.0062) | 0.0079 (0.0041, 0.0116) | 0.0019 (-0.0106, 0.0145) |
| K6 minus P0-v5 | -0.0010 (-0.0020, -0.0001) | -0.0012 (-0.0021, -0.0003) | -0.0056 (-0.0088, -0.0024) | -0.0083 (-0.0130, -0.0034) | 0.0208 (0.0073, 0.0348) |

- **K6 by source.** K6 is below P0-v5, with the interval below zero, in four of the five sources: Ningbo,
  Chapman, Georgia and CPSC 2018. It is above it only at CPSC-Extra, +0.0208 (+0.0073, +0.0348). There, any
  MI is the only condition that enters, and each source weighs one fifth of the metric.
- **Cleaning by source.** On v5-rule labels the largest gain is at CPSC 2018 (+0.0031). On v4 labels the same
  source gives -0.0023 (-0.0060, +0.0011). Rule C9 lowered CPSC 2018's complete-RBBB positives in the test
  group from 341 to 235 [section `support`].
- **H-All by source.** H-All-v5 is above P0-v5 with the interval above zero in every source but CPSC-Extra.

## 2. Every model

Metric (95% interval) with the primary entry rule, the metric with the validation entry rule, and the v4-label
metric of the two P0 models, on the test and validation groups [section `models`]:

| model | test, entry 100 | test, entry 20 | test, v4 labels | validation, entry 100 | validation, entry 20 | validation, v4 labels |
|---|---|---|---|---|---|---|
| P0-v5 | 0.9602 (0.9542, 0.9657) | 0.9714 | 0.9549 (0.9490, 0.9604) | 0.9611 (0.9557, 0.9663) | 0.9701 | 0.9548 (0.9492, 0.9603) |
| P0-v4 | 0.9589 (0.9531, 0.9645) | 0.9708 | 0.9549 (0.9490, 0.9604) | 0.9604 (0.9549, 0.9656) | 0.9692 | 0.9559 (0.9502, 0.9615) |
| H-All-v5 | 0.9640 (0.9582, 0.9692) | 0.9773 |  | 0.9644 (0.9589, 0.9696) | 0.9752 |  |
| H-All-v4 | 0.9660 (0.9604, 0.9711) | 0.9774 |  | 0.9660 (0.9606, 0.9710) | 0.9751 |  |
| K6 | 0.9611 (0.9560, 0.9662) | 0.9678 |  | 0.9588 (0.9532, 0.9640) | 0.9644 |  |

H-All trained on v4 has the highest test value, 0.9660 (0.9604, 0.9711); it was not a secondary comparison.
The validation column comes from the dry run, which scored the calibration groups through the same code.

## 3. By source

Within-source macro AUROC with the primary entry rule [section `sources`]; P0-v5 with its interval:

| set | source | conditions | P0-v5 | P0-v4 | H-All-v5 | H-All-v4 | K6 |
|---|---|---|---|---|---|---|---|
| test | Ningbo | 11 | 0.9895 (0.9882, 0.9907) | 0.9888 | 0.9911 | 0.9912 | 0.9886 |
| test | Chapman | 4 | 0.9879 (0.9853, 0.9903) | 0.9877 | 0.9913 | 0.9912 | 0.9867 |
| test | Georgia | 7 | 0.9654 (0.9611, 0.9695) | 0.9644 | 0.9696 | 0.9684 | 0.9598 |
| test | CPSC 2018 | 5 | 0.9658 (0.9570, 0.9736) | 0.9627 | 0.9738 | 0.9767 | 0.9576 |
| test | CPSC-Extra | 1 | 0.8921 (0.8649, 0.9182) | 0.8907 | 0.8940 | 0.9026 | 0.9129 |
| validation | Ningbo | 11 | 0.9890 (0.9878, 0.9902) | 0.9883 | 0.9916 | 0.9913 | 0.9873 |
| validation | Chapman | 4 | 0.9870 (0.9845, 0.9893) | 0.9869 | 0.9896 | 0.9895 | 0.9864 |
| validation | Georgia | 8 | 0.9679 (0.9632, 0.9723) | 0.9664 | 0.9717 | 0.9711 | 0.9643 |
| validation | CPSC 2018 | 4 | 0.9701 (0.9635, 0.9762) | 0.9698 | 0.9783 | 0.9796 | 0.9612 |
| validation | CPSC-Extra | 1 | 0.8914 (0.8657, 0.9155) | 0.8906 | 0.8911 | 0.8987 | 0.8951 |

Conditions that enter each source's macro on the test groups [section `entering`]:

| source | conditions | entering |
|---|---|---|
| Ningbo | 11 | sinus_tachycardia, sinus_bradycardia, sinus_arrhythmia, af_or_flutter, premature_ventricular_complex, first_degree_av_block, complete_rbbb, left_axis_deviation, lvh, st_depression, t_wave_abnormality |
| Chapman | 4 | sinus_tachycardia, sinus_bradycardia, af_or_flutter, t_wave_abnormality |
| Georgia | 7 | sinus_tachycardia, sinus_bradycardia, af_or_flutter, first_degree_av_block, left_axis_deviation, lvh, t_wave_abnormality |
| CPSC 2018 | 5 | af_or_flutter, premature_ventricular_complex, first_degree_av_block, complete_rbbb, st_depression |
| CPSC-Extra | 1 | mi_any |

- **Entering conditions.** Georgia enters seven conditions on the test groups against eight on validation.
  Complete RBBB has 93 positives there after the rules, below 100. CPSC 2018 enters five against four: PVC has
  116 positives.
- **CPSC-Extra** is the weakest source on both sets: 0.8921 on the test groups, 0.8914 on validation. It
  enters any MI alone.

## 4. Per condition and source (P0-v5)

AUROC, average precision with the prevalence, sensitivity and specificity at the frozen thresholds, with 95%
intervals from record resamples under the validation entry rule. `entry` is 100 when the cell enters the
primary metric, 20 when it enters only the validation rule's resamples (cells below have points only). Cells
without both classes are left out.

Test groups [section `conditions-test`]:

| source | condition | pos / neg | AUROC | AP | prevalence | sensitivity | specificity | entry |
|---|---|---|---|---|---|---|---|---|
| Ningbo | sinus_tachycardia | 1085 / 5828 | 0.999 (0.998, 0.999) | 0.993 (0.990, 0.996) | 0.157 | 1.000 (1.000, 1.000) | 0.913 (0.906, 0.920) | 100 |
| Ningbo | sinus_bradycardia | 2512 / 4403 | 0.999 (0.998, 0.999) | 0.997 (0.996, 0.998) | 0.363 | 0.999 (0.997, 1.000) | 0.895 (0.886, 0.904) | 100 |
| Ningbo | sinus_arrhythmia | 504 / 6440 | 0.987 (0.983, 0.990) | 0.880 (0.854, 0.903) | 0.073 | 0.968 (0.951, 0.983) | 0.910 (0.903, 0.917) | 100 |
| Ningbo | af_or_flutter | 1521 / 5419 | 0.999 (0.999, 0.999) | 0.997 (0.996, 0.998) | 0.219 | 1.000 (1.000, 1.000) | 0.911 (0.903, 0.918) | 100 |
| Ningbo | premature_ventricular_complex | 214 / 6697 | 0.990 (0.985, 0.994) | 0.842 (0.792, 0.885) | 0.031 | 0.981 (0.962, 0.996) | 0.900 (0.893, 0.907) | 100 |
| Ningbo | first_degree_av_block | 166 / 6759 | 0.989 (0.984, 0.993) | 0.753 (0.680, 0.825) | 0.024 | 0.988 (0.969, 1.000) | 0.905 (0.898, 0.912) | 100 |
| Ningbo | complete_rbbb | 260 / 6186 | 0.999 (0.998, 0.999) | 0.978 (0.964, 0.988) | 0.040 | 1.000 (1.000, 1.000) | 0.903 (0.895, 0.911) | 100 |
| Ningbo | incomplete_rbbb | 44 / 6478 | 0.975 (0.966, 0.983) | 0.214 (0.129, 0.336) | 0.007 | 0.909 (0.818, 0.981) | 0.932 (0.926, 0.938) | 20 |
| Ningbo | lafb | 74 / 6638 | 0.987 (0.982, 0.991) | 0.542 (0.433, 0.650) | 0.011 | 1.000 (1.000, 1.000) | 0.921 (0.914, 0.927) | 20 |
| Ningbo | left_axis_deviation | 232 / 6408 | 0.987 (0.984, 0.990) | 0.674 (0.609, 0.744) | 0.035 | 0.991 (0.978, 1.000) | 0.907 (0.899, 0.914) | 100 |
| Ningbo | lvh | 128 / 5127 | 0.999 (0.998, 0.999) | 0.959 (0.936, 0.978) | 0.024 | 1.000 (1.000, 1.000) | 0.941 (0.934, 0.947) | 100 |
| Ningbo | st_depression | 236 / 5776 | 0.972 (0.961, 0.980) | 0.713 (0.650, 0.771) | 0.039 | 0.898 (0.858, 0.933) | 0.916 (0.909, 0.923) | 100 |
| Ningbo | t_wave_abnormality | 1280 / 4765 | 0.966 (0.962, 0.971) | 0.904 (0.890, 0.916) | 0.212 | 0.867 (0.849, 0.886) | 0.922 (0.914, 0.929) | 100 |
| Ningbo | mi_any | 37 / 6422 | 0.962 (0.911, 0.996) | 0.684 (0.532, 0.822) | 0.006 | 0.919 (0.816, 1.000) | 0.941 (0.935, 0.947) | 20 |
| Chapman | sinus_tachycardia | 326 / 1704 | 0.999 (0.998, 1.000) | 0.994 (0.989, 0.997) | 0.161 | 1.000 (1.000, 1.000) | 0.874 (0.859, 0.890) | 100 |
| Chapman | sinus_bradycardia | 795 / 1233 | 0.999 (0.999, 1.000) | 0.999 (0.998, 1.000) | 0.392 | 1.000 (1.000, 1.000) | 0.942 (0.929, 0.954) | 100 |
| Chapman | af_or_flutter | 420 / 1610 | 0.998 (0.997, 0.999) | 0.994 (0.991, 0.997) | 0.207 | 1.000 (1.000, 1.000) | 0.883 (0.867, 0.899) | 100 |
| Chapman | premature_ventricular_complex | 63 / 1967 | 0.983 (0.970, 0.993) | 0.803 (0.701, 0.889) | 0.031 | 0.937 (0.871, 0.986) | 0.906 (0.894, 0.919) | 20 |
| Chapman | first_degree_av_block | 47 / 1957 | 0.992 (0.984, 0.997) | 0.847 (0.756, 0.920) | 0.023 | 0.979 (0.925, 1.000) | 0.914 (0.901, 0.926) | 20 |
| Chapman | complete_rbbb | 83 / 1938 | 0.999 (0.998, 1.000) | 0.984 (0.968, 0.995) | 0.041 | 1.000 (1.000, 1.000) | 0.901 (0.888, 0.914) | 20 |
| Chapman | left_axis_deviation | 76 / 1952 | 0.986 (0.980, 0.991) | 0.653 (0.540, 0.764) | 0.037 | 0.987 (0.956, 1.000) | 0.901 (0.888, 0.915) | 20 |
| Chapman | lvh | 4 / 1587 | 0.995 | 0.715 | 0.003 | 1.000 | 0.928 |  |
| Chapman | st_depression | 81 / 1780 | 0.968 (0.955, 0.981) | 0.629 (0.517, 0.750) | 0.044 | 0.914 (0.849, 0.973) | 0.879 (0.863, 0.894) | 20 |
| Chapman | t_wave_abnormality | 409 / 1455 | 0.955 (0.945, 0.964) | 0.856 (0.819, 0.890) | 0.219 | 0.900 (0.870, 0.928) | 0.896 (0.880, 0.911) | 100 |
| Chapman | mi_any | 8 / 1981 | 0.920 | 0.417 | 0.004 | 0.750 | 0.928 |  |
| Georgia | sinus_tachycardia | 236 / 1791 | 0.995 (0.993, 0.998) | 0.967 (0.948, 0.983) | 0.116 | 0.996 (0.987, 1.000) | 0.864 (0.848, 0.880) | 100 |
| Georgia | sinus_bradycardia | 372 / 1662 | 0.998 (0.997, 0.999) | 0.990 (0.983, 0.995) | 0.183 | 1.000 (1.000, 1.000) | 0.863 (0.847, 0.880) | 100 |
| Georgia | sinus_arrhythmia | 91 / 1948 | 0.957 (0.940, 0.970) | 0.560 (0.454, 0.667) | 0.045 | 0.879 (0.802, 0.944) | 0.887 (0.872, 0.901) | 20 |
| Georgia | af_or_flutter | 147 / 1885 | 0.975 (0.957, 0.991) | 0.927 (0.889, 0.959) | 0.072 | 0.959 (0.924, 0.987) | 0.855 (0.839, 0.870) | 100 |
| Georgia | premature_ventricular_complex | 88 / 1951 | 0.870 (0.828, 0.910) | 0.536 (0.437, 0.639) | 0.043 | 0.716 (0.624, 0.807) | 0.897 (0.884, 0.911) | 20 |
| Georgia | first_degree_av_block | 161 / 1861 | 0.981 (0.972, 0.987) | 0.802 (0.732, 0.873) | 0.080 | 0.981 (0.957, 1.000) | 0.847 (0.829, 0.863) | 100 |
| Georgia | complete_rbbb | 93 / 1894 | 0.994 (0.991, 0.997) | 0.873 (0.802, 0.929) | 0.047 | 1.000 (1.000, 1.000) | 0.902 (0.888, 0.915) | 20 |
| Georgia | incomplete_rbbb | 66 / 1936 | 0.947 (0.919, 0.969) | 0.495 (0.384, 0.637) | 0.033 | 0.924 (0.853, 0.983) | 0.846 (0.830, 0.862) | 20 |
| Georgia | lafb | 33 / 1992 | 0.971 (0.961, 0.981) | 0.298 (0.192, 0.457) | 0.016 | 1.000 (1.000, 1.000) | 0.840 (0.824, 0.857) | 20 |
| Georgia | left_axis_deviation | 185 / 1807 | 0.973 (0.966, 0.979) | 0.715 (0.646, 0.796) | 0.093 | 0.989 (0.973, 1.000) | 0.836 (0.819, 0.853) | 100 |
| Georgia | lvh | 223 / 1682 | 0.942 (0.928, 0.956) | 0.706 (0.643, 0.766) | 0.117 | 0.910 (0.873, 0.947) | 0.840 (0.823, 0.857) | 100 |
| Georgia | t_wave_abnormality | 543 / 1083 | 0.893 (0.878, 0.909) | 0.810 (0.777, 0.841) | 0.334 | 0.823 (0.792, 0.853) | 0.821 (0.798, 0.843) | 100 |
| CPSC 2018 | af_or_flutter | 196 / 1053 | 0.991 (0.982, 0.998) | 0.979 (0.964, 0.991) | 0.157 | 0.980 (0.958, 0.995) | 0.914 (0.897, 0.931) | 100 |
| CPSC 2018 | premature_ventricular_complex | 116 / 1123 | 0.937 (0.901, 0.967) | 0.830 (0.764, 0.891) | 0.094 | 0.914 (0.862, 0.962) | 0.858 (0.837, 0.877) | 100 |
| CPSC 2018 | first_degree_av_block | 137 / 1115 | 0.987 (0.981, 0.992) | 0.908 (0.866, 0.943) | 0.109 | 0.964 (0.928, 0.992) | 0.912 (0.895, 0.929) | 100 |
| CPSC 2018 | complete_rbbb | 235 / 911 | 0.992 (0.987, 0.996) | 0.949 (0.907, 0.984) | 0.205 | 0.991 (0.978, 1.000) | 0.884 (0.862, 0.905) | 100 |
| CPSC 2018 | st_depression | 144 / 1109 | 0.922 (0.902, 0.940) | 0.551 (0.474, 0.639) | 0.115 | 0.910 (0.861, 0.954) | 0.822 (0.800, 0.845) | 100 |
| CPSC-Extra | sinus_tachycardia | 48 / 589 | 0.996 (0.993, 0.999) | 0.953 (0.903, 0.987) | 0.075 | 1.000 (1.000, 1.000) | 0.881 (0.856, 0.906) | 20 |
| CPSC-Extra | sinus_bradycardia | 6 / 574 | 0.997 | 0.815 | 0.010 | 1.000 | 0.866 |  |
| CPSC-Extra | sinus_arrhythmia | 3 / 644 | 0.808 | 0.040 | 0.005 | 0.667 | 0.894 |  |
| CPSC-Extra | af_or_flutter | 47 / 600 | 0.996 (0.990, 0.999) | 0.936 (0.861, 0.992) | 0.073 | 1.000 (1.000, 1.000) | 0.908 (0.885, 0.930) | 20 |
| CPSC-Extra | premature_ventricular_complex | 35 / 607 | 0.915 (0.866, 0.955) | 0.605 (0.435, 0.763) | 0.055 | 0.686 (0.520, 0.839) | 0.891 (0.866, 0.914) | 20 |
| CPSC-Extra | first_degree_av_block | 21 / 624 | 0.990 (0.981, 0.997) | 0.803 (0.628, 0.930) | 0.033 | 1.000 (1.000, 1.000) | 0.907 (0.884, 0.929) | 20 |
| CPSC-Extra | complete_rbbb | 9 / 634 | 0.999 | 0.898 | 0.014 | 1.000 | 0.869 |  |
| CPSC-Extra | incomplete_rbbb | 4 / 639 | 0.875 | 0.060 | 0.006 | 0.500 | 0.937 |  |
| CPSC-Extra | lvh | 29 / 557 | 0.972 (0.941, 0.992) | 0.763 (0.619, 0.891) | 0.049 | 0.966 (0.880, 1.000) | 0.835 (0.803, 0.865) | 20 |
| CPSC-Extra | st_depression | 10 / 241 | 0.802 | 0.170 | 0.040 | 0.500 | 0.838 |  |
| CPSC-Extra | t_wave_abnormality | 6 / 242 | 0.719 | 0.234 | 0.024 | 0.667 | 0.674 |  |
| CPSC-Extra | mi_any | 284 / 363 | 0.892 (0.865, 0.918) | 0.884 (0.851, 0.915) | 0.439 | 0.873 (0.835, 0.909) | 0.763 (0.717, 0.806) | 100 |

Validation groups, the same code on the dry run's scores [section `conditions-validation`]:

| source | condition | pos / neg | AUROC | AP | prevalence | sensitivity | specificity | entry |
|---|---|---|---|---|---|---|---|---|
| Ningbo | sinus_tachycardia | 1109 / 5804 | 0.999 (0.999, 0.999) | 0.995 (0.993, 0.997) | 0.160 | 1.000 (1.000, 1.000) | 0.921 (0.914, 0.928) | 100 |
| Ningbo | sinus_bradycardia | 2480 / 4439 | 0.998 (0.998, 0.999) | 0.997 (0.994, 0.998) | 0.358 | 1.000 (0.999, 1.000) | 0.894 (0.885, 0.902) | 100 |
| Ningbo | sinus_arrhythmia | 526 / 6420 | 0.984 (0.980, 0.987) | 0.864 (0.838, 0.889) | 0.076 | 0.964 (0.948, 0.979) | 0.909 (0.901, 0.916) | 100 |
| Ningbo | af_or_flutter | 1530 / 5416 | 0.999 (0.998, 0.999) | 0.995 (0.994, 0.997) | 0.220 | 1.000 (1.000, 1.000) | 0.912 (0.904, 0.919) | 100 |
| Ningbo | premature_ventricular_complex | 217 / 6695 | 0.991 (0.987, 0.994) | 0.824 (0.765, 0.876) | 0.031 | 0.986 (0.969, 1.000) | 0.906 (0.899, 0.913) | 100 |
| Ningbo | first_degree_av_block | 188 / 6742 | 0.987 (0.982, 0.991) | 0.765 (0.706, 0.824) | 0.027 | 0.984 (0.965, 1.000) | 0.905 (0.898, 0.912) | 100 |
| Ningbo | complete_rbbb | 249 / 6225 | 0.999 (0.997, 1.000) | 0.978 (0.955, 0.993) | 0.038 | 0.996 (0.987, 1.000) | 0.906 (0.899, 0.913) | 100 |
| Ningbo | incomplete_rbbb | 55 / 6472 | 0.975 (0.966, 0.983) | 0.297 (0.193, 0.410) | 0.008 | 0.927 (0.851, 0.984) | 0.936 (0.930, 0.942) | 20 |
| Ningbo | lafb | 74 / 6625 | 0.989 (0.985, 0.993) | 0.549 (0.440, 0.669) | 0.011 | 0.973 (0.933, 1.000) | 0.929 (0.923, 0.935) | 20 |
| Ningbo | left_axis_deviation | 214 / 6411 | 0.990 (0.987, 0.992) | 0.726 (0.661, 0.794) | 0.032 | 0.995 (0.985, 1.000) | 0.909 (0.902, 0.916) | 100 |
| Ningbo | lvh | 101 / 5242 | 0.997 (0.994, 0.999) | 0.922 (0.877, 0.958) | 0.019 | 0.990 (0.967, 1.000) | 0.934 (0.927, 0.940) | 100 |
| Ningbo | st_depression | 222 / 5736 | 0.970 (0.961, 0.978) | 0.698 (0.636, 0.753) | 0.037 | 0.896 (0.855, 0.932) | 0.918 (0.910, 0.925) | 100 |
| Ningbo | t_wave_abnormality | 1246 / 4748 | 0.965 (0.960, 0.970) | 0.899 (0.885, 0.912) | 0.208 | 0.857 (0.837, 0.876) | 0.927 (0.920, 0.934) | 100 |
| Ningbo | mi_any | 26 / 6445 | 0.973 (0.945, 0.995) | 0.422 (0.247, 0.629) | 0.004 | 0.885 (0.750, 1.000) | 0.937 (0.931, 0.942) | 20 |
| Chapman | sinus_tachycardia | 313 / 1722 | 0.999 (0.998, 0.999) | 0.992 (0.986, 0.997) | 0.154 | 1.000 (1.000, 1.000) | 0.861 (0.846, 0.877) | 100 |
| Chapman | sinus_bradycardia | 764 / 1267 | 1.000 (0.999, 1.000) | 0.999 (0.999, 1.000) | 0.376 | 1.000 (1.000, 1.000) | 0.952 (0.939, 0.963) | 100 |
| Chapman | af_or_flutter | 454 / 1582 | 0.999 (0.998, 0.999) | 0.996 (0.994, 0.998) | 0.223 | 1.000 (1.000, 1.000) | 0.872 (0.856, 0.889) | 100 |
| Chapman | premature_ventricular_complex | 64 / 1972 | 0.987 (0.969, 0.997) | 0.870 (0.787, 0.936) | 0.031 | 0.969 (0.919, 1.000) | 0.894 (0.880, 0.908) | 20 |
| Chapman | first_degree_av_block | 50 / 1948 | 0.987 (0.974, 0.995) | 0.810 (0.709, 0.891) | 0.025 | 0.980 (0.931, 1.000) | 0.926 (0.914, 0.937) | 20 |
| Chapman | complete_rbbb | 71 / 1943 | 0.998 (0.997, 1.000) | 0.954 (0.898, 0.990) | 0.035 | 1.000 (1.000, 1.000) | 0.870 (0.856, 0.884) | 20 |
| Chapman | left_axis_deviation | 80 / 1956 | 0.981 (0.975, 0.987) | 0.579 (0.472, 0.701) | 0.039 | 0.988 (0.957, 1.000) | 0.894 (0.880, 0.907) | 20 |
| Chapman | lvh | 4 / 1654 | 0.989 | 0.426 | 0.002 | 1.000 | 0.929 |  |
| Chapman | st_depression | 78 / 1807 | 0.964 (0.939, 0.981) | 0.646 (0.538, 0.752) | 0.041 | 0.949 (0.897, 0.989) | 0.899 (0.885, 0.913) | 20 |
| Chapman | t_wave_abnormality | 390 / 1486 | 0.951 (0.941, 0.960) | 0.827 (0.788, 0.866) | 0.208 | 0.849 (0.813, 0.882) | 0.900 (0.885, 0.915) | 100 |
| Chapman | mi_any | 6 / 1980 | 0.985 | 0.213 | 0.003 | 1.000 | 0.936 |  |
| Georgia | sinus_tachycardia | 258 / 1773 | 0.997 (0.995, 0.998) | 0.975 (0.954, 0.990) | 0.127 | 0.996 (0.988, 1.000) | 0.867 (0.850, 0.882) | 100 |
| Georgia | sinus_bradycardia | 312 / 1719 | 0.997 (0.996, 0.999) | 0.984 (0.970, 0.993) | 0.154 | 1.000 (1.000, 1.000) | 0.863 (0.846, 0.878) | 100 |
| Georgia | sinus_arrhythmia | 92 / 1948 | 0.975 (0.963, 0.984) | 0.626 (0.530, 0.740) | 0.045 | 0.946 (0.892, 0.988) | 0.897 (0.884, 0.910) | 20 |
| Georgia | af_or_flutter | 124 / 1908 | 0.955 (0.927, 0.978) | 0.862 (0.807, 0.915) | 0.061 | 0.911 (0.858, 0.959) | 0.853 (0.836, 0.868) | 100 |
| Georgia | premature_ventricular_complex | 66 / 1975 | 0.869 (0.812, 0.919) | 0.528 (0.405, 0.645) | 0.032 | 0.727 (0.620, 0.831) | 0.893 (0.880, 0.907) | 20 |
| Georgia | first_degree_av_block | 145 / 1878 | 0.979 (0.971, 0.986) | 0.762 (0.682, 0.837) | 0.072 | 0.966 (0.933, 0.993) | 0.860 (0.844, 0.875) | 100 |
| Georgia | complete_rbbb | 104 / 1888 | 0.993 (0.990, 0.996) | 0.862 (0.788, 0.920) | 0.052 | 1.000 (1.000, 1.000) | 0.896 (0.882, 0.909) | 100 |
| Georgia | incomplete_rbbb | 74 / 1935 | 0.950 (0.935, 0.964) | 0.405 (0.302, 0.531) | 0.037 | 0.919 (0.849, 0.976) | 0.851 (0.834, 0.866) | 20 |
| Georgia | lafb | 44 / 1975 | 0.964 (0.949, 0.977) | 0.357 (0.239, 0.509) | 0.022 | 0.955 (0.885, 1.000) | 0.839 (0.823, 0.856) | 20 |
| Georgia | left_axis_deviation | 187 / 1788 | 0.976 (0.968, 0.983) | 0.770 (0.705, 0.841) | 0.095 | 0.995 (0.983, 1.000) | 0.846 (0.830, 0.862) | 100 |
| Georgia | lvh | 230 / 1674 | 0.950 (0.939, 0.960) | 0.745 (0.688, 0.797) | 0.121 | 0.900 (0.860, 0.936) | 0.843 (0.825, 0.860) | 100 |
| Georgia | t_wave_abnormality | 539 / 1094 | 0.896 (0.878, 0.912) | 0.818 (0.785, 0.850) | 0.330 | 0.814 (0.781, 0.846) | 0.829 (0.806, 0.851) | 100 |
| CPSC 2018 | af_or_flutter | 210 / 1032 | 0.994 (0.990, 0.998) | 0.982 (0.971, 0.991) | 0.169 | 0.986 (0.967, 1.000) | 0.911 (0.893, 0.928) | 100 |
| CPSC 2018 | premature_ventricular_complex | 99 / 1133 | 0.957 (0.923, 0.982) | 0.853 (0.788, 0.907) | 0.080 | 0.939 (0.889, 0.981) | 0.867 (0.845, 0.886) | 20 |
| CPSC 2018 | first_degree_av_block | 112 / 1131 | 0.977 (0.959, 0.990) | 0.857 (0.790, 0.917) | 0.090 | 0.946 (0.900, 0.984) | 0.920 (0.903, 0.935) | 100 |
| CPSC 2018 | complete_rbbb | 224 / 910 | 0.995 (0.992, 0.998) | 0.981 (0.966, 0.991) | 0.198 | 0.996 (0.986, 1.000) | 0.902 (0.882, 0.921) | 100 |
| CPSC 2018 | st_depression | 161 / 1082 | 0.913 (0.893, 0.932) | 0.544 (0.471, 0.632) | 0.130 | 0.851 (0.795, 0.904) | 0.832 (0.809, 0.854) | 100 |
| CPSC-Extra | sinus_tachycardia | 63 / 578 | 0.997 (0.994, 0.999) | 0.974 (0.946, 0.994) | 0.098 | 1.000 (1.000, 1.000) | 0.843 (0.815, 0.872) | 20 |
| CPSC-Extra | sinus_bradycardia | 6 / 577 | 0.990 | 0.533 | 0.010 | 1.000 | 0.896 |  |
| CPSC-Extra | sinus_arrhythmia | 2 / 648 | 0.545 | 0.007 | 0.003 | 0.000 | 0.877 |  |
| CPSC-Extra | af_or_flutter | 46 / 602 | 0.995 (0.990, 0.998) | 0.923 (0.846, 0.980) | 0.071 | 1.000 (1.000, 1.000) | 0.910 (0.886, 0.932) | 20 |
| CPSC-Extra | premature_ventricular_complex | 25 / 623 | 0.877 (0.784, 0.951) | 0.318 (0.182, 0.535) | 0.039 | 0.760 (0.583, 0.926) | 0.875 (0.849, 0.898) | 20 |
| CPSC-Extra | first_degree_av_block | 20 / 627 | 0.950 (0.905, 0.984) | 0.490 (0.295, 0.733) | 0.031 | 0.850 (0.667, 1.000) | 0.872 (0.846, 0.898) | 20 |
| CPSC-Extra | complete_rbbb | 20 / 618 | 0.994 (0.981, 1.000) | 0.942 (0.846, 1.000) | 0.031 | 1.000 (1.000, 1.000) | 0.859 (0.831, 0.886) | 20 |
| CPSC-Extra | incomplete_rbbb | 5 / 632 | 0.957 | 0.382 | 0.008 | 1.000 | 0.913 |  |
| CPSC-Extra | lvh | 32 / 550 | 0.965 (0.934, 0.986) | 0.691 (0.517, 0.834) | 0.055 | 0.938 (0.844, 1.000) | 0.855 (0.824, 0.883) | 20 |
| CPSC-Extra | st_depression | 8 / 221 | 0.907 | 0.238 | 0.035 | 0.750 | 0.837 |  |
| CPSC-Extra | t_wave_abnormality | 5 / 224 | 0.814 | 0.090 | 0.022 | 0.800 | 0.652 |  |
| CPSC-Extra | mi_any | 272 / 378 | 0.891 (0.866, 0.916) | 0.852 (0.812, 0.890) | 0.418 | 0.871 (0.828, 0.911) | 0.757 (0.714, 0.799) | 100 |

The cells with the lowest AUROC that enter the primary metric are:

- CPSC-Extra any MI: 0.892 (0.865, 0.918);
- Georgia T-wave abnormality: 0.893 (0.878, 0.909);
- CPSC 2018 ST depression: 0.922 (0.902, 0.940);
- CPSC 2018 PVC: 0.937 (0.901, 0.967).

Among the cells that enter only the validation rule, Georgia PVC has 0.870 (0.828, 0.910), with sensitivity
0.716 at the frozen threshold.

The thresholds were set for 90% specificity on all validation records pooled over six sources (PTB-XL
included). Within a test source the specificity they give ranges, among the cells that enter the primary
metric:

- lowest: CPSC-Extra any MI 0.763 (0.717, 0.806), Georgia T-wave abnormality 0.821 and CPSC 2018 ST depression
  0.822;
- highest: Chapman sinus bradycardia 0.942 and Ningbo LVH 0.941.

## 5. Thresholds

The frozen thresholds of P0-v5 and their pooled validation rates [section `thresholds`]:

| condition | threshold | validation sensitivity | validation specificity | positives / negatives |
|---|---|---|---|---|
| sinus_tachycardia | 0.001413 | 0.999 | 0.9000 | 1824 / 11975 |
| sinus_bradycardia | 0.002443 | 1.000 | 0.9001 | 3626 / 9849 |
| sinus_arrhythmia | 0.037501 | 0.948 | 0.9000 | 697 / 11120 |
| af_or_flutter | 0.016438 | 0.993 | 0.9000 | 2520 / 12565 |
| premature_ventricular_complex | 0.031061 | 0.938 | 0.9001 | 585 / 14459 |
| first_degree_av_block | 0.021130 | 0.963 | 0.9000 | 619 / 14403 |
| complete_rbbb | 0.008109 | 0.997 | 0.9000 | 722 / 13682 |
| incomplete_rbbb | 0.033738 | 0.938 | 0.9000 | 243 / 11082 |
| lafb | 0.009648 | 0.985 | 0.9000 | 275 / 10591 |
| left_axis_deviation | 0.021853 | 0.995 | 0.9001 | 602 / 10899 |
| lvh | 0.045326 | 0.939 | 0.9000 | 577 / 10921 |
| st_depression | 0.070650 | 0.880 | 0.9001 | 569 / 10589 |
| t_wave_abnormality | 0.269694 | 0.832 | 0.9001 | 2440 / 9127 |
| mi_any | 0.103254 | 0.881 | 0.9000 | 824 / 10412 |

Records: 15097. Rule: mlc02_evaluation.specificity_thresholds: lowest observed score with specificity >= 0.90 on all v5 validation records where the condition is defined, pooled over the six sources.

## 6. Calibration (P0-v5)

Brier score per source and condition with its interval on the test groups, beside validation. Also the mean
and the largest gap between mean probability and observed rate in a 10-bin reliability table; the largest gap
counts bins of at least 20 records. The bins are in `challenge_test/score/reliability.csv`
[section `calibration`]:

| source | condition | Brier test | Brier validation | mean gap test | largest gap test | mean gap validation |
|---|---|---|---|---|---|---|
| Ningbo | sinus_tachycardia | 0.0088 (0.0073, 0.0103) | 0.0077 | 0.004 | 0.176 | 0.005 |
| Ningbo | sinus_bradycardia | 0.0111 (0.0093, 0.0130) | 0.0111 | 0.006 | 0.147 | 0.005 |
| Ningbo | sinus_arrhythmia | 0.0212 (0.0188, 0.0237) | 0.0241 | 0.007 | 0.141 | 0.007 |
| Ningbo | af_or_flutter | 0.0103 (0.0087, 0.0118) | 0.0128 | 0.013 | 0.283 | 0.014 |
| Ningbo | premature_ventricular_complex | 0.0114 (0.0095, 0.0134) | 0.0119 | 0.013 | 0.391 | 0.012 |
| Ningbo | first_degree_av_block | 0.0101 (0.0085, 0.0119) | 0.0116 | 0.004 | 0.084 | 0.004 |
| Ningbo | complete_rbbb | 0.0043 (0.0033, 0.0054) | 0.0034 | 0.006 | 0.103 | 0.006 |
| Ningbo | incomplete_rbbb | 0.0069 (0.0055, 0.0085) | 0.0074 | 0.007 | 0.168 | 0.005 |
| Ningbo | lafb | 0.0073 (0.0059, 0.0089) | 0.0071 | 0.003 | 0.092 | 0.004 |
| Ningbo | left_axis_deviation | 0.0163 (0.0141, 0.0184) | 0.0142 | 0.005 | 0.202 | 0.005 |
| Ningbo | lvh | 0.0045 (0.0036, 0.0057) | 0.0049 | 0.011 | 0.170 | 0.010 |
| Ningbo | st_depression | 0.0187 (0.0163, 0.0213) | 0.0187 | 0.009 | 0.122 | 0.008 |
| Ningbo | t_wave_abnormality | 0.0568 (0.0530, 0.0608) | 0.0571 | 0.020 | 0.111 | 0.024 |
| Ningbo | mi_any | 0.0065 (0.0054, 0.0076) | 0.0070 | 0.023 | 0.412 | 0.024 |
| Chapman | sinus_tachycardia | 0.0088 (0.0061, 0.0120) | 0.0090 | 0.004 | 0.051 | 0.007 |
| Chapman | sinus_bradycardia | 0.0057 (0.0036, 0.0081) | 0.0056 | 0.006 | 0.086 | 0.010 |
| Chapman | af_or_flutter | 0.0127 (0.0097, 0.0161) | 0.0109 | 0.013 | 0.106 | 0.016 |
| Chapman | premature_ventricular_complex | 0.0124 (0.0086, 0.0165) | 0.0111 | 0.011 | 0.134 | 0.014 |
| Chapman | first_degree_av_block | 0.0083 (0.0058, 0.0113) | 0.0099 | 0.006 | 0.082 | 0.006 |
| Chapman | complete_rbbb | 0.0052 (0.0031, 0.0075) | 0.0052 | 0.008 | 0.144 | 0.007 |
| Chapman | left_axis_deviation | 0.0187 (0.0147, 0.0232) | 0.0221 | 0.013 | 0.290 | 0.010 |
| Chapman | lvh | 0.0042 | 0.0053 | 0.014 | 0.106 | 0.015 |
| Chapman | st_depression | 0.0231 (0.0183, 0.0279) | 0.0226 | 0.015 | 0.155 | 0.013 |
| Chapman | t_wave_abnormality | 0.0673 (0.0597, 0.0753) | 0.0723 | 0.030 | 0.180 | 0.026 |
| Chapman | mi_any | 0.0086 | 0.0079 | 0.027 | 0.207 | 0.027 |
| Georgia | sinus_tachycardia | 0.0154 (0.0115, 0.0193) | 0.0128 | 0.005 | 0.044 | 0.008 |
| Georgia | sinus_bradycardia | 0.0130 (0.0097, 0.0166) | 0.0138 | 0.011 | 0.117 | 0.009 |
| Georgia | sinus_arrhythmia | 0.0275 (0.0221, 0.0335) | 0.0241 | 0.010 | 0.156 | 0.013 |
| Georgia | af_or_flutter | 0.0135 (0.0094, 0.0177) | 0.0169 | 0.006 | 0.021 | 0.008 |
| Georgia | premature_ventricular_complex | 0.0285 (0.0220, 0.0353) | 0.0219 | 0.016 | 0.298 | 0.010 |
| Georgia | first_degree_av_block | 0.0288 (0.0236, 0.0341) | 0.0276 | 0.011 | 0.263 | 0.013 |
| Georgia | complete_rbbb | 0.0127 (0.0091, 0.0165) | 0.0149 | 0.007 | 0.064 | 0.007 |
| Georgia | incomplete_rbbb | 0.0216 (0.0170, 0.0263) | 0.0268 | 0.008 | 0.123 | 0.008 |
| Georgia | lafb | 0.0183 (0.0146, 0.0222) | 0.0227 | 0.019 | 0.153 | 0.021 |
| Georgia | left_axis_deviation | 0.0393 (0.0326, 0.0453) | 0.0352 | 0.017 | 0.209 | 0.010 |
| Georgia | lvh | 0.0594 (0.0511, 0.0677) | 0.0578 | 0.032 | 0.144 | 0.034 |
| Georgia | t_wave_abnormality | 0.1226 (0.1119, 0.1338) | 0.1183 | 0.045 | 0.122 | 0.032 |
| CPSC 2018 | af_or_flutter | 0.0143 (0.0092, 0.0206) | 0.0157 | 0.007 | 0.023 | 0.010 |
| CPSC 2018 | premature_ventricular_complex | 0.0308 (0.0231, 0.0392) | 0.0267 | 0.017 | 0.085 | 0.013 |
| CPSC 2018 | first_degree_av_block | 0.0310 (0.0236, 0.0384) | 0.0291 | 0.026 | 0.230 | 0.022 |
| CPSC 2018 | complete_rbbb | 0.0288 (0.0206, 0.0376) | 0.0236 | 0.025 | 0.070 | 0.021 |
| CPSC 2018 | st_depression | 0.0697 (0.0592, 0.0800) | 0.0797 | 0.032 | 0.159 | 0.048 |
| CPSC-Extra | sinus_tachycardia | 0.0131 (0.0068, 0.0205) | 0.0148 | 0.008 | 0.011 | 0.017 |
| CPSC-Extra | sinus_bradycardia | 0.0088 | 0.0090 | 0.015 | 0.003 | 0.008 |
| CPSC-Extra | sinus_arrhythmia | 0.0131 | 0.0181 | 0.022 | 0.006 | 0.031 |
| CPSC-Extra | af_or_flutter | 0.0104 (0.0042, 0.0183) | 0.0125 | 0.013 | 0.066 | 0.013 |
| CPSC-Extra | premature_ventricular_complex | 0.0317 (0.0202, 0.0440) | 0.0354 | 0.019 | 0.010 | 0.023 |
| CPSC-Extra | first_degree_av_block | 0.0134 (0.0075, 0.0201) | 0.0214 | 0.008 | 0.002 | 0.019 |
| CPSC-Extra | complete_rbbb | 0.0036 | 0.0061 | 0.009 | 0.004 | 0.011 |
| CPSC-Extra | incomplete_rbbb | 0.0080 | 0.0080 | 0.006 | 0.001 | 0.012 |
| CPSC-Extra | lvh | 0.0255 (0.0170, 0.0351) | 0.0302 | 0.030 | 0.139 | 0.021 |
| CPSC-Extra | st_depression | 0.0481 | 0.0447 | 0.033 | 0.005 | 0.045 |
| CPSC-Extra | t_wave_abnormality | 0.1244 | 0.1536 | 0.203 | 0.306 | 0.239 |
| CPSC-Extra | mi_any | 0.1545 (0.1339, 0.1761) | 0.1550 | 0.141 | 0.334 | 0.127 |

- **Brier scores.** On the test groups they are close to validation's in most cells.
- **Largest Brier scores.** They are at CPSC-Extra any MI (0.1545 test, 0.1550 validation) and in the T-wave
  cells of Georgia and CPSC-Extra.
- **Largest reliability gaps.** Ningbo any MI (0.412) and Ningbo PVC (0.391), in bins of at least 20 records.

## 7. Rules C1-C12, records and support

Each rule on the test groups beside validation [section `rules`]:

- for C1-C4, records selected / excluded;
- for C5-C12, labels selected / made undefined (positives +, negatives -), counted before overlaps / after
  earlier rules.

| rule | test | validation |
|---|---|---|
| C1 | 16 / 16 | 21 / 21 |
| C2 | 10 / 10 | 5 / 5 |
| C3 | 25 / 11 | 27 / 11 |
| C4 | 8 / 1 | 11 / 2 |
| C5 | 5+ 747- / 4+ 722- | 1+ 720- / 1+ 706- |
| C6 | 46+ 0- / 46+ 0- | 50+ 0- / 50+ 0- |
| C7 | 100+ 1774- / 100+ 1754- | 88+ 1900- / 88+ 1877- |
| C8 | 23+ 0- / 23+ 0- | 16+ 0- / 15+ 0- |
| C9 | 110+ 0- / 110+ 0- | 124+ 0- / 124+ 0- |
| C10 | 0+ 13- / 0+ 13- | 0+ 18- / 0+ 18- |
| C11 | 36+ 30- / 36+ 26- | 37+ 26- / 36+ 23- |
| C12 | 0+ 701- / 0+ 643- | 0+ 646- / 0+ 579- |

Records from the split table to dataset v5 [section `records`]:

| set | source | records | short | other exclusion | dropped copy | dropped conflict | included v4 | excluded C1-C4 | included v5 |
|---|---|---|---|---|---|---|---|---|---|
| test | Ningbo | 6983 | 0 | 0 | 0 | 11 | 6972 | 27 | 6945 |
| test | Chapman | 2050 | 0 | 0 | 9 | 11 | 2030 | 0 | 2030 |
| test | Georgia | 2071 | 9 | 0 | 21 | 0 | 2041 | 2 | 2039 |
| test | CPSC 2018 | 1381 | 0 | 0 | 100 | 19 | 1262 | 9 | 1253 |
| test | CPSC-Extra | 687 | 2 | 0 | 13 | 25 | 647 | 0 | 647 |
| validation | Ningbo | 6979 | 0 | 0 | 2 | 5 | 6972 | 26 | 6946 |
| validation | Chapman | 2050 | 0 | 0 | 9 | 5 | 2036 | 0 | 2036 |
| validation | Georgia | 2066 | 7 | 0 | 16 | 0 | 2043 | 2 | 2041 |
| validation | CPSC 2018 | 1375 | 2 | 0 | 100 | 22 | 1251 | 8 | 1243 |
| validation | CPSC-Extra | 692 | 3 | 0 | 8 | 28 | 653 | 3 | 650 |

- **Flow.** The 13,172 test records reduce to 12,952 included in v4 terms and 12,914 in v5. Positives,
  negatives and undefined labels of every source and condition, v5 beside v4, are in section `support`
  (`support_v5.csv`, `support_v4.csv`).
- **Rule counts.** They are of the same order on the test groups as on validation.
- **Hash check.** No test record shares its first-10-s hash with a v4 train or validation record (scan receipt,
  0 in every source).

## 8. Integrity and freeze

Every step's receipt, with its commit, modified tracked files and whether its outputs still have the recorded
SHA-256 [section `integrity`]:

| receipt | commit | modified tracked | wall s | outputs equal |
|---|---|---|---|---|
| challenge_test__cache | 9faec02 | 0 | 39 | 0 of 0 |
| challenge_test__features_cpc | 9faec02 | 0 | 272 | 0 of 0 |
| challenge_test__features_jepa | 9faec02 | 0 | 111 | 0 of 0 |
| challenge_test__features_measurements | 9faec02 | 0 | 18 | 0 of 0 |
| challenge_test__features_xecg | 9faec02 | 0 | 96 | 0 of 0 |
| challenge_test__measure | 9faec02 | 0 | 128 | 4 of 4 |
| challenge_test__publish | 9faec02 | 0 | 3 | 6 of 6 |
| challenge_test__rules | 9faec02 | 0 | 2 | 3 of 3 |
| challenge_test__scan | 9faec02 | 0 | 95 | 2 of 2 |
| challenge_test__score | 9faec02 | 0 | 66 | 6 of 6 |
| freeze__freeze_draft | e9df26e | 0 | 2 | 1 of 1 |
| freeze__thresholds | 89e2f3a | 0 | 1 | 1 of 1 |
| validation__cache | 89e2f3a | 0 | 54 | 0 of 0 |
| validation__check | e9df26e | 0 | 21 | 1 of 1 |
| validation__features_cpc | 89e2f3a | 0 | 356 | 0 of 0 |
| validation__features_jepa | 89e2f3a | 0 | 97 | 0 of 0 |
| validation__features_measurements | 89e2f3a | 0 | 16 | 0 of 0 |
| validation__features_xecg | 89e2f3a | 0 | 86 | 0 of 0 |
| validation__measure | 89e2f3a | 0 | 126 | 4 of 4 |
| validation__publish | 89e2f3a | 0 | 3 | 6 of 6 |
| validation__rules | 89e2f3a | 0 | 2 | 3 of 3 |
| validation__scan | 89e2f3a | 0 | 44 | 2 of 2 |
| validation__score | e9df26e | 0 | 50 | 6 of 6 |

Marker: 1 attempt(s), completed True, freeze commit 9faec02, code commit e9df26e.

- **Dry run.** It ran on the calibration groups through the same code before the freeze and passed every
  reproduction (`validation/check.json`, section `check`, failed: none):
  - v4 and v5 manifest rows cell by cell (13,162 records);
  - the cache bit by bit;
  - Part A, method B, the windows, method A and the analysis table exactly;
  - re-extracted features within 9.54e-06 (ECG-CPC and the measurements bit-identical);
  - stored validation outputs of the five frozen models within 4.44e-15 from the stored features;
  - P0-v5's validation metric 0.9697697420282113 exactly.

  The [freeze record](mlc04-freeze-record-2026-11-11.md) lists these results, the readings and the SHA-256 of
  63 frozen files. The record's SHA-256 is `22acd4564ae566d2d1cd8049c51c3acd9b42be01c621d7487005532c5c522b6c`.
- **Order.**
  - Every loader refused the test role until the freeze record was committed with a clean tracked tree.
  - All test steps ran at `9faec02` with no modified tracked file.
  - The score step ran once and its marker records one attempt, completed.
- **Screening labels.** No binary label and no screening score was computed on the test groups. The space-A
  groups and the `normal` column were emptied in memory before any test-group file was written.

## 9. Departures and doubts

1. **Amendment 1 and the first freeze version.** The orchestrator wrote Amendment 1 after the dry run showed
   that the original comparator was not like-for-like. The amendment landed on the branch a few seconds before
   the code commit `89e2f3a`, and the first freeze record (`0e7797b`) was written without it. That version
   still used 0.9698 and said no amendment had arrived. It was corrected (`e9df26e`, `9faec02`) before any
   test-group record was opened. The guard had opened at `0e7797b`, but no step had been started on the test
   groups.
2. **Rounding of the amendment's bound.** The amendment writes the bound as 0.9511; the code uses the exact
   dry-run value minus 0.010, 0.9510687901468651. The test lower bound (0.9542) is above both, so the reading
   does not depend on it.
3. **First dry run superseded.** The first dry run ran with uncommitted code and the Haswell OpenBLAS kernel.
   Its Part A and method B values differed from the stored ones by about 1e-10, because the MLC-03 steps had
   run with the default kernel. The runner now requires each step's kernel. The dry run was rerun completely
   at the code commit; the first one is kept, unused, in `superseded_first_dry_run/`.
4. **Features.** `outputs/features_challenge_v1/` does not exist on this machine, and it read the centred
   window rather than the first 10 s. Features were therefore extracted now by the MLC-02 code, all three
   encoders on GPU 0; MLC-02 ran ECG-JEPA and ECG-CPC on GPU 1. On validation this reproduced the stored
   features within 9.54e-06.
5. **Readings of the protocol** (freeze record, readings 1-15). The main ones:
   - the test groups carry the MLC-01 role name `test_same_source` with origin `test`;
   - the minimum-positive rule applies the frozen v4 table;
   - Holm uses each comparison's bootstrap p-value against its null value, with the v4-label cleaning interval
     as a condition rather than a fourth test;
   - "v4 labels" means the v4-included records with v4 labels.
6. **Report section added after the read.** `comparisons-by-source` was added to the tables script after the
   test read. It only formats the per-source differences that `evaluation.json` already held. Nothing else
   changed after the read.
7. **Doubt: cleaning.** The cleaning reading ("helps") meets the protocol's rule, but the gain exists only on
   v5-rule labels. On v4 labels the two P0 models do not differ. This fits the rules removing labels that both
   models found hard, more than it fits a better model.
8. **Doubt: K6.** K6's equivalence rests on CPSC-Extra's single entering condition. In the other four sources
   K6 ranks below P0-v5, by -0.0010 to -0.0083, with intervals below zero. The protocol's metric averages over
   sources with equal weight, whatever the number of entering conditions.

## 10. Limits

- **Same hospitals as training.** The test groups come from the five Challenge sources whose train groups
  trained every model and whose calibration groups chose them. SPH remains the only external hospital, and it
  has been read before.
- **Shared patients.** There is no patient identifier, so the split is per record. A share of test patients is
  expected to have training records, as on validation, and the intervals resample records, not patients.
- **Labels.** They are annotation proxies that each source codes differently. They were cleaned by rules C1-C12,
  developed on the train and validation groups of the same sources; a defect that occurs only in the test
  groups can be missed.
- **Encoders.** xECG and ECG-JEPA were pretrained on Chapman and Ningbo waveforms.
- **Model choice.** The rules and thresholds were set on the validation groups, which share hospitals and some
  patients with the test groups, and P0's C per condition was chosen there too.
- **No clinical review.** No cardiologist has reviewed anything, and clinical validation is pending. The
  result is not evidence of clinical performance.

## 11. How it cannot be rerun

The test read is spent:

- `run_mlc04_final_test score --role challenge_test` refuses to run again, because its marker
  (`challenge_test/score/marker.json`) records a completed read.
- `--rerun-after-crash` also refuses a completed read.
- From now on the Challenge test groups count as read for the 14-condition task. Any later change to a model,
  rule, threshold or label definition makes a new pipeline that has no untouched test.
- The steps that hold no score (scan, cache, measure, features, publish, rules) can be repeated from the
  committed code; the score step cannot.
- The screening pipeline's own one-time test is not spent by this experiment.

## Files

`outputs/mlc04_final_test_20261111/` (local, not in Git):

- `validation/` and `challenge_test/`: `scan.parquet`, `manifest_v4.csv`, `part_a_measures.parquet`,
  `part_a_flags.parquet`, `method_b.parquet`, `windows.csv`, `features/`, `manifest_v5.csv`, `rule_flow.csv`,
  `analysis.parquet`, and `score/` (`evaluation.json`, `conditions.csv`, `reliability.csv`, `support_v5.csv`,
  `support_v4.csv`, `outputs.npz`; the marker for the test groups);
- `validation/check.json`;
- `freeze/` (`thresholds.json`, `freeze.json`);
- `receipts/` and `logs/`;
- `superseded_first_dry_run/`.

Waveform caches and feature shards are in `/var/tmp/janetrivera/mlc04/`.

Code: `ecg_experiment/mlc04_data.py`, `mlc04_models.py`, `mlc04_evaluation.py`, `mlc04_checks.py`,
`scripts/experiments/run_mlc04_final_test.py`, `scripts/reports/mlc04_tables.py`, `tests/test_mlc04_final_test.py`.
