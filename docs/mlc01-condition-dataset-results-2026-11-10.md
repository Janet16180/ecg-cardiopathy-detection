# MLC-01 results: the condition dataset

10 November 2026. Executes the frozen [MLC-01 protocol](mlc01-condition-dataset-protocol-2026-11-10.md)
(commit `d4e25c7`). This is a dataset build: no model was fitted and no performance score of any kind was
computed. Every number below is read from the executed outputs in
`outputs/mlc01_condition_dataset_20261110/` (local, not in Git), named in brackets.

- Label mapping and tests committed first: `3044b72` (`ecg_experiment/mlc_labels.py`,
  `tests/test_mlc_labels.py`).
- Build code committed before the run: `b6d3b6c` (`ecg_experiment/mlc01_dataset.py`, `mlc01_tables.py`,
  `mlc01_build.py`, `scripts/data/build_mlc01_dataset.py`, `tests/test_mlc01_dataset.py`). The receipt
  records this commit with no modified tracked file [`receipt.json` `git`].
- Run: `PYTHONPATH=. OPENBLAS_CORETYPE=Haswell OMP_NUM_THREADS=1 .venv/bin/python -m
  scripts.data.build_mlc01_dataset --workers 16`, 471.4 s, peak RSS 1,302,368 KiB in the parent and
  390,636 KiB in a worker [`receipt.json` `run`]. A 100-records-per-source profile ran first into
  `/var/tmp`; it is not a result.

## Final dataset

99,086 of 100,221 enumerated records are included. 63 are shorter than 10 s and 1,072 are removed by the
duplicate rule; no record was unreadable, had other than the twelve named leads, or was not 500 Hz
[`exclusion_flow.csv`].

| role | ptbxl | ningbo | chapman_shaoxing | georgia | cpsc_2018 | cpsc_2018_extra | sph |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 17,418 | 20,919 | 6,097 | 6,117 | 3,763 | 1,877 | 0 |
| validation | 2,183 | 6,972 | 2,036 | 2,043 | 1,251 | 631 | 0 |
| test_same_source | 2,198 | 0 | 0 | 0 | 0 | 0 | 0 |
| test_external | 0 | 0 | 0 | 0 | 0 | 0 | 12,698 |
| sph_calibration | 0 | 0 | 0 | 0 | 0 | 0 | 12,883 |

Roles follow the protocol table: PTB-XL folds 1-8, 9 and 10; the Challenge train and calibration groups;
SPH split into halves by `sha256('experiment042:20261018:' + Patient_ID).digest()[0] < 128` (calibration)
applied to all 25,770 SPH records. The Challenge test groups (13,172 records) were not opened.

## Integrity checks

All checks passed; the build stops on any failure [`receipt.json` `integrity`].

| Check | Expected | Observed |
|---|---|---|
| Challenge split table SHA-256 | `5ecd8215...59cc8` | same |
| Open Challenge records per source and group | the 10 counts of `challenge-splits-v1.md` | equal |
| PTB-XL records in folds 1-8 / 9 / 10 | 17,418 / 2,183 / 2,198 | 17,418 / 2,183 / 2,198 |
| SPH records and patients before duplicate removal | 25,770 and 24,666 | 25,770 and 24,666 |
| Patient hash reproduces the roles of `pr74_fresh_baseline_exact_20261028/features/sph_rows.csv` | 17,402 records | 17,402 |
| PTB-XL and SPH patients in more than one role | 0 and 0 | 0 and 0 |
| Challenge test-group paths scanned | 0 | 0 |
| SPH metadata sample count equals the file | 25,770 | 25,770 |
| Codes absent from their official table (SCP, SNOMED CT, AHA) | none | none |

Every PTB-XL and Challenge header and signal file matched its release `SHA256SUMS.txt` digest (21,799
PTB-XL and 52,652 Challenge records). SPH publishes no per-record checksum: the 25,770 HDF5 files were
hashed into the manifest, and the metadata and code tables matched the download manifest's MD5
[`receipt.json` `raw_files_verified_against_release_checksums`, `sph_small_files_md5`].

The closed-data guard has three layers: the split table is read with `usecols` limited to source, family,
record, path and split; test rows are filtered before any job is built; and every worker refuses a path of
the closed set (tested in `tests/test_mlc01_dataset.py`).

## Exclusion flow

| source | start | unreadable | wrong leads | wrong rate | short | eligible | dropped copy | dropped conflict | final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ptbxl | 21,799 | 0 | 0 | 0 | 0 | 21,799 | 0 | 0 | 21,799 |
| ningbo | 27,920 | 0 | 0 | 0 | 0 | 27,920 | 3 | 26 | 27,891 |
| chapman_shaoxing | 8,197 | 0 | 0 | 0 | 0 | 8,197 | 38 | 26 | 8,133 |
| georgia | 8,273 | 0 | 0 | 0 | 43 | 8,230 | 70 | 0 | 8,160 |
| cpsc_2018 | 5,496 | 0 | 0 | 0 | 10 | 5,486 | 281 | 191 | 5,014 |
| cpsc_2018_extra | 2,766 | 0 | 0 | 0 | 10 | 2,756 | 33 | 215 | 2,508 |
| sph | 25,770 | 0 | 0 | 0 | 0 | 25,770 | 147 | 42 | 25,581 |

No copy crossed roles, so the rule that removes a training or validation copy of a test record removed
nothing. Ningbo JS13118 is not in the split table (its waveform is absent upstream), so it never entered.
Per-role rows are in `exclusion_flow.csv`.

## Patients and record groups

[`group_counts.csv`]

| role | source | records | groups | groups with several records | records in those groups | max per group |
|---|---|---:|---:|---:|---:|---:|
| train | ptbxl | 17,418 | 15,023 | 1,694 | 4,089 | 10 |
| validation | ptbxl | 2,183 | 1,942 | 189 | 430 | 8 |
| test_same_source | ptbxl | 2,198 | 1,904 | 228 | 522 | 6 |
| test_external | sph | 12,698 | 12,246 | 439 | 891 | 4 |
| sph_calibration | sph | 12,883 | 12,400 | 469 | 952 | 4 |

The Challenge sources have no patient identifier, so each record is its own group.

## Space B by evaluation set

Positive / negative / undefined records; `chapman_ningbo` and `cpsc` are the two-source families
[`support_B.csv`].

| condition | val ptbxl | val ningbo | val chapman | val chapman_ningbo | val georgia | val cpsc_2018 | val cpsc_extra | val cpsc | test ptbxl | test sph | sph calibration |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sinus_tachycardia | 83 / 2,100 / 0 | 1,145 / 5,827 / 0 | 314 / 1,722 / 0 | 1,459 / 7,549 / 0 | 265 / 1,778 / 0 | 0 / 0 / 1,251 | 65 / 566 / 0 | 65 / 566 / 1,251 | 82 / 2,116 / 0 | 360 / 12,338 / 0 | 357 / 12,526 / 0 |
| sinus_bradycardia | 64 / 2,119 / 0 | 2,510 / 4,461 / 1 | 769 / 1,267 / 0 | 3,279 / 5,728 / 1 | 318 / 1,725 / 0 | 0 / 0 / 1,251 | 6 / 576 / 49 | 6 / 576 / 1,300 | 64 / 2,134 / 0 | 1,302 / 11,396 / 0 | 1,393 / 11,490 / 0 |
| sinus_arrhythmia | 77 / 2,106 / 0 | 529 / 6,443 / 0 | 0 / 0 / 2,036 | 529 / 6,443 / 2,036 | 93 / 1,950 / 0 | 0 / 0 / 1,251 | 2 / 629 / 0 | 2 / 629 / 1,251 | 77 / 2,121 / 0 | 781 / 11,917 / 0 | 754 / 12,129 / 0 |
| af_or_flutter | 156 / 2,027 / 0 | 1,536 / 5,436 / 0 | 454 / 1,582 / 0 | 1,990 / 7,018 / 0 | 133 / 1,910 / 0 | 213 / 1,038 / 0 | 38 / 593 / 0 | 251 / 1,631 / 0 | 157 / 2,041 / 0 | 379 / 12,319 / 0 | 384 / 12,499 / 0 |
| premature_ventricular_complex | 114 / 2,063 / 6 | 220 / 6,730 / 22 | 64 / 1,972 / 0 | 284 / 8,702 / 22 | 66 / 1,977 / 0 | 113 / 1,138 / 0 | 25 / 606 / 0 | 138 / 1,744 / 0 | 114 / 2,075 / 9 | 517 / 12,181 / 0 | 541 / 12,342 / 0 |
| first_degree_av_block | 104 / 2,079 / 0 | 189 / 6,783 / 0 | 50 / 1,986 / 0 | 239 / 8,769 / 0 | 149 / 1,894 / 0 | 112 / 1,139 / 0 | 18 / 613 / 0 | 130 / 1,752 / 0 | 96 / 2,102 / 0 | 112 / 12,586 / 0 | 123 / 12,760 / 0 |
| complete_rbbb | 54 / 2,128 / 1 | 256 / 6,659 / 57 | 87 / 1,943 / 6 | 343 / 8,602 / 63 | 104 / 1,911 / 28 | 334 / 917 / 0 | 18 / 606 / 7 | 352 / 1,523 / 7 | 54 / 2,144 / 0 | 350 / 12,348 / 0 | 349 / 12,533 / 1 |
| incomplete_rbbb | 111 / 2,071 / 1 | 58 / 6,914 / 0 | 0 / 0 / 2,036 | 58 / 6,914 / 2,036 | 74 / 1,959 / 10 | 0 / 0 / 1,251 | 5 / 619 / 7 | 5 / 619 / 1,258 | 112 / 2,086 / 0 | 619 / 12,079 / 0 | 626 / 12,256 / 1 |
| lafb | 163 / 2,020 / 0 | 76 / 6,896 / 0 | 0 / 0 / 2,036 | 76 / 6,896 / 2,036 | 45 / 1,998 / 0 | 0 / 0 / 1,251 | 0 / 631 / 0 | 0 / 631 / 1,251 | 162 / 2,036 / 0 | 77 / 12,621 / 0 | 74 / 12,809 / 0 |
| left_axis_deviation | 121 / 1,694 / 368 | 216 / 6,756 / 0 | 80 / 1,956 / 0 | 296 / 8,712 / 0 | 190 / 1,853 / 0 | 0 / 0 / 1,251 | 0 / 631 / 0 | 0 / 631 / 1,251 | 134 / 1,670 / 394 | 68 / 12,630 / 0 | 70 / 12,813 / 0 |
| lvh | 210 / 1,946 / 27 | 103 / 6,097 / 772 | 4 / 1,761 / 271 | 107 / 7,858 / 1,043 | 231 / 1,803 / 9 | 0 / 0 / 1,251 | 30 / 601 / 0 | 30 / 601 / 1,251 | 214 / 1,964 / 20 | 100 / 12,598 / 0 | 107 / 12,776 / 0 |
| st_depression | 101 / 2,082 / 0 | 231 / 6,741 / 0 | 78 / 1,958 / 0 | 309 / 8,699 / 0 | 0 / 0 / 2,043 | 161 / 1,090 / 0 | 8 / 623 / 0 | 169 / 1,713 / 0 | 101 / 2,097 / 0 | 497 / 11,293 / 908 | 513 / 11,461 / 909 |
| t_wave_abnormality | 260 / 1,923 / 0 | 1,305 / 5,020 / 647 | 390 / 1,486 / 160 | 1,695 / 6,506 / 807 | 544 / 1,218 / 281 | 0 / 0 / 1,251 | 5 / 410 / 216 | 5 / 410 / 1,467 | 255 / 1,943 / 0 | 1,633 / 11,065 / 0 | 1,618 / 11,265 / 0 |
| mi_any | 521 / 1,643 / 19 | 26 / 6,946 / 0 | 6 / 2,030 / 0 | 32 / 8,976 / 0 | 0 / 2,043 / 0 | 0 / 0 / 1,251 | 267 / 364 / 0 | 267 / 364 / 1,251 | 532 / 1,648 / 18 | 118 / 12,580 / 0 | 137 / 12,746 / 0 |

Training records, positive / negative / undefined [`label_counts_B.csv`]:

| condition | ptbxl | ningbo | chapman_shaoxing | georgia | cpsc_2018 | cpsc_2018_extra |
|---|---:|---:|---:|---:|---:|---:|
| sinus_tachycardia | 661 / 16,757 / 0 | 3,419 / 17,500 / 0 | 922 / 5,175 / 0 | 745 / 5,372 / 0 | 0 / 0 / 3,763 | 156 / 1,721 / 0 |
| sinus_bradycardia | 509 / 16,909 / 0 | 7,615 / 13,300 / 4 | 2,312 / 3,785 / 0 | 954 / 5,159 / 4 | 0 / 0 / 3,763 | 28 / 1,701 / 148 |
| sinus_arrhythmia | 618 / 16,800 / 0 | 1,513 / 19,406 / 0 | 0 / 0 / 6,097 | 264 / 5,853 / 0 | 0 / 0 / 3,763 | 4 / 1,873 / 0 |
| af_or_flutter | 1,257 / 16,161 / 0 | 4,532 / 16,387 / 0 | 1,336 / 4,761 / 0 | 441 / 5,676 / 0 | 587 / 3,176 / 0 | 87 / 1,790 / 0 |
| premature_ventricular_complex | 915 / 16,456 / 47 | 656 / 20,201 / 62 | 176 / 5,921 / 0 | 239 / 5,878 / 0 | 364 / 3,399 / 0 | 94 / 1,783 / 0 |
| first_degree_av_block | 808 / 16,610 / 0 | 576 / 20,343 / 0 | 161 / 5,936 / 0 | 443 / 5,674 / 0 | 418 / 3,345 / 0 | 42 / 1,835 / 0 |
| complete_rbbb | 432 / 16,986 / 0 | 765 / 19,912 / 242 | 238 / 5,834 / 25 | 285 / 5,743 / 89 | 950 / 2,813 / 0 | 22 / 1,833 / 22 |
| incomplete_rbbb | 894 / 16,524 / 0 | 140 / 20,779 / 0 | 0 / 0 / 6,097 | 196 / 5,890 / 31 | 0 / 0 / 3,763 | 13 / 1,843 / 21 |
| lafb | 1,298 / 16,120 / 0 | 227 / 20,692 / 0 | 0 / 0 / 6,097 | 102 / 6,015 / 0 | 0 / 0 / 3,763 | 0 / 1,877 / 0 |
| left_axis_deviation | 1,127 / 13,289 / 3,002 | 708 / 20,211 / 0 | 225 / 5,872 / 0 | 556 / 5,561 / 0 | 0 / 0 / 3,763 | 0 / 1,877 / 0 |
| lvh | 1,708 / 15,535 / 175 | 392 / 18,031 / 2,496 | 7 / 5,370 / 720 | 763 / 5,336 / 18 | 0 / 0 / 3,763 | 85 / 1,792 / 0 |
| st_depression | 807 / 16,611 / 0 | 786 / 20,133 / 0 | 242 / 5,855 / 0 | 0 / 0 / 6,117 | 499 / 3,264 / 0 | 34 / 1,843 / 0 |
| t_wave_abnormality | 2,079 / 15,339 / 0 | 3,951 / 15,020 / 1,948 | 1,156 / 4,418 / 523 | 1,621 / 3,649 / 847 | 0 / 0 / 3,763 | 16 / 1,190 / 671 |
| mi_any | 4,235 / 13,039 / 144 | 97 / 20,822 / 0 | 25 / 6,072 / 0 | 3 / 6,114 / 0 | 0 / 0 / 3,763 | 803 / 1,074 / 0 |

On the final data each of the 14 conditions still has at least 100 positives in the training pool (smallest:
incomplete RBBB 1,243), in SPH (left axis deviation 138) and in PTB-XL folds 9-10 (complete RBBB 108), the
protocol's selection rule.

### Conditions below 20, 50 and 100 positives

The macro rule admits a condition with at least 20 positives and 20 negatives in the set [`support_B.csv`].

| set | below 20 | 20-49 | 50-99 | in macro (of 14) |
|---|---|---|---|---:|
| validation ptbxl | | | complete_rbbb, sinus_arrhythmia, sinus_bradycardia, sinus_tachycardia | 14 |
| validation ningbo | | mi_any | incomplete_rbbb, lafb | 14 |
| validation chapman_shaoxing | incomplete_rbbb, lafb, lvh, mi_any, sinus_arrhythmia | | complete_rbbb, first_degree_av_block, left_axis_deviation, premature_ventricular_complex, st_depression | 9 |
| validation chapman_ningbo | | mi_any | incomplete_rbbb, lafb | 14 |
| validation georgia | mi_any, st_depression | lafb | incomplete_rbbb, premature_ventricular_complex, sinus_arrhythmia | 12 |
| validation cpsc_2018 | incomplete_rbbb, lafb, left_axis_deviation, lvh, mi_any, sinus_arrhythmia, sinus_bradycardia, sinus_tachycardia, t_wave_abnormality | | | 5 |
| validation cpsc_2018_extra | complete_rbbb, first_degree_av_block, incomplete_rbbb, lafb, left_axis_deviation, sinus_arrhythmia, sinus_bradycardia, st_depression, t_wave_abnormality | af_or_flutter, lvh, premature_ventricular_complex | sinus_tachycardia | 5 |
| validation cpsc | incomplete_rbbb, lafb, left_axis_deviation, sinus_arrhythmia, sinus_bradycardia, t_wave_abnormality | lvh | sinus_tachycardia | 8 |
| test ptbxl (fold 10) | | | complete_rbbb, first_degree_av_block, sinus_arrhythmia, sinus_bradycardia, sinus_tachycardia | 14 |
| test sph (evaluation half) | | | lafb, left_axis_deviation | 14 |
| sph calibration half | | | lafb, left_axis_deviation | 14 |

"Below 20" includes sources that do not annotate the condition (0 positives).

### Likelihood sensitivity

PTB-XL positives resting only on diagnostic statements with likelihood below 50, which become undefined in
the sensitivity analysis [`label_counts_B.csv` minus `label_counts_B_lik50.csv`]: myocardial infarction
1,090 (train), 131 (fold 9), 141 (fold 10); LVH 297, 47, 37; complete RBBB 1, 0, 0; first-degree AV block
3, 0, 0; incomplete RBBB 3, 0, 0; LAFB 0, 1, 0; no other condition. In fold 10 this leaves 391 MI and 177
LVH positives [`support_B_lik50.csv`].

## Space A by evaluation set

Positive / negative / undefined [`support_A.csv`]:

| group | val ptbxl | val ningbo | val chapman | val georgia | val cpsc_2018 | val cpsc_extra | test ptbxl | test sph | sph calibration |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MI | 540 / 1,643 / 0 | 26 / 6,946 / 0 | 6 / 2,030 / 0 | 0 / 2,043 / 0 | 0 / 0 / 1,251 | 267 / 364 / 0 | 550 / 1,648 / 0 | 118 / 12,580 / 0 | 137 / 12,746 / 0 |
| STTC | 528 / 1,655 / 0 | 2,068 / 4,904 / 0 | 579 / 1,457 / 0 | 989 / 1,054 / 0 | 195 / 1,056 / 0 | 429 / 202 / 0 | 521 / 1,677 / 0 | 2,480 / 10,218 / 0 | 2,478 / 10,405 / 0 |
| CD | 495 / 1,688 / 0 | 769 / 6,203 / 0 | 256 / 1,780 / 0 | 460 / 1,583 / 0 | 486 / 765 / 0 | 65 / 566 / 0 | 496 / 1,702 / 0 | 1,166 / 11,532 / 0 | 1,200 / 11,683 / 0 |
| HYP | 268 / 1,888 / 27 | 129 / 6,072 / 771 | 5 / 1,760 / 271 | 425 / 1,618 / 0 | 0 / 0 / 1,251 | 42 / 589 / 0 | 262 / 1,917 / 19 | 104 / 12,594 / 0 | 123 / 12,760 / 0 |

Training records (positive / negative / undefined) [`label_counts_A.csv`]: MI 4,379 / 13,039 / 0 at
PTB-XL and 803 / 1,074 / 0 at CPSC-Extra, with 97 or fewer MI positives at Ningbo, Chapman and Georgia;
HYP undefined on
2,492 Ningbo, 720 Chapman and 168 PTB-XL training records (left ventricular high voltage only) and on all
3,763 CPSC 2018 training records.

## Space C (exploratory)

Number of the 30 conditions below each support level, and entering a macro [`support_C.csv`]:

| set | below 20 | below 50 | below 100 | in macro |
|---|---:|---:|---:|---:|
| validation ptbxl | 9 | 12 | 18 | 21 |
| validation ningbo | 8 | 10 | 14 | 22 |
| validation chapman_shaoxing | 13 | 17 | 24 | 17 |
| validation chapman_ningbo | 6 | 8 | 12 | 24 |
| validation georgia | 8 | 14 | 18 | 22 |
| validation cpsc_2018 | 22 | 24 | 24 | 8 |
| validation cpsc_2018_extra | 25 | 28 | 29 | 5 |
| validation cpsc | 19 | 21 | 23 | 11 |
| test ptbxl | 9 | 12 | 19 | 21 |
| test sph | 9 | 11 | 17 | 21 |
| sph calibration | 10 | 12 | 17 | 20 |

Positives per set; `-` marks a source that does not annotate or cannot code the condition:

| condition | val ptbxl | val ningbo | val chapman | val georgia | val cpsc_2018 | val cpsc_extra | test ptbxl | test sph | sph calibration | train (all sources) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| anterior_mi | 282 | 12 | 0 | 0 | - | 11 | 282 | 69 | 76 | 2,336 |
| atrial_fibrillation | 151 | - | 354 | 105 | 213 | 23 | 152 | 321 | 345 | 3,261 |
| atrial_flutter | 7 | 1,536 | 100 | 31 | - | 9 | 7 | 58 | 39 | 4,991 |
| complete_lbbb | 54 | 48 | 34 | 46 | 42 | 8 | 54 | 36 | 46 | 944 |
| inferior_mi | 322 | - | - | - | - | - | 323 | 53 | 65 | 2,593 |
| ischemic_st_t | 250 | 0 | 0 | 312 | - | 108 | 248 | - | - | 3,180 |
| left_atrial_enlargement | 43 | 0 | 0 | 196 | - | 9 | 42 | 2 | 17 | 928 |
| long_qt | 12 | 64 | 11 | 305 | - | 1 | 11 | 11 | 13 | 1,140 |
| low_qrs_voltage | 19 | 153 | 47 | 71 | - | 0 | 18 | 149 | 172 | 1,002 |
| lpfb | 18 | 1 | - | 9 | - | 0 | 18 | 5 | 1 | 155 |
| nonspecific_ivcd | 78 | 118 | 52 | 44 | - | 1 | 79 | - | - | 1,183 |
| paced_rhythm | 29 | 243 | 0 | 21 | - | 0 | 28 | - | - | 1,006 |
| preexcitation | 7 | 11 | 3 | 0 | - | 0 | 8 | 8 | 19 | 127 |
| premature_atrial_complex | 40 | 228 | 57 | 135 | 100 | 16 | 40 | 269 | 269 | 1,886 |
| right_axis_deviation | 15 | 127 | 47 | 13 | - | 0 | 9 | 72 | 89 | 664 |
| rvh | 12 | 25 | 0 | 10 | - | 4 | 12 | 2 | 4 | 234 |
| st_elevation | 3 | 0 | 33 | 24 | 34 | 14 | 3 | 20 | 17 | 334 |
| supraventricular_tachycardia | 5 | 70 | 145 | 15 | - | 4 | 5 | 5 | 8 | 681 |

The other 12 conditions of space C share space B's definitions and counts.

## Normal column

The per-source normal definition is kept for audits only [`label_counts_normal.csv`]: PTB-XL NORM with only
SR otherwise (fold 10: 701 of 2,198); Challenge sinus rhythm as the only code; SPH code 1 alone (evaluation
half 6,850 of 12,698). CPSC-Extra is undefined throughout; the inventory found no CPSC-Extra record with
sinus rhythm as the only code (`inventory_snapshot/label_space_A.csv`).

## Co-occurrence

Share of records by number of positive space-B conditions, training rows and the SPH evaluation half
[`cooccurrence_B_count.csv`, which also holds every other role]:

| source | 0 | 1 | 2 | 3 | 4 | 5 or more |
|---|---:|---:|---:|---:|---:|---:|
| ptbxl | 41.8% | 31.5% | 16.1% | 7.3% | 2.6% | 0.8% |
| ningbo | 17.1% | 54.6% | 20.5% | 6.1% | 1.6% | 0.2% |
| chapman_shaoxing | 20.9% | 54.0% | 18.8% | 5.6% | 0.7% | 0.1% |
| georgia | 29.7% | 41.1% | 22.0% | 6.0% | 1.0% | 0.2% |
| cpsc_2018 | 27.6% | 70.0% | 2.4% | 0% | 0% | 0% |
| cpsc_2018_extra | 41.9% | 43.8% | 13.1% | 1.2% | 0.1% | 0% |
| sph (test_external) | 58.5% | 30.6% | 9.2% | 1.6% | 0.2% | 0.0% |

Most frequent pairs, training rows [`cooccurrence_B_pairs.csv`]: PTB-XL left axis deviation with MI 622
(3.6%) and LAFB with left axis deviation 621 (3.6%); Ningbo AF or flutter with T-wave abnormality 1,556
(7.4%); Chapman AF or flutter with T-wave abnormality 469 (7.7%); Georgia LVH with T-wave abnormality 226
(3.7%); CPSC-Extra sinus tachycardia with MI 81 (4.3%); SPH evaluation half sinus bradycardia with T-wave
abnormality 240 (1.9%). The full 91-pair tables per role and source are in the CSV, and the same for
space C.

## Flat and missing leads

In the first 10 s of the final records [`lead_flags.csv`], at least one lead is flat in 919 Ningbo training
and 283 Ningbo validation records, 13 Chapman, 6 Georgia, 9 CPSC 2018, 5 CPSC-Extra and 1 PTB-XL record;
one CPSC-Extra validation record is flat in all twelve leads; no SPH record has a flat lead. Missing
samples occur in 63 + 19 Ningbo and 3 + 2 Georgia records. No record was excluded for quality; per-lead
flags are manifest columns for the subgroup analysis.

## Duplicates

Exact duplicates share the hash of the first 10 s (int32 microvolts, leads I to V6). Eligible records form
806 groups of more than one record, 1,635 records [`duplicate_accounting.csv`]; in all of them the
whole-record hashes are also identical [`receipt.json` `summary`].

| scope | sources | role | groups | records | kept | dropped copy | dropped conflict |
|---|---|---|---:|---:|---:|---:|---:|
| across sources | chapman_shaoxing+ningbo | train | 43 | 86 | 22 | 22 | 42 |
| across sources | chapman_shaoxing+ningbo | validation | 11 | 22 | 6 | 6 | 10 |
| across sources | cpsc_2018+cpsc_2018_extra | train | 203 | 418 | 63 | 65 | 290 |
| across sources | cpsc_2018+cpsc_2018_extra | validation | 71 | 146 | 27 | 28 | 91 |
| within source | chapman_shaoxing | train / validation | 7 / 3 | 14 / 6 | 7 / 3 | 7 / 3 | 0 / 0 |
| within source | cpsc_2018 | train / validation | 138 / 46 | 277 / 95 | 138 / 46 | 139 / 49 | 0 / 0 |
| within source | cpsc_2018_extra | train / validation | 35 / 10 | 71 / 20 | 25 / 8 | 25 / 8 | 21 / 4 |
| within source | georgia | train / validation | 52 / 16 | 106 / 32 | 52 / 16 | 54 / 16 | 0 / 0 |
| within source | ningbo | train / validation | 1 / 2 | 2 / 4 | 1 / 2 | 1 / 2 | 0 / 0 |
| within source | sph | sph_calibration | 82 | 164 | 70 | 70 | 24 |
| within source | sph | test_external | 86 | 172 | 77 | 77 | 18 |

- No group spans two roles, so 0 records were removed as copies of a test or validation record.
- The protocol names 71 Chapman-Ningbo and 155 CPSC-Extra copies. The 71 is the all-splits count of
  `challenge-splits-v1.md`. In the opened train and calibration groups there are 54 Chapman-Ningbo pairs
  and 274 CPSC 2018-CPSC-Extra groups holding 284 CPSC 2018 and 280 CPSC-Extra records
  [`supplementary/duplicates.json`]; this build does not reproduce the 155.
- Conflicts [`supplementary/duplicate_conflicts.csv`, `duplicates.json`]: 243 groups. 184 are CPSC 2018 with
  CPSC-Extra; in 84 of them every conflicting label is one CPSC 2018 annotates only in part (space-A STTC
  or CD, AF or flutter). The most frequent conflicting labels are space-A STTC (151 groups), premature
  ventricular complexes (39) and premature atrial complexes in space C (25).
- All 168 SPH groups are pairs from one patient; 25 pairs differ in their raw AHA codes, 21 conflict in the
  mapped labels. On the 17,402 records of the earlier SPH baseline table the MLC-01 statuses agree with its
  frozen SPH rule (100 kept, 17,302 unique) [`supplementary/duplicates.json`].
- Near duplicates, which change no rule: 0 groups with the same sign pattern of successive differences
  (unchanged by gain and offset) but a different exact hash, and 0 record pairs sharing six or more identical
  non-flat leads without being exact copies [`near_duplicates_shape.csv`, `near_duplicates_leads.csv`].

## Comparison with the inventory

[`inventory_comparison/`, from `scripts/reports/compare_mlc01_inventory.py`]

1. Reproduction. Re-applying the draft mapping (`vocabulary.py`) to the inventory's record sets reproduces
   all 576 inventory rows exactly (records, positives and undefined), with 0 mismatches. For the Challenge
   groups the inventory used the split's `unique` and `kept` records; they were rebuilt from the
   whole-record hashes with the frozen Ningbo-family rule (51,347 unique and 397 kept, 51,744 in total,
   the inventory's count), without reading any split-table column beyond identity and group.
2. Rule changes, same records. Every changed state is attributed to a protocol rule; 0 are unexplained
   [`rule_transitions.csv`]. Records on the inventory's sets:

| rule | label | change | ptbxl | ningbo | chapman | georgia | cpsc_2018 | cpsc_extra | sph |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| LV high voltage only | lvh | negative to undefined | 222 | 3,267 | 996 | 0 | 0 | 0 | 0 |
| LV high voltage only | space-A HYP | negative to undefined | 214 | 3,262 | 996 | 0 | 0 | 0 | 0 |
| Georgia ST depression | st_depression | positive / negative to undefined | 0 | 0 | 0 | 28 / 8,172 | 0 | 0 | 0 |
| contradictory pair | complete_rbbb | positive to undefined | 1 | 0 | 31 | 41 | 0 | 25 | 1 |
| contradictory pair | incomplete_rbbb | positive to undefined | 1 | 0 | 0 | 41 | 0 | 23 | 1 |
| contradictory pair | complete_lbbb (C) | positive to undefined | 0 | 0 | 31 | 0 | 0 | 3 | 0 |
| SPH code 153 to HYP | space-A STTC | positive to negative | 0 | 0 | 0 | 0 | 0 | 0 | 85 |
| CPSC 2018 not annotated | space-A MI and HYP | negative to undefined | 0 | 0 | 0 | 0 | 5,071 | 0 | 0 |
| unspecified MI territory | anterior_mi (C) | negative to undefined | 0 | 77 | 31 | 3 | 0 | 984 | 0 |
| Chapman fascicular blocks | lpfb (C) | negative to undefined | 0 | 0 | 8,183 | 0 | 0 | 0 | 0 |

3. Record sets [`record_set_changes.csv`]. PTB-XL is unchanged. SPH loses 94 calibration-half and 95
   evaluation-half records to duplicate resolution (the inventory kept every SPH record). Georgia loses 40
   short records, CPSC 2018 10 and CPSC-Extra 10 (the inventory applied no duration rule). The duplicate
   rules differ: the split rule keeps the lowest record name, MLC-01 the copy with the most defined labels,
   so 93 CPSC-Extra and 29 Ningbo records are in MLC-01 but not in the inventory sets, while 50 Chapman,
   47 CPSC 2018 and 3 CPSC-Extra records are in the inventory sets but not in MLC-01. CPSC-Extra training
   MI positives, for example, are 773 in the inventory and 803 in MLC-01, with no rule change
   (`counts.csv`).

## Departures from the protocol

All dated 10 November 2026; none changes a decision of the protocol.

1. SPH files store neither lead names nor the sampling rate. The documented SPH lead order and 500 Hz are
   used; "twelve leads identified by name" is checked by name only for PTB-XL and the Challenge.
2. "Exact waveform hashes" was read as the hash of the first 10 s, the input every model sees, which is
   the more inclusive reading. No group differs in its whole-record hash, so the outcome equals the
   whole-record reading.
3. "Conflicting label sets" was read as one copy positive and another negative for any label of spaces A,
   B or C or the normal column; undefined agrees with anything. Without a conflict the copy with the most
   defined labels is kept (ties: lowest record key). The rule is applied inside every role, the test sets
   included, and copies are removed from the less protected role first (tests, validation, SPH
   calibration, training). The last step removed nothing.
4. Space A. The protocol keeps the existing definitions and moves SPH 153 to HYP. Two three-state rules of
   the protocol were also applied to the groups: CPSC 2018 MI and HYP are undefined (no code among its nine
   classes), and HYP is undefined when the only hypertrophy evidence is left ventricular high voltage, as
   for LVH. CPSC 2018 STTC and CD stay defined and are marked partial.
5. CPSC-Extra "has no normal record": its normal column is undefined rather than 0.
6. Space C. Conditions shared with space B use B's definitions. Anterior MI is undefined on SNOMED codes
   that state no territory (164865005, 164867002, 57054005); LPFB is not annotated at Chapman, which uses
   no fascicular-block code (the draft's own reason for LAFB). Right ventricular hypertrophy keeps the
   draft's undefined state for SPH 153.
7. The likelihood sensitivity flag applies to diagnostic statements. PTB-XL form and rhythm statements
   carry likelihood 0, meaning none was given, and are not treated as below 50.

## Questions for a ruling

These follow the protocol as written; the counts are records of the final dataset that a different ruling
would move [`supplementary/label_doubts.csv`].

- CPSC 2018 partial negatives. Space-A STTC and CD and AF-or-flutter are negative at CPSC 2018 whenever its
  nine classes are absent; 84 of 184 CPSC conflict groups conflict only there.
- Ventricular tachycardia without a PVC code (425856008, 164895002) counts as PVC negative: 33 + 12 Ningbo
  and 1 CPSC-Extra record.
- PTB-XL NST_ (ST change, no direction) without STD_ counts as ST depression negative (555 / 69 / 65 in
  train / fold 9 / fold 10), while SPH 145 without a direction is undefined.
- PTB-XL `heart_axis` empty counts as LAD negative (6,578 / 936 / 954); AXL also (82 / 7 / 12).
- PTB-XL HVOLT alone counts as LVH negative (34 / 6 / 5).
- Ischemia statements without a T-wave statement count as T-wave negative: PTB-XL 1,702 / 219 / 219,
  Georgia 132 / 48, CPSC-Extra 262 / 93 (train / validation).
- CPSC-Extra has 0 LAD and 0 LAFB positives in 2,508 records, and Georgia 3 MI positives in 8,160 (0 in
  validation); they count as negatives, unlike Georgia ST depression.
- LBBB with incomplete RBBB is not a listed contradictory pair (2 CPSC-Extra records).
- The protocol's 155 CPSC-Extra copies are not reproduced; the opened groups hold 280 CPSC-Extra records in
  274 cross-source groups (see Duplicates).
- The protocol's reason for SPH 153 says PTB-XL codes the pattern as LVH plus an ischemic statement, which in
  PTB-XL counts for both HYP and STTC; the decision (HYP only) was followed, moving 85 SPH records from STTC
  positive to negative.

## Limitations

- The labels are annotation statements, not diagnoses, and coding practice differs between sources;
  validation by a cardiologist is pending.
- SPH and PTB-XL fold 10 have been scored by earlier experiments for other targets.
- The Challenge validation groups can share patients with the training groups (no patient identifier).
- Exact duplicates only; copies with any changed sample are not detected, and neither cheap near-duplicate
  check found one.
- The inventory directory was a session scratchpad; its aggregate tables and draft vocabulary are kept in
  `inventory_comparison/inventory_snapshot/` with their SHA-256.

## Files

Outputs (local): `manifest.csv` (one row per enumerated record: identity, role, group, raw path, duration,
file and waveform hashes, lead flags, exclusion and duplicate status, raw codes, label states as 1, 0 or
empty for undefined, `lik50_` flags, normal column), `waveform_scan.parquet` (per-lead and shape hashes),
`mapping.csv`, `annotation_status.csv`, the count tables named above, `receipt.json` (input, code and output
SHA-256, git commit, integrity checks), `build.log`, `inventory_comparison/` and `supplementary/`. Scan
cache: `/var/tmp/janetrivera/mlc01_scan` (chunked, bound to job and code hashes).

Report scripts, run after the build from the worktree with `PYTHONPATH=.`:
`python -m scripts.reports.compare_mlc01_inventory --inventory <labelspace directory>`,
`python -m scripts.reports.describe_mlc01_duplicates` and `python -m scripts.reports.count_mlc01_label_doubts`;
each records its own SHA-256 and the manifest's in its receipt.

Checks: `ruff check ecg_experiment scripts tests conftest.py` passed. `pytest tests/test_mlc_labels.py
tests/test_mlc01_dataset.py`: 33 passed. The full suite in the worktree: 1,616 passed, 2 skipped, 1 failed
(`tests/test_xecg_encoder_motion_v11.py::test_graph_rejects_missing_and_conflicting_leaves`, which rejects
the worktree's symlinked `outputs`; it passes in the main checkout).
