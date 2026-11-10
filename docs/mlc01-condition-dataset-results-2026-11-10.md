# MLC-01 results: the condition dataset

10 November 2026. Executes the frozen [MLC-01 protocol](mlc01-condition-dataset-protocol-2026-11-10.md)
(commit `d4e25c7`) with its Amendment 1 (`8720f86`) and Amendment 2 (`ab833a3`). This is a dataset build:
no model was fitted and no performance score of any kind was computed. Every number below is read from the
executed outputs (local, not in Git), named in brackets.

Three builds exist. v3 is the final dataset (Amendment 2, ruling 4); v1 and v2 stay on disk as the record of
what changed, and their receipts pin earlier versions of the label code, which remain in Git.

| Build | Directory | Code commit | Rules |
|---|---|---|---|
| v1 | `outputs/mlc01_condition_dataset_20261110/` | `b6d3b6c` (labels `3044b72`) | protocol |
| v2 | `outputs/mlc01_condition_dataset_20261110_v2/` | `ee0481f` | + Amendment 1 |
| v3 (final) | `outputs/mlc01_condition_dataset_20261110_v3/` | `41845c2` | + Amendment 2 |

- Every receipt records its commit with no modified tracked file [`receipt.json` `git`].
- v1: `PYTHONPATH=. OPENBLAS_CORETYPE=Haswell OMP_NUM_THREADS=1 .venv/bin/python -m
  scripts.data.build_mlc01_dataset --workers 16` (471.4 s).
- v2 and v3: the same command with `--output-dir <v2 or v3 directory> --reference-dir
  outputs/mlc01_condition_dataset_20261110` (75.2 s and 75.3 s; the waveform scan was reused from the cache,
  whose chunks are bound to the unchanged scan code).
- Both rebuilds require the first build's records, roles and waveform hashes, and v3 also its raw paths, to
  be reproduced exactly. All 100,221 records match, with 0 mismatches
  [`receipt.json` `integrity`, `records_roles_paths_and_waveform_hashes_match_reference` in v3].
- A 100-records-per-source profile into `/var/tmp` preceded the v1 and v2 runs; neither is a result.

## Final dataset (v3)

99,167 of the 100,221 enumerated records are included. 63 are shorter than 10 s and 991 are removed by
the duplicate rule. No record was unreadable, had other than the twelve named leads, or was not 500 Hz
[`exclusion_flow.csv`].

| role | ptbxl | ningbo | chapman_shaoxing | georgia | cpsc_2018 | cpsc_2018_extra | sph |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 17,418 | 20,919 | 6,097 | 6,117 | 3,763 | 1,936 | 0 |
| validation | 2,183 | 6,972 | 2,036 | 2,043 | 1,251 | 653 | 0 |
| test_same_source | 2,198 | 0 | 0 | 0 | 0 | 0 | 0 |
| test_external | 0 | 0 | 0 | 0 | 0 | 0 | 12,698 |
| sph_calibration | 0 | 0 | 0 | 0 | 0 | 0 | 12,883 |

v3 includes the same records as v2 [`v3/comparison_with_before/record_set_changes.csv`, empty]. Only
CPSC-Extra changed from v1: 59 more training records and 22 more validation records. Their copies
had conflicted in v1 on labels that ruling 8 now makes undefined at CPSC 2018
[`comparison_with_before/record_set_changes.csv`].

Roles follow the protocol table:

- PTB-XL: folds 1-8 train, fold 9 validation, fold 10 same-source test.
- Challenge sources: the train and calibration groups.
- SPH: split into halves by `sha256('experiment042:20261018:' + Patient_ID).digest()[0] < 128`
  (calibration), applied to all 25,770 SPH records.

The Challenge test groups (13,172 records) were not opened.

## Integrity checks

All checks passed in both builds; the build stops on any failure [`receipt.json` `integrity`].

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
| v2: records, roles and waveform hashes equal v1 | 100,221, 0 mismatches | 100,221, 0 mismatches |
| v3: records, roles, raw paths and waveform hashes equal v1 | 100,221, 0 mismatches | 100,221, 0 mismatches |

Raw files:

- Every PTB-XL and Challenge header and signal file matched its release `SHA256SUMS.txt` digest: 21,799
  PTB-XL and 52,652 Challenge records.
- SPH publishes no per-record checksum, so the 25,770 HDF5 files were hashed into the manifest. The SPH
  metadata and code tables matched the download manifest's MD5.

The closed-data guard has three layers, all tested in `tests/test_mlc01_dataset.py`:

1. The split table is read with `usecols` limited to source, family, record, path and split.
2. Test rows are filtered out before any job is built.
3. Every worker refuses a path of the closed set.

## Amendment 1 as implemented

The rulings are in `ecg_experiment/mlc_labels.py`, each with a unit test in `tests/test_mlc_labels.py`.
Ruling 9 changes this report only.

1. A record with SNOMED 425856008 (paroxysmal VT) or 164895002 (VT) and no PVC code is undefined for
   premature ventricular complexes. Neither PTB-XL nor SPH has a VT statement.
2. A record with a "nonspecific or ischemic ST-T statement" (table below) and no ST-depression code is
   undefined for ST depression. At SPH, code 145 without a direction modifier and code 146 stay undefined
   as before.
3. A record with the same statements and no T-wave code is undefined for T-wave abnormality.
4. An empty PTB-XL `heart_axis` (coded `axis:missing`) or the value `AXL` is undefined for left axis
   deviation.
5. PTB-XL `HVOLT` alone is undefined for left ventricular hypertrophy. Under departure 4 it also counts as
   left ventricular high voltage for space-A HYP.
6. A source with fewer than 5 positives of a label among all its opened records does not annotate it.
   - Positives are counted after every other rule and before duplicate resolution.
   - The rule is checked for every label of spaces A, B and C in every source.
   - The full table is `minimum_positive_rule.csv`; the caught pairs are in the receipt (`amendment_1`).
7. LBBB with incomplete RBBB is a contradictory pair: both labels are undefined.
8. Space A:
   - SPH 153 counts for HYP. STTC is undefined when 153 is the record's only ST-T evidence, that is, when
     none of 145, 146, 147 and 148 is present.
   - Every CPSC 2018 record is undefined for every group unless positive.

"Nonspecific or ischemic ST-T statement" has the same meaning for rulings 2 and 3
[`receipt.json` `amendment_1.st_t_nonspecific_or_ischemic`]:

| Vocabulary | Codes |
|---|---|
| PTB-XL SCP | `NST_` (non-specific ST changes); the ischemic statements `ISC_`, `ISCAL`, `ISCAS`, `ISCLA`, `ISCAN`, `ISCIN`, `ISCIL` |
| SNOMED CT, nonspecific | 428750005 nonspecific ST-T abnormality, 55930002 ST changes, 164930006 ST interval abnormal |
| SNOMED CT, ischemic | 164861001 myocardial ischemia, 413844008 chronic myocardial ischemia, 413444003 acute myocardial ischemia, 426434006 anterior ischemia, 425419005 inferior ischaemia, 425623009 lateral ischaemia |

### Pairs caught by ruling 6

Ruling 6 caught 23 (source, label) pairs that had defined states before the rule
[`receipt.json` `amendment_1.minimum_positive_pairs_newly_undefined`]. A further 50 flagged pairs were
already undefined throughout, because the source does not annotate or cannot code the label.

| Space | Source | Label | Positives among opened records |
|---|---|---|---:|
| B | cpsc_2018_extra | left_axis_deviation | 0 |
| B | cpsc_2018_extra | lafb | 0 |
| B | georgia | mi_any | 3 |
| A | georgia | MI | 3 |
| C | cpsc_2018_extra | left_axis_deviation, lafb, lpfb, right_axis_deviation, low_qrs_voltage | 0 each |
| C | cpsc_2018_extra | paced_rhythm, long_qt | 3 each |
| C | cpsc_2018_extra | nonspecific_ivcd, preexcitation | 4 each |
| C | chapman_shaoxing | paced_rhythm, left_atrial_enlargement, ischemic_st_t, anterior_mi | 0 each |
| C | chapman_shaoxing | rvh | 4 |
| C | georgia | anterior_mi | 0 |
| C | georgia | preexcitation | 3 |
| C | ningbo | st_elevation, ischemic_st_t | 0 each |
| C | ningbo | lpfb | 4 |

In space B the rule catches exactly the three pairs the amendment names, and none at PTB-XL, SPH or CPSC
2018. v3 catches the same 23 pairs (Amendment 2, ruling 3: the rule applies to every label space).

## Amendment 2 as implemented

1. A PTB-XL record with a subendocardial injury (`INJAS`, `INJAL`, `INJLA`, `INJIN`, `INJIL`), digitalis
   (`DIG`), aneurysm (`ANEUR`) or electrolyte (`EL`) statement and no ST-depression code is undefined for ST
   depression (`ST_CHANGE_WITHOUT_DIRECTION_SCP`; test
   `test_amendment_2_ptbxl_st_changes_without_direction_leave_st_depression_undefined`).
2. Kept as built and stated as limits: SPH 145 without 146 or 147 stays negative for T-wave abnormality;
   CPSC 2018 stays defined for AF or flutter; space C keeps its other ambiguous negatives (ST elevation,
   right axis deviation). The test also checks that these states are unchanged.
3. Ruling 6 of Amendment 1 applies to every label space, as built.
4. The dataset is final after this rebuild.

Effect on records included in both v2 and v3: 383 PTB-XL ST-depression negatives became undefined (305
train, 41 fold 9, 37 fold 10), the same in space C; nothing else changed in spaces A, B or C
[`v3/comparison_with_before/state_transitions.csv`, `before_after_*.csv`].

## Exclusion flow (v3, identical to v2)

| source | start | unreadable | wrong leads | wrong rate | short | eligible | dropped copy | dropped conflict | final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ptbxl | 21,799 | 0 | 0 | 0 | 0 | 21,799 | 0 | 0 | 21,799 |
| ningbo | 27,920 | 0 | 0 | 0 | 0 | 27,920 | 3 | 26 | 27,891 |
| chapman_shaoxing | 8,197 | 0 | 0 | 0 | 0 | 8,197 | 38 | 26 | 8,133 |
| georgia | 8,273 | 0 | 0 | 0 | 43 | 8,230 | 70 | 0 | 8,160 |
| cpsc_2018 | 5,496 | 0 | 0 | 0 | 10 | 5,486 | 363 | 109 | 5,014 |
| cpsc_2018_extra | 2,766 | 0 | 0 | 0 | 10 | 2,756 | 33 | 134 | 2,589 |
| sph | 25,770 | 0 | 0 | 0 | 0 | 25,770 | 147 | 42 | 25,581 |

No copy crossed roles, so the rule that removes a training or validation copy of a test record removed
nothing. Ningbo JS13118 is not in the split table (its waveform is absent upstream), so it never entered.
Per-role rows are in `exclusion_flow.csv`.

## Patients and record groups

[`group_counts.csv`; identical in both builds]

| role | source | records | groups | groups with several records | records in those groups | max per group |
|---|---|---:|---:|---:|---:|---:|
| train | ptbxl | 17,418 | 15,023 | 1,694 | 4,089 | 10 |
| validation | ptbxl | 2,183 | 1,942 | 189 | 430 | 8 |
| test_same_source | ptbxl | 2,198 | 1,904 | 228 | 522 | 6 |
| test_external | sph | 12,698 | 12,246 | 439 | 891 | 4 |
| sph_calibration | sph | 12,883 | 12,400 | 469 | 952 | 4 |

The Challenge sources have no patient identifier, so each record is its own group.

## Space B by evaluation set (v3)

Cells are positive / negative / undefined records. `chapman_ningbo` and `cpsc` are the two-source
families [`support_B.csv`].

| condition | val ptbxl | val ningbo | val chapman | val chapman_ningbo | val georgia | val cpsc_2018 | val cpsc_extra | val cpsc | test ptbxl | test sph | sph calibration |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sinus_tachycardia | 83 / 2,100 / 0 | 1,145 / 5,827 / 0 | 314 / 1,722 / 0 | 1,459 / 7,549 / 0 | 265 / 1,778 / 0 | 0 / 0 / 1,251 | 65 / 588 / 0 | 65 / 588 / 1,251 | 82 / 2,116 / 0 | 360 / 12,338 / 0 | 357 / 12,526 / 0 |
| sinus_bradycardia | 64 / 2,119 / 0 | 2,510 / 4,461 / 1 | 769 / 1,267 / 0 | 3,279 / 5,728 / 1 | 318 / 1,725 / 0 | 0 / 0 / 1,251 | 6 / 598 / 49 | 6 / 598 / 1,300 | 64 / 2,134 / 0 | 1,302 / 11,396 / 0 | 1,393 / 11,490 / 0 |
| sinus_arrhythmia | 77 / 2,106 / 0 | 529 / 6,443 / 0 | 0 / 0 / 2,036 | 529 / 6,443 / 2,036 | 93 / 1,950 / 0 | 0 / 0 / 1,251 | 2 / 651 / 0 | 2 / 651 / 1,251 | 77 / 2,121 / 0 | 781 / 11,917 / 0 | 754 / 12,129 / 0 |
| af_or_flutter | 156 / 2,027 / 0 | 1,536 / 5,436 / 0 | 454 / 1,582 / 0 | 1,990 / 7,018 / 0 | 133 / 1,910 / 0 | 213 / 1,038 / 0 | 49 / 604 / 0 | 262 / 1,642 / 0 | 157 / 2,041 / 0 | 379 / 12,319 / 0 | 384 / 12,499 / 0 |
| premature_ventricular_complex | 114 / 2,063 / 6 | 220 / 6,718 / 34 | 64 / 1,972 / 0 | 284 / 8,690 / 34 | 66 / 1,977 / 0 | 113 / 1,138 / 0 | 27 / 626 / 0 | 140 / 1,764 / 0 | 114 / 2,075 / 9 | 517 / 12,181 / 0 | 541 / 12,342 / 0 |
| first_degree_av_block | 104 / 2,079 / 0 | 189 / 6,783 / 0 | 50 / 1,986 / 0 | 239 / 8,769 / 0 | 149 / 1,894 / 0 | 112 / 1,139 / 0 | 20 / 633 / 0 | 132 / 1,772 / 0 | 96 / 2,102 / 0 | 112 / 12,586 / 0 | 123 / 12,760 / 0 |
| complete_rbbb | 54 / 2,128 / 1 | 256 / 6,659 / 57 | 87 / 1,943 / 6 | 343 / 8,602 / 63 | 104 / 1,911 / 28 | 334 / 917 / 0 | 20 / 625 / 8 | 354 / 1,542 / 8 | 54 / 2,144 / 0 | 350 / 12,348 / 0 | 349 / 12,533 / 1 |
| incomplete_rbbb | 111 / 2,071 / 1 | 58 / 6,914 / 0 | 0 / 0 / 2,036 | 58 / 6,914 / 2,036 | 74 / 1,959 / 10 | 0 / 0 / 1,251 | 5 / 639 / 9 | 5 / 639 / 1,260 | 112 / 2,086 / 0 | 619 / 12,079 / 0 | 626 / 12,256 / 1 |
| lafb | 163 / 2,020 / 0 | 76 / 6,896 / 0 | 0 / 0 / 2,036 | 76 / 6,896 / 2,036 | 45 / 1,998 / 0 | 0 / 0 / 1,251 | 0 / 0 / 653 | 0 / 0 / 1,904 | 162 / 2,036 / 0 | 77 / 12,621 / 0 | 74 / 12,809 / 0 |
| left_axis_deviation | 121 / 751 / 1,311 | 216 / 6,756 / 0 | 80 / 1,956 / 0 | 296 / 8,712 / 0 | 190 / 1,853 / 0 | 0 / 0 / 1,251 | 0 / 0 / 653 | 0 / 0 / 1,904 | 134 / 704 / 1,360 | 68 / 12,630 / 0 | 70 / 12,813 / 0 |
| lvh | 210 / 1,940 / 33 | 103 / 6,097 / 772 | 4 / 1,761 / 271 | 107 / 7,858 / 1,043 | 231 / 1,803 / 9 | 0 / 0 / 1,251 | 33 / 620 / 0 | 33 / 620 / 1,251 | 214 / 1,959 / 25 | 100 / 12,598 / 0 | 107 / 12,776 / 0 |
| st_depression | 101 / 1,772 / 310 | 231 / 5,967 / 774 | 78 / 1,807 / 151 | 309 / 7,774 / 925 | 0 / 0 / 2,043 | 161 / 1,090 / 0 | 8 / 222 / 423 | 169 / 1,312 / 423 | 101 / 1,797 / 300 | 497 / 11,293 / 908 | 513 / 11,461 / 909 |
| t_wave_abnormality | 260 / 1,638 / 285 | 1,305 / 4,931 / 736 | 390 / 1,486 / 160 | 1,695 / 6,417 / 896 | 544 / 1,110 / 389 | 0 / 0 / 1,251 | 5 / 225 / 423 | 5 / 225 / 1,674 | 255 / 1,664 / 279 | 1,633 / 11,065 / 0 | 1,618 / 11,265 / 0 |
| mi_any | 521 / 1,643 / 19 | 26 / 6,946 / 0 | 6 / 2,030 / 0 | 32 / 8,976 / 0 | 0 / 0 / 2,043 | 0 / 0 / 1,251 | 274 / 379 / 0 | 274 / 379 / 1,251 | 532 / 1,648 / 18 | 118 / 12,580 / 0 | 137 / 12,746 / 0 |

Training records, positive / negative / undefined [`label_counts_B.csv`]:

| condition | ptbxl | ningbo | chapman_shaoxing | georgia | cpsc_2018 | cpsc_2018_extra |
|---|---:|---:|---:|---:|---:|---:|
| sinus_tachycardia | 661 / 16,757 / 0 | 3,419 / 17,500 / 0 | 922 / 5,175 / 0 | 745 / 5,372 / 0 | 0 / 0 / 3,763 | 160 / 1,776 / 0 |
| sinus_bradycardia | 509 / 16,909 / 0 | 7,615 / 13,300 / 4 | 2,312 / 3,785 / 0 | 954 / 5,159 / 4 | 0 / 0 / 3,763 | 29 / 1,757 / 150 |
| sinus_arrhythmia | 618 / 16,800 / 0 | 1,513 / 19,406 / 0 | 0 / 0 / 6,097 | 264 / 5,853 / 0 | 0 / 0 / 3,763 | 4 / 1,932 / 0 |
| af_or_flutter | 1,257 / 16,161 / 0 | 4,532 / 16,387 / 0 | 1,336 / 4,761 / 0 | 441 / 5,676 / 0 | 587 / 3,176 / 0 | 113 / 1,823 / 0 |
| premature_ventricular_complex | 915 / 16,456 / 47 | 656 / 20,173 / 90 | 176 / 5,921 / 0 | 239 / 5,878 / 0 | 364 / 3,399 / 0 | 103 / 1,832 / 1 |
| first_degree_av_block | 808 / 16,610 / 0 | 576 / 20,343 / 0 | 161 / 5,936 / 0 | 443 / 5,674 / 0 | 418 / 3,345 / 0 | 51 / 1,885 / 0 |
| complete_rbbb | 432 / 16,986 / 0 | 765 / 19,912 / 242 | 238 / 5,834 / 25 | 285 / 5,743 / 89 | 950 / 2,813 / 0 | 31 / 1,878 / 27 |
| incomplete_rbbb | 894 / 16,524 / 0 | 140 / 20,779 / 0 | 0 / 0 / 6,097 | 196 / 5,890 / 31 | 0 / 0 / 3,763 | 13 / 1,897 / 26 |
| lafb | 1,298 / 16,120 / 0 | 227 / 20,692 / 0 | 0 / 0 / 6,097 | 102 / 6,015 / 0 | 0 / 0 / 3,763 | 0 / 0 / 1,936 |
| left_axis_deviation | 1,127 / 6,629 / 9,662 | 708 / 20,211 / 0 | 225 / 5,872 / 0 | 556 / 5,561 / 0 | 0 / 0 / 3,763 | 0 / 0 / 1,936 |
| lvh | 1,708 / 15,501 / 209 | 392 / 18,031 / 2,496 | 7 / 5,370 / 720 | 763 / 5,336 / 18 | 0 / 0 / 3,763 | 86 / 1,850 / 0 |
| st_depression | 807 / 14,204 / 2,407 | 786 / 17,782 / 2,351 | 242 / 5,343 / 512 | 0 / 0 / 6,117 | 499 / 3,264 / 0 | 34 / 651 / 1,251 |
| t_wave_abnormality | 2,079 / 13,108 / 2,231 | 3,951 / 14,753 / 2,215 | 1,156 / 4,418 / 523 | 1,621 / 3,308 / 1,188 | 0 / 0 / 3,763 | 16 / 660 / 1,260 |
| mi_any | 4,235 / 13,039 / 144 | 97 / 20,822 / 0 | 25 / 6,072 / 0 | 0 / 0 / 6,117 | 0 / 0 / 3,763 | 827 / 1,109 / 0 |

The protocol's selection rule still holds on v3: each of the 14 conditions has at least 100 positives in
each of the three sets below.

| Set | Smallest condition | Positives |
|---|---|---:|
| training pool | incomplete RBBB | 1,243 |
| SPH | left axis deviation | 138 |
| PTB-XL folds 9-10 | complete RBBB | 108 |

### Conditions below 20, 50 and 100 positives (v3)

The macro rule admits a condition with at least 20 positives and 20 negatives in the set [`support_B.csv`].
"Below 20" includes sources that do not annotate the condition, which have 0 positives.

| set | below 20 | 20-49 | 50-99 | in macro (of 14) |
|---|---|---|---|---:|
| validation ptbxl | | | complete_rbbb, sinus_arrhythmia, sinus_bradycardia, sinus_tachycardia | 14 |
| validation ningbo | | mi_any | incomplete_rbbb, lafb | 14 |
| validation chapman_shaoxing | incomplete_rbbb, lafb, lvh, mi_any, sinus_arrhythmia | | complete_rbbb, first_degree_av_block, left_axis_deviation, premature_ventricular_complex, st_depression | 9 |
| validation chapman_ningbo | | mi_any | incomplete_rbbb, lafb | 14 |
| validation georgia | mi_any, st_depression | lafb | incomplete_rbbb, premature_ventricular_complex, sinus_arrhythmia | 12 |
| validation cpsc_2018 | incomplete_rbbb, lafb, left_axis_deviation, lvh, mi_any, sinus_arrhythmia, sinus_bradycardia, sinus_tachycardia, t_wave_abnormality | | | 5 |
| validation cpsc_2018_extra | incomplete_rbbb, lafb, left_axis_deviation, sinus_arrhythmia, sinus_bradycardia, st_depression, t_wave_abnormality | af_or_flutter, complete_rbbb, first_degree_av_block, lvh, premature_ventricular_complex | sinus_tachycardia | 7 |
| validation cpsc | incomplete_rbbb, lafb, left_axis_deviation, sinus_arrhythmia, sinus_bradycardia, t_wave_abnormality | lvh | sinus_tachycardia | 8 |
| test ptbxl (fold 10) | | | complete_rbbb, first_degree_av_block, sinus_arrhythmia, sinus_bradycardia, sinus_tachycardia | 14 |
| test sph (evaluation half) | | | lafb, left_axis_deviation | 14 |
| sph calibration half | | | lafb, left_axis_deviation | 14 |

All 14 conditions still enter the macro at PTB-XL fold 10 and at the SPH evaluation half
[`comparison_with_before/before_after_B.csv`].

Fewer than 20 negatives occurs in v3 only where a source does not annotate the condition at all (0
positives and 0 negatives). No set has a condition with positives but fewer than 20 negatives.

### Likelihood sensitivity

In the sensitivity analysis, a PTB-XL positive that rests only on diagnostic statements with likelihood
below 50 becomes undefined. The counts are unchanged in v2 and v3 [`label_counts_B.csv` minus
`label_counts_B_lik50.csv`]:

| condition | train | fold 9 | fold 10 |
|---|---:|---:|---:|
| myocardial infarction | 1,090 | 131 | 141 |
| left ventricular hypertrophy | 297 | 47 | 37 |
| first-degree AV block | 3 | 0 | 0 |
| incomplete RBBB | 3 | 0 | 0 |
| complete RBBB | 1 | 0 | 0 |
| LAFB | 0 | 1 | 0 |

No other condition is affected. In fold 10 this leaves 391 MI and 177 LVH positives
[`support_B_lik50.csv`].

## Before and after the amendments

[`v2/comparison_with_before/` (v1 to v2) and `v3/comparison_with_before/` (v2 to v3), from
`scripts/reports/compare_mlc01_builds.py`; v1 to v3 below chains the two. No support flag changed between v2
and v3.]

Space B in the two test sets, positive / negative / undefined:

| condition | fold 10 v1 | fold 10 v3 | SPH evaluation v1 | SPH evaluation v3 |
|---|---:|---:|---:|---:|
| sinus_tachycardia | 82 / 2,116 / 0 | 82 / 2,116 / 0 | 360 / 12,338 / 0 | 360 / 12,338 / 0 |
| sinus_bradycardia | 64 / 2,134 / 0 | 64 / 2,134 / 0 | 1,302 / 11,396 / 0 | 1,302 / 11,396 / 0 |
| sinus_arrhythmia | 77 / 2,121 / 0 | 77 / 2,121 / 0 | 781 / 11,917 / 0 | 781 / 11,917 / 0 |
| af_or_flutter | 157 / 2,041 / 0 | 157 / 2,041 / 0 | 379 / 12,319 / 0 | 379 / 12,319 / 0 |
| premature_ventricular_complex | 114 / 2,075 / 9 | 114 / 2,075 / 9 | 517 / 12,181 / 0 | 517 / 12,181 / 0 |
| first_degree_av_block | 96 / 2,102 / 0 | 96 / 2,102 / 0 | 112 / 12,586 / 0 | 112 / 12,586 / 0 |
| complete_rbbb | 54 / 2,144 / 0 | 54 / 2,144 / 0 | 350 / 12,348 / 0 | 350 / 12,348 / 0 |
| incomplete_rbbb | 112 / 2,086 / 0 | 112 / 2,086 / 0 | 619 / 12,079 / 0 | 619 / 12,079 / 0 |
| lafb | 162 / 2,036 / 0 | 162 / 2,036 / 0 | 77 / 12,621 / 0 | 77 / 12,621 / 0 |
| left_axis_deviation | 134 / 1,670 / 394 | 134 / 704 / 1,360 | 68 / 12,630 / 0 | 68 / 12,630 / 0 |
| lvh | 214 / 1,964 / 20 | 214 / 1,959 / 25 | 100 / 12,598 / 0 | 100 / 12,598 / 0 |
| st_depression | 101 / 2,097 / 0 | 101 / 1,797 / 300 | 497 / 11,293 / 908 | 497 / 11,293 / 908 |
| t_wave_abnormality | 255 / 1,943 / 0 | 255 / 1,664 / 279 | 1,633 / 11,065 / 0 | 1,633 / 11,065 / 0 |
| mi_any | 532 / 1,648 / 18 | 532 / 1,648 / 18 | 118 / 12,580 / 0 | 118 / 12,580 / 0 |

- No positive changed in either test set.
- Space B at the SPH evaluation half is unchanged.
- At fold 10, these negatives became undefined: 966 left axis deviation, 300 ST depression (263 under
  Amendment 1 and 37 under Amendment 2), 279 T-wave abnormality and 5 LVH.
- Space A in the test sets [`before_after_A.csv`]:
  - SPH evaluation-half STTC changed from 2,480 / 10,218 / 0 to 2,480 / 10,176 / 42.
  - Fold 10 HYP changed from 262 / 1,917 / 19 to 262 / 1,912 / 24.

History v1 -> v2 -> v3 of every space-B cell that changed in a test set or a per-source validation set
(positive / negative / undefined). The SPH evaluation half changed in no build.

| set | condition | v1 | v2 | v3 |
|---|---|---:|---:|---:|
| test ptbxl (fold 10) | left_axis_deviation | 134 / 1,670 / 394 | 134 / 704 / 1,360 | 134 / 704 / 1,360 |
| test ptbxl (fold 10) | lvh | 214 / 1,964 / 20 | 214 / 1,959 / 25 | 214 / 1,959 / 25 |
| test ptbxl (fold 10) | st_depression | 101 / 2,097 / 0 | 101 / 1,834 / 263 | 101 / 1,797 / 300 |
| test ptbxl (fold 10) | t_wave_abnormality | 255 / 1,943 / 0 | 255 / 1,664 / 279 | 255 / 1,664 / 279 |
| validation ptbxl | left_axis_deviation | 121 / 1,694 / 368 | 121 / 751 / 1,311 | 121 / 751 / 1,311 |
| validation ptbxl | lvh | 210 / 1,946 / 27 | 210 / 1,940 / 33 | 210 / 1,940 / 33 |
| validation ptbxl | st_depression | 101 / 2,082 / 0 | 101 / 1,813 / 269 | 101 / 1,772 / 310 |
| validation ptbxl | t_wave_abnormality | 260 / 1,923 / 0 | 260 / 1,638 / 285 | 260 / 1,638 / 285 |
| validation ningbo | premature_ventricular_complex | 220 / 6,730 / 22 | 220 / 6,718 / 34 | 220 / 6,718 / 34 |
| validation ningbo | st_depression | 231 / 6,741 / 0 | 231 / 5,967 / 774 | 231 / 5,967 / 774 |
| validation ningbo | t_wave_abnormality | 1,305 / 5,020 / 647 | 1,305 / 4,931 / 736 | 1,305 / 4,931 / 736 |
| validation chapman_shaoxing | st_depression | 78 / 1,958 / 0 | 78 / 1,807 / 151 | 78 / 1,807 / 151 |
| validation georgia | mi_any | 0 / 2,043 / 0 | 0 / 0 / 2,043 | 0 / 0 / 2,043 |
| validation georgia | t_wave_abnormality | 544 / 1,218 / 281 | 544 / 1,110 / 389 | 544 / 1,110 / 389 |
| validation cpsc_2018_extra | af_or_flutter | 38 / 593 / 0 | 49 / 604 / 0 | 49 / 604 / 0 |
| validation cpsc_2018_extra | complete_rbbb | 18 / 606 / 7 | 20 / 625 / 8 | 20 / 625 / 8 |
| validation cpsc_2018_extra | first_degree_av_block | 18 / 613 / 0 | 20 / 633 / 0 | 20 / 633 / 0 |
| validation cpsc_2018_extra | incomplete_rbbb | 5 / 619 / 7 | 5 / 639 / 9 | 5 / 639 / 9 |
| validation cpsc_2018_extra | lafb | 0 / 631 / 0 | 0 / 0 / 653 | 0 / 0 / 653 |
| validation cpsc_2018_extra | left_axis_deviation | 0 / 631 / 0 | 0 / 0 / 653 | 0 / 0 / 653 |
| validation cpsc_2018_extra | lvh | 30 / 601 / 0 | 33 / 620 / 0 | 33 / 620 / 0 |
| validation cpsc_2018_extra | mi_any | 267 / 364 / 0 | 274 / 379 / 0 | 274 / 379 / 0 |
| validation cpsc_2018_extra | premature_ventricular_complex | 25 / 606 / 0 | 27 / 626 / 0 | 27 / 626 / 0 |
| validation cpsc_2018_extra | sinus_arrhythmia | 2 / 629 / 0 | 2 / 651 / 0 | 2 / 651 / 0 |
| validation cpsc_2018_extra | sinus_bradycardia | 6 / 576 / 49 | 6 / 598 / 49 | 6 / 598 / 49 |
| validation cpsc_2018_extra | sinus_tachycardia | 65 / 566 / 0 | 65 / 588 / 0 | 65 / 588 / 0 |
| validation cpsc_2018_extra | st_depression | 8 / 623 / 0 | 8 / 222 / 423 | 8 / 222 / 423 |
| validation cpsc_2018_extra | t_wave_abnormality | 5 / 410 / 216 | 5 / 225 / 423 | 5 / 225 / 423 |

The `chapman_ningbo` and `cpsc` family rows change accordingly [`before_after_B.csv`]. The CPSC-Extra
positive increases come from the 22 validation records the duplicate rule now keeps; no label rule added a
positive.

Support flags that changed:

- At CPSC-Extra validation, complete RBBB and first-degree AV block rose from 18 to 20 positives. Both now
  enter that source's macro, which grows from 5 to 7 conditions.
- Fewer than 20 negatives newly occurs for LAFB and left axis deviation at CPSC-Extra and in the CPSC
  family, and for MI at Georgia. These are the pairs that ruling 6 makes undefined throughout.

State changes on records included in both builds, spaces B and A, summed over roles
[`state_transitions.csv` of v2, plus v3 for Amendment 2]. All are negative to undefined, except 3 Georgia MI
positives and 1 incomplete
RBBB positive that became undefined:

| label | records | by source |
|---|---:|---|
| left_axis_deviation | 11,077 | PTB-XL 8,569, CPSC-Extra 2,508 |
| mi_any | 8,160 | Georgia (8,157 negatives, 3 positives) |
| st_depression | 8,018 + 383 | Ningbo 3,125, PTB-XL 2,634 + 383 (Amendment 2), CPSC-Extra 1,596, Chapman 663 |
| t_wave_abnormality | 4,318 | PTB-XL 2,795, CPSC-Extra 718, Georgia 449, Ningbo 356 |
| lafb | 2,508 | CPSC-Extra |
| lvh | 45 | PTB-XL |
| premature_ventricular_complex | 41 | Ningbo 40, CPSC-Extra 1 |
| incomplete_rbbb | 1 | CPSC-Extra (positive, LBBB with incomplete RBBB) |
| space-A MI | 8,160 | Georgia (8,157 negatives, 3 positives) |
| space-A STTC | 4,318 | CPSC 2018 4,235, SPH 83 |
| space-A CD | 3,056 | CPSC 2018 |
| space-A HYP | 45 | PTB-XL |

## Space A by evaluation set (v3, identical to v2)

Positive / negative / undefined [`support_A.csv`]:

| group | val ptbxl | val ningbo | val chapman | val georgia | val cpsc_2018 | val cpsc_extra | test ptbxl | test sph | sph calibration |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MI | 540 / 1,643 / 0 | 26 / 6,946 / 0 | 6 / 2,030 / 0 | 0 / 0 / 2,043 | 0 / 0 / 1,251 | 274 / 379 / 0 | 550 / 1,648 / 0 | 118 / 12,580 / 0 | 137 / 12,746 / 0 |
| STTC | 528 / 1,655 / 0 | 2,068 / 4,904 / 0 | 579 / 1,457 / 0 | 989 / 1,054 / 0 | 195 / 0 / 1,056 | 449 / 204 / 0 | 521 / 1,677 / 0 | 2,480 / 10,176 / 42 | 2,478 / 10,364 / 41 |
| CD | 495 / 1,688 / 0 | 769 / 6,203 / 0 | 256 / 1,780 / 0 | 460 / 1,583 / 0 | 486 / 0 / 765 | 72 / 581 / 0 | 496 / 1,702 / 0 | 1,166 / 11,532 / 0 | 1,200 / 11,683 / 0 |
| HYP | 268 / 1,882 / 33 | 129 / 6,072 / 771 | 5 / 1,760 / 271 | 425 / 1,618 / 0 | 0 / 0 / 1,251 | 45 / 608 / 0 | 262 / 1,912 / 24 | 104 / 12,594 / 0 | 123 / 12,760 / 0 |

## Space C (exploratory, v3)

Number of the 30 conditions below each support level, and number entering a macro [`support_C.csv`]:

| set | below 20 | below 50 | below 100 | in macro |
|---|---:|---:|---:|---:|
| validation ptbxl | 9 | 12 | 18 | 21 |
| validation ningbo | 8 | 10 | 14 | 22 |
| validation chapman_shaoxing | 13 | 17 | 24 | 17 |
| validation chapman_ningbo | 6 | 8 | 12 | 24 |
| validation georgia | 8 | 14 | 18 | 22 |
| validation cpsc_2018 | 22 | 24 | 24 | 8 |
| validation cpsc_2018_extra | 22 | 28 | 29 | 8 |
| validation cpsc | 19 | 22 | 23 | 11 |
| test ptbxl | 9 | 12 | 19 | 21 |
| test sph | 9 | 11 | 17 | 21 |
| sph calibration | 10 | 12 | 17 | 20 |

Positives per set for the 18 conditions outside space B. `-` marks a source that does not annotate the
condition (including through ruling 6) or cannot code it:

| condition | val ptbxl | val ningbo | val chapman | val georgia | val cpsc_2018 | val cpsc_extra | test ptbxl | test sph | sph calibration | train (all sources) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| anterior_mi | 282 | 12 | - | - | - | 11 | 282 | 69 | 76 | 2,337 |
| atrial_fibrillation | 151 | - | 354 | 105 | 213 | 33 | 152 | 321 | 345 | 3,286 |
| atrial_flutter | 7 | 1,536 | 100 | 31 | - | 9 | 7 | 58 | 39 | 4,991 |
| complete_lbbb | 54 | 48 | 34 | 46 | 42 | 7 | 54 | 36 | 46 | 944 |
| inferior_mi | 322 | - | - | - | - | - | 323 | 53 | 65 | 2,593 |
| ischemic_st_t | 250 | - | - | 312 | - | 113 | 248 | - | - | 3,202 |
| left_atrial_enlargement | 43 | 0 | - | 196 | - | 9 | 42 | 2 | 17 | 929 |
| long_qt | 12 | 64 | 11 | 305 | - | - | 11 | 11 | 13 | 1,138 |
| low_qrs_voltage | 19 | 153 | 47 | 71 | - | - | 18 | 149 | 172 | 1,002 |
| lpfb | 18 | - | - | 9 | - | - | 18 | 5 | 1 | 152 |
| nonspecific_ivcd | 78 | 118 | 52 | 44 | - | - | 79 | - | - | 1,181 |
| paced_rhythm | 29 | 243 | - | 21 | - | - | 28 | - | - | 1,004 |
| preexcitation | 7 | 11 | 3 | - | - | - | 8 | 8 | 19 | 120 |
| premature_atrial_complex | 40 | 228 | 57 | 135 | 100 | 21 | 40 | 269 | 269 | 1,888 |
| right_axis_deviation | 15 | 127 | 47 | 13 | - | - | 9 | 72 | 89 | 664 |
| rvh | 12 | 25 | - | 10 | - | 4 | 12 | 2 | 4 | 230 |
| st_elevation | 3 | - | 33 | 24 | 34 | 14 | 3 | 20 | 17 | 334 |
| supraventricular_tachycardia | 5 | 70 | 145 | 15 | - | 6 | 5 | 5 | 8 | 681 |

The other 12 conditions of space C share space B's definitions and counts. Space C is exploratory and keeps
ambiguous negatives that spaces A and B no longer have (see Limitations).

## Normal column

The per-source normal definition is kept for audits only and is unchanged in v2 and v3
[`label_counts_normal.csv`]:

| Source | Normal definition | Count |
|---|---|---|
| PTB-XL | NORM, with SR as the only other statement | fold 10: 701 of 2,198 |
| Challenge | sinus rhythm as the only code | |
| SPH | code 1 alone | evaluation half: 6,850 of 12,698 |
| CPSC-Extra | undefined throughout | |

For CPSC-Extra, the inventory found no record with sinus rhythm as the only code
(`inventory_comparison/inventory_snapshot/label_space_A.csv` in the v1 directory).

## Co-occurrence (v3, identical to v2)

Share of records by number of positive space-B conditions. Training rows are shown, plus the SPH
evaluation half; `cooccurrence_B_count.csv` holds every role:

| source | 0 | 1 | 2 | 3 | 4 | 5 or more |
|---|---:|---:|---:|---:|---:|---:|
| ptbxl | 41.8% | 31.5% | 16.1% | 7.3% | 2.6% | 0.8% |
| ningbo | 17.1% | 54.6% | 20.5% | 6.1% | 1.6% | 0.2% |
| chapman_shaoxing | 20.9% | 54.0% | 18.8% | 5.6% | 0.7% | 0.1% |
| georgia | 29.7% | 41.1% | 22.0% | 6.0% | 1.0% | 0.2% |
| cpsc_2018 | 27.6% | 70.0% | 2.4% | 0% | 0% | 0% |
| cpsc_2018_extra | 40.7% | 44.2% | 13.8% | 1.2% | 0.1% | 0% |
| sph (test_external) | 58.5% | 30.6% | 9.2% | 1.6% | 0.2% | 0.0% |

Most frequent pairs in the training rows [`cooccurrence_B_pairs.csv`]:

| Source | Pair | Records | Share |
|---|---|---:|---:|
| PTB-XL | left axis deviation with MI | 622 | 3.6% |
| PTB-XL | LAFB with left axis deviation | 621 | 3.6% |
| Ningbo | AF or flutter with T-wave abnormality | 1,556 | 7.4% |
| Chapman | AF or flutter with T-wave abnormality | 469 | 7.7% |
| Georgia | LVH with T-wave abnormality | 226 | 3.7% |
| CPSC-Extra | sinus tachycardia with MI | 83 | 4.3% |
| SPH evaluation half | sinus bradycardia with T-wave abnormality | 240 | 1.9% |

The full 91-pair tables per role and source, and the same tables for space C, are in the CSV files.

## Flat and missing leads

Lead flags in the first 10 s of the final records [`lead_flags.csv`]:

| Source | Records with at least one flat lead | Records with missing samples |
|---|---|---|
| Ningbo | 919 training, 283 validation | 63 training, 19 validation |
| Chapman | 13 | 0 |
| Georgia | 6 | 3 training, 2 validation |
| CPSC 2018 | 9 | 0 |
| CPSC-Extra | 5 | 0 |
| PTB-XL | 1 | 0 |
| SPH | 0 | 0 |

One CPSC-Extra validation record is flat in all twelve leads. No record was excluded for quality; the
per-lead flags are manifest columns for the subgroup analysis.

## Duplicates (v3, identical to v2)

Exact duplicates are records with the same hash of the first 10 s (int32 microvolts, leads I to V6).
There are 806 groups of more than one eligible record, holding 1,635 records; they are the same groups as
in v1 [`duplicate_accounting.csv`]. In every group the whole-record hashes are also identical
[`receipt.json` `summary`].

| scope | sources | role | groups | records | kept | dropped copy | dropped conflict |
|---|---|---|---:|---:|---:|---:|---:|
| across sources | chapman_shaoxing+ningbo | train | 43 | 86 | 22 | 22 | 42 |
| across sources | chapman_shaoxing+ningbo | validation | 11 | 22 | 6 | 6 | 10 |
| across sources | cpsc_2018+cpsc_2018_extra | train | 203 | 418 | 122 | 124 | 172 |
| across sources | cpsc_2018+cpsc_2018_extra | validation | 71 | 146 | 49 | 51 | 46 |
| within source | chapman_shaoxing | train / validation | 7 / 3 | 14 / 6 | 7 / 3 | 7 / 3 | 0 / 0 |
| within source | cpsc_2018 | train / validation | 138 / 46 | 277 / 95 | 138 / 46 | 139 / 49 | 0 / 0 |
| within source | cpsc_2018_extra | train / validation | 35 / 10 | 71 / 20 | 25 / 8 | 25 / 8 | 21 / 4 |
| within source | georgia | train / validation | 52 / 16 | 106 / 32 | 52 / 16 | 54 / 16 | 0 / 0 |
| within source | ningbo | train / validation | 1 / 2 | 2 / 4 | 1 / 2 | 1 / 2 | 0 / 0 |
| within source | sph | sph_calibration | 82 | 164 | 70 | 70 | 24 |
| within source | sph | test_external | 86 | 172 | 77 | 77 | 18 |

- No group spans two roles, so no record was removed as a copy of a test or validation record.
- The opened groups contain 54 Chapman-Ningbo pairs and 274 CPSC 2018-CPSC-Extra groups. The CPSC groups
  hold 284 CPSC 2018 and 280 CPSC-Extra records [`supplementary/duplicates.json`]. Under ruling 9, this
  accounting replaces the protocol's earlier figure of 155 CPSC-Extra copies.
- Conflicts [`supplementary/duplicate_conflicts.csv`, `duplicates.json`]:
  - 162 groups conflict in v2 and v3, down from 243 in v1.
  - 103 of them pair CPSC 2018 with CPSC-Extra, down from 184. In 3 of these every conflicting label is
    one CPSC 2018 annotates only in part, AF or flutter.
  - The labels that conflict most often are premature ventricular complexes (39 groups), premature
    atrial complexes in space C (25) and LBBB in space C (24).
- All 168 SPH groups are pairs from one patient. 25 of them differ in their raw AHA codes and 21 conflict
  in the mapped labels. On the 17,402 records of the earlier SPH baseline table, the MLC-01 statuses agree
  with that table's frozen SPH rule (100 kept, 17,302 unique).
- Two cheap near-duplicate checks were run; they change no rule and found nothing
  [`near_duplicates_shape.csv`, `near_duplicates_leads.csv`]:
  - 0 groups share the sign pattern of successive differences, which ignores gain and offset, while
    having different exact hashes.
  - 0 record pairs share six or more identical non-flat leads without being exact copies.

## Comparison with the inventory (v1)

This section compares the first build with the metadata inventory [`inventory_comparison/` in the v1
directory, from `scripts/reports/compare_mlc01_inventory.py`]. The section "Before and after Amendment 1"
gives the differences between v1 and the later builds.

1. Reproduction. The draft mapping (`vocabulary.py`), re-applied to the inventory's record sets,
   reproduces all 576 inventory rows exactly, with 0 mismatches in records, positives or undefined.
   - For the Challenge groups, the inventory used the split's `unique` and `kept` records.
   - Those sets were rebuilt from the whole-record hashes with the frozen Ningbo-family rule: 51,347
     unique and 397 kept, 51,744 in total, the inventory's count.
   - No split-table column beyond identity and group was read.
2. Rule changes on the same records. Every changed state is attributed to a protocol rule; 0 are
   unexplained [`rule_transitions.csv`]. Counts are records in the inventory's sets:

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

3. Record sets [`record_set_changes.csv`]:
   - PTB-XL is unchanged.
   - SPH loses 94 calibration-half and 95 evaluation-half records to duplicate resolution; the inventory
     kept every SPH record.
   - Georgia loses 40 short records, CPSC 2018 10 and CPSC-Extra 10; the inventory applied no duration
     rule.
   - The duplicate rules differ: the split rule keeps the lowest record name, MLC-01 keeps the copy with
     the most defined labels.
   - In v1, 93 CPSC-Extra and 29 Ningbo records are therefore in MLC-01 but not in the inventory sets.
     Conversely, 50 Chapman, 47 CPSC 2018 and 3 CPSC-Extra records are in the inventory sets but not in
     MLC-01.
   - For example, CPSC-Extra training MI positives are 773 in the inventory and 803 in v1, with no rule
     change (`counts.csv`).

## Departures from the protocol

All are dated 10 November 2026, and none changes a decision of the protocol. Departures 1-7 were accepted
with Amendment 1.

1. SPH files store neither lead names nor the sampling rate. The documented SPH lead order and 500 Hz are
   used, so "twelve leads identified by name" is checked by name only for PTB-XL and the Challenge
   sources.
2. "Exact waveform hashes" was read as the hash of the first 10 s, the input every model sees; this is the
   more inclusive reading. No group differs in its whole-record hash, so the outcome equals the
   whole-record reading.
3. "Conflicting label sets" was read as one copy positive and another negative for any label of spaces A,
   B or C or the normal column; undefined agrees with anything.
   - Without a conflict, the copy with the most defined labels is kept (ties: lowest record key).
   - The rule is applied inside every role, the test sets included.
   - Copies are removed from the less protected role first, in the order tests, validation, SPH
     calibration, training. This step removed nothing.
4. Space A keeps the existing definitions, with SPH 153 moved to HYP. CPSC 2018 MI and HYP are undefined,
   and HYP is undefined when the only hypertrophy evidence is left ventricular high voltage, as for LVH.
   Amendment 1 refines this: ruling 8 makes every CPSC 2018 group positive or undefined, and ruling 5 adds
   `HVOLT` to left ventricular high voltage.
5. CPSC-Extra "has no normal record", so its normal column is undefined rather than 0.
6. In space C:
   - Conditions shared with space B use B's definitions.
   - Anterior MI is undefined on SNOMED codes that state no territory (164865005, 164867002, 57054005).
   - LPFB is not annotated at Chapman, which uses no fascicular-block code; this is the draft's own reason
     for LAFB.
   - Right ventricular hypertrophy keeps the draft's undefined state for SPH 153.
7. The likelihood sensitivity flag applies to diagnostic statements only. PTB-XL form and rhythm statements
   carry likelihood 0, meaning none was given, and are not treated as below 50.

Two scope choices made in applying Amendment 1:

8. Ruling 6 is applied to every label of spaces A, B and C, not only space B. Positives are counted over
   all opened records of a source, before exclusions and duplicate resolution.
9. Rulings 2-5 are applied to the conditions they name. Their space-C relatives (ST elevation, right axis
   deviation) are left as before; Amendment 2 keeps them as limits.
10. The v3 reference check compares raw paths with the first build as well as records, roles and waveform
    hashes (the v2 check compared records, roles and hashes).

## Questions settled by Amendment 2

The questions raised after the second build are settled; counts are final v3 records
[`v3/supplementary/open_questions.csv`]:

- PTB-XL ST depression with injury, digitalis, aneurysm or electrolyte statements: now undefined (383
  records); the question no longer appears in v3.
- Kept as built, as limits (Amendment 2, ruling 2):
  - SPH T-wave negatives with code 145 and no 146 or 147: 857 in the calibration half and 844 in the
    evaluation half (379 and 386 with 145 without a direction).
  - CPSC 2018 AF or flutter negatives although flutter is not a CPSC 2018 class: 3,176 training and 1,038
    validation.
  - Space C, ST elevation negatives with a nonspecific or ischemic ST-T statement: PTB-XL 2,542 / 323 / 316
    (train / validation / test), Georgia 2,228 / 759, CPSC-Extra 1,259 / 424, Chapman 698 / 205.
  - Space C, right axis deviation negatives with an empty PTB-XL `heart_axis` or `AXR`: 6,614 / 944 / 961.

The v1 and v2 receipts pin the label and build code of their own commits (`b6d3b6c`, `ee0481f`); those
versions remain in Git. Under Amendment 2, ruling 4, any later label question is logged as a limitation and
not applied.

## Limitations

- The labels are annotation statements, not diagnoses, and coding practice differs between sources.
  Validation by a cardiologist is pending.
- The dataset is final. The negatives kept by Amendment 2, ruling 2 (SPH 145 for T-wave abnormality, CPSC
  2018 AF or flutter, space-C ST elevation and right axis deviation) are known to be less reliable than the
  other negatives; counts are under "Questions settled by Amendment 2".
- SPH and PTB-XL fold 10 have been scored by earlier experiments for other targets.
- The Challenge validation groups can share patients with the training groups, since there is no patient
  identifier.
- Only exact duplicates are detected. Copies with any changed sample are not, and neither cheap
  near-duplicate check found one.
- The inventory directory was a session scratchpad. Its aggregate tables and draft vocabulary are kept,
  with their SHA-256, in `inventory_comparison/inventory_snapshot/` of the v1 directory.

## Files

Each build directory (local) holds:

- `manifest.csv`: one row per enumerated record, with identity, role, group, raw path, duration, file and
  waveform hashes, lead flags, exclusion and duplicate status, raw codes, label states (1, 0 or empty for
  undefined), `lik50_` flags and the normal column.
- `waveform_scan.parquet`: the scan output, including the per-lead and shape hashes.
- `mapping.csv`, `annotation_status.csv` and the count tables named above.
- `receipt.json`: input, code and output SHA-256, git commit and integrity checks.
- `build.log`.

v2 and v3 also hold `minimum_positive_rule.csv`, `comparison_with_before/` (v1 to v2, and v2 to v3) and
`supplementary/`. v1 also holds `inventory_comparison/` and `supplementary/`. The scan cache is `/var/tmp/janetrivera/mlc01_scan`; it is
chunked and bound to the job and code hashes.

Report scripts, run after a build from the worktree with `PYTHONPATH=.`:

- `python -m scripts.reports.compare_mlc01_inventory --inventory <labelspace directory>` (v1)
- `python -m scripts.reports.compare_mlc01_builds --before <v1> --after <v2>` and `--before <v2> --after <v3>`
- `python -m scripts.reports.describe_mlc01_duplicates --dataset <build>`
- `python -m scripts.reports.count_mlc01_label_doubts` (v1)
- `python -m scripts.reports.count_mlc01_open_questions --dataset <v2 or v3>`

Each script records its own SHA-256 and the manifest's in its receipt.

Checks:

- `ruff check ecg_experiment scripts tests conftest.py` passed.
- `pytest tests/test_mlc_labels.py tests/test_mlc01_dataset.py`: 44 passed at `41845c2`.
- The full suite, run once from the worktree at `41845c2` with the final report: 1,629 passed, 2 skipped,
  1 failed. The failing test is `tests/test_xecg_encoder_motion_v11.py::test_graph_rejects_missing_and_conflicting_leaves`;
  it rejects the worktree's symlinked `outputs` (it failed the same way at the v1 commit and passes in the main
  checkout).
