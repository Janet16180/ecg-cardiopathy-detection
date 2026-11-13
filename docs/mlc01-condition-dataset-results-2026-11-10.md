# MLC-01 results: the condition dataset

10 November 2026. This report executes the frozen [MLC-01 protocol](mlc01-condition-dataset-protocol-2026-11-10.md)
(commit `d4e25c7`) together with its three amendments: Amendment 1 (`8720f86`), Amendment 2 (`ab833a3`) and
Amendment 3 (`4a17ca1`). MLC-01 is a dataset build. No model was fitted and no performance score of any kind
was computed. Every number below comes from an executed output, named in brackets. The outputs are local and
not in Git.

Four builds exist. v4 is the final dataset: Amendment 3 makes it final, and no label rule changes from the
first model fit onward. v1 to v3 stay on disk as the record of what changed. Their receipts pin earlier
versions of the label code, and those versions remain in Git.

| Build | Directory | Code commit | Rules |
|---|---|---|---|
| v1 | `outputs/mlc01_condition_dataset_20261110/` | `b6d3b6c` (labels `3044b72`) | protocol |
| v2 | `outputs/mlc01_condition_dataset_20261110_v2/` | `ee0481f` | + Amendment 1 |
| v3 | `outputs/mlc01_condition_dataset_20261110_v3/` | `41845c2` | + Amendment 2 |
| v4 (final) | `outputs/mlc01_condition_dataset_20261110_v4/` | `3475e13` | + Amendment 3 |

- **Clean tree.** Every receipt records its commit with no modified tracked file [`receipt.json` `git`].
- **v1 command.** `PYTHONPATH=. OPENBLAS_CORETYPE=Haswell OMP_NUM_THREADS=1 .venv/bin/python -m
  scripts.data.build_mlc01_dataset --workers 16`, which ran in 471.4 s.
- **v2-v4 command.** The same, plus `--output-dir <build directory> --reference-dir
  outputs/mlc01_condition_dataset_20261110`. These runs took 75.2 s, 75.3 s and 77.7 s, because the
  waveform scan came from the cache, which is bound to the unchanged scan code.
- **Reference check.** Every rebuild must reproduce the first build's records, roles and waveform hashes
  exactly; v3 and v4 must also reproduce its raw paths. In v4 all 100,221 records match, with 0 mismatches
  [`receipt.json` `integrity.records_roles_paths_and_waveform_hashes_match_reference`].

## Final dataset (v4)

99,167 of 100,221 enumerated records are included. 63 are shorter than 10 s and 991 are removed by the
duplicate rule. No record was unreadable, had other than the twelve named leads, or was not 500 Hz
[`exclusion_flow.csv`].

| role | ptbxl | ningbo | chapman_shaoxing | georgia | cpsc_2018 | cpsc_2018_extra | sph |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 17,418 | 20,919 | 6,097 | 6,117 | 3,763 | 1,936 | 0 |
| validation | 2,183 | 6,972 | 2,036 | 2,043 | 1,251 | 653 | 0 |
| test_same_source | 2,198 | 0 | 0 | 0 | 0 | 0 | 0 |
| test_external | 0 | 0 | 0 | 0 | 0 | 0 | 12,698 |
| sph_calibration | 0 | 0 | 0 | 0 | 0 | 0 | 12,883 |

v4 includes exactly the same records as v3 and v2: `v4/comparison_with_before/record_set_changes.csv` is
empty. The only record-set change in the whole history came at v2: 81 more CPSC-Extra records (59 train, 22
validation), whose CPSC 2018 copies stopped conflicting under Amendment 1, ruling 8.

The roles follow the protocol table:

- PTB-XL: folds 1-8, 9 and 10.
- Challenge sources: the train and calibration groups.
- SPH: split into halves by `sha256('experiment042:20261018:' + Patient_ID).digest()[0] < 128`
  (calibration), applied to all 25,770 SPH records.

The Challenge test groups (13,172 records) were not opened.

## Integrity checks

All checks passed in every build, and the build stops on any failure [`receipt.json` `integrity`].

| Check | Expected | Observed (v4) |
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
| Records, roles, raw paths and waveform hashes equal v1 | 100,221, 0 mismatches | 100,221, 0 mismatches |

- **PTB-XL and Challenge files.** Every header and signal file matched its release `SHA256SUMS.txt`
  digest: 21,799 PTB-XL and 52,652 Challenge records.
- **SPH files.** SPH publishes no per-record checksum, so the 25,770 HDF5 files were hashed into the
  manifest. The metadata and code tables matched the download manifest's MD5.
- **Closed-data guard.** There are three layers:
  - the split table is read with `usecols` limited to source, family, record, path and split;
  - test rows are filtered out before any job is built;
  - every worker refuses a path of the closed set.

  All three layers are tested in `tests/test_mlc01_dataset.py`.

## Label rules added by the amendments

All rulings live in `ecg_experiment/mlc_labels.py`, with one unit test each in `tests/test_mlc_labels.py`.
Each one makes an unstated label undefined, and none creates a positive. The ruling 6 of Amendment 1 is
applied in the build from label counts.

### Amendment 1 (v2)

1. **PVC and ventricular tachycardia.** A record with SNOMED 425856008 (paroxysmal VT) or 164895002 (VT)
   and no PVC code is undefined for premature ventricular complexes.
2. **ST depression.** A record with a nonspecific or ischemic ST-T statement (table below) and no
   ST-depression code is undefined for ST depression. At SPH, 145 without a direction modifier and 146 stay
   undefined as before.
3. **T-wave abnormality.** The same statements without a T-wave code leave T-wave abnormality undefined.
4. **PTB-XL axis.** An empty `heart_axis` (coded `axis:missing`) or `AXL` is undefined for left axis
   deviation.
5. **PTB-XL HVOLT.** `HVOLT` alone is undefined for left ventricular hypertrophy. Under departure 4 it is
   also treated as left ventricular high voltage for space-A HYP.
6. **Fewer than five positives.** A source with fewer than 5 positives of a label among all its opened
   records does not annotate that label. Positives are counted after every other rule and before duplicate
   resolution, for every label of spaces A, B and C [`minimum_positive_rule.csv`; receipt `amendment_1`].
7. **LBBB with incomplete RBBB** is a contradictory pair: both labels become undefined.
8. **Space A.** SPH 153 counts for HYP. STTC is undefined when 153 is the record's only ST-T evidence, that
   is, when none of 145, 146, 147 or 148 is present. Every CPSC 2018 group is undefined unless positive.

"Nonspecific or ischemic ST-T statement" for rulings 2 and 3
[`receipt.json` `amendment_1.st_t_nonspecific_or_ischemic`]:

| Vocabulary | Codes |
|---|---|
| PTB-XL SCP | `NST_` (non-specific ST changes); the ischemic statements `ISC_`, `ISCAL`, `ISCAS`, `ISCLA`, `ISCAN`, `ISCIN`, `ISCIL` |
| SNOMED CT, nonspecific | 428750005 nonspecific ST-T abnormality, 55930002 ST changes, 164930006 ST interval abnormal |
| SNOMED CT, ischemic | 164861001 myocardial ischemia, 413844008 chronic myocardial ischemia, 413444003 acute myocardial ischemia, 426434006 anterior ischemia, 425419005 inferior ischaemia, 425623009 lateral ischaemia |

### Amendment 2 (v3)

1. **PTB-XL ST depression.** A subendocardial injury (`INJAS`, `INJAL`, `INJLA`, `INJIN`, `INJIL`),
   digitalis (`DIG`), aneurysm (`ANEUR`) or electrolyte (`EL`) statement with no ST-depression code is
   undefined for ST depression (`ST_CHANGE_WITHOUT_DIRECTION_SCP`).
2. **Kept as built.** Three cases stay as they are and are stated as limits:
   - SPH 145 without 146 or 147 stays negative for T-wave abnormality;
   - CPSC 2018 stays defined for AF or flutter;
   - space C keeps its other ambiguous negatives (ST elevation, right axis deviation).
3. **Scope of ruling 6.** Ruling 6 of Amendment 1 applies to every label space.

### Amendment 3 (v4)

Amendment 3 follows the independent audit (next section). Its rulings apply in every source. They are
applied in spaces A and C only where the same condition appears there.

1. **Left axis deviation and LAFB.** Left anterior fascicular block (SCP `LAFB`, SNOMED 445118002, AHA 101)
   without a left-axis positive leaves left axis deviation undefined, in spaces B and C. In PTB-XL, LAFB
   with `heart_axis` ALAD stays positive.
2. **First-degree AV block.** SNOMED 233917008 (AV block, degree unstated) without a first-degree or
   prolonged-PR code leaves first-degree AV block undefined, in spaces B and C.
3. **T-wave abnormality in PTB-XL.** An `INJ*`, `DIG`, `ANEUR` or `EL` statement without a T-wave code
   leaves T-wave abnormality undefined, in spaces B and C.
4. **Myocardial infarction.** An abnormal Q-wave or poor R-wave progression statement without an infarction
   code leaves myocardial infarction undefined, in space B (`mi_any`) and space A (MI). The space-C
   territory labels are not changed. The codes are:
   - PTB-XL `QWAVE` ("Q waves present");
   - SNOMED 164917005 "qwave abnormal" (abbreviation QAb);
   - SNOMED 365413008 "poor R wave Progression" (PRWP).

   Both SNOMED codes come from `dx_mapping_scored.csv`, whose rows 22 and 18 are listed in `v4/mapping.csv`.
   SPH has no such code.
5. **SPH 153.** SPH 153 without a specific ST or T code (none of 145, 146, 147) leaves ST depression and
   T-wave abnormality undefined, in both SPH halves and in spaces B and C. Ruling 8 of Amendment 1 already
   covers space-A STTC.
6. **Likelihood flag.** The likelihood sensitivity flag ignores diagnostic statements with likelihood 0
   (unknown). Only the flag changes; no label state does.

### Ruling 6 pairs after Amendment 3

The rule was re-run after the new rules and catches the same 23 (source, label) pairs as v2 and v3, with
the same positive counts. A further 50 flagged pairs were already undefined throughout [`v4/receipt.json`
`amendment_1`, compared with `v3/receipt.json`].

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

## Independent audit

A second agent, which did not build the dataset, audited v3. The copy below holds its scripts and tables,
without the waveform parquet files, in `outputs/mlc01_condition_dataset_20261110_v4/audit/`; its
`SHA256SUMS.txt` lists the 42 files.

What it checked and found:

| Check | Finding | File |
|---|---|---|
| Space-B labels re-derived from the raw metadata and the protocol text | 1,402,829 of 1,403,094 cells (all 100,221 records × 14) agree; the 265 differences are first-degree AV block with SNOMED 233917008 (Chapman 140, Ningbo 62, Georgia 60, CPSC-Extra 3; 199 train, 66 validation, none in a test set) | `a3_confusion.csv`, `a3_disagreements.csv`, `a3_summary.json` |
| Closed data | 0 test-group keys, paths or record names in the v1-v3 manifests, in the scan cache (100,221 records) or in the header table (52,652 rows); Ningbo JS13118 not in the split table | `a1_closed_data.json` |
| Roles and patients | 0 role or group mismatches in PTB-XL (21,799) and SPH (25,770); 0 patients and 0 included waveform hashes in two roles | `a2b_roles.json` |
| Waveform hashes | all 100,158 eligible records re-hashed with 0 errors; first-10-s hashes, whole-record hashes and sample counts all equal the manifest | `a2d_hash_check.json` |
| Partial copies | 1,290,949 R-peak-anchored 400 ms windows (leads II and V2) compared; 0 record pairs share a window | `a2g_segments.json`, `a2g_segment_pairs.csv` |
| Duplicate rule | 806 groups, 1,635 records, 0 across roles, 162 conflicting groups (103 CPSC 2018 with CPSC-Extra); statuses reproduce the manifest with 0 mismatches | `a5_duplicates.json` |
| Challenge headers | 200 headers re-read with 0 mismatches | `a3c_header_spotcheck.json` |
| Report numbers | 35 numbers of the v3 report reproduced, 35 matched | `a6_report_numbers.csv` |
| Lint and tests at `0e42212` | ruff passed; 1,629 passed, 2 skipped, 1 failed (the worktree-symlink test) | `ruff.log`, `pytest_full.log`, `pytest_commit.txt` |

Label problems it found, which Amendment 3 applied:

- **LAFB coded as LAD-negative.** 614 records with LAFB were left axis deviation negatives
  [`a4_lafb_lad_negative.csv`]:
  - Ningbo: 297;
  - SPH: 150, of which 76 in the evaluation half (beside 68 positives there);
  - Georgia: 146;
  - PTB-XL: 21.

  v4 makes exactly these 614 undefined (`v4/comparison_with_before/state_transitions.csv`).
- **AV block of unstated degree.** 265 first-degree AV block negatives carried SNOMED 233917008
  [`a3b_ambiguity.csv`, `a3_disagreements.csv`]. v4 makes all 265 undefined.
- **Q wave or poor R-wave progression coded as MI-negative** [`a3b_ambiguity.csv`]:
  - SNOMED 164917005: 126 + 50 Chapman and 468 + 141 Ningbo records (train + validation);
  - SNOMED 365413008: 409 + 113 Ningbo;
  - PTB-XL `QWAVE`: 91 / 5 / 13 (train / fold 9 / fold 10).

  v4 moves 1,378 space-B MI negatives to undefined: Ningbo 1,093, Chapman 176, PTB-XL 109.
- **SPH 153 without an ST or T code.** These were T-wave negatives: 42 in each half [`a3b_ambiguity.csv`].

What it measured without relabelling, now limitations (below):

- `a4_clinical.csv`: heart rate and QRS axis against the rhythm and axis labels;
- `a4b_qrs_width.csv`: a crude QRS width of complete-RBBB positives;
- `a2f_nn_meta.csv`, `a2c_*`: same-patient overlap estimated between Challenge train and validation.

## Exclusion flow (v4, identical to v2 and v3)

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
nothing. Ningbo JS13118 is not in the split table, because its waveform is absent upstream. Per-role rows are
in `exclusion_flow.csv`.

## Patients and record groups

[`group_counts.csv`; identical in every build]

| role | source | records | groups | groups with several records | records in those groups | max per group |
|---|---|---:|---:|---:|---:|---:|
| train | ptbxl | 17,418 | 15,023 | 1,694 | 4,089 | 10 |
| validation | ptbxl | 2,183 | 1,942 | 189 | 430 | 8 |
| test_same_source | ptbxl | 2,198 | 1,904 | 228 | 522 | 6 |
| test_external | sph | 12,698 | 12,246 | 439 | 891 | 4 |
| sph_calibration | sph | 12,883 | 12,400 | 469 | 952 | 4 |

The Challenge sources have no patient identifier, so each record is its own group.

## Space B by evaluation set (v4)

Each cell is positive / negative / undefined records. `chapman_ningbo` and `cpsc` are the two-source
families [`support_B.csv`].

| condition | val ptbxl | val ningbo | val chapman | val chapman_ningbo | val georgia | val cpsc_2018 | val cpsc_extra | val cpsc | test ptbxl | test sph | sph calibration |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sinus_tachycardia | 83 / 2,100 / 0 | 1,145 / 5,827 / 0 | 314 / 1,722 / 0 | 1,459 / 7,549 / 0 | 265 / 1,778 / 0 | 0 / 0 / 1,251 | 65 / 588 / 0 | 65 / 588 / 1,251 | 82 / 2,116 / 0 | 360 / 12,338 / 0 | 357 / 12,526 / 0 |
| sinus_bradycardia | 64 / 2,119 / 0 | 2,510 / 4,461 / 1 | 769 / 1,267 / 0 | 3,279 / 5,728 / 1 | 318 / 1,725 / 0 | 0 / 0 / 1,251 | 6 / 598 / 49 | 6 / 598 / 1,300 | 64 / 2,134 / 0 | 1,302 / 11,396 / 0 | 1,393 / 11,490 / 0 |
| sinus_arrhythmia | 77 / 2,106 / 0 | 529 / 6,443 / 0 | 0 / 0 / 2,036 | 529 / 6,443 / 2,036 | 93 / 1,950 / 0 | 0 / 0 / 1,251 | 2 / 651 / 0 | 2 / 651 / 1,251 | 77 / 2,121 / 0 | 781 / 11,917 / 0 | 754 / 12,129 / 0 |
| af_or_flutter | 156 / 2,027 / 0 | 1,536 / 5,436 / 0 | 454 / 1,582 / 0 | 1,990 / 7,018 / 0 | 133 / 1,910 / 0 | 213 / 1,038 / 0 | 49 / 604 / 0 | 262 / 1,642 / 0 | 157 / 2,041 / 0 | 379 / 12,319 / 0 | 384 / 12,499 / 0 |
| premature_ventricular_complex | 114 / 2,063 / 6 | 220 / 6,718 / 34 | 64 / 1,972 / 0 | 284 / 8,690 / 34 | 66 / 1,977 / 0 | 113 / 1,138 / 0 | 27 / 626 / 0 | 140 / 1,764 / 0 | 114 / 2,075 / 9 | 517 / 12,181 / 0 | 541 / 12,342 / 0 |
| first_degree_av_block | 104 / 2,079 / 0 | 189 / 6,770 / 13 | 50 / 1,948 / 38 | 239 / 8,718 / 51 | 149 / 1,880 / 14 | 112 / 1,139 / 0 | 20 / 632 / 1 | 132 / 1,771 / 1 | 96 / 2,102 / 0 | 112 / 12,586 / 0 | 123 / 12,760 / 0 |
| complete_rbbb | 54 / 2,128 / 1 | 256 / 6,659 / 57 | 87 / 1,943 / 6 | 343 / 8,602 / 63 | 104 / 1,911 / 28 | 334 / 917 / 0 | 20 / 625 / 8 | 354 / 1,542 / 8 | 54 / 2,144 / 0 | 350 / 12,348 / 0 | 349 / 12,533 / 1 |
| incomplete_rbbb | 111 / 2,071 / 1 | 58 / 6,914 / 0 | 0 / 0 / 2,036 | 58 / 6,914 / 2,036 | 74 / 1,959 / 10 | 0 / 0 / 1,251 | 5 / 639 / 9 | 5 / 639 / 1,260 | 112 / 2,086 / 0 | 619 / 12,079 / 0 | 626 / 12,256 / 1 |
| lafb | 163 / 2,020 / 0 | 76 / 6,896 / 0 | 0 / 0 / 2,036 | 76 / 6,896 / 2,036 | 45 / 1,998 / 0 | 0 / 0 / 1,251 | 0 / 0 / 653 | 0 / 0 / 1,904 | 162 / 2,036 / 0 | 77 / 12,621 / 0 | 74 / 12,809 / 0 |
| left_axis_deviation | 121 / 745 / 1,317 | 216 / 6,680 / 76 | 80 / 1,956 / 0 | 296 / 8,636 / 76 | 190 / 1,808 / 45 | 0 / 0 / 1,251 | 0 / 0 / 653 | 0 / 0 / 1,904 | 134 / 704 / 1,360 | 68 / 12,554 / 76 | 70 / 12,739 / 74 |
| lvh | 210 / 1,940 / 33 | 103 / 6,097 / 772 | 4 / 1,761 / 271 | 107 / 7,858 / 1,043 | 231 / 1,803 / 9 | 0 / 0 / 1,251 | 33 / 620 / 0 | 33 / 620 / 1,251 | 214 / 1,959 / 25 | 100 / 12,598 / 0 | 107 / 12,776 / 0 |
| st_depression | 101 / 1,772 / 310 | 231 / 5,967 / 774 | 78 / 1,807 / 151 | 309 / 7,774 / 925 | 0 / 0 / 2,043 | 161 / 1,090 / 0 | 8 / 222 / 423 | 169 / 1,312 / 423 | 101 / 1,797 / 300 | 497 / 11,251 / 950 | 513 / 11,420 / 950 |
| t_wave_abnormality | 260 / 1,605 / 318 | 1,305 / 4,931 / 736 | 390 / 1,486 / 160 | 1,695 / 6,417 / 896 | 544 / 1,110 / 389 | 0 / 0 / 1,251 | 5 / 225 / 423 | 5 / 225 / 1,674 | 255 / 1,633 / 310 | 1,633 / 11,023 / 42 | 1,618 / 11,224 / 41 |
| mi_any | 521 / 1,638 / 24 | 26 / 6,705 / 241 | 6 / 1,980 / 50 | 32 / 8,685 / 291 | 0 / 0 / 2,043 | 0 / 0 / 1,251 | 274 / 379 / 0 | 274 / 379 / 1,251 | 532 / 1,635 / 31 | 118 / 12,580 / 0 | 137 / 12,746 / 0 |

Training records, positive / negative / undefined [`label_counts_B.csv`]:

| condition | ptbxl | ningbo | chapman_shaoxing | georgia | cpsc_2018 | cpsc_2018_extra |
|---|---:|---:|---:|---:|---:|---:|
| sinus_tachycardia | 661 / 16,757 / 0 | 3,419 / 17,500 / 0 | 922 / 5,175 / 0 | 745 / 5,372 / 0 | 0 / 0 / 3,763 | 160 / 1,776 / 0 |
| sinus_bradycardia | 509 / 16,909 / 0 | 7,615 / 13,300 / 4 | 2,312 / 3,785 / 0 | 954 / 5,159 / 4 | 0 / 0 / 3,763 | 29 / 1,757 / 150 |
| sinus_arrhythmia | 618 / 16,800 / 0 | 1,513 / 19,406 / 0 | 0 / 0 / 6,097 | 264 / 5,853 / 0 | 0 / 0 / 3,763 | 4 / 1,932 / 0 |
| af_or_flutter | 1,257 / 16,161 / 0 | 4,532 / 16,387 / 0 | 1,336 / 4,761 / 0 | 441 / 5,676 / 0 | 587 / 3,176 / 0 | 113 / 1,823 / 0 |
| premature_ventricular_complex | 915 / 16,456 / 47 | 656 / 20,173 / 90 | 176 / 5,921 / 0 | 239 / 5,878 / 0 | 364 / 3,399 / 0 | 103 / 1,832 / 1 |
| first_degree_av_block | 808 / 16,610 / 0 | 576 / 20,294 / 49 | 161 / 5,834 / 102 | 443 / 5,628 / 46 | 418 / 3,345 / 0 | 51 / 1,883 / 2 |
| complete_rbbb | 432 / 16,986 / 0 | 765 / 19,912 / 242 | 238 / 5,834 / 25 | 285 / 5,743 / 89 | 950 / 2,813 / 0 | 31 / 1,878 / 27 |
| incomplete_rbbb | 894 / 16,524 / 0 | 140 / 20,779 / 0 | 0 / 0 / 6,097 | 196 / 5,890 / 31 | 0 / 0 / 3,763 | 13 / 1,897 / 26 |
| lafb | 1,298 / 16,120 / 0 | 227 / 20,692 / 0 | 0 / 0 / 6,097 | 102 / 6,015 / 0 | 0 / 0 / 3,763 | 0 / 0 / 1,936 |
| left_axis_deviation | 1,127 / 6,614 / 9,677 | 708 / 19,990 / 221 | 225 / 5,872 / 0 | 556 / 5,460 / 101 | 0 / 0 / 3,763 | 0 / 0 / 1,936 |
| lvh | 1,708 / 15,501 / 209 | 392 / 18,031 / 2,496 | 7 / 5,370 / 720 | 763 / 5,336 / 18 | 0 / 0 / 3,763 | 86 / 1,850 / 0 |
| st_depression | 807 / 14,204 / 2,407 | 786 / 17,782 / 2,351 | 242 / 5,343 / 512 | 0 / 0 / 6,117 | 499 / 3,264 / 0 | 34 / 651 / 1,251 |
| t_wave_abnormality | 2,079 / 12,848 / 2,491 | 3,951 / 14,753 / 2,215 | 1,156 / 4,418 / 523 | 1,621 / 3,308 / 1,188 | 0 / 0 / 3,763 | 16 / 660 / 1,260 |
| mi_any | 4,235 / 12,948 / 235 | 97 / 19,970 / 852 | 25 / 5,946 / 126 | 0 / 0 / 6,117 | 0 / 0 / 3,763 | 827 / 1,109 / 0 |

The protocol's selection rule still holds on v4: each of the 14 conditions has at least 100 positives in the
training pool (smallest: incomplete RBBB, 1,243), in SPH (left axis deviation, 138) and in PTB-XL folds 9-10
(complete RBBB, 108).

### Support levels (v4)

The macro rule admits a condition with at least 20 positives and 20 negatives in a set [`support_B.csv`].
"Below 20" includes sources that do not annotate the condition, which have 0 positives. No support level
changed between v3 and v4 [`v4/comparison_with_before/before_after_B.csv`].

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

- All 14 conditions still enter the macro at PTB-XL fold 10 and at the SPH evaluation half.
- In v4, fewer than 20 negatives occurs only where a source does not annotate the condition at all (0
  positives and 0 negatives). No set has a condition with positives but fewer than 20 negatives.

### Likelihood sensitivity

PTB-XL positives that rest only on diagnostic statements with likelihood above 0 and below 50 become
undefined in the sensitivity analysis [`label_counts_B.csv` minus `label_counts_B_lik50.csv`]:

| condition | train | fold 9 | fold 10 |
|---|---:|---:|---:|
| myocardial infarction | 1,090 | 131 | 141 |
| LVH | 297 | 47 | 37 |
| complete RBBB | 1 | 0 | 0 |
| first-degree AV block | 3 | 0 | 0 |
| incomplete RBBB | 3 | 0 | 0 |
| LAFB | 0 | 1 | 0 |

- This leaves 391 MI and 177 LVH positives in fold 10 [`support_B_lik50.csv`].
- Ruling 6 of Amendment 3 changed no flag: 0 flag cells differ between v3 and v4. The only diagnostic
  statements with likelihood 0 in PTB-XL are `NST_`, `DIG` and `LNGQT`, and none is a space-B positive code.

## History v1 to v4

The tables chain `v2/`, `v3/` and `v4/comparison_with_before/` (from `scripts/reports/compare_mlc01_builds.py`).
They show every space-B cell that changed in a test set or a per-source validation set, as positive /
negative / undefined.

| set | condition | v1 | v2 | v3 | v4 |
|---|---|---:|---:|---:|---:|
| test ptbxl (fold 10) | left_axis_deviation | 134 / 1,670 / 394 | 134 / 704 / 1,360 | 134 / 704 / 1,360 | 134 / 704 / 1,360 |
| test ptbxl (fold 10) | lvh | 214 / 1,964 / 20 | 214 / 1,959 / 25 | 214 / 1,959 / 25 | 214 / 1,959 / 25 |
| test ptbxl (fold 10) | mi_any | 532 / 1,648 / 18 | 532 / 1,648 / 18 | 532 / 1,648 / 18 | 532 / 1,635 / 31 |
| test ptbxl (fold 10) | st_depression | 101 / 2,097 / 0 | 101 / 1,834 / 263 | 101 / 1,797 / 300 | 101 / 1,797 / 300 |
| test ptbxl (fold 10) | t_wave_abnormality | 255 / 1,943 / 0 | 255 / 1,664 / 279 | 255 / 1,664 / 279 | 255 / 1,633 / 310 |
| test sph (evaluation) | left_axis_deviation | 68 / 12,630 / 0 | 68 / 12,630 / 0 | 68 / 12,630 / 0 | 68 / 12,554 / 76 |
| test sph (evaluation) | st_depression | 497 / 11,293 / 908 | 497 / 11,293 / 908 | 497 / 11,293 / 908 | 497 / 11,251 / 950 |
| test sph (evaluation) | t_wave_abnormality | 1,633 / 11,065 / 0 | 1,633 / 11,065 / 0 | 1,633 / 11,065 / 0 | 1,633 / 11,023 / 42 |
| validation ptbxl | left_axis_deviation | 121 / 1,694 / 368 | 121 / 751 / 1,311 | 121 / 751 / 1,311 | 121 / 745 / 1,317 |
| validation ptbxl | lvh | 210 / 1,946 / 27 | 210 / 1,940 / 33 | 210 / 1,940 / 33 | 210 / 1,940 / 33 |
| validation ptbxl | mi_any | 521 / 1,643 / 19 | 521 / 1,643 / 19 | 521 / 1,643 / 19 | 521 / 1,638 / 24 |
| validation ptbxl | st_depression | 101 / 2,082 / 0 | 101 / 1,813 / 269 | 101 / 1,772 / 310 | 101 / 1,772 / 310 |
| validation ptbxl | t_wave_abnormality | 260 / 1,923 / 0 | 260 / 1,638 / 285 | 260 / 1,638 / 285 | 260 / 1,605 / 318 |
| validation ningbo | first_degree_av_block | 189 / 6,783 / 0 | 189 / 6,783 / 0 | 189 / 6,783 / 0 | 189 / 6,770 / 13 |
| validation ningbo | left_axis_deviation | 216 / 6,756 / 0 | 216 / 6,756 / 0 | 216 / 6,756 / 0 | 216 / 6,680 / 76 |
| validation ningbo | mi_any | 26 / 6,946 / 0 | 26 / 6,946 / 0 | 26 / 6,946 / 0 | 26 / 6,705 / 241 |
| validation ningbo | premature_ventricular_complex | 220 / 6,730 / 22 | 220 / 6,718 / 34 | 220 / 6,718 / 34 | 220 / 6,718 / 34 |
| validation ningbo | st_depression | 231 / 6,741 / 0 | 231 / 5,967 / 774 | 231 / 5,967 / 774 | 231 / 5,967 / 774 |
| validation ningbo | t_wave_abnormality | 1,305 / 5,020 / 647 | 1,305 / 4,931 / 736 | 1,305 / 4,931 / 736 | 1,305 / 4,931 / 736 |
| validation chapman_shaoxing | first_degree_av_block | 50 / 1,986 / 0 | 50 / 1,986 / 0 | 50 / 1,986 / 0 | 50 / 1,948 / 38 |
| validation chapman_shaoxing | mi_any | 6 / 2,030 / 0 | 6 / 2,030 / 0 | 6 / 2,030 / 0 | 6 / 1,980 / 50 |
| validation chapman_shaoxing | st_depression | 78 / 1,958 / 0 | 78 / 1,807 / 151 | 78 / 1,807 / 151 | 78 / 1,807 / 151 |
| validation georgia | first_degree_av_block | 149 / 1,894 / 0 | 149 / 1,894 / 0 | 149 / 1,894 / 0 | 149 / 1,880 / 14 |
| validation georgia | left_axis_deviation | 190 / 1,853 / 0 | 190 / 1,853 / 0 | 190 / 1,853 / 0 | 190 / 1,808 / 45 |
| validation georgia | mi_any | 0 / 2,043 / 0 | 0 / 0 / 2,043 | 0 / 0 / 2,043 | 0 / 0 / 2,043 |
| validation georgia | t_wave_abnormality | 544 / 1,218 / 281 | 544 / 1,110 / 389 | 544 / 1,110 / 389 | 544 / 1,110 / 389 |
| validation cpsc_2018_extra | af_or_flutter | 38 / 593 / 0 | 49 / 604 / 0 | 49 / 604 / 0 | 49 / 604 / 0 |
| validation cpsc_2018_extra | complete_rbbb | 18 / 606 / 7 | 20 / 625 / 8 | 20 / 625 / 8 | 20 / 625 / 8 |
| validation cpsc_2018_extra | first_degree_av_block | 18 / 613 / 0 | 20 / 633 / 0 | 20 / 633 / 0 | 20 / 632 / 1 |
| validation cpsc_2018_extra | incomplete_rbbb | 5 / 619 / 7 | 5 / 639 / 9 | 5 / 639 / 9 | 5 / 639 / 9 |
| validation cpsc_2018_extra | lafb | 0 / 631 / 0 | 0 / 0 / 653 | 0 / 0 / 653 | 0 / 0 / 653 |
| validation cpsc_2018_extra | left_axis_deviation | 0 / 631 / 0 | 0 / 0 / 653 | 0 / 0 / 653 | 0 / 0 / 653 |
| validation cpsc_2018_extra | lvh | 30 / 601 / 0 | 33 / 620 / 0 | 33 / 620 / 0 | 33 / 620 / 0 |
| validation cpsc_2018_extra | mi_any | 267 / 364 / 0 | 274 / 379 / 0 | 274 / 379 / 0 | 274 / 379 / 0 |
| validation cpsc_2018_extra | premature_ventricular_complex | 25 / 606 / 0 | 27 / 626 / 0 | 27 / 626 / 0 | 27 / 626 / 0 |
| validation cpsc_2018_extra | sinus_arrhythmia | 2 / 629 / 0 | 2 / 651 / 0 | 2 / 651 / 0 | 2 / 651 / 0 |
| validation cpsc_2018_extra | sinus_bradycardia | 6 / 576 / 49 | 6 / 598 / 49 | 6 / 598 / 49 | 6 / 598 / 49 |
| validation cpsc_2018_extra | sinus_tachycardia | 65 / 566 / 0 | 65 / 588 / 0 | 65 / 588 / 0 | 65 / 588 / 0 |
| validation cpsc_2018_extra | st_depression | 8 / 623 / 0 | 8 / 222 / 423 | 8 / 222 / 423 | 8 / 222 / 423 |
| validation cpsc_2018_extra | t_wave_abnormality | 5 / 410 / 216 | 5 / 225 / 423 | 5 / 225 / 423 | 5 / 225 / 423 |

How to read the history:

- **Positives.** No positive changed in either test set in any build. The CPSC-Extra positive increases at
  v2 come from the 22 validation records that the duplicate rule kept from then on.
- **Support flags.** Only one flag changed in the whole history. At v2, CPSC-Extra validation complete RBBB
  and first-degree AV block rose from 18 to 20 positives each, and that source's macro grew from 5 to 7
  conditions.
- **Space A in the test sets** [`before_after_A.csv`]:
  - SPH evaluation-half STTC changed at v2 from 2,480 / 10,218 / 0 to 2,480 / 10,176 / 42;
  - fold 10 HYP changed at v2 from 262 / 1,917 / 19 to 262 / 1,912 / 24;
  - fold 10 MI changed at v4 from 550 / 1,648 / 0 to 550 / 1,635 / 13.

The v3 to v4 changes on records included in both builds are negative to undefined only
[`v4/comparison_with_before/state_transitions.csv`]:

| label | records | by source |
|---|---:|---|
| mi_any and space-A MI | 1,378 each | Ningbo 1,093, Chapman 176, PTB-XL 109 |
| left_axis_deviation | 614 | Ningbo 297, SPH 150, Georgia 146, PTB-XL 21 |
| t_wave_abnormality | 407 | PTB-XL 324, SPH 83 |
| first_degree_av_block | 265 | Chapman 140, Ningbo 62, Georgia 60, CPSC-Extra 3 |
| st_depression | 83 | SPH |

The same changes appear in space C for the four shared conditions. No space-C territory label changed
[`before_after_C.csv`].

## Space A by evaluation set (v4)

Each cell is positive / negative / undefined [`support_A.csv`].

| group | val ptbxl | val ningbo | val chapman | val georgia | val cpsc_2018 | val cpsc_extra | test ptbxl | test sph | sph calibration |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MI | 540 / 1,638 / 5 | 26 / 6,705 / 241 | 6 / 1,980 / 50 | 0 / 0 / 2,043 | 0 / 0 / 1,251 | 274 / 379 / 0 | 550 / 1,635 / 13 | 118 / 12,580 / 0 | 137 / 12,746 / 0 |
| STTC | 528 / 1,655 / 0 | 2,068 / 4,904 / 0 | 579 / 1,457 / 0 | 989 / 1,054 / 0 | 195 / 0 / 1,056 | 449 / 204 / 0 | 521 / 1,677 / 0 | 2,480 / 10,176 / 42 | 2,478 / 10,364 / 41 |
| CD | 495 / 1,688 / 0 | 769 / 6,203 / 0 | 256 / 1,780 / 0 | 460 / 1,583 / 0 | 486 / 0 / 765 | 72 / 581 / 0 | 496 / 1,702 / 0 | 1,166 / 11,532 / 0 | 1,200 / 11,683 / 0 |
| HYP | 268 / 1,882 / 33 | 129 / 6,072 / 771 | 5 / 1,760 / 271 | 425 / 1,618 / 0 | 0 / 0 / 1,251 | 45 / 608 / 0 | 262 / 1,912 / 24 | 104 / 12,594 / 0 | 123 / 12,760 / 0 |

## Space C (exploratory, v4)

Number of the 30 conditions below each support level, and number entering a macro [`support_C.csv`;
unchanged from v2]:

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
condition (including through ruling 6) or cannot code it. These counts are unchanged since v2:

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

## Normal column, co-occurrence, flat leads

None of these changed after v2: the normal column has no amendment, co-occurrence counts positives only,
and the record set is unchanged.

- **Normal column** [`label_counts_normal.csv`]. The per-source normal definition is kept for audits only:
  - PTB-XL: NORM with SR as the only other statement, 701 of 2,198 in fold 10;
  - Challenge: sinus rhythm as the only code;
  - SPH: code 1 alone, 6,850 of 12,698 in the evaluation half;
  - CPSC-Extra: undefined throughout.
- **Co-occurrence** [`cooccurrence_B_count.csv`, `cooccurrence_B_pairs.csv`]:
  - Records with no positive space-B condition: PTB-XL training 41.8%, Ningbo 17.1%, Chapman 20.9%,
    Georgia 29.7%, CPSC 2018 27.6%, CPSC-Extra 40.7%, SPH evaluation half 58.5%.
  - The most frequent training pair is AF or flutter with T-wave abnormality at Ningbo, in 1,556 records
    (7.4%).
- **Flat and missing leads** [`lead_flags.csv`]:
  - At least one flat lead in the first 10 s: Ningbo 919 training and 283 validation records, Chapman 13,
    Georgia 6, CPSC 2018 9, CPSC-Extra 5, PTB-XL 1, SPH 0. One CPSC-Extra validation record is flat in all
    twelve leads.
  - Missing samples: 63 + 19 Ningbo and 3 + 2 Georgia records.
  - No record was excluded for quality.

## Duplicates

Exact duplicates share the hash of the first 10 s (int32 microvolts, leads I to V6). The 806 groups of more
than one eligible record hold 1,635 records, and their whole-record hashes are identical too
[`duplicate_accounting.csv`, `receipt.json` `summary`]. v4 and v3 are identical here
[`supplementary/duplicates.json`].

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

- No group spans two roles.
- The opened groups hold 54 Chapman-Ningbo pairs and 274 CPSC 2018-CPSC-Extra groups, the latter with 284
  CPSC 2018 and 280 CPSC-Extra records. Per Amendment 1, ruling 9, this accounting replaces the protocol's
  figure of 155 CPSC-Extra copies.
- 162 groups conflict, 103 of them CPSC 2018 with CPSC-Extra; v1 had 243. The most frequent conflicting label
  is premature ventricular complexes, in 39 groups [`supplementary/duplicate_conflicts.csv`].
- All 168 SPH groups are pairs from one patient. On the 17,402 records of the earlier SPH baseline table, the
  statuses agree with its frozen SPH rule.
- Near-duplicate checks found 0 groups with the same sign pattern but a different exact hash, and 0 record
  pairs sharing six or more identical non-flat leads [`near_duplicates_shape.csv`,
  `near_duplicates_leads.csv`]. The audit's window search found no shared segment either (above).

## Comparison with the inventory (v1)

The first build was compared with the metadata inventory [`inventory_comparison/` in the v1 directory]:

- Re-applying the draft mapping to the inventory's record sets reproduces all 576 inventory rows exactly. For
  the Challenge groups this used the split-rule `unique` and `kept` records, rebuilt with the frozen
  Ningbo-family rule: 51,347 unique and 397 kept, 51,744 in total.
- Every state the frozen mapping changes on the same records is attributed to a protocol rule, with 0
  unexplained [`rule_transitions.csv`]. Examples: LV high voltage only (Ningbo 3,267, Chapman 996, PTB-XL 222
  LVH negatives to undefined), Georgia ST depression (28 positives and 8,172 negatives to undefined), the
  contradictory pairs, and SPH 153 (85 STTC positives to negative).
- Record-set differences are explained [`record_set_changes.csv`]:
  - duration exclusions: Georgia 40, CPSC 2018 10, CPSC-Extra 10;
  - SPH duplicate resolution: 94 + 95 records;
  - the two duplicate rules' choice of kept copy.

## Departures from the protocol

All are dated 10 November 2026, and none changes a decision of the protocol. Departures 1-7 were accepted
with Amendment 1.

1. **SPH lead order and rate.** SPH files store neither lead names nor the sampling rate, so the documented
   SPH lead order and 500 Hz are used.
2. **Duplicate hash.** "Exact waveform hashes" was read as the hash of the first 10 s. No group differs in
   its whole-record hash.
3. **Conflicting copies.** "Conflicting label sets" means one copy is positive and another negative on any
   label of spaces A, B or C or the normal column. Without a conflict, the copy with the most defined labels
   is kept. The rule applies inside every role, and copies are removed from the less protected role first.
   That last step removed nothing.
4. **Space A.** Space A keeps the existing definitions, with SPH 153 moved to HYP:
   - CPSC 2018 MI and HYP are undefined;
   - HYP is undefined on left ventricular high voltage alone.

   Later rulings refine this: Amendment 1, ruling 8 makes all CPSC 2018 groups positive or undefined;
   Amendment 1, ruling 5 adds `HVOLT`; Amendment 3, ruling 4 adds Q waves and poor R-wave progression for
   MI.
5. **CPSC-Extra normal column.** It is undefined rather than 0.
6. **Space C rules.**
   - Conditions shared with space B use B's definitions.
   - Anterior MI is undefined on SNOMED infarction codes that state no territory.
   - LPFB is not annotated at Chapman.
   - Right ventricular hypertrophy keeps the draft's undefined state for SPH 153.
7. **Likelihood flag.** It applies to diagnostic statements only. Amendment 3, ruling 6 also excludes
   diagnostic statements with likelihood 0.
8. **Ruling 6 scope.** Ruling 6 of Amendment 1 covers every label space, as Amendment 2, ruling 3
   confirmed.
9. **Scope of later rulings.** Rulings of Amendments 1 and 3 apply in spaces A and C only where the same
   condition appears there.
10. **Raw paths.** The v3 and v4 reference checks also compare raw paths with the first build.

## Limitations

The dataset is final. From the first model fit onward no label rule changes; any later question is logged
as a limitation.

**Labels in general.**

- **Not diagnoses.** The labels are annotation statements, not diagnoses, and coding practice differs
  between sources. Validation by a cardiologist is pending.
- **Earlier scoring.** SPH and PTB-XL fold 10 have been scored by earlier experiments for other targets.
- **Exact duplicates only.** Copies with any changed sample would not be detected. Neither the cheap
  near-duplicate checks nor the audit's window search found one.

**Kept by Amendment 3 ("Not changed"), with the audit's numbers.**

- **SPH 146 alone.** SPH 146 (ST deviation with T-wave change) alone stays a T-wave positive: 521 of the
  1,633 T-wave positives at the SPH evaluation half rest on it, and 531 in the calibration half
  [`audit/a3b_ambiguity.csv`; `support_B.csv`].
- **Bradycardia cut-off.** PTB-XL codes sinus bradycardia at about 50 beats per minute. Among PTB-XL
  bradycardia negatives, 13.4% (`contradict_negative_all` 0.1341) have a measured rate below 60. The same
  share is 0.26% at Chapman, 0.41% at Georgia, 0.47% at Ningbo and 2.4% at SPH [`audit/a4_clinical.csv`].
  The PTB-XL positives have a median rate of 48.8 beats per minute; the other sources' positives have
  medians from 55.0 to 56.0.
- **Complete RBBB completeness.** Complete RBBB in CPSC 2018 and Chapman probably includes incomplete blocks:
  Chapman has no complete/incomplete split, and CPSC 2018's RBBB class does not state completeness. The
  audit's crude width measure [`audit/a4b_qrs_width.csv`] gives these shares of complete-RBBB positives at
  120 ms or more:

  | source | share at 120 ms or more |
  |---|---:|
  | CPSC 2018 | 2.2% |
  | Chapman | 5.5% |
  | SPH | 3.9% |
  | Ningbo | 8.9% |
  | CPSC-Extra | 11.8% |
  | PTB-XL | 14.3% |
  | Georgia | 16.2% |

  The same measure puts the median of PTB-XL complete RBBB at 70 ms and of PTB-XL LBBB at 130 ms. It is
  therefore a relative comparison, not a QRS duration. The amendment's figures (62% and 76% against
  86-96%) are not in the audit copy.
- **Shared patients across Challenge train and validation.** The audit estimates, calibrated on PTB-XL, the
  share of validation records with a same-patient training record [`audit/a2f_nn_meta.csv`, three draws
  each]:

  | family | estimates |
  |---|---|
  | Chapman-Ningbo | 0.118, 0.131, 0.107 |
  | Ningbo | 0.071, 0.068, 0.063 |
  | Georgia | 0.079, 0.096, 0.091 |
  | Chapman | 0.019, 0.006, 0.014 |
  | CPSC | 0.029, 0.063, 0.044 |

  The SPH-calibrated estimates are higher: 0.170-0.239 for Chapman-Ningbo. This affects model choice,
  early stopping and thresholds, not the test sets. PTB-XL fold 9 shares no patient with training
  [`audit/a2b_roles.json`; `receipt.json` `integrity`].

**Kept by Amendment 2, ruling 2** [`v4/supplementary/open_questions.csv`].

- **SPH 145 alone.** SPH T-wave negatives with 145 and no 146 or 147: 857 in the calibration half and 844
  in the evaluation half.
- **CPSC 2018 AF or flutter.** These negatives stay defined although flutter is not a CPSC 2018 class:
  3,176 in training and 1,038 in validation.
- **Space C ST elevation.** ST-elevation negatives that carry a nonspecific or ischemic ST-T statement:
  PTB-XL 2,542 / 323 / 316, Georgia 2,228 / 759, CPSC-Extra 1,259 / 424, Chapman 698 / 205.
- **Space C right axis.** Right-axis negatives with an empty PTB-XL `heart_axis` or `AXR`: 6,614 / 944 /
  961.

**Receipts.** The v1-v3 receipts pin the label and build code of their own commits (`b6d3b6c`, `ee0481f`,
`41845c2`); those versions remain in Git.

## Files

Each build directory (local) holds:

- `manifest.csv`, one row per enumerated record: identity, role, group, raw path, duration, file and
  waveform hashes, lead flags, exclusion and duplicate status, raw codes, label states (1, 0, or empty for
  undefined), `lik50_` flags and the normal column;
- `waveform_scan.parquet`;
- `mapping.csv` and `annotation_status.csv`;
- the count tables named above;
- `receipt.json`;
- `build.log`.

v2-v4 also hold `minimum_positive_rule.csv`, `comparison_with_before/` and `supplementary/`. v4 holds
`audit/`, and v1 holds `inventory_comparison/`. The scan cache is `/var/tmp/janetrivera/mlc01_scan`.

Report scripts, run from the worktree with `PYTHONPATH=.`:

- `scripts.reports.compare_mlc01_inventory`
- `compare_mlc01_builds --before <build> --after <next build>`
- `describe_mlc01_duplicates --dataset <build>`
- `count_mlc01_label_doubts`
- `count_mlc01_open_questions --dataset <build>`

Each script records its own SHA-256 and the manifest's in its receipt.

**Checks:**

- `ruff check ecg_experiment scripts tests conftest.py`: passed.
- `pytest tests/test_mlc_labels.py tests/test_mlc01_dataset.py`: 50 passed at `3475e13`.
- The full suite, run once from the worktree at `3475e13`: 1,635 passed, 2 skipped, 1 failed. The failing test
  is `tests/test_xecg_encoder_motion_v11.py::test_graph_rejects_missing_and_conflicting_leaves`, which rejects
  the worktree's symlinked `outputs`; it failed the same way at every earlier commit and passes in the main
  checkout.
