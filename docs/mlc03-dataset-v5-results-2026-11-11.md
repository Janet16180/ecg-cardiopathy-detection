# MLC-03 Part C results: dataset v5

11 November 2026. This report covers the build of dataset v5 (`outputs/mlc01_condition_dataset_20261111_v5/`) from
the MLC-01 v4 dataset and the cleaning rules C1-C12 of Amendment 1 of the
[MLC-03 protocol](mlc03-data-quality-protocol-2026-11-11.md). No model was fitted or scored in the build. On the
test roles (PTB-XL fold 10; the SPH evaluation half, `test_external`; the SPH calibration half) the frozen rules
were applied by the same code, and only counts are reported. No test-role label was used to choose anything, and
no rule changed after a test-role count existed. No Challenge test group, EchoNext or MIMIC record was opened.
v4 is unchanged on disk.

Every number below comes from a file the build wrote, printed by
`PYTHONPATH=. .venv/bin/python -m scripts.reports.mlc03_v5_tables --section <name>` and cited as [section
`<name>`], or from the build's `receipt.json` and `integrity.json`. Nothing here is a clinical reading; clinical
validation is pending (validación pendiente).

## Summary

- v5 includes 99,008 of the 99,167 records included in v4. The rules excluded 159 records: 117 train, 41 validation,
  1 fold-10 record and no SPH record [section `totals`].
- The label rules made 21,276 labels of the 14 conditions undefined on included records: 1,435 positives and
  19,841 negatives. By role: train 1,065 positives and 14,677 negatives, validation 325 and 3,775, fold 10 7 and
  586, SPH calibration 22 and 411, SPH evaluation half 16 and 392 [section `totals`].
- Integrity gate: on train and validation every rule reproduces its Part A or Part B count exactly, in all 60
  Part A cells and all 212 Part B rows; the readers, the analysis table, method B on a 1,500-record sample and the
  window summaries of all 4,826 long records reproduce Part B's stored inputs exactly [section `gate`].
- v5 equals v4 in every cell no rule touches. Of 12,226,962 cells in v4's 122 columns, 21,595 changed: 159
  `included` and 159 `exclusion_reason` cells, 21,276 label cells (each from positive or negative to undefined) and
  1 likelihood flag. Keys, sources, roles, group ids, families, origins and raw paths are unchanged. No label went
  to positive and none from positive to negative [section `integrity`].
- No condition loses eligibility for a macro (at least 20 positives and 20 negatives) in any evaluation role and
  source. One cell falls below 100 positives: PVC at CPSC 2018 validation, 113 to 99 [section `eligibility`].
- The largest losses are of negatives: sinus bradycardia in PTB-XL (13% of negatives in train, validation and
  fold 10, by C10), LVH (7-15% of negatives in most sources, by C12, C5 and C7), and complete and incomplete RBBB
  in Ningbo (6-7%, by C5 and C7). The largest loss of positives is complete RBBB in CPSC 2018 (28% in train, 33%
  in validation) and Chapman (16% and 18%), by C9 [section `largest-changes`].

## Code, run and receipt

New files, on branch `research/mlc03-data-quality-20261111`:

- `ecg_experiment/mlc03_dataset_v5.py`: readers for any role, the rule masks, their application and the checks.
- `scripts/experiments/run_mlc03_build_v5.py`: steps `gate` (train and validation only) and `build`.
- `tests/test_mlc03_dataset_v5.py`: 19 tests, each rule on small synthetic tables (it touches what it should and
  nothing else), no positive created, order and overlap accounting, the SPH exceptions, refusal of an existing
  output, and the record reader against Part B's (skipped without local data).
- `scripts/reports/mlc03_v5_tables.py`: the tables of this report.

Command: `PYTHONPATH=/tmp/ecg-mlc03 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m
scripts.experiments.run_mlc03_build_v5 build --workers 14`. The build ran at commit `61b1131` with no modified
tracked file, in 126 s, and wrote through a staging directory renamed at the end; it refuses an existing output or
staging directory. The receipt holds the SHA-256 of the 15 inputs and 10 code files, the thread variables, the
package versions (numpy 2.4.6, pandas 3.0.6, scipy 1.17.1, wfdb 4.3.1, h5py 3.16.0, pyarrow 25.0.1), the counts
and the SHA-256 of every output. The uncorrected-memory-error counter was 1 before and after, as in every MLC-03
receipt.

## How the rules were applied

Each rule reuses the code behind its evidence. The Part A flags are read from `flags_train_validation.parquet`
and `flags_test_roles.parquet`. The Part B logic comes from `mlc03_label_quality` (`CONTRADICTIONS`,
`UNREADABLE`, `REQUIRED_LEADS`, `criterion`, `population`, `analysis_table`). Part B's record masks were written
in its run script, which a module does not import, so `part_b_masks` restates them. The gate checks them against
Part B's counts.

| rule | action | definition as applied | evidence | sources |
|---|---|---|---|---|
| C1 | exclude record | Part A `missing_samples` | A-R1 | all |
| C2 | exclude record | Part A `edge_hold` | A-R3 | all |
| C3 | exclude record | Part A `adc_rail` | A-R2 | all but SPH |
| C4 | exclude record | Part A `identity_violation` | A-R7 | all but SPH |
| C5 | label undefined | a lead the condition is read on is flat (manifest `flat_<lead>`, B-R5), or flat, near-flat or without contact (Part A lead lists; I, II, III or aVF for the two axis conditions; B-R11 first part) | B-R5, B-R11 | all |
| C6 | label undefined | the space-B labels of each contradictory pair of `lq.CONTRADICTIONS` | B-R4 | all |
| C7 | label undefined | the 8 conditions of `lq.UNREADABLE` with `c_paced_rhythm` positive | B-R8 | all |
| C8 | label undefined | PVC or AF-or-flutter positive whose method B finding is absent in the first 10 s window and present in a later one | B-R1 | all |
| C9 | label undefined | complete RBBB positive with consensus QRS below 110 ms | B-R2 | Chapman, CPSC 2018 |
| C10 | label undefined | sinus bradycardia negative, sinus evidence, consensus rate below 60 per minute | B-R3 | PTB-XL, CPSC-Extra |
| C11 | label undefined | sinus tachycardia or bradycardia positive outside its bound by more than 5 per minute; negative inside it by more than 5 per minute with sinus evidence | B-R6 | all; negatives not at SPH |
| C12 | label undefined | LVH negative with method B status `ok` and a Sokolow-Lyon or Cornell voltage criterion | B-R10 | all |

Readings, stated where the amendment's wording and the code could differ:

- C9: "QRS below 110 ms beyond tolerance" is the code's `qrs < 110`, which is the 120 ms bound minus its 10 ms
  tolerance. Both readings give the same 429 labels.
- C10: the B-R3 code has no tolerance (consensus rate below 60). Read literally, "beyond tolerance" would mean a
  rate of 55 or less. That reading gives 884 labels instead of 2,595, and all 884 are already among C11's 927
  bradycardia negatives, so C10 would add nothing. The code was followed, and the orchestrator was told before the
  test-role pass. Amendment 2 of the protocol, written after the build, records this reading.
- C11 negatives at SPH: Part B's sinus evidence (`sinus_by_codes`, no non-sinus rhythm code) knows SCP and SNOMED
  codes only. It does not parse SPH's AHA codes, so it would call every SPH record sinus, atrial fibrillation
  included. C11 negatives were therefore not applied at SPH, the same reasoning that keeps C3 and C4 away from SPH.
  C11 positives need no sinus evidence and were applied there. This reading was fixed and reported to the
  orchestrator before any test-role count existed, and Amendment 2 records it, with C7 at SPH as its point 4.
  The labels C6 reads (its 14 pair labels, mapped by `mlc_labels`) all have SPH codes and are annotated at SPH, so
  C6 runs there unchanged; the paced-rhythm label has no SPH code, so C7 has nothing to read.
- C5 is the union of B-R5 and the first part of B-R11. On train and validation the union, 3,173 labels, equals
  B-R11's count, so every B-R5 label (2,774) is also a B-R11 label [sections `rules`, `part-b-totals`].
- C6: the 13 Part B pair rows sum to 273 labels on train and validation. Seven labels sit in two pairs, so C6
  makes 266 distinct labels undefined.
- C9 and C10 use the sources that Part B's outputs give (`item2_separate_coding.csv`, `item2_implied_cutoffs.csv`
  with `cutoff_mask`'s 5 per minute margin); the gate checks the module's constants against them.
- Order. Exclusions come first, C1 to C4: a record's `exclusion_reason` is the first rule that selects it. The
  label rules then act, C5 to C12, on records still included; an excluded record keeps its v4 labels. Each
  `rule_*` column holds the rule's own selection on v4 states, before overlaps, on every v4-included record, so
  its sum is the Part A or Part B count. The final label states do not depend on the order of C5-C12; only the
  attribution in the flow does.
- A label made undefined also has its PTB-XL likelihood flag (`lik50_<condition>`) cleared, since that flag marks
  positives only (one cell, MI).
- Space A, space C and the `normal` column are unchanged; the rules act on the 14 space-B conditions.

Test-role inputs. Part B measured train and validation only. For the test roles the build ran method B once on the
cached first 10 s of all 27,779 included records (each window checked against its manifest hash), and on every
complete 10 s window of the 6,877 SPH records longer than 10 s (385 of them have two or more windows). It used the
same functions and settings, re-expressed without the open-role guard; the gate shows they reproduce Part B's
outputs on train and validation. Method B status was `ok` for 2,189 of 2,198 fold-10 records, 12,872 of 12,883 SPH
calibration records and 12,688 of 12,698 SPH evaluation records; the rest were `no_dominant`. Method A was read from
the stored MLC-02 measurement features (model inputs, not predictions). Both test-role measurement tables are in
the v5 directory.

## Integrity gate (train and validation)

[section `gate`]

| check | result |
|---|---|
| record reader equals Part B's | equal |
| method A equals Part B's | equal |
| analysis table equals Part B's analysis.parquet | equal |
| method B remeasured on 1500 records | equal |
| window summaries of 4826 long records | equal |
| Part A counts, 60 rule-role-source cells | 60 equal |
| Part B counts, 212 rule-condition-source-role rows | 212 equal |

The Part A cells cover train, validation and the three test roles, compared on the flags before the SPH exception
of C3 and C4. Train plus validation totals, found against Part B's `rules_candidate_counts.csv` [section
`part-b-totals`]: B-R1 55 PVC and 21 AF positives; B-R2 429; B-R3 2,595 negatives; B-R4 the 13 pair rows; B-R5
2,774 labels; B-R6 for the rate conditions 48 and 106 positives, 59 and 927 negatives; B-R8 9,628 labels; B-R10
3,883 negatives; B-R11 first part 3,173 labels. The amendment's record counts hold before overlaps: C1 87, C2 15,
C3 116, C4 47 [section `rules`].

## Rules by role

Selected before overlaps / changed after earlier rules: records for C1-C4, labels for C5-C12 [section `rules`]:

| rule | action | train | validation | fold 10 | SPH calibration | SPH test |
|---|---|---|---|---|---|---|
| C1 | exclude records | 66 / 66 | 21 / 21 | 0 / 0 | 0 / 0 | 0 / 0 |
| C2 | exclude records | 10 / 10 | 5 / 5 | 0 / 0 | 0 / 0 | 0 / 0 |
| C3 | exclude records | 89 / 34 | 27 / 11 | 0 / 0 | 0 / 0 | 0 / 0 |
| C4 | exclude records | 34 / 7 | 13 / 4 | 1 / 1 | 0 / 0 | 0 / 0 |
| C5 | labels undefined | 2452 / 2429 | 721 / 707 | 0 / 0 | 7 / 7 | 4 / 4 |
| C6 | labels undefined | 211 / 211 | 55 / 55 | 5 / 5 | 0 / 0 | 0 / 0 |
| C7 | labels undefined | 7436 / 7410 | 2192 / 2162 | 196 / 196 | 0 / 0 | 0 / 0 |
| C8 | labels undefined | 60 / 59 | 16 / 15 | 0 / 0 | 8 / 8 | 11 / 11 |
| C9 | labels undefined | 305 / 305 | 124 / 124 | 0 / 0 | 0 / 0 | 0 / 0 |
| C10 | labels undefined | 2307 / 2307 | 288 / 288 | 275 / 275 | 0 / 0 | 0 / 0 |
| C11 | labels undefined | 983 / 191 | 157 / 61 | 118 / 6 | 14 / 14 | 5 / 5 |
| C12 | labels undefined | 3118 / 2830 | 765 / 688 | 116 / 111 | 405 / 404 | 388 / 388 |

Positives (+) and negatives (-), selected / made undefined [section `label-rules`]:

| rule | train | validation | fold 10 | SPH calibration | SPH test |
|---|---|---|---|---|---|
| C5 | 15+ 2437- / 15+ 2414- | 1+ 720- / 1+ 706- | 0+ 0- / 0+ 0- | 0+ 7- / 0+ 7- | 0+ 4- / 0+ 4- |
| C6 | 211+ 0- / 211+ 0- | 55+ 0- / 55+ 0- | 5+ 0- / 5+ 0- | 0+ 0- / 0+ 0- | 0+ 0- / 0+ 0- |
| C7 | 364+ 7072- / 363+ 7047- | 92+ 2100- / 92+ 2070- | 0+ 196- / 0+ 196- | 0+ 0- / 0+ 0- | 0+ 0- / 0+ 0- |
| C8 | 60+ 0- / 59+ 0- | 16+ 0- / 15+ 0- | 0+ 0- / 0+ 0- | 8+ 0- / 8+ 0- | 11+ 0- / 11+ 0- |
| C9 | 305+ 0- / 305+ 0- | 124+ 0- / 124+ 0- | 0+ 0- / 0+ 0- | 0+ 0- / 0+ 0- | 0+ 0- / 0+ 0- |
| C10 | 0+ 2307- / 0+ 2307- | 0+ 288- / 0+ 288- | 0+ 275- / 0+ 275- | 0+ 0- / 0+ 0- | 0+ 0- / 0+ 0- |
| C11 | 115+ 868- / 112+ 79- | 39+ 118- / 38+ 23- | 2+ 116- / 2+ 4- | 14+ 0- / 14+ 0- | 5+ 0- / 5+ 0- |
| C12 | 0+ 3118- / 0+ 2830- | 0+ 765- / 0+ 688- | 0+ 116- / 0+ 111- | 0+ 405- / 0+ 404- | 0+ 388- / 0+ 388- |

- C3 and C4 overlap C1 and C2 heavily: of 116 rail records, 45 were not already excluded; of 47 identity
  violations, 11.
- C11's negatives mostly overlap C10. In train, 868 bradycardia and tachycardia negatives are selected, but only
  79 are new; in fold 10, 116 and 4.
- The full flow by rule, condition, source and role is `rule_flow.csv`; its `on_excluded_records` and
  `already_undefined` columns hold the overlaps.

## Records

[section `flow`]

| source | role | included_v4 | excluded_C1 | excluded_C2 | excluded_C3 | excluded_C4 | included_v5 |
|---|---|---|---|---|---|---|---|
| PTB-XL | train | 17418 | 0 | 0 | 1 | 3 | 17414 |
| PTB-XL | validation | 2183 | 0 | 0 | 0 | 2 | 2181 |
| PTB-XL | fold 10 | 2198 | 0 | 0 | 0 | 1 | 2197 |
| Chapman | validation | 2036 | 0 | 0 | 0 | 0 | 2036 |
| Chapman | train | 6097 | 0 | 0 | 0 | 0 | 6097 |
| CPSC 2018 | train | 3763 | 0 | 1 | 16 | 4 | 3742 |
| CPSC 2018 | validation | 1251 | 0 | 0 | 6 | 2 | 1243 |
| CPSC-Extra | train | 1936 | 0 | 0 | 10 | 0 | 1926 |
| CPSC-Extra | validation | 653 | 0 | 0 | 3 | 0 | 650 |
| Georgia | train | 6117 | 3 | 0 | 0 | 0 | 6114 |
| Georgia | validation | 2043 | 2 | 0 | 0 | 0 | 2041 |
| Ningbo | train | 20919 | 63 | 9 | 7 | 0 | 20840 |
| Ningbo | validation | 6972 | 19 | 5 | 2 | 0 | 6946 |
| SPH | SPH calibration | 12883 | 0 | 0 | 0 | 0 | 12883 |
| SPH | SPH test | 12698 | 0 | 0 | 0 | 0 | 12698 |

## Labels made undefined per condition

Positives / negatives made undefined on included records, all label rules together [section `undefined`]:

| condition | train | validation | fold 10 | SPH calibration | SPH test |
|---|---|---|---|---|---|
| sinus_tachycardia | 125 / 46 | 42 / 13 | 1 / 4 | 2 / 0 | 1 / 0 |
| sinus_bradycardia | 94 / 2340 | 31 / 298 | 1 / 275 | 12 / 0 | 4 / 0 |
| sinus_arrhythmia | 4 / 0 | 1 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| af_or_flutter | 80 / 0 | 12 / 0 | 2 / 0 | 5 / 0 | 5 / 0 |
| premature_ventricular_complex | 42 / 0 | 13 / 0 | 0 / 0 | 3 / 0 | 6 / 0 |
| first_degree_av_block | 25 / 25 | 4 / 5 | 2 / 0 | 0 / 0 | 0 / 0 |
| complete_rbbb | 341 / 1616 | 130 / 463 | 0 / 28 | 0 / 1 | 0 / 0 |
| incomplete_rbbb | 10 / 1644 | 5 / 468 | 0 / 28 | 0 / 1 | 0 / 0 |
| lafb | 27 / 1029 | 8 / 294 | 1 / 28 | 0 / 2 | 0 / 2 |
| left_axis_deviation | 39 / 764 | 4 / 263 | 0 / 0 | 0 / 2 | 0 / 2 |
| lvh | 30 / 4735 | 4 / 1269 | 0 / 139 | 0 / 405 | 0 / 388 |
| st_depression | 43 / 834 | 10 / 235 | 0 / 28 | 0 / 0 | 0 / 0 |
| t_wave_abnormality | 198 / 734 | 60 / 206 | 0 / 28 | 0 / 0 | 0 / 0 |
| mi_any | 7 / 910 | 1 / 261 | 0 / 28 | 0 / 0 | 0 / 0 |

## Test roles (counts only)

Rules with a nonzero count at a test role, selected / changed (records for C4; positives and negatives for the
label rules) [section `test-roles`]:

| rule | condition | fold 10 | SPH test | SPH calibration |
|---|---|---|---|---|
| C4 |  | 1 / 1 | 0 / 0 | 0 / 0 |
| C5 | complete_rbbb | 0+ 0- / 0+ 0- | 0+ 0- / 0+ 0- | 0+ 1- / 0+ 1- |
| C5 | incomplete_rbbb | 0+ 0- / 0+ 0- | 0+ 0- / 0+ 0- | 0+ 1- / 0+ 1- |
| C5 | lafb | 0+ 0- / 0+ 0- | 0+ 2- / 0+ 2- | 0+ 2- / 0+ 2- |
| C5 | left_axis_deviation | 0+ 0- / 0+ 0- | 0+ 2- / 0+ 2- | 0+ 2- / 0+ 2- |
| C5 | lvh | 0+ 0- / 0+ 0- | 0+ 0- / 0+ 0- | 0+ 1- / 0+ 1- |
| C6 | af_or_flutter | 2+ 0- / 2+ 0- | 0+ 0- / 0+ 0- | 0+ 0- / 0+ 0- |
| C6 | first_degree_av_block | 2+ 0- / 2+ 0- | 0+ 0- / 0+ 0- | 0+ 0- / 0+ 0- |
| C6 | lafb | 1+ 0- / 1+ 0- | 0+ 0- / 0+ 0- | 0+ 0- / 0+ 0- |
| C7 | complete_rbbb | 0+ 28- / 0+ 28- | 0+ 0- / 0+ 0- | 0+ 0- / 0+ 0- |
| C7 | incomplete_rbbb | 0+ 28- / 0+ 28- | 0+ 0- / 0+ 0- | 0+ 0- / 0+ 0- |
| C7 | lafb | 0+ 28- / 0+ 28- | 0+ 0- / 0+ 0- | 0+ 0- / 0+ 0- |
| C7 | lvh | 0+ 28- / 0+ 28- | 0+ 0- / 0+ 0- | 0+ 0- / 0+ 0- |
| C7 | st_depression | 0+ 28- / 0+ 28- | 0+ 0- / 0+ 0- | 0+ 0- / 0+ 0- |
| C7 | t_wave_abnormality | 0+ 28- / 0+ 28- | 0+ 0- / 0+ 0- | 0+ 0- / 0+ 0- |
| C7 | mi_any | 0+ 28- / 0+ 28- | 0+ 0- / 0+ 0- | 0+ 0- / 0+ 0- |
| C8 | premature_ventricular_complex | 0+ 0- / 0+ 0- | 6+ 0- / 6+ 0- | 3+ 0- / 3+ 0- |
| C8 | af_or_flutter | 0+ 0- / 0+ 0- | 5+ 0- / 5+ 0- | 5+ 0- / 5+ 0- |
| C10 | sinus_bradycardia | 0+ 275- / 0+ 275- | 0+ 0- / 0+ 0- | 0+ 0- / 0+ 0- |
| C11 | sinus_tachycardia | 1+ 4- / 1+ 4- | 1+ 0- / 1+ 0- | 2+ 0- / 2+ 0- |
| C11 | sinus_bradycardia | 1+ 112- / 1+ 0- | 4+ 0- / 4+ 0- | 12+ 0- / 12+ 0- |
| C12 | lvh | 0+ 116- / 0+ 111- | 0+ 388- / 0+ 388- | 0+ 405- / 0+ 404- |

- C7 never selects an SPH record: `c_paced_rhythm` is undefined for all 25,770 SPH records in v4, because the
  label code has no SPH code for a paced rhythm. C3 and C4 do not apply to SPH, and C9 has no test-role source.
- C12 is the largest rule at SPH: it selects 793 LVH negatives over both halves and makes 792 undefined. It rests
  on method B's voltages on float16 SPH samples, a storage the measurements were not checked on.
- At fold 10, C7 selects the 28 records with a paced rhythm statement; left axis deviation is not among their
  changed labels.

## Support, v4 against v5

`support_v4_v5.csv` holds positives, negatives and undefined labels of every condition, role and source or family,
v4 beside v5, with `enters_macro` and `below_100` for both [section `support`]. Source cells that lose at least 5%
of their positives or negatives [section `largest-changes`]:

| condition | role | source | positives | negatives | undefined | positive change % | negative change % |
|---|---|---|---|---|---|---|---|
| sinus_bradycardia | train | PTB-XL | 509 -> 506 | 16909 -> 14653 | 0 -> 2255 | -0.5894 | -13.3420 |
| lvh | train | PTB-XL | 1708 -> 1706 | 15501 -> 14332 | 209 -> 1376 | -0.1171 | -7.5414 |
| complete_rbbb | train | Ningbo | 765 -> 731 | 19912 -> 18526 | 242 -> 1583 | -4.4444 | -6.9606 |
| incomplete_rbbb | train | Ningbo | 140 -> 137 | 20779 -> 19349 | 0 -> 1354 | -2.1429 | -6.8819 |
| lafb | train | Ningbo | 227 -> 215 | 20692 -> 19883 | 0 -> 742 | -5.2863 | -3.9097 |
| lvh | train | Ningbo | 392 -> 366 | 18031 -> 15385 | 2496 -> 5089 | -6.6327 | -14.6747 |
| st_depression | train | Ningbo | 786 -> 741 | 17782 -> 17107 | 2351 -> 2992 | -5.7252 | -3.7960 |
| complete_rbbb | train | Chapman | 238 -> 200 | 5834 -> 5829 | 25 -> 68 | -15.9664 | -0.0857 |
| lvh | train | Chapman | 7 -> 7 | 5370 -> 4967 | 720 -> 1123 | 0.0000 | -7.5047 |
| af_or_flutter | train | Georgia | 441 -> 403 | 5676 -> 5673 | 0 -> 38 | -8.6168 | -0.0529 |
| lvh | train | Georgia | 763 -> 757 | 5336 -> 4962 | 18 -> 395 | -0.7864 | -7.0090 |
| premature_ventricular_complex | train | CPSC 2018 | 364 -> 323 | 3399 -> 3385 | 0 -> 34 | -11.2637 | -0.4119 |
| complete_rbbb | train | CPSC 2018 | 950 -> 681 | 2813 -> 2788 | 0 -> 273 | -28.3158 | -0.8887 |
| sinus_tachycardia | train | CPSC-Extra | 160 -> 150 | 1776 -> 1758 | 0 -> 18 | -6.2500 | -1.0135 |
| sinus_bradycardia | train | CPSC-Extra | 29 -> 27 | 1757 -> 1692 | 150 -> 207 | -6.8966 | -3.6995 |
| premature_ventricular_complex | train | CPSC-Extra | 103 -> 93 | 1832 -> 1824 | 1 -> 9 | -9.7087 | -0.4367 |
| incomplete_rbbb | train | CPSC-Extra | 13 -> 12 | 1897 -> 1884 | 26 -> 30 | -7.6923 | -0.6853 |
| lvh | train | CPSC-Extra | 86 -> 86 | 1850 -> 1631 | 0 -> 209 | 0.0000 | -11.8378 |
| sinus_bradycardia | validation | PTB-XL | 64 -> 64 | 2119 -> 1847 | 0 -> 270 | 0.0000 | -12.8362 |
| lvh | validation | PTB-XL | 210 -> 210 | 1940 -> 1801 | 33 -> 170 | 0.0000 | -7.1649 |
| complete_rbbb | validation | Ningbo | 256 -> 249 | 6659 -> 6225 | 57 -> 472 | -2.7344 | -6.5175 |
| incomplete_rbbb | validation | Ningbo | 58 -> 55 | 6914 -> 6472 | 0 -> 419 | -5.1724 | -6.3928 |
| lvh | validation | Ningbo | 103 -> 101 | 6097 -> 5242 | 772 -> 1603 | -1.9417 | -14.0233 |
| complete_rbbb | validation | Chapman | 87 -> 71 | 1943 -> 1943 | 6 -> 22 | -18.3908 | 0.0000 |
| lvh | validation | Chapman | 4 -> 4 | 1761 -> 1654 | 271 -> 378 | 0.0000 | -6.0761 |
| af_or_flutter | validation | Georgia | 133 -> 124 | 1910 -> 1908 | 0 -> 9 | -6.7669 | -0.1047 |
| lvh | validation | Georgia | 231 -> 230 | 1803 -> 1674 | 9 -> 137 | -0.4329 | -7.1547 |
| premature_ventricular_complex | validation | CPSC 2018 | 113 -> 99 | 1138 -> 1133 | 0 -> 11 | -12.3894 | -0.4394 |
| complete_rbbb | validation | CPSC 2018 | 334 -> 224 | 917 -> 910 | 0 -> 109 | -32.9341 | -0.7634 |
| af_or_flutter | validation | CPSC-Extra | 49 -> 46 | 604 -> 602 | 0 -> 2 | -6.1224 | -0.3311 |
| premature_ventricular_complex | validation | CPSC-Extra | 27 -> 25 | 626 -> 623 | 0 -> 2 | -7.4074 | -0.4792 |
| lvh | validation | CPSC-Extra | 33 -> 32 | 620 -> 550 | 0 -> 68 | -3.0303 | -11.2903 |
| sinus_bradycardia | fold 10 | PTB-XL | 64 -> 63 | 2134 -> 1858 | 0 -> 276 | -1.5625 | -12.9335 |
| lvh | fold 10 | PTB-XL | 214 -> 213 | 1959 -> 1820 | 25 -> 164 | -0.4673 | -7.0955 |

Positive counts include the labels of excluded records. No SPH cell loses 5%; the largest SPH change is LVH
negatives at the evaluation half, 12,598 to 12,210 (3.1%).

Eligibility, recomputed per evaluation role and source or family and compared with v4 [section `eligibility`]:
no condition loses eligibility anywhere, and every condition that entered a macro in v4 still does. One cell drops
below 100 positives and becomes imprecise under the MLC-01 rule:

| label | role | level | unit | positive_v4 | positive_v5 | negative_v4 | negative_v5 | loses_eligibility | drops_below_100 |
|---|---|---|---|---|---|---|---|---|---|
| b_premature_ventricular_complex | validation | source | cpsc_2018 | 113 | 99 | 1138 | 1133 | False | True |

At fold 10 and both SPH halves all 14 conditions still enter the macro; at fold 10, sinus tachycardia, sinus
bradycardia, sinus arrhythmia, first-degree AV block and complete RBBB stay below 100 positives, as in v4.

## Integrity of v5 against v4

[section `integrity`, `integrity.json`]

| check | value |
|---|---|
| records | 100221 |
| columns | 122 |
| cells_compared | 12226962 |
| cells_changed | 21595 |
| only_defined_to_undefined | True |
| changes_only_in_rule_columns | True |
| identity_unchanged | True |
| included_only_removed | True |
| support_v4_reproduced | True |

- Changed cells: `included` 159, `exclusion_reason` 159, the 14 label columns 21,276 together (sinus tachycardia
  234, sinus bradycardia 3,055, sinus arrhythmia 5, AF or flutter 104, PVC 64, first-degree AV block 61, complete
  RBBB 2,579, incomplete RBBB 2,156, LAFB 1,391, left axis deviation 1,074, LVH 6,970, ST depression 1,150,
  T-wave abnormality 1,226, MI 1,207), and `lik50_mi_any` 1.
- Every changed label went from 0 or 1 to undefined (`label_transitions` in `integrity.json`). `included` only
  went from True to False.
- `support_v4_reproduced`: the support code reproduces v4's own `support_B.csv` counts before it is used for v5.
- Primary-endpoint eligibility: see above. Record keys, roles and group ids are part of the identity check.

## The v5 manifest

`manifest.csv` has v4's 122 columns, with `included`, `exclusion_reason`, the `b_*` states and the `lik50_*` flags
updated, then 52 new columns: `rule_C1` to `rule_C4` (per record), `rule_C5__<condition>` to
`rule_C12__<condition>` (per record and condition, 30 columns), and the 18 Part A flags as `flag_<name>` (empty for
the 63 records shorter than 10 s, which have no flags). `mlc02_data.load_manifest` reads it as it reads v4.

## Departures and doubts

- The first build attempt stopped at its last step: the receipt could not name the scratch feature files
  repository-relative. Its staging directory was deleted unread, the naming was fixed (`61b1131`) and the build was
  rerun from the start. The successful run is the only output.
- C10 and C11 were applied as read above; the two readings that differed materially were reported to the
  orchestrator before the test-role pass. The literal reading of C10 would have made C10 redundant with C11.
- C7 cannot act at SPH, and C11 negatives were not applied there, because the label code has no SPH sinus or
  paced-rhythm evidence. SPH is therefore cleaned by C5, C8, C11 positives and C12 only.
- The test-role measurements were computed here, not by Part B. They use the same functions and settings, checked
  on train and validation, but were never validated against a reference on SPH, whose float16 storage differs.
- The rules were developed without the test roles, so a defect that occurs only there can be missed, as the
  protocol states.

## How to rerun

From the worktree, with the `data`, `outputs` and `.venv` links: delete or rename the output directory (the builder
refuses an existing one), then

```
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m \
    scripts.experiments.run_mlc03_build_v5 build --workers 14
```

`gate` runs the integrity gate alone (train and validation only). Tables:
`PYTHONPATH=. .venv/bin/python -m scripts.reports.mlc03_v5_tables [--section NAME]`. Tests:
`PYTHONPATH=. .venv/bin/python -m pytest tests/test_mlc03_dataset_v5.py`.

## Files

`outputs/mlc01_condition_dataset_20261111_v5/`: `manifest.csv`, `rule_flow.csv`, `rule_summary.csv`,
`exclusion_flow_v5.csv`, `support_v4_v5.csv`, `eligibility_changes.csv`, `part_a_count_check.csv`,
`part_b_count_check.csv`, `rule_definitions.csv`, `integrity.json`, `measurements_b_test_roles.parquet`,
`windows_test_roles.csv` and `receipt.json`.
