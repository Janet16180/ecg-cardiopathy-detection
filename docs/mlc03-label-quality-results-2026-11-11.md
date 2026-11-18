# MLC-03 Part B: label quality of the train and validation records

11 November 2026. These are the results of Part B of `docs/mlc03-data-quality-protocol-2026-11-11.md`, with its
Addendum 1 (PTB-XL+ and LUDB as measurement references). They were written from the outputs in
`outputs/mlc03_data_quality_20261111/labels/` and `outputs/mlc03_data_quality_20261111/review/labels/` by the
agent that ran them.

Every step was rerun from a clean state at commit `06dfcfe` (receipts in `labels/receipts/`). The tables are
printed by `scripts/reports/mlc03_label_quality_tables.py` (commit `1ac6d6b`); the section that prints each one is
named in brackets. No label was changed, and the rules at the end are proposals for the Part C amendment.

Nothing here is a clinical reading. The review judgements compare a label with what can be measured on the
trace, against a written bound and a stated tolerance. Clinical validation is pending.

## Main results

- Sources code the same measured finding differently. Sixteen of the 17 matched-prevalence analyses have at
  least one inconsistent pair of sources by the protocol's rule; the exception is PVC on records longer than
  10 s, with 2 comparisons.
  - Bradycardia cut-off: PTB-XL codes sinus bradycardia below about 50 per minute (implied cut-off 49.9, 95%
    interval 49.6-50.1) and CPSC-Extra below 48.3; Ningbo, Chapman and Georgia below 61.2, 61.6 and 59.3. At
    50-55 per minute the label's prevalence is 6% in PTB-XL (56/915) and 98-100% in Ningbo, Chapman and Georgia.
  - Complete versus incomplete RBBB: among RBBB-labelled records with QRS 120-130 ms, the share labelled complete
    is 34% in PTB-XL, 55% in Georgia and 98-100% in Ningbo, Chapman and CPSC 2018. Chapman and CPSC 2018 have no
    incomplete-RBBB code.
  - Left axis deviation: at an axis of -45 to -30 degrees it is coded in 100% of PTB-XL records against 49-76%
    elsewhere.
  - First-degree AV block: at PR 200-220 ms it is coded in 71% of CPSC 2018 records against 19-37% elsewhere.
- Labels outside their written measured bound by more than the tolerance:
  - complete-RBBB positives with QRS below 110 ms: 30.7% in CPSC 2018 and 18.2% in Chapman;
  - LVH positives without a voltage criterion: 39.4% in Georgia, 33.3% in PTB-XL;
  - MI positives without Q waves in two leads of one group: 69.2% in PTB-XL, 66.1% in CPSC-Extra;
  - T-wave positives with no inverted or low T in I, II or V3-V6: 55.3% in PTB-XL;
  - PVC positives without a premature beat of different shape in the first 10 s: 57.6% in Georgia;
  - in the long CPSC records, 45 of 340 PVC positives in CPSC 2018 and 10 of 82 in CPSC-Extra show the premature
    beats only after the first 10 s, which is all the models see.
- Confident-learning estimates of wrong labels are 0.3-4.5% pooled by condition. Within a source they are highest
  for T-wave abnormality in CPSC-Extra (18.2%), MI in CPSC-Extra (13.8%) and T-wave abnormality in Georgia
  (10.9%).
- Review by eye of 336 sheets of the most suspected labels (12 per condition and kind):
  - the measurement contradicts the label beyond tolerance in 149 (44%), supports it in 43 (13%), and 144 (43%)
    cannot be told;
  - among rate and rhythm false negatives, 14 of 27 are label errors where the annotation states no rhythm at all,
    against 6 of 21 where it does;
  - reliable Part A signal flags are about as common among label errors (8.7%) as among all records (8.2%).
- Contradictions inside one record are few: 94 records with sinus tachycardia and SVT (84 in Ningbo), 258 with a
  paced rhythm and a T-wave abnormality (238 in Ningbo), 27 with first-degree AV block and AF or flutter, 25 with
  sinus tachycardia and AF or flutter.
- The undefined-entry rules mostly hold on the measurements. For example, 94.3% of CPSC-Extra's "bradycardia of
  unstated origin" records have a sinus rate below 60 per minute.

## Records, roles and guards [records]

Only records with roles `train` and `validation` in the MLC-01 v4 manifest were read.

- Manifest: the loader reads the role column first and never parses the rows of the other roles
  (`load_open_records`; test `test_loader_never_parses_closed_rows`).
- Waveforms and Part A flags: the waveform cache reader and the Part A flag reader raise an error on a closed role.
- PTB-XL+: rows are cut to PTB-XL folds 1-9 by `ecg_id` before any row is parsed (test
  `test_ptbxl_plus_loader_drops_fold_ten`). All 19,601 open PTB-XL records are in folds 1-9.
- Never opened: any Challenge test group, SPH, EchoNext or MIMIC record, the MLC-02 test predictions or test
  tables, and Part A's test-role flag file.

| source | train | validation | records | longer than 10 s |
|---|---|---|---|---|
| PTB-XL | 17418 | 2183 | 19601 | 0 |
| Ningbo | 20919 | 6972 | 27891 | 0 |
| Chapman | 6097 | 2036 | 8133 | 0 |
| Georgia | 6117 | 2043 | 8160 | 0 |
| CPSC 2018 | 3763 | 1251 | 5014 | 3312 |
| CPSC-Extra | 1936 | 653 | 2589 | 1514 |
| total | 56250 | 15138 | 71388 | 4826 |

| method B status | records |
|---|---|
| ok | 71069 |
| no_dominant | 232 |
| few_beats | 86 |
| flat | 1 |

Runs used the CPU only: 7 worker processes, with OpenBLAS and OpenMP at 2 threads each. OpenBLAS chose its own
kernel, `SkylakeX`, because `OPENBLAS_CORETYPE` was unset; every receipt records this. The EDAC uncorrected-error
count was 1 before and after.

The measurement references are public PhysioNet data under `data/raw/`, not committed and without a DVC pointer
(`docs/data-versioning.md` tracks only completed, selected sources registered in `configs/datasets.json`):

- PTB-XL+ 1.0.1 (https://physionet.org/content/ptb-xl-plus/1.0.1/): features only plus the manifest, 228.7 MB
  (136.1 MB on disk); no median beats or fiducial files. SHA-256 of `features/12sl_features.csv`
  `cd805ec672c305d0d5079bb87c83d629b91cf3534ce18d3717a960998d5e8b29`, of `features/unig_features.csv`
  `f95ab01df5c9ed329485fd811578f0c39eabb0667a36ec37c7dda427d9d8f5a5`, of `features/feature_description.csv`
  `5d8358c3b1ec7315adadc4b6dfe9f25a6f9a7c1a870e8ccb911bc59c662eed3b`, and of `SHA256SUMS.txt`
  `1bed47b5cc4584f0b2fd9da66187efd473f43f0410392669aa3435bb8fd5bd10`.
- LUDB 1.0.1 (https://physionet.org/content/ludb/1.0.1/): 29.0 MB (43.0 MB on disk), all 2,805 files verified.
  SHA-256 of `SHA256SUMS.txt` `cccef1f3529519db8f26a333c97a6872a7d0e3e5c64a1448b76b161fd87fb75f`.
- Both are re-verified against their manifests in the `calibrate` and `validate` receipts on every run.

## Measurements and their error [validation, agreement]

Two methods measure each record:

- Method A is MLC-02's frozen extractor (the stored 130 measurements).
- Method B is new (`ecg_experiment/mlc03_measurements.py`).

The consensus keeps method B's value where both agree within 5 per minute (rate), 15 ms (QRS), 25 ms (PR) or
20 degrees (axis). Method B's QRS and P thresholds were chosen on 1,500 PTB-XL training records against PTB-XL+
(QRS fraction 0.04, P fraction 0.2). In the tables, "failed" is the share without a value; for the consensus it
also counts records where the methods disagree.

PTB-XL+ reference (12SL and Uni-G where they agree), PTB-XL validation records outside the calibration sample:

| quantity | method | tolerance | pairs | failed | bias (median difference) | median absolute | 90th percentile absolute | within tolerance |
|---|---|---|---|---|---|---|---|---|
| hr | A | 5.0 | 1357 | 0.1% | 0.226 | 0.244 | 0.479 | 99.3% |
| hr | B | 5.0 | 1357 | 0.0% | 0.228 | 0.246 | 0.476 | 98.1% |
| hr | consensus | 5.0 | 1357 | 1.9% | 0.226 | 0.24 | 0.464 | 99.7% |
| qrs | A | 10.0 | 1227 | 0.2% | -5 | 6 | 14 | 79.6% |
| qrs | B | 10.0 | 1227 | 0.2% | 0 | 4 | 9 | 93.3% |
| qrs | consensus | 10.0 | 1227 | 1.3% | 0 | 4 | 9 | 93.6% |
| pr | A | 20.0 | 1115 | 8.7% | -7 | 7 | 16 | 94.2% |
| pr | B | 20.0 | 1115 | 7.9% | -4 | 6 | 14 | 95.6% |
| pr | consensus | 20.0 | 1115 | 20.4% | -4.5 | 6 | 12 | 99.1% |
| axis | A | 15.0 | 1357 | 0.1% | -1.57 | 4.36 | 13.2 | 93.1% |
| axis | B | 15.0 | 1357 | 0.3% | -1.44 | 4.51 | 13 | 93.1% |
| axis | consensus | 15.0 | 1357 | 1.0% | -1.49 | 4.49 | 12.8 | 93.3% |
| sokolow | A | 0.5 | 1341 | 0.1% | -0.0729 | 0.105 | 0.594 | 88.9% |
| sokolow | B | 0.5 | 1341 | 0.1% | -0.125 | 0.132 | 0.601 | 88.8% |
| cornell | A | 0.5 | 1351 | 0.1% | -0.0675 | 0.0754 | 0.209 | 97.3% |
| cornell | B | 0.5 | 1351 | 0.2% | -0.0961 | 0.0992 | 0.24 | 96.6% |
| st80_V5 | B | 0.05 | 1382 | 0.3% | -0.000494 | 0.00486 | 0.0156 | 98.2% |
| t_V5 | A | 0.1 | 1310 | 0.1% | 0.00466 | 0.00733 | 0.0255 | 98.3% |
| t_V5 | B | 0.1 | 1310 | 0.2% | 0.00474 | 0.00718 | 0.0253 | 97.9% |
| t_II | A | 0.1 | 1328 | 0.1% | 0.00592 | 0.00822 | 0.0252 | 98.5% |
| t_II | B | 0.1 | 1328 | 0.2% | 0.00585 | 0.00806 | 0.0291 | 96.8% |
| qdur_II | B | 10.0 | 1247 | 0.2% | 0 | 0 | 4.5 | 96.1% |
| qdur_V5 | B | 10.0 | 1302 | 0.2% | 0 | 0 | 6 | 95.8% |

LUDB manual delineation (200 records):

| quantity | method | tolerance | pairs | failed | bias (median difference) | median absolute | 90th percentile absolute | within tolerance |
|---|---|---|---|---|---|---|---|---|
| hr | A | 5.0 | 200 | 0.0% | -0.0399 | 0.21 | 1.32 | 94.5% |
| hr | B | 5.0 | 200 | 2.0% | -0.0473 | 0.261 | 1.81 | 94.4% |
| qrs | A | 10.0 | 200 | 2.0% | -28 | 28 | 44.5 | 4.6% |
| qrs | B | 10.0 | 200 | 2.5% | -24 | 24 | 40 | 8.7% |
| pr | A | 20.0 | 175 | 20.0% | -10 | 10 | 28.1 | 82.9% |
| pr | B | 20.0 | 175 | 5.1% | -10 | 10 | 24.5 | 88.0% |

Synthetic records (all noise levels):

| quantity | method | tolerance | pairs | failed | bias (median difference) | median absolute | 90th percentile absolute | within tolerance |
|---|---|---|---|---|---|---|---|---|
| hr | A | 5.0 | 700 | 0.0% | 0 | 0 | 0.881 | 100.0% |
| hr | B | 5.0 | 700 | 0.0% | 0 | 0 | 0.131 | 97.4% |
| qrs | A | 10.0 | 700 | 0.1% | 16 | 16 | 25.2 | 4.1% |
| qrs | B | 10.0 | 700 | 0.6% | 14.6 | 14.6 | 18.4 | 3.4% |
| pr | A | 20.0 | 600 | 12.5% | -21.9 | 22.2 | 33.7 | 21.7% |
| pr | B | 20.0 | 600 | 45.5% | -21 | 21 | 24.9 | 31.8% |
| axis | A | 15.0 | 700 | 0.0% | 0.378 | 2.73 | 6.14 | 99.6% |
| axis | B | 15.0 | 700 | 0.6% | 0.493 | 2.52 | 5.59 | 99.9% |
| st60_II | A | 0.05 | 700 | 0.0% | 0.000621 | 0.00296 | 0.0232 | 96.7% |
| st60_II | B | 0.05 | 700 | 0.6% | 0.000325 | 0.00389 | 0.0171 | 98.4% |
| t_II | A | 0.1 | 700 | 0.0% | 0.00136 | 0.0114 | 0.132 | 85.1% |
| t_II | B | 0.1 | 700 | 0.6% | 0.00344 | 0.0139 | 0.151 | 79.3% |

Synthetic categorical findings:

| finding | with_finding | flagged_with | without_finding | flagged_without |
|---|---|---|---|---|
| rbbb_morphology | 221 | 100.0% | 475 | 1.1% |
| p_present | 597 | 60.5% | 99 | 7.1% |
| irregular | 99 | 100.0% | 597 | 29.6% |
| premature_aberrant | 97 | 100.0% | 599 | 1.5% |

PTB-XL+ findings where 12SL and Uni-G agree (rows: reference; columns: method B):

| finding | reference | False | True |
|---|---|---|---|
| p_present | False | 1268.0 | 298.0 |
| p_present | True | 720.0 | 16652.0 |
| v1_two_positive_peaks | False | 15108.0 | 2762.0 |
| v1_two_positive_peaks | True | 232.0 | 1142.0 |

On LUDB's rhythm statements (`measurement_validation/ludb_findings.csv`), method B flags 11 of 17 AF or flutter
records, and 3.4% of the others. It counts a premature aberrant beat in 6 of 7 records with ventricular
extrasystoles, and in 3.2% of the others.

Method agreement and failures per source:

| source | records | A failed (flat lead / no beats) | B failed | hr: agree / consensus | qrs: agree / consensus | pr: agree / consensus | axis: agree / consensus | P wave present (B) |
|---|---|---|---|---|---|---|---|---|
| PTB-XL | 19601 | 1 / 8 | 49 | 98.3% / 98.2% | 98.6% / 97.7% | 92.6% / 69.2% | 99.3% / 99.0% | 88.6% |
| Chapman | 8133 | 13 / 0 | 6 | 98.5% / 98.4% | 98.7% / 97.9% | 90.9% / 45.9% | 99.3% / 99.1% | 68.1% |
| CPSC 2018 | 5014 | 9 / 7 | 25 | 96.8% / 96.5% | 97.4% / 95.9% | 89.3% / 50.2% | 98.2% / 97.4% | 73.8% |
| CPSC-Extra | 2589 | 5 / 2 | 21 | 97.2% / 96.7% | 98.6% / 97.2% | 90.1% / 59.6% | 99.2% / 98.2% | 81.2% |
| Georgia | 8160 | 6 / 2 | 28 | 98.1% / 97.9% | 98.3% / 97.3% | 90.2% / 58.7% | 99.2% / 98.8% | 85.6% |
| Ningbo | 27891 | 1202 / 87 | 190 | 97.3% / 92.7% | 98.4% / 92.7% | 91.4% / 47.1% | 98.8% / 93.8% | 70.2% |

### Which label checks the measurement error makes unreliable

Reliable checks:

- Heart rate (sinus tachycardia and bradycardia): 99.7% of consensus values are within 5 per minute of PTB-XL+;
  94.4-94.5% of either method's values are within 5 of LUDB, with a 90th-percentile error of 1.3-1.8.
- Frontal axis (LAFB, left axis deviation): 93.3% within 15 degrees of PTB-XL+, bias -1.5 degrees; 99.6-99.9%
  on synthetic records. Only labels within about 15 degrees of -30 or -45 are uncertain.
- Voltage (LVH): Sokolow-Lyon within 0.5 mV for 88.8-88.9%, Cornell for 96.6-97.3%, bias -0.07 to -0.13 mV.
- ST level (ST depression): within 0.05 mV for 98.2%.
- T amplitude (T-wave abnormality): within 0.1 mV for 96.8-98.5%.
- Q duration (MI): within 10 ms for 95.8-96.1%.

Uncertain only near a cut-off:

- QRS duration (complete and incomplete RBBB): method B is within 10 ms of PTB-XL+ for 93.3%, with no bias;
  method A runs 5 ms short. Against LUDB both methods are 24-28 ms short; LUDB's reference QRS runs from the
  earliest lead onset to the latest lead offset, with a median of 121 ms. Labels within about 10 ms of 110 or
  120 ms therefore depend on the convention. The incomplete-RBBB check, a 10 ms window, is the least reliable:
  every reviewed incomplete-RBBB false negative fell in this band.

Unreliable checks:

- PR interval (first-degree AV block): the consensus fails for 20.4% of held-out PTB-XL+ records and exists for
  only 45.9-59.6% of records in the five Challenge sources, against 69.2% in PTB-XL. Against LUDB the bias is
  -10 ms and 82.9-88.0% are within 20 ms. The AV-block counts cover about half the records in Chapman, Ningbo and
  CPSC 2018, and labels with PR within 20 ms of 200 ms are uncertain.
- P-wave presence (the sinus population, and the "no P wave" part of the AF check): method B finds a P wave in
  16,652 of 17,372 records where both PTB-XL+ programs report one, but also in 298 of 1,566 where both report
  none; on synthetic records it finds P waves in only 60.5%.
- RR irregularity (AF and sinus arrhythmia): there is no external reference beyond LUDB's 11 of 17 AF records;
  on synthetic records it also flags 29.6% of regular records, so "AF negatives meeting the definition" are
  inflated on noisy records.
- V1 two-peak morphology (part of the incomplete-RBBB check): it finds 1,142 of 1,374 records with a PTB-XL+ R'
  and wrongly flags 2,762 of 17,870 without one.

Partial by design: the ST, T and MI checks cover only ST depression, T inversion or flattening, and Q waves, not
every finding their codes describe. This is not a measurement error.

## Item 1: label against measurement [definitions, item1, whole]

Each condition has a written measured definition: a bound, its reference, and the population it is read on. For
the rate labels the population is records with sinus evidence: no non-sinus rhythm code, a P wave, and agreeing
methods. "Outside" counts positives that miss the bound and negatives that meet it; "beyond tolerance" counts only
those that miss or meet it by more than the tolerance.

| condition | measure | population | bound | tolerance | reference |
|---|---|---|---|---|---|
| sinus_tachycardia | hr | sinus | heart rate above 100 per minute, sinus rhythm | 5.0 | Mason et al. 2007 (AHA/ACCF/HRS part II) |
| sinus_bradycardia | hr | sinus | heart rate below 60 per minute, sinus rhythm | 5.0 | Mason et al. 2007 (AHA/ACCF/HRS part II) |
| sinus_arrhythmia | rr_range_dominant_b | sinus | longest minus shortest RR between consecutive dominant beats above 120 ms | 20.0 | a commonly used cut-off; no single standard |
| af_or_flutter | rr_nrmssd_b | measured | irregular RR (normalized RMSSD of at least 0.10) without a detectable P wave | 0.03 | AF: irregular RR and no distinct P waves (Hindricks et al. 2021, ESC) |
| premature_ventricular_complex | n_premature_aberrant_any | ten_seconds | at least one premature beat unlike the dominant beat in the first 10 s | 0.0 | premature complex of different QRS morphology |
| first_degree_av_block | pr | p_present | PR interval above 200 ms | 20.0 | Mason et al. 2007 (AHA/ACCF/HRS part II) |
| complete_rbbb | qrs | measured | QRS duration of at least 120 ms | 10.0 | Surawicz et al. 2009 (AHA/ACCF/HRS part III) |
| incomplete_rbbb | qrs | measured | QRS duration 110-119 ms with RBBB morphology | 10.0 | Surawicz et al. 2009 (AHA/ACCF/HRS part III) |
| lafb | axis | measured | frontal QRS axis -45 to -90 degrees | 15.0 | Surawicz et al. 2009 (part III) |
| left_axis_deviation | axis | measured | frontal QRS axis -30 to -90 degrees | 15.0 | Surawicz et al. 2009 (part III) |
| lvh | sokolow_mv | measured | Sokolow-Lyon at least 3.5 mV or Cornell above 2.8 mV (men) / 2.0 mV (women) | 0.5 | Hancock et al. 2009 (AHA/ACCF/HRS part V) |
| st_depression | st_min_mv | measured | ST at least 0.05 mV below the PR level in two leads of one group | 0.05 | Thygesen et al. 2018 (Fourth Universal Definition of MI) |
| t_wave_abnormality | t_inverted | measured | T inverted (at most -0.1 mV) or low (below 0.05 mV) in I, II or V3-V6 | 0.0 | Rautaharju et al. 2009 (AHA/ACCF/HRS part IV) |
| mi_any | q_group | measured | infarct-size Q waves in two leads of one group | 0.0 | Thygesen et al. 2018 (Fourth Universal Definition of MI) |

| condition | source | positives measured | positive median | positives outside (strict / beyond tolerance) | negatives measured | negatives inside (strict / beyond tolerance) |
|---|---|---|---|---|---|---|
| sinus_tachycardia | PTB-XL | 633 | 107.72 | 7 (1.1%) / 5 (0.8%) | 15538 | 83 (0.5%) / 12 (0.1%) |
| sinus_tachycardia | Ningbo | 3100 | 108.615 | 15 (0.5%) / 9 (0.3%) | 14128 | 26 (0.2%) / 9 (0.1%) |
| sinus_tachycardia | Chapman | 1014 | 107.845 | 1 (0.1%) / 1 (0.1%) | 4091 | 7 (0.2%) / 1 (0.0%) |
| sinus_tachycardia | Georgia | 742 | 112.088 | 9 (1.2%) / 4 (0.5%) | 5954 | 54 (0.9%) / 21 (0.4%) |
| sinus_tachycardia | CPSC-Extra | 157 | 110.61 | 6 (3.8%) / 5 (3.2%) | 1886 | 44 (2.3%) / 16 (0.8%) |
| sinus_bradycardia | PTB-XL | 545 | 48.713 | 7 (1.3%) / 2 (0.4%) | 15626 | 2525 (16.2%) / 869 (5.6%) |
| sinus_bradycardia | Ningbo | 8864 | 56.013 | 90 (1.0%) / 34 (0.4%) | 8363 | 127 (1.5%) / 27 (0.3%) |
| sinus_bradycardia | Chapman | 2786 | 55.944 | 27 (1.0%) / 9 (0.3%) | 2319 | 16 (0.7%) / 1 (0.0%) |
| sinus_bradycardia | Georgia | 1172 | 55.146 | 1 (0.1%) / 1 (0.1%) | 5523 | 111 (2.0%) / 15 (0.3%) |
| sinus_bradycardia | CPSC-Extra | 27 | 57.678 | 3 (11.1%) / 2 (7.4%) | 1858 | 75 (4.0%) / 15 (0.8%) |
| sinus_arrhythmia | PTB-XL | 635 | 258.0 | 101 (15.9%) / 89 (14.0%) | 15523 | 2377 (15.3%) / 1750 (11.3%) |
| sinus_arrhythmia | Ningbo | 1424 | 220.0 | 62 (4.4%) / 32 (2.2%) | 15789 | 1701 (10.8%) / 1119 (7.1%) |
| sinus_arrhythmia | Georgia | 326 | 230.0 | 6 (1.8%) / 4 (1.2%) | 6362 | 745 (11.7%) / 587 (9.2%) |
| sinus_arrhythmia | CPSC-Extra | 6 | 305.0 | 1 (16.7%) / 1 (16.7%) | 2036 | 198 (9.7%) / 163 (8.0%) |
| af_or_flutter | PTB-XL | 1409 | 0.27 | 41 (2.9%) / 32 (2.3%) | 18143 | 544 (3.0%) / 511 (2.8%) |
| af_or_flutter | Ningbo | 6002 | 0.264 | 313 (5.2%) / 283 (4.7%) | 21699 | 944 (4.4%) / 836 (3.9%) |
| af_or_flutter | Chapman | 1787 | 0.264 | 79 (4.4%) / 75 (4.2%) | 6340 | 151 (2.4%) / 135 (2.1%) |
| af_or_flutter | Georgia | 568 | 0.214 | 100 (17.6%) / 90 (15.8%) | 7564 | 287 (3.8%) / 274 (3.6%) |
| af_or_flutter | CPSC 2018 | 797 | 0.281 | 5 (0.6%) / 4 (0.5%) | 4192 | 336 (8.0%) / 321 (7.7%) |
| af_or_flutter | CPSC-Extra | 156 | 0.258 | 9 (5.8%) / 8 (5.1%) | 2412 | 121 (5.0%) / 112 (4.6%) |
| premature_ventricular_complex | PTB-XL | 1021 | 1.0 | 151 (14.8%) / 151 (14.8%) | 18479 | 525 (2.8%) / 525 (2.8%) |
| premature_ventricular_complex | Ningbo | 857 | 1.0 | 254 (29.6%) / 254 (29.6%) | 26734 | 1336 (5.0%) / 1336 (5.0%) |
| premature_ventricular_complex | Chapman | 237 | 1.0 | 73 (30.8%) / 73 (30.8%) | 7890 | 248 (3.1%) / 248 (3.1%) |
| premature_ventricular_complex | Georgia | 297 | 0.0 | 171 (57.6%) / 171 (57.6%) | 7835 | 236 (3.0%) / 236 (3.0%) |
| premature_ventricular_complex | CPSC 2018 | 134 | 1.0 | 28 (20.9%) / 28 (20.9%) | 1562 | 131 (8.4%) / 131 (8.4%) |
| premature_ventricular_complex | CPSC-Extra | 48 | 1.0 | 19 (39.6%) / 19 (39.6%) | 1016 | 65 (6.4%) / 65 (6.4%) |
| first_degree_av_block | PTB-XL | 329 | 222.0 | 36 (10.9%) / 8 (2.4%) | 12871 | 583 (4.5%) / 125 (1.0%) |
| first_degree_av_block | Ningbo | 267 | 222.0 | 13 (4.9%) / 1 (0.4%) | 11858 | 336 (2.8%) / 37 (0.3%) |
| first_degree_av_block | Chapman | 84 | 222.0 | 0 (0.0%) / 0 (0.0%) | 3395 | 110 (3.2%) / 10 (0.3%) |
| first_degree_av_block | Georgia | 194 | 222.0 | 12 (6.2%) / 3 (1.5%) | 4426 | 143 (3.2%) / 24 (0.5%) |
| first_degree_av_block | CPSC 2018 | 144 | 220.0 | 10 (6.9%) / 3 (2.1%) | 2123 | 28 (1.3%) / 4 (0.2%) |
| first_degree_av_block | CPSC-Extra | 22 | 216.0 | 3 (13.6%) / 2 (9.1%) | 1398 | 25 (1.8%) / 1 (0.1%) |
| complete_rbbb | PTB-XL | 449 | 146.0 | 21 (4.7%) / 8 (1.8%) | 18693 | 1284 (6.9%) / 867 (4.6%) |
| complete_rbbb | Ningbo | 919 | 134.0 | 123 (13.4%) / 47 (5.1%) | 24662 | 547 (2.2%) / 474 (1.9%) |
| complete_rbbb | Chapman | 296 | 130.0 | 91 (30.7%) / 54 (18.2%) | 7643 | 115 (1.5%) / 76 (1.0%) |
| complete_rbbb | Georgia | 349 | 140.0 | 31 (8.9%) / 15 (4.3%) | 7487 | 413 (5.5%) / 269 (3.6%) |
| complete_rbbb | CPSC 2018 | 1221 | 126.0 | 503 (41.2%) / 375 (30.7%) | 3586 | 155 (4.3%) / 126 (3.5%) |
| complete_rbbb | CPSC-Extra | 51 | 136.0 | 7 (13.7%) / 6 (11.8%) | 2434 | 72 (3.0%) / 48 (2.0%) |
| incomplete_rbbb | PTB-XL | 983 | 100.0 | 874 (88.9%) / 557 (56.7%) | 18159 | 250 (1.4%) / 0 (0.0%) |
| incomplete_rbbb | Ningbo | 189 | 102.0 | 146 (77.2%) / 75 (39.7%) | 25661 | 270 (1.1%) / 0 (0.0%) |
| incomplete_rbbb | Georgia | 257 | 106.0 | 207 (80.5%) / 103 (40.1%) | 7642 | 95 (1.2%) / 0 (0.0%) |
| incomplete_rbbb | CPSC-Extra | 17 | 94.0 | 15 (88.2%) / 13 (76.5%) | 2468 | 33 (1.3%) / 0 (0.0%) |
| lafb | PTB-XL | 1445 | -46.498 | 727 (50.3%) / 168 (11.6%) | 17966 | 440 (2.4%) / 113 (0.6%) |
| lafb | Ningbo | 286 | -58.186 | 71 (24.8%) / 21 (7.3%) | 25870 | 978 (3.8%) / 373 (1.4%) |
| lafb | Georgia | 146 | -54.104 | 48 (32.9%) / 11 (7.5%) | 7912 | 324 (4.1%) / 104 (1.3%) |
| left_axis_deviation | PTB-XL | 1233 | -45.691 | 217 (17.6%) / 47 (3.8%) | 7318 | 9 (0.1%) / 4 (0.1%) |
| left_axis_deviation | Ningbo | 891 | -45.171 | 199 (22.3%) / 58 (6.5%) | 24985 | 883 (3.5%) / 467 (1.9%) |
| left_axis_deviation | Chapman | 300 | -43.806 | 80 (26.7%) / 25 (8.3%) | 7757 | 241 (3.1%) / 118 (1.5%) |
| left_axis_deviation | Georgia | 725 | -38.647 | 233 (32.1%) / 45 (6.2%) | 7188 | 180 (2.5%) / 73 (1.0%) |
| lvh | PTB-XL | 1917 | 3.033 | 889 (46.4%) / 638 (33.3%) | 17394 | 1140 (6.6%) / 55 (0.3%) |
| lvh | Ningbo | 492 | 4.267 | 85 (17.3%) / 39 (7.9%) | 23954 | 1555 (6.5%) / 167 (0.7%) |
| lvh | Chapman | 11 | 3.516 | 4 (36.4%) / 0 (0.0%) | 7125 | 501 (7.0%) / 101 (1.4%) |
| lvh | Georgia | 994 | 2.64 | 510 (51.3%) / 392 (39.4%) | 7111 | 427 (6.0%) / 36 (0.5%) |
| lvh | CPSC-Extra | 119 | 4.262 | 23 (19.3%) / 15 (12.6%) | 2449 | 260 (10.6%) / 58 (2.4%) |
| st_depression | PTB-XL | 907 | -0.078 | 164 (18.1%) / 0 (0.0%) | 15931 | 1714 (10.8%) / 1038 (6.5%) |
| st_depression | Ningbo | 1007 | -0.122 | 81 (8.0%) / 1 (0.1%) | 23585 | 1595 (6.8%) / 925 (3.9%) |
| st_depression | Chapman | 320 | -0.13 | 21 (6.6%) / 0 (0.0%) | 7144 | 728 (10.2%) / 473 (6.6%) |
| st_depression | CPSC 2018 | 658 | -0.045 | 386 (58.7%) / 3 (0.5%) | 4331 | 690 (15.9%) / 410 (9.5%) |
| st_depression | CPSC-Extra | 41 | -0.072 | 15 (36.6%) / 1 (2.4%) | 861 | 121 (14.1%) / 63 (7.3%) |
| t_wave_abnormality | PTB-XL | 2338 | 0.0 | 1293 (55.3%) / 1293 (55.3%) | 14408 | 2100 (14.6%) / 835 (5.8%) |
| t_wave_abnormality | Ningbo | 5213 | 0.0 | 1640 (31.5%) / 1640 (31.5%) | 19552 | 2519 (12.9%) / 371 (1.9%) |
| t_wave_abnormality | Chapman | 1545 | 0.0 | 509 (32.9%) / 509 (32.9%) | 5899 | 654 (11.1%) / 205 (3.5%) |
| t_wave_abnormality | Georgia | 2161 | 1.0 | 541 (25.0%) / 541 (25.0%) | 4397 | 766 (17.4%) / 217 (4.9%) |
| t_wave_abnormality | CPSC-Extra | 20 | 2.0 | 4 (20.0%) / 4 (20.0%) | 874 | 295 (33.8%) / 97 (11.1%) |
| mi_any | PTB-XL | 4751 | 1.0 | 3286 (69.2%) / 3286 (69.2%) | 14542 | 260 (1.8%) / 260 (1.8%) |
| mi_any | Ningbo | 122 | 2.0 | 46 (37.7%) / 46 (37.7%) | 26492 | 1593 (6.0%) / 1593 (6.0%) |
| mi_any | Chapman | 31 | 2.0 | 13 (41.9%) / 13 (41.9%) | 7920 | 117 (1.5%) / 117 (1.5%) |
| mi_any | CPSC-Extra | 1088 | 1.0 | 719 (66.1%) / 719 (66.1%) | 1480 | 34 (2.3%) / 34 (2.3%) |

The labels of records longer than 10 s (CPSC 2018 and CPSC-Extra only) describe the whole recording, while the
models see only the first 10 s. Method B was therefore run on every 10 s window of these records:

| condition | source | label | records | in_first_10s | only_after_10s | in_no_window |
|---|---|---|---|---|---|---|
| premature_ventricular_complex | CPSC 2018 | negative | 2972 | 218 | 79 | 2675 |
| premature_ventricular_complex | CPSC 2018 | positive | 340 | 200 | 45 | 95 |
| premature_ventricular_complex | CPSC-Extra | negative | 1431 | 79 | 31 | 1321 |
| premature_ventricular_complex | CPSC-Extra | positive | 82 | 32 | 10 | 40 |
| premature_ventricular_complex | CPSC-Extra | undefined | 1 | 1 | 0 | 0 |
| af_or_flutter | CPSC 2018 | negative | 2883 | 217 | 67 | 2599 |
| af_or_flutter | CPSC 2018 | positive | 429 | 360 | 15 | 54 |
| af_or_flutter | CPSC-Extra | negative | 1439 | 65 | 24 | 1350 |
| af_or_flutter | CPSC-Extra | positive | 75 | 61 | 6 | 8 |

## Item 2: how the sources code each condition [item2, prevalence]

A pair of sources is inconsistent in a bin of the measurement when the label's prevalences differ by a factor of
two or more, both sources have at least 30 defined records in the bin, and their 95% Wilson intervals do not
overlap.

| analysis | bin comparisons | inconsistent | bins with an inconsistent pair | sources involved |
|---|---|---|---|---|
| sinus_tachycardia | 85 | 1 | 1 | CPSC-Extra, PTB-XL |
| sinus_bradycardia | 68 | 23 | 6 | CPSC-Extra, Chapman, Georgia, Ningbo, PTB-XL |
| sinus_arrhythmia | 45 | 21 | 8 | CPSC-Extra, Georgia, Ningbo, PTB-XL |
| af_or_flutter\|no_p | 91 | 7 | 2 | CPSC 2018, CPSC-Extra, Chapman, Georgia, Ningbo, PTB-XL |
| af_or_flutter\|p_present | 105 | 43 | 7 | CPSC 2018, CPSC-Extra, Chapman, Georgia, Ningbo, PTB-XL |
| premature_ventricular_complex\|10s | 50 | 10 | 3 | CPSC 2018, CPSC-Extra, Chapman, Georgia, Ningbo, PTB-XL |
| premature_ventricular_complex\|longer | 2 | 0 | 0 |  |
| first_degree_av_block | 103 | 8 | 2 | CPSC 2018, CPSC-Extra, Chapman, Ningbo, PTB-XL |
| complete_rbbb | 109 | 40 | 6 | CPSC 2018, CPSC-Extra, Chapman, Georgia, Ningbo, PTB-XL |
| incomplete_rbbb | 42 | 23 | 8 | CPSC-Extra, Georgia, Ningbo, PTB-XL |
| complete_among_rbbb | 64 | 22 | 5 | CPSC 2018, Chapman, Georgia, Ningbo, PTB-XL |
| lafb | 33 | 12 | 6 | Georgia, Ningbo, PTB-XL |
| left_axis_deviation | 61 | 12 | 6 | Chapman, Georgia, Ningbo, PTB-XL |
| lvh | 93 | 63 | 9 | CPSC-Extra, Chapman, Georgia, Ningbo, PTB-XL |
| st_depression | 58 | 28 | 7 | CPSC 2018, CPSC-Extra, Chapman, Ningbo, PTB-XL |
| t_wave_abnormality | 50 | 30 | 5 | CPSC-Extra, Chapman, Georgia, Ningbo, PTB-XL |
| mi_any | 21 | 14 | 4 | CPSC-Extra, Chapman, Ningbo, PTB-XL |

| condition | source | records | positives | cutoff |
|---|---|---|---|---|
| sinus_bradycardia | PTB-XL | 16171 | 545 | 49.9 (49.6, 50.1) |
| sinus_bradycardia | Ningbo | 17227 | 8864 | 61.2 (61.0, 61.4) |
| sinus_bradycardia | Chapman | 5105 | 2786 | 61.6 (61.1, 62.0) |
| sinus_bradycardia | Georgia | 6695 | 1172 | 59.3 (59.2, 59.5) |
| sinus_bradycardia | CPSC-Extra | 1885 | 27 | 48.3 (40.0, 52.4) |
| sinus_tachycardia | PTB-XL | 16171 | 633 | 101.5 (101.2, 102.0) |
| sinus_tachycardia | Ningbo | 17228 | 3100 | 98.2 (97.7, 98.8) |
| sinus_tachycardia | Chapman | 5105 | 1014 | 98.2 (96.9, 100.4) |
| sinus_tachycardia | Georgia | 6696 | 742 | 101.6 (101.1, 102.1) |
| sinus_tachycardia | CPSC-Extra | 2043 | 157 | 105.0 (103.6, 106.5) |

Remaining cases. The protocol makes a condition undefined where a source does not code it separately. This table
lists the cases where a label is still defined although the source never uses the sibling code that would
separate it:

| condition | source | sibling | sibling_codes | positives | negatives |
|---|---|---|---|---|---|
| complete_rbbb | Chapman | incomplete RBBB | 713426002 | 325 | 7777 |
| complete_rbbb | CPSC 2018 | incomplete RBBB | 713426002 | 1284 | 3730 |
| left_axis_deviation | Chapman | LAFB | 445118002 | 305 | 7828 |
| af_or_flutter | Ningbo | atrial fibrillation | 164889003 282825002 314208002 426749004 | 6068 | 21823 |
| af_or_flutter | CPSC 2018 | atrial flutter | 164890007 | 800 | 4214 |
| lvh | Georgia | LV high voltage | 55827005 | 994 | 7139 |
| lvh | CPSC-Extra | LV high voltage | 55827005 | 119 | 2470 |
| st_depression | CPSC 2018 | nonspecific or ischemic ST-T | 164861001 164930006 413444003 413844008 425419005 425623009 426434006 428750005 55930002 | 660 | 4354 |
| first_degree_av_block | PTB-XL | AV block of unstated degree | (not codable) | 912 | 18689 |
| first_degree_av_block | CPSC 2018 | AV block of unstated degree | 233917008 | 530 | 4484 |
| premature_ventricular_complex | PTB-XL | ventricular tachycardia | (not codable) | 1029 | 18519 |
| premature_ventricular_complex | Chapman | ventricular tachycardia | 164895002 425856008 | 240 | 7893 |
| premature_ventricular_complex | Georgia | ventricular tachycardia | 164895002 425856008 | 305 | 7855 |
| premature_ventricular_complex | CPSC 2018 | ventricular tachycardia | 164895002 425856008 | 477 | 4537 |
| sinus_bradycardia | PTB-XL | bradycardia of unstated origin | (not codable) | 573 | 19028 |
| sinus_bradycardia | Chapman | bradycardia of unstated origin | 426627000 | 3081 | 5052 |

Matched prevalence, as prevalence (positives/defined) per bin of the measurement. Bins with fewer than 30 defined
records are in brackets. Only the bins around each cut-off are shown; `--section prevalence --name <analysis>`
prints every bin.

Sinus bradycardia by heart rate (per minute), sinus records:

| bin | PTB-XL | Ningbo | Chapman | Georgia | CPSC-Extra |
|---|---|---|---|---|---|
| 45 to 50 | 0.97 (302/312) | 0.99 (732/740) | 1.00 (252/252) | 0.98 (164/168) | [1/3] |
| 50 to 55 | 0.06 (56/915) | 0.99 (2685/2699) | 1.00 (809/809) | 0.98 (336/344) | [10/21] |
| 55 to 60 | 0.05 (85/1738) | 0.98 (5210/5305) | 0.99 (1640/1655) | 0.87 (600/693) | 0.18 (13/71) |
| 60 to 65 | 0.00 (5/2272) | 0.06 (56/868) | 0.09 (18/198) | 0.00 (0/707) | 0.01 (1/194) |

Complete RBBB among records labelled complete or incomplete RBBB, by QRS (ms):

| bin | PTB-XL | Ningbo | Chapman | Georgia | CPSC 2018 | CPSC-Extra |
|---|---|---|---|---|---|---|
| 90 to 100 | 0.01 (2/278) | 0.13 (7/55) | [18/18] | 0.07 (6/80) | 1.00 (138/138) | [2/7] |
| 100 to 110 | 0.02 (4/230) | 0.18 (15/82) | [22/22] | 0.04 (3/77) | 1.00 (123/123) | [1/3] |
| 110 to 120 | 0.11 (13/122) | 0.64 (76/119) | 1.00 (37/37) | 0.24 (16/66) | 1.00 (128/128) | [1/3] |
| 120 to 130 | 0.34 (47/138) | 0.98 (179/183) | 1.00 (55/55) | 0.55 (37/67) | 1.00 (144/144) | [7/7] |
| 130 to 140 | 0.63 (91/145) | 0.99 (255/257) | 1.00 (58/58) | 0.88 (92/104) | 1.00 (243/243) | [16/17] |

LAFB by frontal axis (degrees):

| bin | PTB-XL | Ningbo | Georgia |
|---|---|---|---|
| -75 to -60 | 0.65 (212/325) | 0.16 (72/445) | 0.25 (34/138) |
| -60 to -45 | 0.62 (433/698) | 0.21 (110/523) | 0.25 (58/234) |
| -45 to -30 | 0.48 (533/1118) | 0.06 (38/629) | 0.09 (33/380) |
| -30 to -15 | 0.07 (116/1742) | 0.01 (6/1050) | 0.00 (1/658) |

Left axis deviation by frontal axis (degrees):

| bin | PTB-XL | Ningbo | Chapman | Georgia |
|---|---|---|---|---|
| -75 to -60 | 0.99 (153/154) | 0.31 (115/376) | 0.41 (36/88) | 0.70 (73/104) |
| -60 to -45 | 0.99 (370/373) | 0.50 (209/415) | 0.47 (59/125) | 0.76 (135/177) |
| -45 to -30 | 1.00 (430/432) | 0.51 (299/591) | 0.49 (98/198) | 0.76 (262/347) |
| -30 to -15 | 0.88 (148/169) | 0.11 (120/1044) | 0.15 (51/338) | 0.26 (170/657) |

First-degree AV block by PR (ms):

| bin | PTB-XL | Ningbo | Chapman | Georgia | CPSC 2018 | CPSC-Extra |
|---|---|---|---|---|---|---|
| 180 to 200 | 0.02 (28/1557) | 0.01 (12/1121) | 0.00 (0/317) | 0.02 (9/419) | 0.06 (7/118) | 0.01 (1/116) |
| 200 to 220 | 0.19 (108/566) | 0.24 (95/394) | 0.22 (29/129) | 0.37 (70/189) | 0.71 (59/83) | 0.27 (9/33) |
| 220 to 240 | 0.56 (115/205) | 0.79 (102/129) | 0.81 (38/47) | 0.83 (62/75) | 0.94 (47/50) | [5/6] |

Reading. Chapman and CPSC 2018 have a single RBBB code, so their "complete" label contains every incomplete RBBB.
In CPSC 2018, 100% of RBBB-labelled records with QRS 90-100 ms are labelled complete (138/138); in PTB-XL the
figure is 1%.

## Item 3: contradictions inside one record [item3; R11 rows of rules_summary]

| first | second | reason | PTB-XL | Ningbo | Chapman | Georgia | CPSC 2018 | CPSC-Extra | total |
|---|---|---|---|---|---|---|---|---|---|
| b_sinus_tachycardia | b_sinus_bradycardia | the rate cannot be above 100 and below 60 per minute | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| b_sinus_tachycardia | b_af_or_flutter | sinus rhythm and atrial fibrillation or flutter exclude each other | 5 | 5 | 0 | 13 | 0 | 2 | 25 |
| b_sinus_tachycardia | c_supraventricular_tachycardia | sinus and supraventricular tachycardia exclude each other | 1 | 84 | 0 | 6 | 0 | 3 | 94 |
| b_sinus_bradycardia | b_af_or_flutter | sinus rhythm and atrial fibrillation or flutter exclude each other | 0 | 1 | 0 | 21 | 0 | 0 | 22 |
| b_sinus_arrhythmia | b_af_or_flutter | sinus rhythm and atrial fibrillation or flutter exclude each other | 0 | 2 | 0 | 3 | 0 | 0 | 5 |
| b_first_degree_av_block | b_af_or_flutter | a PR interval needs P waves, which AF and flutter lack | 10 | 1 | 0 | 15 | 0 | 1 | 27 |
| b_first_degree_av_block | c_preexcitation | pre-excitation shortens the PR interval | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| b_first_degree_av_block | flat lead among II | the condition is read on these leads | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| b_complete_rbbb | b_incomplete_rbbb | QRS of at least and below 120 ms (MLC-01 makes both undefined) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| b_complete_rbbb | c_complete_lbbb | MLC-01 contradictory pair (both made undefined) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| b_complete_rbbb | flat lead among V1 | the condition is read on these leads | 0 | 3 | 0 | 0 | 0 | 0 | 3 |
| b_incomplete_rbbb | c_complete_lbbb | MLC-01 contradictory pair (both made undefined) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| b_incomplete_rbbb | flat lead among V1 | the condition is read on these leads | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| b_lafb | c_lpfb | the two fascicular blocks need opposite frontal axes | 2 | 0 | 0 | 0 | 0 | 0 | 2 |
| b_lafb | c_right_axis_deviation | LAFB needs a frontal axis of -45 degrees or less | 16 | 1 | 0 | 0 | 0 | 0 | 17 |
| b_lafb | b_left_axis_deviation = 0 | LAFB needs an axis of -45 degrees or less, a left axis | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| b_lafb | flat lead among I/aVF | the condition is read on these leads | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| b_left_axis_deviation | c_right_axis_deviation | one frontal axis cannot deviate both ways | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| b_left_axis_deviation | flat lead among I/aVF | the condition is read on these leads | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| c_paced_rhythm | b_incomplete_rbbb | a ventricular-paced QRS replaces intrinsic conduction | 2 | 6 | 0 | 7 | 0 | 0 | 15 |
| c_paced_rhythm | b_complete_rbbb | a ventricular-paced QRS replaces intrinsic conduction | 0 | 32 | 0 | 5 | 0 | 0 | 37 |
| c_paced_rhythm | b_lafb | a ventricular-paced QRS replaces intrinsic conduction | 0 | 13 | 0 | 3 | 0 | 0 | 16 |
| c_paced_rhythm | b_left_axis_deviation | a ventricular-paced QRS replaces intrinsic conduction | 3 | 25 | 0 | 11 | 0 | 0 | 39 |
| c_paced_rhythm | b_lvh | a ventricular-paced QRS replaces intrinsic conduction | 1 | 23 | 0 | 6 | 0 | 0 | 30 |
| c_paced_rhythm | b_st_depression | a ventricular-paced QRS replaces intrinsic conduction | 2 | 51 | 0 | 0 | 0 | 0 | 53 |
| c_paced_rhythm | b_t_wave_abnormality | a ventricular-paced QRS replaces intrinsic conduction | 1 | 238 | 0 | 19 | 0 | 0 | 258 |
| c_paced_rhythm | b_mi_any | a ventricular-paced QRS replaces intrinsic conduction | 4 | 4 | 0 | 0 | 0 | 0 | 8 |
| b_lvh | flat lead among V1/V5/V6 | the condition is read on these leads | 0 | 2 | 0 | 0 | 0 | 1 | 3 |

Labels made unreadable by Part A signal flags (rule R11). The flags come from
`signal/flags_train_validation.parquet` (records with `included == True`; its SHA-256 is in the `rules` receipt).
Only flags the Part A report found reliable are used (`docs/mlc03-signal-quality-results-2026-11-11.md`, branch
`research/mlc03-signal-quality-20261111`, commit `e6869ff`). A label counts when:

- a lead the condition is read on is flat, near-flat or loses contact (for the two axis conditions: I, II, III or
  aVF);
- `reversal_la_ra` or `reversal_neutral` is flagged, for the axis conditions (these passed Part A's check, but that
  check reused the first viewing);
- `amplitude_outlier` is flagged, for LVH.

`reversal_ra_ll` (1 of 12 flagged plots showed the defect) and `rhythm_unmeasurable` (3 of 12) are not used. The
counts are the R11 rows of the rules table below; the largest are 849 complete-RBBB labels and 1,336 LVH labels,
almost all negatives. No Part A flag detects swapped precordial leads.

## Item 4: probable label errors from out-of-fold probabilities [integrity, item4, review]

P0 (the MLC-02 reference model) was refitted by five-fold cross-validation over train plus validation, with folds
grouped by patient for PTB-XL and by record for the Challenge sources, balanced by source, seed 20261111.
Integrity checks before the folds:

- the stored P0 validation selection metric is reproduced exactly (0.9642697112976489, difference 0.0);
- a refit of P0 on its own training data reproduces the stored validation probabilities to within 4.6e-15.

Confident-learning estimates (Northcutt et al. 2021: per-class thresholds at the mean self-confidence, calibrated
confident joint, within each source), as estimated false positives / estimated false negatives with their share
of the defined labels. These are model estimates. On a synthetic set with 400 planted false positives and 200
planted false negatives the estimator stays within 30% of each count and 15% of the total (unit test
`test_confident_learning_recovers_planted_label_noise`); it is not checked on real labels.

| condition | PTB-XL | Ningbo | Chapman | Georgia | CPSC-Extra | pooled | CPSC 2018 |
|---|---|---|---|---|---|---|---|
| sinus_tachycardia | 19 / 48 (0.3%) | 22 / 35 (0.2%) | 4 / 4 (0.1%) | 20 / 27 (0.6%) | 11 / 31 (1.6%) | 71 / 136 (0.3%) |  |
| sinus_bradycardia | 63 / 399 (2.4%) | 21 / 68 (0.3%) | 0 / 4 (0.1%) | 7 / 48 (0.7%) | 2 / 59 (2.5%) | 73 / 202 (0.4%) |  |
| sinus_arrhythmia | 189 / 365 (2.8%) | 56 / 153 (0.7%) |  | 31 / 105 (1.7%) | 2 / 55 (2.2%) | 278 / 516 (1.4%) |  |
| af_or_flutter | 18 / 65 (0.4%) | 8 / 16 (0.1%) | 2 / 12 (0.2%) | 80 / 33 (1.4%) | 3 / 12 (0.6%) | 126 / 148 (0.4%) | 16 / 26 (0.8%) |
| premature_ventricular_complex | 12 / 65 (0.4%) | 10 / 49 (0.2%) | 8 / 19 (0.3%) | 100 / 57 (1.9%) | 45 / 46 (3.5%) | 266 / 259 (0.7%) | 100 / 48 (2.9%) |
| first_degree_av_block | 62 / 210 (1.4%) | 24 / 98 (0.4%) | 9 / 27 (0.4%) | 29 / 74 (1.3%) | 13 / 26 (1.5%) | 155 / 449 (0.8%) | 31 / 31 (1.2%) |
| complete_rbbb | 1 / 79 (0.4%) | 8 / 14 (0.1%) | 6 / 5 (0.1%) | 6 / 45 (0.6%) | 3 / 14 (0.6%) | 78 / 206 (0.4%) | 74 / 33 (2.1%) |
| incomplete_rbbb | 164 / 276 (2.2%) | 33 / 233 (1.0%) |  | 48 / 206 (3.1%) | 10 / 36 (1.8%) | 219 / 639 (1.5%) |  |
| lafb | 37 / 156 (1.0%) | 12 / 122 (0.5%) |  | 19 / 160 (2.2%) |  | 63 / 366 (0.8%) |  |
| left_axis_deviation | 1 / 9 (0.1%) | 9 / 190 (0.7%) | 5 / 112 (1.4%) | 15 / 112 (1.6%) |  | 20 / 322 (0.7%) |  |
| lvh | 175 / 300 (2.5%) | 5 / 31 (0.1%) | 2 / 130 (1.8%) | 169 / 224 (4.8%) | 15 / 72 (3.4%) | 260 / 792 (1.7%) |  |
| st_depression | 129 / 278 (2.4%) | 68 / 159 (0.9%) | 26 / 88 (1.5%) |  | 20 / 60 (8.7%) | 324 / 788 (2.0%) | 123 / 217 (6.8%) |
| t_wave_abnormality | 367 / 721 (6.5%) | 273 / 239 (2.1%) | 99 / 171 (3.6%) | 347 / 371 (10.9%) | 5 / 159 (18.2%) | 1060 / 1491 (4.5%) |  |
| mi_any | 649 / 534 (6.1%) | 13 / 98 (0.4%) | 0 / 57 (0.7%) |  | 262 / 95 (13.8%) | 523 / 777 (2.3%) |  |

Suspect lists and review sheets. For every condition, source and kind, the 50 most suspected labels (the lowest
out-of-fold probability among positives, the highest among negatives) are listed in
`labels/suspects/<condition>.csv` with their measurements, and all 6,614 are plotted under
`review/labels/<condition>/<source>/`. Of these, 336 were read by eye: per condition and kind, rank 1 of every
source, then rank 2, up to 12 sheets (`review/labels/reading_list.csv`). Each judgement and its measurable reason
are in `review/labels/review_judgements.csv` (SHA-256
`175574ea4a9649723a780d02106eedce25d54888ffcf09f610b1e5bc553149f5`):

- label error: the measurement contradicts the label beyond the tolerance (by both methods where both measured),
  and the trace gives no reason to distrust the measurement;
- model error: the measurement supports the label beyond the tolerance;
- cannot tell: the measurement is inside the tolerance band, or the methods disagree; or the trace is noisy or
  paced; or the finding may lie outside the first 10 s; or the check does not cover the code (for example a
  generic T-wave code with tall T waves, or voltage alone for LVH, which this project leaves undefined).

| condition | kind | label error | model error | cannot tell | sheets | label error share |
|---|---|---|---|---|---|---|
| af_or_flutter | false_negative | 3 | 2 | 7 | 12 | 25.0% |
| af_or_flutter | false_positive | 6 | 2 | 4 | 12 | 50.0% |
| complete_rbbb | false_negative | 8 | 1 | 3 | 12 | 66.7% |
| complete_rbbb | false_positive | 9 | 0 | 3 | 12 | 75.0% |
| first_degree_av_block | false_negative | 7 | 3 | 2 | 12 | 58.3% |
| first_degree_av_block | false_positive | 4 | 0 | 8 | 12 | 33.3% |
| incomplete_rbbb | false_negative | 0 | 3 | 9 | 12 | 0.0% |
| incomplete_rbbb | false_positive | 8 | 0 | 4 | 12 | 66.7% |
| lafb | false_negative | 4 | 1 | 7 | 12 | 33.3% |
| lafb | false_positive | 9 | 0 | 3 | 12 | 75.0% |
| left_axis_deviation | false_negative | 8 | 2 | 2 | 12 | 66.7% |
| left_axis_deviation | false_positive | 4 | 1 | 7 | 12 | 33.3% |
| lvh | false_negative | 0 | 0 | 12 | 12 | 0.0% |
| lvh | false_positive | 6 | 1 | 5 | 12 | 50.0% |
| mi_any | false_negative | 4 | 3 | 5 | 12 | 33.3% |
| mi_any | false_positive | 3 | 0 | 9 | 12 | 25.0% |
| premature_ventricular_complex | false_negative | 1 | 1 | 10 | 12 | 8.3% |
| premature_ventricular_complex | false_positive | 6 | 2 | 4 | 12 | 50.0% |
| sinus_arrhythmia | false_negative | 5 | 5 | 2 | 12 | 41.7% |
| sinus_arrhythmia | false_positive | 4 | 6 | 2 | 12 | 33.3% |
| sinus_bradycardia | false_negative | 6 | 1 | 5 | 12 | 50.0% |
| sinus_bradycardia | false_positive | 6 | 1 | 5 | 12 | 50.0% |
| sinus_tachycardia | false_negative | 6 | 5 | 1 | 12 | 50.0% |
| sinus_tachycardia | false_positive | 11 | 0 | 1 | 12 | 91.7% |
| st_depression | false_negative | 7 | 0 | 5 | 12 | 58.3% |
| st_depression | false_positive | 2 | 2 | 8 | 12 | 16.7% |
| t_wave_abnormality | false_negative | 7 | 1 | 4 | 12 | 58.3% |
| t_wave_abnormality | false_positive | 5 | 0 | 7 | 12 | 41.7% |

| kind | label error | model error | cannot tell | sheets |
|---|---|---|---|---|
| false_negative | 66 | 28 | 74 | 168 |
| false_positive | 83 | 15 | 70 | 168 |

| source | kind | label error | model error | cannot tell |
|---|---|---|---|---|
| Chapman | false_negative | 13 | 4 | 7 |
| Chapman | false_positive | 8 | 1 | 15 |
| CPSC 2018 | false_negative | 3 | 0 | 7 |
| CPSC 2018 | false_positive | 5 | 0 | 5 |
| CPSC-Extra | false_negative | 9 | 5 | 13 |
| CPSC-Extra | false_positive | 12 | 3 | 12 |
| Georgia | false_negative | 13 | 5 | 11 |
| Georgia | false_positive | 20 | 2 | 7 |
| Ningbo | false_negative | 9 | 9 | 21 |
| Ningbo | false_positive | 14 | 6 | 19 |
| PTB-XL | false_negative | 19 | 5 | 15 |
| PTB-XL | false_positive | 24 | 3 | 12 |

Rate and rhythm sheets (sinus tachycardia, bradycardia, arrhythmia, AF or flutter), split by whether the
annotation states any rhythm:

| kind | rhythm_stated | label error | model error | cannot tell |
|---|---|---|---|---|
| false_negative | False | 14 | 3 | 10 |
| false_negative | True | 6 | 10 | 5 |
| false_positive | True | 27 | 9 | 12 |

Reliable Part A flags among the reviewed records (`any_reliable_flag` is the union of the 15 flags Part A found
reliable):

| group | records | any_reliable_flag | baseline_wander | baseline_step | isolated_spike | flat_lead | contact_loss |
|---|---|---|---|---|---|---|---|
| all open records | 71388 | 8.2% | 2.9% | 3.2% | 1.3% | 1.7% | 0.5% |
| listed suspects | 6614 | 9.3% | 2.9% | 4.1% | 1.4% | 1.3% | 0.7% |
| listed suspects with a confident-learning flag | 4025 | 8.5% | 2.8% | 4.1% | 1.5% | 1.1% | 0.8% |
| reviewed: label error | 149 | 8.7% | 3.4% | 4.0% | 2.7% | 0.7% | 0.7% |
| reviewed: model error | 43 | 9.3% | 0.0% | 4.7% | 4.7% | 0.0% | 0.0% |
| reviewed: cannot tell | 144 | 12.5% | 5.6% | 7.6% | 4.2% | 1.4% | 2.1% |

What the review shows, from the recorded reasons:

- Rate and rhythm:
  - Sinus tachycardia false positives: all 11 label errors have a measured rate of 54-97 per minute; in one,
    method A counts T waves (136 per minute) while the QRS complexes come every 0.9 s.
  - Sinus tachycardia and bradycardia false negatives: the label errors are mostly regular rates beyond the
    cut-off, with P waves, in records whose annotation has no rhythm code (PTB-XL, Georgia, CPSC-Extra).
  - AF or flutter false positives: the 6 label errors have a regular RR with a P wave before each QRS; several are
    also coded with sinus rhythm or sinus bradycardia.
- Ectopy and PR:
  - PVC false positives: the 6 label errors show all beats alike in the 10 s.
  - PVC false negatives: 10 of 12 cannot be told; wide premature beats are often coded as atrial premature
    complexes, and the measurement cannot tell where they arise.
  - First-degree AV block: the false-positive label errors have PR 140-176 ms; the false-negative ones 226-312 ms.
- Conduction:
  - Complete RBBB false positives: all 9 label errors have a QRS of 98 ms or less by both methods; they come from
    every source except PTB-XL, whose two sheets sit at the 120 ms boundary.
  - Complete RBBB false negatives: the 8 label errors are records coded incomplete RBBB with QRS 128-152 ms,
    records with no RBBB code, and two PTB-XL records coded complete LBBB whose V1 ends positive, with a terminal
    S in I and V6.
  - Incomplete RBBB false positives: the 8 label errors have a QRS below 100 ms, or of 128 ms or more.
  - Incomplete RBBB false negatives: none of the 12 is a label error, because every one lies within 10 ms of a
    cut-off.
- Axis:
  - LAFB false positives: the 9 label errors have an axis of -27 to +47 degrees.
  - Left axis deviation false negatives: all 8 label errors have an axis of -45 to -76 degrees and no axis code of
    any kind.
- LVH:
  - False positives: the 6 label errors have a Sokolow-Lyon index of 2.75 mV or less and a Cornell product below
    the sex threshold.
  - False negatives: all 12 meet a voltage criterion without an LVH code; this project leaves voltage alone
    undefined (rule R10).
- ST, T and MI:
  - ST depression false negatives: the 7 label errors have ST depression of 0.10-0.56 mV in two or more leads of
    one group, in records coded only with other findings.
  - T-wave abnormality false negatives: the 7 label errors have inverted T waves in two or more checked leads.
  - T-wave abnormality false positives: two of the 5 label errors come from PTB-XL statements with likelihood 0
    next to NORM.
  - MI false positives: the 3 label errors are CPSC-Extra "old MI" records with no Q group, normal R progression
    and no ST-T change; one is also coded bradycardia at 104 per minute.

## Item 5: undefined entries and whether their rules hold [item5]

For each undefined entry, "meeting" is the share of its records that meet the condition's measured definition;
"positives" and "negatives" give the same share among the defined positives and negatives of the same source.

| condition | source | reason | records | in_population | meeting | positives | negatives |
|---|---|---|---|---|---|---|---|
| sinus_tachycardia | CPSC 2018 | not annotated: outside CPSC 2018 classes | 5014 | 3528 | 7.8% |  |  |
| sinus_bradycardia | Ningbo | ambiguous: bradycardia of unstated origin | 5 | 1 | 100.0% | 99.0% | 1.5% |
| sinus_bradycardia | Georgia | ambiguous: bradycardia of unstated origin | 4 | 1 | 100.0% | 99.9% | 2.0% |
| sinus_bradycardia | CPSC 2018 | not annotated: outside CPSC 2018 classes | 5014 | 3528 | 9.9% |  |  |
| sinus_bradycardia | CPSC-Extra | ambiguous: bradycardia of unstated origin | 199 | 158 | 94.3% | 88.9% | 4.0% |
| sinus_arrhythmia | Chapman | not annotated: source list | 8133 | 5105 | 11.6% |  |  |
| sinus_arrhythmia | CPSC 2018 | not annotated: outside CPSC 2018 classes | 5014 | 3528 | 18.2% |  |  |
| premature_ventricular_complex | PTB-XL | ambiguous: premature complex of unstated origin | 53 | 52 | 28.8% | 85.2% | 2.8% |
| premature_ventricular_complex | Ningbo | ambiguous: premature complex of unstated origin | 79 | 67 | 17.9% | 70.4% | 5.0% |
| premature_ventricular_complex | Ningbo | ambiguous: premature complex of unstated origin + ventricular tachycardia without a PVC code | 5 | 5 | 20.0% | 70.4% | 5.0% |
| premature_ventricular_complex | Ningbo | ambiguous: ventricular tachycardia without a PVC code | 40 | 38 | 21.1% | 70.4% | 5.0% |
| premature_ventricular_complex | CPSC-Extra | ambiguous: ventricular tachycardia without a PVC code | 1 | 0 |  | 60.4% | 6.4% |
| first_degree_av_block | Ningbo | ambiguous: AV block of unstated degree | 62 | 19 | 0.0% | 41.2% | 1.8% |
| first_degree_av_block | Chapman | ambiguous: AV block of unstated degree | 140 | 54 | 1.9% | 44.4% | 2.1% |
| first_degree_av_block | Georgia | ambiguous: AV block of unstated degree | 60 | 21 | 9.5% | 34.6% | 2.2% |
| first_degree_av_block | CPSC-Extra | ambiguous: AV block of unstated degree | 3 | 3 | 33.3% | 30.2% | 1.2% |
| complete_rbbb | PTB-XL | contradictory pair: incomplete_rbbb | 1 | 1 | 100.0% | 88.4% | 6.7% |
| complete_rbbb | Ningbo | ambiguous: bundle branch block of unstated side | 299 | 294 | 46.6% | 78.8% | 2.1% |
| complete_rbbb | Chapman | contradictory pair: complete_lbbb | 31 | 31 | 77.4% | 63.1% | 1.5% |
| complete_rbbb | Georgia | ambiguous: bundle branch block of unstated side | 76 | 75 | 84.0% | 82.4% | 5.4% |
| complete_rbbb | Georgia | contradictory pair: incomplete_rbbb | 41 | 41 | 19.5% | 82.4% | 5.4% |
| complete_rbbb | CPSC-Extra | contradictory pair: complete_lbbb | 1 | 1 | 100.0% | 86.3% | 2.9% |
| complete_rbbb | CPSC-Extra | contradictory pair: incomplete_rbbb | 34 | 34 | 73.5% | 86.3% | 2.9% |
| incomplete_rbbb | PTB-XL | contradictory pair: complete_rbbb | 1 | 1 | 0.0% | 10.3% | 1.3% |
| incomplete_rbbb | Chapman | not annotated: source list | 8133 | 8127 | 1.0% |  |  |
| incomplete_rbbb | Georgia | contradictory pair: complete_rbbb | 41 | 41 | 14.6% | 18.7% | 1.2% |
| incomplete_rbbb | CPSC 2018 | not annotated: outside CPSC 2018 classes | 5014 | 4989 | 3.2% |  |  |
| incomplete_rbbb | CPSC-Extra | contradictory pair: complete_lbbb | 1 | 1 | 0.0% | 11.8% | 1.3% |
| incomplete_rbbb | CPSC-Extra | contradictory pair: complete_rbbb | 34 | 34 | 8.8% | 11.8% | 1.3% |
| lafb | Chapman | not annotated: source list | 8133 | 8127 | 3.2% |  |  |
| lafb | CPSC 2018 | not annotated: outside CPSC 2018 classes | 5014 | 4989 | 2.6% |  |  |
| lafb | CPSC-Extra | minimum positives | 2589 | 2568 | 5.2% |  |  |
| left_axis_deviation | PTB-XL | ambiguous: LAFB without a left-axis code | 21 | 21 | 33.3% | 81.6% | 0.1% |
| left_axis_deviation | PTB-XL | ambiguous: LAFB without a left-axis code + PTB-XL heart_axis LAD | 577 | 576 | 85.9% | 81.6% | 0.1% |
| left_axis_deviation | PTB-XL | ambiguous: LAFB without a left-axis code + PTB-XL heart_axis empty or AXL | 180 | 179 | 72.1% | 81.6% | 0.1% |
| left_axis_deviation | PTB-XL | ambiguous: PTB-XL heart_axis LAD | 2793 | 2789 | 10.8% | 81.6% | 0.1% |
| left_axis_deviation | PTB-XL | ambiguous: PTB-XL heart_axis empty or AXL | 7423 | 7388 | 4.3% | 81.6% | 0.1% |
| left_axis_deviation | Ningbo | ambiguous: LAFB without a left-axis code | 297 | 292 | 84.6% | 75.1% | 3.3% |
| left_axis_deviation | Georgia | ambiguous: LAFB without a left-axis code | 146 | 146 | 89.0% | 66.6% | 2.5% |
| left_axis_deviation | CPSC 2018 | not annotated: outside CPSC 2018 classes | 5014 | 4989 | 5.0% |  |  |
| left_axis_deviation | CPSC-Extra | minimum positives | 2589 | 2568 | 9.9% |  |  |
| lvh | PTB-XL | ambiguous: left ventricular high voltage | 242 | 241 | 36.5% | 53.6% | 6.6% |
| lvh | Ningbo | ambiguous: left ventricular high voltage | 3268 | 3255 | 57.4% | 82.7% | 6.5% |
| lvh | Chapman | ambiguous: left ventricular high voltage | 991 | 991 | 55.5% | 63.6% | 7.0% |
| lvh | Georgia | ambiguous: ventricular hypertrophy of unstated side | 27 | 27 | 14.8% | 48.7% | 6.0% |
| lvh | CPSC 2018 | not annotated: outside CPSC 2018 classes | 5014 | 4989 | 9.7% |  |  |
| st_depression | PTB-XL | ambiguous: digitalis, aneurysm or electrolyte statement | 156 | 156 | 28.8% | 64.7% | 10.8% |
| st_depression | PTB-XL | ambiguous: digitalis, aneurysm or electrolyte statement + nonspecific or ischemic ST-T statement | 48 | 48 | 70.8% | 64.7% | 10.8% |
| st_depression | PTB-XL | ambiguous: digitalis, aneurysm or electrolyte statement + nonspecific or ischemic ST-T statement + subendocardial injury statement | 1 | 1 | 100.0% | 64.7% | 10.8% |
| st_depression | PTB-XL | ambiguous: digitalis, aneurysm or electrolyte statement + subendocardial injury statement | 7 | 7 | 57.1% | 64.7% | 10.8% |
| st_depression | PTB-XL | ambiguous: nonspecific or ischemic ST-T statement | 2262 | 2259 | 58.8% | 64.7% | 10.8% |
| st_depression | PTB-XL | ambiguous: nonspecific or ischemic ST-T statement + subendocardial injury statement | 60 | 60 | 63.3% | 64.7% | 10.8% |
| st_depression | PTB-XL | ambiguous: subendocardial injury statement | 183 | 183 | 49.2% | 64.7% | 10.8% |
| st_depression | Ningbo | ambiguous: nonspecific or ischemic ST-T statement | 3125 | 3109 | 46.1% | 78.1% | 6.8% |
| st_depression | Chapman | ambiguous: nonspecific or ischemic ST-T statement | 663 | 663 | 51.7% | 86.2% | 10.2% |
| st_depression | Georgia | not annotated: source list | 8160 | 8132 | 13.8% |  |  |
| st_depression | CPSC-Extra | ambiguous: nonspecific or ischemic ST-T statement | 1674 | 1666 | 32.1% | 48.8% | 14.1% |
| t_wave_abnormality | PTB-XL | ambiguous: digitalis, aneurysm or electrolyte statement | 119 | 119 | 58.0% | 44.7% | 14.6% |
| t_wave_abnormality | PTB-XL | ambiguous: digitalis, aneurysm or electrolyte statement + nonspecific or ischemic ST-T statement | 83 | 83 | 56.6% | 44.7% | 14.6% |
| t_wave_abnormality | PTB-XL | ambiguous: digitalis, aneurysm or electrolyte statement + nonspecific or ischemic ST-T statement + subendocardial injury statement | 1 | 1 | 100.0% | 44.7% | 14.6% |
| t_wave_abnormality | PTB-XL | ambiguous: digitalis, aneurysm or electrolyte statement + subendocardial injury statement | 6 | 6 | 100.0% | 44.7% | 14.6% |
| t_wave_abnormality | PTB-XL | ambiguous: nonspecific or ischemic ST-T statement | 2370 | 2367 | 72.8% | 44.7% | 14.6% |
| t_wave_abnormality | PTB-XL | ambiguous: nonspecific or ischemic ST-T statement + subendocardial injury statement | 62 | 62 | 93.5% | 44.7% | 14.6% |
| t_wave_abnormality | PTB-XL | ambiguous: subendocardial injury statement | 168 | 168 | 94.0% | 44.7% | 14.6% |
| t_wave_abnormality | Ningbo | ambiguous: nonspecific or ischemic ST-T statement | 2951 | 2936 | 51.8% | 68.5% | 12.9% |
| t_wave_abnormality | Chapman | ambiguous: nonspecific or ischemic ST-T statement | 683 | 683 | 57.0% | 67.1% | 11.1% |
| t_wave_abnormality | Georgia | ambiguous: nonspecific or ischemic ST-T statement | 1577 | 1574 | 43.6% | 75.0% | 17.4% |
| t_wave_abnormality | CPSC 2018 | not annotated: outside CPSC 2018 classes | 5014 | 4989 | 23.0% |  |  |
| t_wave_abnormality | CPSC-Extra | ambiguous: nonspecific or ischemic ST-T statement | 1683 | 1674 | 54.4% | 80.0% | 33.8% |
| mi_any | PTB-XL | ambiguous: Q wave or poor R-wave progression | 96 | 96 | 10.4% | 30.8% | 1.8% |
| mi_any | PTB-XL | ambiguous: Q wave or poor R-wave progression + subendocardial injury statement | 3 | 3 | 0.0% | 30.8% | 1.8% |
| mi_any | PTB-XL | ambiguous: subendocardial injury statement | 160 | 160 | 4.4% | 30.8% | 1.8% |
| mi_any | Ningbo | ambiguous: Q wave or poor R-wave progression | 1093 | 1087 | 33.3% | 62.3% | 6.0% |
| mi_any | Chapman | ambiguous: Q wave or poor R-wave progression | 176 | 176 | 46.6% | 58.1% | 1.5% |
| mi_any | Georgia | minimum positives | 8160 | 8132 | 5.7% |  |  |
| mi_any | CPSC 2018 | not annotated: outside CPSC 2018 classes | 5014 | 4989 | 2.1% |  |  |

Reading. Every undefined entry has a reason. Most "ambiguous" entries meet the definition at a share between that
of the source's positives and its negatives, which is what an ambiguous code should show. Three exceptions:

- Left axis deviation made undefined because LAFB is coded without a left-axis code: these records meet the
  left-axis definition as often as the positives do (84.6% in Ningbo, 89.0% in Georgia).
- PTB-XL's `heart_axis` value "LAD" without an LAD statement: only 10.8% of these records reach -30 degrees.
- AV block of unstated degree: only 0-33% have a PR above 200 ms.

## Candidate cleaning rules [rules_summary; rules for source and role]

Every rule sets labels undefined. None creates or flips a positive, and none changes a label definition. Counts
are over train and validation: the positives and negatives that would become undefined, each with its share of
the defined positives or negatives in the sources the rule touches.

| rule | condition | sources | positives to undefined | negatives to undefined |
|---|---|---|---|---|
| R1 finding only after the first 10 s | premature_ventricular_complex | CPSC 2018, CPSC-Extra | 55 (9.1%) | 0 (0.0%) |
| R1 finding only after the first 10 s | af_or_flutter | CPSC 2018, CPSC-Extra | 21 (2.2%) | 0 (0.0%) |
| R2 complete RBBB measurably narrow where completeness is not coded | complete_rbbb | Chapman, CPSC 2018 | 429 (26.7%) | 0 (0.0%) |
| R2b complete RBBB where completeness is not coded (whole source) | complete_rbbb | Chapman, CPSC 2018 | 1609 (100.0%) | 11507 (100.0%) |
| R3 sinus bradycardia negative meeting 60/100 per minute in a source whose cut-off differs | sinus_bradycardia | PTB-XL, CPSC-Extra | 0 (0.0%) | 2595 (12.1%) |
| R4 contradictory pair sinus_tachycardia + sinus_bradycardia | sinus_tachycardia | CPSC-Extra | 1 (0.6%) | 0 (0.0%) |
| R4 contradictory pair sinus_tachycardia + sinus_bradycardia | sinus_bradycardia | CPSC-Extra | 1 (3.4%) | 0 (0.0%) |
| R4 contradictory pair sinus_tachycardia + af_or_flutter | sinus_tachycardia | PTB-XL, CPSC-Extra, Georgia, Ningbo | 25 (0.5%) | 0 (0.0%) |
| R4 contradictory pair sinus_tachycardia + af_or_flutter | af_or_flutter | PTB-XL, CPSC-Extra, Georgia, Ningbo | 25 (0.4%) | 0 (0.0%) |
| R4 contradictory pair sinus_bradycardia + af_or_flutter | sinus_bradycardia | Georgia, Ningbo | 22 (0.2%) | 0 (0.0%) |
| R4 contradictory pair sinus_bradycardia + af_or_flutter | af_or_flutter | Georgia, Ningbo | 22 (0.4%) | 0 (0.0%) |
| R4 contradictory pair sinus_arrhythmia + af_or_flutter | sinus_arrhythmia | Georgia, Ningbo | 5 (0.3%) | 0 (0.0%) |
| R4 contradictory pair sinus_arrhythmia + af_or_flutter | af_or_flutter | Georgia, Ningbo | 5 (0.1%) | 0 (0.0%) |
| R4 contradictory pair sinus_tachycardia + supraventricular_tachycardia | sinus_tachycardia | PTB-XL, CPSC-Extra, Georgia, Ningbo | 94 (1.5%) | 0 (0.0%) |
| R4 contradictory pair first_degree_av_block + af_or_flutter | first_degree_av_block | PTB-XL, CPSC-Extra, Georgia, Ningbo | 27 (1.3%) | 0 (0.0%) |
| R4 contradictory pair first_degree_av_block + af_or_flutter | af_or_flutter | PTB-XL, CPSC-Extra, Georgia, Ningbo | 27 (0.4%) | 0 (0.0%) |
| R4 contradictory pair lafb + lpfb | lafb | PTB-XL | 2 (0.2%) | 0 (0.0%) |
| R4 contradictory pair lafb + right_axis_deviation | lafb | PTB-XL, Ningbo | 17 (1.0%) | 0 (0.0%) |
| R9 sinus rate or rhythm label where no rhythm is stated | sinus_tachycardia | PTB-XL, Chapman, CPSC-Extra, Georgia, Ningbo | 0 (0.0%) | 6594 (11.3%) |
| R9 sinus rate or rhythm label where no rhythm is stated | sinus_bradycardia | PTB-XL, Chapman, CPSC-Extra, Georgia, Ningbo | 0 (0.0%) | 6594 (12.9%) |
| R9 sinus rate or rhythm label where no rhythm is stated | sinus_arrhythmia | PTB-XL, CPSC-Extra, Georgia, Ningbo | 0 (0.0%) | 6587 (11.9%) |
| R9b atrial fibrillation or flutter where no rhythm is stated | af_or_flutter | PTB-XL, Chapman, CPSC 2018, CPSC-Extra, Georgia, Ningbo | 0 (0.0%) | 10074 (16.6%) |
| R10 LVH negative meeting a voltage criterion (voltage alone stays undefined) | lvh | PTB-XL, Chapman, CPSC-Extra, Georgia, Ningbo | 0 (0.0%) | 3883 (6.7%) |
| R11 Part A: a lead the condition is read on is flat or loses contact | complete_rbbb | Chapman, CPSC 2018, CPSC-Extra, Georgia, Ningbo | 6 (0.2%) | 843 (1.8%) |
| R11 Part A: a lead the condition is read on is flat or loses contact | incomplete_rbbb | CPSC-Extra, Georgia, Ningbo | 0 (0.0%) | 845 (2.2%) |
| R11 Part A: a lead the condition is read on is flat or loses contact | lafb | PTB-XL, Georgia, Ningbo | 0 (0.0%) | 53 (0.1%) |
| R11 Part A: a lead the condition is read on is flat or loses contact | left_axis_deviation | Chapman, Georgia, Ningbo | 4 (0.2%) | 49 (0.1%) |
| R11 Part A: a lead the condition is read on is flat or loses contact | lvh | PTB-XL, Chapman, CPSC-Extra, Georgia, Ningbo | 4 (0.1%) | 1332 (2.4%) |
| R11 Part A: a lead the condition is read on is flat or loses contact | first_degree_av_block | PTB-XL, Chapman, CPSC 2018, CPSC-Extra, Georgia, Ningbo | 2 (0.1%) | 35 (0.1%) |
| R11 Part A: limb-electrode reversal pattern (LA-RA or neutral) | lafb | PTB-XL, Georgia, Ningbo | 0 (0.0%) | 15 (0.0%) |
| R11 Part A: limb-electrode reversal pattern (LA-RA or neutral) | left_axis_deviation | PTB-XL, Georgia, Ningbo | 2 (0.1%) | 13 (0.0%) |
| R11 Part A: amplitude outlier | lvh | PTB-XL, Chapman, CPSC-Extra, Georgia, Ningbo | 1 (0.0%) | 45 (0.1%) |
| R8 paced rhythm with a conduction or repolarization label | complete_rbbb | PTB-XL, Georgia, Ningbo | 37 (2.0%) | 1250 (2.3%) |
| R8 paced rhythm with a conduction or repolarization label | incomplete_rbbb | PTB-XL, Georgia, Ningbo | 15 (1.0%) | 1282 (2.4%) |
| R8 paced rhythm with a conduction or repolarization label | lafb | PTB-XL, Georgia, Ningbo | 16 (0.8%) | 1281 (2.4%) |
| R8 paced rhythm with a conduction or repolarization label | left_axis_deviation | PTB-XL, Georgia, Ningbo | 39 (1.3%) | 986 (2.4%) |
| R8 paced rhythm with a conduction or repolarization label | lvh | PTB-XL, Georgia, Ningbo | 30 (0.9%) | 1177 (2.4%) |
| R8 paced rhythm with a conduction or repolarization label | st_depression | PTB-XL, Ningbo | 53 (2.8%) | 1074 (2.7%) |
| R8 paced rhythm with a conduction or repolarization label | t_wave_abnormality | PTB-XL, Georgia, Ningbo | 258 (2.6%) | 945 (2.5%) |
| R8 paced rhythm with a conduction or repolarization label | mi_any | PTB-XL, Ningbo | 8 (0.2%) | 1177 (2.9%) |
| R5 a lead the condition is read on is flat | complete_rbbb | Chapman, CPSC 2018, CPSC-Extra, Ningbo | 3 (0.1%) | 803 (2.2%) |
| R5 a lead the condition is read on is flat | incomplete_rbbb | CPSC-Extra, Ningbo | 0 (0.0%) | 806 (2.8%) |
| R5 a lead the condition is read on is flat | lafb | Ningbo | 0 (0.0%) | 1 (0.0%) |
| R5 a lead the condition is read on is flat | left_axis_deviation | Ningbo | 0 (0.0%) | 1 (0.0%) |
| R5 a lead the condition is read on is flat | lvh | PTB-XL, Chapman, CPSC-Extra, Georgia, Ningbo | 3 (0.1%) | 1156 (2.1%) |
| R5 a lead the condition is read on is flat | first_degree_av_block | CPSC-Extra | 0 (0.0%) | 1 (0.2%) |
| R6 positive contradicted by the consensus measurement beyond tolerance | sinus_tachycardia | PTB-XL, Chapman, CPSC-Extra, Georgia, Ningbo | 48 (0.6%) | 0 (0.0%) |
| R6 negative meeting the measured definition beyond tolerance | sinus_tachycardia | PTB-XL, Chapman, CPSC-Extra, Georgia, Ningbo | 0 (0.0%) | 59 (0.1%) |
| R7 as R6 for positives, and flagged by confident learning | sinus_tachycardia | PTB-XL, Chapman, CPSC-Extra, Georgia, Ningbo | 20 (0.3%) | 0 (0.0%) |
| R7 as R6 for negatives, and flagged by confident learning | sinus_tachycardia | PTB-XL, CPSC-Extra, Georgia, Ningbo | 0 (0.0%) | 32 (0.1%) |
| R6 positive contradicted by the consensus measurement beyond tolerance | sinus_bradycardia | PTB-XL, Chapman, CPSC-Extra, Georgia, Ningbo | 106 (0.7%) | 0 (0.0%) |
| R6 negative meeting the measured definition beyond tolerance | sinus_bradycardia | PTB-XL, Chapman, CPSC-Extra, Georgia, Ningbo | 0 (0.0%) | 927 (1.9%) |
| R7 as R6 for positives, and flagged by confident learning | sinus_bradycardia | PTB-XL, CPSC-Extra, Georgia, Ningbo | 11 (0.1%) | 0 (0.0%) |
| R7 as R6 for negatives, and flagged by confident learning | sinus_bradycardia | PTB-XL, CPSC-Extra, Georgia, Ningbo | 0 (0.0%) | 226 (0.5%) |
| R6 positive contradicted by the consensus measurement beyond tolerance | first_degree_av_block | PTB-XL, CPSC 2018, CPSC-Extra, Georgia, Ningbo | 20 (0.8%) | 0 (0.0%) |
| R6 negative meeting the measured definition beyond tolerance | first_degree_av_block | PTB-XL, Chapman, CPSC 2018, CPSC-Extra, Georgia, Ningbo | 0 (0.0%) | 201 (0.3%) |
| R7 as R6 for positives, and flagged by confident learning | first_degree_av_block | PTB-XL, CPSC 2018, CPSC-Extra, Georgia, Ningbo | 15 (0.6%) | 0 (0.0%) |
| R7 as R6 for negatives, and flagged by confident learning | first_degree_av_block | PTB-XL, Chapman, CPSC 2018, CPSC-Extra, Georgia, Ningbo | 0 (0.0%) | 83 (0.1%) |
| R6 positive contradicted by the consensus measurement beyond tolerance | complete_rbbb | PTB-XL, Chapman, CPSC 2018, CPSC-Extra, Georgia, Ningbo | 505 (14.2%) | 0 (0.0%) |
| R6 negative meeting the measured definition beyond tolerance | complete_rbbb | PTB-XL, Chapman, CPSC 2018, CPSC-Extra, Georgia, Ningbo | 0 (0.0%) | 1860 (2.8%) |
| R7 as R6 for positives, and flagged by confident learning | complete_rbbb | Chapman, CPSC 2018, CPSC-Extra, Georgia, Ningbo | 60 (2.0%) | 0 (0.0%) |
| R7 as R6 for negatives, and flagged by confident learning | complete_rbbb | PTB-XL, Chapman, CPSC 2018, CPSC-Extra, Georgia, Ningbo | 0 (0.0%) | 99 (0.1%) |
| R6 positive contradicted by the consensus measurement beyond tolerance | lafb | PTB-XL, Georgia, Ningbo | 200 (10.5%) | 0 (0.0%) |
| R6 negative meeting the measured definition beyond tolerance | lafb | PTB-XL, Georgia, Ningbo | 0 (0.0%) | 590 (1.1%) |
| R7 as R6 for positives, and flagged by confident learning | lafb | PTB-XL, Georgia, Ningbo | 26 (1.4%) | 0 (0.0%) |
| R7 as R6 for negatives, and flagged by confident learning | lafb | PTB-XL, Georgia, Ningbo | 0 (0.0%) | 74 (0.1%) |
| R6 positive contradicted by the consensus measurement beyond tolerance | left_axis_deviation | PTB-XL, Chapman, Georgia, Ningbo | 175 (5.4%) | 0 (0.0%) |
| R6 negative meeting the measured definition beyond tolerance | left_axis_deviation | PTB-XL, Chapman, Georgia, Ningbo | 0 (0.0%) | 662 (1.4%) |
| R7 as R6 for positives, and flagged by confident learning | left_axis_deviation | Chapman, Georgia, Ningbo | 8 (0.5%) | 0 (0.0%) |
| R7 as R6 for negatives, and flagged by confident learning | left_axis_deviation | Chapman, Georgia, Ningbo | 0 (0.0%) | 164 (0.4%) |

Reading, for the Part C amendment:

- Rules that rest on definitions or reliable signal flags, not on a model or a delicate measurement: R4
  (contradictory pairs), R8 (paced rhythm), R5 and R11 (unreadable leads, using only Part A's reliable flags).
  They are small, apart from R8 and the lead-loss negatives.
- Rules that rest on source coding (item 2):
  - R3 removes PTB-XL's and CPSC-Extra's bradycardia negatives at 50-60 per minute, where the other sources code
    bradycardia.
  - R2 and R2b address the single RBBB code of Chapman and CPSC 2018: R2 removes only the 429 measurably narrow
    positives; R2b makes the label undefined in those two sources. The review supports this: all 9 complete-RBBB
    false-positive label errors have a QRS of 98 ms or less.
- R1 matches the first 10 s the models see. It leaves the 95 and 40 PVC positives with no premature beat in any
  window untouched.
- R10 extends the project's rule that left ventricular high voltage alone stays undefined, from the code to the
  measurement. All 12 reviewed LVH false negatives are of this kind.
- R9 and R9b rest on the review (label errors in 14 of 27 rhythm false negatives without a rhythm statement,
  against 6 of 21 with one). They would remove 11-17% of these labels' negatives, the largest data loss of any
  rule.
- R6 and R7 need a reliable measurement. Per the error section, this holds for rate, QRS away from the cut-offs,
  and axis; PR is reliable only where the consensus exists. R7, which also requires a confident-learning flag, is
  the conservative variant.
- No measured rule is proposed for LVH, ST, T or MI positives, because their checks cover only part of what those
  codes describe.

## Departures from the protocol

- Addendum 1: PTB-XL+ 1.0.1 features (folds 1-9 only, cut by `ecg_id` at load) and LUDB 1.0.1 were added as
  measurement references, with receipts. Synthetic records were also used. The consensus error that the protocol
  asks for is reported as well.
- Calibration: method B's QRS and P thresholds were calibrated on 1,500 PTB-XL training records before any label
  check, and validated on PTB-XL validation records outside that sample.
- One change to method B after its first run: it now counts premature aberrant beats at the record edges (commit
  24ba4e5). All 135 shared measurement columns were identical on all 71,388 records afterwards, and every step was
  rerun.
- Review scope: all 6,614 suspect sheets are plotted; reading by eye was limited to 336 sheets, chosen by a fixed
  rule before reading.
- Rules added after the review: R9, R10 and R11 came from patterns seen in the review. They are counted, not
  adopted.

## Limits

- Measurements come from automatic delineation, so the reliable and unreliable checks are as stated above.
- The bounds come from adult guidelines; a label outside a bound is not a wrong diagnosis.
- The reader is not a clinician and saw 12 sheets per condition and kind, chosen from the most suspected; the
  shares are not label-error rates of the sources.
- The confident-learning counts depend on P0's errors.
- No Part A flag detects swapped precordial leads.
- There is no second human reading. Clinical validation is pending.

## Follow-ups proposed for the backlog

1. Clinical review of the 149 label-error sheets and of the rule list.
2. Better P-wave detection and PR measurement in the five Challenge sources.
3. For Part A: five reviewed records with a negative lead I and an axis beyond +/-120 degrees carry no reversal
   flag (ningbo:JS21419, georgia:E02774, cpsc_2018_extra:Q3279, Q0072 and Q3123).
4. Checks for tall T waves, ST elevation, poor R progression and a tall R in V1-V2.
5. A window-based treatment of the CPSC PVC positives with no premature beat in any window.
6. Per-source rate cut-offs; this would change a label definition, so it needs clinical review.
7. Part D: the effect of the adopted rules on the models.

## How to rerun

From the worktree, with the `data`, `outputs` and `.venv` links to the main checkout:

```
PYTHONPATH=. OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -m \
    scripts.experiments.run_mlc03_label_quality <step> --workers 7
```

Run the steps in order: `calibrate`, `measure`, `validate`, `windows`, `labels`, `oof`, `suspects`, `sheets`,
`rules`, `tally`.

- `measure` resumes from `/var/tmp/janetrivera/mlc03/measure_shards`; empty that folder for a fresh run.
- `tally` needs the hand-made `review/labels/review_judgements.csv` and checks that it covers `reading_list.csv`
  exactly.
- The tables come from `PYTHONPATH=. .venv/bin/python -m scripts.reports.mlc03_label_quality_tables
  [--section NAME]`; the sections are `records`, `validation`, `agreement`, `definitions`, `item1`, `whole`,
  `item2`, `prevalence --name <analysis>`, `item3`, `item4`, `review`, `item5`, `rules_summary`, `rules` and
  `integrity`.
- The tests are in `tests/test_mlc03_label_quality.py` (30 tests).
- The last full run took 23.5 minutes.
