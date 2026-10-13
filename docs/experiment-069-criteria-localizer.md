# Experiment 069: measurement-based criteria localizer

Frozen 13 October 2026 (UTC), before any 069 map was built for a scored record and before any LocBench score
of a 069 map. Part of the [localization programme, part 2](localization-program-2-2026-10-13.md); scored on
LocBench v1 ([Experiment 061](experiment-061-locbench.md), with amendments
[2](experiment-061-amendment-2-qrs-strata.md) and [3](experiment-061-amendment-3-noise-standardized.md)).
The user is away and authorized unsupervised experiments; the design choices below were made by the agent
running this experiment and are frozen here.

Before this freeze, only the following were read or run:

- the programme notes, the 061 protocol and amendments, the results reports of 062, 065, 066, 067 and 068, and
  the code of `locbench`, `normative_map062`, `combination066`, `combiner068` and `run_combiner068`;
- the PTB-XL+ `feature_description.csv`, the column names of the three feature files, and pooled unit checks
  over each whole feature file (5th, 50th and 95th percentiles of about 20 columns, value counts of the Uni-G
  morphology codes). These pooled checks covered every PTB-XL row, including folds 9 and 10 and ECGAD
  records; they contain no label, mark or per-record value, but they are disclosed because the brief forbids
  reading folds 9 and 10 outside ECGAD. From here on the feature files are filtered to folds 1-8 and the ECGAD
  development records when loaded;
- feature coverage (missing-value shares) by partition, by record ID only;
- a prototype of the own per-lead measurements on 300 to 800 `fit`-partition records (clean normals and
  records with IMI, ALMI or ASMI statements), compared with 12SL and Uni-G: agreement statistics only. The
  prototype fixed the Q-wave detection threshold (0.02 mV, duration from the joint QRS onset), the variant
  closest to both commercial programs among four tried.

No ECGAD mark, ECGAD statement label, `evaluation`-partition label, or LocBench score of any 069 map has been
read.

## Question

Do per-lead measurements, turned into published ECG criteria, give patient-specific lead and wave evidence
(which leads actually meet a criterion in this ECG) that localizes better than painting the textbook region of
a diagnosis?

## Measurements

Three sources give per-lead and global measurements in mV and ms. A fourth, ECGdeli, is not used: its
`R_Amp_V2` has median -1.14 mV and its global PR median 272 ms, so its amplitude and interval definitions
are not comparable with the others. Its coverage is reported.

1. **own**: from the 500 Hz cache (`locbench_cache.open_cache`, exact float64 signal) with the frozen
   waveform-only delineation of 062 and 066. Beats and components come from
   `combination066.delineated_components` (P, PQ, QRS, ST, T per beat; the same function that built the 066
   and 068 units). The signal is preprocessed with `normative_map062.preprocess` (40 Hz low-pass, PR-segment
   baseline 30-6 ms before each QRS onset, interpolated between beats). The median beat per lead is the
   sample-wise median over every beat whose window from 300 ms before to 700 ms after QRS onset lies inside
   the record. Component boundaries relative to QRS onset are the medians over the same beats. P is "found"
   in a beat when `normative_map062.delineate` gives a finite P onset before QRS onset; P measurements need P
   found in at least half of the beats, otherwise they are missing. Per lead, on the median beat:
   - QRS window from the joint QRS onset to the median QRS offset. Q: if the first sample beyond 0.02 mV in
     absolute value is negative, Q duration runs from QRS onset to the first later sample at or above 0
     (the whole QRS for a QS complex) and Q depth is the most negative value before it; otherwise both are 0.
   - R: the largest positive value in the QRS window; R peak time from QRS onset to it. S: the depth of the
     most negative value after the R peak (0 if R is at most 0.02 mV). Peak-to-peak QRS amplitude. Terminal
     amplitude: mean of the last 40 ms of the QRS window. Net QRS area (for the frontal axis).
   - ST at the J point: mean over the first 10 ms after QRS offset. ST at J+60 and J+80 ms (5-sample means)
     are measured and reported for agreement with Uni-G's 80 ms value; no criterion uses them, because the
     Fourth Universal Definition of MI measures ST deviation at the J point.
   - T amplitude: the value of largest absolute size in the T window (signed). P amplitude likewise in the
     P window. P terminal force in V1: depth times duration (mV x ms) of the negative part after the P peak.
   - Global: QRS duration, PR (median over beats with P found), P duration (likewise), QT (QRS onset to T end),
     RR (median QRS-onset spacing), QTc by the Framingham formula QT + 154 (1 - RR / 1000), frontal QRS axis
     atan2(net area aVF, net area I).
2. **12SL** (`12sl_features.csv`): Q duration, Q depth = |Q_Amp|, R = max(R_Amp, Rp_Amp), S = max(|S_Amp|,
   |Sp_Amp|), ST_Amp (J point), T_Amp, P_Amp, R_PeakTime, QRS_AmpPP; globals QRS_Dur, PR_Int (missing when
   12SL found no P), QT_Int, RR_Mean, P duration P_Off - P_On, R_AxisFrontal wrapped to (-180, 180].
3. **Uni-G** (`unig_features.csv`): the same, with T = T+_Amp if T_Morph > 0 else T-_Amp, P = P+_Amp if
   P_Morph > 0 else P-_Amp (the PTB-XL+ harmonization rule), R peak time = QRS_IntDefl, PR_Int and P_Dur only
   when P_Found_Global = 1, QRS_AxisFront (missing when the axis is flagged indeterminate), P_Term_V1.

Known bias, declared now: the frozen delineation's QRS onset and offset are the 10th and 90th percentiles of
the per-lead supports, so own QRS durations are about 20 ms longer than 12SL's and Uni-G's (prototype on 300
`fit` clean normals: medians 110, 88 and 90 ms; 062 reported the same 110 ms). The prototype also showed weak
own PR agreement (correlation 0.33-0.38).

**Consensus**: for each measurement the median over the available sources (own, 12SL, Uni-G; the mean of two
if one is missing). Terminal amplitude and net area exist only in own; P terminal force in own and Uni-G.
Derived record measurements are computed from the consensus values: Sokolow-Lyon sum S(V1) + max(R(V5), R(V6)),
Cornell sum R(aVL) + S(V3), the largest peak-to-peak QRS amplitude over the limb leads and over the precordial
leads, R - S per lead.

Measurements use no ECGdeli output and no PTB-XL statement. Using 12SL and Uni-G goes beyond the letter of
the 061 map contract (raw waveform, age and sex): they are waveform-derived measurements by commercial
software, prescribed as inputs by the brief, and no interpretive statement of either program is used. The
diagnostic variant `criteria_own` keeps to the contract exactly.

## Criteria table

Sources: Fourth Universal Definition of Myocardial Infarction (Thygesen et al., 2018; UDMI); AHA/ACCF/HRS
Recommendations for the Standardization and Interpretation of the ECG, 2009: Part III intraventricular
conduction (Surawicz et al.), Part IV ST segment, T and U waves and QT (Rautaharju et al.), Part V chamber
hypertrophy (Hancock et al.); Sokolow and Lyon (1949); Cornell voltage (Casale et al., 1985). Age and sex
come from the cache index (PTB-XL sex 0 = male, 1 = female; age missing above 89 counts as 40 or older).
Thresholds are the published values and are not tuned on any data.

| Criterion | Leads | Wave | Condition (all must hold) | Supports (065 heads) |
| --- | --- | --- | --- | --- |
| `ste_v2v3` (UDMI) | V2, V3 | ST | ST_J >= 0.2 mV (men >= 40 y), 0.25 (men < 40), 0.15 (women) | territory MI + injury |
| `ste` (UDMI) | I, II, III, aVL, aVF, V1, V4, V5, V6 | ST | ST_J >= 0.1 mV | territory MI + injury |
| `std` (UDMI) | all but aVR | ST | ST_J <= -0.05 mV | territory ischemia, ISC_, STD_, NST_, DIG |
| `tinv` (UDMI) | I, II, aVL, aVF, V2-V6 | T | T <= -0.1 mV and R - S >= 0 (prominent R, R/S > 1) | territory ischemia, ISC_, INVT, NT_, NDT |
| `q_v2v3` (UDMI) | V2, V3 | QRS | Q duration >= 20 ms (QS included) | territory MI, QWAVE |
| `q_path` (UDMI) | I, II, aVL, aVF, V4, V5, V6 | QRS | Q duration >= 30 ms and Q depth >= 0.1 mV | territory MI, QWAVE |
| `rvh_v1` (Part V) | V1 | QRS | R >= 0.7 mV or R - S >= 0 (either) | RVH, CRBBB, IRBBB |
| `sokolow` | V1, V5, V6 | QRS | S(V1) + max(R(V5), R(V6)) >= 3.5 mV | LVH |
| `r_avl` (Sokolow-Lyon) | aVL | QRS | R(aVL) >= 1.1 mV | LVH |
| `cornell` | aVL, V3 | QRS | R(aVL) + S(V3) >= 2.8 mV (men), 2.0 (women) | LVH |
| `rbbb_v1` (Part III) | V1, V2 | QRS | QRS >= 120 ms and terminal amplitude >= 0 (terminal R') | CRBBB, IRBBB, IVCD |
| `rbbb_lat` (Part III) | I, V6 | QRS | QRS >= 120 ms and terminal amplitude <= 0 (wide terminal S) | CRBBB, IRBBB, IVCD |
| `lbbb_lat` (Part III) | I, aVL, V5, V6 | QRS | QRS >= 120 ms and R peak time >= 60 ms | CLBBB, ILBBB, IVCD |
| `lbbb_v1` (Part III) | V1 | QRS | QRS >= 120 ms and S - R >= 0 (QS or rS) | CLBBB, ILBBB, IVCD |
| `ivcd` (Part III) | all | QRS | QRS >= 110 ms | IVCD, CLBBB, CRBBB, ILBBB, IRBBB, WPW |
| `lafb` (Part III) | I, II, III, aVL, aVF | QRS | frontal axis <= -45 degrees | LAFB |
| `lpfb` (Part III) | I, II, III, aVL, aVF | QRS | frontal axis >= 90 degrees | LPFB, RVH |
| `pr_long` | all | PR (P and PQ) | PR >= 200 ms | 1AVB, LPR |
| `wpw` (Part III) | all | PR and QRS | PR <= 120 ms and QRS >= 120 ms | WPW |
| `qtc_long` (Part IV) | all | T | QTc >= 450 ms (men), 460 (women) | LNGQT |
| `qtc_short` (Part IV) | all | T | QTc <= 390 ms | none (abnormal probability) |
| `low_limb` | I, II, III, aVR, aVL, aVF | QRS | largest limb peak-to-peak <= 0.5 mV | LVOLT |
| `low_prec` | V1-V6 | QRS | largest precordial peak-to-peak <= 1.0 mV | LVOLT |
| `p_tall` (Part V) | II, III, aVF | P | P >= 0.25 mV | RAO/RAE |
| `p_tall_v1` (Part V) | V1 | P | P >= 0.15 mV | RAO/RAE |
| `p_wide` (Part V) | II | P | P duration >= 120 ms | LAO/LAE |
| `ptf_v1` (Part V) | V1 | P | P terminal force >= 4 mV x ms (0.04 mm x s) | LAO/LAE |

Territories (heads of 065): V1-V3 ASMI, AMI, ANEUR, INJAS, ISCAS; V4 AMI, ALMI, ANEUR, INJAS, INJAL, ISCAS,
ISCAL; V5-V6 ALMI, LMI, ILMI, INJAL, ISCAL, ISCLA, ISCIL; I and aVL ALMI, LMI, INJAL, ISCAL, ISCLA; II, III
and aVF IMI, ILMI, ISCIN, ISCIL; aVR none. "Territory MI" are the MI heads of the lead's territory (ASMI,
AMI, ALMI, LMI, ILMI, IMI, ANEUR), "injury" INJAS and INJAL, "ischemia" ISCAS, ISCAL, ISCLA, ISCIN, ISCIL.

Simplifications, declared: each lead is judged alone (the UDMI two-contiguous-lead rule and the "absent LVH
and LBBB" exclusions are not applied, because the map scores leads); III and V1 carry no Q criterion (UDMI
excludes them); aVR carries only global criteria; no "new" (serial) condition is available.

## Continuous evidence

For a criterion with one condition on measurement m and threshold t, the exceedance in lead l is
z = (m - t) / s (for ">=" conditions) or (t - m) / s (for "<="), where s is the robust spread of the same
measurement in the same lead (or of the record measurement, for global and derived measurements) over the
2,666 `fit`-partition clean normals: s = max(IQR / 1.349, floor). The median of that distribution is
reported in the normal reference table; it does not enter z, because the offset is the guideline threshold.
For conditions joined by "and", z is the minimum of the exceedances; for "or" (`rvh_v1`) the maximum.
Evidence e = max(0, z): zero unless the lead meets the criterion, then how far beyond the threshold it is in
units of the normal spread. A missing measurement gives e = 0.

Floors: ST 0.025 mV; R, S, T, Q depth, R - S, terminal amplitude 0.05 mV; P amplitude 0.02 mV; durations and
R peak time 10 ms (PR and QTc 15 ms); axis 15 degrees; Sokolow and Cornell sums 0.2 mV; largest peak-to-peak
0.1 mV; P terminal force 2 mV x ms.

Lead x wave score: E[l, w] = maximum over the criteria that name lead l and wave w of their evidence, for the
five 066 components w = P, PQ, QRS, ST, T (a PR criterion names P and PQ). The map paints E[l, w] over that
component of every delineated beat (`combination066.delineated_components`, the same units as 066 and 068),
zero outside beats, plus 1e-3 times the 061 `amplitude` map as the tie-break (the 066 and 061 `TIE`).
E is not rescaled per record, so record-level metrics compare evidence across records.

## Maps (declared before any score)

1. `criteria` (primary): consensus measurements, criteria table, evidence and painting as above. Label-free:
   no diagnosis model and no region table.
2. `criteria_agree`: `criteria` times a lead reliability r_l. Disagreement D_l is the median, over Q depth,
   R, S, ST_J, T and P with at least two sources, of (largest minus smallest source value) / s, with s the
   consensus normal spread of that lead and measurement. r_l = min(1, q95_l / D_l), where q95_l is the 95th
   percentile of D_l over the `fit` clean normals (r_l = 1 when D_l is 0 or missing). The reading asked for is
   whether it lowers C1_std relative to `criteria`.
3. `criteria_x_mil`: E[l, w] = max over criteria c of w_c,l x e_c,l, with w_c,l the largest 065 bag
   probability (sigmoid of the three-seed ensemble `mil_lse` logit, `outputs/experiment065_mil_v1/bags_*.npz`)
   among the heads the criterion supports in lead l (Supports column and territories), or 1 - p(NORM) for
   `qtc_short`.
4. `units_v2_criteria`: the 068 combiner, by import (`combiner068`), with two features added to the 068
   `units_v2` feature set: the within-record z-scored log1p unit means of `criteria` and of `criteria_x_mil`.
   Training exactly as 068: the same 3,517 `fit` records with region statements and the same union weak labels,
   068's saved unit features (`/tmp/ecg068-work/full/features_train.pkl` and `features_scored.pkl`, with
   cross-fit out-of-fold MIL evidence for training records and the 065 ensemble for scored records), the
   068 out-of-fold bag logits (`oof_bags.npy`) for the training records' `criteria_x_mil`, the same patient
   folds, the same (a, L2) grid and the lowest cross-validated listwise loss, then a final fit on all
   training records and the 066 painting (`combination066.units_map`).
5. `criteria_own` (diagnostic): `criteria` from own measurements only, with own normal spreads. It obeys the
   061 map contract exactly. Its QRS criteria inherit the 20 ms own-QRS bias, so it is a reading of the
   frozen delineation, not a candidate.
6. `criteria_qrs` (secondary): lexicographic with the own QRS component, 2 x [sample in QRS] + E / (1 + E)
   plus the tie-break, as 062's `qrs_lexicographic`: every QRS sample ranks above every non-QRS sample, and
   criteria evidence orders samples inside each.

## Comparisons and decision rules (all ECGAD readings exploratory)

ECGAD development has been compared many times; no ECGAD result here is called confirmed.

- Primary: paired ECGAD development A2 of `criteria` minus 061 `qrs_prior`. Words: "localizes" if the 95%
  interval is above 0, "below" if under 0, otherwise "matches".
- Key secondary: `units_v2_criteria` minus 068 `units_v2`; `criteria_x_mil` minus 066 `textbook`. Also
  reported: every map against `qrs_prior` and `units_v2`; `criteria_agree` and `criteria_own` against
  `criteria`.
- Noise: `criteria_agree` lowers C1_std if the paired difference against `criteria` (same noise records,
  whole-patient draws) has its interval below 0.
- Region test B: `criteria`, `criteria_agree`, `criteria_own` and `criteria_qrs` never see `REGION_TABLE` or
  any statement, so their B1, B2 and B3 are an independent test of whether measured evidence lands on the
  statement's expected leads and waves. `criteria_x_mil` uses the 065 heads (trained on statements, not on
  the region table) and the territory table, which overlaps the region table's lead sets, so its B is only
  partly independent. B is not independent for `units_v2_criteria`, which is trained on region-table weak
  labels.
- Every map gets the full LocBench readings (A1-A6, B1-B3 with strict variants, C1, C2, D1), the 061
  amendment 2 strata (A2_qrs, A2_nonqrs) and the amendment 3 C1_std, with paired strata against `qrs_prior`
  and `units_v2`.

## Patient specificity (descriptive)

- Evidence leads of a record: C = leads with max over waves of E[l, w] > 0 (map `criteria`).
- Region set (evaluation partition; families whose lead set is not all 12 leads): per family, the share of
  records with C empty, the mean Jaccard index of C with the textbook leads L, and the share of records with
  at least one lead of C outside L (departure).
- ECGAD development, marked records with at least one `REGION_TABLE` statement at likelihood >= 50 whose
  union lead set L is not all 12 leads: lead groups C \ L (departures), C ∩ L, L \ C, and neither; the share of
  leads in each group that the cardiologist marked (any mark in samples 100-4899), pooled over records, with
  counts. Departures agree with the marks if their marked share exceeds that of the "neither" leads.

## Fit-partition sanity readings (descriptive, no ECGAD)

- Agreement of own, 12SL and Uni-G on the `fit` clean normals per measurement (median absolute difference,
  Pearson correlation), and the normal reference table (median and s per lead).
- Criterion prevalence (any lead meets it) on `fit` clean normals, and on `fit` records carrying the matching
  statement at likelihood >= 50 (1AVB `pr_long`; CLBBB `lbbb_lat`; CRBBB `rbbb_v1`; LAFB `lafb`; IMI
  `q_path` in II or aVF; ASMI `q_v2v3`; LVH any of `sokolow`, `r_avl`, `cornell`; LNGQT `qtc_long`).

## Integrity checks

- Bar: rescore 061's saved `qrs_prior` ECGAD maps with `locbench.ecgad_record_metrics`; per-record A2 within
  1e-12 of 061's `ecgad.csv` and mean 0.7419383358406554 within 1e-12.
- 068 rebuild: `units_v2` repainted from 068's saved scored features with 068's saved weights equals 068's map
  file (largest absolute difference reported, expected 0) and its per-record A2 matches 068's tables within
  1e-12.
- Units: the unit tables of this run's delineation equal 068's saved unit tables for every training and scored
  record.
- 065 bags: the logits read from `bags_*.npz` equal 066's `mil_bags.npy` for every scored record.
- Separation: no fitted quantity uses an ECGAD or `evaluation` record (normal spreads and q95 use `fit` clean
  normals; the combiner uses the 068 training records); no ECGAD holdout record is mapped or loaded; the
  feature files are filtered to folds 1-8 and ECGAD development.

## Implementation and outputs

`ecg_experiment/criteria069.py`, `scripts/experiments/run_criteria069.py` (phases `measure`, `maps`,
`combiner`, `score`), `tests/test_criteria069.py`. CPU only, at most 8 processes. Large intermediates in
`/tmp/ecg069-work`; outputs in `outputs/experiment069_criteria_v1/` (`result.json` with integrity checks,
every reading, decisions and file hashes; LocBench tables per map; reference tables copied into `ref/` before
the strata and noise CLIs write into them). A smoke run on a few records checks shapes, finiteness and timing
and is not scored. The results report is `docs/experiment-069-criteria-localizer-results.md`.
