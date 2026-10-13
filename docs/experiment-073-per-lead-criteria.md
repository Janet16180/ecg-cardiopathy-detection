# Experiment 073: per-lead interval criteria and beat-level criteria

Frozen 13 October 2026 (UTC), before any 073 map was built and before any LocBench score of a 073 map. Part of
the [localization programme, part 2](localization-program-2-2026-10-13.md); scored on LocBench v1
([Experiment 061](experiment-061-locbench.md), amendments [2](experiment-061-amendment-2-qrs-strata.md) and
[3](experiment-061-amendment-3-noise-standardized.md)). Predecessor:
[Experiment 069](experiment-069-criteria-localizer.md) and its [results](experiment-069-criteria-localizer-results.md).
Backlog entry `loc_per_lead_interval_criteria`. The user is away and authorized unsupervised experiments; the
design choices below were made by the agent running this experiment and are frozen here.

Before this freeze, only the following were read or run:

- the programme notes, the 061 protocol and amendments, the 069 protocol, results report, code and
  `result.json` (069 scores only, no per-record ECGAD value);
- the PTB-XL+ `feature_description.csv` and the column names of the three feature files;
- unit and lead-specificity checks of the per-lead interval columns on the 2,666 `fit`-partition clean
  normals only (rows filtered to those records immediately after reading): 5th, 50th and 95th percentiles per
  lead, the share of records whose 12 lead values are all equal, and the within-record range over leads;
- a prototype of own per-lead intervals (from the per-lead supports of the frozen delineation) on the first 300
  `fit` clean normals, compared with Uni-G: medians by lead and agreement only.

No ECGAD mark, ECGAD statement, `evaluation`-partition label, PTB-XL fold 9 or 10 row outside ECGAD, or score of
any 073 map has been read.

## Question

069 painted its global criteria (QRS duration, PR, QTc) on every lead, so in 72% of marked ECGAD development
records every lead carried evidence and the lead AUROC A3 was at chance (0.495). If each interval criterion is
painted only on the leads whose own interval crosses the threshold, weighted by how far past it, and focal beats
get beat-level criteria from each lead's own beat, does lead localization (A3) recover, and does A2 improve over
`qrs_prior` and over 069 `criteria` and `criteria_qrs`?

## What the pre-freeze checks showed (fit clean normals)

- Uni-G per-lead QRS duration (`QRS_Dur_X`) is lead-specific: no record has all 12 values equal, the median
  within-record range is 30 ms, medians by lead are 72-86 ms, and the maximum over leads equals the global
  duration (median difference 0 ms).
- Uni-G per-lead QT (`QT_Int_X`) is lead-specific (median range 44 ms; medians 376-390 ms).
- Uni-G per-lead PR (`QRS_On_X - P_On_X`) varies only through the lead's QRS onset: `P_On_X` is the same in
  all 12 leads of every record (median range of PR 14 ms). It is the only per-lead PR available with plausible
  values.
- Uni-G per-lead P duration (`P_DurFull_X`) equals the global value in 99.9% of records; it is not per-lead.
  12SL `P_DurFull_X` is lead-specific (median range 50 ms; zero when no P was found in the lead).
- 12SL per-lead QRS duration (sum of Q, R, S, R' and S' durations) equals the global duration within 2 ms in 84%
  of lead values. 12SL `T_Off_X` is centred near 0 ms (5th-95th percentile about -35 to 83 ms) and is not
  documented as an interval, so 12SL gives no per-lead QT.
- ECGdeli per-lead intervals are implausible for normals (median QRS 112-138 ms, PR 216-230 ms, QT up to
  460 ms with 95th percentiles of 500-606 ms); ECGdeli stays excluded, as in 069.
- Own per-lead intervals from the frozen delineation's per-lead supports are biased and noisy: QRS 86-106 ms
  (18 ms above Uni-G, correlation 0.51, within-record lead pattern correlation 0.37), and the low-amplitude
  leads III and aVL give PR medians of 189 and 200 ms, QT 424 and 412 ms and P durations 125 and 137 ms. P is
  found in only 14-30% of beats in V1, V5 and V6.

These checks decided the sources below. No threshold is taken from them.

## Measurements

1. **069 measurements, reused**: the 069 `measure` output (`/tmp/ecg069-work/full/measurements.pkl`, SHA-256
   `0c4aea92…`): own median-beat measurements, harmonized 12SL and Uni-G tables, and the delineated beat
   components of every record. Amplitudes, ST, T, Q waves, R peak time, terminal amplitude, axis, RR and the
   derived voltage sums keep the 069 consensus (median over own, 12SL and Uni-G).
2. **Per-lead intervals** (new), read from the PTB-XL+ feature files with the rows filtered to the measured
   records (all in PTB-XL folds 1-8 or ECGAD development) before any value is used:
   - `QRSDL_X`: Uni-G `QRS_Dur_X`.
   - `PRL_X`: Uni-G `QRS_On_X - P_On_X` when `P_Found_Global` = 1, else missing.
   - `QTCL_X`: Framingham QTc from Uni-G `QT_Int_X` and the 069 consensus RR.
   - `PDURL_X`: 12SL `P_DurFull_X` when above 0, else missing.
3. **Own per-lead intervals** (diagnostic map only): per lead, the median over beats of the frozen delineation's
   per-lead supports (`morphology_boundaries.adaptive_boundaries`, the beats kept by `combination066`):
   QRS offset minus onset, QRS onset minus P onset, T end minus QRS onset (QTc with the own RR), P offset minus
   onset. P measures need a P support in at least half of the lead's beats, else missing.
4. **Beat measurements** (own signal; used by every map except `perlead_median`): for every kept beat b, with the
   signal preprocessed by `normative_map062.preprocess` and the lead median beat m_l built as in 069 (beats whose
   window 300 ms before to 700 ms after QRS onset lies inside the record):
   - prematurity pi_b = (QRS onset of b minus that of the previous kept beat) / the record's median QRS-onset
     spacing; missing for the first beat;
   - QRS deviation delta_bl = RMS over the window [QRS onset of b, + W) of (beat - m_l), divided by
     max(RMS of m_l over the same window, 0.1 mV), with W the record's median joint QRS width; missing when the
     window leaves the record;
   - beat QRS duration d_bl: QRS offset minus onset of lead l's support in beat b (ms).

## Criteria

The 069 criteria table, with these changes (all thresholds published, as in 069):

| Criterion | Change from 069 |
| --- | --- |
| `rbbb_v1`, `rbbb_lat`, `lbbb_lat`, `lbbb_v1`, `ivcd` | QRS duration of the record replaced by `QRSDL` of the same lead |
| `pr_long` | PR replaced by `PRL` of the lead |
| `wpw` | PR and QRS duration replaced by `PRL` and `QRSDL` of the lead |
| `qtc_long`, `qtc_short` | QTc replaced by `QTCL` of the lead |
| `p_wide` | P duration of the record replaced by `PDURL_II` (the AHA criterion names lead II) |
| `lafb` | axis <= -45 degrees and the published lead pattern (AHA 2009 Part III): rS in II, III, aVF (S - R >= 0); qR in aVL (R - S >= 0). No longer painted on I. |
| `lpfb` | axis >= 90 degrees and the published pattern: rS in I and aVL (S - R >= 0); qR in III and aVF (R - S >= 0). No longer painted on II. |

Every other criterion (ST, T, Q waves, voltage, low voltage, P amplitude, P terminal force, `rvh_v1`) is the 069
criterion unchanged; they were already judged on the lead's own measurement or name only one to three leads.

Beat-level criteria (own signal, published or fit-normal thresholds):

| Criterion | Condition in beat b, lead l | Paints (beat b, lead l) |
| --- | --- | --- |
| `beat_premature` | pi_b <= pi01 and delta_bl >= delta99_l | QRS, ST, T |
| `beat_wide` | d_bl >= 120 ms and delta_bl >= delta99_l | QRS |

pi01 is the 1st percentile of pi over every beat of the 2,666 `fit` clean normals, delta99_l the 99th
percentile of delta_bl per lead over the same beats. They are thresholds from fit-partition clean normals; 120 ms
is the published wide-QRS threshold. Beat-level ST or T criteria are not used: no published criterion is defined
per beat, and a median beat measures ST and T more reliably.

## Continuous evidence and painting

As 069: z = (m - t) / s for ">=", (t - m) / s for "<=", with s the floored robust spread (IQR / 1.349) of the
same column over the `fit` clean normals; "and" takes the minimum of the exceedances; e = max(0, z), 0 when a
measurement is missing. Floors as 069, plus `QRSDL` 10 ms, `PRL` 15 ms, `QTCL` 15 ms, `PDURL` 10 ms, pi 0.02,
delta 0.05, beat QRS duration 10 ms.

Median-beat score E[l, w] = maximum over criteria naming lead l and component w (069 `lead_wave`). Beat score
B[b, l, w] = maximum over the beat-level criteria. The map paints max(E[l, w], B[b, l, w]) over component w of
beat b in lead l (the `combination066.delineated_components` units of 069), zero outside beats, plus 1e-3 times
the 061 `amplitude` map. No per-record rescaling.

## Maps (declared before any score)

1. `perlead` (primary): per-lead criteria and beat-level criteria as above; no reliability factor.
2. `perlead_median`: `perlead` without the beat-level criteria.
3. `perlead_shift`: `perlead` with each per-lead interval threshold shifted by the normal offset between the lead
   and the record: t_l = t + (median over `fit` clean normals of the per-lead column - median of the same
   source's global measurement: Uni-G `QRS_Dur_Global`, `PR_Int_Global` with P found, Framingham QTc of
   `QT_Int_Global`, and 12SL `P_Off_Global - P_On_Global`). It asks whether the published thresholds, written for
   global intervals, are too strict per lead.
4. `perlead_agree`: `perlead` times the 069 lead reliability r_l (069 definition, `criteria069.disagreement` and
   `reliability`, q95 from `fit` clean normals), to test dropping the factor.
5. `perlead_qrs`: `perlead` made lexicographic with the own QRS component, as 069 `criteria_qrs`.
6. `perlead_own` (diagnostic): own measurements only (069 own table, own per-lead intervals, own RR, own spreads)
   with the beat-level criteria. It keeps to the 061 map contract; it inherits the biases above.

## Comparisons and decision rules (all ECGAD readings exploratory)

ECGAD development has been compared many times; no ECGAD result here is called confirmed. Intervals are LocBench's
paired whole-patient bootstraps (2,000 draws, seed 61061).

- Primary: paired ECGAD development A2 of `perlead` minus 061 `qrs_prior`. Words: "localizes" if the 95% interval
  is above 0, "below" if under 0, otherwise "matches".
- Key secondary: paired A3 of `perlead` minus 069 `criteria`. "Lead localization recovers" if the interval is
  above 0; otherwise "does not recover".
- Also reported: `perlead` minus 069 `criteria` on A2; `perlead_qrs` minus 069 `criteria_qrs` and minus
  `qrs_prior` on A2 and A3; `perlead` minus `perlead_median` (beat-level criteria) on A2, A3 and A4;
  `perlead_shift` minus `perlead` on A2 and A3; `perlead_own` minus `perlead`.
- Reliability factor: `perlead_agree` minus `perlead` on A2 and C1_std. The factor stays dropped unless the C1_std
  interval is below 0 and the A2 interval is not below 0.
- Region test: B1, B2, B3 of every map against `qrs_prior` and 069 `criteria`. No 073 map uses `REGION_TABLE` or
  any statement, so B is an independent test for all six.
- Every map gets the full LocBench readings (A1-A6, B with strict variants, C1, C2, D1), the amendment 2 strata
  (A2_qrs, A2_nonqrs, paired against `qrs_prior`) and the amendment 3 C1_std (paired against `qrs_prior` and
  069 `criteria`).

## Descriptive readings

- Lead specificity on ECGAD development (declared here; 069 reported it as undeclared): share of marked
  records with evidence on all 12 leads, with none, and the mean number of evidence leads (leads with max over
  waves of E > 0, median-beat evidence), for `perlead` and 069 `criteria`. Beat-level evidence is reported as
  the share of marked records with any beat-level evidence.
- Region set and ECGAD departure readings as in 069 (evidence leads against textbook leads; marked share of
  departure, shared, textbook-only and other leads) for `perlead`.
- On `fit` clean normals: the share meeting each criterion in some lead under `perlead` and under 069's
  consensus, in particular `p_wide` (25% under 069) and `rvh_v1` (5.7%, the V1 R/S rule, which this experiment
  does not change), and the share of records and beats with beat-level evidence.
- On `fit` records with the matching statement at likelihood >= 50: the 069 sanity table (1AVB `pr_long`, CLBBB
  `lbbb_lat`, CRBBB `rbbb_v1`, LAFB `lafb`, IMI `q_path` in II or aVF, ASMI `q_v2v3`, LVH voltage, LNGQT
  `qtc_long`) under `perlead`, `perlead_shift` and 069.

## Integrity checks

- Bar: 061 `qrs_prior` ECGAD maps rescored with `locbench.ecgad_record_metrics`; mean A2 0.7419383358406554
  within 1e-12 and per-record values within 1e-12 of 061's `ecgad.csv`.
- 069 rebuild: 069 `criteria` lead x wave scores recomputed from the 069 measurements with the 069 functions equal
  069's `evidence_scored.npz` (largest absolute difference reported, expected 0); 069 `criteria` and
  `criteria_qrs` ECGAD development maps repainted in this run equal 069's saved `ecgad_maps.npy`, with per-record
  A2 within 1e-12 of 069's `ecgad.csv` and mean A2 0.7240953915617058 and 0.7631057507901341.
- Geometry: the beat components of this run's measure phase equal the 069 saved components for every record.
- Separation: every normal reference and threshold uses `fit` clean normals only; the feature rows read are those
  of the measured records, all in folds 1-8 or ECGAD development; no ECGAD holdout record is mapped or loaded.

## Implementation and outputs

`ecg_experiment/criteria073.py`, `scripts/experiments/run_criteria073.py` (phases `measure`, `maps`, `score`),
`tests/test_criteria073.py`. 069 helpers are imported, not copied. CPU only, at most 8 processes. Work folder
`/tmp/ecg073-work`; outputs in `outputs/experiment073_per_lead_criteria_v1/` (`result.json` with integrity checks,
every reading, decisions and file hashes; LocBench tables per map; reference tables copied into `ref/` before the
strata and noise CLIs write into them). A smoke run on a few records checks shapes and timing and is not scored.
The results report is `docs/experiment-073-per-lead-criteria-results.md`.

## Amendment 1, 13 October 2026, before any score

The smoke run (16 scored, 200 `fit` clean-normal and 40 `fit` sanity records; nothing scored, no ECGAD or
`evaluation` value read) printed the criterion prevalence on its 200 `fit` clean normals. Per-lead `qtc_short`
(QTc <= 390 ms in some lead) was met in 76.5% of them, against 11% for the 069 record-level criterion. A lead's QT
is never longer than the record's QT, and the AHA (Part IV) measures QT in the lead with the longest interval, so
a per-lead "<=" QTc condition fires on normal leads whose T ends early. `qtc_short` therefore keeps the 069
record-level condition (consensus QTc <= 390 ms, painted on all 12 leads) in `perlead`, `perlead_median`,
`perlead_shift`, `perlead_agree` and `perlead_qrs`; in `perlead_own` it uses the own record-level QTc, as
`criteria_own` did. A per-lead short-QT criterion is not possible with these measurements.

No other definition changes. The same smoke readout showed per-lead `pr_long` met in 11% of those normals against
6.5% for the record-level criterion: Uni-G's per-lead PR uses a common P onset, so it is at least the record PR.
That is the frozen design, and it is reported, not changed.
