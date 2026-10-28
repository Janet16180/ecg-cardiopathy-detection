# NBR-03 protocol: replicating NBR-01 on PTB-XL fold 9

Frozen 28 October 2026, before any replacement score. Branch `research/nbr03-ptbxl-replication-20261028`,
stacked on the NBR-02 branch (`8cff2ca`, pull request #79), which is itself stacked on NBR-01. The user approved
this study; the design comes from the lead's brief and the choices below are fixed here.

## Question

Does the NBR-01 method, with the same P0 counterpart unchanged, localize the screening blend's evidence on a
second, independently labelled dataset, PTB-XL fold 9, whose statements include lateral MI and territorial
ischaemia?

## Data rules (stage `checks`, already run; labels and identities only)

- Only PTB-XL `strat_fold` 9 is read: the 2,146 fold-9 rows of the frozen baseline table
  `outputs/pr74_fresh_baseline_exact_20261028/features/ptb_rows.csv` (role `development`, the rows whose
  saved scores exist). Fold 10 is reserved for `final_frozen_test`: the code raises if the table holds a
  fold-10 row (it holds none), every preparation call refuses a row outside fold 9, and the frozen reader
  `read_signal` rejects folds outside 1-9.
- Fold 9 is unseen by both models, verified from their receipts: `all_age.csv` matches its cohort receipt hash
  and its PTB-XL rows come from folds 1-7; `ptb_rows.csv` matches its feature receipt hash and the
  `ptb_baseline` fit rows are folds 1-7 (fold 8 is its threshold calibration, not its fit; the brief's
  "folds 1-8" is therefore one fold wider than the actual fit).
- By patient, none of the 1,917 fold-9 patients appears in the all-age PTB training rows, the `ptb_baseline`
  fit rows, the fold-8 threshold calibration rows, or the 2,666 LocBench `fit` clean normals that fitted the
  P0 counterpart (LocBench covers folds 1-8 only). So no fold-9 ECG needs a patient exclusion.

## Groups (SCP statements only, one group per ECG)

Likelihood rule: a statement counts whatever its likelihood value, exactly as the project's binary label
(`eda.ptbxl.superclass_label`) reads PTB-XL; all 2,146 fold-9 targets agree with that function. Rules, in order:

| Group | Statements | Expected region |
| --- | --- | --- |
| inferior | IMI, ISCIN, INJIN | II, III, aVF |
| anterior | AMI, ASMI, ISCAN, ISCAS, INJAS | V1-V4 |
| lateral | LMI, ISCLA, INJLA | I, aVL, V5, V6 |
| lbbb | CLBBB, with no MI or territorial statement | QRS |
| rbbb | CRBBB, with no CLBBB, MI or territorial statement | QRS |
| stt | NDT, NST_ or ISC_, with no MI, territorial ischaemia, DIG, LNGQT, EL, ANEUR, conduction (CD) or hypertrophy (HYP) statement | ST-T |
| normal | NORM as the only diagnostic statement and a negative label | none |

- A territory ECG has statements from exactly one territory and no multi-territory statement. The
  multi-territory statements ALMI, ILMI, ISCAL, ISCIL, IPLMI, PMI, INJAL and INJIL exclude the ECG.
- IPMI (inferoposterior MI) is judged multi-territory and also excludes the ECG: the posterior part is a
  separate territory, consistent with excluding PMI and IPLMI.
- The subendocardial injury statements INJIN, INJAS and INJLA name a single territory and join it.
- Other statements (for example a bundle branch block with an inferior MI) do not move an ECG out of its
  territory group, as in NBR-01.
- The stt group excludes drug, electrolyte, QT and aneurysm statements and any conduction or hypertrophy
  statement, mirroring NBR-01's SPH rule.

Eligible counts from labels only (ECGs / patients): inferior 185 / 173, anterior 140 / 134, lateral 15 / 14,
lbbb 49 / 42, rbbb 28 / 26, stt 194 / 184, normal 914 / 866.

Lateral has fewer than 40 patients. By the brief's rule the primary contrast uses inferior and anterior, and
lateral is secondary. The `prepare` stage recomputes the territory patient counts after any record that 062
cannot delineate is excluded, and keeps every territory with at least 40 patients in the primary. The
study stops if fewer than two remain.

Sample: NBR-01's rule, `ns.draw_sample`, seed 20261028, one ECG per patient across the sample, 300 patients
per abnormal group and 300 normals. Expected: inferior 173, anterior 134, lateral 14, lbbb 42, rbbb 26, stt
184, normal 300 (873 ECGs).

## Counterpart, regions and models

- Counterpart: NBR-01's P0, unchanged. The model is NBR-01's saved `counterpart/rank0.npz`. Rebuilding 100
  SPH counterparts with this code matched NBR-01's saved ones exactly (0.0 mV).
- Regions: NBR-01's 105 variants, masks and taper, unchanged.
- Covariates: PTB-XL age, with ages above 89 (coded 300) read as missing (062 then uses 90, as for LocBench),
  and PTB-XL sex (0 male, 1 female), 062's own coding.
- Models: `all_age_blend` is primary. `ptb_baseline_blend` and the HGB and xECG components of both are
  secondary. `ptb_baseline` was trained on PTB-XL folds 1-7, so on fold 9 it is in-domain; `all_age` saw
  PTB-XL folds 1-7 among five sources.
- Thresholds: the frozen PTB thresholds of the bundle, set at a 5% budget on fold-8 calibration normals:
  `all_age_blend` 0.9007447632467718 and `ptb_baseline_blend` 0.6768595920359941, components likewise.
  Fold 8 is patient-disjoint from fold 9 (checked), so the flip rates use these thresholds as real
  operating points.

## Primary endpoint

NBR-01's territory contrast for `all_age_blend` with the P0 counterpart, averaged over inferior and anterior:
the expected-lead share of the whole-lead replacement drops in each territory group minus that share in the
other primary territory group. The share and contrast definitions are NBR-01's code, unchanged
(`nbr01_analysis.statistics`); the lateral ECGs are relabelled `lateral_secondary` for this contrast only.

Intervals: 95% whole-patient bootstrap, 2,000 draws, `numpy.random.default_rng(20261028)`, resampling all
analysed patients jointly; every statistic on the same draws.

Decision: localization replicates if the lower bound is above 0. Key qualifier: model minus energy, with the
energy map recomputed on fold 9; only a lower bound above 0 supports localization beyond replaced energy.

## Secondary endpoints (intervals, no decision)

- Every NBR-01 endpoint: wave contrasts, BBB lead contrast, top-1 and top-3 territory hits against energy,
  self-swap territory contrast, flip rates, full-replacement sanity, self-swap artefact and normals.
  Selectivity and flip pools follow NBR-01's definition of "abnormal" (inferior, anterior, lbbb, rbbb, stt).
- The same territory statistics with all three territories (`all_territories/` keys), the per-territory
  parts including lateral (`parts/` keys), and the lateral group's flagged share, changes and flips.
- The HGB and xECG components separately, and everything for `ptb_baseline_blend` (in-domain).
- Selectivity (abnormal over normal mean absolute whole-lead change), as defined in NBR-02.

## Integrity checks

1. The new analysis code, applied to NBR-01's saved SPH scores, reproduces NBR-01's primary
   +0.3056275517940945 exactly.
2. Rebuilt P0 counterparts match NBR-01's on 100 SPH records (0.0 mV; done).
3. Unreplaced fold-9 scores, computed with the exact pipeline from the raw signals, must reproduce the saved
   fold-9 scores in `outputs/pr74_from_raw_validation_20261028/evaluate/ptb_predictions.npz` (these exist for
   both models): largest absolute difference at most 1e-12 for every component and blend, identical
   referral decisions at the PTB thresholds, and PTB thresholds that replay exactly from the saved fold-8
   calibration scores. Pre-registered fallback, NBR-01's Amendment 1 rule: the `parity` stage re-encodes every
   record whose unreplaced xECG features are not bit-identical inside its historical batch of 16 PTB rows
   (the final 13-row batch padded with its last waveform, as in the PR 74 batch-shape fix). The check passes
   if every record over the tolerance reproduces bit for bit there and decisions are identical. Otherwise
   the study stops.
4. Variant SHA-256 between preparation and encoding, as in NBR-01.

## Profile (already run, no score)

20 fold-9 ECGs (3 per abnormal group and 2 normals, seed 20261029): preparation 4.51 CPU-seconds per record
(16 workers, 7.1 s for 20); encoding of 105 variants 1.01 s per record on GPU 0, 0.62 GB peak, 10.6 GB
free. Forecast for 873 ECGs: preparation about 4 minutes and encoding about 8 minutes over both GPUs.

## Page export

`outputs/nbr03_ptbxl_replication_20261028/examples.json`: NBR-01's final schema (v1 fields, `summary`,
self-swap fields) plus `dataset: "ptbxl_fold9"` and an `age_note`. The thresholds in `models` are the PTB
thresholds. 16 examples are drawn within groups by NBR-01's seeded rule, never by result: 3 inferior, 3
anterior, 2 lateral, 2 rbbb, 2 lbbb, 2 stt and 2 normal. `summary.statistics` adds the all-territory, part and
lateral keys to NBR-01's keys.

## Outputs

`outputs/nbr03_ptbxl_replication_20261028/`; work files in `/tmp/nbr03-work`. Code:
`ecg_experiment/localization/nbr03_*.py`, `ecg_experiment/localization/stages.py`,
`scripts/experiments/run_nbr03.py`, `tests/localization/test_nbr03_ptbxl.py`. Report:
`docs/nbr03-ptbxl-replication-results-2026-10-28.md`.

## Limitations stated in advance

- PTB-XL fold 9 has been evaluated before (PR 74, HEADS-01), so it is development data, not a fresh test.
- Lateral has only 14 patients and can only be described.
- The PTB-XL statements were assigned by cardiologists, partly from automatic reports; statement likelihoods
  are ignored, as in the project's label.
