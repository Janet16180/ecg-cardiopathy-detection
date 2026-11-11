# MLC-03 results: data quality and cleaning of the 14-condition dataset

11 November 2026. This is the overall report of MLC-03, written from its outputs by the agent that built dataset v5
and ran Part D. It answers the four questions of the [protocol](mlc03-data-quality-protocol-2026-11-11.md) (with
Addendum 1 and Amendment 1). Parts A and B have their own reports, cited and not repeated:
[signal quality](mlc03-signal-quality-results-2026-11-11.md) and
[label quality](mlc03-label-quality-results-2026-11-11.md). The build has its own report:
[dataset v5](mlc03-dataset-v5-results-2026-11-11.md).

Numbers about the rules, v5 and Part D come from files this agent's code wrote. They are printed by
`PYTHONPATH=. .venv/bin/python -m scripts.reports.mlc03_v5_tables --section <name>` and cited as [section
`<name>`]. Numbers about Parts A and B are cited from their reports. No test role was scored by any model. The
test roles received the frozen rules and are reported as counts only. No Challenge test group, EchoNext or MIMIC
record was opened. Nothing here is a clinical reading. No cardiologist has reviewed any of it; clinical validation
is pending (validación pendiente).

## Answers

1. **Signal.** A small share of records carry a defect that makes the stored signal not a measurement, and
   most of these defects come from one or two sources.
   - In Part A, 6,015 of the 71,388 included train and validation records (8.4%) carry at least one of 18 flags.
     Most are noise flags, which describe real defects of degree [Part A].
   - Four defects mean the stored samples are not a measurement: missing samples (87 records, 82 of them
     Ningbo), a held start or end (15), samples at the 16-bit rail (116), and derived leads inconsistent with
     I and II (47, of which 41 in CPSC 2018 and CPSC-Extra, where most plotted cases were a 16-bit wrap-around)
     [Part A].
   - Defects specific to one source:
     - Ningbo: constant chest leads in 4.3% of its records, and most of the contact loss.
     - PTB-XL: every record holds all leads for the last 90-130 ms.
     - CPSC 2018 and CPSC-Extra: wrap-around, all-lead rail events, and defects after the first 10 s.
     - Georgia: most of the limb-electrode reversals.
     - SPH: float16 storage, 174 identity violations and 64 values of 32.7 mV or more, whose cause was not
       examined [Part A].
   - v5 excludes 159 records: 117 train, 41 validation and 1 at PTB-XL fold 10 [section `totals`].
2. **Labels.** Labels disagree with what can be measured mostly where sources code the same finding differently,
   not at random.
   - Sixteen of 17 matched-prevalence analyses find an inconsistent pair of sources [Part B]:
     - PTB-XL and CPSC-Extra code sinus bradycardia from about 50 per minute, the other sources from about 60.
     - Chapman and CPSC 2018 have a single RBBB code, so their "complete RBBB" holds measurably narrow
       complexes.
     - Left axis deviation and first-degree AV block are coded at different measured values in different
       sources.
   - Contradictions inside one record are few, except labels on paced records.
   - Model-based estimates of wrong labels are 0.3-4.5% per condition, pooled. In 336 sheets of the most
     suspected labels, read by eye by a non-clinician, the measurement contradicted the label beyond tolerance
     in 149 [Part B]. These are not error rates of the sources.
3. **Cleaning.** Twelve rules define v5. They rest on Parts A and B and on no model score; none creates or
   flips a positive. Four rules exclude records. Eight set labels undefined: 21,276 labels on included
   records, 1,435 positives and 19,841 negatives [sections `totals`, `integrity`]. No condition loses macro
   eligibility in any evaluation role or source [section `eligibility`].
4. **Effect (secondary, validation only).** Training on v5 changes neither model's selection metric
   demonstrably.
   - The v5-trained minus v4-trained difference on v5 validation is +0.0008 (-0.0003, +0.0019) for P0 and
     +0.0001 (-0.0006, +0.0010) for H-All.
   - Scoring on v5 labels alone raises both v4-trained models' metric: P0 from 0.9643 to 0.9690, H-All from
     0.9706 to 0.9751 [section `effect`].
   - No per-condition difference survives the Holm adjustment over the 14 conditions. LVH comes closest, and
     its unadjusted interval lies above zero for both models [section `effect-conditions`].

## Parts A and B in short

- **Part A** ([report](mlc03-signal-quality-results-2026-11-11.md)) measured every train and validation
  candidate on the cached first 10 s and the whole raw recording.
  - It defined 18 flags with thresholds set from the train and validation distributions, and checked each by
    eye on 24 flagged records.
  - 15 flags passed; two of them (`reversal_la_ra`, `reversal_neutral`) passed only on a re-viewing.
    `reversal_ra_ll` and `rhythm_unmeasurable` failed.
  - None of the four MLC-02 input pipelines removes clipping, steps, wander, contact loss or flat leads.
    Missing samples enter every pipeline as -32.768 mV.
  - No near-duplicate beyond exact hashes was a copy.
  - The frozen flags were then counted on the test roles.
- **Part B** ([report](mlc03-label-quality-results-2026-11-11.md)) compared labels with two automatic
  measurement methods and their consensus.
  - The measurement error was estimated against PTB-XL+ (folds 1-9), LUDB and synthetic records. Heart rate,
    frontal axis, voltage, ST level, T amplitude and Q duration are reliable enough to judge a label; QRS
    duration is reliable away from the 110 and 120 ms cut-offs; PR, P-wave presence and RR irregularity are not.
  - It measured source coding by matched prevalence, listed contradictions inside one record, estimated wrong
    labels from out-of-fold P0 probabilities by confident learning, read 336 suspect sheets, and counted
    candidate rules R1-R11.

## Adopted rules and their counts

Amendment 1 adopted C1-C12. Each is applied by the code that produced its evidence. The gate reproduced every
Part A and Part B count on train and validation exactly: 60 Part A cells and 212 Part B rows [section `gate`].
Selected before overlaps / changed after earlier rules; records for C1-C4, labels for C5-C12 [section `rules`]:

| rule | definition (evidence) | train | validation | fold 10 | SPH calibration | SPH test |
|---|---|---|---|---|---|---|
| C1 | exclude: missing samples in the first 10 s (A-R1) | 66 / 66 | 21 / 21 | 0 / 0 | 0 / 0 | 0 / 0 |
| C2 | exclude: constant head or tail (A-R3) | 10 / 10 | 5 / 5 | 0 / 0 | 0 / 0 | 0 / 0 |
| C3 | exclude: samples at the 16-bit rail, integer sources (A-R2) | 89 / 34 | 27 / 11 | 0 / 0 | 0 / 0 | 0 / 0 |
| C4 | exclude: derived leads inconsistent with I and II, integer sources (A-R7) | 34 / 7 | 13 / 4 | 1 / 1 | 0 / 0 | 0 / 0 |
| C5 | undefined: a lead the condition is read on is flat, near-flat or without contact (B-R5, B-R11) | 2452 / 2429 | 721 / 707 | 0 / 0 | 7 / 7 | 4 / 4 |
| C6 | undefined: both labels of a contradictory pair (B-R4) | 211 / 211 | 55 / 55 | 5 / 5 | 0 / 0 | 0 / 0 |
| C7 | undefined: conduction, axis, hypertrophy, ST, T and MI labels with a paced rhythm (B-R8) | 7436 / 7410 | 2192 / 2162 | 196 / 196 | 0 / 0 | 0 / 0 |
| C8 | undefined: PVC or AF positive measured only after the first 10 s (B-R1) | 60 / 59 | 16 / 15 | 0 / 0 | 8 / 8 | 11 / 11 |
| C9 | undefined: complete RBBB positive with QRS below 110 ms, Chapman and CPSC 2018 (B-R2) | 305 / 305 | 124 / 124 | 0 / 0 | 0 / 0 | 0 / 0 |
| C10 | undefined: sinus bradycardia negative below 60 per minute, PTB-XL and CPSC-Extra (B-R3) | 2307 / 2307 | 288 / 288 | 275 / 275 | 0 / 0 | 0 / 0 |
| C11 | undefined: rate label contradicted by the consensus heart rate (B-R6) | 983 / 191 | 157 / 61 | 118 / 6 | 14 / 14 | 5 / 5 |
| C12 | undefined: LVH negative that meets a voltage criterion (B-R10) | 3118 / 2830 | 765 / 688 | 116 / 111 | 405 / 404 | 388 / 388 |

- The amendment's train and validation counts hold before overlaps: C1 87, C2 15, C3 116, C4 47, and every
  label rule against Part B's `rules_candidate_counts.csv` [section `part-b-totals`].
- Readings, stated in the [v5 report](mlc03-dataset-v5-results-2026-11-11.md):
  - C10 follows the B-R3 code, a rate below 60 with no tolerance. The literal "beyond tolerance" would give
    884 labels instead of 2,595, all already in C11.
  - C11's negatives are not applied at SPH, because the sinus-evidence code does not read SPH's codes.
  - C3 and C4 skip SPH, as the amendment says.
  - C7 cannot act at SPH, because no SPH code marks a paced rhythm in the label code.
- At SPH the rules change little: no record is excluded, and 22 + 16 positives and 411 + 392 negatives become
  undefined, mostly LVH negatives by C12 [section `totals`].
- The largest losses [section `largest-changes`]:
  - negatives of sinus bradycardia in PTB-XL (13%, C10), of LVH (7-15% in most sources) and of complete and
    incomplete RBBB in Ningbo (6-7%);
  - positives of complete RBBB in CPSC 2018 (28% of train, 33% of validation) and Chapman (16%, 18%), by C9.
- PVC at CPSC 2018 validation falls from 113 to 99 positives, below the 100 that MLC-01 calls precise
  [section `eligibility`].

## Not adopted, and why

From Amendment 1, unchanged by the build:

- Lead masking in the model input (A-R4 to A-R7): it changes the input pipeline, not the dataset. C5 covers the
  labels.
- Limb-electrode reversal (A-R8, A-R9; the reversal part of B-R11): the passing check reused the first viewing,
  and `reversal_ra_ll` failed.
- Amplitude outliers (A-R10; the amplitude part of B-R11): the sample includes plausible large complexes.
- Noise flags (wander, mains, high frequency, steps, spikes): real, but a matter of degree. They stay in the v5
  manifest as `flag_<name>` for stratified reporting.
- Complete RBBB undefined in all of Chapman and CPSC 2018 (B-R2b): it removes 13,116 labels, most of them
  consistent with the measurement. C9 removes the measurably narrow positives.
- Rhythm negatives where no rhythm is stated (B-R9, B-R9b): 6,600 to 10,100 negatives on the evidence of 27
  sheets, and the measurements that could test it are unreliable.
- Measurement against label for complete RBBB, LAFB, left axis deviation and first-degree AV block (B-R6 for
  those conditions), and every rule that needs a model flag (B-R7).

These, and the source-coding differences no rule removes (left axis deviation, first-degree AV block, complete
against incomplete RBBB by QRS band), wait for clinical review.

## Part D: effect on validation

Recipes as in MLC-02:

- P0: one L2 logistic regression per condition on standardized, row-normalised pooled xECG. C is chosen per
  condition from {0.01, 0.1, 1, 10, 100} on validation. OpenBLAS ran its default kernel (SkylakeX, recorded in
  every receipt).
- H-All: the one-hidden-layer network on all features, learning rate 3e-4, weight decay 1e-4. It was trained
  with seeds 20261110-12 for 100 epochs, and each seed's epoch was chosen on the validation selection metric.
  H-All ran on GPU 0 under the shared lock, after a three-epoch profile (0.43 s per epoch, 0.89 GB peak).

Fitting used v5 train (56,133 records); scoring used v5 validation (15,097 records and their v5 labels). The
v4-trained models are MLC-02's stored validation predictions on the same records. Intervals are paired
patient-cluster bootstraps, 2,000 draws within each validation source (records in the Challenge sources), with
no redraw needed. The selection metric's interval combines the sources draw by draw. Code:
`ecg_experiment/mlc03_effect.py`, `scripts/experiments/run_mlc03_effect.py`, `tests/test_mlc03_effect.py`;
outputs in `outputs/mlc03_data_quality_20261111/effect/`.

Integrity, before any v5 fit [`effect/integrity.json`, `candidates/H-All/candidate.json`]:

- The stored predictions reproduce MLC-02's selection metrics exactly: P0 0.9642697112976489 and H-All
  0.9705630596869196.
- P0 refitted on v4 train at its chosen C reproduces its stored validation probabilities within 4.6e-15.
- H-All's first seed rerun on v4 gives bit-identical weights and the same best epoch (11).

Selection metric [section `effect`]:

| model | trained on | scored on | selection metric | 95% interval |
|---|---|---|---|---|
| P0 | v4 | v4 | 0.9643 | |
| P0 | v4 | v5 | 0.9690 | 0.9659-0.9719 |
| P0 | v5 | v5 | 0.9698 | 0.9664-0.9728 |
| P0 | v5 minus v4 | v5 | +0.0008 | -0.0003 to +0.0019 |
| H-All | v4 | v4 | 0.9706 | |
| H-All | v4 | v5 | 0.9751 | 0.9722-0.9778 |
| H-All | v5 | v5 | 0.9752 | 0.9726-0.9776 |
| H-All | v5 minus v4 | v5 | +0.0001 | -0.0006 to +0.0010 |

- Most of the rise from v4 to v5 comes from the scored labels, not from training. The cleaned validation set
  drops labels where a measurement disagrees, and the models disagree with many of the same labels. The rise
  shows the models agree better with v5's labels. It does not show that v5's labels are more often right;
  that needs a clinical reading.
- The same conditions enter each source's macro in v4 and v5 validation (61 source-condition cells).
- Settings chosen on v5: P0's C changed for 5 conditions. The new values are sinus tachycardia 100 (v4 10), AF
  or flutter 1 (10), PVC 10 (0.1), complete RBBB 1 (10) and incomplete RBBB 10 (100). H-All's best epochs were
  8, 5 and 9 (v4: 11, 10 and 14) [`candidates/*/candidate.json`].

Per condition: the within-source AUROC averaged over the sources where the condition enters. The table gives
v5-trained minus v4-trained on v5 validation, with intervals; p is Holm-adjusted over the 14 conditions [section
`effect-conditions`].

| condition | P0 difference (95% interval) | P0 Holm p | H-All difference (95% interval) | H-All Holm p |
|---|---|---|---|---|
| sinus tachycardia | +0.0003 (+0.0000, +0.0007) | 0.40 | +0.0003 (+0.0000, +0.0007) | 0.56 |
| sinus bradycardia | +0.0010 (+0.0004, +0.0016) | 0.06 | +0.0001 (-0.0006, +0.0006) | 1.00 |
| sinus arrhythmia | +0.0003 (-0.0001, +0.0006) | 1.00 | -0.0002 (-0.0014, +0.0009) | 1.00 |
| AF or flutter | +0.0007 (-0.0010, +0.0024) | 1.00 | +0.0006 (-0.0007, +0.0020) | 1.00 |
| PVC | +0.0010 (-0.0065, +0.0081) | 1.00 | +0.0001 (-0.0037, +0.0035) | 1.00 |
| first-degree AV block | +0.0000 (-0.0002, +0.0002) | 1.00 | -0.0009 (-0.0025, +0.0004) | 1.00 |
| complete RBBB | +0.0020 (-0.0000, +0.0053) | 0.58 | +0.0024 (+0.0001, +0.0072) | 0.44 |
| incomplete RBBB | +0.0001 (-0.0028, +0.0027) | 1.00 | -0.0027 (-0.0054, -0.0002) | 0.44 |
| LAFB | +0.0003 (-0.0001, +0.0008) | 1.00 | -0.0003 (-0.0015, +0.0009) | 1.00 |
| left axis deviation | +0.0002 (-0.0002, +0.0007) | 1.00 | -0.0004 (-0.0011, +0.0002) | 1.00 |
| LVH | +0.0029 (+0.0011, +0.0045) | 0.06 | +0.0034 (+0.0010, +0.0060) | 0.14 |
| ST depression | +0.0002 (-0.0002, +0.0005) | 1.00 | -0.0013 (-0.0029, +0.0002) | 1.00 |
| T-wave abnormality | +0.0001 (-0.0003, +0.0005) | 1.00 | -0.0008 (-0.0020, +0.0003) | 1.00 |
| any MI | +0.0013 (-0.0002, +0.0032) | 1.00 | -0.0015 (-0.0039, +0.0010) | 1.00 |

Condition by condition:

- No condition changes after the Holm adjustment, for either model.
- Unadjusted intervals entirely above zero:
  - P0: sinus tachycardia (lower end 0.00002), sinus bradycardia and LVH;
  - H-All: sinus tachycardia (lower end 0.000003), complete RBBB and LVH.
- One unadjusted interval lies entirely below zero: incomplete RBBB for H-All (-0.0027).
- Each of these sits where a rule removed many training labels:
  - LVH: C12, C5 and C7 removed LVH negatives, including those that meet a voltage criterion;
  - sinus bradycardia: C10 removed PTB-XL's negatives at 50-60 per minute;
  - complete RBBB: C9 removed the narrow positives of Chapman and CPSC 2018.
- These are secondary, unadjusted observations among 28 comparisons, not findings.
- Within single sources, 11 of P0's 61 source-condition intervals and 7 of H-All's exclude zero
  [section `effect-sources-nonzero`].
  - P0 has five cells with a within-source Holm p below 0.05: PTB-XL PVC, Ningbo sinus bradycardia and PVC,
    Georgia sinus bradycardia and LVH. All are gains of 0.0010 to 0.0064.
  - H-All has none.
  - The largest H-All change in one source is MI at CPSC-Extra, -0.0076 (-0.0132, -0.0017), unadjusted.
- Macro AUROC per source moves by -0.0005 to +0.0018 between the v4- and v5-trained models
  [section `effect-macro-source`].

Part D measures what cleaning changes on validation. It does not decide the rules, and it says nothing about
the test roles. Whether training on v5 changes performance at SPH or PTB-XL fold 10 needs its own protocol.

## Limits

- **No second human reading.** The Part A and Part B visual checks were read once, by a non-clinician. The 149
  "label error" sheets are measurement-based judgements, not diagnoses, and no cardiologist has reviewed any of
  them. Clinical validation is pending.
- **Measurement reliability.**
  - The label rules C9-C12 rest on automatic delineation. Its error was estimated in Part B, and it is
    reliable for heart rate and voltage.
  - Near the cut-offs the measurement itself is uncertain. C9 sits 10 ms below the 120 ms bound, and C10 uses
    the plain bound of 60 per minute.
  - Noisy records, which are the records of interest, are where delineation fails.
- **Rules developed without the test roles.** A defect that occurs only at fold 10 or SPH can be missed.
  The test-role measurements were computed with the same code but never validated there.
- **SPH storage unexamined.**
  - SPH stores float16 samples. Its 174 identity violations and 64 large values were not examined, which needs
    the user's approval.
  - C3 and C4 do not apply to SPH, C7 and C11's negatives cannot act there, and C12 rests on voltages measured
    on float16 data.
  - SPH is therefore cleaned less than the training sources.
- **Label definitions unchanged.** No rule changes what a label means. The source-coding differences that remain
  are still in v5 and wait for clinical review: PTB-XL and CPSC-Extra positives of sinus bradycardia still follow
  a cut-off near 50 per minute (C10 removes only their negatives between it and 60), and complete against
  incomplete RBBB, left axis deviation and first-degree AV block are coded differently by source.
- **Patient overlap.** The Challenge sources have no patient identifier, so Part D's validation shares some
  patients with training, as in MLC-02.
- **Part D** uses validation records chosen for both the C grid and the epoch. Its numbers are optimistic for
  both training sets alike.

## How to rerun

From the worktree, with the `data`, `outputs` and `.venv` links to the main checkout:

1. Parts A and B: see their reports.
2. Dataset v5: `PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m
   scripts.experiments.run_mlc03_build_v5 build --workers 14`. It refuses an existing output; `gate` runs the
   train and validation checks alone.
3. Part D, with `OPENBLAS_CORETYPE` unset: `PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=2 .venv/bin/python
   -m scripts.experiments.run_mlc03_effect <step>`, with steps `integrity`, `p0 --workers 6 --threads 2`,
   `profile` and `h-all` (both with `CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 --threads 4`), then
   `evaluate`. The last full Part D run took 74 s (integrity), 260 s (P0), 214 s (H-All) and 27 s (evaluate)
   [`effect/receipts/`].
4. Tables: `PYTHONPATH=. .venv/bin/python -m scripts.reports.mlc03_v5_tables [--section NAME]`.
5. Tests: `tests/test_mlc03_dataset_v5.py` (19) and `tests/test_mlc03_effect.py` (3).

Every receipt records the commit with no modified tracked file (`61b1131` for the build and the fits, `108192c`
for the evaluation, which differs only by the tables script). The uncorrected-memory-error counter was 1 before
and after the build and is 1 in every Part D receipt.

## Files

- `outputs/mlc01_condition_dataset_20261111_v5/`: the v5 manifest, the flow, support and integrity tables, the
  test-role measurements and the receipt ([v5 report](mlc03-dataset-v5-results-2026-11-11.md)).
- `outputs/mlc03_data_quality_20261111/effect/`:
  - `integrity.json`;
  - `candidates/P0/` and `candidates/H-All/`: v5-trained models with `model.json`, validation predictions,
    `candidate.json` and the profile;
  - `selection_metric.csv`, `difference_by_condition.csv`, `difference_by_source.csv`, `auroc_by_source.csv`,
    `effect.json` and `receipts/`.
- `outputs/mlc03_data_quality_20261111/signal/`, `labels/` and `review/`: Parts A and B.
