# MLC-01: dataset, test sets and evaluation rules for a condition-type classifier

Prospective, 10 November 2026. The user asked for a carefully built dataset and test set, and a
serious evaluation, for a classifier that names the type of cardiac condition on a 12-lead ECG.
This document is frozen before any dataset build, fit or score. MLC-01 builds the dataset and
computes no model score. MLC-02 (models) gets its own protocol, written after the build and
before any fit; the evaluation rules below bind it.

Inputs to this design, all metadata only: a survey of the earlier condition work (HEADS-01,
KEY-01 to KEY-04, Experiments 022, 024, 032, 033), a literature review, and a label inventory
over PTB-XL, SPH and the Challenge train and calibration groups. No Challenge test-group header
and no EchoNext record was opened.

## Decisions

1. **The target is multilabel.** Each condition is a separate yes/no output. Conditions co-occur
   on 13% (SPH) to 51% (Georgia) of records, and keeping only single-condition records would
   change the population (Strodthoff et al., 2021; Liu et al., 2022). "Multiclass" results the
   team needs (top-1, top-2 named condition) are derived from the same outputs.
2. **Label space B, 14 conditions, is primary**: sinus tachycardia, sinus bradycardia, sinus
   arrhythmia, atrial fibrillation or flutter, premature ventricular complexes, first-degree AV
   block (including prolonged PR), complete RBBB, incomplete RBBB, left anterior fascicular block,
   left axis deviation, left ventricular hypertrophy, ST depression, T-wave abnormality, any
   myocardial infarction. Rule used to choose them, applied before any score: codable in all
   three vocabularies (SCP, SNOMED CT, AHA), at least 100 positives in the training pool, in SPH
   and in PTB-XL folds 9-10 together, and not nested inside another chosen condition.
3. **Label space A** (MI, STTC, CD, HYP with the existing definitions) is secondary and links the
   work to KEY-04. **Label space C** (30 conditions) is exploratory: 16 of its 18 additions have
   fewer than 50 positives in some evaluation set.
4. **Three label states per record and condition: positive, negative, undefined.** Undefined
   entries are masked in the loss and left out of that condition's metrics. They are never
   counted as negatives (Suh et al., 2021).
5. **No untouched human-labelled patient-level test set exists.** SPH and PTB-XL fold 10 have
   been scored before, for the binary target, the four groups and six rhythm findings. We say so
   in every report. The protection is this protocol: a fixed model list, selection on validation
   data only, and one primary comparison. The Challenge test groups stay closed (see Roles).
6. The binary proxy label, the referral-budget operating point and the rule that left ventricular
   high voltage alone is undefined do not change. No age-subgroup analysis is added. The labels
   are annotation statements, not diagnoses; validation by a cardiologist is pending.

## Label rules

The mapping is code, `ecg_experiment/mlc_labels.py`, committed with unit tests before the build
and derived from the inventory table (309 condition-code rows). Rules:

- Positive: the record carries a positive code of the condition. For PTB-XL diagnostic statements
  any likelihood counts, as in the PTB-XL benchmark. Sensitivity analysis: statements with
  likelihood below 50 become undefined.
- Undefined when the source does not annotate the condition or its negatives are unreliable:
  - CPSC 2018 for every condition outside its nine classes; CPSC-Extra has no normal record.
  - Left ventricular hypertrophy on a record that carries only left ventricular high voltage.
  - ST depression at SPH when code 145 has no direction modifier, and in Georgia, which codes ST
    depression on 0.34% of records against about 4% elsewhere.
  - Left axis deviation in PTB-XL when `heart_axis` is LAD (only ALAD is positive).
  - Myocardial infarction in PTB-XL when the only evidence is a subendocardial injury statement.
  - Contradictory pairs on one record (complete with incomplete RBBB, LBBB with RBBB): both
    conditions undefined.
- SPH code 153 (ST-T change due to hypertrophy) counts for HYP in space A, not for STTC, as PTB-XL
  codes the same pattern as LVH plus an ischemic statement. In space B it sets no condition.
- Unstated-completeness RBBB codes follow the official Challenge equivalence (complete RBBB).
- Negative otherwise.
- "Normal" is not a class. A record with no positive condition is "none of the 14", which is not
  the same as a normal ECG. Per-source normal definitions are kept as a column for audits.

Coding practice differs between sources (sinus bradycardia 37% in Chapman and Ningbo against 2.9%
in PTB-XL; MI 24% in PTB-XL against 1% at SPH). We do not correct for this. We report every
metric within a source and never on sources pooled together.

## Records

- Sources: PTB-XL 1.0.3 at 500 Hz, the Challenge train and calibration groups of
  `docs/challenge-splits-v1.md` (recovered split table, SHA-256 checked), and SPH. MIMIC-IV-ECG,
  CODE-15% and EchoNext are not used: their labels are machine-made or not ECG statements.
- Inclusion: readable file, twelve leads identified by name, 500 Hz, at least 10 s. Every model
  sees the first 10 s, which is what the screening device records. Shorter records are excluded
  and counted. Sensitivity analysis for longer records: mean score over consecutive 10 s windows.
- No record is excluded for signal quality, because that would make the test sets easier. Flat
  or missing leads are flagged in the manifest and reported as a subgroup. Ningbo JS13118 is
  excluded as corrupted upstream.
- Duplicates: exact waveform hashes are compared across all opened sets before roles are
  assigned. A copy in training or validation of any test record is removed from training or
  validation. Among duplicates inside training, conflicting label sets drop every copy; otherwise
  one copy is kept. The 71 Chapman-Ningbo and 155 CPSC-Extra copies are handled by this rule.
- Preprocessing is not part of the dataset. The manifest points at raw files. Each model applies
  its own documented input pipeline; a pretrained encoder gets exactly the pipeline it was
  pretrained with.

## Roles

| Role | Records | Use |
|---|---|---|
| Train | PTB-XL folds 1-8; Challenge train groups | Fitting |
| Validation | PTB-XL fold 9; Challenge calibration groups | Model choice, early stopping, thresholds, calibration |
| Test, same sources | PTB-XL fold 10 | Scored once per frozen model |
| Test, external hospital | SPH evaluation half (patient hash of `docs/raw-role-bootstrap-protocol-2026-10-28.md`, applied to all SPH records) | Primary endpoint |
| SPH calibration half | The other SPH patients | One named secondary analysis only: refit of thresholds and calibration on local data |
| Closed | Challenge test groups; EchoNext test | Not opened |

PTB-XL and SPH roles are patient-disjoint. The Challenge sources have no patient identifier, so
their train and calibration groups can share patients; this is a stated limit of the validation
data, not of the two test sets. SPH never enters fitting, model choice or threshold setting.

Source shift during development is measured without the test sets, by leave-one-family-out
cross-validation over train and validation: PTB-XL, Chapman with Ningbo, Georgia, CPSC with
CPSC-Extra (Leinonen et al., 2024).

The Challenge test groups (13,172 records) are the only human-read labels no model of ours has
been scored on. Opening them is a separate, one-time step that needs the user's explicit
approval after the pipeline is frozen. Their limits are already known: record-level split, and
two encoders were pretrained on Chapman and Ningbo waveforms.

## Evaluation rules (binding for MLC-02)

- **Primary endpoint:** macro AUROC over the 14 conditions on the SPH evaluation half. A
  condition enters a set's macro only if it has at least 20 positives and 20 negatives there.
- **Primary comparison:** the model selected on validation against the predecessor baseline
  (independent logistic regressions on pooled xECG embeddings, as in KEY-04, refit on this
  dataset). The selected model is better if the lower end of the paired 95% interval of the
  macro AUROC difference is above zero. This is the only confirmatory test.
- **Selection metric:** the mean, over validation sources, of the within-source macro AUROC.
  Nothing is selected, tuned or thresholded on a test set, including epochs and seeds.
- **Intervals:** 2,000 bootstrap resamples of patients (records where no patient identifier
  exists), the same resamples for every model, redrawn when a condition loses all positives.
- **Per condition, per test set:** AUROC, average precision with the prevalence beside it,
  sensitivity and specificity at the threshold that gave 90% specificity on validation,
  calibration slope and intercept, Brier score. Conditions with fewer than 100 positives in a
  test set are marked imprecise. Per-condition model comparisons are Holm-adjusted over 14.
- **Derived multiclass view:** top-1 and top-2 accuracy of the named condition on records with
  exactly one positive condition, and exact-set accuracy on all records, at validation thresholds.
- **Audits:** source identification from the model's features; results on records with one
  condition against several; by sex; with flat leads; both sensitivity analyses above; label
  space A beside KEY-04's numbers.
- **Integrity check before new work:** reproduce KEY-04's independent-logistic macro AUROC at SPH
  (0.9608) from the cached features.
- **Pretraining overlap** of each encoder with every evaluation source is written into the MLC-02
  protocol before any score.
- Training runs use three seeds; all three are reported. A real GPU profile gates full training.
- Any departure from this document is logged in the results report with its date and reason.

## MLC-01 deliverables

`ecg_experiment/mlc_labels.py` and tests; a build script; a manifest under
`outputs/mlc01_condition_dataset_20261110/` (one row per record: source, family, role, patient
or record group, raw path, duration, lead flags, waveform hash, the 14 label states, the space A
and space C states, the normal column); a receipt with input hashes; and a results report with
the exclusion flow, the counts per condition, role and source, the duplicate accounting and the
co-occurrence table. The build reproduces the inventory's counts or explains each difference.

## References

Leinonen et al. (2024), Comput Biol Med 183:109271. Liu et al. (2022), Sci Data 9:272.
Reyna et al. (2022), Physiol Meas 43(8). Strodthoff et al. (2021), IEEE JBHI 25(5):1519-1528.
Suh et al. (2021), Computing in Cardiology, doi:10.22489/CinC.2021.019. Wagner et al. (2020),
Sci Data 7:154.

## Amendment 1 (10 November 2026, after the first build, before any fit or score)

The first build raised label questions the rules above did not settle. The rulings follow one
principle: a statement that does not say whether the condition is present makes the label
undefined, not negative. No model had been fitted and no score computed when these were written.

1. Premature ventricular complexes: a record with ventricular tachycardia and no PVC code is
   undefined.
2. ST depression: a nonspecific or ischemic ST-T statement with no ST-depression code is
   undefined (PTB-XL `NST_` and the ischemic statements, and their SNOMED counterparts).
3. T-wave abnormality: an ischemic or nonspecific ST-T statement with no T-wave code is undefined.
4. Left axis deviation in PTB-XL: an empty `heart_axis` and the value `AXL` are undefined.
5. Left ventricular hypertrophy in PTB-XL: `HVOLT` alone is undefined, as for the other sources.
6. A source with fewer than five positives of a condition among its opened records does not
   annotate that condition: every record of that source is undefined for it. This covers left
   axis deviation and left anterior fascicular block in CPSC-Extra and myocardial infarction in
   Georgia. The check is on label counts only.
7. LBBB with incomplete RBBB joins the contradictory pairs.
8. Space A: SPH code 153 sets HYP positive and leaves STTC undefined unless another ST-T code is
   present. CPSC 2018 records are undefined for STTC and CD unless positive, since that source
   annotates only some of the conditions in those groups.
9. The "155 CPSC-Extra copies" quoted above came from an earlier audit with another scope. The
   build's own accounting (274 CPSC 2018 and CPSC-Extra groups) replaces it.

The build is rerun with these rules. The results report states the counts before and after.

## Amendment 2 (10 November 2026, after the second build, before any fit or score)

1. ST depression in PTB-XL: a record with a subendocardial injury (`INJ*`), digitalis (`DIG`),
   aneurysm (`ANEUR`) or electrolyte (`EL`) statement and no ST-depression code is undefined.
   These statements describe ST changes without giving a direction we can map.
2. Kept as built, and stated as limits:
   - SPH code 145 (ST deviation) without code 146 or 147 stays negative for T-wave abnormality:
     SPH has a separate code for ST deviation with T-wave change, so 145 alone says the T wave
     was not flagged. The SPH labels do not change in either amendment.
   - CPSC 2018 stays defined for atrial fibrillation or flutter although flutter is not one of
     its nine classes.
   - Space C keeps its remaining ambiguous negatives (ST elevation, right axis deviation). It is
     exploratory, and its report says so.
3. Ruling 6 of Amendment 1 applies to every label space, as built.
4. The dataset is final after this rebuild (`outputs/mlc01_condition_dataset_20261110_v3/`). Any
   later label question is logged as a limitation, not applied, so that the labels cannot be
   adjusted after models have been scored. The first two builds stay on disk as the record of
   what changed; their receipts pin earlier versions of the label code, which remain in Git.

## Amendment 3 (10 November 2026, after an independent audit, before any fit or score)

A second agent, which did not build the dataset, re-derived every label from the raw metadata
and the text above, re-hashed every waveform, and compared labels with simple measurements
(heart rate, QRS axis, QRS width). It agreed with the build on 1,402,829 of 1,403,094 label
cells, found no closed record, no patient or waveform shared between roles, and no error in the
report's numbers. It also found label problems that the rules above did not foresee. Ruling 4 of
Amendment 2 closed the labels so that they could not be tuned against model scores. No model has
been fitted or scored, so that purpose is intact, and we apply the audit's findings once.

Applied (each makes an unstated label undefined; none creates a positive):

1. Left axis deviation: a record with left anterior fascicular block and no left-axis code is
   undefined in every source. The block implies a left axis, and SPH, Ningbo and Georgia do not
   code the axis separately: at the SPH evaluation half 76 such records were negatives beside 68
   positives.
2. First-degree AV block: SNOMED 233917008 (AV block, degree unstated) without a first-degree or
   prolonged-PR code is undefined (265 records, none in a test set).
3. T-wave abnormality in PTB-XL: an `INJ*`, `DIG`, `ANEUR` or `EL` statement with no T-wave
   code is undefined, as Amendment 2 did for ST depression.
4. Myocardial infarction: an abnormal Q-wave or poor R-wave progression statement with no
   infarction code is undefined (PTB-XL `QWAVE`, and the SNOMED codes for abnormal Q wave and
   poor R-wave progression).
5. SPH code 153 without a specific ST or T code leaves ST depression and T-wave abnormality
   undefined, matching ruling 8 of Amendment 1.
6. The likelihood sensitivity flag ignores diagnostic statements whose likelihood is 0, which
   PTB-XL uses for "unknown".

Not changed, and reported as limitations with the audit's numbers:

- SPH code 146 (ST deviation with T-wave change) stays T-wave positive; 521 of the 1,633 T-wave
  positives at the SPH evaluation half rest on it alone. The SNOMED sources code the comparable
  free-text statement as a nonspecific ST-T change, which is undefined here.
- PTB-XL codes sinus bradycardia at about 50 beats per minute and the other sources at about
  60: 13.4% of PTB-XL bradycardia negatives are sinus-rhythm records below 60. We do not relabel
  from measurements.
- Complete RBBB in CPSC 2018 and Chapman probably includes incomplete blocks (62% and 76% of
  positives have a measured QRS of at least 120 ms, against 86-96% elsewhere).
- The Challenge train and validation groups share patients: the audit estimates 6-13% of
  validation records in Ningbo, Georgia and the Chapman-Ningbo family have a same-patient
  training record. This affects model choice, early stopping and thresholds, not the test sets.
  PTB-XL fold 9 has no such overlap.

The build is rerun as `outputs/mlc01_condition_dataset_20261110_v4/`, which is final. The audit's
scripts and tables are copied to `outputs/mlc01_condition_dataset_20261110_v4/audit/`. From the
first model fit onward no label rule changes.
