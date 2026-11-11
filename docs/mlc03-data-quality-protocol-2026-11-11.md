# MLC-03: data quality and cleaning for the 14-condition classifier

Prospective, 11 November 2026. Frozen before any quality measurement of this experiment. It runs
on the MLC-01 dataset v4 (`outputs/mlc01_condition_dataset_20261110_v4/`) and follows MLC-02
(`docs/mlc02-condition-models-results-2026-11-10.md`). The roles, label states and evaluation
rules of MLC-01 bind this document.

## Why

MLC-02 found no demonstrated gain over the reference model at the external hospital, validation
thresholds that do not hold there, and conditions whose labels differ between sources (incomplete
RBBB, left axis deviation, left anterior fascicular block). The models are no longer the weakest
part; the data are. The user's instruction of 11 November 2026 is to make the investigation of
cleaning and quality the most important part, and that every available dataset may be used.

## Questions

1. Signal: which records carry a waveform defect that a model should not learn from or be scored
   on, how many are there per source and role, and which defects are specific to one source?
2. Labels: where does a label disagree with what can be measured on the same ECG, where do
   sources code the same condition differently, and how many labels are probably wrong?
3. Cleaning: which rules, each justified by evidence that does not use a model's test score,
   define a cleaned dataset (v5)?
4. Effect (secondary): what does training and validating on v5 change?

## What may be read

- Rules, thresholds and every choice are developed on train and validation records only.
- PTB-XL fold 10 and both SPH halves: once a rule is frozen it is applied to them mechanically
  and only counts are reported (records flagged, by flag and source). Their labels and the stored
  MLC-02 test predictions are not used to choose or tune anything. No new model is scored on them
  in this experiment.
- Challenge test groups: the user allowed their use on 11 November 2026. This experiment does
  not open them. They are kept for one final test of the frozen cleaned pipeline, in a later
  experiment with its own protocol; the frozen rules of this experiment will be applied to them
  there.
- EchoNext and MIMIC-IV-ECG are not part of this experiment: they carry no human-read labels for
  the 14 conditions.

## Part A: signal quality (every train and validation record)

Computed on the cached first 10 s (500 Hz, mV) and, where the recording is longer or shorter, on
the raw file. One row per record, one column per measure, in
`outputs/mlc03_data_quality_20261111/signal/`:

1. Integrity: non-finite samples, zero-padded or constant head or tail, recording shorter than
   10 s, sampling rate and resolution as declared, ADC saturation or clipping (runs at the extreme
   value), quantisation step.
2. Lead content: flat or near-flat lead, duplicated leads, a lead that is an exact copy or a
   sign-flip of another.
3. Lead consistency: residual of III = II - I, aVR = -(I + II)/2, aVL = I - II/2 and
   aVF = II - I/2. A large residual means derived leads were not derived, were swapped, or are
   mislabelled. Limb-lead reversal patterns (for example a positive aVR complex, or lead I
   inverted with normal precordial progression) are scored by simple, stated rules.
4. Amplitude and units: per-lead robust amplitude, QRS amplitude, records and sources whose
   scale differs by a constant factor from the rest (unit or gain error).
5. Noise: baseline wander (power below 0.5 Hz), mains interference (power at 50 and 60 Hz
   against neighbours), high-frequency noise (above 40 Hz), abrupt steps and spikes, electrode
   contact loss.
6. Beats: R-peak detection on at least two leads, heart rate, agreement between detectors and
   between leads; a record on which no rhythm can be measured is flagged.
7. Preprocessing already in the pipeline: what each encoder's input pipeline and the 100 Hz
   path do to these defects (filters, resampling, normalisation), checked on real flagged records.
8. Near-duplicates beyond exact hashes: correlation of waveforms within and across sources and
   roles, extending `near_duplicates_*.csv` of MLC-01; any pair across roles is reported.

Each flag has a written definition and a threshold. A threshold is set from the distribution on
train and validation, with a reason, and checked by eye: for every flag, a fixed random sample
of 24 flagged and 24 unflagged records is drawn (seed 20261111) and plotted to a contact sheet
under `outputs/mlc03_data_quality_20261111/review/`. The report states, per flag, how many of
the 24 flagged plots show the defect.

## Part B: label quality (train and validation labels)

1. Label against measurement, per condition and source, using the 130 measurements and new
   ones where needed: heart rate against sinus tachycardia and bradycardia (and each source's
   implied cut-off); RR irregularity and P-wave evidence against AF or flutter and sinus
   arrhythmia; PR interval against first-degree AV block; QRS duration and V1 morphology against
   complete and incomplete RBBB; frontal axis against left axis deviation and LAFB; voltage
   criteria against LVH; ST level and T-wave polarity against ST depression and T-wave
   abnormality. Reported as distributions and as counts of labels outside stated physiological
   bounds, with the measurement's own error estimated on records where two methods agree.
2. Source coding: for each condition, which sources code it, with which codes, and at what
   prevalence given the same measurement (for example prevalence of the bradycardia label among
   records with a measured rate below 50). A condition that a source does not code separately
   must be undefined there; any remaining case is listed.
3. Contradictions inside one record: label pairs that cannot both hold, and positives whose
   record fails a Part A flag that makes the condition unreadable.
4. Probable label errors from models, on train and validation only: out-of-fold probabilities
   of the reference model (P0) by five-fold patient-grouped cross-validation over train plus
   validation, never a model that saw the record. Per condition and source: the estimated number
   of wrong labels by the confident-learning count, and the 50 highest-ranked suspected false
   positives and false negatives, each with its measurements, plotted to review sheets. These
   lists are evidence for a reader; a model's disagreement alone never changes a label.
5. Undefined entries: how many, why, and whether the rule that made them undefined holds on the
   measurements.

## Part C: cleaning rules and dataset v5

Written as an amendment to this protocol after Parts A and B, before any model is refitted:

- A rule may exclude a record or set a label to undefined. It never creates or flips a
  positive, and no label definition changes (the project waits for clinical review for that).
- Each rule cites its evidence from Parts A or B and the count it affects per source and role.
- A rule is adopted on quality evidence, not because a model scores higher with it.
- Rules about unreadable signals apply to every role. Rules that set labels undefined apply to
  every role with the same code logic, as MLC-01's amendments did.
- The rebuild is `outputs/mlc01_condition_dataset_20261111_v5/`, with the flow from v4 and the
  same integrity checks. v4 stays as built.

## Part D: effect (secondary, validation only)

P0 and H-All are refitted on v5 train with their MLC-02 recipes and scored on v5 validation,
beside the same models trained on v4 and scored on the v5 validation records: selection metric
and per-condition AUROC with paired bootstrap intervals. This measures what cleaning changes; it
does not decide the rules. No test role is scored.

## Reading the result

- A flag is reported as reliable only if at least 20 of its 24 sampled flagged plots show the
  defect.
- A condition is reported as inconsistently labelled between sources if its prevalence at a
  matched measurement differs between two sources by a factor of two or more with
  non-overlapping 95% intervals.
- Estimated label-error counts are estimates from a model and are reported with that caveat;
  nothing in this experiment is a clinical validation, which is pending.

## Limits known in advance

Measurements come from automatic delineation and fail on noisy records, which are the records of
interest. The Challenge sources have no patient identifier. No second human reading exists.
Quality rules were not developed on the test roles, so a defect that occurs only there can be
missed.

## Resources and deliverables

CPU work with limited threads; GPU only for Part D, with the shared lock. New modules and
scripts only, with tests. `docs/mlc03-signal-quality-results-2026-11-11.md`,
`docs/mlc03-label-quality-results-2026-11-11.md`, the amendment with the rules, the v5 build and
its report, `docs/mlc03-data-quality-results-2026-11-11.md` written from the outputs, queue and
backlog updates.

## Addendum 1 (11 November 2026, before any label-quality result)

Two public PhysioNet references are added to estimate the error of the measurements used in
Part B, beside the two-method agreement named there:

1. PTB-XL+ 1.0.1 feature tables (measurements of two commercial algorithms on PTB-XL). Only
   rows of PTB-XL folds 1 to 9 are read; fold 10 rows are not.
2. LUDB 1.0.1 (200 records with manual wave onsets and offsets).

Neither supplies a label for the 14 conditions, and neither enters training. A measurement whose
error against these references is too large to judge a label is reported as such.
