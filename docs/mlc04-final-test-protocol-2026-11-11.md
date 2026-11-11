# MLC-04: one final test of the 14-condition classifier on the Challenge test groups

Prospective, 11 November 2026. Frozen before any Challenge test-group record of this experiment
is opened. Approved by the user on 11 November 2026 ("go ahead with the challenge test"), after
being told that the Challenge test groups are the only human-read labels no model of ours has
been scored on, and that this reading can be done once.

It follows MLC-01 (dataset and roles), MLC-02 (models; P0 stays the reference), MLC-03 (cleaning
rules C1-C12, dataset v5) and KEY-05 (condition keys).

## What is frozen

Nothing is fitted, tuned or chosen in this experiment. Everything below already exists:

- the label code of MLC-01 (Amendment 3 labels) and the cleaning rules C1-C12 of MLC-03 with
  Amendments 1 and 2, as code;
- P0 trained on v5 (MLC-03 Part D): the classifier under test;
- P0 trained on v4 (MLC-02), H-All trained on v5 and on v4 (MLC-03 Part D, MLC-02), and the
  KEY-05 key K6: comparisons;
- thresholds: for each condition, the threshold of the v5-trained P0 fitted on v5 validation by
  the MLC-02 threshold rule;
- the metric: the MLC-02 selection metric, the macro AUROC within each source (conditions with
  at least 100 positives and 100 negatives in that source's test group), averaged over the
  sources.

The labels are ECG-annotation proxies, not diagnoses. No clinical review has taken place.

## Data

- The Challenge test groups of `docs/challenge-splits-v1.md`: 13,172 records of Ningbo, Chapman,
  Georgia, CPSC 2018 and CPSC-Extra. They share hospitals with the training data, so this is a
  test on new records of known hospitals, not on a new hospital; SPH remains the only external
  hospital. The sources have no patient identifier: the unit is the record, about 6-13% of
  patients are estimated to be shared with train and validation (MLC-01), and intervals are
  record-level and somewhat too narrow.
- EchoNext test and MIMIC are not opened. PTB-XL fold 10 and SPH are not rescored.
- The binary screening pipeline (`docs/pipeline-v2.md`) is not scored here: no binary label and
  no screening score is computed on the test groups, so its own one-time final test is not
  spent by this experiment. The waveforms and the 14-condition labels are read.

## Steps, in this order

1. Code and dry run without the test groups. The whole path (labels, Part A flags, Part B
   measurements, rules C1-C12, features, predictions, metrics) is written as one runner that
   takes a role as argument. It is run on the Challenge validation (calibration) groups and
   must reproduce: the v5 manifest rows of those records cell by cell; the stored features of
   those records (exactly if stored features are reused, or within 1e-4 if re-extracted); the
   stored validation predictions of every frozen model; and the stored validation metric of the
   v5-trained P0. The loaders refuse the test role unless a freeze file exists.
2. Freeze. The code commit, the SHA-256 of every model, threshold and feature file, and the dry
   run result are written to `docs/mlc04-freeze-record-2026-11-11.md` and committed.
3. The one test read. Labels are built for the test groups with the same code; the rules are
   applied; features are taken from an existing extraction made by the same code as the
   training features if one exists (verified on validation in step 1), otherwise extracted now;
   the frozen models score the records. The runner writes a marker and refuses a second run.
4. Report.

## Primary result

The selection metric of the v5-trained P0 on the test groups with v5-rule labels, with a record
bootstrap interval (2,000, `ecg_experiment/intervals.py`), beside its validation value (0.9698).
Reading: the validation estimate is called confirmed if the test interval's lower bound is
above the validation value minus 0.010; not confirmed if the upper bound is below it;
otherwise undecided. This measures how much the validation numbers of this project were
flattered by model choice and rule development on validation.

## Secondary, paired on the same records, Holm over the three

1. Cleaning: v5-trained minus v4-trained P0, on v5-rule labels and, separately, on v4 labels
   (no rule applied). Cleaning is said to help only if the first interval is above zero and the
   second is not below zero.
2. Larger model: H-All (v5) minus P0 (v5).
3. Key: K6 minus P0 (v5), read with the KEY-05 margin of -0.010.

## Descriptive

Per condition and per source: AUROC, average precision, support. Sensitivity and specificity at
the frozen thresholds, with intervals, and the same on validation. Calibration (Brier score and
reliability) of the v5-trained P0. Counts of each rule C1-C12 on the test groups, beside
validation. The metric with the 20-positive entry rule of validation, for comparability. No
review sheets and no reading of individual test records.

## What this result may and may not be used for

It is reported once and no model, rule, threshold or label definition is changed in response
to it inside this project line; any later change makes a new pipeline that has no untouched
test. After this experiment the Challenge test groups count as read for the 14-condition
task.

## Limits known in advance

Same hospitals as training; shared patients across the split; record-level intervals; labels
cleaned by rules developed on train and validation of the same sources; coding differences
between sources remain; no clinical review.

## Resources and deliverables

One GPU at most, with the shared lock, only if features must be extracted; otherwise CPU, at
most 8 threads. New modules (`ecg_experiment/mlc04_*.py`), scripts and synthetic-data tests; no
pinned file edited. Outputs in `outputs/mlc04_final_test_20261111/`. Report
`docs/mlc04-final-test-results-2026-11-11.md` written from the outputs. Queue and backlog
updates, and a line in `AGENTS.md`'s closed-data paragraph is proposed to the user, not edited.
