# KEY-05: condition keys on xECG embeddings for the 14 conditions

Prospective, 11 November 2026. Frozen before any analysis of this experiment. Requested by the
user on 11 November 2026: an in-depth study of vector keys per cardiopathy on the encoder
behind the project's best binary detector (xECG).

It follows KEY-01 to KEY-04 (`docs/condition-keys-results-2026-11-03.md`,
`docs/long-section-keys-results-2026-11-06.md`, `docs/reference-ecg-audit-results-2026-11-06.md`;
whole-record key averages reached macro AUROC 0.896 against 0.958 for a joint classifier on four
annotation groups at SPH), MLC-02 (logistic regressions on pooled xECG, P0, reach 0.9643 on
validation for 14 conditions) and MLC-03 (cleaned dataset v5).

A key is one vector per condition, built from reference ECGs without iterative training; a new
ECG is scored by its similarity to the key. The labels are ECG-annotation proxies, not
diagnoses, and no clinical review has taken place.

## Questions

1. How close does one key per condition come to a trained classifier on the same embedding, for
   each of the 14 conditions, and which construction of the key matters?
2. Where a key falls short, why: covariance of the embedding, co-occurring conditions, the
   source, several forms of one condition, or too few reference ECGs?
3. How many reference ECGs does a key need, and how stable is it under resampling of patients
   and under leaving a source family out?
4. Does each key follow the measurement that defines its condition?

## Data

- MLC-01 dataset v5 (`outputs/mlc01_condition_dataset_20261111_v5/`), included records, defined
  labels only. Train builds keys. Validation chooses between constructions and gives every
  descriptive number. The test roles (PTB-XL fold 10, SPH evaluation half) are read once, at
  the end, for the frozen comparison only. The SPH calibration half is not used. No Challenge
  test group, EchoNext or MIMIC record is opened.
- Features: pooled xECG embeddings (1,024 values,
  `outputs/mlc02_condition_models_20261110/features/xecg.npy`), standardized on train.
- Reference classifier: P0 refitted on v5 by MLC-03 Part D
  (`outputs/mlc03_data_quality_20261111/effect/`), one logistic regression per condition on the
  same features.
- Integrity, before any analysis: the feature rows match the v5 manifest keys; the stored
  validation metric of the v5-trained P0 on v5 labels is reproduced exactly from its stored
  predictions; v5 label counts per role equal those of the v5 receipt.

## Key constructions (all fitted on train only)

| Id | Key | Score |
|---|---|---|
| K1 | Mean of the positives | cosine |
| K2 | Mean of the positives minus mean of the negatives | dot product |
| K3 | Mean of the positives minus mean of records with no positive label | dot product |
| K4 | K2 after whitening with the pooled within-class covariance (Ledoit-Wolf shrinkage); the linear discriminant direction | dot product |
| K5 | K2 computed inside each train source and averaged with equal source weights | dot product |
| K6 | K4 with the source means removed first (source-centred features) | dot product |
| K7 | m keys per condition: k-means centres of the positives, m in {2, 4, 8}, each minus the negative mean, whitened as K4 | maximum over the m keys |
| N1 | Not a key, a bound for reference-set methods: share of positives among the 20 nearest train neighbours (cosine) | share |
| P0 | Reference classifier | probability |

K1 to K6 are single keys. K7 and N1 show what a single direction cannot hold. No construction
has a parameter tuned on validation other than the choice between constructions.

## Analyses on validation

1. Per construction: AUROC and average precision per condition, macro AUROC over the 14
   conditions with a patient bootstrap interval (2,000, `ecg_experiment/intervals.py`), and the
   paired difference from P0. Per source as well as pooled.
2. Anatomy of the gap to P0. The steps K1, K2, K4, K6, K7, P0 are read as a ladder; each
   step's paired gain is reported per condition. Cosine between each key and P0's weight vector.
3. Co-occurring and related conditions. For each ordered pair (A, B): the AUROC of key A
   among records negative for A, with B-positives against B-negatives (0.5 means key A ignores
   B). Reported for all pairs with at least 100 such positives, and discussed for the related
   pairs: complete and incomplete RBBB; sinus tachycardia and AF or flutter; LAFB and left axis
   deviation; LVH and ST or T changes; the MI conditions among themselves. Then the same after
   projecting the other 13 keys out of key A.
4. Source. Cosine of each key with the directions that separate the sources (multinomial
   logistic regression on train); AUROC within source against pooled.
5. Number of references. Keys K2 and K4 built from n positives and n negatives, n in {5, 10,
   20, 50, 100, 200, 500, all}, 50 random draws each: median and 5-95% range of validation
   AUROC per condition. For K4 the covariance is estimated once from all train records
   (unlabelled use) and also from the drawn records only.
6. Stability. Cosine between keys built on 200 patient bootstrap resamples of train. Leave one
   source family out: key built without the family, AUROC on that family's validation records,
   against the key built with it.
7. Agreement with the defining measurement, within source, on records with a reliable
   measurement (MLC-03 Part B): Spearman correlation and the curve of key score against the
   measurement. Heart rate for sinus tachycardia and bradycardia; QRS duration for the bundle
   branch blocks; frontal axis for left axis deviation and LAFB; Sokolow-Lyon and Cornell
   voltage for LVH; ST level and T amplitude for the ST and T conditions; Q duration for the MI
   conditions. PR interval and RR irregularity are reported but marked unreliable.
8. Review sheets. For each condition, the 12 highest-scoring validation negatives and the 12
   lowest-scoring positives of the selected construction are plotted under
   `outputs/key05_condition_keys_20261111/review/`; the analyst records what is measurable in
   them. Non-clinical reading.

Token-level keys (keys on the encoder's time tokens instead of the pooled vector) are out of
scope here: KEY-01 found them worse on four groups, and they need new feature extraction. They
go to the backlog with what this experiment learns.

## Selection and the one test read

After the validation analyses, a selection record is committed: the single-key construction
(K1 to K6) with the highest validation macro AUROC, and, separately, the best of K7. Then, once:

- the selected single key, the selected K7 and the v5-trained P0 are scored on PTB-XL fold 10
  and the SPH evaluation half, v5 labels, conditions with at least 100 positives and 100
  negatives in the role;
- primary comparison: selected single key minus P0, macro AUROC at the SPH evaluation half,
  paired patient bootstrap (2,000), 95% interval.

Decision rule: the single key is called equivalent to the classifier for ranking if the
interval's lower bound is above -0.010; clearly worse if its upper bound is below -0.010;
otherwise undecided. The same reading is given per condition without a decision. Nothing is
promoted to the screening pipeline by this experiment; P0 stays the reference whatever the
result.

## Limits known in advance

Selection and description share the validation set, so validation numbers of the selected
construction are optimistic; the test read is the guard. SPH has been read by earlier
experiments and is development data. SPH labels are cleaned less than the other sources
(MLC-03 Amendment 2). The measurements are those of MLC-03, some unreliable. Keys depend on
which records the sources chose to label.

## Resources and deliverables

CPU only, at most 8 threads. New modules (`ecg_experiment/key05_*.py`), scripts and tests on
synthetic data; no pinned file is edited. Outputs in `outputs/key05_condition_keys_20261111/`.
Report `docs/key05-condition-keys-results-2026-11-11.md` written from the outputs, with the
construction table, the ladder, the pair table, the reference-count curves, stability, the
measurement agreement, the review reading, departures and doubts. Queue and backlog updates.
