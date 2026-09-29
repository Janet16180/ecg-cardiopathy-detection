# Experiment 027b: calibrating the screening threshold on several hospitals

**Frozen 29 September 2026, before any Challenge score is computed and before any new SPH operating point is
read.** This is the rerun of [Experiment 027](experiment-027-calibrated-threshold.md) that the Ningbo data
made possible (backlog item `multisource_calibration`). The user authorized the reruns of the experiments
Ningbo could affect. Only aggregate 027 and 022 results and metadata (record counts, labels, feature
receipts) were read before this freeze.

## Question

027 fitted Platt scaling and the at-least-95%-sensitivity threshold on PTB-XL calibration patients, from the
same hospital as training. At SPH the threshold missed 95% (JEPA 0.916, xECG 0.922; CPC overshot at 0.971
with specificity 0.303) and the probabilities ran far too high (calibration intercepts −1.1 to −1.9).

Does fitting the same Platt mapping and threshold on PTB-XL plus the calibration groups of four more
hospital collections (Chapman/Ningbo, Georgia, CPSC) make the operating point transfer to SPH better than
PTB-XL calibration alone?

## What stays fixed

- **Readouts.** `cpc_standard`, `jepa_standard` and `xecg_standard`, fitted on PTB-XL training only, exactly
  as in 027. The run refits them with 027's `reproduce_heads`, which requires 022's development
  probabilities to 1e-9. Nothing in a head changes; only the calibration data change.
- **Method.** Platt scaling on the head's logit (`screening_threshold.fit_platt`, `C=1e6`) and the highest
  calibrated probability at which at least 95% of the calibration positives are referred
  (`screening_threshold.screening_threshold`). An ECG is referred when its calibrated probability is at or
  above the threshold.
- **Label.** The standard label: the PTB-XL superclass label of Experiment 020 for PTB-XL, the primary
  label of Experiment 022 for SPH, and the primary label of the
  [Challenge mapping](challenge-label-mapping.md) for the Challenge sources (positive for any MI, STTC, CD
  or HYP code; negative for sinus rhythm alone). The secondary labels are not used.
- **SPH.** The 21,008 `use_evaluation` ECGs of `data/processed/sph_clean_v1` with a primary label (20,364
  patients, 7,190 positive) and the saved 022 probabilities, loaded and checked with 027's `sph_rows`.

## Calibration data

- **PTB-XL.** The 564 calibration ECGs of 027 (504 patients, 348 positive), with 027's saved calibration
  features (`outputs/experiment027_calibrated_threshold_v1/calibration_features.npz`, hash checked against
  027's `result.json`; its ECG IDs must equal 027's calibration rows).
- **Challenge.** The `calibration` groups of the frozen [Challenge split](challenge-splits-v1.md)
  (`data/processed/challenge_splits_v1`, `rows.csv` hash checked against its metadata), `evaluable` rows
  only (a defined primary label, duplicates resolved), that have frozen-encoder features in
  `outputs/features_challenge_v1/` (the npz hashes checked against its `metadata.json`, whose status must be
  `complete`). A record of exactly ten seconds is used whole; a longer record gives its centred ten-second
  window (start `(samples − 5000) // 2`, the `ssl_center_crop` rule); shorter or nonfinite records have no
  features. The features were made with 022's encoders and input paths, which the extraction checked against
  022's profile (final extraction of 29 September, after a first pass that skipped every record longer
  than ten seconds; only record lists and counts of either pass were read here).

Counts from the split manifest and the feature record lists (before any score); the runner stops if a
count differs.

| Family | Source | ECGs | Positive | Negative |
| --- | --- | ---: | ---: | ---: |
| PTB-XL | PTB-XL | 564 | 348 | 216 |
| `chapman_ningbo` | Chapman/Shaoxing | 1,018 | 745 | 273 |
| `chapman_ningbo` | Ningbo | 3,414 | 2,509 | 905 |
| `georgia` | Georgia | 1,718 | 1,372 | 346 |
| `cpsc` | CPSC 2018 | 872 | 689 | 183 |
| `cpsc` | CPSC-Extra | 590 | 590 | 0 |

The Challenge pool is 7,612 ECGs, 78% positive; PTB-XL calibration is 62% positive and SPH 34%.

## Arms

Each arm fits one Platt mapping and one threshold per head.

| Arm | Calibration data | Role |
| --- | --- | --- |
| `ptbxl` | PTB-XL only | the 027 reference; must reproduce 027's Platt coefficients, thresholds and SPH calibrated probabilities to 1e-9 |
| `pooled` | PTB-XL plus every Challenge calibration ECG, unweighted | **primary** |
| `balanced` | the same ECGs, weighted so each of the four families has equal total weight | secondary |
| `loso_chapman_ningbo`, `loso_georgia`, `loso_cpsc` | `pooled` without one Challenge family | secondary |

- **`balanced` weights.** A record of family f gets weight N / (4 n_f), with N the pool size and n_f the
  family's records, so the weights average 1. Platt is fitted with these sample weights (same `C`), and the
  threshold is the highest calibrated probability at which the weighted share of positives referred is at
  least 95%. With unit weights both reduce to the unweighted functions (tested).
- A nonpositive Platt slope stops the run.

## Evaluation

Nothing is fitted on an evaluation set. Each set is scored once.

1. **SPH (primary test), every arm.** At the threshold: sensitivity, specificity, PPV and NPV at the observed
   prevalence, false referrals per 1,000 negatives; at assumed prevalences of 5% and 1%, PPV and referrals
   per 1,000 screened; Brier score, 10-bin ECE, calibration intercept and slope of the calibrated
   probability (all as in 027, `screening_threshold.operating_point`).
2. **A held-out family (secondary new-site replication).** For each Challenge family f, arms `ptbxl` and
   `loso_f` on the calibration-group ECGs of f, which neither arm was fitted on. Same metrics. For
   `chapman_ningbo` it is also reported without the Ningbo records that have a lead stored as zero for the
   whole record (70 of the 3,414; the [Ningbo manifest](clean-ningbo-v1.md) asks for both).
3. **In sample (reference only).** Each arm on its own calibration ECGs.

The Challenge **test** groups stay closed. The pooled arm calibrates on the same hospitals as those groups,
so they would not test transfer to a new site, and the held-out-family readout answers that without them.
They remain unused for a later final stage (for example `multisource_lso`).

## Bootstrap

- **SPH:** 2,000 paired whole-patient draws, seed 29029: each draw resamples SPH patients with replacement and
  keeps every ECG of a sampled patient. The same draws serve every arm and head. Thresholds and Platt fits are
  held fixed.
- **Held-out families:** 2,000 paired record-level draws (the sources have no patient IDs, and exact
  duplicates are already removed), seed 29029, a fresh generator per set.
- Per draw: sensitivity, specificity, Brier score and 10-bin ECE of every arm and head. Draws without a
  positive or a negative are skipped. Intervals are the 2.5 and 97.5 percentiles.
- Contrasts, arm minus `ptbxl`, per head: the **deviation** `|sensitivity − 0.95|`, sensitivity,
  specificity, Brier score and ECE.

## Primary comparison and decision rule

- **Primary comparison:** on SPH, for each head, the difference in deviation
  `D = |sens(pooled) − 0.95| − |sens(ptbxl) − 0.95|`, with its paired bootstrap interval.
  - `pooled` **transfers better** for a head if the interval of D lies below 0, **worse** if above 0, and
    otherwise the two are not distinguished.
- **Decision:** multi-hospital calibration (`pooled`) replaces PTB-XL-only calibration as the project's way
  of fixing the operating point if it transfers better for at least two of the three heads and worse for
  none. Otherwise PTB-XL-only calibration stays the reference, and the results say whether any arm came
  closer.
- **The 027 transfer rule is also applied to every arm:** the threshold transfers to SPH if the SPH
  sensitivity interval includes 0.95.
- Specificity, Brier score and ECE differences are secondary. They do not change the decision, but the
  results must say whether a closer sensitivity was bought with specificity, and whether the probabilities
  themselves improved.
- Secondary arms and the held-out-family readout are described with the same interval language (interval
  excludes 0 or not) and no decision attaches to them.

## Caveats written into the results

- The Challenge split is by record; a patient with several recordings can appear in calibration and
  test groups. This does not touch SPH, whose bootstrap is by patient.
- ECG-JEPA and xECG report Chapman and Ningbo in their self-supervised pretraining data, so those two
  encoders have likely seen the `chapman_ningbo` waveforms without labels. The CPC encoder was pretrained on
  the 40k PTB-XL and MIMIC pool (Experiment 004) and has seen no Challenge record. No head saw a Challenge
  label. SPH remains unseen by every encoder.
- The Challenge negative (sinus rhythm alone) is a stricter normal than PTB-XL's NORM and SPH's code 1, and
  CPSC-Extra has no negatives at all. Pooling moves the calibration prevalence further from SPH's (78%
  Challenge against 34% SPH), which by itself pushes calibrated probabilities up.
- CPSC records longer than ten seconds are read through one centred ten-second window, which may miss a
  finding annotated elsewhere in the record. The `cpsc` held-out readout rests on 183 negatives.
- The label is an ECG annotation proxy, not confirmed disease or a referral decision. One fit per arm and
  head; the bootstrap holds the fits fixed.

## Execution

`scripts/experiments/run_multisource_calibration027b.py`, one CPU stage (at most 4 threads), run once after
`outputs/features_challenge_v1/metadata.json` reports every source complete. It hashes every input, performs
the checks above, and writes the Challenge head probabilities, the calibrated probabilities and an aggregate
`result.json` to `outputs/experiment027b_multisource_calibration_v1/`. Shared code is in
`ecg_experiment/multisource_calibration.py`; no frozen module changes. The feature extraction code is that of
commit `022d920` (branch of the GPU extraction), whose hashes the feature metadata records and the run copies
into its identity. Results go to
`docs/experiment-027b-multisource-calibration-results.md`.
