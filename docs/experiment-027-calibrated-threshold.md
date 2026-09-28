# Experiment 027: a calibrated screening threshold, checked on SPH

**Frozen 28 September 2026, before any calibration, development or SPH score is computed at a threshold.**
The user authorized new follow-up experiments. Experiment 022 showed that the CPC, ECG-JEPA and xECG heads keep
their AUROC ranking at an independent hospital (SPH). AUROC hides what screening needs: a fixed operating point.
The project's rule (see [model findings](model-findings-report.md), "Platt calibration") is a Platt mapping
and the highest threshold that reaches at least 95% sensitivity on the PTB-XL calibration patients. This
experiment applies that rule to the three Experiment 022 v3 heads and reads the operating point once on PTB-XL
development and once on SPH.

## Question

At the project's operating point (at least 95% sensitivity, chosen on the calibration patients), what
sensitivity, specificity and false-referral rate do CPC, ECG-JEPA and xECG give on PTB-XL development and at
the new SPH hospital? Does the threshold keep 95% sensitivity there?

## Heads

`cpc_standard`, `jepa_standard` and `xecg_standard` of Experiment 022 v3, refitted exactly as there:

- `cpc_standard` with `external_readout.ptb_heads` from the saved Experiment 020 features, which must reproduce
  Experiment 020's development probabilities to 1e-9;
- `jepa_standard` and `xecg_standard` with the 022 runner's `fit_encoder_heads`, from the ECG-JEPA and xECG
  caches plus 022's saved `ptb_features.npz` (hash checked against 022's `profile.json`).

Each refitted head must reproduce 022's saved development probabilities
(`outputs/experiment022_sph_external_v3/development_predictions.npz`, hash checked against 022's
`result.json`) to 1e-9 before any calibration score is computed. Nothing in the heads changes.

A head's logit is `logit(p)` of its probability `p`, computed the same way for every set.

## Rows

Counts below come from metadata and labels only, before this freeze.

| Set | Rows | ECGs | Patients | Positive | Negative |
| --- | --- | ---: | ---: | ---: | ---: |
| Calibration | `heldout_references.csv` rows with `split == "calibration"`, standard label defined | 564 | 504 | 348 | 216 |
| Development, full | 022 development rows with a standard label | 1,572 | 1,413 | 884 | 688 |
| Development, original | the original subset of those rows | 1,306 | 1,173 | 843 | 463 |
| SPH | `use_evaluation` rows of `data/processed/sph_clean_v1` with a primary label | 21,008 | 20,364 | 7,190 | 13,818 |

- The calibration rows are all PTB-XL fold 9 and share no patient with the training or development rows; the
  run checks both. `data/processed/training_union_500hz_v1/heldout_references.csv` and the copy that
  `full_development.cohorts` reads (`outputs/data_quality/clean_rerun_preflight_v1/`) must have the same hash.
  The 15 fold-9 ECGs of calibration patients outside the references file (`cohorts()["closed"]`) are not used.
- The label is the standard PTB-XL superclass label of Experiment 020 for PTB-XL and the primary SPH label of
  Experiment 022 (`rows.csv`, hash checked against the manifest receipt).
- PTB-XL test (fold 10) waveforms, features, labels and scores are never read. The ECG-JEPA cache and the CPC
  waveform pool also hold test ECGs; they are memory-mapped and only rows selected by ECG ID are read.

## Calibration features

- **ECG-JEPA:** from `data/processed/pretrained/ecg-jepa-full-public`, selected by ECG ID. All 564 are present.
- **CPC:** no CPC feature cache holds calibration rows (Experiment 020 saved training and development only).
  The 250 Hz CPC pool (`data/processed/cpc_pool_40k`, hashes checked against its `complete.json`) holds all 564
  signals. Features are extracted with the Experiment 020 path (`ptb_cpc_features.ptb_signals` and
  `pooled_features`, unchanged starting encoder and normalization).
- **xECG:** the Experiment 016 cache holds no calibration rows. They are extracted with the 022 xECG path
  (`ptb_xecg_input` and `xecg_features`, released checkpoint).
- The GPU work runs under the shared `gpu_lock` in blocking mode, so it waits for any running job (Experiment
  023) to release the GPU.
- **Integrity, before any extraction is used:** recomputed features of 32 evenly spaced training ECGs must match
  the cached ones to 1e-4 (maximum absolute difference) for each encoder: CPC against Experiment 020's saved
  training features, ECG-JEPA and xECG with the 022 runner's `integrity` check.

## Calibration and threshold

Fitted on the calibration rows only, per head:

1. **Platt scaling.** A logistic regression of the label on the head's logit, fitted as in
   `evaluation._fit_calibrator` (`C=1e6`, lbfgs). A nonpositive slope stops the run.
2. **Threshold.** The highest calibrated probability at which at least 95% of calibration positives have a
   calibrated probability at or above it (`evaluation.select_threshold`). An ECG is referred when its
   calibrated probability is at or above the threshold.

## Evaluation

Nothing is fitted. Each set is scored once. Development uses the saved 022 development probabilities (equal to
the refitted heads) and SPH the saved 022 SPH probabilities
(`outputs/experiment022_sph_external_v3/predictions.npz`, hash checked; its ECG and patient IDs must match the
manifest's evaluation rows).

Per head, on the calibration set (in sample, for reference), full development, original development and SPH:

- at the threshold: sensitivity, specificity, PPV and NPV at the set's observed prevalence, and false referrals
  per 1,000 negatives (1,000 times one minus specificity);
- at assumed screening prevalences of 5% and 1%, from the set's sensitivity and specificity: PPV, false
  referrals and total referrals per 1,000 people screened (`evaluation.scenario_ppv`);
- calibration of the calibrated probability: Brier score, calibration intercept (maximum-likelihood `a` in
  `logit P(y=1) = a + logit(p)`, ideal 0), calibration slope (unpenalized `b` in `logit P(y=1) = c + b logit(p)`,
  ideal 1), and 10-bin equal-width expected calibration error (`evaluation.metrics`).

## Contrasts

On full development and on SPH, 2,000 paired whole-patient bootstrap draws, seed 27027 (a fresh generator per
set). Each draw resamples patients with replacement, keeps every ECG of a sampled patient, and applies each
head's fixed threshold. Reported with 2.5 and 97.5 percentiles:

- specificity differences `xecg − cpc`, `jepa − cpc` and `xecg − jepa`;
- each head's sensitivity and specificity.

## Prespecified reading

- The threshold **transfers** to SPH for a head if that head's SPH sensitivity interval includes 0.95.
  The point estimate is also stated as above or below 0.95.
- A head is **better for screening** than another on a set if their specificity difference interval excludes 0
  in its favor.
- Development is quoted beside SPH but no test compares the two populations.

## Caveats written into the results

- The calibration patients come from the same hospital (and devices) as training, so the threshold is an
  in-distribution choice.
- SPH prevalence on the primary label is about 34%, far above what a student population would show; PPV and
  NPV at the observed prevalence do not carry over, hence the 5% and 1% scenarios.
- The label is an ECG annotation proxy, not confirmed disease or a referral decision.
- One fit per head; the bootstrap does not include the uncertainty of the Platt fit or threshold.

## Execution

`scripts/experiments/run_calibrated_threshold027.py`, one stage, run once. It hashes every input, performs the
checks above, and writes local calibration features and calibrated probabilities and an aggregate `result.json`
to `outputs/experiment027_calibrated_threshold_v1/`. The GPU part is 32 integrity rows per encoder plus 564 CPC
and 564 xECG calibration ECGs, so no cost gate is used. Results go to
`docs/experiment-027-calibrated-threshold-results.md`.
