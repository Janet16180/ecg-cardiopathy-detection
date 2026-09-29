# Experiment 022b: the readout trained on PTB-XL plus several hospitals, read on SPH

**Frozen 29 September 2026, before any readout is fitted on Challenge labels and before any new SPH, PTB-XL
development or Challenge score is computed.** This runs two backlog items as one experiment:
`ningbo_sph_transfer` (the rerun of [Experiment 022](experiment-022-sph-external-readout.md) with Ningbo) and
`multisource_lso` (a multi-source readout with leave-one-source-family-out evaluation). The user authorized
the reruns of the experiments Ningbo could affect. Before this freeze only aggregate 022, 027, 027b and 026b
results and metadata were read: record counts, labels, feature record lists and window starts, and the
training quality-policy reasons of the Challenge training rows (a waveform check that uses no score and no
model).

## Question

The user asks: **does adding Ningbo make things better or worse?**

022 fitted a logistic readout on frozen features of PTB-XL training ECGs only. At SPH, an unseen Chinese
hospital, it ranked ECGs at AUROC 0.915 (xECG), 0.911 (ECG-JEPA) and 0.876 (CPC). 027 and 027b showed that
its operating point does not transfer: a threshold for 95% sensitivity gives 0.91-0.92 at SPH. 026b showed
that adding Chapman/Ningbo normals to an unsupervised normal reference helped SPH clearly.

Does fitting the same readout on PTB-XL plus the labeled training ECGs of four more collections (Ningbo and
Chapman/Shaoxing, Georgia, CPSC) rank SPH ECGs better than PTB-XL alone? And which source family carries any
change, in particular Ningbo?

## What stays fixed

- **Readout.** `full_development.fit_logistic`'s head: a `StandardScaler` fitted on the arm's training rows,
  then L2 logistic regression with `C=0.01`, `lbfgs`, `tol=1e-8`, `max_iter=5000`, in float64
  (`external_readout.fit_logistic_c` with `C=0.01`). No hyperparameter changes and none is searched.
- **Encoders.** The frozen `xecg`, `jepa` and `cpc` features of 022. **xECG is primary**, as in 026b; JEPA
  and CPC are secondary.
- **PTB-XL.** 022's training rows with a standard label and features, in 022's order: 17,083 ECGs (14,822
  patients, 9,840 positive). CPC from Experiment 020's saved features, JEPA and xECG from 022's caches and
  `ptb_features.npz` (all hashes checked against their receipts).
- **SPH.** The 21,008 `use_evaluation` ECGs of `data/processed/sph_clean_v1` with a primary label (20,364
  patients, 7,190 positive), with 022's saved features (`outputs/experiment022_sph_external_v3/features.npz`,
  hash checked against 022's result).
- **Label.** The standard label: the PTB-XL superclass label for PTB-XL, 022's primary label for SPH, and
  the primary label of the [Challenge mapping](challenge-label-mapping.md) for the Challenge sources (positive
  for any MI, STTC, CD or HYP code; negative for sinus rhythm alone). The secondary labels are not used.

## Challenge training rows

From the frozen [Challenge split](challenge-splits-v1.md) (`data/processed/challenge_splits_v1/rows.csv`,
hash checked against its metadata): the `train` groups, `evaluable` rows (defined primary label, duplicates
resolved), with a feature row in `outputs/features_challenge_v1/` (npz hashes checked against its
`metadata.json`, status `complete`).

**Training quality policy, as 026b applied it, now to positives and negatives alike.** Ningbo rows must be
`use_training` in the [clean Ningbo manifest](clean-ningbo-v1.md). For Chapman, Georgia, CPSC 2018 and
CPSC-Extra, `ecg_quality.assess` is run on the exact ten-second window the features were computed from (the
saved `window_start`), read with the official checksums verified; a record with any exclusion reason is left
out. The runner reuses 026b's `quality_reasons`.

Counts from the split manifest, the feature record lists and the quality check (no score); the runner stops if
a count differs.

| Family | Source | Evaluable train with features (positive) | Policy exclusions | Training ECGs (positive) |
| --- | --- | ---: | ---: | ---: |
| `ptbxl` | PTB-XL | 17,083 (9,840) | (022) | 17,083 (9,840) |
| `chapman_ningbo` | Ningbo | 10,238 (7,531) | 286 | 9,952 (7,371) |
| `chapman_ningbo` | Chapman/Shaoxing | 3,054 (2,234) | 4 | 3,050 (2,231) |
| `georgia` | Georgia | 5,145 (4,118) | 9 | 5,136 (4,113) |
| `cpsc` | CPSC 2018 | 2,613 (2,062) | 13 | 2,600 (2,049) |
| `cpsc` | CPSC-Extra | 1,774 (1,774) | 18 | 1,756 (1,756) |

The Challenge training rows total 22,494 (17,520 positive, 78%); PTB-XL training is 58% positive.

## Arms

Each arm is one readout per encoder, fitted on PTB-XL training plus the listed Challenge training rows.

| Arm | Challenge training rows added | Training ECGs | Role |
| --- | --- | ---: | --- |
| `ptbxl` | none | 17,083 | the 022 reference |
| `pooled` | all four families, unweighted | 39,577 | **primary** |
| `balanced` | the same rows, each of the four families (`ptbxl` included) with equal total weight | 39,577 | secondary |
| `ptbxl_chapman_ningbo` | Chapman/Ningbo only | 30,085 | secondary: isolates the Chapman/Ningbo family |
| `ptbxl_ningbo` | Ningbo only | 27,035 | secondary: the Ningbo source alone |
| `loso_chapman_ningbo` | Georgia and CPSC | 26,575 | secondary |
| `loso_georgia` | Chapman/Ningbo and CPSC | 34,441 | secondary |
| `loso_cpsc` | Chapman/Ningbo and Georgia | 35,221 | secondary |

- **`balanced`.** A row of family f gets weight N / (4 n_f) (027b's `family_weights`), so the weights
  average 1. Both the scaler and the logistic regression take these sample weights; with unit weights the
  weighted fit equals the unweighted one (tested).
- **Reproduction.** The `ptbxl` arm must reproduce 022's saved SPH and development probabilities of
  `cpc_standard`, `jepa_standard` and `xecg_standard`. The expected difference is 0 and the SPH AUROC must be
  identical; the run stops if any probability differs by more than 1e-9, and reports the largest difference.

## Operating point: each arm's own calibration data

As in [027b](experiment-027b-multisource-calibration.md): Platt scaling on the head's logit
(`screening_threshold.fit_platt`) and the highest calibrated probability that refers at least 95% of the
calibration positives, fitted per arm and head on **the calibration groups of the same sources the arm was
trained on**:

| Arm | Calibration ECGs |
| --- | --- |
| `ptbxl` | PTB-XL calibration (564, 348 positive; 027's saved features); must reproduce 027's Platt fits and thresholds to 1e-9 |
| `pooled` | PTB-XL plus every Challenge calibration ECG (8,176) |
| `balanced` | the same, with 027b's family weights (027b's weighted Platt and threshold) |
| `ptbxl_chapman_ningbo` | PTB-XL plus Chapman/Ningbo (4,996) |
| `ptbxl_ningbo` | PTB-XL plus Ningbo (3,978) |
| `loso_f` | PTB-XL plus the Challenge calibration groups without family f (3,744, 6,458, 6,714) |

The Challenge calibration groups are 027b's: evaluable rows with features, all quality (Chapman 1,018 with
745 positive, Ningbo 3,414 with 2,509, Georgia 1,718 with 1,372, CPSC 2018 872 with 689, CPSC-Extra 590 with
590). PTB-XL calibration is used only to fit Platt scaling and thresholds, never as a test.

## Evaluation

Nothing is fitted or selected on an evaluation set. Each set is scored once.

1. **SPH (primary test), every arm and encoder.** AUROC (primary metric) and average precision. Brier score
   and calibration intercept and slope of the raw readout probability. At the arm's own threshold:
   sensitivity, specificity, Brier score, 10-bin ECE and calibration intercept and slope of the Platt
   probability (`screening_threshold.operating_point`).
2. **Held-out family (the `multisource_lso` readout, secondary).** For each Challenge family f, its
   calibration-group ECGs (Chapman/Ningbo 4,432 with 3,254 positive, Georgia 1,718 with 1,372, CPSC 1,462 with
   1,279). The new-site comparison is `loso_f` against `ptbxl`: neither was trained or calibrated on f. Every
   other arm is also scored; `pooled` on f is the in-distribution reference (it saw f's training rows, and
   patients may repeat across the record split). The same metrics as SPH. Chapman/Ningbo is also reported
   without the 70 Ningbo records with a lead stored as zero for the whole record.
3. **PTB-XL development (secondary).** 022's 1,572 labeled full-development ECGs (1,413 patients, 884
   positive), and its original (1,306, 843 positive) and added (266, 41 positive) subsets. The added subset is
   022's hard-case set. AUROC and AP only: does a multi-source readout cost at the home site, and does it rank
   the hard cases differently?

The Challenge **test** groups, PTB-XL calibration as a test, and the PTB-XL test ECGs stay closed. No age or
other subgroup analysis is done.

## Bootstrap

- **SPH and PTB-XL development:** 2,000 paired whole-patient draws, seed 31031 (`normal_manifold`
  `patient_resamples`, `bootstrap_metrics` and `contrast` for AUROC and AP; 027b's `bootstrap_rates` for the
  operating-point rates, which draws the same patients from the same seed). The same draws serve every arm
  and encoder of a set. Fits and thresholds are held fixed.
- **Held-out families:** 2,000 paired record-level draws, seed 31031, a fresh generator per set (the
  sources have no patient IDs).
- Draws with one class are skipped and counted. Intervals are the 2.5 and 97.5 percentiles.
- Contrasts, per encoder: every arm minus `ptbxl` on every set (AUROC, AP; on SPH and the families also
  sensitivity, specificity, Brier score, ECE and the deviation `|sensitivity − 0.95|`), and `pooled` minus
  `loso_chapman_ningbo` on SPH (what Chapman/Ningbo adds to the other sources).

## Primary comparison and decision rule

- **Primary comparison:** on SPH, xECG, the AUROC of `pooled` minus that of `ptbxl`, with its paired
  whole-patient bootstrap interval.
- **Decision:**
  - The multi-source readout (`pooled`) **replaces** the PTB-XL-only readout if that interval lies entirely
    above 0.
  - If it lies entirely below 0, adding the hospitals **hurts** and PTB-XL-only stays.
  - Otherwise the two are not distinguished and PTB-XL-only stays, since it is the simpler fit.
- **Size.** With 21,008 SPH ECGs, differences of about 0.002 can have intervals that exclude 0. The results
  call a difference below 0.005 AUROC "negligible in practice" even when its interval excludes 0 (the
  priorities page's 0.005-0.01 noise level). This wording does not change the decision.
- **Does Ningbo help? (prespecified answer).** Read on SPH, xECG AUROC, from two contrasts:
  (a) `ptbxl_chapman_ningbo` minus `ptbxl` (the family added alone) and (b) `pooled` minus
  `loso_chapman_ningbo` (the family added to the others). Ningbo **helps** if both intervals lie above 0,
  **hurts** if both lie below 0, and otherwise its effect is **not established**. `ptbxl_ningbo` minus
  `ptbxl` (the Ningbo source without Chapman), JEPA, CPC, the operating point and PTB-XL development are
  described alongside, with no separate decision.
- **Operating point (secondary):** for each arm and head, the SPH deviation difference
  `D = |sens(arm) − 0.95| − |sens(ptbxl) − 0.95|` in 027b's language (better if its interval lies below 0,
  worse if above 0), and 027's transfer rule (the threshold transfers if the SPH sensitivity interval
  includes 0.95). The results must say whether a closer sensitivity was bought with specificity.
- All other contrasts are described by where their interval lies (above 0, below 0, includes 0), with no
  decision.

## Caveats written into the results

- ECG-JEPA and xECG report Chapman and Ningbo in their self-supervised pretraining data, so they have likely
  seen the `chapman_ningbo` waveforms without labels. The Chapman/Ningbo held-out readout is therefore not an
  unseen-hospital test for those two encoders. CPC was pretrained on PTB-XL and MIMIC (Experiment 004) and has
  seen no Challenge record. No encoder saw a Challenge label; SPH is unseen by every encoder.
- The Challenge negative (sinus rhythm alone) is a stricter normal than PTB-XL's NORM and SPH's code 1, so
  the added rows teach a slightly different boundary. CPSC-Extra adds positives only, and the Challenge rows
  are 78% positive against 58% for PTB-XL and 34% for SPH, which moves the raw probabilities up; the
  operating point is set by each arm's own Platt calibration.
- The Challenge split is by record; a patient may have ECGs in the train and calibration groups, which
  flatters an arm on a family it was trained on. SPH and the `loso` readouts do not depend on this.
- CPSC records longer than 10 s are read through their centred window. The CPSC calibration group has 183
  negatives.
- The arms differ in training size (17,083 to 39,577). The readouts are fixed linear heads on frozen
  features, so more rows mainly steady the fit; a gain may come from size as well as from the hospitals.
- One fit per arm and encoder, one run, one checkpoint per encoder. The bootstrap holds the fits fixed. The
  label is an ECG annotation proxy, not confirmed disease or a referral decision. PTB-XL development was
  inspected by earlier experiments.

## Execution

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 uv run --no-sync python -u -m scripts.experiments.run_multisource_readout022b
```

One CPU stage: the quality check in the main process, then one process per encoder with one BLAS thread
each (three threads in all, within the four allowed). `scripts/experiments/run_multisource_readout022b.py`
uses the frozen 022, 027, 027b and 026b runners and modules and the new
`ecg_experiment/multisource_readout.py`; no frozen module changes. It hashes every input, source and this
protocol into the result, performs the checks above, refuses to overwrite an existing run, and writes
`outputs/experiment022b_multisource_readout_v1/` (`result.json`, `predictions.npz`, `training_rows.csv`,
`run.log`). Results go to `docs/experiment-022b-multisource-readout-results.md`.
