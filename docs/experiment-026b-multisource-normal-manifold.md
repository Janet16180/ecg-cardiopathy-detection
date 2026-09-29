# Experiment 026b: the normal-ECG manifold fitted on normals from several hospitals

Frozen 29 September 2026, before any distance score is computed on any arm other than 026's. This is the
rerun of [Experiment 026](experiment-026-normal-manifold.md) that the Ningbo data made possible (backlog item
`multisource_normal_manifold`). The user authorized the reruns of the experiments Ningbo could affect. Before
this freeze only aggregate 026, 027b and 022 results and metadata were read: record counts, labels, feature
record lists and window starts, and the quality-policy reasons of the Challenge training normals (a waveform
check that uses no score and no model).

## Question

026 fitted a Mahalanobis distance from normal ECGs (PCA-64, Ledoit-Wolf) on PTB-XL NORM ECGs only. On xECG
features it ranked PTB-XL development ECGs at AUROC 0.923, but SPH at 0.858, against 0.915 for the supervised
Experiment 022 probe. ECG-JEPA fell from 0.910 to 0.836 and CPC from 0.836 to 0.791.

Does fitting the same score on normal ECGs from several hospitals (PTB-XL plus the Challenge training
groups: Chapman/Ningbo, Georgia, CPSC) rank SPH ECGs better than the PTB-XL-only fit? Put differently: is
the SPH gap a too-narrow picture of "normal", which more hospitals would widen, or a property of SPH?

## What stays fixed

- **Score.** `normal_manifold.fit_mahalanobis` and `mahalanobis_scores`, unchanged: a `StandardScaler`, then
  `PCA(n_components=64, svd_solver="full")`, then `LedoitWolf` on the 64 components, all fitted on the fit set
  in float64; the score is the squared Mahalanobis distance. No hyperparameter changes and none is searched.
  The kNN score of 026 is not rerun.
- **Encoders.** `cpc`, `jepa` and `xecg`, the frozen features of 026. xECG is primary, as in 026's SPH
  question; JEPA and CPC are secondary.
- **PTB-XL and SPH features and rows.** Loaded exactly as 026 does (`run_normal_manifold026.identity`,
  `load_sph`; `run_label_efficiency025.select_rows`, `load_features`), with every cache checked against its
  receipt.
- **Label.** The standard label: PTB-XL's superclass label, SPH's primary label (Experiment 022), and the
  primary label of the [Challenge mapping](challenge-label-mapping.md) (positive for any MI, STTC, CD or HYP
  code; negative for sinus rhythm alone).

## Fit sets

The Challenge rows come from the frozen [Challenge split](challenge-splits-v1.md)
(`data/processed/challenge_splits_v1/rows.csv`, hash checked against its metadata): the `train` groups,
`evaluable` rows (defined primary label, duplicates resolved), primary label 0, with a feature row in
`outputs/features_challenge_v1/` (npz hashes checked against `metadata.json`, status `complete`).

- **Training quality policy.** A Challenge normal enters a fit set only if it passes the project's training
  quality policy (`ecg_experiment/ecg_quality.py`).
  - Ningbo: `use_training` of the [clean Ningbo manifest](clean-ningbo-v1.md) (hash checked against the
    split's recorded input).
  - Chapman, Georgia and CPSC 2018 have no quality flags in a manifest, so the policy (`ecg_quality.assess`)
    is applied to the exact ten-second window the features were computed from (the saved `window_start`),
    read with the official checksums verified (`challenge_features.read_verified`). A record with any
    exclusion reason is left out.
  - CPSC-Extra has no primary negatives.
- **PTB-XL.** 026's fit set: the 5,872 NORM-only training ECGs of Experiment 025's pool (5,537 patients).

Counts, from the split manifest, the feature record lists and the quality check (no score); the runner stops
if a count differs.

| Family | Source | Evaluable train negatives with features | Excluded by the policy | Fit-set ECGs |
| --- | --- | ---: | ---: | ---: |
| `ptbxl` | PTB-XL | 5,872 | (026) | 5,872 |
| `chapman_ningbo` | Ningbo | 2,707 | 126 (`use_training` false) | 2,581 |
| `chapman_ningbo` | Chapman/Shaoxing | 820 | 1 | 819 |
| `georgia` | Georgia | 1,027 | 4 | 1,023 |
| `cpsc` | CPSC 2018 | 551 | 0 | 551 |

The Challenge normals total 4,974; with PTB-XL, 10,846.

## Arms

| Arm | Fit set | ECGs | Role |
| --- | --- | ---: | --- |
| `ptbxl` | PTB-XL normals only | 5,872 | the 026 reference |
| `pooled` | PTB-XL plus every Challenge training normal, unweighted | 10,846 | **primary** |
| `balanced` | the same four families, 551 ECGs drawn from each | 2,204 | secondary |
| `loso_chapman_ningbo` | `pooled` without Chapman/Ningbo | 7,446 | secondary |
| `loso_georgia` | `pooled` without Georgia | 9,823 | secondary |
| `loso_cpsc` | `pooled` without CPSC | 10,295 | secondary |

- **`balanced`.** Families weighted equally by equal counts: each of `ptbxl`, `chapman_ningbo`, `georgia` and
  `cpsc` gives as many ECGs as the smallest family (CPSC, 551), drawn without replacement
  (`multisource_manifold.equal_family_subsample`, seed 30030, families in sorted order). The scaler, PCA and
  Ledoit-Wolf estimator take no sample weights, so equal counts are the simple way to weight equally. This arm
  also has a smaller fit set, which the results must keep in mind.
- **Reproduction.** The `ptbxl` arm must reproduce 026's saved `mahalanobis` scores of every encoder
  (`development_scores.csv` and `sph_scores.csv`, hashes checked against 026's `result.json`). The expected
  difference is 0; the run stops if any score differs by more than 1e-9 relative, and reports the largest
  difference.

## Evaluation

Nothing is fitted or selected on an evaluation set. Each set is scored once.

1. **SPH (primary test).** The 21,008 `use_evaluation` ECGs with a primary label (20,364 patients, 7,190
   positive), every arm and encoder: AUROC (primary metric) and average precision.
2. **Challenge calibration groups, per family (secondary).** The `calibration` groups of the split,
   `evaluable` rows with features, all quality (held-out data keep their difficult cases): Chapman/Ningbo
   4,432 ECGs (3,254 positive), Georgia 1,718 (1,372), CPSC 1,462 (1,279; CPSC-Extra adds positives only).
   Every arm is scored. The new-site question is `loso_f` against `ptbxl` on family f: neither was fitted on
   f. `pooled` on f is the in-distribution reference (it saw f's training normals; patients may repeat
   across the record split). Chapman/Ningbo is also reported without the 70 Ningbo records with a lead
   stored as zero for the whole record, as the Ningbo manifest asks.
3. **PTB-XL development (secondary).** 026's 1,306 development ECGs (1,173 patients, 843 abnormal), binary
   task, every arm: does a wider normal reference cost ranking at the home site?

The Challenge **test** groups and the PTB-XL calibration and test ECGs stay closed. No age or other subgroup
analysis is done.

## Bootstrap

- **SPH and PTB-XL development:** 2,000 paired whole-patient draws, seed 30030 (`normal_manifold`
  `patient_resamples`, `bootstrap_metrics` and `contrast`): patients are resampled with replacement and every
  ECG of a sampled patient kept. The same draws serve every arm and encoder of a set. Fits are held fixed.
- **Challenge families:** 2,000 paired record-level draws, seed 30030, a fresh generator per set (the sources
  have no patient IDs; the record is the unit, as in 027b).
- Draws with one class are skipped and counted. Intervals are the 2.5 and 97.5 percentiles of the paired
  differences.
- Contrasts, per encoder, AUROC and AP:
  - SPH and development: every arm minus `ptbxl`.
  - Family f: `loso_f` minus `ptbxl`, and `pooled` minus `loso_f` (what f's own normals add).

## Primary comparison and decision rule

- **Primary comparison:** on SPH, xECG, the AUROC of `pooled` minus that of `ptbxl`, with its paired
  whole-patient bootstrap interval.
- **Decision:**
  - The multi-hospital normal reference (`pooled`) **replaces** the PTB-XL-only reference for the
    distance-from-normal score if that interval lies entirely above 0.
  - If the interval lies entirely below 0, pooling **hurts** and PTB-XL-only stays.
  - Otherwise the two are not distinguished and PTB-XL-only stays the reference, since it is the simpler fit.
- **How much of the gap closes** (descriptive): the change in xECG SPH AUROC as a share of 026's gap to the
  Experiment 022 xECG probe (0.915 − 0.858 = 0.057).
- JEPA and CPC, the secondary arms, the AP differences, the development task and the Challenge families are
  described in the same interval language (interval above 0, below 0, or includes 0). No decision attaches to
  them.
- **New-site reading (secondary):** a new hospital's normals are "predicted" by the other hospitals if
  `loso_f` minus `ptbxl` has an interval above 0 on family f. How much remains is read from `pooled` minus
  `loso_f`.

## Caveats written into the results

- ECG-JEPA and xECG report Chapman and Ningbo in their self-supervised pretraining data, so they have likely
  seen the `chapman_ningbo` waveforms without labels. CPC was pretrained on PTB-XL and MIMIC (Experiment 004)
  and has seen no Challenge record. No Challenge label was used by any encoder. SPH is unseen by every encoder.
- The Challenge negative (sinus rhythm alone) is a stricter normal than PTB-XL's NORM and SPH's code 1. A
  normal in any source is an ECG annotation, not proof of health, and none of these are young adults.
- The Challenge split is by record; a patient may have normals in the train group and ECGs in the calibration
  group, which flatters `pooled` on the families it saw. The `loso` readouts and SPH do not depend on this.
- CPSC records longer than 10 s are read through their centred window. The CPSC calibration group has 183
  negatives.
- The fit set changes in size between arms (2,204 to 10,846). A larger fit set alone steadies the PCA and
  covariance; `balanced` is the smallest.
- A distance from normal flags anything unusual, including filtering, device and noise differences. SPH is
  heavily band-pass filtered; this experiment does not test filtering. One fit per arm; the bootstrap holds
  the fits fixed.

## Execution

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 uv run --no-sync python -u -m scripts.experiments.run_multisource_manifold026b
```

One CPU stage, one process per encoder with one BLAS thread each (three threads in all, within the four
allowed). `scripts/experiments/run_multisource_manifold026b.py` uses the frozen `normal_manifold.py` and the
new `ecg_experiment/multisource_manifold.py`; no frozen module changes. It hashes every input, source and this
protocol into the result, performs the checks above, refuses to overwrite an existing run, and writes
`outputs/experiment026b_multisource_manifold_v1/` (`result.json`, `sph_scores.csv`,
`challenge_scores.csv`, `development_scores.csv`, `fit_sets.csv`). Results go to
`docs/experiment-026b-multisource-normal-manifold-results.md`.
