# Experiment 035: why the pooled readout loses on the PTB-XL hard added subset

**Frozen 29 September 2026, before any score of this experiment is computed.** This runs the backlog item
`label_harmonization_hard_subset`. The user asked to keep experiments running during a machine migration.
Before this freeze only the following were read: the aggregate results of 020, 022, 022b and 025b; PTB-XL
metadata (SCP codes, NORM likelihoods, devices, ages, sex, heart axis); the Challenge SNOMED codes and the
frozen split; 022b's `training_rows.csv` (labels and quality reasons, no score); the descriptive summary below
(`scripts/experiments/describe_hard_subset035.py`, which reads no feature and no score); one pass of the
runner with `--counts`, which builds every arm's training rows and runs the training quality policy on the
Challenge rows 022b did not check, then stops before any readout is fitted; and one timing fit of the xECG
readout on Challenge training rows, from which nothing was scored. 022b's predictions were not opened.

## Question

022b made the readout fitted on PTB-XL plus the Challenge training groups (`pooled`) the adopted pipeline; 030
and 033 build on it. It costs little on ordinary PTB-XL development ECGs but a lot on 022's hard added subset:

| xECG AUROC | `ptbxl` | `pooled` | `pooled` − `ptbxl` |
| --- | ---: | ---: | --- |
| Ordinary development (1,306, 843 positive) | 0.960 | 0.955 | −0.005 [−0.009, −0.000] |
| Hard added subset (266, 41 positive) | 0.726 | 0.655 | −0.072 [−0.113, −0.032] |

JEPA and CPC lose 0.066 and 0.069 there (CPC falls to 0.506). In 025b the same contrast was −0.053 with
025's smaller PTB-XL pool. Is the loss a label-definition mismatch between the PTB-XL superclass label and the
Challenge SNOMED mapping, which could be fixed, or a genuine cost of pooling hospitals?

## What the hard subset is (descriptive, no scores)

The hard subset is every fold-9 ECG the project label dropped whose patient is not a calibration patient (020).
The project label keeps a NORM record only when NORM and SR are its sole codes, so **every one of the 266
lists NORM plus something else**. The standard label then splits them:

| Group | ECGs | Most common codes besides NORM | Median age | Male |
| --- | ---: | --- | ---: | ---: |
| Hard negatives | 225 | SR 138, ABQRS 70, SARRH 37, PVC 22, STACH 21, SBRAD 20, VCLVH 16, PAC 12, LPR 12, NT_ 7, STD_ 7, LVOLT 7, AFIB 4 | 52 | 0.41 |
| Hard positives | 41 | IRBBB 24, IVCD 7, SARRH 7, NST_ 5, 1AVB 4 (35 are CD+NORM, 4 STTC+NORM) | 53 | 0.51 |
| Ordinary negatives | 463 | SR 455 only | 52 | 0.55 |
| Ordinary positives | 843 | no NORM; IMI 184, ASMI 177, LVH 136, LAFB 128, NDT 128, AFIB 107 | 69 | 0.53 |

- The hard negatives are normal ECGs with a rhythm or form statement (sinus variants, ectopic beats, voltage,
  borderline PR or ST-T form statements). Their NORM likelihood is lower than for ordinary normals (100 in 95
  of 225, against 404 of 463).
- The hard positives are normal-looking ECGs with a minor conduction finding, mostly incomplete RBBB, at the
  age of the normals rather than of the ordinary positives.
- Devices are spread like the ordinary development set, except that `CS100 3` (the device the features
  identify best, 020) has 2 hard ECGs against 47 ordinary ones. About 10% of hard and of ordinary reports were
  generated automatically; every development report was validated by a human.
- **Under the Challenge mapping, no hard negative would be a negative.** Translating their PTB-XL codes to
  the Challenge rule (`hard_subset.ptbxl_under_challenge`), 199 of the 225 would be undefined (never shown to a
  readout) and 26 positive (LPR, STD_, NT_, INVT count as the Challenge's CD and STTC codes). With the
  Challenge's secondary label (sinus variants allowed), 68 would be negative. The 455 SR-only ordinary
  negatives stay negative under both. All 41 hard positives are positive under both mappings.
- **The Challenge training rows teach that these rhythm and form codes mean "abnormal".** Among 022b's kept
  Challenge training rows, 48% of Chapman/Ningbo positives and 30% of Georgia positives list a sinus variant
  (bradycardia, tachycardia, arrhythmia), and 37% and 30% list a PTB-XL-compatible rhythm or form code (PVC,
  PAC, low voltage, high voltage, Q waves, axis deviation); **0% of the negatives list either**, because the
  primary negative is sinus rhythm alone. In PTB-XL both kinds of ECG are negatives when NORM is present.
- **The training analogues are already in 022b.** The 1,724 PTB-XL training ECGs without a project label
  (353 positive) look like the hard subset (IRBBB 203 and IVCD 80 among the positives; SARRH 312, SBRAD 276,
  STACH 178, PVC 132 among the negatives). 025b's pool lacked them, but 022b's 17,083 PTB-XL training rows
  include all of them with their standard label, so the 022b loss is not caused by their absence.

The prespecified working hypothesis is therefore the Challenge negative definition, a label-selection
confound: in the pooled rows, sinus variants and benign form findings occur only in positives. The arms below
test it against the other candidate causes.

## Readout, encoders and data (fixed)

- **Readout.** 022b's: a train-only `StandardScaler` and L2 logistic regression, `C=0.01`, `lbfgs`,
  `tol=1e-8`, `max_iter=5000`, float64 (`multisource_readout.fit_readout`; weighted arms use its weighted
  scaler and fit). No hyperparameter is changed or searched.
- **Encoders.** The frozen xECG, ECG-JEPA and CPC features of 022 and 022b. **xECG is primary**; JEPA and CPC
  are secondary.
- **PTB-XL training.** 022b's 17,083 ECGs (14,822 patients, 9,840 positive) with the standard label.
- **Challenge training.** The frozen split's `train` groups, duplicate status `unique` or `kept`, with a
  feature row, under the arm's label rule, passing the training quality policy. 022b's quality result is
  reused for the 22,824 rows it checked; the 11,054 others with a label under some rule are checked with
  026b's `quality_reasons` (Ningbo `use_training`, the saved feature window for the other sources). From the
  `--counts` pass, 611 of them are excluded (607 Ningbo, 3 Chapman, 1 Georgia).
- **Integrity.** `ptbxl`, `pooled`, the three `loso_` arms and `ptbxl_chapman_ningbo` must reproduce 022b's
  saved development and SPH probabilities for every encoder to 1e-9; the run stops otherwise.

## Arms (each changes one thing relative to 022b's `pooled`)

| Arm | Change | Training ECGs (positive) |
| --- | --- | ---: |
| `ptbxl` | reference: PTB-XL only (022) | 17,083 (9,840) |
| `pooled` | reference: 022b's adopted readout | 39,577 (27,360) |
| **(a) families** | | |
| `loso_chapman_ningbo`, `loso_georgia`, `loso_cpsc` | one Challenge family left out (022b) | 26,575; 34,441; 35,221 |
| `ptbxl_chapman_ningbo`, `ptbxl_georgia`, `ptbxl_cpsc` | one Challenge family alone | 30,085; 22,219; 21,439 |
| **(b) Challenge labels** | | |
| `secondary_negative` | Challenge negatives may list the benign sinus variants (the mapping's secondary label) | 47,889 (27,360) |
| `ptbxl_negative` | negatives: no superclass code, a sinus code present, all other codes PTB-XL-compatible rhythm or form codes (PVC, PAC, low QRS voltage, LV high voltage, abnormal Q waves, axis deviation) | 50,020 (27,360) |
| `form_ignored` | the Challenge codes whose PTB-XL counterparts are form statements only (ST depression, ST elevation, T-wave inversion, T-wave abnormal, prolonged PR) no longer make a record positive; records left without a superclass code drop out | 35,316 (23,099) |
| `ptbxl_rule` | the PTB-XL rule as closely as the codes allow: `form_ignored` positives and `ptbxl_negative` negatives that may also list those form codes (two changes to the one mapping; secondary) | 47,809 (23,099) |
| **(c) source** | | |
| `source_indicator` | three 0/1 columns for the Challenge families; PTB-XL rows and every scored ECG get 0 | 39,577 |
| `prevalence_matched` | within each Challenge family, positives and negatives reweighted to PTB-XL's training prevalence (0.576), family totals kept | 39,577 |
| `ptbxl_half` | PTB-XL rows carry half of the total weight (43% in `pooled`) | 39,577 |
| **(d) the dropped PTB-XL ECGs** | | |
| `dropped_upweighted` | the 1,724 dropped training ECGs carry the share of weight they have in the PTB-XL-only fit (10.1% instead of 4.4%) | 39,577 |
| `without_dropped` | the 1,724 removed (diagnostic, as in 025b's pool; not a candidate fix) | 37,853 (27,007) |

Weighted arms use weights that average 1, in the scaler and the logistic fit. Rules are implemented in
`ecg_experiment/hard_subset.py` (tested in `tests/test_hard_subset.py`); the `primary` and `secondary` rules
must reproduce the split's labels on every row. Per-family counts are in the `--counts` output and are checked
by the runner. The source indicator does not change the ranking within any scored set, since every scored
ECG gets the same value.

**Device or filter shift** is not tested with a new feature: a high-pass or device control would need new
feature extraction on the GPU, which this CPU-only run does not do. `source_indicator` tests whether letting
the readout separate the hospitals helps, and the per-device error analysis below is descriptive.

## Evaluation

1. **Hard added subset (primary set):** 266 ECGs of 258 patients, 41 positive: the labeled full-development
   rows with `original` false.
2. **Ordinary development:** the 1,306 original development ECGs (1,173 patients). **Full development:** all
   1,572 (1,413 patients).
3. **SPH:** the 21,008 `use_evaluation` ECGs with a primary label (7,190 positive). SPH has been read by
   022-033; here it is a guard against a fix that costs external ranking, not a final test.
4. **Cross-fitted training analogues (secondary, xECG only):** the 1,724 dropped PTB-XL training ECGs. The
   17,083 PTB-XL training rows are split into 5 folds by patient (`hard_subset.patient_folds`, seed 39039);
   each arm is refitted without one fold's PTB-XL patients (all Challenge rows of the arm kept, weights
   recomputed on the fold's rows) and scores that fold's dropped ECGs. This is a larger analogue of the hard
   subset (353 positives against 41).

Metric: AUROC (average precision reported alongside). Every contrast is **arm minus `pooled`**, with a paired
whole-patient bootstrap from `ecg_experiment.intervals.paired_auroc_difference`: 2,000 draws, seed 39039, the
same draws for every arm of a set; fits are held fixed. The loss is `ptbxl` minus `pooled`.

The PTB-XL calibration and test ECGs and the Challenge calibration and test groups are not read. No age or
other demographic subgroup analysis is done.

## Primary contrast and decision rule

- **Primary:** on the hard subset, xECG, AUROC of each candidate arm (every arm except `ptbxl`, `pooled` and
  `without_dropped`) minus `pooled`.
- **Recovery** = (arm − `pooled`) / (`ptbxl` − `pooled`) on the hard subset.
- **Verdict per arm (xECG):**
  - `fix`: recovery ≥ 0.5, the hard-subset interval lies above 0, and SPH AUROC falls by at most 0.005
    (arm − `pooled` ≥ −0.005);
  - `trade_off`: recovery ≥ 0.5 with the interval above 0, but SPH falls by more than 0.005;
  - `suggestive`: recovery ≥ 0.5 and SPH kept, but the hard-subset interval includes 0;
  - `no_recovery`: otherwise.
- **Decision:** if any arm is a `fix`, the preferred arm is the `fix` with the highest xECG SPH AUROC, and the
  recommendation is to refit the adopted pooled readout with that change (`adopt_preferred_arm`). If none is,
  `pooled` stays (`no_fix`). The preferred arm is **confirmed on the analogues** if its cross-fitted
  arm − `pooled` interval also lies above 0; this is reported and does not change the decision.
- **What it means (prespecified):** a (b) arm as the preferred fix means a label-definition mismatch that is
  fixable; a (c) or (d) arm, a weighting or source effect; an (a) arm, a single family. Arms that recover but
  cost SPH mean a genuine trade-off. If nothing recovers half, the tested levers do not explain the loss.
- Ordinary development and SPH AUROC are reported for every arm with intervals, so that no fix silently loses
  SPH; an arm that costs more than 0.005 on ordinary development is flagged in the results. JEPA and CPC get
  the same verdicts, described without a separate decision.

## Secondary descriptive analyses (after scoring)

- **Where the hard ECGs rank.** For every arm (xECG), each hard ECG's percentile among the 463 ordinary
  development negatives' scores, averaged within code groups: hard negatives with ST-T or PR form statements
  (STD_, STE_, INVT, TAB_, NT_, LOWT, LPR), else sinus variants, else ectopy (PVC, PAC, PRC(S), BIGU, TRIGU,
  SVARR), else voltage or Q waves (LVOLT, VCLVH, HVOLT, QWAVE), else other; hard positives with IRBBB, other
  CD (IVCD, 1AVB, LAFB, LPFB), else other. This shows whether `pooled` loses by lifting hard negatives or by
  lowering hard positives, and which group an arm moves.
- **Devices.** The same percentile for `ptbxl` and `pooled`, per device with at least 20 hard ECGs.

## Caveats written into the results

- The hard subset has 41 positives; its intervals are wide. The analogues help, but are PTB-XL training ECGs
  scored by cross-fitting, not held-out development patients.
- Development patients and SPH were inspected by earlier experiments. The hypothesis above was formed from
  metadata of the same development ECGs that are scored.
- One fit per arm and encoder. The arms differ in training size (21,439 to 50,020), so a label arm also
  changes size and prevalence (`secondary_negative` makes the Challenge rows 57% positive instead of 78%);
  `prevalence_matched` separates prevalence from the label definition.
- JEPA and xECG were pretrained without labels on Chapman and Ningbo waveforms. The labels are ECG annotation
  proxies, not confirmed disease or referral decisions.

## Execution

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 uv run --no-sync python -m scripts.experiments.describe_hard_subset035
PYTHONPATH=. OMP_NUM_THREADS=1 uv run --no-sync python -u -m scripts.experiments.run_hard_subset035 --counts
PYTHONPATH=. OMP_NUM_THREADS=1 uv run --no-sync python -u -m scripts.experiments.run_hard_subset035
```

One CPU stage with three worker processes of one BLAS thread each: one per encoder for the main fits, then the
five xECG cross-fit folds. No GPU. The runner hashes every input, this protocol and its sources into the
result, refuses to overwrite an existing `result.json`, and writes `outputs/experiment035_hard_subset_v1/`
(`descriptive.json`, `result.json`, `predictions.npz`, `training_rows.csv`, `run.log`). Results go to
`docs/experiment-035-hard-subset-results.md`.
