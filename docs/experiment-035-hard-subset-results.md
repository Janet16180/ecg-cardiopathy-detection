# Experiment 035 results: why the pooled readout loses on the PTB-XL hard added subset

Completed 30 September 2026 under the [frozen protocol](experiment-035-hard-subset.md) (committed as
`f2d0666`; the run recorded the same file hash, `55c68ad1…b21a`). Backlog item
`label_harmonization_hard_subset`. Run once on CPU in 2,775 s (48 s of it the training quality check), three
worker processes with one BLAS thread each, no GPU. Local outputs are in `outputs/experiment035_hard_subset_v1/`
(`descriptive.json`, `result.json` SHA-256 `5a502fe9…ce22`, `predictions.npz`, `training_rows.csv`, `run.log`).
No PTB-XL calibration or test ECG and no Challenge calibration or test ECG was read. No age or other
demographic subgroup analysis was done.

Before the run, a synthetic smoke test of the runner functions (random features, no real score) checked that
the code ran end to end. Nothing was changed after it.

## Integrity

- The six arms 022b also fitted (`ptbxl`, `pooled`, the three `loso_` arms, `ptbxl_chapman_ningbo`)
  reproduced 022b's saved development and SPH probabilities exactly for all three encoders (largest
  difference 0.0).
- Every input matched its receipt, and every count matched the protocol: 17,083 PTB-XL training ECGs (1,724
  without a project label, 353 positive), 11,054 Challenge rows newly checked for quality (611 excluded), and
  every arm's training size. The `primary` and `secondary` rules reproduced the split's labels on every row.
- Every fit converged (255-366 iterations for xECG). No bootstrap draw was skipped.

Intervals are 95% paired whole-patient bootstrap intervals (`ecg_experiment.intervals`, 2,000 draws, seed
39039). Every contrast is arm minus `pooled`.

## 1. The arms on the hard subset and at SPH (xECG, primary)

The loss to recover: `ptbxl` minus `pooled` on the hard subset is +0.072 [+0.032, +0.112] (0.726 against 0.655).
Recovery is the share of it an arm wins back.

| Arm | Hard (266, 41 pos.) | Arm − `pooled`, hard | Recovery | Ordinary − `pooled` | SPH | SPH − `pooled` | Verdict |
| --- | ---: | --- | ---: | --- | ---: | --- | --- |
| `ptbxl` | 0.726 | +0.072 [+0.032, +0.112] | | +0.005 | 0.915 | −0.024 | reference |
| `pooled` | 0.655 | | | | 0.939 | | reference |
| (a) `loso_chapman_ningbo` | 0.677 | +0.023 [−0.000, +0.045] | 0.31 | +0.002 | 0.926 | −0.012 [−0.013, −0.011] | no recovery |
| (a) `loso_georgia` | 0.664 | +0.010 [−0.004, +0.023] | 0.13 | +0.001 | 0.939 | +0.000 | no recovery |
| (a) `loso_cpsc` | 0.660 | +0.005 [−0.014, +0.024] | 0.07 | +0.001 | 0.935 | −0.003 | no recovery |
| (a) `ptbxl_chapman_ningbo` | 0.676 | +0.021 [+0.002, +0.040] | 0.30 | +0.003 | 0.936 | −0.003 | no recovery |
| (a) `ptbxl_georgia` | 0.696 | +0.041 [+0.011, +0.072] | 0.57 | +0.004 | 0.920 | −0.019 [−0.021, −0.017] | trade-off |
| (a) `ptbxl_cpsc` | 0.697 | +0.042 [+0.009, +0.076] | 0.59 | +0.002 | 0.924 | −0.015 [−0.016, −0.013] | trade-off |
| (b) `secondary_negative` | 0.685 | +0.030 [−0.004, +0.064] | 0.42 | +0.000 | 0.930 | −0.009 [−0.011, −0.008] | no recovery |
| (b) `ptbxl_negative` | 0.709 | +0.054 [+0.012, +0.095] | 0.76 | −0.004 [−0.009, +0.000] | 0.928 | −0.010 [−0.012, −0.008] | trade-off |
| (b) `form_ignored` | 0.673 | +0.018 [+0.008, +0.030] | 0.26 | +0.000 | 0.937 | −0.002 [−0.002, −0.001] | no recovery |
| (b) `ptbxl_rule` | 0.712 | +0.058 [+0.014, +0.100] | 0.80 | −0.007 [−0.013, −0.002] | 0.913 | −0.026 [−0.028, −0.023] | trade-off |
| (c) `source_indicator` | 0.655 | +0.001 [−0.013, +0.015] | 0.01 | +0.001 | 0.938 | −0.000 | no recovery |
| (c) `prevalence_matched` | 0.649 | −0.006 [−0.022, +0.010] | −0.08 | −0.001 | 0.939 | +0.001 | no recovery |
| (c) `ptbxl_half` | 0.667 | +0.012 [+0.006, +0.019] | 0.17 | +0.001 | 0.938 | −0.001 | no recovery |
| (d) `dropped_upweighted` | 0.712 | +0.058 [+0.038, +0.080] | 0.81 | −0.002 [−0.003, +0.000] | 0.939 | +0.000 [−0.000, +0.001] | fix |
| (d) `without_dropped` | 0.562 | −0.093 | | −0.001 | 0.937 | −0.002 | diagnostic |

Ordinary development is the 1,306 original ECGs (843 positive); SPH has 21,008 ECGs (7,190 positive).
Full development (1,572): `pooled` 0.927, `dropped_upweighted` 0.931 (+0.004 [+0.002, +0.006]), `ptbxl` 0.939.
Hard-subset average precision: `pooled` 0.255, `dropped_upweighted` 0.282, `ptbxl` 0.293.

## 2. The larger analogue: the 1,724 dropped PTB-XL training ECGs, cross-fitted (xECG)

353 positives, five patient folds.

| Arm | AUROC | Arm − `pooled` |
| --- | ---: | --- |
| `ptbxl` | 0.683 | +0.101 [+0.084, +0.119] |
| `pooled` | 0.581 | |
| `dropped_upweighted` | 0.660 | +0.079 [+0.071, +0.088] |
| `ptbxl_rule` | 0.660 | +0.079 [+0.061, +0.096] |
| `ptbxl_cpsc` | 0.664 | +0.083 [+0.070, +0.096] |
| `ptbxl_negative` | 0.645 | +0.064 [+0.049, +0.079] |
| `ptbxl_georgia` | 0.639 | +0.058 [+0.044, +0.071] |
| `loso_chapman_ningbo` | 0.630 | +0.049 [+0.039, +0.059] |
| `secondary_negative` | 0.628 | +0.047 [+0.034, +0.060] |
| `ptbxl_half` | 0.606 | +0.024 [+0.021, +0.028] |
| `loso_georgia`, `form_ignored` | 0.599, 0.598 | +0.018, +0.017 |
| `ptbxl_chapman_ningbo` | 0.595 | +0.014 [+0.006, +0.022] |
| `loso_cpsc`, `source_indicator`, `prevalence_matched` | 0.576, 0.575, 0.566 | −0.005 to −0.015 |
| `without_dropped` | 0.484 | −0.097 [−0.107, −0.087] |

The analogues rank the arms in the same order as the hard subset, with intervals a quarter as wide.

## 3. Where the hard ECGs move (xECG, descriptive)

Mean percentile of each hard ECG among the 463 ordinary development normals (higher means scored more
abnormal). For negatives lower is better; for positives higher is better.

| Hard group | ECGs | `ptbxl` | `pooled` | `dropped_upweighted` | `ptbxl_negative` | `secondary_negative` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Negatives, sinus variant | 73 | 0.611 | 0.695 | 0.649 | 0.560 | 0.577 |
| Negatives, ectopy | 31 | 0.676 | 0.720 | 0.643 | 0.693 | 0.748 |
| Negatives, ST-T or PR form | 26 | 0.801 | 0.850 | 0.833 | 0.853 | 0.845 |
| Negatives, voltage or Q waves | 24 | 0.674 | 0.684 | 0.667 | 0.648 | 0.677 |
| Negatives, other (mostly SR with ABQRS) | 71 | 0.596 | 0.599 | 0.590 | 0.584 | 0.586 |
| Positives, IRBBB | 24 | 0.873 | 0.821 | 0.855 | 0.825 | 0.816 |
| Positives, other CD | 12 | 0.838 | 0.860 | 0.856 | 0.854 | 0.855 |

- `pooled` loses in two ways: it scores normal ECGs with sinus bradycardia, tachycardia or arrhythmia and with
  ectopic beats as more abnormal, and it scores incomplete RBBB as less abnormal.
- The Challenge relabeling arms fix the sinus-variant negatives (below even `ptbxl`) but not the ectopy
  negatives or the IRBBB positives. Upweighting the dropped PTB-XL ECGs moves all three groups back toward
  `ptbxl`.
- By device, `pooled` lifts the hard negatives on every device with at least 20 hard ECGs, by 0.03 to 0.07
  (CS-12 E 0.505 to 0.537, AT-6 6 0.766 to 0.840). No device stands out, so device shift is not the
  explanation. The hard subset has almost no `CS100 3` ECGs.

## 4. Secondary encoders

The same pattern holds. Loss: JEPA +0.066 [+0.026, +0.106], CPC +0.069 [+0.028, +0.110].

| Arm | JEPA hard − `pooled` | JEPA SPH − `pooled` | JEPA verdict | CPC hard − `pooled` | CPC SPH − `pooled` | CPC verdict |
| --- | --- | --- | --- | --- | --- | --- |
| `dropped_upweighted` | +0.056 [+0.035, +0.078] | −0.001 | fix (0.86) | +0.041 [+0.022, +0.059] | −0.003 | fix (0.60) |
| `ptbxl_negative` | +0.051 [+0.018, +0.085] | −0.006 | trade-off | +0.051 [+0.004, +0.097] | −0.016 | trade-off |
| `secondary_negative` | +0.035 [+0.005, +0.065] | −0.005 | trade-off (0.53) | +0.040 [−0.002, +0.081] | −0.012 | no recovery |
| `ptbxl_rule` | +0.078 [+0.040, +0.119] | −0.019 | trade-off | +0.044 [−0.010, +0.097] | −0.031 | no recovery |
| `loso_chapman_ningbo` | +0.027 [+0.004, +0.049] | −0.015 | no recovery | +0.057 [+0.029, +0.087] | −0.013 | trade-off |

Recovery in parentheses. For CPC, Chapman/Ningbo carries most of the loss (leaving it out recovers 0.83),
which it does not for xECG or JEPA. JEPA's `secondary_negative` loses 0.0052 at SPH, just past the 0.005
line. Source indicators and prevalence matching recover nothing for any encoder.

## Prespecified reading

- Decision: `adopt_preferred_arm`, with `dropped_upweighted`. It is the only `fix` for xECG: recovery
  0.81, hard-subset +0.058 [+0.038, +0.080] with the interval above 0, SPH +0.000 [−0.000, +0.001], ordinary
  development −0.002 [−0.003, +0.000] (within the 0.005 flag). It is also a `fix` for JEPA and CPC.
- Confirmed on the analogues: +0.079 [+0.071, +0.088] on the 1,724 cross-fitted training ECGs.
- Trade-offs: `ptbxl_negative`, `ptbxl_rule`, `ptbxl_georgia` and `ptbxl_cpsc` recover more than half with
  intervals above 0 but cost 0.010-0.026 SPH AUROC.
- Prespecified meaning of a (d) fix: a weighting effect, not a single family and not a fix in the labels.

## What it means, in plain language

- The hard ECGs are normal recordings with a small extra finding (a sinus variant, an extra beat, a
  borderline form statement) and normal-looking recordings with incomplete RBBB. The Challenge training labels
  never show the first kind as normal: their negative is sinus rhythm alone, so every Challenge ECG with sinus
  bradycardia or an extra beat that reaches the readout is a positive.
- That label difference is real: relabeling the Challenge negatives the PTB-XL way removes most of the loss
  on the hard subset. But it costs about 0.01 AUROC at SPH, whose own normal (code 1 alone) is as strict as the
  Challenge's. The label definition matters in both directions.
- What PTB-XL itself contributes is its 1,724 training ECGs of exactly this hard kind. In the PTB-XL-only fit
  they are 10% of the rows; pooling dilutes them to 4.4%. Giving them back their 10% share (weight 2.3 each,
  0.94 for every other row) recovers 81% of the loss, keeps SPH at 0.939, and costs 0.002 on ordinary development. Removing them
  (`without_dropped`) drops the hard subset to 0.562, which is 025b's finding.
- So the loss is neither a device effect nor a fixed price of pooling. It is fixable, by weighting rather than
  by relabeling.

## What this means for the adopted pipeline

- The adopted readout (022b `pooled`, used by 030 and 033) can be refitted with `dropped_upweighted` at no
  measured cost at SPH: xECG hard subset 0.655 to 0.712, full PTB-XL development 0.927 to 0.931, SPH 0.939 to
  0.939. The 0.014 gap to PTB-XL-only on the hard subset stays, as does the 0.024 SPH gain of pooling.
- 030 and 033 were read at SPH with `pooled`; their SPH ranking should change very little (SPH AUROC differs
  by less than 0.001), but thresholds and referral budgets would need recomputing if the readout is swapped.
  That is the user's call; nothing was changed in those experiments.
- Keep the Challenge primary label for now. Harmonizing it to PTB-XL's normal helps PTB-XL and hurts SPH;
  which "normal" the university screen should use (is sinus bradycardia in a student normal?) is a question
  for the cardiologist, and the answer decides which label to train on.

## Surprises

- Prevalence was not the problem: matching the Challenge families to PTB-XL's 58% positive share changed
  nothing (−0.006), and neither did source indicators (+0.001).
- Leaving out Chapman/Ningbo helped CPC a lot (recovery 0.83) but xECG and JEPA much less (0.31, 0.40).
- Adding a single small family (Georgia or CPSC) cost less on the hard subset than adding Chapman/Ningbo, but
  cost much more at SPH, where Chapman/Ningbo is the useful family.
- `ptbxl_rule`, the fullest harmonization, cost 0.026 at SPH, more than either of its parts.

## Caveats

- The hard subset has 41 positives; its intervals are wide. The analogues are PTB-XL training ECGs scored by
  cross-fitting, not held-out patients, but they rank the arms the same way.
- The hypothesis was formed from the metadata of the same development ECGs that were scored, and PTB-XL
  development and SPH were read by earlier experiments. SPH served as a guard here, not a final test.
- The weight of `dropped_upweighted` was set by a prespecified rule (the share in the PTB-XL-only fit), not
  tuned; other weights were not tried. The arms differ in size and prevalence.
- One fit per arm and encoder; the bootstrap holds fits fixed. ECG-JEPA and xECG were pretrained without
  labels on Chapman and Ningbo waveforms. The labels are ECG annotation proxies, not confirmed disease or
  referral decisions. Device or filter shift was tested only indirectly (source indicator and per-device
  shifts), not with new features.

## Reproduce

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 uv run --no-sync python -m scripts.experiments.describe_hard_subset035
PYTHONPATH=. OMP_NUM_THREADS=1 uv run --no-sync python -u -m scripts.experiments.run_hard_subset035 --counts
PYTHONPATH=. OMP_NUM_THREADS=1 uv run --no-sync python -u -m scripts.experiments.run_hard_subset035
```

The runner refuses to overwrite an existing `result.json`. It needs the 020, 022 and 022b outputs, the
JEPA and xECG caches, `outputs/features_challenge_v1/` and the raw Chapman and Georgia records for the quality
check of the newly used rows.
