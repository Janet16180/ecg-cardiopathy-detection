# Experiment 020 results: full-development readout with demographic baselines

Completed 28 September 2026 under the [frozen protocol and its v2 addendum](experiment-020-full-development-readout.md).
Development patients only; no calibration or test ECG was featurized or scored. Encoder: the unchanged
starting CPC encoder. Local outputs are in `outputs/experiment020_full_development_v2/`.

**Integrity.** The project-label heads reproduce Experiment 018 on the original 1,306 ECGs: AUROC is
identical at both budgets, and probabilities agree to 1.3e-6 (full) and 2.0e-7 (limited).

**Cohort.** Full development has 1,604 ECGs from 1,438 patients: the original 1,306, plus 298 fold-9 ECGs
the project label had dropped. 15 dropped ECGs of calibration patients stay closed. 32 ECGs have no
diagnostic statement and no standard label, which leaves 1,572 (884 positive) for standard-label analyses.

## 1. The dropped ECGs are the hard cases

AUROC on the standard superclass label (NORM only versus any MI, STTC, CD or HYP):

| Head | Full development (1,572) | Original (1,306) | Added (266, 41 positive) |
| --- | ---: | ---: | ---: |
| `cpc_project_full` (the historical head) | 0.879 | 0.921 | 0.506 |
| `cpc_standard` (trained on the standard label) | 0.889 | 0.916 | 0.575 |
| `age_sex_standard` | 0.756 | 0.779 | 0.503 |
| `cpc_age_sex_standard` | 0.890 | 0.917 | 0.571 |

On the ECGs the project label dropped, the historical head is at chance (0.506). Those ECGs are mostly
normal recordings with an extra rhythm or form statement, and abnormal findings that PTB-XL lists next to
NORM. Including them lowers the headline AUROC from 0.921 to 0.879. Training the head on the standard
label recovers part of the gap: +0.0098 AUROC on full development, 95% interval [+0.0051, +0.0146].

The 41 added negatives with sinus bradycardia or tachycardia explain some of the drop. Without them,
`cpc_project_full` scores 0.891 instead of 0.879 on full development.

## 2. The encoder is far better than age and sex alone

| Contrast on full development | AUROC difference | 95% interval |
| --- | ---: | --- |
| `cpc_standard` minus `age_sex_standard` | +0.132 | [+0.109, +0.155] |
| `cpc_age_sex_standard` minus `cpc_standard` | +0.001 | [-0.001, +0.003] |

Age and sex alone reach 0.756, so the demographic baseline is substantial. Still, the CPC features add
0.13 AUROC on top of it, and adding age and sex to the CPC features changes nothing measurable: the encoder
already carries that information.

## 3. Device

`cpc_standard` AUROC by device, for devices with at least 50 ECGs and 10 of each class:

| Device | ECGs | Positive | AUROC |
| --- | ---: | ---: | ---: |
| AT-6 C | 88 | 78 | 0.928 |
| AT-6 6 | 165 | 92 | 0.890 |
| CS-12 | 441 | 294 | 0.888 |
| AT-6 C 5.8 | 53 | 34 | 0.872 |
| AT-60 3 | 132 | 78 | 0.865 |
| AT-6 C 5.5 | 297 | 166 | 0.862 |
| CS-12 E | 330 | 96 | 0.822 |

A logistic probe on the frozen features identifies the `CS100 3` device on development ECGs with AUROC
0.974, although it has only 49 development ECGs, too few for its own disease AUROC. The features clearly
encode the device. This device makes up 35% of the training folds, and 74% of its labeled training ECGs are positive. The experiment
shows the shortcut is possible; it does not measure how much the head uses it.

## Interpretation

- Results reported on the project label overstate performance on PTB-XL as a whole. On the ECGs it drops,
  the model does no better than chance.
- The label definition matters more than recent architecture changes: training on the standard label
  improved full-development AUROC by about 0.01. Contrasts in Experiments 011-019 were typically 0.001-0.01.
- Demographics are not what the model relies on. The device signal is strong and deserves a controlled
  test, such as training without `CS100 3` or balancing device by label.
- This is one encoder and one seed. The original development patients were inspected before, and the added
  subset is small (41 positives), so its AUROC is imprecise.

## Reproduce

```bash
OMP_NUM_THREADS=1 uv run --no-sync python -m scripts.experiments.run_full_development_readout020 --stage profile
OMP_NUM_THREADS=1 uv run --no-sync python -m scripts.experiments.run_full_development_readout020 --stage run
```
