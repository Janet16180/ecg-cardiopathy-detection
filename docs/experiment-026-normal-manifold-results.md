# Experiment 026 results: one-class screening by distance from the normal-ECG manifold

Completed 28 September 2026 under the [frozen protocol](experiment-026-normal-manifold.md). Development
patients and SPH only; no PTB-XL calibration or test ECG was fitted or scored. CPU only, from cached frozen
features, one run. Local outputs are in `outputs/experiment026_normal_manifold_v1/` (`result.json` with every
input, source and protocol hash, `development_scores.csv`, `sph_scores.csv` and `run.log`).

**Integrity.**
- Every cache matched its extraction receipt. The Experiment 025 draws and predictions, and the Experiment 022
  SPH features and rows, matched their recorded hashes.
- The refitted `probe_all` of each encoder reproduced Experiment 025's saved N = all probabilities exactly
  (maximum difference 0.0).
- All 20 `probe_100` draws had Experiment 025's subset hashes and reproduced its AUROCs to 1.1e-16.
- No bootstrap draw was invalid in any task.

**Rows.**
- Fit set: 5,872 NORM-only training ECGs from 5,537 patients, from Experiment 025's pool of 15,359 ECGs.
- Development: 1,306 ECGs from 1,173 patients, 843 abnormal and 463 NORM-only.
- SPH: 21,008 ECGs from 20,364 patients, 7,190 positive.
- The held-out-superclass probes were trained on 10,982 ECGs without MI (5,110 positive), 11,198 without STTC
  (5,326), 11,784 without CD (5,912) and 13,242 without HYP (7,370).
- All counts match the protocol.

The 64 PCA components kept 94.8% of the standardized fit-set variance for ECG-JEPA, 90.1% for xECG and 91.7%
for CPC. Runtime was 1,602 seconds, with three encoder processes in parallel (JEPA 1,565 s, xECG 1,587 s,
CPC 1,547 s). The run completed on the first attempt.

All intervals are 95% paired whole-patient bootstrap intervals (2,000 draws, seed 26026). `probe_100` is the
mean over the 20 Experiment 025 draws with 100 labels.

## 1. Primary: Mahalanobis distance in the ECG-JEPA space

On the binary task, `mahalanobis` on ECG-JEPA scores AUROC **0.910** and AP **0.956**. The comparisons:

| Contrast | AUROC difference [95% interval] | AP difference [95% interval] |
| --- | --- | --- |
| `mahalanobis` - `probe_all` | -0.049 [-0.062, -0.038] | -0.023 [-0.030, -0.017] |
| `mahalanobis` - `probe_100` | -0.012 [-0.023, -0.000] | -0.004 [-0.009, +0.002] |

## 2. Every score and encoder, binary task

| Encoder | `mahalanobis` | `knn` | `probe_100` | `probe_all` | Mahalanobis - probe_all | Mahalanobis - probe_100 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| ECG-JEPA | 0.910 | 0.702 | 0.922 | 0.959 | -0.049 [-0.062, -0.038] | -0.012 [-0.023, -0.000] |
| xECG | 0.923 | 0.749 | 0.921 | 0.962 | -0.039 [-0.050, -0.028] | +0.002 [-0.008, +0.013] |
| CPC | 0.836 | 0.559 | 0.883 | 0.921 | -0.085 [-0.105, -0.066] | -0.047 [-0.068, -0.026] |

The values are AUROCs. The AP of `mahalanobis` is 0.956 (JEPA), 0.962 (xECG) and 0.917 (CPC). The AP of
`probe_all` is 0.979, 0.981 and 0.961.

`knn` is far below the probes on every encoder, by 0.21-0.36 AUROC against `probe_all`. Every `knn`
interval excludes 0.

## 3. Held-out conditions

In each task, S is the positive class and NORM-only ECGs are the negatives. `probe_without_S` never saw an
ECG with S. The table gives AUROCs and, in the last column, `mahalanobis` minus `probe_without_S`.

| S (positives) | Encoder | `mahalanobis` | `probe_without_S` | `probe_all` | Mahalanobis - probe_without_S |
| --- | --- | ---: | ---: | ---: | --- |
| MI (390) | ECG-JEPA | 0.930 | 0.955 | 0.970 | -0.025 [-0.038, -0.011] |
| | xECG | 0.950 | 0.956 | 0.975 | -0.006 [-0.017, +0.007] |
| | CPC | 0.866 | 0.933 | 0.945 | -0.066 [-0.090, -0.046] |
| STTC (360) | ECG-JEPA | 0.937 | 0.946 | 0.979 | -0.008 [-0.022, +0.005] |
| | xECG | 0.945 | 0.942 | 0.980 | +0.004 [-0.011, +0.018] |
| | CPC | 0.876 | 0.927 | 0.957 | -0.052 [-0.077, -0.029] |
| CD (348) | ECG-JEPA | 0.933 | 0.946 | 0.972 | -0.013 [-0.027, +0.001] |
| | xECG | 0.947 | 0.937 | 0.973 | +0.010 [-0.006, +0.028] |
| | CPC | 0.861 | 0.930 | 0.942 | -0.070 [-0.093, -0.047] |
| HYP (180) | ECG-JEPA | 0.924 | 0.918 | 0.959 | +0.006 [-0.018, +0.029] |
| | xECG | 0.911 | 0.913 | 0.956 | -0.001 [-0.025, +0.023] |
| | CPC | 0.860 | 0.878 | 0.900 | -0.018 [-0.058, +0.019] |

Every `knn` contrast with `probe_without_S` is negative, from -0.10 to -0.37, and every interval excludes 0.

S-only sensitivity: the positives here have no other abnormal superclass.

| S (positives) | Encoder | `mahalanobis` | `probe_without_S` | Mahalanobis - probe_without_S |
| --- | --- | ---: | ---: | --- |
| MI (164) | ECG-JEPA | 0.859 | 0.901 | -0.042 [-0.069, -0.015] |
| | xECG | 0.900 | 0.902 | -0.002 [-0.026, +0.024] |
| STTC (179) | ECG-JEPA | 0.893 | 0.897 | -0.004 [-0.031, +0.022] |
| | xECG | 0.906 | 0.890 | +0.016 [-0.011, +0.044] |
| CD (124) | ECG-JEPA | 0.877 | 0.880 | -0.003 [-0.034, +0.032] |
| | xECG | 0.906 | 0.865 | +0.040 [+0.006, +0.075] |
| HYP (41) | ECG-JEPA | 0.796 | 0.682 | +0.114 [+0.035, +0.195] |
| | xECG | 0.722 | 0.655 | +0.067 [-0.011, +0.155] |

CPC S-only differences are -0.085 (MI), -0.052 (STTC) and -0.101 (CD); each interval excludes 0. For HYP the
CPC difference is +0.093 [-0.023, +0.208]. HYP-only has 41 positives, so its intervals are wide.

## 4. Per superclass and subclass against the full probe

Superclass versus NORM, `mahalanobis` minus `probe_all` AUROC:

| S | ECG-JEPA | xECG | CPC |
| --- | --- | --- | --- |
| MI | -0.040 [-0.055, -0.027] | -0.025 [-0.035, -0.014] | -0.079 [-0.102, -0.059] |
| STTC | -0.041 [-0.055, -0.028] | -0.034 [-0.047, -0.023] | -0.082 [-0.106, -0.059] |
| CD | -0.039 [-0.053, -0.027] | -0.026 [-0.040, -0.013] | -0.082 [-0.105, -0.059] |
| HYP | -0.035 [-0.057, -0.016] | -0.044 [-0.067, -0.024] | -0.040 [-0.077, -0.005] |

Against `probe_100`, `mahalanobis` on xECG is within ±0.024 for every superclass, and every interval includes
0. On ECG-JEPA it is lower for MI (-0.015) and STTC (-0.024), with intervals excluding 0. It is not
distinguishable for CD (+0.002) or HYP (+0.017).

Subclass versus NORM (15 subclasses). The table gives AUROCs, and `mahalanobis` minus `probe_all` for ECG-JEPA
and xECG. `probe_100` is a point estimate only.

| Subclass (positives) | JEPA `mahalanobis` | JEPA `probe_100` | JEPA `probe_all` | JEPA difference | xECG `mahalanobis` | xECG difference | CPC `mahalanobis` |
| --- | ---: | ---: | ---: | --- | ---: | --- | ---: |
| CLBBB (40) | 1.000 | 0.998 | 1.000 | -0.000 [-0.001, +0.000] | 1.000 | -0.000 [-0.001, +0.000] | 0.990 |
| CRBBB (37) | 0.999 | 0.993 | 1.000 | -0.001 [-0.004, -0.000] | 1.000 | -0.000 [-0.002, +0.000] | 0.955 |
| LAO/LAE (32) | 0.949 | 0.933 | 0.950 | -0.001 [-0.032, +0.034] | 0.944 | -0.021 [-0.064, +0.010] | 0.900 |
| ISCA (66) | 0.988 | 0.990 | 0.996 | -0.008 [-0.022, +0.001] | 0.989 | -0.006 [-0.016, +0.004] | 0.954 |
| _AVB (60) | 0.964 | 0.963 | 0.975 | -0.011 [-0.033, +0.004] | 0.964 | -0.019 [-0.042, -0.003] | 0.963 |
| ISC_ (81) | 0.984 | 0.992 | 0.999 | -0.015 [-0.031, -0.003] | 0.994 | -0.005 [-0.009, -0.001] | 0.942 |
| AMI (229) | 0.971 | 0.971 | 0.992 | -0.021 [-0.033, -0.011] | 0.978 | -0.013 [-0.022, -0.006] | 0.912 |
| NST_ (47) | 0.948 | 0.955 | 0.972 | -0.025 [-0.044, -0.008] | 0.961 | -0.023 [-0.049, -0.004] | 0.946 |
| IVCD (47) | 0.914 | 0.927 | 0.947 | -0.034 [-0.069, -0.005] | 0.945 | +0.014 [-0.025, +0.061] | 0.890 |
| ISCI (32) | 0.943 | 0.966 | 0.982 | -0.039 [-0.080, -0.007] | 0.961 | -0.020 [-0.047, +0.000] | 0.923 |
| LAFB/LPFB (143) | 0.935 | 0.927 | 0.978 | -0.043 [-0.065, -0.023] | 0.954 | -0.029 [-0.049, -0.013] | 0.812 |
| LVH (136) | 0.915 | 0.899 | 0.957 | -0.043 [-0.068, -0.018] | 0.897 | -0.054 [-0.081, -0.030] | 0.849 |
| IMI (227) | 0.906 | 0.935 | 0.959 | -0.053 [-0.074, -0.032] | 0.932 | -0.035 [-0.052, -0.019] | 0.838 |
| STTC (152) | 0.887 | 0.935 | 0.962 | -0.075 [-0.101, -0.050] | 0.898 | -0.064 [-0.090, -0.040] | 0.782 |
| IRBBB (74) | 0.897 | 0.878 | 0.974 | -0.077 [-0.117, -0.041] | 0.910 | -0.061 [-0.098, -0.030] | 0.797 |

Bundle branch blocks are the easiest subclasses for the distance score, as for the probes. Nonspecific STTC,
IRBBB and IMI have the largest gaps.

## 5. SPH (external, scored once)

Primary label, 21,008 ECGs, 7,190 positive. The one-class scores were fitted on PTB-XL NORM ECGs only. The
probes are Experiment 022's heads, fitted on PTB-XL with all labels. No interval was computed for SPH.

| Encoder | `mahalanobis` AUROC / AP | `knn` AUROC / AP | Experiment 022 probe AUROC / AP |
| --- | --- | --- | --- |
| ECG-JEPA | 0.836 / 0.792 | 0.599 / 0.522 | 0.911 / 0.882 |
| xECG | 0.858 / 0.806 | 0.688 / 0.602 | 0.915 / 0.885 |
| CPC | 0.791 / 0.736 | 0.615 / 0.499 | 0.876 / 0.831 |

The table below gives SPH AUROCs per superclass, each against the SPH negatives.

| S (positives) | JEPA `mahalanobis` | JEPA probe | xECG `mahalanobis` | xECG probe | CPC `mahalanobis` | CPC probe |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MI (255) | 0.872 | 0.965 | 0.944 | 0.978 | 0.812 | 0.922 |
| STTC (5,037) | 0.858 | 0.925 | 0.871 | 0.927 | 0.820 | 0.900 |
| CD (2,365) | 0.811 | 0.893 | 0.847 | 0.899 | 0.751 | 0.840 |
| HYP (227) | 0.923 | 0.973 | 0.927 | 0.972 | 0.910 | 0.936 |

On PTB-XL development, `mahalanobis` trails the full probe by 0.04-0.05 on JEPA and xECG. At SPH it trails
by 0.057 (xECG) and 0.075 (JEPA); these are point differences with no interval.

## Prespecified reading

- **Primary (`mahalanobis`, ECG-JEPA): competitive.** The one-class AUROC is 0.049 below `probe_all`, just
  inside the 0.05 margin. The rule uses the point estimate; the interval [-0.062, -0.038] extends past the
  margin.
- **Primary, unseen conditions: useful for STTC, CD and HYP, not for MI.** The upper bound of the one-class
  minus `probe_without_S` interval is at least 0 for STTC (+0.005), CD (+0.001) and HYP (+0.029). It is
  -0.011 for MI. The S-only sensitivity gives the same pattern: MI below, STTC, CD and HYP not below.
- **Secondary, xECG `mahalanobis`:** competitive (-0.039), and useful for all four held-out superclasses. The
  S-only version agrees, and for CD-only it is above the held-out probe (+0.040 [+0.006, +0.075]).
- **Secondary, CPC `mahalanobis`:** not competitive (-0.085). It is useful only for HYP, where the interval is
  wide.
- **Secondary, `knn` on every encoder:** not competitive, and not useful for any held-out superclass. With
  only 41 HYP-only positives, the HYP-only sensitivity intervals include 0.

## Findings

- A Mahalanobis distance from the normal ECGs, fitted without a single abnormal label, reaches AUROC 0.910
  (ECG-JEPA) and 0.923 (xECG) on PTB-XL development. That is 0.04-0.05 below a probe trained on 15,359 labels.
  It is about equal to a probe trained on 100 labels: -0.012 for ECG-JEPA and +0.002 for xECG.
- When a probe has never seen a superclass, the xECG distance score detects that superclass about as well as
  the probe. For all four superclasses the intervals include 0. ECG-JEPA is similar except for MI, where the
  held-out probe stays 0.025 ahead.
- These scores depend on the encoder. On the CPC features, the same method is 0.085 below its probe. The
  mean-kNN cosine distance is weak on every encoder (0.56-0.75).
- The distance score is closest to the probe for conduction blocks (CLBBB, CRBBB, _AVB) and for anterior
  ischemia. It is furthest for nonspecific ST-T changes, IRBBB and inferior MI.
- At SPH, the one-class scores rank below the Experiment 022 probes: 0.836-0.858 against 0.911-0.915 for
  JEPA and xECG. Their gap is larger than on PTB-XL development. The xECG distance score still ranks SPH MI
  at 0.944.

## Caveats

- The development patients were inspected by earlier experiments; these results are exploratory. The primary
  reading rests on a point estimate 0.0006 inside the margin.
- NORM in PTB-XL is an ECG annotation, not proof of health. The fit set is an older clinical population, not
  a student cohort. Normal ECGs of young adults may differ from this reference, so the distances would need
  to be checked against normal ECGs from that population.
- A distance from normal flags anything unusual, including noise, device and demographic shifts. Experiment
  024 found device and heart rate strongly encoded in these spaces. A high score is not a diagnosis, and this
  experiment did not test which non-cardiac factors raise it.
- `probe_without_S` still sees conditions that co-occur with S. The S-only tables address this, but the
  HYP-only task has 41 positives.
- The larger gap at SPH comes from one run with no interval. SPH is far more band-pass filtered than PTB-XL,
  and this experiment did not test whether that shift explains the gap. No threshold, sensitivity or
  specificity is reported.
- Each encoder is one feature set (one checkpoint, preprocessing and pooling).

## Reproduce

```bash
PYTHONPATH=. OMP_NUM_THREADS=4 uv run --no-sync python -m scripts.experiments.run_normal_manifold026
```

The runner refuses to overwrite an existing `result.json`.
