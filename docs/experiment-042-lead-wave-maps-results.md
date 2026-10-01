# Experiment 042 results: per-lead maps from ECG-JEPA patches and beat-aligned waves

Completed 30 September 2026 under the [frozen protocol](experiment-042-lead-wave-maps.md) (commit `175356a`,
Amendment 1 in `1f6a3da`; run from `810a3e7`). PTB-XL development data only; no calibration or test ECG was
read. Outputs are in `outputs/experiment042_lead_wave_maps_v1/` (`result.json` with every input, source and
protocol hash, `unit_scores.npz`, `figures/` and `run.log`); figures are copied to
[docs/figures/experiment-042/](figures/experiment-042/). The run took 544 seconds, including 53 seconds of
model time to extract ECG-JEPA tokens of 7,476 ECGs. This experiment also serves as the 041b fallback.

## Readings

| Arm | Keeps premature-beat localization | Lead gain | Detection gain | Benign gain | Improves on 041 |
| --- | --- | --- | --- | --- | --- |
| B, beat-aligned waves (`U_B`) | **Yes**: +0.066 [-0.040, +0.166] | No | No | **Yes**: -0.192 [-0.346, -0.038] | **Yes** |
| J, ECG-JEPA patches (`U_J`) | No: -0.404 [-0.539, -0.267] | No | No | No: -0.135 [-0.288, 0.000] | No |

Arm B improves on Experiment 041, so the protocol's demonstration notebook is written for arm B only
([`notebooks/12-jr-beat-wave-maps.ipynb`](../notebooks/12-jr-beat-wave-maps.ipynb)). Neither arm passed the
primary lead test.

## Integrity

- Experiment 026's whole-ECG JEPA Mahalanobis reproduced to a relative 2.3e-16, `probe_all` to 1.7e-16
  (OpenBLAS Haswell kernels). The streaming Mahalanobis matched `normal_manifold.fit_mahalanobis` to a
  relative 7.9e-15.
- For all 7,178 extracted ECGs in the JEPA cache, the token mean equals the cached feature to 1.5e-6.
- Token order: adding 1 mV to (lead 0, patch 0), (lead 1, patch 25) and (lead 7, patch 49) changed tokens 0,
  75 and 399 the most, in all four check ECGs.
- The mean `G_J` contribution equals the cached-feature logit to 3.9e-6; the 48 `G_B` shares sum to the logit
  to 1.4e-14.
- The 041 section scores matched their receipt; the scored rows equal 041's, in order.
- Arm B kept 63,479 fit-set beats. Every fit, pool and scored ECG had at least two kept beats, so no ECG
  was excluded. As in 041, 73 of the 84 PVC ECGs had a rule-detected premature beat.

## Amendment 1

The training-only smoke test showed that one covariance pooled over the 8 JEPA leads lets lead II dominate.
In a training check, lead II held the top token for 79 of 150 normal ECGs, and the worst-token AUROC was
0.319; with one reference per lead it was 0.735. The protocol was amended before any development score, and
`U_J` and `U_J_kmeans` use one reference per lead.

## Premature beats: the beat-aligned map is the most precise

| Map | Top unit on a premature beat | Chance | Hit - chance [95% interval] | Minus 041 `U` [95% interval] |
| --- | ---: | ---: | --- | --- |
| 041 `U` (time sections) | 0.822 | 0.153 | +0.669 [0.582, 0.758] | |
| `U_B` | 0.932 | 0.197 | +0.734 [0.671, 0.791] | +0.066 [-0.040, +0.166] |
| `U_J` | 0.411 | 0.147 | +0.264 [0.142, 0.380] | -0.404 [-0.539, -0.267] |
| `U_J_kmeans` | 0.260 | 0.147 | +0.114 [0.012, 0.216] | -0.555 [-0.679, -0.428] |
| `G_J` | 0.575 | 0.147 | +0.429 [0.311, 0.540] | -0.240 [-0.390, -0.099] |

## Lead localization: no arm passed the primary test

Top lead in V1-V4 (anterior hit) and in II, III or aVF (inferior hit; arm J has only II), anterior-only (146)
against inferior-only (149) infarcts. The primary test is the anterior contrast of `U_B` and `U_J`.

| Map | Anterior hit: AMI / IMI | Anterior contrast [95% interval] | Inferior hit: IMI / AMI | Inferior contrast [95% interval] |
| --- | --- | --- | --- | --- |
| `U_B` | 0.534 / 0.450 | +0.085 [-0.038, +0.202] | 0.396 / 0.281 | +0.115 [-0.002, +0.230] |
| `U_B_median` | 0.486 / 0.403 | +0.084 [-0.038, +0.196] | 0.309 / 0.219 | +0.090 [-0.014, +0.194] |
| `G_B` | 0.384 / 0.329 | +0.055 [-0.055, +0.168] | 0.221 / 0.110 | +0.112 [0.026, 0.195] |
| `U_J` | 0.479 / 0.416 | +0.063 [-0.051, +0.182] | 0.154 / 0.075 | +0.079 [0.007, 0.154] |
| `U_J_kmeans` | 0.589 / 0.483 | +0.106 [-0.009, +0.227] | 0.128 / 0.082 | +0.045 [-0.026, +0.115] |
| `G_J` | 0.240 / 0.201 | +0.038 [-0.057, +0.134] | 0.114 / 0.034 | +0.080 [0.021, 0.138] |

Every point estimate is in the expected direction, and three secondary inferior contrasts (`G_B`, `U_J`,
`G_J`) have intervals above zero. But anterior leads hold the top unit in 40-50% of inferior infarcts too:
the top lead says little about where the infarct is.

## Detection, red marks and benign variants

| Map | AUROC (worst unit) | Minus 041 `U` [95% interval] | Any red: normal | Positive | Benign | Benign minus 041 `U` [95% interval] |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| 041 `U` | 0.777 | | 0.052 | 0.230 | 0.269 | |
| `U_B` | 0.740 | -0.036 [-0.069, 0.000] | 0.052 | 0.286 | 0.077 | -0.192 [-0.346, -0.038] |
| `U_B_median` | 0.755 | -0.021 [-0.056, +0.012] | 0.052 | 0.321 | 0.058 | -0.212 [-0.346, -0.096] |
| `G_B` | 0.810 | +0.033 [-0.001, +0.067] | 0.052 | 0.388 | 0.096 | -0.173 [-0.308, -0.058] |
| `U_J` | 0.745 | -0.032 [-0.069, +0.004] | 0.052 | 0.146 | 0.135 | -0.135 [-0.288, 0.000] |
| `U_J_kmeans` | 0.802 | +0.025 [-0.011, +0.058] | 0.052 | 0.196 | 0.115 | -0.154 [-0.288, -0.038] |
| `G_J` | 0.912 | +0.135 [0.108, 0.164] | 0.052 | 0.693 | 0.038 | -0.231 [-0.365, -0.115] |

Mean red units per ECG: `U_B` 0.31 in normal ECGs, 9.5 in positives and 16.8 in premature-beat ECGs (out of
549 units on average: about 11 beats, 12 leads, 4 waves); `G_J` 14.6 in positives out of 400.

## Example figures

| Example | `U_B` red units | `U_J` | `G_J` |
| --- | --- | --- | --- |
| [NORM 47](figures/experiment-042/NORM_47.png) | none | none | none |
| [PVC 219](figures/experiment-042/PVC_219.png) | 2: QRS of the premature beat (8.30 s) in II and aVF | none | 1, lead I at 4.8 s |
| [IMI 8](figures/experiment-042/IMI_8.png) | none | none | none |
| [AMI 184](figures/experiment-042/AMI_184.png) | none | none | 26 |
| [CLBBB 287](figures/experiment-042/CLBBB_287.png) | 1: V1 ST at 9.18 s | 1: V1 at 6.0 s | 51 |
| [LVH 30](figures/experiment-042/LVH_30.png) | none | none | none |
| [Benign 69](figures/experiment-042/benign_69.png) | none | none | none |

## Interpretation

- Cutting the ECG at real beats and comparing each wave with the same wave of healthy beats gave the most
  precise map so far. Its top unit was on the premature beat in 93% of PVC ECGs, and it put red on 7.7% of
  benign-variant ECGs, against 27% for 041's time sections. A slow or irregular sinus rhythm changes the
  spacing of beats, not their shape, and the beat-aligned map compares shapes only.
- Its worst unit is still a weak screen (0.740). Like 041, it finds one strange beat, not a finding spread
  over every beat.
- ECG-JEPA's patch tokens did not localize premature beats nearly as well (41% top-patch hits). Its
  label-guided map detects well (0.912) and rarely marks benign variants (3.8%), but on positives it marks
  broadly (14.6 of 400 patches).
- The infarct lead test is weak evidence either way. The statements give no region, old infarcts can be
  subtle, and the top-lead summary is crude. The consistently positive point estimates suggest some lead
  information, but nothing here shows that a map points to the infarct's leads.

## Caveats

- Arm B's wave windows are fixed offsets from R. In wide-QRS rhythms (bundle branch block, PVC), the QRS spills
  into the ST window, and the T window does not follow the QT interval.
- Arm J cannot mark III, aVR, aVL or aVF.
- The benign-variant set is small (52; one ECG moves the share by 1.9 points), and the premature-beat rule is
  automatic.
- Development data were read by earlier experiments; the results are exploratory.

## Follow-ups (added to the backlog)

- `section_map_with_ecg_score` (updated): screen with the whole-ECG readout and explain with `U_B`.
- `rate_adaptive_wave_windows`: QRS and T windows that follow the measured QRS width and QT interval.
- `cardiologist_region_marks`: a page where the cardiologist marks abnormal leads and waves on 30-50 ECGs, the
  ground truth the lead test lacks.
- `jepa_patch_localization041b`: closed by this experiment.
