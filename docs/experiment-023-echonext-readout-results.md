# Experiment 023 results: echo-confirmed structural heart disease on EchoNext

Completed 28 September 2026 under the [frozen protocol with addenda v2 and v3](experiment-023-echonext-readout.md).
This is the project's first label that does not come from reading the ECG: structural heart disease measured
on an echocardiogram. No encoder was trained, nothing was tuned on validation patients, and the EchoNext test
split stayed closed. EchoNext is credentialed data, so this page holds aggregates only. Features and
predictions stay in `outputs/experiment023_echonext_v3/`.

**Integrity.**
- The cache receipt matched: release checksums, array and row hashes, and the build sources.
- The Experiment 020 heads and the Experiment 022 v3 ECG-JEPA and xECG PTB-XL heads were refitted. They
  reproduced their saved development probabilities exactly (maximum difference 0.0).
- ECG-JEPA and xECG features recomputed for 32 PTB-XL training ECGs through the historical 500 Hz path
  matched the caches exactly (difference 0.0).
- **250 Hz input diagnostic (reported, not a gate).** The same 32 ECGs were passed through the 250 Hz path
  used for EchoNext and compared with the cached features:
  - ECG-JEPA: mean cosine similarity 0.999995, minimum 0.999965;
  - xECG: mean 0.999998, minimum 0.999980.

  So starting at 250 Hz does not by itself change what these encoders see.
- Rows: 71,823 usable training ECGs from 26,023 patients (37,744 positive, prevalence 0.526), and 4,575
  usable validation ECGs, one per patient (1,974 positive, prevalence 0.431).
- The profile projected 4,435 s against the 7,200 s ceiling. Extraction took 1,881 s.

## 1. Primary: the frozen CPC features detect echo-confirmed disease

Composite `shd_moderate_or_greater_flag` on the usable validation ECGs:

| Head | AUROC | AP |
| --- | ---: | ---: |
| `xecg_tabular` | 0.840 | 0.812 |
| `xecg` | 0.838 | 0.809 |
| `jepa_tabular` | 0.824 | 0.793 |
| `jepa` | 0.823 | 0.790 |
| `cpc_tabular` | 0.813 | 0.780 |
| `cpc` (primary) | **0.812** | **0.779** |
| `ptb_cpc_standard` | 0.758 | 0.696 |
| `ptb_xecg_standard` | 0.758 | 0.672 |
| `ptb_jepa_standard` | 0.752 | 0.679 |
| `tabular` | 0.737 | 0.682 |
| `age_sex` | 0.655 | 0.570 |

AP should be read against the prevalence of 0.431.

## 2. Contrasts

Paired patient bootstrap, 2,000 draws, seed 23023:

| Contrast | AUROC difference | 95% interval |
| --- | ---: | --- |
| `cpc` minus `tabular` (primary) | +0.075 | [+0.063, +0.086] |
| `cpc_tabular` minus `tabular` (primary) | +0.076 | [+0.065, +0.087] |
| `cpc` minus `age_sex` | +0.157 | [+0.140, +0.175] |
| `jepa` minus `cpc` | +0.011 | [+0.005, +0.018] |
| `xecg` minus `cpc` | +0.027 | [+0.019, +0.033] |
| `jepa_tabular` minus `tabular` | +0.087 | [+0.076, +0.099] |
| `xecg_tabular` minus `tabular` | +0.103 | [+0.090, +0.115] |

No draw was invalid. Every interval excludes zero.

## 3. Secondary results

**All validation rows.** `cpc` scores AUROC 0.807 and AP 0.770 on all 4,626 validation ECGs (1,990
positive), including the 51 that failed the quality rules. On the usable rows it scores 0.812, so the
exclusions change little.

**PTB-XL heads applied unchanged.** The ECG-abnormality heads fitted on PTB-XL already rank echo-confirmed
disease well above chance:
- `ptb_cpc_standard` 0.758, `ptb_xecg_standard` 0.758 and `ptb_jepa_standard` 0.752;
- all three are above the cart measurements and demographics (`tabular`, 0.737);
- all three are 0.05-0.08 below heads fitted on EchoNext.

Unlike the fitted heads, the stronger encoders give no advantage here. An abnormal ECG and a diseased heart
overlap, but they are not the same target.

## 4. Component labels

These use rows with a measured echo value only. A separate head is fitted for each label on the usable
training rows. AUROC with AP in parentheses; there are no intervals.

| Component | Val positives / rows | CPC | Tabular | JEPA | xECG |
| --- | ---: | ---: | ---: | ---: | ---: |
| LVEF ≤ 45% | 857 / 4,077 | 0.859 (0.655) | 0.741 (0.433) | 0.888 (0.723) | 0.896 (0.756) |
| RV systolic dysfunction | 362 / 4,074 | 0.842 (0.366) | 0.728 (0.232) | 0.865 (0.437) | 0.871 (0.473) |
| Tricuspid regurgitation | 301 / 4,076 | 0.806 (0.315) | 0.732 (0.198) | 0.833 (0.340) | 0.843 (0.389) |
| Mitral regurgitation | 281 / 4,077 | 0.797 (0.246) | 0.757 (0.197) | 0.810 (0.239) | 0.823 (0.274) |
| Aortic stenosis | 251 / 4,072 | 0.791 (0.217) | 0.834 (0.271) | 0.827 (0.289) | 0.841 (0.302) |
| PASP ≥ 45 mmHg | 577 / 2,258 | 0.714 (0.448) | 0.629 (0.342) | 0.742 (0.490) | 0.757 (0.522) |
| TR velocity ≥ 3.2 m/s | 267 / 1,652 | 0.709 (0.306) | 0.629 (0.217) | 0.753 (0.358) | 0.761 (0.426) |
| LV wall thickness ≥ 13 mm | 867 / 4,066 | 0.724 (0.374) | 0.656 (0.319) | 0.736 (0.422) | 0.743 (0.425) |
| Pericardial effusion | 52 / 3,931 | 0.718 (0.030) | 0.666 (0.025) | 0.739 (0.037) | 0.712 (0.036) |
| Aortic regurgitation | 62 / 4,072 | 0.677 (0.027) | 0.658 (0.033) | 0.705 (0.054) | 0.767 (0.063) |

Pulmonary regurgitation had only 21 validation positives and was not scored, as prespecified.

What the component table shows:
- Reduced ejection fraction and right-ventricular dysfunction are the most detectable components.
- xECG is best or close to best on almost every component.
- Aortic stenosis is the one component where the tabular baseline beats CPC: 0.834 against 0.791. It is
  strongly age-related, which probably explains this. xECG slightly exceeds the baseline (0.841), and JEPA
  nearly matches it (0.827).
- The effusion and aortic-regurgitation rows have 52 and 62 positives, so they are imprecise.

## Interpretation

- **The primary question is answered yes.** A fixed logistic head on the unchanged CPC encoder detects
  echo-confirmed structural heart disease at AUROC 0.812. That is 0.075 above the ECG cart's own
  measurements with demographics, with an interval that clearly excludes zero.
- **Adding the tabular inputs to CPC adds almost nothing** (0.813 against 0.812). The waveform features
  already carry what the cart measurements and age hold.
- **The released frozen encoders are again better, but by less.** Compared with CPC:

  | Setting | ECG-JEPA | xECG |
  | --- | ---: | ---: |
  | EchoNext (this experiment) | +0.011 | +0.027 |
  | SPH ([Experiment 022](experiment-022-sph-external-readout-results.md)) | +0.036 | +0.039 |
  | PTB-XL at 1,000 labels ([Experiment 025](experiment-025-label-efficiency-results.md)) | +0.037 | +0.036 |

  On this target xECG is ahead of ECG-JEPA by 0.015 AUROC. That is a point estimate; the protocol did not
  prespecify that contrast. Here, unlike on PTB-XL and SPH, the two encoders are not tied. xECG with the
  tabular inputs is the best head overall (0.840).
- **Taken together with 022 and 025:** a frozen xECG with a simple readout remains the strongest
  representation the project has tested, now also on a label that does not come from reading an ECG.

## Caveats

- **Input shift.** EchoNext waveforms were median filtered, clipped and standardized by the dataset authors,
  and have no physical unit.
  - The per-lead mapping onto PTB-XL statistics recovers the scale an encoder expects, but not the filtering.
  - The smaller JEPA and xECG margins may partly reflect how each encoder reacts to that preprocessing,
    rather than the target. This experiment cannot separate the two.
- **Population.** EchoNext comes from one academic centre, and every ECG belongs to a patient who had an
  echocardiogram. Prevalence is 0.43 in validation. These AUROCs describe that referred population, not
  screening.
- **Components.** A component is filled only for ECGs within a year of an echo that measured it. PASP and TR
  velocity are measured on far fewer rows, so those rows form a different subset. Component results have no
  intervals and no multiplicity control.
- **Tabular baseline.** It uses the cart's automated measurements. Those serve only as a comparator here, not
  as a label.
- **Scope.** Development validation only, with one fit per head and one seed per encoder. There is no
  calibration or threshold, so sensitivity and specificity are not reported. The test split was not opened.

## Reproduce

```bash
.venv/bin/python scripts/experiments/run_echonext_readout023.py --stage profile
.venv/bin/python scripts/experiments/run_echonext_readout023.py --stage run
```

The run's extraction took 1,881 s:

| Part | Train (s) | Validation (s) |
| --- | ---: | ---: |
| Read | 265 | 18 |
| CPC | 53 | 4 |
| ECG-JEPA | 620 | 40 |
| xECG | 827 | 53 |
