# Experiment 027 results: a calibrated screening threshold, checked on SPH

Completed 28 September 2026 under the [frozen protocol](experiment-027-calibrated-threshold.md), run once at
commit `0e3bf00`. Local outputs are in `outputs/experiment027_calibrated_threshold_v1/` (`result.json`
SHA-256 `7f9c6d0d…40b5`). No PTB-XL test (fold 10) waveform, feature, label or score was read.

## Integrity

- The three refitted heads reproduced 022's saved development probabilities: largest differences 5.6e-17
  (CPC), 2.2e-16 (JEPA) and 1.4e-16 (xECG), against a 1e-9 tolerance.
- Recomputed features of 32 training ECGs matched the caches: CPC 7.2e-7 against Experiment 020's saved
  features, JEPA and xECG 0.0, against a 1e-4 tolerance.
- Calibration features: ECG-JEPA from its cache (all 564 present); CPC (from the 250 Hz pool) and xECG
  (from the raw records) extracted on the V100. The run waited 295 s for the GPU lock (Experiment 023) and then
  used the GPU for 97 s. Total runtime was 692 s.
- Row counts matched the protocol: calibration 564 ECGs (504 patients, 348 positive), full development 1,572
  (884 positive), original development 1,306 (843 positive), SPH 21,008 (7,190 positive).

## Calibration fit

| Head | Platt slope | Platt intercept | Threshold (calibrated probability) |
| --- | ---: | ---: | ---: |
| `cpc_standard` | 1.113 | +0.410 | 0.244 |
| `jepa_standard` | 1.084 | +0.442 | 0.281 |
| `xecg_standard` | 1.111 | +0.582 | 0.248 |

On the calibration ECGs themselves (in sample) each threshold refers 331 of 348 positives (sensitivity 0.951),
with specificity 0.542 (CPC), 0.755 (JEPA) and 0.694 (xECG).

## Operating points

Sensitivity and specificity with 95% whole-patient bootstrap intervals (2,000 draws, seed 27027). PPV and NPV
at the set's own prevalence. "False referrals" are per 1,000 negative ECGs.

**SPH, primary label** (21,008 ECGs, prevalence 0.342):

| Head | Sensitivity | Specificity | PPV | NPV | False referrals / 1,000 negatives |
| --- | --- | --- | ---: | ---: | ---: |
| `cpc_standard` | 0.971 [0.967, 0.974] | 0.303 [0.295, 0.310] | 0.420 | 0.952 | 697 |
| `jepa_standard` | 0.916 [0.909, 0.922] | 0.663 [0.655, 0.671] | 0.586 | 0.938 | 337 |
| `xecg_standard` | 0.922 [0.916, 0.928] | 0.646 [0.638, 0.654] | 0.575 | 0.941 | 354 |

**PTB-XL full development** (1,572 ECGs, prevalence 0.562):

| Head | Sensitivity | Specificity | PPV | NPV | False referrals / 1,000 negatives |
| --- | --- | --- | ---: | ---: | ---: |
| `cpc_standard` | 0.930 [0.912, 0.945] | 0.490 [0.453, 0.527] | 0.701 | 0.845 | 510 |
| `jepa_standard` | 0.938 [0.920, 0.954] | 0.699 [0.663, 0.734] | 0.800 | 0.897 | 301 |
| `xecg_standard` | 0.951 [0.936, 0.965] | 0.663 [0.626, 0.699] | 0.784 | 0.914 | 337 |

**PTB-XL original development** (1,306 ECGs, prevalence 0.645; point estimates only):

| Head | Sensitivity | Specificity | PPV | NPV | False referrals / 1,000 negatives |
| --- | ---: | ---: | ---: | ---: | ---: |
| `cpc_standard` | 0.938 | 0.538 | 0.787 | 0.827 | 462 |
| `jepa_standard` | 0.947 | 0.762 | 0.879 | 0.887 | 238 |
| `xecg_standard` | 0.957 | 0.737 | 0.869 | 0.905 | 263 |

## At screening prevalences

Expected outcomes per 1,000 people screened, from each set's sensitivity and specificity.

| Set | Head | PPV at 5% | False referrals at 5% | Referrals at 5% | PPV at 1% | False referrals at 1% | Referrals at 1% |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SPH | `cpc_standard` | 0.068 | 662 | 711 | 0.014 | 690 | 700 |
| SPH | `jepa_standard` | 0.125 | 320 | 366 | 0.027 | 334 | 343 |
| SPH | `xecg_standard` | 0.120 | 337 | 383 | 0.026 | 351 | 360 |
| Development | `cpc_standard` | 0.088 | 485 | 531 | 0.018 | 505 | 514 |
| Development | `jepa_standard` | 0.141 | 286 | 333 | 0.031 | 298 | 307 |
| Development | `xecg_standard` | 0.129 | 320 | 368 | 0.028 | 334 | 343 |

## Calibration of the probabilities

| Set | Head | Brier | Intercept (ideal 0) | Slope (ideal 1) | ECE, 10 bins |
| --- | --- | ---: | ---: | ---: | ---: |
| SPH | `cpc_standard` | 0.202 | −1.85 | 0.78 | 0.237 |
| SPH | `jepa_standard` | 0.127 | −1.18 | 0.76 | 0.115 |
| SPH | `xecg_standard` | 0.124 | −1.13 | 0.75 | 0.105 |
| Development, full | `cpc_standard` | 0.136 | −0.23 | 0.79 | 0.053 |
| Development, full | `jepa_standard` | 0.102 | −0.24 | 0.83 | 0.030 |
| Development, full | `xecg_standard` | 0.100 | −0.33 | 0.81 | 0.032 |
| Development, original | `cpc_standard` | 0.113 | +0.18 | 0.95 | 0.032 |
| Development, original | `jepa_standard` | 0.082 | +0.21 | 1.05 | 0.025 |
| Development, original | `xecg_standard` | 0.078 | +0.13 | 1.03 | 0.019 |

At SPH every head's calibrated probabilities are far too high on average (intercepts −1.1 to −1.9) and too
spread out (slopes about 0.75). On the original development ECGs calibration is close to ideal.

## Specificity contrasts at the calibrated thresholds

| Contrast | SPH | Development, full |
| --- | --- | --- |
| xECG − CPC | +0.343 [+0.334, +0.352] | +0.173 [+0.136, +0.211] |
| JEPA − CPC | +0.360 [+0.351, +0.369] | +0.209 [+0.170, +0.247] |
| xECG − JEPA | −0.017 [−0.024, −0.011] | −0.036 [−0.066, −0.008] |

## Prespecified reading

- **The threshold does not transfer to SPH for any head.** No SPH sensitivity interval includes 0.95. For JEPA
  (0.916) and xECG (0.922) sensitivity fell below the target. For CPC it rose above it (0.971), and specificity
  fell to 0.303: CPC's threshold became more conservative, not less.
- **JEPA and xECG are both better for screening than CPC**, on SPH and on development: every interval of their
  specificity difference from CPC excludes 0.
- **JEPA is better for screening than xECG by the rule**, on SPH and on development (intervals exclude 0). The
  difference is small (1.7 and 3.6 points), and at these thresholds xECG also had slightly higher sensitivity
  (SPH 0.922 against 0.916; development 0.951 against 0.938). Each head sits at a different sensitivity, so
  this contrast mixes a small ranking difference with a small threshold difference; it is not a clean
  dominance.

## What this means for screening students

Strictly from these numbers, and with the caveats below:

- **A threshold chosen for 95% sensitivity on PTB-XL does not deliver 95% at a new hospital.** With JEPA or
  xECG it caught about 92% of SPH positives, so roughly 1 in 12 positive ECGs was missed instead of 1 in 20.
  With CPC it caught 97%, but it flagged 70% of the normal ECGs. A threshold would have to be re-chosen on
  calibration data from the population where it is used.
- **The encoder matters much more for the referral workload than AUROC suggests.** The AUROC gap between CPC
  and JEPA or xECG at SPH was about 0.04 (Experiment 022). At the screening threshold it becomes about 35
  points of specificity: roughly 700 against 340 false referrals per 1,000 people without the finding.
- **Most referrals would be false.** If 1% of screened students had a positive ECG and the SPH sensitivity and
  specificity held, JEPA would refer about 343 of every 1,000 students, of whom about 9 would be positive (PPV
  0.027); CPC would refer about 700 (PPV 0.014). At 5% prevalence, JEPA's PPV rises to 0.125. At any plausible
  student prevalence, this operating point refers about a third of everyone, which a screening programme would
  have to judge acceptable before using it.
- **The probabilities themselves should not be read as risks outside PTB-XL.** At SPH they overstate the
  chance of a positive ECG by a large margin (ECE 0.11 to 0.24), partly because the calibration patients have
  a much higher prevalence (62%) than SPH (34%), and a student population would be lower still.

## Caveats

- The calibration patients come from the same hospital and devices as training, so the threshold is an
  in-distribution choice; the SPH shortfall is exactly the kind of shift this cannot anticipate.
- The calibration set is small (348 positives): 17 missed positives define the threshold, so the threshold
  itself is imprecise. The bootstrap intervals hold the Platt fit and threshold fixed and do not include that
  uncertainty.
- SPH prevalence (34%) and the development prevalence (56%) are far above a student population. PPV and NPV at
  the observed prevalence do not carry over; the 5% and 1% rows assume sensitivity and specificity stay as
  measured, which this experiment shows is not guaranteed.
- SPH is heavily band-pass filtered compared with PTB-XL (see the [SPH EDA review](sph-echonext-eda-review.md)),
  and the SPH label follows the PTB-XL superclass rule only as closely as AHA codes allow.
- The label is an ECG annotation proxy, not confirmed disease or a referral decision.
- One fit per head, one run. PTB-XL development has been inspected many times.

## Reproduce

```bash
PYTHONPATH=. OMP_NUM_THREADS=4 uv run --no-sync python -u -m scripts.experiments.run_calibrated_threshold027
```

Two launch attempts failed before any score was computed: `/usr/bin/time` is not installed, and the worktree
lacked `third_party/` (encoder checkpoints), which was then linked read-only to the shared checkout's copy.
The third launch is the only run that computed scores.
