# Experiment 027b results: calibrating the screening threshold on several hospitals

Completed 29 September 2026 under the [frozen protocol](experiment-027b-multisource-calibration.md)
(frozen at commit `1a9a5b3`), run once at commit `d5b5d4c` on CPU in 85 s. Local outputs are in
`outputs/experiment027b_multisource_calibration_v1/` (`result.json` SHA-256 `49cd8115…1958`). No PTB-XL
test ECG and no Challenge test-group ECG was read.

## Integrity

- The three refitted PTB-XL heads reproduced 022's development probabilities (largest differences 5.6e-17,
  2.2e-16 and 1.4e-16, against 1e-9).
- The `ptbxl` arm reproduced 027 exactly: Platt slopes and intercepts, thresholds and all 21,008 calibrated
  SPH probabilities differ by 0.0.
- Calibration counts matched the protocol: PTB-XL 564 (348 positive); Challenge 7,612 (Chapman 1,018, Ningbo
  3,414, Georgia 1,718, CPSC 2018 872, CPSC-Extra 590), 78% positive. SPH 21,008 ECGs of 20,364 patients;
  all 2,000 bootstrap draws were valid.

## Calibration fits

| Arm | CPC slope / intercept / threshold | JEPA | xECG |
| --- | --- | --- | --- |
| `ptbxl` (027) | 1.11 / +0.41 / 0.244 | 1.08 / +0.44 / 0.281 | 1.11 / +0.58 / 0.248 |
| `pooled` (primary) | 0.83 / +0.30 / 0.389 | 0.85 / +0.59 / 0.392 | 0.75 / +0.68 / 0.388 |
| `balanced` | 0.85 / +0.51 / 0.394 | 0.83 / +0.67 / 0.388 | 0.77 / +0.74 / 0.387 |
| `loso_chapman_ningbo` | 0.78 / +0.83 / 0.470 | 0.70 / +0.82 / 0.445 | 0.65 / +0.89 / 0.460 |
| `loso_georgia` | 0.87 / +0.23 / 0.376 | 0.95 / +0.74 / 0.376 | 0.84 / +0.81 / 0.373 |
| `loso_cpsc` | 0.86 / +0.05 / 0.368 | 0.89 / +0.31 / 0.375 | 0.76 / +0.43 / 0.352 |

The thresholds are on each arm's own calibrated scale, so they are not comparable across arms; what matters
is which ECGs they refer. In sample every arm refers 95.0-96.0% of its own calibration positives.

## SPH, primary label

Sensitivity and specificity with 95% whole-patient bootstrap intervals (2,000 draws, seed 29029).

| Arm | Head | Sensitivity | Specificity | Brier | ECE | Intercept | Slope |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `ptbxl` | CPC | 0.971 [0.966, 0.974] | 0.303 [0.295, 0.311] | 0.202 | 0.237 | −1.85 | 0.78 |
| `ptbxl` | JEPA | 0.916 [0.909, 0.922] | 0.663 [0.655, 0.671] | 0.127 | 0.115 | −1.18 | 0.76 |
| `ptbxl` | xECG | 0.922 [0.916, 0.928] | 0.646 [0.637, 0.654] | 0.124 | 0.105 | −1.13 | 0.75 |
| `pooled` | CPC | 0.944 [0.938, 0.949] | 0.446 [0.437, 0.454] | 0.195 | 0.236 | −1.55 | 1.05 |
| `pooled` | JEPA | 0.912 [0.905, 0.919] | 0.676 [0.668, 0.684] | 0.140 | 0.164 | −1.35 | 0.98 |
| `pooled` | xECG | 0.920 [0.914, 0.926] | 0.649 [0.640, 0.657] | 0.140 | 0.168 | −1.30 | 1.11 |
| `balanced` | CPC | 0.956 [0.951, 0.961] | 0.383 [0.374, 0.391] | 0.215 | 0.269 | −1.78 | 1.02 |
| `balanced` | JEPA | 0.920 [0.914, 0.926] | 0.646 [0.638, 0.654] | 0.146 | 0.177 | −1.43 | 1.00 |
| `balanced` | xECG | 0.925 [0.919, 0.931] | 0.635 [0.626, 0.643] | 0.144 | 0.175 | −1.37 | 1.08 |
| `loso_chapman_ningbo` | CPC | 0.964 [0.960, 0.969] | 0.351 [0.344, 0.359] | 0.254 | 0.326 | −2.05 | 1.11 |
| `loso_chapman_ningbo` | JEPA | 0.930 [0.924, 0.936] | 0.612 [0.604, 0.621] | 0.169 | 0.226 | −1.59 | 1.18 |
| `loso_chapman_ningbo` | xECG | 0.927 [0.922, 0.933] | 0.623 [0.615, 0.631] | 0.167 | 0.223 | −1.54 | 1.28 |
| `loso_georgia` | CPC | 0.940 [0.934, 0.945] | 0.463 [0.455, 0.471] | 0.188 | 0.223 | −1.52 | 0.99 |
| `loso_georgia` | JEPA | 0.918 [0.912, 0.925] | 0.654 [0.646, 0.662] | 0.145 | 0.168 | −1.49 | 0.87 |
| `loso_georgia` | xECG | 0.925 [0.920, 0.932] | 0.631 [0.622, 0.639] | 0.143 | 0.171 | −1.42 | 0.99 |
| `loso_cpsc` | CPC | 0.927 [0.920, 0.933] | 0.510 [0.502, 0.519] | 0.173 | 0.193 | −1.32 | 1.01 |
| `loso_cpsc` | JEPA | 0.887 [0.880, 0.895] | 0.734 [0.726, 0.742] | 0.125 | 0.121 | −1.07 | 0.94 |
| `loso_cpsc` | xECG | 0.911 [0.905, 0.918] | 0.681 [0.674, 0.689] | 0.125 | 0.130 | −1.05 | 1.09 |

### Primary comparison: `pooled` minus `ptbxl`

`D` is the change in `|sensitivity − 0.95|`; negative means closer to the target.

| Head | D | Sensitivity change | Specificity change | Brier change | ECE change | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| CPC | −0.014 [−0.023, −0.006] | −0.027 [−0.030, −0.023] | +0.143 [+0.137, +0.149] | −0.007 [−0.008, −0.007] | −0.001 [−0.002, −0.001] | better |
| JEPA | +0.004 [+0.002, +0.005] | −0.004 [−0.005, −0.002] | +0.013 [+0.011, +0.015] | +0.013 [+0.013, +0.014] | +0.049 [+0.048, +0.050] | worse |
| xECG | +0.002 [+0.001, +0.003] | −0.002 [−0.003, −0.001] | +0.003 [+0.002, +0.004] | +0.017 [+0.016, +0.018] | +0.063 [+0.062, +0.064] | worse |

### Secondary arms, D against `ptbxl`

| Arm | CPC | JEPA | xECG |
| --- | --- | --- | --- |
| `balanced` | −0.014 [−0.017, −0.012] | −0.004 [−0.006, −0.003] | −0.003 [−0.004, −0.002] |
| `loso_chapman_ningbo` | −0.006 [−0.008, −0.005] | −0.014 [−0.017, −0.012] | −0.005 [−0.007, −0.004] |
| `loso_georgia` | −0.010 [−0.019, −0.001] | −0.002 [−0.004, −0.001] | −0.003 [−0.005, −0.002] |
| `loso_cpsc` | +0.003 [−0.006, +0.012] | +0.028 [+0.025, +0.032] | +0.011 [+0.008, +0.013] |

Where JEPA or xECG came closer to 0.95, specificity fell by 0.9 to 5.1 points, and the Brier score and ECE
got worse in every such case.

## Held-out family (secondary new-site replication)

Each Challenge family's calibration-group ECGs, scored by the PTB-XL-only arm and by the arm calibrated on
PTB-XL plus the other two families. Record-level bootstrap (2,000 draws, seed 29029).

| Held-out family (ECGs, positive) | Head | Sensitivity, `ptbxl` | Sensitivity, `loso` | Specificity, `ptbxl` | Specificity, `loso` | D |
| --- | --- | --- | --- | --- | --- | --- |
| Chapman/Ningbo (4,432, 3,254) | CPC | 0.985 [0.980, 0.989] | 0.980 [0.975, 0.984] | 0.262 | 0.312 | −0.005 [−0.007, −0.003] |
| | JEPA | 0.962 [0.955, 0.968] | 0.970 [0.964, 0.975] | 0.670 | 0.626 | +0.008 [+0.005, +0.011] |
| | xECG | 0.957 [0.950, 0.964] | 0.959 [0.953, 0.966] | 0.646 | 0.625 | +0.002 [+0.001, +0.004] |
| Georgia (1,718, 1,372) | CPC | 0.958 [0.947, 0.968] | 0.937 [0.924, 0.949] | 0.347 | 0.465 | +0.006 [−0.015, +0.024] |
| | JEPA | 0.969 [0.960, 0.978] | 0.969 [0.960, 0.978] | 0.408 | 0.396 | +0.000 [+0.000, +0.000] |
| | xECG | 0.964 [0.954, 0.974] | 0.966 [0.956, 0.975] | 0.425 | 0.416 | +0.001 [+0.000, +0.004] |
| CPSC (1,462, 1,279) | CPC | 0.955 [0.943, 0.966] | 0.901 [0.884, 0.917] | 0.421 | 0.601 | +0.045 [+0.019, +0.063] |
| | JEPA | 0.912 [0.897, 0.927] | 0.892 [0.875, 0.909] | 0.765 | 0.798 | +0.020 [+0.013, +0.028] |
| | xECG | 0.922 [0.907, 0.937] | 0.912 [0.896, 0.927] | 0.694 | 0.732 | +0.010 [+0.005, +0.016] |

Without the 70 Ningbo records that have a lead stored as zero, the Chapman/Ningbo rows change by at most
0.012 in specificity and 0.001 in sensitivity, and every D is the same to three decimals.

The calibration intercepts on the held-out families run in both directions: near 0 at Chapman/Ningbo and
Georgia, and +1.3 to +1.9 at CPSC (probabilities too low, prevalence 87%), against −1.1 to −1.9 at SPH
(probabilities too high, prevalence 34%).

## Prespecified reading

- **Primary: multi-hospital calibration is not adopted.** `pooled` brought CPC closer to 95% but moved JEPA
  and xECG slightly further away (intervals of D above 0). The rule needed two of three heads better and none
  worse. PTB-XL-only calibration stays the reference.
- **No arm's threshold transfers to SPH for any head.** No SPH sensitivity interval includes 0.95. The
  closest JEPA or xECG result was `loso_chapman_ningbo` JEPA at 0.930 [0.924, 0.936].
- The secondary `balanced`, `loso_chapman_ningbo` and `loso_georgia` arms moved every head closer to 0.95, by
  0.2 to 1.4 points for JEPA and xECG. `loso_cpsc` moved JEPA and xECG further away. These are descriptive; no
  decision attaches to them.
- **Held-out families:** calibrating on the other hospitals did not bring a held-out hospital closer to 95%
  than PTB-XL alone for any head, except CPC at Chapman/Ningbo (by 0.5 points). At CPSC it was worse for all
  three heads.

## What this means for screening at a new site

- **Adding hospitals to the calibration set does not fix the operating point at a hospital it has not seen.**
  For JEPA and xECG, the multi-hospital threshold refers almost exactly the same SPH ECGs as the PTB-XL
  threshold: SPH sensitivity stays at 0.91-0.92, about 1 positive in 12 missed instead of 1 in 20.
- **The shortfall is a property of the new site, not of a too-narrow calibration set.** The PTB-XL threshold
  already gave 96-97% sensitivity at Chapman/Ningbo and Georgia, 91-92% at CPSC, and 92% at SPH. The pooled
  hospitals agree with PTB-XL on where the 95% cut lies, so pooling them cannot anticipate a site where it
  lies elsewhere. SPH is heavily band-pass filtered, and CPSC is read through a centred window of longer
  records, which are plausible reasons for their shifts.
- **CPC is the exception because its PTB-XL threshold was too conservative at SPH** (0.971, specificity
  0.303). Multi-hospital calibration pulled it to 0.944 and cut false referrals from about 700 to 550 per
  1,000 negative ECGs, but CPC still refers far more people than JEPA or xECG (specificity 0.45 against
  0.65-0.68).
- **Multi-hospital calibration makes the probabilities worse as risks.** The slopes at SPH move to about 1
  (from 0.75), but the pooled set is 78% positive, so the intercepts drop further (−1.3 to −1.6) and the ECE
  of JEPA and xECG rises from 0.11 to 0.16-0.17. A student population, with far lower prevalence, would see
  even more inflated probabilities.
- **The practical answer is local recalibration.** A new site, such as the university, needs its own labeled
  ECGs to choose the threshold; the backlog item `site_recalibration` asks how many are needed. Until then, a
  threshold chosen elsewhere should be expected to miss 95% by several points in either direction.

## Surprises

- The pooled Platt slopes are below 1 (0.75-0.85) on the calibration set but about 1 at SPH, while 027's
  PTB-XL slopes were above 1 in calibration and 0.75 at SPH. Mixing hospitals spreads the calibration set's
  scores, which happens to match SPH's spread, but not its prevalence.
- Georgia is an easy site for sensitivity and a hard one for specificity: every head refers 96-97% of
  positives but only 35-47% of negatives pass. Its primary negatives (sinus rhythm alone) are few and may be
  a harder normal than PTB-XL's.

## Caveats

- The Challenge split is by record; patients may repeat across groups. SPH, the primary test, is bootstrapped
  by patient and does not depend on this.
- ECG-JEPA and xECG were pretrained on Chapman and Ningbo without labels; the Chapman/Ningbo held-out readout
  is not an unseen-hospital test for those two encoders. CPC has seen no Challenge record.
- The Challenge negative (sinus rhythm alone) differs from PTB-XL's NORM and SPH's normal code, and CPSC-Extra
  has no negatives, so the pooled prevalence (78%) is far from SPH's (34%). This explains the worse
  intercepts; it is part of what pooling these sources means, not an error in the fit.
- CPSC records longer than 10 s were read through one centred window; the CPSC held-out readout rests on 183
  negatives.
- One fit per arm and head, one run. The bootstrap holds the fits fixed and does not include their
  uncertainty. The label is an ECG annotation proxy, not confirmed disease or a referral decision.

## Reproduce

```bash
PYTHONPATH=. OMP_NUM_THREADS=4 uv run --no-sync python -u -m scripts.experiments.run_multisource_calibration027b
```

The run needs `third_party/` (the encoder checkpoints, which the feature caches are checked against); in the
worktree it was linked read-only to the shared checkout's copy. The Challenge features are the final
extraction of `outputs/features_challenge_v1/` (centred 10 s windows, extraction code of commit `022d920`),
whose hashes the run records in its identity.
