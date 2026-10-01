# Experiment 049 results: a label-free focal-versus-diffuse explanation switch

Completed 1 October 2026 under the [frozen protocol](experiment-049-focal-switch.md) (commit `adb5d08`; run
from `2dc1a53`). Saved scores and saved head parameters only: nothing was trained. Besides saved scores and
cached xECG features, the run read only the 84 PVC records' PTB-XL waveforms (041's premature-beat
windows). SPH features were read only for the integrity checks. No calibration, test or EchoNext record was
read. Outputs are in `outputs/experiment049_focal_switch_v1/` (`result.json` with every input, source and
protocol hash, `explanations.npz`, `run.log`). The run took 90 seconds on CPU, 74 of them in loading and
integrity checks.

## Readings

| Case | `focal_switch` hit - chance minus PVC switch's (lower bound > -0.10) | Anterior contrast minus PVC switch's (lower bound > 0) | Against the PVC switch | Against `U_B` alone |
| --- | --- | --- | --- | --- |
| `logistic_concat`, q = 0.975 (primary) | -0.405 [-0.536, -0.275] | -0.002 [-0.110, +0.102] | **Does not improve** | Does not improve |
| q = 0.95 | -0.212 [-0.336, -0.108] | +0.018 [-0.088, +0.122] | Does not improve | Does not improve |
| q = 0.99 | -0.693 [-0.814, -0.563] | +0.007 [-0.111, +0.120] | Does not improve | Does not improve |
| E referral, q = 0.975 | -0.407 [-0.542, -0.276] | +0.002 [-0.108, +0.109] | Does not improve | Does not improve |
| `combined_50` referral, q = 0.975 | -0.478 [-0.577, -0.371] | -0.013 [-0.125, +0.093] | Does not improve | Does not improve |

The focal switch fails both primary conditions in every case. The secondary arms give the same reading:

- `focal_and_pvc` has the same premature-beat numbers as `focal_switch` and fails both conditions.
- `focal_or_pvc` keeps premature-beat localization (difference 0.000), but its anterior gain over the PVC
  switch has an interval that includes zero, so it does not improve on the PVC switch.
- Against `U_B` alone, `focal_or_pvc` and the PVC switch improve (048's reading again); `focal_switch` and
  `focal_and_pvc` lose premature-beat localization.

No example figures were drawn.

## Integrity

- All of 048's checks passed again: 047's checks, 044's file hashes, and z_PVC on SPH against 044 (largest
  difference 4.3e-15).
- z_WPW on SPH reproduces 044's `sph_v2_z_wpw` (largest difference 1.3e-15), and F reproduces
  `sph_v2_z_combined` (4.3e-15), within 1e-10.
- 048's `explanations.npz` matches its receipt, and its `z_pvc` equals the recomputed development z_PVC
  exactly. 048's analysis, recomputed with seed 48048 for its four cases, equals 048's `result.json` exactly.
- All 1,604 `U_B` maps follow the beat-lead-wave unit order. Beats per ECG: 5 to 26, median 11.

## Focal-ratio distributions

Focal ratio = maximum beat score / median beat score (all ECGs of each group, not only referred ones).
Thresholds on the 463 normals: f(0.95) = 8.89, f(0.975) = 22.42, f(0.99) = 91.67.

| Group | n | Median | 97.5th percentile | Switch on, q = 0.95 | q = 0.975 | q = 0.99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NORM-only normals | 463 | 1.65 | 22.4 | 0.052 | 0.026 | 0.011 |
| Positives | 843 | 1.61 | 130.6 | 0.147 | 0.098 | 0.038 |
| PVC ECGs | 84 | 18.8 | 188.3 | 0.667 | 0.429 | 0.155 |
| PVC ECGs with a window | 73 | 20.0 | 207.1 | 0.712 | 0.438 | 0.164 |
| Benign variants | 52 | 1.53 | 26.5 | 0.077 | 0.038 | 0.000 |
| Anterior-only infarcts | 146 | 1.67 | 82.1 | 0.178 | 0.137 | 0.027 |
| Inferior-only infarcts | 149 | 1.71 | 64.1 | 0.128 | 0.081 | 0.020 |

The ratio separates the groups' medians as intended: PVC ECGs have a median of about 19, infarcts about
1.7, the same as normals. But the normals' upper tail is long (97.5th percentile 22.4), so a threshold at
the 97.5th percentile is above the median PVC ECG.

## Referred ECGs, primary case

Referral is 047's: 792 ECGs (24 normals, 680 positives, 49 PVC ECGs with a window, 136 anterior-only and
112 inferior-only infarcts). Intervals use seed 49049 with 2,000 whole-patient draws.

| Explanation | Hit | Chance | Hit - chance | Anterior contrast | Inferior contrast |
| --- | ---: | ---: | --- | --- | --- |
| `focal_switch` | 0.469 | 0.185 | +0.285 [0.157, 0.419] | +0.186 [0.067, 0.305] | +0.161 [0.058, 0.266] |
| `focal_and_pvc` | 0.469 | 0.185 | +0.285 [0.157, 0.419] | +0.188 [0.071, 0.310] | +0.176 [0.074, 0.278] |
| `focal_or_pvc` | 0.898 | 0.208 | +0.690 [0.603, 0.766] | +0.187 [0.061, 0.307] | +0.150 [0.039, 0.255] |
| PVC switch (048) | 0.898 | 0.208 | +0.690 [0.603, 0.766] | +0.189 [0.064, 0.308] | +0.164 [0.052, 0.272] |
| `U_B` alone | 0.898 | 0.208 | +0.690 [0.603, 0.766] | +0.031 [-0.096, +0.161] | +0.105 [-0.017, +0.227] |
| `attention_jepa` alone | 0.020 | 0.160 | -0.140 [-0.178, -0.092] | +0.197 [0.083, 0.310] | +0.212 [0.107, 0.317] |

**Cost on infarcts and coverage of PVC ECGs.** Referred ECGs sent to `U_B`:

| Arm | Anterior-only (of 136) | Inferior-only (of 112) | PVC with a window (of 49) | Positives (of 680) |
| --- | ---: | ---: | ---: | ---: |
| `focal_switch` | 19 | 10 | 24 | 74 |
| `focal_and_pvc` | 13 | 7 | 24 | 57 |
| `focal_or_pvc` | 74 | 46 | 49 | 309 |
| PVC switch (048) | 68 | 43 | 49 | 292 |

The focal switch cuts the infarct cost from 68 to 19 anterior and from 43 to 10 inferior infarcts. But it
reaches only 24 of the 49 PVC ECGs. All 24 are also above the PVC switch, so `focal_and_pvc` covers the same
PVC ECGs.

## Sensitivity cases

| Case | `focal_switch` hit - chance | Anterior contrast | Sent to `U_B`: anterior / inferior / PVC with a window |
| --- | --- | --- | --- |
| q = 0.95 | +0.479 [0.348, 0.599] | +0.207 [0.085, 0.330] | 25 / 13 / 34 of 136 / 112 / 49 |
| q = 0.99 | -0.003 [-0.100, +0.114] | +0.196 [0.080, 0.311] | 3 / 2 / 8 |
| E, q = 0.975 | +0.286 [0.157, 0.426] | +0.197 [0.074, 0.314] | 20 / 10 / 25 of 139 / 108 / 51 |
| `combined_50`, q = 0.975 | +0.257 [0.158, 0.365] | +0.170 [0.042, 0.288] | 19 / 11 / 32 of 136 / 116 / 73 |

`combined_50` referral fits thresholds of 0.322 (readout logit) and 3.336 (F) on the development normals.
It refers 816 ECGs, 22 of them normals and all 84 PVC ECGs, including example 219. Under it, the PVC switch
puts the explanation on the premature beat in 0.932 of the 73 PVC ECGs with a window (hit - chance +0.734
[0.674, 0.787]), with an anterior contrast of +0.183 [0.059, 0.302]. That is `U_B`'s full premature-beat
localization on every PVC ECG.

## Post-hoc check (not prespecified)

A scratch script recomputed each ECG's beat scores from the saved `U_B` units. The 12 normals above
f(0.975) do not have unusually small median beat scores (median 59.3, against 60.2 for all normals). They
have one very large beat (median maximum beat score 7,306, against 120 for all normals and 3,428 for PVC
ECGs with a window). So the long tail comes from single outlying beats in NORM-only normals, from
unlabelled ectopic beats or artefacts, not from a small denominator.

## Interpretation

- **The focal ratio measures what it was meant to.** PVC ECGs have a median ratio of about 19 against 1.7
  for infarcts and normals, and the switch sends far fewer infarcts to `U_B` than 048's PVC switch (19 against
  68 anterior-only).
- **The normals' threshold is too high to use.** About 3% of NORM-only normals have one outlying beat as
  extreme as a PVC's, so the 97.5th percentile (22.4) is above the median PVC ECG. The switch reaches only
  24 of 49 referred PVC ECGs, and the hit rate falls from 90% to 47%. Even at q = 0.95 it reaches 34 of 49.
- **048's infarct cost barely affects the lead contrast.** The PVC switch sends 68 anterior infarcts to
  `U_B`, yet its anterior contrast (+0.189) equals the focal switch's (+0.186) and is close to the attention
  layer's (+0.197; paired difference -0.009 [-0.121, +0.111]). By this metric, the cost that motivated this
  experiment is small: the contrasts do not show that the infarcts sent to `U_B` lose lead information on
  average. The cost remains a display question: an infarct explained by a rhythm map.
- **The PVC switch with `combined_50` referral is the best configuration so far.** It refers every PVC ECG,
  puts the explanation on the premature beat in 93% of those with a window, and keeps the infarct-lead
  contrast.

## Caveats

- The focal ratio depends on the beat count (5 to 26 beats here). Bigeminy and frequent PVCs raise the
  median beat score and lower the ratio.
- The NORM-only normals are not clean of single outlying beats; the switch threshold inherits them.
- 49 (73 under `combined_50`) PVC ECGs and 112-136 infarcts per group; the premature-beat rule is automatic;
  the attention layer marks 8 leads; infarct location is a whole-ECG statement.
- `combined_50` here is fitted on the 463 development normals, not on local normals as in pipeline v3.
- Development data, read by Experiments 041-048; the results are exploratory.

## Deviations

None.

## Follow-up ideas

- Make the PVC switch with `combined_50` referral the candidate explanation rule. Its thresholds should be
  refitted on the local normals with the referral thresholds, and it should be shown to the cardiologist with
  the 048 figures.
- A robust focal score: the second-largest beat excluded from the median, or the share of beats above a
  per-ECG outlier threshold. Both still need a threshold that the normals' outlying beats do not set. One
  option is the 97.5th percentile of the normals with their single worst beat removed. It is worth trying
  only if the infarct display cost matters to the cardiologist.
- Review the 12 normals with a focal ratio above 22.4 with the cardiologist, as possible unlabelled ectopy
  or artefacts in the NORM-only reference set; if they are ectopic beats, this affects every map threshold
  fitted on these normals.
