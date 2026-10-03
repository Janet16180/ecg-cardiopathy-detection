# Experiment 051 results: a beat-sum wave map

Completed 1 October 2026 under the [frozen protocol](experiment-051-beat-sum-map.md) (commit `84c89b2`; run
from `88e4ffb`). This experiment used 042's saved units only: nothing was trained or extracted. Besides
saved scores, the run read only the 84 PVC records' PTB-XL waveforms, to rebuild 041's premature-beat windows.
No calibration, test or EchoNext record was read. Outputs are in `outputs/experiment051_beat_sum_v1/`
(`result.json` with every input, source and protocol hash, `beat_maps.npz`, `run.log`). The run took 96
seconds on CPU, 86 of them in loading and integrity checks.

## Readings

| Map | Hit - chance minus `U_B`'s (lower bound > -0.10) | Anterior contrast (lower bound > 0) | Benign any red minus `U_B`'s (upper bound < 0) | Worst-unit AUROC minus `U_B`'s (lower bound > 0) | Reading |
| --- | --- | --- | --- | --- | --- |
| `beat_sum` (primary) | -0.070 [-0.131, -0.004] | +0.030 [-0.091, +0.150] | -0.019 [-0.058, 0.000] | -0.006 [-0.017, +0.004] | **Does not improve** |
| `beat_sum_relative` | -0.070 [-0.131, -0.004] | +0.030 [-0.091, +0.150] | 0.000 [-0.077, +0.077] | -0.188 [-0.217, -0.157] | Does not improve |

Neither map keeps premature-beat localization by 042's overlap rule, and neither gains on lead, benign or
detection. The strict R-time hit, which the protocol made descriptive, would keep localization. But the
conclusion does not depend on which hit rule is used, because no gain condition is met. No example figures
were drawn.

## Integrity

All of 049's checks passed again:

- The 042-048 receipts and reproductions.
- z_PVC and z_WPW against 044's SPH values (4.3e-15 and 1.3e-15).
- The `U_B` layout for all 1,604 ECGs.

049's `explanations.npz` matches its receipt. 049's five cases, recomputed with seed 49049, equal its
`result.json` exactly, and so does its PVC switch threshold (1.654).

## Maps on all 1,604 ECGs

Thresholds on the 463 NORM-only normals: `U_B` 1,158.8, `beat_sum` 7,733.9, `beat_sum_relative` 3.44.
Intervals use seed 51051 with 2,000 whole-patient draws.

| Map | Worst-unit AUROC | AP | Any red: normal | Positive | PVC | Benign | Hit | Chance | Hit - chance | Anterior contrast | Inferior contrast |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `U_B` | 0.740 | 0.839 | 0.052 | 0.286 | 0.774 | 0.077 | 0.932 | 0.197 | +0.734 [0.669, 0.791] | +0.085 [-0.038, +0.202] | +0.115 [-0.012, +0.229] |
| `beat_sum` | 0.734 | 0.840 | 0.052 | 0.279 | 0.798 | 0.058 | 0.945 | 0.281 | +0.665 [0.598, 0.727] | +0.030 [-0.091, +0.150] | +0.034 [-0.086, +0.144] |
| `beat_sum_relative` | 0.552 | 0.729 | 0.052 | 0.170 | 0.774 | 0.077 | 0.945 | 0.281 | +0.665 [0.598, 0.727] | +0.030 [-0.091, +0.150] | +0.034 [-0.086, +0.144] |

The positive any-red share minus `U_B`'s is -0.007 [-0.026, +0.011] for `beat_sum` and -0.116 [-0.143,
-0.090] for `beat_sum_relative`. Dividing by the ECG's median beat sum does not change which beat is on top,
so both beat maps have the same premature-beat and lead numbers.

**Why the overlap hit - chance falls.** `beat_sum`'s top beat overlaps a premature-beat window more often
than `U_B`'s top unit (hit 0.945 against 0.932). But a beat unit spans 0.70 s, so more of an ECG's units
overlap a window (chance 0.281 against 0.197). The difference comes from the chance term, not from missed
premature beats.

## Strict premature-beat hit (descriptive)

Here a hit counts only when the R time of the top unit's beat lies inside a premature-beat window.

| Map | Hit | Chance | Hit - chance |
| --- | ---: | ---: | --- |
| `U_B` | 0.548 | 0.168 | +0.380 [0.273, 0.491] |
| `beat_sum` | 0.575 | 0.168 | +0.407 [0.295, 0.519] |

The paired difference is +0.027 [-0.041, +0.097], so `beat_sum` keeps localization by this rule. `U_B`'s
strict hit (0.548) is much lower than its overlap hit (0.932). A post-hoc check (not prespecified; a scratch
script on the saved units and windows) looked at the 28 premature-beat ECGs where `U_B` hits by overlap but
not by the strict rule. In 26 of them, `U_B`'s top unit is the T piece of the beat just before the premature
beat. That piece spans R + 0.20 to R + 0.45 s and contains the early premature complex. In the other 2 it is
a T piece of another beat. The strict rule credits these to the preceding beat, so it undercounts `U_B` as
much as the overlap rule may overcount wide beat units. Neither rule is clearly right for both maps.

## Where the maps disagree (descriptive)

- `U_B`'s top unit and `beat_sum`'s top beat are in different beats in 807 of the 1,604 ECGs (50%).
- Among the 73 premature-beat ECGs they differ in 21. By the strict rule, the premature beat is `beat_sum`'s
  top beat in 12 of them and `U_B`'s top unit's beat in 10.

## ECG 287 (complete LBBB)

The beat sums reproduce the user's figures. They are 2,770-4,714 for the regular beats and 6,797 for the
odd wide beat at R = 6.114 s, which is `beat_sum`'s top beat (2.18 times the ECG's median beat sum). `U_B`'s
top unit is V1 ST of the beat at 9.104 s, score 1,416. So `beat_sum` points at the odd beat, as the user
expected. But at its own threshold (7,733.9, the 95th percentile of the normals' worst beat sum) that beat is
not red. The ECG's every beat is abnormal in the same way, so the sum adds the LBBB's V1 ST in every beat,
and normals reach high sums too.

## Secondary: `beat_sum` as the rhythm layer of 048's explanation

The rule is 048's PVC switch with 049's `combined_50` referral: 816 referred ECGs, and all 73 PVC ECGs with
a window are referred.

| Rhythm layer | Hit | Chance | Hit - chance | Anterior contrast | Inferior contrast |
| --- | ---: | ---: | --- | --- | --- |
| `U_B` (048/049; equals 049's numbers) | 0.932 | 0.197 | +0.734 [0.669, 0.791] | +0.183 [0.065, 0.312] | +0.161 [0.050, 0.262] |
| `beat_sum` | 0.945 | 0.281 | +0.665 [0.598, 0.727] | +0.177 [0.060, 0.303] | +0.168 [0.058, 0.271] |

The paired differences, `beat_sum` minus `U_B`, are -0.070 [-0.131, -0.004] for hit - chance, -0.006
[-0.037, +0.026] for anterior and +0.007 [-0.039, +0.051] for inferior. The hit - chance lower bound is
below -0.10, so by 048's rule `beat_sum` does not keep the rhythm layer's localization; the lead contrasts are
unchanged. 048's rule keeps `U_B`.

## Interpretation

- **The user's observation is right, and summing does find odd beats.** On ECG 287, `beat_sum` picks the odd
  wide beat that `U_B` misses. Among PVC ECGs where the two maps disagree, `beat_sum` is on the premature
  beat about as often as `U_B` (12 against 10 by the strict rule).
- **It does not improve on `U_B` overall.** Its detection is the same (AUROC 0.734 against 0.740), and it
  marks benign variants slightly less (5.8% against 7.7%; interval up to 0.000). It loses the lead contrasts
  (anterior +0.030), because the top piece within the top beat is a weaker lead pointer than the single worst
  piece. By 042's overlap rule its wider units also cost premature-beat localization.
- **A sum over a whole beat favours ECGs with abnormal beats everywhere.** On ECG 287 the odd beat wins
  within the ECG, but its sum (6,797) is below the normals' 95th percentile (7,734): 5% of NORM-only normals
  have a beat whose sum is higher.
- **`beat_sum_relative`** finds the same top beats, but as a detector it is weak (AUROC 0.552), as the
  protocol expected. It is a measure of within-ECG oddity.
- For the notebook's display problem, the evidence suggests showing `beat_sum`'s top beat **in addition to**
  `U_B`'s top unit when the two differ, rather than replacing `U_B`. This is a display choice; it was not
  tested here.

## Caveats

- Both premature-beat hit rules have a bias: the overlap rule rewards wide units in hit and penalizes them
  in chance, and the strict rule credits an early premature complex to the preceding beat. A rule based on
  the premature complex's own R peak and QRS span would compare the two maps more fairly.
- 73 premature-beat ECGs and 146 and 149 infarcts; the premature-beat rule is automatic; infarct location is
  a whole-ECG statement.
- Development data, read by Experiments 041-050; the results are exploratory.

## Deviations

None.

## Follow-up ideas

- A fairer premature-beat hit for maps of different unit sizes. A hit would require the top unit to overlap
  the premature complex's QRS (R - 0.06 to R + 0.08 s) rather than the 0.5 s window, with chance computed the
  same way. This needs a protocol, because it changes 041's established test.
- A two-mark display: on switch-on ECGs, show `U_B`'s top unit and, when it differs, `beat_sum`'s top beat
  with its 3 top pieces. This is for the cardiologist review, not a metric question.
- A within-ECG oddity score combined with the absolute score (for example `beat_sum` times
  `beat_sum_relative`) to rank beats inside a referred ECG only. Referral would not use it, since
  `beat_sum_relative` detects poorly.
