# Experiment 053 results: calibrating fixed-offset normal-reference units

Normal calibration did **not** improve the lead localization proxy. The primary persistent
map's symmetric regional contrast fell from 0.100 to 0.080; its paired gain was
-0.020 [95% interval -0.133, +0.091], below the frozen +0.10 decision threshold.
The focal map also failed the premature-beat noninferiority guardrail. No candidate is
promoted for localization and no demonstration notebook is warranted by this result.

Completed 2 October 2026 from source commit `3705960`, under the
[frozen protocol](experiment-053-calibrated-units.md), including its pre-score tie amendment.
The final local receipt is `outputs/experiment053_calibrated_units_v2/result.json`;
unit maps, patient partitions, complete waveform hashes, source/input/output hashes,
fixed review figures and the live log are in that directory. The CPU run took
197.6 seconds with two numerical threads. No GPU, new data download, Challenge test,
EchoNext test or calibration cohort was used.

## Integrity and execution lineage

All original Experiment 042 U_B unit distances reproduced **bit for bit**: maximum
absolute and relative differences were both zero. Unit leads/time spans and each
legacy first-argmax top lead matched exactly. Legacy anterior contrast 0.08458214581226442
and inferior contrast 0.1151512365541969, their 042 patient-bootstrap bounds, detection
AUROC 0.7403160060362431, PVC top-unit hit 0.9315068493150684, chance
0.19715822833268443 and hit-minus-chance 0.7343486209823841 reproduced exactly.
The scored order was unchanged, all 7,476 ECGs had at least two complete beats,
and 73 of 84 PVC records met the unchanged automatic eligibility rule.
Required baseline summary checks executed before any candidate fit or score.

The first run, source `793a578`, completed scoring and figures but failed while writing
its patient partition: PTB-XL patient identifiers are namespaced strings, not integers.
That scored directory remains intact as `outputs/experiment053_calibrated_units_v1/`,
with `failure.json` recording the error and source/artifact hashes. A new runner,
`run_calibrated_localization053_v2.py`, changes only the output directory and serialization
of patient identifiers to strings. All scientific recipes and inputs remained unchanged;
no candidate primary metrics were inspected to choose this correction. The completed
v2 runner refuses any existing output directory. Both runner versions remain frozen.

## Independent normal fitting and marks

The 5,872 normal training ECGs / 5,537 patients were split by whole patient before fitting:

| Purpose | ECGs | Patients | Kept beats |
| --- | ---: | ---: | ---: |
| Covariance reference | 4,108 | 3,875 | 44,317 |
| Per-lead/block empirical tails | 1,171 | 1,107 | 12,690 |
| Independent mark control | 593 | 555 | |

There was no overlap between these patient groups or development patients. No abnormal
label fitted a covariance, tail, threshold or aggregation recipe. The development
labels only evaluated the frozen endpoints.

Every candidate mark threshold is the 95th percentile of ECG maxima in the independent
training-normal control, held out of both covariance and tail fitting. The observed
control any-mark rates were 5.06% for persistent maps, 4.55% for calibrated focal and
4.05% for equal-width focal; discreteness explains the latter differences from 5%.
The U_B continuity threshold 1158.8408102760714 remains the old threshold fitted on
463 development normal records. Its reference includes the independent control, so the
reported U_B control threshold 1639.3895939745082 is an in-sample diagnostic and is not
an independent normal validation result. Candidate versus continuity mark-rate differences
therefore have this threshold-provenance limitation.

The calibrated persistent map's normal-control top-block shares were P 20.57%,
QRS 29.68%, ST 25.30%, T 24.45%. The equal-35-point control was 19.22%, 28.54%,
24.58%, 27.66%. These block names denote frozen offsets from R, not validated physiological
boundaries.

## Frozen primary lead endpoint

Anterior membership is the fraction of unique tied maximal leads in V1-V4; inferior
membership uses II, III and aVF. The anterior contrast compares AMI-only with IMI-only;
the inferior contrast reverses those groups. The symmetric contrast is their mean.
All ties use absolute tolerance 1e-12 and whole-patient intervals use 2,000 draws,
seed 53053. Paired gains subtract the baseline within each ECG before resampling.

| Map | Anterior contrast [95% interval] | Inferior contrast [95% interval] | Symmetric contrast | Paired symmetric gain [95% interval] |
| --- | --- | --- | ---: | --- |
| `U_B` | +0.085 [-0.035, +0.202] | +0.115 [+0.003, +0.223] | 0.100 | +0.000 [+0.000, +0.000] |
| `calibrated_focal` | +0.073 [-0.044, +0.184] | -0.010 [-0.090, +0.065] | 0.032 | -0.068 [-0.158, +0.026] |
| `calibrated_persistent` | +0.075 [-0.041, +0.181] | +0.085 [-0.004, +0.169] | 0.080 | -0.020 [-0.133, +0.091] |
| `chi_square_persistent` | +0.143 [+0.032, +0.255] | +0.056 [-0.052, +0.162] | 0.100 | -0.000 [-0.101, +0.099] |
| `raw_distance_persistent` | +0.138 [+0.025, +0.249] | +0.088 [-0.024, +0.198] | 0.113 | +0.013 [-0.085, +0.111] |
| `equal35_focal` | +0.089 [-0.025, +0.201] | +0.025 [-0.051, +0.101] | 0.057 | -0.043 [-0.136, +0.051] |
| `equal35_persistent` | +0.071 [-0.048, +0.181] | +0.058 [-0.034, +0.146] | 0.065 | -0.035 [-0.146, +0.075] |

The primary persistent map failed the required paired gain >=0.10 with lower bound
above zero, although both regional point estimates had the expected sign. The same-reference
raw-distance median ablation also had no supported gain (+0.013 [-0.085, +0.111]).
Its positive own anterior contrast and the chi-square map's positive own anterior contrast
do not demonstrate improvement over U_B: their paired symmetric gains include zero.
Secondary equal-width maps cannot replace the failed primary and did not improve the
point estimate. Empirical tails and common coordinate counts therefore do not resolve
the localization problem in these data.

## Detection, independent marks and focal support

| Map | AUROC | Paired AUROC gain [95% interval] | Development normal any mark | Positive any mark | Benign any mark |
| --- | ---: | --- | ---: | ---: | ---: |
| `U_B` | 0.740 | +0.000 [+0.000, +0.000] | 5.18% | 28.59% | 7.69% |
| `calibrated_focal` | 0.758 | +0.018 [+0.007, +0.029] | 3.02% | 22.78% | 1.92% |
| `calibrated_persistent` | 0.769 | +0.029 [+0.005, +0.052] | 3.67% | 35.94% | 9.62% |
| `chi_square_persistent` | 0.764 | +0.024 [+0.002, +0.047] | 3.24% | 24.91% | 3.85% |
| `raw_distance_persistent` | 0.748 | +0.008 [-0.014, +0.029] | 3.46% | 24.20% | 3.85% |
| `equal35_focal` | 0.760 | +0.020 [+0.008, +0.031] | 3.02% | 23.01% | 5.77% |
| `equal35_persistent` | 0.774 | +0.033 [+0.010, +0.057] | 4.32% | 35.82% | 5.77% |

The primary persistent map modestly improved detection AUROC (+0.029 [+0.005, +0.052]),
but detection was not this experiment's localization endpoint. Benign marks rose from
4/52 to 5/52, an increase of 0.019 [-0.058, +0.096]; the point increase passed the frozen
<=0.05 guardrail. No claim of improved clinical screening is made from this development
comparison.

| Timed map | Premature top-unit hit | Chance | Hit-minus-chance | Paired excess gain [95% interval] |
| --- | ---: | ---: | ---: | --- |
| `U_B` | 0.932 | 0.197 | 0.734 | +0.000 [-0.000, +0.000] |
| `calibrated_focal` | 0.896 | 0.197 | 0.699 | -0.035 [-0.070, -0.008] |
| `equal35_focal` | 0.899 | 0.197 | 0.702 | -0.033 [-0.068, -0.005] |

Calibrated focal localization lost 0.035 [-0.070, -0.008] hit-minus-chance relative to U_B.
The lower bound was below the prespecified -0.05 margin, so the focal guardrail failed.
Persistent units do not have an observed beat time and cannot establish focal support.

The empirical tail cap is log(12,691). Calibrated focal unit saturation was 0.128% of
units averaged across development ECGs; persistent saturation was 0.105%. Despite that
small unit fraction, tied maximal leads occurred in 8.73% of focal ECGs and 2.37% of
persistent ECGs. Averaging tied leads/units prevents the array's first lead/beat from
winning by order. Chi-square tails reached the floating-point floor in some cases;
its maximal-lead tie rate was 4.86%. Raw median and U_B had zero maximal-lead ties.

## Fixed local waveform review

All seven prespecified examples were rendered, without selecting examples by performance.
Figures stay local at `outputs/experiment053_calibrated_units_v2/figures/`. Persistent
marks are shown repeated on the same block of every complete beat solely to show the
block support; they are not a prediction that every displayed beat is abnormal.

| Fixed example | U_B marked units | Calibrated focal marked units | Persistent marked lead/block channels |
| --- | ---: | ---: | ---: |
| NORM 47 | 0 | 0 | 0 |
| PVC 219 | 2 | 1 | 0 |
| IMI 8 | 0 | 0 | 0 |
| AMI 184 | 0 | 0 | 0 |
| CLBBB 287 | 1 | 12 | 4 |
| LVH 30 | 0 | 0 | 0 |
| benign 69 | 0 | 0 | 0 |

Visual inspection of AMI 184, PVC 219 and CLBBB 287 agrees with these executed counts:
AMI 184 remains unmarked; PVC 219 retains one focal marked unit at the unusual late beat
and no persistent marks; CLBBB 287 has more repeated block marks than U_B. The narrow
fixed-offset highlights in a wide-complex ECG do not verify physiological wave ownership
or exact abnormal boundaries. The unchanged unmarked AMI and IMI examples are failures
to reveal an explanation at the chosen budget and are included with the marked examples.

## Verification and implications

Seven focused scientific tests passed: conservative empirical ties, finite tails,
channel-scale invariance, equal-width endpoint preservation, disjoint infarct patients,
unique-lead tie averaging, presorted-tail equivalence and tied focal overlap. Ruff passed
for the new library, both runners and tests before their corresponding launches.
The results remain exploratory PTB-XL development proxies; exact abnormal lead and wave
annotations are absent. Clinical labels and age strata were unchanged.

Follow-up evidence for the backlog: channel-tail calibration should not be promoted as a
localization fix; future candidate maps should retain separate focal/persistent outputs
and tie-aware paired comparisons. Improving detection or balancing normal block preferences
is insufficient to claim correct localization. A successor needs a different morphology
model or measured boundaries, evaluated against the same paired baseline; cardiologist
region marks remain the direct validation missing from these proxy endpoints.
