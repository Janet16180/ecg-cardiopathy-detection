# Experiment 053: training-normal calibration of lead and fixed-offset units

Frozen 2 October 2026 before any score. This is a normal-reference experiment, with no
abnormal diagnostic labels used in fitting or recipe selection. It addresses unequal
Mahalanobis dimensions in Experiment 042 (P 48, QRS 35, ST 30, T 63).
These are fixed offsets from detected R peaks, not validated physiological delineations.

Use exactly 042's 5,872 normal training ECGs (5,537 patients) and 1,604 development ECGs.
First reconstruct all-training normal references and reproduce every cached U_B unit to
relative error below 1e-8, exact top leads, detection AUROC and premature-beat rates.
No Challenge or EchoNext test data, calibration patients, downloads or GPU are involved.

Shuffle sorted normal training patient IDs with seed 53053. Assign the first floor(70%)
to reference fitting, the next floor(20%) to channel calibration, and the remainder to
independent mark-threshold control. Keep all records of a patient together and assert
patient and record disjointness against development. The recipe is fixed before scoring.

Fit 48 Ledoit-Wolf references on raw 042 beat pieces of reference patients. For each
lead and fixed-offset block, sort all calibration-patient beat Mahalanobis distances.
For an observed distance d, score -log((1 + count(calibration >= d))/(N+1)). This uses
conservative right tails, including ties. No development normal fits a channel CDF.
The primary persistent map takes the median of calibrated scores across kept beats,
one score per lead and block. Its top lead is the largest of the 48 scores. A separate
focal map retains all beat/lead/block units and their original time support.

Primary endpoint: mean(anterior contrast, inferior contrast) minus the same mean for
reproduced U_B; anterior is AMI-only minus IMI-only for V1-V4, inferior is IMI-only
minus AMI-only for II/III/aVF. Pair candidate and baseline within each ECG, resampling
patients independently inside each infarct group. Use shared patient bootstrap helpers,
2,000 draws, seed 53053. A worthwhile primary improvement requires gain >=0.10 and
95% lower bound >0 and both candidate regional contrasts positive.
Both groups are unchanged from 042:
146 AMI-only ECGs / 133 patients; 149 IMI-only / 139 patients.

Fix each candidate's mark threshold at the 95th percentile of its ECG maximum on the
independent training-normal control patients. Report development normal, positive and
benign any-mark rates, detection AUROC, and marked-unit fraction. For continuity only,
report 042's existing threshold fitted on 463 development normals as a separate baseline.
Also compute a baseline threshold using the independent control; this is explicitly
in-sample for baseline reference and is a diagnostic, never an independent safety claim.
A success must not increase benign any-mark by >0.05 versus the continuity baseline.
Focal support requires premature hit-minus-chance loss versus U_B 95% lower bound >-0.05.
Primary persistent units have no time interpretation or premature-beat endpoint.

Prespecified secondary controls: (1) repeat the same reference/calibration procedure after
linear interpolation of each block to 35 points, preserving original time support; (2)
transform each raw distance with the chi-square survival for its original coordinate
count, and median across beats. These distinguish distribution calibration from a
change in sample dimensions. Secondary results cannot replace a failed primary recipe.
Report top-block distribution on independent normals to quantify dimension preference.

Save input IDs, patient partition, source/input/protocol hashes, unit scores and a receipt.
Use the exact fixed examples NORM 47, PVC 219, IMI 8, AMI 184, CLBBB 287, LVH 30 and benign 69
from 042, regardless of outcome; all waveform figures stay local under outputs.
Write results from executed outputs, including failures and limitations. Clinical ground
truth for exact leads and offsets is absent; infarct code locations and automatic PVC
windows are development proxies. No claim of anatomical localization is authorized.

## Pre-score amendment: tied empirical tails and matched aggregation control

Empirical tails saturate above the largest calibration distance. For every regional
endpoint, identify the maximal block score of each lead and include all leads within
absolute tolerance 1e-12 of the ECG maximum. Regional membership is the fraction of
these tied top leads in the region, for both candidate and baseline. Report tie and
tail-saturation rates. Legacy first-argmax U_B metrics are reproduced separately,
so the new tie-aware endpoint does not obscure predecessor integrity.
Add a descriptive raw-distance persistent map from exactly the candidate's 70%
normal reference, taking median distances over beats, with the same independent
control threshold. This separates the effect of calibration from aggregation;
it cannot replace the primary result.
