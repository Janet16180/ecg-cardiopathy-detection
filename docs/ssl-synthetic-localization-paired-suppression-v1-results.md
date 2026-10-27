# Paired learning and background suppression on generated ECGs

19 October 2026. The fixed combined objective improved average introduced-change ranking and reduced
control peaks, but its complete scientific rule **failed: 14 of 15 gates passed**. The repeated-edit
noninferiority guard failed. All nine technical integrity gates and the independent audit passed,
with zero mismatches. The failed scientific verdict is retained; no arm is selected for deployment.

The [protocol](ssl-synthetic-localization-paired-suppression-v1.md) was committed as `4500999` before
the coordinator opened the preceding replication scores. Identity-only admission `03ef46d` and
source `25fd1b8` were committed before the actual profiles and fits. Four fixed arms were trained:
B is unchanged weighted BCE; P adds the fixed paired support hinge and outside-support consistency;
K adds the mean top-600 control-probability penalty; PK adds both. All coefficients are one.
The paired hinge margin is 0.2. Hinge and consistency are bundled and cannot be separately attributed.

## Primary comparison and all fitted arms

| PK versus B primary | Mean | Exploratory crossed 95% interval |
|---|---:|---|
| `PK_minus_B_direct_AP_focal` |0.062735|[0.014439, 0.118658]|
| `PK_minus_B_direct_AP_repeated` |0.021080|[-0.011608, 0.057661]|
| `PK_minus_B_direct_AP_pooled` |0.041908|[0.005981, 0.080065]|
| `B_minus_PK_top600_sham` |0.277670|[0.215482, 0.352617]|
| `B_minus_PK_top600_nuisance` |0.274799|[0.210851, 0.352396]|
| `B_minus_PK_top600_pooled` |0.276235|[0.213717, 0.352266]|
| `PK_focal_response_ap` |0.139952|[0.081186, 0.209753]|
| `PK_focal_selectivity` |0.044514|[0.027988, 0.064832]|
| `PK_repeated_response_ap` |0.118584|[0.073089, 0.175777]|
| `PK_repeated_selectivity` |0.031879|[0.020766, 0.044454]|

The pooled AP gain exceeds the fixed 0.02 point margin and has a positive interval lower bound.
The repeated-edit gain is +0.021080, but its lower bound −0.011608 is below the required −0.01.
Fit 1 actually loses 0.008928 repeated AP; its pooled gain is only 0.004561. Fit 2 and 3 pooled
gains are 0.053780 and 0.067382. These are three fixed initialization/schedule pairs, not 192
independent fits. All individual gains, ranges and conditional intervals remain in the receipt.

| Fit | Arm | Focal AP | Repeated AP | Equal-mode AP | Control top600 mean | Sham max>0.5 | Nuisance max>0.5 |
|---|---|---:|---:|---:|---:|---:|---:|
|1|B|0.104453|0.116814|0.110634|0.357441|31/64|31/64|
|1|P|0.195904|0.254119|0.225011|0.463063|46/64|48/64|
|1|K|0.050622|0.070335|0.060478|0.088244|0/64|0/64|
|1|PK|0.122503|0.107886|0.115194|0.115160|0/64|0/64|
|2|B|0.075394|0.127937|0.101665|0.345624|31/64|31/64|
|2|P|0.206215|0.226535|0.216375|0.476079|47/64|49/64|
|2|K|0.119121|0.111937|0.115529|0.079013|1/64|1/64|
|2|PK|0.154477|0.156414|0.155445|0.118428|3/64|2/64|
|3|B|0.068556|0.107279|0.087918|0.500181|51/64|51/64|
|3|P|0.147914|0.184945|0.166429|0.592237|52/64|53/64|
|3|K|0.087313|0.077449|0.082381|0.117343|0/64|0/64|
|3|PK|0.159629|0.150971|0.155300|0.140955|1/64|1/64|

Across the three fits, mean equal-mode AP is B 0.100072, P 0.202605, K 0.086129, PK 0.141980.
These probabilities and the 0.5 display cutoff are uncalibrated. Controls mean no introduced edit,
not a healthy patient; these counts are not clinical false-positive rates.

The predeclared factorial contrasts expose a tradeoff. P improves pooled AP by 0.102533
[0.069728, 0.139598] over B, while increasing control top600 probability by 0.109377
[0.086018, 0.137434]. K reduces that control score by 0.306216, but pooled AP changes by
−0.013943 [−0.051880, 0.019458], and repeated AP decreases by 0.030770. PK gains pooled AP
over K by 0.055851 [0.034709, 0.082523], while losing 0.060625 [−0.116057, −0.008617]
against P. More suppression is therefore not evidence of uniformly better localization.
All 42 predefined secondary contrasts are reported below, without multiplicity adjustment,
best-arm selection, coefficient tuning or a replacement success rule.

Fixed fit-1 galleries retain indices 2000–2007 for both edit modes and all four arms, on one
0–1 color scale. K/PK are dimmer, but temporal and lead alignment remain uneven; dimmer maps
alone do not establish cleaner localization. The aggregate figure shows every fitted arm.

## Exact closure and execution

The preceding replication aggregate reproduced exactly. Four CPU generator closure forwards
recovered original arrays exactly. Actual GPU generation used three profiled inputs and one
exact replay, then all 64 scientific seeds. All fresh raw signals and canonical shams were unique within the new64; fresh shams were disjoint
from the preceding192shams. Final arrays were finite and passed the declared support-energy and
lead-algebra checks.
First-three bank outputs matched their profiles, with unchanged released generator state.

Each objective profile restored the saved step-one model and full Adam state and exactly
reproduced steps two and three. Saved native first-step logits and batch indices independently
recover BCE, hinge, consistency, control penalty and total objective within the predeclared
1e-6 tolerance. This adds no verification inference. All three B arms then reproduce every
preceding supervised loss, final model and full Adam tensor exactly for all 400 updates.

The original 80 TRAIN backgrounds and 320 variants were unchanged. Each of the twelve fits used
400 updates, 3,200 presentations and the same positive weight, architecture and Adam recipe.
All fits finished before scoring the fresh panel. There were 4 CPU and 68 GPU generator forwards,
20 profile updates plus 1,200 B replay and 3,600 new-objective updates; 4 TRAIN-profile and
384 fresh scientific inference batches. No old-panel inference, search updates or real ECG reads ran.

Fresh generator seeds are 2026105000–2026105063, editor indices 2000–2063; initialization seeds
2026103801–3803 and schedule seeds 2026103901–3903 are paired across all four arms.
The crossed bootstrap uses 2,000 draws, seed 2026105501, resampling three fit rows then one
shared 64-background vector; conditional intervals use 2026105502. All draws are retained.
Three-row uncertainty is exploratory and does not provide clinical or patient coverage.

## Resources, audit and identities

Main terminal CPU was 163.665497943 s, whole-stage wall 388.866524710 s and GPU-lock wall
269.419470983 s. Forecasts were 177.488724 CPU /203.665759 whole-wall /190.159490 GPU-lock s.
Both wall forecasts underestimate execution. Timing matching loss/component persistence did not
solve the forecasting discrepancy; map persistence remains outside the inference-rate sample.
All fixed 1,800 CPU/wall, 1,200 GPU-lock second and 1 GiB output caps nevertheless passed.
Peak allocated GPU memory was 154,269,184 bytes, process RSS 2,618,961,920 bytes and main final
storage 875,916,535 bytes. GPU0 was an actual RTX 3090 with torch2.6.0+cu124,
deterministic one-thread settings, TF32 disabled and the shared project GPU lock.

The independent audit verified 402 bindings, 33,671 metric scalars and 4,343 case scalars with
zero mismatches, including all twelve saved map panels, profiles, complete states, fifteen gates
and factorial contrasts. It used no model forward or optimizer update. Saved-state arithmetic
and byte checks are independent; runtime operation-path claims additionally rely on frozen source.
Audit terminal CPU was 35.808072573 s, wall 39.504808745 s, GPU zero.
Main plus audit CPU is 199.473570516 s. The fourteen verified
current-machine stages total 390.154950634 CPU /
447.981392133 GPU-lock seconds. Every retained failure
is added once; absent older historical costs remain unknown. Measurement ends after receipt
write/hash and before terminal-ledger serialization, under the reserved-tail convention.

| Identity | SHA-256 |
|---|---|
|summary.json|`8850ab4119e9d964aa6177ed33edc93559b810dd96038304da6d364685360111`|
|prospective_ledger.json|`29d9afbe3dad2cc11ab5257b6cf96c90cd3521284c6b04fcc6ad79980338c8ef`|
|independent_audit_v1/summary.json|`07f8a0d5a50c57d4355576ee2d24fc552b390ab99be5aed211c83f2f5fedb4bd`|
|independent_audit_v1/prospective_ledger.json|`705cc964022f4f1d78d08d7d1eff75a665f02c150e311dacf2e1c4503ede5496`|
|ecg_experiment/ssl_synthetic_paired_suppression.py|`d402cf3758cad75f00eb54e378e3545b916b0d3842622458a18092f38bb0c384`|
|ecg_experiment/ssl_synthetic_paired_suppression_run.py|`3845cec36eb733f47853f1c1acc9b3cfd9e3a80b6a13680694a2cea4035367cd`|

The public Pulse2Pulse checkpoint remains
`e646eae78b7c9e48db0d38f094059ab89b53479bbc174a900ae3086517761827`,
architecture revision `f28e4b792e79c503b8b7e0912be9d304a3a7177b`, Hub revision
`262e0e67c0101cd4989d7c2f66f16d6d54756e70`. Weights were loaded locally, not retrained.

All generated signals, masks, predictions, complete models/Adam states, profile logits, losses,
source snapshots and three quality/localization figures remain local beneath
`outputs/ssl_synthetic_localization/generated_paired_suppression_v1/`.
No data, weights, predictions or figures are staged in Git. See the
[independent audit report](ssl-synthetic-localization-paired-suppression-independent-audit-v1-results.md).

The [temporal/lead diagnosis](ssl-synthetic-localization-temporal-diagnosis-v1.md) was committed
as `ec86255` before the coordinator read this study's outcomes. It uses only these spent saved maps
to distinguish within-lead temporal ranking from lead-presence ranking, without a new success rule.
This result establishes a fixed-recipe engineering tradeoff on introduced changes in unconditional
generated signals. Physical calibration, physiological validity, disease conditioning, real disease
masks and benefit from real-ECG SSL remain unproven. Clinical labels and closed tests are unchanged.
PR #67 remains draft.

## Complete predefined scientific gates

| Gate | Passed |
|---|---|
|`PK_minus_B_direct_AP_pooled_mean`|True|
|`PK_minus_B_direct_AP_pooled_lower`|True|
|`B_minus_PK_top600_pooled_mean`|True|
|`B_minus_PK_top600_pooled_lower`|True|
|`PK_minus_B_direct_AP_focal_guard`|True|
|`PK_focal_response_ap_lower`|True|
|`PK_focal_selectivity_lower`|True|
|`PK_minus_B_direct_AP_repeated_guard`|False|
|`PK_repeated_response_ap_lower`|True|
|`PK_repeated_selectivity_lower`|True|
|`fit_1_direct_AP_gain_positive`|True|
|`fit_2_direct_AP_gain_positive`|True|
|`fit_3_direct_AP_gain_positive`|True|
|`B_minus_PK_top600_sham_lower`|True|
|`B_minus_PK_top600_nuisance_lower`|True|

## Complete predefined factorial secondaries

| Contrast | Mean | Exploratory crossed 95% interval |
|---|---:|---|
|`P_minus_B__focal_edited_ap`|0.100543|[0.055936, 0.156323]|
|`K_minus_B__focal_edited_ap`|0.002884|[-0.053924, 0.055637]|
|`PK_minus_K__focal_edited_ap`|0.059851|[0.027753, 0.099569]|
|`PK_minus_P__focal_edited_ap`|-0.037808|[-0.092952, 0.019472]|
|`half_P_minus_B_plus_PK_minus_K__focal_edited_ap`|0.080197|[0.051244, 0.112301]|
|`half_K_minus_B_plus_PK_minus_P__focal_edited_ap`|-0.017462|[-0.065302, 0.026768]|
|`PK_minus_P_minus_K_plus_B__focal_edited_ap`|-0.040692|[-0.106081, 0.015804]|
|`P_minus_B__repeated_edited_ap`|0.104523|[0.064657, 0.152162]|
|`K_minus_B__repeated_edited_ap`|-0.030770|[-0.058662, -0.009023]|
|`PK_minus_K__repeated_edited_ap`|0.051850|[0.027112, 0.090681]|
|`PK_minus_P__repeated_edited_ap`|-0.083443|[-0.151869, -0.027031]|
|`half_P_minus_B_plus_PK_minus_K__repeated_edited_ap`|0.078186|[0.051685, 0.104888]|
|`half_K_minus_B_plus_PK_minus_P__repeated_edited_ap`|-0.057106|[-0.101610, -0.022434]|
|`PK_minus_P_minus_K_plus_B__repeated_edited_ap`|-0.052673|[-0.111090, 0.000049]|
|`P_minus_B__pooled_edited_ap`|0.102533|[0.069728, 0.139598]|
|`K_minus_B__pooled_edited_ap`|-0.013943|[-0.051880, 0.019458]|
|`PK_minus_K__pooled_edited_ap`|0.055851|[0.034709, 0.082523]|
|`PK_minus_P__pooled_edited_ap`|-0.060625|[-0.116057, -0.008617]|
|`half_P_minus_B_plus_PK_minus_K__pooled_edited_ap`|0.079192|[0.059356, 0.100478]|
|`half_K_minus_B_plus_PK_minus_P__pooled_edited_ap`|-0.037284|[-0.079931, -0.001338]|
|`PK_minus_P_minus_K_plus_B__pooled_edited_ap`|-0.046683|[-0.087649, -0.003158]|
|`P_minus_B__sham_top600`|0.107570|[0.083959, 0.135949]|
|`K_minus_B__sham_top600`|-0.306978|[-0.379711, -0.245240]|
|`PK_minus_K__sham_top600`|0.029307|[0.017477, 0.042461]|
|`PK_minus_P__sham_top600`|-0.385241|[-0.452313, -0.323294]|
|`half_P_minus_B_plus_PK_minus_K__sham_top600`|0.068439|[0.052476, 0.088196]|
|`half_K_minus_B_plus_PK_minus_P__sham_top600`|-0.346109|[-0.415729, -0.284966]|
|`PK_minus_P_minus_K_plus_B__sham_top600`|-0.078263|[-0.099658, -0.058414]|
|`P_minus_B__nuisance_top600`|0.111184|[0.087649, 0.139280]|
|`K_minus_B__nuisance_top600`|-0.305454|[-0.378100, -0.243992]|
|`PK_minus_K__nuisance_top600`|0.030655|[0.019573, 0.043729]|
|`PK_minus_P__nuisance_top600`|-0.385983|[-0.454456, -0.324013]|
|`half_P_minus_B_plus_PK_minus_K__nuisance_top600`|0.070920|[0.054667, 0.090565]|
|`half_K_minus_B_plus_PK_minus_P__nuisance_top600`|-0.345718|[-0.415887, -0.285274]|
|`PK_minus_P_minus_K_plus_B__nuisance_top600`|-0.080529|[-0.101493, -0.061786]|
|`P_minus_B__pooled_top600`|0.109377|[0.086018, 0.137434]|
|`K_minus_B__pooled_top600`|-0.306216|[-0.379570, -0.244720]|
|`PK_minus_K__pooled_top600`|0.029981|[0.018977, 0.043078]|
|`PK_minus_P__pooled_top600`|-0.385612|[-0.453257, -0.323721]|
|`half_P_minus_B_plus_PK_minus_K__pooled_top600`|0.069679|[0.053744, 0.089384]|
|`half_K_minus_B_plus_PK_minus_P__pooled_top600`|-0.345914|[-0.416005, -0.284894]|
|`PK_minus_P_minus_K_plus_B__pooled_top600`|-0.079396|[-0.100567, -0.060380]|
