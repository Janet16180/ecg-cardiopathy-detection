# Matched control training on real ECG backgrounds results

18 October 2026. Both fixed training arms and the single frozen evaluation completed. The SSL-initialized arm had greater counterfactual ranking response to the introduced voltage changes in this fixed-seed study, especially focal changes. The complete prospective advancement rule failed: the random-initialized arm did not establish positive target-versus-candidate selectivity, and the nuisance-control flag increase exceeded the allowed five percentage points. No threshold, checkpoint or rule was adjusted afterward.

## Localization and response

R began from the original random encoder; S began from the existing real-data SSL encoder. Both used the same original localization head, architecture, 500 real training patients, 2,000 fixed introduced-change/control inputs, identical 600-step schedule and fresh optimizer. No additional SSL was performed. Evaluation retained all 128 separate patients and all 512 cases. The 64 calibration patients and these evaluation patients exclude the earlier 192-patient transfer cohort. All are development evidence.

Average precision (AP) measures ranking of the exact introduced-change support, not diagnostic accuracy. Baseline-site AP scores the matching unedited control map against the SAME intervention mask; it measures alignment before the introduction. Response is edited AP minus baseline-site AP. Each patient contributes equally, averaging focal and repeated modes.

| Arm | Edited AP | Baseline-site AP | AP response |
| --- | ---: | ---: | ---: |
| R | 0.346928 | 0.339938 | 0.006990 [0.002668, 0.011757] |
| S | 0.511194 | 0.266235 | 0.244958 [0.203916, 0.286321] |

The primary S minus R AP-response difference was **0.237969 [0.198587, 0.278683]**. Edited-map AP itself improved by 0.164266 [0.131586, 0.197375], passing the frozen noninferiority guard. These are paired whole-patient 95% bootstrap intervals, 2,000 draws, seed 20261025; no draws were skipped. They describe patient sampling for one training seed, not training variability.

| Mode | R edited AP | S edited AP | S minus R edited AP, 95% interval | S minus R response, 95% interval |
| --- | ---: | ---: | ---: | ---: |
| focal | 0.077251 | 0.427709 | 0.350458 [0.294107, 0.408261] | 0.338975 [0.277900, 0.398174] |
| repeated | 0.616604 | 0.594678 | -0.021926 [-0.040893, -0.002710] | 0.136962 [0.112536, 0.162375] |

The pooled response gain decomposes into an edited-AP gain of 0.164266 plus 0.073703 lower baseline-site AP for S. Thus it is not wholly an increase in edited-map localization. Repeated-change response improved while repeated edited AP was lower; this does not establish better repeated-change localization.

The focal result gives a clearer introduced-change signal than edited AP alone: S focal AP rose from baseline-site 0.073131 to 0.427709, while R rose from 0.061649 to 0.077251. For repeated changes, R already aligned strongly before introduction (baseline-site AP 0.618227 versus edited AP 0.616604). Its higher repeated edited AP therefore does not establish greater responsiveness. S repeated AP rose from 0.459339 to 0.594678.

## Context and fixed area coverage

| Method | Overall AP | Focal AP | Repeated AP | Overall top1% precision | Overall top1% recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| R | 0.346928 | 0.077251 | 0.616604 | 0.397826 | 0.093986 |
| S | 0.511194 | 0.427709 | 0.594678 | 0.507044 | 0.261820 |
| uniform | 0.039793 | 0.006114 | 0.073472 | 0.039793 | 0.010000 |
| absolute_voltage | 0.064261 | 0.011434 | 0.117088 | 0.004609 | 0.001763 |
| algorithmic_anchor_prior | 0.100605 | 0.015978 | 0.185232 | 0.100605 | 0.026036 |

Introduced-support prevalence was 0.039793 overall, 0.006114 focal and 0.073472 repeated. The uniform map equals that prevalence. The phase prior uses all algorithmic candidate windows/leads with privileged timing, not the realized intervention lead or focal beat. Top1% allocates 300 native cells with expected credit across boundary ties; it does not represent complete abnormal-region coverage.

| Arm / mode | Top1% precision | Top1% recall |
| --- | ---: | ---: |
| R / focal | 0.075729 | 0.102848 |
| R / repeated | 0.719922 | 0.085125 |
| S / focal | 0.326641 | 0.440978 |
| S / repeated | 0.687448 | 0.082661 |

## Frozen flags and advancement

Each arm’s threshold was the empirical higher 95th percentile of the 64 calibration patients’ joint unchanged/nuisance maximum-map scores; flags use strict greater-than. Calibration bytes and both map slices were frozen before any B inference and remained unchanged afterward. These are introduced-change flags, not clinical sensitivity or clinical false-positive rates.

| Arm | Threshold | Unchanged flags /128 | Nuisance flags /128 | Added-change flags /256 |
| --- | ---: | ---: | ---: | ---: |
| R | 0.923113227 | 1 (0.78%) | 1 (0.78%) | 2 (0.78%) |
| S | 0.967530906 | 7 (5.47%) | 8 (6.25%) | 121 (47.27%) |

S minus R unchanged-control flag difference was 0.046875 [0.007812, 0.093750]; nuisance-control difference was 0.054688 [0.007812, 0.101562]. The latter point increase is 5.46875 percentage points, exceeding the frozen five-point maximum. The threshold is retained.

| Prospective gate | Passed |
| --- | --- |
| `response_gain_at_least_minimum` | yes |
| `response_gain_lower_bound_positive` | yes |
| `edited_ap_noninferiority` | yes |
| `R_average_precision_change_lower_bound_positive` | yes |
| `R_selectivity_contrast_lower_bound_positive` | no |
| `S_average_precision_change_lower_bound_positive` | yes |
| `S_selectivity_contrast_lower_bound_positive` | yes |
| `sham_point_increase_bounded` | yes |
| `nuisance_point_increase_bounded` | no |

Both arms had positive pooled AP-response intervals, but R’s target-minus-unedited-candidate probability-response contrast was 0.000346 [−0.000648,0.001336], so its required positive lower bound failed. S’s contrast was 0.210333 [0.176431,0.243532]. These within-arm uncalibrated probability changes are not cross-arm probability calibration. Seven of nine gates passed; the full conjunction is false. A large relative response gain with a weak comparator response still requires interpretation and replication under a separately frozen design.

## Execution and scope

All 692 backgrounds and 2,768 variants qualified on the first fixed selection: no integrity/anchor/case failures, replacements or dropped cases; 2,768 exact replays and 1,384 exact positive/control baselines. The batch schedule includes all four variants for each selected patient set. Each arm received 9,600 presentations, 4,800 positive and 4,800 zero introduced-mask, on the same schedule. This is a different label budget and training recipe from the original simulator pilot; the new patients also differ from the earlier frozen transfer. Comparing their AP values cannot isolate a causal effect of real-background training or matching alone.

Only the observed edited waveform enters the network. The model does not receive the paired baseline, algorithmic anchor, introduced mask, patient identity or edit parameters. Native abnormalities remain unknown; a zero mask labels absence of the introduced modification, not health. The algorithmic anchors and controlled pulses are not expert-verified pathological boundaries. No clinical label, referral rule or closed test changed.

The actual profile passed model/AdamW/RNG recovery after one saved update and two exact continued updates for both arms. It took 12.166 wall/12.153 CPU seconds; the two fixed fits took 55.857/55.853 seconds. Evaluation took 22.368/22.305 seconds. Whole-stage GPU-wall was 90.391 seconds and cumulative scientific CPU before final independent audits was 535.774 seconds. All runtime/storage limits passed.

The independent training audit passed 79,348 checks with zero mismatches. It independently reconstructed the schedule, native class weight, input exposures and CPU tensor hashes, and checked saved losses, checkpoint components and recovery records. Final scientific optimizer states were not saved, so optimizer-step counts remain checked through source-bound execution records; the independent audit did not rerun updates.

The independent saved-map audit passed 195,526 checks with zero mismatches, reproducing the complete new summary, thresholds, all 13 response metrics, paired intervals and all nine decision gates. It verified 228 file bindings with 11 explicit waveform-payload exceptions. Neither audit opened raw or processed waveform payloads or ran inference. Consequently, raw-voltage per-case scores and anchor/mask generation are verified through frozen declarations and aggregate checks, rather than independent waveform regeneration.

Both audits together consumed 9.370591353 CPU seconds. Complete cumulative scientific CPU was 545.144970413 seconds; new-stage CPU 111.012039341 seconds and cumulative GPU-wall 226.412812073 seconds. Actual stream storage before the metrics audit’s final receipt was 3,478,092,294 bytes, below 4 GiB. The original simulator advancement and this study’s full advancement both remain failed. No further fits, extra seeds, threshold changes or pushes are scheduled by these results.

## Follow-up decisions

Keep the observed SSL response gain as a single-seed engineering finding. Diagnose the random arm’s strong pre-edit repeated-phase alignment and the SSL control flags using saved outputs before specifying another fit. Its nuisance-control rate was 6.25%, versus 5.47% on unchanged controls; the failed rule concerns the S minus R nuisance-control increase, not proof that adding noise caused every flag. Do not relax that rule after observing the result.

Any new optimization or replication study needs a separate committed comparison and resource profile. The present evaluation patients are now development evidence. Recover and audit compatible expert annotations before clinical localization comparisons. A GAN remains a separate hypothesis requiring a measured benefit over controlled editing at a matched budget.

## Local immutable evidence

| Receipt | SHA-256 |
| --- | --- |
| `preparation/summary.json` | `38a9c533be62157d0696c14468da9f4d908338069b54c6fe3bbe2790bdca2f18` |
| `profile/summary.json` | `07cdf8c7517aa686337b16cb9128102c6e3d1c9954cc1c1edea26f6fc0658d76` |
| `training/summary.json` | `1896e9c638f27590d4a90c582536926dfbc85ffe57db1f08da12f7e3da8bf44c` |
| `evaluation/summary.json` | `5041fab51ce83a8ea2bc89beed347c0981d5e53de8852233e31410400f07e9a4` |
| `training_audit/summary.json` | `55b50f8c9705aeacfce7e19334a1b987a0ebb527b3924f2ff362eb099a10359a` |
| `metrics_audit/summary.json` | `464a438c0a3e48ed7a60bf2f7ea2195fcb23b26369c45394c00fa8aaaa118976` |

Source, checkpoint, data, calibration and output bindings remain in the local receipts. Patient IDs, waveforms, masks, weights and probability maps are excluded from Git. The qualification and profile/training reports provide their detailed execution accounting.
