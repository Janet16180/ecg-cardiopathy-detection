# Saved-map control and phase diagnostic results

18 October 2026. The one forward diagnostic completed using existing saved maps only. Seven of S’s
eight evaluation nuisance-control flags were already flagged on the same patients’ unchanged
controls; one appeared with the nuisance bundle and none disappeared. R’s one control flag persisted
in both versions. The failed parent veto still compares eight S nuisance flags with one R nuisance
flag, a seven-patient difference. The within-S net addition of one patient answers a different
question and does not repair that failure.

R’s output already favored algorithmic candidate windows before intervention and changed little
after addition, especially for repeated edits. S showed a much larger introduced-change response.
These are saved-output patterns, not proof of which internal features either model uses. Both parent
advancement failures remain unchanged; no threshold, model or decision criterion was fitted or
altered.

## Exact paired control flags

Patient-bootstrap intervals are conditional on these fixed fitted models and this one training
schedule/initialization; they do not estimate training-order or pretraining-seed uncertainty.

Flags use the original strict maximum-map threshold: R 0.923113226890564 and S 0.9675309062004089.
Calibration and evaluation patients remain separate. Intervals are paired whole-patient 95%
bootstrap intervals, 2,000 draws, seed 20261026, with no skipped draws. Calibration rows are
fitted-data checks, not independent specificity estimates. Zero intervention support does not label
a patient healthy.

| Role / arm | Patients | Neither | Unchanged only | Nuisance only | Both | Nuisance minus unchanged rate, 95% interval |
|---|---:|---:|---:|---:|---:|---:|
| calibration / R | 64 | 61 | 1 | 1 | 1 | 0.000000 [-0.046875, 0.046875] |
| calibration / S | 64 | 61 | 1 | 0 | 2 | -0.015625 [-0.046875, 0.000000] |
| evaluation / R | 128 | 127 | 0 | 0 | 1 | 0.000000 [0.000000, 0.000000] |
| evaluation / S | 128 | 120 | 0 | 1 | 7 | 0.007812 [0.000000, 0.023438] |

Each arm’s joint calibration-control count was three of 64, consistent with the empirical higher-95%
threshold and strict comparison. No recalibration occurred. In evaluation, S unchanged/nuisance
rates remain 7/128 (5.47%) and 8/128 (6.25%); R remains 1/128 for both.

| Role / control | Neither arm | R only | S only | Both arms | S minus R rate, 95% interval |
|---|---:|---:|---:|---:|---:|
| calibration / sham | 59 | 2 | 3 | 0 | 0.015625 [-0.046875, 0.093750] |
| calibration / nuisance | 60 | 2 | 2 | 0 | 0.000000 [-0.062500, 0.062500] |
| evaluation / sham | 120 | 1 | 7 | 0 | 0.046875 [0.007812, 0.093750] |
| evaluation / nuisance | 119 | 1 | 8 | 0 | 0.054688 [0.007812, 0.101562] |

The evaluation flag sets for R and S were disjoint. The failed nuisance veto remains S minus R =
7/128 = 0.0546875, above its frozen 0.05 point limit. The original gate uses its original
seed-20261025 inference; the diagnostic’s seed-20261026 intervals do not replace that decision.

## Native maximum scores and signed margins

Signed margins subtract each arm’s own frozen threshold. Raw probabilities and margins are
within-arm descriptions, not calibrated confidence comparisons across arms. The bundle combines
gain, drift and noise; this analysis cannot isolate their individual effects or native pathology.

| Role / arm | Mean unchanged maximum | Mean nuisance maximum | Nuisance minus unchanged maximum, 95% interval |
|---|---:|---:|---:|
| calibration / R | 0.862743 | 0.862232 | -0.000511 [-0.004053, 0.002332] |
| calibration / S | 0.736016 | 0.804024 | 0.068008 [0.043137, 0.096634] |
| evaluation / R | 0.868967 | 0.869710 | 0.000742 [-0.000591, 0.002060] |
| evaluation / S | 0.748816 | 0.803471 | 0.054655 [0.039418, 0.069807] |

For S’s seven persistent evaluation flags, the mean maximum changed by only 0.000133. The one new
nuisance flag crossed from margin −0.018751 to +0.006713, a maximum increase of 0.025464. The 120
patients flagged in neither version nevertheless had a mean maximum increase of 0.058079. Thus one
extra threshold crossing does not mean only one map responded to nuisance.

| Role / arm / fixed transition | Count | Mean unchanged margin | Mean nuisance margin | Mean maximum change |
|---|---:|---:|---:|---:|
| calibration / R / neither | 61 | -0.063386 | -0.064155 | -0.000769 |
| calibration / R / sham_only | 1 | 0.000365 | -0.004565 | -0.004930 |
| calibration / R / nuisance_only | 1 | -0.013729 | 0.007205 | 0.020934 |
| calibration / R / both | 1 | 0.016222 | 0.014429 | -0.001793 |
| calibration / S / neither | 61 | -0.243499 | -0.172072 | 0.071426 |
| calibration / S / sham_only | 1 | 0.010660 | -0.000185 | -0.010845 |
| calibration / S / nuisance_only | 0 | null | null | null |
| calibration / S / both | 2 | 0.012922 | 0.016090 | 0.003168 |
| evaluation / R / neither | 127 | -0.054786 | -0.054148 | 0.000638 |
| evaluation / R / sham_only | 0 | null | null | null |
| evaluation / R / nuisance_only | 0 | null | null | null |
| evaluation / R / both | 1 | 0.027172 | 0.041122 | 0.013951 |
| evaluation / S / neither | 120 | -0.234208 | -0.176129 | 0.058079 |
| evaluation / S / sham_only | 0 | null | null | null |
| evaluation / S / nuisance_only | 1 | -0.018751 | 0.006713 | 0.025464 |
| evaluation / S / both | 7 | 0.018323 | 0.018456 | 0.000133 |

All fixed-category counts, means, medians, minima/maxima and linear 5th/95th percentiles are
retained in the local receipt, including explicit null summaries for empty categories. The groups
were prospectively specified and are descriptive; no subgroup was selected for a new test.

## Algorithmic phase alignment before the introduction

Candidate C contains every algorithmic window on every lead, independent of the realized edited lead
or focal anchor. Control maps have no added intervention. A higher mean in C than outside C is
consistent with an output pattern aligned to those windows. It does not identify a causal internal
feature or expert-verified ST boundary.

| Role / arm / control | Mean in C | Mean outside C | C minus outside, 95% interval |
|---|---:|---:|---:|
| calibration / R / sham | 0.327706 | 0.030865 | 0.296841 [0.281839, 0.308255] |
| calibration / R / nuisance | 0.326359 | 0.031547 | 0.294811 [0.279329, 0.307128] |
| calibration / S / sham | 0.173084 | 0.014682 | 0.158402 [0.145669, 0.172538] |
| calibration / S / nuisance | 0.188677 | 0.016932 | 0.171746 [0.159596, 0.185212] |
| evaluation / R / sham | 0.329176 | 0.032163 | 0.297013 [0.289870, 0.303864] |
| evaluation / R / nuisance | 0.328515 | 0.032978 | 0.295537 [0.288171, 0.302501] |
| evaluation / S / sham | 0.172818 | 0.015037 | 0.157780 [0.149339, 0.166485] |
| evaluation / S / nuisance | 0.184045 | 0.016485 | 0.167560 [0.159429, 0.175577] |

R already had mean candidate output about 0.329 versus 0.032 outside on unchanged evaluation
controls. At the future intervention sites T, its baseline mean was about 0.646. Its apparent
repeated-edit ranking therefore coexists with a strong pre-addition output pattern. This does not
show that R ignores the waveform entirely or establish why its training produced that pattern.

## Exact intervention-site levels and response

For all 256 positive cases per arm, T is exact support, U is candidate minus T and O is outside
candidate. Baseline uses the same patient’s frozen matching unchanged/nuisance map. Baseline/control
waveform equality is checked through frozen hashes and nuisance metadata; this saved-map diagnostic
does not reopen waveforms to regenerate that truth. Every mean, difference, absolute change and
margin uses float64 reduction of saved native float32 maps.

| Arm / mode | Baseline T | Edited T | Baseline U | Edited U | Baseline O | Edited O |
|---|---:|---:|---:|---:|---:|---:|
| R / all | 0.645988 | 0.646895 | 0.284954 | 0.285516 | 0.032656 | 0.032355 |
| R / focal | 0.649195 | 0.650496 | 0.322951 | 0.323018 | 0.032656 | 0.032592 |
| R / repeated | 0.642780 | 0.643294 | 0.246958 | 0.248013 | 0.032656 | 0.032117 |
| S / all | 0.369282 | 0.631002 | 0.153414 | 0.204801 | 0.015893 | 0.016128 |
| S / focal | 0.376023 | 0.634601 | 0.174441 | 0.181461 | 0.015893 | 0.015926 |
| S / repeated | 0.362542 | 0.627403 | 0.132387 | 0.228140 | 0.015893 | 0.016330 |

| Arm / mode | Baseline-site AP | Edited AP | AP change, 95% interval | Change of T-minus-U contrast, 95% interval | Mean absolute map change | Mean case maximum absolute change |
|---|---:|---:|---:|---:|---:|---:|
| R / all | 0.339938 | 0.346928 | 0.006990 [0.003028, 0.011672] | 0.000346 [-0.000611, 0.001288] | 0.001432 | 0.035782 |
| R / focal | 0.061649 | 0.077251 | 0.015603 [0.007828, 0.024805] | 0.001233 [0.000165, 0.002389] | 0.000263 | 0.027198 |
| R / repeated | 0.618227 | 0.616604 | -0.001623 [-0.003010, -0.000224] | -0.000541 [-0.001645, 0.000477] | 0.002602 | 0.044366 |
| S / all | 0.266235 | 0.511194 | 0.244958 [0.201691, 0.288422] | 0.210333 [0.175036, 0.244423] | 0.031621 | 0.508324 |
| S / focal | 0.073131 | 0.427709 | 0.354578 [0.288835, 0.419946] | 0.251558 [0.207918, 0.294660] | 0.005177 | 0.439099 |
| S / repeated | 0.459339 | 0.594678 | 0.135339 [0.110627, 0.159594] | 0.169108 [0.141287, 0.195251] | 0.058065 | 0.577549 |

R’s repeated AP declined slightly from baseline-site 0.618227 to edited 0.616604; its mean absolute
map change was 0.002602. The overall R target-minus-U response interval still includes zero. S’s
overall T mean increased from 0.369282 to 0.631002, with T-minus-U response 0.210333
[0.175036,0.244423]. S had a clear within-arm target response. Comparative rank-response evidence
is the AP analysis below; raw probability magnitudes across arms are not calibrated confidence
contrasts.

| Mode | S minus R AP response, 95% interval |
|---|---:|
| all | 0.237969 [0.195541, 0.281287] |
| focal | 0.338975 [0.275202, 0.403755] |
| repeated | 0.136962 [0.111995, 0.161958] |

All point AP/response values reproduced the parent case metrics. The new seed-20261026 intervals are
descriptive diagnostics, not a replacement primary analysis. The receipt also stores all seventeen
per-case metric families with mode-specific and equal-within-patient pooled intervals, plus paired
R/S differences. Baseline-site AP measures ordering at the same fixed edit mask before addition; it
is not native pathology accuracy. These results cannot identify internal causal feature reliance.

## Fixed training-loss accounting

All 600 losses per arm were finite. The six saved progress milestones per arm matched receipt losses
at steps 100,200,…,600 exactly; there were not 600 independently logged milestones. Steps use the
same fixed patient/variant schedule. Their dependent losses receive no confidence intervals and
select no checkpoint.

| Arm | First 10 mean | Last 10 mean |
|---|---:|---:|
| R | 0.968116 | 0.422145 |
| S | 0.934176 | 0.289658 |

| Arm / steps | Mean weighted BCE | SD (ddof 0) | Minimum | Maximum |
|---|---:|---:|---:|---:|
| R / 1–100 | 0.745090 | 0.156266 | 0.459520 | 1.034773 |
| R / 101–200 | 0.481379 | 0.086821 | 0.352983 | 0.892705 |
| R / 201–300 | 0.431903 | 0.065422 | 0.297200 | 0.634648 |
| R / 301–400 | 0.426045 | 0.069427 | 0.331667 | 0.615249 |
| R / 401–500 | 0.416224 | 0.056791 | 0.314271 | 0.636487 |
| R / 501–600 | 0.410959 | 0.058465 | 0.314909 | 0.633250 |
| S / 1–100 | 0.576628 | 0.177036 | 0.370261 | 1.032526 |
| S / 101–200 | 0.430138 | 0.061195 | 0.330627 | 0.598531 |
| S / 201–300 | 0.406356 | 0.059437 | 0.284608 | 0.592378 |
| S / 301–400 | 0.381484 | 0.067668 | 0.245727 | 0.600347 |
| S / 401–500 | 0.345509 | 0.059010 | 0.192739 | 0.515797 |
| S / 501–600 | 0.316140 | 0.063053 | 0.164498 | 0.462400 |

R’s block mean decreased from 0.745090 to 0.410959 and S’s from 0.576628 to 0.316140. R’s later
blocks changed less. These optimization trajectories alone do not explain internal mechanisms,
establish localization validity, or justify more steps, another seed or checkpoint selection.

## Execution, retained stop and scope

The first immutable attempt stopped before diagnostic rows because its scientific tolerance was
incorrectly applied to a rounded accounting literal; see the [v1 stop
report](ssl-synthetic-localization-paired-diagnostic-results.md). The prospective [forward
policy](ssl-synthetic-localization-paired-diagnostic-v2.md) uses exact SHA-anchored receipt costs.
Original v1 code/configuration/failure were preserved, and no scientific endpoint or 1e-12
reproduction tolerance changed.

Forward code `da37e05` reused frozen arithmetic, predecessor admission and file-scope helpers
without modifying their globals. Protocol `2e19b1e`, new adapter and tests were committed and
root-integrated before execution. Sixteen artificial CPU tests passed in 0.49 seconds, Ruff passed,
and independent source review plus explicit V2 READY preceded the single forward run.

All eight parent summary blocks, frozen thresholds, all nine gates and 1,024 parent case flag/max
values reproduced. Every selected diagnostic case remained: 384 paired-control rows and 512 positive
rows across arms. There was no waveform/clinical/checkpoint payload read, model deserialization, new
inference, optimizer update, GPU use, threshold change, patient selection or new advancement rule.

Only the four prospective probability/mask arrays were opened. All 104 source, 230 input and two
output bindings are recorded, with 58 forbidden-payload digests checked only against the anchored
passed-audit catalogs. No generic waveform-opening receipt verifier was used. Checks were repeated
after calculation. Thus waveform identity, algorithmic anchors and physical support truth remain
verified through source-bound declarations, not independent waveform regeneration.

Actual v2 wall/CPU cost was 14.856351360/14.853658143 seconds. The retained failed attempt adds
1.387159193/1.386906838 seconds once, giving charged main-attempt wall/CPU totals
16.243510553/16.240564981, below the common 120-second limit. Cumulative scientific CPU after the
main diagnostic was 561.385535394 seconds; paired-stage CPU 127.252604322 seconds. The completed
independent audit is charged separately below.

Before the final main receipt, combined diagnostic-family outputs were 888,536 bytes and stream
storage 3,479,051,275 bytes, below 16 MiB/4 GiB with a 1 MiB final-receipt reserve. CUDA was hidden
and numerical threads were one. The forward run had no stop; the earlier accounting stop remains
retained.

## Independent verification

The independent audit passed 57,450 checks with zero mismatches at absolute numerical tolerance
1e-12. It reproduced all 384 paired-control rows, 512 positive rows, seven aggregate sections,
seventeen positive-statistic fields, paired patient intervals, fixed loss blocks, all eight parent
summary blocks, frozen thresholds and all nine decision gates of the failed conjunction. It freshly
verified 188 permitted files and retained 58 authority-only forbidden-payload exceptions.

The auditor used separate numerical reductions rather than production diagnostic calculators. It
performed no waveform regeneration, checkpoint/model read, inference, optimization or GPU action.
Calibration remained a fitted-data integrity check. Physical waveform identity, native masks and
algorithmic anchors still depend on the frozen parent declarations; this audit establishes
saved-output arithmetic within that scope, not native pathological localization or clinical
confidence.

Audit cost was 8.549395527 wall and 8.548247630 CPU seconds. All attempts plus this audit consumed
24.792906080 wall and 24.788812611 CPU seconds, below the 240-second combined allowance; main
attempts and audit also passed their separate 120-second ceilings. Complete cumulative CPU was
569.933783024 seconds and paired-stage CPU 135.800851952 seconds, including the retained v1 stop
once. All output and whole-stream storage gates passed. The original and paired full advancement
verdicts remain failed, with no new fitting or relaxed rule.

## Local immutable receipts

| Receipt | SHA-256 |
|---|---|
| `diagnostic_v1/failure.json` | `3d41af8ff1c592ee1b7f0a1ac89d9744e796aef3b72cee64b71155151565bb67` |
| `diagnostic_v2/summary.json` | `4d095ece86c26a665e647477c2713007a51ec12bcb4fe8615c94438945a698f7` |
| `diagnostic_v2/audit/summary.json` | `3180a8ea87a626294a629302a842184cda4f7ac1f438bfcef631e3199682e3c8` |

Paths are under `outputs/ssl_synthetic_localization/paired_background_v1/`. Local per-case rows,
masks, maps and identifiers remain untracked; only this aggregate report enters Git. Work remains
local without pushes or PR updates. No new fit follows automatically from this diagnostic.
