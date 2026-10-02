# Experiment 054 results: aligned residuals with normal phase calibration

The method improves the automatic premature-complex timing proxy, but fails both independent synthetic
location checks. It is not promoted. The existing referral detector and explanation pipeline remain
unchanged. This is a mixed exploratory result, not evidence of correct clinical wave or anatomical
localization.

Completed 2 October 2026 under the [frozen protocol](experiment-054-aligned-phase-residual.md) and
[exact patient manifest](experiment-054-cohorts.json), committed before scoring as `da21b67`.
Implementation and the prospective calibration-channel permutation clarification were committed as
`2eb46c5`. The CPU run took 117.18 seconds with two threads and no GPU. Live outputs are
`outputs/experiment054_aligned_phase_residual_v1/`: receipt, normal references, development and synthetic
maps, per-record tables, calibration/development shifts, local template arrays, 20 figures, and run log.
No waveform was downloaded, uploaded or sent to an external service. No calibration, closed test,
EchoNext or MIMIC waveform was read.

## Primary timing comparison

All 1,604 development ECGs were usable. Evaluation uses the same 140 ms candidates and exact common
eligible sample mask as Experiment 052. Both maps average over tied maxima, with the frozen numerical
tolerance. The target remains the automatic early complex's R - 0.06 through R + 0.08 second support;
RR timing is not a score feature.

| On 73 PVC ECGs with automatic targets | Aligned phase (primary) | Aligned only (descriptive) | U_B fixed support |
| --- | ---: | ---: | ---: |
| Window-center temporal hit | 0.630 | 0.671 | 0.488 |
| Uniform eligible-window chance | 0.048 | 0.048 | 0.048 |
| Hit minus chance, patient interval | 0.582 [0.491, 0.671] | 0.623 [0.519, 0.727] | 0.440 [0.334, 0.543] |
| Median distance from closest target R | 60 ms | 42 ms | 72 ms |
| Mean target overlap / displayed width | 0.549 | 0.578 | 0.504 |
| Records with tied maxima | 69 | 0 | 40 |
| Mean maximum tie count | 33.88 | 1.00 | 5.92 |

The primary paired gain is **+0.142 [+0.001, +0.287]**, exceeding the required +0.10 point gain, with a
lower bound narrowly above zero. This passes the real-data proxy rule. Intervals use 2,000 whole-patient
draws with seed 54054. The baseline point estimates equal 052's; its interval differs because the
successor uses its prespecified new bootstrap seed.

The primary has a greater center-hit expectation than U_B on 33 records, a smaller one on 29, and the
same one on 11. These counts include fractional tied-top hits. The automatic proxy is reused
development evidence, not independently annotated clinical truth. The larger aligned-only center-hit
is descriptive and cannot replace the frozen primary recipe.

## Independent known-support edits

The exact manifest separates 1,000 normal-reference patients, 300 threshold-control patients and 200
new synthetic patients. All three sets are mutually patient disjoint, exclude the development patients,
and exclude 052's 200 synthetic patients. All promised records were usable; there were no replacements
or exclusions. The reference contains finite record maxima for all 48 lead/phase channels, with exactly
1,000 references in every channel. Previous project use of training normals is not a new clinical
holdout; no 053 model or parameter enters this recipe.

| New synthetic edit | Primary exact time-and-lead joint hit, patient interval | Primary worst-score change from unchanged copy | Supporting rule |
| --- | --- | --- | --- |
| Single-lead 0.15 mV ST bump | 0.125 [0.080, 0.175] | +0.0328 [+0.0176, +0.0500] | Fails 80% localization |
| Three-lead stretched QRS transplant | 0.593 [0.528, 0.660] | +0.5569 [+0.4649, +0.6493] | Fails 80% localization |
| ST moved to another interior beat | 0.113 [0.070, 0.158] | +0.0374 [+0.0202, +0.0572] | Descriptive |
| ST with cyclic lead and reference permutation | 0.125 [0.080, 0.175] | +0.0328 [+0.0176, +0.0500] | Descriptive |

The primary's median distance from the exact edit center is 2.200 seconds for ST and 66 ms for QRS.
Mean support overlap divided by the fixed displayed width is 0.089 and 0.570. Statistically positive
score changes do not compensate for selecting the wrong time or lead. The conservative CDF caps and
ties are included in the joint success, rather than resolved by a favorable display choice.

The aligned-only ablation also fails: ST joint hit is 0.075 [0.040, 0.110] and QRS is 0.390
[0.325, 0.455]. Its positive paired score-change intervals do not rescue localization. These edits are
engineering ground truth with fixed original R anchors; they do not simulate validated diseases. The
template-based synthetic QRS transplant is especially favorable to a template residual method.
Because 054 uses different synthetic patients and seed from 052, their percentages are not a paired
method comparison.

Promotion required the real timing rule and both synthetic location rules together. The conjunction
fails, so `promote_for_review` is false. No threshold was relaxed and no alternative recipe was selected
after seeing the result.

## Normal and nuisance controls

Thresholds come from the 300 separate unchanged control patients, not the synthetic holdout itself.
The primary threshold is 5.52246; aligned-only is 0.675283. On the 200 synthetic-holdout unchanged ECGs,
the primary flags 6%, and the constant-offset-plus-slow-drift nuisance copies also flag 6%. The aligned
ablation flags 8% in both cases. Thus neither has the prespecified descriptive concern of a nuisance
increase exceeding five percentage points.

The primary nuisance changes the mean worst score by +0.0320 [+0.0079, +0.0581], despite leaving the flag
share unchanged. The aligned-only change is +0.00734 [+0.00233, +0.01229]. This is residual sensitivity
to a smooth baseline change; unchanged flags do not prove full nuisance invariance. Synthetic ST
lead-permutation results exactly match the original ST results when calibration channels follow the
waveform permutation.

## Ties, alignment and the motivating example

The empirical record-max CDF reaches its cap log(1001) on 76 of the 1,604 development ECGs. Even below
the cap, its step scores create ties. On the 73 PVC records, 69 have tied maxima, with a mean of 33.88
maximal windows. This limits how precise the displayed single mark can be; primary metrics average all
maxima. The deterministic display uses a median-time tied maximum and is only illustrative.

Across 18,447 development beats, 81.4% use zero alignment shift; 2.62% use the maximum allowed absolute
shift of five samples. Alignment never fits per-beat amplitude. Extended templates predict each sample
back on its original timeline, without changing target R positions or padding eligible score windows.

ECG 287's calibrated display is V2 at **6.05–6.19 seconds**, on the visibly odd wide beat. It has 14 tied
maxima, an expected QRS-center hit of 0.929, and mean distance of 35 ms from the automatic R target.
Aligned-only displays V2 at 6.07–6.21 seconds, with a 26 ms distance. U_B still displays V1 at
9.17–9.31 seconds, 3.126 seconds from that target. This individual improvement is reproduced, while the
synthetic controls prevent promotion of the method as a general locator.

All seven fixed examples have local map-comparison figures and original-beat/aligned-template overlays.
The lowest-ID real gain is 287; the lowest-ID real loss is 219. For 219, calibrated tied-max center-hit
is 0.542 versus 1.000 for both U_B and aligned-only. Lowest-ID synthetic joint successes and failures
for both edit types are saved (success 590, failure 595). The maps, calibration references and
development/synthetic per-record tables are persisted, so examples are auditable rather than selected
only for favorable appearance.

## Integrity, verification and limitations

Before any 054 score, the full historical U_B metrics reproduced exactly. All frozen 052 protocol,
source/input/output hashes matched. Its entire primary metric dictionary, complete descriptive map
metrics and all synthetic summary fields, including nuisance intervals, reproduced exactly from
stored artifacts. Patient separation and manifest hashes passed. The new receipt includes the exact
cohorts, source/protocol hashes and 6,208 read waveform header/data hashes. Six new scientific invariant
tests plus 052's seven tests passed, and Ruff passed before launch. Tests cover shared-shift correction,
original-time mapping, amplitude-change preservation, offset/translation/lead equivariance, finite
extended-edge predictions, phase coverage and conservative equal-value CDF tails.

No source was changed after scoring. No outcome-driven method, target, support, split or threshold
change occurred. The phase bins are alignment strata, not clinical wave delineations. The real target
is automatic and timing-derived; lead truth is available only for artificial edits. Normal reference
calibration improves the exploratory timing proxy but does not establish correct lead or wave
localization. A within-record template also subtracts persistent abnormalities and cannot replace
the existing referral detector or identify an anatomical lesion.

## Next decision

The current local data cannot validate clinical fragment correctness. Further recipe variants on the
same automatic timing proxy would not resolve that limitation. The next meaningful decision is to
obtain independent beat/region annotations, through cardiologist review or an approved public
beat-annotated dataset, and freeze a localization evaluation around those labels. No new data was
downloaded and no additional method variant is launched after this result.
