# Experiment 055 results: independent physiological wave-boundary validation

The frozen full-wave candidate failed its promotion rule. It improved record-macro
P/QRS/T overlap from 0.557 to 0.651, a gain of +0.095 [95% interval +0.080, +0.109],
but the required gain was at least +0.10 and P-wave overlap deteriorated. The
QRS and T components showed substantial improvements against independent cardiologist
boundaries; these are useful secondary findings, not permission to replace the failed
primary recipe with a selected component after scoring.

Completed 2 October 2026 in 151.9 seconds, CPU only, from source commit `7164258`.
The [protocol](experiment-055-ludb-wave-boundaries.md) was committed in `5069b24`,
with annotation-format amendment `e0f4c5e` before any score. Live receipt:
`outputs/experiment055_ludb_boundaries_v1/result.json`; local per-record comparisons,
predictions, data audit, source/input/output hashes and fixed figures are in that folder.
No annotation-guided fit, threshold tuning or recipe selection occurred. All 200 LUDB
subjects were evaluation-only; they are now development evidence for any successor.
No closed project test, new clinical label or age subgroup was read or created.

## Public source and exact predecessor integrity

[LUDB 1.0.1](https://physionet.org/content/ludb/1.0.1/) was downloaded locally after
sandbox DNS restrictions required an authorized network escalation. Slow HTTP transfer
required resuming a partial archive; the completed archive hash is in the launch receipt.
Its 2,806 extracted files include 2,805 files covered by the published SHA256SUMS.
Every published checksum matched. Version 1.0.1 has the corrected waveform scaling;
all 200 records passed 12-lead, 500 Hz, 5,000-sample, finite-value and physical-unit audits.
All leads are supplied in mV and were reordered by name to the project's canonical order.
The [official README](https://physionet.org/content/ludb/1.0.1/README) states 200 records
from 200 subjects, supporting whole-record subject bootstrap intervals.

Before LUDB scoring, the live Experiment 042 score archive matched its recorded hash,
and its cached unit maps reproduced all prespecified scalar metrics exactly (maximum
absolute discrepancy zero): detection AUROC 0.7403160060362431; anterior contrast
0.08458214581226442; inferior contrast 0.1151512365541969; PVC top-unit hit
0.9315068493150684, chance 0.19715822833268443 and excess 0.7343486209823841;
normal, positive and benign any-red rates 0.05183585313174946,
0.2858837485172005 and 0.07692307692307693. The 73 eligible PVC records and
all 1,604 development rows matched their predecessor. This check used the frozen
cached reference; no baseline covariance or clinical labeling was changed.

## Annotation quality and evaluation denominators

A whole-dataset format audit executed before any score. The source contains 31 lead
streams with nonstandard/incomplete sequences. Twenty complete groups with typed
peaks 1–3 samples just outside an otherwise valid boundary span were recovered under
the pre-score 10 ms policy; their original onset and offset were preserved. Of these,
13 peaks were one sample outside, six were two samples outside and one was three
samples outside. No event was reused and a distant neighboring peak was not borrowed.
The source left 240 explicitly audited unknown annotation events (193 invalid/distant
candidate-group events and 47 partial events); missing onset/offset/type was not imputed.
These are event counts, not a claim of 240 distinct missing waves.

| Wave | Published wave count | Raw typed peaks | Valid complete/recovered intervals | Eligible truth intervals | Edge-excluded truth | Eligible truth without a matched prediction slot |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P | 16,797 | 16,797 | 16,797 | 16,785 | 12 | 10 |
| QRS | 21,966 | 21,965 | 21,862 | 21,843 | 19 | 0 |
| T | 19,666 | 19,661 | 19,650 | 19,650 | 0 | 96 |

All 21,862 valid annotated QRS peaks matched a raw R anchor within the prespecified
150 ms limit. P and T association was one-to-one with annotated beats and used the
same raw anchor assignment for both arms. Association uses complete truth including
edge waves before applying the interior eligibility filter; rare edge competition
can affect an interior assignment equally across both arms.

Every eligible complete truth interval contributes, including unmatched slots and
rejected predictions with IoU zero. Unknown/incomplete annotation fragments are
unknown boundary coverage, not normal samples. Entire annotated spans had to lie
in [0.5,9.5] seconds. Missing predictions receive a 500 ms boundary-error penalty.
Records without a class do not invent truth for that class: P truth appears in 176
records; QRS and T truth appear in all 200. The primary averages available classes
equally inside each subject and then subjects equally. This prevents the much more
frequent QRS waves or multiple beats/leads from acting as independent subjects.

## Primary comparison and physiological wave endpoints

Both arms used the identical frozen raw-signal R detector. Fixed offsets came from
042; candidate derivative-energy QRS and amplitude/noise P/T boundaries were frozen
without annotated training. Intervals are sample spans; overlap is intersection-over-union.
Whole-subject bootstrap uses 2,000 draws, seed 55055, with shared patient resampling helpers.

| Wave | Fixed IoU | Adaptive IoU | Paired subject-macro gain [95% interval] | Frozen per-wave guardrail |
| --- | ---: | ---: | --- | --- |
| P | 0.502 | 0.418 | -0.084 [-0.114, -0.054] | Fail |
| QRS | 0.668 | 0.839 | +0.171 [+0.160, +0.181] | Pass |
| T | 0.491 | 0.668 | +0.176 [+0.148, +0.202] | Pass |
| Primary record-macro | 0.557 | 0.651 | +0.095 [+0.080, +0.109] | Fail: gain <0.10 and P deteriorates |

QRS overlap improved by +0.171 [+0.160, +0.181] without a missing QRS prediction.
T overlap improved by +0.176 [+0.148, +0.202]. P overlap fell by -0.084
[-0.114, -0.054], with many rejected candidate slots. The full candidate therefore
cannot be promoted despite supported improvements in its QRS/T components.

| Wave and method | Onset error, ms | Offset error, ms | Boundaries within 30 ms | Missing predictions, subject-macro |
| --- | ---: | ---: | ---: | ---: |
| P, fixed | 51.1 | 45.0 | 28.9% | 0.1% |
| P, adaptive | 222.8 | 215.7 | 47.8% | 41.4% |
| QRS, fixed | 15.8 | 31.8 | 69.8% | 0.0% |
| QRS, adaptive | 8.0 | 10.8 | 96.4% | 0.0% |
| T, fixed | 48.7 | 106.6 | 20.1% | 0.5% |
| T, adaptive | 67.1 | 80.2 | 61.7% | 8.7% |

QRS endpoints within 30 ms rose from 69.8% to 96.4%. P's apparent increase in
within-30-ms endpoints does not undo its many missing predictions: all rejected
slots remain in the IoU and error denominators. T onset error increased because
its failures carry the frozen penalty, even while average interval overlap improved.

| Wave | True duration median [25th,75th percentile], ms | Fixed proposed duration, ms | Adaptive valid proposed duration median [25th,75th percentile], ms |
| --- | --- | ---: | --- |
| P | 96 [82,112] | 190 | 112 [88,144] |
| QRS | 94 [84,106] | 140 | 100 [90,110] |
| T | 180 [154,204] | 250 | 184 [154,240] |

Adaptive duration summaries condition on valid predictions; they do not remove
missing predictions from the primary result.

## Unannotated slots and fixed local review

| Wave | Fixed prediction slots without eligible annotation | Adaptive slots without eligible annotation | Adaptive rejected interior slots |
| --- | ---: | ---: | ---: |
| P | 7,549 | 4,311 | 10,147 |
| QRS | 2,481 | 2,481 | 0 |
| T | 4,770 | 4,352 | 2,149 |

These unannotated-slot counts cannot establish false positives or wave-presence
specificity: absence may reflect incomplete/source coverage. Nor can fewer proposed
P waves excuse missing a known annotated P wave. Subject-level overlap remains the
promotion endpoint.

All fixed review records 1, 50, 100, 150 and 200 were plotted on leads II and V1,
with cardiologist spans in green, fixed endpoints in blue and adaptive endpoints
in red. Figures stay local under `outputs/experiment055_ludb_boundaries_v1/figures/`.
Visual inspection of records 1 and 100 showed tighter QRS boundaries and missing
candidate P/T slots; the accompanying per-record comparisons preserve these failures.
No example was selected by its outcome.

## Verification and next experiment

Twelve focused scientific tests passed and Ruff passed before launch. Checks cover
fixed-offset equality, conservative incomplete-source parsing, near-span peak recovery,
no distant-peak borrowing, one-to-one matching, excess atrial truth, unmatched truth
kept at zero, equal class weights, polarity/amplitude invariance, no constant-noise
wave invention and native JSON receipt scalar types. The source and scored outputs
remain frozen and were not retuned after these results.

The useful successor idea is a hybrid that keeps the fixed P window and uses the
new QRS/T boundaries, but this was not the frozen primary recipe and has not been
validated here as a promoted model. It needs a new prospective protocol and independent
boundary data; all 200 LUDB records have now been scored. Improving physiological
wave boundaries still does not establish that a red wave is pathological, and later
integration with a normal-reference anomaly map needs its own frozen localization
comparison. Send this evidence to the parent-maintained backlog rather than changing
clinical definitions or lowering the failed promotion threshold.
