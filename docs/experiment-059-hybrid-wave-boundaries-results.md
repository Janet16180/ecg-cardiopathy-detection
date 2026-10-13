# Experiment 059 results: improved QRS boundary localization

The frozen hybrid passed every prospective review gate on the 72 QTDB records
whose original source was not European ST-T. Joint QRS intersection-over-union
rose from 0.681702 to 0.793725: paired gain +0.112023, record-bootstrap 95% CI
[0.088979, 0.136357]. QRS boundaries within 30 ms rose from 65.4813% to 91.5352%,
a +26.0539 percentage-point gain [20.8740, 31.6513]. This is an improvement in
physiological QRS boundary localization compared with the frozen fixed windows.
It does not identify a diseased wave or improve the frozen classifier.

These numbers come from the completed local receipt
`outputs/experiment059_qtdb_hybrid_v2/result.json` and its 105 record JSON/NPZ
pairs. The recipe keeps fixed P and uses unchanged Experiment 055 QRS/T
morphology, taking the earliest onset/latest offset across valid channels.
It uses raw waveform R detection rather than expert anchors for inference.
No QTDB labels fitted or selected the method.

## Frozen gates and complete primary population

All 72 non-EDB records supplied eligible complete QRS truth: 2,582 intervals.
Every truth interval contributed; unmatched or missing predictions would score
IoU zero and incur 500 ms boundary error. Neither arm missed a primary QRS slot.
Each record received equal weight, followed by 2,000 paired original-record
bootstrap draws, seed 59059. Universal patient IDs are unavailable, so these
are record intervals, not independently verified patient intervals.

| Prospective gate | Executed result | Passed |
| --- | --- | --- |
| QRS IoU point gain >=0.10 | +0.112023 | yes |
| QRS IoU lower 95% bound >0 | +0.088979 | yes |
| QRS boundary-within-30-ms point gain >=0.10 | +0.260539 | yes |
| T-offset MAE worsening upper bound <=10 ms | +2.621 ms | yes |
| P predictions exactly equal | verified in all 105 records | yes |

The T guard includes all known annotated offsets, including intervals whose
onset is unknown and failed slots penalized at 500 ms. Its distinct population
contains 70 non-EDB records; two have no known T offset. Record-macro T-offset
MAE was 92.698 ms fixed and 75.017 ms hybrid, a deterioration of -17.681 ms
[-35.987, +2.621]. This supports the prespecified noninferiority guard; it does
not establish a statistically certain improvement in every T-wave endpoint.

## Wave-specific findings and failures

| Complete truth endpoint | Records / waves | Fixed | Hybrid | Paired IoU gain, 95% CI |
| --- | --- | --- | --- | --- |
| P IoU | 65 / 2,153 | 0.519085 | 0.519085 | 0.000000 [0, 0] |
| QRS IoU | 72 / 2,582 | 0.681702 | 0.793725 | +0.112023 [0.088979, 0.136357] |
| T IoU | 36 / 1,006 | 0.543385 | 0.605241 | +0.061857 [-0.024673, 0.147744] |

QRS onset MAE fell from 16.570 to 8.959 ms; QRS offset MAE fell from 32.812
to 19.367 ms. P remains a limited fixed window: only 33.123% of its complete
boundaries are within 30 ms. Retaining P avoids Experiment 055's regression,
but does not solve P delineation.

Complete T-wave overlap has an interval crossing zero. Its onset MAE worsened
from 52.747 to 75.282 ms, offset MAE from 91.668 to 94.543 ms, and record-macro
missing-prediction rate from 0 to 7.298%. The complete-T subset differs from
the all-known-offset guard; neither subset is silently substituted for the other.
The hybrid should therefore be reviewed primarily for QRS support, with T
limitations visible.

Every source-family QRS point gain was positive. Descriptive family gains were
MIT arrhythmia +0.114663 (15 records), MIT ST +0.157191 (6), MIT supraventricular
+0.128516 (13), MIT normal sinus +0.135601 (10), MIT long-term +0.118876 (4), and
sudden death +0.079180 (24). These are not independently selected successes.
The sudden-death family exposes a T weakness: its known-offset MAE deterioration
was +25.885 ms [-18.923, +77.291], despite the aggregate guard passing.

All-105 descriptive QRS IoU was 0.676624 to 0.786113, gain +0.109489
[0.088581, 0.126765]. The 33 EDB-derived excerpts separately gave +0.103961
[0.069029, 0.136831]. Those records were excluded from the primary because
Experiment 057 had already accessed their original source. They are not a
new independent replication.

## Data, provenance and integrity

Downloaded [QTDB 1.0.0](https://physionet.org/content/qtdb/1.0.0/) raw data,
headers and expert `.q1c` annotations locally: 315 official-checksummed files,
71,003,581 bytes, plus RECORDS and SHA256SUMS. All 105 records were processed
at native 250 Hz with two channels. Their lengths are 225000 samples (53),
224999 (29), or 224993 (23); none was padded, excluded or resampled.

Annotation-format auditing accounted for 8,336 complete groups and 2,844
onset-missing groups (2,130 T and 714 U), with zero unknown events, malformed
groups or displaced peaks. Native typed counts were P 3,194, QRS 3,623,
T 3,542, U 821; the source paper describes 3,622 selected beats, whereas the
verified current annotation files contain 3,623 QRS intervals. The evaluator
preserves the official files without forcing this descriptive total to agree.
QRS symbols A, B, Q and V were included alongside N. All 3,623 expert QRS
peaks matched independent raw anchors; 111,126 raw anchors were inferred overall.

Sparse annotation coverage is explicit: 107,799 P and 107,369 QRS prediction
slots lacked eligible truth in both arms. T had 107,450 such fixed slots and
104,031 valid hybrid slots. These slots are UNKNOWN, not false positives or
evidence of specificity. Experts viewed both channels together, so joint
support is evaluated against joint truth without a best-channel oracle.
The [published source mapping](https://physionet.org/physiobank/database/qtdb/doc/node4.html)
also identifies estimated gains for sudden-death recordings; no absolute
clinical amplitude claim is made.

Before any QTDB metric, exact Experiment 055 integrity passed: 17 pinned
sources, 2,806 inputs and 408 immutable outputs verified, all 200 record
summaries reconstructed, and every saved per-wave metric, primary mean and
paired interval reproduced exactly. The only omitted predecessor hash is
its mutable run.log, whose final message was appended after receipt hashing.
Predecessor primary remains 0.5565263444483334 to 0.6514050075554896,
gain 0.0948786631071562 [0.08041680553663626, 0.10901857121219544].

Protocol commits are c4dd7d5 and the pre-score native-format clarification
ec7c041. Initial source 43c5cff passed predecessor integrity but stopped at
its fixed-P identity guard before any QTDB metric or record artifact: a negative
recording-edge P onset was incorrectly converted to missing. Preserve its
launch/failure/integrity receipts and empty records directory under
`outputs/experiment059_qtdb_hybrid_v1`. New source cacf7e8 copies fixed P verbatim;
its new helper, runner, regression test and execution note preserve the failed
sources. The correction changed no scientific method or decision rule.

V2 completed in 32.757 seconds with two CPU threads, no GPU, and approximately
19.58 MB of local artifacts. Eighteen focused tests and Ruff passed. Synthetic
signals tested invariance and matching behavior only; no synthetic accuracy
result was promoted. The receipt pins 317 input hashes and 217 immutable
output hashes. All source/protocol hashes and per-record raw artifacts remain
local; Git contains only source, tests and this aggregate report.

Fixed local review figures are sel100, sel232, sel16265, sel30 and sele0106.
They include both channels with joint expert shading, fixed blue and hybrid
red boundaries; missing T onsets are unshaded. These examples were prescribed
before scoring and were not selected for favorable performance.

An independent scientist verified 15 source, 317 input and 217 output hashes,
reconstructed all 10,359 saved wave measurements and missing penalties, checked
unique matching and P identity in all 105 records, and independently rebuilt
the primary and T-guard bootstrap intervals to within 1e-13. The audit confirms
every gate and the QRS-only claim, with receipt
`outputs/experiment059_qtdb_hybrid_audit_v1/audit.json`. Inference accepts only
waveforms and sampling frequency; expert labels enter the evaluator, not the
boundary function. Boundary-within-30-ms percentages count start/end endpoints,
not the fraction of entirely correct QRS intervals.

The local `outputs/localization_iteration_review_2026_10_02/index.html` provides
simpler previous/adaptive panels for the first annotated QRS in four prescribed
records and the largest record-level QRS regression. Expert timing is joint
across channels. Display-only baseline shifts do not affect any measurement.

The result warrants reviewing adaptive QRS support in the localization layer.
It has not yet demonstrated that replacing fixed supports improves U_B
pathology localization, has no lead-specific truth, and is not a test of the
university screening population. P, T and noisy-record behavior remain limited.
All 105 QTDB records are now development data. A subsequent integration study
must freeze its score/support comparison on new independent truth; no retuning
or subgroup selection on this evaluation can serve as another independent test.
