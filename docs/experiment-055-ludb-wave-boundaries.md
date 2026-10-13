# Experiment 055: label-free wave boundaries against independent LUDB annotations

Frozen 2 October 2026 before any boundary score. Experiment 042's P/QRS/ST/T
pieces are fixed offsets from R, so they do not establish which physiological
wave owns a red interval. This experiment asks whether deterministic morphology
boundaries overlap cardiologist-delineated P/QRS/T waves more accurately.
Success is anatomical wave-boundary localization, not identification of a
pathological wave or a new diagnosis.

## Data and independence

Download the public [LUDB 1.0.1](https://physionet.org/content/ludb/1.0.1/)
archive locally, audit its published SHA256SUMS, lead names, mV conversion,
500 Hz sampling and annotation grammar. Version 1.0.1 corrects waveform scaling.
The [official README](https://physionet.org/content/ludb/1.0.1/README) states
200 records from 200 subjects, so a whole-record bootstrap is a whole-subject
bootstrap. All 200 records are evaluation-only: no annotated training, model
selection, threshold fitting or recipe tuning. File-format inspection is permitted
before freezing; no annotation scores or label-directed waveform inspection is.
Metadata diagnoses, ages and sex do not enter the experiment. All raw data and
per-record outputs stay local and outside Git. No closed project test is read.

Before candidate scores, load live Experiment 042 cached U_B maps, verify their
recorded archive hash and reproduce its exact AUROC, first-argmax lead contrasts,
PVC hit/chance/excess and continuity normal/positive/benign marks from metadata.
The cached reference remains frozen; this experiment changes boundaries only.

## Frozen candidate

Both arms use exactly `fragment_localization.r_peaks` on raw 12-lead signals.
Inference uses raw waveforms only. The baseline uses 042 offsets P[-250,-60],
QRS[-60,+80], T[+200,+450] ms from R; ST has no independent annotation endpoint.
The candidate is per lead, with zero-phase second-order 0.5–30 Hz Butterworth
filtering and a separate second-order 12 Hz low-pass for P/T shapes.

- QRS: absolute temporal derivative, Gaussian smoothing sigma 6 ms. Search
  R[-120,+160] ms; choose the envelope maximum within R[-60,+60] ms.
  Keep the contiguous support around it above 12% of that peak, bridge gaps
  <=10 ms, add 8 ms on both sides, and clip to the search window. Minimum
  width is 20 ms centered on the selected peak. This does not use wave labels.
- P: search from max(R-350 ms, previous R+200 ms) to
  min(R-80 ms, inferred QRS onset-30 ms).
- T: search from inferred QRS offset+40 ms to
  min(R+600 ms, next R-150 ms, R+0.65*local RR).
  Local RR is the median of available preceding/following R distances;
  default 1 s for isolated R. The terminal beat uses the preceding RR.
- For P/T, subtract the line through the median first and last 10 ms of the
  search, use absolute smoothed amplitude (Gaussian sigma 8 ms), find its
  maximum, and estimate noise as 1.4826 times MAD in R[-100,-60] ms.
  Reject a slot if peak amplitude <=3*noise. Otherwise keep the first to
  last samples above max(15% of peak, 2*noise) within +/-100 ms (P) or
  +/-180 ms (T) of the maximum, add 6 ms each side and clip to the search.
  Disconnected lobes within this support are retained as one wave interval.
  Empty/invalid search slots return missing predictions.

All time limits are converted with round(seconds*500). There is one fixed
candidate recipe, no secondary parameter sweep and no post-score retuning.
Synthetic checks may test ordering, polarity/amplitude invariance and matching
without any LUDB evaluation waveforms or annotations.

## Matching, missingness and primary endpoint

Parse complete `(, p/N/t, )` onset/peak/offset triplets per lead. Audit and
report every incomplete or unexpected sequence; never silently discard it.
Only truth waves with their entire onset/offset in [0.5,9.5] seconds are eligible.
An absent/unmarked wave is not assumed to be a normal sample or a false-negative
annotation. Count candidate/baseline prediction slots without an associated
eligible annotation separately; do not claim wave-presence specificity.

Use the same score-independent one-to-one minimum-absolute-distance matching
between raw detected R anchors and each lead's annotated QRS peaks, maximum
150 ms. Match P to the nearest following annotated QRS peak within 450 ms,
and T to the nearest preceding annotated QRS peak within 650 ms. Only this
post-inference evaluator may use annotation anchors. Association is one-to-one
per wave class, preferring the nearest compatible wave when necessary;
any unassociated truth wave remains eligible and scores IoU=0 in both arms.
Unmatched QRS anchors, missing candidate slots or malformed boundaries score
IoU=0, never disappear from the endpoint.

Interval intersection-over-union uses continuous sample spans [onset,offset].
First average IoU over eligible waves/leads of each P/QRS/T class within a
record, then average the present classes equally within that record, then
average the 200 records equally. Records with no eligible truth in any class
score zero and are counted. Primary comparison is the paired per-record
candidate-minus-fixed-baseline mean IoU. A worthwhile improvement requires
point gain >=0.10, bootstrap 95% lower bound >0, and nonnegative point gains
for each of P, QRS and T. Bootstrap 2,000 whole-subject draws, seed 55055,
using shared patient grouping/resampling helpers. No subgroup model selection.

Secondary descriptive measurements: onset/offset absolute error (missing slot
penalty 500 ms), share of boundaries within 30 ms, per-wave IoU, QRS anchor
match coverage, annotated truth without assignment, rejected candidate slots,
unannotated prediction slots, duration distributions and raw R-anchor counts.
These cannot substitute for a failed primary decision.

## Artifacts and limits

CPU only, at most two numerical threads; download <=50 MB, audit+run <=20 minutes,
local artifacts <=250 MB. Preserve failed/completed runs and source files.
Store official/input/source/protocol hashes, raw metadata audit, subject IDs,
per-wave measurements, paired bootstrap intervals and a complete receipt.
Plot fixed records 1, 50, 100, 150 and 200 on leads II and V1, including failures;
all waveform figures remain local. Write the aggregate results report from
executed outputs and send follow-up evidence to the parent-maintained backlog.

This dataset supplies independently annotated boundaries, not proof that a model's
red wave is diseased. An improved delineator must later be integrated with a
normal-reference anomaly map in a new frozen comparison. No clinical label
change or age-subgroup analysis is part of this experiment.

## Pre-score annotation-format amendment

A whole-dataset format-only audit found 31 lead streams with incomplete or
nonstandard sequences before any waveform boundary score. Examples include
`N, (, )` where the typed peak is just before the annotated onset, `N, )`
with missing onset, and `(, )` without a typed peak. These are source coverage
issues; missing boundaries must not be reconstructed from inference.

Accept a contiguous, disjoint three-event group containing exactly one onset,
one typed p/N/t peak and one offset when onset<offset. Preserve its original
onset/offset; a peak outside the span is accepted only within 10 ms of the
nearest endpoint, with its displacement and group explicitly audited.
This prevents a distant peak from a neighboring partial wave being borrowed.
Every accepted event is used once. Partial/untyped groups, nonpositive spans
or more distant peaks remain UNKNOWN annotation coverage, with their samples
and symbols recorded locally; they cannot enter an interval-IoU denominator.
The primary includes every eligible complete valid/recovered wave, including
unmatched/missing predicted slots at IoU0. Report published typed-wave totals,
parsed valid counts, recovered groups and unknown events separately. Do not
require valid counts to equal published totals. Inference, matching tolerances,
primary weights and decision rule are unchanged. All 200 records remain evaluation-only.
Use "physiological wave-boundary localization" for claims, without implying
an anatomical heart region or that the wave is diseased.
