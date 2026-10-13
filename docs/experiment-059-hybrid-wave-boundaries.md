# Experiment 059: hybrid wave boundaries on independent QTDB excerpts

Freeze before any Experiment 059 score. Experiment 055 improved QRS and T
overlap but worsened P overlap. Its 200 LUDB records are now development data.
This successor retains the frozen fixed P window and tests unchanged 055
QRS/T morphology on a separate expert-delineated public source. The classifier
and abnormality score remain frozen. This tests physiological wave-boundary
localization, not whether a particular wave is pathological.

## Data and source independence

Use [QTDB 1.0.0](https://physionet.org/content/qtdb/1.0.0/): 105 two-channel,
15-minute records at 250 Hz, with manually selected beats annotated in `.q1c`
(expert 1's audited second pass). Do not use automatic `.pu*` annotations.
Download raw `.dat`, `.hea`, `.q1c`, RECORDS and SHA256SUMS locally; verify
every selected file against the official checksums. No credentialed or closed
project test is accessed. Metadata diagnosis and demographic labels are unused.

The [source table](https://physionet.org/physiobank/database/qtdb/doc/node4.html)
identifies 33 `sele*` European ST-T excerpts; exclude all 33 from the primary,
because Experiment 057 already accessed their source. The other 72 records
are the evaluation-only primary. All 105 may be described secondarily, clearly
separating EDB-derived records. No annotation-guided fitting, parameter tuning,
subject selection or waveform inspection precedes evaluation. Source-family and
original-record IDs are audited. QTDB does not establish universal subject IDs
across sources, so bootstrap units are original records, not claimed independent
patients. Report this limitation and descriptive source-family results.

Experts [viewed both channels simultaneously](https://physionet.org/physiobank/database/qtdb/doc/node5.html)
and supplied one time annotation; `.q1c` channel 0 is not verified lead-specific
truth. Freeze joint predicted support as earliest onset/latest offset among
valid predictions from the two channels. No best-channel oracle is allowed.
The sudden-death source has estimated, uncalibrated gains; relative morphology
is used without clinical absolute-amplitude claims. Unannotated beats are UNKNOWN,
not absent waves or negative samples.

## Frozen inference and integrity

Both arms use waveform-only `fragment_localization.r_peaks` at native 250 Hz.
Fixed baseline is 042 P[-250,-60], QRS[-60,+80], T[+200,+450] ms from R.
Hybrid P is exactly that fixed P window; QRS and T use frozen Experiment 055
`morphology_boundaries.adaptive_boundaries`, unchanged physical time constants
and missing-wave rejection. Collapse two-channel candidate intervals into the
joint union described above. Both channels missing gives a missing interval.
The fixed intervals are already identical across channels.

Before any 059 candidate metric, verify 055 pinned source/input hashes and
immutable output hashes, reconstruct its record summaries from saved per-wave
measurements, and reproduce its aggregate baseline/candidate IoU and paired
bootstrap interval exactly. Explicitly exclude its mutable run.log from output
hash verification because its completion message was appended after hashing;
do not modify the predecessor. Record live 056/057 completion identities for
context, without claiming their clinical endpoints are this experiment's comparator.

## Annotation parsing and endpoints

Parse expert onset `(`, typed peak and offset `)` groups, preserving samples.
Recognize p, t, u and all WFDB QRS beat symbols, with boundary type `.num`
0=P, 1=QRS, 2=T, 3=U where available. A complete group has exactly one onset,
one typed peak and one offset, onset<offset, with the peak inside the span or
within 10 ms of an endpoint. Events cannot be reused. A typed peak followed
by its compatible offset but lacking onset is valid for offset measurement
only; never invent its onset or score it as complete interval truth. Audit
malformed, partial, recovered, and unknown events. U is counted but not scored.
If a format audit exposes ambiguity before scores, freeze its conservative
resolution in a prospective amendment; after scoring no recipe may change.

Every complete QRS interval within [0.5,899.5] seconds is primary-eligible.
Use score-independent one-to-one raw-R/QRS-peak matching, minimum absolute
distance and maximum 150 ms, as in 055. P associates one-to-one to the following
truth QRS within 450 ms; T to the preceding within 650 ms. Only evaluation uses
truth anchors. Every eligible truth contributes: unmatched R, unassigned wave
or missing prediction yields IoU 0 and boundary error 500 ms. No conditioning
on successfully detected waves. IoU uses continuous sample spans. For complete
truth average both onset/offset absolute errors or within-30-ms indicators;
T offset-only truth contributes its known offset to the T guardrail.

Primary is paired record-macro joint QRS IoU gain, hybrid minus fixed, across
the 72 non-EDB records (records lacking eligible QRS score 0 and are counted).
A reviewable improvement requires ALL: point IoU gain >=0.10, two-sided 95%
bootstrap lower bound >0, point gain in QRS boundaries within 30 ms >=0.10,
and T-offset MAE deterioration upper 95% bootstrap bound <=10 ms. The T guard
includes every annotated T offset in the interior, complete or onset-missing,
including unmatched/missing predictions with the 500 ms penalty. Records with
no known T offset are counted and excluded from this distinct guard population.
Hybrid/fixed P predictions must be exactly equal. Use 2,000 whole-original-record
paired draws, seed 59059, shared interval helpers. No secondary outcome can
replace a failed gate.

Secondary: per-wave complete IoU, onset/offset MAE, missing predictions, matched
anchor coverage, unmatched truth, annotation unknown counts, duration distributions,
all-105 and EDB-only endpoints, and source-family descriptive outcomes. Predictions
outside annotated sparse slots are counted as unknown, not false positives.
Synthetic compact-wave stress checks may exercise width/rate/noise/polarity
without fitting parameters; their results are development diagnostics and cannot
establish real or clinical improvement. A shared synthetic generator may be used
if available, otherwise deterministic analytic test signals suffice for invariants.

## Execution and artifacts

CPU only, two numerical threads maximum, no GPU. Public download <=100 MB,
audit/scoring <=30 minutes excluding network, local outputs <=500 MB. Commit
protocol then source before execution. Refuse any existing output directory;
preserve failure receipts and create a new successor identity for fixes.
Store source/protocol/data hashes, complete format/source audits, record measurements,
bootstrap populations, all gates, and receipt. Fixed representative local figures
are sel100, sel232, sel16265, sel30 and sele0106 (EDB diagnostic only), without
selection on outcomes. Raw signals and per-record outputs remain ignored/local.
Write aggregate results from executed outputs and send follow-up evidence to the
parent-maintained backlog. Do not retune on any of the 105 QTDB records after scoring.

## Pre-score native-format clarification

The whole-source header audit found 53 records with 225000 samples, 29 with
224999, and 23 with 224993, all two channels at 250 Hz. Preserve these native
lengths (899.972–900 seconds), without padding, exclusion or resampling. The
frozen interior eligibility limit of 899.5 seconds is valid in every record.
Annotation-only audit found 8336 complete intervals, 2130 onset-missing T and
714 onset-missing U intervals, with zero malformed events or displaced peaks.
All boundary class numbers were compatible with the typed peaks. No truth
onset is inferred. Matching, inference, denominators and gates are unchanged.
