# NTT-02: ECG-native masked-reference candidate selection

Prospective protocol, 18 October 2026. This bounded CPU study tests two-stage masked-reference matching:
pool physical ECG features inside selected support windows, average independent support patients into
criterion-specific prototypes, then rank already supplied target windows by cosine similarity. It does
not execute SAM2/DINOv2, propose windows, diagnose records or measure free localization. Released answers
are generated sparse silver choices; other intervals have unknown clinical status.

## Cohort and integrity gate

First reproduce every aggregate reasoning and local-label value in the committed NTT-01 receipt exactly,
using its existing quarantined audit. This replay is an integrity operation, not new allocation/scoring.
Only source identifiers and patient/fold metadata may be examined before eligibility. New annotation and
signal analysis uses folds 1–8, excludes every patient with any record in folds 9/10, and uses only PR67's
train role: SHA256(`ssl_synthetic_partition_v1:20261018:` + canonical integer patient ID), integer digest
modulo 10,000 less than 8,000. Validation and engineering-test patients are reserved. No Challenge,
EchoNext, MIMIC, ECGAD holdout, new download, GPU or neural training. Preserve source/input byte hashes.

Include positive diagnosis cases, finding-identification answer exactly Yes, a named criterion, and
single-answer wave questions with at least two distinct valid absolute-time options within ten seconds.
Preserve diagnosis, criterion, question, finding polarity, selected interval, lead answers when present,
source IDs, patient and units. Beat-ordinal ranges are counted and excluded rather than silently coerced.
Use criterion text plus exact wave-question text as the reference class; report every group and exclusion.
Keep at most three questions per patient/class, sorted by case and step. A group is eligible only with
at least 12 distinct patients and at least three support patients outside each of three deterministic
inner patient folds (SHA256(`ntt02_inner:` + patient ID) modulo 3). No score-dependent class selection.

## Fixed method and controls

Use native 100 Hz WFDB physical signals in mV, preserving all 12 leads in source order. A lead answer in
the same reasoning step does not prove an exact wave/lead link; store it but use all leads for every
method and claim time-window candidate selection only. Subtract each lead's full-record median. Inside
each mask linearly resample physical amplitudes to 32 bins per lead, flatten and normalize to unit L2.
Average instance vectors within each reference patient, then average patients and normalize again.
Rank candidate vectors by cosine with this class prototype. Ties choose the earliest stored option.
No labels or unselected target windows enter fitting.

For each inner fold select fixed independent support patients by SHA256(seed, class, patient), budgets
1 and 3, nested, seeds 20261018, 20261019, 20261020. All eligible out-of-fold patients are targets. All
supports/targets lie inside PR67 train. Target patients are disjoint from their reference bank.
Compare on identical target questions with option-zero; uniform-choice expectation; duration matching
(negative absolute log ratio to median support duration); absolute-center matching; QRS-phase matching
(negative distance to support median center-to-nearest-QRS distance, QRS proposals from all-lead derivative
energy peaks); physical RMS-amplitude matching; derivative-energy matching; wrong-window reference
(first unselected support option); and shuffled-class reference (next eligible class in lexical order,
independently drawn out of fold). Controls use the same reference budget and target options.

## Endpoint and decision

Primary endpoint: patient-macro exact selected-option agreement, averaged across three support draws,
for budget-three morphology minus duration-only matching. Form 95% paired patient bootstrap intervals
with 2,000 draws using shared patient grouping/resampling helpers. All questions of each patient travel
together and patients receive equal weight. Secondary endpoints: chosen-versus-selected interval IoU,
all baseline/control accuracies, per-class results, budget/draw sensitivity and candidate width/position
distributions. Options explicitly contain the selected answer: the proposal oracle is one by construction,
not model coverage. No negative-finding endpoint or abstention is inferred.

Advance only to a new independent reference-quality study if the primary interval lower bound exceeds
zero AND morphology exceeds every simple baseline and both reference controls in point estimate.
Otherwise record an inconclusive/negative result; do not tune features, supports, class rules or thresholds
after scores. A pass still establishes candidate-choice agreement with generated answers only.
The first eligible class/patient/question must be illustrated with support and target physical traces,
seconds/mV and exact option coordinates, including failure if that preselected example fails.

## Execution and artifacts

Commit protocol before scores and implementation before the scored run. Use the existing venv, at most
two numerical threads, one CPU process, no new dependencies and an initial scoring budget of ten minutes.
Preserve failed outputs; reruns need fresh output directories and disclosed reasons. Write local
per-question predictions, support manifests, waveform hashes, aggregate JSON, receipt and an SVG/PNG
example under `outputs/ntt02_native_reference_transfer/`. Commit aggregate results and code only;
source identifiers/waveforms stay local. Root owns queue/backlog updates. No push/PR publication.
