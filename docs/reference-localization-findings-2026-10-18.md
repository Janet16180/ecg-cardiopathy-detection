# Reference localization, morphology clusters and explanations: executed findings

18 October 2026. The current evidence supports a narrower, testable thesis objective: identify a named
ECG finding and return the lead, wave region and measurement that support it. None of the executed
studies identifies the underlying cause of heart disease. Binary ECG annotations, record findings,
wave boundaries, sufficient clinical evidence and model explanations are different targets.

All work described here is local. The repository uses the personal identity `Janet16180` /
`clarajanet16180@outlook.com`; no further push or publication is authorized. Two worker agents used
separate worktrees, with the coordinator integrating and checking results. Protocols and implementations
were committed before scores. Sources, weights, patient roles and outputs are recorded in local receipts.

## Reference matching can select a wave without identifying its pathological feature

[NTT-02](ntt-02-native-reference-transfer-results.md) evaluated 1,507 supplied candidate-window questions
from 769 patients and 32 eligible classes. Native morphology matching reached 75.12%, beating duration
matching at 58.67% by 16.45 percentage points, with patient-bootstrap 95% interval [13.31, 19.52].
But a generic QRS-phase comparator reached 87.07%. The advancement gate failed. Success on alternatives
that differ in wave type is insufficient evidence of finding-specific pathological localization.

[NTT-03](ntt-03-visual-reference-pilot-results.md) executed frozen DINOv2-S and SAM2-Tiny components on
physical ECG renderings. This is a compact engineering variant of
[No Time to Train](https://arxiv.org/abs/2507.02798), not its original large/large configuration.
With finding, lead and candidate windows supplied, visual reference matching selected all six nominated
silver answers; wrong-window references selected none in the primary one-second view. Generic phase
already selected all six in a contextual comparison. These six cases demonstrate functioning matching
and sensitivity to the supplied reference region; they do not estimate clinical localization accuracy.

Automatic proposals were a separate bottleneck. On three full-trace examples, SAM offered a proposal
with temporal intersection-over-union at least 0.5 for one nominated interval. The semantically selected
proposal overlapped none of the three nominated intervals. Another recurring beat may still carry the
same finding, so zero overlap with one nominated answer does not establish clinical incorrectness.
The frozen engineering gate failed without subsequent tuning.

The next frozen comparison, [NTT-04](ntt-04-same-wave-finding-discrimination.md), gives every class the
same automatically detected QRS region and canonical V1/V6 view. It compares normal, complete right
bundle-branch block and complete left bundle-branch block source findings with patient-disjoint
reference banks and raw-morphology/scalar controls. This removes the easy wave-type distinction.
It remains record-finding discrimination with a programmed region, not an independently adjudicated
pathology mask. Results will be added only after execution and artifact review.

## Annotation scarcity includes a mismatch in what the labels mean

[NTT-01](ntt-01-reference-label-audit-results.md) audited public PTB reasoning annotations and independently
reproduced existing label counts. In permitted folds, the source provides 2,499 cases from 2,322 ECGs
and 2,174 patients, with 4,854 selected wave answers: 4,813 numeric timestamp ranges and 41 valid symbolic
QRS-index spans. A timestamp-only parser would miss those 41 spans. Selected QA answers provide sparse
silver evidence, not exhaustive independently drawn specialist masks. Unselected windows are unknown.

The source's single-answer lead subset has a narrow answer-content imbalance: option zero is correct
in 164/255 questions, largely because V5 is the answer in 144/168 V5/V6 R-amplitude questions. This is
an answer-option baseline, not a disease-localization score. NTT-02's prospective role gate protected
all new waveform/reference/scoring operations; its predecessor-integrity replay nevertheless reparsed
reserved-role aggregate annotations before gating. Its report discloses that access limitation explicitly.
Later studies avoid that broad replay.

Prior studies through 073 in [PR61](https://github.com/Janet16180/ecg-cardiopathy-detection/pull/61)
already tested delineation, abnormal-beat evidence, native retrieval and replacement faithfulness.
Their ECGAD holdout is spent, and original marks/tables are absent locally. They cannot be treated as a
new untouched evaluation or claimed exactly reproduced here. The separate PR67 evaluation roles also
remain reserved. All new reference and feature-model scoring uses only its training-role patients.

A useful clinical annotation contract should accept multiple equivalent beats, distinguish absent
findings from unmarked evidence, and record the named finding, leads, time or beat coordinates,
measurement and provenance. Independent specialist agreement remains necessary to evaluate clinical
highlighting. Neither clustering nor SHAP creates that ground truth.

## Stable morphology clusters need not define stable disease groups

[CLUSTER-01](cluster-01-physical-morphology-results.md) extracted 53 physical features for 1,999 permitted
patients. It compared k-means, Gaussian mixtures, spectral clustering and HDBSCAN under raw and
nuisance-residualized views, interpreting overlapping diagnoses only after unsupervised selection.
The selected residual spectral solution had silhouette 0.204 and median stability ARI 0.884, but its
worst refit was 0.484, below the frozen 0.50 requirement. HDBSCAN assigned every patient to noise.
The gate failed; the stable secondary mixture was not substituted after observing scores.

[CLUSTER-02](cluster-02-stability-selection-results.md) prospectively prioritized stability and refitted
all preprocessing on fitting patients. Across 36 configurations and 288 subsample refits, selected
raw, residualized and amplitude-normalized mixtures each passed the within-view stability requirement;
consensus also qualified. Yet cross-view ARIs were 0.686, 0.272 and 0.205, failing the required 0.50
agreement. The most stable shape mixture had annotation-combination AMI 0.039 versus quality 0.080 and
beat consistency 0.091. Its stability did not establish a disease taxonomy.

This is evidence of representation-dependent morphology organization in this explored PTB cohort.
Amplitude can carry real physiology, so disagreement after removing amplitude does not prove artifact.
Feature/delineation reliability and transfer to another permitted development source remain open.
No unsupervised partition here establishes etiology or an independent-source clinical result.

## SHAP can be exact while answering different reference questions

[SHAP-01](shap-01-linear-background-sensitivity-results.md) fitted one fixed linear annotation-proxy model
on the physical features, then computed exact interventional Shapley contributions to its logit.
Every feature belonged to one of twelve lead groups or a global duration/rhythm/axis group. The model,
scaler and both backgrounds used only internal training patients; 400 separate internal evaluation
patients supplied the sensitivity endpoint. This is exploratory evaluation within an already studied
cohort, not an independent clinical test.

Changing only the background from all training patients to strict-normal training patients changed the
leading absolute group for 175/400 patients (43.75%) and the leading positive group for 145/400 (36.25%).
At least one group changed sign for 316/400 patients. Predictions were identical. The maximum additive
reconstruction error was 5.33e-15. For this linear game, each group's background shift is constant across
patients and is cancelled by the changed base value.

These observations do not show that SHAP computed an incorrect answer. They show that selecting the
reference population selects the comparison question. A leading lead attribution cannot automatically
be interpreted as the anatomical disease site or specialist evidence. Waveform-level and conditional
attributions, correlated-lead handling, replacement validity and independent expert overlap were not
evaluated in this study.

## Reviewable artifacts and remaining decisions

Every study has a prospective protocol, committed implementation, completed local receipt, tests and a
results report. The NTT-03 integrated suite passed 1,266 tests with six skipped before the later studies
were integrated. Subsequent focused checks passed; final union verification will be recorded after NTT-04.
No original pinned dependency file, frozen historical method, clinical label definition or closed test
was changed. No full GPU training run was started.

The [queue](experiment-queue.md) records execution status; the [ranked backlog](experiment-priorities.md)
records follow-ups without scheduling them. The main next decision is whether same-wave reference
recognition adds useful morphology information beyond physical baselines. A clinical highlighting claim
also requires independent annotation of acceptable finding evidence, rather than treating every
unselected beat as normal.
