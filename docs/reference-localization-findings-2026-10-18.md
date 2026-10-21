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

[NTT-04](ntt-04-same-wave-finding-discrimination-results.md) then gave every class the same automatic
QRS region and canonical V1/V6 view. Sixty target patients, twenty per finding, were separate from
27 reference patients across three fixed support draws. Physical full-beat reference matching reached
**92.78% balanced accuracy**, versus **82.78%** for compact DINO QRS-pooled features. The primary visual
minus strongest-baseline difference was **−10.00 percentage points**, paired whole-patient 95% interval
**[−17.73, −2.34]**. The strongest baseline was reselected in every bootstrap draw. Raw QRS-only matching
reached 88.33%; the combined duration/amplitude/energy comparator reached 72.22%. The visual advancement
gate failed. The run took 55.98 seconds on CPU, with no selected-record replacement or weight training.

This removes the supplied wave-type-choice shortcut and gives a concrete priority: preserve physical
reference matching as the benchmark before adding the visual representation. It does not prove that the
original large/large paper fails on ECGs. DINO tokens remain globally contextual even when pooled inside
QRS; the prospectively added full-beat physical baseline addresses that input-context difference. Both
methods classify source record findings with a programmed region, without independently marked pathology.
The confidence interval conditions on the fixed support banks. Source annotations also co-occurred:
12/20 right-bundle-block targets had MI codes versus 2/20 left-bundle-block targets. The comparison does
not isolate causal BBB morphology. No class-specific fusion or further tuning was performed afterward.

DINO had higher left-bundle-block recall in this small cohort (95.00% versus 88.33% for physical full-beat
matching), while physical matching had higher normal and right-bundle-block recall. The overall result
is not uniform superiority in every class. These descriptive differences are hypotheses for a new
prospective study, not a basis for selecting a combined model on these same targets.

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
highlighting. Neither clustering nor SHAP creates that ground truth. The [draft review contract](ecg-evidence-annotation-guide.md)
and machine-readable schema make these requirements concrete; they contain no clinical labels and still
require semantic adapter checks and cardiologist review.

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

## Native-signal follow-up

After the initial seven studies, the user requested adapting the reference method to signals.
[NTT-05](ntt-05-native-signal-adapter-results.md) implements lead/time token pooling, patient-balanced
finding banks, compatible embedding-space checks, explicit abstention and same-record temporal redundancy
decay. All 26 engineering invariants and 180 saved predecessor choices reproduced. This adds a concrete
image-free adapter, not a new clinical result: native trained checkpoints and prior dense caches are absent
locally. The latest integrated suite passed 1,331 tests with six skipped in 73.10 seconds; Ruff passed.
The original seven-study verification below remains a historical snapshot rather than a rewritten receipt.

## Reviewable artifacts and remaining decisions

Every study has a prospective protocol, committed implementation, completed local receipt, tests and a
results report. The final integrated CPU suite passed **1,304 tests with six skipped in 72.99 seconds**;
repository-wide Ruff passed. Independent artifact reconstruction reproduced NTT-04's 180 patient-draw predictions, all nine
methods and the 2,000-draw primary interval without model inference or waveform reads.
No original pinned dependency file, frozen historical method, clinical label definition or closed test
was changed. No full GPU training run was started. The [verification receipt](../reports/local_reference_research_verification.json)
records 464 matching frozen-source hash comparisons across all seven studies, test results, schema
structural checks and personal-identity scope.

The [queue](experiment-queue.md) records execution status; the [ranked backlog](experiment-priorities.md)
records follow-ups without scheduling them. The next research decision is how physical reference matching
and clinically sufficient evidence transfer to another permitted cohort, with coexisting annotations and
support uncertainty addressed prospectively. A clinical highlighting claim also requires independent
annotation of acceptable finding evidence, rather than treating every unselected beat as normal.
