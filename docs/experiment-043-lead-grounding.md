# Experiment 043: waveform information in finding-conditioned lead grounding

Prospective protocol, 18 October 2026. Freeze before any grounding model fit or performance metric.
The user authorized autonomous multi-agent research without cardiologist access. This study uses
existing public annotations and cached measurements, with no GPU or download of ECG data.

## Endpoint and annotation provenance

Question: do waveform measurements improve the benchmark's finding-specific lead answers beyond
the diagnosis/finding question and offered lead identities? The task is conditional lead grounding,
not disease-presence screening, dense pathology segmentation, anatomical disease localization or
an explanation of the Experiment 041 classifier. Preserve the official task; introduce no clinical
rules, new disease mapping or age analysis.

Use only the local public PTB-XL JSONL at
`data/raw/ecg-reasoning-benchmark/90a53f1/ptbxl.jsonl`, SHA-256
`353ddc413b68c751a0e2c5c44a11a182de4519e278c5ae49fe0bf089c674b39b`.
Its 13,659,066 bytes have Git blob SHA-1 `da62062c76828d53297e76f00ff02153e5713246`, matching
the authored tree `90a53f107023c7c196cbf08de183b871b2a5798c` and corrected release `0.0.2`
(`7c84653c8606557657e76b8357e0958163802f26`). The release update was
`04686428ae78f9733a202ff4c72cc31d0e38af79`. This identity was established through local hashing
and remote tree/history metadata; no annotation payload was downloaded. Bind these identities in
the execution receipt. [Authored tree](https://api.github.com/repos/Jwoo5/ecg-reasoning-benchmark/git/trees/90a53f1?recursive=1),
[corrected release](https://api.github.com/repos/Jwoo5/ecg-reasoning-benchmark/git/trees/0.0.2?recursive=1).

The benchmark derives paths through automated delineation, measurements and diagnostic logic,
with specialist review of 143 representative paths across its two sources and supervised author
review. The local payload here is PTB-XL only. Matching its answers can partly reconstruct that
annotation pipeline; it is not independent clinical confirmation. [Construction paper](https://arxiv.org/html/2603.14326v1).
No MIMIC annotation file, credentialed data, closed Challenge/EchoNext test or waveform is opened.

## Qualification and roles

Match the payload hash and Git blob exactly. Join official PTB patient/fold metadata (SHA-256
`7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b`) by ECG ID. Exclude
fold 10 before processing its grounding steps. Fit folds 1–7, leave fold 8 unused, evaluate fold 9
once. All questions from a patient stay in the same role. Read the unchanged 130-feature raw cache
from Experiment 041, with no selected-feature transformations or label-derived preprocessing.

Take only `question_type == "lead_grounding"`. Context is the exact pair `(target_dx, question)`.
The target diagnosis name is the question's supplied condition; neither its presence label
`dx_label` nor any prior correct answer is a predictor. Preserve exact question strings, including
the separate ST-elevation wordings. Each offered option must be exactly `Lead ` followed by one
canonical lead identity. Parse only this literal prefix; verify answer text against answer indices.
Target is the official correct-answer lead set. Unoffered leads are unannotated and never labeled,
predicted or included in the metrics. Preserve the offered set in canonical lead order, independently
of option positions. Do not merge distinct questions or union their answers.

Case key: `(ecg_id, context, canonical offered-lead set)`. Deduplicate identical copies; conflicting
answers for a key stop qualification. Missing joins, unsupported option/answer semantics, patient
role overlap or malformed targets stop before any fit. Require at least 50 development patients
and 30 development patients with an informative context as defined below. Report all role,
duplication, missingness and answer-cardinality counts, including empty/all-positive queries.
Empty or all-positive answer sets stop this frozen execution because the setup audit found none;
do not silently drop them.

A read-only setup audit, with no models or performance calculations, found fit 1,117 queries /
823 ECGs / 776 patients; unused fold 8 154 / 110 / 99; development 137 / 100 / 97. It found zero
missing joins, duplicate/conflicting keys, invalid literal options or answer-index mismatches.
All 23 training contexts vary in answers. Informative exact context/offered-set groups cover 69
development queries and 56 patients. Reproduce this accounting before fitting.

## Fixed controls, candidate and training

Expand each query into its offered canonical leads. Target is correct-answer membership; sample
weight is `1 / number_of_offered_leads`, giving each query equal total training weight. Apply the
same rows and weights to both fits and lookup counts.

Shared predictors: training-coded exact context, canonical lead identity and 12 binary indicators
of the offered set. Context category codes come from training only; unseen contexts map to missing.
Lead identity codes follow the canonical 12-lead order. No ECG/patient ID, original option position,
`dx_label`, `path_idx`, answer, earlier reasoning answer or demographic becomes a predictor.

1. Primary control `context_hgb`: only the shared context/offer predictors.
2. Candidate `wave_hgb`: the shared predictors, all 130 raw measurements in their frozen order,
   and these nine candidate-lead measurements selected by exact column name:
   `std`, `ptp`, `kurt`, `r_amp`, `s_depth`, `q_depth`, `qrs_area`, `st`, `t_amp`.
   Preserve missing values and replace nonfinite numeric values with missing; no new clinical cutpoints.

Both classifiers are histogram gradient boosting with identical settings:
`max_iter=200`, `learning_rate=0.05`, `max_leaf_nodes=15`, `min_samples_leaf=30`,
`l2_regularization=10`, `early_stopping=False`, `random_state=20261018`, native categorical
context/lead columns and other installed scikit-learn defaults. One fit each, no tuning, extra seed,
validation-based selection or refit. Require every numeric candidate feature to have a finite fitting
observation to avoid the installed estimator's all-missing-column binning failure.

3. Secondary control `context_lookup`: training-only hierarchical per-lead prevalence. First use
   global lead weighted positives/total with Beta(1,1). Shrink each context/lead probability toward
   that global probability with strength 10; shrink each context/offered-set/lead probability toward
   its context/lead probability with strength 10. Use the same query row weights. Unseen exact sets
   fall back to context/lead, unseen contexts to global lead, unseen leads to 0.5. Strength 10 means
   ten units of weighted evidence. Bind counts and parameters. Do not choose among controls after scoring.

## Primary comparison and decision

Compute average precision within each query over offered leads, using scikit-learn's tie-aware AP.
Average queries within patient, then patients equally. Primary is patient-macro AP of `wave_hgb`
minus `context_hgb`. Use shared `patient_groups` / `patient_resample` with 2,000 paired whole-patient
draws, seed 20261018, percentile 2.5/97.5 interval. Resample patient-level query means, preserving all
queries of each drawn patient; no single-class draw can arise after valid per-query AP calculation.

The informative subset is determined solely from training: exact `(context, offered-set)` groups
with at least two different correct-answer lead sets. Calculate the same patient-macro difference
on the development patients' informative queries only. It is a prespecified guard, not a selected
subgroup or separate clinical target.

Return `advance_benchmark_grounding` only if primary gain is at least 0.05, its paired 95% lower
bound is strictly positive, informative-subset primary gain is strictly positive, and overall
`wave_hgb` minus `context_lookup` patient-macro AP is nonnegative. Otherwise return
`do_not_advance_grounding`. No model or target change follows a failed gate.

Secondary: each model's patient-macro AP and paired contrasts to lookup; query/patient counts;
informative-subset AP; Brier score averaged within query then patient; exact offered-lead-set accuracy
using a fixed strict `score > 0.5`; and descriptive context counts/AP (no context-specific intervals
or decision claims). Report all answer cardinalities and constant-target contexts. Fold 8 stays unused.

## Resources, artifacts and independent checks

CPU only, four numerical threads at most and eight across concurrent work. Profile the actual two
fixed fit paths and inference on the fitting role with artificial data tests before the real run;
no GPU is used. Execution has a 3,600-second wall timeout and 256 MiB output cap. Failures remain
in exclusive directories. Only a new unpinned module, thin runner and scientific tests may be added.

Write `outputs/experiment043_lead_grounding_v1/` exclusively: qualification counts, fitted models,
context/lookup state, aligned per-query offered targets/predictions and patient aggregates, paired
intervals/gates, source/input/protocol SHA-256 bindings, versions, Git revision and timing. Use shared
path and interval helpers; raw identifiers and predictions remain local. No waveform reads or model
fits outside the two frozen arms. Existing 041/042 artifacts and verdicts remain unchanged.

Before launch, require independent source review and artificial tests for answer parsing, option-order
invariance, unavailable-lead exclusion, patient roles, context coding/fallback, query weights,
informative-group training-only definition, AP ties/patient averaging and the exact decision. After
execution, independently recompute the saved query/patient statistics and decision without new
inference or fits. The coordinator writes the output-derived results report, updates both queues
and the ranked backlog, and reports the task's annotation and clinical limits even if it passes.
