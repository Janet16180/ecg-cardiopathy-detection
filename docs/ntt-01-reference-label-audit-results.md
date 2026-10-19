# NTT-01 results: references need finding-level meaning, not just diagnosis labels

18 October 2026. The [protocol](ntt-01-reference-label-audit.md) was committed as `288dad1`, followed by a
[quarantine amendment](ntt-01-amendment-safe-grounding.md) and reviewed implementation `9089031`, before
the audit ran. The executed audit took 1.522 seconds on CPU with two numerical threads. The complete
aggregate receipt is [reports/ntt01_reference_label_audit.json](../reports/ntt01_reference_label_audit.json);
the original result and status remain under `outputs/ntt01_reference_label_audit/`.

The useful finding is a specific label opportunity with a specific limit: a public PTB reasoning release
provides thousands of selected evidence intervals and lead answers, but they are generated, sparse and
task-conditioned. It can help construct references; it is not a new independent test of specialist-level
abnormality localization. The original SAM2/DINOv2 method was not executed, and no new diagnostic or
waveform-localization accuracy is reported.

## What was executed

Downloaded only the public `data/ptbxl.jsonl` at upstream commit
`90a53f107023c7c196cbf08de183b871b2a5798c`, Git blob
`da62062c76828d53297e76f00ff02153e5713246`, SHA-256
`353ddc413b68c751a0e2c5c44a11a182de4519e278c5ae49fe0bf089c674b39b`.
Joined source identifiers to local PTB-XL identifiers/patient/fold columns before accessing reasoning.
Folds 9/10 were counted by identity and quarantined; their diagnoses, reasoning and signals were not
interpreted. No MIMIC annotation, credentialed signal, Challenge test, EchoNext test or ECGAD holdout was read.

| Coverage | Executed count |
| --- | ---: |
| All released PTB QA cases, identifiers only | 3,084 |
| Unique ECGs / patients, identifiers only | 2,871 / 2,698 |
| Quarantined fold-9/10 cases | 585 |
| Permitted fold-1–8 cases | 2,499 |
| Permitted unique ECGs / patients | 2,322 / 2,174 |
| Extra cases beyond unique permitted ECGs | 177 |
| Maximum cases for one permitted ECG | 4 |
| Diagnoses represented in permitted cases | 17 |
| Positive / negative permitted diagnosis cases | 1,129 / 1,370 |
| Permitted cases without any grounding question | 172 |

There were zero unknown ECG IDs, invalid identifiers or duplicate case IDs. A QA case is not an independent
ECG or patient. The count of 3,084 is the pinned file's actual PTB case count, resolving which count to use
for this revision rather than relying on inconsistent prose totals in the initial paper.

## What the grounding actually contains

| Type, permitted cases only | Questions | Single-answer | Multiple-answer |
| --- | ---: | ---: | ---: |
| Lead grounding | 1,271 | 255 | 1,016 |
| Wave grounding | 4,854 | 4,854 | 0 |
| Measurement grounding | 2,202 | 2,202 | 0 |

Every answer agreed structurally with its option indices. Of the wave answers, 4,813 are explicit time
intervals within ten seconds; their median width is 120 ms, with 70–420 ms range. All 9,626 timestamp
endpoints lie on a 10 ms grid; this describes formatting, not measured boundary accuracy. The other 41 are
symbolic intervals between numbered QRS waves, not malformed annotations. The frozen receipt's
`invalid_selected_time_ranges=41` means **not parsed by the timestamp-only
parser**. Post-hoc inspection found no invalid numeric interval: all 41 need a beat-index coordinate adapter.

These are selected answers to particular questions. They do not enumerate all acceptable evidence or label
every unselected sample normal. Their numerical validity and agreement with option indices do not establish
clinical correctness. Preserve timestamp and beat-relative coordinates separately when building references.

## A narrow signal-free shortcut

The prespecified baseline always selected option zero, on single-answer grounding questions only. It read
no ECG, diagnosis or question meaning. Uniform-choice expectation is the mean of `1 / option_count` over
the same questions; no question-independent confidence intervals are claimed.

| Grounding type | Correct / eligible | Stored-order accuracy | Uniform-choice expectation | Frozen flag |
| --- | ---: | ---: | ---: | --- |
| Lead | 164 / 255 | 64.31% | 39.76% | Yes |
| Wave | 1,138 / 4,854 | 23.44% | 24.58% | No |
| Measurement | 395 / 2,202 | 17.94% | 23.78% | No |

The lead result excludes 1,016 multiple-answer questions. Post-hoc decomposition explains most of the flag:
168 eligible questions offer only V5/V6, and V5, option zero, is correct in 144 (85.71%). These ask which
lead has the larger R amplitude for an LVH criterion, not which lead is diseased. The remaining
five-option questions contribute 20 correct option-zero choices out of 87. This is a restricted answer
distribution shortcut, not evidence that the entire reasoning benchmark is solvable without ECGs or that
published models exploited it. The wave and measurement position baselines did not trigger the frozen flag.
New evaluations should report simple signal-free baselines and preserve task-specific denominators.

## The existing local labels also explain part of the difficulty

On 17,418 PTB ECGs from 15,023 patients in folds 1–8, the existing label helpers and an independent direct
implementation agreed exactly:

| Rule | Positive | Negative | Undefined |
| --- | ---: | ---: | ---: |
| Diagnostic superclass binary | 9,841 | 7,243 | 334 |
| Existing strict project rule | 9,488 | 5,872 | 2,058 |

Of 13,404 SR-coded ECGs, 7,135 also carry an abnormal diagnostic superclass: sinus rhythm cannot define
a normal reference bank. There are 353 NORM-coded records with an abnormal superclass, 1,724 NORM-coded
records with additional non-SR codes, and 3,779 ECGs with at least two abnormal superclasses. These are
coexisting source annotations, not proof of labeling errors. A single binary target removes finding-level
information, and mutually exclusive disease clusters would misrepresent many records. No labels were changed.

These counts reproduce the independent agent's descriptive audit and the existing helper definitions. They
do not reproduce the historical experiment cohort: its quality exclusions and original manifests are absent.

## Why this matters for No Time to Train

The [paper](https://arxiv.org/html/2507.02798v4) matches masked reference objects to candidate image masks;
the [official custom-data instructions](https://github.com/miquel-espinosa/no-time-to-train) require annotated
reference examples. Training-free target inference does not remove the need to define the reference object.
For this project that object must encode a named finding and its supporting lead/time relationship, rather
than simply an unusual shape or a disease name.

The new [reasoning paper](https://arxiv.org/html/2603.14326v1) uses delineation, measurements and diagnostic
rules with expert review. Treat these as silver references. Its [release notes](https://github.com/Jwoo5/ecg-reasoning-benchmark/releases)
also document corrected grounding samples and an earlier inference-history bug; pin both annotations and
inference code. The repository's MIT software license does not by itself establish separate waveform or
annotation redistribution rights. This study distributes aggregate counts only.

PR61 already supplies report-level evidence that detection and plausible maps can coexist with failed
localization comparisons. Its original ECGAD marks and score tables are unavailable here; no historical
localization metric was reconstructed. The spent holdout was not reused. Some maps/checkpoints survive in
temporary caches, which does not replace missing truth or prove their provenance.

Decision: retain the public PTB annotations as a candidate supplementary reference source, with a typed
coordinate adapter and split reconciliation before use. Do not call them a new expert test set. The next
comparison should test whether abnormal masked supports add patient-specific evidence beyond diagnosis/QRS
priors. Detailed stages and later clustering/SHAP work are in the [research plan](ecg-reference-localization-plan.md).

## Checks and deviations

The focused quarantine, schema, identifier and failure-preservation suite passed 24 tests; the full CPU
suite passed 1,232 tests with six skipped in 114.95 seconds. Repository-wide Ruff passed. An independent
scientific/code review checked the denominator and quarantine rules; a separate standard-library count
reconstruction passed all ten expected-count checks ([review](../reports/ntt01_independent_review.json)). Sources
matched their committed bytes before execution, and public annotation bytes matched the frozen upstream blob.

The timestamp-format and V5/V6 decompositions are explicitly post hoc and choose no model. A preliminary
LUDB archive download timed out before this audit; it supplied no evaluated input. LUDB was then excluded
from new work after discovering the completed predecessor study. The prospective audit itself completed
without a scoring failure. No GPU or original-paper model was run.

Reproduce into a fresh output directory:

```bash
UV_CACHE_DIR=/tmp/ecg-ntt01-uv OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
MPLCONFIGDIR=/tmp/ecg-ntt01-mpl uv run --no-sync python -m scripts.reports.audit_reference_labels \
  --output outputs/ntt01_reference_label_audit_replay
```

The pinned public JSONL must already be present at the receipt's repository-relative path; it stays local.
