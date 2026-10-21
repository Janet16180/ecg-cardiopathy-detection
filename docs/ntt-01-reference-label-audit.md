# NTT-01: reference-label and benchmark audit

Prospective audit, 18 October 2026. The user authorized planning followed by autonomous investigation of
[No Time to Train](https://arxiv.org/abs/2507.02798). This is a CPU data audit, not a replication of SAM2/DINOv2
inference or a diagnostic localization experiment. Its identifier avoids the existing 041–073 studies.

## Question and decision

Can the public PTB portion of ECG-Reasoning-Benchmark provide independent expert masks for reference-based
ECG localization, or only supplementary rule-derived evidence? Before adopting it, measure its actual
record/patient coverage, diagnostic overlap, grounding schema and answer-position shortcuts.

Promote it only as supplementary silver supervision if source provenance confirms generated labels. Never
promote it to independent expert ground truth on the strength of schema coverage or question accuracy.
An answer-position baseline at least 0.10 above its uniform-position expectation is a shortcut flag,
not evidence about waveform interpretation. Report the flag either way. No search over alternative models.

## Inputs and frozen analysis

- Pin official repository `Jwoo5/ecg-reasoning-benchmark` commit
  `90a53f107023c7c196cbf08de183b871b2a5798c`; download only `data/ptbxl.jsonl` and public documentation.
  Do not download MIMIC annotations or waveforms. Record file SHA-256 and Git blob identity.
- Join source ECG identifiers to local PTB-XL 1.0.3 metadata, using only identifiers, patient identifiers
  and stratification folds for the join. For folds 9/10, count identifiers only: never read their reports,
  diagnosis dictionaries, signals, marks or model outputs. No Challenge, EchoNext or ECGAD holdout access.
- Count QA cases, unique ECGs/patients, repeat cases per ECG, diagnoses, positive/negative cases,
  step/question types, absent/invalid IDs and duplicate case IDs. Require every declared source to be PTB.
- Count lead, wave and measurement grounding questions; verify each answer agrees with its answer indices
  and option strings. Report empty answers, multiple answers, absent grounding and malformed time ranges.
  Parse selected wave ranges in seconds, require 0 <= start < end <= 10, and describe widths and quantization.
  These are selected multiple-choice ranges, not automatically exhaustive abnormality masks.
- For single-answer grounding questions, compute the exact-answer accuracy of always choosing option zero,
  and the mean `1 / number_of_options` expectation. Report lead/wave/measurement separately. No waveform
  predictor is fitted and no clinical accuracy is computed.
- Recompute the prior exploratory folds 1–8 metadata findings using the existing superclass/project-label
  helpers. Preserve their definitions. Use a separate direct implementation as the integrity comparator;
  exact equality is required. Reports from PR61 remain historical evidence: absent original ECGAD marks
  and scoring tables prevent claiming an exact reproduction of its model metrics.

No new independent sample is created by joining a second annotation file onto an existing ECG. Count folds
and repeated patients before proposing a support/calibration/evaluation split. Preserve all existing
quarantines, including ECGAD and the closed non-ECGAD fold-10 records identified in PR61. This audit makes
no allocation of patients to model training.

## Analysis already seen before this freeze

Two agents inspected the paper, official documentation, PR61 protocols/reports and local file availability.
They also computed exploratory folds 1–8 label counts (17,418 ECGs; 7,135 SR-coded ECGs with abnormal diagnostic
codes; 3,779 with multiple abnormal superclasses). These are disclosed descriptive findings, not prospective
hypothesis tests. No released reasoning JSONL, grounding counts or answer-position result was read.
The paper and README describe algorithm/rule-derived reasoning with expert review; the audit will not
pretend that this provenance conclusion was discovered through a blinded experiment.

## Resources, receipts and limits

Use the existing `.venv` through `uv --no-sync`, one process and at most two numerical threads. Download
at most 30 MB; CPU analysis at most five minutes; no GPU, new neural training or dependency changes.
Write source/input/protocol hashes, source commit, package versions, commands and aggregate findings under
`outputs/ntt01_reference_label_audit/`. Keep downloaded annotations and identifiers local. Commit only code,
tests, protocols and aggregate reports. Preserve any failed run and use a new output directory for reruns.

Write the results report from executed outputs, update both queue documents, and rank follow-up candidates.
Keep original-paper image inference, ECG-native masked-reference transfer, clustering and SHAP as distinct
future studies with their own validation gates. Failure to recover old artifacts must be reported; it does
not justify fabricating predecessor reproduction or opening a spent/closed evaluation set.
