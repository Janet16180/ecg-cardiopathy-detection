# Corrected local ECG research app

This app adapts PR 74's sidebar, ECG mark and visual palette around new reproducible development results.
It displays the primary strict-age/matched-control and secondary expanded-age cohort summaries,
exact confusion matrices across saved score cutoffs, and a quality-gated analysis of a newly provided
canonical ECG. The threshold view also compares the existing age-preferred and no-age-preference
25,000-record cohorts side by side for the selected evaluation site. Each fixed 50:50 blend uses its own
calibration-only 5% normal-budget cutoff; the exploratory slider never changes those comparison rows.
Both cohorts use the same quality and label-admission rules. English and Spanish are available.

The old PR's serialized models, alternative diagnosis target, validation prediction CSVs and diagnosis
hit-rate claims are not imported. No missing notebooks or historical training steps are reconstructed by
assertion. Unsupported multilabel diagnoses are unavailable. Localization is visibly work in progress;
there is no invented interval plot or claim of improved classification.

## Run locally

Use the repository's Python 3.11 environment and existing lock. FastAPI, Uvicorn and HTTPX are already in
`uv.lock` through the tracking environment; the frozen root dependency files do not need changes.

```bash
uv sync --locked --inexact --group tracking
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-sync --group tracking python -m uvicorn ui.app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. The default artifact root is the current code checkout. In an isolated worktree,
use the already provisioned main environment, set `PYTHONPATH` to the worktree, and configure the trusted
local data checkout at startup:

```bash
ECG_APP_ARTIFACT_ROOT=/home/janetrivera/ecg-cardiopathy-detection \
PYTHONPATH=/tmp/ecg-pr74-ui OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /home/janetrivera/ecg-cardiopathy-detection/.venv/bin/python -m uvicorn \
  ui.app:app --host 127.0.0.1 --port 8000
```

Choose one fixed artifact mode at startup: `ECG_APP_ARTIFACT_MODE=cached` (the default) or
`ECG_APP_ARTIFACT_MODE=from_raw`. The mode is shown in the interface and binds its evaluation,
cohort summaries, model bundle and verified loader together. It cannot be changed through an API request.
There is no fallback to another mode when an artifact is missing or unfinished.

For a completed from-raw reproduction:

```bash
ECG_APP_ARTIFACT_MODE=from_raw OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-sync --group tracking python -m uvicorn ui.app:app --host 127.0.0.1 --port 8000
```

The from-raw mode uses only these additional fixed namespaces:

| Artifact | Fixed from-raw location |
| --- | --- |
| Primary cohort manifests and summary | `outputs/cohort25_recovery_20261028/cold_start/strict_matched/` |
| Secondary expanded-age cohorts | `outputs/cohort25_recovery_20261028/cold_start/cohorts_25k/` |
| Evaluation, completion receipt and model bundle | `outputs/pr74_from_raw_validation_20261028/evaluate/` |
| Verified bundle loader | `ecg_experiment.pr74_raw_inference.load_bundle` |

Follow the [complete reproduction commands](../docs/pr74-from-raw-reproduction-2026-10-28.md),
including the batch-shape correction and explicit baseline selection, and the registered
[raw study protocol](../docs/pr74-raw-study-protocol-2026-10-28.md) and
[fresh baseline protocol](../docs/pr74-fresh-baseline-protocol-2026-10-28.md). The from-raw pipeline must
finish before its results or models are available. Run its
receipt-producing data reconstruction and `scripts.experiments.run_pr74_raw_study` stages; do not copy or
rename cached completion receipts into those locations. A clean checkout without those completed
artifacts will correctly show unavailable results.

Only the operator configures that root. No API accepts a filesystem path. No raw dataset directory is
mounted, no reserved record is sampled, and no recording history is saved. Missing or inconsistent
artifacts produce an unavailable state rather than archived or fabricated metrics.

## Artifact contract and reproduction

Complete the new data reconstruction, frozen-split exclusion/reference recovery, and label-evidence
assembly before corrective model training. Follow `docs/cohort-age-assembly-protocol-2026-10-28.md` and
the corrective evaluation protocol/runner committed with the model pipeline. The primary comparison follows `docs/cohort-strict-young-matched-protocol-2026-10-28.md`: strict
ages 18–25 versus a control matched on source, label-evidence rank and target. The completed primary
cohorts contain 770 hospital ECGs each, not an Italian student sample. Expanded-age selections with a
25,000 ceiling remain secondary. Selection seed is 42 and eligibility never weakens to fill a ceiling. Training and evaluation
seeds and checkpoint hashes belong to that fixed pre-score model protocol and its executed receipts.

The default cached mode reads only these fixed local paths:

| Artifact | Gate |
| --- | --- |
| `outputs/cohort_strict_young_matched_20261028/{receipt.json,summary.json}` | Complete primary strict/matched scope and matching fixed manifest/summary hashes |
| `outputs/cohort_age_label_priority_20261028/{receipt.json,summary.json}` | Complete secondary expanded-age scope and matching fixed manifest/summary hashes |
| `outputs/pr74_corrective_evaluation_20261028/evaluate/{completion.json,screening_evaluation.json}` | Completed evaluate stage, matching aggregate hash and development-only schema |
| `outputs/pr74_corrective_evaluation_20261028/evaluate/model_bundle.joblib` | Only the evaluator's fixed locally generated bundle, verified by `ecg_experiment.pr74_screening_inference.load_bundle` before deserialization |

The inference loader and `ecg_experiment.cuda_runtime20261028` must be present from the corrective
evaluation implementation. The app automatically validates and configures the installed CUDA runtime
headers before importing the frozen xLSTM path, including when inference runs on CPU. It respects an
explicit `CUDA_HOME` setting; no manual hidden variable, download, installation or GPU acquisition is
required in the provisioned pinned environment. Aggregate viewing and quality-only checks do not load
xECG.

The app never loads
PR 74's old joblibs, accepts an uploaded pickle, or imports its legacy inference modules. Bundle absence
is a supported state. A bundle file's presence alone is not reported as successful inference validation.

The aggregate JSON schema is version 1, `complete=true`, `scope=reused_development`,
`score_interpretation=uncalibrated`. Each model has an ID, label, cohort and PTB/SPH evaluation pools.
Each pool includes the fixed threshold, calibration-normal counts, `metrics` and a `sweep` of ascending
finite cutoffs with exact TP/TN/FP/FN and rates. Undefined zero-denominator rates are JSON `null`.
The app checks matrix totals, rate arithmetic and population consistency before displaying them. It
projects allowed aggregate fields and does not expose patient prediction arrays or raw receipt content.

## Threshold interpretation

The fixed operating point uses a 5% normal-referral budget fitted on calibration normals. The comparator
is strictly **score > threshold**; ties are not flagged. The evaluation slider visits saved cutoffs and
uses their exact saved counts, without interpolation. The saved grid must include the exact frozen cutoff; the reset returns to that calibrated
threshold. The separate frozen-point matrix retains its actual fixed-threshold counts.

Every matrix shows positive and negative reference denominators, sensitivity, specificity, precision,
referral workload and sample prevalence. Offline evaluation retains all admitted reference ECGs, including difficult signals. These matrices
describe classifier evaluation; they do not validate the combined quality-abstention and referral workflow.
The slider is descriptive development exploration. It does not
change the frozen model or select a deployment threshold. Scores are uncalibrated model scores, not disease
probabilities. Below threshold means not flagged at that operating point; it does not establish no risk.

PTB-XL and SPH are reused development data. These displays do not demonstrate Italian-student
performance or an untouched final test. Zero observed exclusion-reference overlap covers recovered PTB
folds 8–10, original Challenge reserved groups, and existing SPH qualification references. EchoNext test
remains closed. Official xECG pretraining includes CODE, INCART, Chapman and Ningbo; source overlap and
actual checkpoint provenance must be disclosed by the execution receipt.

## New recording contract

Choose a plain CSV with this exact header followed by 5000 rows of twelve samples:

```text
I,II,III,aVR,aVL,aVF,V1,V2,V3,V4,V5,V6
```

Samples must already be physical mV at 500 Hz, making exactly ten seconds. The app does not infer units,
resample, crop longer records, reorder undocumented leads, parse WFDB archives or deserialize NumPy/pickle
uploads. Blank/NaN samples remain missing and cause quality abstention; they are never replaced by zero.
Requests are limited to 4 MiB. The frozen `ecg_experiment.ecg_quality.assess` policy runs before model
loading, feature imputation or inference. Exclusion reasons cause abstention without a score or referral
result; the UI suggests repeat acquisition or clinical review, and never treats abstention as negative.
Review flags are retained separately. The verified runtime can also abstain if extraction fails.

## Checks

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-sync --group tracking python -m pytest tests/test_corrected_ui.py -q
uv run --no-sync --group tracking ruff check ui tests/test_corrected_ui.py
node --test tests/test_ui_matrix.mjs
node --check ui/static/app.js
```

Tests use synthetic signals and aggregate fixtures, not protected recordings. Some restricted tool
sandboxes block asyncio inter-thread wakeup sockets; run the ASGI route tests in the normal local
environment when that restriction applies. The actual host tests must still pass; do not skip them. They cover missing/changed
artifacts, no raw dataset endpoints, invalid-signal abstention before model loading, missing-model behavior,
strict threshold ties, zero-denominator matrices, exact saved-cutoff behavior and rejected path/pickle
fields. These software checks do not establish clinical validity.

For an optional real-browser check, start the app and a local Firefox geckodriver, then run:

```bash
geckodriver --host 127.0.0.1 --port 4444
python -m scripts.validation.check_corrected_ui_browser \
  --app-url http://127.0.0.1:8000 --driver-url http://127.0.0.1:4444 \
  --output-dir docs/figures/pr74-corrective
```

Use `--firefox-binary` when the browser needs an explicit executable. This check uses only aggregate
API data: it checks the fixed default, exact slider endpoints, the failed primary-comparison conclusion,
the secondary table's exact model/site counts and cutoffs, slider independence, actual cohort counts,
mobile layout, Spanish switching and status-only localization. It saves no raw ECG
or individual prediction. The screenshot files are concrete review artifacts for this corrective PR.
