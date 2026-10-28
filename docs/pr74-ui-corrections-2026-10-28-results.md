# Corrected local app validation

28 October 2026. The new `ui/` source adapts PR 74's visual palette and sidebar around the corrected
receipt-bound pipeline. It imports none of PR 74's old serialized models, prediction CSVs, alternate target
or unrestricted dataset viewer. The local interface has English and Spanish views.

## Executed checks

- All 27 application tests passed on the normal host environment. They cover missing and modified
  artifacts, no raw dataset endpoints, quality abstention before model loading, missing-model behavior,
  strict threshold ties, null rates, primary cohort contract checks, exact calibrated-cutoff availability,
  automatic runtime-header setup before lazy model loading, streamed request-size rejection, and
  fixed cached/from-raw namespace pairing without fallback.
  The frozen quality helper emits its existing spectral warnings for the synthetic all-flat input;
  that input still abstains and produces no score.
- All four Node tests passed: exact saved-cutoff extremes/ties, empty and single-class denominators,
  explicit canonical CSV parsing without missing-value zero imputation, and fixed age-preference
  comparison mappings by model identity and evaluation site. JavaScript syntax and
  targeted Ruff checks passed, including the UI and browser-check source.
- The real aggregate API validated the completed 15-model, two-site evaluation and its fixed
  `ptb_baseline_blend` default. The primary comparison's failed development-advantage decision is visible;
  the app does not select a new default from the displayed metric rankings.
- A headless Firefox check against the running localhost app passed exact all-flagged/no-flagged matrix
  endpoints, the frozen default, primary conclusion, both actual 770-record cohort cards, mobile layout,
  Spanish switching and status-only localization. There was no horizontal mobile overflow.
- The secondary 25,000-record comparison table passed real browser checks against the aggregate API
  for both PTB and SPH: correct `younger_blend`/`all_age_blend` rows, each model's own calibrated cutoff,
  and exact TP/FN/FP/TN counts. Moving the exploratory slider did not change this fixed comparison.
  Both cohort labels rendered in Spanish; the wide table scrolls inside its mobile container.
- Real API inference passed on one explicitly admitted PTB training ECG after checking its canonical
  waveform hash. The verified default blend returned a finite score and a consistent strict comparison.
  The verified runtime also loaded successfully with `CUDA_HOME` initially unset, using only already
  installed runtime headers. No individual score or record identity was exported. No protected recording was read by this check.

Restricted tool sandboxes stalled even minimal synchronous FastAPI routes because their inter-thread
wakeup sockets were blocked. The actual host tests passed; no test was silently skipped or dependency
changed to work around that restriction. The request guard uses a bounded ASGI stream and rejects an
oversize chunk before extending its buffer or invoking validation/inference.

The fixed `cached` and `from_raw` startup modes were exercised with synthetic completed receipts,
including rejection of cached fallback when fresh artifacts are absent. Each mode pairs its own cohort
summaries, evaluation and verified bundle loader. Real cohort metadata passed in both namespaces, each
with 770 strictly young and 770 matched control records. While the fresh pipeline was running, its
results remained unavailable and the app did not substitute cached results. After completion, a real
HTTP ASGI check returned 200 for `/api/status`, `/api/evaluation` and `/api/cohorts`, all explicitly in
`from_raw` mode. The app verified fresh aggregate SHA-256
`3dd9bac300ad5002dac0312a24088eefb1c6b6aaaa73e31c3aacde4710670086`, loaded all 15 models and 30
model/site payloads, retained the fixed baseline default and displayed the unchanged failed primary
comparison. This routing check generated no new model score or waveform read.

## Presentation contract

The fixed operating point comes from the 5% calibration-normal budget. The rule is strictly
`score > threshold`; ties are not flagged. The separate evaluation slider visits saved exact cutoff
matrices, without interpolating counts or changing the frozen operating point. It displays TP/TN/FP/FN,
reference denominators, sensitivity, specificity, precision, referral workload and sample prevalence.
Offline evaluation retains all admitted reference ECGs, including difficult signals. The matrices
describe the classifier, not a validated combined quality-abstention and referral workflow. Quality
abstention is explicitly no result, with repeat acquisition or clinical review suggested. Scores are
labeled uncalibrated. Below threshold means not flagged at that cutoff, not absence of risk.

The primary strict-age/matched-control cohorts remain distinct from the secondary expanded-age 25,000
selections. Source composition, label-evidence ranks, exact-age summaries and missing/censored ages are
reported from completed aggregate artifacts. They are hospital ECGs, not an Italian student sample.
PTB/SPH evaluations are reused development evidence, not an untouched final test or clinical validation.
The threshold view directly compares the existing age-preferred and no-age-preference 25,000-record
cohorts, using their fixed 50:50 blends and separate calibration-only 5% normal-budget cutoffs. Both
cohorts retain the same quality and label-admission rules. This display uses completed results and
requires no additional cohort selection or training.

Localization is explicitly optional work in progress. No unverified interval plot, confirmed disease
location or improvement claim is shown, and localization cannot affect the referral result. Unsupported
multilabel diagnosis predictions are unavailable. No closed localization experiment was resumed.

## Review artifacts

The screenshots contain aggregate dashboard information only:

- [Threshold explorer](figures/pr74-corrective/evaluation.png).
- [Cohort comparison](figures/pr74-corrective/cohorts.png).
- [Mobile cohort view](figures/pr74-corrective/cohorts-mobile.png).

Commands, prerequisites, the canonical ECG input contract, artifact allowlists and reproducibility steps
are in [`ui/README.md`](../ui/README.md). Browser checks use the standard-library WebDriver client in
`scripts/validation/check_corrected_ui_browser.py`; no browser package or root dependency change was needed.
