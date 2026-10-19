# Experiment 042: external development transfer of the CPU screen

Prospective protocol, 18 October 2026. Freeze before any SPH model score or new extraction.
The user authorized autonomous work with existing data, no cardiologist and no closed tests.
Experiment 041 is the fixed predecessor; no model choice or setting here depends on its result.

## Question and scope

Does the fixed nonlinear raw-measurement screen outperform its matched logistic comparator on
SPH after a separate local-normal referral calibration? This tests transfer of the CPU implementation,
not the stronger historical xECG pipeline. SPH has already been read in Experiments 022–037 and is
development data. It is never used to fit these classifiers. Existing annotations are proxies for
ECG findings, not confirmed structural disease or regional localization.

## Model and extraction identities

Load exactly the two Experiment 041 saved models, their ordered 130 raw features and PTB thresholds.
Require a completed 041 integrity gate; bind its result, model and prediction file hashes. No refit,
new head, parameter search, model selection or seed replication. An unfavorable 041 development
decision remains unchanged regardless of this separate external comparison.

The compatible raw-measurement algorithm is embedded in `notebooks/Avance2.8.ipynb`, cells 15 and 20.
Its six extraction functions, concatenated in the notebook's recorded order with their trailing
newlines, plus `repr(PLAUSIBLE)`, hash to prefix `ee39c8f6f7`, matching the original feature cache.
Create a forward reusable module preserving its arithmetic and exposing an array entry point:
float64, samples by 12 canonical leads, 500 Hz, mV. The older `eda/avance2_features.py` is different
and must not substitute for this algorithm. SPH is already in mV; do not divide it by 1,000.

Before SPH preparation, regenerate the 32 lowest numeric ECG IDs in PTB fold 1 present in the raw
cache, through the new array entry point and the notebook-compatible PTB loader. Compare every
numeric measurement after the original `%.6g` serialization, including matching missingness, and
status/plausibility fields. Stop if any differs. This technical qualification produces no disease
score and does not validate the physiological accuracy of delineation. Use the same six-significant-
digit storage precision when preparing and reloading SPH features for inference.

## SPH source, quality and labels

Read only the public SPH raw source through `data/raw/sph/sph`, linked per source into this checkout.
Never write through that symlink, read the other checkout's outputs, or touch its Git state.
Use existing `sph.verify_small_files`, `sph.table`, `sph.read_window`, `ecg_quality.assess` and
`sph.duplicate_status` rather than reimplementing their policies. Bind source hashes, small-file
verification, per-window hashes and the prepared feature/identity table. First 10 seconds only,
float32 12-by-5,000 mV as returned by `read_window`, then transpose and convert to float64.

Preserve the existing SPH `primary` label: any existing MI/STTC/CD/HYP mapping is positive, code 1
alone is negative, all other code combinations undefined. Keep class flags for descriptive overlap
counts. No new clinical mapping, age analysis or label expansion. The PTB and SPH definitions are
not identical; report the known difference, especially in the negative cohort.

Prepare all source records to resolve exact-window duplicates using the existing raw AHA strings.
Retain `unique` and `kept` records; exclude conflicting duplicates and extra copies. Retain technical
quality failures in evaluation, as in the existing SPH evaluation policy. Preserve extraction status
and missing measurements; do not selectively drop hard ECGs. Flat/no-beat paths produce missing
measurements and remain scoreable. Unexpected I/O or numeric failures stop with a receipt.

## Patient roles and local calibration

Before model scoring, assign each patient deterministically by the first byte of
SHA-256(`experiment042:20261018:` + patient ID encoded as UTF-8): byte below 128 is calibration,
otherwise evaluation. All records from a patient share a role. Assert no overlap and bind the
ordered role table before inference. Calibration/evaluation each require both primary classes and
at least 100 target negatives, or stop. Only defined primary labels enter the performance comparison.

For each model, use all retained calibration negatives to fit its own 5% referral threshold:
k=floor(0.05 m), threshold = (k+1)-th highest score, refer strictly above it. Report achieved
calibration referral rate and ties. These normals are annotated SPH normals, not recruited students.
No positive calibration label affects a threshold or fit. No evaluation label affects calibration.

## Frozen endpoints and decision

Primary: evaluation-patient AUROC difference, boosting minus logistic. Use
`intervals.paired_auroc_difference`, 2,000 whole-patient draws, seed 20261018, with percentile 95%
interval and skipped draws. Return `support_external_development` only when:

- primary point difference is at least 0.01 and lower 95% bound is strictly positive;
- boosting sensitivity at its local-normal threshold is at most 0.02 below logistic sensitivity;
- boosting refers no more than 8% of evaluation target negatives.

Otherwise return `do_not_advance_external`. These are development gates and cannot authorize
deployment, pipeline replacement or a final test.

Secondary: each arm's AUROC, average precision, Brier, fixed-threshold sensitivity/specificity and
whole-patient intervals using the same shared conventions; paired sensitivity difference; descriptive
miss counts among positive MI/STTC/CD/HYP records, allowing overlap; and the evaluation sensitivity
and negative referral rate obtained by transporting the original PTB thresholds without SPH
calibration. No per-class fitting, explainability/localization claim or search for favorable subsets.

## Operational profile, outputs and closure

CPU only, at most four numerical workers/threads and eight total across concurrent agents. Before
full preparation, profile the first 64 lexically sorted SPH records through the actual window read,
feature extraction, quality assessment and hashing path, without model inference. Forecast full
preparation conservatively as twice the observed per-record wall time times the source record count
plus 120 seconds; require at most 3,600 seconds. Require projected total output at most 512 MiB.
If either fails, preserve profile and stop. Profile records may be reread during preparation; account
for that overhead. The full stage has a two-hour wall budget, including qualification and intervals.

Use exclusive stage directories under `outputs/experiment042_measurement_transfer_v1/`; never
overwrite earlier receipts. Persist qualification, profile, preparation and evaluation summaries,
source/protocol/041/input SHA-256 identities, versions, timing and Git revision. Save aligned feature
and prediction payloads locally; paths use `paths.to_stored`. No private record IDs, waveform data,
features or predictions enter Git or an external API. The coordinator writes the results report
from executed outputs, requests independent source/arithmetic verification, updates both queues and
the ranked backlog, and preserves every negative result. No Challenge or EchoNext data are opened.
