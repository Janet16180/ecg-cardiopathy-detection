# KEY-01: condition keys from whole ECGs and automatically selected token regions

Prospective protocol, 3 November 2026. Commit before any new condition score. The user authorized
the comparison proposed in this session. No new encoder training, downloads, image-model run,
clinical label definition, age subgroup or closed-test evaluation is authorized by this study.

## Question and relationship to No Time to Train

Does selecting contiguous regions of frozen ECG embeddings improve class-mean reference keys
over whole-record keys? The motivation is the memory/prototype matching in Espinosa et al.,
[No time to train!](https://arxiv.org/html/2507.02798v1). This is an ECG adaptation, not a
reproduction of its SAM2/DINOv2 image pipeline. Here the encoder's fixed patches propose regions;
reference similarity selects among them. There is no evidence that SAM masks delineate ECG findings.

Existing local NTT studies are not on this main branch. Their saved aggregates are prerequisites:
`outputs/ntt04_same_wave/result.json`, `outputs/ntt06_native_encoder/result.json`, and
`outputs/ntt15_native_candidate_cpu/evaluation/result.json`. NTT-04 and NTT-06 did not beat their
best physical comparator; NTT-15 did not beat its best phase comparator. Preserve these outcomes.
The available directly executable predecessor is HEADS-01: reproduce its saved SPH point metrics
exactly before generating a new score. Experiments 024/024b reports motivate the whole-record
prototype comparator, but their raw outputs are absent; do not claim to reproduce them.

## Inputs and roles

Use only the completed HEADS-01 caches in `outputs/heads01_token_superclass_20261028`:
`extract_training/{rows.csv,pooled.npy,tokens.npy,completion.json}`,
`extract_evaluation/{rows.csv,pooled.npy,tokens.npy,completion.json}`, and
`qualify/{labels.csv,completion.json}`. Verify their artifact hashes before use.
Training is exactly the existing all-age 25k cohort in
`outputs/cohort_age_label_priority_20261028/all_age.csv`; its hash must equal the cohort binding
in the HEADS-01 qualification receipt. Align every array by unique record ID, never row assumption.
All training rows must have `split == train`; no SPH training row is permitted.

Use the existing SPH baseline rows at
`outputs/pr74_fresh_baseline_exact_20261028/features/sph_rows.csv` with their unchanged
`MI,STTC,CD,HYP,target,patient_key,evaluation` columns. This table's digest must equal the
HEADS-01 evaluation receipt's binding. Only the 10,434 rows marked evaluation are scored:
3,584 proxy-positive, 6,850 proxy-negative. No calibration ECG is used to select keys or windows.
SPH is reused external development, not an untouched test. No PTB evaluation, Challenge test,
EchoNext data, raw waveforms, benchmark grounding payloads or new labels are opened.

Four independent class memberships are the existing annotation groups MI, STTC, CD and HYP.
Their union must reproduce the binary target in training and evaluation. Multiple labels remain
multiple labels. These are ECG annotation proxies, not confirmed diseases or new clinical targets.
All rows, including measurement-quality flags already admitted by the predecessor, remain included.
Known patient identities must not overlap across training and evaluation. Source families and
record-split limitations remain as documented in HEADS-01; encoder pretraining is not class-unseen.

## Feature geometry and qualification

The released xECG configuration has 25 samples per patch at 100 Hz: 40 contextual tokens over
10 seconds, each 1,024 features. Regions are token-index spans across all leads, not independent
re-encodings of cropped waveforms and not physiological P/QRS/T boundaries. A token can carry
context outside its nominal 250 ms patch. Region indices are model evidence, not clinical localization.

Fit a float64 StandardScaler on the 25,000 full-record pooled vectors only. Standardize a window's
mean raw token vector with this same scaler, then L2-normalize; no query-fitted transformation.
Each record has equal weight within each key. A key is the L2-normalized mean of its unit vectors.
Use the existing `embedding_geometry.prototype` implementation for class means.

Qualification: validate shapes/dtypes, finite values in used rows and all identities. The relative
L2 error of the arithmetic mean of all 40 float16 cached tokens against the saved float32 pooled
vector must be <= 0.001 for every used training/evaluation row. Otherwise stop for unresolved
pooling/padding semantics; do not silently drop rows or treat padding as a candidate region.
This tolerance accounts only for cache quantization. It is not an exact feature reproduction claim.
The predecessor's saved classification metrics must still reproduce exactly from saved scores.

## Fixed arms

For each group c, build an initial positive key from training rows with c present and an initial
normal key from training rows with target 0. Other abnormal rows are not support negatives.
Each score below is cosine to the positive key minus cosine to the normal key.

- `whole`: score the saved full-record vector against the initial keys.
- `window_whole_keys`: score every proposed region against initial keys and take the maximum.
- `window_selected_keys` (candidate): for each positive training record, select the region with
  highest initial positive-minus-normal margin. For each normal training record and class, select
  the region with lowest such margin. Average those selected unit vectors into new positive and
  class-specific normal keys, once. No repeated refinement or parameter optimization. At query,
  take the maximum positive-minus-normal score across proposed regions.
- `fixed_selected_keys`: repeat the same single selection/refinement and query maximum, using
  only ten nonoverlapping four-token (one-second) spans, to control the benefit of variable windows.
- `shuffled_keys`: cyclically permute the four candidate positive and class-specific normal keys
  by one group at inference, preserving window candidates. Diagnostic control, not a decision arm.

Proposed variable windows are every stride-one span of lengths 2, 4 and 8 tokens (0.5, 1 and
2 seconds): 109 candidates per record. Fixed ordering is length ascending, then start ascending;
ties select the first. No threshold or window-size search. Candidate selection is weakly supervised
by record labels; the selected regions have no regional reference truth. Save keys, selected
training spans, query maximizing spans and all scores locally for arithmetic verification.

## Primary endpoint and decision

Site: SPH evaluation rows, all four one-versus-rest annotation memberships. For each group,
every other evaluated record is a negative, including ECGs with other abnormal groups. The endpoint
is the unweighted mean of the four AUROCs, not only condition-versus-normal discrimination.

Primary contrast: macro AUROC of `window_selected_keys` minus `whole`. Paired whole-patient
bootstrap, 2,000 draws, seed 20261103, shared `intervals.patient_groups` and `patient_resample`.
Every draw resamples the same patients for every group/arm; skip and count a draw if any group
has one class. Report the 2.5/97.5 percentile interval. Uncertainty conditions on the fixed support
cohort, fitted scaler/keys and pretrained encoder; it does not include retraining or support sampling.

Advance region selection only if (1) primary point gain >= 0.01, (2) its 95% lower bound > 0,
and (3) the candidate macro AUROC is at least that of `fixed_selected_keys`. Gate 3 is descriptive,
not a noninferiority claim. No arm or pipeline is promoted automatically. A failed gate is retained.

Secondary descriptive endpoints: per-group AUROC and average precision on all evaluated rows;
the same metrics restricted to group-present versus target-0 records; macro average precision;
`window_whole_keys` and shuffled controls; histograms of selected lengths and starts. Report
annotation counts, patient counts, support label co-occurrence and source counts. Scores are not
calibrated probabilities; this ranking experiment makes no referral-budget or clinical accuracy claim.

## Integrity, execution and artifacts

Recompute HEADS-01's saved SPH evaluation metrics for every saved model from its saved
`sph_predictions.npz` and thresholds in `evaluate/evaluation.json`, requiring exact equality
of every field in `point_metrics`. No refit, age-subgroup score or renewed old experiment.
Capture predecessor artifact hashes and the new committed protocol/source revision. Never modify
old receipts or pinned code. Scientific implementation lives in new library/runner/test files.

Use two numerical threads, batch at most 128 records, CPU only. Cache arrays are memory-mapped.
Profile the actual first 128 training rows before full construction, with no query metrics; if the
forecast for feature processing exceeds 1,800 seconds, record a resource stop. Bound the complete
run to 3,600 seconds; save failure/resource receipts rather than partial results masquerading as final.
No GPU lock is required because no GPU is used. Preserve failed attempts; refuse output overwrite.

Output: `outputs/condition_keys_20261103/` (private arrays, keys, spans, profile, integrity, result).
Publish only aggregate results in `reports/condition-keys-2026-11-03/aggregate.json` and
`docs/condition-keys-results-2026-11-03.md`, update both queue documents and rank follow-ups.
Write the report from executed outputs, including negative findings and exact scope limits.

This experiment does not test a condition with no reference examples. It also does not establish
recognition of unseen disease categories: all four groups already exist in the reference data.
Reference-scarce or withheld-category transfer needs a separate prospective protocol.

## Notebook and descriptive clusters (user addition before scoring)

The user additionally requested an executed Jupyter notebook explaining the complete method,
clusters and prediction results. This addition is frozen before fitting/scoring KEY-01.
Fit PCA (16 components, randomized solver, seed 20261103) on standardized unit whole-record
training embeddings; use the first two coordinates for plots. Fit k-means with k = 5, n_init = 10,
seed 20261103 on the 16 training PCA coordinates, then assign SPH evaluation rows without refitting.
These unsupervised clusters are descriptive and do not enter condition keys or primary endpoints.
Five clusters do not imply five mutually exclusive diseases. Report training cluster sizes, source-family
composition and AMI with source family and the exact four-bit annotation combination. At SPH, report
cluster sizes and each group's annotation prevalence by cluster; no query-driven cluster naming or
prediction mapping. Save only aggregate cluster tables, contingency counts and PCA plot grids in
publishable artifacts, keeping per-record coordinates and memberships local. Notebook cells load
executed aggregate outputs and show the prototype algorithm, patient/source roles, primary and
secondary performance, key similarities, window-selection distributions and cluster patterns.
Its conclusions remain descriptive; project recommendations belong in the separate results report.
