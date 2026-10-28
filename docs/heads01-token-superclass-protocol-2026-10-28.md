# HEADS-01: token-pooled xECG readout and superclass-decomposed heads

Committed 28 October 2026 before any new model fit, token extraction or score. Successor of the
[corrective PR 74 evaluation](pr74-corrective-evaluation-protocol-2026-10-28.md) and its
[from-raw reproduction](pr74-from-raw-reproduction-results-2026-10-28.md). Reused development data
only: no Challenge test group, EchoNext test, PTB fold 10 or final frozen test is opened.

## Why these two ideas

Training data diversity, not head or fusion changes, produced the last large SPH gain (PTB-only 0.686
to all-age 25k 0.774 sensitivity), and the remaining admitted pool adds only 3,568 mostly non-Chinese
ECGs, while the PTB-only head and fusion screens (SCREEN-02 to 05) never changed the xECG
representation or the single binary target. We therefore test the two ranked backlog readouts that do:
`attention_readout` in its deterministic first form (mean plus max pooling of the 40 frozen xECG
tokens that the official readout averages away) and `multitask_head` (separate superclass-versus-normal
heads combined as a mixture likelihood ratio), both on the unchanged all-age 25k cohort.

A concurrent study (`pr75` training-only hyperparameters) tunes HGB settings and the logistic C on the
same cohort. This study changes neither; it keeps every predecessor hyperparameter fixed.

## Arms

All arms are fitted on the completed all-age 25,000-ECG cohort
(`outputs/cohort25_recovery_20261028/cold_start/cohorts_25k/all_age.csv`; 7,393 negatives,
17,607 positives) with the frozen predecessor recipes below. Every blend is the fixed arithmetic mean
of its HGB and xECG component scores (0.5 each); no fusion weight is chosen here.

| Arm | HGB component (130 measurements) | xECG component | Role |
| --- | --- | --- | --- |
| R `all_age` (current best) | binary HGB | binary logistic on the saved 1,024 mean-pooled features | reference, must reproduce |
| D `ptb_baseline` (default) | saved PTB-only model | saved PTB-only model | reference, saved predictions |
| A `token_pool` | R's binary HGB, unchanged | binary logistic on 2,048 features: saved mean ‖ new token max | primary candidate |
| B `superclass` | four superclass heads, mixture | four superclass heads on the 1,024 mean features, mixture | primary candidate |
| C `token_superclass` | B's mixture HGB | four superclass heads on the 2,048 features, mixture | secondary, descriptive |

Token max: the frozen released xECG returns its pooled vector and the 40 final-block tokens
(`[40, 1024]`, 25 samples per token at 100 Hz). The official pooled vector is their mean. The new
input is the elementwise maximum over the 40 tokens, excluding the tokens the official pooling masks as
padding (none in a normal recording), concatenated after the saved mean features.

Superclass heads: for each superclass k in MI, STTC, CD and HYP, one head is fitted on the cohort's
negatives plus every positive carrying k (a record with several superclasses enters each of its heads).
Each head uses the same fixed recipe as its binary counterpart. With `n_0` negatives and `n_k` k-positives
in that head, its log-likelihood ratio is the head's decision function minus `log(n_k / n_0)`. The
component score is the mixture log-likelihood ratio `S = log sum_k (n_k / sum_j n_j) exp(LLR_k)`,
returned on the probability scale of the binary heads as `sigmoid(S + log(n_pos / n_0))`, where `n_pos`
and `n_0` are the cohort's binary counts. The binary target is unchanged; the superclass flags are the
documented components of its existing definition.

## Superclass labels and their integrity gate

Challenge flags come from the `# Dx` line of each admitted training header, after checking the
header's SHA-256 against the official release checksums, mapped by the unchanged
`challenge_labels.record_labels`. PTB flags come from `ptbxl_database.csv` `scp_codes` with the unchanged
`eda.ptbxl.diagnostic_classes`, restricted to admitted training IDs before any other use. For all 28,568
admitted pool records, the reconstructed primary label must equal the cohort target, and the union of the
four flags must equal the target. Any mismatch stops the study. Only training records are read.

A metadata-only feasibility check before this commit (no model, no score) found zero mismatches and these
all-age positives per head: MI 2,041, STTC 12,356, CD 6,049, HYP 2,376 (4,540 positives carry more than
one superclass), each against the 7,393 negatives. The qualify stage repeats this check with receipts.

## Primary comparison and decision rule

Site: SPH, the exact existing roles: 6,968 calibration normals set every threshold; 10,434 evaluation
ECGs (3,584 proxy-positive, 6,850 proxy-negative, 10,116 patients) are scored. For each model, sort the
calibration normal scores descending, take index `floor(0.05 * 6968)` as the threshold and refer on strict
`score > threshold` (the unchanged `pr74_screening_metrics.budget_threshold`).

Primary contrasts: SPH sensitivity of A blend minus R blend, and of B blend minus R blend. Each uses a
paired whole-patient bootstrap of the frozen referral decisions (2,000 draws, seed 20261028,
`intervals.two_class_resamples`, single-class draws skipped and reported). With two primary candidates,
each decision interval is the two-sided 97.5% percentile interval (Bonferroni); the 95% interval is
also reported.

A candidate establishes an improvement over the current best only if all three gates pass:

1. point sensitivity gain at least 0.02 (the predecessors' practical bar, unchanged);
2. lower bound of its 97.5% paired interval above zero;
3. SPH evaluation specificity no more than 0.005 below R's (point guard; not a noninferiority test).

A candidate passing gates 2 and 3 but not gate 1 is reported as a statistically supported gain below the
practical bar, not as an established improvement. If both candidates pass, both are reported; neither is
promoted automatically, and the default model stays `ptb_baseline_blend` until the user freezes a pipeline.

Comparisons against the default: A minus D and B minus D sensitivity with 95% paired intervals, same
draws and seed. These are reported for context and carry no decision.

Secondary, descriptive only: arm C against R; every component; paired AUROC differences
(`intervals.paired_auroc_difference`, same draws and seed, 95%); AUROC, average precision, specificity,
precision, referral fractions, TP/TN/FP/FN and threshold sweeps; SPH sensitivity by superclass flag
(MI, STTC, CD, HYP) for R, A, B, C and D without intervals; the same models at PTB (fold 8 normals
calibrate, fold 9 evaluates). No age subgroup is analysed and no label definition changes.

All model fitting finishes before the first new SPH or PTB score. No result changes an arm, a recipe,
a threshold rule or a gate. The uncertainty is conditional on the fitted models and calibration sample.

## Fixed recipes and seeds

- HGB: `cpu_screening041.HGB_CONFIG` (200 iterations, learning rate 0.05, 15 leaves, 30 samples per leaf,
  L2 10, no early stopping, `random_state=20261018`), native missing values, the 130 frozen measurements.
- Logistic: `full_development.fit_logistic` (training-only StandardScaler, L2, C=0.01, lbfgs, tol 1e-8,
  max_iter 5,000, no class weights, `random_state=42`); non-convergence stops the study.
- Bootstrap: 2,000 whole-patient draws, `numpy.random.default_rng(20261028)`.
- Threads: at most two numerical threads per process (`threadpool_limits(2)`, OMP/OpenBLAS/MKL at 2,
  `torch.set_num_threads(2)`).

## Predecessor integrity check, before anything new

1. Verify the completed receipts and artifact hashes of the from-raw evaluation
   (`outputs/pr74_from_raw_validation_20261028/{extract,evaluate}`), the exact baseline features
   (`outputs/pr74_fresh_baseline_exact_20261028/features`) and both cohort receipts.
2. From the saved SPH and PTB predictions, recompute every one of the 30 model/site calibrations,
   confusion matrices and AUROCs; all must equal `screening_evaluation.json` exactly, and the saved
   strict-versus-matched primary interval must reproduce exactly.
3. Refit the all-age and age-preferred binary heads from the saved features; their SPH and PTB scores must
   equal the saved predictions to 1e-12, reproducing `all_age_blend` 2,774/3,584 and `younger_blend`
   2,760/3,584 caught at SPH. `ptb_baseline_blend` (2,460/3,584) is reproduced from its saved model.

Any failure stops the study before extraction.

## Smoke test, profile and extraction

Smoke test (about ten minutes, no evaluation score): reconstruct labels for a small training slice,
extract tokens for 32 admitted training records on the GPU and check that the token mean equals the
saved pooled features, then run the fitting, scoring, bootstrap and export code on synthetic features
and labels. Its outputs stay under a separate `smoke/` directory and are never reported as results.

GPU profile, per device used: 48 records of that part through the real raw read, checksum checks,
preprocessing and frozen encoder. Gates: at least 4 GiB free at start, added peak allocation below 3 GiB,
`2 * measured seconds per record * part records + 300` below two hours, and token output below 6 GiB in
total. A failed gate stops extraction and is reported. Use the shared lock of the device used
(`/tmp/ecg_project_gpu.lock` for GPU 0, `/tmp/ecg_project_gpu1.lock` for GPU 1, as in
`pr74_fresh_baseline`), never touch other processes, and check `nvidia-smi` before launching.

Extraction parts: `training` = the 19,666 admitted Challenge rows in the predecessor's extraction order,
then the 8,902 admitted PTB pool rows; `evaluation` = all 17,402 SPH rows, then the 4,275 PTB fold 8 and
9 rows. Batches of 16 follow each source block's order. Each part saves the pooled vector, the token
maximum (float32) and the 40 tokens (float16, for a later learned-attention follow-up). Every pooled vector
must match its saved predecessor feature within 1e-5 absolute, or the part fails.

## Data roles

SPH never fits a model, scaler, label, weight or feature transform; its calibration normals only set
thresholds and its evaluation rows are only scored. Training uses admitted original Challenge training
records and PTB folds 1 to 7 from the all-age cohort. The training/evaluation identity overlap check of the
predecessor (`pr74_screening_study.assert_evaluation_independence`) runs before scoring each site.
Released xECG pretraining exposure and the proxy nature of the label remain limitations.

## Outputs

Local, never committed: `outputs/heads01_token_superclass_20261028/` with `qualify/`, `smoke/`,
`profile_<part>/`, `extract_<part>/` and `evaluate/` (labels, tokens, models, per-record predictions).
Each stage writes a completion receipt binding source, protocol, input and artifact SHA-256 values.
Committed: this protocol, source, tests, an aggregate-only `reports/heads01-token-superclass-2026-10-28/aggregate.json`,
the results report written from the outputs, both queue files and the ranked backlog.
