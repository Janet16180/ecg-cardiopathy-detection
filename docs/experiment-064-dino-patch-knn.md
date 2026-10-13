# Experiment 064: SimDINO beat-patch tokens scored by phase-matched kNN

Frozen 12 October 2026, before any LocBench score of this method. Part of the
[localization programme](localization-program-2026-10-12.md); scored on [LocBench v1](experiment-061-locbench.md).

## Question

Do self-supervised features of 40 ms single-lead patches of complete beats, learned with a SimDINOv2 CLS
objective plus iBOT-style masked-patch distillation, and scored AnomalyDINO-style by their distance to the
nearest patches of clean normal beats at the same lead and R-relative phase, localize cardiologist-marked
abnormalities better than the best LocBench v1 baseline? And do the learned features add anything over the
same kNN on the raw waveform patches?

The idea answers three of the programme's diagnoses of 041-060: tokens are lead x 40 ms instead of 250 ms
all-lead tokens; the reference is a population bank of clean normals with a per-lead, per-phase comparison
instead of a within-ECG residual; and beats are cut around detected R peaks instead of fixed record windows.

## Tokens

- Preprocessing (every record, pretraining and scoring): raw 500 Hz mV, second-order Butterworth band-pass
  0.5-40 Hz applied forward and backward, polyphase downsampling to 250 Hz, clipping at +-8 mV. One global
  scale: amplitudes stay in mV, with no per-beat or per-lead normalization.
- R peaks: the frozen detector `ecg_experiment.fragment_localization.r_peaks` on the raw 500 Hz record
  (file SHA-256 prefix `268fa8a5672d2765`), halved and rounded to 250 Hz.
- Beat window: R - 400 ms to R + 560 ms (240 samples at 250 Hz). Each lead is cut into 24 patches of 40 ms
  (10 samples), giving 12 x 24 = 288 tokens in lead-major order. The window covers a P wave with a PR
  interval up to about 300 ms and a T wave ending up to 560 ms after R.
- Embedding: a shared linear map of the 10 samples to width 256, plus a learned lead embedding (12) and a
  learned R-relative phase embedding (24), plus a CLS token.
- Pretraining windows are stored with 40 ms extra on each side (260 samples) for crop jitter, and only beats
  whose stored window lies inside the record are kept. Scoring keeps every detected beat; patches that are
  not entirely inside the record are excluded from attention (key mask) and from scoring.

## Network and objective

- Backbone: pre-norm ViT, 6 blocks, width 256, 8 heads, MLP width 1,024, final LayerNorm (4.75 M parameters).
  Two projection heads (CLS and patch), each 256-1,024-1,024-256 with GELU and no prototype layer (SimDINO).
- Views per beat: two global views (all 288 tokens) and four local views (6 random leads x 12 contiguous
  phases, 72 tokens; a local view is a token subset, so its lead and phase embeddings stay true). Every view
  gets its own crop offset (0-40 ms jitter of the R alignment), a global gain in [1/1.1, 1.1], a per-lead
  offset N(0, 0.02 mV) and white noise with a standard deviation drawn from U(0, 0.02) mV.
- iBOT masks on both student global views: one to three blocks, each a random lead subset (every lead kept
  with probability U(0.2, 1), at least one lead) over 2-8 contiguous phases; masked patches are replaced by a
  learned mask token. Mean masked share about 0.23.
- Teacher: EMA of the student (backbone and heads), momentum from 0.996 to 1 on a cosine schedule; it sees
  the two global views unmasked, under `no_grad`.
- Loss = CLS compression + 0.1 x CLS coding-rate expansion + masked-patch cosine distance
  + 0.1 x patch coding-rate expansion.
  - CLS terms: the unchanged upstream xECG `SimDINOv2Loss` (`third_party/bench-xecg`, imported through
    `cpc_simdino040.upstream_loss`, eps 0.05, coefficient 0.1 as in Experiment 040). Compression averages
    the cosine similarity of the 6 student CLS projections with the 2 teacher CLS projections, excluding
    same-view pairs; expansion is the coding rate of the student global-view projections over the batch.
  - Patch term: mean over masked tokens of 1 - cosine(student patch projection, teacher patch projection of
    the same view and token).
  - Patch expansion: the same coding-rate expansion on 2,048 random student patch projections per batch.
    This is an addition to SimDINOv2 (whose expansion acts only on CLS), declared here to keep patch
    features, which the scoring uses, from collapsing.
- Optimization: AdamW, batch 256 beats sampled uniformly with replacement, peak learning rate 5e-4 with 5%
  linear warmup and cosine decay to 1% of the peak, weight decay 0.05 on matrices, gradient clipping at 3,
  bfloat16 autocast with fp32 loss reductions, seed 64064. The number of updates is fixed from the GPU
  profile below. The final EMA teacher is used for scoring; no checkpoint is selected by any score.

## Pretraining data

The curated 500 Hz mV block of cohorts v4 (`clean_100k_v4`, 55,175 rows: PTB-XL, Ningbo, Chapman, Georgia,
CPSC, CPSC-Extra), minus every PTB-XL record that is not in the LocBench `fit` partition. The v4 PTB-XL rows
are all folds 1-8, so this removes exactly the 5,793 `evaluation` records; folds 9-10 and the ECGAD records
are not in v4. The runner raises if an ECGAD record, an `evaluation` record, a Challenge test or
calibration record (by record ID or waveform hash) or Ningbo JS13118 is present.

- EchoNext (filtered, clipped, standardized, no physical unit) is left out because the method keeps mV;
  CODE-15 is left out because its amplitude unit is unresolved; MIMIC is not in the curated block.
- Records with a flat lead (standard deviation below 1 microvolt) are dropped: none were found.
- Result: 49,382 records (Ningbo 19,803, PTB-XL 11,612, Chapman 6,127, Georgia 6,102, CPSC 3,933,
  CPSC-Extra 1,805) and 575,249 beats, cached as float16 on local disk (`/tmp/ecg064-beats`, 3.6 GB,
  built in 4.8 minutes). The ECGAD, `evaluation`, fold 9 and fold 10 records never enter pretraining.

## Normal patch bank and calibration

- Only the 2,666 `fit`-partition clean normals of `docs/locbench-partition-v1.json`. They are split by
  patient (seed 64064) into 2,130 bank records and 536 calibration records.
- Bank: every patch of every bank beat whose 24 patches all lie inside the record (about 91% of beats,
  about 19,000 beats), with no subsampling; the full bank fits on the GPU (about 3 GB in float16).
- Distance: for a test patch at lead l and phase p, the mean distance to its k = 3 nearest bank patches
  among lead l and phases p-1, p, p+1 (the neighbouring bins absorb R-alignment error).
  Learned features: cosine distance of L2-normalized teacher backbone patch tokens. Raw ablation: Euclidean
  distance of the 10 raw samples (mV).
- Calibration: the same distance for every valid patch of the calibration normals (not in the bank), and
  its mean and standard deviation per lead and phase bin. The calibrated score is
  z = (distance - mean) / standard deviation, so a score means "how unusual for this lead and phase".

## Maps

Patch scores are broadcast to the 20 samples (500 Hz) of each patch; where beats overlap the maximum is
taken; samples outside every beat get the record map's minimum. Maps are float32 (n, 12, 5000) per the 061
contract and are stored as float16 under `outputs/experiment064_dino_knn_v1/` for the ECGAD development
records and every `evaluation`-partition record (5,999 records), a superset of the 4,871 IDs of
`locbench.scored_ecg_ids()`.

## Variants

| Variant | Features | Score | Role |
| --- | --- | --- | --- |
| `dino_z` | learned | calibrated z | **primary** |
| `dino_dist` | learned | raw kNN cosine distance (AnomalyDINO as published) | secondary |
| `dino_z_rr` | learned | `dino_z` + max(0, calibrated prematurity z) of the beat, on all its patches | secondary (rhythm) |
| `qrs_lex_dino` | learned + prior | 100 x (061 `qrs_prior` >= 0.5) + stored `dino_z` | secondary (declared prior combination) |
| `raw_z` | raw 40 ms waveform patches | calibrated z | ablation: what the network adds |
| `raw_dist` | raw 40 ms waveform patches | raw kNN Euclidean distance | ablation |
| `random_z` | the untrained network (same seed) | calibrated z | optional diagnostic, if GPU time allows |

Prematurity of a beat is max(0, 1 - preceding RR / median RR of the record) (0 for the first beat),
calibrated by its mean and standard deviation over calibration-normal beats. It is added so that premature
beats with normal morphology, which beat cropping hides from the patch tokens, can still be flagged.

`qrs_lex_dino` ranks every sample inside the 061 `qrs_prior` QRS intervals above every sample outside them,
and orders samples within each stratum by `dino_z` (lexicographic). It uses the saved 061 `qrs_prior` maps
(label-free, from the 059 hybrid delineation) and is stored as float32. It exists because the cardiologist
marks concentrate on QRS complexes (Experiment 061); the anomaly evidence alone is `dino_z`.

Maps are stored after the monotone transform sign(x) log(1 + |x|), which keeps every rank and prevents
float16 overflow (65,504).

## Primary comparison and decision rule

- Primary: paired difference in A2 (within-record point AUROC over the marked ECGAD development records)
  between `dino_z` and the 061 bar, `qrs_prior` (A2 0.7419 [0.7082, 0.7743], B1 macro 0.7283, C1 0.3529,
  D1 0.6321; `docs/experiment-061-locbench-results.md`), from `run_locbench_score --compare
  outputs/experiment061_locbench_v1/tables/qrs_prior`: 2,000 whole-patient bootstrap draws, seed 61061.
- Decision: 064 counts as localizing if the 95% interval of `dino_z` minus `qrs_prior` on A2 lies above 0.
  Learned features count as adding to the raw waveform if `dino_z` minus `raw_z` has a 95% interval above 0.
- QRS-stratified reading (061 amendment 2): `A2_qrs` (inside the 059 hybrid QRS: does the map pick the marked
  complexes and leads) and `A2_nonqrs` (P, PR, ST, T), from `run_locbench_strata` with `qrs_prior` as the
  reference, reported for every variant with paired differences against `residual056` (A2_qrs 0.607
  [0.570, 0.644], A2_nonqrs 0.671) and `qrs_prior` (A2_qrs 0.539, A2_nonqrs 0.661).
- Noise reading (061 amendment 3): `C1_std` from `run_locbench_noise` (lead scores standardized by the
  per-lead median and IQR over the evaluation clean normals; 0.5 = not drawn to noise; `qrs_prior` 0.560,
  `U_B` 0.720, `residual056` 0.707). A map whose `C1_std` 95% interval lies above 0.60 is flagged as drawn to
  noise, whatever its A2. Raw C1 is reported unchanged.
- Also reported for every variant: A1-A6, B1 macro and per family, B2, B3, C1, C2 and D1 with intervals, and
  paired A2 differences against `qrs_prior`, `residual056` and `raw_z`. The secondary variants and ablations
  do not change the decision; a secondary variant that beats the bar is reported as a lead for a new
  protocol.
- Iteration on ECGAD development is allowed: every re-scored version needs a dated amendment below
  committed before it is scored, and every scored version is reported. The ECGAD holdout half is never read
  here.

## Integrity checks

- Reproduce the predecessor: rescore the saved 061 `qrs_prior` maps (`outputs/experiment061_locbench_v1/maps/`)
  with the LocBench scorer and match its A2, B1 macro, C1 and D1 in `outputs/experiment061_locbench_v1/result.json`
  exactly.
- Pool exclusions are asserted by the runner (above); target records must be disjoint from bank and
  calibration records (asserted).
- Pinned inputs recorded in the receipt: the pool manifest SHA-256, the partition and ECGAD split SHA-256,
  the upstream loss file, the model checkpoint SHA-256, and the code commit.
- Unit tests (`tests/test_dino_knn064.py`) and Ruff pass before scoring.

## GPU profile and budget

Real profile (`outputs/experiment064_dino_knn_v1/gpu_profile.json`), RTX 3090 shared with vLLM, under the
project GPU lock, memory capped at 5.1 GB (free memory minus 0.7 GB at the time): batch 256 and batch 128
both ran out of memory without activation checkpointing, so the backbone recomputes each block in the
backward pass (numerically the same gradients; tested). With checkpointing, batch 256: 0.434 s per update
(50 timed updates after 10 warm-up), peak allocated 3.70 GB. Student 7.90 M parameters (backbone 4.75 M).

Pretraining is fixed at **15,000 updates** of 256 beats (3.84 M beat views, 6.7 passes over the 575,249
beats), projected at 1.8 hours from the profile, inside the 2-hour budget with a margin for contention on
the shared GPU. The run checkpoints every 15 minutes and resumes the same schedule if interrupted. The
scoring pass (features and kNN for about 70,000 beats) is projected at well under 30 minutes and runs as a
separate GPU job.

## Outputs

`outputs/experiment064_dino_knn_v1/`: `gpu_profile.json`, `pretrain_log.jsonl`, `pretrain_summary.json`,
`model.pt`, `maps_<variant>.npy` (float16), `ecg_ids.npy`, `calibration.npz`, `maps_receipt.json`, the
LocBench score outputs and `result.json`; the report is `docs/experiment-064-dino-patch-knn-results.md`.
