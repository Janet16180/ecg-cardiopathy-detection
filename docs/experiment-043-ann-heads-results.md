# Experiment 043 results: ANN heads, an attention head on ECG-JEPA tokens and the tutor's CNN + transformer

Completed 1 October 2026 under the [frozen protocol](experiment-043-ann-heads.md) (commit `4b81ee4`; networks
and training loop in `3bb6e39`). Each stage recorded the hash of its runner, of `ecg_experiment/ann_heads.py`
and of the protocol in its `result.json`, and these equal the committed files: Stage 1 ran from `3bb6e39` with
the runner committed as `6228bff`, Stage 2 from `6228bff` (committed as `aad3023`) and Stage 3 from `22638d9`
(committed as `fcf9711`). Development data only: PTB-XL fold 9 and SPH, both read by earlier experiments. No
PTB-XL calibration or test ECG and no Challenge calibration or test-group ECG was scored. Outputs are in
`outputs/experiment043_ann_heads_v1/stage{1,2,3}/` (`result.json` with every input, source and protocol hash,
`predictions.npz`, `run.log`; Stages 2 and 3 also `token_maps.npz`, Stage 2 `cache_rows.csv`). A
training-rows-only smoke test preceded every stage. No figure was drawn.

## Readings

R is pipeline v2's readout (Experiment 037). An arm beats R if its SPH AUROC difference has a lower bound above
0 and its full-development difference is at least -0.005; it matches R if the SPH lower bound is above -0.01
and the full difference is at least -0.01.

| Arm | Stage | SPH minus R [95% interval] | Full minus R [95% interval] | Reading |
| --- | ---: | --- | --- | --- |
| `logistic_concat` (xECG + JEPA, logistic) | 1 | +0.0017 [+0.0008, +0.0026] | +0.0041 [+0.0008, +0.0077] | **Beats R** |
| `mlp_xecg` | 1 | -0.0017 [-0.0035, +0.0000] | +0.0042 [-0.0013, +0.0096] | Matches R |
| `mlp_concat` | 1 | -0.0006 [-0.0025, +0.0013] | +0.0035 [-0.0025, +0.0099] | Matches R |
| `logistic_jepa` | 1 | -0.0052 [-0.0070, -0.0034] | -0.0005 [-0.0063, +0.0056] | Matches R |
| `attention_jepa` | 2 | +0.0017 [-0.0003, +0.0039] | +0.0063 [-0.0002, +0.0133] | Matches R |
| `cnn_transformer` (tutor's, from scratch) | 3 | -0.0097 [-0.0119, -0.0075] | -0.0106 [-0.0179, -0.0033] | **Below R** |

A map improves on 042's `U_B` if it keeps premature-beat localization (lower bound of its hit - chance minus
`U_B`'s above -0.10) and gains at least one of: anterior lead contrast lower bound above 0, benign any-red
difference upper bound below 0, worst-unit AUROC difference lower bound above 0.

| Map | Keeps premature-beat localization | Lead gain | Benign gain | Detection gain | Improves on `U_B` |
| --- | --- | --- | --- | --- | --- |
| `U_B_gated` | Yes: 0.000 [-0.041, +0.041] | No | No: -0.038 [-0.096, 0.000] | No | No |
| `attention_jepa` | No: -0.854 [-0.914, -0.776] | Yes | No | Yes | No |
| `cnn_transformer` | No: -0.662 [-0.768, -0.560] | No | No | No | No |

`logistic_concat` beats R, so the protocol's demonstration notebook is due; it has not been written yet.
Experiment 044 adopted this readout as [pipeline v3](pipeline-v3.md).

## Integrity

- Training rows: the xECG matrix (55,011 × 1,024), the binary rows, labels and design weights, and the
  development and SPH rows equal Experiment 037's `training_data` exactly. Counts: 39,577 binary rows
  (17,083 PTB-XL), 27,360 positive, 1,724 dropped PTB-XL rows.
- Refitting R (316 iterations, OpenBLAS Haswell kernels) reproduced 037's saved probabilities to 1.1e-16
  (development) and 2.2e-16 (SPH). R's AUROCs equal 037's on all four sets: full 0.93056, ordinary 0.95356,
  hard 0.71241, SPH 0.93869.
- 037's predictions, 035's predictions and 042's unit scores match their receipts. Stages 2 and 3 found R and
  the validation split equal to Stage 1's (largest difference 0.0); Stage 3 found Stage 2's predictions equal
  to their receipt.
- The recomputed `U_B` point metrics (threshold, AUROC, AP, red shares, hit and chance rates, hit - chance,
  lead-contrast group means) equal 042's `result.json` exactly in every stage.
- Validation split: 3,732 of 37,316 groups (PTB-XL patients, Challenge records), 3,939 rows (2,733 positive,
  1,652 PTB-XL). The networks trained on the other 35,638 rows (24,627 positive).
- Stage 2 cache: for all 62,189 rows (39,577 training, 1,604 development, 21,008 labeled SPH), the float32
  token mean equals the cached JEPA feature to 2.0e-6 (tolerance 1e-4), checked before float16 storage. Every
  Challenge window was read with its official checksums at its saved window start, every Ningbo window
  matched the manifest hash, and every SPH window matched its manifest hash.
- Stage 3 rehashed the window cache; its data SHA-256 (`a712335a…`) equals the cache receipt and Stage 2's.
- The per-token contributions sum to the logit minus the bias to 1.6e-6 (Stage 2) and 1.5e-6 (Stage 3).
- Every paired interval used 2,000 whole-patient draws with seed 43043; no draw was skipped.

## Stage 1: MLP and logistic heads on frozen features (CPU)

| Arm | Full | Ordinary | Hard | SPH | SPH AP |
| --- | ---: | ---: | ---: | ---: | ---: |
| R | 0.9306 | 0.9536 | 0.7124 | 0.9387 | 0.9164 |
| `mlp_xecg` | 0.9347 | 0.9546 | 0.7519 | 0.9370 | 0.9175 |
| `logistic_concat` | 0.9347 | 0.9566 | 0.7462 | 0.9404 | 0.9192 |
| `mlp_concat` | 0.9341 | 0.9539 | 0.7646 | 0.9381 | 0.9198 |
| `logistic_jepa` | 0.9300 | 0.9530 | 0.7242 | 0.9335 | 0.9091 |

`mlp_concat` minus `logistic_concat`: SPH -0.0023 [-0.0040, -0.0007], full -0.0006 [-0.0061, +0.0052]. On
the same features, the MLP ranks SPH worse than the logistic head.

| Arm | Seed | Best epoch / epochs run | Validation AUROC | Full | Hard | SPH |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `mlp_xecg` | 43043 | 10 / 21 | 0.9634 | 0.9315 | 0.7419 | 0.9318 |
| `mlp_xecg` | 43044 | 7 / 18 | 0.9648 | 0.9310 | 0.7501 | 0.9340 |
| `mlp_xecg` | 43045 | 7 / 18 | 0.9647 | 0.9331 | 0.7435 | 0.9369 |
| `mlp_concat` | 43043 | 6 / 17 | 0.9651 | 0.9283 | 0.7515 | 0.9342 |
| `mlp_concat` | 43044 | 6 / 17 | 0.9655 | 0.9324 | 0.7722 | 0.9365 |
| `mlp_concat` | 43045 | 4 / 15 | 0.9650 | 0.9347 | 0.7569 | 0.9374 |

Best epochs count from 0. Every single seed is below its three-seed mean at SPH, and the MLPs stopped after 15-21 of the 200 allowed
epochs. The logistic heads converged in 422 (`logistic_concat`) and 297 (`logistic_jepa`) iterations.

**`U_B_gated`.** Gating 042's beat-wave scores by the sign of the `G_B` readout share kept 54.9% of the
881,184 units.

| Map | Red threshold | Worst-unit AUROC | Any red: normal / positive / PVC / benign | Hit / chance | Hit - chance [95% interval] | Anterior contrast | Inferior contrast |
| --- | ---: | ---: | --- | --- | --- | --- | --- |
| `U_B` | 1158.8 | 0.740 | 0.052 / 0.286 / 0.774 / 0.077 | 0.932 / 0.197 | +0.734 [0.667, 0.789] | +0.085 [-0.031, +0.212] | +0.115 [0.006, 0.233] |
| `U_B_gated` | 799.0 | 0.736 | 0.052 / 0.305 / 0.821 / 0.038 | 0.932 / 0.197 | +0.734 [0.675, 0.786] | +0.091 [-0.034, +0.217] | +0.108 [-0.002, +0.225] |

`U_B_gated` minus `U_B`: hit - chance 0.000 [-0.041, +0.041] (the top unit is the same in all 73 premature-beat
ECGs), benign any red -0.038 [-0.096, 0.000], worst-unit AUROC -0.004 [-0.012, +0.003]. Benign variants with
red fell from 4 of 52 to 2, but the interval reaches 0, so the gain does not pass.

## Stage 2: attention head on frozen ECG-JEPA tokens (GPU)

| Arm | Full | Ordinary | Hard | SPH | SPH AP | Hard AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R | 0.9306 | 0.9536 | 0.7124 | 0.9387 | 0.9164 | 0.2817 |
| `attention_jepa` | 0.9368 | 0.9548 | 0.8098 | 0.9404 | 0.9194 | 0.3813 |

`attention_jepa` minus `logistic_jepa` (the same features): SPH +0.0069 [+0.0052, +0.0088], full +0.0068
[+0.0011, +0.0126]. It misses "beats R" because its SPH lower bound is -0.0003.

| Seed | Best epoch / epochs run | Validation AUROC | Full | Hard | SPH |
| ---: | --- | ---: | ---: | ---: | ---: |
| 43043 | 4 / 9 | 0.9603 | 0.9351 | 0.7979 | 0.9360 |
| 43044 | 5 / 10 | 0.9604 | 0.9361 | 0.8078 | 0.9400 |
| 43045 | 7 / 12 | 0.9600 | 0.9346 | 0.8073 | 0.9397 |

| Map | Red threshold | Worst-unit AUROC | Any red: normal / positive / PVC / benign | Hit / chance | Hit - chance [95% interval] | Anterior contrast | Inferior contrast |
| --- | ---: | ---: | --- | --- | --- | --- | --- |
| `attention_jepa` | 0.2746 | 0.897 | 0.052 / 0.472 / 0.405 / 0.115 | 0.027 / 0.147 | -0.119 [-0.156, -0.070] | +0.190 [0.080, 0.299] | +0.205 [0.112, 0.300] |

`attention_jepa` minus `U_B`: hit - chance -0.854 [-0.914, -0.776], benign any red +0.038 [-0.077, +0.154],
worst-unit AUROC +0.157 [0.126, 0.189]. The attention map's top token falls on the premature beat in 2.7% of
PVC ECGs, below the 14.7% expected by chance. It marks 47% of positives red against `U_B`'s 29% at the same
normal rate, and it is the only map in 042-043 whose anterior and inferior contrasts both have intervals above
0 (anterior hit 0.445 in anterior-only against 0.255 in inferior-only infarcts; inferior hit
0.356 against 0.151). It cannot mark leads III, aVR, aVL or aVF, which ECG-JEPA does not take.

## Stage 3: the tutor's CNN + transformer from scratch (GPU)

Input: the cached canonical 500 Hz windows decimated to 250 Hz, each lead's per-record median subtracted, then
divided by that lead's SD over the training part. The protocol does not say whether the SD is weighted; it
was taken unweighted, pooled over every sample of the 35,638 training-part records around their pooled mean
(lead SDs 0.13-0.35 mV). Augmentation was applied to training batches only.

| Arm | Full | Ordinary | Hard | SPH | SPH AP |
| --- | ---: | ---: | ---: | ---: | ---: |
| R | 0.9306 | 0.9536 | 0.7124 | 0.9387 | 0.9164 |
| `cnn_transformer` | 0.9199 | 0.9452 | 0.7003 | 0.9290 | 0.9034 |

`cnn_transformer` minus `attention_jepa`: SPH -0.0114 [-0.0138, -0.0091], full -0.0169 [-0.0249, -0.0092].
Minus `logistic_concat`: SPH -0.0114 [-0.0136, -0.0093], full -0.0148 [-0.0219, -0.0076].

| Seed | Best epoch / epochs run | Validation AUROC | Full | Hard | SPH |
| ---: | --- | ---: | ---: | ---: | ---: |
| 43043 | 13 / 19 | 0.9435 | 0.9080 | 0.6956 | 0.9171 |
| 43044 | 17 / 23 | 0.9477 | 0.9104 | 0.6694 | 0.9143 |
| 43045 | 8 / 14 | 0.9449 | 0.9134 | 0.6857 | 0.9225 |

The three-seed mean (SPH 0.9290) is above every seed (0.914-0.923). Validation AUROC peaked at 0.944-0.948,
below the frozen-feature heads' 0.960-0.966.

| Map | Red threshold | Worst-unit AUROC | Any red: normal / positive / PVC / benign | Hit / chance | Hit - chance [95% interval] | Anterior contrast | Inferior contrast |
| --- | ---: | ---: | --- | --- | --- | --- | --- |
| `cnn_transformer` | 0.0179 | 0.776 | 0.052 / 0.036 / 0.048 / 0.019 | 0.219 / 0.147 | +0.073 [-0.017, +0.166] | +0.071 [-0.014, +0.154] | +0.062 [-0.034, +0.157] |

`cnn_transformer` minus `U_B`: hit - chance -0.662 [-0.768, -0.560], benign any red -0.058 [-0.154, +0.019],
worst-unit AUROC +0.036 [-0.007, +0.080]. Its red tokens mark positives (3.6%) less often than normals (5.2%).

## Runtime and GPU profile

| Stage | Hardware | Profile | Projected (ceiling) | Actual |
| ---: | --- | --- | --- | ---: |
| 1 | CPU, 4 threads | none required | | 198 s |
| 2 | GPU 0 (RTX 3090, shared) | extraction 29 ms read + 6.8 ms model per record (128 records); 40 ms per optimizer step (30 steps after 3 warm-up) | 3,385 s (7,200 s) | 1,602 s |
| 3 | GPU 0 | 77 ms per optimizer step (30 steps after 3 warm-up) | 5,495 s (14,400 s) | 3,075 s |

- Stage 1: the four arms took 12.9 s (`mlp_xecg`), 14.1 s (`logistic_concat`), 16.6 s (`mlp_concat`) and
  4.8 s (`logistic_jepa`).
- Stage 2: extraction of 62,189 ECGs took 1,189 s (582 s in the model, 421 s waiting for reads, 176 s for
  checks and writes); three seeds trained in 340 s, 8.0-10.5 s per epoch on average by seed.
- Stage 3: preprocessing took 125 s; three seeds trained in 2,831 s, 50.1 s per epoch on average (largest
  50.8 s). GPU memory peaked at 1.94 GiB allocated (3.97 GiB reserved).
- Every CUDA step held the shared GPU lock.
- The caches stay on local NVMe in the session scratchpad, `exp043_cache/` (float16 tokens
  `[62,189, 400, 768]`, 36 GB; float32 windows `[62,189, 12, 5000]`, 14 GB), with their row order in
  `rows.csv` and `receipt.json`.

## Interpretation

- A non-linear head on frozen features did not help. The MLP on xECG and the MLP on xECG + JEPA both match
  R, and on the joint features the MLP is below the logistic head at SPH. Adding the JEPA features to the
  logistic readout is what helped: it is the only arm that beats R.
- The attention head on JEPA tokens is the strongest single network on development data (full 0.9368, hard
  0.8098), and it gains clearly over the logistic head on the same JEPA features. At SPH it equals
  `logistic_concat` (0.9404) without passing the rule against R.
- The tutor's CNN + transformer trained from scratch on 35,638 ECGs is below R by about 0.01 at SPH and on full
  development, and below both frozen-feature heads. The best epochs (counted from 0) were 8, 13 and 17 of the 40 allowed.
- Gating `U_B` by the readout share kept its premature-beat localization exactly and halved its benign
  red marks (4 to 2 of 52), but 52 ECGs are too few for that to pass.
- The two network maps split detection from localization. The attention map detects well and points to the
  expected leads, but its top token rarely sits on the premature beat. `U_B` is the opposite. Experiment 045
  combined the two layers under one budget; the combination lost premature-beat localization.

## Caveats

- The networks saw 90% of the training rows; R and the logistic arms saw all of them.
- The Challenge validation groups are records, so a patient with several records can sit in both parts.
- Three seeds and fixed recipes, with no hyperparameter search: a negative result says these recipes did not
  help, not that the architectures cannot.
- The attention contributions are each network's own decomposition, not a validated explanation.
- The attention map covers 8 leads; the premature-beat rule is automatic (73 ECGs); the benign set is small
  (52; one ECG moves the share by 1.9 points); infarct location comes from whole-ECG statements.
- Development data and SPH were read by earlier experiments, and Experiment 044 then chose its readout from
  these results; the numbers are exploratory.

## Follow-ups

Added to the [backlog](experiment-backlog.json): `benign_referrals_v3`, `ecg_level_explanation_rule`,
`weighted_ensemble`, `cnn_transformer_pretraining` and `jepa_finetune_attention` (deferred: in Experiment 016,
fine-tuning xECG end to end lost to its frozen probe).

## Reproduce

```bash
OPENBLAS_CORETYPE=Haswell PYTHONPATH=. OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES= \
    .venv/bin/python -m scripts.experiments.run_ann_heads043 --stage 1
OPENBLAS_CORETYPE=Haswell PYTHONPATH=. OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 \
    .venv/bin/python -m scripts.experiments.run_ann_heads043 --stage 2 --cache <local cache folder>
OPENBLAS_CORETYPE=Haswell PYTHONPATH=. OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 \
    .venv/bin/python -m scripts.experiments.run_ann_heads043 --stage 3 --cache <local cache folder>
```

Each stage refuses to overwrite its output and writes into `<output>.partial` first. `--smoke` runs a stage on
training rows only. Stage 2 reuses a cache only after rehashing it against its receipt.
