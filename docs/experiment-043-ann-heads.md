# Experiment 043: ANN heads, attention maps and the tutor's CNN + transformer

**Frozen 30 September 2026, before any score of this experiment is computed.** After Experiment 042, the user
asked whether a neural network (ANN) at the end, predicting whether the whole ECG has a cardiopathy, would do
better, and whether the tutor's simple CNN + transformer had been tested (it had not). The user chose three
arms (an MLP head, an attention head on the map, and the tutor's CNN + transformer) and asked that the extra
improvements be ordered by their probability of success. The user then asked for the work to continue
overnight without questions, on the PR #61 branch, without merging.

The stages run in order of the expected probability of success. Each stage writes its own outputs, so a
failure in a later stage cannot affect an earlier one.

| Stage | Arms | Hardware |
| --- | --- | --- |
| 1 | MLP head on xECG; logistic and MLP heads on xECG + JEPA features; label-gated beat-wave map | CPU |
| 2 | Attention head on frozen ECG-JEPA tokens (prediction plus a lead and time map) | GPU |
| 3 | The tutor's CNN + transformer, trained from scratch, with an attention head and augmentation | GPU |

Deferred to the backlog with their reasons: fine-tuning ECG-JEPA end to end (in Experiment 016, end-to-end
fine-tuning of xECG lost to its frozen linear probe, 0.945 and 0.931 against 0.962) and an attention head on
beat-wave pieces (the linear median-beat readout `G_B` reached only 0.810 in 042).

## Comparator and rows

- **Pipeline v2's readout R** (`docs/pipeline-v2.md`, Experiment 037): xECG features, 39,577 training rows
  (17,083 PTB-XL, 22,494 Challenge training-group rows passing the quality policy; 27,360 positive), the 035
  `dropped_upweighted` sample weights, `multisource_readout.fit_readout`. Its saved development and SPH
  probabilities (`outputs/experiment037_pipeline_v2_v1/predictions.npz`, hash checked against the 037
  receipt) are the comparator: AUROC 0.9306 on full PTB-XL development, 0.9387 at SPH.
- **Training rows, labels and weights:** exactly R's, built with Experiment 037's `training_data` (imported
  from its runner, because the row logic lives in pinned scripts; copying it would duplicate about 200 lines).
  JEPA features come from the same caches in the same row order.
- **Validation split for early stopping:** 10% of training groups, drawn once with seed 43043 and shared by
  every arm and seed. A group is a patient for PTB-XL and a record for the Challenge sources, which have no
  patient IDs. The networks train on the other 90%. R and the logistic arms use all rows, as R did.
- **Evaluation:** full PTB-XL development (1,572 ECGs, 884 positive, 1,413 patients; also the 1,306 ordinary
  and 266 hard rows, as 037) and SPH (21,008 labeled evaluation ECGs, 7,190 positive, 20,364 patients). Both
  were read by earlier experiments; this is development data. Calibration and test groups stay closed.
- **Map rows:** the 042 rows on the 1,604 full-development ECGs (premature-beat set 84, benign variants 52,
  anterior-only 146 and inferior-only 149 infarcts, 463 NORM-only normals).

## Arms

All networks use PyTorch, float32, AdamW, a weighted binary cross-entropy with R's sample weights, the
validation AUROC for early stopping (best epoch kept), and seeds 43043, 43044 and 43045. An arm's
prediction is the mean logit of its three seeds. Seeds and epochs are reported, never selected on
development or SPH.

### Stage 1 (CPU)

- `mlp_xecg`: the 1,024 xECG features, standardized with the weighted training mean and SD; one hidden layer
  of 256 ReLU units, dropout 0.2, one output. Learning rate 1e-3, weight decay 1e-4, batch 256, at most 200
  epochs, patience 10.
- `logistic_concat`: `fit_readout` on the 1,792 concatenated xECG and JEPA features with R's weights.
- `mlp_concat`: `mlp_xecg` on the concatenated features.
- `logistic_jepa`: `fit_readout` on JEPA features alone, the same-feature comparator for stage 2.
- `U_B_gated` (map): 042's `U_B` unit score where the `G_B` share of that unit's lead and wave is above 0,
  and 0 elsewhere, from 042's saved scores.

### Stage 2 (GPU)

- Tokens: ECG-JEPA's 400 tokens (042 `jepa_tokens`) of every training, development and SPH ECG, stored as
  float16 in a local scratch cache. Inputs: PTB-XL `ptb_jepa_input`; Challenge `jepa_input` of the
  `canonical_window` at 022b's saved window start, read with `read_verified`; SPH `jepa_input` of
  `sph.read_window`, checked against the manifest window hash.
- `attention_jepa`: an additive multiple-instance head. Each token goes through LayerNorm, a linear layer to
  128, GELU and dropout 0.1. Attention weights are a softmax over the tokens of a two-layer tanh scorer
  (128 to 64 to 1), and each token has a linear logit f. The ECG logit is the sum of the weights times f,
  plus a bias. Learning rate 3e-4, weight decay 1e-2, batch 64, at most 30 epochs, patience 4.
- Its map: the per-token contribution, weight times f, averaged over the three seeds.

### Stage 3 (GPU)

- Input: the canonical 10 s, 500 Hz window of each ECG, from the same readers as stage 2, decimated to 250 Hz
  with `scipy.signal.resample_poly(·, 1, 2)`. Each lead has its median subtracted and is divided by that
  lead's SD over the training part.
- `cnn_transformer` (the tutor's architecture): a CNN shared by all leads turns each lead into 50 tokens of
  0.2 s: Conv1d 1 to 32 (kernel 7), 32 to 64 (kernel 5, stride 5) and 64 to 128 (kernel 5, stride 5), each with
  GELU, then 128 to 128 (kernel 3, stride 2). Learned lead and time embeddings are added. Then a transformer
  encoder over all 600 tokens: 4 pre-norm layers, 4 heads, width 128, feed-forward 256, dropout 0.1. Then the
  stage 2 attention head.
- Augmentation (training only): per-record amplitude scale drawn from U(0.9, 1.1), Gaussian noise with SD
  0.01 of the standardized signal, and each lead zeroed with probability 0.1.
- Learning rate 5e-4, weight decay 0.05, batch 64, gradient-norm clip 1.0, at most 40 epochs, patience 5.
- Its map: the per-token contribution on the 12-lead by 50-patch grid, averaged over seeds.

## Statistics

- **Primary, per detection arm:** AUROC at SPH and on full PTB-XL development, each minus R's, with 2,000
  paired whole-patient bootstrap draws (`intervals.paired_auroc_difference`, seed 43043).
- Secondary: AP; the ordinary and hard development subsets; every seed's AUROC; `mlp_concat` minus
  `logistic_concat`; `attention_jepa` minus `logistic_jepa`.
- **Maps** (`U_B_gated`, `attention_jepa`, `cnn_transformer`): 042's definitions and contrasts against 042's
  `U_B`: premature-beat hit - chance, lead contrasts, benign any-red, worst-unit AUROC. The red threshold is
  the 95th percentile of the worst-unit scores of the 463 NORM-only ECGs.

## Prespecified reading

- A detection arm **beats R** if its SPH difference's lower bound is above 0 and its full-development
  difference is at least -0.005. It **matches R** if the SPH lower bound is above -0.01 and the
  full-development difference is at least -0.01. Otherwise it is **below R**.
- A map **improves on 042's `U_B`** by 042's rule: it keeps premature-beat localization (the lower bound of its
  hit - chance minus `U_B`'s is above -0.10), and at least one of its lead contrast's lower bound is above 0,
  its benign any-red difference's upper bound is below 0, or its worst-unit AUROC difference's lower bound is
  above 0.
- A demonstration notebook is written only if an arm beats R or a map improves on `U_B`, as for 042.

## Integrity

- R's rows: counts as 037 (39,577; 27,360 positive). Refitting R reproduces 037's saved development and SPH
  probabilities to 1e-10 (OpenBLAS Haswell kernels).
- 037's predictions and 042's unit scores match their receipts.
- Stage 2: for every extracted ECG, the token mean equals its cached JEPA feature (PTB-XL cache, Challenge
  `features_challenge_v1`, SPH 022 `features.npz`) to 1e-4.
- Stages 2 and 3: every Challenge window is read at its saved start and every Ningbo and SPH window matches its
  manifest hash.
- Stages 2 and 3 profile before training (128 records for extraction, 30 optimizer steps per model) and
  stop if the projected stage time exceeds 2 hours (stage 2) or 4 hours (stage 3).

## Caveats

- The networks see 90% of the training rows; R saw all of them.
- The Challenge validation groups are records, so a patient with several records can sit in both parts.
- Three seeds and fixed recipes, with no hyperparameter search: a negative result says this recipe did not
  help, not that the architecture cannot.
- The attention contributions are the network's own decomposition, not a validated explanation; the map tests
  check them as in 042.
- Development data were read by earlier experiments; the results are exploratory.

## Execution

```bash
OPENBLAS_CORETYPE=Haswell PYTHONPATH=. OMP_NUM_THREADS=4 uv run --no-sync python -m scripts.experiments.run_ann_heads043 --stage 1
OPENBLAS_CORETYPE=Haswell PYTHONPATH=. OMP_NUM_THREADS=4 uv run --no-sync python -m scripts.experiments.run_ann_heads043 --stage 2
OPENBLAS_CORETYPE=Haswell PYTHONPATH=. OMP_NUM_THREADS=4 uv run --no-sync python -m scripts.experiments.run_ann_heads043 --stage 3
```

`scripts/experiments/run_ann_heads043.py` uses `ecg_experiment/ann_heads.py`, `ecg_experiment/lead_wave_maps.py`
and `ecg_experiment/fragment_localization.py`. Stage outputs go to
`outputs/experiment043_ann_heads_v1/stage{1,2,3}/` (`result.json`, `predictions.npz`, maps, `run.log`); stages
2 and 3 take the GPU lock. Results go to `docs/experiment-043-ann-heads-results.md`.
