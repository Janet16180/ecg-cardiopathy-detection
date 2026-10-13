# Pipeline v4: the candidate student screening pipeline

Adopted by [Experiment 046](experiment-046-pipeline-v4-results.md) on 1 October 2026 (decision `adopt_v4`).
It is [pipeline v3](pipeline-v3.md) with one change: the binary readout is the unfitted mean of two logits,
v3's readout R3 and the attention head A on frozen ECG-JEPA tokens (Experiment 043 Stage 2's
`attention_jepa`, the ensemble E of Experiment 045). The finding heads, their standardization, the threshold
rule and the local-normal requirement are v2's and v3's, unchanged. This file lists everything a later
one-time final test (`final_frozen_test`) would freeze. It describes a candidate, not a validated screen:
every number behind it comes from development data (PTB-XL development, the Challenge calibration groups and
SPH, all read by earlier experiments), and 045 chose E partly on SPH. No final test has been run.

On SPH, v4 caught 0.012 more athlete-criteria abnormal ECGs than v3 at a 5% budget with 200 local normals
(+0.0121 [+0.0082, +0.0161]) and 0.020 more than v2, with about the same share of normals referred (5.82%
against 5.74%). It also referred more "other" ECGs such as sinus bradycardia (27.5% against 24.9% for v3 and
22.3% for v2). v2 and v3 stay documented in `pipeline-v2.md` and `pipeline-v3.md` so that a final test can
report all three.

## 1. Features

- Two frozen encoders on a 10 s, 12-lead, 500 Hz window in mV, canonical lead order, exactly as v3
  (`pipeline-v3.md` section 1): xECG (1,024 pooled features) and ECG-JEPA, both from the same checkpoints and
  input paths.
- ECG-JEPA is used twice: its pooled 768-dimensional `encoder.representation` (as in v3, for R3) and its 400
  output tokens before pooling (`lead_wave_maps.jepa_tokens`: the encoder's patch embedding, blocks and
  normalization on `external_encoders.jepa_input`; token 50 × lead + patch, leads I, II, V1-V6, 0.2 s
  patches). The pooled feature is the mean of the tokens (checked to 1e-4 on every cached ECG; largest
  difference 2e-6). Tokens are stored as float16 and widened to float32 for A.

## 2. Binary readout E

- R3: exactly v3's (`pipeline-v3.md` section 2), the 035 `dropped_upweighted` logistic fit on the 1,792
  concatenated xECG and JEPA features; its logit is the decision value.
- A: `ann_heads.AttentionHead(768)`: each token goes through LayerNorm, a linear layer to 128, GELU and
  dropout 0.1 (off at scoring); attention weights are a softmax over the 400 tokens of a 128-64-1 tanh
  scorer; each token has a linear logit; the ECG logit is the attention-weighted sum of the token logits plus
  a bias. About 108,000 parameters per network.
- A's training (043 Stage 2's recipe): v2's 39,577 training rows, labels and weights (`pipeline-v2.md`
  section 2), split by group with seed 43043 into 35,638 fit and 3,939 validation rows; AdamW, learning rate
  3e-4, weight decay 1e-2, batch 64, at most 30 epochs, patience 4, weighted binary cross-entropy, the epoch
  with the best validation AUROC kept; seeds 43043, 43044 and 43045 (best epochs 4, 5 and 7), each with
  `torch.manual_seed(seed)` before the network is built. A's logit is the mean of the three networks' logits.
- E = (R3 logit + A logit) / 2. No weight, scale or threshold is fitted. The pipeline ranks by expit(E); the
  thresholds are order statistics of the local normals, so this is ranking by E.

## 3. Finding heads and the combined finding score F (unchanged from v2)

The xECG PVC and WPW heads of 032, standardized on the 1,707 Challenge calibration clear normals, and
F = max(z_PVC, z_WPW), as in `pipeline-v3.md` section 3 (PVC mean −5.539442, SD 0.825770; WPW mean −9.857676,
SD 1.768115).

## 4. The referral rule and threshold (unchanged from v2)

`finding_screen.split_thresholds(normals, budget_per_mille, (50,))` with columns expit(E) then F, as in
`pipeline-v2.md` section 4: k = floor(b m); F's threshold is the (r + 1)-th highest normal F with
r = floor(50 k / 1000); E's threshold is the highest-ranked normal such that at most k − 1 of the m normals
lie above either threshold; an ECG is referred when E or F is strictly above its threshold; k ≥ 1 is required.
Budget choices 1%, 2%, 5% (reference) or 10%.

## 5. The local-normal requirement (unchanged from v2)

- The threshold must come from the site's own normal ECGs, read as normal by the cardiologist with the athlete
  criteria in mind.
- Size: with 200 normals the achieved rate at a 5% target ranges 3.4-8.9% across draws (5th-95th percentile);
  with 1,000 normals 4.3-6.7%, within ±1 point in 73% of draws (77% with v3, 88% with v2). On SPH at 5%, v4
  refers between 0.01 points fewer and 0.24 points more of the evaluation normals than v3, depending on the
  number of normals.

## 6. Expected operating numbers (SPH, development data)

Evaluation half of the SPH simulated site, 200 local normals, per 1,000 ECGs at an assumed 5% prevalence of
athlete-criteria abnormal ECGs:

| Budget | Referrals per 1,000 | Abnormal (binary label) caught | Athlete-criteria findings caught | Normals referred | "Other" ECGs referred |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1% | 47 | 0.62 | 0.66 | 1.45% | 0.12 |
| 2% | 61 | 0.70 | 0.72 | 2.57% | 0.17 |
| 5% | 96 | 0.79 | 0.81 | 5.82% | 0.27 |
| 10% | 146 | 0.85 | 0.87 | 10.78% | 0.39 |

PVC ECGs are referred 0.96-0.98 of the time and WPW 0.79 at 5% (15 ECGs). These are the numbers a final test
would check, not guarantees.

## 7. Inference cost

Per ECG on one RTX 3090, inputs in memory, after a warm-up (Experiment 046, 128 SPH ECGs):

| Step | Time per ECG |
| --- | ---: |
| xECG features, with preprocessing | 6.4 ms |
| ECG-JEPA tokens, with the Fourier resampling (the pooled JEPA feature is their mean) | 6.4 ms |
| The three attention heads together | 0.28 ms |
| R3, the finding heads and the rule (CPU) | negligible |

Reading the waveform is extra. v4 adds to v3 only the token output (no second ECG-JEPA pass, since the pooled
feature is the token mean) and the three heads, about 0.3 ms per ECG on a GPU; training them peaked at about 1 GiB
of GPU memory. The site needs both encoder checkpoints, the R3 and finding-head parameters and the three
attention networks' weights.

## 8. Code paths and hashes to reproduce

Frozen parameters (load these to score new ECGs; do not refit):

- R3 and the finding heads: `outputs/experiment044_pipeline_v3_v1/pipeline_v3_heads.npz`, SHA-256
  `54c5049aa0830b05a6c971c2a5234c18bddd136ad9c7592da0b4549b0786f8ce` (keys `v3_*`, `pvc_*`, `wpw_*` and the
  standardization constants; `pipeline-v3.md` section 7). Score with `ecg_experiment.pipeline_v2.score_parameters`;
  R3's logit is `logit(score)` or the linear part `(x − mean) / scale · coef + intercept`.
- A: `outputs/experiment046_pipeline_v4_v1/attention_jepa_weights.npz`, SHA-256
  `1af36b6832573f16b20c07785d3224cbb177ab8cb42ab9bc7a49cb04e7ac4fa0`, keys `seed{43043,43044,43045}.{name}`
  of each network's `state_dict`. Load with `ecg_experiment.pipeline_v4.load_state(AttentionHead(768), arrays,
  seed)` and score with `ann_heads.predict`; this reproduces the saved logits exactly. The retrained
  networks give logits bit-identical to those Experiment 043 saved (043 saved no weights).
- E: `ecg_experiment.pipeline_v4.ensemble_logit(R3 logit, A logit)`.
- Experiment 046 result: `outputs/experiment046_pipeline_v4_v1/result.json` `f92e7f2e…a90b`;
  `predictions.npz` `21d5af6870bd81d34c3d7e17d5da9ad9a1987b259c93d30125bbbae88ddec442`; `draws.csv`
  `3fe6dda24acc144e58be72d03054068816d4aea0bce6934f147f0b1e8be5bd2e`.

Code (hashes as recorded by the 046 run, commit `b0455d9`):

| Path | SHA-256 |
| --- | --- |
| `ecg_experiment/pipeline_v4.py` | `f936fcdfa863150102d601a870160f0e4ebc20c6c2f604b05092c78c536f13ed` |
| `ecg_experiment/pipeline_v3.py` | `dce65303c6c2e8d131664782ce94f66e839b4d01a83f4aaf71425336a182dcd8` |
| `ecg_experiment/pipeline_v2.py` | `1748c062efa9cb71630a50122d268c74e1ddaa87ec5d6e3f6bf52439dc3c8b53` |
| `ecg_experiment/finding_screen.py` | `2cb793ac7094e5fba4d3a239f154ff8e8bdb8ee780f63e6fc9dace3658687648` |
| `ecg_experiment/ann_heads.py` | `19a90a7f16228afec66372becb4030484346eef0455b3c24d54a9ee8748a29d5` |
| `ecg_experiment/lead_wave_maps.py` | `45d03882feb994536e5cda4161dff0e3eb429d056b1860e35616ccc9ed271c10` |
| `ecg_experiment/external_encoders.py` | `b21784379805e126fb5f0068c0cef53efb346bffe1ccd9f15f7fced9306a7c48` |
| `ecg_experiment/sph.py` | `44dd7808f5d8a6d82e6b6004a3731412e5376298493ff7c5a8f7bf7e1313536c` |
| `ecg_experiment/multisource_readout.py` | `6cf20dff174ace45e870fb4f65776c654f954fb96a82a7c739c70946b7887bc5` |
| `ecg_experiment/intervals.py` | `b45103664e8c2e063b90c8223b64e7bbe6394e54796e0613538bd0b53dfe99af` |
| `ecg_experiment/referral_budget.py` | `c2af973802e269c29c143b30e54147c05bba5513548149b8174269aec13ff56c` |
| `ecg_experiment/gpu.py` | `d827f44648d390651dba28d9f1aa5d0cf9c27611c60a09414848f6ff9b38f58d` |
| `scripts/experiments/run_pipeline_v4_046.py` | `554ad9cf6b264cfbe20b0da347002daf577bb768a444c01993edfd6ec934bd32` |
| `scripts/experiments/run_pipeline_v3_044.py` | `78a903b51d563627eacd73d9cf9f407b870bc08daaec40dc86cafcb3e626939e` |
| `scripts/experiments/run_ann_heads043.py` | `b5aa8fe03cbb110144ef1f5bffc25e2281ab12dece020f9020684716bda05c89` |
| `scripts/experiments/run_pipeline_v2_037.py` | `d05868abbd982b283bb5e06329a8ed62b1a19d25db5a18b8ddf7681fbc3a3bde` |
| `scripts/experiments/run_finding_screen033.py` | `0a15d541ab8dae3c554880f708bcc101f7d62bf457775941e3b3926d487ec879` |
| `pyproject.toml` | `9df6d56e8f27348d067eb0cf3b6506f5efbbc03154b96b6b3987f4bb4854279e` |
| `uv.lock` | `20c6e373cba70e7fe859ba59697f6145c34b42418eddc955f82d29a165d5db7c` |
| `docs/experiment-046-pipeline-v4.md` | `92d98a193a49c65f2ddb2218aecf88894277b01ad2d21966e3bae2097496ae59` |

The full list of source and input hashes is in the 046 `result.json` (`identity`). Torch 2.6.0+cu124, OpenBLAS
Haswell kernels.

Retrain from scratch (reproduces the saved weights and scores exactly on this machine; one GPU, about 11
minutes, of which about 3 read the SPH waveforms):

```bash
OPENBLAS_CORETYPE=Haswell OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 \
    CUDA_HOME=.venv/lib/python3.11/site-packages/nvidia/cuda_runtime \
    .venv/bin/python -u -m scripts.experiments.run_pipeline_v4_046
```

It needs Experiment 043's token cache on local disk (`exp043_cache/`, 38 GB of tokens, built by 043 Stage 2).

## 9. What a final test must still decide

Everything in `pipeline-v3.md` section 8, and in addition:

- Whether to report v2 and v3 beside v4. On SPH, v4 catches 0.003-0.013 more composite ECGs than v3 with
  `combined_50` at every budget and size; the interval includes 0 only at 1% with 500 or 2,000 normals.
- The ECG-JEPA token path (the encoder's output before pooling) and the three attention networks must run at
  the site, on a GPU or with a CPU timing check first.
- The "other" ECGs: v4 refers 27.5% of them at 5%, against 24.9% (v3) and 22.3% (v2). At a student site
  many would be normal variants (sinus bradycardia, sinus tachycardia, atrial premature beats). The
  cardiologist should see this trade before a final test fixes v4.

## 10. Explanation (added 1 October 2026; not part of the referral specification above)

The recommended explanation of a referred ECG is Experiment 048's PVC switch
([results](experiment-048-pvc-switch-results.md)). Mark the top unit of 042's beat-wave map `U_B` when the
xECG PVC head's z-score is above the 97.5th percentile of the normals' z-scores; otherwise mark the top token
of the attention head A's per-token contributions (`pvc_switch.switch_explain`,
`explanation_rule.rule_marks`). On PTB-XL development it kept every `U_B` premature-beat hit (90% of referred
PVC ECGs) and gained infarct-lead information over `U_B` (anterior contrast +0.189 [0.067, 0.307]). It sends
about half of the referred anterior infarcts to `U_B`. [Experiment 049](experiment-049-focal-switch-results.md)
found no better switch. With the `combined_50` referral, the PVC switch put the explanation on the premature
beat in 93% of the PVC ECGs. The thresholds were fitted on PTB-XL development normals, not local normals.
The CPU cost is in [inference-timing-cpu.md](inference-timing-cpu.md).
