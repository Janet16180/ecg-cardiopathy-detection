# Experiment 050 results: attention finding heads on frozen ECG-JEPA tokens

Completed 1 October 2026 under the [frozen protocol](experiment-050-attention-finding-heads.md) (committed as
`ae1c879`; the run recorded the same file hash, `79ec5b86…eeae`). Run once at commit `e65735f` in 1,172 s on
one RTX 3090 (GPU 0, under the shared GPU lock), at most four CPU threads. Local outputs are in
`outputs/experiment050_attention_findings_v1/` (`result.json` SHA-256 `0ed107ff…3a55`, `predictions.npz`,
`token_maps.npz`, `attention_finding_weights.npz`, `run.log`). A training-rows-only smoke run
(`outputs/experiment050_attention_findings_smoke/`, 80 s) preceded it. No PTB-XL calibration or test ECG, no
Challenge calibration or test-group ECG and no EchoNext record was read. Besides tokens, the run read the
waveforms of the 15,434 new training rows and the 84 PVC records' PTB-XL waveforms (041's windows).

## Readings

| Question | Prespecified test | Result | Reading |
| --- | --- | --- | --- |
| (a) PVC detection, SPH (primary) | attention − xECG AUROC, lower bound > 0 beats, > −0.005 matches | −0.0068 [−0.0128, −0.0015] | **Below** |
| (b) PVC map localization | hit − chance minus `U_B`'s, lower bound > −0.10 keeps | −0.045 [−0.148, +0.054] | **Does not keep** |

Because (b) does not keep premature-beat localization, the secondary explanation rule was not run, as
prespecified.

## Integrity

- Every input matched its receipt. 043's training rows equal 037's. The rebuilt stacked table equals 043's on
  the 39,577 binary rows, and its groups equal 043's.
- 043's and 046's token caches matched their recorded receipts (rehashed, 49 s). The 15,434 new training rows
  were read and encoded (201 s); every token mean matched its cached JEPA feature (largest difference 1.3e-6,
  tolerance 1e-4).
- The xECG PVC and WPW heads reproduced 044's saved SPH z-scores (largest difference 4.3e-15 and 1.3e-15) and
  044's SPH AUROCs exactly.
- `U_B` reproduced 042, and `U_B` and `attention_jepa` reproduced 043 Stage 2's `maps` block exactly.
- Fresh networks loaded from `attention_finding_weights.npz` reproduced every seed's development logits
  exactly (0.0).

## Training

| Head | Fit rows (positive) | Validation rows (positive) | Best epochs (50050, 50051, 50052) | Validation AUROC |
| --- | --- | --- | --- | --- |
| `attention_pvc` | 49,418 (2,194) | 5,492 (243) | 3, 3, 6 | 0.975, 0.975, 0.975 |
| `attention_wpw` | 46,130 (114) | 5,099 (10) | 3, 6, 3 | 0.894, 0.883, 0.895 |

Profile before any extraction or training: 0.032 s per record to read and encode, 0.018 s per optimizer step;
projected 3,376 s against the 2-hour ceiling. Actual: 201 s extraction, 390 s (PVC) and 423 s (WPW) training
and scoring, 1,172 s in all, at most 1.0 GiB of GPU memory.

## (a) Detection

Paired whole-patient bootstrap, 2,000 draws, seed 50050. The xECG heads are pipeline v2-v4's.

| Head and set | ECGs (positive) | Attention AUROC | xECG AUROC | Attention − xECG [95% CI] | Reading | Attention AP | xECG AP |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: |
| PVC, SPH (primary) | 25,577 (1,058) | 0.9830 | 0.9898 | −0.0068 [−0.0128, −0.0015] | below | 0.957 | 0.945 |
| PVC, full development | 1,600 (84) | 0.9959 | 0.9941 | +0.0018 [−0.0008, +0.0051] | matches | 0.929 | 0.919 |
| WPW, SPH | 25,566 (27) | 0.9668 | 0.9916 | −0.0248 [−0.0652, +0.0012] | below | 0.417 | 0.565 |
| WPW, full development | 1,604 (6) | 0.878 | 0.939 | −0.061 [−0.177, 0.000] | below | 0.614 | 0.678 |

Seed AUROCs: PVC at SPH 0.9831, 0.9827 and 0.9825 (all below the xECG head); WPW at SPH 0.983, 0.906 and
0.976, so one seed pulls the mean down.

## (b) The PVC head's token map

042's tests on the 1,604 full-development ECGs, seed 50050 (the `U_B` and `attention_jepa` rows equal 042's
and 043's point values):

| Map | Hit (73 ECGs) | Chance | Hit − chance [95% CI] | Anterior contrast | Inferior contrast | Worst-unit AUROC (binary label) |
| --- | ---: | ---: | --- | --- | --- | ---: |
| `U_B` (042) | 0.932 | 0.197 | +0.734 [0.671, 0.787] | +0.085 [−0.042, +0.208] | +0.115 [+0.004, +0.230] | 0.740 |
| `attention_jepa` (043, binary label) | 0.027 | 0.147 | −0.119 [−0.155, −0.073] | +0.190 [0.078, 0.304] | +0.205 [0.112, 0.299] | 0.897 |
| `attention_pvc` (this experiment) | 0.836 | 0.147 | +0.689 [0.595, 0.775] | +0.112 [0.023, 0.207] | +0.086 [0.024, 0.154] | 0.681 |

| Paired contrast, `attention_pvc` minus | Hit − chance | Benign any red | Worst-unit AUROC |
| --- | --- | --- | --- |
| `U_B` | −0.045 [−0.148, +0.054] | 0.000 [−0.115, +0.096] | −0.059 [−0.097, −0.019] |
| `attention_jepa` | +0.808 [0.716, 0.895] | −0.038 [−0.154, +0.077] | −0.216 [−0.251, −0.184] |

**Reading: does not keep.** The point difference from `U_B` is small (−0.045), but the lower bound
(−0.148) is below the −0.10 margin. By 042's full rule, `attention_pvc` does not improve on `U_B`: it gains
the lead contrast (anterior lower bound above 0) but does not keep localization. Against `attention_jepa` it
gains +0.81 in hit − chance.

At its own 95th-percentile threshold (−0.0005: normal ECGs' contributions are near zero), the PVC map is red
on 99% of PVC ECGs but also marks many units: 291 red tokens per PVC ECG on average, 30 per positive and 1.3
per normal.

**Example ECG 219 (PVC).** Premature-beat window 8.258-8.758 s. `attention_pvc`'s top token is V5 at
8.4-8.6 s, on the premature beat. `U_B`'s top unit is aVF at 8.298-8.438 s, also on it. `attention_jepa`'s
top token is V2 at 5.2-5.4 s, not on it.

## What it means, in plain language

- Training the attention head on the PVC label makes its map point at premature beats: its top token falls on
  the premature beat in 84% of PVC ECGs, against 3% for the binary-label head. That is close to `U_B` (93%),
  but with 73 ECGs the interval cannot rule out a loss of more than the prespecified 0.10, so the rule says
  it does not keep localization.
- As a detector it does not replace the xECG PVC head. At SPH its AUROC is 0.983 against 0.990 (below), and
  on PTB-XL development it matches. Its average precision at SPH is higher (0.957 against 0.945), so it ranks
  the top PVC ECGs well but misses more of the harder ones.
- The WPW head is worse than the xECG head (27 SPH positives, 124 training positives; one seed is weak).
- Pipelines v3 and v4 keep their xECG finding heads; 048's explanation rule keeps `U_B` as its rhythm layer.

## Caveats

- Development data, read by earlier experiments; the results are exploratory.
- The networks saw 90% of their rows; the xECG heads saw all of them.
- 73 premature-beat ECGs (one moves the hit rate by 1.4 points), 27 SPH and 6 development WPW ECGs.
- The PVC label is ECG-level; the premature-beat windows come from an automatic rule.
- The contributions are the networks' own decomposition, not a validated explanation.
- Three seeds of one recipe, no hyperparameter search.

## Deviations

None. The secondary explanation rule was not run because its prespecified condition, (b) keeping
localization, did not hold. The new tokens are in the scratch cache `exp050_cache/` (`tokens.npy` data
SHA-256 `e07d0051…d551`), outside the repository.

## Follow-up ideas

- An unfitted mean of the xECG PVC z-score and the attention PVC logit (both standardized on the same
  normals) as the PVC finding score: the attention head has higher average precision at SPH, so the two may
  be complementary, as in 045 for the binary readout. It needs a protocol and a pipeline comparison like 046.
- Repeat the map test on more premature-beat ECGs (for example the SPH PVC ECGs with 041's rule) to narrow the
  interval; on 73 ECGs the point difference from `U_B` was only −0.045.
- A two-colour explanation for switch-on ECGs (048's follow-up) using `attention_pvc` for rhythm and
  `attention_jepa` for morphology, both from the same tokens, so the site needs no beat-wave pipeline. Its
  localization would still have to pass the margin above.
- A WPW head needs more positives (124 in training) before it is worth retraining.

## Reproduce

```bash
OPENBLAS_CORETYPE=Haswell OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 \
    .venv/bin/python -u -m scripts.experiments.run_attention_findings050
```

The runner refuses to overwrite an existing output. It needs 043's and 046's token caches on local disk, the
032, 035, 037, 042, 043, 044, 046 and 048 outputs, and the PTB-XL and Challenge training waveforms (it reuses
`exp050_cache/` when its receipt holds).
