# Experiment 046 results: pipeline v4, pipeline v3 with the ensemble readout

Completed 1 October 2026 under the [frozen protocol](experiment-046-pipeline-v4.md) (committed as `5170a37`;
the run recorded the same file hash, `92d98a19…ae59`). Run once at commit `b0455d9` in 671 s on one RTX 3090
(GPU 0, beside the vLLM server, under the shared GPU lock), at most four CPU threads, OpenBLAS Haswell
kernels. Local outputs are in `outputs/experiment046_pipeline_v4_v1/` (`result.json` SHA-256
`f92e7f2e…a90b`, `draws.csv`, `predictions.npz`, `attention_jepa_weights.npz`, `run.log`). A
training-rows-only smoke run (`outputs/experiment046_pipeline_v4_smoke/`, 167 s) preceded it. No PTB-XL
calibration or test ECG, no Challenge calibration or test-group ECG and no EchoNext record was read. No age or
other demographic subgroup analysis was done. The specification of the adopted pipeline is in
[pipeline-v4.md](pipeline-v4.md).

**SPH is development data.** Experiments 022-045 have read it, and 045 chose the ensemble partly on SPH AUROC.
This is a simulation of a new site, not a final test.

## Integrity

- Every input matched its receipt (037, 043 Stages 1 and 2, 044, 035, and the 030 and 032 outputs that 033's
  loaders check). The xECG rows, labels, weights and evaluation rows of 043's `training_data` equal 037's.
- The R3 refit (422 iterations) reproduced 044's saved v3 logits and probabilities exactly on SPH (25,577)
  and development (1,604), largest difference 0.0, and 043 Stage 1's `logistic_concat` to within 1.1e-14.
  Scoring 044's `pipeline_v3_heads.npz` reproduced the refit to 6.7e-16. 037's saved v2 scores equal 044's
  copy, and 033's z-scores equal 044's saved ones exactly.
- 043's token cache matched its receipt (row keys and both data hashes rehashed, 47 s).
- The 4,569 SPH evaluation ECGs without a binary label were read with the manifest hash check and encoded;
  every token mean matched 022's saved JEPA feature (largest difference 1.5e-6, tolerance 1e-4).
- **The retrained attention head A is bit-identical to 043's.** The protocol required r ≥ 0.999 and AUROCs
  within 0.001. The retrained 3-seed mean logits equal 043's saved ones exactly on the 21,008 labeled SPH ECGs
  and the 1,572 full-development ECGs (largest difference 0.0, r = 1.0, AUROC difference 0.0: 0.94039 SPH,
  0.93683 full development), and so does each seed. The best epochs are 043's (4, 5 and 7). Fresh networks
  loaded from `attention_jepa_weights.npz` reproduced every seed's development logits exactly (0.0).
- The v2 and v3 draws reproduced all 10,600 v2 and all 10,600 v3 rows of 044's `draws.csv` exactly, and the
  v2 rows reproduced 037's (largest difference 0.0). v3 − v2 reproduces 044's primary contrast exactly.
- v2's AUROCs equal 037's and v3's equal 043 Stage 1's. 033's local-pool selection still selects
  `combined_50` for v2 and v3.
- The matched-rate check reproduced every v2 and v3 number of 044's rate-matched result (largest
  difference 0.0).
- One of the 2,000 resamples held no high-grade AV block ECG and is left out of that outcome's interval, as in
  037 and 044.

Intervals are 95% paired whole-patient bootstrap intervals over the 12,320 evaluation patients (037's
resamples, seed 41041). Point values are means over 037's 200 local-normal draws. All rows use 033's
`combined_50` rule unless they say "binary alone". v2 and v3 are the frozen pipelines; v4 replaces only v3's
binary readout with E, the mean of the R3 logit and A's 3-seed mean logit.

## Primary: composite sensitivity at 5%, 200 local normals

| | v2 | v3 | v4 | v4 − v3 [95% CI] | v4 − v2 [95% CI] |
| --- | ---: | ---: | ---: | --- | --- |
| Athlete-criteria composite (4,052) | 0.789 | 0.797 | 0.809 | **+0.0121 [+0.0082, +0.0161]** | +0.0201 [+0.0152, +0.0258] |
| Binary label (3,584) | 0.768 | 0.777 | 0.791 | +0.0138 [+0.0094, +0.0183] | +0.0230 [+0.0173, +0.0293] |
| Normals referred (6,895) | 5.45% | 5.74% | 5.82% | +0.08 pp [−0.17, +0.32] | +0.36 pp [+0.05, +0.68] |
| **"Other" ECGs referred (1,812)** | **0.223** | **0.249** | **0.275** | **+0.026 [+0.017, +0.035]** | **+0.052 [+0.040, +0.063]** |
| PVC (531) | 0.967 | 0.966 | 0.965 | −0.0015 [−0.0040, +0.0003] | −0.0018 [−0.0062, +0.0016] |
| WPW (15) | 0.816 | 0.765 | 0.794 | +0.029 [0.000, +0.096] | −0.022 [−0.073, 0.000] |
| Composite without a binary label (468) | 0.955 | 0.954 | 0.953 | −0.0009 [−0.0052, +0.0033] | −0.0023 [−0.0091, +0.0043] |
| MI (122) | 0.919 | 0.921 | 0.917 | −0.004 [−0.028, +0.019] | −0.002 [−0.030, +0.026] |
| STTC (2,507) | 0.820 | 0.827 | 0.840 | +0.0133 [+0.0086, +0.0183] | +0.0203 [+0.0140, +0.0267] |
| CD (1,184) | 0.688 | 0.700 | 0.713 | +0.0130 [+0.0048, +0.0222] | +0.0248 [+0.0135, +0.0369] |
| HYP (117) | 0.885 | 0.914 | 0.925 | +0.011 [−0.005, +0.033] | +0.040 [+0.013, +0.074] |
| AF/flutter (370) | 0.989 | 0.988 | 0.989 | +0.0008 [−0.0033, +0.0055] | +0.0001 [−0.0069, +0.0069] |
| Long QT (10) | 1.000 | 0.998 | 0.988 | −0.010 [−0.033, 0.000] | −0.013 [−0.042, 0.000] |

**Decision: `adopt_v4`.** The pre-registered rule adopts v4 if the lower bound of v4 − v3 is above 0; it is
+0.0082. v4 catches about 12 more of every 1,000 athlete-criteria abnormal ECGs than v3 (20 more than v2), and
it is ahead of v3 in 77.5% of the 200 local-normal draws (behind in 22%). v4 − v2 is read the same way
(`adopt_v4` against v2 too).

**The cost for students: "other" ECGs.** v4 refers 27.5% of the "other" ECGs (sinus bradycardia or
tachycardia, atrial premature beats and similar, which SPH counts neither as caught nor as false referrals),
against 24.9% for v3 and 22.3% for v2. Each pipeline step since v2 has added about 2.6 points. At a student
site many of these would be normal variants, so a part of the gain will be paid in referrals of healthy
students that the normal-only false-referral rate does not show. The false-referral rate on true normals
barely moves (5.82% against 5.74%, interval including 0).

Per 1,000 ECGs at an assumed 5% prevalence, v4 gives 95.7 referrals against v3's 94.4 and v2's 91.3, and
catches 40.5 of the 50 athlete-criteria abnormal ECGs (39.9-41.0) against 39.9 (39.3-40.5) and 39.5.

## By budget, 200 and 1,000 local normals

| Budget | m | v3 composite | v4 composite | v4 − v3 composite | v3 binary | v4 binary | v3 rate | v4 rate | v3 "other" | v4 "other" | v4 − v3 "other" |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1% | 200 | 0.650 | 0.655 | +0.005 [+0.002, +0.009] | 0.616 | 0.620 | 1.56% | 1.45% | 0.103 | 0.119 | +0.016 [+0.010, +0.022] |
| 2% | 200 | 0.716 | 0.723 | +0.007 [+0.004, +0.011] | 0.688 | 0.695 | 2.55% | 2.57% | 0.150 | 0.170 | +0.020 [+0.013, +0.028] |
| 5% | 200 | 0.797 | 0.809 | +0.012 [+0.008, +0.016] | 0.777 | 0.791 | 5.74% | 5.82% | 0.249 | 0.275 | +0.026 [+0.017, +0.035] |
| 10% | 200 | 0.859 | 0.869 | +0.010 [+0.006, +0.014] | 0.843 | 0.855 | 10.49% | 10.78% | 0.371 | 0.391 | +0.020 [+0.010, +0.029] |
| 1% | 1,000 | 0.646 | 0.652 | +0.006 [+0.002, +0.011] | 0.618 | 0.624 | 1.20% | 1.08% | 0.087 | 0.106 | +0.019 [+0.011, +0.026] |
| 2% | 1,000 | 0.715 | 0.727 | +0.011 [+0.007, +0.016] | 0.686 | 0.699 | 2.09% | 2.21% | 0.140 | 0.163 | +0.024 [+0.015, +0.032] |
| 5% | 1,000 | 0.798 | 0.810 | +0.012 [+0.008, +0.017] | 0.776 | 0.790 | 5.46% | 5.46% | 0.243 | 0.269 | +0.027 [+0.016, +0.038] |
| 10% | 1,000 | 0.861 | 0.870 | +0.009 [+0.004, +0.013] | 0.845 | 0.855 | 10.29% | 10.56% | 0.372 | 0.388 | +0.017 [+0.005, +0.027] |

At 5% the v4 − v3 composite difference is +0.008 to +0.013 at every m from 50 to 2,000, and every lower bound
is above 0 (smallest +0.0052, at m = 50). Of the 23 `combined_50` screens, the interval includes 0 only at 1%
with 500 normals (+0.0025 [−0.0015, +0.0069]) and 1% with 2,000 (+0.0043 [−0.0006, +0.0094]). v4 − v3 in
the false-referral rate is between −0.18 and +0.28 points; its interval includes 0 except at 1% with 200, 500
and 2,000 normals (v4 lower) and at 2% with 50 and 5% with 100 normals (v4 higher). The "other" share is
higher with v4 in every one of the 23 screens, by +0.014 to +0.028, every interval above 0.

Achieved rates across draws at 5% (5th-95th percentile): with 200 normals 3.4-8.9% for v4 (3.5-8.5% for v3);
with 1,000 normals 4.3-6.7% (4.4-6.6%), within ±1 point in 73% of draws against 77%.

## The binary readout alone

At 5% with 200 normals, E alone catches 0.768 of the composite ECGs against R3's 0.758 (+0.011 [+0.007,
+0.015]) and 0.793 of the binary-positive ECGs against 0.780 (+0.013 [+0.009, +0.018]), at a rate of 5.95%
against 5.89%. It refers fewer PVC ECGs than R3 (0.613 against 0.625, −0.011 [−0.024, +0.002]) and more
"other" ECGs (0.274 against 0.250, +0.024 [+0.014, +0.033]). The PVC head still carries the PVC ECGs in the
full pipeline (0.965).

033's finding rule adds about the same with E as with R3: +0.041 [+0.035, +0.048] composite at 5% with 200
normals (v3 +0.040), for a binary-label cost of 0.003 (v3 0.003). The local-pool selection with v4 still
selects `combined_50` (local composite 0.801, local binary cost 0.0032).

## Matched false-referral rate (prespecified, descriptive)

Thresholds set on the 6,895 evaluation normals themselves (an oracle no site has), re-estimated in each
resample, so every pipeline refers 4.97% of them at a 5% budget:

| Screen at a matched 4.97% | v2 | v3 | v4 | v4 − v3 | v4 − v2 |
| --- | ---: | ---: | ---: | --- | --- |
| `combined_50`, composite | 0.785 | 0.792 | 0.805 | +0.0126 [+0.0039, +0.0213] | +0.0200 [+0.0084, +0.0287] |
| `combined_50`, binary label | 0.761 | 0.770 | 0.784 | +0.0142 [+0.0044, +0.0238] | +0.0232 [+0.0100, +0.0332] |
| `combined_50`, "other" referred | 0.208 | 0.231 | 0.262 | +0.031 [+0.015, +0.044] | +0.054 [+0.033, +0.068] |
| binary alone, composite | 0.734 | 0.742 | 0.756 | +0.0143 [+0.0038, +0.0234] | +0.0227 [+0.0110, +0.0351] |

At a matched rate the gain is the same as in the primary comparison (+0.0126 against +0.0121), and unlike
v3 − v2 in 044 its interval stays above 0. The extra "other" referrals are not a rate effect either. With the
thresholds set on the whole local pool (6,923 normals), v4 refers 5.47% of the evaluation normals against
v3's 5.38% and catches 0.812 of the composite ECGs against 0.797. At the other matched budgets v4 − v3 is
+0.018 [+0.002, +0.035] (1%), +0.003 [−0.008, +0.020] (2%) and +0.007 [−0.001, +0.013] (10%).

## AUROC (for completeness)

| Set | ECGs (positive) | v2 | v3 | A | v4 (E) | v4 − v3 [95% CI] | v4 − v2 [95% CI] |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| PTB-XL hard added subset | 266 (41) | 0.712 | 0.746 | 0.810 | 0.795 | +0.049 [+0.018, +0.084] | +0.083 [+0.039, +0.132] |
| PTB-XL ordinary development | 1,306 (843) | 0.9536 | 0.9566 | 0.9548 | 0.9595 | +0.0029 [0.0000, +0.0058] | +0.0060 [+0.0019, +0.0102] |
| PTB-XL full development | 1,572 (884) | 0.9306 | 0.9347 | 0.9368 | 0.9405 | +0.0058 [+0.0027, +0.0094] | +0.0099 [+0.0052, +0.0149] |
| SPH | 21,008 (7,190) | 0.9387 | 0.9404 | 0.9404 | 0.9442 | +0.0038 [+0.0028, +0.0049] | +0.0055 [+0.0041, +0.0069] |

Because A is bit-identical to 043's, E's AUROCs equal 045's on all four sets; only the intervals differ
(seed 41041 here, 45045 in 045).

## Cost

- GPU profile before any extraction or training: 0.066 s per record to read and encode (128 records),
  0.0135 s per optimizer step; projected 1,147 s for the whole run against the 2-hour ceiling. Actual: 197 s
  to read and encode the 4,569 new SPH ECGs (140 s of it waiting on reads from `/home`), 271 s to train and
  score the three seeds (80-103 s each, 9-12 epochs), 1.0 GiB peak GPU memory, 671 s in all.
- Inference per ECG on the RTX 3090, timed on 128 newly read SPH ECGs held in memory: ECG-JEPA tokens
  6.4 ms (with the Fourier resampling), xECG features 6.4 ms (with its preprocessing), the three attention
  heads 0.28 ms together; the logistic readouts and finding heads are negligible. Reading the waveform is
  extra. The xECG features computed here match 022's saved ones to 2.1e-5 and the token means 022's JEPA
  features to 1.1e-6.

## Numbers for the cardiologist meeting

Pipeline v4 (`combined_50`), threshold set on 200 local normal ECGs, assuming 5% of students have an
athlete-criteria abnormal ECG:

| Budget (share of normals flagged) | Referrals per 1,000 | Share of abnormal caught | Share of athlete-criteria findings caught | Normals referred | "Other" ECGs referred |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1% | 47 | 0.62 | 0.66 | 1.45% | 0.12 |
| 2% | 61 | 0.70 | 0.72 | 2.57% | 0.17 |
| 5% | 96 | 0.79 | 0.81 | 5.82% | 0.27 |
| 10% | 146 | 0.85 | 0.87 | 10.78% | 0.39 |

At 5% that is about 96 referrals per 1,000 students and 40-41 of the 50 abnormal ones caught, against 94 and
40 with v3.

## What it means, in plain language

- Averaging the attention head on ECG-JEPA tokens with v3's readout catches more abnormal ECGs at every
  budget: about 12 more of every 1,000 at 5%, mostly ST-T and conduction findings, as many PVC ECGs. The gain
  passes the pre-registered rule, so v4 becomes the candidate pipeline.
- Unlike v3 over v2, v4 does not refer clearly more true normals at the same budget, and the gain holds when
  the false-referral rate is held equal.
- It does refer more "other" ECGs: 27.5% against 24.9% (v3) and 22.3% (v2). Sinus bradycardia, sinus
  tachycardia and atrial premature beats are common in healthy students, so the cardiologist should see this
  before a final test fixes v4. v4 and v3 together now refer about 5 points more of these ECGs than v2.
- v4 needs the ECG-JEPA token output and three small attention networks at the site, on top of v3's two
  encoders. On a GPU that adds well under a millisecond per ECG; the encoders dominate.

## Surprises

- The retrained attention head was bit-identical to 043's on the GPU, so the protocol's tolerance was not
  needed.
- The SPH AUROC gain (+0.0038) turned into +0.012 at the 5% budget, a larger ratio than v3's (+0.0017 to
  +0.008).
- WPW moved up (0.794 against 0.765) and long QT down (0.988 against 0.998); 15 and 10 ECGs, intervals reach
  0.
- With 200 normals v4's achieved rate stays within ±1 point of 5% in 46% of draws against 52.5% for v3.

## Caveats

- SPH is development data (022-045), an older Chinese hospital cohort at 34% prevalence, whose normals are
  hospital normals with sinus rhythm alone. 045 chose E partly on SPH AUROC, so this is a third look at the
  same data, not an independent confirmation.
- The local pool and evaluation half come from the same hospital, the most favourable case.
- WPW (15), high-grade AV block (12) and long QT (10) have few evaluation ECGs; MI and HYP have 122 and 117.
- The labels are ECG annotation proxies, not confirmed disease or referral decisions.
- One fit per logistic head and one recipe with three seeds for A; only the local normals and the evaluation
  patients vary. A saw 90% of the training rows.
- The matched-rate check uses oracle thresholds; it describes, it does not test.
- The inference times are from one GPU with the inputs in memory, after a warm-up; they are not a deployment
  benchmark.

## Deviations

None. The 4,569 SPH ECGs without a binary label were scored as prespecified; their tokens are in the scratch
cache `exp046_cache/` (`tokens.npy` data SHA-256 `b0c28495…7b7e5`), outside the repository.

## Reproduce

```bash
OPENBLAS_CORETYPE=Haswell OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 \
    CUDA_HOME=.venv/lib/python3.11/site-packages/nvidia/cuda_runtime \
    .venv/bin/python -u -m scripts.experiments.run_pipeline_v4_046
```

The runner refuses to overwrite an existing output. It needs the 030, 032, 033, 035, 037, 043 (Stages 1 and 2),
044 and 045 outputs, 043's token cache on local disk, the JEPA and xECG caches and checkpoints, the SPH
manifest and the SPH waveforms of the 4,569 ECGs without a binary label (it reuses `exp046_cache/` when its
receipt holds).
