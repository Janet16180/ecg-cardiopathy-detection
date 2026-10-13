# Experiment 063 results: masked cross-lead and cross-beat reconstruction

Run on 12 October 2026 under the [protocol](experiment-063-masked-reconstruction.md) (commit eabadb8) and its
amendments 1-3 (commits 310f1ec, 75c034d, f789186, all before any score) and amendment 4 (commit 77b7d82, after
the v1 scores and before v2 was scored). Every number comes from `outputs/experiment063_inpaint_v1/result.json`
(LocBench summaries in `locbench/<variant>/summary.json`, strata and noise readings in
`locbench/<variant>/tables/`). Intervals are 95% whole-patient bootstraps, 2,000 draws, seed 61061. No draw was
skipped.

## Decision

- v1, anomaly-only `primary` (mean z over the three mask families): negative. A2 0.6017 [0.5767, 0.6270]
  against `qrs_prior` 0.7419; paired difference -0.1402 [-0.1716, -0.1095].
- v2, `raw_qrs_combined` (061 QRS indicator plus the unnormalized error evidence, declared in amendment 4 after
  the v1 scores): inconclusive. A2 0.7317 [0.6945, 0.7668]; paired difference -0.0103 [-0.0221, +0.0015].
  It was chosen after seeing development scores, so this reading is optimistic.

Neither version beats the bar. Neither is carried forward as a localization candidate on A2. The v2 map, and the
`raw` error as one component of a combined map, are reasonable inputs for Experiment 066. Their noise and
detection readings are better than every 061 baseline (below).

## Main readings

| Map | A2 | B1 macro | C1 | C1_std | D1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `primary` (v1) | 0.6017 [0.5767, 0.6270] | 0.5707 [0.5623, 0.5794] | 0.7469 [0.7206, 0.7730] | 0.7016 [0.6723, 0.7315] | 0.7464 [0.7299, 0.7628] |
| `phase` | 0.6167 [0.5918, 0.6416] | 0.5896 [0.5810, 0.5989] | 0.7468 [0.7212, 0.7730] | 0.7113 [0.6836, 0.7399] | 0.7372 [0.7209, 0.7539] |
| `block` | 0.5961 [0.5721, 0.6201] | 0.5666 [0.5583, 0.5754] | 0.7629 [0.7366, 0.7888] | 0.7056 [0.6770, 0.7344] | 0.7425 [0.7263, 0.7585] |
| `whole` | 0.6019 [0.5771, 0.6273] | 0.5666 [0.5581, 0.5753] | 0.7428 [0.7152, 0.7713] | 0.6788 [0.6486, 0.7099] | 0.7496 [0.7336, 0.7654] |
| `raw` (error in mV) | 0.7091 [0.6758, 0.7411] | 0.6655 [0.6578, 0.6733] | 0.5636 [0.5372, 0.5915] | 0.5918 [0.5621, 0.6228] | 0.7805 [0.7657, 0.7943] |
| `qrs_combined` | 0.7028 [0.6664, 0.7389] | 0.7002 [0.6933, 0.7073] | 0.6297 [0.6006, 0.6608] | 0.5984 [0.5692, 0.6291] | 0.7658 [0.7500, 0.7804] |
| `raw_qrs_combined` (v2) | 0.7317 [0.6945, 0.7668] | 0.7145 [0.7076, 0.7216] | 0.5312 [0.5041, 0.5594] | 0.5371 [0.5079, 0.5676] | 0.7801 [0.7651, 0.7939] |
| 061 `qrs_prior` (bar) | 0.7419 [0.7082, 0.7743] | 0.7283 [0.7212, 0.7356] | 0.3529 [0.3248, 0.3843] | 0.560 | 0.6321 [0.6135, 0.6501] |
| 061 `residual056` | 0.6796 [0.6504, 0.7096] | 0.6368 | 0.8005 | 0.707 [0.680, 0.733] | 0.5934 |
| 061 `U_B` | 0.6315 [0.5968, 0.6644] | 0.6676 | 0.6732 | 0.720 [0.692, 0.746] | 0.7621 |

The 061 rows without intervals are point estimates derived from the 061 results report (bar plus paired
difference) or quoted from 061 amendment 3.

## Paired differences on ECGAD development

Paired against `qrs_prior` and `residual056` on the same draws (this map minus the baseline):

| Map | Metric | minus `qrs_prior` | minus `residual056` |
| --- | --- | ---: | ---: |
| `primary` | A2 | -0.1402 [-0.1716, -0.1095] | -0.0779 [-0.1138, -0.0428] |
| `primary` | A1 | -0.0734 [-0.1059, -0.0406] | -0.0440 [-0.0850, -0.0011] |
| `primary` | A3 (lead) | +0.0975 [+0.0148, +0.1790] | -0.0369 [-0.1263, +0.0528] |
| `primary` | A5 (record) | +0.0839 [+0.0027, +0.1652] | +0.1488 [+0.0720, +0.2269] |
| `raw` | A2 | -0.0328 [-0.0487, -0.0172] | +0.0295 [-0.0026, +0.0614] |
| `raw` | A1 | +0.0326 [+0.0147, +0.0503] | +0.0619 [+0.0186, +0.1081] |
| `raw_qrs_combined` | A2 | -0.0103 [-0.0221, +0.0015] | +0.0520 [+0.0198, +0.0854] |
| `raw_qrs_combined` | A1 | +0.0239 [+0.0108, +0.0373] | +0.0532 [+0.0138, +0.0956] |
| `raw_qrs_combined` | A3 (lead) | +0.1057 [+0.0272, +0.1780] | -0.0287 [-0.1232, +0.0678] |
| `raw_qrs_combined` | A4 (time) | -0.0383 [-0.0534, -0.0249] | +0.0978 [+0.0620, +0.1310] |
| `raw_qrs_combined` | A5 (record) | +0.1360 [+0.0735, +0.2013] | +0.2010 [+0.1142, +0.2923] |
| `raw_qrs_combined` | A6 (top 5%) | +0.0361 [+0.0137, +0.0584] | +0.0540 [-0.0000, +0.1067] |

Other ECGAD readings of `raw_qrs_combined`: A3 0.5622 [0.5022, 0.6209], A4 0.8123 [0.7761, 0.8441], A5 0.8488
[0.7879, 0.9034], A6 0.3933 [0.3338, 0.4578] against a chance level of 0.1861.

## QRS strata (061 amendment 2)

| Map | A2_qrs | minus `qrs_prior` | minus `residual056` | A2_nonqrs | minus `qrs_prior` | minus `residual056` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `primary` | 0.5567 [0.5213, 0.5896] | +0.0180 [-0.0171, +0.0529] | -0.0500 [-0.1043, +0.0023] | 0.5899 [0.5583, 0.6223] | -0.0713 [-0.1021, -0.0395] | -0.0813 [-0.1254, -0.0394] |
| `phase` | 0.5433 [0.5105, 0.5748] | +0.0046 [-0.0298, +0.0372] | -0.0634 [-0.1187, -0.0101] | 0.6094 [0.5796, 0.6397] | -0.0517 [-0.0814, -0.0214] | -0.0617 [-0.1027, -0.0230] |
| `block` | 0.5541 [0.5216, 0.5847] | +0.0154 [-0.0186, +0.0483] | -0.0526 [-0.1067, -0.0012] | 0.5853 [0.5536, 0.6170] | -0.0758 [-0.1060, -0.0437] | -0.0858 [-0.1278, -0.0457] |
| `whole` | 0.5632 [0.5270, 0.5969] | +0.0246 [-0.0124, +0.0604] | -0.0434 [-0.0961, +0.0089] | 0.5886 [0.5550, 0.6224] | -0.0726 [-0.1046, -0.0408] | -0.0826 [-0.1298, -0.0369] |
| `raw` | 0.5968 [0.5644, 0.6266] | +0.0582 [+0.0329, +0.0830] | -0.0098 [-0.0604, +0.0384] | 0.6435 [0.6081, 0.6783] | -0.0176 [-0.0455, +0.0115] | -0.0276 [-0.0730, +0.0144] |

89 records enter A2_qrs and 92 enter A2_nonqrs. `qrs_combined` and `raw_qrs_combined` have the strata of `primary` and
`raw` respectively, because inside each stratum they order samples by a monotone function of that evidence.
Reference strata: `qrs_prior` 0.539 / 0.661 (its amplitude tiebreak), `residual056` 0.607 / 0.671.

## Region set (evaluation partition)

B1 per family, point estimates (macro and intervals above):

| Map | 1AVB | CLBBB | CRBBB | IMI | LAFB | LNGQT | ST injury | anterior MI | ischemia |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `primary` | 0.490 | 0.608 | 0.599 | 0.716 | 0.719 | 0.512 | 0.457 | 0.514 | 0.522 |
| `raw` | 0.402 | 0.797 | 0.828 | 0.855 | 0.832 | 0.528 | 0.455 | 0.770 | 0.523 |
| `raw_qrs_combined` | 0.375 | 0.971 | 0.922 | 0.929 | 0.921 | 0.480 | 0.454 | 0.897 | 0.482 |

B2, lead AUROC (expected leads):

| Map | CRBBB | IMI | LAFB | ST injury | anterior MI | ischemia | macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `primary` | 0.593 [0.524, 0.653] | 0.764 [0.734, 0.793] | 0.732 [0.701, 0.762] | 0.518 [0.421, 0.605] | 0.389 [0.358, 0.421] | 0.509 [0.452, 0.565] | 0.584 [0.562, 0.606] |
| `raw` | 0.612 [0.565, 0.657] | 0.754 [0.724, 0.781] | 0.642 [0.614, 0.668] | 0.578 [0.488, 0.670] | 0.587 [0.556, 0.617] | 0.476 [0.427, 0.525] | 0.608 [0.588, 0.629] |
| `raw_qrs_combined` | 0.616 [0.569, 0.660] | 0.773 [0.742, 0.800] | 0.674 [0.647, 0.700] | 0.469 [0.381, 0.567] | 0.575 [0.545, 0.603] | 0.474 [0.426, 0.523] | 0.597 [0.576, 0.618] |

`raw_qrs_combined` B1 macro is below the bar (-0.0138 [-0.0209, -0.0071]), the strict macro matches it (+0.0018
[-0.0033, +0.0070]), and B2 macro is +0.0209 [-0.0002, +0.0443] above it. The error maps do carry lead
information for inferior MI (B2 IMI 0.75-0.77, as residual056's 0.80) and LAFB. They carry none for ST and T
findings (ST injury, ischemia, long QT and 1AVB all near or below 0.5).

## Training, calibration and runtime

- Model: 6.35 M parameters. Training set: 2,400 `fit` clean normals; 266 records from 257 held-out patients used
  for validation.
- GPU profile (RTX 3090): 0.048 s per update at batch 64, peak 1.69 GB, 0.051 s per record for the 140-mask
  inference.
- Training: early stop at update 16,500 of a 60,000 maximum; best validation Gaussian NLL -3.0225 at update
  12,500; 14.4 minutes. Validation NLL went from -2.55 (update 500) to -3.02, and validation MSE on hidden samples
  from 0.0061 to 0.0032 mV^2. One transient spike at update 6,000 (NLL 160) recovered by the next validation.
  Training stopped while the cosine learning rate was still high, so the model is probably undertrained.
- Calibration on the held-out normals under the inference masks: median z 0.48-0.50 (Gaussian 0.67), share of
  z above 2 2.1-2.3% (Gaussian 4.6%), above 3 0.38-0.49% (Gaussian 0.27%), mean absolute error 24-27 uV.
  Sigma is too wide in the bulk and too narrow in the tails. The noise augmentation, present in training but
  absent from validation, explains the first part.
- Maps: 198 ECGAD development and 5,801 evaluation records. Phase-family coverage was 94-95% of lead-samples;
  the rest takes the other families' value. The evaluation maps took about 6 minutes of GPU time after the
  lock was acquired (2,000 records in 11.2 minutes of logged time that included the wait, 5,801 in 15.1). The
  whole experiment took about 90 minutes of wall time, much of it waiting for the shared GPU lock.

## Integrity

All passed (`result.json` `integrity`):

1. The scorer reproduced the bar's A2 exactly: 0.7419383358406554 in both 061 and this run.
2. Determinism: maps for 5 ECGAD development records, regenerated in a separate process, matched the saved maps
   exactly (maximum absolute difference 0.0 for every variant).
3. Data: 0 training or validation records are ECGAD records or in the `evaluation` partition.
4. No-leak unit test: changing hidden input samples leaves the output unchanged (`tests/test_inpaint063.py`, 8
   tests pass).
5. Every map is finite with shape (n, 12, 5000), checked at write time. The QRS indicator used by the combined
   variants matched 061's `qrs_prior >= 0.5` on all 198 ECGAD development records (0 mismatched lead-samples).

## Why the primary failed

- Dividing by sigma removes the QRS emphasis. Only 21% of the `primary` map's top 5% of ECGAD development
  lead-samples fall inside QRS (base rate 16%), against 74% for `raw`. The cardiologist marks are concentrated
  on QRS (35.9% of marks), so an evenly spread map loses A2 and A4.
- The z maps are drawn to noise (C1_std 0.68-0.71, like `U_B` and `residual056`), the opposite of the design
  intent. In flat segments sigma is small, so the real PTB-XL noise annotated in the noise set produces large z.
  The synthetic electrode noise in training did not teach the model to widen sigma enough for this noise. The
  `raw` lead score is dominated by QRS-sized errors instead, and stays near 0.5 after lead standardization
  (C1_std 0.59; 0.54 for `raw_qrs_combined`).
- Phase masking alone (`phase`) is slightly better than block masking on A2 (0.617 against 0.596) and on B1. It
  does not recover persistent ST/T findings: B1 for ST injury and ischemia stays near 0.5 in every variant.
- What the reconstruction error does well is detection and record ranking: D1 0.746-0.781 against 0.632 for the
  bar and 0.762 for `U_B`, and A5 up to 0.849.

## Deviations

- Amendment 1: ECGAD maps stored in float32, as in 061.
- Amendment 4: a post-hoc v2 primary, scored once; no other version was scored.
- Running `run_locbench_strata` with the 061 `qrs_prior` and `residual056` folders as inputs rewrote their
  `strata.json` and `strata.csv`. It is a deterministic recomputation and the residual056 values match 061
  amendment 2 (0.607 [0.570, 0.644]). The v2 strata, and the strata paired against `residual056`, were computed
  with the same library functions and written only into this experiment's folders.
- Challenge-train normals were not used (declared in the protocol).

## Follow-up ideas

- Combine the `raw` error (as QRS-morphology and detection evidence, low noise attraction) with `residual056`
  and the QRS prior in Experiment 066. The two are close on A2_qrs, and `raw` is far less drawn to noise.
- Train noise-aware sigma on real PTB-XL noise: the `fit`-partition records with noise annotations, used as
  unlabeled noisy examples, or a mixture of observation noise and model sigma. Then retest z maps on C1_std.
- Train to convergence: patience counted against a smoothed validation curve, or a fixed cosine budget. Early
  stopping at 28% of the schedule left the model undertrained.
- ST/T localization needs a different signal. Phase masks over the ST/T window with the whole precordial or limb
  group hidden at once, or a beat-aligned target, could stop the model copying ST shifts from neighbouring leads.
- Add Challenge-train normals as a domain-robustness variant, with a per-source sigma.
