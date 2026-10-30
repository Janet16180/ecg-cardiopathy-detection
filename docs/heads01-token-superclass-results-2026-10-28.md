# HEADS-01 results: neither readout change improved SPH sensitivity

28 October 2026. Both prespecified candidates failed the decision rule of the committed
[protocol](heads01-token-superclass-protocol-2026-10-28.md) (commit `69c6f2b`; code `4554816`).
The current best, the all-age 25k 50:50 blend, stays the best development model, and the default
remains `ptb_baseline_blend`. No pipeline was promoted and nothing was changed after the scores.

| SPH, frozen 5% normal budget | Sensitivity | Specificity | AUROC | Gain over current best, pp [97.5% paired interval] | Decision |
| --- | ---: | ---: | ---: | --- | --- |
| Current best: all-age blend | 0.773996 | 0.947737 | 0.936770 | reference | |
| A: token-pooled xECG (mean + max) | 0.771484 | 0.947737 | 0.935949 | −0.251 [−0.825, +0.331] | not established |
| B: superclass mixture heads | 0.777065 | 0.949489 | 0.939240 | +0.307 [−0.224, +0.892] | not established |
| Default: PTB-only blend | 0.686384 | 0.949489 | 0.917495 | | |

Each sensitivity uses the 10,434 SPH evaluation ECGs (3,584 proxy-positive, 6,850 proxy-negative,
10,116 patients). Each threshold came only from the 6,968 SPH calibration normals: 348 of them
scored strictly above it for every model. Intervals are paired whole-patient bootstraps with 2,000 draws,
seed 20261028, and no skipped draws. The 97.5% level is the Bonferroni level for two primary candidates.

## Decision gates

| Gate | A token pool | B superclass |
| --- | --- | --- |
| 1. Point gain at least 0.02 | fail (−0.00251) | fail (+0.00307) |
| 2. Lower 97.5% bound above zero | fail (−0.00825) | fail (−0.00224) |
| 3. Specificity drop at most 0.005 | pass (0.0) | pass (+0.00175) |
| Established improvement | no | no |
| Statistically supported gain below the practical bar | no | no |

B caught a net 11 more positive ECGs (2,785 against 2,774) and referred 12 fewer negative ECGs
(346 against 358). A caught 9 fewer (2,765) with the same 358 negatives referred. These are ECG counts,
not distinct patients. The 95% intervals were [−0.770, +0.254] pp for A and [−0.166, +0.809] pp for B.

## Against the default

| Candidate minus `ptb_baseline_blend` | Sensitivity, pp [95%] | Specificity change, pp | AUROC difference [95%] |
| --- | --- | ---: | --- |
| A token pool | +8.510 [+7.445, +9.660] | −0.175 | +0.01845 [+0.01508, +0.02204] |
| B superclass | +9.068 [+7.994, +10.184] | 0.000 | +0.02175 [+0.01834, +0.02509] |

Both gains come mostly from the multisource cohort that the current best already uses; the current best
itself is 8.761 pp above the default (2,774 against 2,460 caught).

## Secondary results (descriptive, no decision)

All SPH models, each with its own calibration-normal threshold:

| Model | AUROC | AP | Sensitivity | Specificity | TP | FN | FP | TN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ptb_baseline_blend | 0.917495 | 0.888419 | 0.686384 | 0.949489 | 2,460 | 1,124 | 346 | 6,504 |
| all_age_blend | 0.936770 | 0.917951 | 0.773996 | 0.947737 | 2,774 | 810 | 358 | 6,492 |
| token_pool_blend | 0.935949 | 0.915182 | 0.771484 | 0.947737 | 2,765 | 819 | 358 | 6,492 |
| superclass_blend | 0.939240 | 0.920159 | 0.777065 | 0.949489 | 2,785 | 799 | 346 | 6,504 |
| token_superclass_blend (C) | 0.938674 | 0.914621 | 0.779297 | 0.948321 | 2,793 | 791 | 354 | 6,496 |
| all_age_hgb | 0.925757 | 0.894833 | 0.745815 | 0.945255 | 2,673 | 911 | 375 | 6,475 |
| all_age_xecg | 0.933510 | 0.914178 | 0.765625 | 0.948905 | 2,744 | 840 | 350 | 6,500 |
| token_pool_xecg | 0.929695 | 0.906499 | 0.745815 | 0.952409 | 2,673 | 911 | 326 | 6,524 |
| superclass_hgb | 0.931122 | 0.901048 | 0.750279 | 0.950511 | 2,689 | 895 | 339 | 6,511 |
| superclass_xecg | 0.934616 | 0.915571 | 0.773996 | 0.945839 | 2,774 | 810 | 371 | 6,479 |
| token_superclass_xecg | 0.932370 | 0.905621 | 0.761440 | 0.950365 | 2,729 | 855 | 340 | 6,510 |

| Contrast at SPH | Sensitivity, pp [95%] | Specificity change, pp | AUROC difference [95%] |
| --- | --- | ---: | --- |
| A minus current best | −0.251 [−0.770, +0.254] | 0.000 | −0.00082 [−0.00185, +0.00014] |
| B minus current best | +0.307 [−0.166, +0.809] | +0.175 | +0.00247 [+0.00154, +0.00333] |
| C minus current best | +0.530 [−0.028, +1.092] | +0.058 | +0.00190 [+0.00061, +0.00309] |
| Token-pooled xECG minus binary xECG | −1.981 [−2.702, −1.225] | +0.350 | −0.00382 [−0.00556, −0.00211] |
| Superclass HGB minus binary HGB | +0.446 [−0.190, +1.025] | +0.526 | +0.00536 [+0.00404, +0.00671] |
| Superclass xECG minus binary xECG | +0.837 [+0.302, +1.376] | −0.307 | +0.00111 [+0.00011, +0.00204] |

What these show, without changing the decision:

- Adding the token maximum hurt the xECG readout at SPH: −1.98 pp sensitivity and −0.0038 AUROC, both
  intervals below zero. At PTB its AUROC was also lower (0.910887 against 0.916739). On the first 5,000
  rows of each part, the median per-feature standard deviation of the maximum was 0.742 in training and
  0.812 in the SPH-first evaluation part, against 0.533 and 0.361 for the mean. That the maximum carries
  more site-specific variation is a hypothesis these numbers suggest, not a tested result.
- The superclass mixture raised the ranking of both components (HGB AUROC +0.0054, xECG +0.0011, both
  intervals above zero) and the blend's AUROC by +0.0025, but this moved the 5% operating point by only
  +0.31 pp. Arm C, which combines both changes, was secondary and carries no decision.

SPH sensitivity within each superclass flag of the positive evaluation ECGs (no intervals; a record can
carry several flags):

| Flag | Positives | Default | Current best | A | B | C |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MI | 118 | 0.9153 | 0.9407 | 0.9407 | 0.9492 | 0.9492 |
| STTC | 2,521 | 0.7180 | 0.8358 | 0.8334 | 0.8362 | 0.8362 |
| CD | 1,166 | 0.6501 | 0.6655 | 0.6621 | 0.6732 | 0.6810 |
| HYP | 104 | 0.8462 | 0.9423 | 0.9423 | 0.9327 | 0.9327 |

Conduction findings have the lowest sensitivity for every model, in line with Experiment 030.

PTB (fold 8: 889 normals set the threshold; fold 9: 2,146 ECGs, 1,232 positive), blend sensitivity and
AUROC: default 0.779221 / 0.950182, current best 0.669643 / 0.923255, A 0.663961 / 0.921002,
B 0.685877 / 0.929420, C 0.685065 / 0.927960. The PTB-only default stays best at its own hospital.

## Integrity, smoke test, profile and run

- Predecessor check (qualify stage, 381 s): all 30 saved model/site points, including thresholds,
  confusion matrices, AUROC and sweeps, and the saved strict-versus-matched primary interval reproduced
  exactly from the saved predictions. Refitting the all-age and age-preferred heads reproduced every saved
  SPH and PTB score with a maximum absolute difference of 0.0, so `all_age_blend` 2,774 and `younger_blend`
  2,760 caught of 3,584 and `ptb_baseline_blend` 2,460 were reproduced before anything new ran.
- Superclass labels: for all 28,568 admitted pool records, the reconstructed label and the union of the
  four flags equal the cohort target (zero mismatches). The all-age heads had MI 2,041, STTC 12,356,
  CD 6,049 and HYP 2,376 positives against 7,393 negatives.
- Smoke test (no evaluation score): 32 training token means were bit-identical to the saved features and
  the synthetic fit/score/bootstrap path ran. It is not a result.
- GPU profiles (48 records each): training on GPU 0 used 619,681,792 peak bytes with 10.6 GB free and a
  conservative forecast of 1,858 s; evaluation on GPU 1 used the same peak with 4.63 GB free and a forecast
  of 4,655 s. All gates passed. Each part held its own device lock; no other process was touched.
- Extraction ran both parts in parallel: 28,568 training records in 1,927 s and 21,677 evaluation
  records in 1,699 s. Token means were bit-identical to the saved features for all 19,666 Challenge and all
  17,402 SPH rows; PTB rows differed by at most 5.7e-6 (8,896 of 8,902 and 4,272 of 4,275 identical),
  within the 1e-5 gate.
- Evaluation (242 s) refitted the reference inside the stage and again matched the saved scores to 1e-12
  before scoring the candidates. Fit times: binary HGB 1.8 s, binary xECG 10.3 s, token xECG 19.8 s,
  superclass HGB 3.9 s, superclass xECG 15.9 s, token superclass xECG 36.6 s.

The [aggregate](../reports/heads01-token-superclass-2026-10-28/aggregate.json) (SHA-256
`48479c43d124bef2ff9b1943af722722cd05e41a06935dd196a56ae993d6d8a7`) holds every point, sweep, interval
and receipt hash without patient data. Local outputs, including the 4.2 GB token cache, are in
`outputs/heads01_token_superclass_20261028/`; logs are `outputs/heads01_*_20261028.log`.

## Scope and limitations

SPH and PTB folds 8 and 9 are reused development data, and SPH has now been read by many experiments.
SPH never fitted a model, scaler, label or weight; its calibration normals only set thresholds. No
Challenge test group, EchoNext test set, PTB fold 10 or final frozen test was opened. The intervals are
conditional on the fitted models and the calibration sample; they do not include training-sample or
threshold-calibration uncertainty. Labels remain ECG-annotation proxies, released xECG pretraining saw
Chapman/Ningbo, and nothing here measures performance in Italian students. No age subgroup was analysed
and no label definition changed. The concurrent `pr75` hyperparameter study was not used or affected.

## Follow-ups (added to the backlog)

- `xecg_learned_attention_readout`: a learned attention pool over the cached tokens. Lower value now that
  max pooling hurt transfer.
- `superclass_mixture_resampling`: whether B's AUROC gain survives refitting on resampled training sets.
- `full_pool_fixed_heads`: the fixed heads on all 28,568 admitted records.
- `conduction_miss_analysis`: what the missed SPH conduction ECGs have in common (descriptive).
