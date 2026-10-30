# NBR-02 results: the personalized counterpart sharpens lead localization but lowers selectivity, so P0 stays

28 October 2026, under the committed [protocol](nbr02-personal-counterpart-protocol-2026-10-28.md)
(`3d8cace`; code `33dfd2e`). This branch is stacked on NBR-01 (started from `fb20de7`); its pull request
waits until NBR-01 is on `main`. Every number comes from `outputs/nbr02_personal_counterpart_20261028/`
(`analyze/results.json`, `fitsets/fitsets.json`, `fitsets/cohort_checks.json`, `prepare/prepare.csv`,
`profile/profile.json`, `encode_*/encode.json`, `export/export.json`). Intervals are 95% whole-patient
bootstraps, 2,000 draws, seed 20261028, with every statistic of every arm on the same draws; no draw was
skipped for any statistic. Scores are `all_age_blend` probabilities unless stated.

## Answer

- Decision: C6 does not replace P0. The page keeps NBR-01's population counterpart.
- A failed. Selectivity fell from 0.935 [0.812, 1.070] (P0) to 0.783 [0.678, 0.896] (C6); the paired
  difference is -0.152 [-0.205, -0.104]. The personalized template cut the mean absolute whole-lead change
  on abnormal ECGs by 29% (0.0207 to 0.0147, -0.0060 [-0.0068, -0.0052]) but on normals by only 15% (0.0221
  to 0.0188, -0.0033 [-0.0042, -0.0025]). The ratio fell because abnormal changes shrank more, not because
  of artefacts raising abnormal scores.
- B passed. The territory contrast rose from +0.306 to +0.386 [+0.320, +0.454]; C6 minus P0 is +0.080
  [+0.041, +0.123], above the -0.05 margin.
- C passed. The abnormal full-replacement drop is 0.245 [0.227, 0.261]. It is less than half of P0's 0.519:
  after full C6 replacement the mean abnormal score is 0.665, above the 0.657 threshold, and only 36.2% of
  flagged abnormal ECGs lose the flag (P0 76.3%).
- Personalization, not the cohort, drives every effect. Rank 6 against rank 0 changes selectivity by -0.130
  (P fit set) and -0.145 (C fit set); the C cohort against the P normals changes it by -0.007 [-0.023,
  +0.009] at rank 0 and -0.023 [-0.050, +0.003] at rank 6.
- The rank-6 subspace absorbs part of the abnormality. Templates move further from the population template
  for abnormal groups (0.037-0.044 mV) than for normals (0.026 mV), and at least one C6 coordinate sits on
  the 2.58 SD clip bound for 79% of LBBB, 37% of RBBB and 6% of normal ECGs.

## Primary decision

| Part | Statistic | Point [95%] | Bound | Result |
| --- | --- | --- | --- | --- |
| A. Selectivity improves | C6 - P0 selectivity | -0.152 [-0.205, -0.104] | lower > 0 | failed |
| B. Territory non-inferior | C6 - P0 territory contrast | +0.080 [+0.041, +0.123] | lower > -0.05 | passed |
| C. Operator check | C6 abnormal full-replacement drop | +0.245 [+0.227, +0.261] | lower > 0 | passed |
| C6 replaces P0 | all three | | | no |

## The four arms (`all_age_blend`)

| Arm | Selectivity | Abnormal lead change | Normal lead change | Territory | Minus energy | Wave | Top-1 hit | Abnormal full drop | Full flip |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| P0 (NBR-01) | 0.935 [0.812, 1.070] | 0.0207 | 0.0221 | +0.306 [+0.248, +0.363] | +0.190 | +0.547 | 0.490 | 0.519 | 0.763 |
| P6 | 0.805 [0.695, 0.924] | 0.0150 | 0.0186 | +0.376 [+0.309, +0.442] | +0.240 | +0.457 | 0.654 | 0.245 | 0.356 |
| C0 | 0.928 [0.807, 1.062] | 0.0210 | 0.0226 | +0.305 [+0.247, +0.363] | +0.175 | +0.550 | 0.500 | 0.538 | 0.773 |
| C6 | 0.783 [0.678, 0.896] | 0.0147 | 0.0188 | +0.386 [+0.320, +0.454] | +0.235 | +0.390 | 0.712 | 0.245 | 0.362 |

Lead change is the mean absolute score change over the 12 whole-lead replacements; full flip is the share of
flagged abnormal ECGs whose flag the full replacement removes. Each arm's energy map was recomputed from its
own replacement energies: its territory contrast is 0.115 (P0), 0.136 (P6), 0.129 (C0) and 0.151 (C6),
and its wave contrast falls from 0.178 (P0) to 0.065 (C6). The self-swap territory contrast is +0.109
[+0.014, +0.209] for every arm, because the self-swap is shared.

## 2 x 2 decomposition (paired, `all_age_blend`)

| Statistic | Cohort, rank 0 (C0 - P0) | Cohort, rank 6 (C6 - P6) | Personalization, P (P6 - P0) | Personalization, C (C6 - C0) | C6 - P0 |
| --- | --- | --- | --- | --- | --- |
| Selectivity | -0.007 [-0.023, +0.009] | -0.023 [-0.050, +0.003] | -0.130 [-0.177, -0.088] | -0.145 [-0.200, -0.096] | -0.152 [-0.205, -0.104] |
| Territory contrast | -0.001 [-0.009, +0.006] | +0.010 [-0.010, +0.033] | +0.070 [+0.036, +0.106] | +0.081 [+0.042, +0.123] | +0.080 [+0.041, +0.123] |
| Territory minus energy | -0.015 [-0.026, -0.005] | -0.005 [-0.026, +0.018] | +0.050 [+0.012, +0.087] | +0.060 [+0.020, +0.103] | +0.045 [+0.001, +0.088] |
| Wave contrast | +0.003 [-0.005, +0.013] | -0.067 [-0.089, -0.048] | -0.089 [-0.123, -0.057] | -0.160 [-0.194, -0.124] | -0.156 [-0.191, -0.121] |
| Top-1 territory hit | +0.010 [-0.022, +0.044] | +0.058 [+0.000, +0.126] | +0.164 [+0.073, +0.256] | +0.212 [+0.116, +0.308] | +0.221 [+0.126, +0.318] |
| Abnormal full drop | +0.019 [+0.017, +0.021] | +0.000 [-0.005, +0.005] | -0.274 [-0.292, -0.257] | -0.293 [-0.313, -0.274] | -0.274 [-0.294, -0.255] |
| Normal full drop | +0.008 [+0.007, +0.009] | +0.010 [+0.006, +0.014] | -0.042 [-0.052, -0.033] | -0.040 [-0.051, -0.030] | -0.032 [-0.043, -0.022] |
| Abnormal one-lead flip | +0.008 [+0.002, +0.014] | +0.000 [-0.009, +0.009] | -0.036 [-0.051, -0.022] | -0.043 [-0.060, -0.027] | -0.036 [-0.052, -0.020] |
| Abnormal full flip | +0.009 [+0.002, +0.017] | +0.006 [-0.016, +0.028] | -0.407 [-0.445, -0.370] | -0.410 [-0.446, -0.374] | -0.401 [-0.438, -0.365] |

## Secondary results

Components and the secondary model (C6 - P0):

| Score | Selectivity | Territory contrast |
| --- | --- | --- |
| `all_age` HGB | -0.064 [-0.128, -0.007] | +0.107 [+0.062, +0.155] |
| `all_age` xECG | -0.206 [-0.266, -0.152] | +0.018 [-0.032, +0.065] |
| `ptb_baseline_blend` | -0.139 [-0.186, -0.093] | +0.055 [+0.021, +0.088] |

The territory gain comes mostly from the HGB measurements; the xECG component's territory contrast does not
change, and its selectivity falls most.

Other `all_age_blend` endpoints, C6 against P0: top-3 territory hit 0.952 against 0.817 (+0.135 [+0.069,
+0.206]); QRS-in-BBB contrast 0.411 against 0.550 and ST-T contrast 0.370 against 0.543; BBB lead
contrast 0.390 against 0.429 (-0.039 [-0.064, -0.016]); flip rates among flagged abnormal ECGs: grid cell
0.066 against 0.085, lead 0.088 against 0.124, component 0.240 against 0.395. The normals' full drop is
0.086 [0.068, 0.105] against 0.119.

Mean score after full replacement, C6 against P0, by group: inferior 0.573 against 0.207, anterior 0.645
against 0.346, LBBB 0.878 against 0.717, RBBB 0.680 against 0.530, stt 0.645 against 0.254, normal 0.090
against 0.057.

Template movement (mean absolute difference over the 12 x 380 grid, mV) and clipping:

| Group | P6 - P0 | C6 - C0 | C0 - P0 | C6 - P0 | P6 any coordinate clipped | C6 any coordinate clipped | C6 mean clipped (of 6) |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| inferior | 0.0377 | 0.0395 | 0.0083 | 0.0398 | 0.367 | 0.327 | 0.57 |
| anterior | 0.0377 | 0.0413 | 0.0086 | 0.0420 | 0.364 | 0.364 | 0.49 |
| LBBB | 0.0438 [0.0401, 0.0476] | 0.0438 [0.0399, 0.0477] | 0.0086 | 0.0439 | 0.912 | 0.794 | 1.47 |
| RBBB | 0.0400 [0.0385, 0.0416] | 0.0399 [0.0383, 0.0416] | 0.0086 | 0.0413 | 0.445 | 0.371 | 0.54 |
| stt | 0.0374 | 0.0386 | 0.0082 | 0.0397 | 0.343 | 0.287 | 0.34 |
| normal | 0.0265 [0.0253, 0.0277] | 0.0256 [0.0244, 0.0269] | 0.0082 | 0.0284 | 0.147 | 0.063 | 0.08 |

The two population templates differ by about 0.008 mV for every group. The rank-6 projection moves LBBB
templates most and clips at least one LBBB coordinate in 79-91% of records: a global BBB pattern is partly
explained away as normal variation, up to the clip bound. The C cohort's wider normal variation clips
normals less often (6.3% against 14.7%).

## Integrity

| Check | Result |
| --- | --- |
| 1. Rebuilt P0 counterparts against NBR-01's saved counterparts | 0.0 mV on all 1,037 records; 062 knots identical for all |
| Refitted P0 model against NBR-01's `rank0.npz` | identical in all six fields |
| 2. NBR-01 statistics recomputed by the new analysis code from NBR-01's saved scores | all 769 point values identical; primary +0.3056275517940945 exactly |
| Unreplaced and self-swap scores | NBR-01's arrays, not recomputed |
| 3. Cohort fit set | 4,355 Challenge rows all `train`, full-signal hash equal to the split table's for all 4,355; zero record or hash overlap with Challenge test or calibration; zero SPH rows, paths or hash matches; PTB-XL folds 1-7 only; no EchoNext, MIMIC or CODE-15 |
| 4. Variant SHA-256 per arm between preparation and encoding | all 3 x 1,037 matched |

Fit sets: P 2,666 of 2,666 usable. C 7,393 rows, minus 54 with missing age (Ningbo 41, Georgia 13), 0
unknown sex and 0 without two complete beats, leaving 7,339 (PTB-XL 3,038, Ningbo 2,540, Chapman 819,
Georgia 788, CPSC 2018 154). The six P6 eigenvalues are 1,249, 592, 514, 417, 310 and 233; the C6
eigenvalues are 1,118, 632, 509, 427, 319 and 226 (standardized units).

## Run

Fit sets 286 s (16 workers). Profile on 20 ECGs: preparation 3.43 CPU-seconds per record, encoding 1.43 s per
record for 156 variants. Full run: preparation 146 s with 16 workers; encoding 519 ECGs on GPU 0 in 847 s
and 518 on GPU 1 in 743 s, in parallel, each under its own lock, peak 0.62 GB; bootstrap 85 s.

## Deviations and disclosures

1. The `fitsets` and `profile` stages ran before the protocol was committed. They read no SPH score; the
   fit-set counts and checks they produced are quoted in the protocol. Their code was not changed before
   the commit.
2. `data/processed/challenge_splits_v1/rows.csv` is not present locally. The check used the recovered copy
   `outputs/cohort25_recovery_20261028/challenge/full/rows.csv`, whose SHA-256 equals the value in
   `docs/challenge-splits-v1.md`.
3. The decision rule was reviewed before any score and kept unchanged (see the protocol). The result shows
   the case the review anticipated in reverse: selectivity fell because the numerator shrank.
4. New variants were encoded in batches of the record's own new variants, while the unreplaced scores come
   from NBR-01's encoding; NBR-01 measured such batch effects at up to 2.1e-8 on the blend.

## What it means

- A personalized normal counterpart is not a better "normal version of this student" for the page: the
  rank-6 normal subspace explains away part of what makes an abnormal ECG abnormal (most for LBBB), so
  replacing everything with it leaves most abnormal ECGs flagged.
- The same absorption removes global, normal-variation differences and leaves the remaining changes more
  localized, which is why lead localization sharpens: the top single lead falls in the MI territory for 71%
  of territory ECGs (P0 49%), and the territory contrast beyond energy rises by +0.045 [+0.001, +0.088].
- Fitting on the multi-source training normals instead of 2,666 PTB-XL normals changes nothing material at
  either rank. The population template was not the limiting factor; personalization was the only active
  ingredient.
- Under the frozen rule the page keeps P0. A page could still show the C6 lead ranking as a second view,
  but that is a design question for the user, not a result of this study.

## Limitations

- Same as NBR-01: label-derived validation on SPH development data, old-MI territories only, no clinician
  marks, model-specific attribution on the probability scale.
- Selectivity is a ratio of probability changes; abnormal scores near 1 compress their changes in every arm.
- The rank (6) and clip bound (2.58 SD) are 062's and were not tuned; a smaller bound could absorb less
  abnormality, but choosing it on SPH would need its own protocol.

## Page export

`outputs/nbr02_personal_counterpart_20261028/examples.json` (6,459,208 bytes; validated by
`nbr02_export.validate`): NBR-01's 16 IDs in the same order, with the C6 counterpart, scores and energies;
signals, components, unreplaced and self-swap scores are NBR-01's. `summary.statistics` holds 1,841
statistics with intervals, under arm-prefixed keys (`C6/all_age_blend/territory`) and contrast keys
(`C6-P0/all_age_blend/selectivity`); `summary.decisions` holds parts A, B and C and `C6_replaces_P0: false`;
top-level `arm: "C6"` and `fit_set` counts by source. Because C6 did not pass, the page should keep NBR-01's
P0 file as its default.

## Follow-ups (added to the backlog)

- `nbr_subspace_absorption`: measure, label-free, how much of each SPH group's median-beat deviation the
  062 rank-6 subspace explains, and whether a fit that excludes clipped records changes it.
- `nbr_bounded_personalization`: a protocol to test a smaller clip bound or rank chosen on PTB-XL fold 9,
  then confirmed once on SPH, aiming to keep abnormal changes while shrinking normal ones.
- `nbr_dual_view_page`: a page design question for the user, P0 for flips and full replacement and C6 for
  the lead ranking.
- `nbr_record_specific_counterpart` is marked done: its rank-6 projection variant is this study.
