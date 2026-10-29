# NBR-04 results: leaving the region out does not stop the absorption; neither arm replaces P0

28 October 2026, under the committed [protocol](nbr04-leave-region-out-protocol-2026-10-28.md) (`e0b32ee`;
code `ffd0e67`). Same branch as NBR-03, stacked on #79. Every number comes from
`outputs/nbr04_leave_region_out_20261028/` (`analyze/results.json`, `score/integrity.json`,
`fitsets/fitsets.json`, `prepare/prepare.csv`, `profile/profile.json`, `encode_*/encode.json`,
`export/export.json`). Intervals are 97.5% whole-patient bootstraps, 2,000 draws, seed 20261028, with every
statistic of every arm of one dataset on the same draws; no draw was skipped for any decision statistic.
Scores are `all_age_blend` probabilities unless stated.

## Answer

- Neither L6 nor L6D replaces P0. The page keeps NBR-01's population counterpart; the export carries L6, as
  the protocol fixed for this case.
- L6 behaves almost exactly like NBR-02's C6. Leaving the replaced region out of the personalization changes
  SPH selectivity by only +0.008 [-0.004, +0.021] against C6 and the territory contrast by -0.002 [-0.010,
  +0.007]. So the absorption does not come from the replaced region itself: the other leads and components
  predict the abnormal shape through the subspace's correlations.
- L6D, which also removes the four superclass directions, recovers part of the selectivity (+0.087 [+0.026,
  +0.148] over L6) and the wave contrast, but loses the whole localization gain: its territory contrast
  equals P0's (-0.010 [-0.032, +0.012]).
- Both arms fail part A (selectivity below P0) and part C (component flips lost beyond the 0.05 margin), and
  neither is confirmed on PTB-XL fold 9.

## Primary decision (SPH, D on PTB-XL fold 9)

| Part | Rule | L6 | L6D |
| --- | --- | --- | --- |
| A. Selectivity, arm - P0 | lower > 0 | -0.144 [-0.206, -0.093], failed | -0.058 [-0.115, -0.002], failed |
| B. Territory, arm - P0 | lower > -0.05 | +0.078 [+0.033, +0.128], passed | -0.010 [-0.032, +0.012], passed |
| C. Component flip rate, arm - P0 | lower > -0.05 | -0.076 [-0.105, -0.048], failed | -0.048 [-0.072, -0.025], failed |
| D. PTB selectivity, arm - P0 | point > 0 | -0.148 [-0.220, -0.087], failed | -0.022 [-0.066, +0.018], failed |
| D. PTB territory, arm - P0 | lower > -0.05 | +0.011 [-0.018, +0.038], passed | -0.003 [-0.023, +0.017], passed |
| D confirmed | both D parts | no | no |
| Replaces P0 | A, B, C and D | no | no |

Tie rule (not needed): L6D minus L6 selectivity is +0.087 [+0.026, +0.148]. No arm was chosen; the export is
L6.

## The arms on SPH (`all_age_blend`)

| Arm | Selectivity | Abnormal lead change | Normal lead change | Territory | Minus energy | Wave | Top-1 hit | Component flip | Full flip |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| P0 (NBR-01) | 0.935 [0.796, 1.093] | 0.0207 | 0.0221 | +0.306 [+0.240, +0.373] | +0.190 | +0.547 | 0.490 | 0.395 | 0.763 |
| C6 (NBR-02) | 0.783 [0.660, 0.915] | 0.0147 | 0.0188 | +0.386 [+0.306, +0.462] | +0.235 | +0.390 | 0.712 | 0.240 | 0.362 |
| L6 | 0.791 [0.668, 0.922] | 0.0150 | 0.0190 | +0.384 [+0.305, +0.463] | +0.235 | +0.446 | 0.712 | 0.319 | 0.773 |
| L6D | 0.877 [0.745, 1.028] | 0.0182 | 0.0208 | +0.296 [+0.226, +0.366] | +0.164 | +0.535 | 0.538 | 0.347 | 0.773 |

Full replacement in L6 and L6D is C0 by construction, so their full flip rate is C0's (0.773; P0 0.763), and
the abnormal full drop is 0.538 against P0's 0.519. At the component level both arms still keep more
flagged abnormal ECGs flagged than P0.

Selected paired contrasts on SPH:

| Statistic | L6 - P0 | L6D - P0 | L6D - L6 | L6 - C6 | L6D - C6 |
| --- | --- | --- | --- | --- | --- |
| Selectivity | -0.144 [-0.206, -0.093] | -0.058 [-0.115, -0.002] | +0.087 [+0.026, +0.148] | +0.008 [-0.004, +0.021] | +0.095 [+0.033, +0.159] |
| Abnormal lead change | -0.0057 [-0.0066, -0.0049] | -0.0025 [-0.0030, -0.0020] | +0.0032 [+0.0025, +0.0040] | +0.0003 | +0.0035 |
| Normal lead change | -0.0032 [-0.0042, -0.0022] | -0.0014 [-0.0026, -0.0002] | +0.0018 [+0.0007, +0.0030] | +0.0002 | +0.0020 |
| Territory | +0.078 [+0.033, +0.128] | -0.010 [-0.032, +0.012] | -0.088 [-0.141, -0.040] | -0.002 [-0.010, +0.007] | -0.090 [-0.141, -0.042] |
| Territory minus energy | +0.045 [-0.007, +0.096] | -0.026 [-0.064, +0.009] | -0.071 [-0.122, -0.021] | -0.000 | -0.071 |
| Wave | -0.100 [-0.139, -0.062] | -0.012 [-0.034, +0.011] | +0.088 [+0.051, +0.129] | +0.056 [+0.020, +0.092] | +0.144 [+0.104, +0.187] |
| Top-1 territory hit | +0.221 [+0.111, +0.336] | +0.048 [-0.012, +0.114] | -0.173 [-0.271, -0.071] | +0.000 | -0.173 |
| Component flip | -0.076 [-0.105, -0.048] | -0.048 [-0.072, -0.025] | +0.028 [-0.003, +0.058] | +0.079 [+0.054, +0.106] | +0.108 [+0.072, +0.145] |

Components and the secondary model, selectivity minus P0 on SPH: HGB L6 -0.061 [-0.129, +0.004], L6D -0.050
[-0.126, +0.022]; xECG L6 -0.196 [-0.260, -0.136], L6D -0.059 [-0.104, -0.016]; `ptb_baseline_blend` L6 -0.132
[-0.186, -0.084], L6D -0.073 [-0.128, -0.021].

## PTB-XL fold 9 (`all_age_blend`)

| Arm | Selectivity | Territory | Minus energy | Wave | Top-1 hit | Component flip |
| --- | --- | --- | --- | --- | ---: | ---: |
| P0 (NBR-03) | 0.982 [0.838, 1.158] | +0.098 [+0.044, +0.148] | +0.038 | +0.196 | 0.424 | 0.466 |
| L6 | 0.834 [0.703, 0.992] | +0.109 [+0.051, +0.166] | +0.065 | +0.110 | 0.469 | 0.358 |
| L6D | 0.960 [0.819, 1.129] | +0.095 [+0.046, +0.146] | +0.038 | +0.190 | 0.381 | 0.403 |

Contrasts against P0: L6 selectivity -0.148 [-0.220, -0.087], territory +0.011 [-0.018, +0.038], component flip
-0.108 [-0.155, -0.063]; L6D selectivity -0.022 [-0.066, +0.018], territory -0.003 [-0.023, +0.017], component
flip -0.063 [-0.095, -0.031]. On PTB-XL even L6's localization gain is small.

## Clipping and template movement

Share of a record's 51 region fits with any coordinate at the clip bound, and mean absolute template movement
from the population (C0) template on 062's grid, by group:

| Group (SPH) | L6 clip | L6D clip | L6 movement, mV | L6D movement, mV |
| --- | ---: | ---: | --- | --- |
| inferior | 0.321 | 0.194 | 0.0395 | 0.0293 |
| anterior | 0.386 | 0.122 | 0.0413 | 0.0287 |
| LBBB | 0.783 | 0.644 | 0.0439 [0.0395, 0.0482] | 0.0444 [0.0395, 0.0495] |
| RBBB | 0.377 | 0.400 | 0.0399 [0.0382, 0.0419] | 0.0366 [0.0352, 0.0379] |
| stt | 0.291 | 0.224 | 0.0386 | 0.0315 |
| normal | 0.068 | 0.075 | 0.0258 | 0.0252 |

On PTB-XL fold 9: LBBB clip 0.849 (L6) and 0.721 (L6D), movement 0.0443 and 0.0494 mV; normals clip 0.075 and
0.094, movement 0.0252 and 0.0243 mV; lateral clip 0.314 and 0.285.

Removing the superclass directions moves the templates of MI and ST-T ECGs less (inferior 0.0395 to 0.0293
mV), but not those of LBBB (0.0439 to 0.0444 mV), which still clip in 64% of region fits: the conduction
pattern is not well described by one mean direction per superclass.

## Integrity

| Check | Result |
| --- | --- |
| 1. L6 full-replacement counterpart against NBR-02's C0 | bit-identical for all 1,037 SPH ECGs |
| 1. L6 full-replacement scores against NBR-02's C0 full-replacement scores | largest difference 0.0 (HGB and xECG), so the pre-registered batch exception was not needed |
| 2. P0 primary statistics reproduced by the NBR-04 code | SPH +0.3056275517940945 and PTB-XL fold 9 +0.09848614265592237, exact |
| 3. C0 and C6 share coefficients and scale; L6D keeps them | passed |
| 4. Variant SHA-256 per arm between preparation and encoding | all 2 x 1,899 matched |
| 5. 062 knots against the NBR-01 and NBR-03 plans | identical for all 1,899 ECGs |
| Disease directions, rerun on committed code | identical arrays to the first run |

## Run

Fit set and directions 343 s on the committed rerun (1,062 s on the first run). Profile: preparation 2.07
CPU-seconds per record and encoding 1.25 s per record (104 variants); L6 full equal to C0 for 10 of 10.
Preparation of 1,899 ECGs 191 s with 16 workers; encoding 949 and 950 ECGs in 1,234 and 1,275 s on the two
GPUs in parallel, each under its lock; bootstrap and analysis 215 s.

## Deviations and disclosures

1. The first `fitsets` run finished its computation but failed to write its receipt, because a source file
   (`nbr04_export.py`) was edited while it ran. The stage was rerun after the protocol and code were committed;
   the L6D model and the directions were identical to the first run.
2. The `fitsets` and `profile` stages read no replacement score, and the protocol, committed before the
   `prepare` stage, quotes the fit-set numbers.
3. The rule was reviewed before any score and kept unchanged.

## What it means

- Personalizing the counterpart with 062's normal subspace absorbs part of an abnormal ECG's deviation even
  when the replaced region is hidden from the fit, because the abnormality in the rest of the record is
  enough to move the coordinates. Leave-region-out is therefore not a fix.
- Removing one mean direction per superclass reduces the absorption for MI and ST-T patterns, and with it
  the localization gain disappears. The localization gain of C6 and L6 and their absorption look like the
  same effect: the personalized template follows the record wherever the subspace can, which removes global
  differences and leaves local ones, but also makes abnormal records look partly normal.
- No arm passes the frozen rule, so P0 stays the page counterpart. Three counterparts now point the same way:
  the population template is the most selective and keeps the most component-level flips.

## Limitations

- Same as NBR-01 to NBR-03: development data, label-derived expectations, probability-scale attribution.
- The superclass directions are means; patterns with large within-class variation (LBBB) are not removed by
  them. A richer removal (a disease subspace per class) was not tested.
- The canonical-bin mapping of components (PR segment, QRS, ST and T segments) approximates the 066 timeline
  labels used for the replacement masks.

## Page export

`outputs/nbr04_leave_region_out_20261028/examples.json` (6,615,335 bytes; validated): arm L6 (no arm passed),
NBR-01's 16 SPH IDs in the same order. The `counterpart` array is a composite: each lead comes from its
whole-lead leave-region-out template (`counterpart_note`). Signals, components, unreplaced and self-swap
scores are NBR-01's. `summary.statistics` holds 3,081 statistics with 97.5% intervals for P0, C6, L6 and L6D on
SPH and, under `ptbxl_fold9/`, for P0, L6 and L6D on fold 9; `summary.decisions` holds parts A to D for both
arms, the tie statistic, `chosen: null` and `exported_arm: "L6"`. The page should keep NBR-01's file as its
default.

## Follow-ups (added to the backlog)

- `nbr_bounded_personalization` is lowered in value: neither leaving the region out nor removing mean
  disease directions kept localization without absorption.
- `nbr_disease_subspace_removal`: remove a per-class disease subspace (several principal directions of each
  superclass's residuals, especially conduction), not one mean direction, and test LBBB absorption first,
  label-free on the templates, before any scoring.
