# NBR-04 results: no personalized counterpart replaces P0; removing disease directions trades localization for selectivity

28 October 2026, under the committed [protocol](nbr04-leave-region-out-protocol-2026-10-28.md) (`e0b32ee`;
code `ffd0e67`) and its Amendment 1 (`fea21be`, timing note `9221846`; code `72cdaff`, `3686841`).
Amendment 1 added a third arm, D6, at the user's request. The request was sent at 16:17 local time (23:17 UTC),
before both the NBR-03 protocol (16:26) and this protocol (17:10); the agent first saw it after committing the
two-arm report (`34b7bfb`, `0c8957b`, 17:50) and committed the amendment at 17:55, before any D6 score. The
amendment was therefore written with the L6 and L6D results known; D6 is the only unseen arm. Same branch as
NBR-03, stacked on #79. Every number comes from
`outputs/nbr04_leave_region_out_20261028/` (`analyze/results.json` for the original two-arm analysis,
`amendment_analyze/results.json` for the amended three-arm analysis, `score/integrity.json`,
`amendment_check/check.json`, `amendment_rescore/rescore.json`, `fitsets/fitsets.json`, the `prepare`,
`profile`, `encode_*` and `export` stages
and their `amendment_` counterparts). Intervals are whole-patient bootstraps, 2,000 draws, seed 20261028, with
every statistic of every arm of one dataset on the same draws: 98.33% for the amended decision, 97.5% for the
original two-arm decision, 95% for the factorial effects. No draw was skipped for any decision or factorial
statistic. Scores are `all_age_blend` probabilities unless stated.

## Answer

- No arm replaces P0, under the amended three-arm rule or under the original two-arm rule. The page keeps
  NBR-01's population counterpart; the export carries L6, as fixed for this case.
- D6 (disease directions removed, whole-record coordinates) also fails: selectivity minus P0 -0.060 [-0.120,
  -0.003] (A), component flips minus P0 -0.058 [-0.083, -0.032] (C), and no PTB-XL confirmation (selectivity
  point -0.018).
- In the 2 x 2 factorial, leaving the region out does nothing for selectivity or localization on either
  dataset (SPH selectivity +0.008 [-0.003, +0.019] without removal, territory -0.002 [-0.009, +0.005]).
  Removing the four superclass directions raises selectivity (SPH +0.092 [+0.041, +0.147]; PTB-XL +0.135
  [+0.082, +0.190]) but lowers the territory contrast (SPH -0.091 [-0.135, -0.051]) and the top-1 territory hit
  (SPH -0.192 [-0.283, -0.102]). The interactions are near zero except for component flips.
- So the personalized template's localization gain and its absorption of the abnormality go together: what
  removes one removes the other, and neither variant beats the population template on selectivity.

## Addendum of 28 October 2026: Amendment 1, three arms

### Amended primary decision (98.33% intervals, Bonferroni over three arms)

| Part | Rule | L6 | D6 | L6D |
| --- | --- | --- | --- | --- |
| A. Selectivity, arm - P0 (SPH) | lower > 0 | -0.144 [-0.209, -0.089], failed | -0.060 [-0.120, -0.003], failed | -0.058 [-0.119, -0.001], failed |
| B. Territory, arm - P0 (SPH) | lower > -0.05 | +0.078 [+0.031, +0.132], passed | -0.011 [-0.038, +0.015], passed | -0.010 [-0.034, +0.013], passed |
| C. Component flip rate, arm - P0 (SPH) | lower > -0.05 | -0.076 [-0.107, -0.047], failed | -0.058 [-0.083, -0.032], failed | -0.048 [-0.074, -0.023], failed |
| D. PTB-XL selectivity, arm - P0 | point > 0 | -0.148, failed | -0.018, failed | -0.022, failed |
| D. PTB-XL territory, arm - P0 | lower > -0.05 | +0.011 [-0.020, +0.040], passed | -0.011 [-0.033, +0.010], passed | -0.003 [-0.024, +0.017], passed |
| Replaces P0 | A, B, C and D | no | no | no |

No arm passed, so the choice rule was not used; P0 stays and L6 is exported. Recomputed at 98.33%, none of
L6's or L6D's A-D outcomes changed from the 97.5% analysis: for both, A failed, B passed, C failed, the PTB-XL
territory part passed and the PTB-XL selectivity part failed, so D stayed unconfirmed. L6D's part C lower bound
moved from -0.072 to -0.074, and its part A lower bound from -0.115 to -0.119.

All five arms on SPH (`all_age_blend`, point values; selectivity and territory with 98.33% intervals):

| Arm | Leave-out | Removal | Selectivity | Territory | Top-1 hit | Component flip | Full flip | Abnormal lead change | Normal lead change |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| P0 | | | 0.935 [0.786, 1.105] | +0.306 [+0.235, +0.378] | 0.490 | 0.395 | 0.763 | 0.0207 | 0.0221 |
| C6 | no | no | 0.783 [0.656, 0.928] | +0.386 [+0.302, +0.467] | 0.712 | 0.240 | 0.362 | 0.0147 | 0.0188 |
| L6 | yes | no | 0.791 [0.662, 0.938] | +0.384 [+0.300, +0.467] | 0.712 | 0.319 | 0.772 | 0.0150 | 0.0190 |
| D6 | no | yes | 0.875 [0.738, 1.035] | +0.295 [+0.220, +0.371] | 0.519 | 0.337 | 0.563 | 0.0179 | 0.0204 |
| L6D | yes | yes | 0.877 [0.738, 1.040] | +0.296 [+0.221, +0.371] | 0.538 | 0.347 | 0.772 | 0.0182 | 0.0208 |

On PTB-XL fold 9: P0 0.982 / +0.098, C6 0.829 / +0.109, L6 0.834 / +0.109, D6 0.964 / +0.087, L6D 0.960 / +0.095
(selectivity / territory). D6's full replacement unflags 56.3% of flagged abnormal SPH ECGs (C6 36.2%, P0 76.3%;
the leave-out arms equal C0 by construction); its abnormal full drop is 0.384. At least one D6 coordinate is
clipped for 64.7% of LBBB ECGs on SPH (C6 79.4%) and 6.7% of normals.

### Factorial effects (paired, 95%)

| Effect (`all_age_blend`) | Selectivity | Territory | Top-1 hit | Component flip |
| --- | --- | --- | --- | --- |
| SPH: leave-out without removal (L6 - C6) | +0.008 [-0.003, +0.019] | -0.002 [-0.009, +0.005] | +0.000 [-0.027, +0.028] | +0.079 [+0.057, +0.102] |
| SPH: leave-out with removal (L6D - D6) | +0.002 [-0.014, +0.018] | +0.001 [-0.005, +0.008] | +0.019 [+0.000, +0.051] | +0.010 [-0.007, +0.028] |
| SPH: removal without leave-out (D6 - C6) | +0.092 [+0.041, +0.147] | -0.091 [-0.135, -0.051] | -0.192 [-0.283, -0.102] | +0.097 [+0.068, +0.130] |
| SPH: removal with leave-out (L6D - L6) | +0.087 [+0.036, +0.141] | -0.088 [-0.131, -0.048] | -0.173 [-0.260, -0.086] | +0.028 [+0.002, +0.055] |
| SPH: interaction | -0.006 [-0.025, +0.013] | +0.003 [-0.007, +0.013] | +0.019 [-0.017, +0.061] | -0.069 [-0.096, -0.043] |
| PTB-XL: leave-out without removal | +0.005 [-0.006, +0.017] | +0.001 [-0.008, +0.010] | -0.003 [-0.014, +0.007] | +0.071 [+0.039, +0.106] |
| PTB-XL: leave-out with removal | -0.004 [-0.017, +0.007] | +0.008 [+0.003, +0.014] | -0.007 [-0.020, +0.006] | -0.020 [-0.042, +0.003] |
| PTB-XL: removal without leave-out | +0.135 [+0.082, +0.190] | -0.021 [-0.045, +0.002] | -0.085 [-0.129, -0.039] | +0.136 [+0.090, +0.183] |
| PTB-XL: removal with leave-out | +0.126 [+0.073, +0.180] | -0.014 [-0.037, +0.010] | -0.088 [-0.132, -0.043] | +0.045 [+0.005, +0.088] |
| PTB-XL: interaction | -0.010 [-0.027, +0.006] | +0.008 [-0.003, +0.018] | -0.003 [-0.020, +0.013] | -0.091 [-0.133, -0.052] |

Leaving the region out raises the whole-component flip rate only when the disease directions are kept (SPH
+0.079 against +0.010 with removal), which is what makes that interaction negative.

## Original two-arm analysis (frozen protocol, 97.5% intervals)

The following sections report the analysis as frozen at `e0b32ee`, unchanged.

### Two-arm decision (SPH, D on PTB-XL fold 9)

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

### The arms on SPH (`all_age_blend`)

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

### PTB-XL fold 9 (`all_age_blend`)

| Arm | Selectivity | Territory | Minus energy | Wave | Top-1 hit | Component flip |
| --- | --- | --- | --- | --- | ---: | ---: |
| P0 (NBR-03) | 0.982 [0.838, 1.158] | +0.098 [+0.044, +0.148] | +0.038 | +0.196 | 0.424 | 0.466 |
| L6 | 0.834 [0.703, 0.992] | +0.109 [+0.051, +0.166] | +0.065 | +0.110 | 0.469 | 0.358 |
| L6D | 0.960 [0.819, 1.129] | +0.095 [+0.046, +0.146] | +0.038 | +0.190 | 0.381 | 0.403 |

Contrasts against P0: L6 selectivity -0.148 [-0.220, -0.087], territory +0.011 [-0.018, +0.038], component flip
-0.108 [-0.155, -0.063]; L6D selectivity -0.022 [-0.066, +0.018], territory -0.003 [-0.023, +0.017], component
flip -0.063 [-0.095, -0.031]. On PTB-XL even L6's localization gain is small.

### Clipping and template movement

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
| Amendment: NBR-02's SPH C6 counterparts rebuilt by the amendment's code path | 0.0 mV on 100 records |
| Amendment: NBR-04's P0, L6 and L6D point statistics recomputed by the amendment's analysis | 4,974 of 4,974 identical, including both P0 primaries |
| Amendment: D6 and C6 variant SHA-256 between preparation and encoding | all matched (1,037 SPH, 862 fold 9) |
| Amendment: saved L6 and L6D scores re-scored from their saved measurements and xECG features | largest difference 0.0, energies identical, on both samples |

## Run

Fit set and directions 343 s on the committed rerun (1,062 s on the first run). Profile: preparation 2.07
CPU-seconds per record and encoding 1.25 s per record (104 variants); L6 full equal to C0 for 10 of 10.
Preparation of 1,899 ECGs 191 s with 16 workers; encoding 949 and 950 ECGs in 1,234 and 1,275 s on the two
GPUs in parallel, each under its lock; bootstrap and analysis 215 s.

Amendment 1: profile 1.13 CPU-seconds and 0.78 s of encoding per record (20 ECGs); preparation of D6 on both
samples and C6 on fold 9 in 129 s with 16 workers; encoding 949 and 950 ECGs in 727 and 825 s on the two GPUs;
scoring and the three-arm analysis 332 s.

## Deviations and disclosures

1. The first `fitsets` run finished its computation but failed to write its receipt, because a source file
   (`nbr04_export.py`) was edited while it ran. The stage was rerun after the protocol and code were committed;
   the L6D model and the directions were identical to the first run.
2. The `fitsets` and `profile` stages read no replacement score, and the protocol, committed before the
   `prepare` stage, quotes the fit-set numbers.
3. The rule was reviewed before any score and kept unchanged.
4. Amendment 1's request was sent at 16:17 local time (23:17 UTC), before both protocols, and the agent first
   saw it at about 17:50, after the two-arm report. The amendment was committed at 17:55 with the L6 and L6D
   results known. It added D6, which had not been scored, and replaced the decision rule by the three-arm rule.
   The wider 98.33% intervals could only make passing harder for L6 and L6D, and none of their outcomes changed.
5. The amendment's export replaced the first `examples.json`. The arm (L6), the 16 examples, their counterparts
   and scores are identical in both versions; only the summary changed (all five arms, the amended decision and
   the factorial effects).

## What it means

- Personalizing the counterpart with 062's normal subspace absorbs part of an abnormal ECG's deviation even
  when the replaced region is hidden from the fit, because the abnormality in the rest of the record is
  enough to move the coordinates. Leave-region-out is therefore not a fix: in the factorial it changes neither
  selectivity nor localization on either dataset.
- Removing one mean direction per superclass is the only factor that moves selectivity, and it removes the
  localization gain by about the same amount it was gained. The localization gain of C6 and L6 and their
  absorption behave as one effect: the personalized template follows the record wherever the subspace can,
  which removes global differences and leaves local ones, but also makes abnormal records look partly normal.
- No arm passes either rule, so P0 stays the page counterpart. Four personalized variants now point the same
  way: the population template is the most selective and keeps the most component-level flips.

## Limitations

- Same as NBR-01 to NBR-03: development data, label-derived expectations, probability-scale attribution.
- The superclass directions are means; patterns with large within-class variation (LBBB) are not removed by
  them. A richer removal (a disease subspace per class) was not tested.
- The canonical-bin mapping of components (PR segment, QRS, ST and T segments) approximates the 066 timeline
  labels used for the replacement masks.

## Page export

`outputs/nbr04_leave_region_out_20261028/examples.json` (7,276,396 bytes; validated), written under Amendment
1: arm L6 (no arm passed), NBR-01's 16 SPH IDs in the same order. The `counterpart` array is a composite: each
lead comes from its whole-lead leave-region-out template (`counterpart_note`). Signals, components,
unreplaced and self-swap scores are NBR-01's. `summary.statistics` holds 5,810 statistics, each with 98.33%
(`low`, `high`) and 95% (`low95`, `high95`) intervals, for P0, C6, L6, D6 and L6D on SPH and, under
`ptbxl_fold9/`, on fold 9, including the `factorial/` effects; `summary.decisions` holds parts A to D for the
three arms, `passing: []`, `chosen: null` and `exported_arm: "L6"`. The page should keep NBR-01's file as its
default.

## Follow-ups (added to the backlog)

- `nbr_bounded_personalization` is lowered in value: neither leaving the region out nor removing mean
  disease directions kept localization without absorption.
- `nbr_disease_subspace_removal`: remove a per-class disease subspace (several principal directions of each
  superclass's residuals, especially conduction), not one mean direction, and test LBBB absorption first,
  label-free on the templates, before any scoring.
