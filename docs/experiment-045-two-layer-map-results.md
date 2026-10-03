# Experiment 045 results: a two-layer explanation map and a detection ensemble

Completed 1 October 2026 under the [frozen protocol](experiment-045-two-layer-map.md) (commit `cab7341`; run
from `555761a`). Saved scores only: nothing was trained or extracted. Besides the saved scores, the run read
only the 84 PVC records' PTB-XL waveforms, to rebuild 041's premature-beat windows. No calibration, test or
EchoNext record was read. Outputs are in `outputs/experiment045_two_layer_map_v1/` (`result.json` with every
input, source and protocol hash, `combined_map.npz`, `ensemble.npz`, `run.log`). The run took 170 seconds on
CPU.

## Readings

| Question | Prespecified test | Result | Reading |
| --- | --- | --- | --- |
| A. Two-layer map against `U_B` | keeps premature-beat localization, plus one gain | loses localization: -0.309 [-0.418, -0.202] | **Does not improve** |
| B. Ensemble E against v3's readout | SPH lower bound > 0, full >= -0.005 | SPH +0.0038 [0.0028, 0.0049], full +0.0058 | **Beats** |
| B. Ensemble E against R (secondary) | same rule | SPH +0.0055 [0.0042, 0.0069], full +0.0099 | **Beats** |

The combined map did not improve on `U_B`, so no example figures were drawn.

## Integrity

- The 042 `unit_scores.npz` and the 043 Stage 1 `predictions.npz` and Stage 2 `predictions.npz` and
  `token_maps.npz` match the hashes in their `result.json` files. 043 Stage 2's recorded hashes of 042's
  outputs match the files.
- The rows align: the 1,604 scored ECG IDs are the same in 042, in the 043 token maps and in the development
  rows. The two stages' development record IDs are equal. The Stage 2 SPH IDs equal the Stage 1 SPH IDs at the
  21,008 labeled positions. The premature-beat rule gave 73 ECGs, with 042's exclusions (0 with too few
  peaks, 11 without a premature beat).
- `U_B` recomputed with seed 42042 equals 042's `result.json` exactly, for every field checked: threshold,
  AUROC, AP, any red, mean red units, sensitivity at the budget with its interval, both lead contrasts, PVC
  count, hit, chance, and hit - chance with its interval.
- With 043's `map_metrics` and seed 43043, `U_B` and `attention_jepa` reproduce 043 Stage 2's `maps` block
  exactly, and `map_contrasts` reproduces its `attention_jepa_minus_U_B` block exactly.
- The saved logits reproduce 043's AUROC and AP of R, `logistic_concat` and `attention_jepa` on full,
  ordinary, hard and SPH exactly, and their differences from R (seed 43043) exactly.

## Thresholds

On the 1,001-value grid, the share of the 463 normals with either layer red is never 23 / 463. Near 5% the
count moves in steps of two (20, 22, 24, 26), because one quantile crosses an order statistic of both layers'
463 normal scores at the same q. The nearest achievable share is 24 / 463 = 0.0518, which 22 contiguous grid
values give (q = 0.9719 to 0.9740). Their median is **q = 0.9729**:

| Layer | Threshold at q = 0.9729 | Threshold alone (q = 0.95) | Normals red, layer alone at q |
| --- | ---: | ---: | ---: |
| 1, `U_B` | 1,829.6 | 1,158.8 | 13 (2.8%) |
| 2, `attention_jepa` | 0.395 | 0.275 | 13 (2.8%) |

Two normals are red in both layers, so 24 are red in all: the same budget as each single layer at its 95th
percentile.

## A. The two-layer map

Every interval here uses seed 45045, so the single-layer intervals differ slightly from 042's and 043's.
The point values are identical.

| Map | Hit | Chance | Hit - chance [95% interval] | Anterior contrast | Inferior contrast |
| --- | ---: | ---: | --- | --- | --- |
| `U_B` | 0.932 | 0.197 | +0.734 [0.673, 0.788] | +0.085 [-0.034, +0.202] | +0.115 [-0.000, +0.228] |
| `attention_jepa` | 0.027 | 0.147 | -0.119 [-0.157, -0.072] | +0.190 [0.082, 0.296] | +0.205 [0.111, 0.302] |
| combined | 0.603 | 0.177 | +0.425 [0.311, 0.539] | +0.226 [0.110, 0.328] | +0.212 [0.114, 0.309] |

| Map | Any red: normal | Positive (sensitivity [95% interval]) | PVC | Benign | Worst-unit AUROC |
| --- | ---: | --- | ---: | ---: | ---: |
| `U_B` | 0.052 | 0.286 [0.255, 0.317] | 0.774 | 0.077 | 0.740 |
| `attention_jepa` | 0.052 | 0.472 [0.435, 0.508] | 0.405 | 0.115 | 0.897 |
| combined | 0.052 | 0.423 [0.388, 0.458] | 0.679 | 0.115 | not applicable |

Anterior and inferior hits of the combined map: 0.541 of anterior-only against 0.315 of inferior-only
infarcts have an explanation lead in V1-V4; 0.362 against 0.151 have one in II, III or aVF. Mean red units per
ECG (both layers added): 0.13 in normals, 6.86 in positives, 12.38 in PVC ECGs and 0.15 in benign variants.

Share explained by layer 1 (`U_B` red at q): normals 0.028, positives 0.238, PVC ECGs 0.643, benign variants
0.038, anterior-only infarcts 0.274, inferior-only 0.154.

**Combined minus `U_B` alone, per ECG:**

| Contrast | Difference [95% interval] | Test | Passed |
| --- | --- | --- | --- |
| Hit - chance (73 PVC ECGs) | -0.309 [-0.418, -0.202] | lower bound > -0.10 | No |
| Anterior contrast of the combined map | +0.226 [0.110, 0.328] | lower bound > 0 | Yes |
| Benign any red (52) | +0.038 [-0.058, +0.135] | upper bound < 0 | No |
| Positive any red (843) | +0.138 [0.109, 0.166] | lower bound > 0 | Yes |

The combined map gains two of the three (lead and positive any red) but fails the required premature-beat
condition, so by the prespecified rule it does not improve on `U_B`.

## B. The detection ensemble

E is the mean of the `logistic_concat` logit and the `attention_jepa` 3-seed mean logit.

| Readout | Full development | Ordinary | Hard | SPH | SPH AP |
| --- | ---: | ---: | ---: | ---: | ---: |
| R (pipeline v2) | 0.9306 | 0.9536 | 0.7124 | 0.9387 | 0.9164 |
| `logistic_concat` (pipeline v3) | 0.9347 | 0.9566 | 0.7462 | 0.9404 | 0.9192 |
| `attention_jepa` | 0.9368 | 0.9548 | 0.8098 | 0.9404 | 0.9194 |
| E | 0.9405 | 0.9595 | 0.7950 | 0.9442 | 0.9245 |

| Paired AUROC difference (seed 45045) | SPH [95% interval] | Full development [95% interval] | Reading |
| --- | --- | --- | --- |
| E minus `logistic_concat` (primary) | +0.0038 [0.0028, 0.0049] | +0.0058 [0.0025, 0.0092] | beats |
| E minus R | +0.0055 [0.0042, 0.0069] | +0.0099 [0.0054, 0.0145] | beats |
| `logistic_concat` minus R | +0.0017 [0.0008, 0.0027] | +0.0041 [0.0008, 0.0075] | |
| `attention_jepa` minus R | +0.0017 [-0.0003, +0.0038] | +0.0063 [-0.0004, +0.0128] | |

## Interpretation

- **The two-layer map.** One 5% budget shared by two layers gives each layer about half of it: at q = 0.9729
  each layer marks 2.8% of normals instead of 5.2%. Layer 1's higher threshold drops `U_B` red on PVC ECGs
  from 77% to 64%. The rest fall through to the attention layer, which almost never points at the premature
  beat, so the hit rate falls from 93% to 60%. The attention layer brings what was expected: the anterior
  contrast rises from +0.085 to +0.226, with an interval above zero, and 42% of positives are red instead of
  29% at the same normal budget. Benign variants are marked as often as by the attention layer alone (11.5%,
  against 7.7% for `U_B`). The trade is built into the design: at a fixed joint budget, every normal ECG the
  attention layer marks is one that `U_B` cannot.
- **The ensemble.** Averaging the two readouts' logits beats each one. The SPH gain over pipeline v3's
  readout is small (+0.004 AUROC) but has a tight interval, and it holds on full development (+0.006). On the
  hard subset, E (0.795) sits between `logistic_concat` (0.746) and `attention_jepa` (0.810). E is an unfitted
  average, so nothing was tuned on development or SPH. But it needs both the xECG and ECG-JEPA features plus
  the attention head's tokens at inference.

## Caveats

- The positive any-red gain measures detection at a fixed normal budget, not localization.
- The attention layer marks only 8 leads; infarct location is a whole-ECG statement; the premature-beat rule
  is automatic; the benign set is small (52; one ECG moves the share by 1.9 points).
- E's referral budget, calibration and operating point were not assessed; the AUROC gain does not by itself
  change pipeline v3.
- Development and SPH were read by earlier experiments; the results are exploratory.

## Deviations

- The first full run computed every number and renamed its folder, then failed while writing the final
  "done" line to the old `.partial` log path. The runner was fixed (`555761a`, logging only), the first
  output was moved aside, and the experiment was run again. The second run's quantile, maps, contrasts and
  ensemble blocks equal the first run's exactly. The reported outputs are from the second run.
- None in the analysis. The protocol's caveat expected a 23 / 463 share; the grid could not reach it, and the
  rule chose 24 / 463, as prespecified.
