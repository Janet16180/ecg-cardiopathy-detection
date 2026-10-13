# Experiment 070 results: replacement faithfulness test of the localization maps

Run on 13 October 2026 under the [protocol](experiment-070-replacement.md), committed as 4c196c3 before any
replacement score. Every number below comes from `outputs/experiment070_replacement_v1/result.json` (written by
the `analyze` phase at commit 4600c04), `detector_a.json`, `score_a.json`, `score_b.json`, `prepare.json` and
`post_hoc_fresh_random.json`. Intervals are 95% whole-patient bootstraps, 2,000 draws, seed 61061. S(k) is the
logit drop after replacing a map's top k% of lead-samples by the normal counterpart, minus the mean drop after
replacing 8 matched random regions (same lead-samples per lead and per wave label, moved to random leads and
positions inside the same wave label). Higher means the detector relies more on the map's region than on a
region of the same size and wave composition.

## Answer

- Every map except `residual056` points to regions that the independent detector relies on more than matched
  random regions: S(10%) is above 0 for 12 of the 14 maps on 1,500 abnormal `evaluation` records.
- The primary comparison goes against the confirmed map. `units_v2` minus `qrs_prior` on S(10%) is -0.608
  [-0.702, -0.518]: by this test the disease-blind QRS indicator is more faithful to detector (a) than `units_v2`.
  Detector (b), the 065 ensemble, gives the same sign: -0.138 [-0.179, -0.098].
- The test does not rank maps the way the ECGAD marks do. Across the 14 maps, the Spearman correlation between
  S(10%) and ECGAD development A2 is 0.174 [-0.007, 0.327], and -0.033 [-0.258, 0.159] without the `random` map.
- The test mostly rewards maps that select where the counterpart differs most from the record. The maps with the
  highest S(10%) are the 063 reconstruction error `raw` (1.607) and the disease-blind `amplitude` map (1.492).
  Across maps, S(10%) under detector (a) correlates 0.758 with how much more signal energy the map region
  replaces than its controls (a post-hoc reading).
- One integrity check failed: the `random` map's S(10%) is -0.0054 [-0.0103, -0.0010], whose interval excludes 0.
  It is about 1% of the smallest other faithful map's S(10%) (`U_B`, 0.529), it changes no decision, and its cause
  was not found (see Integrity).

## Main table (set A: 1,500 abnormal `evaluation` records, one per patient)

| Map | ECGAD dev A2 | S(10%), detector (a) | Verdict (a) | W (top 3 leads), (a) | S(10%) on normals, (a) | S(10%), detector (b) |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `random` (061) | 0.500 | -0.005 [-0.010, -0.001] | anti-faithful | -0.037 [-0.109, +0.031] | +0.001 | -0.005 [-0.010, -0.001] |
| `amplitude` (061) | 0.698 | +1.492 [+1.380, +1.604] | faithful | -0.362 [-0.448, -0.276] | +0.266 | +0.168 [+0.130, +0.206] |
| `qrs_prior` (061) | 0.742 | +1.186 [+1.090, +1.284] | faithful | -0.308 [-0.395, -0.221] | +0.145 | +0.246 [+0.224, +0.268] |
| `U_B` (061) | 0.632 | +0.529 [+0.462, +0.592] | faithful | +0.262 [+0.173, +0.355] | +0.044 | +0.005 [-0.033, +0.046] |
| `residual056` (061) | 0.680 | +0.031 [-0.008, +0.072] | not separable | -0.145 [-0.227, -0.065] | -0.036 | +0.125 [+0.077, +0.173] |
| `normative_v1` (062) | 0.694 | +0.893 [+0.819, +0.970] | faithful | +0.699 [+0.611, +0.791] | +0.268 | +0.196 [+0.162, +0.232] |
| `qrs_lexicographic` (062) | 0.753 | +0.727 [+0.669, +0.786] | faithful | +0.678 [+0.586, +0.773] | +0.090 | +0.132 [+0.113, +0.151] |
| `raw` (063) | 0.709 | +1.607 [+1.503, +1.715] | faithful | +0.228 [+0.136, +0.322] | +0.375 | +0.098 [+0.062, +0.132] |
| `dino_dist` (064) | 0.712 | +1.060 [+0.975, +1.150] | faithful | +0.207 [+0.116, +0.296] | +0.067 | +0.074 [+0.036, +0.112] |
| `mil_lse` (065) | 0.662 | +1.159 [+1.090, +1.226] | faithful | +1.229 [+1.147, +1.313] | +0.347 | +0.593 [+0.550, +0.636] * |
| `factor` (066) | 0.725 | +1.374 [+1.305, +1.447] | faithful | +1.249 [+1.170, +1.326] | +0.308 | +0.939 [+0.891, +0.984] * |
| `textbook` (066) | 0.749 | +0.557 [+0.488, +0.628] | faithful | +0.633 [+0.542, +0.725] | +0.050 | +0.246 [+0.213, +0.278] * |
| `units_learned` (066) | 0.760 | +0.668 [+0.607, +0.727] | faithful | +0.774 [+0.682, +0.862] | +0.092 | +0.122 [+0.089, +0.155] * |
| `units_v2` (068) | 0.767 | +0.577 [+0.518, +0.636] | faithful | +0.709 [+0.617, +0.799] | +0.068 | +0.108 [+0.073, +0.143] * |
| `units_v2` minus `qrs_prior` | | -0.608 [-0.702, -0.518] | qrs_prior more faithful | +1.017 [+0.889, +1.141] | -0.077 [-0.113, -0.042] | -0.138 [-0.179, -0.098] |

\* Circular under detector (b): `mil_lse` is the detector's own evidence; `factor`, `textbook`, `units_learned`
and `units_v2` use 065 bag probabilities or evidence. ECGAD dev A2 is the mean of each map's saved per-record
table (92 records, 90 patients). Normals: 400 `evaluation` clean normals, one per patient; their intervals are in
`result.json`.

## The detectors

- Detector (a): a 624,449-parameter 1D ResNet trained on 7,436 `fit` records with random 9-second crops, stopped on
  1,321 held-out `fit` records of 1,149 patients. Its held-out AUROC is 0.972 (best epoch 9 of 16; 35 s of
  training), above the declared 0.85, so it was adopted. On the whole `evaluation` detection set (4,374 records,
  2,974 positive) its AUROC is 0.967. No `fit` record is an ECGAD record.
- Detector (b): the 065 three-network `mil_lse` ensemble, abnormal logit = minus the mean NORM bag logit. Its
  float32 GPU path differs from 065's bf16 CPU-preprocessing path by at most 0.013 logit (mean 0.003) on set A.
- Mean original logit on set A: 5.77 [5.52, 6.00] under (a) and 2.76 [2.62, 2.89] under (b); on normals -1.40 and
  -1.71.

## The normal counterpart works as intended

Replacing the whole scored range (all 12 leads, samples 100-4899) by the counterpart lowers the set-A logit by
7.11 [6.90, 7.30] under detector (a), from 5.77 to -1.33. That is close to the mean logit of the untouched
normals (-1.40). Under (b) the drop is 3.52 [3.42, 3.62]. On normals the same full replacement also lowers the
logit, by 1.55 [1.43, 1.66] under (a), because the population template is a more typical normal beat than most
individual normals. Examples are in `operator_examples.png` (three set-A records, `units_v2` at 10% and one
matched control).

## How S depends on the region size

| Detector (a), set A | S(5%) | S(10%) | S(20%) | AOPC_matched | AOPC_raw |
| --- | ---: | ---: | ---: | ---: | ---: |
| `qrs_prior` | +1.583 | +1.186 | +0.186 | +0.985 | 3.201 |
| `units_v2` | +0.646 | +0.577 | +0.311 | +0.512 | 2.781 |
| `units_v2` minus `qrs_prior` | -0.936 [-1.044, -0.832] | -0.608 [-0.702, -0.518] | +0.125 [+0.065, +0.189] | -0.473 [-0.543, -0.401] | -0.420 [-0.505, -0.337] |
| `raw` | +1.526 | +1.607 | +1.401 | +1.511 | 3.445 |
| `amplitude` | +1.423 | +1.492 | +1.499 | +1.472 | 3.416 |
| `factor` | +1.048 | +1.374 | +1.495 | +1.306 | 3.384 |
| `random` | -0.004 | -0.005 | -0.003 | -0.004 | 1.143 |

- `qrs_prior`'s S falls from 1.583 at 5% to 0.186 at 20%. Its top 10% is 96% QRS (composition below), and at 20%
  it covers nearly every QRS lead-sample, so its matched control, which must stay inside QRS, replaces almost the
  same samples. A composition-matched control cannot reward a map that covers a whole wave label.
- The paired difference changes sign at 20% (+0.125 [+0.065, +0.189]) for that reason, but over the three sizes
  `units_v2` stays below `qrs_prior` (AOPC_matched -0.473 [-0.543, -0.401]). Without any control, AOPC_raw also
  favours `qrs_prior` (-0.420 [-0.505, -0.337]).

Composition of each map's top 10% on set A (share of P, PQ, QRS, ST, T, other): `qrs_prior` 0.00 / 0.01 / 0.96 /
0.02 / 0.00 / 0.00; `units_v2` 0.01 / 0.02 / 0.65 / 0.12 / 0.19 / 0.00; `amplitude` 0.02 / 0.00 / 0.61 / 0.02 /
0.35 / 0.00; `raw` 0.03 / 0.01 / 0.64 / 0.04 / 0.25 / 0.03; `random` 0.15 / 0.08 / 0.16 / 0.11 / 0.25 / 0.24.

## Lead localization alone (whole-lead variant)

Replacing a map's top 3 leads against 20 random 3-lead sets ranks the maps differently from S(10%):

- Highest W under detector (a): `factor` 1.249, `mil_lse` 1.229, `units_learned` 0.774, `units_v2` 0.709,
  `normative_v1` 0.699, `qrs_lexicographic` 0.678, `textbook` 0.633.
- Below 0: `amplitude` -0.362, `qrs_prior` -0.308 and `residual056` -0.145. Their top leads are the largest leads
  (for `qrs_prior` through its amplitude tiebreak), and replacing those leads lowers the logit less than replacing
  3 random leads.
- `units_v2` minus `qrs_prior` on W is +1.017 [+0.889, +1.141] under (a) and +0.801 [+0.709, +0.898] under (b).
  On which leads matter, the test agrees with ECGAD's lead reading A3, where the 065-based maps also beat
  `qrs_prior`.

## Validation against the ECGAD marks

| Reading | Spearman, S(10%) against development A2 (14 maps) |
| --- | ---: |
| Set A, detector (a), joint patient bootstrap (declared) | +0.174 [-0.007, +0.327] |
| Same without `random` | -0.033 [-0.258, +0.159] |
| Same, bootstrap over maps | +0.174 [-0.545, +0.701] |
| Set A, detector (b) | +0.226 [+0.077, +0.424] |
| Set A, detector (b), without `random` | +0.033 [-0.154, +0.280] |
| Set E (198 ECGAD development records, exploratory), detector (a) | +0.103 [-0.095, +0.292] |
| Set E, detector (b) | +0.064 [-0.064, +0.389] |

- By the declared rule, no agreement is shown: the interval for the declared reading contains 0. Under detector
  (b) the interval lies above 0, but the agreement disappears without `random` (+0.033), so it comes from the
  one map at A2 0.5 and S 0.
- `replacement_vs_a2.png` shows the scatter. The maps with the highest A2 (`units_v2`, `units_learned`,
  `qrs_lexicographic`, `textbook`) sit in the middle or lower half on S(10%) under both detectors.
- What this means, as declared: the regions that a detector relies on are not the regions that the cardiologists
  marked, or the test measures something else. The next section points to the second: under detector (a), much of
  S follows how much signal a replacement changes.

## What S measures (post-hoc readings, not declared)

Replaced energy, the summed squared counterpart difference over a region, is not matched by the control. On set
A, the mean log ratio of a map region's energy to its controls' energy is 0.83 for `amplitude`, 0.71 for `raw`,
0.59 for `normative_v1`, 0.54 for `dino_dist`, 0.42 for `qrs_prior`, 0.07 for `units_v2`, -0.02 for `textbook`,
-0.36 for `residual056` and -0.001 for `random`.

- Across the 14 maps, S(10%) under detector (a) correlates 0.758 with this log ratio on set A (0.692 on normals).
  Within each map, record by record, the correlation is 0.18-0.65, and it is 0.286 even for `random`.
- Under detector (b) the across-map correlation is 0.033 on set A (0.525 on normals), and within maps it is
  -0.13 to 0.26. The 065 networks respond to their own evidence regions (`factor` 0.939, `mil_lse` 0.593) rather
  than to replaced energy.
- So under detector (a), a map that selects high-amplitude QRS and T samples, where a population template differs
  most from an individual record, scores high whether or not it knows anything about disease. That explains
  `amplitude` and `raw` at the top and the textbook-painting maps, whose regions replace about as much energy as
  their controls, in the lower half.
- Normals: S(10%) is also above 0 on clean normals for most maps (for example `raw` 0.375, `mil_lse` 0.347,
  `amplitude` 0.266), about a quarter of the set-A value for those maps. For `units_v2` (0.068 against 0.577) and
  `qrs_prior` (0.145 against 1.186) the normal value is smaller relative to set A.

Mean S(10%) under detector (a) by detection superclass of the set-A records (descriptive; MI 565, STTC 657, CD
644, HYP 270 records; a record counts in each superclass it has): `qrs_prior` 1.34 / 0.68 / 2.04 / 1.24,
`units_v2` 0.57 / 0.37 / 0.90 / 0.45, `factor` 1.39 / 1.25 / 1.65 / 1.36, `amplitude` 1.74 / 0.93 / 2.40 / 1.64.
Every map except `random` and `residual056` has its highest S in conduction disturbances and its lowest in ST/T
changes; `factor` varies least (1.25 to 1.65).

## Integrity

| Check | Result |
| --- | --- |
| 1. Detector (a) held-out AUROC at least 0.85 | 0.972; passed |
| 2. Replacing nothing | the replaced signal equals the record bit for bit (unit test); logits of the zero-mask pass differ from a separate forward pass by at most 3.8e-6 (a) and 3.4e-5 (b), from batch composition; passed |
| 3. `random` S(10%) interval contains 0 | -0.0054 [-0.0103, -0.0010] under (a), -0.0055 [-0.0105, -0.0008] under (b); **failed** |
| 4. Full replacement lowers the set-A logit | 7.11 [6.90, 7.30] under (a); passed |
| 5a. 065 bag logits recomputed with `holdout067.mil_summaries` | identical to 065's saved bags on all 2,098 records (largest difference 0.0); passed |
| 5b. Rebuilt maps against saved maps | all 10 rebuilt maps identical to the scored development maps on 198 records, to 066/068's float32 evaluation maps (1,500 and 400 records), to 062's `qrs_lexicographic`, and to 061's float16 maps on the 548 set-A records they cover (largest difference 0.0); U_B, which exceeds the float16 range, rechecked on 50 records with 061's clipping: identical; passed |
| 5c. Saved development `qrs_prior` and `units_v2` maps rescored from the development marks | per-record A2 within 2.2e-16 of the saved tables (92 records); passed |
| 5d. A2 table means | equal the published values for all 14 maps (to 5e-5) and the six 067 declared values (to 1e-12); passed |
| 6. Detector (b) GPU path against the 065 CPU path | largest difference 0.013 logit (described, no threshold) |

Check 3 failed by the declared rule, so the control is flagged. What was checked afterwards:

- The controls are exact bijections of the scored lead-samples: every saved lead permutation and every
  within-label source map is a permutation (all records, all 8 draws). So the expected S of a uniform random
  region is exactly 0 for any detector.
- The replaced energy, a detector-free statistic, shows no bias for the `random` map (mean difference against
  its controls +1.55 mV^2, z = 1.5) or for 6,000 fresh uniform subsets (z = 0.45).
- Fresh uniform subsets under detector (a) (10 per set-A record, each against that record's 8 saved controls,
  `post_hoc_fresh_random.json`) give a mean S(10%) of -0.0025 (standard error 0.0008).
- The two detectors' per-record `random` scores correlate only 0.10, yet both means are -0.005.

No explanation was found. The deviation is 0.0025-0.005 logit, against S(10%) of 0.53-1.61 for the 12 maps
other than `random` and `residual056`, and 0.031 for `residual056`. A bias of that size could only move a verdict
whose interval lies within about 0.005 of 0; none does. A bias shared by all maps would also cancel in the
primary paired difference (-0.608).

## Deviations and disclosures

- The scoring of detector (b) first ran in 40-record chunks per GPU-lock acquisition; another experiment held the
  lock for long stretches, so after 120 records it was stopped and resumed from its saved progress with the lock
  held for up to 10 minutes of continuous computing per acquisition (commit 4600c04). The 120 records already
  scored were kept. Scoring code is unchanged; only the locking loop changed.
- The analysis and figures phases, the replaced-energy descriptive and the fresh-subset diagnostic were written
  after the protocol and after the first scores were seen. The energy reading and the fresh-subset diagnostic are
  post hoc and marked as such; no decision uses them.
- The U_B float16 comparison stored during preparation saturated (U_B reaches about 6e7, 061 clipped its float16
  file to 65,504). It was replaced by a recheck with 061's clipping and cast order on 50 records.
- Detector (b) runs in float32 instead of 065's bf16, with 065's filter as an exact matrix (declared).
- The analysis code repeats about 20 lines of 067's map-building runner because scripts are not imported.
- No ECGAD holdout record was mapped or read; ECGAD development marks were read only to rescore two saved maps
  (check 5c). Nothing was fitted on ECGAD or `evaluation` records.

## Runtime

| Stage | Time |
| --- | ---: |
| Detector (a) profile and training (GPU) | 35 s of training; 0.017 s per step, 0.24 GB peak |
| Preparation: 065 evidence (GPU, 13 s), maps, counterparts and plans (8 processes) | 207 s |
| Scoring, detector (a), 2,098 records x 414 passes | 35 s of GPU time |
| Scoring, detector (b) | about 40 minutes of GPU time; 1.8 hours of wall time with lock waits |
| Analysis | 41 s |

Large intermediates (3.8 GB) are in `/tmp/ecg070-work/full`.

## What it means

- A label-free replacement test can be built: the counterpart is accepted as normal by an independent detector
  (fully replaced abnormal records score like normals), and composition-matched controls make the `random` map
  score near 0.
- As a judge of localization maps, it disagrees with the cardiologist marks. Under detector (a) it mainly rewards
  maps that select where a population template differs most from the record (high-amplitude QRS and T), so the
  disease-blind `amplitude` map ranks second. Under detector (b) it rewards the maps built from that detector's own
  evidence. Neither ranking follows A2.
- The confirmed `units_v2` is less faithful than `qrs_prior` on S(10%) under both detectors, but it picks leads
  that matter to both detectors far better (W +1.017 and +0.801). That repeats the ECGAD finding: the 065-based
  maps carry lead information, and their time placement inside the lead is not what a detector relies on most.
- Faithfulness to a model and agreement with a cardiologist are different targets. This test cannot replace new
  marks for choosing a localization map.

## Follow-up ideas (for the backlog)

1. Energy-matched control: draw control regions matched on wave label and on replaced energy (or score the drop
   per unit of replaced energy) and retest whether any map beats its control beyond energy.
2. Record-specific counterpart: use the record's own dominant normal-looking beats (or the 062 rank-6 projection)
   instead of the population mean, so replacement changes only the abnormal part and normals are left almost
   unchanged.
3. Find the source of the small negative `random` bias (-0.0025 to -0.005) with a CPU replay of a few records in
   float64 and a check of the taper at the scored-range edges.
4. Use the whole-lead variant as a label-free lead-localization check for future maps; it agrees with ECGAD A3 in
   ranking the 065-based maps above `qrs_prior`.
5. Insertion test: insert a map's region from the record into its normal counterpart and measure the rise in the
   abnormal logit, which is less sensitive to how much a deletion changes the signal.
