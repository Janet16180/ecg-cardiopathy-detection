# NBR-01 results: the blend's lead evidence follows the MI territory, beyond replaced energy

28 October 2026, under the committed [protocol](nbr01-normal-beat-replacement-protocol-2026-10-28.md)
(`8c07b07`) and its Amendment 1 (`8064761`, written after integrity check 2 failed and before any replacement
score was read). Code `5c7e882`, `76012cd` and `eef3380`; post-hoc descriptives `60c65f7`. Every number comes
from `outputs/nbr01_normal_beat_replacement_20261028/` (`analyze/results.json`, `analyze/posthoc.json`,
`analyze/counts.csv`, `score/integrity.json`, `parity/parity.json`, `counterpart/counterpart.json`,
`profile/profile.json`, `select/selection.json`, `prepare/prepare.csv`). Intervals are 95% whole-patient
bootstraps, 2,000 draws, seed 20261028, all statistics on the same draws; no draw was skipped for any
statistic. Scores are probabilities; a drop is the original score minus the score after replacement.

## Answer

- Primary endpoint passed. For `all_age_blend`, the share of whole-lead evidence in the labelled MI territory
  exceeds the share in the same leads for the other territory by +0.306 [+0.248, +0.363] (inferior MI in II,
  III, aVF against anterior MI in V1-V4; 49 and 55 patients).
- Key qualifier passed. The disease-blind energy map gives +0.115 [+0.078, +0.153], and the paired difference,
  model minus energy, is +0.190 [+0.134, +0.246]. Unlike Experiment 070's detector, the blend's whole-lead
  evidence follows the labelled territory more than it follows where the normal counterpart changes the
  signal most.
- The localization is coarse. Inferior evidence sits mostly in lead II (mean share 0.37) and aVR (0.24),
  with little in III (0.04) and aVF (0.08); anterior evidence is spread over aVR (0.21) and V1-V5 (0.09-0.11
  each). The largest single-lead drop lies in the expected leads for 49.0% of territory ECGs, no better than
  the energy map (56.7%, difference -0.077 [-0.214, +0.064]).
- Waves: replacing the QRS dominates for RBBB (mean QRS share 0.79, V1 0.68 of the lead evidence) and the
  ST-T dominates for ST-T statements (0.85). The wave contrast is +0.547 [+0.509, +0.584] against +0.178 for
  energy.
- Single regions rarely remove a referral: among 668 flagged abnormal ECGs, one grid cell does so for 8.5%,
  one whole lead for 12.4%, one whole component for 39.5%, and the full replacement for 76.3%. The page
  should show graded drops, not only flips.
- The self-swap control changes the score a fifth to a seventh as much as the counterpart, but its lead
  shares also show a weaker territory contrast (+0.109 [+0.014, +0.209]); the model-minus-self-swap gap is +0.196
  [+0.099, +0.290] (post hoc).
- The PTB-only default (`ptb_baseline_blend`) has larger point values on every contrast: territory +0.450
  [+0.386, +0.510], +0.335 [+0.269, +0.394] beyond energy. No paired comparison between the two models was
  declared.

## Primary endpoint: territory discrimination

Lead attribution is the positive part of the 12 whole-lead drops normalized to shares (no `all_age_blend`
ECG had zero positive drops). The contrast averages the two territory groups SPH can label; SPH has no
lateral MI statement.

| Score | Model contrast | Energy map | Model minus energy |
| --- | --- | --- | --- |
| `all_age_blend` (primary) | +0.306 [+0.248, +0.363] | +0.115 [+0.078, +0.153] | +0.190 [+0.134, +0.246] |
| `all_age` HGB | +0.226 [+0.176, +0.281] | +0.115 | +0.111 [+0.061, +0.164] |
| `all_age` xECG | +0.375 [+0.300, +0.453] | +0.115 | +0.260 [+0.184, +0.333] |
| `ptb_baseline_blend` | +0.450 [+0.386, +0.510] | +0.115 | +0.335 [+0.269, +0.394] |
| `ptb_baseline` HGB | +0.390 [+0.329, +0.452] | +0.115 | +0.275 [+0.212, +0.335] |
| `ptb_baseline` xECG | +0.550 [+0.473, +0.622] | +0.115 | +0.434 [+0.357, +0.506] |

Decision under the protocol: territory localization is shown (lower bound above 0), and it goes beyond the
replaced energy (paired lower bound above 0). Post hoc, the two territory parts were +0.325 (inferior) and
+0.287 (anterior) for `all_age_blend`, against +0.091 and +0.139 for energy.

Mean lead shares (post hoc, point values; the expected leads are in bold where they lead):

| Group (n) | `all_age_blend`: largest shares | Energy map: largest shares |
| --- | --- | --- |
| inferior (49) | **II 0.37**, aVR 0.24, aVF 0.08, V5 0.08 | V3 0.17, V4 0.16, V2 0.14, V5 0.12 |
| anterior (55) | aVR 0.21, **V1 0.11**, **V2 0.10**, V5 0.10 | **V3 0.22**, **V4 0.19**, **V2 0.18**, V5 0.11 |
| rbbb (299) | **V1 0.68**, II 0.07, V2 0.05, aVR 0.05 | V1 0.17, V2 0.16, V4 0.14, V5 0.13 |
| lbbb (34) | II 0.18, aVR 0.12, V1 0.11, **V6 0.11** | V2 0.35, V3 0.24, V1 0.17, V4 0.09 |
| stt (300) | aVR 0.27, II 0.18, V5 0.15, V4 0.09 | V3 0.19, V4 0.17, V2 0.16, V5 0.12 |
| normal (300) | II 0.12, aVR 0.12, V5 0.12, V4 0.11 | V2 0.18, V3 0.17, V4 0.14, V5 0.11 |

The energy map puts most of every group's share in V2-V4, the largest-amplitude leads, which is why it
favours the anterior group. The blend's inferior share in II comes mostly from the xECG component (II 0.63);
HGB puts more on aVR (0.38).

## Secondary endpoints

Wave and BBB lead contrasts (`all_age_blend`):

| Contrast | Model | Energy map | Model minus energy |
| --- | --- | --- | --- |
| Wave (mean of the two below) | +0.547 [+0.509, +0.584] | +0.178 [+0.158, +0.200] | +0.368 [+0.332, +0.404] |
| QRS share, BBB against other abnormal | +0.550 [+0.508, +0.591] | +0.193 [+0.172, +0.213] | +0.357 [+0.317, +0.399] |
| ST-T share, stt against other abnormal | +0.543 [+0.504, +0.585] | +0.164 [+0.139, +0.189] | +0.379 [+0.341, +0.418] |
| BBB lead (RBBB V1-V2, LBBB I, aVL, V5-V6) | +0.429 [+0.388, +0.469] | -0.190 [-0.215, -0.162] | +0.619 [+0.567, +0.670] |

`ptb_baseline_blend`: wave +0.574 [+0.536, +0.609] (+0.395 [+0.361, +0.428] beyond energy), BBB lead +0.482
[+0.420, +0.540]. By component of `all_age`: wave HGB +0.534, xECG +0.584; BBB lead HGB +0.425 [+0.376,
+0.469], xECG +0.238 [+0.197, +0.282].

Mean component shares (P, QRS, ST-T; post hoc) for `all_age_blend`: RBBB 0.04 / 0.79 / 0.17, LBBB 0.08 /
0.30 / 0.62, stt 0.03 / 0.12 / 0.85, inferior 0.02 / 0.39 / 0.60, anterior 0.05 / 0.39 / 0.56, normal 0.19 /
0.33 / 0.48. LBBB evidence lies mainly in the ST-T. The counterpart is warped to each beat's own QRS onset,
QRS offset and T end, so it keeps the record's QRS duration and QT; evidence carried by a wide QRS cannot be
removed by this operator, which is a likely reason for the LBBB pattern and for the low LBBB flip rates below.

Territory hit rates (104 territory ECGs; chance, weighted by group size, 0.294 for top-1 and 0.685 for
top-3):

| Score | Top-1 | Minus energy | Top-3 | Minus energy |
| --- | --- | --- | --- | --- |
| Energy map | 0.567 [0.468, 0.663] | | 0.625 [0.529, 0.716] | |
| `all_age_blend` | 0.490 [0.391, 0.588] | -0.077 [-0.214, +0.064] | 0.817 [0.742, 0.887] | +0.192 [+0.061, +0.327] |
| `all_age` HGB | 0.269 [0.186, 0.361] | -0.298 [-0.415, -0.179] | 0.798 [0.719, 0.873] | +0.173 [+0.038, +0.297] |
| `all_age` xECG | 0.625 [0.532, 0.717] | +0.058 [-0.089, +0.216] | 0.856 [0.784, 0.919] | +0.231 [+0.110, +0.352] |
| `ptb_baseline_blend` | 0.654 [0.556, 0.742] | +0.087 [-0.051, +0.223] | 0.827 [0.753, 0.898] | +0.202 [+0.074, +0.323] |

Decision flips of `all_age_blend`: share of flagged ECGs for which one replacement leaves the score at or
below the threshold 0.65697.

| Group | Flagged | Any grid cell | Any lead | Any component | Full | Self-swap grid | Self-swap lead |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| inferior | 45 / 49 | 0.133 | 0.244 | 0.467 | 0.933 | 0.000 | 0.000 |
| anterior | 52 / 55 | 0.096 | 0.077 | 0.288 | 0.846 | 0.000 | 0.000 |
| lbbb | 34 / 34 | 0.000 | 0.000 | 0.029 | 0.353 | 0.000 | 0.000 |
| rbbb | 293 / 299 | 0.038 | 0.096 | 0.235 | 0.683 | 0.003 | 0.000 |
| stt | 244 / 300 | 0.143 | 0.164 | 0.648 | 0.869 | 0.012 | 0.016 |
| all abnormal | 668 / 737 | 0.085 [0.065, 0.108] | 0.124 [0.100, 0.150] | 0.395 [0.358, 0.433] | 0.763 [0.730, 0.794] | 0.006 [0.000, 0.012] | 0.006 [0.001, 0.012] |
| normal | 15 / 300 | 0.600 | 0.733 | 0.933 | 1.000 | 0.067 | 0.000 |

`ptb_baseline_blend` (all abnormal, 632 flagged): grid 0.134 [0.108, 0.161], lead 0.179 [0.150, 0.208],
component 0.584 [0.543, 0.622], full 0.959 [0.942, 0.974].

Full-replacement sanity and operator check. For `all_age_blend` the mean abnormal score falls from 0.910
[0.897, 0.923] to 0.391 [0.371, 0.413]; the drop is 0.519 [0.499, 0.539], so the operator check (lower bound
above 0) passes. The replaced abnormal ECGs stay above the normals' original mean (0.176 [0.155, 0.197]) but
fall below the threshold on average; by group the mean full-replacement score is inferior 0.207, anterior
0.347, stt 0.254, RBBB 0.530 and LBBB 0.717. Full replacement also lowers normal scores, from 0.176 to 0.058
(drop 0.119 [0.099, 0.138]), as in 070: the population template is more typical than most real normals. For
`ptb_baseline_blend` the abnormal drop is 0.672 [0.654, 0.688] (to 0.221) and the normal drop 0.178 [0.156,
0.202].

Self-swap artefact (`all_age_blend`, mean absolute score change):

| | Grid cell, counterpart | Grid cell, self-swap | Whole lead, counterpart | Whole lead, self-swap |
| --- | --- | --- | --- | --- |
| Abnormal | 0.0079 [0.0072, 0.0087] | 0.0017 [0.0014, 0.0019] | 0.0207 [0.0190, 0.0224] | 0.0031 [0.0027, 0.0035] |
| Normal | 0.0106 [0.0096, 0.0117] | 0.0046 [0.0037, 0.0054] | 0.0221 [0.0198, 0.0245] | 0.0085 [0.0071, 0.0099] |

Full self-swap raises the score slightly: drop -0.012 [-0.015, -0.010] for abnormal and -0.048 [-0.057,
-0.039] for normal ECGs. Splicing and beat-to-beat variation are therefore small next to the counterpart on
abnormal ECGs, and about 40% of its size on normals. Self-swap lead shares give a territory contrast of +0.109
[+0.014, +0.209] for `all_age_blend` and +0.005 [-0.096, +0.109] for `ptb_baseline_blend`; post hoc, the
counterpart minus self-swap contrast is +0.196 [+0.099, +0.290] and +0.445 [+0.327, +0.562].

Normals (300): 15 are flagged by `all_age_blend` (12 by `ptb_baseline_blend`). Single grid cells change
normal scores at least as much as abnormal ones (0.0106 against 0.0079), and 9 of the 15 flagged normals are
unflagged by one grid cell. On a normal ECG an attribution mostly shows where the record differs from the
population template.

## Integrity

| Check | Result |
| --- | --- |
| 1. 070 operator: refitted 062 rank-0 model (2,666 fit clean normals, 2,566 patients) against 070's saved counterparts | largest difference 0.0 mV on 100 set-A and 100 set-N records; passed |
| Profile parity (20 ECGs, before the protocol) | measurements identical, xECG features bit-identical for 20 of 20 |
| 2. Unreplaced scores against the saved SPH scores, gate 1e-12 | **failed as registered**: HGB 0.0; xECG up to 4.12e-8 (`all_age`) and 2.46e-8 (`ptb_baseline`); blends up to 2.06e-8 and 1.23e-8 |
| 2, Amendment 1: historical-batch parity | the 2 records over tolerance (A25752, A25760) are the only ones whose features were not bit-identical (largest feature difference 7.15e-6); re-encoded in their historical batch both are bit-identical and their xECG scores match exactly; passed |
| Referral decisions of the unreplaced scores | identical for both blends |
| 3. Bundle thresholds against the calibration-normal replay | identical |
| 4. Variants rebuilt by the encoding stage against the preparation stage (SHA-256) | all 1,037 matched |
| 5. Raw SPH bytes and canonical windows | verified on every read |

Amendment 1 attributed the gap to batch content. The parity stage showed that both records sit in the final
SPH batch, rows 17,392-17,401, which held only 10 rows in the saved extraction. The cause is therefore batch
shape (10 against 16 here), the effect already documented in `pr74-batch-shape-reproduction-2026-10-28.md`.
The rule and its result are unchanged. The registered check stays failed; the amended check passed.

## Run

- Sample: 1,038 ECGs drawn from labels only (inferior 49, anterior 55, lbbb 34, rbbb 300, stt 300, normal
  300; one per patient). One RBBB ECG (A19078) was excluded because no counterpart or component mask could be
  built, leaving 1,037. The territory groups kept 49 and 55 patients, so the territory endpoint stayed
  primary.
- Preparation: 114 s with 16 single-thread workers. Encoding of 105 variants per ECG: 519 ECGs on GPU 0 in
  607 s and 518 on GPU 1 in 550 s, in parallel, each under its own lock, peak 0.62 GB. Bootstrap: 20 s.
- Profile forecast was 61 minutes on one GPU; the run took about 10 minutes of encoding on two.

## Deviations and disclosures

1. Integrity check 2 failed at its registered 1e-12 gate; the study continued only under Amendment 1, whose
   rule and code were committed before any replacement score was read. Its written diagnosis (batch content)
   was wrong about the cause (batch shape); see Integrity.
2. The first encoding launch stopped on a pandas mask error in `prepared_rows` before any ECG was encoded;
   it was fixed in `76012cd` and relaunched.
3. The prepare receipt's per-status counts were overwritten by the receipt's own `status` key; the counts
   come from `prepare/prepare.csv` (1,037 prepared, 1 excluded).
4. The profile ran on code that was not yet committed (it produced no scores); the counterpart stage was
   rerun on committed code with the same result.
5. `analyze/posthoc.json` (territory parts, share profiles, the self-swap paired gap and the zero-drop count)
   was computed after the declared results by `scripts/reports/describe_nbr01.py`; no decision uses it.

## What it means

- The screening blend's lead evidence for MI is anatomically plausible in direction and goes beyond where a
  normal template differs most from the record, the confound that dominated Experiment 070. The page can show
  lead and wave attributions as model evidence with that support.
- It is not a precise localizer. Inferior evidence concentrates in lead II and aVR rather than III and aVF,
  and the top single lead is in the expected territory only about half the time. aVR carries a large share
  for every abnormal group except RBBB, most of it from the HGB measurements.
- Most referrals do not hinge on one region: one lead removes the flag for 12.4% of flagged abnormal ECGs.
  Counterfactual text such as "if V2's ST-T looked normal, this student would not be referred" will be true
  for few students; graded drops are the main display.
- Because the counterpart keeps each beat's timing, it cannot show evidence carried by QRS width or QT; LBBB
  stays flagged after full replacement in 22 of 34 cases.
- Scores near 1 compress drops (for example an inferior MI scored 0.999 has a largest single-lead drop of
  0.007 in the export), so shares of small drops carry the attribution for saturated ECGs.

## Limitations

- SPH territorial statements are old MI only, with no lateral MI and no lead information for ST-T
  statements; 104 territory patients bound the precision of the primary endpoint.
- Labels are ECG-annotation proxies, and SPH has been read by many experiments.
- The attribution is model-specific and counterfactual: it shows where the score responds, not where disease
  is. Single-lead replacement leaves the limb leads mutually inconsistent, and how each model reads that is
  not measured.
- Intervals are conditional on the fitted models, the template and the sample.

## Page export

`outputs/nbr01_normal_beat_replacement_20261028/examples.json` (6,225,836 bytes; schema version 1, validated
by `nbr01_export.validate`): 16 examples drawn within groups without regard to any score, 4 inferior, 4
anterior (the two lateral slots moved there), 2 RBBB, 2 LBBB, 2 ST-T and 2 normal. Each model's scores also
carry `hgb_original` and `xecg_original`. The `counterpart` array is the operator's full counterpart; outside
the replaced component ranges (record edges and gaps between beat windows) it is flattened and was never
used.

## Follow-ups (added to the backlog)

- `nbr_lateral_territory_ptb`: repeat the territory test on PTB-XL fold 9, whose statements include lateral
  MI and territorial ischaemia.
- `nbr_logit_drops`: re-analyse the saved scores with drops on the logit scale, which avoids saturation near 1.
- `nbr_timing_counterpart`: a counterpart that also normalizes QRS duration and QT, to expose interval
  evidence such as LBBB width.
- `nbr_limb_lead_consistent`: replace I and II and recompute the derived limb leads, to test whether the
  lead II and aVR emphasis comes from inconsistent limb leads.
- `nbr_record_specific_counterpart`: a counterpart from the record's own normal-looking beats (070 follow-up
  2), to reduce changes on normal ECGs.
- `nbr_page_clinician_review`: a cardiologist reviews the page examples, once one is available.
