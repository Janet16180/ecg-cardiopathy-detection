# Saved-map counterfactual-response audit v2

18 October 2026. Both adapted models respond strongly to the programmed intervention, rather than
achieving their entire edited-map AP through unchanged background-site agreement. On profile B,
R's mean target score rises by 0.70847 and S's by 0.72174 relative to the exact unedited counterpart.
Their target-minus-other-candidate response contrasts are 0.56211 and 0.57721. The frozen-encoder
arm F has essentially no positive target/selectivity response. These descriptive findings do not
change the [original pilot's failed advancement](ssl-synthetic-localization-pilot-results.md).
No new model inference, training, checkpoint deserialization, threshold fitting or GPU work occurred.

## Frozen scope and exact counterpart admission

The [original diagnostic protocol](ssl-synthetic-localization-counterfactual.md) was committed as
`cd7b1eb` before any diagnostic metric. Its first execution stopped on a missing explicit original
summary binding, with zero pairs verified and zero diagnostic profiles completed; the
[v1 failure report](ssl-synthetic-localization-counterfactual-results.md), source bytes and failed
receipt remain unchanged. The [forward provenance amendment](ssl-synthetic-localization-counterfactual-v2.md)
was committed as `adafd30`; adapter/tests were committed as `d1978bd` and independently reviewed
before this execution. All numerical matching, metric and bootstrap helpers are the unchanged,
receipt-pinned v1 functions. No frozen module globals were patched.

The additive independent reproduction closure explicitly binds the original evaluation summary
and passed original numerical audit, with zero mismatches. It freshly verified 141 bindings and
766 comparisons, matching the stored independently recomputed original aggregates, thresholds,
denominators and failed decision. Its SHA-256 is
`abff7844814238a75922280d59932a28f856c20abf5b522407ef120f1ea444f2`.
The prior full original audit passed with zero mismatches and remains intact. This closure repairs
an explicit provenance-schema omission; it does not repair or replace any model score.

All 1,024 positive cases were admitted before any new diagnostic: 256 subjects × focal/persistent ×
A/B. For every case, its saved paired baseline is byte-identical to the same latent subject's observed
sham input when nuisance=False, or nuisance-control input when True. Seed, latent background,
nuisance parameters and profile match exactly. The existing same-arm control map therefore supplies
the baseline response without rerunning the model. Both profiles retain all subjects and variants;
there are 3,072 arm/positive-case comparisons and 1,536 arm/nuisance-control comparisons. No region
was empty and no case was dropped. All 9,216 edited-map AP/top-area metric values were reproduced
exactly against the original local case tables during this audit.

Regions are fixed from original intervention truth: T is actual support above 0.005 mV; U is the
unedited part of the broad all-beat/all-lead candidate domain; O is outside that domain. T is contained
in the candidate domain. B region sizes [minimum,median,maximum] are T=[30,267,3930],
U=[5112,10594.5,16790] and O=[13164,18150,23088] cells out of 30,000. A counts are retained separately
in the receipt. Focal and persistent values are averaged equally within each subject before the
256-subject mean. Intervals use 2,000 shared whole-subject bootstrap draws, seed 20261019, with no
skipped draws. They condition on this one training seed and these simulated backgrounds.

## Held-out-profile B input response

The score change is saved p_edited minus the exact counterpart's saved p_baseline, computed in
float64. Recording thresholds were fitted separately in the original experiment; the sigmoid values
themselves are not probability-calibrated. The response scale is descriptive, especially across arms.

| Arm | Mean change in T | Mean change in U | Mean change in O | T-minus-U contrast [95% interval] |
| --- | ---: | ---: | ---: | ---: |
| R | +0.70847 | +0.14637 | -0.000262 | +0.56211 [0.53452,0.58872] |
| S | +0.72174 | +0.14452 | -0.000057 | +0.57721 [0.54686,0.60599] |
| F | -0.00156 | -0.00118 | -0.000095 | -0.00038 [-0.00217,0.00145] |

Focal T-minus-U contrasts are 0.68845/0.69669/-0.00157 for R/S/F; persistent contrasts are
0.43576/0.45773/0.00081. For the adapted arms, persistent edits also raise unedited candidate scores
substantially: mean U changes are 0.27225 for R and 0.26874 for S, compared with 0.02049/0.02031 for
focal edits. Responsiveness is therefore not the same as perfectly confined support recovery.
Near-zero signed O means do not bound absolute or maximum off-target changes; those were not the
predeclared outside-region endpoints.

## Edited AP versus unchanged baseline-site agreement

Both maps are scored against the SAME fixed intervention mask T. The baseline-map column asks how
well pre-intervention scores already rank the future edited sites. It is a site-agreement/phase-prior
diagnostic, not pathology accuracy on a recording without that intervention.

| B arm | Edited AP [95% interval] | Baseline-site AP [95% interval] | AP change [95% interval] |
| --- | ---: | ---: | ---: |
| R | 0.63796 [0.59592,0.68060] | 0.20085 [0.19054,0.21166] | +0.43711 [0.39886,0.47428] |
| S | 0.64060 [0.59900,0.68347] | 0.18156 [0.17225,0.19085] | +0.45904 [0.41717,0.49953] |
| F | 0.15787 [0.14254,0.17472] | 0.14410 [0.13291,0.15559] | +0.01377 [0.00450,0.02354] |

| B arm | Focal baseline-site AP → edited AP | Persistent baseline-site AP → edited AP |
| --- | ---: | ---: |
| R | 0.04326 → 0.62974 | 0.35844 → 0.64618 |
| S | 0.04132 → 0.62921 | 0.32179 → 0.65199 |
| F | 0.02607 → 0.05383 | 0.26213 → 0.26192 |

F's persistent AP is virtually unchanged from its pre-intervention site agreement. Its persistent
AP change is -0.00021, interval [-0.01059,+0.01042]. Its modest macro AP change does not accompany a
positive mean target response or a positive target-selectivity interval. R/S show substantially larger
intervention-dependent ranking changes, while still retaining background-site agreement.

Top-1%-area precision changes on B are +0.43581 [0.39080,0.48005] for R, +0.44351
[0.39857,0.48722] for S and +0.01277 [-0.00265,0.02872] for F. Corresponding target-recall changes
are +0.33730 [0.31372,0.36113], +0.33998 [0.31569,0.36522] and +0.01317 [0.00652,0.02027].
These use the original 300-cell area and expected-tie handling; no score-window restriction was added.

The exploratory S-minus-R AP-change contrast is +0.02194 [0.01162,0.03191]. It decomposes into the
original edited-AP gap (+0.00264) minus S's lower baseline-site AP gap (-0.01929). Most of this new
contrast therefore comes from different baseline-site agreement, not a newly demonstrated edited-AP
improvement. Focal/persistent AP-change contrasts are +0.00141 [-0.01202,0.01335] and +0.04247
[0.03054,0.05483]. The exploratory score-selectivity contrast S-minus-R is +0.01511 [0.00391,0.02577];
uncalibrated score scales prevent a common clinical effect interpretation. None of these diagnostics
replaces the original primary endpoint or creates a new advancement rule.

## Nuisance responses at unchanged thresholds

Every subject's nuisance and sham maps were compared using the ORIGINAL fixed arm threshold and
strict greater-than flags. Nuisance combines the original noise, drift and gain; this audit does not
attribute the response to any one component. A/B controls are identical and yield the same results.

| Arm | Mean absolute full-map nuisance change [95% interval] | Sham → nuisance flags / 256 | Within-arm flag-rate change [95% interval] |
| --- | ---: | ---: | ---: |
| R | 0.01736 [0.01563,0.01915] | 2 → 19 | +0.06641 [0.03906,0.09775] |
| S | 0.01818 [0.01612,0.02041] | 1 → 23 | +0.08594 [0.05078,0.12109] |
| F | 0.01437 [0.01337,0.01538] | 0 → 17 | +0.06641 [0.03906,0.09766] |

These within-arm nuisance-versus-sham changes are different quantities from the original comparative
S-minus-R control veto. No new nuisance pass/fail cutoff was invented. The flags reproduce the
original counts, and their sensitivity remains visible despite the original between-arm veto passing.

## A secondary, provenance and cost

Secondary A edited/baseline-site/AP-change means are 0.64904/0.20576/+0.44329 for R,
0.65651/0.18611/+0.47040 for S and 0.17533/0.15322/+0.02211 for F. A T-minus-U contrasts are
0.57631/0.59515/0.00032. Its exploratory S-minus-R AP-change is +0.02712 [0.01727,0.03632], with the
same baseline-site-agreement limitation. All A/B per-arm and per-mode intervals, region denominators,
fixed-area changes and nuisance comparisons remain in the full receipt/local case tables.

The successful forward run took 34.253 wall seconds and 34.245 CPU seconds with CUDA hidden and all
OPENBLAS/OMP/MKL/NUMEXPR limits one. No new prediction or waveform arrays were written; total new
output is 3,397,556 bytes, under the 100 MiB ceiling. The cumulative recorded CPU charge is a
conservative 389.701 seconds against 14,400. It retains the original pilot, numerical audit,
checkpoint supplement, stopped 1.689-second counterfactual attempt and binding closure. The closure's
0.744-second metadata-only fallback is charged both within its own total and as a separately bound
historical failure; this conservative duplicate was not subtracted from immutable receipts.

Twelve combined artificial CPU tests passed in 0.55 seconds and Ruff passed. Source review approved
the forward adapter before execution. After completion, all 59 source bindings, 142 input bindings
and three local output hashes were freshly verified again. The unused real-training waveform cache
was checked only against its frozen metadata digest and never reopened. No checkpoint was
deserialized, real waveform payload opened or GPU/model API used. Original evaluation/calibration,
weights, native probability maps and the failed advancement verdict remain unchanged.

The source-bound result is
`outputs/ssl_synthetic_localization/pilot_v1/counterfactual_audit_v2/summary.json`, SHA-256
`209f3c1b560c181c295f16042dd982e50c6f17da051acac903a411311a6bfc79`.
The same directory retains admission and separate A/B local case-response JSON files; all model maps
remain at their original evaluation paths. The prior v1 failure SHA-256 is
`8a842ebb36acee66fa22c82026444f1d61afe702301d328c8c1f55068cea06f4`.

This is a spent-development diagnostic of a narrow simulated intervention. It shows input response
for the adapted models and limited response for the frozen head, while exposing background-site
agreement and nuisance sensitivity. It does not establish clinical causality, disease validity,
complete pathology support, real-background transfer or fresh confirmation. The original failed
SSL advancement remains the experiment's decision.
