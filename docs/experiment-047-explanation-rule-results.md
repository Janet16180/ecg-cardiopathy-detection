# Experiment 047 results: an explanation rule keyed to the referral decision

Completed 1 October 2026 under the [frozen protocol](experiment-047-explanation-rule.md) (commit `12f6c28`;
run from `54b61be`). Saved scores only: nothing was trained or extracted. Besides the saved scores, the run
read only the 84 PVC records' PTB-XL waveforms, to rebuild 041's premature-beat windows, and SPH predictions
for the 043 AUROC check. No calibration, test or EchoNext record was read. Outputs are in
`outputs/experiment047_explanation_rule_v1/` (`result.json` with every input, source and protocol hash,
`explanations.npz`, `run.log`). The run took 64 seconds on CPU, 62 of them in loading and integrity checks.

## Readings

| Comparison | Premature-beat test (hit - chance difference, lower bound > -0.10) | Anterior contrast of the rule (lower bound > 0) | Reading |
| --- | --- | --- | --- |
| Rule against `U_B` alone (primary) | -0.136 [-0.232, -0.057]: fails | +0.215 [0.099, 0.336]: passes | **Does not improve** |
| Rule against `attention_jepa` alone | +0.694 [0.574, 0.808]: passes | +0.215 [0.099, 0.336]: passes | Improves |
| Sensitivity, E as the readout, against `U_B` | -0.148 [-0.242, -0.058]: fails | +0.223 [0.112, 0.344]: passes | Does not improve |

The rule did not improve on `U_B` alone by the primary reading, so no example figures were drawn.

## Integrity

- The 042, 043 Stage 1 and Stage 2 outputs match the hashes in their `result.json` files, and 043 Stage 2's
  recorded hashes of 042's outputs match. 045's `ensemble.npz` matches its receipt.
- The rows align: the two stages' development record IDs are equal and their ECG IDs equal 042's 1,604
  scored rows; the Stage 2 SPH IDs equal the Stage 1 SPH IDs at the labeled positions. The premature-beat
  rule gave 73 ECGs, with 042's exclusions (0 with too few peaks, 11 without a premature beat).
- On all ECGs, `U_B` recomputed with seed 42042 equals 042's `result.json` exactly in every field checked
  (threshold 1,158.8, AUROC, AP, any red, mean red units, sensitivity at the budget, both lead contrasts, PVC
  count, hit, chance and hit - chance). With seed 43043, `U_B` and `attention_jepa` reproduce 043 Stage 2's
  `maps` and `attention_jepa_minus_U_B` blocks exactly; `attention_jepa`'s threshold is 0.275.
- The saved logits reproduce 043's AUROC and AP of R, `logistic_concat` and `attention_jepa` on full,
  ordinary, hard and SPH, and their differences from R, exactly. E recomputed from the saved logits equals
  045's saved development E exactly.

## Referral

The readout threshold is the 0.95 quantile of the 463 normals' `logistic_concat` logits, 0.271. 792 of the
1,604 ECGs are referred.

| Group | Size | Referred | Rate |
| --- | ---: | ---: | ---: |
| NORM-only normals | 463 | 24 | 0.052 |
| Positives | 843 | 680 | 0.807 [0.779, 0.833] |
| PVC ECGs | 84 | 60 | 0.714 |
| PVC ECGs with a premature-beat window | 73 | 49 | 0.671 |
| Benign variants | 52 | 6 | 0.115 |
| Anterior-only infarcts | 146 | 136 | 0.932 |
| Inferior-only infarcts | 149 | 112 | 0.752 |

## Explanations of the referred ECGs

Every interval uses seed 47047 with 2,000 whole-patient draws. Premature beats: 49 referred ECGs with a
window. Lead contrasts: 136 referred anterior-only against 112 referred inferior-only infarcts. The
"anterior hits" column gives the share with an explanation lead in V1-V4, anterior-only against
inferior-only.

| Explanation | Hit | Chance | Hit - chance | Anterior contrast | Anterior hits | Inferior contrast |
| --- | ---: | ---: | --- | --- | --- | --- |
| Rule | 0.755 | 0.201 | +0.554 [0.436, 0.667] | +0.215 [0.099, 0.336] | 0.537 / 0.321 | +0.194 [0.088, 0.305] |
| `U_B` alone | 0.898 | 0.208 | +0.690 [0.598, 0.767] | +0.031 [-0.103, +0.170] | 0.522 / 0.491 | +0.105 [-0.014, +0.225] |
| `attention_jepa` alone | 0.020 | 0.160 | -0.140 [-0.178, -0.093] | +0.197 [0.079, 0.315] | 0.412 / 0.214 | +0.212 [0.101, 0.319] |

| Paired difference | Hit - chance | Anterior contrast | Inferior contrast |
| --- | --- | --- | --- |
| Rule minus `U_B` | -0.136 [-0.232, -0.057] | +0.184 [0.042, 0.328] | +0.089 [-0.040, +0.224] |
| Rule minus `attention_jepa` | +0.694 [0.574, 0.808] | +0.018 [-0.084, +0.126] | -0.018 [-0.092, +0.055] |

Share of referred ECGs explained by `U_B` (the rest by `attention_jepa`):

| Referred group | Referred | Explained by `U_B` |
| --- | ---: | ---: |
| All | 792 | 0.327 |
| Normals | 24 | 0.000 |
| Positives | 680 | 0.318 |
| PVC ECGs | 60 | 0.817 |
| PVC ECGs with a window | 49 | 0.857 |
| Benign variants | 6 | 0.167 |
| Anterior-only infarcts | 136 | 0.331 |
| Inferior-only infarcts | 112 | 0.179 |

When `attention_jepa` supplied the explanation, it was itself red at its own threshold on 0.499 of those ECGs.

## Sensitivity: E as the readout

E's threshold on the same normals is 0.181; 794 ECGs are referred (24 normals, 682 positives, 62 PVC ECGs,
51 with a window, 4 benign variants, 139 anterior-only and 108 inferior-only infarcts). `U_B` explains 0.327
of them, 0.843 of the PVC ECGs with a window.

| Explanation | Hit - chance | Anterior contrast | Inferior contrast |
| --- | --- | --- | --- |
| Rule | +0.545 [0.425, 0.655] | +0.223 [0.112, 0.344] | +0.192 [0.090, 0.299] |
| `U_B` alone | +0.693 [0.608, 0.767] | -0.001 [-0.135, +0.124] | +0.064 [-0.049, +0.185] |
| `attention_jepa` alone | -0.141 [-0.178, -0.093] | +0.211 [0.105, 0.321] | +0.210 [0.103, 0.319] |
| Rule minus `U_B` | -0.148 [-0.242, -0.058] | +0.223 [0.086, 0.360] | +0.127 [-0.006, +0.256] |
| Rule minus `attention_jepa` | +0.686 [0.569, 0.800] | +0.011 [-0.087, +0.110] | -0.019 [-0.095, +0.050] |

The reading is the same as with pipeline v3's readout.

## Post-hoc check (not prespecified)

A scratch script read the saved `explanations.npz` and 045's `load_inputs` and asked, for each referred PVC
ECG with a window that the rule sent to `attention_jepa`, whether each map's top unit was on the premature
beat. With `logistic_concat` there are 7 such ECGs (IDs 1321, 2750, 3862, 4686, 6115, 7936, 15892). On all 7,
`U_B`'s top unit is on the premature beat and `attention_jepa`'s is not. Their `U_B` worst-unit scores are 109
to 1,016, below the 1,158.8 threshold. With E there are 8 (the same 7 and ECG 5934, `U_B` worst 1,136.5),
with the same result. Among all 73 PVC ECGs with a window, the 10th, 25th and 50th percentiles of `U_B`'s
worst-unit score are 543, 1,461 and 3,428.

## Interpretation

- Keying the explanation to the referral decision removes 045's budget split, and it helps: on referred PVC
  ECGs, the rule's top unit is on the premature beat in 76% of ECGs, against 60% for 045's combined map on all
  PVC ECGs. But the 7 referred PVC ECGs on which `U_B` is not red go to the attention layer, which misses all
  of them, while `U_B`'s top unit was right on every one. The hit rate falls from 90% to 76%, and the paired
  difference's lower bound (-0.232) is below the -0.10 margin.
- The lead information is kept: the rule's anterior contrast (+0.215) equals `attention_jepa`'s (+0.197;
  paired difference +0.018 [-0.084, +0.126]) and is above `U_B`'s by +0.184 [0.042, 0.328]. The inferior
  contrast is close to `attention_jepa`'s (+0.194 against +0.212), but its gain over `U_B` (+0.089
  [-0.040, +0.224]) includes zero. The attention layer explains two thirds of the referred anterior-only and
  four fifths of the referred inferior-only infarcts.
- Against `attention_jepa` alone, the rule is better by the secondary reading: it adds premature-beat
  localization (+0.694) without losing lead information. So the rule is a better explanation than the
  attention map, but not than `U_B` for premature beats.
- `U_B`'s red threshold is the wrong switch. Being red means "this ECG is unusual for a normal", not "the top
  unit is meaningful". On PVC ECGs, `U_B`'s top unit is often right even when its score is below the normals'
  95th percentile.

## Caveats

- 49 referred PVC ECGs: one ECG moves the hit rate by 2 points. The premature-beat rule is automatic.
- The attention layer marks only 8 leads; infarct location is a whole-ECG statement; the contributions are
  the networks' own decompositions.
- The referral and map thresholds are fitted on the same 463 normals that set the referral rate.
- The post-hoc check was run after the reading and does not change it.
- Development data, read by Experiments 041-045; the results are exploratory.

## Deviations

None. The rule did not improve, so no figures were drawn, as prespecified.

## Follow-up ideas

- Switch on the rhythm finding, not on `U_B`'s redness: on a referred ECG, show `U_B` when 041's
  premature-beat rule (or an R-R irregularity score) fires, otherwise `attention_jepa`. The post-hoc check
  suggests this would recover the 7 lost hits, but it needs R peaks on all 1,604 ECGs, and its cost on
  infarct ECGs with ectopic beats must be measured.
- Show both layers on referred ECGs, in two labeled colours (rhythm: `U_B`; morphology: attention), with no
  switch, and score each layer on its own task. This is a display choice for the cardiologist, not a metric
  question.
- A `U_B` threshold fitted on referred ECGs (for example the quantile that keeps 95% of `U_B` hits on
  referred PVC ECGs) would need a separate fitting set, to avoid choosing it on the 49 ECGs it is scored on.
