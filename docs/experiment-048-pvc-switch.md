# Experiment 048: switch the explanation on the PVC finding head, from saved scores

**Frozen 1 October 2026, before any score of this experiment is computed.** Experiment 047 referred ECGs
with pipeline v3's readout and explained each referred ECG with `U_B`'s top unit when `U_B` was red,
otherwise with `attention_jepa`'s top token. The rule kept the attention map's infarct-lead information
(anterior contrast +0.215 [0.099, 0.336]) but lost part of `U_B`'s premature-beat localization (paired hit -
chance -0.136 [-0.232, -0.057], below the -0.10 margin). A post-hoc check found that the 7 referred PVC ECGs
the rule sent to the attention layer all had `U_B`'s top unit on the premature beat, with `U_B` worst-unit
scores below its red threshold. `U_B` being red was the wrong switch.

This experiment switches on a learned rhythm finding instead: pipeline v3's PVC finding head. It does not
use 041's premature-beat rule as the switch, because that rule also defines the premature-beat ground truth.
Saved scores and saved head parameters only; nothing is trained.

Question: on referred ECGs, does "show `U_B` when the PVC head is high, otherwise `attention_jepa`" keep
`U_B`'s premature-beat localization and gain `attention_jepa`'s infarct-lead information?

## Inputs and rows

- Everything 047 reads (045's `load_inputs`: 042's `U_B` units and rows, 043 Stage 1 and Stage 2 predictions
  and token maps), 045's `ensemble.npz`, and 047's `result.json` and `explanations.npz`.
- Experiment 044: `outputs/experiment044_pipeline_v3_v1/pipeline_v3_heads.npz` (keys `pvc_mean`,
  `pvc_scale`, `pvc_coef`, `pvc_intercept`, `pvc_logit_mean`, `pvc_logit_sd`) and `predictions.npz`
  (`sph_v2_z_pvc`, `sph_ecg_ids`, `development_record_ids`), both hash-checked against 044's `result.json`.
  The xECG PVC head is shared by pipelines v2 and v3 (`docs/pipeline-v3.md` section 3, "unchanged from v2";
  044's results: the refitted xECG PVC head reproduced 037's z-scores exactly, and v3's finding heads score
  as v2's to within 6e-17). It was trained on PTB-XL training rows and Challenge training rows (032), not on
  the development ECGs, with ECG-level PVC labels, not 041's rule.
- xECG features of the 1,604 development ECGs and the 25,577 SPH evaluation ECGs: 043's `ptb_inputs` and
  `sph_inputs` (the cached features read by 043 and 044). Rows: 042's 1,604 full-development ECGs, as 047.

## PVC score

- z_PVC = (clipped logit of `pipeline_v2.score_parameters(heads, "pvc", x_xECG)` - `pvc_logit_mean`) /
  `pvc_logit_sd`, with the logit of probabilities clipped to [1e-15, 1 - 1e-15]
  (`finding_screen.clipped_logits`), as `finding_screen.finding_z` with 044's saved constants.
- Switch threshold s(q): the q-quantile (NumPy's default linear interpolation) of the 463 NORM-only normals'
  z_PVC. Primary q = 0.975; sensitivity q = 0.95 and q = 0.99. The switch is on when z_PVC is strictly above
  s(q).

## Referral and explanation

- Referral as 047: `logistic_concat`'s development logit strictly above the 0.95 quantile of the 463 normals'
  logits (threshold 0.271, 792 referred ECGs). Sensitivity: E (045's ensemble) at its own 0.95 quantile.
- For a referred ECG, the explanation unit is `U_B`'s top unit if the switch is on, otherwise
  `attention_jepa`'s top token. Non-referred ECGs get no explanation.
- Comparators on the same referred ECGs: `U_B` alone and `attention_jepa` alone (047's rows, recomputed with
  this experiment's seed), and, descriptively, 047's rule (switch on `U_B` red at 1,158.8).

## Metrics, on referred ECGs only (047's definitions and helpers)

- Premature beats: on referred ECGs among the 73 with a premature-beat window, hit, chance (share of the
  supplying layer's units on a premature-beat window) and hit - chance with a whole-patient bootstrap.
- Lead contrasts: anterior (explanation lead in V1-V4, referred anterior-only minus referred inferior-only)
  and inferior (II, III or aVF, inferior-only minus anterior-only), patients resampled within each group.
- Share of referred ECGs explained by each layer, overall and per group (normal, positive, PVC, PVC with a
  window, benign, anterior, inferior).
- Cost on infarcts: the number and share of referred anterior-only and inferior-only infarcts that the switch
  sends to `U_B`.
- Referral rates per group, as 047 (they do not depend on the switch).
- Paired contrasts, rule minus each comparator: hit - chance per referred PVC ECG with a window (bootstrap of
  the per-ECG difference), and the lead contrasts via the per-ECG indicator difference
  (`explanation_rule.paired_difference`).

## Prespecified reading

**Primary** (`logistic_concat` referral, q = 0.975). The rule improves on `U_B` alone if both:

1. the lower bound of the rule's hit - chance minus `U_B`'s, on referred PVC ECGs with a window, is above
   -0.10; and
2. the lower bound of the rule's anterior contrast is above 0.

**Secondary.**

- Against `attention_jepa` alone, the same two conditions (`explanation_rule.rule_reading`).
- Sensitivity: q = 0.95 and q = 0.99 with `logistic_concat` referral, and E referral with q = 0.975, each
  read by the primary rule.
- Descriptive: rule minus 047's rule, paired on the same referred ECGs (`logistic_concat` referral).

If the primary reading says the rule improves, figures of the seven 041 example ECGs (IDs 47, 219, 8, 184,
287, 30, 69) are drawn with a new plotting helper (`ecg_experiment/pvc_switch.py`, two colours by layer:
`U_B` red, `attention_jepa` blue) in three columns: the rule (for a referred ECG, the supplying layer's units
above that layer's own 0.95-quantile threshold, plus its top unit; nothing for a non-referred ECG), `U_B`
alone and `attention_jepa` alone (red units at each map's own threshold). They are saved under the run's
`figures/` and copied to `docs/figures/experiment-048/`. No notebook.

## Statistics

- 2,000 whole-patient bootstrap draws with seed 48048 for every interval of this experiment. The integrity
  reproductions use the seeds of the runs they reproduce (42042, 43043, 47047).
- Development data, read by Experiments 041-047; the results are exploratory. The switch threshold is fitted
  on the same 463 normals as the referral and map thresholds. q = 0.975 was chosen before any z_PVC of a
  development ECG was computed.

## Integrity, before any score of this experiment

- 047's checks: input receipts, row alignment, 73 premature-beat ECGs with 042's exclusions, `U_B` against
  042 exactly, `U_B` and `attention_jepa` against 043 Stage 2 exactly, the readout logits against 043's
  AUROCs exactly, E against 045's saved E exactly.
- 044's `pipeline_v3_heads.npz` and `predictions.npz` match the hashes in 044's `result.json`; the heads file
  also matches the hash in `docs/pipeline-v3.md` (`54c5049a…f8ce`).
- 044's `sph_ecg_ids` equal 043's SPH ECG IDs and 044's `development_record_ids` equal the development record
  IDs. z_PVC recomputed on the 25,577 SPH ECGs reproduces 044's saved `sph_v2_z_pvc` to an absolute 1e-10.
- 047's `explanations.npz` matches its receipt, and 047's analysis (its runner's `analyse`, seed 47047)
  recomputed on both readouts equals 047's `result.json` `readouts` block exactly.

Any failure stops the run before the experiment's scores.

## Caveats

- The PVC head fires on ECGs with PVCs anywhere; it says nothing about where the premature beat is. The
  hit still comes from `U_B`'s top unit.
- The PVC head also fires on some infarct ECGs; that is the cost measured above. An infarct with ectopic
  beats sent to `U_B` may be explained by the ectopic beat, which is correct for the rhythm but hides the
  infarct leads.
- The premature-beat rule is automatic; the attention layer marks only 8 leads; infarct location is a
  whole-ECG statement; the contributions are the networks' own decompositions.

## Execution

```bash
CUDA_VISIBLE_DEVICES= OPENBLAS_CORETYPE=Haswell OMP_NUM_THREADS=4 \
.venv/bin/python -m scripts.experiments.run_pvc_switch048
```

`scripts/experiments/run_pvc_switch048.py` uses `ecg_experiment/pvc_switch.py` (new) and
`ecg_experiment/explanation_rule.py` (047's, unchanged, because 047's receipt pins it), and the 042, 043,
045 and 047 runners. It hashes its inputs, sources and this protocol, refuses to overwrite a run, writes into
a `.partial` folder and renames it to `outputs/experiment048_pvc_switch_v1/` (`result.json`, `run.log`,
`explanations.npz` and, if the rule improves, `figures/`). The results go to
`docs/experiment-048-pvc-switch-results.md`.
