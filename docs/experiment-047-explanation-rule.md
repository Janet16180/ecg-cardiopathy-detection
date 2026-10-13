# Experiment 047: an explanation rule keyed to the referral decision, from saved scores

**Frozen 1 October 2026, before any score of this experiment is computed.** Experiment 045 combined two maps
by splitting one 5% red budget between them. Each layer then marked only 2.8% of normals, and the share of
PVC ECGs whose explanation sat on the premature beat fell from 93% (`U_B` alone) to 60%. The budget split was
the cause: every normal ECG one layer marks is one the other cannot. This experiment removes the split. The
referral decision comes from pipeline v3's readout alone, and the maps only choose what to show on an ECG
that is already referred. `U_B` keeps its own red threshold (042's 95th percentile), so it explains every
referred ECG on which it would be red alone. Saved scores only; nothing is trained or extracted.

Question: on referred ECGs, does the rule "show `U_B` when `U_B` is red, otherwise `attention_jepa`" keep
`U_B`'s premature-beat localization and gain `attention_jepa`'s infarct-lead information?

## Inputs and rows

- Experiment 042: `outputs/experiment042_lead_wave_maps_v1/unit_scores.npz` (`U_B` units with lead, start
  and end) and `result.json`. Rows: 042's `select_rows` on the 1,604 full-development ECGs: 463 NORM-only
  normals and 843 positives (the 1,306 binary-evaluation ECGs), 84 PVC ECGs (73 with a rule-detected
  premature beat), 52 benign variants, 146 anterior-only (AMI) and 149 inferior-only (IMI) infarcts.
- Experiment 043 Stage 2: `stage2/token_maps.npz` (`attention_jepa` 3-seed mean token contributions, 8 leads
  by 50 patches of 0.2 s) and `stage2/predictions.npz` (`attention_jepa` 3-seed mean logit).
- Experiment 043 Stage 1: `stage1/predictions.npz` (`logistic_concat`, pipeline v3's readout, and R logits on
  the 1,604 development ECGs and the SPH evaluation ECGs).
- Experiment 045: `outputs/experiment045_two_layer_map_v1/ensemble.npz` (E, the mean of the
  `logistic_concat` and `attention_jepa` logits), for the sensitivity analysis and an integrity check.
- The loading, receipt checks and row alignment are 045's `load_inputs`, imported from its runner. The only
  waveform reading is that of the 84 PVC records' PTB-XL waveforms (041's premature-beat rule, via 042's
  `premature_targets`) and, if the rule improves, the seven example ECGs. SPH predictions are loaded only for
  the integrity check of 043's AUROCs. No calibration, test or EchoNext record is read.

## Referral

- The readout logit of an ECG is `logistic_concat`'s saved development logit.
- The referral threshold is the 0.95 quantile (NumPy's default linear interpolation) of the 463 NORM-only
  normals' readout logits. An ECG is referred when its logit is strictly above the threshold. This is the 5%
  budget on the same normals that the maps use for their red thresholds (Experiment 030's operating point is
  fitted on local normals; this is its development analogue, not a deployable threshold).
- Referral rates are reported for normals, positives (with a whole-patient bootstrap interval), the 84 PVC
  ECGs, the 73 with a premature-beat window, the 52 benign variants and the anterior-only and inferior-only
  infarcts.

## Explanation rule

- Layer 1 is `U_B` with 042's red threshold t1 = 1,158.8 (the 0.95 quantile of `U_B`'s worst-unit scores on
  the same 463 normals, recomputed and required to equal 042's `result.json`). Layer 2 is `attention_jepa`.
- For a referred ECG, the explanation unit is `U_B`'s top unit if `U_B`'s worst unit is strictly above t1,
  otherwise `attention_jepa`'s top token (whether or not `attention_jepa` is red). This is 045's `explain`
  with t1 as the first threshold; the second threshold (043's 0.95 quantile for `attention_jepa`) only sets a
  descriptive red flag. Non-referred ECGs get no explanation.
- Comparators on the same referred ECGs: `U_B` alone (its top unit on every referred ECG) and
  `attention_jepa` alone (its top token on every referred ECG).

## Metrics, on referred ECGs only

For each of the three explanations (rule, `U_B` alone, `attention_jepa` alone):

- **Premature beats:** on the referred ECGs among the 73 with a premature-beat window, `hit` is 1 if the
  explanation unit's time span overlaps a premature-beat window, and `chance` is the share of the supplying
  layer's units that overlap one (041's rule, 042's `premature_hit`). Reported: number of ECGs, hit, chance,
  and hit - chance with a whole-patient bootstrap interval.
- **Lead contrasts:** the explanation lead is the lead of the explanation unit. Anterior contrast: share
  of referred anterior-only infarcts with an explanation lead in V1-V4 minus the share of referred
  inferior-only infarcts. Inferior contrast: share in II, III or aVF, inferior-only minus anterior-only
  (`attention_jepa` has only II of these). Patients are resampled within each group (042's
  `two_group_difference`).
- **Share explained by each layer** (rule only): among referred ECGs overall and per group (normal,
  positive, PVC, PVC with a window, benign, anterior, inferior).

**Paired contrasts, rule minus each comparator:**

- Hit - chance: per referred PVC ECG with a window, the rule's hit - chance minus the comparator's, with a
  whole-patient bootstrap of the mean (`fragment_localization.bootstrap_mean`).
- Lead contrasts (descriptive): the per-ECG difference of the lead indicator, rule minus comparator, run
  through `two_group_difference`, which equals the difference of the two contrasts with a paired interval.

## Prespecified reading

**Primary.** The rule improves on `U_B` alone if both:

1. the lower bound of the rule's hit - chance minus `U_B`'s, on referred PVC ECGs with a window, is above
   -0.10; and
2. the lower bound of the rule's anterior contrast (referred AMI-only against referred IMI-only) is above 0.

**Secondary.**

- Against `attention_jepa` alone, the same two conditions with `attention_jepa` as the comparator in (1).
  Condition (2) is the same statistic as in the primary reading; it is listed for completeness.
- Sensitivity: the whole analysis repeated with E (045's ensemble, from its saved `ensemble.npz`) as the
  readout, with its own 0.95-quantile threshold on the same normals, read by the primary rule.

If the primary reading says the rule improves, figures of the seven 041 example ECGs (IDs 47, 219, 8, 184,
287, 30, 69) are drawn with `lead_wave_maps.plot_lead_marks`, three columns: the rule (for a referred ECG,
the supplying layer's units above that layer's own 0.95-quantile threshold, plus its top unit; no marks for
a non-referred ECG), `U_B` alone and `attention_jepa` alone (each with its red units at its own threshold).
They are saved under the run's `figures/` and copied to `docs/figures/experiment-047/`. No notebook.

## Statistics

- 2,000 whole-patient bootstrap draws with seed 47047 for every interval of this experiment. The integrity
  reproductions use the seeds of the runs they reproduce (42042, 43043).
- Development data, read by Experiments 041-045; the results are exploratory. The referred subsets are
  smaller than the full groups, so the intervals are wider than 042's and 045's.

## Integrity, before any score of this experiment

- The 042, 043 Stage 1 and Stage 2 outputs match the hashes in their `result.json` files, and 043 Stage 2's
  recorded hashes of 042's outputs match (045's `load_inputs`). 045's `ensemble.npz` matches the hash in its
  `result.json`.
- Rows align (045's checks); the premature-beat rule gives 73 ECGs with 042's exclusions (0 with too few
  peaks, 11 without a premature beat).
- Single maps on all ECGs: `U_B` recomputed with seed 42042 equals 042's `result.json` exactly (045's
  `check_042`); `U_B` and `attention_jepa` with seed 43043 equal 043 Stage 2's `maps` block and its
  `attention_jepa_minus_U_B` block exactly (045's `check_043_maps`).
- Readout logits: R, `logistic_concat` and `attention_jepa` reproduce 043's AUROC and AP on full, ordinary,
  hard and SPH and their differences from R exactly (045's `check_043_detection`). E recomputed from the
  saved logits equals 045's saved development E exactly.

Any failure stops the run before the experiment's scores.

## Caveats

- Referral uses the readout's out-of-sample development logits; the threshold is fitted on the same normals
  it is evaluated on, as the maps' thresholds are.
- The premature-beat rule is automatic; the attention layer marks 8 leads only; infarct location is a
  whole-ECG statement; the contributions are the networks' own decompositions, not validated explanations.
- Restricting to referred ECGs changes the groups: an AMI that is not referred has no explanation, so the
  contrasts describe what a cardiologist would see, not the maps' accuracy on all infarcts.

## Execution

```bash
CUDA_VISIBLE_DEVICES= OPENBLAS_CORETYPE=Haswell OMP_NUM_THREADS=4 \
.venv/bin/python -m scripts.experiments.run_explanation_rule047
```

`scripts/experiments/run_explanation_rule047.py` uses `ecg_experiment/explanation_rule.py` (new),
`ecg_experiment/two_layer_map.py`, and the 042, 043 and 045 runners. It hashes its inputs, sources and this
protocol, refuses to overwrite a run, writes into a `.partial` folder and renames it to
`outputs/experiment047_explanation_rule_v1/` (`result.json`, `run.log`, `explanations.npz` and, if the rule
improves, `figures/`). The results go to `docs/experiment-047-explanation-rule-results.md`.
