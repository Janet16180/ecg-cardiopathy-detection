# Experiment 049: a label-free focal-versus-diffuse switch for the explanation, from saved scores

**Frozen 1 October 2026, before any score of this experiment is computed.** Experiment 048 explained a
referred ECG with `U_B`'s top unit when the xECG PVC head's z-score was above the 97.5th percentile of the
normals', and otherwise with `attention_jepa`'s top token. It kept all of `U_B`'s premature-beat hits
(hit - chance +0.690 [0.599, 0.766]) and gained lead information over `U_B` (anterior contrast +0.189
[0.067, 0.307]). But the PVC head is not specific. It sent half of the referred anterior-only infarcts
(68 / 136) and 38% of the inferior-only infarcts (43 / 112) to `U_B`, whose top lead says little about infarct
location.

The idea here is that a premature beat is one beat unlike the ECG's other beats, while an infarct or a
bundle branch block changes every beat. A switch on the ratio of the worst beat to the typical beat should
separate the two. It uses neither 041's RR-based premature-beat rule, which defines the ground truth, nor any
label. Saved scores and saved head parameters only; nothing is trained.

## Inputs and rows

- Everything 048 reads: 045's `load_inputs` (042's `U_B` units and rows, 043 Stage 1 and Stage 2), 045's
  `ensemble.npz`, 047's and 048's `result.json` and `explanations.npz`, and 044's `pipeline_v3_heads.npz`
  and `predictions.npz`.
- xECG features of the 1,604 development and 25,577 SPH ECGs (043's `ptb_inputs`, `sph_inputs`). Rows:
  042's 1,604 full-development ECGs.

## Focal ratio

- `U_B` units are saved per ECG in the order of `lead_wave_maps.beat_unit_map`: unit index
  = (beat x 12 + lead) x 4 + wave. Each beat therefore has 48 consecutive units. The run checks the layout
  before using it: the unit count is a multiple of 48, the lead index of unit u is (u // 4) mod 12, and
  within a beat every lead has the same four wave start times.
- Beat score: the maximum over the beat's 48 units. Focal ratio of an ECG: its maximum beat score divided by
  its median beat score (NumPy `median`).
- Switch threshold f(q): the q-quantile (linear interpolation) of the 463 NORM-only normals' focal ratios.
  Primary q = 0.975; sensitivity q = 0.95 and q = 0.99. The focal switch is on when the ratio is strictly
  above f(q).

## PVC and WPW scores

- z_PVC as in 048 (`pvc_switch.pvc_z` with 044's heads). 048's PVC switch: z_PVC above s(0.975) = 1.654,
  the 0.975 quantile of the normals' z_PVC.
- z_WPW: `pvc_switch.pvc_z(heads, x, prefix="wpw")`, the same formula with the `wpw_*` arrays. F = max(z_PVC,
  z_WPW), as in pipeline v3.

## Arms (explanation of a referred ECG)

- `focal_switch` (primary): `U_B`'s top unit if the focal switch is on, otherwise `attention_jepa`'s top token.
- `focal_and_pvc`: `U_B` only when both the focal switch and 048's PVC switch are on.
- `focal_or_pvc`: `U_B` when either is on.
- Comparators: `U_B` alone, `attention_jepa` alone, and 048's `pvc_switch`.

For the q sensitivity analyses, q changes the focal threshold only; the PVC switch stays at s(0.975).

## Referral

- Primary: 047's, the `logistic_concat` development logit strictly above the 0.95 quantile of the 463
  normals' logits (792 referred ECGs).
- Secondary: E (045's ensemble) at its own 0.95 quantile.
- Secondary, `combined_50`: pipeline v3's referral rule (033's `combined_50`, `finding_screen.split_thresholds`
  with a 50 per-mille budget and a 50 per-mille share for F) fitted on the 463 development normals. The
  matrix columns are the `logistic_concat` logit and F. With k = floor(0.05 x 463) = 23, F's threshold is the
  (r + 1)-th highest normal F, r = floor(50 x 23 / 1000) = 1. The readout's threshold is chosen by
  `split_thresholds` so that at most k - 1 = 22 normals are above either threshold. An ECG is referred when
  it is strictly above either threshold (`finding_screen.referred_matrix`). The logit is used in place of
  044's probability; the two are monotone, so the referrals are the same. This variant refers PVC-only ECGs
  that the readout alone does not refer, such as example 219.

## Metrics, on referred ECGs only (047's definitions and helpers)

For each arm and comparator: premature-beat hit, chance and hit - chance on referred ECGs among the 73 with
a window; anterior and inferior lead contrasts on the referred anterior-only and inferior-only infarcts; the
share explained by each layer; and the cost on infarcts, the number of referred anterior-only and
inferior-only infarcts sent to `U_B`, reported next to 048's. Paired contrasts, arm minus each comparator,
use `explanation_rule.paired_difference` (hit - chance per ECG, and the lead contrasts through the per-ECG
indicator difference). The run also reports the focal ratio's median and 97.5th percentile in normals,
positives, PVC ECGs (all 84 and the 73 with a window), benign variants and the anterior-only and inferior-only
infarcts, and the share of each group with the focal switch on.

## Prespecified reading

**Primary** (`logistic_concat` referral, q = 0.975). `focal_switch` improves on 048's PVC switch if both:

1. the lower bound of its hit - chance minus the PVC switch's, on referred PVC ECGs with a window, is above
   -0.10; and
2. the lower bound of its anterior contrast minus the PVC switch's (paired) is above 0.

**Secondary.**

- Each arm against `U_B` alone by 048's rule: the hit - chance difference's lower bound above -0.10, and the
  arm's own anterior contrast's lower bound above 0.
- `focal_and_pvc` and `focal_or_pvc` against the PVC switch by the primary rule.
- Sensitivity: q = 0.95 and q = 0.99, E referral, and `combined_50` referral, each read by the primary rule
  and by the rule against `U_B`.

If the primary reading says `focal_switch` improves, figures of the seven 041 example ECGs (IDs 47, 219, 8,
184, 287, 30, 69) are drawn with `pvc_switch.plot_layer_marks` (two colours by layer) under the `combined_50`
referral, so that 219 is shown. They have three columns: `focal_switch` (the supplying layer's units above
that layer's own threshold, plus its top unit; nothing for a non-referred ECG), `U_B` alone and
`attention_jepa` alone. They are saved under the run's `figures/` and copied to
`docs/figures/experiment-049/`.

## Statistics

- 2,000 whole-patient bootstrap draws with seed 49049. The integrity reproductions use the seeds of the
  runs they reproduce (42042, 43043, 47047, 48048).
- Development data, read by Experiments 041-048; the results are exploratory. All thresholds are fitted on
  the same 463 normals. q = 0.975 was chosen before any focal ratio was computed.

## Integrity, before any score of this experiment

- 048's integrity checks: 047's checks (receipts, rows, 73 premature-beat ECGs, `U_B` against 042, both maps
  against 043, readout logits against 043, E against 045, 047's analysis reproduced exactly); 044's file
  hashes; z_PVC on SPH against 044's `sph_v2_z_pvc` to 1e-10.
- z_WPW on SPH reproduces 044's `sph_v2_z_wpw` to an absolute 1e-10, and F reproduces `sph_v2_z_combined` to
  1e-10.
- 048's `explanations.npz` matches its receipt, its `z_pvc` equals the recomputed development z_PVC exactly,
  and 048's analysis (its runner's `analyse`, seed 48048) recomputed for all four of its cases equals 048's
  `result.json` `cases` block exactly.
- The `U_B` unit layout check above passes for all 1,604 ECGs.

Any failure stops the run before the experiment's scores.

## Caveats

- The focal ratio depends on the number of beats: with two beats the ratio is at most 2. A premature beat
  in a short or slow recording, or several premature beats (bigeminy), make the worst beat less of an
  outlier.
- A focal artefact (a noisy beat) also raises the ratio. An infarct with an ectopic beat can switch on, which
  is correct for the rhythm but hides the infarct leads.
- The premature-beat rule is automatic; the attention layer marks 8 leads; infarct location is a whole-ECG
  statement; the contributions are the networks' own decompositions.

## Execution

```bash
CUDA_VISIBLE_DEVICES= OPENBLAS_CORETYPE=Haswell OMP_NUM_THREADS=4 \
.venv/bin/python -m scripts.experiments.run_focal_switch049
```

`scripts/experiments/run_focal_switch049.py` uses a new `ecg_experiment/focal_switch.py`,
`ecg_experiment/pvc_switch.py` and `ecg_experiment/explanation_rule.py` (both unchanged, because 047's and
048's receipts pin them), and the 042-048 runners. It hashes its inputs, sources and this protocol, refuses
to overwrite a run, writes into a `.partial` folder and renames it to `outputs/experiment049_focal_switch_v1/`
(`result.json`, `run.log`, `explanations.npz` and, if the primary arm improves, `figures/`). The results go
to `docs/experiment-049-focal-switch-results.md`.
