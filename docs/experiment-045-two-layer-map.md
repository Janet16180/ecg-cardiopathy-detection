# Experiment 045: a two-layer explanation map and a detection ensemble, from saved scores

**Frozen 1 October 2026, before any score of this experiment is computed.** Experiments 042 and 043 left two
maps with complementary strengths. Experiment 042's beat-aligned wave map `U_B` puts its top unit on a
rule-detected premature beat in 93% of PVC ECGs (hit - chance +0.734), but its top lead says little about where
an infarct is (anterior contrast +0.085 [-0.038, +0.202]). Experiment 043's `attention_jepa` contribution map
points infarcts to the right leads (anterior +0.190 [0.080, 0.299], inferior +0.205 [0.112, 0.300]) but almost
never to premature beats (top token on the premature beat in 2.7%, below the 14.7% chance). The two 043
readouts with the best detection, `logistic_concat` (pipeline v3's binary readout, Experiment 044) and
`attention_jepa`, use different heads on overlapping features. This experiment asks two questions on saved
scores only; nothing is trained or extracted.

- **A.** Does a two-layer map (show `U_B` when it is red, otherwise `attention_jepa`) improve on `U_B` alone
  by 042's rule?
- **B.** Does the mean of the two readouts' logits detect better than pipeline v3's readout?

## Inputs and rows

- Experiment 042: `outputs/experiment042_lead_wave_maps_v1/unit_scores.npz` (`U_B` units with lead, start
  and end) and `result.json`. Rows: 042's `select_rows` (imported from its runner) on the 1,604
  full-development ECGs: 1,306 binary-evaluation ECGs (843 positive, 463 NORM-only normals), 84 PVC ECGs
  (73 with a rule-detected premature beat), 52 benign variants, 146 anterior-only and 149 inferior-only
  infarcts.
- Experiment 043 Stage 2: `stage2/token_maps.npz` (`attention_jepa`, the 3-seed mean per-token contribution,
  400 tokens = leads I, II, V1-V6 by 50 patches of 0.2 s) and `stage2/predictions.npz` (`attention_jepa`'s
  3-seed mean logit on the 1,604 development and 21,008 labeled SPH ECGs).
- Experiment 043 Stage 1: `stage1/predictions.npz` (`logistic_concat` and R logits on the 1,604 development
  and 25,577 SPH evaluation ECGs, of which 21,008 are labeled).
- Detection sets: 043's `evaluation_sets` (imported from its runner) built from Experiment 032's PTB-XL and
  SPH rows: full development 1,572 (884 positive, 1,413 patients), ordinary 1,306, hard 266, SPH 21,008
  (7,190 positive, 20,364 patients).
- Premature-beat windows: 041's rule via 042's `premature_targets`, which reads the 84 PVC records' raw
  PTB-XL waveforms (development data, read by 041-043) to detect R peaks. This is the only waveform reading,
  besides the example figures. No calibration, test or EchoNext record is read.

## A. Two-layer explanation map

**Layers.** Layer 1 is `U_B`: units are (beat, lead, wave), with time spans. Layer 2 is `attention_jepa`:
units are (lead, 0.2 s patch), in 8 leads. A layer's worst-unit score is the maximum over the ECG's units.

**Thresholds.** For a quantile q, layer k's threshold t_k(q) is the q-quantile (NumPy's default linear
interpolation, as in 042) of layer k's worst-unit scores on the 463 NORM-only normals. Layer k is red on an
ECG when its worst-unit score is strictly above t_k(q). An ECG is red when either layer is red. On the grid
q = 0.9000, 0.9001, ..., 1.0000 (1,001 values), q is chosen so that the share of the 463 normals that are red
is nearest to 0.05 (the achievable shares are k / 463). If several grid values give that share, q is the
median of those grid values (the lower middle value if their number is even). q uses only the normals'
scores; no label of any other ECG.

**Explanation.** The explanation unit of an ECG is layer 1's top unit if layer 1 is red, otherwise layer 2's
top unit (whether or not layer 2 is red). Every ECG has one.

**Tests, 042's definitions on the same rows:**

- Premature beats (73 ECGs): `hit` is 1 if the explanation unit's time span overlaps a premature-beat window;
  `chance` is the share of the supplying layer's units that overlap one; hit - chance with a whole-patient
  bootstrap interval.
- Lead contrasts: the explanation lead is the lead of the explanation unit. Anterior contrast: share with an
  explanation lead in V1-V4, anterior-only minus inferior-only infarcts. Inferior contrast: share in II, III or
  aVF (layer 2 has only II), inferior-only minus anterior-only. Patients are resampled within each group.
- Any red: share of red ECGs among normals, positives, PVC ECGs and benign variants. The ECG-level any-red
  sensitivity at the 5% budget is the positives' share, with a whole-patient bootstrap interval. Mean red
  units (both layers' red units added) per group.
- The worst-unit AUROC does not apply to the combined map and is not computed for it. The two single layers'
  worst-unit AUROCs are reported for reference.
- Descriptive: the share of each group explained by layer 1.

**Single-layer rows.** `U_B` and `attention_jepa` are scored alone with their own 95th-percentile thresholds
(042's rule), so the table has three rows: `U_B`, `attention_jepa` and `combined`.

**Paired contrasts, combined minus `U_B`, per ECG:** hit - chance on the 73 premature-beat ECGs; benign
any-red on the 52 benign variants; positive any-red on the 843 positives. Each is a whole-patient bootstrap of
the per-ECG difference (`fragment_localization.bootstrap_mean`).

**Prespecified reading (042's rule, adapted as above).** The combined map improves on `U_B` alone if

1. it keeps premature-beat localization: the lower bound of its hit - chance minus `U_B`'s is above -0.10;
   and
2. it gains at least one of: (a) its anterior contrast's lower bound is above 0; (b) the benign any-red
   difference's upper bound is below 0; (c) the positive any-red difference's lower bound is above 0.

If it improves, figures of the seven 041 example ECGs (47, 219, 8, 184, 287, 30, 69) are drawn with both
layers' red marks per lead (`lead_wave_maps.plot_lead_marks`, one column per layer) and copied to
`docs/figures/experiment-045/`. No notebook is written by this experiment.

## B. Detection ensemble

- E = (`logistic_concat` logit + `attention_jepa` mean logit) / 2, on the development and labeled SPH ECGs.
  No weight, scale or threshold is fitted.
- **Primary:** AUROC of E minus `logistic_concat` (pipeline v3's readout) on SPH and on full development,
  paired whole-patient bootstrap, 2,000 draws (`intervals.paired_auroc_difference`).
- **Prespecified reading:** E **beats** v3's readout if the SPH difference's lower bound is above 0 and the
  full-development difference is at least -0.005; it **matches** if the SPH lower bound is above -0.01 and the
  full difference is at least -0.01; otherwise it is **below** (043's `detection_reading`).
- Secondary: E minus R (pipeline v2's readout) on SPH and full development, read by the same rule; AUROC and
  AP of E, `logistic_concat`, `attention_jepa` and R on full, ordinary, hard and SPH.

## Statistics

- 2,000 whole-patient bootstrap draws with seed 45045 for every interval of this experiment, including the
  three map rows. The integrity reproductions use the seeds of the runs they reproduce (42042 for 042, 43043
  for 043), so the single-layer intervals in this experiment's table can differ slightly from the reported
  ones while their point values are identical.
- Development and SPH were read by earlier experiments; the results are exploratory.

## Integrity, before any score of this experiment

- The 042 `unit_scores.npz` and the 043 Stage 1 `predictions.npz` and Stage 2 `predictions.npz` and
  `token_maps.npz` match the hashes in their `result.json`; 043 Stage 2's recorded hash of 042's
  `result.json` matches the file.
- Rows align: 042's scored ECG IDs equal the saved `U_B` and `attention_jepa` IDs; the Stage 1 and Stage 2
  development record IDs are equal; the Stage 2 SPH IDs equal the Stage 1 SPH IDs at the labeled positions.
  The premature-beat windows give 73 ECGs and 042's exclusions (0 too few peaks, 11 without a premature beat).
- `U_B` recomputed with seed 42042 reproduces 042's `result.json` exactly: threshold, AUROC, AP, any red, mean
  red units, sensitivity at the budget, both lead contrasts, PVC count, hit, chance and hit - chance.
- `U_B` and `attention_jepa` recomputed with 043's `map_metrics` and seed 43043 reproduce 043 Stage 2's
  `maps` block exactly, and `map_contrasts` reproduces its `attention_jepa_minus_U_B` block exactly.
- The saved logits reproduce 043's AUROC and AP of R, `logistic_concat` and `attention_jepa` on all four sets
  exactly, and their differences from R (seed 43043) exactly.

Any failure stops the run before the experiment's scores.

## Caveats

- The share nearest 0.05 is 23 / 463 = 0.0497 if the grid reaches it, while each single layer at its 95th
  percentile marks 24 / 463 = 0.0518. The combined map would then be compared at a budget one normal ECG
  stricter than `U_B`.
- The positive any-red gain (c) mostly measures detection at a fixed normal budget, not localization.
- Layer 2 cannot mark III, aVR, aVL or aVF. Infarct location is a whole-ECG statement, not a marked region.
- The attention contributions are the network's own decomposition, not a validated explanation.
- Averaging logits of two heads with different scales is a fixed, unfitted choice.

## Execution

```bash
CUDA_VISIBLE_DEVICES= OPENBLAS_CORETYPE=Haswell OMP_NUM_THREADS=4 \
.venv/bin/python -m scripts.experiments.run_two_layer_map045
```

`scripts/experiments/run_two_layer_map045.py` uses `ecg_experiment/two_layer_map.py` and imports row and
metric helpers from the 042 and 043 runners. It hashes its inputs, sources and this protocol, refuses to
overwrite a run, writes into a `.partial` folder and renames it to
`outputs/experiment045_two_layer_map_v1/` (`result.json`, `run.log`, `combined_map.npz`, `ensemble.npz` and,
if the map improves, `figures/`). The results go to `docs/experiment-045-two-layer-map-results.md`.
