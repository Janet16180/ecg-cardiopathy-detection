# Experiment 050: attention finding heads on frozen ECG-JEPA tokens

**Frozen 1 October 2026, before any score of this experiment is computed.** Experiment 043's attention head
on frozen ECG-JEPA tokens (`attention_jepa`) was trained on the binary label and almost never puts its top
token on a premature beat (2.7% of the 73 premature-beat ECGs against a 14.7% chance; 043 Stage 2), probably
because PVC is not part of the binary label. Pipelines v2-v4 detect PVC and WPW with xECG logistic finding
heads (032), and Experiment 048 explains PVC ECGs with 042's beat-wave map `U_B`. This experiment trains the
same attention head on the finding labels instead and asks:

- **(a)** Do attention heads on the tokens, trained on 032's `ventricular_ectopy` (PVC) and `preexcitation`
  (WPW) labels, detect PVC and WPW at least as well as the xECG finding heads of pipelines v3 and v4?
- **(b)** Does the PVC head's per-token contribution map point at premature beats?

The user authorized follow-up experiments. Before this freeze only the protocols and results reports of
042-048, the aggregate `result.json` fields of 044 (finding AUROCs) and 048 (the primary case), the key names
and shapes of saved files, and label counts of the rows below were read. No score of these heads exists.

## Rows, labels and tokens

- Rows and labels: exactly those of 032's `ventricular_ectopy` and `preexcitation` readouts, as 037, 044 and
  046 build them (043's `training_data`, `readouts[group]`: rows of the 55,011 stacked training rows where
  the group label is defined). PVC: 54,910 rows, 2,437 positive. WPW: 51,229 rows, 124 positive. Unweighted,
  like the xECG heads.
- Validation split: 10% of the groups of all 55,011 stacked rows (`ann_heads.validation_mask`, seed 50050;
  groups as 043: patient for PTB-XL, source and record for the Challenge sources). Each head trains on its
  rows outside the validation groups and stops early on its rows inside them. Counts and positives of each
  part are reported.
- The stacked reader table (PTB-XL record, or Challenge source, path, window start and window hash) is
  rebuilt for all 55,011 rows from 043's and 032's inputs, as 043's `training_data` builds it for the binary
  rows. Its binary rows must equal 043's `stacked` table exactly, and its groups 043's `groups`.
- Tokens: 043's cache (`<scratchpad>/exp043_cache/`: the 39,577 binary training rows, the 1,604 development
  and the 21,008 labeled SPH ECGs), reused only if its row keys and both data hashes match its receipt;
  046's cache (`<scratchpad>/exp046_cache/`: the 4,569 SPH evaluation ECGs without a binary label), reused
  only if it matches 046's recorded receipt; and a new cache (`<scratchpad>/exp050_cache/`) for the 15,434
  stacked rows in either finding readout that are not binary rows (PVC 15,419, of them 699 positive; WPW
  14,264, none positive). The new rows are extracted with 043's `extract_cache` and reader (PTB-XL
  `ptb_jepa_input`; Challenge `read_verified` at the saved window start, Ningbo windows checked against the
  manifest hash; `lead_wave_maps.jepa_tokens`). Every token mean must equal the cached JEPA feature of its
  row to 1e-4.

## Heads

- Architecture and recipe: 043 Stage 2's, unchanged. `ann_heads.AttentionHead(768)`,
  `Recipe(3e-4, 1e-2, 64, 30, 4)` (AdamW, batch 64, at most 30 epochs, patience 4), unweighted binary
  cross-entropy (all weights 1), `ann_heads.train` (best validation-AUROC epoch kept). Seeds 50050, 50051 and
  50052, with `torch.manual_seed(seed)` before each network is built. A head's prediction is the mean logit of
  its three seeds; its map is the mean per-token contribution (weight times token logit) of its three seeds.
- Two heads: `attention_pvc` and `attention_wpw`. Each seed's best weights are saved to
  `attention_finding_weights.npz`; fresh networks loaded from the file must reproduce each seed's
  development logits to 1e-6.
- Scored: the 1,604 PTB-XL full-development ECGs and the 25,577 SPH evaluation ECGs.

## Comparator

The xECG PVC and WPW heads of pipelines v2-v4 (`outputs/experiment044_pipeline_v3_v1/pipeline_v3_heads.npz`,
keys `pvc_*` and `wpw_*`), as z-scores (`pvc_switch.pvc_z` with each head's prefix). On SPH they must
reproduce 044's saved `sph_v2_z_pvc` and `sph_v2_z_wpw` to 1e-10, and their SPH AUROCs must equal 044's
(PVC 0.98982, WPW 0.99161) to 1e-12. On development they are scored from the cached xECG features.

## (a) Detection

- Sets: SPH evaluation ECGs with the group label defined (both halves, as 032 and 044): PVC 25,577 ECGs
  (1,058 positive), WPW 25,566 (27 positive); PTB-XL full development with the label defined: PVC 1,600 (84
  positive), WPW 1,604 (6 positive).
- **Primary:** SPH AUROC of `attention_pvc` minus the xECG PVC head, paired whole-patient bootstrap, 2,000
  draws, seed 50050 (`intervals.paired_auroc_difference`). Reading: **beats** if the lower bound is above 0,
  **matches** if it is above −0.005, otherwise **below**.
- Secondary (read by the same rule, no decision): the PVC difference on full development; the WPW
  differences on SPH and development (few positives; counts reported); AUROC and average precision of each
  head; each seed's AUROC; best epochs.

## (b) Map

- Rows: 042's 1,604 full-development ECGs (045's `load_inputs`), with 042's premature-beat windows (041's
  rule; 73 ECGs). The map of an ECG is `lead_wave_maps.jepa_unit_map` of `attention_pvc`'s mean
  contributions (400 tokens: leads I, II, V1-V6 by 50 patches of 0.2 s).
- Metrics: 043's `map_metrics` with seed 50050 on `U_B` (042), `attention_jepa` (043's saved token maps) and
  `attention_pvc`: hit, chance and hit − chance on the 73 ECGs, the lead contrasts, any red, worst-unit AUROC
  of the binary label. Paired contrasts with 043's `map_contrasts`: `attention_pvc` minus `U_B`, and
  `attention_pvc` minus `attention_jepa`.
- **Primary for (b):** `attention_pvc`'s hit − chance minus `U_B`'s, paired per ECG; **keeps** premature-beat
  localization if the lower bound is above −0.10 (042's margin), otherwise **does not keep**. Also reported:
  042's full map reading (`map_reading`) for `attention_pvc` against `U_B`.
- The lead and time of `attention_pvc`'s top token on the PVC example ECG 219, and whether it overlaps a
  premature-beat window, beside `U_B`'s and `attention_jepa`'s top units.

## Secondary, only if (b) keeps premature-beat localization: a one-network explanation rule

048's explanation rule with the attention PVC head as both the switch and the rhythm layer:

- Referral and rows as 048's primary case: `logistic_concat`'s development logit above the 0.95 quantile of
  the 463 NORM-only normals' logits (792 referred ECGs).
- Switch: `attention_pvc`'s mean logit strictly above the 0.975 quantile (NumPy's default interpolation) of
  the 463 normals' `attention_pvc` logits (as 048's q = 0.975 on its own z-score; a quantile does not depend
  on the standardization).
- Explanation of a referred ECG: `attention_pvc`'s top token if the switch is on, otherwise
  `attention_jepa`'s top token. 048's rule: `U_B`'s top unit if 048's xECG z_PVC is above 048's s(0.975),
  otherwise `attention_jepa`'s.
- Metrics (048's, `explanation_rule.explanation_metrics` and `paired_difference`, seed 50050): hit − chance
  on referred PVC ECGs with a window, the anterior and inferior contrasts, and the infarcts sent to the rhythm
  layer (count and share of referred anterior-only and inferior-only infarcts, and of referred PVC ECGs with a
  window).
- Reading against 048 (descriptive, no decision): the 050 rule **improves on 048** if the lower bound of its
  hit − chance minus 048's is above −0.10 and the lower bound of its anterior-contrast difference (050 minus
  048) is above 0; it **keeps 048's localization** if only the first holds.
- Integrity: 048's rule, recomputed here with seed 48048, must reproduce 048's `result.json` primary case
  (`cases["logistic_concat_q0.975"]`: `referred`, `sent_to_U_B` and `maps["rule"]`) exactly.

## Integrity, before any score of this experiment

- Every input matches its receipt: 042's `unit_scores.npz`; 043 Stage 1 `predictions.npz`; 043 Stage 2
  `predictions.npz` and `token_maps.npz`; 044's `pipeline_v3_heads.npz` and `predictions.npz`; 048's
  `explanations.npz`; 037 and 035 (through 043's loaders).
- 043's training rows equal 037's (`check_against_037`); the rebuilt stacked table and groups as above.
- The three caches as above.
- `U_B` reproduces 042 (045's `check_042`), and `U_B` and `attention_jepa` reproduce 043 Stage 2's `maps`
  block (045's `check_043_maps`).
- The comparator checks above.

Any failure stops the run.

## Profile and stopping

Before any extraction or training the runner times the extraction of 128 of the new rows and 30 optimizer
steps (043's `profile_extraction` and `profile_training`) and stops if the projected time of the run
(extraction, both heads with every epoch run, scoring) exceeds 2 hours.

## What to expect (reasoning before any score)

The xECG PVC head reaches 0.990 at SPH; there is little room above it, so "matches" is the likely reading
for (a), and "beats" would need a tight interval. A head trained on PVC labels should attend to the tokens of
the ectopic beat, so its hit rate should rise far above `attention_jepa`'s 2.7%. To keep localization by the
−0.10 margin against `U_B` (hit − chance +0.734) it needs a hit − chance of about +0.65 or more. Tokens are
0.2 s patches of 8 leads, coarser than `U_B`'s beat waves, so a premature beat's QRS may straddle two patches;
the outcome is uncertain. WPW has 124 training positives and 27 SPH positives; its intervals will be wide.

## Closed data and exclusions

- No PTB-XL calibration or test ECG, no Challenge calibration or test-group ECG, no EchoNext record. SPH is
  development data (read by 022-048).
- Besides tokens, the run reads the 84 PVC records' PTB-XL waveforms (041's windows, as 042-048) and the
  waveforms of the 15,434 new training rows.
- No age or other subgroup analysis; no label definition changes.
- New files only: `ecg_experiment/attention_findings.py`, `tests/test_attention_findings.py`,
  `scripts/experiments/run_attention_findings050.py`, this protocol and the results report. The runner imports
  the 043, 044 and 045 runners (accepted exceptions) and the 042 and 032 functions they import.

## Caveats written into the results

- Development data, read by earlier experiments; exploratory.
- The networks see 90% of their rows; the xECG heads saw all of them.
- The PVC label is ECG-level; the premature-beat windows come from an automatic rule.
- The contributions are the networks' own decomposition, not a validated explanation.
- Three seeds of one recipe, with no hyperparameter search.

## Execution

```bash
OPENBLAS_CORETYPE=Haswell OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 \
    .venv/bin/python -u -m scripts.experiments.run_attention_findings050 --smoke \
    --output outputs/experiment050_attention_findings_smoke
OPENBLAS_CORETYPE=Haswell OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 \
    .venv/bin/python -u -m scripts.experiments.run_attention_findings050
```

GPU 0 under `ecg_experiment.gpu.gpu_lock("cuda")` for all CUDA work. The smoke mode uses training rows only
(4,000 of the PVC readout's rows, a held-out quarter by group, 2 epochs, 64 new rows extracted into a smoke
cache; no development or SPH ECG is scored). The runner hashes inputs, sources and this protocol, writes to
`<output>.partial` and renames it, and refuses to overwrite. The full run writes
`outputs/experiment050_attention_findings_v1/` (`result.json`, `predictions.npz`, `token_maps.npz`,
`attention_finding_weights.npz`, `run.log`). Results go to
`docs/experiment-050-attention-finding-heads-results.md`.
