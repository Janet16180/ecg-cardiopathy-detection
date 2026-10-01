# Experiment 048 results: switch the explanation on the PVC finding head

Completed 1 October 2026 under the [frozen protocol](experiment-048-pvc-switch.md) (commit `1e0fec8`; run
from `3bd6cf9`). Saved scores and saved head parameters only: nothing was trained. Besides saved scores and
cached xECG features, the run read the 84 PVC records' PTB-XL waveforms (041's premature-beat windows) and
the seven example ECGs' waveforms for the figures. SPH features were read only for the integrity check. No
calibration, test or EchoNext record was read. Outputs are in `outputs/experiment048_pvc_switch_v1/`
(`result.json` with every input, source and protocol hash, `explanations.npz`, `figures/`, `run.log`).
Figures are copied to [docs/figures/experiment-048/](figures/experiment-048/). The run took 82 seconds on CPU.

## Readings

All cases use `logistic_concat` referral (792 referred ECGs) unless marked E. The primary case is q = 0.975.

| Case | Hit - chance minus `U_B`'s (lower bound > -0.10) | Rule's anterior contrast (lower bound > 0) | Against `U_B` | Against `attention_jepa` |
| --- | --- | --- | --- | --- |
| q = 0.975 (primary) | +0.000 [0.000, 0.000] | +0.189 [0.067, 0.307] | **Improves** | Improves |
| q = 0.95 | +0.000 [0.000, 0.000] | +0.156 [0.033, 0.280] | Improves | Improves |
| q = 0.99 | -0.020 [-0.061, 0.000] | +0.146 [0.028, 0.264] | Improves | Improves |
| E referral, q = 0.975 | +0.000 [0.000, 0.000] | +0.195 [0.072, 0.314] | Improves | Improves |

The rule improves on `U_B` alone by the primary reading, and on `attention_jepa` alone by the secondary one.
Every sensitivity case gives the same readings.

## Integrity

- All of 047's checks passed again. The input receipts match. The rows align. There are 73 premature-beat
  ECGs, with 042's exclusions (0 with too few peaks, 11 without a premature beat). `U_B` equals 042's numbers
  exactly; `U_B` and `attention_jepa` equal 043 Stage 2's exactly. The readout logits reproduce 043's AUROC
  and AP exactly, and E equals 045's saved E exactly.
- 047's `explanations.npz` matches its receipt. 047's analysis, recomputed with seed 47047, equals 047's
  `result.json` `readouts` block exactly for both readouts.
- 044's `pipeline_v3_heads.npz` and `predictions.npz` match 044's receipt, and the heads hash appears in
  `docs/pipeline-v3.md`. 044's SPH IDs and development record IDs equal 043's. z_PVC recomputed on the
  25,577 SPH ECGs reproduces 044's `sph_v2_z_pvc` with a largest difference of 4.3e-15, within the 1e-10
  tolerance. The xECG PVC head is the one pipelines v2 and v3 share (`pipeline-v3.md` section 3; 044's
  results report that the refitted head reproduced 037's z-scores exactly).

## Switch thresholds and where the switch is on

Thresholds are quantiles of the 463 normals' z_PVC: s(0.95) = 1.20, s(0.975) = 1.65, s(0.99) = 2.19.

| Group, all ECGs (not only referred) | q = 0.95 | q = 0.975 | q = 0.99 |
| --- | ---: | ---: | ---: |
| Normals | 0.052 | 0.026 | 0.011 |
| Positives | 0.469 | 0.371 | 0.275 |
| PVC ECGs (84) | 1.000 | 1.000 | 0.988 |
| PVC ECGs with a window (73) | 1.000 | 1.000 | 0.986 |
| Benign variants | 0.192 | 0.115 | 0.058 |
| Anterior-only infarcts | 0.596 | 0.479 | 0.329 |
| Inferior-only infarcts | 0.423 | 0.336 | 0.228 |

## Referred ECGs, primary case

Referral is 047's: threshold 0.271, 792 referred ECGs (24 normals, 680 positives, 60 PVC ECGs, 49 with a
window, 6 benign variants, 136 anterior-only and 112 inferior-only infarcts). Intervals use seed 48048,
2,000 whole-patient draws. Anterior hits give the share with an explanation lead in V1-V4, anterior-only
against inferior-only.

| Explanation | Hit | Chance | Hit - chance | Anterior contrast | Anterior hits | Inferior contrast |
| --- | ---: | ---: | --- | --- | --- | --- |
| Rule (PVC switch) | 0.898 | 0.208 | +0.690 [0.599, 0.766] | +0.189 [0.067, 0.307] | 0.537 / 0.348 | +0.164 [0.055, 0.273] |
| `U_B` alone | 0.898 | 0.208 | +0.690 [0.599, 0.766] | +0.031 [-0.097, +0.162] | 0.522 / 0.491 | +0.105 [-0.015, +0.231] |
| `attention_jepa` alone | 0.020 | 0.160 | -0.140 [-0.178, -0.091] | +0.197 [0.084, 0.314] | 0.412 / 0.214 | +0.212 [0.109, 0.312] |
| 047's rule (`U_B` red) | 0.755 | 0.201 | +0.554 [0.441, 0.672] | +0.215 [0.092, 0.334] | 0.537 / 0.321 | +0.194 [0.086, 0.297] |

| Paired difference | Hit - chance | Anterior contrast | Inferior contrast |
| --- | --- | --- | --- |
| Rule minus `U_B` | +0.000 [0.000, 0.000] | +0.158 [0.029, 0.287] | +0.060 [-0.058, +0.172] |
| Rule minus `attention_jepa` | +0.830 [0.730, 0.913] | -0.009 [-0.132, +0.105] | -0.047 [-0.152, +0.056] |
| Rule minus 047's rule | +0.136 [0.056, 0.228] | -0.027 [-0.128, +0.067] | -0.029 [-0.116, +0.054] |

All 49 referred PVC ECGs with a window go to `U_B`, so the rule's premature-beat numbers equal `U_B`'s.

**Cost on infarcts.** The switch sends 68 of the 136 referred anterior-only infarcts (50%) and 43 of the 112
referred inferior-only infarcts (38%) to `U_B`. 047's rule sent 33% and 18%.

Share of referred ECGs explained by `U_B`: 0.400 overall, 0.000 of normals, 0.429 of positives, 1.000 of PVC
ECGs, 0.167 of benign variants (1 of 6). `U_B` was itself red on 0.498 of the ECGs it explained, and
`attention_jepa` on 0.444 of the ECGs it explained.

### Sensitivity cases

| Case | Sent to `U_B`: anterior / inferior / PVC with a window | Rule hit - chance | Rule anterior contrast | Rule inferior contrast |
| --- | --- | --- | --- | --- |
| q = 0.95 | 84 / 54 / 49 of 136 / 112 / 49 | +0.690 [0.599, 0.766] | +0.156 [0.033, 0.280] | +0.141 [0.030, 0.256] |
| q = 0.975 | 68 / 43 / 49 | +0.690 [0.599, 0.766] | +0.189 [0.067, 0.307] | +0.164 [0.055, 0.273] |
| q = 0.99 | 48 / 28 / 48 | +0.670 [0.575, 0.752] | +0.146 [0.028, 0.264] | +0.147 [0.042, 0.249] |
| E, q = 0.975 | 69 / 42 / 51 of 139 / 108 / 51 | +0.693 [0.605, 0.767] | +0.195 [0.072, 0.314] | +0.163 [0.052, 0.277] |

With E, the rule minus `U_B` anterior contrast is +0.195 [0.065, 0.324]; with q = 0.99 it is +0.115
[-0.028, +0.263].

## Example figures

Three columns per figure: the rule, coloured by the supplying layer (`U_B` red, `attention_jepa` blue), then
`U_B` alone and `attention_jepa` alone at their own thresholds. Under `logistic_concat` referral, only 2 of
the 7 examples are referred, and `U_B` explains both. AMI 184 has z_PVC 1.71, so it goes to `U_B`, which marks
aVL at 6.21 s, not an anterior lead. CLBBB 287 has z_PVC 6.94; `U_B` marks V1 at 9.18 s. The PVC example
(219, z_PVC 6.99) is not referred by the binary readout, so the rule column is empty; its `U_B` column marks
the premature beat in II and aVF. NORM 47, IMI 8, LVH 30 and benign 69 are not referred.

## Interpretation

- **The switch fixes 047's failure.** The PVC head is on for every PVC ECG with a premature-beat window, so
  the rule keeps all of `U_B`'s premature-beat hits (90%, against 76% for 047's rule; +0.136 [0.056, 0.228]).
  It is not circular: the head was trained on ECG-level PVC labels from the training split, not on 041's
  rule.
- **The lead information is mostly kept.** The anterior contrast (+0.189) sits between `U_B`'s (+0.031) and
  `attention_jepa`'s (+0.197). Its interval excludes zero, and it is above `U_B`'s by +0.158 [0.029,
  0.287]. Against `attention_jepa` and against 047's rule, the anterior and inferior differences have
  intervals that include zero.
- **The cost is real.** The PVC head is not specific: it is on for 48% of all anterior-only and 34% of all
  inferior-only infarcts, 12% of benign variants and 37% of positives. Among referred infarcts, half the
  anterior and 38% of the inferior ones are explained by `U_B`, whose top lead carries no infarct-location
  information. AMI 184 is an example: its explanation is in aVL. The contrast survives because the other
  half still goes to attention. The CLBBB example (z_PVC 6.94) suggests that wide-QRS ECGs drive the head.
- **The threshold matters little between 0.95 and 0.99.** q = 0.99 sends fewer infarcts to `U_B` (48 and 28)
  but loses one PVC ECG and does not raise the anterior contrast (+0.146). q = 0.975 has the highest anterior
  contrast of the three, but the three intervals overlap widely; this does not support tuning q.

## Caveats

- 49 referred PVC ECGs, 136 and 112 referred infarcts; one PVC ECG moves the hit rate by 2 points.
- Referral here is the binary readout alone. Full pipeline v3 also refers on the finding score F =
  max(z_PVC, z_WPW), which would refer PVC example 219. Rule results on F-referred ECGs were not assessed.
- The switch, referral and map thresholds are all fitted on the same 463 normals.
- The premature-beat rule is automatic; the attention layer marks 8 leads; infarct location is a whole-ECG
  statement; the contributions are the networks' own decompositions.
- Development data, read by Experiments 041-047; the results are exploratory.

## Deviations

- None from the protocol. The brief suggested putting the new plotting helper in
  `ecg_experiment/explanation_rule.py`. Because 047's receipt pins that file, the protocol put it in a new
  `ecg_experiment/pvc_switch.py`, and `explanation_rule.py` was not changed.

## Follow-up ideas

- Refer with full pipeline v3 (readout or F at the combined budget) and run the same rule, so PVC-only ECGs
  such as 219 are explained.
- A more specific switch: z_PVC above threshold and a QRS width below an LBBB cutoff, or the PVC head minus
  a conduction head. Its aim is to keep PVC ECGs on `U_B` while sending conduction and infarct ECGs to
  attention. It needs a protocol and a check of how much it costs on PVC ECGs.
- Show both layers on referred ECGs with switch-on (rhythm marks from `U_B` and morphology marks from
  attention, in two colours), so infarcts with a high PVC score keep their lead marks.
- Review the seven figures, and a sample of switch-on infarcts, with the cardiologist.
