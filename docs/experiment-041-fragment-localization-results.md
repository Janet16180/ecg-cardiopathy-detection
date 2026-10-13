# Experiment 041 results: marking the abnormal sections of an ECG

Completed 30 September 2026 under the [frozen protocol](experiment-041-fragment-localization.md) (commit
`086cc87`; run from `27c392a`). PTB-XL development data only; no calibration or test ECG was read. Outputs
are in `outputs/experiment041_fragment_localization_v1/` (`result.json` with every input, source and
protocol hash, `section_scores.npz`, the fitted `reference.pkl`, `figures/` and `run.log`). Figures are copied
to [docs/figures/experiment-041/](figures/experiment-041/). The run took 275 seconds, including 51 seconds
of GPU work on 7,476 ECGs.

## Readings

| Prespecified reading | Statistic | Result |
| --- | --- | --- |
| Part 1 localizes | `U` mean(hit - chance) on premature beats, lower bound > 0 | **Yes**: +0.669 [0.582, 0.758] |
| Part 1 keeps detection | `U` AUROC at most 0.05 below the whole-ECG Mahalanobis | **No**: 0.777 against 0.923 |
| Part 2 improves detection | `G` minus `U` AUROC, lower bound > 0 | **Yes**: +0.149 [0.123, 0.175] |
| Part 2 marks fewer benign variants | `G` minus `U` benign any-red share, upper bound < 0 | **No**: 0.000 [-0.115, +0.135] |

Because Part 1 did not keep detection, the protocol's JEPA fallback (Experiment 041b) is triggered. It needs
its own frozen protocol.

## Integrity

- The refitted whole-ECG xECG Mahalanobis reproduced Experiment 026 to a relative 3.1e-16 and `probe_all` to
  1.1e-16. With this machine's default OpenBLAS kernels (SkylakeX), `probe_all` differed by 9.7e-7, above
  the 1e-8 tolerance. Haswell kernels (`OPENBLAS_CORETYPE=Haswell`) reproduce it exactly, so the runner
  now requires them and records them in the receipt. The tolerance was not changed.
- For all 7,178 extracted ECGs present in the xECG cache, the mean of the 40 tokens equals the cached feature
  to 3.1e-5 (tolerance 1e-4).
- Section order: adding 1 mV to the first 25 input samples changed token 0 the most, and to the last 25
  samples token 39 the most, in all eight check ECGs.
- The mean of the `G` sections equals the cached-feature logit to 5.0e-6 (tolerance 1e-3).
- All row counts matched the protocol: fit 5,872, pool 15,359, binary evaluation 1,306 (843 positive),
  full development 1,604, premature-beat set 84 (81 patients), benign-variant set 52.

## Part 1: the unsupervised map finds premature beats, but its worst section is a weak screen

The beat rule found a premature beat in 73 of the 84 PVC ECGs; 11 had none, and none had too few R peaks.

| Map | Top section on a premature beat | Chance | Hit - chance [95% interval] |
| --- | ---: | ---: | --- |
| `U` (primary) | 0.822 | 0.153 | +0.669 [0.582, 0.758] |
| `U_kmeans` | 0.822 | 0.153 | +0.669 [0.581, 0.757] |
| `U_uncentred` | 0.795 | 0.153 | +0.641 [0.550, 0.737] |
| `G` | 0.767 | 0.153 | +0.614 [0.513, 0.713] |
| `G_gated` | 0.822 | 0.153 | +0.669 [0.582, 0.758] |

Binary detection (1,306 development ECGs, ECG score = worst section):

| Score | AUROC | AP | Minus `U` AUROC [95% interval] |
| --- | ---: | ---: | --- |
| Whole-ECG xECG Mahalanobis (Experiment 026) | 0.923 | 0.962 | +0.147 [0.122, 0.172] |
| Whole-ECG readout, the mean of the `G` sections (Experiment 026 `probe_all`) | 0.962 | | |
| `U` | 0.777 | 0.849 | |
| `U_kmeans` | 0.832 | 0.890 | +0.055 [0.041, 0.069] |
| `U_uncentred` | 0.713 | 0.812 | -0.064 [-0.081, -0.046] |
| `G` | 0.926 | 0.960 | +0.149 [0.123, 0.175] |
| `G_gated` | 0.848 | 0.899 | +0.072 [0.056, 0.088] |

At the 5% budget, a red section appeared in 23.0% [20.2, 25.8] of positive development ECGs with `U`,
against 76.7% [73.5, 79.6] with `G`. All 84 premature-beat ECGs had a red section with every map except
`U_uncentred` (98.8%).

Position centring helped: without it, AUROC fell by 0.064 and the mean normal score was 100.7 in the first
section and 77.8 in the last, against 62.2 in the middle. With centring, the first and last sections still
score slightly higher than the middle (79.5 and 61.1 against 64.1).

## Part 2: the label-guided map detects better but marks broadly, and benign variants equally

| Map | Mean red sections: normal | Positive | Premature beat | Benign variant | Benign with any red |
| --- | ---: | ---: | ---: | ---: | ---: |
| `U` | 0.08 | 0.86 | 6.55 | 0.46 | 0.269 |
| `U_kmeans` | 0.08 | 2.05 | 8.18 | 0.81 | 0.346 |
| `U_uncentred` | 0.07 | 0.78 | 6.33 | 0.52 | 0.365 |
| `G` | 0.08 | 15.35 | 17.86 | 0.44 | 0.269 |
| `G_gated` | 0.07 | 1.92 | 8.00 | 0.83 | 0.327 |

By construction, 5.2% of the 463 NORM-only ECGs have a red section with every map. The benign-variant set
(sinus bradycardia or arrhythmia with a normal diagnosis) had a red section 27% of the time with both `U` and
`G`, about five times the normal rate; the difference was 0.000 [-0.115, +0.135].

## Example figures

The examples are the lowest ECG ID in each group, chosen by rule.

| Example | `U` red sections | `G` red sections |
| --- | --- | --- |
| [NORM 47](figures/experiment-041/NORM_47.png) | none | none |
| [PVC 219](figures/experiment-041/PVC_219.png) | 3, at 8.25-9.0 s, on the one premature beat | 2, at 8.5-9.0 s, on the same beat |
| [IMI 8](figures/experiment-041/IMI_8.png) | none | none |
| [AMI 184](figures/experiment-041/AMI_184.png) | none | 30 of 40 |
| [CLBBB 287](figures/experiment-041/CLBBB_287.png) | 3, at 6.0-6.75 s, on the one beat whose shape differs in every lead | 26 of 40 |
| [LVH 30](figures/experiment-041/LVH_30.png) | none | none |
| [Benign 69](figures/experiment-041/benign_69.png) | 2, at 0.75 s and 2.0 s | none |

## Interpretation

- Per-section distance from normal finds a focal, intermittent abnormality well. On premature beats, its top
  section was on the beat 82% of the time against 15% by chance, without any label.
- Taking the worst section loses the evidence that whole-ECG averaging accumulates. Findings present in every
  beat, such as infarction, hypertrophy and bundle branch block, raise many sections a little and none a lot.
  The `U` worst-section AUROC (0.777) is 0.147 below the whole-ECG distance (0.923). The two scores answer
  different questions: whether the ECG is abnormal, and where it is most unusual.
- The label-guided contributions detect well even as a worst section (0.926), but less well than their own
  mean, which is the readout (0.962, from Experiment 026's saved `xecg_probe_all` scores). They still point at
  premature beats (77%). For diffuse findings they mark most of
  the record (15.4 red sections on average in positives), which is honest for a finding in every beat but is
  not a location.
- The clustering variant (the tutor's suggestion, `U_kmeans`) detected better than the single Gaussian
  (+0.055) and localized premature beats equally.
- Neither map separated benign rhythm variants from findings. The labels used here never contrast benign
  variants with disease: the fit set and the readout's negatives exclude them.

## Deviations and notes

- `CUDA_HOME` was set to the venv's `nvidia/cuda_runtime` folder because the vendored xLSTM computes a CUDA
  include path at import, and this machine has no CUDA toolkit. The vanilla backend compiles nothing.
- Before the full run, the smoke test (training ECGs only) showed that the NORM example group included
  benign-variant ECGs, which also have a standard label of 0. The example now comes from the 463 NORM-only
  binary-evaluation ECGs, as the protocol defines NORM-only. This changed no metric.
- The probe is fitted with one BLAS thread, as in Experiment 026. Thread count did not cause the 9.7e-7
  difference, but it is kept for consistency.

## Caveats

- The location check covers premature beats only, found by an automatic rule that also catches premature
  atrial beats. 11 of 84 PVC ECGs had no detected premature beat.
- A red band covers all 12 leads; it cannot say which lead is abnormal.
- Every PVC ECG in the binary evaluation also has another abnormal superclass, so a hit may partly reflect a
  co-occurring finding.
- The benign-variant set is small (52 ECGs; one ECG moves its share by 1.9 points), defined from PTB-XL
  statements, and outside the readout's training distribution.
- Development data were read by earlier experiments; the results are exploratory.

## Follow-ups (added to the backlog)

- `jepa_patch_localization041b`: the triggered fallback, ECG-JEPA patch tokens with per-lead maps.
- `section_map_with_ecg_score`: report the whole-ECG score for the screening decision and the section map
  only for explanation, and test that combination on the premature-beat and benign sets.
- `benign_variant_contrast`: a readout or reference that sees benign variants as negatives (needs the
  cardiologist, since it changes what counts as normal).
- `beat_annotated_localization`: a 12-lead database with expert beat labels (INCART) for a stronger location
  check (needs a download).
