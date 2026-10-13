# Experiment 065 results: finding-specific multiple-instance heads

Run on 12 October 2026 under the [protocol](experiment-065-finding-mil.md) (frozen as b5e64f8) and its
amendments 1 (8b0c928) and 2 (bfe1171), both committed before any LocBench score. Maps were written at commit
1533b11 and scored at e2aa2a8; the two commits in between changed only the bar reproduction check (see
Deviations). Every number below comes from `outputs/experiment065_mil_v1/result.json` (SHA-256
`ab963ed1…939b`) and the per-variant `locbench/<variant>/summary.json`. Intervals are 95% whole-patient
bootstraps, 2,000 draws, seed 61061 (LocBench's convention). Training used only `fit`-partition records. No
ECGAD record, `evaluation` record, fold 9 or 10, SPH, Challenge or EchoNext record was used for fitting or
selection, and no ECGAD holdout label was read.

## Decision

The primary map `mil_lse` is **below** the bar. Its A2 is 0.6624 [0.6272, 0.6953]. `qrs_prior`'s A2 is
0.7419, and the paired difference is −0.0796 [−0.1177, −0.0393]. So `mil_lse` is not a candidate for
Experiment 067 on the primary reading.

None of the declared variants reaches the bar on A2 either. The prior-plus-evidence variant `mil_lse_qrs`
comes closest at −0.0302 [−0.0534, −0.0051].

The maps do carry lead information that no 061 baseline has:

- B2 macro, the lead AUROC on the region set: 0.719, against 0.576 for `qrs_prior`. The paired difference is
  +0.143 [+0.120, +0.168].
- A3, the ECGAD lead AUROC: 0.644, against 0.457 for `qrs_prior` (paired +0.188 [+0.108, +0.268]) and 0.591
  for `residual056`.

What the maps cannot do is say where in time the abnormality is. A4, the time AUROC, is 0.694, against
0.851 for `qrs_prior` (paired −0.157 [−0.188, −0.122]).

## Integrity

- No training or held-out row is an ECGAD or `evaluation` record (asserted). Training used 9,851 `fit`
  records; early stopping and the sanity checks used 1,766 held-out `fit` records from 1,502 patients.
- Fresh networks loaded from the saved weights reproduced the held-out bag logits exactly. The largest
  difference was 0.0 for all five networks (tolerance 1e-4).
- Bar reproduction: 061's saved `qrs_prior` ECGAD maps were rescored with `locbench.ecgad_record_metrics`.
  The 92 per-record A2 values match 061's `ecgad.csv` to within 2.2e-16, which is one unit in the last place
  in 46 records. The mean is 0.7419383358406556, against 0.7419383358406554 saved.
- The 059 hybrid QRS mask that `mil_lse_qrs` uses equals 061's `qrs_prior >= 0.5` on all 198 ECGAD
  development records. QRS covers 16.6% of samples over the 4,871 scored records.

## Training

There are 49 statement heads, including NORM as the reference head. Each network trains 0.74 to 0.79 million
parameters at about 20 s per epoch on one RTX 3090.

| Network | Best epoch | Epochs run | GPU seconds (epochs) | Held-out macro AUROC (heads with ≥10 positives) |
| --- | ---: | ---: | ---: | ---: |
| lse_65065 | 15 | 22 | 438 | 0.9440 |
| lse_65066 | 13 | 20 | 402 | 0.9436 |
| lse_65067 | 15 | 22 | 451 | 0.9439 |
| attention_65065 | 20 | 27 | 559 | 0.9395 |
| crosslead_65065 | 11 | 18 | 377 | 0.9491 |
| `mil_lse` (mean of the three seeds) | | | | 0.9497 |

Held-out detection AUROC of `mil_lse` per head (heads with at least 5 held-out positives):

| Head | AUROC | Head | AUROC | Head | AUROC | Head | AUROC |
| --- | ---: | --- | ---: | --- | ---: | --- | ---: |
| 1AVB | 0.959 | ILBBB | 0.967 | LAFB | 0.994 | NT_ | 0.953 |
| ABQRS | 0.903 | ILMI | 0.973 | LAO/LAE | 0.891 | PAC | 0.866 |
| AFIB | 0.987 | IMI | 0.971 | LMI | 0.980 | PACE | 0.993 |
| ALMI | 0.984 | INJAL | 0.977 | LNGQT | 0.946 | PVC | 0.987 |
| AMI | 0.936 | INJAS | 0.986 | LOWT | 0.896 | QWAVE | 0.860 |
| ASMI | 0.980 | INVT | 0.951 | LPFB | 0.993 | RAO/RAE | 0.996 |
| CLBBB | 0.998 | IRBBB | 0.988 | LPR | 0.967 | RVH | 0.985 |
| CRBBB | 0.995 | ISCAL | 0.941 | LVH | 0.965 | STD_ | 0.917 |
| DIG | 0.946 | ISCAS | 0.989 | LVOLT | 0.894 | SVARR | 0.900 |
| EL | 0.922 | ISCIL | 0.939 | NDT | 0.908 | WPW | 0.953 |
| ISC_ | 0.971 | ISCIN | 0.971 | NORM | 0.930 | IVCD | 0.833 |
| ISCLA | 0.917 | NST_ | 0.904 | | | | |

AFLT, ANEUR and BIGU have fewer than 5 held-out positives and are not scored.

## Sanity checks (held-out `fit` records)

| Check | Reading | Rule | Result |
| --- | --- | --- | --- |
| Randomization, untrained networks (1,071 abnormal records) | median Spearman 0.005 (IQR −0.073 to 0.095) | < 0.5 | pass |
| Randomization, instance layer re-initialized | median −0.184 (IQR −0.299 to −0.061) | < 0.5 | pass |
| Amplitude correlation, trained | median 0.176 (IQR 0.080 to 0.279) | flag if > 0.5 | does not track amplitude |
| Amplitude correlation, untrained (reference) | median −0.000 | | |
| Expected leads (own evidence, lead AUROC) | 13 of 13 heads above 0.5 | ≥ 9 of 13 | pass |
| Noisy-lead AUROC of the combined map (315 records) | 0.492 | descriptive | not drawn to noisy leads |

Mean lead AUROC of each head's evidence against its `REGION_TABLE` leads, on that head's held-out positives:

| Head | n | Lead AUROC | Head | n | Lead AUROC |
| --- | ---: | ---: | --- | ---: | ---: |
| IMI | 121 | 0.968 | INJAS | 14 | 0.741 |
| ASMI | 174 | 0.910 | INJAL | 16 | 0.908 |
| AMI | 24 | 0.760 | ISCAS | 12 | 0.786 |
| ALMI | 30 | 0.911 | ISCAL | 67 | 0.890 |
| LMI | 10 | 0.838 | ISCIN | 14 | 0.947 |
| CRBBB | 50 | 0.862 | ISCLA | 15 | 0.843 |
| LAFB | 133 | 0.531 | | | |

Each head points to its clinically expected leads. The exception is LAFB, which is defined by the axis, a
relation between leads. These are development readings on `fit` data, not LocBench scores.

## LocBench v1

Paired differences against `qrs_prior` (variant minus `qrs_prior`):

| Variant | A2 (primary for `mil_lse`) | A2 − `qrs_prior` | A2 − `residual056` | A2 − `U_B` |
| --- | --- | --- | --- | --- |
| `mil_lse` | 0.6624 [0.6272, 0.6953] | −0.0796 [−0.1177, −0.0393] | −0.0173 [−0.0585, +0.0233] | +0.0308 [−0.0153, +0.0773] |
| `mil_lse_max` | 0.6578 [0.6220, 0.6914] | −0.0841 [−0.1234, −0.0426] | −0.0219 [−0.0641, +0.0186] | +0.0263 [−0.0203, +0.0737] |
| `mil_attention` | 0.6691 [0.6442, 0.6941] | −0.0728 [−0.1055, −0.0392] | −0.0105 [−0.0434, +0.0205] | +0.0376 [+0.0006, +0.0755] |
| `mil_crosslead` | 0.6785 [0.6469, 0.7116] | −0.0635 [−0.0912, −0.0334] | −0.0012 [−0.0406, +0.0361] | +0.0469 [+0.0060, +0.0902] |
| `mil_lse_qrs` | 0.7118 [0.6761, 0.7464] | −0.0302 [−0.0534, −0.0051] | +0.0321 [−0.0050, +0.0678] | +0.0802 [+0.0351, +0.1257] |

Every reading:

| Metric | `qrs_prior` (bar) | `mil_lse` | `mil_lse_max` | `mil_attention` | `mil_crosslead` | `mil_lse_qrs` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 pooled point AUROC | 0.695 | 0.807 | 0.791 | 0.762 | 0.788 | 0.631 |
| A2 within-record | 0.742 | 0.662 | 0.658 | 0.669 | 0.679 | 0.712 |
| A3 lead | 0.457 | 0.644 | 0.637 | 0.598 | 0.635 | 0.648 |
| A4 time | 0.851 | 0.694 | 0.694 | 0.687 | 0.699 | 0.737 |
| A5 record | 0.713 | 0.949 | 0.945 | 0.964 | 0.954 | 0.866 |
| A6 precision at top 5% (chance 0.186) | 0.357 | 0.367 | 0.378 | 0.329 | 0.372 | 0.376 |
| A2_qrs (061 amendment 2) | 0.539 | 0.569 | 0.572 | 0.533 | 0.577 | 0.569 |
| A2_nonqrs | 0.661 | 0.611 | 0.603 | 0.644 | 0.560 | 0.611 |
| B1 macro | 0.728 | 0.737 | 0.731 | 0.724 | 0.711 | 0.758 |
| B2 macro | 0.576 | 0.719 | 0.717 | 0.698 | 0.744 | 0.730 |
| B3 macro | 0.774 | 0.776 | 0.777 | 0.746 | 0.719 | 0.783 |
| B1 / B2 / B3 macro, strict | | 0.806 / 0.759 / 0.865 | 0.799 / 0.755 / 0.866 | 0.803 / 0.729 / 0.860 | 0.808 / 0.783 / 0.815 | 0.868 / 0.772 / 0.892 |
| C1 noisy lead | 0.353 | 0.528 | 0.521 | 0.577 | 0.550 | 0.522 |
| C1_std (061 amendment 3) | 0.560 | 0.583 | 0.585 | 0.551 | 0.597 | 0.553 |
| C2 noise record | 0.479 | 0.492 | 0.491 | 0.497 | 0.515 | 0.514 |
| D1 detection | 0.632 | 0.962 | 0.959 | 0.944 | 0.968 | 0.883 |

`mil_lse` paired against `qrs_prior` on the other bar metrics:

- B1 macro: +0.0087 [−0.0025, +0.0196].
- B2 macro: +0.1434 [+0.1203, +0.1681].
- B3 macro: +0.0025 [−0.0109, +0.0162].
- C1: +0.1755 [+0.1421, +0.2080].
- D1: +0.3303 [+0.3111, +0.3484].
- A1: +0.1114 [+0.0619, +0.1670].
- A3: +0.1876 [+0.1081, +0.2678].
- A4: −0.1566 [−0.1884, −0.1222].

C1_std of `mil_lse` is 0.583 [0.557, 0.608]. That is above 0.5, so the map is mildly drawn to noise beyond
its usual lead levels. It is less drawn than `U_B` (0.720) or `residual056` (0.707).

QRS strata, paired:

| Variant | A2_qrs − `qrs_prior` | A2_qrs − `residual056` | A2_nonqrs − `qrs_prior` | A2_nonqrs − `residual056` |
| --- | --- | --- | --- | --- |
| `mil_lse` | +0.030 [−0.022, +0.085] | −0.038 [−0.103, +0.027] | −0.050 [−0.093, −0.004] | −0.060 [−0.108, −0.012] |
| `mil_lse_max` | +0.033 [−0.019, +0.088] | −0.035 [−0.101, +0.030] | −0.058 [−0.101, −0.012] | −0.068 [−0.117, −0.019] |
| `mil_attention` | −0.006 [−0.056, +0.045] | −0.074 [−0.124, −0.022] | −0.017 [−0.052, +0.020] | −0.027 [−0.068, +0.015] |
| `mil_crosslead` | +0.038 [−0.002, +0.080] | −0.030 [−0.086, +0.027] | −0.101 [−0.152, −0.048] | −0.111 [−0.154, −0.064] |
| `mil_lse_qrs` | as `mil_lse` (same order within each stratum) | | | |

Region set per family (`mil_lse`; `qrs_prior` B2 in brackets where informative):

| Family | B1 | B2 | B3 |
| --- | ---: | ---: | ---: |
| 1AVB | 0.555 | | 0.546 |
| CLBBB | 0.881 | | 0.983 |
| CRBBB | 0.905 | 0.784 (0.728) | 0.933 |
| IMI | 0.884 | 0.884 (0.289; `residual056` 0.804) | 0.875 |
| LAFB | 0.777 | 0.535 (0.251) | 0.968 |
| LNGQT | 0.476 | | 0.529 |
| ST injury | 0.708 | 0.637 (0.820) | 0.697 |
| anterior MI | 0.856 | 0.773 (0.815) | 0.928 |
| ischemia | 0.591 | 0.702 (0.551) | 0.529 |

## What it means

- Supervision puts the evidence in the right leads. On held-out `fit` data the IMI head's evidence ranks
  II, III and aVF above the other leads (lead AUROC 0.97), and ASMI's ranks V1-V3 above the others (0.91).
  On the region set, the IMI lead AUROC is 0.88, the best of any LocBench map so far. `residual056`, the only 061
  map with lead information, reaches 0.80.
- The maps do not say when within the record. The instance logits come from a receptive field of about
  1.1 s, and log-mean-exp pooling with r = 1 rewards broad firing. A lead with a positive finding therefore
  lights up over most of its beats and segments. Inside QRS complexes the maps rank marked against unmarked
  samples slightly better than `qrs_prior` (A2_qrs +0.03, interval includes 0). Outside QRS they rank worse
  than `qrs_prior`'s 0.001 amplitude tiebreak (A2_nonqrs −0.05). The cardiologist marks favour QRS, so
  `qrs_prior` wins A2 through time alone.
- Adding the QRS indicator (`mil_lse_qrs`) recovers most of the gap. The remaining loss on A2 (−0.030) comes
  from the non-QRS stratum. B1 macro improves on the bar: 0.758 against 0.728, paired +0.029
  [+0.020, +0.039].
- Short-duration and interval findings stay weak: 1AVB (PR), LNGQT (T), ischemia (ST-T). The heads detect
  them well (held-out AUROCs 0.96, 0.95 and 0.92-0.99), but their evidence does not concentrate on the expected
  component.
- Cross-lead mixing raises detection and A2 slightly, but blurs components (B3 0.719) and the non-QRS
  stratum (A2_nonqrs 0.560). The lead-wise primary is the better localizer.
- The randomization and amplitude checks show that the maps depend on the trained weights and are not an
  amplitude edge detector.

## Deviations

- Bar reproduction: the protocol asked for an exact match of A2. 46 of 92 per-record values differ from
  061's by one unit in the last place (2.2e-16), and the mean differs by 2e-16, so the check uses a 1e-12
  tolerance. The first two scoring attempts stopped at this check, before any variant was scored.
- The `seconds` field of the history files includes time spent waiting for the GPU lock
  (lse_65066: 1,269 s against 402 s of epochs). The epoch seconds above are the GPU time.

## Runtime

- Training: 2,227 GPU seconds across the five networks (about 37 minutes).
- Maps phase: held-out predictions, sanity checks and maps for 198 ECGAD and 5,801 `evaluation` records.
  About 6,800 s wall time, most of it waiting for the shared GPU lock while Experiment 064 pretrained.
- Scoring: about 2 minutes per variant on CPU.
- Peak GPU memory: about 2.5 GB.

## Failure analysis

- The time localization is too coarse. Every instance sees about 1.1 s, and the pooling does not force
  sparse evidence, so the instance logits mark leads rather than waves.
- The supervision says which finding is present but not which wave. A head for an ST-T statement can rely
  on any part of the beat that correlates with the statement.
- The ECGAD marks concentrate on QRS (36% of marks on 16% of samples), which rewards a time-only prior. On
  the clinical region set, where the components come from ECGdeli, the maps match the bar on B1 and beat it
  on B2.

## Follow-up ideas (for the backlog)

1. Factorized map for Experiment 066: lead weights from the MIL heads, times a finding-specific component
   prior from the 059 delineation (QRS for MI and bundle branch blocks, PR for 1AVB, ST and T for
   ischemia, injury and LNGQT), weighted by p_k. This combines the MIL lead information with the time
   information that wins A2. It needs a protocol, and is scored on ECGAD development against `qrs_prior`.
2. Beat x lead x component instances, pooled from 059 delineation units, so that MIL evidence is forced onto
   one wave. Add top-k or larger-r pooling, or a sparsity penalty on positive bags.
3. Smaller receptive field (about 300 ms) with a separate rhythm branch for AFIB, PVC and PAC.
4. A lead-quality map learned from the noise annotations (066), because C1_std is 0.58.
5. The `mil_lse_qrs` B1 gain (+0.029 over the bar) and the B2 gain (+0.14) are worth confirming on the
   ECGAD holdout in Experiment 067 only if a factorized variant (idea 1) also reaches the A2 bar.

## Reproduce

```bash
cd /tmp/ecg-wt-065
export OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python -u -m scripts.experiments.run_mil065 train
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. .venv/bin/python -u -m scripts.experiments.run_mil065 maps
CUDA_VISIBLE_DEVICES= PYTHONPATH=. .venv/bin/python -u -m scripts.experiments.run_mil065 score
```

Outputs: `outputs/experiment065_mil_v1/`. The folder holds the weights, held-out bags, histories,
`maps_<variant>_{ecgad_dev,evaluation}.npy` (ECGAD maps in float32, `evaluation` maps in float16),
`finding_maps_*.npz`, `bags_*.npz`, and `locbench/<variant>/` with `summary.json`, `tables/`, `strata.csv`,
`noise_std.csv` and `readings065.json`.
