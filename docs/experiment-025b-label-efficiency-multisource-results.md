# Experiment 025b results: label efficiency with pooled PTB-XL and Challenge labels

Completed 29 September 2026 under the [frozen protocol](experiment-025b-label-efficiency-multisource.md)
(frozen at commit `16c32bd`; the run recorded the same file hash, `9a54ac6b…477d`). Backlog item
`rerun_025_ningbo`. Run once at commit `c571c88` on CPU in 2,068 s, three encoder processes with one BLAS
thread each. Local outputs are in `outputs/experiment025b_label_efficiency_multisource_v1/` (`result.json`
SHA-256 `8c28968d…4da6`, `draws.csv`, `all_budget_predictions.npz`, `run.log`). No PTB-XL test or calibration
ECG and no Challenge test-group ECG was read. No age or other subgroup analysis was done.

After the protocol commit and before the runner was written, `origin/main` (PRs #24-#28) and
`origin/fix/manifold-window-call` (PR #29, the two-line `canonical_window` call fix in the 026b quality helper)
were merged into the branch (`d4f3978`, `96bdd76`). Neither merge touched the protocol file, which is
byte-identical to `16c32bd`, and neither changed an input: the Challenge feature files have the same hashes
as in 022b, and the runner checks this.

Before the full run, a scratch pass of the runner's preparation stage and a reduced fit (a few CPC draws,
20 bootstrap draws) checked that the code ran end to end. It printed no score and nothing was changed after it.

## Integrity

- The `ptbxl` arm reproduced 025 for CPC, ECG-JEPA and xECG: every draw's subset hash, record and positive
  counts matched, the largest AUROC and AP difference was 1.1e-16, and the N = all development
  probabilities differed by 0.0.
- Every input matched its receipt: the 020, 022, 025 and 022b artifacts, the JEPA and xECG caches, the
  Challenge features (also equal to 022b's), the Challenge split and the Ningbo manifest.
- The counts matched the protocol: 15,359 PTB-XL pool ECGs (9,487 positive), 22,494 Challenge training
  ECGs (17,520 positive), a pooled pool of 37,853 ECGs from 35,845 patient units (27,007 positive), and every
  evaluation set.
- Every fit converged (28-359 iterations). No bootstrap draw was skipped.
- The pooled draws spread across sources as pre-registered: on average 40-41% PTB-XL at every budget,
  33-34% Chapman/Ningbo, 14% Georgia and 11-12% CPSC.

Intervals are 95% paired bootstrap intervals (2,000 draws, seed 32032) of the draw-mean AUROC: by patient
for SPH and PTB-XL development, by record for the Challenge families. "k/20" is the number of the 20 label
draws in which the first readout beats the second.

## 1. SPH (primary set: 21,008 ECGs, 7,190 positive)

Mean AUROC over 20 draws; N = all is one fit. `+ Challenge` is N PTB-XL labels plus all 22,494 Challenge
labels; at N = all it would be the same fit as `pooled`.

| N labels | xECG `ptbxl` | xECG `pooled` | xECG `+ Challenge` | JEPA `ptbxl` | JEPA `pooled` | CPC `ptbxl` | CPC `pooled` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.864 (0.021) | 0.890 (0.014) | 0.938 | 0.861 | 0.886 | 0.809 | 0.833 |
| 250 | 0.880 (0.014) | 0.902 (0.014) | 0.938 | 0.883 | 0.902 | 0.836 | 0.856 |
| 500 | 0.895 (0.011) | 0.911 (0.008) | 0.938 | 0.894 | 0.910 | 0.853 | 0.868 |
| 1,000 | 0.902 (0.009) | 0.919 (0.006) | 0.938 | 0.902 | 0.917 | 0.864 | 0.876 |
| 2,000 | 0.909 (0.005) | 0.926 (0.004) | 0.939 | 0.909 | 0.923 | 0.872 | 0.885 |
| 4,000 | 0.912 (0.004) | 0.930 (0.002) | 0.939 | 0.912 | 0.927 | 0.876 | 0.890 |
| all | 0.915 | 0.937 | - | 0.914 | 0.935 | 0.882 | 0.900 |

SD over draws in parentheses for xECG; the `+ Challenge` SD is 0.000-0.001 at every budget (JEPA 0.935-0.936,
CPC 0.896-0.900). SPH average precision for xECG at N = 250: 0.835 (`ptbxl`), 0.868 (`pooled`), 0.919
(`+ Challenge`); at N = all 0.887 and 0.917.

`pooled` − `ptbxl` on SPH:

| N | xECG | JEPA | CPC |
| ---: | --- | --- | --- |
| 100 | +0.026 [+0.024, +0.028], 17/20 | +0.025 [+0.023, +0.027], 15/20 | +0.024 [+0.021, +0.026], 15/20 |
| **250** | **+0.022 [+0.020, +0.024], 16/20** | +0.019 [+0.017, +0.021], 18/20 | +0.020 [+0.017, +0.023], 18/20 |
| 500 | +0.017 [+0.015, +0.018], 18/20 | +0.016 [+0.015, +0.018], 19/20 | +0.016 [+0.014, +0.018], 19/20 |
| **1,000** | **+0.016 [+0.015, +0.018], 17/20** | +0.015 [+0.013, +0.016], 19/20 | +0.012 [+0.010, +0.014], 18/20 |
| 2,000 | +0.018 [+0.016, +0.019], 20/20 | +0.014 [+0.012, +0.016], 19/20 | +0.013 [+0.011, +0.015], 19/20 |
| 4,000 | +0.018 [+0.017, +0.020], 20/20 | +0.015 [+0.013, +0.017], 20/20 | +0.013 [+0.011, +0.015], 20/20 |
| all | +0.022 [+0.020, +0.025] | +0.020 [+0.018, +0.023] | +0.018 [+0.016, +0.020] |

Every interval lies above 0. `+ Challenge` − `ptbxl` is larger still: xECG +0.073 at N = 100, +0.058 at
250, +0.036 at 1,000 and +0.027 at 4,000, 20/20 draws and intervals above 0 at every budget and for every
encoder.

**Label efficiency at SPH.** The `ptbxl` arm's SPH AUROC with all 15,359 PTB-XL labels (xECG 0.915, JEPA
0.914, CPC 0.882) is not reached by the `ptbxl` arm at any smaller budget. `pooled` reaches it with 1,000
labels for xECG and JEPA and 2,000 for CPC; `+ Challenge` reaches it with 100 PTB-XL labels for every
encoder.

## 2. PTB-XL development (secondary)

Ordinary development ECGs (1,306, 843 positive), mean AUROC:

| N | xECG `ptbxl` | xECG `pooled` | xECG `+ Challenge` | JEPA `ptbxl` | JEPA `pooled` | CPC `ptbxl` | CPC `pooled` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 0.921 | 0.913 | 0.907 | 0.922 | 0.906 | 0.883 | 0.867 |
| 250 | 0.931 | 0.923 | 0.911 | 0.934 | 0.918 | 0.897 | 0.876 |
| 1,000 | 0.947 | 0.938 | 0.922 | 0.947 | 0.937 | 0.910 | 0.897 |
| 4,000 | 0.957 | 0.946 | 0.940 | 0.955 | 0.946 | 0.916 | 0.906 |
| all | 0.962 | 0.954 | - | 0.959 | 0.954 | 0.921 | 0.913 |

`pooled` − `ptbxl` for xECG is −0.008 to −0.011 at every budget (intervals below 0; `ptbxl better` by the
draw rule from N = 1,000 up), and −0.008 [−0.012, −0.004] at N = all. JEPA loses 0.009-0.016 and CPC
0.010-0.021, both below 0 at every budget. `+ Challenge` costs more: xECG −0.014 to −0.025.

Hard added subset (266 ECGs, 41 positive), mean AUROC:

| N | xECG `ptbxl` | xECG `pooled` | xECG `+ Challenge` | JEPA `ptbxl` | JEPA `pooled` | CPC `ptbxl` | CPC `pooled` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 250 | 0.532 | 0.509 | 0.481 | 0.532 | 0.484 | 0.521 | 0.442 |
| 1,000 | 0.563 | 0.515 | 0.495 | 0.570 | 0.512 | 0.511 | 0.448 |
| 4,000 | 0.597 | 0.521 | 0.522 | 0.609 | 0.538 | 0.505 | 0.453 |
| all | 0.615 | 0.562 | - | 0.629 | 0.580 | 0.506 | 0.465 |

`pooled` − `ptbxl` at N = all: xECG −0.053 [−0.087, −0.017], JEPA −0.049 [−0.088, −0.013], CPC −0.042
[−0.075, −0.009]. At N = 1,000 and above the xECG and JEPA intervals lie below 0; at N = 250 the xECG
interval includes 0 (−0.024 [−0.060, +0.011]). With 41 positives these values are imprecise.

## 3. Challenge calibration families (secondary; in distribution for `pooled` and `+ Challenge`)

xECG AUROC, `ptbxl` → `pooled` (mean over draws):

| Family (ECGs, positive) | N = 250 | N = 1,000 | N = all |
| --- | --- | --- | --- |
| Chapman/Ningbo (4,432, 3,254) | 0.942 → 0.972, +0.030 [+0.027, +0.033] | 0.958 → 0.980, +0.023 | 0.965 → 0.992, +0.027 |
| Georgia (1,718, 1,372) | 0.889 → 0.912, +0.023 [+0.018, +0.029] | 0.906 → 0.927, +0.022 | 0.917 → 0.945, +0.028 |
| CPSC (1,462, 1,279) | 0.898 → 0.908, +0.011 [+0.003, +0.018] | 0.917 → 0.926, +0.009 | 0.932 → 0.951, +0.019 |

All xECG intervals lie above 0. At CPSC the JEPA and CPC contrasts include 0 below N = all (JEPA +0.000 and
+0.001, CPC −0.006 at N = 250 and 1,000). `+ Challenge`, which saw every training ECG of these hospitals,
scores 0.993-0.994 on Chapman/Ningbo, 0.951-0.953 on Georgia and 0.943-0.949 on CPSC for xECG.

## 4. Encoder ranking

Within every arm, on SPH and ordinary development, at N = 250 and 1,000: ECG-JEPA and xECG are each better
than CPC (20/20 draws), and xECG versus JEPA shows no clear difference. The differences are the same size in
both arms (SPH `pooled`, N = 250: JEPA − CPC +0.046, xECG − CPC +0.046; `ptbxl`: +0.047 and +0.044). At
N = all on SPH, xECG − JEPA is +0.001 [−0.001, +0.003] for `ptbxl` and +0.003 [+0.001, +0.004] for
`pooled`.

## Prespecified reading

- **Primary verdicts (SPH, xECG, `pooled` − `ptbxl`):** N = 250 **not distinguished** (+0.022, interval
  above 0, but only 16 of 20 draws positive); N = 1,000 **not distinguished** (+0.016, interval above 0,
  17 of 20 draws).
- **Decision: `mixed`.** By the protocol, PTB-XL-only stays the default pool for a small labeled set. The
  pooled draws win at the budgets where the draw rule is met (xECG at 500, 2,000 and 4,000 with 18-20 of 20
  draws; JEPA and CPC from 250 up), and every SPH interval lies above 0; neither gain is negligible.
- **Encoder ranking: holds.** No JEPA − CPC, xECG − CPC or xECG − JEPA verdict differs between `ptbxl` and
  `pooled` on SPH or ordinary development at N = 250 or 1,000.
- **Secondary:** `+ Challenge` beats `ptbxl` at SPH and at every Challenge family at every budget (20/20
  draws; the CPC interval at CPSC includes 0 from N = 1,000 up). It loses on ordinary PTB-XL development at
  every budget (intervals below 0, except JEPA at N = 100, −0.009 [−0.022, +0.003]). Pooling costs 0.008-0.011 on ordinary development (xECG) and about
  0.05 on the hard added subset.

## What this means for the project

- The decision is `mixed` because the xECG draws are noisy, not because pooling fails at SPH. The average
  gain from pooling at a fixed label count is about +0.016 to +0.026 AUROC at every budget and for every
  encoder, with intervals above 0, but at N = 250 and 1,000 four and three of the 20 xECG draws still favour
  the PTB-XL-only pool. Per the protocol, a lab choosing a pool of 250 or 1,000 labels for PTB-XL-like
  deployment keeps PTB-XL alone; for a new hospital like SPH, pooling tends to help on average.
- The practical answer is the `+ Challenge` arm. Taking every public Challenge label and adding even
  100 home labels gives SPH 0.938 (xECG), above the 0.915 of all 15,359 PTB-XL labels. Once the 22,494
  public labels are in, more home labels do not move SPH (0.938 to 0.939) but they do recover home-site
  performance (ordinary development 0.907 at N = 100 to 0.940 at N = 4,000).
- Every pooled readout ranks the home site slightly worse. The ordinary cost is small (about 0.01); the
  cost on the hard subset is 0.05-0.08. That is the question for the new backlog item
  `label_harmonization_hard_subset`.
- Nothing here changes the encoder choice. JEPA and xECG stay about 0.04 AUROC above CPC whichever pool the
  labels come from.

## Surprises

- The interval and the draw rule disagree for the primary contrast. The `pooled` and `ptbxl` draws that
  share a seed are different ECGs from different pools, so the per-draw difference carries both draws'
  sampling noise (xECG `ptbxl` SD 0.014 at N = 250). The draw-mean interval holds the draws fixed and is
  narrow (±0.002). JEPA and CPC pass the draw rule at the same budgets where xECG does not.
- The hard added subset scores much lower here than in 022 and 022b: xECG `ptbxl` at N = all is 0.615
  against 0.726 in 022b. The 1,724 PTB-XL training ECGs outside 025's pool, which is limited to the released
  caches, are exactly the ECGs the project label dropped (all 1,724 have no project label; 353 are
  positive). They are the training analogues of the hard added subset. A pool without them ranks that subset
  close to chance (0.51-0.63), whatever the encoder. This was counted after the run, from the pool tables
  only.
- With all Challenge labels in, the PTB-XL labels barely change the SPH curve (SD across draws 0.001). The
  pooled-arm label-efficiency curve at SPH is the `+ Challenge` curve shifted down: 1,000 pooled labels
  match 15,359 PTB-XL labels.
- In `+ Challenge`, JEPA beats xECG on ordinary development at every budget (0/20 draws favour xECG, −0.002
  to −0.005), while xECG beats JEPA on SPH (20/20, +0.002 to +0.003). Both differences are small.

## Caveats

- ECG-JEPA and xECG report Chapman and Ningbo in their self-supervised pretraining data, so they have likely
  seen the Chapman/Ningbo waveforms without labels. CPC (Experiment 004, PTB-XL and MIMIC) has seen no
  Challenge record and shows the same pattern. No encoder saw a Challenge label; SPH is unseen by every
  encoder.
- The Challenge negative (sinus rhythm alone) is a stricter normal than PTB-XL's NORM and SPH's code 1, and
  the pooled draws are 71% positive against 62% for PTB-XL draws.
- The PTB-XL pool is 025's 15,359 ECGs, not 022b's 17,083, so `pooled` at N = all is not 022b's `pooled`
  fit (SPH 0.937 here against 0.939) and the hard subset is not comparable with 022b (see Surprises).
- Challenge records stand in for patients. A Challenge patient with two records can be drawn twice, and the
  family readouts are in distribution for the pooled arms.
- The hard added subset has 41 positives, so its values are imprecise.
- Draw variability reflects label sampling on fixed evaluation sets. The development patients were inspected
  by earlier experiments. One checkpoint per encoder, one run. The label is an ECG annotation proxy, not a
  clinical outcome or a referral decision.

## Reproduce

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 uv run --no-sync python -u -m scripts.experiments.run_label_efficiency025b
```

The runner refuses to overwrite an existing `result.json`. It needs the 020, 022, 025 and 022b outputs, the
JEPA and xECG caches, the Challenge split and `outputs/features_challenge_v1/`.
