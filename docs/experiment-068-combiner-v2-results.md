# Experiment 068 results: learned unit combiner v2

Run on 13 October 2026 (UTC) under the [protocol](experiment-068-combiner-v2.md), committed as 5e22f3c before
the cross-fit networks were trained and before any 068 map was scored. Networks, features, maps and scores
come from code commit 87bee2b. Every number below comes from `outputs/experiment068_combiner_v2/result.json`
(SHA-256 `14dd3def…2548`), which embeds `crossfit_summary.json`, `fit_receipt.json` and the LocBench
summaries. Intervals are LocBench's 95% whole-patient bootstraps: 2,000 draws, seed 61061, no skipped draws.
A2 uses the 92 marked ECGAD development records of 90 patients. No ECGAD holdout record was mapped, and no
ECGAD or `evaluation` record entered any fitted quantity.

## Decision

The primary map `units_v2` does not beat the bar. Its A2 is 0.7671 [0.7365, 0.7952], against 0.7419 for
`qrs_prior`. The paired difference is +0.0252 [-0.0130, +0.0647], so under the frozen rule `units_v2`
**matches** the bar.

`units_v2` is the highest development A2 so far. It beats 066 `units_learned` by +0.0068 [+0.0012, +0.0125]
and 066 `textbook` by +0.0178 [+0.0016, +0.0354]. Both intervals are above 0, but the gains are small.

Its development A2 (0.7671) exceeds `units_learned`'s 0.7604, so by the declared rule Experiment 067 adds
`units_v2` to its candidates.

## Main readings

| Map | A2 | A2 minus `qrs_prior` | A3 | A4 | A2_qrs | A2_nonqrs | C1_std | D1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `units_v2` (primary) | 0.7671 [0.7365, 0.7952] | +0.0252 [-0.0130, +0.0647] | 0.655 | 0.785 | 0.638 | 0.733 | 0.540 | 0.617 |
| `units_v2_nolead` (ablation) | 0.7633 [0.7331, 0.7909] | +0.0213 [-0.0168, +0.0611] | 0.662 | 0.780 | 0.631 | 0.727 | 0.512 | 0.628 |
| `units_v2_hgb` (nonlinear) | 0.7295 [0.6917, 0.7650] | -0.0124 [-0.0656, +0.0382] | 0.626 | 0.782 | 0.616 | 0.705 | 0.511 | 0.699 |
| `units_v1_data` (diagnostic) | 0.7618 [0.7318, 0.7906] | +0.0199 [-0.0179, +0.0600] | 0.653 | 0.779 | 0.639 | 0.724 | 0.541 | 0.611 |
| 066 `units_learned` | 0.7604 | +0.0184 [-0.0195, +0.0582] | 0.654 | 0.780 | 0.641 | 0.710 | 0.537 | 0.633 |
| 066 `textbook` | 0.7494 | +0.0074 [-0.0346, +0.0511] | 0.599 | 0.798 | 0.571 | 0.726 | 0.486 | 0.657 |
| 061 `qrs_prior` (bar) | 0.7419 [0.7082, 0.7743] | | 0.457 | 0.851 | 0.539 | 0.661 | 0.560 | 0.632 |
| 061 `residual056` | 0.6796 | | 0.591 | 0.715 | 0.607 | 0.671 | 0.707 | 0.593 |

The reference rows come from the copied tables in `ref/`: the strata and C1_std were recomputed in this run,
and the other values are the reference side of the paired comparisons. The `units_learned` minus `qrs_prior`
interval is quoted from the 066 report.

Other ECGAD readings for `units_v2`: A1 0.744, A5 0.622 and A6 0.393 (chance 0.186), against A1 0.695, A5
0.713 and A6 0.357 for `qrs_prior`.

## Paired comparisons (A2, same draws)

| Comparison | Difference |
| --- | ---: |
| `units_v2` minus `qrs_prior` (primary) | +0.0252 [-0.0130, +0.0647] |
| `units_v2` minus `units_learned` | +0.0068 [+0.0012, +0.0125] |
| `units_v2` minus `textbook` | +0.0178 [+0.0016, +0.0354] |
| `units_v2` minus `units_v2_nolead` | +0.0038 [+0.0021, +0.0055] |
| `units_v2` minus `units_v1_data` | +0.0053 [+0.0021, +0.0085] |
| `units_v2` minus `units_v2_hgb` | +0.0376 [+0.0207, +0.0562] |
| `units_v2` minus `residual056` | +0.0875 [+0.0513, +0.1229] |
| `units_v1_data` minus `units_learned` | +0.0015 [-0.0025, +0.0054] |
| `units_v2_nolead` minus `units_learned` | +0.0029 [-0.0025, +0.0086] |
| `units_v2_hgb` minus `units_learned` | -0.0309 [-0.0494, -0.0136] |

`units_v2` against `qrs_prior` on the other axes:

- A3 (lead): +0.199 [+0.123, +0.273].
- A4 (time): -0.066 [-0.102, -0.027].
- A2_qrs: +0.100 [+0.047, +0.153].
- A2_nonqrs: +0.071 [+0.020, +0.123].

`units_v2` against `units_learned`: A3 +0.001 [-0.012, +0.014], A4 +0.005 [-0.006, +0.015].

## What changed the A2, step by step

The diagnostic map `units_v1_data` has 066's 24 features trained on this experiment's training set. It
separates the two changes:

| Step | Map | A2 | Paired step |
| --- | --- | ---: | ---: |
| 066 combiner, 289 records | `units_learned` | 0.7604 | |
| Same features, 3,517 records, cross-fit MIL | `units_v1_data` | 0.7618 | +0.0015 [-0.0025, +0.0054] |
| Add softened lead factor and interactions | `units_v2` | 0.7671 | +0.0053 [+0.0021, +0.0085] |

More training data changed nothing measurable. The new features gave the small gain. It sits outside QRS:
A2_nonqrs is 0.733 for `units_v2` against 0.724 for `units_v1_data` and 0.710 for `units_learned`, while
A2_qrs is 0.638, 0.639 and 0.641.

## Cross-fit (fit partition)

- Five patient folds of the 11,617 `fit` records (seed 66066). Each fold network trained on 7,869-7,905
  records, stopped early on 1,387-1,436 records and predicted 2,312-2,335 out-of-fold records. Best epochs
  were 13-16, and early-stopping macro AUROCs 0.933-0.942.
- Out-of-fold macro AUROC (heads with at least 10 positives and 10 negatives): 0.9338 over all 11,617
  records, 0.9354 on the 1,766 rows that were 065's held-out rows. 065's per-seed held-out values were
  0.9440, 0.9436 and 0.9439, and its three-seed ensemble reached 0.9497 on those rows. The cross-fit is about
  0.01 below one 065 network, as expected from training on 68% of `fit` instead of 85%.
- The weakest out-of-fold heads were IVCD (0.810), EL (0.832), BIGU (0.844) and LNGQT (0.845).

### Calibration gap

Train-time MIL features come from the single-seed cross-fit, test-time features from the 065 ensemble.
Non-NORM bag probabilities:

| Records | Source | Mean p | Median | 90th pct | 99th pct | Mean per-record sum | Mean per-record max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 065 held-out rows (1,766) | cross-fit, out of fold | 0.0283 | 0.0012 | 0.040 | 0.707 | 1.358 | 0.452 |
| 065 held-out rows (1,766) | 065 ensemble | 0.0266 | 0.0010 | 0.036 | 0.699 | 1.279 | 0.440 |
| Training records (3,517) | cross-fit, out of fold | 0.0519 | 0.0030 | 0.113 | 0.883 | 2.490 | 0.748 |
| Scored records (4,871) | 065 ensemble | 0.0288 | 0.0011 | 0.040 | 0.739 | 1.382 | 0.457 |

On the same records the cross-fit logits are on average 0.16 higher than the ensemble's (mean absolute
difference 0.70), and the two correlate at 0.926. The gap between the training and scored rows is mostly
population: every training record carries a region statement.

## Training set and combiners

- 3,517 training records of 2,983 patients, none dropped; 2,636,400 units, 452,045 weakly positive. 1,229
  records carry two or more region statements, and 467 fail the region-set rhythm rule (both included, as
  declared).
- Records per family: anterior MI 1,185, IMI 892, LAFB 842, ischemia 572, 1AVB 417, CRBBB 297, CLBBB 294,
  lateral MI 188, ST injury 147, long QT 59, WPW 45.

Cross-validation on the training records (five patient folds; held-out listwise loss, lower is better;
held-out within-record weak-label AUROC):

| Model | Chosen setting | CV loss | CV weak-label AUROC |
| --- | --- | ---: | ---: |
| `units_v2` | a = 0.25, L2 = 1e-3 | 5.4011 | 0.9224 |
| `units_v2_nolead` | L2 = 1e-3 | 5.4048 | 0.9220 |
| `units_v1_data` | L2 = 1e-3 | 5.4037 | 0.9218 |
| `units_v2_hgb` | fixed | | 0.9233 |
| 066 weights, 3,228 training records outside 066's 289 | | | 0.9218 |

Across the twelve `units_v2` settings the CV loss ranged from 5.4011 to 5.4328, and at a fixed L2 the three
values of a differed by less than 0.001. The weak labels did not pin down how soft the lead factor should be.

The largest `units_v2` weights:

- `textbook` +1.01;
- components QRS +0.64, T -0.54, ST -0.16, PQ +0.17, P -0.11;
- interactions `textbook` x PQ +0.54, x P +0.27, x ST +0.16, x QRS +0.06, x T -0.02;
- leads V3 +0.39, III +0.23, I -0.38, V5 -0.33, V6 -0.24.

The MIL lead features carry small weights: `factor` +0.08, `factor_soft25` -0.11, `mil` -0.02. Other evidence:
q +0.09, amplitude -0.08, normative +0.06, residual +0.001. `units_v1_data`'s weights are close to 066's
(`textbook` +1.06 against +1.25; QRS +0.58 against +0.66; T -0.61 against -0.72).

## Noise

| Map | C1 | C1_std |
| --- | ---: | ---: |
| `units_v2` | 0.496 [0.471, 0.520] | 0.540 [0.511, 0.569] |
| `units_v2_nolead` | 0.470 [0.448, 0.493] | 0.512 [0.484, 0.540] |
| `units_v2_hgb` | 0.453 [0.426, 0.479] | 0.511 [0.484, 0.538] |
| `units_v1_data` | 0.498 [0.473, 0.522] | 0.541 [0.511, 0.569] |
| `units_learned` | 0.518 | 0.537 [0.508, 0.565] |
| `qrs_prior` | 0.353 | 0.560 [0.532, 0.589] |

The MIL lead features pull slightly toward noisy leads: raw C1 of `units_v2` minus `units_v2_nolead` is
+0.026 [+0.016, +0.036]. Every 068 map has C1_std between 0.51 and 0.55.

## Region set (not independent)

All four maps are trained on `REGION_TABLE` weak labels, so their B readings are not independent tests. For
completeness, B1 / B2 / B3 macro: `units_v2` 0.882 / 0.871 / 0.875; `units_v2_nolead` 0.882 / 0.871 /
0.877; `units_v2_hgb` 0.885 / 0.875 / 0.878; `units_v1_data` 0.882 / 0.873 / 0.873; `units_learned` 0.879 /
0.872 / 0.873; `qrs_prior` 0.728 / 0.576 / 0.774.

## Integrity

| Check | Result |
| --- | --- |
| Bar: 061 `qrs_prior` ECGAD maps rescored | per-record A2 within 2.2e-16; mean 0.7419383358406556 against 0.7419383358406554; passed |
| 066 `units_learned` saved tables rescored | per-record A2 within 1.1e-16; mean 0.7603613741425231 against 0.7603613741425232; passed |
| 066 `units_learned` rebuilt from this run's features with 066's weights | largest map difference 0.0; per-record A2 within 1.1e-16; passed |
| Test-time MIL (066's recomputed 065 bag logits) | identical to 065's saved bags on all 4,871 scored records; passed |
| Cross-fit | out-of-fold macro AUROC 0.934 (065 per seed 0.944) |
| Data separation | 11,617 cross-fit and 3,517 combiner records, none ECGAD or `evaluation`, none outside `fit`; no combiner record is a `fit` clean normal; no ECGAD holdout record mapped; passed |
| Maps | all finite; every scored and training record had delineated beats |

## Deviations and disclosures

- None from the protocol. Two choices that differ from the brief were declared in the protocol before any
  score: `units_v2_hgb` uses a pointwise logistic loss, because no ranking-loss library is installed and the
  dependencies are pinned; and the diagnostic map `units_v1_data` was added to separate data from features.
  It is not a candidate.
- The strata and noise CLIs wrote `strata.*` and `noise_std.*` into the copied reference folders under
  `ref/`. No 061, 065 or 066 file was written.
- Smoke outputs are in `outputs/experiment068_combiner_smoke/` and are not results.

## Runtime

| Phase | Time |
| --- | ---: |
| GPU cross-fit, five folds (GPU0, shared lock) | 1,765 s |
| Unit features, 8,388 records (8 processes) | 237 s |
| CV selection, final fits, maps | 358 s |
| Integrity checks, four LocBench scorings, strata, noise | 535 s |

Profile before training: 0.115 s per step, 2.5 GB peak GPU memory, about 14 s per epoch (measured 15-16 s).
Intermediates (preprocessed signals, fold weights, out-of-fold evidence, unit features, maps) are in
`/tmp/ecg068-work/full` and are not kept in `outputs/`.

## Why it moved so little

1. The 066 combiner was not data-limited. Twelve times more training records, multi-statement records and
   cross-fit evidence moved A2 by +0.0015 [-0.0025, +0.0054], and the refitted weights are close to 066's.
   On the fit partition, 066's weights already reach a weak-label AUROC of 0.9218 on 3,228 records they never
   saw, the same as the refitted model's CV value (0.9218).
2. The weak labels reward one thing: paint the diagnosed region, with a QRS-over-T preference and per-lead
   offsets. `textbook` keeps the largest weight in every linear model, and at their chosen settings the
   linear variants differ by at most 0.0006 in CV weak-label AUROC. The extra features can only make small
   adjustments.
3. The useful adjustment was a component-specific diagnosis weight. The interactions give the diagnosed
   region more weight when it is P or PQ (+0.27 and +0.54). The gain over `units_v1_data` appears outside
   QRS (A2_nonqrs 0.733 against 0.724), not inside it.
4. The MIL lead evidence earns a little: +0.0038 [+0.0021, +0.0055] A2 over `units_v2_nolead`, at the cost of
   +0.026 raw C1. How soft the lead factor should be was not identified on the fit partition.
5. Weak-label fit and ECGAD transfer diverge. The boosted trees fit the weak labels best (CV 0.9233, highest B
   macro) and transfer worst (A2 0.7295, -0.0309 [-0.0494, -0.0136] against `units_learned`). More capacity
   learns the table, not the cardiologists.
6. The bar gap is about time. `units_v2` is far ahead of `qrs_prior` on leads (A3 +0.199) and within QRS
   (A2_qrs +0.100), but behind on time (A4 -0.066 [-0.102, -0.027]). 066 found that the ECGAD marks favour
   QRS more than the statements predict. Region-statement weak labels cannot teach that preference.

## What it means

- `units_v2` is the best development map so far (A2 0.7671). It matches, but does not beat, the
  disease-blind bar, and its edge over `units_learned` (+0.007) is small enough that the holdout half in 067
  may not separate them.
- Learned combiners trained on `REGION_TABLE` weak labels have reached their ceiling on this benchmark. More
  data, more features or more capacity do not move ECGAD A2 by more than about 0.007.
- `units_v2` joins the 067 candidate list under the declared rule.

## Follow-up ideas (for the backlog)

1. A different training signal for the combiner, not more of the same weak labels. Options: beat-level labels
   from the 065 rhythm heads (PVC, PAC) for focal beats, or a QRS-time preference learned from `fit` records
   with PTB-XL+ ECGdeli component boundaries (B readings are already not independent for these maps).
2. Record-level scaling: multiply the min-max unit map by the record's 065 abnormal probability. A2 is
   unchanged, but A1, A5 and D1 should recover (D1 is 0.617 for `units_v2` against 0.632 for the bar).
3. Apply the lead-quality weight only to the MIL lead features, which carry the noisy-lead pull (+0.026 raw
   C1), instead of the whole map.
4. Report a statement-stratified A2 in 067 (066 follow-up 4) for `units_v2`, `units_learned` and the bar.
5. Fix a in the factor (0.25) instead of searching it in later work: the CV could not distinguish 0.25,
   0.5 and 0.75.

## Reproduce

```bash
cd /tmp/ecg-wt-068
export PYTHONPATH=/tmp/ecg-wt-068 OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m scripts.experiments.run_combiner068 profile
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m scripts.experiments.run_combiner068 crossfit
CUDA_VISIBLE_DEVICES= .venv/bin/python -u -m scripts.experiments.run_combiner068 features
CUDA_VISIBLE_DEVICES= .venv/bin/python -u -m scripts.experiments.run_combiner068 fit
CUDA_VISIBLE_DEVICES= .venv/bin/python -u -m scripts.experiments.run_combiner068 score
```

The `features` and `score` phases read 066's work folder `/tmp/ecg066-work/full` (065 ensemble evidence and
the `units_learned` maps for the rebuild check).
