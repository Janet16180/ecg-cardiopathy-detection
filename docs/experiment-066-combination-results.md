# Experiment 066 results: factorized combination maps and lead-quality suppression

Run on 13 October 2026 (UTC) under the [protocol](experiment-066-combination.md), committed as 71b3440
before any map was built or scored. Maps and scores come from code commit d0ade37. Every number below comes
from `outputs/experiment066_combination_v1/result.json` (SHA-256 `e66fbc55…`, written by the `score` and
`separation` phases) or from `descriptives.json` (`e20dcea0…`, undeclared descriptive readings, marked as
such). Intervals are LocBench's 95% whole-patient bootstraps: 2,000 draws, seed 61061, no skipped draws. A2
uses the 92 marked ECGAD development records of 90 patients. No ECGAD holdout record was mapped, and no ECGAD
or `evaluation` record entered any fitted quantity.

## Decision

The primary map `factor` does not beat the bar. Its A2 is 0.7247 [0.6940, 0.7553]; `qrs_prior` has 0.7419.
The paired difference is -0.0173 [-0.0558, +0.0212], so under the frozen rule `factor` **matches** the bar.

`factor` localizes worse than the diagnose-then-paint baseline `textbook`: paired A2 -0.0247 [-0.0498,
-0.0016]. The MIL lead evidence therefore did not add localization beyond the diagnosis on ECGAD.

The learned combiner `units_learned` has the highest development A2 of any map so far, 0.7604 [0.7310,
0.7885]. Against `qrs_prior` its paired difference is +0.0184 [-0.0195, +0.0582], which is not significant.

The lead-quality suppression is not adopted. `factor_q` lowers C1_std by 0.016 against `factor`; the rule
needed 0.05.

067 candidates, as declared: `qrs_prior`, 062 `qrs_lexicographic`, `factor`, and `units_learned` (the 066
secondary with the higher development A2: 0.7604 against 0.7214 for `factor_q`).

## Main readings

| Map | A2 | A2 minus `qrs_prior` | A3 | A4 | A2_qrs | A2_nonqrs | C1_std | D1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `factor` (primary) | 0.7247 [0.6940, 0.7553] | -0.0173 [-0.0558, +0.0212] | 0.639 | 0.793 | 0.617 | 0.680 | 0.580 | 0.637 |
| `textbook` | 0.7494 [0.7169, 0.7815] | +0.0074 [-0.0346, +0.0511] | 0.599 | 0.798 | 0.571 | 0.726 | 0.486 | 0.657 |
| `textbook_true` (diagnostic) | 0.7582 [0.7259, 0.7898] | +0.0163 [-0.0294, +0.0593] | 0.583 | 0.826 | 0.572 | 0.716 | 0.521 | 0.999 |
| `factor_q` | 0.7214 [0.6905, 0.7527] | -0.0205 [-0.0602, +0.0187] | 0.638 | 0.798 | 0.612 | 0.677 | 0.563 | 0.639 |
| `units_learned` | 0.7604 [0.7310, 0.7885] | +0.0184 [-0.0195, +0.0582] | 0.654 | 0.780 | 0.641 | 0.710 | 0.537 | 0.633 |
| 061 `qrs_prior` (bar) | 0.7419 [0.7082, 0.7743] | | 0.457 | 0.851 | 0.539 | 0.661 | 0.560 | 0.632 |
| 062 `qrs_lexicographic` | 0.7530 | +0.0111 [-0.0084, +0.0312] | 0.575 | 0.801 | 0.591 | 0.651 | 0.544 | 0.817 |
| 061 `residual056` | 0.6796 | | 0.591 | 0.715 | 0.607 | 0.671 | 0.707 | 0.593 |

The `qrs_prior`, `residual056` and `qrs_lexicographic` strata and C1_std were recomputed in this run from copied
tables (`reference_readings` in `result.json`); their other values are quoted from the 061 and 062 reports.

Other ECGAD readings: A6 (precision at top 5%, chance 0.186) is 0.434 for `factor`, 0.383 for `textbook`, 0.403
for `units_learned` and 0.357 for `qrs_prior`. `factor` against `qrs_prior` on A6: +0.077 [+0.013, +0.142].
A5 (record AUROC) is 0.607 for `factor` and 0.624 for `units_learned`, against 0.713 for `qrs_prior`, because
every 066 map is scaled to a maximum of 1 per record.

## Paired comparisons (A2, same draws)

| Comparison | Difference |
| --- | ---: |
| `factor` minus `qrs_prior` (primary) | -0.0173 [-0.0558, +0.0212] |
| `factor` minus `textbook` | -0.0247 [-0.0498, -0.0016] |
| `factor` minus `qrs_lexicographic` | -0.0283 [-0.0682, +0.0100] |
| `factor_q` minus `factor` | -0.0033 [-0.0057, -0.0010] |
| `units_learned` minus `qrs_prior` | +0.0184 [-0.0195, +0.0582] |
| `units_learned` minus `factor` | +0.0357 [+0.0114, +0.0615] |
| `units_learned` minus `textbook` | +0.0110 [-0.0074, +0.0304] |
| `units_learned` minus `qrs_lexicographic` | +0.0073 [-0.0267, +0.0413] |
| `factor` minus `residual056` | +0.0450 [+0.0070, +0.0816] |
| `units_learned` minus `residual056` | +0.0807 [+0.0444, +0.1167] |

On the other axes, `factor` against `qrs_prior`:

- A3 (lead): +0.183 [+0.103, +0.263].
- A4 (time): -0.058 [-0.098, -0.017].
- A2_qrs: +0.078 [+0.027, +0.132].
- A2_nonqrs: +0.019 [-0.024, +0.062].

`units_learned` against `qrs_prior`:

- A3: +0.198 [+0.123, +0.274].
- A4: -0.071 [-0.107, -0.030].
- A2_qrs: +0.102 [+0.049, +0.155].
- A2_nonqrs: +0.048 [-0.004, +0.100].

`factor` against `textbook`: A3 +0.040 [-0.002, +0.085], A4 -0.005 [-0.020, +0.009].

## Noise

| Map | C1 | C1_std |
| --- | ---: | ---: |
| `factor` | 0.523 [0.497, 0.551] | 0.580 [0.554, 0.606] |
| `factor_q` | 0.483 [0.455, 0.513] | 0.563 [0.537, 0.589] |
| `textbook` | 0.482 [0.456, 0.509] | 0.486 [0.458, 0.514] |
| `units_learned` | 0.518 [0.492, 0.542] | 0.537 [0.508, 0.565] |
| `qrs_prior` | 0.353 | 0.560 [0.532, 0.589] |

The suppression moved raw C1 by -0.040 [-0.047, -0.033] against `factor`, but C1_std fell only from 0.580 to
0.563, a drop of 0.016. Its A2 cost was -0.0033 [-0.0057, -0.0010]. The lower bound passes the -0.01 limit,
but the C1_std drop misses the 0.05 threshold, so the suppression is not adopted.

`textbook` is the only map with C1_std near 0.5 (0.486): its lead weights come from a table, not from the
signal. `textbook_true` gives a noise-set record (a normal ECG) no statement, so its map there is the
amplitude tiebreak alone, and its C1 equals `amplitude`'s (0.333).

### Lead-quality classifier (held-out, `fit` partition)

- Training rows: 117,203 leads. Positives: 9,155 named noisy leads from 2,129 records. Negatives: every lead
  of 9,004 records without a noise annotation.
- Out-of-fold lead AUROC, pooled: 0.761.
- Within-record noisy-lead AUROC on the 2,129 `fit` records naming 1-11 leads: 0.728.
- The same on the 613 `fit` records that pass the noise-set rule: 0.708.
- On scored records, q has mean 0.080 and median 0.056; its 95th percentile is 0.227, and only 0.01% of
  scored leads have q above 0.5. The suppression weight (1 - q)^2 has median 0.891 and 5th percentile 0.598
  (computed from `quality/q.npz`), so the suppression is mild.
- The largest standardized coefficients are the burst ratio (-0.60), the flat fraction (-0.56) and the mains
  share (+0.27). The flat fraction averages 0.076 on `fit` leads, so it mostly measures quantized flat baseline
  segments rather than dropouts.

## Why `factor` did not beat the bar, and why `textbook` did better than `factor`

1. **The MIL lead weights cost A2, not the rhythm route.** An undeclared 2 x 2 decomposition rebuilt the maps
   of the ECGAD development records with each choice swapped (`descriptives.json`, `decomposition_A2`; the
   rebuilt `factor` matches the scored tables to 2.2e-16).

   | Lead source | Rhythm heads: MIL time evidence | Rhythm heads: whole beat |
   | --- | ---: | ---: |
   | MIL lead evidence | 0.7247 (`factor`) | 0.7081 |
   | Table lead indicator | 0.7508 | 0.7494 (`textbook`) |

   Swapping the lead source changes mean A2 by 0.026 to 0.041. Swapping the rhythm route changes it by 0.001 to
   0.017.

2. **MIL lead weights help inside QRS and hurt outside it.** Within QRS, `factor`'s A2_qrs is 0.617 against
   0.571 for `textbook`. Outside QRS, its A2_nonqrs is 0.680 against 0.726. A3 rises (0.639 against 0.599),
   but a lead weight multiplies the whole lead. A likely reason for the loss: a marked lead with a small MIL
   weight falls below unmarked components of strongly weighted leads. ECGAD marks are coarse (on average 7.23
   marked leads per marked record, `descriptives.json`), so narrowing the lead set loses more marked
   lead-samples than it gains.

3. **The marks favour QRS beyond what the statements predict.** By this experiment's delineation, QRS holds
   34.0% of marked lead-samples on 17.0% of samples, and T holds 26.5% of marks on 23.4% of samples
   (`marked_share_by_component`). Even with the true PTB-XL statements, painting their textbook regions reaches
   A4 0.826 against `qrs_prior`'s 0.851, and A2 only +0.016 [-0.029, +0.059] above the bar (`textbook_true`).
   For several statements `textbook_true` loses to `qrs_prior`: per-record mean A2 for LAFB is 0.720 against
   0.801, CRBBB 0.675 against 0.830, and LPR 0.567 against 0.694. The textbook region is a poor proxy for
   where these cardiologists marked.

4. **Where `factor` wins and loses** (undeclared per-statement means over marked records with the statement,
   five or more records):

   | Statement (records) | `qrs_prior` | `textbook` | `factor` | `units_learned` |
   | --- | ---: | ---: | ---: | ---: |
   | LVH (20) | 0.760 | 0.734 | 0.778 | 0.781 |
   | IRBBB (11) | 0.768 | 0.855 | 0.855 | 0.815 |
   | ISC_ (16) | 0.754 | 0.746 | 0.764 | 0.777 |
   | 1AVB (18) | 0.686 | 0.721 | 0.625 | 0.771 |
   | PVC (13) | 0.773 | 0.764 | 0.730 | 0.752 |
   | AFIB (11) | 0.735 | 0.718 | 0.661 | 0.681 |
   | CLBBB (5) | 0.824 | 0.764 | 0.695 | 0.720 |
   | NDT (7) | 0.710 | 0.588 | 0.626 | 0.602 |

   Records with several statements count in each row. `factor` beats `qrs_prior` on A2 in 42 of 92 records,
   `textbook` in 51 and `units_learned` in 50.

## The learned combiner

Trained on 289 held-out `fit` records of 259 patients (212,976 units, 28,272 weakly positive), none dropped.
The listwise loss fell from 6.582 at zero weights to 5.129. The largest weights are:

- `textbook` unit score +1.25;
- component QRS +0.66, T -0.72, ST -0.33, PQ +0.28;
- leads III +0.28, V3 +0.36, I -0.36, V5 -0.37, V6 -0.36.

The other evidence features carry small weights: normative +0.10, `factor` +0.08, amplitude -0.05, `mil`
-0.03, q +0.03 and `residual056` +0.002. So the combiner is mostly `textbook` with a learned QRS-over-ST-T
preference. That preference, learned from the `REGION_TABLE` weak labels, moves it toward the bar's time
structure.

## Region set (not independent for these maps)

`textbook`, `factor` and `units_learned` use the `REGION_TABLE` prior or its weak labels, so their B readings
are not independent tests, and `textbook_true`'s are circular. Reported for completeness:

| Map | B1 macro | B2 macro | B3 macro |
| --- | ---: | ---: | ---: |
| `factor` | 0.780 [0.772, 0.788] | 0.720 [0.702, 0.737] | 0.861 [0.848, 0.873] |
| `textbook` | 0.877 | 0.875 | 0.868 |
| `units_learned` | 0.879 | 0.872 | 0.873 |
| `factor_q` | 0.778 | 0.717 | 0.860 |
| `textbook_true` | 0.927 | 0.948 | 0.903 |
| `qrs_prior` | 0.728 | 0.576 | 0.774 |

`factor`'s B2 for LAFB is 0.526 (`textbook` 0.945); 065 found the same LAFB weakness.

## Integrity

| Check | Result |
| --- | --- |
| Bar: 061 `qrs_prior` ECGAD maps rescored | per-record A2 within 2.2e-16, mean 0.7419383358406556 against 0.7419383358406554 saved; passed |
| 065 `mil_lse` rebuilt from its saved maps and scored with the CLI | per-record A2 identical (difference 0), A2 0.662367873235951 equal to 065's; passed |
| 065 recomputation from the saved weights (5,160 records, GPU) | bag logits identical on 4,871 records; evidence within 2.4e-4 on 2,550 overlapping records (float16 rounding); passed |
| `residual056` recomputation | identical to 061's scored ECGAD maps on 198 records |
| 062 `normative_v1` path | the saved maps are used for scored records; three scored records recomputed with `record_maps` in the smoke checks matched the saved float16 maps exactly |
| Data separation (`integrity.separation`) | 11,617 quality-classifier records and 289 combiner records, none ECGAD or `evaluation`; every combiner record is a held-out 065 row and none is a `fit` clean normal; no ECGAD holdout record mapped; passed |
| Maps | all finite; every scored record had delineated beats (median 12) |

## Deviations and disclosures

- The data-separation asserts ran inside the `mil`, `quality` and `maps` phases and would have stopped the run.
  Their counts were written to `result.json` afterwards by a `separation` phase added after scoring. It reads
  only partition tables and saved IDs.
- The decomposition, the per-statement means and the component shares come from an undeclared descriptive
  script (`scripts/reports/describe_combination066.py`) written after the scores. They explain the results
  and are not candidates.
- Negatives of the quality classifier come from every `fit` record without a noise note, not only clean
  normals (declared in the protocol as a refinement of the brief). The Einthoven/Goldberger feature was
  dropped before any score because PTB-XL derives III, aVR, aVL and aVF from I and II.
- The strata and noise CLIs wrote `strata.*` and `noise_std.*` into the copied reference folders under
  `ref/`. No 061, 062 or 065 file was written.
- Smoke outputs are in `outputs/experiment066_combination_smoke/` and are not results.

## Runtime

CPU at most 8 processes, plus GPU0 for the MIL recomputation:

| Phase | Time |
| --- | ---: |
| MIL recomputation (GPU) | 37 s |
| Quality classifier | 17 s |
| Maps and combiner | 178 s |
| Integrity checks, five LocBench scorings, strata, noise | 897 s |

Wall time from start to the end of scoring was about 20 minutes. The intermediate maps (5.8 GB) live in
`/tmp/ecg066-work/full` and are not kept in `outputs/`.

## What it means

- Combining the MIL heads' lead evidence with a clinical component prior keeps most of the bar's time
  structure (A4 0.793 against 0.709 for 062 alone and 0.694 for 065 alone). It is the best map so far at
  picking marked QRS complexes and leads: A2_qrs 0.617 for `factor`, 0.641 for `units_learned`, against 0.607
  for `residual056`. It does not beat the disease-blind bar on A2, because the ECGAD marks favour QRS more than
  clinical statements predict.
- Multiplying by MIL lead weights is too sharp for coarse marks. A table lead set, or a learned blend
  dominated by it, ranks better.
- The lead-quality classifier learns the noise notes moderately (held-out within-record AUROC 0.73). At
  calibrated probabilities it barely changes the maps, so it cannot remove the residual pull to noisy leads.
- `units_learned` is the strongest development map so far, but its gain over the bar is not significant and
  was trained on `REGION_TABLE` weak labels. It must be confirmed once on the holdout half in Experiment 067.

## Follow-up ideas (for the backlog)

1. Soften the lead factor in `factor`: replace `L_k` by a mixture `(1 - a) + a L_k` with small a, or by a rank
   transform, so that MIL lead evidence reorders leads inside QRS without pushing marked leads below unmarked
   components. Fit `a` on `fit` region statements, not on ECGAD.
2. A balanced quality classifier (class-weighted, or q rescaled to its `fit` noisy-lead quantiles) with a
   larger gamma. The present q is too small to change the maps. Decide the scale on `fit` noise records.
3. Per-beat weak labels for the combiner from 065 rhythm heads (PVC, PAC), so that `units_learned` learns focal
   beats; at present every beat of a record shares the same label.
4. Report a statement-stratified A2 for 067. `textbook_true` loses to `qrs_prior` for LAFB, CRBBB and LPR,
   which suggests that ECGAD marks for axis and conduction findings sit on QRS complexes rather than in the
   textbook leads. This is worth asking the cardiologist about.
5. Add a QRS-over-ST-T time weight, learned on `fit` region statements, as an explicit factor of `factor`
   instead of relying on the combiner to learn it.

## Reproduce

```bash
cd /tmp/ecg-wt-066
export PYTHONPATH=/tmp/ecg-wt-066 OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u -m scripts.experiments.run_combination066 mil
CUDA_VISIBLE_DEVICES= .venv/bin/python -u -m scripts.experiments.run_combination066 quality
CUDA_VISIBLE_DEVICES= .venv/bin/python -u -m scripts.experiments.run_combination066 maps
CUDA_VISIBLE_DEVICES= .venv/bin/python -u -m scripts.experiments.run_combination066 score
.venv/bin/python -m scripts.experiments.run_combination066 separation
.venv/bin/python -m scripts.reports.describe_combination066
```
