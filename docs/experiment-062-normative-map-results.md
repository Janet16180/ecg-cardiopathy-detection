# Experiment 062 results: normative phase-aligned population map

The primary map does not beat the bar. `normative_v1` reaches A2 0.6938 [0.6577, 0.7293] on ECGAD development,
against `qrs_prior` at 0.7419: paired difference -0.0481 [-0.0916, -0.0066]. Under the frozen decision rule
(lower bound above zero), the verdict is negative. The map is the best of all 062 and 061 maps at detection
(D1 0.855) and at picking marked leads (A3 0.635), and within QRS complexes it beats `qrs_prior`
(A2_qrs +0.051 [+0.019, +0.084]). It loses on time localization (A4 0.709 against 0.851). The declared
prior-combined variant `qrs_lexicographic` reaches A2 0.7530, +0.0111 [-0.0084, +0.0312] over the bar, which
is not significant.

Run on 12 October 2026 under the [protocol](experiment-062-normative-map.md) (commit 11b1dca) and its
Amendments 1-3 (commits ad401c7, d29e919, 3db0aa4). All three amendments were committed before any 062 score.
Code commit 8059fab. All numbers come from `outputs/experiment062_normative_v1/result.json`. Intervals are
LocBench's 95% whole-patient bootstraps: 2,000 draws, seed 61061, no skipped draws. Paired differences are this
map minus the baseline, on the same draws.

## Primary comparison

| ECGAD development (92 records, 90 patients) | A2 |
| --- | ---: |
| `normative_v1` (primary) | 0.6938 [0.6577, 0.7293] |
| `qrs_prior` (bar) | 0.7419 [0.7082, 0.7743] |
| Paired difference | -0.0481 [-0.0916, -0.0066] |
| Paired against `residual056` (0.6796) | +0.0142 [-0.0270, +0.0540] |
| Paired against `U_B` (0.6315) | +0.0623 [+0.0369, +0.0898] |
| Paired against `amplitude` (0.6978) | -0.0039 [-0.0384, +0.0306] |

The map beats `qrs_prior` on A2 in 45 of the 92 records.

## All readings against the bar

| Metric | `normative_v1` | Paired vs `qrs_prior` |
| --- | ---: | ---: |
| A1 pooled point AUROC | 0.8366 [0.8034, 0.8688] | +0.1414 [+0.1008, +0.1823] |
| A2 within-record (primary) | 0.6938 [0.6577, 0.7293] | -0.0481 [-0.0916, -0.0066] |
| A3 lead AUROC (64 records) | 0.6353 [0.5611, 0.7049] | +0.1788 [+0.1034, +0.2485] |
| A4 time AUROC | 0.7091 [0.6656, 0.7517] | -0.1415 [-0.1897, -0.0937] |
| A5 record AUROC | 0.8854 [0.8327, 0.9308] | +0.1727 [+0.0975, +0.2508] |
| A6 precision at top 5% (chance 0.186) | 0.4203 [0.3506, 0.4878] | +0.0632 [+0.0161, +0.1158] |
| B1 macro region AUROC | 0.6972 [0.6822, 0.7114] | -0.0311 [-0.0446, -0.0174] |
| B1 macro, strict | 0.7094 [0.6953, 0.7225] | -0.1282 [-0.1412, -0.1163] |
| B2 macro lead AUROC | 0.5995 [0.5770, 0.6221] | +0.0238 [-0.0004, +0.0486] |
| B3 macro component AUROC | 0.7030 [0.6855, 0.7198] | -0.0709 [-0.0859, -0.0552] |
| C1 noisy-lead AUROC | 0.6173 [0.5888, 0.6484] | +0.2644 [+0.2283, +0.3019] |
| C1_std (061 Amendment 3) | 0.6217 [0.5937, 0.6515] | not computed |
| C2 noise-record AUROC | 0.6227 [0.5850, 0.6548] | +0.1439 [+0.0944, +0.1935] |
| D1 detection AUROC | 0.8552 [0.8440, 0.8663] | +0.2231 [+0.2036, +0.2430] |

Flags under the decision rule:

- Noise concern: raised. C1 exceeds the bar's 0.353 by far more than 0.05. On the lead-standardized reading,
  C1_std is 0.622, between `qrs_prior` (0.560) and `residual056` (0.707) and below `U_B` (0.720).
- Detection concern: not raised. D1 is 0.855 against 0.632 for the bar and 0.762 for `U_B`, the best 061 detector.

## Where the evidence lands (061 Amendment 2 strata)

| Map | A2_qrs | A2_nonqrs |
| --- | ---: | ---: |
| `qrs_prior` | 0.5386 [0.5092, 0.5686] | 0.6611 [0.6312, 0.6901] |
| `residual056` | 0.6067 [0.5704, 0.6441] | 0.6711 [0.6356, 0.7070] |
| `normative_v1` | 0.5896 [0.5530, 0.6245] | 0.6447 [0.6027, 0.6852] |
| `normative_v1` minus `qrs_prior` | +0.0509 [+0.0190, +0.0835] | -0.0164 [-0.0548, +0.0205] |
| `normative_v1` minus `residual056` | -0.0171 [-0.0662, +0.0343] | -0.0264 [-0.0816, +0.0249] |

Inside the QRS mask, the map picks the marked complexes and leads better than the bar's amplitude tiebreak,
and about as well as `residual056`. Outside the mask it is no better than either. The A2 shortfall therefore
comes from ranking QRS against non-QRS samples, which is what `qrs_prior` encodes (A4 0.851 against 0.709).

## Variants (all declared before scoring; only `normative_v1` is primary)

| Variant | A2 | Paired A2 vs `qrs_prior` | A3 | A4 | B1 macro | C1 | C1_std | D1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `normative_v1` | 0.6938 | -0.0481 [-0.0916, -0.0066] | 0.6353 | 0.7091 | 0.6972 | 0.6173 | 0.6217 | 0.8552 |
| `max_combination` | 0.6825 | -0.0595 [-0.1021, -0.0181] | 0.6104 | 0.7077 | 0.6937 | 0.6487 | 0.6322 | 0.8597 |
| `persistent_only` | 0.6691 | -0.0729 [-0.1204, -0.0252] | 0.6434 | 0.7040 | 0.7001 | 0.5382 | 0.5409 | 0.8724 |
| `focal_only` | 0.6175 | -0.1244 [-0.1529, -0.0951] | 0.4922 | 0.5946 | 0.5917 | 0.6709 | 0.6914 | 0.6650 |
| `no_subspace` | 0.7091 | -0.0328 [-0.0788, +0.0116] | 0.6163 | 0.7306 | 0.7088 | 0.5826 | 0.5966 | 0.8587 |
| `no_noise_weight` | 0.6896 | -0.0524 [-0.0948, -0.0118] | 0.6028 | 0.6957 | 0.6959 | 0.7103 | 0.7145 | 0.8520 |
| `qrs_lexicographic` | 0.7530 | +0.0111 [-0.0084, +0.0312] | 0.5752 | 0.8007 | 0.7323 | 0.5403 | 0.5437 | 0.8169 |

`qrs_lexicographic` against `qrs_prior` also gives A3 +0.119 [+0.060, +0.175], A6 +0.047 [+0.025, +0.070] and
B2 macro +0.032 [+0.008, +0.055]. Its B3 is -0.019 [-0.031, -0.007]. Against `residual056` its A2 gain is
+0.073 [+0.041, +0.105]. Its strata are A2_qrs 0.591 and A2_nonqrs 0.652.

The ablations support three points:

- Persistent and focal evidence are complementary. Summing them beats either alone (A2 0.694 against 0.669
  and 0.618) and beats taking the maximum (0.683).
- Noise attraction comes from the focal component. `persistent_only` has C1_std 0.541, while `focal_only`
  has 0.691. The noise terms and the cross-lead coherence lower the primary's C1_std from 0.715
  (`no_noise_weight`) to 0.622.
- The rank-6 bounded subspace does not help. `no_subspace` has the higher A2 (0.709), A4 and B1, and its A3
  is lower (0.616 against 0.635). These are descriptive readings; no variant replaces the primary.

## Region set by family (`normative_v1`)

| Family | B1 | B2 |
| --- | ---: | ---: |
| 1AVB | 0.7022 [0.6541, 0.7492] | all leads |
| CLBBB | 0.9037 [0.8853, 0.9190] | all leads |
| CRBBB | 0.8855 [0.8641, 0.9046] | 0.6919 [0.6469, 0.7344] |
| IMI | 0.6686 [0.6427, 0.6947] | 0.5564 [0.5170, 0.5992] |
| LAFB | 0.7388 [0.7162, 0.7626] | 0.5996 [0.5677, 0.6296] |
| LNGQT | 0.6661 [0.5746, 0.7548] | all leads |
| ST injury | 0.4242 [0.3685, 0.4821] | 0.5723 [0.4803, 0.6689] |
| anterior MI | 0.6794 [0.6571, 0.7016] | 0.5811 [0.5442, 0.6192] |
| ischemia | 0.6059 [0.5794, 0.6313] | 0.5957 [0.5530, 0.6384] |

For 1AVB (0.702 against 0.382) and LNGQT (0.666 against 0.526), the map beats `qrs_prior` on B1, where the
interval evidence and the 1:1 pre-QRS window act as designed. ST injury stays below 0.5.

## Descriptive failure analysis (ECGAD development, per-record A2 by statement at likelihood 50 or more)

| Statement | Records | `normative_v1` | `qrs_prior` | `residual056` |
| --- | ---: | ---: | ---: | ---: |
| 1AVB | 18 | 0.760 | 0.686 | 0.612 |
| PVC | 11 | 0.796 | 0.768 | 0.703 |
| CLBBB | 5 | 0.832 | 0.824 | 0.728 |
| LAFB | 12 | 0.793 | 0.801 | 0.710 |
| CRBBB | 10 | 0.811 | 0.830 | 0.715 |
| ASMI | 20 | 0.723 | 0.731 | 0.673 |
| LVH | 20 | 0.678 | 0.760 | 0.727 |
| IMI | 8 | 0.698 | 0.818 | 0.675 |
| IRBBB | 11 | 0.691 | 0.768 | 0.692 |
| ISC_ (any ischemia) | 16 | 0.689 | 0.754 | 0.697 |
| NDT | 7 | 0.534 | 0.710 | 0.683 |
| NST_ | 4 | 0.557 | 0.782 | 0.734 |

Records with several statements count in each row. The map wins on conduction and rhythm findings with a
timing signature: 1AVB, PVC (A3 1.000 on those records) and CLBBB. It loses on LVH, IMI, IRBBB and nonspecific
ST-T findings. In those records the marks sit on QRS complexes while the map spreads evidence into ST, T and the
pre-QRS window.

## Integrity checks

- 061 reproduction (Amendment 1). Experiment 061's scored float32 `residual056` ECGAD maps were rescored with
  `ecgad_record_metrics`. They matched the saved per-record A2, A3, A4, A6 and marked share to at most
  2.2e-16, and re-summarizing the saved tables matched 061's A2, B1 macro, C1 and D1 values and intervals to at
  most 1.1e-16. This is a deviation from "exactly": the tolerance was 1e-12, and the residual differences are
  float round-off from the CSV round trip.
- The delineation sources `morphology_boundaries.py` and `fragment_localization.py` have the SHA-256 recorded in
  Experiment 059's receipt.
- The model was fitted on all 2,666 `fit` clean normals (2,566 patients, 30,472 complete beats). No record failed,
  and PR was missing in 51. The fit-normal medians were heart rate 68.6 bpm, QRS 110 ms, QT 388 ms and PR 168 ms.
- 4,871 scored records were mapped (`locbench.scored_ecg_ids()`), with no failure and no ECGAD holdout record
  (the runner raises on one). All maps are finite, and the largest value is about 93, well inside float16 range.
- Rebuilding the maps after adding `qrs_lexicographic` reproduced all six earlier variant maps bit for bit
  (identical SHA-256 in `maps_receipt_build1.json` and `maps_receipt.json`).

## Deviations and disclosures

- Design choices used only `fit`-partition records, listed in the protocol: ECGdeli agreement on 120 fit clean
  normals for the joint-fiducial quantiles, and descriptive lead, extra-beat and noise checks on fit-partition
  abnormal and noisy records. The ECGdeli check nudges the delineation toward the region-set boundaries on
  normals.
- Amendment 1 adopted `qrs_prior` as the bar from 061's completed receipt before the 061 report existed. The
  report later confirmed it.
- Amendments 2 and 3 added the strata and C1_std readings and one prior-combined variant, all before scoring.
- The 061 reference tables were copied into `outputs/experiment062_normative_v1/locbench/ref061/`, so no 061 file
  was written. ECGAD marks were only accessed through LocBench's own benchmark loader, which keeps the
  development half.
- No second version was scored on ECGAD development.

## Runtime

CPU only, at most 8 processes:

| Stage | Time |
| --- | ---: |
| Fit | 68 s |
| Maps, 7 variants for 4,871 records | 80 s |
| Scoring, including integrity check and secondary readings | 931 s |

Maps take about 3.5 GB as float16 under `outputs/experiment062_normative_v1/`.

## Follow-up ideas (for the backlog)

1. Time prior from normals rather than from disease. Calibrate each segment by how often it carries abnormality
   in the `fit`-partition abnormal records, or learn a segment weight on the `fit` partition region statements
   (no ECGAD). This addresses the A4 gap without a hand-set QRS prior.
2. Refit with rank 0 or 2 and a tighter coordinate bound. The rank-6 subspace absorbed abnormal shape (the
   `no_subspace` A2 was 0.709); confirm on the `fit` partition before any ECGAD score.
3. A QRS-specific persistent score at finer resolution (QRS in 100 bins, unsmoothed), aimed at the LVH, IMI and
   IRBBB losses where the marks sit on QRS.
4. Use `normative_v1` as the anomaly channel in Experiment 066. It brings the best detection (D1 0.855) and lead
   evidence (A3 0.635), and with a QRS prior it reaches A2 0.753. Its focal channel needs the lead-quality map,
   since noise attraction comes from there.
5. Replace the cross-lead coherence with an explicit limb-lead (Einthoven) consistency check, to separate
   electrode artifacts from physiological beat changes.
