# Experiment 071 results: same-patient serial change maps

Run on 13 October 2026 under the [protocol](experiment-071-serial.md), committed as a2abb1a with the code before
any evaluation-partition or ECGAD reading of a 071 map. The primary readings, the reference models and the
secondary maps come from that commit; the secondary-pair readings and a label fix in the noise table come from
350ccd5. Every number below comes from
`outputs/experiment071_serial_v1/result.json` (SHA-256 `83b87445…6e8d`), which embeds `counts.json`,
`fit_summary.json`, `result_primary.json`, `secondary_receipt.json`, `secondary_pairs.json`, the LocBench
summary of the secondary map and both integrity files; per-record values are in `tables/`. Intervals are 95%
whole-patient bootstraps with 2,000 draws and seed 61061. Paired differences use the same records for both maps.
No ECGAD holdout record was read, and only `fit`-partition pairs entered any fitted quantity.

## Decision

On the 20 primary `forward` evaluation pairs (16 patients), the change map `serial` has a unit AUROC (B1u) of
0.616 [0.529, 0.702] for the expected lead x wave units of the new statement. The 157 evaluation normal-normal
pairs, read against the same mix of statements, give 0.536 [0.526, 0.545].

| Primary comparison (`forward`, B1u) | Difference | Decision |
| --- | --- | --- |
| `serial` minus normal-normal null | +0.080 [-0.005, +0.171] | not distinguishable |
| `serial` minus 062 `normative_v1` (paired) | -0.007 [-0.170, +0.120] | not distinguishable |
| `serial` minus `cross` (paired) | -0.028 [-0.144, +0.090] | not distinguishable |

The lower bound of the first comparison misses 0 by 0.005, so under the frozen rule the serial signal is not
shown. With 20 records the intervals are about 0.17 to 0.29 wide; differences of the size seen on `fit` pairs
(+0.075) cannot be resolved. The null flag was not raised (null B1u 0.536 lies in [0.45, 0.55]).

## Main table (`forward`, 20 evaluation records)

| Map | B1u | `serial` minus map (paired) | Records where `serial` is higher |
| --- | ---: | ---: | ---: |
| `serial` (primary) | 0.616 [0.529, 0.702] | | |
| `serial_interval` | 0.587 [0.483, 0.691] | +0.028 [-0.072, +0.124] | 11 |
| `cross` (matched other patient) | 0.644 [0.570, 0.714] | -0.028 [-0.144, +0.090] | 7 |
| 062 `normative_v1` | 0.622 [0.502, 0.764] | -0.007 [-0.170, +0.120] | 11 |
| 065 `mil_lse` | 0.673 [0.583, 0.745] | -0.057 [-0.197, +0.097] | 7 |
| 061 `qrs_prior` | 0.614 [0.462, 0.777] | +0.002 [-0.157, +0.177] | 8 |
| 066 `textbook` (circular) | 0.897 [0.800, 0.963] | -0.281 [-0.386, -0.167] | 2 |
| 068 `units_v2` (circular) | 0.888 [0.788, 0.958] | -0.272 [-0.384, -0.156] | 4 |

`textbook` and `units_v2` paint the region table that defines the targets, so their rows are an upper reference,
not a competitor. Among the maps that do not use the table, none is separable from another on these 20 records.

Sample-level LocBench readings on the same records:

| Map | B1 | B2 (14 records) | B3 |
| --- | ---: | ---: | ---: |
| `serial` | 0.587 [0.543, 0.630] | 0.521 [0.347, 0.692] | 0.581 [0.490, 0.663] |
| `cross` | 0.589 [0.521, 0.655] | 0.693 [0.494, 0.823] | 0.533 [0.442, 0.611] |
| `normative_v1` | 0.601 [0.520, 0.699] | 0.583 [0.432, 0.702] | 0.590 [0.503, 0.690] |
| `mil_lse` | 0.699 [0.620, 0.763] | 0.696 [0.565, 0.781] | 0.809 [0.730, 0.878] |
| `qrs_prior` | 0.670 [0.558, 0.793] | 0.543 [0.313, 0.724] | 0.732 [0.619, 0.849] |
| `units_v2` | 0.878 [0.793, 0.933] | 0.876 [0.779, 0.955] | 0.877 [0.792, 0.945] |

`serial` against `normative_v1`: B1 -0.015 [-0.112, +0.066], B2 -0.062 [-0.150, +0.039]. The serial map picks
the expected leads no better than chance (B2 0.521).

## Secondary pair sets

| Reading (B1u) | `forward` (20) | `both` (39) | `finding_free` (74) |
| --- | ---: | ---: | ---: |
| `serial` | 0.616 [0.529, 0.702] | 0.608 [0.543, 0.672] | 0.551 [0.504, 0.599] |
| Normal-normal null for that set | 0.536 [0.526, 0.545] | 0.551 [0.540, 0.563] | 0.526 [0.519, 0.532] |
| `serial` minus null | +0.080 [-0.005, +0.171] | +0.056 [-0.013, +0.122] | +0.025 [-0.020, +0.074] |
| `cross` | 0.644 [0.570, 0.714] | 0.627 [0.569, 0.687] | 0.606 [0.560, 0.650] |
| `cross` minus its null | +0.112 [+0.036, +0.182] | +0.081 [+0.021, +0.141] | +0.087 [+0.039, +0.132] |
| `normative_v1` | 0.622 [0.502, 0.764] | 0.588 [0.501, 0.684] | 0.638 [0.579, 0.693] |
| `serial` minus `normative_v1` | -0.007 [-0.170, +0.120] | +0.020 [-0.094, +0.124] | -0.088 [-0.154, -0.024] |
| `serial` minus `cross` | -0.028 [-0.144, +0.090] | -0.020 [-0.095, +0.052] | -0.055 [-0.106, +0.003] |
| `serial` minus `qrs_prior` | +0.002 [-0.157, +0.177] | -0.062 [-0.173, +0.059] | -0.044 [-0.122, +0.031] |

- On the largest set (`finding_free`, partner without any region statement at 50 or more), the serial map is
  below the population map 062 by 0.088, and its excess over the null (+0.025) is not distinguishable from 0.
- The cross-patient map, whose reference is a matched normal ECG of another person, exceeds its own null on all
  three sets. The patient's own baseline did not make the map more specific than a matched stranger's.
- Sample-level, on `finding_free`, `serial` minus `normative_v1` is B1 -0.083 [-0.128, -0.039] and B2 -0.100
  [-0.175, -0.027].

Family macro B1u (families with at least 3 records; descriptive): `forward` `serial` 0.583 [0.519, 0.681],
`cross` 0.675 [0.628, 0.743], `normative_v1` 0.601 [0.524, 0.714], `serial` minus `normative_v1` -0.018
[-0.133, +0.092] over five families (1AVB, LAFB, ST injury, anterior MI, lateral MI); 540 of 2,000 draws missed a
family and were skipped.

## Null and noise checks

- Normal-normal null by single statement (157 evaluation pairs): B1u 0.59-0.62 for QRS statements (IMI 0.611,
  CLBBB 0.624, LAFB 0.616), 0.52-0.59 for ST and T statements, 0.403 for WPW and 0.320 [0.300, 0.340] for 1AVB.
  The serial map is structurally low in PQ units and high in QRS units even when nothing changed. This is the
  bias the protocol foresaw from the spread floor; the composition-matched null absorbs it in the primary
  contrast, but it affects every comparison with other maps (for example 1AVB below).
- Noise (97 evaluation pairs naming 1-11 noisy leads, 49 of them normal-normal): `serial` noisy-lead AUROC 0.506
  [0.450, 0.560], lead-standardized 0.543 [0.487, 0.596]; `cross` 0.527 [0.476, 0.575] and 0.548 [0.493, 0.599].
  Neither interval excludes 0.5, so no clear pull to noisy leads was measured.

## By family and time gap (descriptive, very small counts)

Mean B1u on `forward` evaluation records:

| Family (records) | `serial` | `cross` | `normative_v1` | `qrs_prior` | `units_v2` |
| --- | ---: | ---: | ---: | ---: | ---: |
| anterior MI (6) | 0.657 | 0.662 | 0.566 | 0.882 | 0.937 |
| 1AVB (4) | 0.395 | 0.555 | 0.680 | 0.421 | 0.762 |
| LAFB (3) | 0.557 | 0.658 | 0.831 | 0.906 | 1.000 |
| ST injury (3) | 0.586 | 0.845 | 0.267 | 0.218 | 0.885 |
| lateral MI (3) | 0.720 | 0.656 | 0.660 | 0.939 | 0.950 |
| LNGQT (2) | 0.841 | 0.620 | 0.760 | 0.162 | 0.819 |
| ischemia (2) | 0.746 | 0.539 | 0.501 | 0.222 | 0.789 |
| IMI (1) | 0.766 | 0.421 | 0.579 | 0.842 | 1.000 |

- The serial map does worst for 1AVB (0.395), where the null is lowest (0.320) because of the PQ bias.
- For ST-T findings (ST injury, ischemia, LNGQT) the serial map is above `normative_v1` and `qrs_prior` on these
  2-3 records each; in the `both` set the same holds for ST injury (7 records: 0.698 against 0.300 and 0.376)
  and ischemia (3 records: 0.668 against 0.383 and 0.257).
- Time gap, `forward`: 12 pairs 30 days apart or less have `serial` 0.648 (`normative_v1` 0.550); 8 pairs more
  than 30 days apart have 0.567 (0.730). In `finding_free` the gap made little difference (0.545 and 0.565).
- Records 21686 and 21687 (one patient) were recorded 3 and 6 minutes after a normal ECG and carry new ASMI and
  ALMI statements; `serial` reaches 0.733 and 0.675 on them (`tables/pair_readings.csv`).

## Secondary: change maps as weak supervision (`serial_units`, exploratory)

Training: 221 `fit` pairs of 177 patients (finding-free partner, either order), none dropped; 169,416 units,
33,896 labelled as the top 20% of each record's `serial` change. L2 1e-3 was chosen (held-out listwise loss
6.3994; grid 6.3994-6.4526).

| Held-out within-record AUROC for the change labels (`fit`, five patient folds) | Value |
| --- | ---: |
| `serial_units` (fitted) | 0.713 |
| Frozen 068 `units_v2` weights on the same units | 0.582 |

The region-table combiner predicts which units change in a patient only slightly better than chance. The
fitted model's largest weights are unit-type priors: PQ -1.38, P +0.72, ST +0.66, T +0.33, QRS -0.32, and the
lead-quality probability q +0.48 (068 `units_v2`: PQ +0.17, P -0.11, ST -0.16, T -0.54, QRS +0.64, q +0.09).
It learned the change map's own structure, including the PQ bias and a preference for noisier leads.

LocBench readings of `serial_units` (ECGAD development is exploratory):

| Reading | `serial_units` | Minus `qrs_prior` | Minus 068 `units_v2` |
| --- | ---: | ---: | ---: |
| A2 (92 records) | 0.6870 [0.6545, 0.7182] | -0.0550 [-0.0977, -0.0126] | -0.0802 [-0.1179, -0.0423] |
| A3 lead (64) | 0.5852 [0.5223, 0.6492] | +0.1287 [+0.0786, +0.1806] | -0.0701 [-0.1283, -0.0086] |
| A4 time (91) | 0.7656 [0.7247, 0.8032] | -0.0850 [-0.1264, -0.0476] | -0.0189 [-0.0599, +0.0198] |
| A2_qrs (89) | 0.5675 [0.5218, 0.6134] | +0.0289 [-0.0111, +0.0707] | |
| A2_nonqrs (92) | 0.7016 [0.6723, 0.7309] | +0.0404 [+0.0120, +0.0662] | |
| B1 macro | 0.7214 [0.7108, 0.7320] | -0.0069 [-0.0168, +0.0031] | -0.1602 [-0.1728, -0.1484] |
| B2 macro | 0.6467 [0.6272, 0.6653] | +0.0709 [+0.0486, +0.0917] | -0.2239 [-0.2438, -0.2039] |
| B3 macro | 0.7608 [0.7474, 0.7735] | -0.0131 [-0.0248, -0.0019] | -0.1147 [-0.1300, -0.0995] |
| C1_std (299 records) | 0.599 [0.569, 0.629] | `qrs_prior` 0.560 | `units_v2` 0.540 |
| D1 | 0.5470 [0.5287, 0.5645] | -0.0851 [-0.1097, -0.0603] | -0.0699 [-0.0920, -0.0465] |

The strata are paired against `qrs_prior`; `units_v2` has A2_qrs 0.638 and A2_nonqrs 0.733 on the same run. On
ECGAD development `serial_units` is below both references on A2, and it is drawn to noisy leads more than either
(C1_std 0.599). On the evaluation change pairs it is closer to its own training signal: B1u 0.641 [0.550, 0.720]
on `forward`, 0.647 [0.581, 0.708] on `both` and 0.626 [0.564, 0.687] on `finding_free`, where it exceeds the
serial map itself by +0.075 [+0.023, +0.125] and equals `normative_v1` (-0.013 [-0.085, +0.059]); it stays far
below `units_v2` (-0.209 to -0.246).

## Integrity

| Check | Result |
| --- | --- |
| 061 `qrs_prior` ECGAD development A2 from saved maps | per record within 2.2e-16, mean 0.7419383358406556 against 0.7419383358406554 saved; passed before the primary readings and again before LocBench |
| Saved comparator maps (062 `normative_v1`, 065 `mil_lse`, 066 `textbook`, 068 `units_v2`) | SHA-256 equal to their receipts |
| 068 `units_v2` repainted from saved features and weights, 24 records | maximum difference 0.0 |
| Beat summaries | 5,713 records, none without two complete beats; no pair lost a map |
| Readings defined | every pair and null pair had a finite B1u for every map |
| Data separation | reference models from 306 `fit` normal-normal pairs (same patient) and 306 cross-patient `fit` pairs; cross partners drawn from 4,913 `fit` normals; secondary trained on `fit` pairs only (asserted) |

The same-patient reference is much tighter than the cross-patient one: the median QRS-bin spread is 0.037 mV
against 0.139 mV, and the T-bin spread 0.041 against 0.090 mV (`fit_summary.json`).

## Design data

The recipe was fixed on `fit` pairs before the protocol commit (`develop_fit/`, in-sample for the reference
model). On the 33 `fit` `forward` pairs `serial` reached B1u 0.635 against an in-sample null of 0.560
(+0.075 [-0.001, +0.137]), `normative_v1` 0.670 and `qrs_prior` 0.747. The evaluation result (0.616 against
0.536) is close to that `fit` reading.

## Deviations and disclosures

- The primary reading is the record mean of B1u rather than the family macro (1-6 records per family); the macro
  is reported above.
- Interval-change evidence was moved from the primary map to the `serial_interval` variant after the `fit`
  development readings, before the protocol commit.
- The secondary-pair readings (`secondary_pairs.json`) were added to the code after the primary scores and
  before the secondary map was scored; the protocol declared them.
- The strata and noise CLIs wrote into copies of the reference tables under `locbench/ref/`; no 061 or 068 file
  was written.
- The primary score phase was run a second time after relabelling the normal-pair rows of
  `tables/noise_readings.csv` from `null` (which pandas reads as missing) to `normal_pair`; every reading in
  `result_primary.json` was identical apart from the run time.
- Smoke outputs (`outputs/experiment071_serial_smoke/`) used `fit` pairs only and are not results.

## Runtime

CPU only, at most 8 threads or processes: beat summaries 37 s (`features_receipt.json`), primary readings 50 s
(`result_primary.json`), secondary fit and 4,871 maps 69 s (`secondary_receipt.json`), LocBench scoring with
strata and noise 97 s (`locbench_receipt.json`); the reference-model fit was not timed. The secondary maps
(1.2 GB) and beat summaries are in `/tmp/ecg071-work/full`.

## What it means

- Same-patient differences between a normal and a later abnormal PTB-XL ECG light the expected lead x wave
  units only modestly (B1u about 0.55-0.62), not clearly more than normal-normal differences on 20 primary
  pairs, and not more than the population map 062. On 74 finding-free pairs they are worse than 062.
- The patient-specific reference is tighter, but a matched stranger's ECG gave an equal or better map. The
  change between two recordings carries electrode-placement and reading changes that are as large as the
  finding, and the new statement's textbook region is a coarse target.
- The signal may be real for ST-T findings, where population maps fail (ST injury and ischemia), but those
  families have 2-7 records here.
- Training a single-ECG model on change maps teaches it the change map's biases (PQ, noisy leads), not where
  findings are; it does not help on ECGAD development.

## Follow-up ideas (for the backlog)

1. Pool all PTB-XL serial pairs, including folds 9 and 10 outside ECGAD, only after the user approves it, to
   reach about 100 primary pairs; the present design is powered only for differences above about 0.1.
2. Per-cell null calibration of the change map instead of a spread floor, to remove the PQ bias (1AVB null
   0.320) before any comparison with other maps.
3. Use serial change as evidence only for ST-T findings (ST injury, ischemia, long QT), combined with a
   population map, and test that on the `both` and `finding_free` sets.
4. Explain the cross-patient result: compare same-patient and matched-stranger differences per lead to see
   whether limb-electrode placement or reading differences dominate the same-patient changes.
5. Check the near-simultaneous pairs (minutes apart, new statements) as a reading-variability sample: they bound
   how much of a statement change is a reading change rather than a waveform change.
