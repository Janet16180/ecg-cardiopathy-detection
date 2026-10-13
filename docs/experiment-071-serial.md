# Experiment 071: same-patient serial change maps

Frozen 13 October 2026 before any evaluation-partition or ECGAD score. Part of the
[localization programme, part 2](localization-program-2-2026-10-13.md); scored with LocBench v1
([Experiment 061](experiment-061-locbench.md) and its amendments 2 and 3). The user authorized unsupervised
experiments for the day; the decisions below were taken by the running agent and are written down here.

## Question

When a patient has a normal ECG and a later ECG with a new statement, does the beat-aligned difference between
them light the leads and waves of the new statement, more than differences between two normal ECGs of the same
patient, and more than a population-referenced map of the later ECG alone (062 `normative_v1`)? This is the only
patient-specific localization signal available without new cardiologist marks. A secondary arm asks whether
change maps on `fit` pairs can teach a single-ECG model something the region table does not.

## Data and counts

All pairs lie inside PTB-XL folds 1-8. PTB-XL keeps each patient in one fold and the LocBench partition
(`docs/locbench-partition-v1.json`) splits folds 1-8 by patient, so every pair falls wholly in `fit` or wholly
in `evaluation`. Order comes from `recording_date`. No ECGAD record, fold 9 or 10, Challenge or EchoNext record
is used. Counts come from `outputs/experiment071_serial_v1/counts.json`, produced from metadata only
(`run_serial071 count`).

Folds 1-8 hold 17,418 ECGs of 15,023 patients; 1,694 patients have more than one ECG (4,089 ECGs), and 298 have
at least one ECG with NORM and another without.

Record definitions (code: `ecg_experiment/serial071.py::record_flags`):

- Rhythm rule: rhythm statements only among SR, SARRH, SBRAD and STACH (the LocBench region-set rule), and no
  pacemaker note.
- Normal: NORM at likelihood at least 50, no `REGION_TABLE` statement at any likelihood, rhythm rule
  (7,424 records).
- Finding-free: no `REGION_TABLE` statement at likelihood 50 or more, rhythm rule (11,044 records).
- Abnormal: at least one `REGION_TABLE` statement at likelihood 50 or more, rhythm rule. Its new statements are
  those absent from the partner's statement list at any likelihood; a pair needs at least one.

Pair sets (one pair per abnormal record, with the partner closest in time; the map is painted on the abnormal
record):

| Set | Partner | Order | `fit` pairs (patients) | `evaluation` pairs (patients) |
| --- | --- | --- | ---: | ---: |
| `forward` (primary) | normal | partner strictly earlier | 33 (29) | 20 (16) |
| `both` | normal | either order (ties prefer earlier) | 66 (60) | 39 (32); 20 forward |
| `finding_free` | finding-free | partner strictly earlier | 118 (96) | 74 (56) |
| Normal-normal null | consecutive normals of one patient | painted on the later | 306 (237) | 157 (131) |

`evaluation` pairs per family (a record counts in each family of its new statements):

| Family | `forward` | `both` | `finding_free` |
| --- | ---: | ---: | ---: |
| anterior MI | 6 | 8 | 12 |
| 1AVB | 4 | 4 | 18 |
| LAFB | 3 | 8 | 10 |
| ST injury | 3 | 7 | 7 |
| lateral MI | 3 | 5 | 5 |
| ischemia | 2 | 3 | 18 |
| LNGQT | 2 | 3 | 2 |
| IMI | 1 | 5 | 15 |
| WPW | 0 | 2 | 1 |

Five `forward` evaluation records carry two new statements. The median gap is 8.3 days for `forward`
evaluation pairs (quartiles 3.6 and 113 days) and 50 days for evaluation null pairs. Every evaluation abnormal
record has an ECGdeli file and a saved 062, 065, 066 and 068 map.

These counts are small. With 20 primary records from 16 patients, a difference of a few hundredths of AUROC will
not be distinguishable from zero, and no family has enough records for a family-level claim. Claims are worded
accordingly.

## Change map (`serial`, primary)

1. Beats: the frozen 062 recipe (`normative_map062.record_features`: waveform-only delineation, 40 Hz low-pass,
   PR-segment baseline, QRS alignment shift, piecewise-linear warp to 380 bins per lead, dominant-beat median).
2. Difference: later minus earlier median beat per lead and bin, in mV.
3. Standardization: the median and 1.4826 MAD of the same difference over the 306 `fit` normal-normal pairs, per
   lead and bin (`fit_null`). The spread is floored at the 10th percentile of all 4,560 lead-bin spreads
   (0.0120 mV on the `fit` pairs), so that cells whose spread is near zero by construction (next to the
   baseline knot) cannot dominate. z = (difference - centre) / floored spread.
4. Magnitude: local root mean square of z over 9 bins (062 `smooth_abs`).
5. Painting: the same grid map on every complete beat of the later ECG through that beat's warp (062
   `to_timeline`, maximum where beats overlap, uncovered samples at the minimum): float32 (12, 5000), canonical
   leads, higher means more change. A pair where either ECG has fewer than two complete beats gets no map and is
   counted.

Declared variants (same pairs, descriptive):

- `serial_interval`: `serial` plus interval-change evidence, the absolute standardized change in QRS duration,
  QT and PR (centre and 1.4826 MAD over `fit` null pairs, spread floored at 4 ms because intervals are quantized
  to samples) added to the QRS, T and pre-QRS bins of every lead.
- `cross`: the same recipe with a cross-patient reference. The partner is a `fit` normal ECG of another patient
  with the same sex, age within 5 years (then 10, then any; ages above 89 match each other) and the nearest heart
  rate; the standardization comes from 306 cross-patient matched `fit` normal-normal pairs (each `fit` null
  pair's later ECG with its matched other-patient normal). It asks whether the patient's own baseline matters.

## Comparators on the same abnormal records

From saved maps, hash-checked against their receipts: 062 `normative_v1` (primary comparator), 065 `mil_lse`,
066 `textbook` and 068 `units_v2` (both built from the region table, so circular for these targets), and 061
`qrs_prior` rebuilt with the frozen `locbench_baselines061.qrs_prior_map`.

## Readings

Targets come from PTB-XL+ ECGdeli global fiducials, as in LocBench, so they do not depend on the map's own
delineation. Units are lead x wave: P, PQ (PR interval minus P), QRS, ST and T, each the union over beats within
samples 100-4899 (`unit_masks`). The expected units of a record are the union over its new statements of
`REGION_TABLE` leads x components (PR = P and PQ).

- B1u (primary unit reading): within-record AUROC of the unit means of a map, expected against other units
  (`unit_auroc`). Empty units are left out; a record without both classes is counted and left out.
- B1, B2 and B3: LocBench's sample-level region, lead and component AUROCs for the union region (they equal
  `locbench.region_record_metrics` for one statement).
- Normal-normal null: on each evaluation null pair, B1u of its `serial` map against pseudo-statements, averaged
  over the statement sets of the change set being compared, weighted by how often each set occurs there
  (`null_reading`). The same for `serial_interval` and `cross`. Also the null B1u per single statement.
- Noise: on evaluation pairs (change and null) whose two ECGs together name 1-11 noisy leads, the within-record
  AUROC of the map's lead scores (95th percentile) against those leads, raw and standardized per lead by the
  median and IQR over evaluation null pairs with no noise note (as 061 amendment 3). 0.5 means no attraction.
- Descriptive: per-family means, family macro (families with at least 3 records, whole-patient bootstrap,
  draws missing a family skipped), and means by time gap (30 days or less against longer).

Statistics: means over records, 95% whole-patient bootstrap intervals with 2,000 draws and seed 61061
(`intervals.patient_groups` and `patient_resample`). Paired differences use records where both maps are
defined. The change-minus-null difference resamples the union of the two groups' patients together.

## Primary comparison and decision rule

Set `forward`, evaluation partition, map `serial`, reading B1u, record mean.

1. Serial signal: `serial` B1u on change pairs minus the composition-matched null B1u. The change map localizes
   new statements beyond normal-normal differences if the 95% lower bound is above 0.
2. Patient-specific against population: paired `serial` minus 062 `normative_v1` B1u on the same records. Above
   0 if the lower bound is above 0, below if the upper bound is below 0, otherwise not distinguishable.
3. Reported with the same rule, not decisive: `serial` minus `cross` (does the own baseline help), and every
   other comparator, on all three sets and all four readings.

The brief named a family macro as the primary reading. With 1-6 records per family in `forward`, a macro would
weight single records as heavily as families, so the record mean is primary and the macro is reported beside it.
A null B1u outside [0.45, 0.55] is flagged as a structural bias of the map; the primary contrast is
composition-matched, so the flag does not change the decision.

## Secondary: change maps as weak supervision (`serial_units`)

Exploratory, declared secondary. It is the only route by which serial information reaches ECGAD.

- Training pairs: every `fit` abnormal record with a finding-free partner in either order (221 pairs, 177
  patients), its `serial` map painted on it with the `fit` reference model.
- Units and features: Experiment 068's saved unit table (066 delineation, beat x lead x component) and its
  `units_v2` feature set at a = 0.25, read from `/tmp/ecg068-work/full/features_train.pkl` (out-of-fold MIL
  evidence for `fit` records).
- Labels: the units in the top 20% of the record's `serial` unit means.
- Model: 068's listwise linear ranker (`combiner068.fit_listwise`), L2 chosen from 068's grid by the lowest
  patient-grouped five-fold held-out listwise loss (folds by `combination066.patient_folds`, seed 71071).
  Reported beside it: the held-out within-record AUROC of the change labels for the fitted model and for the
  frozen 068 `units_v2` weights, which says how much of the change the region-table combiner already predicts.
- Maps: unit scores painted on every LocBench scored record with 066 `units_map` from 068's saved scored-record
  features, then scored with the LocBench CLI, paired against `qrs_prior` and 068 `units_v2`, with the strata and
  noise CLIs on copies of the reference tables. ECGAD development A2 is exploratory, as every part-2 ECGAD
  reading. The secondary map is also read on the primary pairs.

## Integrity checks

- Reproduce 061 `qrs_prior` ECGAD development A2 0.7419383358406554 from its saved maps, per record within
  1e-12 (before the primary readings and again before LocBench scoring).
- SHA-256 of every saved comparator map equals the one in its producing receipt.
- 068 `units_v2` repainted from the saved features and weights on 24 scored records matches the saved maps
  (within 1e-6).
- No ECGAD holdout record and no evaluation record enters any fit (asserted in each phase).

## Design data disclosure

The recipe above was chosen on `fit` pairs only (`run_serial071 develop`, outputs in `develop_fit/`, in-sample
for the reference model) after a smoke test on 12 `fit` pairs per set. No evaluation or ECGAD reading of any 071
map existed. On the 33 `fit` `forward` pairs the first recipe, which added interval-change evidence to every
lead, gave B1u 0.547 against an in-sample null of 0.556; the waveform-only map gave 0.635 against 0.560. A
scratch comparison on `fit` pairs with the null read on a held-out half of the `fit` null pairs (floor against
no floor, per-cell and per-segment null calibration, with and without interval terms) gave change-minus-null
differences of +0.07 to +0.10 for every waveform-only variant and -0.03 to +0.03 with interval terms. The
simplest waveform-only variant (floored z, no calibration) was kept as primary and interval evidence became the
`serial_interval` variant. On the same `fit` pairs 062 `normative_v1` reached 0.670 and `qrs_prior` 0.747, so a
loss against the population map is possible. The 10% floor lowers the PQ units in normal-normal maps (fit null
B1u for 1AVB 0.31, for QRS statements about 0.60-0.63); the composition-matched null absorbs this bias.

## Deviations from the brief

- Primary reading is the record mean of B1u, not the family macro (counts above).
- Interval-change evidence is a variant, not part of the primary map (design data above).
- Reverse and finding-free pairs are declared secondary sets with full readings, not only descriptives.

## Outputs

`outputs/experiment071_serial_v1/`: `counts.json`, `pairs.csv`, `null_pairs.csv`, `features_receipt.json`,
`fit_summary.json`, `integrity_primary.json`, `result_primary.json` and `tables/` (pair, null, noise and unit
readings), `secondary_receipt.json`, `integrity.json`, `locbench/serial_units/` with copied references in
`locbench/ref/`, and `result.json`. Large intermediates (beat summaries, reference models, secondary maps) stay
in `/tmp/ecg071-work`. The running agent writes `docs/experiment-071-serial-results.md` from these outputs.
