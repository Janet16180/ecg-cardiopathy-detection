# Experiment 069 results: measurement-based criteria localizer

Run on 13 October 2026 (UTC) under the [protocol](experiment-069-criteria-localizer.md), committed as 12debaa before
any 069 map was built for a scored record. The code was committed as 23d043f and produced every map and score. Every
number below comes from `outputs/experiment069_criteria_v1/result.json` (SHA-256 `6c1f88ad…`) unless it is marked
as an undeclared descriptive from `descriptives_undeclared.json` (`88543d4d…`). Intervals are LocBench's 95%
whole-patient bootstraps (2,000 draws, seed 61061, no skipped draws). A2 uses the 92 marked ECGAD development
records of 90 patients. ECGAD development has been compared many times, so every ECGAD reading here is
exploratory and none is called confirmed.

## Decision

The primary map `criteria` does not beat the bar. Its A2 is 0.7241 [0.6926, 0.7531], against 0.7419 for
`qrs_prior`. The paired difference is -0.0178 [-0.0608, +0.0203], so under the declared words `criteria`
**matches** the bar. It has the higher A2 in 41 of the 92 records. It is below 068 `units_v2` (-0.0430 [-0.0798,
-0.0105]) and not separable from 066 `textbook` (-0.0253 [-0.0649, +0.0113]).

Key secondaries:

- `units_v2_criteria` minus `units_v2`: +0.0008 [-0.0011, +0.0026]. Adding the criteria features changes the
  learned combiner by nothing measurable.
- `criteria_x_mil` minus `textbook`: -0.0276 [-0.0658, +0.0073]. Weighting the criteria by the 065 finding
  probabilities does not beat painting the predicted diagnosis's textbook region.
- `criteria_agree` lowers C1_std by 0.0032 [-0.0069, -0.0004]. That meets the declared rule (interval below 0),
  but the drop is tiny, and it costs A2 -0.0042 [-0.0078, -0.0010] against `criteria`.

Measured criteria therefore do not localize better than the textbook region on ECGAD. They do land on the
statement's expected leads and waves better than a disease-blind map in the independent region test, and the leads
they add beyond the textbook leads are marked by the cardiologists about as often as the textbook leads.

## Main table

| Map | A2 | vs `qrs_prior` | vs `units_v2` | A3 | A4 | A2_qrs | A2_nonqrs | B1 macro | B2 macro | C1_std | D1 |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `criteria` (primary) | 0.7241 | -0.0178 [-0.0608, +0.0203] | -0.0430 [-0.0798, -0.0105] | 0.495 | 0.796 | 0.542 | 0.684 | 0.777 | 0.630 | 0.526 | 0.876 |
| `criteria_agree` | 0.7199 | -0.0220 [-0.0650, +0.0166] | -0.0472 [-0.0839, -0.0149] | 0.529 | 0.793 | 0.539 | 0.678 | 0.774 | 0.609 | 0.522 | 0.874 |
| `criteria_x_mil` | 0.7218 | -0.0201 [-0.0606, +0.0189] | -0.0453 [-0.0792, -0.0137] | 0.500 | 0.804 | 0.548 | 0.684 | 0.779 | 0.642 | 0.531 | 0.895 |
| `criteria_own` (diagnostic) | 0.7251 | -0.0168 [-0.0536, +0.0205] | -0.0420 [-0.0775, -0.0076] | 0.505 | 0.781 | 0.560 | 0.674 | 0.766 | 0.600 | 0.537 | 0.851 |
| `criteria_qrs` (secondary) | 0.7631 | +0.0212 [-0.0003, +0.0434] | -0.0040 [-0.0364, +0.0269] | 0.492 | 0.840 | 0.551 | 0.684 | 0.780 | 0.630 | 0.546 | 0.809 |
| `units_v2_criteria` | 0.7679 | +0.0260 [-0.0119, +0.0649] | +0.0008 [-0.0011, +0.0026] | 0.665 | 0.791 | 0.637 | 0.734 | 0.883 | 0.870 | 0.522 | 0.576 |
| 068 `units_v2` (reference) | 0.7671 | | | 0.655 | 0.785 | 0.638 | 0.733 | 0.882 | 0.871 | 0.541 | 0.617 |
| 066 `textbook` (reference) | 0.7494 | | | 0.599 | 0.798 | 0.571 | 0.726 | 0.877 | 0.875 | 0.486 | 0.657 |
| 061 `qrs_prior` (bar) | 0.7419 | | | 0.457 | 0.851 | 0.539 | 0.661 | 0.728 | 0.576 | 0.560 | 0.632 |

Reference strata and C1_std were recomputed in this run from tables copied into `ref/`; the `qrs_prior` A3, A4,
B and D1 values are the reference side of the paired comparisons.

Other readings for `criteria`: A1 0.815 (paired against `qrs_prior` +0.120 [+0.081, +0.156]), A5 0.914, A6 0.407
(chance 0.186; against the bar +0.050 [-0.008, +0.110]), C1 0.363, C2 0.489. Records with the higher A2 than
`qrs_prior` (read from the saved tables): `criteria_agree` 38, `criteria_x_mil` 39, `criteria_own` 43,
`criteria_qrs` 46 and `units_v2_criteria` 52 of 92.

## Where the evidence lands on ECGAD

`criteria` against `qrs_prior`:

- A3 (which lead): +0.039 [-0.021, +0.095]. Its A3 of 0.495 is at chance.
- A4 (when): -0.055 [-0.095, -0.014].
- A2_qrs: +0.003 [-0.025, +0.031]. Inside the QRS it ranks no better than the bar's amplitude tie-break.
- A2_nonqrs: +0.023 [-0.009, +0.056].

Against `units_v2`, `criteria` is behind on leads (A3 -0.160 [-0.244, -0.077]) and inside the QRS (A2_qrs -0.096
[-0.144, -0.050]). It is level on time (A4 +0.011 [-0.031, +0.053]).

The lead reading is at chance because the criteria fire on most leads of a marked record. In the undeclared
descriptive, 72% of the 94 marked development records have evidence on all 12 leads. 3% have none. The all-lead
criteria fire often in these records: `ivcd` (QRS >= 110 ms) in 43%, `pr_long` in 38% and `qtc_long` in 23%. Some
lead-specific criteria are also common: `std` in 60%, `tinv` in 46%, `rvh_v1` in 46% and `p_wide` in 46%.
A map that paints QRS on every lead whenever QRS >= 110 ms says nothing about which lead.

`criteria_qrs` adds the QRS time prior lexicographically. Its A2 rises by +0.0390 [+0.0134, +0.0661] over
`criteria`, to 0.7631. That is +0.0212 [-0.0003, +0.0434] over the bar and level with `units_v2` (-0.0040
[-0.0364, +0.0269]), with no training on the region table. Its A4 is 0.840, against 0.851 for the bar. Its A3
(0.492) is still at chance. The gain comes from time structure, not from lead evidence.

## Region test (independent for the label-free maps)

`criteria`, `criteria_agree`, `criteria_own` and `criteria_qrs` never see `REGION_TABLE` or any statement, so their
B readings independently test whether measured evidence lands on each statement's expected leads and waves.
`criteria_x_mil` is only partly independent: its 065 heads were trained on statements, and its territory table
overlaps the region table's lead sets. B is not independent for `units_v2_criteria`, which is trained on
region-table weak labels.

`criteria` against `qrs_prior`:

- B1 (region): +0.048 [+0.036, +0.059].
- B2 (lead): +0.054 [+0.036, +0.071].
- B3 (component): +0.034 [+0.019, +0.046].

It stays far below the textbook map, which knows the region by construction (B1 -0.101 [-0.117, -0.085], B2 -0.246
[-0.270, -0.224]). Its B2 of 0.630 is also below the 0.719 that 065's statement-trained MIL heads reached (065
report).

| Family | B1 `criteria` | B2 `criteria` | B1 `qrs_prior` |
| --- | ---: | ---: | ---: |
| CLBBB | 0.949 | all leads | |
| CRBBB | 0.910 | 0.707 | |
| 1AVB | 0.869 | all leads | 0.382 (062 report) |
| anterior MI | 0.782 | 0.768 | |
| LAFB | 0.796 | 0.439 | |
| IMI | 0.752 | 0.505 | |
| LNGQT | 0.680 | all leads | 0.526 (062 report) |
| ischemia | 0.628 | 0.600 | |
| ST injury | 0.623 | 0.759 | |
| macro | 0.777 | 0.630 | 0.728 |

Where a criterion names the right leads and the record meets it, the evidence lands well: conduction and PR
findings on B1, and anterior MI and ST injury on B2. The lead test fails for IMI (0.505) and LAFB (0.439). For IMI,
III carries no UDMI Q criterion. For LAFB, the axis criterion paints I, II, III, aVL and aVF together, while the
region table's limb set also includes aVR.

## Patient specificity

Region set (852 `evaluation` records in families whose lead set is not all 12 leads): the evidence leads differ
from the textbook leads in 79% of records. They average 7.1 leads, of which 4.3 lie outside the textbook set, and
the mean Jaccard index is 0.33. Every CRBBB record has evidence on all 12 leads, because `ivcd` paints the whole
QRS of every lead. In 5.9% of records no criterion is met in any lead.

ECGAD development: 34 marked records have a region statement whose lead set is not all 12 leads. On those records
we sorted each lead by whether it carries criteria evidence and whether it is a textbook lead, and counted how
often the cardiologist marked it:

| Lead group | Marked / leads | Marked share |
| --- | ---: | ---: |
| Departure (evidence, not textbook) | 96 / 158 | 0.61 |
| Shared (evidence and textbook) | 132 / 189 | 0.70 |
| Textbook only (no evidence) | 12 / 18 | 0.67 |
| Neither | 9 / 43 | 0.21 |

Undeclared descriptive, leaving out the five criteria that name all 12 leads (`ivcd`, `pr_long`, `wpw`,
`qtc_long`, `qtc_short`): the evidence leads shrink to 4.0 per region-set record, 69% of records still depart, and
on ECGAD the shares become 0.68 (63 / 92) departure, 0.73 (94 / 129) shared, 0.64 (50 / 78) textbook only and 0.39
(42 / 109) neither.

The leads a record adds beyond the textbook set are marked about as often as textbook leads and far more often than
the remaining leads. The measured evidence is therefore patient-specific in a way the cardiologists agree with. It
does not raise A2, because ECGAD marks are coarse (about 7 leads per marked record) and already cover most of these
leads, and because the evidence ranks waves poorly in time. These are small exploratory counts without intervals.

## Noise and the agreement factor

| Map | C1 | C1_std | C1_std paired |
| --- | ---: | ---: | --- |
| `criteria` | 0.363 | 0.526 [0.500, 0.551] | vs `qrs_prior` -0.035 [-0.067, -0.004] |
| `criteria_agree` | 0.352 | 0.522 [0.497, 0.548] | vs `criteria` -0.0032 [-0.0069, -0.0004] |
| `criteria_x_mil` | 0.362 | 0.531 | vs `criteria` +0.005 [-0.005, +0.015] |
| `criteria_own` | 0.405 | 0.537 | vs `criteria` +0.011 [-0.025, +0.046] |
| `units_v2_criteria` | 0.479 | 0.522 | vs `units_v2` -0.018 [-0.028, -0.008] |

The criteria maps are only mildly drawn to noisy leads, less than the bar. The noise-set records are normal ECGs,
so most of their leads meet no criterion, and the maps there reduce to the amplitude tie-break.

The agreement factor barely acts. On `fit` clean normals only 5.0% of leads have a reliability below 1, by
construction near the 95th-percentile cut. A likely reason the factor misses noise, not measured here, is that
every source measures a median beat, which averages much of the noise out before the sources can disagree. On this
evidence, source disagreement is not a useful noise detector.

## The combiner with criteria features

`units_v2_criteria` was trained as 068 was, on the 3,517 `fit` records with region statements (none dropped; 2,636,400
units). The lowest CV loss chose a = 0.25 and L2 = 1e-3, the same as 068. The CV loss was 5.3979, against 068's
5.4011, and the CV weak-label AUROC 0.9231, against 0.9224. The criteria features got small weights: `criteria`
+0.079 and `criteria_x_mil` -0.012, against `textbook` +1.007. On ECGAD, A3 rose by +0.010 [+0.001, +0.020] and A4
by +0.007 [+0.001, +0.015] against `units_v2`, with A2 unchanged. D1 fell to 0.576, because the map is rescaled to
[0, 1] per record.

## Fit-partition checks (descriptive)

Share of `fit` records with a statement that meet the matching criterion (consensus measurements):

| Statement | Records | Criterion | Met |
| --- | ---: | --- | ---: |
| CLBBB | 294 | `lbbb_lat` | 0.97 |
| CRBBB | 297 | `rbbb_v1` | 0.89 |
| 1AVB | 417 | `pr_long` | 0.85 |
| LVH | 966 | any voltage criterion | 0.72 |
| IMI | 892 | `q_path` in II or aVF | 0.63 |
| ASMI | 1,065 | `q_v2v3` | 0.56 |
| LAFB | 842 | `lafb` | 0.52 |
| LNGQT | 59 | `qtc_long` | 0.44 |

Share of the 2,666 `fit` clean normals that meet a criterion in some lead: `p_wide` 25%, `qtc_short` 9.2%, `std`
6.8%, `ptf_v1` 6.7%, `rvh_v1` 5.7%, `ivcd` 5.3%, `pr_long` 4.2%, `ste` 4.2%, every other criterion under 3%. With
own measurements alone, `ivcd` fires in 57% of normals and `q_path` in 27%. This is the expected effect of the
frozen delineation's long QRS, and `criteria_own` should be read with it in mind.

Measurement agreement on `fit` clean normals (median absolute difference; Pearson r):

- QRS duration: 12SL against Uni-G 4 ms (0.87); own against either 20 ms (0.66-0.72).
- PR: 12SL against Uni-G 4 ms (0.88); own 10-11 ms (0.38-0.44).
- QTc: 4 ms (0.94); own 10-12 ms (0.71).
- Axis: 1 degree (0.99); own 5 degrees (0.92).
- P duration: 6 ms between the commercial programs, but r is only 0.58; own r 0.13-0.15.
- Amplitudes agree closely: R(V5) r 0.99-1.00, T(V2) r 1.00.

## Integrity

| Check | Result |
| --- | --- |
| Bar: 061 `qrs_prior` ECGAD maps rescored | per-record A2 within 2.2e-16; mean 0.7419383358406556 against 0.7419383358406554; passed |
| 068 `units_v2` repainted from its saved features and weights | largest map difference 0.0; per-record A2 within 1.1e-16; passed |
| Unit tables of this run against 068's | equal for all 8,388 training and scored records |
| 065 bag logits (`bags_*.npz`) against 066's `mil_bags.npy` | identical on 4,871 scored records |
| Separation | 2,666 normal-reference and 3,517 combiner records, none ECGAD or `evaluation`, all in `fit`; no ECGAD holdout record mapped; passed |
| Measurements | 11,669 records measured; every one had delineated beats; 12SL, Uni-G and ECGdeli rows present for all |

## Deviations and disclosures

- Before the freeze, pooled unit checks (a few percentiles, morphology code counts) were run over the whole PTB-XL+
  feature files, including folds 9 and 10 and ECGAD rows. No label, mark or per-record value was read. The runs
  filtered the files to folds 1-8 and ECGAD development. This is disclosed in the protocol.
- The ECGAD and region-set descriptive that leaves out the all-lead criteria, and the ECGAD criterion prevalences,
  are undeclared. They were computed after the scores to explain them, with the script kept beside the outputs
  (`describe_specific_undeclared.py`). They read ECGAD development marks, are exploratory and choose nothing.
- 12SL and Uni-G measurements go beyond the 061 map contract, as declared. `criteria_own`, which keeps to the
  contract, has an A2 of 0.7251, level with `criteria` (+0.0010 [-0.0285, +0.0282]).
- A bug in the reliability function (zero instead of one for leads without disagreement) was caught by the unit
  tests and fixed before the full run. No scored map used the faulty version.
- The strata and noise CLIs wrote into the copied reference folders under `ref/`. No 061, 066 or 068 file was
  written.

## Runtime

CPU only, at most 8 processes: measurements 104 s (11,669 records), evidence and maps 96 s, combiner (12-setting
CV, final fit, two maps) 283 s, scoring with integrity, six LocBench scorings, strata and noise 675 s. That is about
20 minutes in total. The maps (7 GB, float32) are in `/tmp/ecg069-work/full`.

## Why it did not beat the bar

1. Guideline criteria are written per record, not per lead. The global criteria (QRS duration, PR, QTc) paint every
   lead, and they fire in a large share of abnormal ECGs. In 72% of marked ECGAD records, every lead has evidence,
   so the lead ranking is at chance (A3 0.495).
2. The marks favour QRS (066, 068). The criteria put much of their evidence on ST, T and P: ST depression, T
   inversion, long QTc and wide P. Without a QRS time prior the map loses on time (A4 -0.055). With the prior
   (`criteria_qrs`) it reaches 0.763, but the gain is the prior's.
3. Inside QRS, the hinge leaves most QRS units at zero, and ties fall back on the amplitude tie-break. So A2_qrs
   equals the bar's (0.542 against 0.539).
4. Measured evidence is patient-specific and agrees with the marks: departure leads are marked 61% of the time,
   against 21% for other leads. But coarse marks already include most of these leads, so the evidence does not
   change the ranking A2 rewards.

## Follow-up ideas (for the backlog)

1. Lead-resolve the global criteria: paint QRS-duration, PR and QTc evidence only on the leads where that lead's
   own interval exceeds the threshold (per-lead QRS, PR and QT durations exist in 12SL, Uni-G and ECGdeli), instead
   of all leads.
2. Use criteria evidence as patient-specific weak labels for the combiner, in place of or beside the `REGION_TABLE`
   region. 068 found those labels saturated, and the criteria labels are independent of the region table.
3. Combine the label-free `criteria_qrs` (0.763 without any region-table training) with a learned per-lead QRS
   weighting fitted on `fit` records, and test it on the next marked set.
4. Beat-level criteria (per-beat measurements against the median beat) for focal findings such as PVCs, which a
   median beat removes.
5. Drop the source-disagreement reliability factor. It cannot see noise that the median beat averages out. A
   noise feature measured on raw beats (as in the 066 quality classifier) is the better route.
6. Ask the cardiologist whether a wide P (25% of clean normals by the 120 ms rule) and an R/S > 1 in V1 should
   count, since both fire often in normals.

## Reproduce

```bash
cd /tmp/ecg-wt-069
export PYTHONPATH=/tmp/ecg-wt-069 OPENBLAS_CORETYPE=Haswell OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
.venv/bin/python -u -m scripts.experiments.run_criteria069 measure
.venv/bin/python -u -m scripts.experiments.run_criteria069 maps
.venv/bin/python -u -m scripts.experiments.run_criteria069 combiner
.venv/bin/python -u -m scripts.experiments.run_criteria069 score
```

The `combiner` phase reads 068's work folder `/tmp/ecg068-work/full` (unit features, out-of-fold bags, the
`units_v2` map), and the score phase reads 066's `mil_bags.npy` for the bag check.
