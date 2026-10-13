# Experiment 067 results: holdout confirmation of the frozen localization maps

Run on 13 October 2026 (UTC) under the [protocol](experiment-067-holdout.md), committed as be67817 before any
holdout map existed. The code was committed as c5bbbaf. The holdout maps were built at that commit, and their
manifest was committed as 01aaf0c ([docs/experiment-067-maps-manifest.json](experiment-067-maps-manifest.json))
before any holdout mark was read. The holdout marks were then read once, by one scoring run. That run
completed on its first attempt (`score_attempts.jsonl`: one start, one complete, no crash). Every number below
comes from `outputs/experiment067_holdout_v1/result.json` (SHA-256 `27c9cd69…bfbb5`), its per-record tables, or
`maps_check.json`, unless a row says it comes from the development rehearsal. Intervals are 95% whole-patient
bootstraps with 2,000 draws and seed 61061, and no draws were skipped. Paired differences use the same draws for
both maps.

## Decision

The primary map `units_v2` is **confirmed** against the bar. On the 96 marked holdout records (92 patients),
its A2 is 0.7641 [0.7326, 0.7935] and `qrs_prior`'s is 0.7268 [0.6958, 0.7557]. The paired difference is
+0.0374 [+0.0121, +0.0649]: the whole interval lies above 0. Read per record from the saved tables, `units_v2`
has the higher A2 in 59 of the 96 records.

## Main table (holdout, ECGAD metrics)

| Map | Holdout A2 | A2 minus `qrs_prior` | Development A2 | Shift (holdout minus development) | A3 | A4 | A2_qrs | A2_nonqrs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `units_v2` (primary) | 0.7641 [0.7326, 0.7935] | +0.0374 [+0.0121, +0.0649] | 0.7671 | -0.0030 [-0.0464, +0.0404] | 0.616 | 0.807 | 0.594 | 0.737 |
| `units_learned` | 0.7589 [0.7272, 0.7873] | +0.0322 [+0.0067, +0.0595] | 0.7604 | -0.0014 [-0.0450, +0.0405] | 0.607 | 0.807 | 0.594 | 0.724 |
| `qrs_prior` (bar) | 0.7268 [0.6958, 0.7557] | | 0.7419 | -0.0152 [-0.0590, +0.0304] | 0.441 | 0.817 | 0.527 | 0.690 |
| `factor` | 0.7244 [0.6937, 0.7542] | -0.0024 [-0.0310, +0.0261] | 0.7247 | -0.0003 [-0.0434, +0.0429] | 0.638 | 0.821 | 0.605 | 0.678 |
| `qrs_lexicographic` | 0.7086 [0.6767, 0.7405] | -0.0181 [-0.0374, +0.0009] | 0.7530 | -0.0444 [-0.0880, +0.0027] | 0.569 | 0.752 | 0.558 | 0.663 |
| `textbook_true` (diagnostic) | 0.7671 [0.7376, 0.7948] | +0.0403 [+0.0110, +0.0724] | 0.7582 | +0.0089 [-0.0362, +0.0509] | 0.599 | 0.848 | 0.587 | 0.731 |
| `textbook` | 0.7550 [0.7228, 0.7841] | +0.0282 [+0.0002, +0.0571] | 0.7494 | +0.0056 [-0.0395, +0.0500] | 0.588 | 0.815 | 0.563 | 0.728 |
| `amplitude` | 0.7123 [0.6869, 0.7349] | -0.0145 [-0.0282, +0.0000] | 0.6978 | +0.0145 [-0.0210, +0.0505] | 0.429 | 0.808 | 0.527 | 0.690 |
| `residual056` | 0.6885 [0.6583, 0.7194] | -0.0382 [-0.0710, -0.0010] | 0.6796 | +0.0089 [-0.0335, +0.0521] | 0.570 | 0.733 | 0.604 | 0.693 |
| `U_B` | 0.6485 [0.6173, 0.6789] | -0.0783 [-0.1078, -0.0470] | 0.6315 | +0.0169 [-0.0281, +0.0635] | 0.560 | 0.677 | 0.564 | 0.632 |
| `random` | 0.5009 [0.4998, 0.5020] | -0.2258 [-0.2549, -0.1950] | 0.5001 | +0.0008 [-0.0007, +0.0023] | 0.506 | 0.501 | 0.496 | 0.502 |

The first five rows are the candidates; the rest are references. Records per reading: A2 96, A3 79, A4 95,
A2_qrs 92, A2_nonqrs 95. Holdout totals: 201 records, 190 patients, 96 marked. The QRS strata use the holdout
`qrs_prior` maps (value at least 0.5) as the mask. Inside and outside that mask, `qrs_prior` ranks samples by its
1e-3 amplitude tiebreak alone, so its strata equal `amplitude`'s. The shift intervals combine independent
bootstraps of the two halves (holdout seed 61061, development seed 61062). The development A2 values were
recomputed from this experiment's development maps (see Integrity).

Development values for the same columns, for comparison (from the development rehearsal described under
Deviations; they equal the 061-068 reports): A3 / A4 / A2_qrs / A2_nonqrs were 0.655 / 0.785 / 0.638 / 0.733
for `units_v2`, 0.654 / 0.780 / 0.641 / 0.710 for `units_learned`, 0.457 / 0.851 / 0.539 / 0.661 for
`qrs_prior`, 0.639 / 0.793 / 0.617 / 0.680 for `factor` and 0.575 / 0.801 / 0.591 / 0.651 for
`qrs_lexicographic`.

## Secondary paired comparisons (A2, holdout)

These comparisons are not adjusted for multiplicity and are descriptive.

| Comparison | Difference |
| --- | ---: |
| `units_v2` minus `qrs_prior` (primary) | +0.0374 [+0.0121, +0.0649] |
| `units_learned` minus `qrs_prior` | +0.0322 [+0.0067, +0.0595] |
| `factor` minus `qrs_prior` | -0.0024 [-0.0310, +0.0261] |
| `qrs_lexicographic` minus `qrs_prior` | -0.0181 [-0.0374, +0.0009] |
| `units_v2` minus `textbook` | +0.0091 [-0.0047, +0.0215] |
| `units_v2` minus `units_learned` | +0.0052 [-0.0002, +0.0107] |
| `textbook` minus `qrs_prior` | +0.0282 [+0.0002, +0.0571] |
| `textbook_true` minus `qrs_prior` | +0.0403 [+0.0110, +0.0724] |

On development, `units_v2` beat `textbook` by +0.0178 [+0.0016, +0.0354] and `units_learned` by +0.0068
[+0.0012, +0.0125] (068 report). On the holdout both intervals contain 0.

## Other axes against the bar (holdout, paired with `qrs_prior`)

| Map | A1 | A3 (lead) | A4 (time) | A5 (record) | A6 (top 5%) | A2_qrs | A2_nonqrs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `units_v2` | +0.024 [-0.007, +0.058] | +0.175 [+0.103, +0.249] | -0.009 [-0.041, +0.024] | +0.005 [-0.105, +0.117] | +0.095 [+0.051, +0.141] | +0.067 [+0.023, +0.116] | +0.047 [+0.007, +0.091] |
| `units_learned` | +0.017 [-0.016, +0.053] | +0.166 [+0.093, +0.242] | -0.009 [-0.042, +0.026] | +0.031 [-0.079, +0.138] | +0.107 [+0.062, +0.155] | +0.067 [+0.023, +0.114] | +0.034 [-0.006, +0.078] |
| `factor` | -0.054 [-0.090, -0.018] | +0.197 [+0.131, +0.260] | +0.005 [-0.027, +0.038] | -0.093 [-0.212, +0.029] | +0.137 [+0.081, +0.192] | +0.078 [+0.037, +0.122] | -0.011 [-0.049, +0.028] |
| `qrs_lexicographic` | +0.081 [+0.046, +0.118] | +0.128 [+0.069, +0.190] | -0.065 [-0.094, -0.037] | +0.162 [+0.076, +0.253] | +0.052 [+0.027, +0.080] | +0.031 [+0.000, +0.059] | -0.026 [-0.059, +0.007] |
| `textbook` | +0.017 [-0.027, +0.065] | +0.147 [+0.077, +0.218] | -0.001 [-0.032, +0.031] | -0.058 [-0.166, +0.053] | +0.056 [+0.008, +0.105] | +0.036 [-0.010, +0.085] | +0.039 [+0.003, +0.076] |
| `residual056` | -0.055 [-0.100, -0.008] | +0.129 [+0.021, +0.243] | -0.084 [-0.115, -0.053] | -0.102 [-0.215, +0.016] | -0.001 [-0.056, +0.056] | +0.077 [+0.022, +0.129] | +0.003 [-0.043, +0.052] |
| `U_B` | +0.013 [-0.015, +0.038] | +0.119 [+0.059, +0.178] | -0.140 [-0.181, -0.097] | +0.085 [-0.004, +0.178] | -0.012 [-0.050, +0.029] | +0.037 [-0.002, +0.076] | -0.058 [-0.099, -0.017] |

Holdout point values of `qrs_prior`: A1 0.724, A5 0.663, A6 0.296. The A6 chance level (mean marked share) is
0.144 [0.115, 0.174] on the holdout, against 0.186 on development. `units_v2` reaches A1 0.748, A5 0.668 and A6
0.391. On development, `units_v2` minus `qrs_prior` on A4 was -0.066 [-0.102, -0.027].

## Statement-stratified A2 (descriptive)

Mean holdout A2 per `REGION_TABLE` family (statements at likelihood at least 50) and for marked records with
none of them (`other`). The table shows groups with at least 5 marked records. A record counts in every family
it has. There are no intervals.

| Group (records) | `qrs_prior` | `qrs_lexicographic` | `factor` | `units_learned` | `units_v2` | `textbook` | `textbook_true` | `residual056` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| anterior MI (33) | 0.754 | 0.745 | 0.738 | 0.773 | 0.775 | 0.759 | 0.765 | 0.673 |
| other (26) | 0.688 | 0.667 | 0.685 | 0.722 | 0.725 | 0.708 | 0.721 | 0.690 |
| ischemia (21) | 0.670 | 0.600 | 0.731 | 0.745 | 0.760 | 0.746 | 0.791 | 0.641 |
| LAFB (13) | 0.873 | 0.852 | 0.760 | 0.839 | 0.850 | 0.832 | 0.835 | 0.745 |
| 1AVB (12) | 0.658 | 0.685 | 0.630 | 0.706 | 0.700 | 0.672 | 0.722 | 0.629 |
| CRBBB (12) | 0.856 | 0.836 | 0.742 | 0.802 | 0.821 | 0.800 | 0.817 | 0.753 |
| IMI (10) | 0.842 | 0.845 | 0.764 | 0.826 | 0.831 | 0.841 | 0.879 | 0.752 |
| CLBBB (5) | 0.835 | 0.818 | 0.741 | 0.730 | 0.735 | 0.782 | 0.769 | 0.733 |
| lateral MI (5) | 0.815 | 0.810 | 0.800 | 0.818 | 0.815 | 0.820 | 0.782 | 0.672 |

ST injury (2 marked records) and long QT (1) are below the minimum.

- `units_v2` has its largest gains over `qrs_prior` in ischemia (0.760 against 0.670), 1AVB (0.700 against
  0.658), `other` (0.725 against 0.688) and anterior MI (0.775 against 0.754).
- `qrs_prior` stays best for the conduction and axis findings: LAFB 0.873, CRBBB 0.856 and CLBBB 0.835. The
  development rehearsal shows the same order for CLBBB (0.824 against 0.729 for `units_v2`), CRBBB (0.830
  against 0.736) and LAFB (0.801 against 0.764) (`rehearsal_development/statement_groups.csv`).
- `qrs_lexicographic` has the lowest candidate value of any group in ischemia (0.600). `factor` has the
  lowest candidate value in LAFB (0.760) and CRBBB (0.742).
- `textbook_true` (the true statements painted) is highest in ischemia (0.791) and IMI (0.879), but below
  `qrs_prior` in LAFB, CRBBB and CLBBB.

## Integrity

| Check | Result |
| --- | --- |
| Frozen inputs: 065 weights, 065 result, 062 model and 061 U_B references against 066's `result.json` | all equal; passed |
| Development maps from the wrapper against the scored 061, 062, 066 and 068 development maps (198 records, 11 maps) | identical for all 11 maps (largest difference 0.0); passed |
| Recomputed 065 bag logits of the development records against `bags_ecgad_dev.npz` | largest difference 0.0; passed |
| Lead quality q of the development records against 066's `q.npz` | largest difference 0.0; passed |
| Map rebuild at the code commit | the holdout and development map hashes equal those of the first build |
| Map hashes and record orders at scoring against the committed manifest (22 files) | passed |
| ECGAD label file against `ecgad-benchmark-v1.md` | passed |
| Development A2 recomputed at scoring | per-record differences at most 2.2e-16 against the saved tables; means equal the declared values to 1e-12 for all 11 maps; passed |
| Holdout `random` A2 in [0.48, 0.52] | 0.5009; passed |
| Scoring attempts | 1 (complete) |

## Deviations and disclosures

- The maps phase ran twice. The first build ran before the code was committed. It was rebuilt at the code
  commit so that the manifest records that commit, and every map hash was identical in the two builds.
- An undeclared rehearsal ran before the single scoring run. It ran the exact scoring code on the development
  half (development maps and development marks, with the holdout record list replaced by the development one),
  to catch crashes before the holdout marks were read. It reproduced 068's development primary
  (+0.0252 [-0.0130, +0.0647]) and the 061-068 development strata. Its outputs are in
  `outputs/experiment067_holdout_v1/rehearsal_development/`. Its files named `*holdout*` hold development
  records. The development statement-group means quoted above come from it. No holdout data entered it.
- The protocol says the development A2 in the shift is recomputed from the saved development tables. It was
  recomputed from this experiment's development maps and development marks instead, and those values match
  the saved tables to at most 2.2e-16 per record.
- `device_setup` and `load_networks` repeat about 15 lines of the 066 runner, because scripts are not imported.
- `textbook_true` read the holdout PTB-XL statements in the map stage, and the statement groups read them in
  the scoring run. Both were declared. The holdout marks were read only by the one scoring run. Everything after
  it, including the per-record win counts, comes from its saved tables.
- No ECGAD record entered any fitted quantity: nothing was fitted.

## Runtime

| Stage | Time |
| --- | ---: |
| Maps phase, first build (development check plus holdout, GPU0 for the MIL evidence) | 33 s |
| Maps phase, rebuild at the code commit | 26 s (MIL on GPU 1.4-1.9 s per half) |
| Development rehearsal of the scoring code | 239 s |
| Single holdout scoring run (CPU, 8 threads) | 237 s |

Maps live in `/tmp/ecg067-work/{development,holdout}/` (11 float32 files of 48 MB per half) and are not kept in
Git.

## What the holdout says about the programme (061-068)

- The top of the development ranking held. `units_v2` and `units_learned` were the two best candidates on
  development (0.7671 and 0.7604), and they are the two best on the holdout (0.7641 and 0.7589). Each moved by
  less than 0.004. The rest of the ranking did not hold. `qrs_lexicographic` was third on development (0.7530)
  and is last of the candidates on the holdout (0.7086, shift -0.0444 [-0.0880, +0.0027]). The bar
  `qrs_prior` fell from 0.7419 to 0.7268. None of the shift intervals excludes 0.
- The confirmation comes from two moves: `units_v2` held its development A2, and the bar fell. Its margin over
  `qrs_prior` grew from +0.0252 on development to +0.0374 on the holdout. The holdout marks are sparser: the
  mean marked share is 0.144 against 0.186. On the holdout, `qrs_prior`'s time AUROC is 0.817, against 0.851 on
  development.
- What the holdout confirms is diagnose-then-paint with a learned component weighting. `textbook` alone also
  beats the bar on the holdout (+0.0282 [+0.0002, +0.0571]). On the holdout, `units_v2` is not separable from
  `textbook` (+0.0091 [-0.0047, +0.0215]) or from `units_learned` (+0.0052 [-0.0002, +0.0107]). So the MIL lead
  evidence and the 068 features add little that this sample can resolve. Painting the true statements
  (`textbook_true`) reaches 0.7671, close to `units_v2`'s 0.7641 (no paired interval was declared for this
  pair), which suggests that the predicted diagnoses cost little on this benchmark.
- Lead axis: held. Every candidate other than the bar, and both textbook maps, beat `qrs_prior` on A3 by
  +0.13 to +0.20 on the holdout (`units_v2` +0.175 [+0.103, +0.249]). On development the range was +0.12 to
  +0.20.
- Within-QRS axis: held. `units_v2` A2_qrs beats the bar by +0.067 [+0.023, +0.116] (development +0.100),
  `factor` by +0.078 and `residual056` by +0.077.
- Outside QRS: `units_v2` keeps a gain of +0.047 [+0.007, +0.091] (development +0.071 [+0.020, +0.123]).
- Time axis: the development deficit of the learned maps did not reproduce. `units_v2` minus `qrs_prior` on
  A4 is -0.009 [-0.041, +0.024] on the holdout, against -0.066 [-0.102, -0.027] on development.
  `qrs_lexicographic` (-0.065), `residual056` (-0.084) and `U_B` (-0.140) still lose on time.
- Label-free maps stay below the bar on A2: `residual056` -0.0382 and `U_B` -0.0783. The 062 normative channel
  keeps its lead and within-QRS information (`qrs_lexicographic` A3 +0.128, A2_qrs +0.031). It loses on time
  (A4 -0.065 [-0.094, -0.037]) and is no better outside QRS (A2_nonqrs -0.026 [-0.059, +0.007]).
- By statement, the ECGAD marks for conduction and axis findings (LAFB, CRBBB, CLBBB) still favour the
  disease-blind QRS indicator over every textbook-based map, on both halves. Asking the cardiologist about this
  (066 follow-up 4) is still the open question.
- The holdout half is now spent. It cannot select or tune any later localization method.

## Follow-up ideas (for the backlog)

1. A new untouched localization test set is needed for any method after 068, for example a second annotator
   pass or new cardiologist marks on SPH or local records. The ECGAD holdout cannot be reused.
2. Separate textbook painting from the combiner on a larger marked set: on 96 records, `units_v2` minus
   `textbook` (+0.009) cannot be resolved.
3. A conduction-aware component weight: for LAFB, CRBBB and CLBBB, the QRS indicator beats every textbook map
   on both halves. Ask the cardiologist whether ECGAD marks for these findings cover all leads' QRS complexes,
   then build this from `fit` data only.
