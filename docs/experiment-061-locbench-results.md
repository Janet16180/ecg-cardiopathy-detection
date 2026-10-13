# Experiment 061 results: LocBench v1 and baselines

Run on 12 October 2026 under the [frozen protocol](experiment-061-locbench.md) (commit 351d9db) and its
Amendment 1 (commit eadf3ba, written before any score). Code commit 301aaca, no tracked modifications. All
numbers come from `outputs/experiment061_locbench_v1/result.json`. Intervals are 95% whole-patient bootstraps,
2,000 draws, seed 61061; no draw was skipped for any metric. Every integrity check passed, so the benchmark is
adopted. The bar for Experiments 062-066 is `qrs_prior`.

## The bar

`qrs_prior` (1 inside the Experiment 059 hybrid QRS intervals, 0 elsewhere, plus 1e-3 times amplitude) has the
highest A2 point estimate. Frozen:

| Bar metric | `qrs_prior` |
| --- | ---: |
| A2, ECGAD within-record point AUROC (92 records, 90 patients) | 0.7419 [0.7082, 0.7743] |
| B1 macro, region AUROC over 9 families (1,047 records, 944 patients) | 0.7283 [0.7212, 0.7356] |
| C1, noisy-lead AUROC (299 records, 293 patients) | 0.3529 [0.3248, 0.3843] |
| D1, detection record AUROC (4,374 records, 2,974 positive, 3,838 patients) | 0.6321 [0.6135, 0.6501] |

Paired differences against the bar on the same draws (other baseline minus `qrs_prior`):

| Baseline | A2 | B1 macro | C1 | D1 |
| --- | ---: | ---: | ---: | ---: |
| random | -0.2418 [-0.2743, -0.2081] | -0.2283 [-0.2356, -0.2210] | +0.1493 [+0.1102, +0.1865] | -0.1357 [-0.1619, -0.1103] |
| amplitude | -0.0442 [-0.0589, -0.0294] | -0.0713 [-0.0742, -0.0684] | -0.0198 [-0.0366, -0.0024] | +0.0036 [+0.0013, +0.0062] |
| U_B | -0.1104 [-0.1475, -0.0744] | -0.0607 [-0.0715, -0.0500] | +0.3203 [+0.2831, +0.3542] | +0.1300 [+0.1081, +0.1507] |
| residual056 | -0.0623 [-0.0922, -0.0327] | -0.0915 [-0.1005, -0.0824] | +0.4476 [+0.4050, +0.4865] | -0.0387 [-0.0667, -0.0110] |

## ECGAD development (198 records, 94 marked)

| Metric | random | amplitude | qrs_prior | U_B | residual056 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A1 pooled point AUROC | 0.4999 [0.4994, 0.5004] | 0.6789 [0.6277, 0.7290] | 0.6953 [0.6429, 0.7460] | 0.7188 [0.6594, 0.7744] | 0.6659 [0.6244, 0.7070] |
| A2 within-record point AUROC (primary) | 0.5001 [0.4991, 0.5012] | 0.6978 [0.6705, 0.7234] | 0.7419 [0.7082, 0.7743] | 0.6315 [0.5968, 0.6644] | 0.6796 [0.6504, 0.7096] |
| A3 lead AUROC | 0.4891 [0.4293, 0.5490] | 0.4389 [0.3761, 0.5046] | 0.4565 [0.3916, 0.5223] | 0.4603 [0.3982, 0.5232] | 0.5909 [0.5221, 0.6598] |
| A4 time AUROC | 0.4993 [0.4976, 0.5010] | 0.8170 [0.7854, 0.8447] | 0.8506 [0.8183, 0.8781] | 0.6462 [0.6051, 0.6864] | 0.7146 [0.6838, 0.7436] |
| A5 record AUROC | 0.5213 [0.4355, 0.5992] | 0.7206 [0.6415, 0.7910] | 0.7128 [0.6338, 0.7823] | 0.7878 [0.7186, 0.8482] | 0.6478 [0.5634, 0.7200] |
| A6 precision at top 5% | 0.1859 [0.1495, 0.2265] | 0.3394 [0.2776, 0.4014] | 0.3572 [0.2960, 0.4199] | 0.3257 [0.2622, 0.3923] | 0.3393 [0.2794, 0.4012] |
| A6 chance (mean marked share) | 0.1861 [0.1494, 0.2270] | same | same | same | same |

Populations: A1 and A5 198 records (190 patients); A2 92 records (two marked records have every scored
lead-sample marked); A3 64 records with 1-11 marked leads; A4 91 records (three have a mark at every scored
sample); A6 94 records (92 patients). The A1 point estimate is the exact Mann-Whitney value; its bootstrap uses
the 65,536-bin rank grid, which matched the exact value within 3.0e-8 for every baseline (tolerance 1e-4).

## Region set (evaluation partition, 1,081 records)

Families with at least 20 records: 1AVB 96, CLBBB 94, CRBBB 39, IMI 240, LAFB 216, LNGQT 24, ST injury 28,
anterior MI 226, ischemia 84. WPW (15) and lateral MI (19) are below 20 and enter no family row or macro.

### B1, within-record region AUROC

| Family | random | amplitude | qrs_prior | U_B | residual056 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1AVB | 0.4997 [0.4990, 0.5003] | 0.4182 [0.3999, 0.4361] | 0.3818 [0.3623, 0.4012] | 0.6536 [0.6302, 0.6769] | 0.4692 [0.4561, 0.4828] |
| CLBBB | 0.4997 [0.4991, 0.5003] | 0.7918 [0.7823, 0.8010] | 0.9701 [0.9650, 0.9749] | 0.8514 [0.8359, 0.8662] | 0.7233 [0.7047, 0.7400] |
| CRBBB | 0.5007 [0.4990, 0.5024] | 0.7981 [0.7821, 0.8137] | 0.9209 [0.9129, 0.9284] | 0.8552 [0.8304, 0.8778] | 0.6499 [0.6216, 0.6765] |
| IMI | 0.5002 [0.4994, 0.5010] | 0.7048 [0.6952, 0.7147] | 0.8958 [0.8911, 0.9004] | 0.6744 [0.6510, 0.6981] | 0.8041 [0.7929, 0.8148] |
| LAFB | 0.4999 [0.4993, 0.5005] | 0.7419 [0.7352, 0.7489] | 0.9059 [0.9014, 0.9105] | 0.7767 [0.7599, 0.7931] | 0.7548 [0.7455, 0.7639] |
| LNGQT | 0.5008 [0.4995, 0.5022] | 0.5931 [0.5606, 0.6244] | 0.5259 [0.4982, 0.5535] | 0.5454 [0.4915, 0.5988] | 0.5543 [0.5324, 0.5776] |
| ST injury | 0.4996 [0.4976, 0.5014] | 0.5412 [0.4853, 0.5997] | 0.5238 [0.4733, 0.5788] | 0.4764 [0.4345, 0.5172] | 0.5862 [0.5626, 0.6088] |
| anterior MI | 0.4993 [0.4985, 0.5001] | 0.7499 [0.7417, 0.7581] | 0.9065 [0.9014, 0.9114] | 0.6370 [0.6029, 0.6717] | 0.6080 [0.5901, 0.6260] |
| ischemia | 0.5005 [0.4998, 0.5011] | 0.5743 [0.5557, 0.5924] | 0.5241 [0.5065, 0.5403] | 0.5386 [0.5159, 0.5618] | 0.5818 [0.5622, 0.6010] |
| macro | 0.5000 [0.4996, 0.5004] | 0.6570 [0.6492, 0.6654] | 0.7283 [0.7212, 0.7356] | 0.6676 [0.6573, 0.6780] | 0.6368 [0.6304, 0.6433] |

### B2, lead AUROC (families whose lead set is not all 12 leads)

| Family | random | amplitude | qrs_prior | U_B | residual056 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CRBBB | 0.4728 [0.4187, 0.5273] | 0.7224 [0.6809, 0.7636] | 0.7284 [0.6859, 0.7680] | 0.7019 [0.6683, 0.7359] | 0.4263 [0.3849, 0.4671] |
| IMI | 0.4965 [0.4718, 0.5210] | 0.2725 [0.2495, 0.2975] | 0.2894 [0.2642, 0.3153] | 0.5289 [0.4935, 0.5633] | 0.8037 [0.7736, 0.8328] |
| LAFB | 0.5040 [0.4823, 0.5269] | 0.2411 [0.2218, 0.2638] | 0.2511 [0.2285, 0.2756] | 0.5023 [0.4624, 0.5367] | 0.6613 [0.6254, 0.6955] |
| ST injury | 0.4812 [0.4281, 0.5414] | 0.7855 [0.7190, 0.8456] | 0.8199 [0.7546, 0.8793] | 0.5510 [0.4483, 0.6507] | 0.3229 [0.2613, 0.3865] |
| anterior MI | 0.5096 [0.4844, 0.5346] | 0.8219 [0.7951, 0.8467] | 0.8151 [0.7868, 0.8407] | 0.5508 [0.5078, 0.5912] | 0.3378 [0.3021, 0.3730] |
| ischemia | 0.4915 [0.4520, 0.5312] | 0.5237 [0.4685, 0.5759] | 0.5506 [0.4936, 0.6029] | 0.4819 [0.4319, 0.5332] | 0.4272 [0.3785, 0.4742] |
| macro | 0.4926 [0.4760, 0.5097] | 0.5612 [0.5438, 0.5777] | 0.5757 [0.5579, 0.5935] | 0.5528 [0.5302, 0.5739] | 0.4965 [0.4779, 0.5145] |

### B3, component AUROC (time score against expected components)

| Family | random | amplitude | qrs_prior | U_B | residual056 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1AVB | 0.4996 [0.4979, 0.5015] | 0.3540 [0.3274, 0.3803] | 0.3356 [0.3079, 0.3625] | 0.6630 [0.6259, 0.6970] | 0.4148 [0.3992, 0.4318] |
| CLBBB | 0.5015 [0.4991, 0.5040] | 0.8302 [0.8192, 0.8411] | 0.9725 [0.9674, 0.9772] | 0.8691 [0.8517, 0.8846] | 0.7317 [0.7066, 0.7546] |
| CRBBB | 0.5021 [0.4983, 0.5062] | 0.8745 [0.8594, 0.8891] | 0.9691 [0.9623, 0.9755] | 0.9024 [0.8816, 0.9241] | 0.6838 [0.6519, 0.7150] |
| IMI | 0.5003 [0.4989, 0.5018] | 0.8205 [0.8131, 0.8274] | 0.9660 [0.9623, 0.9697] | 0.6222 [0.5871, 0.6563] | 0.6863 [0.6743, 0.6992] |
| LAFB | 0.4990 [0.4974, 0.5005] | 0.8235 [0.8168, 0.8301] | 0.9530 [0.9487, 0.9574] | 0.7351 [0.7070, 0.7631] | 0.6675 [0.6558, 0.6808] |
| LNGQT | 0.5008 [0.4964, 0.5047] | 0.6902 [0.6508, 0.7273] | 0.6238 [0.5901, 0.6554] | 0.5796 [0.5151, 0.6443] | 0.6256 [0.5928, 0.6586] |
| ST injury | 0.4979 [0.4922, 0.5030] | 0.5924 [0.5429, 0.6376] | 0.5756 [0.5299, 0.6195] | 0.4480 [0.3934, 0.5083] | 0.7041 [0.6886, 0.7201] |
| anterior MI | 0.5002 [0.4987, 0.5016] | 0.7798 [0.7711, 0.7880] | 0.9516 [0.9470, 0.9559] | 0.6218 [0.5855, 0.6595] | 0.6913 [0.6771, 0.7051] |
| ischemia | 0.5006 [0.4988, 0.5025] | 0.6829 [0.6589, 0.7064] | 0.6176 [0.5942, 0.6387] | 0.5567 [0.5288, 0.5846] | 0.6873 [0.6672, 0.7069] |
| macro | 0.5002 [0.4992, 0.5013] | 0.7165 [0.7082, 0.7248] | 0.7739 [0.7666, 0.7812] | 0.6664 [0.6525, 0.6798] | 0.6547 [0.6480, 0.6620] |

### Strict variant (no other diagnostic statement)

Families with at least 20 strict records: CLBBB 75, IMI 147, LAFB 114, anterior MI 105, ischemia 38
(479 records, 443 patients; B2 macro 404 records without CLBBB).

| Macro | random | amplitude | qrs_prior | U_B | residual056 |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1 strict | 0.4998 [0.4994, 0.5002] | 0.7041 [0.6987, 0.7101] | 0.8375 [0.8331, 0.8421] | 0.6924 [0.6768, 0.7064] | 0.6994 [0.6895, 0.7100] |
| B2 strict | 0.4945 [0.4745, 0.5144] | 0.4608 [0.4336, 0.4879] | 0.4775 [0.4496, 0.5047] | 0.5167 [0.4867, 0.5455] | 0.5698 [0.5423, 0.5993] |
| B3 strict | 0.5006 [0.4996, 0.5017] | 0.7859 [0.7789, 0.7928] | 0.8918 [0.8865, 0.8974] | 0.6678 [0.6491, 0.6860] | 0.6954 [0.6854, 0.7052] |

Per-family strict values are in `result.json` (`B1/<family>/strict` and so on). The strict B2 values for IMI
are 0.2735 (amplitude), 0.2911 (qrs_prior), 0.5412 (U_B) and 0.8425 [0.8146, 0.8697] (residual056).

## Noise and detection

| Metric | random | amplitude | qrs_prior | U_B | residual056 |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 noisy-lead AUROC (299 records) | 0.5023 [0.4783, 0.5255] | 0.3331 [0.3054, 0.3637] | 0.3529 [0.3248, 0.3843] | 0.6732 [0.6439, 0.7016] | 0.8006 [0.7776, 0.8213] |
| C2 noise set vs clean normals (299 vs 1,400) | 0.5015 [0.4647, 0.5368] | 0.4831 [0.4466, 0.5192] | 0.4788 [0.4404, 0.5150] | 0.7136 [0.6817, 0.7413] | 0.6459 [0.6103, 0.6775] |
| D1 detection (4,374 records) | 0.4964 [0.4774, 0.5148] | 0.6357 [0.6170, 0.6534] | 0.6321 [0.6135, 0.6501] | 0.7621 [0.7474, 0.7775] | 0.5934 [0.5753, 0.6111] |

## Integrity

| Check | Result |
| --- | --- |
| 042 reproduction: U_B rebuilt from the 5,872-record (5,537-patient) reference, 63,479 beats, on the first 20 saved fold-9 records | geometry identical, maximum relative score difference 0, maximum time difference 0; passed |
| 056 reproduction: INCART I01, I02, I03 rescored with the saved reference (hash matches the 056 receipt) | 2,795, 2,642 and 2,454 beats, same beats as saved; maximum relative difference 4.6e-14 (aligned residual) and 3.1e-16 (U_B fixed); passed |
| `random` A2 in [0.48, 0.52] | 0.5001; passed |
| Exact float64 recovery from the cache equals `wfdb.rdsamp` | 20 of 20 records equal; passed |
| ECGAD labels and match table equal the hashes in `ecgad-benchmark-v1.md` | passed (labels `8b73c64b...`, match `fa3e09c9...`) |
| Split and partition hashes recorded | `ecgad-split-v1.json` `4de33156...`, `locbench-partition-v1.json` `eaedad07...` |
| A1 binned against exact | largest absolute difference 3.0e-8 |

ECGAD sanity reading (descriptive, development marks only). In the 9 marked development records with IMI at
likelihood at least 50, 34.3% of marked lead-samples lie in II, III and aVF (3 of 12 leads, 25%) and 23.5% in
V1-V4. In the 21 marked records with ASMI, 31.5% lie in V1-V4 (4 of 12 leads, 33.3%) and 21.7% in II, III and
aVF.

## Counts and exclusions

- ECGAD development: 198 records, 190 patients, 94 marked; holdout labels were not read and no holdout or
  duplicate record was scored.
- Region set: 1,081 candidates, none excluded (the three `evaluation` records without an ECGdeli file are not
  region candidates; no candidate lacked an expected component in samples 100-4899). The 1,081 ECGdeli files
  hash to `086ca1ce...`.
- Noise set 299 records (293 patients); clean normals 1,400 (1,343 patients); detection 4,374 (2,974 positive,
  3,838 patients). Scored records in all: 4,871.
- U_B refit: 2,666 `fit` clean normals (2,566 patients), 28,664 kept beats, none without two beats.
- No record needed a constant map: every record had at least two kept beats for U_B, three complete beats for
  `residual056`, and at least one hybrid QRS interval for `qrs_prior`.
- Runtime: 336 seconds in all, of which 156 seconds built the five maps of 4,871 records with 8 processes.

## Interpretation

- A pure wave prior is the strongest baseline on the primary metric. `qrs_prior` knows nothing about disease,
  yet reaches A2 0.7419, above U_B by 0.1104 and above `residual056` by 0.0623, both with paired intervals
  excluding zero. A descriptive reading of its maps (`descriptives.json`, from
  `scripts/reports/describe_locbench061.py`) shows why: the hybrid QRS covers 16.1% of the scored ECGAD
  samples but holds 35.9% of the marked lead-samples. The cardiologist marks concentrate on QRS complexes, so
  much of the ECGAD within-record signal is "where the QRS is".
- Lead information is weak for every baseline on ECGAD. A3 is below 0.5 for amplitude (0.4389), `qrs_prior`
  (0.4565) and U_B (0.4603); only `residual056` is above chance, 0.5909 [0.5221, 0.6598].
- The region test is dominated by the same prior. `qrs_prior` B1 is 0.896-0.970 for the five QRS families
  (CLBBB, CRBBB, IMI, LAFB, anterior MI). Its B1 lead-and-component region therefore mostly measures
  agreement between two delineators on the QRS, and its B2 lead AUROC follows amplitude: 0.2894 for IMI and
  0.2511 for LAFB, below chance, consistent with precordial leads having the largest amplitude (median 95th-percentile
  amplitude of clean normals 0.43-0.45 mV in V2-V3 against 0.11-0.14 mV in III, aVL and aVF). For the
  non-QRS families (1AVB, LNGQT, ST injury, ischemia) `qrs_prior` B1 is 0.38-0.53.
- `residual056` carries the only lead information for inferior findings: B2 IMI 0.8037 [0.7736, 0.8328], LAFB
  0.6613, against 0.24-0.29 for amplitude and `qrs_prior`. It points away from the expected leads for anterior
  MI (B2 0.3378) and ST injury (0.3229).
- Maps that compare with a reference are drawn to noise. C1 is 0.8006 for `residual056` and 0.6732 for U_B,
  and both separate noisy from clean normal records (C2 0.6459 and 0.7136). Amplitude and `qrs_prior` sit
  below 0.5 on C1 (0.33-0.35); in the noise set 61.2% of records name limb leads only, and limb leads have
  smaller amplitude, so this is more likely an amplitude effect than noise robustness.
- U_B detects best (D1 0.7621, A5 0.7878, A1 0.7188) but localizes worst of the non-random maps on A2. A1
  rewards detection: U_B has the highest A1 and the lowest non-random A2, which supports A2 as the primary.
- For Experiments 062-066, beating the bar means beating a disease-blind QRS indicator on A2 (0.7419) and on
  B1 macro (0.7283), while staying near 0.5 on C1 (the bar's own C1, 0.3529, is below chance for the reason
  above). A method that only learns where the QRS is will look good on A2, A4 and the QRS families of B1, and
  should be read against B2, A3 and the non-QRS families.

## Outputs and use by later experiments

`outputs/experiment061_locbench_v1/`: `result.json`, `run.log`, `map_build.csv`, `ub_references.pkl`,
`descriptives.json`,
`tables/<method>/` (per-record CSVs, A1 histograms and the float32 ECGAD maps that were scored) and
`maps/<method>.npy` (float16 maps of the 1,578 ECGAD development, region and noise records, with
`.ids.json`). The U_B float16 file first held 23,587 infinite samples in 38 records because U_B reaches
6.0e7, beyond float16; after the run it was rewritten from the scored float32 maps with values clipped to
65,504 (`locbench.save_maps` now clips). This file is for display only and entered no score.

A later method scores with
`PYTHONPATH=. .venv/bin/python -m scripts.experiments.run_locbench_score --maps <npy> --ids <json> --name <name>
--out <dir> --compare outputs/experiment061_locbench_v1/tables/qrs_prior`, which writes every metric and the
paired differences on the same draws. Rescoring the `qrs_prior` maps with this command reproduced every
metric and interval, with paired differences against the saved tables below 4e-16.

## Limitations

- ECGdeli boundaries are automatic, so the region targets are approximate, and the QRS families partly reward
  any QRS detector.
- The region truth is a statement-level expectation, not a mark. Old infarcts may show nothing in the expected
  leads.
- The ECGAD development half has 94 marked records from 92 patients; A3 rests on 64 records.
