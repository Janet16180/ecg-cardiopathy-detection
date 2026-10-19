# Experiment 042 results: CPU screening improvement transfers to SPH

18 October 2026. The fixed PTB-XL-trained boosting model passed all four external development
gates at SPH. AUROC was **0.907938**, versus **0.867394** for the matched logistic model;
paired difference **+0.040545 [0.035482, 0.045825]**. At separately calibrated local-normal
thresholds, sensitivity increased from 0.562500 to 0.668248, with normal referral rates 0.047153
and 0.048613. No classifier was fitted or updated using SPH.

The independent saved-output audit of Experiments 041 and 042 passed **462 checks with zero
mismatches**, including thresholds, metrics, intervals, decisions, patient roles, duplicates and
artifact identities. This is verification of the executed calculation; SPH is a reused development
hospital, and neither result supplies clinical confirmation or a final test.

## Cohort and patient roles

All 25,770 SPH records were processed through the first 10 seconds in canonical lead order,
500 Hz and mV. Existing exact-window duplicate rules retained 25,434 unique records and 143
representatives, dropping 143 extra copies and 50 conflicting copies. No quality-based exclusion
was applied to evaluation. There were 98 technical-quality flags across the complete source;
38 labeled evaluation records carried one. Extraction returned 25,748 `ok` and 22 `sin_latidos`
statuses; no-beat measurements remained missing and were retained.

The patient role hash was fixed before scoring. Only existing defined primary labels entered the
comparison. Across all source records, 4,605 had undefined primary labels.

| Role after duplicate/label rules | ECGs | Patients | Positive | Target-negative |
|---|---:|---:|---:|---:|
| Calibration | 10,574 | 10,248 | 3,606 | 6,968 |
| Evaluation | 10,434 | 10,116 | 3,584 | 6,850 |

Patients did not cross roles. Only the 6,968 calibration negatives determined the two local
thresholds. Each referred 348 of those normals (4.9943%), strictly above the 349th-highest normal
score. Logistic threshold was 0.7130950117578835; boosting threshold 0.7821236835503446.

## Frozen evaluation

| Endpoint | Logistic | Boosting |
|---|---:|---:|
| AUROC, patient-bootstrap 95% interval | 0.867394 [0.859807, 0.875345] | 0.907938 [0.901564, 0.914280] |
| Average precision | 0.815103 | 0.875520 |
| Sensitivity at local-normal threshold | 0.562500 [0.545659, 0.578876] | 0.668248 [0.652496, 0.683541] |
| Normal referral rate | 0.047153 [0.042151, 0.052228] | 0.048613 [0.043446, 0.053808] |
| Brier score | 0.136122 | 0.116615 |
| Positive ECGs missed | 1,568 | 1,189 |
| Target-negative ECGs referred | 323 | 333 |

The paired sensitivity difference was +0.105748 [0.093659, 0.118539]. Boosting caught a net 379
additional label-positive ECGs while referring a net 10 additional target-negative ECGs under these
two independently calibrated rules. Both contrasts used 2,000 whole-patient draws, seed 20261018,
with zero skipped draws; fitted classifiers and thresholds stayed fixed. This interval does not
include uncertainty from retraining or selecting a different calibration sample.

Every frozen gate passed: AUROC gain at least 0.01, positive lower bound, sensitivity difference
at least −0.02, and boosting normal referral at most 8%. Decision: `support_external_development`.
Boosting still missed 1,189/3,584 positive annotations at this operating point. The result supports
continued development, not deployment.

Descriptive superclass misses, allowing overlap, were MI 11→10 of 118, STTC 1,102→779 of 2,521,
CD 468→409 of 1,166, and HYP 22→15 of 104. These counts retain existing labels; they do not define
new clinical conditions or establish subgroup efficacy.

Transporting the original PTB thresholds instead of local calibration gave normal referral rates
2.9343% (logistic) and 6.4380% (boosting), with sensitivities 50.5301% and 70.3962%. The higher
boosting sensitivity at the transported threshold comes with a higher normal referral rate.
The primary comparison uses the prospectively specified local-normal calibration for both arms.

## Qualification, execution and independent verification

The reusable measurement port preserved the notebook extraction arithmetic. Before SPH preparation,
32 predetermined PTB fold-1 records reproduced all 130 cached numeric measurements exactly after
the original six-significant-digit serialization: maximum absolute difference 0.0, identical
missingness, extraction status and plausibility fields. This is an algorithm-reproduction check,
not expert verification of waveform delineation.

The real 64-record read/extract/quality/hash profile took 2.703 seconds. Its conservative full
preparation forecast, 2,292.223 seconds and about 102 MB, passed the frozen limits. Full preparation
took 1,193.384 seconds and evaluation 37.910 seconds on one numerical thread. Together with
qualification and profiling, measured stage time was 1,235.337 seconds. Output occupied 36,322,026
bytes before the separate audit. There were no GPU runs, classifier updates, data downloads or
failed 042 stages. The independent audit added 26.992 seconds for both experiments.

Fifteen artificial transfer tests, exact notebook-arithmetic comparison and independent source
review passed before execution. The final integrated repository suite passed 1,328 tests,
with six skips, after restoring the upstream source already pinned by CI. A further focused audit
regression passed; Ruff and package build passed. The initial broad test attempt's missing-source
failures are preserved in local logs; no frozen experiment source was changed to repair them.

The independent auditor used saved masks/labels/probabilities and declared source/metadata hashes,
without model inference, fitting or waveform reads. It reconstructed strict normal thresholds,
all arm and paired metrics/intervals, superclass misses, local/transport decisions, exact-window
duplicate retention and pre-inference patient roles. It did not independently regenerate waveform
truth, rerun training or establish the clinical validity of annotations.

## Interpretation and limits

The improvement survives a different hospital using a small CPU classifier and existing raw-signal
measurements. Its task remains annotation screening. SPH negatives are code 1 alone, whereas
PTB negatives have NORM as their only diagnostic superclass; the two label definitions and cohorts
are not identical. No athlete-specific or student-population label was created, and no age analysis
was performed. SPH had already informed earlier research. The historical xECG pipeline uses other
features and training cohorts and was not a matched comparator here; this result does not replace it.
Challenge and EchoNext closed tests remain unopened.

The separately frozen [lead-grounding study](experiment-043-lead-grounding.md) tests a conditional
benchmark task with public generated annotations. It cannot turn these binary scores into verified
disease-location masks. Follow-ups belong in the ranked backlog; no additional fit follows this result.

## Local evidence

Source was committed as `6ce0382`; qualification/profile/preparation used that revision. Evaluation
used `f53853d`, with the same bound scientific sources. Protocol hashes and all environment/input
bindings are in each stage's `completion.json`. The two classifiers are exactly the saved 041v2
artifacts; their bytes stayed unchanged.

| Artifact | SHA-256 |
|---|---|
| `outputs/experiment042_measurement_transfer_v1/evaluation/results.json` | `838e93491d0a80551b86e1400af08ddbe770ffbd07f197adabf8ef8599de4179` |
| `evaluation/predictions.csv` under that directory | `4f0ff3b189457500e0cb3476ace5aa4ccb7a93ee91bf6c47ea53cee7035e4189` |
| `preparation/measurements.csv` under that directory | `73567e3038611c46c5d74efc05614772af824826045ed5fcb7e99acdb049f019` |
| `evaluation/completion.json` under that directory | `d17324c743040772568401229e79bb4a89667d74081ddea391e483a36c98ae23` |
| `outputs/measurement_screening_arithmetic_audit_v1/audit.json` | `965a3b9ba742a19c723ddfaee79a1204c81dbf7b8a16e177df7bba49eb4e46dc` |

Payloads and record/patient identities stay local. The four exclusive stages can be reproduced
with `uv run --no-sync python -m scripts.experiments.run_measurement_screening042`
and `qualify`, `profile`, `prepare`, `evaluate`, specifying a new `--output` directory. Preserve the
completed run and use the same saved 041v2 model artifact. See the [protocol](experiment-042-measurement-transfer.md)
for exact role, quality, resource and endpoint definitions.
