# Corrective evaluation: the strict-age advantage was not established

28 October 2026. The prespecified strict 18–25 training cohort and its source/evidence/target-matched
general-age control each contained 770 hospital ECGs. Their fixed 50:50 HGB/xECG scores achieved
SPH sensitivity 70.40% and 69.92%, respectively, after separate calibration using the same 5% normal
referral budget. The paired difference was **+0.474 percentage points, 95% interval
[−0.722, +1.638] points**. Neither the required two-point gain nor positive lower bound was achieved.
The age advantage was not established. No pipeline was promoted and no model setting was changed
after evaluation. This does not estimate retraining variability or prove age has no effect.

The committed [protocol](pr74-corrective-evaluation-protocol-2026-10-28.md) was amended to the strict
comparison before new extraction, fitting or comparative scores. The broader 25k age-preferred and
all-age selections remained secondary. They are predominantly older hospital cohorts, not student
cohorts. The exact aggregate, including all 15 models, both sites and ordered threshold sweeps, is
[available without local patient data or weights](../reports/pr74-corrective-evaluation-2026-10-28/aggregate.json).

## Primary comparison and denominators

| Training cohort, fixed blend | AUROC | Sensitivity | TP | FN | FP | TN | Normal referral | All referred |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Strict 18–25, 770 ECGs | 0.926219 | 70.40% | 2,523 | 1,061 | 330 | 6,520 | 4.82% | 27.34% |
| Matched general ages, 770 ECGs | 0.914614 | 69.92% | 2,506 | 1,078 | 335 | 6,515 | 4.89% | 27.23% |

The evaluation denominator is identical: 10,434 SPH ECGs from 10,116 patients, with 3,584 proxy-positive
and 6,850 proxy-negative ECGs. Strict training caught a net 17 more positive ECGs and referred five
fewer negative ECGs. These are record counts, not 17 distinct newly identified patients. AUROC is
a descriptive secondary ranking metric and is not the fraction correctly diagnosed.

Each model's SPH threshold used only 6,968 calibration negatives; 348 were strictly above threshold.
The five-percent budget concerns normal calibration ECGs, not all ECGs in a population. Evaluation
normal referral need not equal five percent. Every matrix uses `score > threshold`; equal scores
are not referred. The saved slider sweeps describe these existing development scores and cannot
choose or change a deployment threshold.

The primary interval uses 2,000 paired whole-patient draws, seed 20261028, with no skipped draws.
It is conditional on the two fitted models and this calibration sample. It does not incorporate
new training samples, calibration-sample uncertainty or a new hospital. The cohorts match source,
label-evidence rank and target counts (506 negatives, 264 positives); other case mix may differ.
Their overlap of 31 training records and 25 known patients is documented and allowed. This design
does not isolate a causal effect of age.

## Secondary broad-cohort results

| Training cohort, fixed blend | AUROC | Sensitivity | TP | FN | FP | TN | Normal referral |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Age-preferred 25k | 0.936451 | 77.01% | 2,760 | 824 | 330 | 6,520 | 4.82% |
| All-age 25k | 0.936770 | 77.40% | 2,774 | 810 | 358 | 6,492 | 5.23% |
| Original PTB-only reference | 0.917495 | 68.64% | 2,460 | 1,124 | 346 | 6,504 | 5.05% |

The broad selections improve these descriptive hospital-development points over the original
PTB-only reference. Training size, sources, class prevalence, annotation evidence and admission
differ, so this is not an isolated age or architecture effect. The secondary comparison cannot
rescue the failed primary age hypothesis or select a clinical winner. The UI default remains the
prespecified PTB-only reference rather than whichever new model scores highest.

The strict cohort's exact mean age is 21.51 years; the matched control's is 57.46. The broad
age-preferred cohort has only 770 records aged 18–25 and exact-age mean 59.95; the all-age selection
has 668 and mean 62.17. Privacy-censored PTB ages are reported separately, not treated as true age 90.
These source-hospital ECGs do not establish performance in Italian university students.

## Integrity, resource use and reproducibility scope

Qualification reproduced every saved predecessor point metric, calibrated threshold and integer
matrix, the known original 50:50 reference, and saved HGB/xECG predictions to absolute tolerance
1e-12. Thirty-two raw PTB records reproduced the frozen 130-feature cache exactly after historical
six-significant-digit serialization. Thirty-two admitted PTB xECG vectors then replayed exactly.

A real profile decoded and extracted 80 admitted Challenge training ECGs through both feature
paths. It used 622,368,768 peak GPU bytes, with 10,598,547,456 initially free; its conservative
two-times-plus-300-second forecast was 3,914.6 seconds. All resource gates passed. Full feature
extraction of 19,666 admitted Challenge training records took 785.446 seconds. Fixed readout fits
took 1.247 seconds (strict), 0.321 (matched), 6.705 (age-preferred 25k) and 6.773 (all-age 25k).
Evaluation and aggregate export took 28.538 seconds, including the paired bootstrap.

The first profile attempt failed during xLSTM import because CUDA_HOME was unset, before inference.
The unchanged-code rerun used the installed PyTorch runtime at
`.venv/lib/python3.11/site-packages/nvidia/cuda_runtime`; no package or backbone changed. The failed
log remains local. The fresh bootstrap includes automatic discovery of these installed headers.

| Completed local stage | Receipt SHA-256 |
| --- | --- |
| Qualification | `b8d7400a6faa23cdd9f7a58a2c07b294ee942cb40bc54942c929e429b23cce7f` |
| Profile | `2c994c936302325099aab6d673e81e8fe895a33eb3fe9c4734e7ea7b02b18de1` |
| Training extraction | `8cd01beccd47520a6f11da9ce6b63f1d54c166c3b7fa3f4fc2a271dbc8f3eb95` |
| Evaluation | `2d6a0e59fac54ef9215ea2bfc3ecb1f139dc3d855f61b356b08da0cfdae0e7b6` |

The aggregate SHA-256 is `36389457555c74e86ca352595a36b2b51353c1f804308c65d208ad1735a52f58`,
identical to the receipt-bound local evaluation artifact. Models, embeddings and patient predictions
remain local. This executed run reused verified historical PTB/SPH feature artifacts; the separate
[fresh bootstrap protocol](pr74-fresh-baseline-protocol-2026-10-28.md) removes those input dependencies.
The [completed fresh verification report](pr74-from-raw-reproduction-results-2026-10-28.md) records
the full new raw PTB/SPH extraction, the preserved numerical correction and explicit raw-table/
training-feature reuse. All 30 scientific evaluation payloads and the primary interval match exactly.

All classifier/scaler fitting used admitted training ECGs only. The released xECG encoder was frozen;
only CPU readouts were trained. No new score was computed on the original Challenge held-out groups,
PTB fold 10 or EchoNext test. The covered record/patient/hash reference checks do not audit every
possible alternative window or establish patient independence for sources without patient IDs.
Released xECG's prior Chapman/Ningbo exposure remains a limitation. Labels remain source-specific
ECG-annotation proxies, not confirmed cardiopathy or referral truth. SPH and PTB evaluation are reused
development evidence, and scores are uncalibrated rather than population disease probabilities.

The matrices measure the fixed models offline on every admitted development record. They do not
validate the application's complete quality-abstention plus referral workflow. For example, the
fresh raw administrative receipt (`bebdc12bbecb800b4f6db29cf8414b893a13a4c9cb6e4e3f8ceef8d7f663682f`)
marks descriptive signal-quality failures in 32 of 6,968 admitted SPH calibration ECGs and 38 of
10,434 evaluation ECGs; these rows remain in the prespecified scoring population. The application
can instead abstain before prediction on failed quality or physiological feature extraction.
An abstention needs review and is never a negative prediction or evidence of health. No primary
denominator or endpoint was changed after seeing results.
