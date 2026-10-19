# Experiment 041 results: a stronger CPU screening baseline

18 October 2026. The fixed histogram gradient boosting model passed all four development gates.
On PTB-XL fold 9, AUROC was **0.946512**, compared with **0.918070** for the matched logistic
regression. The paired difference was **+0.028442 [0.021141, 0.036076]**. At separately calibrated
5% normal-referral thresholds, sensitivity increased from 0.670455 to 0.781656. This is improvement
on reused development data, not clinical validation or a final test.

## Executed comparison

Both models used the same 130 raw signal measurements and the same fitting patients. No age, sex,
selected Avance transformation, PCA or supervised feature selection entered either model.
Logistic imputation and standardization were fitted only on the fitting role. Boosting used native
missingness, fixed settings and no early stopping. No search or additional fit followed the result.

| Role | Official folds | ECGs | Patients | Positive | Target-negative |
|---|---|---:|---:|---:|---:|
| Fit | 1–7 | 14,954 | 12,968 | 8,600 | 6,354 |
| Normal-only threshold calibration | 8 | 2,129 | 1,854 | 1,240 | 889 |
| Development evaluation | 9 | 2,146 | 1,917 | 1,232 | 914 |

Patient roles were disjoint. Only the 889 calibration negatives determined each threshold;
calibration positives were counted but did not determine a threshold. The strict order-statistic
rule referred 44/889 calibration normals in each arm (4.9494%). Thresholds were 0.7786042356752809
for logistic and 0.7222200091857669 for boosting; referral means strictly greater than the threshold.

| Fold-9 endpoint | Logistic | Boosting |
|---|---:|---:|
| AUROC, patient-bootstrap 95% interval | 0.918070 [0.906343, 0.929177] | 0.946512 [0.937365, 0.955153] |
| Average precision | 0.944800 | 0.962080 |
| Sensitivity at frozen threshold | 0.670455 [0.640938, 0.698316] | 0.781656 [0.756641, 0.805556] |
| Normal referral rate | 0.030635 [0.019506, 0.042798] | 0.047046 [0.033769, 0.061471] |
| Brier score | 0.114299 | 0.092523 |
| Positive ECGs missed | 406 | 269 |
| Target-negative ECGs referred | 28 | 43 |

The paired sensitivity difference was +0.111201 [0.090623, 0.132002]. Both the AUROC and
sensitivity contrasts used 2,000 whole-patient draws, seed 20261018, with zero skipped single-class
draws. Model fits and thresholds were held fixed in these intervals; they do not cover refitting
or calibration-cohort uncertainty. The two realized normal referral rates differ, so the sensitivity
contrast is at the two prospectively calibrated operating rules rather than exactly equal achieved
specificity. All four gates passed: AUROC gain at least 0.01, positive lower bound, sensitivity
difference at least −0.02, and boosting normal referral no more than 8%. Decision: `advance_development`.

Descriptive misses decreased in each existing superclass: MI 117→82 of 540 positive ECGs, STTC
154→75 of 528, CD 116→78 of 495, HYP 73→63 of 268. Classes overlap; these counts are not exclusive
diagnoses, new clinical definitions or localization labels.

## Integrity, failure preservation and resources

The first attempt stopped before integrity fitting or either new fit: its validator overlooked the
historical flat-lead exclusion in the Avance notebook. The failure remains at
`outputs/experiment041_cpu_screening_v1/failure.json`. The separately committed
[forward correction](experiment-041-cpu-screening-execution-v2.md) preserved the original code and
every scientific setting. Its exact-cohort check excluded the one previously excluded labeled
flat-lead record and retained the historical no-beat failures.

The v2 run reproduced historical baseline validation AUROC 0.9280305990508397 and average
precision 0.9511171977377277 exactly. Refitted and saved baseline probabilities had maximum
absolute difference **0.0**, passing the original 1e-12 tolerance. This historical model used 111
transformed features and folds 1–8; it is an integrity comparator, not the matched raw-feature
logistic arm in the primary comparison. The baseline notebook, initially absent on the inspected
research branch, was present after integration of current main before execution.

The new logistic fit converged in 351 iterations without reaching its 5,000-iteration cap;
boosting completed its fixed 200 iterations. The complete successful run took 15.202 seconds with
at most four numerical threads, no GPU, no download and no new waveform access. Its fit stages
took 0.383 and 0.643 seconds respectively. Thirty artificial tests and independent implementation
review passed before v2 launch; integrated checks including the transfer and shared helpers passed
66 tests. Saved-output independent arithmetic verification is pending.

## Scope and next step

The target preserves Avance's diagnostic-superclass annotations: any MI/STTC/CD/HYP is positive;
NORM as the only diagnostic superclass is negative. It differs from the older strict NORM/SR
project rule and does not implement the international athlete criteria. Fold 9 informed earlier
work, and the nonlinear hypothesis was already supported by historical notebook output. This
experiment improves the reproducible CPU implementation and calibration separation; it supplies
no untouched confirmatory sample, student-population estimate, confirmed disease diagnosis or
anatomical/temporal localization evidence. No fold-10 score was computed, and Challenge/EchoNext
closed tests were not opened.

[Experiment 042](experiment-042-measurement-transfer.md), frozen before this result, compares the
same two fitted models at SPH with independent patient roles and local-normal calibration. It is
the next check of this result's usefulness. This run does not replace the historical xECG pipeline.

## Local artifacts and reproduction

Run source was committed as `1bc6bac`; protocol and forward amendment were committed before any
new fit. Python 3.11.13, NumPy 2.4.6, pandas 3.0.6, SciPy 1.17.1 and scikit-learn 1.9.1 were used.
Full source/input/protocol hashes and versions are in the result receipt.

| Artifact under `outputs/experiment041_cpu_screening_v2/` | SHA-256 |
|---|---|
| `result.json` | `d08893ae3eb4fd9f764f458572cb7f6c14a55606c64932a70a02b6de40291370` |
| `fitted_models.joblib` | `a66a331a8be658aee8729b4431f68f646003bbafdd456136a78a5260ea3ef071` |
| `predictions.npz` | `d4c17b0d51374006e8e7c71537e1867a9c8389a390695f000446c1ef45e771a1` |
| `integrity.json` | `ae6b8fa81028dae87c1bdceb6957209c2ea58d28b590f67767bbcc8dc989e18b` |
| `cohort.json` | `525da04510271e12f61391d8b37e5aad35a4104543ba4ebd9ff6c7f06f718f96` |

The original v1 failure receipt hashes to
`a107b37345e3f6c439223a4be6c900deaf4138653da3ecf21db9257ea8e347a7`.
Models, feature and patient identities and predictions stay local and untracked. To reproduce
under a new exclusive output directory, preserving the completed run:

```bash
UV_CACHE_DIR=/tmp/ecg-uv-cache OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  uv run --no-sync python -m scripts.experiments.run_cpu_screening041_v2 \
  --threads 4 --output outputs/experiment041_reproduction
```
