# CLF-01 results: at SPH the frozen HGB beats the tuned SVM, MLP and logistic regression

28 October 2026, under the committed [protocol](clf01-measurement-classifiers-protocol-2026-10-28.md)
(`25923c0`; code `d2bda20`). All models were trained on the all-age 25k cohort with the same 130
measurements. Each threshold was set from the 6,968 SPH calibration normals (5% budget, strict `>`; 348
referred for every model). Scores are from the 10,434 SPH evaluation ECGs (3,584 positive).

| SPH | Sensitivity | Specificity | AUROC | TP | FP |
| --- | ---: | ---: | ---: | ---: | ---: |
| HGB (frozen) | 0.745815 | 0.945255 | 0.925757 | 2,673 | 375 |
| RBF SVM (C=3, gamma=0.0056) | 0.703683 | 0.947883 | 0.913229 | 2,522 | 357 |
| MLP (128, relu) | 0.704799 | 0.949781 | 0.911772 | 2,526 | 344 |
| Logistic regression (C=0.1) | 0.635603 | 0.956058 | 0.890986 | 2,278 | 301 |
| HGB + xECG blend (current best) | 0.773996 | 0.947737 | 0.936770 | 2,774 | 358 |
| SVM + xECG blend | 0.756696 | 0.949781 | 0.930654 | 2,712 | 344 |
| MLP + xECG blend | 0.759766 | 0.948467 | 0.930631 | 2,723 | 353 |
| LR + xECG blend | 0.735770 | 0.952117 | 0.926512 | 2,637 | 328 |

Paired differences against HGB (or against the HGB blend). Sensitivity intervals are 98.33% (Bonferroni
over three challengers) with 2,000 whole-patient draws, seed 20261028, and no skipped draws. AUROC
intervals are 95%.

| Contrast | Sensitivity, pp [98.33%] | Specificity change, pp | AUROC difference [95%] | Rule |
| --- | --- | ---: | --- | --- |
| SVM − HGB (primary) | −4.213 [−5.469, −3.016] | +0.263 | −0.01253 [−0.01571, −0.00946] | HGB better |
| MLP − HGB | −4.102 [−5.342, −2.875] | +0.453 | −0.01399 [−0.01721, −0.01080] | HGB better |
| LR − HGB | −11.021 [−12.555, −9.329] | +1.080 | −0.03477 [−0.03971, −0.02984] | HGB better |
| SVM blend − HGB blend | −1.730 [−2.687, −0.827] | +0.204 | −0.00612 [−0.00770, −0.00457] | HGB better |
| MLP blend − HGB blend | −1.423 [−2.281, −0.521] | +0.073 | −0.00614 [−0.00770, −0.00462] | HGB better |
| LR blend − HGB blend | −3.823 [−4.914, −2.689] | +0.438 | −0.01026 [−0.01231, −0.00810] | HGB better |

None of the challengers beat HGB. The interval for every one of them lies entirely below zero, alone and
in the blend. Their slightly higher specificity does not offset their loss of 151 to 395 caught positive
ECGs.

Comparison with the internal evidence: in PTB-XL cross-validation (Avance 4.8) the SVM and MLP trailed
gradient boosting by only 0.004 to 0.006 AUROC, about one fold standard deviation. At the external hospital
they trail HGB by 0.013 to 0.014 AUROC and about 4 pp of sensitivity at the operating point, and logistic
regression trails by 0.035 AUROC and 11 pp. Internal validation therefore understated how much better HGB
transfers. Even so, the PR 74 claim that boosting is among the best models holds at SPH. Its claim that
the SVM is nearly as good does not.

PTB fold 9 (secondary, 889 calibration normals) was closer: sensitivity HGB 0.628, SVM 0.651, MLP 0.635,
LR 0.582; AUROC 0.9175, 0.9181, 0.9173 and 0.8890.

## Integrity and run

The refitted all-age HGB and xECG matched the saved PR 74 SPH and PTB scores with a maximum difference of
0.0. The saved all-age HGB and blend SPH points (threshold, matrix, AUROC) reproduced exactly before any
challenger was fitted. Fit times were SVM 48.1 s (5,727 support vectors; no subsample needed), MLP 2.4 s
(early stopping after 22 iterations) and LR 0.3 s. The run used the CPU only, with two threads and no GPU
lock.

## Limitations

- The challengers keep the Avance 4.8 hyperparameters, which were tuned on 111 selected PTB-XL features.
  Here they run on 130 multisource measurements, and gamma was not rescaled.
- SPH is reused development data. The intervals are conditional on the fitted models and the
  calibration sample.
- Labels are ECG-annotation proxies.

Aggregate: `reports/clf01-measurement-classifiers-2026-10-28/aggregate.json`. Local outputs:
`outputs/clf01_measurement_classifiers_20261028/`. Log: `outputs/clf01_run_20261028.log`.
