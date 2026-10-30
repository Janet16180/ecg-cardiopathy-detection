# Avance 5: executed delivery

The completed Spanish notebook is `notebooks/Avance5.8.ipynb`. Its final version has 15 code cells,
executed sequentially with no errors; it contains four diagnostic figures, threshold and localization
screenshots, a Python localization example and paired fine-tuning confusion matrices. Source and results are under
`ecg_experiment/avance5.py` and `reports/avance5_final/`. Patient predictions and fitted objects
remain in ignored `outputs/avance5_final/`. Its HTML export is local, in that same output directory.

## Main comparison

All nine new recipes used the exact verified all-age 25,000-ECG supervised cohort. Three grouped
training folds selected random-forest `max_features=sqrt`, histogram boosting `max_leaf_nodes=31`,
`learning_rate=.08`, and xECG logistic `C=.1`. Known patients remained together; unavailable patient
identities still limit independence. The equal boosting/xECG blend had the highest training OOF
AUROC, **0.980406**, and was selected before hospital-development scoring. Stacking used SVM/MLP
OOF predictions; its non-nested internal diagnostic was explicitly excluded from model selection.

| Recipe | SPH AUROC | Average precision | Sensitivity | Specificity | TP | FN | FP | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Equal boosting/xECG blend | 0.937352 | 0.918310 | 0.773996 | 0.950657 | 2774 | 810 | 338 | 6512 |
| Frozen xECG + logistic | 0.932119 | 0.911947 | 0.760882 | 0.950365 | 2727 | 857 | 340 | 6510 |
| Histogram boosting | 0.927218 | 0.895611 | 0.747210 | 0.944818 | 2678 | 906 | 378 | 6472 |
| Random forest | 0.925041 | 0.897761 | 0.725725 | 0.950365 | 2601 | 983 | 340 | 6510 |
| Stacking SVM/MLP | 0.914478 | 0.887567 | 0.707589 | 0.947445 | 2536 | 1048 | 360 | 6490 |
| Equal SVM/MLP vote | 0.914432 | 0.887539 | 0.708426 | 0.947737 | 2539 | 1045 | 358 | 6492 |

The blend-minus-xECG AUROC difference is **+0.005233 [0.003372, 0.007172]**, paired whole-patient
bootstrap, 2,000 draws, seed 20261029, zero skipped draws. These intervals condition on fitted models
and their calibration samples. Each hospital uses its own strict five-percent normal-calibration
budget; SPH has 6,968 calibration normals and 10,434 evaluation ECGs (3,584 positive, 6,850 negative).

The selected blend's PTB fold-9 AUROC is **0.925880**, sensitivity 0.685065, specificity 0.962801.
Its gap from training OOF reflects the different evaluation population and source/label mixture;
it prevents interpreting a high pooled training score as established cross-hospital generalization.
The new mixture differs from the earlier fixed HGB/.01-logistic recipe; their results are separately
attributed. Before new fits, the older all-age blend matrix reproduced exactly: 2774/810/358/6492.

Final component fitting took 1.613 s for boosting and 2.110 s for xECG's head; their mixture reuses
these fits. This excludes prior encoder pretraining, feature extraction, OOF generation and search.
Random-forest tuning took 241.024 s; boosting tuning 5.346 s; xECG tuning 4.260 s. All candidate
settings, OOF costs and final-fit times are in the executed tables.

## Pretrained representations

With identical PTB folds 1–7 fitting, fold 8 normal calibration, fold 9 evaluation and logistic C=.01,
released xECG achieved AUROC **0.943080**, JEPA **0.941003**, and released S4 CPC **0.918773**.
Thus xECG was the strongest of these three frozen representations for this particular task. This
does not isolate an architecture effect: released pretraining data, budgets and representations differ.
The 25k multisource fitting did not improve the separate PTB-only development point. No universal
or student-population superiority is claimed.

## Other evidence in the notebook

The original strict-age result and its uncertainty are preserved: the 770-record young/matched
blends differed by +0.474 sensitivity points [−0.722, +1.638], failing the prespecified two-point gate.
Published US athlete proportions inform explicitly conditional workload arithmetic, never validated
Italian disease forecasts. The app's independent JSON readers validated its cohort summaries,
15 aggregate model entries and NBR-01 localization export. No existing joblib was deserialized.

The notebook describes the actual normal-reference replacement display, its territory/top-lead
limitations, SHAP background sensitivity and the need for specialist lead/interval labels. It cites
the three model papers and two US screening studies. AI assistance is declared accurately;
individual contributions inherited from Avance 4 are labeled as a proposed responsibility allocation.

Targeted checks: Ruff, Git whitespace checks, five synthetic tests of patient separation and
scaler/head algebra, exact previous-matrix replay, and complete sequential notebook execution.
No original Challenge test, PTB fold 10 or EchoNext test was scored. The completed full-backbone
GPU [successor result](avance5-finetune-results-2026-10-29.md) did not improve the operating point;
the frozen blend remains selected and the notebook includes both outcomes.
