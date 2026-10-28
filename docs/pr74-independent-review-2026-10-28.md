# Independent review of PR 74

Reviewed 27 October 2026 (local time). PR: https://github.com/Janet16180/ecg-cardiopathy-detection/pull/74

- Title: “Add the ECG app with its two models and the balanced dataset rebuild …”
- Author: AndreGutierrezTecMx.
- Reviewed head: `cd642d74740b7d52c79e2a1b24707c17355c6a78`.
- Base at review: `ea1ee5140e5277b65a31ad3bf048c0b705d125ee` (`main`).
- Scope: 53 added files, 89,614 inserted lines, largely a local FastAPI presentation app, metadata for a balanced 20,000-ECG dataset, validation prediction CSVs for experiments 057–063, and two serialized models. No review comment was posted, and no branch was pushed or merged.
- Isolated checkout: `/tmp/ecg-pr74-review`. Review scripts: `/tmp/pr74_audit.py` and `/tmp/pr74_multilabel_audit.py`; executed output: `/tmp/pr74-audit-results.txt`; recomputed metric table: `/tmp/pr74-recomputed-metrics.csv`.

## Assessment

The evidence supports a useful exploratory classifier and an informative demonstration, with several important presentation and evaluation limitations. It does not support selecting this as a validated student-screening system. The weak point is the evaluation contract and operating point, not the absence of a large or novel neural network. The selected models are ordinary gradient boosting on engineered ECG features; using those models is reasonable. The observed AUROCs are credible as calculations on the supplied validation predictions. Their clinical and external generalization interpretation remains unresolved.

There are two particularly concrete reporting defects: a mislabeled denominator inflates the displayed top-three diagnosis hit rate, and an unsupported account of the xECG pretraining source undermines the overlap assurance. Missing training/protocol sources prevent verifying important claims beyond the supplied output arithmetic.

## Ranked findings

### 1. P1 — Restore the experiment provenance before treating these comparisons as reproducible research

Locations: `ecg_experiment/balanceado_v2.py:1` (references `docs/experiment-057-062-protocolo-v2.md`), `ui/experimentos.py:39`, `ui/experimentos.py:40`, `ui/modelo.py:1`, `ui/modelo.py:95`, `ui/modelo.py:132`, `scripts/data/crear_balanceado.py:3`.

The PR repeatedly attributes its data selection, training, comparison, thresholds, and pretraining exclusions to notebooks 28–39 and protocols for 057–063. None of those numbered notebooks or protocols exists in the reviewed Git tree. The added label CSV is the only numbered notebook artifact in that range. There is no runner for retraining either selected model or reproducing experiments 059–063. The serialized models, validation CSVs, and common feature/split functions are present. The waveform reconstruction script reconstructs a fixed selection; it does not reproduce the selection procedure, training, or model selection.

Consequently, the following cannot be independently established from this PR: which checkpoint initialized xECG; whether all evaluation patients/records were excluded from SimDINO pretraining; optimizer, hyperparameters, stopping and checkpoint selection; whether preprocessing and the 111-feature selection used only allowed training data; exact label mapping/selection logic; or the claimed timing and protocol chronology. The helper Preprocessor does fit its transforms on the argument it receives, which is good, but its caller during training is absent. This is an evidence gap, not evidence of fabrication or definite leakage.

Requested change: commit the referenced pre-score protocols, dataset selection and training sources, checkpoint/corpus hashes and exclusion receipts, commands/environment, and a results report. Explain whether this is intentionally only a presentation PR and keep unreproducible comparisons explicitly exploratory. Update both experiment queues if adding actual experiments.

### 2. P1 — Correct the diagnosis hit-rate denominator and expose unsupported diagnoses

Locations: `outputs/experiment063_multietiqueta_v2/aciertos_val.csv:8`, `ui/modelo.py:138`, `ui/modelo.py:145`, `ui/experimentos.py:260`, `ui/modelo.py:191`.

The UI labels diagnosis hit rates as a fraction of ECGs with cardiopathy in validation. There are 1,000 such ECGs. Recomputing top-three predictions from `probabilidades_val.csv` and truth from `ecgs.csv` gives **830 hits / 1,000 = 83.0%**. The exported/displayed **86.4583%** equals **830 / 960**, restricting the denominator to positives with at least one of the 23 modeled diagnoses. The other 40 positives have only excluded diagnoses. These include preexcitation/WPW (7), ventricular tachycardia (2), and complete AV block (1), among other labels. No new clinical interpretation of these labels is required to establish the denominator error.

The category top-one figure is different: 764 / 1,000 = 76.4%, which does match its label. The top-three constant-frequency baseline also uses the restricted denominator: 50.6% of all positives versus displayed 52.7083% of eligible positives. Thus the multilabel model still improves substantially over that baseline after correction; this finding does not erase its useful signal.

Requested change: state denominator and label coverage separately for each metric; report all-positive and supported-label performance; identify diagnoses outside scope in the inference UI. Do not label restricted-support top-three hits as overall diagnosis accuracy. Also clarify that top-three hit rate only requires one true label in a list and is not complete multilabel correctness or clinical diagnostic accuracy.

### 3. P1 — The shipped 99%-sensitivity operating point is unsuitable evidence for the project's referral-budget decision

Locations: `ui/modelo.py:68`, `ui/modelo.py:94`, `ui/modelo.py:131`, `ecg_experiment/balanceado_v2.py:128`, `outputs/experiment063_multietiqueta_v2/deteccion_val.csv:7`.

The interface explicitly chooses its operating point for 99% sensitivity on the same validation set used for model comparison. Its ROC specificity is correctly computed; the clinical implication is poor selectivity, not an arithmetic bug. With all training labels:

| Model | AUROC | Validation sensitivity | Specificity | Normal ECGs flagged | All balanced validation ECGs flagged |
| --- | ---: | ---: | ---: | ---: | ---: |
| Binary gradient boosting | 0.933688 | 990/1000 = 99.0% | 32.5% | 675/1000 = 67.5% | 1665/2000 = 83.25% |
| Multilabel combined score | 0.935326 | 991/1000 = 99.1% | 30.2% | 698/1000 = 69.8% | 1689/2000 = 84.45% |

With 2,000 training labels, binary specificity at 99% sensitivity averages 21.14% across five seeds. Full-data SVM specificity is 34.1%, better than either shipped model at that point despite a slightly lower AUROC. This illustrates why choosing by AUROC rank need not optimize a screening workflow.

The UI does disclose that these are validation numbers and that the test remains unused, which is appropriate. Fitting and displaying the threshold on validation is acceptable for exploratory calibration; it is not an independent confirmation of future 99% sensitivity. The threshold has no separate evaluation here, and the same validation set is used for per-label maximum-F1 thresholds. No claim of final-test leakage can be made just from that reuse.

The current project guide requires operating points from a referral budget on local normals (Experiment 030), not a fixed-sensitivity target. Requested change: call this a demo threshold and assess/freeze the target and referral budget with the cardiologist before any approved final evaluation. Report false referrals and missed positives with uncertainty. Do not open the protected tests to resolve this review.

### 4. P1 — xECG's claimed pretraining provenance and no-overlap assurance are unsupported

Location: `ui/experimentos.py:318` (“Pesos ya preentrenados con EchoNext, que no tiene ECGs del dataset balanceado.”). Related: `ui/experimentos.py:322` says the pretraining contributes little.

The official xECG paper lists CODE, INCART, Chapman and Ningbo as pretraining sources, not EchoNext. Primary evidence: https://arxiv.org/html/2509.10151v1 (Methods, Pretraining datasets; lines 326–346 in web rendering). The official released model is https://huggingface.co/riccardolunelli/xECG_base_model_v1 and links the authors' code, https://github.com/dlaskalab/bench-xecg/. The project's existing `docs/experiment-007-xecg.md:18` and `:26` already record the official checkpoint hashes and pretraining sources.

The balanced dataset includes Chapman and Ningbo. **If** experiment 059 used that official checkpoint, those portions cannot be described as pretraining-unseen evaluation, and the EchoNext sentence is false. If a different checkpoint actually pretrained or adapted on EchoNext was used, that provenance must be supplied; the PR omits notebook 33 and a checkpoint receipt, so I cannot identify the actual weights from this PR alone. Source overlap is established for the official method; exact record inclusion/overlap for the absent checkpoint is not established by this review.

The supplied scores show fine-tuned xECG AUROC 0.873380 vs 0.865775 from scratch at 2,000 labels, a difference of about 0.0076. These numbers alone do not isolate a general benefit of pretraining without verifying initialization, adaptation, equal tuning effort, and training success. Do not infer that xLSTM or foundation models are intrinsically unsuitable.

Requested change: supply exact checkpoint identifier/hash and training receipt; correct the source description; audit overlap and clearly mark seen-source arms. Keep evaluation families excluded when pretraining a new encoder.

### 5. P1 — Preserve the project's frozen test boundary in the new split and viewer

Locations: `ecg_experiment/balanceado_v2.py:91`, `ui/datos.py:453`, `ui/datos.py:460`, `ui/datos.py:469`, `ui/preparar_senales.py:82`, `ui/preparar_senales.py:86`, `scripts/data/crear_balanceado.py:189`.

The new helper independently randomizes Challenge records into a new 70/10/20 split. It never reads or checks the existing frozen Challenge split. The viewer samples from every record of the selected dataset without filtering the project’s closed groups, and preparation samples waveforms from all source records. Therefore these execution paths do not enforce the established holdout boundary. A new split named test does not preserve the pre-existing closed test.

The existing `docs/challenge-splits-v1.md` says every subsequent Challenge experiment must retain the frozen assignment and describes grouped signal duplicates and source families. The PR's new split does not use those assignments or groups. I did not read frozen-test scores, waveforms, or labels, and did not compare record IDs against protected manifests, so I am not reporting a measured count of test records already consumed by this branch. This is a missing boundary check in the committed code and a required provenance audit, not an allegation of knowingly using the closed tests.

Requested change: filter/reject closed records before reconstruction, EDA sampling, training, and pretraining, and provide an assignment-overlap audit without opening test waveforms or scores. If the user intentionally authorizes a different research track, document the consequences for the original holdout first.

### 6. P2 — Model scores are presented as disease probabilities without calibration evidence

Locations: `ui/modelo.py:84`, `ui/modelo.py:113`, `ui/modelo.py:130`, `ui/static/textos.js:67`, `ui/static/textos.js:68`, `ui/static/textos.js:73`, `ui/static/textos.js:94`.

The binary output comes directly from `predict_proba` on an artificially 50/50 dataset. The multilabel output is `1 − product(1 − p_category)`, which is the union probability only under independence assumptions; the UI itself acknowledges category dependence (`ui/experimentos.py:271`). For an elementary example, two identical events each with probability 0.5 have union probability 0.5, while the formula yields 0.75. The formula can still be a useful ranking score, and its AUROC was recomputed correctly, but that does not calibrate its probability interpretation.

Neither model has a calibration assessment or a demonstrated transport to the student population's label prevalence. AUPRC and VPP from this enriched validation sample are similarly specific to its prevalence. The UI additionally says “Sin riesgo de cardiopatía” below threshold, which is stronger than “below the referral threshold.” The application does state that the result is not a diagnosis, a useful qualification that should be retained.

Requested change: describe an uncalibrated model score and threshold result, or provide an approved representative calibration study before using population probability language. Do not equate AUROC 0.934 with 93.4% accuracy, 93.4% diagnostic certainty, or demonstrated sensitivity in new students.

## Other material limitations and software observations

- **Target definition is provisional and different from the established project proxy.** `notebooks/28-clasificacion-etiquetas.csv:26` onward adds broad arrhythmia, axis/morphology and other categories; `:91` labels incomplete RBBB as cardiopathy. `ui/datos.py:253` calls the label mapping provisional pending cardiologist review. Isolated LV high voltage remains undefined (`:23`–`:25`), which follows the guide. Treat this as a separate unapproved target definition and do not compare its AUROC numerically with previous project experiments as though they predict the same outcome. No diagnosis or athlete-criteria clinical adjudication was performed in this review.
- **Population and source generalization are untested.** The balanced cohort's median ages are 43 (positive) and 45 (negative), range 1–59. Only 2,937/20,000 records are age 18–29. This is a cohort-description check, not a new age-subgroup performance analysis. All seven sources occur in training and validation. Source-stratified random validation does not test a new hospital; source-only AUROC is already 0.6524. Counterbalancing evidence: binary full-data within-source AUROCs remain substantial (CPSC 0.878, Georgia 0.921, PTB-XL 0.902, Ningbo 0.964, Chapman 0.984), so the evidence does not support dismissing the entire result as a dataset-source shortcut. PTB Diagnostic has only 18 validation records, so its 0.861 estimate is particularly unstable.
- **Model choice is not decisively established.** Binary GB has mean AUROC 0.915270 versus SVM 0.908444 with 2,000 labels. Recomputed paired patient-group bootstrap intervals support GB improvement in only 3/5 seeds, matching the export and failing the displayed 4/5 rule. The full-data multilabel-minus-binary difference is 0.001638 with exported interval [−0.003828, 0.007398]. It does not establish superiority. Multilabel 2,000-label noninferiority is met in only 1/5 seeds by the displayed −0.01 lower-bound rule. The interface accurately discloses these failures. Repeated seeds on the same validation records do not replace independent validation of the selected method.
- **Bad-signal abstention is missing.** `ui/registro.py:112` replaces missing samples with zero, `:116`–`:125` produces warnings, and `:154` still predicts any input of at least 10 seconds. `ecg_experiment/caracteristicas_avance2.py:133`–`:150` marks flat leads or too few beats as failed extraction; `ui/modelo.py:41` immediately transforms/imputes features and predicts, discarding that status. All-flat or insufficiently measured inputs can therefore receive a risk category based largely on imputation, although dataset reconstruction explicitly excludes flat/invalid inputs. Reject or abstain on failures before presenting a prediction. This was identified by tracing code, not by running an endpoint on patient data.
- **CI is not green.** GitHub reports CPU tests/package build SUCCESS and Ruff lint FAILURE for the reviewed SHA. The lint log contains many actual violations in the new helper/rebuild files. No tests were added for the app or new research paths. The configured Ruff command excludes `ui/`, so the successful CPU/package check should not be represented as an application validation. The failed lint is a merge gate under AGENTS; it does not itself show that the model is scientifically poor.
- **Model novelty/size is not an objection.** A gradient-boosted tree ensemble over 111 ECG measures is a defensible baseline for 2,000–14,000 labels. The PR's strongest evidence is actually for that simple approach in its specific development sample. A graph model having slightly better mean AUROC or a foundation model having more parameters does not by itself make either preferable.

## Checks completed and positive findings

1. Recomputed every one of the 257 rows in `comparacion_metricas_val.csv` from the saved prediction CSVs, aligned by `fila`: AUROC, average precision/AUPRC, specificity at 95% and 99% sensitivity. All match to floating-point error (<1.2e−16). Average precision is the specific AUPRC convention used.
2. Recomputed all six multilabel union-score AUROCs from category prediction columns; they match `deteccion_val.csv`. Its saved binary comparison predictions reproduce the binary AUROCs.
3. Reconstructed `partition()` from the manifest and `budgets()` from training rows: zero mismatches against the committed partition/budget columns. Train = 13,999, validation = 2,000, new test = 4,001; validation is exactly 1,000 positives / 1,000 normals.
4. No declared patient group crosses this new split. Challenge has no patient IDs, so record-disjointness cannot establish patient-disjointness for those sources.
5. No duplicate `huella_origen` or `huella_senal` exists within the 20,000 selected records, hence no cross-split identical hash in those supplied hashes. This does not audit partial-window/near duplicates, header differences, known patient relations without IDs, or overlap with external pretraining.
6. Recomputed five paired GB–SVM AUROC differences with the shared whole-group bootstrap helper (1,000 draws, seed 2026); 3/5 intervals entirely exceed zero, consistent with the reported decision. The shared helper and patient-aware intervals are good practice.
7. The reconstruction script verifies source hashes and float16 waveform hashes, and writes a temporary array before replacing the output. This is a meaningful reproducibility strength for its selected waveform array.
8. Syntax compilation passed for all 14 new Python modules/scripts under the required Python 3.11 interpreter. This does not test application behavior.
9. The UI consistently labels reported research metrics validation, declares test unused, shows source-only baselines and per-source performance, labels the old SimDINO checkpoint contaminated, and discloses uncertainty/failure of decision rules. These are substantive strengths.

I did not train models, use the GPU, download data/checkpoints, execute the reconstruction or app preparation scripts, unpickle the committed joblib/pickle artifacts, or open protected Challenge/EchoNext test data. The review validates supplied prediction arithmetic, public PR manifests, and code paths; it does not certify that serialized models reproduce those predictions on reconstructed waveforms.

## Questions for the author

1. Where are notebooks 28–39 and the pre-registered protocols/results reports? Can the selected models be rebuilt from a clean checkout with exact commands and hashes?
2. Which exact xECG checkpoint was used, and why is EchoNext named? Which corpus and record exclusions produced the 77,610-example SimDINO cohort?
3. How was the established Challenge split respected before any selection, signal preparation, training and pretraining? What does “test unused” mean relative to the original project holdout?
4. Who approved the new target taxonomy, and which diagnoses and benign variants are intended to trigger referral? Which diagnoses are absent from the 23-label model?
5. Why was a 99%-sensitivity threshold selected despite flagging 67.5–69.8% of validation normals, and what referral capacity should the student screening workflow serve?
6. Can the top-three hit rate be reported with both denominators, and can population probability language be replaced with model-score language until calibration is supported?

## Suggested conclusion to communicate

“The app is a useful prototype and its saved validation AUROC calculations check out. A simple gradient boosting model is a reasonable choice here. However, the current evidence does not establish a better or deployable student-screening model: the operating point flags most normals, the label target and population differ, the training provenance is incomplete, the pretraining-overlap statement needs correction, and one diagnosis success figure uses a narrower denominator than the UI says. Fix and document these issues, preserve the frozen tests, and agree on the clinical target and referral budget before any final evaluation.”
