# Important papers for our ECG project

**Last checked:** 24 September 2026.

This reading guide collects the most relevant papers found during our experiments and literature search. The emphasis is raw 12-lead ECGs, learning from unlabeled recordings, transfer with limited labels, and eventual use of cardiologists' text reports. It includes recent work and the older papers needed to understand it. It is a curated guide, not an exhaustive systematic review.

Paper findings and **our project interpretation** are distinguished below. Publication and resource availability refer to the linked versions checked on this date. A repository link does not necessarily mean that working code or trained weights have been released.

## Start here

| Reading priority | Paper | Why it matters for us | Project status |
| --- | --- | --- | --- |
| 1 | [ECG-JEPA](#1-ecg-jepa) | Latent prediction; strongest frozen encoder in our completed comparison | Released encoder tested |
| 2 | [ECG-FM](#2-ecg-fm) | Practical wav2vec-style ECG pretraining; basis of our continued training | Frozen, fine-tuned, and adapted variants tested |
| 3 | [HuBERT-ECG](#3-hubert-ecg) | Cluster-based masked prediction and the reported 9.1-million-example corpus | Small model tested |
| 4 | [ECG foundation-model benchmark / ECG-CPC](#4-ecg-foundation-model-benchmark-and-ecg-cpc) | Broad comparison; a concrete state-space alternative to large transformer encoders | Benchmark and ECG-CPC not reproduced |
| 5 | [BenchECG and xECG](#26-benchecg-and-xecg) | Released xLSTM foundation model for a matched binary transfer study | Experiment 007 planned |
| 6 | [MERL](#10-merl) | Public implementation of learning from paired ECGs and reports | Not tested |
| 7 | [CoRe-ECG](#7-core-ecg) | Combines contrastive and reconstruction objectives; addresses lead shortcuts | Not tested |
| 8 | [Calibration](#19-on-calibration-of-modern-neural-networks) | Essential background for the requested confidence percentage | Our experiments use a separate calibration split |

For measured results, read [our model findings report](model-findings-report.md). For the ongoing larger-data comparison, read [experiment 003](experiment-003-mimic.md). Numbers in that report are our measurements and should not be attributed to the papers.

## A. Core models and comparative evidence

### 1. ECG-JEPA

**Citation:** Sehun Kim. *Learning General Representation of 12-Lead Electrocardiogram with a Joint-Embedding Predictive Architecture.* arXiv:2410.08559, first posted 2024; version 5 revised April 2026. **Preprint in the source checked.**

**Sources:** [Paper](https://arxiv.org/abs/2410.08559) · [Official code and checkpoint links](https://github.com/sehunfromdaegu/ECG_JEPA).

- **Idea:** Predict the representations of masked ECG regions using a teacher representation, instead of reconstructing every waveform sample. Cross-Pattern Attention (CroPA) structures attention for multilead signals.
- **Why we should read it:** It offers an alternative to both waveform reconstruction and HuBERT-style discrete targets. The main pretraining study uses roughly 180,000 public ECG examples, making its scale especially relevant to our current expansion.
- **Availability:** The repository provides code and links to pretrained checkpoints. We used its multiblock checkpoint.
- **Our interpretation:** This is the strongest immediate reference from our completed experiments. Its frozen encoder with all eligible public training labels achieved **0.9545 AUROC in our experiment**, not in a result quoted from this paper.
- **Caution:** Our compact JEPA-inspired model is a different implementation. The latest manuscript includes additional experiments; do not assume every released checkpoint used every dataset mentioned in the latest paper. Exact checkpoint provenance still matters.

### 2. ECG-FM

**Citation:** McKeen et al. *ECG-FM: an open electrocardiogram foundation model.* **JAMIA Open**, 8(5), ooaf122, 2025.

**Sources:** [Published paper](https://doi.org/10.1093/jamiaopen/ooaf122) · [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12530324/) · [Official repository](https://github.com/bowang-lab/ECG-FM) · [Weights](https://huggingface.co/wanglab/ecg-fm).

- **Idea:** Adapt wav2vec-style masked learning to ECGs, combined with contrastive learning across temporal views and lead masking.
- **Why we should read it:** It connects the user's original wav2vec idea to an available ECG model and a reproducible implementation.
- **Availability:** Code and pretrained weights are public. Its implementation uses `fairseq-signals`; it is not simply a standard Transformers `AutoModel` checkpoint.
- **Our interpretation:** It is a useful baseline for testing whether continued learning on a larger unlabeled pool improves transfer.
- **Caution:** Released weights already encountered MIMIC-IV-ECG and PTB-XL through their pretraining sources. Our additional MIMIC experiment is continued adaptation. Our implemented temporal contrastive objective is also simpler than the paper's complete pretraining objective.

### 3. HuBERT-ECG

**Citation:** Coppola et al. *HuBERT-ECG as a self-supervised foundation model for broad and scalable cardiac applications.* **medRxiv preprint**, first posted 2024; version 4 posted June 2026.

**Sources:** [Current paper](https://www.medrxiv.org/content/10.1101/2024.11.14.24317328v4) · [Earlier full methods and source datasets](https://www.medrxiv.org/content/10.1101/2024.11.14.24317328v3.full) · [Official repository and model links](https://github.com/Edoar-do/HuBERT-ECG).

- **Idea:** Build discrete targets by clustering ECG representations, then predict hidden targets from masked inputs. Large-scale pretraining is followed by task-specific transfer.
- **Why we should read it:** This is the source of the **approximately 9.1 million reported training examples** discussed earlier, and a direct application of HuBERT to ECGs.
- **Availability:** Code and pretrained model variants are released. The complete pooled raw corpus is not offered as one public download. Its source datasets have different access conditions; full Ribeiro/CODE access is restricted, while CODE-15% is a separate public subset.
- **Our interpretation:** Compare larger released variants before attempting to reconstruct the entire pretraining corpus.
- **Caution:** We tested only HuBERT-small. The reported sample count is not a count of 9.1 million unique patients. Historical pretraining includes PTB-XL and MIMIC; its methods also list SPH. Access to weights does not establish access to all training records.

### 4. ECG foundation-model benchmark and ECG-CPC

**Citation:** M A Al-Masud, Juan Miguel Lopez Alcaraz, and Nils Strodthoff. *Benchmarking ECG Foundational Models: A Reality Check Across Clinical Tasks.* **ICLR 2026**. The current arXiv version uses the shortened title *Benchmarking ECG FMs: A Reality Check Across Clinical Tasks*; the first preprint appeared in 2025.

**Sources:** [Paper](https://arxiv.org/abs/2509.25095) · [ICLR record](https://openreview.net/forum?id=xXRqWpt3Xr) · [Official benchmark and checkpoints](https://github.com/AI4HealthUOL/ecg-fm-benchmarking).

- **Idea:** Compare eight foundation models across 26 tasks and 12 public datasets, including different transfer settings and training-set sizes.
- **Why we should read it:** It directly addresses whether a larger or newer model is actually a better choice. Its compact **ECG-CPC structured state-space model** is an especially relevant alternative architecture.
- **Availability:** Benchmark code and checkpoint links are public. We downloaded and checksum-verified the [35.6 MB official checkpoint archive](https://figshare.com/articles/dataset/ECG-CPC_Checkpoint_zip/30192604), released under CC BY 4.0 and described as pretrained on HEEDB.
- **Our experiment:** [Experiment 004](experiment-004-cpc.md) adds this released S4 checkpoint as a frozen-feature reference, alongside a separate matched comparison of compact local CPC objectives.
- **Measured status:** The released checkpoint's frozen mean-pooled features reached test AUROC 0.93794 with full labels and 0.92347 with the 10% manifest; [implementation and results](released-ecg-cpc.md). These exploratory linear-probe results do not reproduce the paper's full benchmark or establish the best possible fine-tuned performance.

## B. Learning from unlabeled waveforms and lead structure

### 5. CLOCS

**Citation:** Dani Kiyasseh, Tingting Zhu, and David A. Clifton. *CLOCS: Contrastive Learning of Cardiac Signals Across Space, Time, and Patients.* **ICML 2021**, PMLR 139:5606–5615.

**Sources:** [Paper](https://proceedings.mlr.press/v139/kiyasseh21a.html) · [Official code](https://github.com/danikiyasseh/CLOCS).

- **Idea:** Construct related views through ECG leads, temporal segments, and patient identity, and learn by contrasting them with other signals.
- **Why we should read it:** It provides the ECG-specific background for our temporal-view adaptation and for handling repeated recordings from one patient.
- **Our interpretation:** The definition of a positive pair is an important experimental choice. Two halves of an ECG may differ when a finding is transient; forcing their representations to agree could remove useful information.
- **Project status:** Our adaptation is conceptually related, but it is not a complete reproduction of CLOCS.

### 6. Lead-agnostic self-supervised learning

**Citation:** Jungwoo Oh et al. *Lead-agnostic Self-supervised Learning for Local and Global Representations of Electrocardiogram.* **CHIL 2022**, PMLR 174:338–353.

**Sources:** [Paper](https://proceedings.mlr.press/v174/oh22a.html) · [Official implementation framework](https://github.com/Jwoo5/fairseq-signals).

- **Idea:** Learn local and global representations while using lead masking to support different lead configurations.
- **Why we should read it:** It supplies useful context for multilead pretraining and for the framework used by ECG-FM.
- **Our interpretation:** Lead masking and learning across leads have substantial prior work. They cannot, by themselves, justify a novelty claim for our custom architecture.
- **Availability:** The implementation framework is public; a checkpoint specifically reproducing this paper was not verified for this guide.

### 7. CoRe-ECG

**Citation:** Zehao Qin et al. *CoRe-ECG: Advancing Self-Supervised Representation Learning for 12-Lead ECG via Contrastive and Reconstructive Synergy.* **arXiv preprint**, 2026, arXiv:2604.11359.

**Source:** [Paper](https://arxiv.org/abs/2604.11359).

- **Idea:** Combine reconstruction with contrastive representation learning, frequency-aware augmentation, and masking across time and leads.
- **Why we should read it:** It explicitly addresses shortcuts caused by dependencies between ECG leads. This is relevant to designing masks that require useful learning rather than easy recovery from another lead.
- **Our interpretation:** It is a useful comparison for our custom lead-based objectives, although our implementation did not reproduce its method. Any future hybrid objective needs ablations of its component losses.
- **Availability:** No author-provided code or pretrained weights were verified in this review.
- **Project status:** Not tested; its paper results are not evidence that it would outperform our current reference on our task.

### 8. ECG-NAT

**Citation:** Mahsa Gazeran et al. *ECG-NAT: A Self-supervised Neighborhood Attention Transformer for Multi-lead Electrocardiogram Classification.* **arXiv preprint**, 2026, arXiv:2605.13194.

**Sources:** [Paper](https://arxiv.org/abs/2605.13194) · [Authors' repository](https://github.com/Mahsagazeran/ECG-NAT).

- **Idea:** Use hierarchical neighborhood attention to capture short waveform patterns and longer temporal context. Masked pretraining is followed by supervised contrastive and classification objectives.
- **Why we should read it:** It is a recent architecture alternative for morphology and rhythm at different time scales.
- **Our interpretation:** It is useful prior work for our multiscale model; multiscale attention alone is not an original contribution.
- **Availability:** At the check date, the repository was a README announcing code and model release upon acceptance. **Do not treat it as a ready-to-run implementation.**
- **Project status:** Not tested.

### 9. ACL-ECG

**Citation:** Wenhan Liu et al. *ACL-ECG: Anatomy-Aware Contrastive Learning for Multi-Lead Electrocardiograms.* **Sensors**, 26(3):1080, 2026.

**Source:** [Published paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12900096/).

- **Idea:** Use encoders for individual leads, anatomical lead groups, and contrastive objectives that reflect those groups.
- **Why we should read it:** It provides a physiological basis for organizing lead representations and choosing which views should agree.
- **Our interpretation:** Grouping every lead identically may overlook useful differences. This paper motivates testing explicit lead structure with a matched baseline.
- **Availability:** No author-provided code or pretrained weights were verified in the article review.
- **Project status:** Not tested. Anatomical grouping is existing work, so our thesis would need a more specific contribution.

## C. Learning from ECGs and cardiologists' reports

These papers become particularly relevant when we receive the university's paired waveforms and expert text. They also show how public ECG/report pairs could support pretraining before local access. A report is a source of supervision even if it has not been converted into a binary label.

### 10. MERL

**Citation:** Che Liu et al. *Zero-Shot ECG Classification with Multimodal Learning and Test-time Clinical Knowledge Enhancement.* **ICML 2024**, PMLR 235:31949–31963.

**Sources:** [Paper](https://proceedings.mlr.press/v235/liu24bg.html) · [Official code and pretrained checkpoint links](https://github.com/cheliu-computation/MERL-ICML2024).

- **Idea:** Align ECG representations with report text, then use text descriptions and clinical knowledge for classification.
- **Why we should read it:** It is a practical, available starting point for using expert text without reducing every report immediately to one binary target.
- **Availability:** The repository links code and pretrained ResNet/ViT checkpoints.
- **Our interpretation:** A waveform encoder pretrained with report supervision could still feed a simple binary classifier at deployment.
- **Caution:** Classification using text prompts is not the same as generating a faithful medical report. Text access, language differences, and the meaning of our referral target would need separate treatment.
- **Project status:** Not tested.

### 11. ECG-CLIP

**Citation:** Michael Ko, Matteo Gadaleta, Eric J. Topol, Evan D. Muse, and Giorgio Quer. *Development and external validation of a contrastive learning foundation model for ECG-based prediction of cardiovascular diseases and outcomes.* **The Lancet Digital Health**, online September 2026, article 101092.

**Sources:** [Published article](https://doi.org/10.1016/j.landig.2026.101092) · [Publisher abstract](https://www.sciencedirect.com/science/article/pii/S2589750026001159).

- **Idea:** Combine masked waveform reconstruction with ECG/report contrastive pretraining. The study uses more than 1.7 million ECG/report pairs and evaluates transfer using MIMIC data.
- **Why we should read it:** It is recent evidence for combining raw signals with clinician-written interpretations and studying transfer when downstream labels are scarce.
- **Our interpretation:** It supports investigating report-assisted pretraining as another route beyond waveform-only HuBERT or wav2vec objectives.
- **Availability:** The pretraining data are a Scripps clinical collection. Public code and weights were not verified for this guide.
- **Caution:** Its disease and outcome tasks differ from our university screening endpoint. Reported transfer performance is not a validation of our application.

## D. Background methods worth understanding

These are methodological references from speech or computer vision, not ECG screening evaluations.

### 12. wav2vec 2.0

**Citation:** Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.* **NeurIPS 2020**.

**Source:** [Paper](https://papers.neurips.cc/paper_files/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html).

- **Idea:** Mask parts of a latent sequence and learn to identify quantized targets through a contrastive objective, then transfer the representation using labels.
- **Why we should read it:** It explains the original idea behind the user's proposal and a major part of ECG-FM's ancestry.
- **Our interpretation:** The strategy is relevant to scarce ECG labels, but speech preprocessing, augmentations, and pretrained weights are not automatically suitable for multilead ECGs.

### 13. Original HuBERT

**Citation:** Wei-Ning Hsu et al. *HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units.* 2021; linked author manuscript arXiv:2106.07447.

**Source:** [Paper](https://arxiv.org/abs/2106.07447).

- **Idea:** Use clustering to create target units and learn to predict those units where the input is masked; targets can be refined through further clustering.
- **Why we should read it:** It explains the distinction between HuBERT's target construction and wav2vec 2.0's contrastive quantization approach.
- **Our interpretation:** Cluster quality, temporal resolution, and the information retained by target units are important choices for ECG adaptation.
- **Caution:** The original paper concerns speech. HuBERT-ECG is the application-specific reference for our project.

### 14. FixMatch

**Citation:** Kihyuk Sohn et al. *FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence.* **NeurIPS 2020**.

**Sources:** [Paper](https://proceedings.neurips.cc/paper/2020/hash/06964dce9addb1c5cb5d6e3d9838f733-Abstract.html) · [Official code](https://github.com/google-research/fixmatch).

- **Idea:** Generate pseudo-labels on weakly augmented unlabeled examples; retain confident predictions and train for agreement on stronger augmentations.
- **Why we should read it:** Unlabeled data can also help during downstream classifier training, after representation pretraining.
- **Our interpretation:** A carefully adapted consistency-learning baseline could be useful at 1% or 10% label budgets. It has not been tested here.
- **Caution:** The original experiments use images. ECG augmentations must preserve relevant findings, and in a population dominated by negative cases, confidence-based selection could reinforce majority-class errors. High pseudo-label confidence does not guarantee correctness.

## E. Dataset papers and primary dataset references

### 15. PTB-XL

**Citation:** Patrick Wagner, Nils Strodthoff, et al. *PTB-XL, a large publicly available electrocardiography dataset.* **Scientific Data**, 7:154, 2020.

**Sources:** [Paper](https://www.nature.com/articles/s41597-020-0495-6) · [Dataset v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/).

- **Contribution:** Public 12-lead waveforms, ECG statements, reports, patient metadata, and recommended folds.
- **Why we should read it:** It is the basis of our current labels, splits, and downstream experiments.
- **Version detail:** The original paper describes 21,837 ECGs. **Our v1.0.3 experiments use 21,799**; cite the release as well as the paper when reporting counts.
- **Our interpretation:** The dataset's multiple statement categories require a documented binary mapping. A normal ECG annotation is not proof that a person has no heart disease. Our retained-label proxy and exclusions are documented in the experiment report.

### 16. MIMIC-IV-ECG

**Citation:** Brian Gow et al. *MIMIC-IV-ECG: Diagnostic Electrocardiogram Matched Subset*, version 1.0, **PhysioNet dataset release**, 2023. This entry is a primary dataset reference, not a model paper.

**Source:** [Dataset description and citation](https://physionet.org/content/mimic-iv-ecg/1.0/).

- **Contribution:** Approximately 800,000 ten-second, 500 Hz, 12-lead ECGs with patient and recording identifiers.
- **Why we should read it:** It supports our current selection of 200,000 additional unlabeled recordings.
- **Availability:** Version 1.0 waveforms are open access. Access to waveforms does not grant access to every linked clinical dataset or cardiologist report.
- **Our interpretation:** It is suitable for larger waveform-only experiments, with patient grouping and duplicate checks. Its clinical source population differs from university students.
- **Caution:** MIMIC was already used to pretrain released ECG-FM and HuBERT models. Reusing it does not establish transfer to unseen pretraining data.

### 17. Ribeiro / CODE and large-scale supervised learning

**Citation:** Antônio H. Ribeiro et al. *Automatic diagnosis of the 12-lead ECG using a deep neural network.* **Nature Communications**, 11:1760, 2020.

**Sources:** [Paper](https://www.nature.com/articles/s41467-020-15432-4) · [Official code](https://github.com/antonior92/automatic-ecg-diagnosis) · [CODE-15% public subset](https://zenodo.org/records/4916206).

- **Idea:** Train a neural network on more than two million labeled ECG exams to recognize six specified abnormalities.
- **Why we should read it:** It is both an important large-scale supervised reference and background for the CODE data used by later foundation models. Its annotation process also draws on clinical reports.
- **Availability:** The paper's complete training collection requires research access. CODE-15% is a separate public release containing 345,779 exams; it is not the complete corpus.
- **Our interpretation:** Public labels can complement self-supervision. Strong supervised models remain essential comparisons.
- **Caution:** Absence of the six target abnormalities does not define an exhaustive healthy category or our referral endpoint.

### 18. SPH: standardized diagnostic statements

**Citation:** Hui Liu et al. *A large-scale multi-label 12-lead electrocardiogram database with standardized diagnostic statements.* **Scientific Data**, 9:272, 2022.

**Sources:** [Paper](https://www.nature.com/articles/s41597-022-01403-5) · [Public dataset and associated resources](https://springernature.figshare.com/collections/A_large-scale_multi-label_12-lead_electrocardiogram_database_with_standardized_diagnostic_statements/5779802/1).

- **Contribution:** 25,770 ECGs from 24,666 patients, with standardized diagnostic statements, patient information, and recordings lasting 10–60 seconds.
- **Why we should read it:** The annotation structure is relevant to harmonizing abnormality labels across public datasets.
- **Our interpretation:** It is a candidate additional source after adapting the signal contract and defining compatible labels.
- **Caution:** It is not automatically an untouched external test: [HuBERT's pretraining methods](https://www.medrxiv.org/content/10.1101/2024.11.14.24317328v3.full) list SPH. Audit exposure separately for each checkpoint. Variable recording duration also differs from our present ten-second input requirement.

## F. Confidence and evaluation under rare positives

### 19. On Calibration of Modern Neural Networks

**Citation:** Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. *On Calibration of Modern Neural Networks.* **ICML 2017**, PMLR 70:1321–1330.

**Source:** [Paper](https://proceedings.mlr.press/v70/guo17a.html).

- **Idea:** Investigate the mismatch between neural-network confidence and observed correctness, and evaluate post-hoc calibration, including temperature scaling.
- **Why we should read it:** The requested confidence percentage needs evaluation independently of classification accuracy.
- **Our interpretation:** Keep calibration patients separate from model fitting and final testing. Our current implementation uses Platt scaling, not a reproduction of every method in this paper.
- **Caution:** Calibration on PTB-XL does not establish calibrated disease probabilities for students or protect against dataset shift.

### 20. Precision–recall evaluation with imbalanced data

**Citation:** Takaya Saito and Marc Rehmsmeier. *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets.* **PLOS ONE**, 10(3):e0118432, 2015.

**Source:** [Paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432).

- **Idea:** Explain why precision–recall analysis exposes aspects of performance that ROC curves can obscure when positives are rare.
- **Why we should read it:** The intended university population is expected to contain few positive cases, making unnecessary referrals an important practical outcome.
- **Our interpretation:** Report AUROC alongside precision–recall measures, sensitivity, specificity, and false referrals at a chosen threshold. Precision and average precision depend on prevalence; values from a hospital benchmark should not be treated as student-population estimates.

## G. Simple predictive objectives suggested for CPC

### 21. Word2vec: CBOW and skip-gram

**Citation:** Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. *Efficient Estimation of Word Representations in Vector Space.* 2013.

**Source:** [Paper](https://arxiv.org/abs/1301.3781).

- **Idea:** CBOW predicts a word from surrounding context; skip-gram predicts surrounding words from the center word.
- **ECG hypothesis:** Apply these predictive relationships to learned continuous ECG segment features. CBOW-style averaging is simple but may discard useful temporal order. Ensure context does not already contain the target through overlapping waveform support or recurrent state.

### 22. Word2vec: negative sampling

**Citation:** Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. *Distributed Representations of Words and Phrases and their Compositionality.* NeurIPS 2013.

**Source:** [Paper](https://arxiv.org/abs/1310.4546).

- **Idea:** Learn representations using logistic discrimination of observed pairs versus sampled noise pairs, alongside word-frequency subsampling and phrase representations.
- **ECG hypothesis:** Compare this logistic loss with InfoNCE using the same encoder, positive pairs, and sampled negatives. CPC already uses contrastive negative candidates; the distinct intervention is the logistic objective. Faster training or improved ECG transfer remains unproven.
- **Protocol:** The authorized [word2vec-style experiment](experiment-005-word2vec.md) compares the objectives with matched 16-negative draws. It is separate from the frozen first CPC experiment and has no completed results yet.

## H. Learned targets and adaptive token boundaries

These are separate design choices: a model can learn better discrete targets while retaining fixed temporal patches. HuBERT and ECG2TOK do not, by themselves, solve variable boundary placement.

### 23. ECG2TOK: ECG Pre-Training with Self-Distillation Semantic Tokenizers

**Citation:** Xiaoyan Yuan, Wei Wang, Han Liu, Jian Chen, and Xiping Hu. IJCAI 2025, pages 9990–9998.

**Sources:** [Official proceedings](https://www.ijcai.org/proceedings/2025/1110) · [Paper](https://www.ijcai.org/proceedings/2025/1110.pdf).

- **Idea:** Use self-distillation and online clustering to create discrete ECG prediction targets, refined through iterative training.
- **Relevance:** Direct precedent for the user's idea of improving the learned ECG vocabulary. The method still embeds fixed, nonoverlapping patches; it changes target construction rather than discovering variable temporal boundaries.
- **Project judgment:** A compact CPC encoder with an auxiliary cluster-prediction head would be a separate, simpler adaptation, not a reproduction or an established improvement. Preserve continuous features so coarse clusters do not become the only path carrying rare morphology.

### 24. Beat-Synchronous Tokenization for ECG Transformers

**Citation:** Ahmed Sameh, Nolan Wilson, Max Enderlein, and Yogatheesan Varatharajah. Preprint, 31 August 2026.

**Source:** [Paper](https://arxiv.org/abs/2608.30367).

- **Idea:** Compare fixed patches with heartbeat-aligned tokens under matched Transformer settings, including MIMIC pretraining and PTB-XL evaluation.
- **Relevance:** Direct evidence for studying physiological boundaries separately from the encoder family. Beat resampling was competitive, while their simple adaptive-pooling variant performed substantially worse; alignment alone was insufficient.
- **Project judgment:** Preserve original durations and rhythm timing, audit peak-detector failures, and test a fixed-grid fallback. The paper's classification tasks and metrics differ from our binary proxy.

### 25. Dynamic Chunking for End-to-End Hierarchical Sequence Modeling

**Citation:** Sukjun Hwang, Brandon Wang, and Albert Gu. 2025 preprint introducing H-Net.

**Source:** [Paper](https://arxiv.org/abs/2507.07955).

- **Idea:** Learn content-dependent chunk boundaries jointly with sequence modeling, replacing a separately chosen text tokenizer.
- **Relevance:** This is closer to genuinely learned boundary selection than conventional frequency-based BPE. Its evidence concerns language and other sequence settings, not our ECG task.
- **Project judgment:** An ECG boundary module would need a token-budget constraint, duration information, and tests for attention to noise or missed rare events. This is a more complex follow-up than iterative target clustering or a controlled beat-alignment study.

## I. Released recurrent ECG foundation model

### 26. BenchECG and xECG

**Citation:** Riccardo Lunelli, Angus Nicolson, Samuel Martin Pröll, Sebastian Johannes Reinstadler, Axel Bauer, and Clemens Dlaska. *BenchECG and xECG: a benchmark and baseline for ECG foundation models.* **npj Digital Medicine**, published 14 September 2026.

**Sources:** [Published paper](https://www.nature.com/articles/s41746-026-03196-y_reference.pdf) · [Official repository](https://github.com/dlaskalab/bench-xecg) · [Released weights and model card](https://huggingface.co/riccardolunelli/xECG_base_model_v1).

- **Idea:** BenchECG evaluates transfer across varied ECG tasks. xECG combines a bidirectional xLSTM encoder with SimDINOv2 self-supervised pretraining. The current published paper reports a 0.838 fine-tuned BenchECG score; its older arXiv version contains different numbers, so use the published version for this claim.
- **Availability:** The authors link a 57-million-parameter, 228 MB safetensors checkpoint and code for a pooled signal-level classification head. The model is designed for 100 Hz raw ECG with canonical 12-lead order. The model card lists MIT licensing; the repository also limits its intended use to research rather than clinical care.
- **Pretraining exposure:** The published source list contains full CODE, Chapman and Ningbo, and INCART, with no PTB-XL or MIMIC-IV-ECG listed. Patient or waveform overlap with our PTB split has not been independently audited. CODE-15% overlaps full CODE.
- **Our interpretation:** This is a useful released reference for direct fine-tuning on the existing binary PTB proxy. [Experiment 007](experiment-007-xecg.md) queues full-label and 10%-label runs after Experiment 006. The checkpoint is downloaded and CPU backend checks passed; GPU profiling and training remain queued, with no project result yet.
- **Caution:** A comparison with our earlier models cannot isolate architecture or pretraining effects, and the repeatedly examined PTB test set is exploratory rather than a fresh external holdout.

## J. Recent vision self-supervision for ECG adaptation

### 27. DINOv3

[Paper](https://arxiv.org/abs/2508.10104) · [Official code](https://github.com/facebookresearch/dinov3).
Its Gram-anchoring idea motivates preserving relationships among local ECG features during continued training. Our [Experiment 008](experiment-008-vision-ssl.md) tests a temporal adaptation with a frozen xECG reference, not image weights or a reproduction of the complete DINOv3 training system. Gains in vision do not imply gains on ECGs.

### 28. LeJEPA

[Paper](https://arxiv.org/abs/2511.08544) · [Author code](https://github.com/galilai-group/lejepa).
Combines view agreement with SIGReg distribution regularization. A useful future objective comparison can hold the ECG encoder and views fixed. The xECG repository already includes a LeJEPA path, so applying the name alone is not a novelty claim. This objective comparison is researched but not part of Experiment 008's four runs.

### 29. V-JEPA 2.1

[Paper](https://arxiv.org/abs/2603.14482) · [Official code](https://github.com/facebookresearch/vjepa2).
Dense latent prediction and supervision of visible as well as masked regions offer ideas for retaining short ECG patterns. Our visible-token reference loss uses a different ECG-specific setup; it is not a V-JEPA reproduction. See the [research review](vision-to-ecg-research.md) for candidate methods, controls, and compute tradeoffs.

## K. Independent architectures from NLP, genomics and vision

**Status: researched on 24 September 2026. KDA/CKDA, the genomic hybrid and Mamba-3 are now authorized and queued for implementation as Experiments 011–013.** None has run or been attached to the automatic GPU scheduler yet. See the [persistent queue](experiment-queue.md) and [cross-domain designs](cross-domain-architecture-candidates.md). These are independent model families, separate from xECG adaptation and the CPC objective/readout studies.

### 30. Gated DeltaNet-2

*Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention.* Technical report, 21 May 2026. [Paper](https://arxiv.org/abs/2605.22791) · [Official code](https://github.com/NVlabs/GatedDeltaNet-2).

Separates channel-wise erase and write control in recurrent linear attention. The reported language/retrieval comparisons motivate testing memory updates, but do not establish gains for compact ECG models. A waveform adaptation needs a new signal stem, matched initialization/data/objective and a V100 backend check. This is a follow-up candidate, not an additional immediate experimental arm.

### 31. Kimi Linear and Kimi Delta Attention

*Kimi Linear: An Expressive, Efficient Attention Architecture.* 2025. [Authors' repository and technical report](https://github.com/MoonshotAI/Kimi-Linear).

KDA refines gated delta memory with channel-wise forgetting; the published model combines it with attention. Our proposed test uses a small KDA sequence mixer learned on ECG, not the released 48B-parameter language model. Ordinary KDA is the required comparator for any CKDA range-extension claim.

### 32. Complex KDA

*Complex KDA: Understanding and Enhancing the Expressivity of Kimi Delta Attention.* Preprint, **21 September 2026**. [Paper](https://arxiv.org/abs/2609.24797) · [Official code](https://github.com/OpenEuroLLM/ComplexKDA).

Extends the channel gate to `[-1,1]` and the delta coefficient to `[0,2]`, allowing signed transitions and rotational state dynamics. The periodic-waveform experiment makes this a focused ECG hypothesis. Its official repository also states that the toy task uses shifts of one fixed groove and that GRU is strongest on that task; language performance is similar to ordinary KDA. Thus the evidence motivates a controlled test, not an expectation of beating our GRU or a claim that the method has never been used on signals.

### 33. RWKV-7

*RWKV-7 “Goose” with Expressive Dynamic State Evolution.* 2025. [Paper](https://arxiv.org/abs/2503.14456) · [Official code](https://github.com/RWKV/RWKV-LM).

A generalized delta recurrence with dynamic state evolution offers another route to a compact ECG context model. The source results concern language/state tracking. Preserve waveform tokenization and a matched training objective when evaluating its recurrence; do not infer ECG usefulness from text benchmarks or transfer vocabulary embeddings as physiological categories.

### 34. Mamba-3

*Mamba-3: Improved Sequence Modeling using State Space Principles.* ICLR 2026; preprint 16 March 2026. [Paper](https://arxiv.org/abs/2603.15569).

Combines a richer discretized recurrence, complex-valued state evolution and multi-input/multi-output modeling. Oscillatory state dynamics are a plausible signal hypothesis; the paper's language gains and inference efficiency do not establish a benefit for short ECG sequences or our V100. It is authorized as [Experiment 013](experiment-013-mamba3-plan.md), with implementation and device compatibility checks pending.

### 35. Evo 2 and StripedHyena 2

*Genome modelling and design across all domains of life with Evo 2.* **Nature**, 4 March 2026. [Paper](https://www.nature.com/articles/s41586-026-10176-5) · [Official code](https://github.com/ArcInstitute/evo2).

StripedHyena 2 mixes short, medium and long input-dependent convolution operators with attention. A compact ECG adaptation could test whether that mixture helps local morphology and relationships across beats. This transfers a model design; it does not treat DNA weights or genomic token likelihoods as ECG knowledge. Genomic sequence lengths, data volume and model size differ greatly from our task.

### 36. TiViT

*Unlocking Pretrained Vision Transformers for Time Series Classification.* GCPR 2026 Oral. [Paper](https://arxiv.org/abs/2506.08641) · [Official code](https://github.com/ExplainableML/TiViT).

Stacks time-series segments into images and probes intermediate frozen vision features. The current repository lists OpenCLIP, SigLIP 2, DINOv2 and MAE. This supplies an inexpensive cross-domain-weight pilot, subject to a faithful twelve-lead input conversion and pretrained-versus-random feature control. Its reported TiViT-only benchmark average exceeds Mantis on UCR but is lower on UEA, so avoid a universal superiority claim. See the [independent vision-model proposals](vision-to-ecg-architecture-candidates.md).

## How these papers change our next decisions

The following are project judgments, not results reported by the cited authors:

1. **Performance reference:** Keep released ECG-JEPA and a supervised CNN in the comparison. Our existing results make them more informative references than assuming a newer architecture must win.
2. **Architecture alternative:** Investigate the benchmark's ECG-CPC implementation as a concrete state-space option with released resources. Compare it under the same labels and evaluation rules.
3. **Larger unlabeled pool:** Finish the matched-update ECG-FM comparison before interpreting the benefit of the 200,000 additional MIMIC recordings.
4. **Reports:** MERL is an accessible starting point once appropriate ECG/report pairs are available. ECG-CLIP provides a recent complementary study, but its implementation availability is less clear.
5. **Custom architecture:** Read CoRe-ECG, ECG-NAT, ACL-ECG, and lead-agnostic learning before making an originality claim. New combinations need specific ablations and useful measured gains.
6. **Confidence and screening:** Include calibration and false-referral evaluation in model selection. A high benchmark AUROC alone does not establish performance for the intended young population.

## Related project documents

- [Complete model findings report](model-findings-report.md)
- [Initial experiment and label definition](experiment-001.md)
- [Larger MIMIC experiment](experiment-003-mimic.md)
- [Custom architecture and prior-work comparison](custom-architecture.md)
- [Published ECG-JEPA checkpoint details](jepa-notes.md)
- [HuBERT and ECG-FM implementation notes](pretrained-notes.md)
- [Public-data strategy](public-data-strategy.md)
- [Independent architectures from NLP, genomics and vision](cross-domain-architecture-candidates.md)
- [Independent vision-model proposals](vision-to-ecg-architecture-candidates.md)
