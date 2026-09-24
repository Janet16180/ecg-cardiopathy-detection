# ECG screening research: complete model findings

**Initial report:** 23 September 2026; **latest update:** 24 September 2026, 17:27 UTC  
**Study:** Public-data experiments for a master's project on ECG screening with limited local annotations  
**Scope:** The original 37-run analysis below, with subsequent study updates and linked reports. The original ranking is historical, not the current complete ranking.

## Morphology templates: Experiment 017 complete, preliminary positive signal

All six five-epoch runs completed at 17:31:38 UTC. On the 1,306-record development cohort:

| Arm | Full-label AUROC | 1,518-label AUROC | Limited-label fold specificity | Limited-label fold sensitivity |
| --- | ---: | ---: | ---: | ---: |
| No-branch CPC | 0.9571 | 0.9431 | 0.6134 | 0.9490 |
| Matched convolution | 0.9620 | 0.9462 | 0.6609 | 0.9502 |
| Morphology templates | 0.9619 | 0.9487 | 0.6868 | 0.9478 |

The template arm improved limited-label AUROC by 0.00256 over matched convolution and 0.00562 over no branch, meeting the frozen development screen. Its fold specificity was 2.59 percentage points above convolution while sensitivity was 0.24 points lower. At full labels, template and convolution AUROC were essentially tied; convolution had higher specificity, with slightly lower sensitivity. Thus the results support testing the added local branch, with a possible limited-label benefit from distance matching. They do not establish template superiority at both budgets or a new state of the art.

A matched second seed and artifact/template dominance checks are required before calibration/test. No calibration or test predictions were made. Epoch choice uses development AUROC; patient-fold threshold evaluation does not eliminate model-selection optimism. Completion hashes were verified. [Full report](../outputs/experiment017_morphology_templates_v2/report.md). The queue is complete and the follow-up has not been scheduled.

## Cached teacher distillation: Experiment 015 complete

All four matched five-epoch runs finished. Full-label development AUROC was **0.9575 control versus 0.9591 distillation**; patient-fold specificity was 0.6955 versus 0.7084. At 1,518 labels, AUROC was **0.9378 versus 0.9361**, and specificity was 0.6026 versus 0.5853. The small full-label gain fell below the predefined AUROC/specificity criteria, and the limited-label result declined. The prespecified screen failed; no second seed, calibration fit or test prediction followed. This single-seed result addresses this global cached-feature loss, not every form of distillation. [Matched development report](../outputs/experiment015_jepa_cpc_distillation/report.md).

The loading and recovery fixes also passed real GPU checks. Training took about 159 seconds summed over the four arms, following a roughly six-minute cold RAM preload during profiling; the separate training process needed only 5.34 seconds to preload. Corrected Experiment 017 subsequently completed; its results and required replication are reported above.

## Latest inexpensive screen: Experiment 014

Cached JEPA + ordinary frozen CPC score fusion did not pass its development gate at either label budget. Both selected JEPA alone. Patient-fold specificity was 0.7473 with full labels and 0.6998 with 1,518 labels; the 75%-JEPA blend reached 0.7430 and 0.6825, respectively. These are development estimates at thresholds selected on other development patient folds, not new test results. No calibration fit or test predictions were made. The CPC constituent was a frozen probe, not the stronger fine-tuned CPC encoder. The CPU screen completed in 19.95 seconds after preflight and a matched limited-label probe refit. [Full report and verification](../outputs/experiment014_jepa_cpc_fusion/report.md).

## Latest completed studies: 24 September

The compact CPC studies use 56,875 training ECGs. [Experiment 004](../outputs/experiment004_cpc_40k/report.md) demonstrated an improvement from unlabeled CPC pretraining at the 10% label budget. [Experiment 005](../outputs/experiment005_cpc_word2vec/report.md) found no clear advantage for word2vec-style negative sampling over sampled InfoNCE. The newly completed findings are:

### Tokenization: learned boundaries help within the chunk family

| Experiment 006 arm | Full-label test AUROC | 10%-label test AUROC |
| --- | ---: | ---: |
| Native-grid CPC continuation | 0.957 | 0.924 |
| Auxiliary learned cluster targets | 0.955 | 0.925 |
| Fixed chunks | 0.946 | 0.913 |
| Heartbeat-aligned chunks | 0.944 | 0.915 |
| Learned chunks | 0.951 | 0.919 |

Learned chunks exceeded fixed chunks by **+0.0054** with full labels (paired patient 95% CI **+0.0013 to +0.0098**, rounded) and **+0.0066** with 10% labels (CI **+0.0013 to +0.0121**). Both chunk variants remained below the native-grid continuation point estimate. Fixed chunking reduced AUROC versus native continuation at both budgets, with intervals excluding zero. Neither heartbeat boundaries nor auxiliary cluster targets showed a clear AUROC benefit over their respective matched controls. Chunk/native comparisons also change temporal compression and readout; they do not isolate boundary choice alone. [Full results and protocol](../outputs/experiment006_cpc_tokenization/report.md).

### Frozen CPC readout: ordinary local features beat prediction mismatch

| Experiment 009 feature set | Full-label test AUROC | 10%-label test AUROC |
| --- | ---: | ---: |
| Context only | 0.9192 | 0.9096 |
| Context + ordinary local features | 0.9322 | 0.9182 |
| Context + prediction mismatch | 0.9247 | 0.9100 |

The mismatch branch underperformed the equally wide ordinary-feature branch by **−0.0075** (full-label CI **−0.0134 to −0.0014**) and **−0.0082** (10%-label CI **−0.0130 to −0.0033**). Adding ordinary features improved over context-only pooling, although that comparison also increases classifier width. This supports retaining local feature information in a frozen probe; it does not show that the prediction-mismatch mechanism helps. These frozen logistic probes should not be interpreted as a direct architecture comparison against the end-to-end fine-tuned models. [Full results](../outputs/experiment009_cpc_prediction_mismatch/report.md).

Both studies use one training seed and the previously inspected PTB-XL test cohort. Patient-bootstrap intervals capture patient sampling uncertainty, not retraining variability. Neither validates a healthy-person label or university referral performance.

### Released xECG fine-tuning: complete

| Experiment 007 budget | Test AUROC | Sensitivity | Specificity | Brier |
| --- | ---: | ---: | ---: | ---: |
| Full labels | 0.9374 | 95.06% | 59.77% | 0.0964 |
| 10% labels | 0.9283 | 94.73% | 60.63% | 0.1040 |

Both runs finished at 04:53 UTC. The full-label run selected epoch 1 of 9; the 10% run selected epoch 2 of 10. Later epochs failed to improve development AUROC despite falling training loss. The best checkpoints were retained for evaluation. The observed scores do not exceed our strongest compact CPC full-label point estimates, but differences in pretraining and optimization prevent attribution to architecture alone. Fine-tuning instability or overfitting remains a hypothesis, not a diagnosed cause. [Full report and checked artifact hashes](../outputs/experiment007_xecg/report.md).

Experiment 008 passed its corrected microbatch-4 profile, but its roughly 27.7-hour forecast led to deferral. Experiment 010 was also deferred after actual disk I/O invalidated its short profile; one complete native SSL epoch is preserved. Neither has a classification result. A new bounded RAM loader passed full-epoch GPU profiling for Experiment 015; its matched distillation pilot has completed, as reported above. Experiment 017 completed after a verified checkpoint-comparison correction; its development findings are above. These implementation/profile checks are not new model-performance findings. See the [persistent queue](experiment-queue.md) for live execution and recovery status. Downloaders continue independently.

## Original 37-run executive summary (23 September)

The strongest observed system was the **released ECG-JEPA encoder with a linear classifier trained on all 15,360 eligible public training labels**. On the fixed 1,896-record PTB-XL test set, it achieved **AUROC 0.9545**, **95.2% sensitivity**, **71.8% specificity**, and **Brier score 0.0804**. Its AUROC patient-cluster bootstrap interval was **0.9455–0.9629**. This is the leading practical baseline from these experiments, not a statistically established universal architecture winner.

The main findings are:

1. **Using available public labels helped.** Increasing the exposed training labels from 1,518 to 15,360 improved observed AUROC for each of the three systems tested at both budgets: CNN, frozen ECG-FM, and frozen ECG-JEPA.
2. **Published ECG-JEPA provided the strongest frozen representation in this study.** With 10% of eligible training patients labeled, its mean AUROC was 0.9420 across three label samples, compared with 0.9226 for ECG-FM and 0.9194 for HuBERT-small.
3. **A small supervised CNN was a strong reference.** Its mean AUROC of 0.9217 exceeded the locally pretrained compact transformer variants. Self-supervision was useful in some matched comparisons, but did not automatically produce the strongest practical system.
4. **Additional ECG-FM self-supervision showed no clear AUROC benefit.** Both adaptation contrasts against direct fine-tuning had paired confidence intervals spanning zero. Higher specificity occurred alongside lower test sensitivity.
5. **The small Georgia pooling pilot did not establish a benefit from more unlabeled sources.** Adding 948 accepted ECGs produced an AUROC change of −0.00103 versus PTB-XL-only adaptation, with a paired 95% interval of −0.00239 to +0.00028. This narrow pilot cannot settle the value of substantially larger, balanced multicenter training.
6. **The custom architecture produced a negative result.** Adding a lead-difference target improved average AUROC relative to ordinary latent prediction, but both SSL variants underperformed the same encoder trained from scratch.

These results concern an **ECG annotation proxy**, not confirmed heart disease, overall health, or cardiologist referral need. The retained test cohort is much older and has far more abnormal ECGs than the intended university population. The measured operating points do not yet justify a student-screening deployment.

## 1. Research objective and scope

The intended application is nurse-led acquisition of raw 12-lead ECGs, followed by a binary recommendation about further assessment. The university dataset is expected to contain approximately 20,000 recordings, with cardiologist text reports for only about 1%; access is pending. This study therefore used public signals and simulated limited annotation by hiding training labels.

The study tested four questions:

- Which available encoder performs best under a 10% training-patient label budget?
- Does learning from the remaining training waveforms improve the downstream classifier?
- Do a custom architecture or an additional public waveform source improve performance?
- What changes when all eligible public training labels are used?

The user explicitly allowed public labels and larger public datasets to supplement scarce local labels. The 10% setting is a controlled benchmark; it is not a restriction that must be imposed on the eventual public-data training strategy. All signals used here were real recordings. No synthetic ECG waveforms were generated.

The 37 runs comprise 12 original compact-model runs, nine frozen published-encoder probes, two direct published-encoder fine-tunes, two ECG-FM adaptation runs, nine custom-architecture runs, and three full-public-label runs. Hyperparameter candidates and interrupted runtime profiling are not counted as separate evaluated runs.

## 2. Data, labels, and isolation

### 2.1 PTB-XL

PTB-XL v1.0.3 contains 21,799 ten-second 12-lead ECGs and provides patient identifiers, structured ECG statements, reports, and recommended folds. We used the released 100 Hz and 500 Hz waveform versions according to each encoder's requirements. Dataset background is documented by [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/); the counts below come from this experiment's manifests.

| Partition or budget | ECG recordings | Role |
| --- | ---: | --- |
| Official training folds 1–8 | 17,418 | Waveforms available for our self-supervision |
| Eligible labeled training records | 15,360 | Maximum public-label supervised budget |
| Seed-42 exposed training labels | 1,518 | 913 abnormal-proxy and 605 normal-proxy labels |
| Seed-42 training records treated as unlabeled | 15,900 | Includes hidden eligible labels and unresolved cases |
| Development subset of fold 9 | 1,306 | Checkpoint and hyperparameter selection |
| Calibration subset of fold 9 | 564 | Probability calibration and operating threshold |
| Retained fold-10 test set | 1,896 | Final evaluation; 1,195 positives and 701 negatives |

The labeled seed-42 subset contains 1,335 of 13,352 eligible training patients: approximately 10%. Because patients can contribute multiple ECGs, this corresponds to 9.88% of eligible training recordings and 8.72% of all training recordings. Seeds 43 and 44 use different patient samples, with slightly different record counts.

The main seed-42 experiment uses **5,284 annotated ECGs in total** when development, calibration, and test labels are counted. The full-label experiment uses **19,126** across those partitions. Existing annotations are also consulted to define eligibility. Consequently, this is retrospective label masking, not proof that only 10% of a new cohort would need annotation.

### 2.2 Binary target

The implemented target is a conservative diagnostic-abnormality proxy:

- **Positive:** at least one diagnostic code in a non-NORM superclass, with no NORM code.
- **Negative:** an explicit NORM code and no other codes except SR, indicating sinus rhythm.
- **Unresolved:** other combinations, including rhythm-only records, conflicting normal/abnormal statements, and absent diagnostic statements.

Unresolved training waveforms can still support SSL. Unresolved evaluation records were excluded: 313 from validation and 302 from test. Test coverage is therefore **86.3%**, or 1,896 of the 2,198 original test-fold ECGs. This selection may simplify the task, and it relies on annotations that would not be available to decide eligibility at deployment. A binary operational system has not been validated on the excluded cases.

The negative label means that the retained annotation satisfies this rule; it does not establish that the person is healthy. Some clinically relevant rhythm findings are outside this provisional target. Local cardiologist reports will require a reviewed mapping to the project's intended endpoint.

### 2.3 Georgia pilot

We downloaded all 999 records in the `g1` folder of the [Georgia Challenge 2020 collection](https://physionet.org/files/challenge-2020/1.0.2/training/georgia/). Official file checksums were verified. The signal contract required 12 standard leads, millivolt units, 500 Hz, 5,000 samples, and finite values.

Ten recordings failed the duration requirement. Another 41 were exact decoded-waveform duplicates against the accumulated audit/pool. All 21,799 PTB-XL waveforms were included in that identity audit. This left **948 additional records**, producing a combined SSL pool of **18,366 ECGs**.

Georgia diagnoses were not used. Patient identities were unavailable, so Georgia grouping identifiers represent recordings rather than verified individual people. The duplicate audit does not establish cross-source patient independence. This was a convenience subset, not a random sample or the complete Georgia dataset.

## 3. Evaluation protocol

Training, validation, and test patients were separated using the official PTB-XL folds. Validation was further divided by patient into development and calibration subsets. Development AUROC selected checkpoints and linear-probe regularization. Test labels were not used by the optimization code.

Logistic, or Platt, calibration was fitted on the calibration subset. The operating threshold was the highest threshold achieving at least **95% sensitivity on calibration ECGs**. This is an illustrative research operating point; it does not guarantee 95% sensitivity on the test set, on patients as aggregated units, or on future students.

| Metric | Meaning in this report |
| --- | --- |
| AUROC | Ability to rank positive-proxy ECGs above negative-proxy ECGs across thresholds; higher is better |
| Average precision (AP) | Precision–recall summary; strongly affected by prevalence; higher is better |
| Sensitivity | Fraction of positive-proxy test ECGs flagged at the fixed threshold |
| Specificity | Fraction of negative-proxy test ECGs not flagged at that threshold |
| Brier score | Mean squared probability error against the binary proxy; lower is better |
| ECE | Descriptive binned calibration error; dependent on the binning procedure |

All outcomes are **ECG-level**. Confidence intervals use 500 patient-cluster bootstrap resamples, retaining each sampled patient's ECGs together. The intervals are conditional on fitted models and selected thresholds; they do not include retraining uncertainty, calibration/threshold estimation uncertainty, or domain shift. Paired adaptation comparisons align ECG IDs, patient IDs, and labels before resampling.

Three label seeds measure variation in exposed training patients and downstream optimization. They reuse the same evaluation cohort. The compact SSL runs also share one unlabeled pretraining initialization across label seeds. These are not three independent external validations.

The overall study is **exploratory**: later experiments were designed after earlier results were inspected. Per-run development/test separation does not turn the whole evolving comparison into a prospectively locked benchmark. Paired intervals are not adjusted for multiple comparisons.

## 4. Models and training methods

### 4.1 Supervised CNN and transformer references

The CNN uses three convolutional stages with 32, 64, and 128 channels, GroupNorm, GELU, and mean/max pooling. Including the classifier, it has **184,001 parameters**. The scratch transformer uses joint 12-lead 250 ms patches, 40 tokens per recording, width 96, four attention heads, and three transformer layers; its classifier has **257,953 parameters**.

Both receive 100 Hz signals, per-record lead demeaning, and training-only per-lead RMS scaling. Compact downstream runs use a maximum of 40 epochs and ten-epoch early-stopping patience. These references test whether more complex pretraining actually improves on modest supervised models at the same label budget.

### 4.2 Compact MAE and compact JEPA-inspired model

The masked autoencoder uses the same compact transformer and reconstructs hidden waveform patches. The compact JEPA-inspired model instead predicts masked EMA-teacher latent targets, using a representation variance penalty. Both mask half the time patches in contiguous 500 ms spans, receive 50 SSL epochs on training waveforms only, and then use the matched downstream transformer optimizer.

These are local compact implementations. They are **not reproductions of the released ECG-JEPA model** or other published masked-ECG systems.

### 4.3 Released HuBERT-small, ECG-FM, and ECG-JEPA

The released encoders were evaluated through frozen feature extraction followed by standardized, regularized logistic regression. The regularization parameter was selected on development AUROC from six predefined values. HuBERT-small and ECG-FM were also fine-tuned end to end at seed 42, using a backbone learning rate of 1e-5, classifier learning rate of 1e-3, batch size 16, a 20-epoch maximum, and patience five.

| Released encoder | Input used in this study | Representation |
| --- | --- | --- |
| HuBERT-small | Official filtering/min–max pipeline from 500 Hz; two five-second views, flattened by lead and decimated | Mean token features per view, then mean across views; 512 dimensions |
| ECG-FM | 500 Hz, 12 leads, lead-wise z-score over ten seconds, then two five-second views | Mean token features per view, then mean across views; 768 dimensions |
| Published ECG-JEPA | 500 Hz waveform resampled to 250 Hz; I, II, V1–V6; no amplitude normalization | Authors' mean-pooled encoder representation; 768 dimensions |

Official implementations and weights are documented by the [HuBERT-ECG authors](https://github.com/Edoar-do/HuBERT-ECG), [ECG-FM authors](https://github.com/bowang-lab/ECG-FM), and [ECG-JEPA paper](https://arxiv.org/abs/2410.08559). Exact checkpoint hashes, source revisions, and preprocessing are recorded in the local [pretrained notes](pretrained-notes.md) and [ECG-JEPA notes](jepa-notes.md).

Only HuBERT-small was tested here. The findings do not establish the performance of HuBERT-base, HuBERT-large, or supervised Cardio-Learning checkpoints. Published ECG-JEPA was evaluated as a frozen encoder; end-to-end fine-tuning was not tested.

### 4.4 Continued ECG-FM self-supervision

Two additional runs started from the same released ECG-FM checkpoint: one adapted to PTB-XL training waveforms, the other to PTB-XL plus accepted Georgia waveforms. Both used one fixed pass, learning rate 1e-6, batch size 32, and a CMSC-style symmetric contrastive objective between the two temporal views. Known same-patient PTB-XL pairs were positives. The objective is a simplified adaptation experiment, not the complete original ECG-FM pretraining loss.

The one-epoch budget was selected from runtime profiling before adapted-model downstream evaluation. The pools required 545 and 574 optimizer updates respectively, so pooling also added some compute. Downstream fine-tuning settings and exposed labels matched direct ECG-FM fine-tuning. Encouraging agreement between two recording halves could suppress transient information present in only one half; that is a possible limitation of the objective, not a demonstrated explanation for the results.

### 4.5 Custom lead-aware multiscale architecture

The custom encoder uses eight leads, 400 ms morphology patches, one-second rhythm windows, cross-attention, and two temporal transformer layers shared across leads. It has **204,864 encoder parameters**, or **205,441 including the classifier**. Three matched conditions were tested: scratch training, ordinary masked latent prediction, and latent prediction with an additional lead-difference target.

The additional target predicts the difference between a teacher's representation for one lead and the mean representation of the other leads at that time. Teacher targets are centered within each training batch separately at each lead/time coordinate to reduce trivial prediction of static position identity. One lead and two additional two-second lead spans are masked before both student branches. The teacher sees the full preprocessed waveform and receives no gradients.

Both SSL conditions use 20 epochs and batch size 256. Tests verify that changing masked preprocessed values cannot alter student features and that a constant lead/time teacher pattern produces zero centered targets. Between-record teacher and residual variance remained nonzero. Those checks address simple collapse or visibility failures; they do not establish that the learned features are useful.

This is a custom hypothesis with **unverified novelty**. Multiscale processing, lead-aware learning, and latent prediction already have related literature. The [architecture report](custom-architecture.md) documents prior work and exact implementation details.

## 5. Results at the 10% label budget

### 5.1 Primary label sample: seed 42

The following table uses the same 1,518 labeled training ECGs and 1,896 test ECGs for all rows. Rankings are descriptive. Pretraining histories, parameter counts, input representations, and compute are not matched across all families.

| Model | AUROC (95% patient-cluster CI) | AP | Sensitivity | Specificity | Brier |
| --- | --- | --- | --- | --- | --- |
| Published ECG-JEPA, frozen | 0.9407 (0.9303–0.9511) | 0.9693 | 94.0% | 67.2% | 0.0928 |
| ECG-FM, adapted + fine-tuned | 0.9315 (0.9207–0.9419) | 0.9636 | 94.2% | 61.3% | 0.1025 |
| ECG-FM, pooled adaptation | 0.9305 (0.9192–0.9411) | 0.9632 | 94.4% | 61.9% | 0.1033 |
| ECG-FM, fine-tuned | 0.9289 (0.9174–0.9403) | 0.9626 | 95.6% | 50.5% | 0.1038 |
| HuBERT-small, fine-tuned | 0.9270 (0.9146–0.9379) | 0.9613 | 95.7% | 50.4% | 0.1048 |
| CNN, supervised | 0.9233 (0.9115–0.9346) | 0.9595 | 94.5% | 55.9% | 0.1080 |
| ECG-FM, frozen | 0.9230 (0.9112–0.9353) | 0.9595 | 94.6% | 58.8% | 0.1084 |
| HuBERT-small, frozen | 0.9204 (0.9080–0.9332) | 0.9560 | 95.3% | 54.1% | 0.1101 |
| Compact JEPA + fine-tuning | 0.9069 (0.8937–0.9205) | 0.9479 | 95.1% | 48.4% | 0.1196 |
| Transformer, supervised | 0.8914 (0.8767–0.9061) | 0.9372 | 94.3% | 49.2% | 0.1299 |
| Compact MAE + fine-tuning | 0.8891 (0.8747–0.9048) | 0.9365 | 93.8% | 49.4% | 0.1313 |
| Lead multiscale, supervised | 0.8877 (0.8729–0.9032) | 0.9394 | 93.5% | 47.2% | 0.1317 |
| Lead multiscale, innovation SSL | 0.8794 (0.8635–0.8950) | 0.9348 | 93.1% | 45.9% | 0.1373 |
| Lead multiscale, latent SSL | 0.8650 (0.8482–0.8824) | 0.9286 | 92.1% | 40.8% | 0.1438 |

### 5.2 Variation across three label samples

Mean ± sample standard deviation is reported only where all three seeds were evaluated. The seed-specific values avoid hiding unfavorable runs.

| Model | Seed 42 AUROC | Seed 43 AUROC | Seed 44 AUROC | Mean AUROC ± SD | Mean sensitivity | Mean specificity |
| --- | --- | --- | --- | --- | --- | --- |
| Published ECG-JEPA, frozen | 0.9407 | 0.9455 | 0.9397 | 0.9420 ± 0.0031 | 93.5% | 70.6% |
| ECG-FM, frozen | 0.9230 | 0.9219 | 0.9228 | 0.9226 ± 0.0006 | 94.2% | 58.5% |
| CNN, supervised | 0.9233 | 0.9228 | 0.9190 | 0.9217 ± 0.0023 | 95.1% | 52.6% |
| HuBERT-small, frozen | 0.9204 | 0.9216 | 0.9163 | 0.9194 ± 0.0028 | 94.7% | 56.5% |
| Compact JEPA + fine-tuning | 0.9069 | 0.9080 | 0.9010 | 0.9053 ± 0.0038 | 95.0% | 48.5% |
| Transformer, supervised | 0.8914 | 0.9008 | 0.8975 | 0.8966 ± 0.0048 | 94.6% | 48.5% |
| Compact MAE + fine-tuning | 0.8891 | 0.8876 | 0.8765 | 0.8844 ± 0.0069 | 93.9% | 48.0% |
| Lead multiscale, supervised | 0.8877 | 0.8773 | 0.8858 | 0.8836 ± 0.0055 | 92.6% | 48.6% |
| Lead multiscale, innovation SSL | 0.8794 | 0.8632 | 0.8619 | 0.8682 ± 0.0098 | 94.1% | 41.3% |
| Lead multiscale, latent SSL | 0.8650 | 0.8542 | 0.8637 | 0.8610 ± 0.0059 | 93.1% | 36.9% |

### 5.3 Findings by model

**Supervised CNN.** Mean AUROC was 0.9217. It substantially exceeded the scratch transformer and both compact SSL transformer variants in this setup. A plausible interpretation is that its convolutional structure and optimization fit the available labeled sample well, but this study does not isolate that mechanism. It should remain a practical reference rather than being discarded because it is simpler.

**Supervised transformer.** Mean AUROC was 0.8966. Its performance demonstrates that adopting a transformer alone did not solve label scarcity. This is a finding about this small architecture and training configuration, not all transformers.

**Compact MAE.** Mean AUROC was 0.8844, approximately 0.0121 below its matched scratch transformer. It underperformed that reference in all three seeds. The waveform reconstruction objective did not improve downstream discrimination under the tested budget. Possible contributions from reconstruction difficulty, optimization, or target choice remain hypotheses.

**Compact JEPA-inspired model.** Mean AUROC was 0.9053, approximately 0.0087 above the matched scratch transformer, with improvement in all three seeds. This is evidence that its pretraining helped that encoder under these settings. It still fell below the CNN and released encoders; it should not be confused with the much stronger published ECG-JEPA result.

**HuBERT-small, frozen.** Mean AUROC was 0.9194. It provided useful features with a simple classifier, but did not exceed the CNN mean in this experiment. This does not test the value of all HuBERT model sizes or the complete set of methods in its paper.

**HuBERT-small, fine-tuned.** Seed-42 AUROC increased from 0.9204 for the frozen probe to 0.9270. Test sensitivity rose from 95.3% to 95.7%, while specificity fell from 54.1% to 50.4%. The single-seed comparison supports considering fine-tuning, but does not establish a reliable advantage across label samples.

**ECG-FM, frozen.** Mean AUROC was 0.9226, close to the CNN mean. It had the smallest AUROC spread among the three frozen encoders in these three runs, although three seeds are insufficient to establish broad stability. Its mean specificity was 58.5% at mean test sensitivity 94.2%.

**ECG-FM, fine-tuned.** Seed-42 AUROC increased from 0.9230 to 0.9289 compared with its frozen probe. Sensitivity increased from 94.6% to 95.6%, while specificity fell from 58.8% to 50.5%. A better ranking metric therefore did not translate into a lower false-positive rate at the separately calibrated operating point.

**ECG-FM, PTB-XL-only adaptation.** AUROC was 0.9315, versus 0.9289 without additional SSL. The paired interval includes zero, so the observed gain is inconclusive. Specificity increased to 61.3%, but sensitivity decreased to 94.2%. The adaptation did not meet a demonstrated 95% test-sensitivity requirement simply because calibration targeted 95%.

**ECG-FM, pooled adaptation.** AUROC was 0.9305, slightly below PTB-XL-only adaptation. Adding the Georgia pilot did not demonstrate a benefit. The result applies to 948 additional records, one pass, and this objective; it is not evidence against using substantially more data.

**Published ECG-JEPA, frozen.** Mean AUROC was 0.9420, the highest observed mean among the repeated 10%-label systems. Mean specificity was 70.6%, but mean sensitivity was 93.5%, below the calibration target. Its practical performance warrants further study, while its different pretraining data, scale, and preprocessing prevent attributing the result solely to architecture.

**Custom multiscale encoder, supervised.** Mean AUROC was 0.8836, below the simpler CNN and the original scratch transformer. It provides the necessary matched control for the two custom SSL objectives.

**Custom multiscale encoder, ordinary latent SSL.** Mean AUROC was 0.8610, below its scratch reference in every seed. The implemented masked objective did not help this encoder. Successful optimization and nonzero variance did not guarantee useful downstream representations.

**Custom multiscale encoder, added lead-difference objective.** Mean AUROC was 0.8682. Relative to ordinary SSL, the three paired changes were +0.01439, +0.00896, and −0.00182. The mean increase of 0.00718 was not consistent across all seeds, and this variant remained below scratch training in every seed. The tested custom SSL approach should not replace the stronger existing baselines on the basis of these results.

### 5.4 Paired adaptation comparisons

| Contrast | AUROC change | 95% paired patient-cluster interval |
| --- | --- | --- |
| ECG-FM, adapted + fine-tuned minus ECG-FM, fine-tuned | +0.00263 | [-0.00308, +0.00813] |
| ECG-FM, pooled adaptation minus ECG-FM, fine-tuned | +0.00160 | [-0.00463, +0.00765] |
| ECG-FM, pooled adaptation minus ECG-FM, adapted + fine-tuned | -0.00103 | [-0.00239, +0.00028] |

All intervals span zero. These comparisons provide no clear AUROC benefit from either adaptation or the Georgia addition. They concern discrimination; the operating-point sensitivity/specificity tradeoffs must be considered separately.

## 6. Results using all eligible public training labels

The full-label experiments expose 15,360 training ECG labels, keeping development, calibration, and test sets unchanged. Each was evaluated at seed 42.

| Model | AUROC (95% patient-cluster CI) | AP | Sensitivity | Specificity | Brier |
| --- | --- | --- | --- | --- | --- |
| Published ECG-JEPA, frozen | 0.9545 (0.9455–0.9629) | 0.9765 | 95.2% | 71.8% | 0.0804 |
| CNN, supervised | 0.9435 (0.9339–0.9533) | 0.9703 | 96.2% | 59.1% | 0.0920 |
| ECG-FM, frozen | 0.9345 (0.9229–0.9453) | 0.9657 | 93.5% | 67.0% | 0.1003 |

| Model | 1,518 labels: AUROC | 15,360 labels: AUROC | Observed change |
| --- | --- | --- | --- |
| Published ECG-JEPA, frozen | 0.9407 | 0.9545 | +0.0138 |
| CNN, supervised | 0.9233 | 0.9435 | +0.0202 |
| ECG-FM, frozen | 0.9230 | 0.9345 | +0.0114 |

All three observed AUROC changes were positive. These are single-seed changes, without a paired significance analysis for the label-budget comparison. They support using public annotations fully while maintaining the 10% setting as a controlled experiment. They do not quantify how many university annotations will ultimately be needed.

For the strongest observed system, the full-label frozen ECG-JEPA classifier, the test confusion matrix was:

| Actual proxy label | Flagged abnormal | Not flagged |
| --- | ---: | ---: |
| Abnormal | 1,138 | 57 |
| Normal | 198 | 503 |

Its test accuracy was 86.55%, precision 85.18%, and negative predictive value 89.82% at the observed 63.0% positive prevalence. **AUROC 0.9545 is not 95.45% classification accuracy.** The model still flagged 28.25% of retained negative-proxy ECGs and missed 4.77% of retained positive-proxy ECGs.

## 7. Confidence percentages and the screening use case

Calibration is implemented, but the output estimates the probability of the **defined annotation proxy in this evaluation setting**. A value of 90% is not a validated statement that a student has a 90% probability of heart disease, nor that the model is 90% certain about a clinically correct referral.

The full-label ECG-JEPA model had Brier score 0.0804 and ten-bin ECE 0.0135 on this test set. These are useful descriptive measurements, not proof of calibration in a different population. Age, prevalence, devices, acquisition quality, and clinical endpoint can all change the relevance of those scores.

### Hypothetical 1% prevalence example

Suppose, purely for arithmetic, that positive-proxy prevalence were 1% and the full-label ECG-JEPA model retained exactly its measured sensitivity and specificity. Per 1,000 people, assuming one ECG per person, the expected counts would be approximately:

- **9.5** true positive flags;
- **279.6** false positive flags;
- **0.5** missed positives;
- **289.2** total flags, with **3.3%** positive predictive value.

This is not a forecast for the university. Neither the assumed prevalence nor transportability of sensitivity/specificity has been established, and proxy positives are not confirmed disease. The example shows why a high AUROC alone does not establish a manageable referral workload. The hypothetical 1% prevalence is separate from the original 1% annotation rate.

For the intended use case, the important next evaluation is sensitivity and false referrals at a clinician-agreed operating point, alongside calibration and performance on the actual young population. That clinical operating point has not been validated here.

## 8. Pretraining exposure and interpretation limits

The HuBERT and ECG-FM released SSL checkpoints include historical PTB-XL exposure. Our downstream patient splits prevent new supervised fitting on test patients, but do not undo prior waveform exposure during released-model pretraining. Georgia was also represented in the released ECG-FM training collection. Additional Georgia adaptation therefore changes the recent training distribution; it does not establish access to a previously unseen pretraining source. Provenance is recorded in the [pretrained notes](pretrained-notes.md).

For published ECG-JEPA, the repository's pretraining commands reference Shaoxing and CODE15. We have not independently verified every dataset used for the exact released weight file, so this report does not treat it as a proven completely unexposed external-test control.

Further limitations are:

- **Population mismatch:** the retained test cohort has median age 64 and only 105 ECGs from people aged 18–30.
- **Endpoint mismatch:** annotated ECG findings are not equivalent to disease or referral need.
- **Excluded cases:** 13.7% of the original test fold is unresolved under the proxy and not evaluated.
- **Unequal pretraining scale and compute:** local SSL uses tens of thousands of signals; released models carry representations learned from much larger historical collections.
- **Limited repetition:** end-to-end foundation-model fitting, adaptation, and full-label runs have one label seed each.
- **Limited adaptation budget:** one fixed SSL epoch cannot represent exhaustive continued-pretraining optimization.
- **Small pooling increment:** 948 Georgia records add approximately 5.4% to the PTB-XL SSL pool; this is not a large-data scaling study.
- **Unknown cross-source patient identity:** exact signal deduplication cannot exclude all same-person or nonexact duplicate records.
- **Exploratory selection:** the experiment evolved after previous test results were seen; results need confirmation on a newly reserved cohort.
- **Missing evaluations:** no original 1%-label run, no local university test, no published ECG-JEPA end-to-end fine-tune, no HuBERT-base/large comparison, and no large multicenter scaling curve have been completed.

## 9. Larger-data direction

### Is HuBERT's 9.1-million-example corpus available?

The HuBERT-ECG authors report pretraining on approximately 9.1 million ECG examples assembled from 11 source datasets. Their released code and weights are available, but the pooled raw corpus is not offered as one downloadable dataset. The large Ribeiro/CODE component requires research access; its public CODE-15% subset is a separate, smaller release. The exact count of unique original acquisitions behind the stated example count should not be inferred as 9.1 million unique patients or independent ten-second recordings. See the [current HuBERT paper](https://www.medrxiv.org/content/10.1101/2024.11.14.24317328v4.full) and [authors' repository](https://github.com/Edoar-do/HuBERT-ECG).

Our use of small labeled PTB-XL subsets does **not** mean that the released encoders learned their representations only from those records. They already incorporate large historical pretraining corpora. More raw data for further pretraining is a separate experiment from using those pretrained representations or exposing more public training labels. No university recordings were used.

| Candidate | Approximate scale | Access and measured/published storage | Proposed role, not yet executed |
| --- | --- | --- | --- |
| [MIMIC-IV-ECG v1.0](https://physionet.org/content/mimic-iv-ecg/1.0/) | About 800,000 ECGs from about 160,000 patients | Waveforms are open access; about 33.8 GB compressed or 90.4 GB uncompressed | Substantially larger SSL pool, sampled by patient with a documented scale series |
| [CODE-15%](https://zenodo.org/records/4916206) | 345,779 exams | Public; approximately 46.3 GB across archive parts, before extraction | Additional SSL and carefully reviewed public-label experiments; automated normality labels are not adjudicated referral outcomes |
| [Shandong Provincial Hospital](https://springernature.figshare.com/collections/A_large-scale_multi-label_12-lead_electrocardiogram_database_with_standardized_diagnostic_statements/5779802/1) | 25,770 ECGs | Public waveform archive about 2.28 GB; variable duration | Candidate labeled source or reserved evaluation cohort after label harmonization and checkpoint-specific exposure auditing |
| Full HuBERT training collection | About 9.1 million reported examples | Multiple source releases; Ribeiro/CODE research access needed | Long-term corpus reconstruction, not an immediately available single download |

The [SPH dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9174207/) describes standardized diagnostic statements and clinical annotation review, making it worth investigating for endpoint compatibility. Its label definitions still need an explicit comparison with our proxy. SPH is included in HuBERT's historical pretraining collection, so it would not be an unseen-waveform external test for that encoder; the authors list the source in their [pretraining data description](https://www.medrxiv.org/content/10.1101/2024.11.14.24317328v1.full).

MIMIC's open waveform access should not be confused with access to every linked clinical table or cardiologist report. The proposed first use is waveform-only SSL. Released ECG-FM and HuBERT already have historical exposure to MIMIC, so a new experiment must state whether it is additional adaptation or fresh training; it must not claim a previously unexposed corpus for those weights.

At preparation of this report, the workspace had about **4.8 GB free**. The user subsequently expanded the disk; a filesystem check on 23 September 2026 confirmed approximately 202 GiB free. A **roughly 200 GB free-space target** is a practical estimate for a MIMIC-first stage using streamed waveform loading or compact caches, allowing raw signals, possible archive retention, derived data, and checkpoints. An additional full 500 Hz float32 waveform cache alone would require approximately 192 GB, so that cache is excluded from this estimate. It is not the source download size or a guarantee for every cache format. Combining multiple complete large corpora will require a revised storage budget. No MIMIC-scale or CODE-scale download or training is included among the 37 completed runs. The subsequent user-approved expansion is capped at **200,000 additional MIMIC recordings**, estimated at **24–30 GB** of raw files, rather than downloading the full corpus. Its separate [experiment 003 protocol](experiment-003-mimic.md) tracks the next comparison; preparation and running jobs are not counted as completed model findings here.

The next plan should reserve patients and evaluation cohorts before training, use staged downloads and data manifests, and measure throughput before committing to a full-corpus schedule. Disk expansion addresses storage, but training time and validation quality remain separate constraints.

## 10. Recommended next experiments

These are research priorities inferred from the measured results, not additional completed findings.

1. **Keep full-label published ECG-JEPA and the supervised CNN as reference systems.** Their practical comparison provides a stronger starting point than discarding a useful simple model in favor of architecture novelty.
2. **Test genuinely larger public training pools.** Use a predefined scaling series, for example 20k, 50k, 100k, and a larger feasible point, with matched held-out evaluation and both fixed-update and fixed-pass comparisons. This separates data-size effects from extra optimization.
3. **Use public labels wherever endpoints can be harmonized.** For incompatible source labels, use SSL or source-specific training targets; do not equate sinus rhythm with normality. If multi-task labels are used for training, the eventual prediction interface can still be binary.
4. **Evaluate stronger transfer settings.** Published ECG-JEPA fine-tuning, larger HuBERT encoders, and carefully scoped supervised pretrained weights are candidates. Any checkpoint that already used PTB-XL labels needs a different unexposed downstream evaluation for a credible new test.
5. **Protect a local evaluation cohort once access arrives.** Define the report-to-label rule with the clinical collaborator, separate patients before fitting, and reserve data for calibration and final evaluation. Using local test waveforms in SSL would make that evaluation transductive; preserve a separate untouched cohort for future-student claims.
6. **Prioritize the intended screening operating point.** Measure false referrals, missed findings, calibration, and robustness to acquisition problems. The current high-sensitivity operating points have substantial false-flag rates whose clinical acceptability remains unestablished.
7. **Pursue custom architecture changes through controlled ablations.** The tested lead-difference objective is a documented negative result. Any revised objective should be selected on development data and confirmed without repeatedly selecting favorable outcomes from the current test set.

## 11. Reproducibility and verification

Runs used a Tesla V100 16 GB GPU and PyTorch 2.6.0 with CUDA 12.4. Released-model dependencies were isolated in a separate environment. Model weights, configuration files, histories, calibration artifacts, probabilities, and per-run metrics are stored under `outputs/experiment001/` and `outputs/experiment002_public_labels/`. Public waveforms and large artifacts are git-ignored.

Verification included official download checksums, patient split checks, hidden-label manifest checks, probability/threshold metric tests, clustered bootstrap checks, finite-gradient checks, teacher-gradient isolation, and masked-input visibility checks. The automated unit suite passed **13 tests** during the experiment. Scientific and code reviews checked preprocessing and provenance. These checks support implementation integrity; they do not constitute clinical validation.

The original three-seed feature-extraction union files contained label columns, but the extractors copied only identifiers and waveform paths and did not provide labels to the encoders. Each probe fitted only its own original labeled training manifest. The newer feature-union preparation utility explicitly emits label-free manifests. This distinction is retained rather than rewriting earlier provenance files.

Key artifacts:

- [Compact results and plots](experiment001-results.md)
- [10%-label metrics for all 34 runs](experiment001-metrics.json)
- [Full-public-label metrics for three runs](experiment002-metrics.json)
- [Paired adaptation comparisons](paired-adaptation-comparisons.json)
- [Custom architecture methods and findings](custom-architecture.md)
- [Public-data pooling strategy](public-data-strategy.md)
- [Reproduction instructions](../README.md)

The existing `experiment001-results.pdf` is the comparison **figure**, not a PDF of this complete narrative report.

![Primary 10%-label comparison: ranking, operating points, ROC, and calibration](experiment001-results.png)

## Appendix A. Complete evaluated-run ledger

Each row represents one final evaluation after development-based selection. `10%` denotes patient-level label sampling; its exact training record count appears separately. The full-label rows use every eligible public training record. All rows use the same 1,896 test ECGs.

| Model | Label budget | Seed | Training labels | AUROC | AP | Sensitivity | Specificity | Brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CNN, supervised | 10% | 42 | 1,518 | 0.9233 | 0.9595 | 94.5% | 55.9% | 0.1080 |
| CNN, supervised | 10% | 43 | 1,551 | 0.9228 | 0.9581 | 95.7% | 50.9% | 0.1082 |
| CNN, supervised | 10% | 44 | 1,559 | 0.9190 | 0.9571 | 95.1% | 51.1% | 0.1108 |
| ECG-FM, adapted + fine-tuned | 10% | 42 | 1,518 | 0.9315 | 0.9636 | 94.2% | 61.3% | 0.1025 |
| ECG-FM, fine-tuned | 10% | 42 | 1,518 | 0.9289 | 0.9626 | 95.6% | 50.5% | 0.1038 |
| ECG-FM, frozen | 10% | 42 | 1,518 | 0.9230 | 0.9595 | 94.6% | 58.8% | 0.1084 |
| ECG-FM, frozen | 10% | 43 | 1,551 | 0.9219 | 0.9595 | 93.7% | 58.2% | 0.1088 |
| ECG-FM, frozen | 10% | 44 | 1,559 | 0.9228 | 0.9596 | 94.2% | 58.5% | 0.1086 |
| ECG-FM, pooled adaptation | 10% | 42 | 1,518 | 0.9305 | 0.9632 | 94.4% | 61.9% | 0.1033 |
| Published ECG-JEPA, frozen | 10% | 42 | 1,518 | 0.9407 | 0.9693 | 94.0% | 67.2% | 0.0928 |
| Published ECG-JEPA, frozen | 10% | 43 | 1,551 | 0.9455 | 0.9716 | 93.1% | 74.6% | 0.0889 |
| Published ECG-JEPA, frozen | 10% | 44 | 1,559 | 0.9397 | 0.9689 | 93.3% | 70.0% | 0.0936 |
| HuBERT-small, fine-tuned | 10% | 42 | 1,518 | 0.9270 | 0.9613 | 95.7% | 50.4% | 0.1048 |
| HuBERT-small, frozen | 10% | 42 | 1,518 | 0.9204 | 0.9560 | 95.3% | 54.1% | 0.1101 |
| HuBERT-small, frozen | 10% | 43 | 1,551 | 0.9216 | 0.9564 | 94.1% | 61.1% | 0.1087 |
| HuBERT-small, frozen | 10% | 44 | 1,559 | 0.9163 | 0.9536 | 94.8% | 54.4% | 0.1127 |
| Compact JEPA + fine-tuning | 10% | 42 | 1,518 | 0.9069 | 0.9479 | 95.1% | 48.4% | 0.1196 |
| Compact JEPA + fine-tuning | 10% | 43 | 1,551 | 0.9080 | 0.9490 | 95.6% | 46.2% | 0.1195 |
| Compact JEPA + fine-tuning | 10% | 44 | 1,559 | 0.9010 | 0.9452 | 94.3% | 50.8% | 0.1234 |
| Lead multiscale, innovation SSL | 10% | 42 | 1,518 | 0.8794 | 0.9348 | 93.1% | 45.9% | 0.1373 |
| Lead multiscale, innovation SSL | 10% | 43 | 1,551 | 0.8632 | 0.9246 | 94.8% | 36.1% | 0.1469 |
| Lead multiscale, innovation SSL | 10% | 44 | 1,559 | 0.8619 | 0.9224 | 94.5% | 41.8% | 0.1484 |
| Lead multiscale, latent SSL | 10% | 42 | 1,518 | 0.8650 | 0.9286 | 92.1% | 40.8% | 0.1438 |
| Lead multiscale, latent SSL | 10% | 43 | 1,551 | 0.8542 | 0.9228 | 92.6% | 35.7% | 0.1498 |
| Lead multiscale, latent SSL | 10% | 44 | 1,559 | 0.8637 | 0.9256 | 94.6% | 34.2% | 0.1449 |
| Lead multiscale, supervised | 10% | 42 | 1,518 | 0.8877 | 0.9394 | 93.5% | 47.2% | 0.1317 |
| Lead multiscale, supervised | 10% | 43 | 1,551 | 0.8773 | 0.9355 | 91.9% | 45.8% | 0.1382 |
| Lead multiscale, supervised | 10% | 44 | 1,559 | 0.8858 | 0.9383 | 92.5% | 52.9% | 0.1336 |
| Compact MAE + fine-tuning | 10% | 42 | 1,518 | 0.8891 | 0.9365 | 93.8% | 49.4% | 0.1313 |
| Compact MAE + fine-tuning | 10% | 43 | 1,551 | 0.8876 | 0.9346 | 94.6% | 45.9% | 0.1325 |
| Compact MAE + fine-tuning | 10% | 44 | 1,559 | 0.8765 | 0.9294 | 93.3% | 48.8% | 0.1382 |
| Transformer, supervised | 10% | 42 | 1,518 | 0.8914 | 0.9372 | 94.3% | 49.2% | 0.1299 |
| Transformer, supervised | 10% | 43 | 1,551 | 0.9008 | 0.9447 | 95.7% | 44.2% | 0.1244 |
| Transformer, supervised | 10% | 44 | 1,559 | 0.8975 | 0.9421 | 93.7% | 52.1% | 0.1266 |
| CNN, supervised | All eligible | 42 | 15,360 | 0.9435 | 0.9703 | 96.2% | 59.1% | 0.0920 |
| ECG-FM, frozen | All eligible | 42 | 15,360 | 0.9345 | 0.9657 | 93.5% | 67.0% | 0.1003 |
| Published ECG-JEPA, frozen | All eligible | 42 | 15,360 | 0.9545 | 0.9765 | 95.2% | 71.8% | 0.0804 |

## Conclusion

The study establishes a reproducible public-data baseline and shows that public annotations should be used fully. Released ECG-JEPA features produced the strongest observed practical system, while a small CNN remained competitive with several much more elaborate approaches. Compact latent prediction helped one matched transformer, but the custom multiscale SSL architecture and small Georgia adaptation pilot did not establish improvements over their relevant controls.

The next substantial gain should be tested through larger and better matched data, stronger transfer baselines, and evaluation against the actual local clinical endpoint. The present results support continuing the research; they do not yet establish a reliable binary health or referral decision for university students.
