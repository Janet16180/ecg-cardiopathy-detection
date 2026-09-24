# Simple next steps for ECG CPC

**Research note, 23 September 2026.** These are proposed experiments, not measured improvements. They preserve the compact CNN plus two-layer GRU and the existing patient partitions. No architecture was selected from PTB test results. The authorized sampled-InfoNCE versus SGNS experiment should finish as its own comparison before combining changes.

**Primary recommendation:** after the loss comparison, test a mixture of temporal negatives and negatives from other patients of the same data source. This directly probes whether forcing repeated beats apart is limiting CPC, adds no model parameters, and retains temporal negatives to resist patient-identity shortcuts. The pooling change below is an even cheaper preliminary experiment because it needs no new pretraining.

The fixed pool has 56,875 training ECGs: 39,457 MIMIC and 17,418 PTB recordings. Transfer has 15,360 full-label or 1,518 limited-label training examples. The released S4 model is an external pretrained reference with different data exposure; its performance cannot isolate an architectural effect against this GRU model. A recent controlled ECG study supports continuing to investigate CPC, but does not establish which of the changes below will help our binary task. [Al-Masud and Strodthoff, 2026 preprint](https://arxiv.org/abs/2605.12241); [ECG foundation-model benchmark, ICLR 2026](https://arxiv.org/abs/2509.25095).

## Ranked experiments, simplest first

### 1. Preserve a strong finding present in only one half

**Change:** keep the 512-dimensional classifier input, but concatenate the mean across all context tokens with the maximum across all context tokens. The current implementation averages the two half-wise maxima. Thus, a large activation in one half is diluted by the other half. The new rule is `concat(mean(contexts, halves/time), max(contexts, halves/time))`; it adds no parameters or encoder passes.

**Hypothesis:** the downstream readout will preserve occasional morphology or rhythm findings more faithfully. This is an inference from the actual pooling code, not a claim that any hidden channel is already a validated abnormality detector.

**Control:** use the same fixed pretrained checkpoint, classifier initialization, label manifests and optimization schedule to compare existing pooling with global mean/max pooling. First use frozen features to isolate the readout, then compare fine-tuning only if promising on development patients. Keep both halves independent inside the causal encoder.

**Risk and diagnostic:** max pooling can favor a brief artifact. Report development performance stratified by prespecified signal-quality flags and, where labels support it, rhythm/morphology groups. Check which half contains the maximum activation and whether gains survive excluding clearly corrupted training/development recordings under a frozen rule. Record-level labels alone cannot prove detection of a transient event.

### 2. Mix negative sources without allowing source identity to solve the task — primary recommendation

**Change:** retain 16 negative draws per query, but use eight distant positions from the same half and eight positions from other patients of the **same source**. Exclude the anchor patient across all visits. Select an eligible patient uniformly, then one of that patient's batch recordings, then a token; this avoids automatically favoring patients with many visits. Keep the existing within-half exclusion around the positive target. Use the existing CNN features, heads and temperature.

**Hypothesis:** repeated, similar beats in a five-second half are imperfect semantic negatives. Reducing their share may reduce pressure to distinguish clinically equivalent local patterns. Other-patient negatives also contain similar normal beats, so this is a test of negative composition, not a complete false-negative correction. The general problem is established in [Debiased Contrastive Learning](https://proceedings.neurips.cc/paper/2020/hash/63c3ddcc7b23daa1e42dc41f9a44a873-Abstract.html); its class-prior correction should not be imported blindly into continuous ECG tokens.

**Minimal control:** compare 16 same-half draws with the eight/eight mixture using sampled InfoNCE; hold the loss fixed to isolate the sampler. Keep the original minibatch sampler in both arms. Predeclare a fallback to same-half draws if a batch lacks an eligible other patient from that source, and report how often this occurs. Match draw counts, replacement policy and seed. If the mixed arm improves development performance, add a 16 other-patient control to distinguish a benefit of mixture from simply making instance identity useful. Repeat with SGNS only as a later interaction test.

**Risk and diagnostic:** even within one source, another patient's acquisition characteristics can make negatives easy. Retaining temporal negatives limits but does not eliminate that shortcut. Log positive-versus-negative scores separately for each negative source, patient/source predictability from frozen training features, and downstream development performance by age, sex and available diagnostic groups. Do not choose negatives using held-out diagnoses, reports or model predictions. Do not automatically turn repeated beats or repeated visits into positive pairs; a rare beat or a new visit may express a different condition.

### 3. Vary prediction delays while matching their average

**Change:** retain three heads, assigning them horizon bands `{4,5,6}`, `{8,9,10}` and `{12,13,14}` tokens. Draw one horizon per band for each batch with a dedicated saved RNG. Each head therefore learns a small range of future delays. Compare against fixed horizons `(5,9,13)`, which have the same expected delays; retain `(4,8,12)` as the existing reference.

**Hypothesis:** small delay variation may reduce reliance on one exact beat phase and improve transfer across heart rates. This extends the multi-horizon forecasting principle of [CPC](https://arxiv.org/abs/1807.03748); it is not a published result for this implementation.

**Control:** use the common set of valid query positions for the fixed and variable arms, so extra targets near sequence boundaries do not change the training exposure. Keep the count of three horizon losses and their equal weighting. Log loss by actual horizon and development performance across prespecified heart-rate groups.

**Risk and diagnostic:** one head must represent several delays and may learn less precise morphology. A better SSL score does not establish a better classifier. Do not jitter the shortest horizon down to three: with the present resampling filter and tokenizer, that can remove the intended separation between raw query and target samples. Rate-dependent delays derived from a beat detector would be a separate, more complex experiment with failure modes on ectopy and irregular rhythms.

### 4. Give a target token slightly more morphology context

**Change:** set dilation two in the last causal convolution and increase its left padding accordingly. Kernel sizes, channel counts, stride and parameter count stay fixed. The tokenizer receptive field increases from 33 to 49 resampled samples, approximately 132 to 196 ms. This remains a partial morphology window, not a complete beat representation.

**Hypothesis:** a target spanning more local shape may be more useful than predicting very short waveform fragments. ECG CPC already has empirical support as a representation-learning method; this particular dilation choice is an untested adaptation. [Mehari and Strodthoff](https://arxiv.org/abs/2103.12676).

**Control:** use horizons `(5,9,13)` in both the original and dilated tokenizer arms. This separates dilation from the necessary forecast-gap change. Keep negative count, valid query positions and optimizer fixed, and evaluate morphology-related development groups where the annotation permits.

**Leakage risk:** at 250 Hz the tokenizer stride is 16 samples; the resampling support contributes ten samples on each side. A conservative separation calculation is `16*h - (receptive_field - 1) - 20`. At horizon four this is `12` samples for the present tokenizer but `-4` for the dilated one. Horizon five restores positive separation (`12` samples) for the wider tokenizer. Recompute support from the actual filter and add an input-perturbation causality check before any run. Increasing receptive field while retaining the shortest current horizon would invalidate the intended prediction gap.

### 5. Agree across views of the same moment, preserving time-local findings

**Change:** add a small auxiliary contrastive term between time-aligned local context windows from two lead views of the same half. Keep one full-lead view and create a second view that drops at most one lead, using a fixed mask for the half. Start with a fixed weight of `0.05`; pool aligned one-second windows rather than matching the two different five-second halves. Contrast against same-source, other-patient windows and exclude other windows from the anchor patient.

**Hypothesis:** agreement at the same moment can learn limited acquisition robustness without requiring a transient finding in one half to recur in the other. ECG-specific contrast across leads and time has precedent in [CLOCS](https://proceedings.mlr.press/v139/kiyasseh21a.html), and local/global hybrid pretraining is used by [ECG-FM](https://pmc.ncbi.nlm.nih.gov/articles/PMC12530324/). Neither source establishes that this particular auxiliary objective is optimal.

**Controls:** compare CPC alone, CPC with the same random lead dropout but no agreement loss, and CPC plus aligned agreement. Use the same lead-mask distribution in the two augmented arms. Report the extra forward-pass cost; match total compute in a sensitivity comparison if a gain appears. Keep cross-half CMSC as its existing separate experiment rather than adding both auxiliary losses at once.

**Risk and diagnostic:** a finding visible primarily in a dropped lead can be suppressed by enforced agreement. Start with one dropped lead rather than arbitrary lead subsets, and check lead-local morphology groups on development data. Limb leads are algebraically related, so successful lead-view matching is not itself evidence that important morphology was learned. Whole-record agreement, time warping, sign flips and aggressive amplitude normalization would require separate justification.

## Scientific review of the authorized word2vec comparison

For a positive cosine score `s+` and 16 sampled negative scores `s_j`, both divided by temperature `0.1`, the proposed objectives are:

```text
InfoNCE = -s+ + log(exp(s+) + sum_j exp(s_j))
SGNS    = softplus(-s+) + sum_j softplus(s_j)
```

Average each over queries, halves, recordings and the three horizons. The SGNS expression is the conventional positive term plus summed negative terms from [Mikolov et al.](https://arxiv.org/abs/1310.4546). This tests a word2vec-style objective on learned continuous ECG features; it does not reproduce word2vec's discrete vocabulary, separate word tables or frequency-based noise distribution.

### Loss normalization and score bias

The primary SGNS sum is defensible and should be stated explicitly. Dividing the **entire** expression by 17 is a constant rescaling, preserving its mathematical optimum but changing optimization dynamics. Averaging only the negative terms changes their weight relative to the positive and defines a different objective. Neither should be silently substituted. Equal learning rate, weight decay and clipping thresholds constitute matched settings, but cannot establish equal optimization difficulty or that either loss has been tuned to its best setting.

At all-zero scores, InfoNCE is `log(17) ≈ 2.833`; SGNS is `17*log(2) ≈ 11.784`. Their raw loss curves are not directly comparable. Record the unclipped gradient norm, clipping frequency, positive/negative score distributions and finite-step checks. Frequent clipping in one arm can obscure an objective comparison. A normalized-SGNS sensitivity run or a small, equally sized development-selected learning-rate grid can address that later; it should not become an unreported rescue for one arm.

There is also an offset issue. If every pair has the same score `s`, SGNS is minimized at `s = -log(16) ≈ -2.773`, reflecting one positive to sixteen negatives. InfoNCE is invariant to a common offset. With normalized scores and no bias, SGNS must express this preference through embedding geometry. Therefore an informative secondary control is a **fixed score bias** `b=-log(16)`, applied to every SGNS score before softplus. It adds no trainable parameters and makes zero cosine similarity correspond to the prior score. A learned scalar initialized there is another distinct experiment. Bias is not required to define SGNS; omitting it in the primary arm is valid, but a poor no-bias result would not settle whether independent logistic discrimination is useful. The use of a score bias for imbalanced sigmoid contrastive training has precedent in [SigLIP](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhai_Sigmoid_Loss_for_Language_Image_Pre-Training_ICCV_2023_paper.pdf).

### Targets, random draws and collapse

- **Target drift:** retain gradients through the target CNN in both arms, as in the current CPC code. Shared trainable targets change throughout training. Detaching targets or adding an EMA teacher to only SGNS would confound the comparison. In particular, positive alignment and lower SGNS loss can reflect a changing score offset, not more useful features.
- **Identical draws:** initialize an independent negative-sampling generator identically for both arms; save and restore it with checkpoints. Sampling must consume the same count of random values in each arm. Diagnostic forwards must preserve or use a separate sampling RNG so logging cannot change future training examples. A small deterministic check should verify identical sampled indices after initialization and resume.
- **Replacement policy:** the present sampler draws with replacement. That is compatible with the stated loss, but 16 draws need not contain 16 distinct negatives. Retain duplicate multiplicity in both losses and report the policy. Do not sample a positive as a negative; keep the full temporal exclusion in both arms.
- **Collapse:** inspect normalized CNN token variance across time and patients, normalized context variance, off-diagonal cosine similarity and effective rank on a fixed training-only diagnostic set. A raw pooled variance alone can miss angular or token-level collapse. Track positive retrieval rank as well as loss. Neither negative sampling nor a finite decreasing loss guarantees useful representations; same-half sampling can also miss collapse to patient-specific features.
- **Fair comparison:** match encoder/head initial tensors, data order, dropout RNG, normalization, precision, optimizer, number of updates, target-gradient policy and downstream selection. Do not claim a compute improvement from 16 negatives if the implementation still computes all query/target scores before gathering candidates. The existing full-candidate arm is a separate candidate-count reference.

Static review of `ecg_experiment/cpc_word2vec.py` and `scripts/run_cpc_word2vec.py` found matching positive/negative score dimensions and consistent metric keys. Both arms reset the model, dropout, loader and negative-sampler seeds; epoch checkpoints preserve the separate sampler state; diagnostics consume no random draws. The paired comparison reports SGNS minus InfoNCE after checking patient/ECG/label alignment and retains each model's calibration threshold. The runner now logs clipping frequency, preclip gradient norms, normalized token variance and pooled pair cosine. These are appropriate first diagnostics, with the limits described above. This is scientific and static code review; executable validation is performed separately by the implementation owner.

## Decision protocol and limits

Use development patients to select between the prespecified alternatives; reserve calibration patients for calibration and threshold fitting. Freeze each comparison before inspecting its test results. Because the PTB test set has already been used, any subsequent reported result remains exploratory. For a promising idea, repeat the complete pretraining/transfer pair with three seeds; a patient bootstrap measures evaluation-sample uncertainty and does not replace retraining variability.

Report full-label and limited-label transfer, discrimination, calibration, and specificity at the frozen high-sensitivity threshold, together with relevant development subgroups and signal-quality checks. Keep the unchanged CPC control in every new study. Start with one modification, then test a combination only after its components have been assessed.

Young university referral is a different deployment question from detecting the present PTB binary label. A small PTB gain or an age subgroup analysis does not validate that use. A future patient-separated student cohort with clinically defined referral labels and representative prevalence is required to establish the actual referral benefit. These proposals improve the research design without assuming that broader claim.
