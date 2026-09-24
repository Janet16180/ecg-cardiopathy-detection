# Experiment 004: a simple CPC improvement using downloaded MIMIC ECGs

**Protocol date:** 23 September 2026.  
**Status:** Complete. The compact-model suite finished on 24 September 2026; see the [results report](../outputs/experiment004_cpc_40k/report.md). Full-label AUROC was 0.950 for scratch, 0.955 for CPC, and 0.957 for CPC plus temporal agreement. The paired interval for the last comparison includes zero. Released S4 frozen-feature evaluations are also complete (see below).

## Question

Does adding a small global temporal-agreement loss improve a compact predictive CPC encoder, with its architecture, training data, initialization, and optimization budget held constant?

The user requested experiments using the approximately 40,000 MIMIC recordings already downloaded, combined with available public data, while preserving CPC's simplicity. The larger 200,000-record download continues independently.

## Paper motivation and scope

[Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) learns representations by predicting future features. [ECG-FM](https://doi.org/10.1093/jamiaopen/ooaf122) combines local learning with temporal-view agreement. The [recent ECG foundation-model benchmark](https://arxiv.org/abs/2509.25095) also motivates studying token-level and sequence-level representation quality together.

This is a **local compact CPC prototype with a GRU**, trained from scratch. The benchmark's released ECG-CPC instead uses an S4 context model and different pretraining data and settings; its [official configuration](https://raw.githubusercontent.com/AI4HealthUOL/ecg-fm-benchmarking/main/code/conf/config_cpc_ecg_s4_heedb.yaml) is a methodological reference. These runs do not reproduce that published model or establish an improvement over its reported results. A favorable local result would motivate testing the same objective change with S4 later.

No novelty claim is made for combining CPC and temporal contrastive learning. The experiment measures whether that combination helps this task under a bounded budget.

## Data

- **MIMIC selection:** take the deterministic whole-patient prefix of our already fixed, seed-42 MIMIC selection, stopping before the next patient would exceed 40,000 records. Selection does not depend on diagnosis or model predictions. Do not skip missing records to favor those that downloaded fastest.
- **PTB-XL training:** the existing 17,418 official fold 1–8 recordings, including recordings whose labels are hidden or unresolved.
- **Validation and test:** retain the existing PTB-XL patient partitions. None of their recordings enter gradient-based pretraining or normalization fitting.
- **Audit:** verify official checksums for selected MIMIC files; require standard 12-lead, 500 Hz, ten-second, finite physical-mV signals; remove exact decoded-waveform duplicates within MIMIC and against all 21,799 PTB-XL recordings. Access to held-out waveforms here is solely an identity check to exclude duplicates.
- **Freeze:** record accepted ECGs, patient counts, source counts, exclusions, and manifest hashes before either SSL arm starts. Actual accepted counts may be below the 40,000-record cap. The nominal combined training pool is approximately 57,000 ECGs.
- **Labels:** use no MIMIC reports, machine diagnoses, or outcome labels. Primary supervised transfer uses all 15,360 eligible PTB training labels. Secondary transfer uses the existing seed-42 set of 1,518 labels.

Separate manifests and audit state live under `data/processed/mimic_ssl_40k_cpc/`. The new preparation does not modify the running 200,000-record preparation or its manifests.

The fixed prefix contains **39,996 MIMIC ECGs from 7,920 patients**, before waveform exclusions. All selected header/signal files passed official SHA256 checks. The selection manifest SHA256 is `2098d2fc81bcbb45e41ed478386d5988a9d71415ae3fe4001ba05240e7f435e3`.

The completed audit accepted **39,457 MIMIC ECGs from 7,892 patients**. It excluded 512 nonfinite waveforms and 27 exact MIMIC duplicates; no MIMIC waveform matched the 21,799 PTB waveforms checked. The combined **training pool is 56,875 ECGs**. Including validation/test, the cache contains 60,641 recordings; `signals.npy` SHA256 is `2f34c9adeb5e6b474e71a1dca23c0f31caacc0572258c3db9d10664bfe883133`.

## Preprocessing and prediction integrity

Decode and reorder leads to `I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`. Resample each five-second half independently from 500 Hz to 250 Hz using `scipy.signal.resample_poly`, then concatenate the two halves in a float32 memory-mapped cache. Each recording contains 12 × 2,500 samples.

Use fixed per-lead mean and standard deviation fitted on the combined **training** signals. Do not demean or standardize using each recording's complete time window: that would give a causal prefix statistics computed from its future. The supervised scratch reference uses the same training-pool normalization, so its weights are supervised but its normalization also uses the available unlabeled signals.

The resampling filter is offline and has finite future support. Prediction gaps must exceed the combined filter support and tokenizer overlap; causal network layers alone would not establish separation in the original waveform. With the default 41-tap filter for downsampling by two, each resampled sample depends on at most 20 original samples on either side. Our shortest forecast leaves a positive gap between the query's raw-sample support and the target's support. Separate half resampling prevents shared raw samples across the temporal-agreement pair.

## Compact architecture

Both SSL arms and the supervised reference use the same encoder:

1. Four causal 1D convolution blocks with channels `[64, 128, 192, 256]`, kernels `[5, 3, 3, 3]`, and strides `[2, 2, 2, 2]`.
2. Explicit left padding, per-token channel LayerNorm, and GELU. No BatchNorm across time.
3. A two-layer unidirectional GRU with hidden size 256 and interlayer dropout 0.1.
4. Process each five-second half independently, resetting recurrent context at the boundary.

The tokenizer has stride 16 and receptive field 33 samples at 250 Hz. The parameter count must be measured from code and recorded; the design is approximately one million parameters. There are no added attention blocks or separate teacher networks.

Measured implementation counts: **1,041,024 encoder parameters**, **1,237,632 total pretraining parameters** in either SSL arm, and **1,041,537 supervised parameters**. CMSC adds no parameters.

For supervised classification, concatenate mean and maximum context features within each half, average the resulting half representations, and use a linear binary head.

## Matched self-supervised objectives

### Baseline: predictive CPC

Use three bias-free linear heads to predict future encoder features from the GRU context at horizons 4, 8, and 12 tokens, approximately 256, 512, and 768 ms. Use cosine InfoNCE with temperature 0.1, averaging valid query positions and horizons. Skip the first three context tokens.

Negatives come from other temporal positions of the **same recording half**, avoiding a task solved solely by recording identity. Exclude negative positions within three tokens of the positive target; always retain the actual positive. Gradients flow through the target encoder features as in ordinary CPC. No quantization or EMA teacher is added.

### Proposed variant: CPC plus temporal agreement

Add `0.1 × CMSC` to the same CPC loss. Mean-pool GRU context separately within each half, normalize features, and contrast corresponding halves against other recordings in the batch at temperature 0.1. Use both anchor directions. Matching recording halves are positive pairs; exclude off-diagonal pairs belonging to the same namespaced patient, since repeated visits need not express the same findings.

Both variants execute the same backbone passes. The added term has no trainable parameters. There are no differences in augmentation or lead masking. Random initialization, shuffled data order, batch size, optimizer, and pretraining budget are matched. Different sources may still make some cross-record negatives easier, a limitation of the combined pool.

Temporal agreement may suppress findings present in only one half. A smaller SSL loss would not itself establish a better screening representation; downstream results determine whether the change helped.

## Optimization and comparisons

Initial fixed plan, subject only to a documented runtime/memory profile before downstream results:

| Setting | SSL | Supervised transfer |
| --- | --- | --- |
| Seed | 42 | 42 |
| Batch size | 128 | 128 |
| Budget | 20 epochs per SSL arm | Maximum 40 epochs; patience 8 |
| Optimizer | AdamW, weight decay 0.01 | AdamW, weight decay 0.01 |
| Learning rate | 1e-3 with two-epoch warmup and cosine decay to 0.1 of base | Encoder 3e-4; classification head 1e-3 |
| Augmentation | None | None |
| Selection | Final fixed-budget SSL state | Best development AUROC |

Train both SSL encoders before evaluating their downstream test performance. Fine-tune scratch, CPC, and CPC-plus-CMSC encoders at each of the two label budgets: **six downstream runs**. The same SSL checkpoint is reused across the two label budgets for each pretrained variant.

Record runtime, parameter counts, actual batches and examples, source and manifest hashes, package versions, and source snapshots. Use atomic epoch checkpoints with optimizer and random states so interrupted runs can resume without silently resetting their budgets.

### Budget freeze and execution

A V100 synthetic-input profile (batch 128, five warmup and twenty measured updates) measured 41.3 ms/update for CPC and 46.2 ms/update for CPC+CMSC, with 1.05/1.07 GB peak allocated GPU memory. These measurements exclude data loading and are **not** wall-clock training estimates. The profile supports retaining the planned **20 SSL epochs per arm**; no downstream results were examined to make that decision. The durable queue also records a real-cache runtime profile before training.

Run `.venv-pretrained/bin/python -m scripts.run_cpc_experiment --stage all --variant all --labels all --device cuda --threads 1`. The current coordinated launch is recorded in `outputs/experiment004_cpc_40k/launch.json`; its log is `outputs/experiment004_cpc_40k/runner.log`. It waits for the audited cache, then temporarily suspends only the idle Experiment 003 orchestrator while this suite uses the GPU. The 200k downloader continues. The orchestrator is resumed when the suite exits; resumable epoch checkpoints survive an interrupted training process. Do not launch a second copy while this queue is active.

## Evaluation

Use the existing 1,306-record development split, 564-record calibration split, and 1,896-record test set. Fit Platt calibration separately for each model on calibration patients. Select the highest threshold attaining at least 95% sensitivity on that calibration set, then freeze it for testing.

Report AUROC, average precision, sensitivity, specificity, Brier score, ECE, confusion matrices, and patient-bootstrap intervals. The primary comparison at each label budget is CPC-plus-CMSC minus CPC. Also compare both SSL arms with the matching scratch model. Use paired patient resampling with aligned ECG IDs and labels, following the project's existing 500-resample convention.

The PTB test set has already been examined in previous experiments, so these are exploratory comparisons. One seed does not measure retraining variability. Do not infer improved student-population performance, clinical referral validity, or architectural superiority from a small point-estimate change.

Generated histories, predictions, paired comparisons, and a separate results summary will live under `outputs/experiment004_cpc_40k/`. Preparation and runtime profiles are not completed model findings.

## Additional reference: released S4 ECG-CPC

At the user's request, also evaluate the authors' released pretrained S4 model. The [official repository](https://github.com/AI4HealthUOL/ecg-fm-benchmarking) links a [35,616,949-byte checkpoint archive](https://figshare.com/articles/dataset/ECG-CPC_Checkpoint_zip/30192604), licensed CC BY 4.0. The release describes HEEDB pretraining. Its archive matched the official MD5 `7cdd0d54786b4d98248afc6dce63beea`; our SHA256 is `94972d80645ac479acfee439da910bd0df1532dce6fae088e65f7529cc6af15d`.

The packaged configuration specifies 240 Hz, 2.5-second inputs, no normalization, four convolution layers of width 512, and a causal S4 context model with state dimension 8. Preserve this preprocessing and validate strict checkpoint loading. First evaluate frozen features with the existing regularized linear probe and the same full/10% label manifests, development selection, calibration, and test patients. This is a pretrained reference with different architecture and data exposure; it is outside the matched local CPC objective comparison. Availability of the checkpoint does not establish its performance on our task.

The released model passed strict loading, numerical-backend checks, and real-ECG CPU/GPU inference checks. Extraction on 19,126 PTB ECGs and both probes completed. Test AUROC was **0.93794 with full labels** and **0.92347 with the 10% manifest**. [Released-model implementation notes and results](released-ecg-cpc.md) document the native PyTorch Cauchy backend, checkpoint buffer handling, preprocessing, metrics, and limitations. These results concern the released frozen S4 encoder; the matched local-model study remains separate.

## Follow-up idea: word2vec objectives

The user proposed CBOW, skip-gram, and negative sampling. [Word2vec](https://arxiv.org/abs/1301.3781) and [its negative-sampling extension](https://arxiv.org/abs/1310.4546) provide relevant objective choices. CPC already predicts neighboring future representations and uses negative candidates, but its InfoNCE softmax differs from word2vec's independent logistic positive/negative terms.

A clean next ablation retains the compact encoder and future horizons and compares **sampled InfoNCE with 16 negatives** against **word2vec-style logistic loss with the same 16 negatives**. Keep cosine scoring, temperature, candidate exclusions, data order, initialization, and budget fixed. Use a dedicated negative-sampling RNG so added sampling does not change dropout or minibatch order. The current full-candidate InfoNCE arm provides a separate reference. The user has now authorized this as [Experiment 005](experiment-005-word2vec.md), separate from this frozen six-run study.

Skip-gram could predict past and future *CNN features* from a center feature; causal context prediction already supplies a related future-only objective. CBOW could average surrounding CNN features to predict the omitted center. Avoid including overlapping target samples or using a future GRU context that already contains the target. Unordered averaging may lose timing information, and repeated normal-looking beats can become false negatives. These are hypotheses to evaluate, not established performance improvements. No discrete vocabulary is needed to test the continuous-feature losses; word-frequency subsampling would require an additional definition of ECG token frequency.
