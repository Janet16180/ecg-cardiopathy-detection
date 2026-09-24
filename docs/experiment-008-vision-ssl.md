# Experiment 008: preserve local ECG features during continued xECG adaptation

**Protocol date:** 24 September 2026. **Status:** deferred at the user's request because the profiled suite projected about 27.7 hours. The original queue stopped at 04:56 UTC after a CUDA out-of-memory error in the fourth profile arm, before any comparative training. The first three arms profiled at microbatch 8, reaching 16.1–16.2 GB peak allocation. The frozen original command and failure are preserved in `outputs/experiment_queue/` and `outputs/experiment008_vision_ssl/priority_queue.log`. A successor queue in `outputs/experiment_queue_recovery_008/` used a common 008 microbatch of **4**, with **16 accumulation steps** and the unchanged effective batch of 64. The coding-rate expansion is therefore computed on four records in every arm; those statistics differ from a microbatch of eight. This change was made before any adaptation arm trained. All four V100 profile arms passed at 14:54 UTC. The successor began arm A training at 14:54:47 UTC, then was interrupted at 15:06:30 UTC. The log records updates through 30, but no `resume.pt` or completed encoder was saved; checkpoints begin at update 100. Future 008 training would restart arm A. See the successor manifest, launch receipt, interruption status and `outputs/experiment008_vision_ssl/profile_cuda.json`. Earlier experiment sources remain frozen.

The completed profile exercised two optimizer steps, all teacher and student passes, Adam state, and checkpoint save/restore in each arm. Peak allocated GPU memory ranged from 9.16 to 9.19 GB. Measured steady updates were 19.70–20.00 seconds; extrapolation gives about 22.0 hours for the four SSL arms, plus about 5.65 hours for the eight transfers based on Experiment 007 timing. These are runtime projections, not measured experiment completion or performance results. The profile receipt SHA-256 is `2de0d0497c2a37178153197f7af45efd23bbefb46d76952077360afb393048ef`. Arm A training update 1 was logged under child PID 77297 at approximately 14:55 UTC; check the live log for subsequent progress.

**Verification:** seven model/recovery tests passed, including coding-rate value/gradient equivalence, teacher stop-grad, visible-only losses, exact sampler/mask/EMA/optimizer/RNG recovery, and the released model's forward/token alignment. A separate cache test verified interrupted-prefix recovery and changed-source rejection. A synthetic report check covered all eight paired comparisons and difference direction. Source snapshots, package versions, and verification receipts are under `outputs/experiment008_vision_ssl/provenance/`. These are implementation checks, not ECG performance findings.

The train-only input cache is complete. The priority coordinator requires successful completion of Experiment 007, then runs the full GPU profile before the four-arm suite. It coordinates the older MIMIC runner and prevents overlapping GPU jobs. Its exact active command is in `outputs/experiment_queue/launch.json`; do not restart the superseded coordinator command.

## Question and scope

Can preserving the released xECG encoder's local temporal structure during self-supervised adaptation improve our fixed PTB-XL abnormality proxy? Does giving more retention weight to unusual **visible** waveform regions help beyond applying the same retention objective uniformly?

The backbone remains the strictly loaded 57,021,472-parameter released xECG used in [Experiment 007](experiment-007-xecg.md). Inputs remain physical-mV, twelve-lead, full ten-second waveforms at 100 Hz, with 40 tokens of 25 samples. No ECG image rendering or vision-model weights are involved. DINOv3 contributes the idea of preserving local feature relationships with a reference encoder; its demonstrated motivation is dense-feature degradation during long vision training, so benefit in this short ECG adaptation is a hypothesis. [DINOv3 paper](https://arxiv.org/abs/2508.10104).

The released xECG training source already contains masked-token SimDINOv2 and a LeJEPA/SIGReg path. Masking, an EMA teacher and Gaussian regularization alone are therefore not proposed as novel xECG additions. This suite retains a simplified version of xECG's existing SimDINOv2 losses as the continuation control and tests local retention explicitly. It is not an exact reproduction of the original pretraining recipe, which used different views, large batches, an auxiliary reconstruction path and much more data. See the [authors' SSL trainer](https://github.com/dlaskalab/bench-xecg/blob/release/bench_xecg/trainers/ssl_pretrainer.py) and [loss implementation](https://github.com/dlaskalab/bench-xecg/blob/release/bench_xecg/utils/loss_utils.py).

## Fixed comparisons

Use four adaptation arms, each initialized independently from the same official checkpoint. Experiment 007 supplies the released-checkpoint transfer baseline without adaptation.

| Arm | Adaptation objective | Comparison answered |
| --- | --- | --- |
| A: continuation | Masked EMA token prediction + pooled EMA consistency + coding-rate expansion | Does this bounded continuation help relative to 007? |
| B: Gram retention | A + frozen-reference temporal Gram loss | Does preservation of temporal feature relationships help? |
| C: uniform local retention | B + uniform visible-token frozen-reference loss | Does additional pointwise local retention help? |
| D: rarity-weighted local retention | B + the same visible-token loss with bounded within-record rarity weights | Does the proposed weighting help beyond C? |

**D versus C is the primary novelty comparison.** B versus A and C versus B are mechanistic comparisons. D versus A combines several changes and cannot attribute their effects separately. Do not select one of the four arms using test results. Keep all four results, including failures and negative results.

## Data and views

Use exactly the audited **56,875 training records**, comprising 39,457 MIMIC and 17,418 PTB-XL ECGs, from the existing `cpc_pool_40k/rows.csv` training rows. The complete CPC cache also contains validation and test records; selecting every cache row would leak evaluation waveforms into adaptation. Preserve source-qualified patient identities and the existing audit of exact decoded-waveform duplicates. The current 200,000-record acquisition must not expand this experiment after protocol freeze.

Build a separate cache directly from the audited raw 500 Hz records using `preprocess_xecg`: full-record FFT resampling to 100 Hz, canonical lead order, physical mV, no per-record or per-lead normalization. Do not resample the CPC cache, whose two five-second halves were filtered independently. Require shape `[1000,12]`, finite values, verified units and matching raw record identities. Any new exclusion fails preparation for investigation; do not silently change the fixed pool. The resulting training array is approximately 2.73 GB in float32. Never use development, calibration or test signals to fit any adaptation statistic.

Every record supplies one clean teacher waveform and two independently masked student views of the same complete ten seconds. For each student view, select exactly 8 consecutive tokens out of 40, with the starting index sampled uniformly from 0 through 32; the two views use independent starts. Apply the released learned mask token **after** the patch embedding. Preserve all token positions and all twelve leads. The masks are prediction corruption, not an assertion that the corrupted waveform has an unchanged diagnostic label. Clean teacher targets preserve the complete record; no partial recording is treated as a separate positive ECG. Different recordings from the same patient are not positive pairs.

Use no random cropping, lead deletion, time reversal, time warping, polarity inversion or amplitude rescaling in this first suite. Mask placement and record order are identical across arms, from streams independent of model RNG. Keep each record's two views together in its microbatch. Track source composition and repeated patient identities in batches; the fixed record sampler is shared across arms, so repeated recordings cannot explain an arm difference.

## Encoder and targets

Let the trainable student be `S`, an EMA copy be `E`, and a frozen copy of the official release be `F`. Both teachers run in evaluation mode under `no_grad`. Initialize all three from the same verified release; retain the release's mask token. Set adaptation dropout and stochastic depth to zero for every arm. Update `E` once after each optimizer step with cosine momentum from 0.99 to 0.9999 across the fixed update budget; never update `F`.

The released Hugging Face model has no pretrained SSL projection head. Use its 1024-dimensional backbone features directly, with float32 L2 normalization only inside cosine and Gram losses. Average raw token features before normalizing the pooled feature. No new predictor or projection head is needed. The student and EMA may adapt all encoder parameters, including the mask token.

The implementation must call patch embedding, explicit token replacement, core and pooling while retaining an explicit all-valid temporal mask. The Hugging Face `get_padding_mask` marks a patch as padding from the first waveform sample being zero across leads; do not reuse that heuristic for masked input. Verify that the unmasked adaptation path agrees numerically with the released forward on valid real ECGs. The nine released blocks flip the sequence before blocks 2–9, yielding eight flips and restoring the original token order. Test position alignment explicitly before comparing tokens or constructing temporal neighborhoods. These features are bidirectional contextual features, not causal forecasts or independent beat embeddings.

## Losses and exact reductions

For record `b`, student view `v`, and token `p`, let normalized student, EMA and frozen features be `s[b,v,p]`, `e[b,p]` and `f[b,p]`. Use epsilon `1e-8` for normalization. Let `M[b,v]` contain the eight masked positions and `V[b,v]` the 32 visible positions. Compute record/view losses first, then average, so one recording cannot dominate through its number of positions.

**Common continuation loss:**

`L_A = L_mask + L_pool + 0.1 * L_expand`.

`L_mask` is mean `1 - dot(s,e)` over masked positions. `L_pool` is mean cosine distance between each masked student's pooled feature and the clean EMA pooled feature. These are same-record targets; no age, sex, patient ID, report or diagnosis is an input.

`L_expand` retains the upstream SimDINOv2 coding-rate expansion with `eps=0.05`, averaged over the two student views. For a microbatch of `m` L2-normalized pooled vectors in `Z[m,d]`, `d=1024`, use:

`-0.5 * eps * sqrt(m / (d * min(d,m))) * logdet(I_m + d/(m*eps) * Z @ Z.T)`.

This equals the upstream feature-space log determinant by the determinant lemma. Verify value and gradient agreement in a numerical test. Compute it over the actual microbatch, not over an imaginary effective batch of 64: gradient accumulation does not combine batch-statistical losses. Freeze the same microbatch size for all arms after profiling and report it. Expansion discourages collapse; monitoring still has to verify that useful feature variation survives.

**Gram retention:** on each view's visible positions, form `G_s = s_V @ s_V.T` and `G_f = f_V @ f_V.T`. `L_gram` is mean squared difference over off-diagonal pairs, averaged across records and views. B, C and D add `1.0 * L_gram`. This applies DINOv3's relational-retention idea to temporal ECG tokens with a frozen released anchor. Using only visible positions avoids demanding recovery of an unpredictable local event from fully hidden evidence. The frozen reference and visible-only restriction are protocol choices, not claims to reproduce all DINOv3 details.

**Uniform local retention:** C additionally adds `0.25 * L_visible`, where `L_visible` is mean `1 - dot(s,f)` over visible positions, records and views.

**Proposed rarity weighting:** D replaces C's uniform average with a weighted average of the exact same visible-token distances and retains coefficient 0.25. Compute rarity entirely from the clean frozen features, with no learned weighting network:

1. For each token `p`, consider tokens `q` satisfying `abs(p-q) > 1` in the same ten-second record. This excludes immediate temporal neighbors.
2. Let `r[p] = 1 - mean(top3_q(dot(f[p], f[q])))`. A token that has fewer close feature matches elsewhere in its own recording receives a higher score.
3. Compute midranks of the 40 rarity scores, rank zero through 39, then `w[p] = 1 + rank[p]/39`. Exact ties receive their common midrank, so equal scores give equal weights.
4. For each visible set, use `sum(w[p] * distance[p]) / sum(w[p])`. Stop gradients through all scores and weights. Raw weights lie in `[1,2]`; normalization keeps the loss scale comparable with C.

This deliberately does not upweight **masked** rare-event prediction. It attempts to preserve unusual information that remains present in the student input. Temporal feature rarity is an unsupervised heuristic; it may emphasize ordinary phase boundaries, noise or artifacts rather than pathology. It is not an arrhythmia detector and makes no claim that rare features are clinically important. Clipping the weight ratio and comparing D with C make that hypothesis testable. Inspect a fixed training-only set of weighted waveforms and retain those diagnostics with the run.

## Bounded training and transfer

Fix **1,000 optimizer updates per adaptation arm**, effective batch 64 records, two student views per record, seed 42. Shuffle the fixed training records without replacement, then reshuffle when exhausted. Every arm sees the same 64,000 record draws. The attempted microbatch 8 and eight accumulation steps failed GPU profiling with CUDA OOM in arm D. Use the common microbatch 4 and 16 accumulation steps, subject to a successful four-arm GPU profile before comparative training. This changes the coding-rate expansion statistics; report and retain this common choice across arms.

Use AdamW, encoder learning rate `1e-5`, weight decay 0.04, 50-update linear warmup and cosine decay to `1e-6`, gradient norm clipping 1.0, and FP32 on the V100. Apply weight decay to matrix/kernel weights only; exclude biases, normalization parameters and mask token. Fix this parameter grouping and all scalar loss coefficients across arms. These are conservative continuation settings, not optimized hyperparameters. Select the **final update-1000 student**, not the EMA teacher or a favorable intermediate checkpoint, for downstream transfer.

Evaluate each final encoder under both Experiment 007 label budgets, using exactly its supervised optimizer policy, initialization of the new one-logit head, fixed splits, epoch limit, patience and checkpoint selection by development AUROC. There are eight supervised transfers in this suite. The full-label budget is primary; the 1,518-label budget is secondary. Calibration patients fit Platt calibration and choose the existing at-least-95%-sensitivity threshold; freeze both before evaluating test patients. Preserve the existing metrics and paired patient-bootstrap procedure. Pair D–C predictions on the same patients and report the interval for the difference; do not infer significance by comparing separate confidence intervals. Record every arm's development and test results, without a test-selected winner.

Matched adaptation updates do not make 007 compute matched: it receives no continuation updates. Even across A–D, auxiliary losses have small compute differences. Record wall time, actual examples and view count, updates, peak GPU memory, and all forward passes. For transparent runtime comparison, compute the frozen teacher features in every arm for shared diagnostics, even when their training coefficient is zero; no gradients pass through teachers.

## Required gates, diagnostics and provenance

Before training, validate exact cohort identities/hashes, no held-out adaptation records, release weight hash and strict loading, unmasked forward equivalence, token/mask alignment, teacher stop-grad, each objective's gradient path, coding-rate equivalence, finite real-batch forward/backward/update, and exact resume continuation including EMA, optimizer, scheduler, record order, mask RNG and all other RNG state. Check that C and D coincide when all rarity weights are equal; changing the rarity stream must not alter input batches or masks. Every save records arm identity and frozen protocol hash to prevent cross-arm resume.

The real GPU profile must include at least two optimizer steps for the largest arm with both students, both teachers, optimizer states and checkpoint save/restore. Use one common configuration. If memory does not fit, reduce the common microbatch and increase accumulation before any arm starts; do not silently apply different sizes. Profile throughput and record projected total suite duration before the queue starts adaptation. A failed backend, finite-value or data-identity gate stops this queue entry without disturbing earlier experiments. Temporary profiles must not become adaptation initialization.

Every 100 updates, save resumable state and log each loss term, pre-clipping gradient norm, student/teacher feature norms, pooled feature variance/effective rank, same-record versus other-record similarity, EMA-to-release drift, visible Gram error, and rarity-weight distribution. Use the same fixed eight training-only diagnostic records for all arms: four PTB-XL and four MIMIC records selected with a separate recorded seed, without consuming training sampler, mask or model RNG. Save their identities and waveforms with the diagnostic weights. The actual training loss is not comparable across different objectives. Nonfinite loss/parameters stop the run. Record near-collapse and feature drift as observed failures rather than modifying loss weights mid-suite.

Save source snapshots and hashes, adapter/backend versions, raw-input and output-cache hashes, audited selection manifests, arm configurations, data/mask seeds, GPU details, run status, adaptation checkpoint hash and downstream checkpoint hashes. Implement Experiment 008 in new files and reuse frozen Experiment 007 utilities without changing their behavior. Queue only after the existing Experiment 007 coordinator completes successfully and the GPU lock is acquired; record the actual launch command/PID and status file afterward.

## Interpretation limits

One seed and an already inspected PTB-XL test cohort support exploratory comparisons only. Bootstrap intervals describe patient sampling uncertainty, not retraining variability. The released pretraining corpus and cross-dataset identity limitations documented in Experiment 007 still apply. A successful result establishes a benefit under this checkpoint, adaptation budget and proxy-label task; it does not establish a better xLSTM architecture, a clinical referral system, general university-student performance, or superiority over modern vision SSL generally. Failure to improve is plausible: the release already learned from millions of ECGs, and anchoring may restrict useful adaptation.
