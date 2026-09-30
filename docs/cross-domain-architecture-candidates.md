# Independent ECG architectures from NLP, genomics and vision

**Research date: 24 September 2026. Status: KDA/CKDA, the genomic hybrid and Mamba-3 are user-authorized and queued for implementation as Experiments 011–013.** See the [persistent queue](experiment-queue.md). They have not been implemented or attached to the automatic GPU scheduler yet. These designs address the user's request for a new model family transferred to ECG. They do not modify xECG. The CPC mismatch and cross-lead work remains separate.

## Two architectural hypotheses worth testing

The strongest focused next experiment is **a small Kimi Delta Attention model versus Complex KDA**, with the current GRU as a practical reference. It tests whether a different recurrent memory update helps a repeating but variable waveform. A second, more ambitious direction is **a compact StripedHyena-style mixture of local and long temporal convolutions**, inspired by genomic sequence modeling. Both would learn directly from twelve-lead waveforms; neither requires assigning ECG samples text or DNA vocabulary IDs.

These are promising mechanisms, with evidence in their source domains, and **no measured improvement on our ECG task**. Recent publication, excellent language/genomics scores, or a familiar biological analogy is not evidence that a model will outperform the compact CPC baseline. No claim is made that these families have never appeared in signal research.

| Priority | Borrowed mechanism | ECG hypothesis | Primary control |
| --- | --- | --- | --- |
| 1 | KDA memory with CKDA's extended transition ranges | Signed/rotational state dynamics may preserve useful changing waveform phase | Ordinary KDA with otherwise identical architecture |
| 2 | StripedHyena-style operators at several temporal scales | Local morphology and relationships across beats may benefit from explicit scale diversity | Same compact hybrid with long operators replaced by local operators |

## 1. NLP memory: KDA versus Complex KDA

Kimi Linear uses Kimi Delta Attention (KDA), a refinement of the gated delta memory update; its large released language model is not the proposed ECG model. [Kimi Linear authors' repository](https://github.com/MoonshotAI/Kimi-Linear).

**Recent evidence.** Complex KDA was submitted on **21 September 2026**. It extends channel-wise transition gates to `[-1,1]` and the delta coefficient to `[0,2]`, enabling rotations while retaining the analyzed diagonal-plus-rank-one, non-expansive transitions. The paper reports improved extrapolation among its tested KDA range variants on synthetic state tracking and periodic audio continuation; its language results are similar to ordinary KDA. [CKDA paper](https://arxiv.org/abs/2609.24797).

The authors' code makes a useful limitation explicit: the audio experiment studies circular shifts of a single fixed synthetic groove, and **GRU is the strongest model on that particular task**. This supports a mechanistic experiment on phase preservation, not a claim that CKDA is already better than GRU for physiological signals. The released CPU reference and tests are useful starting points; neither proves V100 training compatibility. [Official implementation and audio-task description](https://github.com/OpenEuroLLM/ComplexKDA).

**Proposed compact model.** Retain the existing causal twelve-lead CNN waveform stem, its 256-wide output, independent five-second halves, 79 token positions, and mean/max downstream readout. Replace the GRU context mixer with a small stack of KDA blocks; instantiate CKDA using the same blocks and only the documented transition-range changes. Use the same number of blocks, head dimensions, projections, normalization, feedforward paths and dropout in KDA and CKDA. Start with width 256 and two blocks, then count actual parameters and profile; the exact compact configuration is not yet frozen.

Train all models from scratch on the fixed **56,875-record** SSL pool. Copy the same random stem and prediction-head initial states across arms; context modules receive their documented initializers. Importing a trained GRU checkpoint into only one arm would confound architecture with pretraining. There is no assumption that language-model embeddings or text-pretrained weights have ECG semantics.

**Minimal fair experiment.** Three arms: original GRU-CPC, KDA-CPC, CKDA-CPC. KDA and CKDA must be exactly parameter matched; compare their recurrence ranges as the primary effect. Keep the GRU comparison transparent about parameter count. If a parameter-matched GRU is desired, set its width/depth and any common output projection before observing outcomes; report the original GRU too rather than quietly replacing it.

Keep the existing causal CPC objective, horizons 4/8/12, raw-support exclusions, fixed training-only normalization, downstream pooling, label subsets and patient splits. A sensible initial budget is the original **20 SSL epochs**, same batch/examples and optimizer family, followed by the same 10% and full-label transfers. This is a proposed budget, subject to a real-device profile before a protocol is frozen. Report updates, examples, wall time and memory: equal epochs do not imply equal computation. If optimization needs tuning, grant KDA and CKDA the same small development-only grid and report every tried setting.

A later mechanistic ablation can separate the two range extensions with four settings: original ranges, signed channel gates only, expanded delta coefficient only, both. Do not attribute a two-change result uniquely to signed gates without this ablation. A fresh bidirectional masked-prediction study is also possible, but it needs matching bidirectional controls and an input-level mask before any clean waveform information reaches the encoder. Comparing bidirectional masked CKDA directly with causal CPC would mix architecture, available information and training objective.

**What could fail.** ECG timing changes, ectopy and noise are not a stationary oscillator. Persistent phase tracking could emphasize common repetitive patterns while suppressing isolated evidence. A short 79-token context may not exercise the long-sequence strengths of these language models. A theoretically non-expansive state transition does not guarantee stable end-to-end gradients or accurate classification. Optimized recurrent kernels may require hardware features unavailable on the V100; a small correct PyTorch reference could run slower than GRU. Verify recurrence value/gradient agreement and causality before any timing claim.

**Other recent NLP results.** Gated DeltaNet-2 separates erase and write control; Mamba-3 uses a richer discretization, complex-valued state evolution and a MIMO formulation. Their published language comparisons do not establish their ranking at one or two million parameters on ECG. Mamba-3 now has an authorized [Experiment 013 implementation plan](experiment-013-mamba3-25k.md); Gated DeltaNet-2 remains a possible follow-up. [Gated DeltaNet-2, May 2026](https://arxiv.org/abs/2605.22791), [Mamba-3, March 2026](https://arxiv.org/abs/2603.15569). RWKV-7 is another generalized-delta recurrence from language modeling, with the same need for compact waveform adaptation and a matched control. [RWKV-7](https://arxiv.org/abs/2503.14456).

## 2. Genomics: a compact mixture of temporal scales

**Source mechanism.** Evo 2's StripedHyena 2 architecture combines short explicit, medium regularized and long implicit convolution operators with attention. The Nature paper was published on **4 March 2026**. Its genome results motivate testing a mixture of scales, while its billion-parameter size, million-position context and training corpus are far from this ECG setting. [Evo 2 paper](https://www.nature.com/articles/s41586-026-10176-5), [authors' repository](https://github.com/ArcInstitute/evo2).

**Proposed transfer.** Build a new small waveform model with a twelve-lead CNN stem and a handful of width-256 gated temporal blocks. Preserve the input-dependent gating and explicit/implicit temporal-filter idea from the source implementation. Use local operators for short morphology and longer causal operators for relationships across the five-second half. Include a single causal attention block only if it is shared by the proposed arm and its control. Do not call a generic dilated CNN an Evo 2 reproduction.

A clean initial pair uses the same stem, depth, width, gates, attention placement, readout and causal CPC objective. The proposed arm mixes short and long operators. The control restricts all temporal operators to local support, retaining comparable trainable parameter counts and the same FFT/direct convolution implementation where possible. An alternative strict architectural comparison is against a matched Transformer trained with the same stem and loss, but that tests the complete hybrid package rather than the specific contribution of long filters.

All weights are learned on ECG; no released DNA vocabulary, token likelihood or genomic checkpoint is used as an ECG diagnostic score. Use the same fixed pool, 20-epoch candidate budget and downstream manifests as the NLP experiment if this family is later selected. No special genomic preprocessing or expansion to the ongoing 200k acquisition belongs in the initial comparison.

**Why it is second.** Scaling long convolutions from genomic lengths down to 79 tokens may remove their main advantage. Filter construction, padding, FFT circular-wrap prevention and local/long operator balance create more implementation work than the KDA range comparison. The small model is an adaptation inspired by the architecture, not an exact StripedHyena 2 replication. Size and runtime are estimates until implemented and profiled; importing the large released Evo model is not a V100 plan.

## 3. Vision: independent architectures to test on ECG

**Research date: 24 September 2026. Status: proposals only; no training queued.** These are independent models, with no xECG backbone.

| Candidate | What transfers | Cheapest informative comparison | Main uncertainty |
| --- | --- | --- | --- |
| TiViT with OpenCLIP | A frozen image-pretrained transformer and its intermediate features | Image-pretrained versus randomly initialized frozen backbone with the same ECG conversion and probe | Rendering/stacking can obscure amplitude, temporal detail or lead relations |
| Compact 1D ConvNeXt V2 | Depthwise temporal blocks, hierarchical resolution and global response normalization | Same compact model with versus without GRN, same SSL objective and data | Benefits from image scales and spatial statistics may disappear on ECG |
| DINOv3 ViT-S blocks with a waveform stem | Image-pretrained attention/MLP blocks in a new raw-signal model | Pretrained versus random transformer blocks, identical new stem and adaptation budget | New input distribution and temporal positions may defeat useful weight transfer |

### Cheapest existing cross-domain recipe: TiViT

TiViT converts segmented time series into stacked grayscale images and probes intermediate frozen vision features. Its current official implementation lists OpenCLIP, SigLIP 2, DINOv2 and MAE; DINOv3 is not listed there. This is actual input/model transfer, rather than borrowing a training loss. [Paper](https://arxiv.org/abs/2506.08641), [official implementation](https://github.com/ExplainableML/TiViT).

Start with a supported modest OpenCLIP ViT-B/16 checkpoint, freeze it, extract a small predefined set of intermediate layer features, and fit regularized probes on the existing 10% and full label manifests. The first implementation decision is a documented twelve-lead conversion that preserves lead identity, duration and physical amplitude scaling. Compare it with the same conversion and randomly initialized frozen transformer to identify the contribution of image pretraining. Development patients choose layer/probe regularization; calibration and test patients must not choose image layout, layer or normalization. A frozen time-series model can be a contextual reference, with its differing pretraining exposure reported.

The authors' UCR/UEA results motivate a trial but do not establish performance on our twelve-lead binary proxy. In the repository's reported table, TiViT alone is higher than Mantis on UCR and lower on UEA; fusion improves the reported averages. Do not present it as a universal winner. The cheapest pilot is inference plus small probes, not an automatic claim that a large ViT is faster than our compact CNN/GRU.

### Lower-parameter architecture transfer: 1D ConvNeXt V2

The official ConvNeXt V2 release contains Atto/Femto variants, fully convolutional masked-autoencoder pretraining and global response normalization (GRN). It is a **2023** family, useful for compact design rather than a claim about the latest 2026 vision leader. [Official code and checkpoints](https://github.com/facebookresearch/ConvNeXt-V2).

Construct a small one-dimensional version with twelve input channels, temporal depthwise kernels, pointwise channel mixing and a few downsampling stages. Aim for roughly 1–3 million parameters, measured before protocol freeze. Train from scratch on the fixed 56,875 ECG pool; this tests an architectural pattern without assuming that a 2D image kernel has a unique correct 1D conversion. Compare GRN against an otherwise identical no-GRN model under one shared masked-reconstruction objective, with masking applied to the input before convolutions and loss evaluated only on withheld samples. This is a dense ECG adaptation, not an exact FCMAE reproduction.

Both variants receive identical masks, a shared small decoder, fixed training-only amplitude normalization and equal examples/updates. A new GRU or CNN reference trained under that same reconstruction objective is needed for claims against those architectures. Comparing against an old CPC run alone would mix objective and architecture. GRN over the temporal axis is not causal; do not substitute this model into future CPC without redesigning and verifying its normalization and raw-support path.

### Ambitious image-weight transfer: DINOv3 transformer blocks on raw ECG

The official DINOv3 release includes distilled ViT-S/16 (21M parameters) and ViT-S+/16 (29M); access to its pretrained weights follows the authors' request procedure. These image-model counts are not counts for a new ECG adapter. [Official DINOv3 release](https://github.com/facebookresearch/dinov3).

Replace the image patch projection with a learned twelve-lead temporal convolution producing the transformer's expected embedding width. Keep a bounded token count, for example 100 nonoverlapping 100 ms waveform tokens per ten-second record after fixed preprocessing. Transfer only compatible transformer block/norm tensors with an explicit audited load list. Adapt positional handling to one-dimensional time and preserve any required special-token contract; DINOv3's image position machinery cannot be assumed to accept raw waveform tokens unchanged. Do not silently reshape leads into RGB channels or describe modified inputs as an unchanged image model.

The decisive pair is **image-pretrained blocks versus random blocks**, with exactly the same new waveform stem, temporal-position implementation, masked-waveform or feature-prediction objective, data, parameter count, adaptation updates and downstream readout. Trainability schedules must also match. A frozen-block stem-only pilot has smaller optimizer state, but gradients still traverse the frozen blocks to train the stem; it needs a real V100 profile. Avoid a large teacher or extra loss in only the pretrained arm.

This is an investigation of cross-domain weights, not a DINOv3 ECG novelty claim. Vision models and DINO-style objectives already appear in signal research. An ECG result would be needed before calling the approach promising on this task. Hiera's hierarchical masked-autoencoder architecture is another established **2023** option, but adding it now would broaden the sweep before these sharper questions are answered. [Official Hiera implementation](https://github.com/facebookresearch/hiera).

No candidate above has passed a local V100 profile or produced ECG results. Keep external-source benchmark claims, implementation readiness and measured project performance separate. The existing authorized queue remains unchanged by this section.

## Other domains and evidence standards

The independent vision shortlist in section 3 covers frozen TiViT/OpenCLIP features, a compact one-dimensional ConvNeXt V2, and transfer of DINOv3 transformer blocks through a new waveform stem. They test image-weight transfer or vision-derived architecture, unlike 008's borrowing of a vision SSL loss while retaining xECG.

Borrowing HuBERT's general lesson means designing waveform inputs and a learning problem that suit the signal, then measuring transfer. A masked discrete-target method would require training-only target construction and appropriate controls; simply attaching text IDs to amplitude bins is not evidence that language pretraining transfers. Experiment 006 already studies learned future target codes, so another code-prediction objective alone would not satisfy the request for a new architecture.

For any chosen candidate, freeze the architecture, data/label manifests, initialization policy, source revisions, optimizer/search budget and primary comparison before test access. Retain calibration-only threshold selection and report actual test sensitivity alongside specificity, AUROC, AP and calibration. Add paired patient-bootstrap differences and, for a promising result, matched additional seeds. Inspect noise sensitivity and diagnostic subgroups where labels allow. The current test set has already been examined; published source-domain state of the art and an isolated point-estimate gain do not establish an ECG state-of-the-art or clinical result.

**Current decision:** implement queued Experiment 011 (KDA versus CKDA with GRU reference), then 012 (the genomic hybrid), then 013 (Mamba-3). These are authorized backlog entries with no implementation, runtime measurements or performance results yet. The [JSON catalog](experiment-queue.json) records their exact next actions and dependencies.
