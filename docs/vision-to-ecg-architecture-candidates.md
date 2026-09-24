# Independent vision architectures to test on ECG

**Research date: 24 September 2026. Status: proposals only; no training queued.** These are independent models, with no xECG backbone. The broader priority and NLP/genomics proposals are in [cross-domain architecture candidates](cross-domain-architecture-candidates.md).

| Candidate | What transfers | Cheapest informative comparison | Main uncertainty |
| --- | --- | --- | --- |
| TiViT with OpenCLIP | A frozen image-pretrained transformer and its intermediate features | Image-pretrained versus randomly initialized frozen backbone with the same ECG conversion and probe | Rendering/stacking can obscure amplitude, temporal detail or lead relations |
| Compact 1D ConvNeXt V2 | Depthwise temporal blocks, hierarchical resolution and global response normalization | Same compact model with versus without GRN, same SSL objective and data | Benefits from image scales and spatial statistics may disappear on ECG |
| DINOv3 ViT-S blocks with a waveform stem | Image-pretrained attention/MLP blocks in a new raw-signal model | Pretrained versus random transformer blocks, identical new stem and adaptation budget | New input distribution and temporal positions may defeat useful weight transfer |

## Cheapest existing cross-domain recipe: TiViT

TiViT converts segmented time series into stacked grayscale images and probes intermediate frozen vision features. Its current official implementation lists OpenCLIP, SigLIP 2, DINOv2 and MAE; DINOv3 is not listed there. This is actual input/model transfer, rather than borrowing a training loss. [Paper](https://arxiv.org/abs/2506.08641), [official implementation](https://github.com/ExplainableML/TiViT).

Start with a supported modest OpenCLIP ViT-B/16 checkpoint, freeze it, extract a small predefined set of intermediate layer features, and fit regularized probes on the existing 10% and full label manifests. The first implementation decision is a documented twelve-lead conversion that preserves lead identity, duration and physical amplitude scaling. Compare it with the same conversion and randomly initialized frozen transformer to identify the contribution of image pretraining. Development patients choose layer/probe regularization; calibration and test patients must not choose image layout, layer or normalization. A frozen time-series model can be a contextual reference, with its differing pretraining exposure reported.

The authors' UCR/UEA results motivate a trial but do not establish performance on our twelve-lead binary proxy. In the repository's reported table, TiViT alone is higher than Mantis on UCR and lower on UEA; fusion improves the reported averages. Do not present it as a universal winner. The cheapest pilot is inference plus small probes, not an automatic claim that a large ViT is faster than our compact CNN/GRU.

## Lower-parameter architecture transfer: 1D ConvNeXt V2

The official ConvNeXt V2 release contains Atto/Femto variants, fully convolutional masked-autoencoder pretraining and global response normalization (GRN). It is a **2023** family, useful for compact design rather than a claim about the latest 2026 vision leader. [Official code and checkpoints](https://github.com/facebookresearch/ConvNeXt-V2).

Construct a small one-dimensional version with twelve input channels, temporal depthwise kernels, pointwise channel mixing and a few downsampling stages. Aim for roughly 1–3 million parameters, measured before protocol freeze. Train from scratch on the fixed 56,875 ECG pool; this tests an architectural pattern without assuming that a 2D image kernel has a unique correct 1D conversion. Compare GRN against an otherwise identical no-GRN model under one shared masked-reconstruction objective, with masking applied to the input before convolutions and loss evaluated only on withheld samples. This is a dense ECG adaptation, not an exact FCMAE reproduction.

Both variants receive identical masks, a shared small decoder, fixed training-only amplitude normalization and equal examples/updates. A new GRU or CNN reference trained under that same reconstruction objective is needed for claims against those architectures. Comparing against an old CPC run alone would mix objective and architecture. GRN over the temporal axis is not causal; do not substitute this model into future CPC without redesigning and verifying its normalization and raw-support path.

## Ambitious image-weight transfer: DINOv3 transformer blocks on raw ECG

The official DINOv3 release includes distilled ViT-S/16 (21M parameters) and ViT-S+/16 (29M); access to its pretrained weights follows the authors' request procedure. These image-model counts are not counts for a new ECG adapter. [Official DINOv3 release](https://github.com/facebookresearch/dinov3).

Replace the image patch projection with a learned twelve-lead temporal convolution producing the transformer's expected embedding width. Keep a bounded token count, for example 100 nonoverlapping 100 ms waveform tokens per ten-second record after fixed preprocessing. Transfer only compatible transformer block/norm tensors with an explicit audited load list. Adapt positional handling to one-dimensional time and preserve any required special-token contract; DINOv3's image position machinery cannot be assumed to accept raw waveform tokens unchanged. Do not silently reshape leads into RGB channels or describe modified inputs as an unchanged image model.

The decisive pair is **image-pretrained blocks versus random blocks**, with exactly the same new waveform stem, temporal-position implementation, masked-waveform or feature-prediction objective, data, parameter count, adaptation updates and downstream readout. Trainability schedules must also match. A frozen-block stem-only pilot has smaller optimizer state, but gradients still traverse the frozen blocks to train the stem; it needs a real V100 profile. Avoid a large teacher or extra loss in only the pretrained arm.

This is an investigation of cross-domain weights, not a DINOv3 ECG novelty claim. Vision models and DINO-style objectives already appear in signal research. An ECG result would be needed before calling the approach promising on this task. Hiera's hierarchical masked-autoencoder architecture is another established **2023** option, but adding it now would broaden the sweep before these sharper questions are answered. [Official Hiera implementation](https://github.com/facebookresearch/hiera).

No candidate above has passed a local V100 profile or produced ECG results. Keep external-source benchmark claims, implementation readiness and measured project performance separate. The existing authorized queue remains unchanged by this note.
