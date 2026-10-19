# NTT-05: native signal reference adapter, engineering protocol

19 October 2026. User requested adapting No Time to Train to signals rather than rendered ECG images.
The original method uses a frozen image encoder, proposed masks, instance/class reference aggregation,
cosine matching and semantic-aware soft merging. This successor preserves that structure while changing
both the representation and the proposal domain. It is an adaptation, not a reproduction of the paper.

Commit this protocol before implementation execution. Commit executable sources before its single
engineering verification. No new clinical performance comparison, pretrained model training, downloads,
GPU work, push, PR or clinical label change is scheduled. Native ECG checkpoint files and prior encoder
feature caches are currently absent locally. Random weights cannot replace trained weights in a clinical
comparison. Leave learned-native recognition/localization explicitly unexecuted.

## Proposed native system

| Original component | Native replacement |
| --- | --- |
| Image DINO feature map | Frozen ECG encoder token vectors with declared lead/time anchors and full context metadata |
| SAM image masks | Automatic time intervals with explicit lead sets, from fixed multiscale windows or a beat/wave detector |
| Masked instance vector | Overlap-weighted pooling of native tokens inside the candidate interval/lead set |
| Class memory bank | Equal instance-to-patient and patient-to-finding reference aggregation |
| Cosine match | Candidate-to-finding similarity, with explicit minimum score and runner-up margin |
| Soft merging | Temporal redundancy score decay for overlapping candidates with the same finding and lead set |

A reference is a named finding on a specified region, with source provenance; a whole-record label alone
does not become a specialist region mark. Candidate generators must not use target answers. A missing
reference class or low similarity can cause abstention. Similarity/margin thresholds require a separate
prospective training-role calibration before interpreting them clinically. No numeric cutoff is claimed
clinically validated in this engineering study.

Two supported feature contracts: lead/time tokens, such as ECG-JEPA's lead-major patches; and time tokens
that already mix all input leads, such as xECG/CPC. A mixed-lead token cannot be split into lead-specific
information by selecting an index. Reject a request that slices such a token's lead set. Lead-anchored
JEPA tokens can be pooled by their anchors, but their attention sees other times/leads. Store contextual
scope explicitly; an anchor or pooling mask is not a proven causal receptive field or pathology mask.
Do not invent a uniform sample/token mapping for an encoder without verifying its actual geometry.

The adapter accepts already computed token arrays and exact interval anchors; it does not load visual
models, read waveforms, download weights or silently infer an encoder geometry. A separate lead/time
patch adapter supports an explicitly declared nonoverlapping patch size/sample rate. Future model
integration must verify source revision, pretrained checkpoint hash, canonical leads, resampling,
normalization, class/register tokens, padding and effective context before scoring.

## Engineering decisions frozen before execution

Primary gate: the generic prototype path must reproduce all 180 saved NTT-04 physical full-median-beat
reference choices exactly, and balanced accuracy 167/180, without any model inference or waveform read.
That score is predecessor integrity, not a new clinical result. Verify result SHA256
`79cfa5130ccc8d275e398ddb15b056a1affe1beb812fa32c52e12a8c2211e789`, immutable feature/selection/prediction
hashes and training-role identities first. No broad label replay. Use the saved raw full-context vectors,
three fixed support banks and exact patient separation. Keep existing NTT-04 finding order for ties.

Synthetic invariant tests must cover physical-to-token coordinate mapping, sub-token interval occupancy,
lead selection versus mixed-lead rejection, missing/invalid coverage, finite nonzero vectors, equal
patient weighting under unequal instance counts, target/support patient isolation, explicit abstention,
same-finding/lead overlap decay, independent recurring intervals, and deterministic candidate generation.
These tests verify arithmetic/contracts, not representation quality, clinical localization or any disease
sensitivity. They do not generate a synthetic diagnostic accuracy headline.

Soft decay is a declared 1D adaptation: lower-ranked candidates of the same finding and exact lead set
receive multiplicative `sqrt(1 - IoS * max(0, cosine))` against already higher-ranked candidates; IoS is
intersection divided by the lower candidate's duration. Negative semantic similarity cannot increase its
score. Preserve the candidates and adjusted scores; do not union disjoint recurring beats or call
unselected regions normal. This is not the original mask implementation verbatim.

Primary gate passes only if predecessor choices and the required invariant suite match. Even a pass
admits only the engineering adapter. Learned-native finding recognition and acceptable expert evidence
agreement remain unrun. Runtime ceiling five minutes, two numerical CPU threads, existing venv, no new
dependency. Source/protocol/input/output hashes, timings, exact checks and limitations go under
`outputs/ntt05_native_signal_adapter/`; aggregate result and report may enter Git, identifiers/vectors stay
local. Root updates both queue documents and ranked backlog. No automatic follow-up inference.

## Next scored study, not authorized by this receipt

After recovering a declared pretrained native checkpoint, freeze a separate protocol comparing native
encoder references with physical waveform references and the visual variant on identical permitted
patients, support budgets and proposals. Keep wave-phase and within-wave controls, support uncertainty,
coexisting annotations, absent findings and source/quality controls. Evaluate proposal coverage, finding
recognition and specialist evidence agreement separately. PR67's validation/engineering-test roles,
Challenge/EchoNext tests and the spent ECGAD holdout are not new evaluation resources.
