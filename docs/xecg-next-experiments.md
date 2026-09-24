# xECG experiments after the current adaptation study

**Design note: 24 September 2026. Status: proposals, not implemented or queued.** The authorized CPC mismatch and cross-lead studies have separate protocols. This note investigates xECG improvements beyond [Experiment 008](experiment-008-vision-ssl.md); it does not change that experiment or reserve GPU time.

## Recommendation and what the release already supplies

Start with a **frozen-encoder readout that can combine distributed evidence with a brief strong response**. It needs one feature extraction and inexpensive classifier training. If a larger architectural experiment is warranted, test a **zero-initialized morphology adapter spanning adjacent patch boundaries**, retaining every released xECG weight. Both are hypotheses about this task; their value and methodological novelty are unestablished.

The inspected release uses physical-mV signals at 100 Hz, a 12-to-1024 linear convolution with kernel and stride 25, and 40 nonoverlapping tokens for ten seconds. Its nine xLSTM blocks reverse token order before blocks 2–9, giving bidirectional context; final tokens return to their original order. Average/max pooling and token outputs already exist. The authors also provide patch-level classification. Neither token classification nor max pooling is a new xECG mechanism. [Released encoder](https://huggingface.co/riccardolunelli/xECG_base_model_v1/blob/main/xECG.py), [authors' task heads](https://github.com/dlaskalab/bench-xecg/blob/release/xecg/downstream_models.py).

Experiment 008 already tests masked EMA prediction, coding-rate expansion, frozen-reference Gram retention, and uniform versus rarity-weighted visible-token retention. The proposals below change downstream aggregation or the waveform-to-token interface. They do not repackage those existing losses.

## Priority 1: distributed evidence plus a brief response

**Question.** Does ordinary mean pooling leave useful variation across the 40 contextual tokens inaccessible to the current linear classifier? A positive result would show a readout benefit; it would not prove that the encoder forgot local morphology. Its contextual tokens already incorporate information from other times.

Strictly load the original released checkpoint used by Experiment 007, chosen before any 008 test ranking. Freeze it in evaluation mode. Extract its 40 final 1024-wide token vectors once per existing eligible PTB record, retaining record IDs and split membership. Use the complete-record all-valid mask path already checked in `xecg_adaptation.encode_tokens`, and verify agreement with ordinary unmasked inference on real records. No feature standardization, learned statistics, or model selection may use calibration or test patients. FP32 token storage for 19,126 records is approximately 3.13 GB before metadata; a disk cache permits head training without further encoder passes.

Let `z_p` be token `p`, `mu = mean_p(z_p)`, and `g(z) = Linear(32,1)(GELU(Linear(1024,32)(z)))`. Keep one global linear term `w dot mu + b`, then compare these heads:

| Arm | Record logit | Purpose |
| --- | --- | --- |
| Linear reference | `w dot mu + b` | Existing frozen mean-pool reference |
| Mean local branch | `w dot mu + b + mean_p(g(z_p))` | Same local network and parameters as proposed branch |
| Smooth local branch | `w dot mu + b + logsumexp_p(g(z_p)) - log(40)` | Proposed soft emphasis on stronger token responses |
| Max local branch | `w dot mu + b + max_p(g(z_p))` | Same-parameter established extreme-pooling comparator |

Use logsumexp temperature 1, fixed before results. Initialize the local network's final linear layer, including its bias, to zero in all three branch arms. Copy identical global-head and hidden-layer initial states across arms. All branch heads initially match the linear reference. Initial gradients can reach the final local layer; its preceding hidden layer starts learning once that layer moves from zero. This creates approximately 33,000 additional trainable parameters, which must be counted from the implementation.

The primary comparison is smooth minus mean local branch: extra parameters, per-token nonlinearity, feature exposure, and optimization are matched. Smooth minus max tests whether a boundedly smooth response helps relative to an established hard maximum. The smaller linear arm is a practical reference, not a capacity-matched attribution control. The fixed smooth temperature does not bound score magnitudes; monitor whether learned token-score scale makes the branch behave like a maximum.

Use both existing labeled manifests: 1,518 and 15,360 records. Initial proposed head schedule is AdamW at `1e-3`, weight decay `0.01`, batch 128, 40-epoch ceiling, patience 8, development-AUROC selection, and shared seed/data order. No expensive encoder adaptation or SSL is required. Freeze this schedule before launching and report convergence; do not grant one head a larger tuning budget. Four heads at two budgets make eight small supervised runs after a single extraction. Full-label performance is primary; the 10% result tests label efficiency.

**Failures to look for.** The maximum or smooth branch may magnify artifacts, electrode transients, or arbitrary contextual-token outliers. A token score is not a clinically validated event localization. A repetitive abnormality may benefit mainly from the retained global branch. The broad binary proxy need not be determined by a single positive instance. Inspect a fixed training-only set and report concentration of local scores, association with signal amplitude/quality where available, and whether the selected responses are stable across seeds. Do not treat unusual waveforms as disease or equate study-negative records with healthy people.

**Related work.** ECG multiple-instance learning predates these proposals; for example, Shanmugam et al. study record outcomes using heartbeat instances. That is related motivation, with a different population, duration and endpoint. [Multiple Instance Learning for ECG Risk Stratification, 2019](https://proceedings.mlr.press/v106/shanmugam19a.html). Beat-SSL combines heartbeat and rhythm contrastive learning, providing another precedent for local/global complementarity; this readout is not a reproduction of its training method. [Beat-SSL, 2026](https://arxiv.org/abs/2601.16147).

**Interpretation boundary.** Compare frozen heads with other frozen heads. Experiment 007 fine-tunes the backbone, so a difference from 007 alone does not isolate pooling. If frozen results motivate trainable-backbone confirmation, predefine a separate smooth-versus-mean comparison using the same 007 optimizer and both label budgets. Do not select an 008 checkpoint or a head on test results.

## Priority 2: preserve the checkpoint and add a boundary-spanning morphology adapter

**Question.** Does allowing a small nonlinear waveform stem to see both sides of a 250 ms patch boundary improve transfer with the released sequence model? The linear patch embedding maps 300 input values to 1024 values; a large patch is not automatically an information bottleneck. The hypothesis concerns a useful local inductive bias and transfer optimization, not proof of lost waveform information.

Use the exact existing 100 Hz, twelve-lead input and original 40-token grid. Avoid changing sample rate, lead order, token count, recurrent depth, or the released weights in this initial architecture test. For physical waveform `x`, original patch embedding `E(x)`, and a small adapter `A(x)`, send `E(x) + A(x)` into the unchanged xLSTM stack.

One implementable adapter is:

1. A depthwise temporal convolution, twelve channels, kernel 25, stride 1, symmetric padding of 12 samples, without bias.
2. A pointwise 12-to-64 convolution with bias and GELU.
3. Average the resulting sample-level features within the original consecutive 25-sample patches, yielding `[40,64]`.
4. A bias-free 64-to-1024 projection, initialized to zero, yielding `[40,1024]` residuals.

The adapter has about 67,000 parameters; count exact values in code. The original 57,021,472 released parameters still load strictly into their original submodule. At initialization the modified encoder must numerically reproduce the release. Earlier adapter layers should have ordinary nonzero initialization: zeroing the whole branch would prevent useful learning. The proposed stem uses approximately 490 ms of waveform support for an interior pooled adapter token. It is intended for offline full-record classification, consistent with xECG's bidirectional context, and makes no causal-inference claim.

Use three arms, with the same final mean-pool linear classifier:

| Arm | Adapter input processing | What it establishes |
| --- | --- | --- |
| Released reference | No adapter | Practical cost/performance reference |
| Within-patch adapter | Run the same depthwise filter separately inside each 25-sample patch with the same zero padding; then mix, pool, project | Added trainable local capacity and nonlinear waveform processing |
| Across-boundary adapter | Run the identical filter across the complete signal, then mix, pool, project | Extra benefit from waveform context spanning patch boundaries |

Across-boundary minus within-patch is primary. Both have identical trainable tensors, initialization and downstream readout. Internal zero-padding artifacts are a limitation of the within-patch control and must be inspected; the comparison estimates the whole boundary-continuity change, not a unique physiological mechanism. Both share the same outer recording-boundary treatment.

For a bounded first pilot, freeze the released encoder parameters and train only adapter plus classifier. Proposed ceiling: 1,000 optimizer updates per arm and label budget, effective batch 32, AdamW `1e-3`, weight decay `0.01`, 50-update warmup followed by cosine decay to `1e-4`, development evaluation every 100 updates, and development-AUROC checkpoint selection. The reference trains only its classifier for the same example/update budget. Start real-device profiling at microbatch 2 with accumulation 16, then freeze one supported microbatch for both adapter arms before comparison. Both existing label budgets remain necessary; do not replace the 10% sample with a new draw. This is a suggested launch design, not an executed protocol.

**V100 limitation.** Freezing the xLSTM parameters saves their optimizer states but does not eliminate its backward activations: gradients must traverse the stack to reach the input adapter. Its runtime/memory must be measured with forward, backward, optimizer state, and save/restore. Do not claim the pilot is as cheap as frozen-feature head training. If the supported backend cannot fit, investigate activation checkpointing with a documented numerical/recovery check before revising the design; no silent change to the objective or architecture.

**Failures to look for.** The stem may learn noise, instrument/source signatures or amplitude cues. The pretrained stack may suppress the small residual, or the residual may grow until it disrupts pretrained feature scales. Track residual-to-original-embedding norm, feature variation, gradients and development calibration. A negative result could mean that the frozen stack cannot use the added stem; it would not establish that local morphology is unhelpful. Keep any subsequent whole-backbone adaptation as a separate comparison.

**Related work and distinction.** Local morphology plus broader rhythm is an established ECG direction, including Beat-SSL and the 2026 ECG-NAT preprint's hierarchical temporal architecture. [ECG-NAT](https://arxiv.org/abs/2605.13194). The proposed controlled adapter is a project-specific hypothesis around the released xECG interface. A literature search would still be needed before claiming a novel architecture. This is distinct from 008 because it changes the input representation rather than preserving teacher features with another retention loss.

## Shared evaluation and decision rules

Retain the patient splits, eligible records, 10% label identities, Platt calibration, calibration-selected threshold attaining at least 95% sensitivity, and paired patient bootstrap used in Experiments 007–008. Report both observed test sensitivity and specificity at that frozen threshold: calibration targeting 95% does not guarantee 95% sensitivity on test patients. Include AUROC, AP, Brier score, parameters, forward/backward counts, wall time and peak memory. Compare paired differences directly and retain all arm results.

The existing test cohort has already been examined, and one seed does not establish retraining stability. These are exploratory task improvements, not evidence of clinical referral validity, performance among university students, or a claim that normal ECG appearance establishes health. Checkpoint pretraining-overlap limitations from Experiment 007 remain applicable.

**Order:** finish already authorized studies; extract frozen features and run the inexpensive readout comparison first if subsequently authorized; consider the controlled morphology adapter afterward. No xECG training or queue mutation was performed for this design note. Any launch requires its own frozen protocol, implementation checks and actual V100 profile; these prerequisites are technical readiness requirements, not evidence of completed performance.
