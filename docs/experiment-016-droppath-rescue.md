# Experiment 016 rescue: residual stochastic-depth semantics

Astra design, 25 September 2026. Proposed successor; not implemented or launched.
One experiment at a time. Preserve every historical source, checkpoint and
receipt. This protocol uses development outcomes only; calibration and test
remain closed.

## Why this experiment

The original 016 frozen probe reached 0.96195 development AUROC, but two-epoch
fine-tuning selected 0.94513 with a random head and 0.93138 with the probe head.
Before any updates, the probe's BCE on 128 fixed training records jumped from
0.236 in evaluation mode to 1.345 in training mode; median feature cosine was
0.303. Those observations identify a mode mismatch, not its cause.

The pinned `third_party/checkpoints/xecg/xECG.py` provides a concrete candidate.
Its DropPath wrapper returns `block(x)/(1-p)` when retained and `x` when dropped.
The pinned `xlstm/blocks/xlstm_block.py` already includes identity residuals.
Writing the complete block as `B(x)`, the conditional expected wrapper output
is therefore `B(x) + p*x`, not `B(x)`. Even an identity block is changed. This
is an algebraic fact about the inspected implementation, not yet evidence
that it explains the performance decline. The upstream [released source](https://huggingface.co/riccardolunelli/xECG_base_model_v1/blob/b060532/xECG.py)
contains the same behavior. The released configuration uses dropout 0 and
drop-path 0; our historical fine-tuning used a depthwise 0-to-0.5 schedule.

A residual-preserving alternative is `x + mask/(1-p) * (B(x)-x)`, whose
conditional expectation is `B(x)`. This follows the residual-branch placement
in [stochastic depth](https://arxiv.org/abs/1603.09382) and its
[Torchvision implementation](https://docs.pytorch.org/vision/main/_modules/torchvision/ops/stochastic_depth.html).
It preserves the conditional mean of one block, not the expectation of the
entire nonlinear network. Disabling stochastic depth provides a distinct
control for stochastic variation itself.

## Fixed comparison

Use the clean 15,359 full-label PTB training records from
`outputs/data_quality/clean_rerun_preflight_v1`, the existing 1,306 development
records, and the original patient partition. Do not use the limited-label
condition or expanded union in this first mechanistic study. Strictly load
the same released xECG weights and vanilla backend as 016. Preserve its 100 Hz
physical-mV twelve-lead waveform transformation and cache identity.

Refit the frozen probe on those 15,359 training feature rows. Use train-only
StandardScaler, logistic regression with `C=0.01` fixed from historical 016,
`lbfgs`, 3,000 iterations, and seed 42. There is no new regularization search.
Verify cached feature, waveform and record/patient identities. Fold the scaler
into the affine classifier and check native logits against the probe before
training. The resulting frozen probe is the practical reference.

Every trained arm starts with this identical probe head and released backbone:

| Arm | Maximum drop probability | Training operation on complete block B |
| --- | ---: | --- |
| Legacy | 0.5 | `where(mask, B(x)/(1-p), x)` |
| Residual | 0.5 | `x + mask/(1-p)*(B(x)-x)` |
| Off | 0 | `B(x)` |

Retain the nine-block linear probability schedule, sequence reversals, pooling,
normalization and all block computation. Legacy and Residual use identical
per-record masks at each corresponding block/update. Off evaluates every block
and may consume/discard the same mask stream for explicit RNG matching. Use
separate, saved RNG streams for mask draws and record permutation. At inference
all three execute `B(x)` and initially give identical features and logits.

Run seed 42 for exactly two epochs, 30,718 record exposures and 480 optimizer
updates per arm. No early stopping or schedule extension. Retain 016 AdamW:
encoder LR 3e-5, head LR 1e-3, layerwise decay 0.75, weight decay 0.1, gradient
clip 3, microbatch 16, effective batch 64, one-epoch warmup then cosine decay.
Use identical ordered records and loss weighting, including the short final
batch, and fresh optimizer states. All other dropout remains zero. No learning
rate, drop-probability or epoch search is allowed inside this experiment.

## Mechanism checks before training

First prove both branches analytically and with deterministic mask tests:
identity B, affine residual B, p=0/evaluation equivalence, kept/dropped examples,
and gradients of inputs and block weights. Verify local and released source
hashes; install the experimental wrapper only on newly created model objects.
Do not edit downloaded xECG source or historical `load_xecg` behavior.

Use the first 64 positives and first 64 negatives in the clean training
manifest for diagnostics. Save their IDs. With identical frozen weights, run
evaluation once and training mode with eight paired mask seeds 16000–16007.
Report mean/range of BCE, AUROC, logit shift, feature cosine and feature norms
for all arms. Check that Off train/eval agree within documented FP32 tolerance.
Save blockwise norm shifts and exact per-block conditional-mean checks on
captured fixed inputs. Global BCE need not improve merely because blockwise
expectations are corrected. Diagnostic draws must not consume training RNG.
If numerical, gradient, source or train/eval equivalence checks fail, stop
before training and report the specific correctness failure.

## Outcomes and interpretation

The primary performance contrast is **Residual minus Legacy at final epoch 2**.
Report Off minus Legacy and Residual minus Off, all epoch-0/1/2 outcomes, AP,
BCE, and five-fold patient-group sensitivity/specificity with the established
95%-sensitivity threshold policy. Best-of-two development AUROC is descriptive
only; it cannot replace the fixed final-epoch primary endpoint. This avoids
using a favorable stopping point to explain the original decline.

At initialization, after the first optimizer update and at each epoch end,
evaluate the fixed training diagnostic set in evaluation mode. Log probe-head
norms, layerwise relative parameter change, gradient norms/clipping, feature
cosine versus the release, and logits under the original frozen probe head on
the current backbone. These separate immediate mode disturbance, feature drift
and classifier adaptation. They do not independently prove mediation.

Use 2,000 paired patient-cluster development bootstrap draws, seed 16016, for
the three prespecified final-epoch AUROC differences. Resample patients with
all their records, report valid draws and label single-class draws invalid.
These intervals measure patient sampling, not training-seed uncertainty.

Decision rules are practical exploratory screens, not claims of significance:

- **Correction supports the mechanism:** Residual beats Legacy by at least
  0.005 final-epoch AUROC and its average initial train/eval logit shift is
  smaller. If it still trails the frozen probe by more than 0.002, report a
  partial explanation with no useful fine-tuning rescue yet.
- **Practical rescue worth replication:** Residual or Off beats Legacy by at
  least 0.005, is within 0.002 AUROC of the frozen probe or better, and loses
  no more than 0.005 cross-fold sensitivity versus the probe at epoch 2.
  If both pass, prefer Residual only when it exceeds Off by at least 0.002;
  otherwise choose Off as the simpler recipe. This choice is frozen in advance.
- If only Off passes, the evidence supports removing this stochastic-depth
  package; it does not establish the scaling correction as the sole cause.
  If neither passes, stop this rescue path and retain the frozen probe.
- Any passing recipe still needs a separately costed seed-43 matched replication
  of all three arms before calibration/test. Seed 43 changes optimization only,
  never patient/label partitions. This proposal does not schedule that follow-up.

## Execution and budget

Sol should add a small experimental DropPath module, a new staged runner and
meaningful tests; reuse stable library utilities without importing experiment
entry points into reusable code. Freeze successor sources, protocol, cohort,
cache, feature, released-weight and environment hashes before any GPU run.
Profile a complete epoch, development pass and full checkpoint roundtrip for
each of the three arms, including real input verification/loading. Discard all
profile models. Measure diagnostics/probe/report costs as well.

Require `measured preparation + measured profiles + 1.25 * (six projected
training epochs + remaining diagnostics/report/checkpoint time) <= 7200s`.
Use the slower complete profile pass for every projected epoch. Historical 016
passes were about 515 seconds, so this appears feasible but must be measured.
If the gate fails, do not reduce epochs or omit controls after seeing results.
Save the profile and propose a separately frozen smaller study.

GPU work remains sequential under the shared lock with live process checks.
Checkpoint model, optimizer, scheduler, sampler, RNG and exact update position;
verify resumed next-update equivalence on the same device and CPU tensor
serialization equality. Preserve downloaders. Actual commands, sources,
receipts, outcomes and wall times belong in a new output directory and both
queue documents. No current executable manifest is modified in place.

## Why the other failed paths wait

| Path | Evidence and priority decision |
| --- | --- |
| 014 fusion | Both original and clean probe fusion gates failed. The CPC probe was weaker than fine-tuned CPC; stronger-model fusion is cheap and plausible, but changing predictors tests complementarity rather than explaining this failure. It also adds another selection loop on the same development patients. |
| 015 distillation | Full-label AUROC +0.0016 and limited-label -0.0017; feature variance stayed around 91–95% of control, so collapse is not supported. Changing teacher target/loss weight without a gradient-conflict diagnostic would be speculative tuning. |
| 008 adaptation | Deferred for estimated 27.7-hour suite and OOM at the original microbatch, not a completed negative model result. Resolving xECG transfer behavior first is cheaper than adding costly SSL objectives atop it. |
| 010 cross-lead | Deferred after measured random-I/O cost, not predictive failure. New staging may solve runtime, but requires a fresh complete-path profile and controlled continuation study; it offers no existing performance signal to rescue. |
| 006 token/chunk | Chunking lost to the native grid; learned boundaries improved the fixed-chunk control without recovering native performance. The successful native-grid 017 work already pursues the stronger morphology direction. |
| 009 mismatch | Residual features underperformed matched ordinary features at both budgets. Removing predicted information can discard useful signal, but that explanation has not been isolated; a new residual formulation lacks the direct implementation-level defect found here. |

These are diagnostic-annotation proxy experiments, not clinical validation.
Earlier work has inspected the historical test population, so even a future
test result on it would not be a new external confirmation cohort.
