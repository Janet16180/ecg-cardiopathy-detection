# Experiment 016 v7: frozen-backbone readout audit

**Astra design frozen 25 September 2026 before implementation or outcome
calculation.** This is one bounded diagnostic study. Stop further xECG encoder
fine-tuning; no learning-rate, clipping, duration or regularization sweep is
justified by the current evidence. The only proposed fitting is a fixed linear
probe on already frozen representations. Calibration and test stay closed.

## What the v6 result establishes

[V6](experiment-016-droppath-rescue-v6-results.md) found Residual minus Legacy
AUROC +0.03552, with a paired patient interval +0.02646 to +0.04470. Off achieved
almost the same improvement. Under the matched one-epoch recipe, changing the
DropPath rule therefore materially reduced harm. The conditional-mean defect
is real, and its correction is a useful intervention in this setting. The
result does not establish that the defect is the sole mediator of the gain,
or that Residual is better than removing DropPath: Residual minus Off was
+0.00042 with interval -0.00657 to +0.00758.

Residual remained 0.01372 AUROC below the frozen probe; Off remained 0.01413
below. Off has no initial train/eval mode difference, so stochastic-depth mode
mismatch cannot explain all remaining decline. This is not merely a probability
calibration problem: the AUROC reduction means discrimination/ranking changed.
The experiment does not isolate encoder changes from head updates, the joint
optimizer, or the endpoint of the rising-learning-rate warmup.

On the fixed balanced 128-record training diagnostic, final feature cosine to
the release was about 0.977 for Residual and Off, while mean absolute logit
shift under the original probe was about 1.97 and 2.05. Legacy had the highest
feature cosine, about 0.993, despite the worst AUROC. Therefore global cosine
or small parameter changes cannot certify preserved task information or prove
its destruction. The relevant discriminating direction may change without a
large global rotation.

Fixed-set evaluation-gradient norms were 26.77 initially and 105.33/24.56/38.04
finally for Legacy/Residual/Off. These are gradients of a selected, balanced
training subset, not full-training gradient or clipping-frequency histories.
The initial zero-LR update changed no model tensors, although optimizer moments
can update. Neither these norms nor the missing final training-batch norm
justify selecting a new learning rate or clip threshold. One optimization seed,
one warmup epoch and a repeatedly inspected development cohort also limit
generality. A patient bootstrap does not measure training-seed variation.

## One prioritized hypothesis

**The final Off backbone retains train-learnable linear discrimination that
the jointly updated classifier fails to recover.** Test whether a newly fitted
fixed-recipe linear readout recovers most of its development gap to the release.
Off is primary because it eliminates stochastic-depth mode mismatch entirely.
Residual and Legacy are prespecified secondary controls, not alternative winners
to select if the Off result fails.

This asks about a recoverable readout component. Re-fitting also changes the
head's optimization and regularization relative to joint AdamW, so a positive
result cannot isolate head nonconvergence from regularization or prove no
representation information was lost. Conversely, failure under this fixed probe
does not prove all possible nonlinear readouts must fail.

## Frozen states, data and readouts

Use four backbone states: the original released xECG, and the exact final
Legacy/Residual/Off checkpoints from
`outputs/experiment016_droppath_rescue_v6/{legacy,residual,off}/resume.pt`.
Verify their hashes against completion receipts and audit evidence, with exactly
240 completed updates and the seed-43 cohort identity. Never resume optimizer
training or use a v5 profile checkpoint.

Use the same 15,359 clean full-label training ECGs and 1,306 development ECGs
from 1,173 patients. Preserve patient partitions, input order, physical-mV
twelve-lead 100 Hz preprocessing, vanilla backend and exact 1,024-wide pooled
representation. Extract with `eval()` and gradients disabled. Every encoder
tensor remains immutable. The released feature cache may be reused after
matching its waveform/identity/checkpoint/feature hashes; select only the
designated clean train and development rows from it.

For each backbone independently, fit StandardScaler **only on its clean
training features**, followed by LogisticRegression with C=0.01, `lbfgs`,
max_iter=3000, default intercept, and random_state=42. This is the same probe
recipe in the released reference. No development-selected C, fit iterations,
feature subset or backbone checkpoint. Freeze row order and solver/environment.
Nonconvergence at the fixed limit is a failed probe fit, not permission to tune.
The released refit must reproduce the recorded 0.9619404 AUROC and stored
reference logits within a declared numerical tolerance before any contrasts.

On each adapted backbone, additionally evaluate its saved joint-training head
and the original frozen released-probe head. These require no fitting. Verify
the joint-head logits reproduce the saved v6 development predictions. The
readout matrix is therefore three heads on each of the three adapted backbones,
with the released probe as common reference. Keep the distinction between
raw native features and standardized probe features explicit when evaluating
saved affine heads.

## Primary contrast and stopping rules

The primary contrast is **Off refitted-probe AUROC minus Off saved joint-head
AUROC** on the unchanged development patients. Report both absolute values and
the Off-refit-minus-released-probe difference. Residual/Legacy re-probe effects
and original-head-on-adapted-feature effects are secondary descriptions.
Report AUROC, AP, BCE, and the unchanged five patient-group threshold folds
targeting 95% sensitivity on the other four folds. Save all logits locally.

Use 2,000 paired whole-patient bootstrap draws with analysis seed 16017, common
across all contrasts, and report invalid/single-class draws. Do not select heads
or refit models inside bootstrap samples. Explicitly retain the conditional,
postmortem nature of inference on this inspected development cohort.

- **Recoverable readout component:** Off refit improves over its joint head by
  at least 0.005 AUROC. Report the paired interval; if it includes zero, describe
  sampling uncertainty rather than confirming the component.
- **Near-complete practical recovery:** additionally Off refit is within 0.002
  AUROC of the released probe and loses at most 0.005 patient-fold sensitivity.
  This supports linear readout recovery on the final Off representation. It is
  not a new independent test result or proof that fine-tuning helps.
- **Potential useful adaptation:** Off refit exceeds the released probe by at
  least 0.002 AUROC, with the same sensitivity condition. It is only a signal
  for separately frozen replication; do not launch it automatically.
- Otherwise retain the released frozen probe. If every re-probed adapted state
  still trails it by more than 0.002, record failure to recover comparable
  linear discrimination under the fixed probe and close this xECG rescue line.
  Do not proceed to an LR, clipping, early-stopping or C sweep.

Even if a secondary arm looks better, do not change the primary hypothesis,
claim Off passed, select it for calibration/test or automatically schedule a
follow-up. All outcomes end this bounded diagnostic; any next experiment needs
a separate decision and protocol.

## Cost and execution handoff

No GPU optimizer steps are required. Historical full train/development release
extraction was about 190 seconds, but that is a planning reference only. About
205 MB of new float32 feature arrays are needed for three adapted backbones.
Use the shared GPU lock for sequential inference, preserve downloaders, and
write new files under `outputs/experiment016_frozen_readout_audit_v7` only.

Sol should implement new source/test/entry-point files, not modify any v5/v6
source or receipt. Provide `check`, `extract`, `probe`, and `report` stages.
Before feature extraction, validate identity, strict checkpoint loading,
frozen/eval mode and release-cache/native-head agreement on a fixed training
sample. Meaningful tests cover ID joins, train-only scaling, state immutability,
joint-head reproduction, feature/head dimensional contracts and decision gates.
Freeze a new source map, actual commands and executable manifest.

Extract Off first on the complete 16,665-record train-plus-development path,
including load, inference, checkpoint identity checks and feature writes.
This produces declared frozen features and measures actual full-pass cost; it
is not encoder training or a disposable performance profile. Hash and retain
the resulting cache without calculating comparative outcomes yet. Time the
fixed released-control probe fit on CPU to measure its real solver path.

Freeze the 7,200-second gate:

```text
T = max(190 seconds, measured complete Off extraction seconds)
P = max(60 seconds, measured released-control probe fit seconds)
projected_total = actual preparation/Off/control-fit time
                  + 1.25 * (2*T + 3*P + 300 seconds)
require projected_total <= 7200 seconds
```

The allowance covers remaining control checks, bootstrap and reports; update
the gate with actual elapsed times after each stage. If it fails, stop without
partial performance selection. Historical evidence suggests minutes rather
than another fine-tuning suite, but no completion time is promised. Preserve
resumable feature chunks with identity hashes. Store scaler/probe coefficients,
convergence receipts, state/feature hashes and all negative results. Update both
queue documents and finish with no running queue or automatic follow-up.
