# Experiment 016 v10: second-seed encoder-update replication

**Astra protocol frozen 25 September 2026 before implementation or seed-47
outcomes.** This is one development-only replication, handed to Sol only after
design. It preregisters the local analysis before execution; it is not an
external registry submission. Preserve all v5–v9 sources, executable manifests,
profiles, checkpoints and predictions. Calibration and test remain closed.

## Question and prediction

[V9](experiment-016-encoder-motion-v9-results.md) found that permitting encoder
updates reduced joint-head AUROC relative to its compute-matched frozen control:
F minus M was +0.006067, with a paired-patient interval of +0.001696 to +0.010901.
The readout-gap difference was +0.008224. M's fixed-C refit retained measured
linear usefulness. These are encouraging results from one optimization seed,
46, on repeatedly inspected development patients.

Repeat the identical intervention with **optimization seed 47**. The single
prioritized hypothesis is that the joint-head harm and enlarged readout gap
persist under a different training-record permutation. Prediction: at update
240, F joint minus M joint is at least +0.005 AUROC; M refit remains within
0.002 of F refit or better; M's excess readout gap is at least +0.005.

This changes training randomness, principally record order. It does **not**
change pretrained encoder initialization, patient sampling, task, or data
cleaning. “Second encoder-training seed” must not be described as a second
pretrained encoder or an independent patient validation.

## Fixed scientific execution

Inherit [v9's protocol](experiment-016-encoder-motion-v9.md) and its verified
production mechanics in full, except for seed 46 → 47 and the new artifact
identity. No third arm, hyperparameter search, additional epoch, or selected
checkpoint belongs in this experiment.

| Item | Frozen requirement |
| --- | --- |
| Data | Same 15,359 clean labeled training ECGs; 1,306 development ECGs from 1,173 patients; same row order, patient partitions, labels, excluded ECG 12722, preprocessing and waveform cache |
| Initialization | Identical released xECG and original clean C=.01 probe head, with fresh optimizer moments; never a v9/profile/bridge checkpoint |
| Architecture | Same vanilla backend, raw physical-mV twelve-lead 100 Hz input, all dropout and stochastic depth Off |
| M | Original encoder base LR 3e-5 and layerwise decay .75; head LR .001 |
| F | Encoder base/scheduled LR exactly zero; all encoder gradients and optimizer groups retained; head identical to M |
| Optimization | FP32 V100, AdamW betas (.9,.999), epsilon 1e-8, weight decay .1 with identical grouping including head bias; BCE; global clip norm 3 including encoder gradients |
| Exposure | Same seed-47 permutation in M and F; microbatch 16/effective batch 64; 15,359 exposures, 240 updates, last effective batch 63 |
| Schedule | Original warmup multiplier k/240 at update indices k=0,...,239; stop at 240 |
| Final probes | Independently fit the same train-only StandardScaler plus C=.01 lbfgs probe, max_iter=3000, random_state=42, on each final encoder's features; require convergence |

F keeps `requires_grad=True`, full backward, encoder gradients in clipping and
full Adam state. No detached/cached-feature training or head-only clipping.
F's encoder parameters and buffers must remain bitwise unchanged every update.
Actual clip factors may differ after the encoder trajectories separate; this
is a downstream consequence included in the total intervention effect.

Retain the exact v9 data path in production, including full final train and
development feature extraction **for both arms**, native/affine logit checks,
the F released-feature/refit positive control, checkpointing every 40 updates
and on interruption, and final same-device replay. Do not introduce a cached F
extraction shortcut when claiming executable/profile compatibility.

Log every update's batch digest, losses, learning rates, encoder/head/global
preclip norms, clip factor and head-step norm. Retain the same fixed 128 training
ECGs and diagnostics at updates 0, 1, 60, 120, 240, restoring mode and RNG.
There is no intermediate development selection. The final endpoint is the
same one-epoch warmup endpoint, not a converged-fine-tuning comparison.

## Measurements and decisions frozen before seed 47

The sole primary contrast is `D47 = AUROC(F_joint) - AUROC(M_joint)` at update
240. Report all four readouts and the unchanged released probe with AUROC, AP,
BCE and v9's five patient-group threshold-fold sensitivity/specificity screen.
Define:

```text
R47 = AUROC(M_refit) - AUROC(F_refit)
gap_M47 = AUROC(M_refit) - AUROC(M_joint)
gap_F47 = AUROC(F_refit) - AUROC(F_joint)
G47 = gap_M47 - gap_F47
```

Use **2,000 paired whole-patient bootstrap draws, seed 16020**, with all ECGs
of each sampled patient retained, multiplicity preserved, and one common draw
across every readout and both optimization seeds. Report invalid single-class
draws, valid count and percentile 95% intervals; fewer than 1,900 valid draws
makes interval inference unresolved rather than authorizing a bootstrap change.
Bootstrap RNG never seeds training or label selection.

- **Joint-head harm replicated:** D47 >= .005. Report separately whether
  its interval lower bound exceeds zero. An interval containing zero is an
  uncertain positive point-estimate screen, not a confident replication.
- **Narrower readout-gap pattern replicated:** additionally R47 >= -.002
  and G47 >= .005. Report both secondary intervals without treating R47's
  point threshold as equivalence/noninferiority proof.
- D47 between 0 and .005 is directional agreement below the frozen practical
  threshold; D47 <= 0 is nonreplication of the predicted direction. If D47
  passes but R47 < -.002, the readout interpretation fails and representation
  loss may contribute. If G47 fails, report that separately. Do not change
  thresholds, seeds, budget or decision labels after seeing outcomes.
- An incomplete/invalid M/F pair yields **no completed replication result**.
  Preserve partial artifacts and the stopping reason, without interpreting
  an unmatched arm as the primary contrast.

Display seed 46 and 47 point estimates side by side. The existing v9 report and
its seed-16019 intervals remain immutable. In the **new** report compute each
contrast's arithmetic mean across the two fixed optimization seeds and its
paired-patient interval using the shared seed-16020 draws. Also show the two
individual values and their range. Average contrasts, not logits; this is not
an ensemble or a chosen-seed result. The combined mean cannot override a failed
seed-47 gate. Recomputed seed-46 intervals under the common draws are labeled
as combined-analysis intervals, not replacements for the published v9 ones.

Patient draws quantify conditional patient-sampling variation given these two
fixed trained pairs. Do not treat four sets of predictions as extra patients,
bootstrap two seeds to claim reliable seed-population uncertainty, or present
the development set as independent of the investigation. Two consistent seeds
strengthen robustness to order; two seeds cannot establish a universal effect.

## Profile reuse requires executable compatibility

V9 actually profiled both complete pipelines, including all 240 updates,
logging, development inference, full feature extraction, fixed probe and
checkpoint replay. These verified full-path receipts may serve as v10 cost
evidence; the short bridge below does **not** replace them.

Bind and verify these historical identities before allowing reuse:

| Evidence | SHA-256 |
| --- | --- |
| v9 profile manifest | `9d6a2b033db352c9974adfbbf77a1b2d3cada5c5bb4107c3f5095fefddb1a676` |
| v9 profile source map | `9cb55fc92d1930fb3d10539a157213609c17190078900b27d96d068d6eff2a09` |
| v9 production manifest | `0b3b2456e94ea1a21a6bae24d720d893460aa78f1d2351406d2c47659396d45f` |
| v9 production source map | `f52010d13e4f3b03a06605a17d5ba12721e59e35df2f2e9d81d8fa64e310b405` |
| v9 report | `938bc656167a01afa5cf38100a1f042a98ad9d2a843390fcb2f1f8c7f7eaf50b` |
| v9 reusable training module | `a5311a0c6a03ed4ed7f63759b7ccfe1963dd9da4d733c82e579cd34aa4482839` |
| v9 execution script | `decc4f2aaa09c5ae0e387b96a71d59b5a43334f6487b58a43cef4ce229d9316c` |
| M profile completion receipt | `8bc6503c04db9dfbf3d9763aa98430cf433504f03580bd48349c637be0c11ea6` |
| F profile completion receipt | `74eff7c7804883cead2217547c91af7d73e3f09705b135094eae0f90615e952b` |

Rehash source-map entries, predecessor completion artifacts and full input
fingerprints, not merely these top-level JSON files. Verify v9 profile.json and
cost_gate.json against their completion receipts, including P, H and passed
correctness flags. Record the new source revision, complete source map,
protocol hash, exact commands and environment. Require the same dependency
versions/lockfile, vendored encoder implementation, device class, precision,
CUDA backend/settings, cache/input identity and execution graph. Record current
GPU/driver information against available historical launch/environment evidence.

Create a machine-readable **compatibility receipt** mapping v10's production
functions and dependencies to their hashed v9 implementations. Prefer importing
the unchanged reusable training primitives. If orchestration is copied into
new files, verify function AST identity wherever unchanged, and retain a
reviewed, explicit diff for necessary changes. Allowed changes are output/
run identity, seed 47, compatibility/bridge/gate orchestration and the new
two-seed report. Renamed bindings must still resolve to the same scientific
implementations. No global mutation/monkeypatch of v9 paths or seeds. An opaque
wrapper around a changed training path is not compatibility evidence.

Include a focused CPU check of common initialized tensors, seed-47 order,
optimizer/scheduler groups and initial logits; confirm its permutation differs
from seed 46 while matching between M/F. Inherited unit checks remain valid
only for byte-identical code. Add checks for new identity/gate/bootstrap logic,
including rejecting wrong seed, reused old checkpoints, missing artifacts,
changed executable paths and incomplete pairs. If compatibility cannot be
established, **stop before production**. Do not replace full profiling with
minibatch extrapolation or silently relax the rule. A changed scientific path
requires another frozen design and full profiles outside this replication.

## Bounded GPU bridge and inherited cost gate

After compatibility and CPU checks, use the shared GPU lock and a fresh,
separate **seed-47 bridge**, bounded to **300 seconds total GPU-stage wall
time**. Run M then F, with one active GPU model at a time. Each performs exactly
the first three real effective-batch updates from the full seed-47 permutation,
with the unchanged 240-update scheduler; use full-size microbatches/gradients.
Check common initial tensors/order and the first two updates' batch identity,
loss, norms, clip factor and head-step norm using v9's tolerances. M must move
after the first nonzero step; F must remain bitwise fixed, with both heads
moving. Exercise save/restore and two sequential replays of the next update
from this bridge checkpoint, using exact CPU serialization and v9's FP32
`atol=1e-8, rtol=1e-6` next-update criterion, with exact counters/order/RNG/
scheduler. Record device and peak allocated/reserved memory. No development
metrics are evaluated in the bridge.

Bridge artifacts are verification only. Delete neither historical evidence nor
failed bridge receipts, and never resume bridge weights for production. Fresh
production starts from the released tensors and seed 47 again. A bridge timeout,
OOM, changed dependency/input, failed replay, identity or gradient-control check
stops this experiment. Do not reduce microbatch, precision, gradient scope or
diagnostics to squeeze through.

Preserve the **7,200-second ceiling** with conservative historical profile
charging, even though those profiles have already been run:

```text
H = 2693.7965717150364 seconds  # verified v9 preparation plus both full profiles
P = 1098.4113020410005 seconds  # slower verified complete profile pipeline
V = all new measured compatibility/CPU-check/bridge/pre-production preparation
projected_total = H + V + 1.25 * (2*P + 300)
require projected_total <= 7200
```

Thus V must be at most **1,385.1751731824625 seconds**. Account for each new
verification/stage-preparation cost once; an unrecorded repeated verification
is not free. The final 300-second allowance covers joint bootstrap/report and
its input/artifact verification. V9's single-seed report took 81.27 seconds;
the new common-draw two-seed report is plausible within this allowance. Do not
rerun v9 training. Record actual new elapsed time separately from this
conservative envelope, so inherited cost is not reported as newly consumed
compute or omitted from the planning gate.

Before production, and after M before F, require the gate to pass using observed
elapsed costs. After n completed arms use:

```text
H + V + measured_completed_pipeline_seconds
  + 1.25 * ((2-n)*max(P, slowest_completed_pipeline_seconds) + 300) <= 7200
```

Keep checking the actual charged elapsed ceiling at safe checkpoint boundaries;
stop with a resumable artifact if it is exhausted. A runtime breach is reported
as a breach even if artifacts have finished. No endpoint choice is permitted
to solve a cost failure. The production M/F pair, mandatory probes, validation
and report are required for completion. Re-profile is not an automatic fallback.

## Sol handoff and paper scope

Suggested new output root: `outputs/experiment016_encoder_motion_replication_v10`.
Use new reusable/CLI/test files for new logic, and staged `check`, `bridge`,
`train`, `report` execution. Freeze a new bridge-only executable manifest and
source map first; a separate production successor binds the passing bridge,
compatibility and cost receipts. Keep historical manifests unchanged. Store
both new completion receipts, all endpoint artifacts and hashes, per-update
trajectories, bootstrap metadata and the combined report. Resume only an exact
seed-47 production identity. Check live coordination/process identities and
the V100 before GPU work; preserve downloaders; run sequentially. Update both
queue documents at each state transition. This protocol authorizes no unrelated
experiment or automatically scheduled third seed.

A successful replication could support a cautious investigation claim: under
this fixed clean ECG adaptation recipe, two record-order seeds show a worse
joint readout when encoder updates are enabled, despite largely retained
linearly decodable discrimination. It would strengthen the controlled failure
analysis, not establish feature nonstationarity as the exclusive mediator,
general fine-tuning harm, a novel architecture, or clinical benefit. The
[v9 prior-art discussion](experiment-016-encoder-motion-v9.md#handoff-and-paper-interpretation)
still applies: selective ECG adaptation and probe-then-fine-tune recipes already
exist. Failure to replicate is useful evidence of seed sensitivity and must
remain in the paper record. Additional checkpoints/tasks and an honestly
independent evaluation remain necessary for broader claims; they are not opened
by this protocol. Retain the released probe as the practical reference.
