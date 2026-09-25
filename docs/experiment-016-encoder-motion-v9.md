# Experiment 016 v9: matched encoder-update intervention

**Astra design frozen 25 September 2026 before implementation or v9 outcomes.**
This is one development-only experiment on one V100. Sol implements and runs
only after this design handoff. Preserve all historical sources and receipts;
calibration and test remain closed. No follow-up is scheduled automatically.

## Why one further experiment is justified

[V8](experiment-016-head-mechanism-v8-results.md) rejected its standardized-
coordinate discrimination hypothesis. Raw AdamW head-only fitting recovered
AUROC to 0.96249/0.96185 after 240 updates on fixed Off features, without the
explicit probe penalty; changing that regularization package added little.
This narrows the question but does not isolate moving features: v8 also used
CPU float64, fresh head moments, additional exposures, a different starting
head, and a head-only rather than full-model clipping norm.

V9 changes **whether encoder parameters update**, while holding the head's
initialization, optimizer, exposure count, precision and clipping policy fixed.
Its prediction is that keeping the representation stationary prevents much of
the joint-head discrimination loss under the original one-epoch budget. It is
not another learning-rate search or a test of converged fine-tuning.

## Two matched arms and data

Use the same 15,359 clean full-label PTB training ECGs, 1,306 development ECGs
from 1,173 patients, frozen patient partitions, released xECG weights, vanilla
backend, physical-mV twelve-lead 100 Hz inputs and exact clean C=.01 initial
probe used in v6. Both arms start from identical backbone and native affine
probe-head tensors. Do not start from an adapted v6 checkpoint or v8 head.
All dropout and stochastic depth are **Off** in both arms.

Use the new optimization seed **46**, not inspected encoder seeds 42/43 or
v8 head seeds 44/45. It seeds global training RNG and the common record
permutation; it never changes patient or label selection. Use profile-only
seed **16090** for all timing/checkpoint checks. Production always starts fresh.

| Arm | Encoder updates | Head updates | Forward/backward path |
| --- | --- | --- | --- |
| M: Moving | Original layerwise AdamW encoder rates | Original head AdamW | Full model |
| F: Frozen | Exactly zero encoder learning rates | Identical head AdamW | Full model, including encoder gradients |

The crucial control is that **F still computes every encoder gradient and
includes it in global gradient clipping**. Keep encoder `requires_grad=True`;
do not detach features, use cached features during training, omit the encoder
from clipping, or set its gradients to zero. Keep identical parameter groups,
weight-decay settings and Adam state allocation, but set the encoder groups'
base learning rates to zero before constructing the scheduler. This disables
both gradient and AdamW decay changes to F's encoder tensors while retaining
the same backward graph and clipping rule. Verify F's entire encoder state,
including buffers, is bitwise unchanged throughout the epoch.

In M retain encoder base LR 3e-5 and the existing 0.75 layerwise decay. Both
heads have LR .001, AdamW betas (.9,.999), epsilon 1e-8, weight decay .1 on the
same groups (including head bias), fresh zero optimizer moments, BCE and
global gradient clipping at 3.0. Preserve the exact existing grouping for
patch embedding, nine blocks, other backbone parameters and head.

Both arms run FP32 on the same V100 with microbatch 16, effective batch 64,
the same ordered records and accumulation weighting. One complete epoch is
**15,359 exposures and 240 updates**, with a 63-record last effective batch.
Apply the same first-epoch warmup multiplier `k/240` for update indices
`k=0,...,239` and stop. Do not compress the old two-epoch schedule, add a
second epoch, or select a better intermediate checkpoint. The first zero-LR
update leaves parameters unchanged but can change optimizer moments.

The clipping **policy and included parameters** match. Actual gradient norms
and clip factors can diverge after encoder updates cause model states to
diverge; those differences are downstream consequences of the intervention.
Do not describe this as holding realized clip factors numerically equal or
isolating feature drift from every optimizer-mediated consequence. The primary
estimand is the total effect of permitting encoder updates under this fixed
training policy. F is a compute-matched diagnostic control, not an efficient
implementation recommendation for ordinary linear probing.

## Required readout diagnostic

After both fresh arms finish, freeze their final backbones and obtain pooled
training/development features in eval mode with no gradients. Fit the exact
v7 train-only StandardScaler plus C=.01, `lbfgs`, max_iter=3000,
random_state=42 probe independently on each final backbone. Require convergence;
no C grid, coefficient selection or development stopping. This mandatory
diagnostic measures linear usefulness under one common fitting recipe.

Because F's backbone is unchanged, its refit is a released-feature positive
control and should reproduce the clean reference up to recorded inference/
solver tolerance. Reusing released features is allowed only after exact encoder
identity and cached-feature/input verification; profile conservatively as if
both final feature sets were extracted. Verify final native joint-head logits
against affine logits from each feature cache. Preserve the original clean
probe's coefficients and predictions; do not rewrite them with new fits.

## Measurements and prespecified decisions

The sole primary contrast is **F joint-head minus M joint-head development
AUROC at update 240**. Report the released initial probe and all four final
readouts: M/F joint heads and M/F refitted probes. Secondary quantities are
the M-versus-F refit difference and the difference in readout gaps:

```text
gap_M = AUROC(M_refit) - AUROC(M_joint)
gap_F = AUROC(F_refit) - AUROC(F_joint)
gap_difference = gap_M - gap_F
```

Report AUROC, AP, BCE, and the unchanged five patient-group threshold-fold
sensitivity/specificity screen targeting 95% sensitivity on the other folds.
Use 2,000 paired whole-patient bootstrap draws, seed 16019, shared across all
contrasts, reporting invalid single-class draws and 95% intervals. These
condition on a single new optimization seed; patient resampling is not seed
replication, and the development patients remain repeatedly inspected.

- **Encoder-update harm screen:** F minus M is at least +.005 AUROC. If its
  paired interval includes zero, call it a positive point-estimate screen with
  unresolved sampling uncertainty. A strictly positive interval strengthens
  evidence for this fixed-budget intervention, not for all ECG fine-tuning.
- **Readout-gap interpretation supported:** additionally M refit is no more
  than .002 AUROC below F refit, and `gap_difference >= .005`. This supports
  encoder updates creating a larger gap between the jointly trained head and
  a common linear readout, rather than a comparable loss in measured linear
  usefulness. Report uncertainty for both secondary quantities; these practical
  thresholds do not establish statistical equivalence of representations.
- If F beats M but M refit also falls more than .002 below F refit, report
  evidence compatible with both representation and readout contributions.
  Do not claim all harm is head lag or irreversible forgetting.
- If F and M both degrade similarly, the moving-encoder explanation fails
  this matched test; shared head optimization/clipping/warmup remains a possible
  cause, not an established finding. If M is as good or better, retain that
  counterevidence and do not change the primary hypothesis after the result.
- The released probe remains the practical reference unless a separately
  evaluated recipe proves useful. No pattern launches more training or opens
  calibration/test. A one-epoch warmup result cannot establish longer-budget
  generalization or a clinical benefit.

Record the global, encoder-only and head-only preclip gradient norms, scalar
clip factor, head update norm and applied learning rates **at every update**.
This fixes the missing epoch-wide clipping history in earlier studies. Record
training loss and batch-ID digest at the same granularity. On the fixed
training-only 128-record diagnostic set, evaluate at initialization and after
updates 1, 60, 120 and 240: current-head BCE/logits, old-probe logits, feature
cosine to release, and head/encoder parameter changes. Restore model mode and
RNG around diagnostics; no extra development checkpoint selection is allowed.

## Correctness and feasible full-path gate

Sol must implement the zero-LR encoder intervention in new files. Before any
production run, verify common initial tensors/logits/optimizer state and batch
order. The first backward gradient norms, clipping and head optimizer updates
must agree across M/F within the established FP32 tolerance. After the first
nonzero step M's encoder must change, F's must not, and both heads must update.
Compare against an explicit full-gradient, zero-encoder-step reference on a
small deterministic example. Guard against accidental backbone AdamW decay,
scheduler restoration of nonzero F learning rates, mode differences, hidden
feature caching, and using CPU float64 training for F.

Checkpoint every 40 updates and on interruption, including model, full optimizer,
scheduler, exact batch position, permutation and global RNG. CPU serialization
is exact. Use the established v5 same-device resumed-update tolerances
`atol=1e-8, rtol=1e-6` for floating tensors, with exact batches/counters/RNG/
scheduler. Replay sequentially so multiple full models/optimizer copies do not
exceed V100 memory. Keep at most one active training model on the GPU.

Require a **fresh full profile of both M and F** with seed 16090. Each includes
the entire real-data epoch, the prescribed per-update logging/diagnostics,
development pass, checkpoint write/load/replay, complete final train/development
feature extraction and the fixed probe fit. Profile models, features and fitted
heads are disposable cost/correctness artifacts, never production results.
Historical timings are feasibility evidence only: v6 complete passes were about
530–542 seconds, and v7 extraction about 184 seconds. F's deliberate full
backward means it is not forecast as a cheap cached-head run.

Freeze the 7,200-second gate:

```text
P = slower measured complete M/F profile pipeline seconds
projected_total = measured preparation plus both profile pipelines
                  + 1.25 * (2*P + 300 seconds)
require projected_total <= 7200 seconds
```

The two projected pipelines cover the two fresh one-epoch production arms and
their mandatory refits. The allowance covers final bootstrap/report/receipts.
On historical rates this appears feasible, but new logging, I/O, checkpointing,
and F's full gradients must be included in measured P. If the gate or memory
check fails, stop; do not silently remove full-model clipping gradients, shorten
the cohort/budget or discard the refit control. After each arm update remaining
cost with measured time. An incomplete pair produces no completed primary result.

## Handoff and paper interpretation

Suggested new output: `outputs/experiment016_encoder_motion_v9`; staged
`check`, `profile`, `train`, `report` runner with reusable logic and focused
tests in new files. Preserve frozen v5–v8 modules and immutable manifests.
Bind the protocol, source/input/environment/checkpoint hashes and exact commands
in a verified successor manifest. Use the shared GPU lock, check live process
identities, preserve downloaders, run sequentially, and update both queue
documents through completion. No other experiment belongs in this manifest.

ECG adaptation is established prior art. The September 2026 Scientific Reports
study by [Monachino et al.](https://www.nature.com/articles/s41598-026-71351-2)
compares partial fine-tuning, BitFit, LayerNorm tuning and linear probing with
full fine-tuning on Self-DANA, including in-domain and out-of-domain scenarios.
It already supports the broader point that selective adaptation can improve
ECG generalization. Freezing an ECG encoder is therefore not our novelty claim.

[Zhou et al.'s ECGFounder post-training study](https://arxiv.org/abs/2509.12991)
already combines preview linear probing with full fine-tuning and regularization,
including stochastic depth, on PTB-XL tasks. Its residual formulation preserves
the identity outside the dropped branch. Our xECG observations must not be
presented as a new LP-FT strategy or as a refutation of their different model,
tasks and training protocol.

A possible paper contribution is narrower: **a controlled account of how an
ECG adaptation run can acquire a poor joint readout while retaining linearly
useful features, with the encoder-update intervention separated from head
optimizer, clipping and precision changes.** V9 can strengthen or falsify that
account under one fixed recipe. It cannot alone prove that nonstationarity is
the exclusive causal mediator, establish a general new method, or establish
clinical usefulness. A credible paper still needs encoder-seed replication,
additional ECG checkpoints/tasks and genuinely independent evaluation; these
remain roadmap items rather than automatically scheduled work.
