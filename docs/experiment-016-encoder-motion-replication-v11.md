# Experiment 016 v11: verified seed-47 encoder-update replication

**Astra design frozen 25 September 2026 before v11 implementation or seed-47
outcomes.** This is the xECG experiment following the completed CPC local-readout
study, under the user's sequential design/implementation instruction. It is a
new execution attempt at the [v10 scientific protocol](experiment-016-encoder-motion-replication-v10.md),
which stopped before GPU updates. This document freezes the recovery mechanism
and cost decision, not a new scientific comparison. Sol must establish the
measured gate before production. There is no passing v11 cost receipt yet.

Preserve all v9/v10 protocols, sources, manifests, logs, checkpoints and receipts.
Use new v11 files and output identities. Run one experiment at a time on the
shared V100; preserve downloaders and do not restart the legacy MIMIC scheduler.
Calibration/test remain closed. This local protocol is not an external registry
submission, and neither a bridge nor a CPU check is a replication result.

## Why version 11 is needed

[V10's stop report](experiment-016-encoder-motion-replication-v10-results.md)
records a nonexistent `torch.cuda.get_device_count()` call after expensive input
verification. Its CPU check took 452.593 s; the failed child's pre-GPU path took
411.145 s. Repeating that path could not pass v10's remaining cost allowance.
There was no seed-47 GPU update or development prediction, so seed 47 remains
uninspected for this comparison. The failure provides engineering/cost evidence,
not counterevidence to [v9's one-seed finding](experiment-016-encoder-motion-v9-results.md).

The frozen v9 profile/full source maps contain 124/146 entries and about
3.34/4.85 GB of mapped files, with considerable overlap. V10 separately walks
these maps, their completion artifacts, and nested v9/v8/v6/v5 input verifiers.
Those overlapping reads justify eliminating duplicate reads of the **same
verified bytes within a verification pass**. They do not justify trusting an old
JSON receipt instead of current source, checkpoint or waveform bytes.

## Scientific design remains identical to v10

Inherit v10 and v9's full scientific execution, diagnostics and interpretation.
The following requirements are explicit to prevent engineering changes from
changing the experiment:

| Item | Frozen requirement |
| --- | --- |
| Cohort | Same 15,359 clean full-label training ECGs; 1,306 development ECGs from 1,173 patients; excluded ECG 12722, row order, labels, patient partitions and input cache unchanged |
| Initialization | Released xECG weights plus original clean C=.01 affine probe head; identical M/F tensors and fresh optimizer moments; no profile, bridge, v9 or v10 checkpoint initialization |
| Input/model | Physical-mV twelve-lead 100 Hz waveform inputs, vanilla backend, dropout and stochastic depth Off, FP32 on the V100 |
| Randomness | Optimization seed 47 for both arms' global RNG and common full-record permutation; never a new pretrained encoder initialization or patient split |
| M | Encoder base LR 3e-5, layerwise decay .75; head LR .001 |
| F | Exactly zero encoder base and scheduled LRs, with encoder `requires_grad=True`, full backward, unchanged optimizer groups/state and encoder gradients included in global clipping |
| Shared optimizer | AdamW betas (.9,.999), epsilon 1e-8, decay .1 with historical grouping including head bias, BCE, global norm clip 3.0 |
| Exposure/schedule | Microbatch 16, effective batch 64, one epoch/15,359 exposures/240 updates, last batch 63, multiplier k/240 at k=0,...,239 |
| Endpoint | Update 240 only; no selected checkpoint, extra epoch, arm, seed or hyperparameter search |
| Final probes | Full train/development feature extraction for **both** final encoders; independent train-only StandardScaler + C=.01 lbfgs, max_iter=3000, random_state=42; require convergence |

F's whole encoder state, including buffers, stays bitwise unchanged after every
update; M moves at the first nonzero step and both heads then move. Retain native
versus affine-logit agreement and F's released-feature/refit positive control.
Do not replace F training or final extraction with cached-feature shortcuts.
Keep per-update batch digests, loss, LRs, global/encoder/head preclip norms, clip
factor and head-step magnitude. Preserve the fixed training-only 128-record
diagnostic at updates 0, 1, 60, 120, 240, with RNG/mode restoration. The policy
matches between arms; realized clipping may differ downstream of encoder motion.

## Cheap preflight must precede large input reads

Implement an independently callable v11 preflight. Before any multi-GB map walk,
historical checkpoint load, waveform hash or model construction, it must:

1. Parse the exact CLI/stage, resolve output paths and source imports, check
   new-source hashes and new manifest schema, and reject old output roots or
   wrong production seed. No historical executable file is patched in place.
2. Check that the actual installed torch exposes callable `cuda.device_count`,
   `cuda.is_available`, `cuda.get_device_name` and all other APIs used by the
   device receipt. A focused CPU test exercises the exact helper with a strict
   fake CUDA API that has `device_count` and deliberately lacks
   `get_device_count`; also test missing-device and malformed-query failures.
3. On a GPU launch, after obtaining the shared lock and checking actual host
   process identities, invoke the real helper with `torch.cuda.device_count()`
   and `nvidia-smi`. Confirm the single V100 16 GB, installed torch/CUDA and
   backend settings; fail before hashing if unavailable or incompatible. This
   query is not a training update. Record the current driver, while explicitly
   noting that v9's historical driver version was not recorded; do not claim
   exact historical driver equality. The subsequent bridge tests compatibility.

This ordering applies to the executable coordinator too: a child preflight
after a coordinator's expensive verification is too late. Use a new versioned
launcher/coordinator if needed, preserving the existing coordinator. Bind the
preflight receipt to the exact command, manifest/source hashes and environment;
test that a failing device query occurs before a hash-reader sentinel is called.
Every actual preflight/verification invocation is timed and charged.

## Provenance closure and executable compatibility

Retain v10's pinned v9 profile/production manifest and source-map hashes, report
hash, training-module/script hashes and M/F profile completion hashes. Before
reuse, verify their actual bytes and resolve every required source-map and
completion-artifact reference transitively. Validate the historical completion
state, manifest identity, profile correctness, measured H/P values and all
input fingerprints. V10's failed manifest/status/log/stop receipt remain failure
evidence; they are never a successful predecessor gate.

A new verifier may flatten the historical verification graph into a union of
required leaf paths and expected SHA-256 values. It must provide a machine-readable
coverage receipt mapping **each original verification obligation** to the
current verified leaf or explicit semantic assertion. Hash each distinct leaf
once per pass, reject conflicting expected hashes, missing leaves, path escape,
unresolved references and unexpected mutable/symlink substitutions. Include the
vendored source tree and the complete release/cache/input identities, not just
the top-level maps. Check file identity/size/timestamps before and after a read;
abort on concurrent modification. Record unique and logical-reference counts,
bytes read, measured time and resolved edges. A source-map hash alone does not
verify its leaves; a completion JSON alone does not verify its artifacts.

All semantic checks in the old input path still apply: cohort counts and exact
ordered ECG/patient/label identities, disjoint partitions, clean exclusion,
cache row mapping and preprocessing, release/feature/probe relationships, and
environment. A new input materializer may consume the verified leaf table and
reconstruct the **exact old v9 input fingerprint and data objects** without
recursively rehashing the same leaves. Demonstrate equality to the pinned v10
check's nested v9 fingerprint and the current verified underlying inputs. Keep
an explicit reviewable list of inherited assertions and their new locations.
Do not monkeypatch global hash functions or historical loaders to return cached
answers. Routine split-manifest integrity checks do not authorize calibration
or test prediction, feature extraction, fitting or endpoint inspection.

Deduplication is scoped to this validated pass and its consuming process, with
verified input identity kept under observation and no concurrent writer. A new
process/launch must rehash the required closure; a prior receipt, mtime, file
size or an advisory lock alone cannot replace fresh byte verification. Do not
claim cross-process reuse unless a separately reviewed immutable filesystem
snapshot/content guarantee actually exists. The default v11 path assumes none.
Any unavoidable coordinator and child scans must both be charged. Report-only
verification may use a smaller closure only if its exact computational inputs
and the authenticated transitive provenance links are enumerated; it must still
rehash every artifact the report actually reads. This cannot weaken production
input checks or silently omit historical obligations.

Create a compatibility receipt mapping every production callable and dependency
to v9's hashed implementation. Prefer unchanged training primitives; for copied
orchestration, compare ASTs and review the explicit diff. Only seed/output
identity, new preflight/verification and cost orchestration, and the unchanged
v10 two-seed analysis are permitted differences. Validate resolved runtime
bindings as well as AST text. No changed precision, data transformation,
optimizer/gradient path, extraction or checkpoint behavior is compatible.

Run focused CPU checks of common initialization/logits and zero optimizer
moments, complete seed-47 permutation equality across M/F and difference from
46, group LRs and all 240 warmup multipliers. Test rejection of wrong seed,
bridge/old-run checkpoint, altered/missing leaf, conflicting/transitively missing
hash, incomplete pair, and cost-ledger omissions. Preserve v10's passed CPU
receipt as historical evidence, not as a substitute for checking new logic.
If compatibility cannot be established, stop; full re-profiling or a changed
scientific path needs a separate design and is not an automatic fallback.

## Bridge, production and replay

After the CPU/identity checks, run a new seed-47 bridge, M then F, with one active
GPU model at a time and no development metrics. Bound its entire GPU stage,
including initialization, checkpoint operations and replay, to **300 seconds**;
charge preceding source verification/preparation separately. Each arm uses the
first three full real effective batches from the unchanged seed-47 permutation
and 240-update scheduler. Compare common starts and the first two updates'
batch identities, loss, gradient norms, clip factors and head-step magnitudes
under v9's established tolerances. Check M movement/F identity and both heads.

Exercise exact CPU checkpoint serialization and two sequential replays of the
next update from the bridge checkpoint, with exact order, counters, RNG and
scheduler, and `atol=1e-8, rtol=1e-6` for post-update floating tensors. Record
maximum discrepancies and allocated/reserved GPU memory. No simultaneous full
model/optimizer copies. Timeout, OOM or failed correctness stops the attempt;
never lower microbatch, precision, gradient scope or diagnostics to pass.

Production begins fresh from release/probe, M then F, only after a passing
bridge and measured cost gate. Never resume bridge weights. Checkpoint every
40 updates and on interruption, retaining full model/optimizer/scheduler,
permutation/position and global RNG. Only exact v11 production identity may
resume, and all prior attempt time remains charged. Retain final sequential
same-device replay, complete extraction/refits and artifact hashes for both
arms. A partial pair has no completed replication result.

## Complete-path measured cost decision

The **7,200-second ceiling is unchanged**. V11 is a separately authorized
attempt, not a retroactive revision of v10's failed gate. Keep a cumulative
investigation ledger displaying v10's approximately 1,086.936 s observed new
work and uncertainty separately, plus all v11 actual work; do not report that
the combined historical investigation fits two hours. V11 retains v10's
conservative charge for the already-completed full v9 profiles:

```text
H = 2693.7965717150364 s  # v9 preparation + both complete profile pipelines
P = 1098.4113020410005 s  # slower complete v9 pipeline, never a minibatch estimate
E = all measured v11 executed stage wall time to date
n = number of complete v11 production pipelines (0, 1 or 2)
Pstar = max(P, each measured complete v11 production pipeline)
Qremaining = conservative measured forecast of all remaining stage preparation,
             source/input/artifact verification and coordinator overhead not in Pstar
R = 300 s for remaining two-seed bootstrap, reporting and completion work
projected = H + E + 1.25 * ((2-n)*Pstar + Qremaining + R)
require projected <= 7200 s
```

E includes preflight, CPU checks, real verification measurements, bridge,
manifest checks/launch preparation, imports/loading, stage/coordinator overhead,
finished production pipelines, failed invocations/retries and artifact audits.
Use a persistent ledger of nonoverlapping wall intervals: nested child timing
is not added twice to its enclosing coordinator interval, but no outer startup,
repeated scan or failed attempt disappears. Separate engineering/design time
from executable study stages; starting a new command never resets E. Include
any intentional job wait inside an active launch. Charge source/data rechecks
performed while preparing executable manifests as study work too.

Measure a complete unique-leaf verification pass and actual stage preparation
before GPU bridge admission; use the slower measured applicable scan/preparation
for each future occurrence, including both coordinator and child if duplicated.
Do not estimate hashing from a subset of a large file or assume warm-cache
speed. Qremaining must enumerate future invocations and additional generated
artifact reads; use measured throughput/size and the slower observed overhead
for those not yet present. R retains at least 300 s until final reporting; if a
timing check or observed report preparation requires more, increase the reserve
and reapply the gate. No step may be omitted to fit the ceiling.

Before the bridge, add its full 300 s cap to Qremaining until its actual time
enters E. Before production, all remaining verification, including a separate
production manifest/launch, must already be represented. Recheck after that
actual launch preparation, before M, after M/before F, and before reporting.
Initially H + 1.25*(2*P + 300) = **5814.824827 s**, leaving at most
**1385.175173 s** for observed new work and the inflated remaining preparation.
That is an upper allowance, **not evidence the redesigned verifier fits**.
The old repeated path is not a passing v11 forecast. A complete measured scan,
CPU check and enumerated future preparation must demonstrate the allowance.

At production checkpoint boundaries, record H + actual E and conservative
remaining completion cost; stop with a resumable artifact if the remaining
budget is insufficient. Timeouts include checkpoint/save overhead; a breach
must be reported even if artifacts eventually finish. The final receipt must
show H + actual total v11 stage time <= 7200, plus the separate actual/cumulative
ledger. Keep R charged conservatively until reporting has completed, then
replace it with actual report time already in E. If the honest gate fails,
record a cost-stop and **do not launch/retry production, drop controls, change
scientific thresholds or quietly replace this cost rule**.

## Frozen analysis and interpretation

Keep v10's sole primary contrast and all numerical decisions:

```text
D47 = AUROC(F_joint47) - AUROC(M_joint47)
R47 = AUROC(M_refit47) - AUROC(F_refit47)
G47 = [AUROC(M_refit47)-AUROC(M_joint47)]
      - [AUROC(F_refit47)-AUROC(F_joint47)]
```

Joint-head harm replicates at D47 >= .005, with a separate flag for a strictly
positive interval. The narrower readout-gap pattern additionally requires
R47 >= -.002 and G47 >= .005. Intervals including zero remain uncertain;
0 < D47 < .005 is directional agreement below the practical threshold and
D47 <= 0 fails direction replication. Failed R47/G47 conditions must be stated
separately. No combined estimate overrides a failed seed-47 screen.

Report all four final readouts and released probe with AUROC, AP, BCE and the
unchanged five patient-group threshold-fold sensitivity/specificity screen.
Use exactly 2,000 paired whole-patient draws with seed **16020**, preserving all
records and multiplicities of each sampled patient, with common draws across
readouts and seeds. Verify exact ECG/patient/label alignment before pairing.
Report percentile 95% intervals, invalid single-class draws and valid count;
fewer than 1,900 valid draws leaves interval inference unresolved. Do not replace
the patient bootstrap with row resampling or change draws after inspection.

Show seed 46 and 47 contrasts and their range, plus the arithmetic mean of each
contrast with common-draw intervals. Average contrasts, not logits. Preserve
v9's original seed-16019 report/intervals; label seed-46 intervals recomputed
under seed 16020 as new combined-analysis intervals. These intervals condition
on two fixed optimization seeds and repeatedly inspected development patients.
They are neither seed-population uncertainty nor independent patient validation.

The interpretation remains a fixed-budget total encoder-update effect that may
involve realized clipping and optimizer dynamics. Success does not isolate
feature motion as the exclusive mediator, establish general fine-tuning harm,
introduce a new LP-FT method, or demonstrate clinical utility. Failure remains
evidence about seed sensitivity. Retain the released probe as the practical
reference. No third seed, other experiment, calibration or test follows
automatically.

## Implementation handoff and completion

Use `outputs/experiment016_encoder_motion_replication_v11` and new reusable,
CLI and test files. Expected receipts are `preflight.json`, `verification.json`,
`check.json`, `compatibility.json`, `bridge.json`, `cost_gate.json`, persistent
`cost_ledger.json`, per-arm completion/replay artifacts, `report.json` and final
completion verification. Freeze a bridge-only manifest/source map first; a
separate production successor binds the passed bridge/check/compatibility/gate.
The new launcher must preserve all existing queue safety checks, exact manifest
identity and the shared GPU lock. Freeze exact commands, source revision and
hashes before each launch; retain failed attempts unchanged.

Both queue documents track design, implementation, checks, bridge/gate and
completion truthfully. **Design complete does not mean runnable or cost-passed.**
Sol's first deliverable is a tested preflight and measured verification/cost
receipt. Stop if a compatible, honest complete-path projection cannot fit.
