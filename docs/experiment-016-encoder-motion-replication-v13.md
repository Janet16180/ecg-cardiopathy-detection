# Experiment 016 v13: prospective seed-47 replication with remaining-work accounting

**Design frozen 25 September 2026, before any seed-47 outcome. Implementation,
complete CPU checks and measured admission remain pending.** The user's active
request is CPC followed by xECG, using all 15,359 clean training labels. CPC is
complete. This separately identified prospective design continues that request;
it does not reopen or rewrite the failed v10/v11 attempts or the
[v12 design stop](experiment-016-encoder-motion-replication-v12-results.md).
V12's self-imposed no-successor rule does not cancel the user's request.
No executable manifest or GPU launch is authorized by design completion alone.

The decision is **conditionally feasible for implementation, not yet runnable**.
The historical initial envelope is 7,096.074827 s against the unchanged 7,200 s
ceiling. V13 resolves the double-counted incomplete arm prospectively: elapsed
work enters the actual ledger once, while only unfinished work is forecast.
Its conservative progress and pace rules below are frozen before execution.
They may still stop a slow run. There is one real-data admission/bridge process
and one freshly verified production process, with no automatic retry or resume.

## Scientific comparison remains fixed

Inherit the entire [v10 scientific protocol](experiment-016-encoder-motion-replication-v10.md),
[v9 controls](experiment-016-encoder-motion-v9.md), and
[v11 provenance/compatibility obligations](experiment-016-encoder-motion-replication-v11.md).
Only version/output identity, verified materialization, launch organization and
cost orchestration change. Preserve the pinned v10 table of historical hashes.

- Same 15,359 clean full-label training ECGs, excluded ECG 12722, exact ordered
  labels/rows, fixed patient partitions and 1,306 development ECGs/1,173 patients.
  Same released xECG, physical-mV twelve-lead 100 Hz cache and train-only fitting.
  Calibration/test prediction, fitting, feature extraction and endpoint access
  remain closed; split-integrity checks do not open those endpoints.
- Both arms start afresh from identical released tensors and original clean
  C=.01 affine probe head, with zero optimizer moments. Seed **47** controls
  global training RNG and common complete permutation, different from seed 46.
  Never initialize from an old, profile or bridge checkpoint.
- Vanilla backend, dropout/stochastic depth Off, FP32 V100, microbatch 16,
  effective batch 64, 15,359 exposures, 240 updates, last effective batch 63.
  Keep the original `k/240` warmup for k=0,...,239 and update-240 endpoint.
- M uses encoder base LR 3e-5, layerwise decay .75 and head LR .001. F has
  exactly zero encoder rates, retaining encoder gradients, optimizer groups
  and full Adam state, full backward and global norm clipping at 3.0. Preserve
  BCE, AdamW betas (.9,.999), epsilon 1e-8, decay .1 and historical grouping,
  including head bias. Different realized clipping after motion is permitted.
- Require every-update bitwise F encoder/buffer identity, M movement at the
  first nonzero step, common initial gradients, native/affine logits and the
  F released-feature/refit control. Retain per-update diagnostics and the fixed
  128 training ECG diagnostic at 0/1/60/120/240 with RNG/mode restoration.
  Checkpoint every 40 updates and on interruption; preserve exact CPU
  serialization and sequential same-device replay with `atol=1e-8, rtol=1e-6`
  and exact counters, order, RNG and scheduler. Keep one active GPU model.
- Fully extract final train/development features for **both** encoders and fit
  independent train-only StandardScaler + C=.01 lbfgs probes, max_iter=3000,
  random_state=42, requiring convergence. No cached-F shortcut, shorter epoch,
  reduced cohort, precision change or omitted control may rescue a failed gate.

The sole primary contrast is `D47 = AUROC(F_joint47)-AUROC(M_joint47)`.
The harm screen is **D47 >= .005**, with a separate strictly-positive-interval
flag. The narrower readout pattern additionally requires
`R47 = AUROC(M_refit47)-AUROC(F_refit47) >= -.002` and
`G47 = (M_refit47-M_joint47)-(F_refit47-F_joint47) >= .005`, using AUROC
throughout. Preserve v10's directional agreement below .005, nonreplication
at D47 <= 0, failed-secondary and uncertain-interval labels. A partial/invalid
pair has no completed replication result; no combined mean overrides failure.

Report all four readouts and released probe with AUROC/AP/BCE and unchanged
five patient-group threshold folds. Use exactly 2,000 common paired whole-patient
bootstrap draws, seed **16020**, with exact ECG/patient/label alignment,
patient multiplicity, percentile 95% intervals and invalid-draw counts. Fewer
than 1,900 valid draws leaves interval inference unresolved. Display both seed
contrasts, range and arithmetic mean with common-draw intervals; never average
logits. Preserve v9's original seed-16019 report and label recomputed seed-46
intervals as new combined-analysis intervals. These are two fixed optimization
seeds on repeatedly inspected development patients, not seed-population
uncertainty or independent validation. The endpoint is an ECG annotation proxy.
The intervention estimates total encoder-update effects including downstream
clipping/optimizer dynamics, not an exclusive mediator or clinical benefit.
Retain the released probe as the practical reference.

## Verified execution without redundant preflight launches

Use new reusable/CLI/test files and
`outputs/experiment016_encoder_motion_replication_v13`. Do not edit frozen
v9-v12 sources, protocols, manifests, stop receipts or hash records. The current
v11 CLI is preflight-only; its verifier returns a receipt and its semantic
function returns no production objects. A complete versioned materializer,
launcher, CPU check, compatibility map, ledger and gate hooks must be implemented.

Freeze a bridge-only manifest/source map before the first real-data invocation.
It binds the protocol, new sources, full imported executable closure, exact
command, revision, dependencies, output identity and cost inventory. In its
single consuming process: cheap CLI/source/API checks; nonblocking GPU lock;
actual host-process/device check; fresh complete historical byte verification;
materialization and complete CPU correctness; measured admission gate; bridge.
Do not run a separately hashing coordinator or a separate real-data preflight
command. A unavoidable extra scan/process must be charged and still fit; it
does not create another allowed admission invocation.

Before a large read, exercise the actual installed helper using callable
`torch.cuda.device_count()` and validate all other used APIs. Confirm one
V100 16 GB, Torch 2.6.0+cu124/CUDA 12.4, historical dependencies/backend and
current driver, without claiming unrecorded historical driver equality.
Preserve downloader source/data paths and processes and do not restart the
legacy runner. Hold the shared lock for each whole consuming process. No
intentional lock wait, unlocked preflight/use gap or simultaneous full models.

In **each consuming process**, freshly hash the full transitive historical
source/input/completion closure and that process's new executable artifacts.
Version v11's verified graph with every original hash obligation and semantic
assertion mapped to current bytes; reject omissions, conflicts, path escape,
symlinks, replacement or concurrent changes. Include vendored source members,
source-map leaves and completion-artifact leaves, not only top-level JSON.
Record logical obligations, unique leaves, bytes and nonoverlapping timings.
Keep verified input identity under observation through use; mtime, size, a
prior receipt or the GPU lock never replaces a new process's fresh byte pass.

Materialize the actual clean train/development rows, waveform mapping/cache,
released features, probe and fixed diagnostic inputs from that same validated
graph. Reproduce exactly v10's pinned nested v9 fingerprint and retain all
historical cohort, label, split, cache, release and train-only assertions.
Do not call recursively hashing historical loaders or monkeypatch their hash
functions. A compatibility receipt maps **all** production callables and
dependencies to authenticated v9/v10 implementations, including runtime bindings
and AST comparison plus explicit orchestration-only diffs. Failure is a no-go;
full reprofiling or a changed scientific path is not an automatic fallback.

In the bridge process, the complete CPU check verifies identical initialized
M/F tensors/four real training ECG logits, zero optimizer moments, full seed-47
permutation equality/difference from 46, all parameter-group rates and all 240
warmup multipliers. Free CPU models before the bridge. Recheck the cost gate
after actual preparation and **before any GPU update**.

The bridge keeps exactly v10's M-then-F first three real effective batches,
240-update schedule, common-start/first-two-update comparisons, M movement,
F identity, CPU checkpoint serialization and two sequential next-update replays.
No development metrics. Its **entire** GPU stage, including initialization,
checkpoint I/O, replay, cleanup and completion work, has a 300 s cap.
Timeout/OOM/identity/correctness failure closes v13.

After a passed bridge, freeze a separate production successor binding the
passed CPU, compatibility, bridge and gate receipts and all final source hashes.
It freshly rehashes the complete closure, including both new bridge checkpoints,
and starts M then F afresh from release/probe. Its single process also produces
the frozen two-seed report and completion audit. Verify all newly generated
artifacts actually read, and final completion-artifact hashes. A same-process
historical pass does not waive new-artifact verification. No production resume.

Before real-data admission, focused synthetic CPU tests must cover strict fake
CUDA API compatibility, failure before any hash-reader sentinel, full scheduler
and permutation logic, altered/missing/conflicting/transitively omitted leaves,
wrong seed/checkpoint/source binding, incomplete pairs, ledger omissions and
every cost boundary below. No real-data dry-run invocation precedes admission.

## Frozen remaining-work cost rule

The **7,200 s total planning ceiling**, 25% forecast margin and historical full
pipeline evidence remain unchanged. V13 replaces v12's incomplete-arm equation
prospectively; it does not reinterpret v12's failed frozen rule.

```text
H = 2693.7965717150364 s    # historical v9 preparation + both full profiles
P = 1098.4113020410005 s    # slower historical full pipeline
E = all actual v13 executable study-stage wall time so far, once
Pstar = max(P, every completed measured v13 production pipeline)
U = number of unstarted production arms
Aremaining = forecast unfinished fraction of the one active pipeline, or 0
Qremaining = enumerated remaining nonpipeline work, including stop reserve
Rremaining >= 300 s until reporting/completion finishes
projected = H + E + 1.25*(U*Pstar + Aremaining + Qremaining + Rremaining)
GO requires projected <= 7200 s
```

For an active arm, `a` is its actual pipeline elapsed time and is already in E.
Progress `f` is credited only for completed scientific work under this frozen
map. It is never wall time divided by P and never simply updates/240:

| Completed work | Cumulative f |
| --- | ---: |
| Initialization but no complete update | 0 |
| k updates, including due diagnostics, integrity checks and checkpoint work | k/480, at most .50 |
| All 240 updates plus complete native development inference, train/development extraction, affine/control checks and converged final refit | .75 |
| Required sequential replay and pipeline cleanup complete | 1.00 |

During training record update durations through all due ancillary work. Use
40-update blocks for pace comparison; the first block includes initialization
and update-0 diagnostics. At safe checkpoints 40/80/.../240, and the .75/1.00
milestones, each completed block contributes its actual duration divided by
its credited delta-f. Keep the maximum of all such ratios within the arm.
For intermediate per-update guards only complete blocks receive the block-pace
statistic; `a/f` still includes all observed active time. At f=0 use Pstar and
any previously measured applicable full-pipeline pace floor, without division.

```text
Tpace = max(Pstar, a/f when f>0, all completed block duration/delta-f)
Aremaining = (1-f)*Tpace                       # f in [0,1]
```

This intentionally allocates half the initial pipeline forecast to its entire
post-training tail; it is a conservative frozen planning allocation, **not** a
claim that historical phase times were measured. Historical logs lack those
phase timestamps. At update 240 half the arm forecast remains. The first tail
block retains its full .25 credit until all extraction/refit checks pass; replay
retains the final .25 until complete. Existing historical/full-path timings and
actual bridge observations must support the inventory; if any required tail
component needs a larger defensible forecast, add its uncovered excess to Q
and reapply the gate. No optimistic credit for iterations, bytes or time alone.

The full-arm pace floor can increase with slower observed work; never lower P,
Pstar or recorded block maxima on faster observations. `a/f` may fluctuate but
cannot remove those floors. Unstarted F still costs a **full Pstar**; completed
M enters E and updates Pstar, with no second full M forecast. At f=1 transition
atomically from active to completed, update Pstar, and evaluate the next gate.
Each checkpoint receipt records E/U/a/f/Pstar/block paces, A/Q/R, projection,
source identity and decision. Cost guards never inspect development metrics.

Use an external monotonic start before Python imports for each executable
study command, including synthetic study checks and manifest preparation.
Persist disjoint intervals; nested timings partition an interval rather than
add to it. Count failed invocations, imports, scans, binding/checks, I/O,
initialization, checkpoints, cleanup, reporting and stop work exactly once.
Engineering reading/writing is separate. New commands cannot reset E. Pipeline
phase timers must nest in this ledger and not omit startup or work between arms.

**Historical timer caveat:** v9/v10 construct `complete_pipeline_seconds`
before hashing final `resume.pt`, feature and other artifacts. Those hashes,
completion JSON writing/verification and later report-input reads therefore
belong explicitly in Q/R, even when the retained scientific runner does them.
Do not assume the historical P covers them, and do not charge nested measured
hash time twice. A broader new pipeline timer must document its boundary and
remove only the corresponding duplicate forecast from Q, without lowering
the historical floor or losing any actual E interval.

Recheck before bridge, after bridge, after production verification, before M,
each update/at mandatory checkpoints, after training and every tail milestone,
after M/before F, and before reporting. Atomic long operations need a watchdog
and conservative finish/stop allowance; no gate waits for exhaustion. Retain
at least 30 s for safe checkpoint/stop until GPU work and cleanup finish, using
the larger of 30 s and the slowest measured applicable stop/checkpoint bound.
This reserve is part of Q, not free. Report and close any actual breach even
if artifacts subsequently finish. Completion requires a valid pair/report and
`H + actual total v13 executable stage time <= 7200`.

## Inventory and conditional feasibility

The initial allowances below are frozen planning floors, not measured v13
stage durations. Do not lower them after a faster warm-cache measurement.
Completed components transfer to actual E; uncompleted components remain
reserved. For each future occurrence use the larger of its floor and slower
applicable measurements/whole-pass byte-throughput forecast. Partial scans
cannot forecast a whole pass. Inflate uncovered additional work rather than
borrowing silently from another category.

| Q component | Seconds |
| --- | ---: |
| Fresh bridge-process historical closure and semantics | 200 |
| Fresh production closure including both bridge checkpoints | 225 |
| Full CPU materialization/correctness beyond the closure | 100 |
| Entire bounded GPU bridge | 300 |
| Synthetic checks, source/manifest binding, startup/host checks, additional artifact audits/completion and safe stop reserve | 200 |
| Total Q | 1,025 |

The last 200 s must have a concrete prelaunch subinventory: study tests,
source-only freeze/read checks, imports and host checks for both processes,
the two production checkpoint/feature hash sets and their rereads, completion
writes/audits, and the >=30 s safe-stop reserve. Report bootstrap/input reads
and final audit are assigned explicitly either there or to R, never omitted
or forecast twice. Planned artifacts must be enumerated with bytes and read
counts. Initial-source hashing and executable study tests count in E even
before a bridge manifest exists.

V11's closure measured 288 leaves/1,051 obligations/6,493,086,486 bytes in
172.721 s; use the slower 176.726584 s as the historical whole-pass floor.
Two approximately 684,552,140-byte bridge checkpoints add about 37.3 s to a
fresh production pass at that rate. Each v9 final arm artifact set is about
753 MB, making two initial final-artifact hash sets approximately 41 s at
the same throughput, before any repeat audit. CPU probes previously totaled
45.441 s but were incomplete; 100 s is a planning allowance whose sufficiency
the complete CPU check must establish. R retains at least 300 s for two-seed
bootstrap/report/completion (historical single-seed reporting was 81.27 s).
Increase any insufficient allowance; the ceiling stays fixed.

Initially, with no active arm, `U=2, E=0`:

```text
H + 1.25*(2*P + 1025 + 300) = 7096.0748268175375 s
initial inflated headroom = 103.9251731824625 s
```

This arithmetic is a design envelope, **not a passing measured gate** or a
promise that the conservative milestone forecast admits execution. Actual
preparation replaces its completed reserves; repeated scans/checks can consume
the headroom. If the 1,025 s planned preparation were completed, a hypothetical
active arm with f=.25 and a=.25P at the historical pace would project
6,771.174120 s, versus 6,839.824827 s immediately before M. The incomplete
arm's elapsed .25P appears only in E; .75P remains forecast, plus full F.
This is an arithmetic example, not a measured runtime. A slower active pace
raises the unfinished-work forecast and can cause a legitimate cost stop.

Before real-data invocation require a full executable inventory and passing
synthetic tests, with no unidentified mandatory cost assigned zero. Perform
a pre-GPU feasibility audit by replaying the frozen weighted gate at every
mandatory milestone using authenticated v9 phase timings wherever available
and explicitly conservative phase bounds elsewhere. Do not infer phase times
from current file mtimes. The inspected v9 profile receipts give full pipeline
times but no phase breakdown; the checkpoint console log has no per-checkpoint
timestamps. Treat the 50/25/25 allocation as a forecast requiring this audit,
not as missing measurements manufactured from total P. If the rule's own
weighted pace floors make the complete path impossible under supported bounds,
or those bounds cannot be supported, close before GPU rather than waiting for
an inevitable production checkpoint stop. Preserve the phase-bound audit in
the executable inventory and repeat it with actual bridge/preparation evidence.
Complete the real CPU check and fresh closure within the single bridge process and
require an actual measured admission projection before any bridge update.
Production requires that same evidence plus a passing bounded bridge and
freshly measured production-admission gate. If compatibility, materialization,
tail bounds, audit costs or stop timing cannot be supported within the ceiling,
record **no-go before GPU**, preserving all evidence. Conditional feasibility
does not permit launching merely to discover an unbudgeted mandatory step.

## Bounded handoff and separate historical spending

There is at most **one real-data admission/bridge invocation and one production
invocation**. Synthetic engineering defects may be corrected before admission,
with all executable study-check time retained. Any real-data identity,
compatibility, CPU correctness, bridge, cost, OOM, timeout, production or report
failure closes v13. Preserve interruption checkpoints as evidence without
resuming them. No report-only rescue, extra arm/seed, calibration/test promotion
or automatic version-reset chain follows. Do not alter the frozen formula,
weights, thresholds or lower allowances once execution starts.

Display v10's approximately **1,086.935835 s** and v11's **579.826784 s lower
bound** separately: at least approximately **1,666.762619 s** before v13, with
recorded uncertainty and omitted startup retained. V12 executed no study stage.
Add all v13 actual work to a separate cumulative investigation ledger. H is a
historical planning charge, not newly consumed compute; never claim the full
investigation or failed-version sequence cost at most two hours.

Expected evidence: design/protocol hash; source closure and semantic coverage;
CPU/compatibility receipts; executable inventory and persistent cost ledger;
bridge manifest/receipt; separate production manifest; per-arm checkpoints,
updates, diagnostics, features/refits, replay and completion hashes; frozen
two-seed report and final audit, or a precise immutable stop receipt. Bind exact
commands, revision and source/input hashes before execution. Update both queue
documents through implementation, checks, admission, bridge and completion.
At this handoff none of those execution gates is represented as passed.
