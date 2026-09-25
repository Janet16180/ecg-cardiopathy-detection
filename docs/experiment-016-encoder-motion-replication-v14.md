# Experiment 016 v14: prospective seed-47 replication

**Design frozen 25 September 2026 before seed-47 outcomes. Feasible for
implementation; not yet runnable.** This continues the user's CPC-then-xECG
request after CPC completed. It preserves all v9–v13 evidence and their recorded
decisions. Implementation, one complete CPU check, executable compatibility,
a bounded V100 bridge and measured cost admission must pass before production.
One experiment at a time; calibration/test remain closed.

## Prospective cost decision

The queue's `priority_policy` identifies two hours as a **working GPU-pilot
planning gate**, adopted after the rejected 22-hour suite, not a user-specified
cumulative research limit. The queue requires full data-path costs, all arms,
evaluation and checkpoint writes. Charging historical v9 preparation/profile
time H again was an additional v10–v13 protocol choice. Those frozen attempts
remain closed under their original rules.

V14 prospectively applies **7,200 seconds to all new executable study work**,
including CPU checks, verification, preparation, bridge, production and report.
Historical profile reuse is allowed only after the compatibility and bridge
requirements below. Historical spending is disclosed separately, never erased
or described as free. This is a changed accounting policy, not a claim that
the complete investigation fits two hours, and does not permit repeated version
resets or an unbounded attempt. No new numeric permission is needed to implement
the user's requested study under the project's actual planning convention.

| Already incurred evidence | Recorded seconds |
| --- | ---: |
| v9 preparation and both complete profiles, H | 2,693.796572 |
| v9 two production pipelines | 1,955.290292 |
| v9 report | 81.265179 |
| v10 failed CPU/check/bridge attempt, approximate | 1,086.935835 |
| v11 executable work, lower bound | 579.826784 |
| v12 executable study stages | 0 |
| v13 synthetic executable checks, lower bound | 9.394027 |

These disjoint recorded components total **at least approximately 6,406.509 s**
for v9–v13. V10's wall time is approximate; unmetered outer launches, final
artifact hashing and other omissions mean this is not a complete historical
total. V10–v13 alone account for at least approximately **1,676.157 s**.
Earlier xECG studies, CPC and engineering time are additional and not included
in either subtotal. Report this history beside actual v14 elapsed work; never
claim the combined spending is below two hours.

## Unchanged scientific endpoint

Inherit **all scientific execution, controls, decisions and interpretation**
from [v10](experiment-016-encoder-motion-replication-v10.md), with v14 artifact
identity. Its historical hash table remains the pinned reference. In particular:

- Exactly 15,359 clean full-label training ECGs (exclude ECG 12722), 1,306
  development ECGs/1,173 patients, identical ordered rows, labels, patient
  partitions, physical-mV twelve-lead 100 Hz preprocessing and waveform cache.
  Train-only fitting; no calibration/test features, predictions or evaluation.
- Fresh identical released xECG tensors and clean C=.01 probe head, fresh Adam
  moments, **seed 47** and common full training permutation. Never initialize
  from a bridge, profile, prior run or partial production checkpoint.
- M original encoder LR 3e-5/layerwise decay .75 versus F exactly zero encoder
  LR, retaining full encoder gradients/Adam state/global clipping. Both use
  head LR .001, FP32 V100, dropout/stochastic depth Off, BCE, AdamW (.9,.999),
  epsilon 1e-8, decay .1 including head bias, global clip 3, microbatch 16,
  effective batch 64, 240 updates/15,359 exposures, final batch 63, k/240 warmup.
- Preserve every-update F encoder/buffer identity, M movement, common initial
  gradients, per-update logs, fixed 128-training-ECG diagnostics at 0/1/60/120/240,
  mode/RNG restoration, checkpoints every 40 updates and on interruption,
  sequential same-device replay (`atol=1e-8, rtol=1e-6`; exact counters/order/RNG).
- Fully extract final train/development features for **both** arms, check
  native/affine logits and F released-feature identity, and independently fit
  train-only StandardScaler/C=.01 lbfgs probes, max_iter=3000, random_state=42,
  requiring convergence. No cached-F shortcut or omitted diagnostic.

Primary `D47 = AUROC(F_joint)-AUROC(M_joint)` passes at **>= .005**, with a
separate interval-lower-bound-above-zero flag. The narrower readout pattern
additionally requires `R47 = AUROC(M_refit)-AUROC(F_refit) >= -.002` and
`G47 = (M_refit-M_joint)-(F_refit-F_joint) >= .005` in AUROC units. Preserve
v10's directional/nonreplication/uncertainty labels and incomplete-pair rule.
Use its unchanged five patient-group threshold folds, AUROC/AP/BCE, 2,000
common paired whole-patient bootstrap draws, seed **16020**, multiplicities,
95% percentile intervals and >=1,900 valid draws. Show both seeds, their range
and arithmetic mean of contrasts with shared draws; never average logits or
let a combined mean override a failed seed-47 screen. Preserve the original
v9 report/seed-16019 intervals. Two fixed optimization seeds on repeatedly
inspected development patients do not establish independent validation,
seed-population uncertainty, an exclusive causal mediator or clinical benefit.
Retain the released probe as the practical reference.

## Lean verified implementation and launches

Create new reusable/CLI/test files and
`outputs/experiment016_encoder_motion_replication_v14`. Reuse v9 scientific
primitives and v10 orchestration by explicit versioned copies/imports. Do not
edit historical sources or receipts, monkeypatch globals, alter scientific
bindings or implement another transitive-verifier/materializer framework.
The **existing v10 `_verify_map`/`inputs` followed by v9's verified input path
is acceptable**, including repeated hashes. Preserve every inherited source,
input, split, cohort, release, cache and completion-artifact check; meter every
repeat. Do not replace fresh byte verification with old receipts or metadata.

Record a compatibility map to authenticated v9/v10 callables, source hashes
and runtime bindings, AST identity for unchanged science and an explicit diff
for identity, early API checks, timers, cost hooks and combined launch/report
orchestration. Bind the complete imported executable/source closure, protocol,
revision, exact command, dependencies/lockfile, environment and input identities
in new manifests/source maps. Reuse v9's **measured full M/F pipelines** only
if their manifests, maps, completion artifacts, input fingerprints and passed
correctness flags authenticate and this map passes. A bridge alone is not a
full profile. Changed science/input/environment means stop, not an automatic
reprofile. Record present driver and historical device evidence honestly;
the historical driver version was not recorded.

Use one CPU-check invocation, one bridge-only coordinator/child launch and one
fresh production-plus-report successor launch. A single focused synthetic
test/lint command may precede the real-data check; include its time. Extra
source-map creation/validation reads are allowed and counted, not free retries.
Freeze each manifest before use; production binds passing CPU/compatibility/
bridge/cost receipts and hashes both bridge checkpoints. Report runs in the
production process without a fourth recursively hashing input load.

Before any expensive read, exercise callable **`torch.cuda.device_count()`**
and the other installed API names on the cheap path; never use nonexistent
`get_device_count`. A synthetic fake exposing only genuine names guards this.
The single real-data CPU check compares initialized M/F tensors/four training
logits, empty moments, complete common seed-47 permutation distinct from 46,
optimizer groups and all 240 warmup multipliers. Inherited checks apply only
to authenticated unchanged implementations. Test the new cost logic, identity
rejection and incomplete-pair guard without extra real-data rehearsal.

Before each GPU launch check live coordination/process identity and V100
availability, preserve downloaders, and take the shared nonblocking GPU lock.
Use one active model. The bridge retains v10's M then F first three full-size
effective-batch updates and 240-update schedule, first-two-update comparisons,
M movement/F identity, CPU serialization and two sequential next-update
replays. Its **whole GPU stage is capped at 300 s**, including initialization,
checkpoint I/O, replay, cleanup and completion work, with no development
metrics. A timeout/OOM/correctness failure closes v14. Production starts fresh
from release/probe and runs M then F plus the required report/artifact audit.
There is no automatic retry, resume, additional seed or endpoint change.

## Incremental complete-path budget and watchdog

Use measured historical `P = 1098.4113020410005 s`, the slower complete v9
profile pipeline, with a **25% margin on remaining work**. Historical P ended
before final artifact hashing: account for those hashes explicitly below.

| Initial remaining component | Seconds | Basis |
| --- | ---: | --- |
| Complete CPU check including historical input reads | 550 | v10 measured 452.593 |
| Bridge launch, coordinator/input verification and preparation | 600 | v10 failed whole launch measured about 544.343 before bridge updates |
| Entire GPU bridge | 300 | fixed hard cap |
| Production launch/input verification including bridge artifacts | 600 | same observed path plus two approximately 684.6 MB checkpoints |
| Synthetic checks, source-map/manifest preparation, final arm artifact hashes and repeated completion audits outside P | 300 | explicit extra allowance; enumerate reads before launch |
| Two full production pipelines | 2P | authenticated full-data profile |
| Two-seed report/input rereads and final report completion | 300 | v9 single-seed report 81.265; additional shared-draw analysis |
| Safe checkpoint/stop reserve | 60 | raise to slower observed safe-stop cost if necessary |

Initial projection is **1.25*(2350 + 2P + 300 + 60) = 6,133.528255 s**,
leaving **1,066.471745 s** below 7,200. This is a historically supported
planning forecast, **not a measured v14 pass**. Before launch enumerate every
scheduled process/hash/read; increase inadequate allowances. In particular
include new and seed-46 final artifact rereads, each roughly 753 MB per arm,
with no timing gap at historical P's boundary. No closure deduplication is
required to fit this initial forecast.

Let E be all actual v14 executable study-command wall time, counted once from
before imports/coordinator startup through completion, including failures,
checks and source-manifest preparation. Nested timers partition wall time.
Engineering reading/writing is separate. Preserve the ledger across processes;
new manifests cannot reset E. Let Q be enumerated unfinished nonpipeline
allowances, R the unfinished report allowance (300 until report completion),
S the stop reserve (>=60 while stopping may be needed), and U unstarted arms.
Let Pstar be max(P, every completed new full-pipeline time).

```text
projected_new = E + 1.25*(U*Pstar + active_remaining + Q + R + S)
GO iff projected_new <= 7200
```

Before an arm, `active_remaining=0` and that arm is in U. During initialization
and until the first 40-update checkpoint, reserve one Pstar for its remaining
pipeline. At each completed 40-update checkpoint, record the elapsed block
including due diagnostics/checkpoint work (first block includes initialization).
Set `r` to the slowest observed complete block seconds/update in either arm.
Forecast active remaining as **`(240-k)*r + Pstar`** while its post-training
tail is unfinished. The extra Pstar conservatively reserves a *whole historical
pipeline* for the unknown development/extraction/refit/replay tail; it is not
an invented measured phase duration. Do not add elapsed active work again.
At k=240 only that tail remains. Retain this conservative tail allowance until
all tail work finishes; its spent portion is not claimed as scientific progress.
If observed tail time exceeds Pstar, raise the remaining tail allowance to the
larger observed duration and reapply the gate. On full completion set active
remaining to zero, update Pstar and reserve a full Pstar for any unstarted F.
Keep final artifact hashes in Q unless explicitly included in the new timer.

These are prospective forecast updates, not a requirement for missing historical
40-update measurements before GPU. The 300 s bridge admits only compatibility;
the two measured historical full pipelines justify initial cost feasibility.
At runtime a slow first block or tail may legitimately stop the attempt. For
scale, at E=2,530 s after a hypothetical 180 s first block, U=1, k=40 and
r=4.5 s/update, Q=0, R=300, S=60, projection is **6,851.028255 s**. This is
an arithmetic feasibility example, not a promised observed pace.

Recheck before/after bridge, after fresh production verification, before arms,
at every 40-update checkpoint, before/after tail work, between arms and before
reporting. A monotonic watchdog enforces the remaining actual wall deadline
through long operations and requests safe termination with S remaining; retain
checkpoints on interruption. Stop on a failed forecast or actual 7,200 s breach
and record the incomplete result. If work cannot stop promptly, report any
overrun honestly. Do not inspect development metrics to govern cost. Completion
requires the valid paired endpoints, probes, replays, report, artifact audit and
actual incremental E <= 7,200; report historical and cumulative spending beside E.

The design is implementable using the proven data/scientific path with small
versioned orchestration changes. Mark it runnable only after implementation
checks; freeze the verified production successor only after a passing bridge
and measured admission. Design completion itself launches nothing.
