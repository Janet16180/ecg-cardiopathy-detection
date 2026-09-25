# Experiment 016 v12: final bounded seed-47 replication attempt

**Design frozen 25 September 2026; implementation and measured admission pending.**
This is a separately identified final execution attempt, following the
[v10 device-helper failure](experiment-016-encoder-motion-replication-v10-results.md)
and [v11 cost stop](experiment-016-encoder-motion-replication-v11-results.md).
Neither attempt reached a seed-47 GPU update or development prediction. Their
stops remain valid. V12 is not a retry inside either historical allowance and
does not make the investigation a two-hour study. This document authorizes no
launch by itself; the executable checks and gates below are mandatory.

## Decision and evidence

A final attempt is **conditionally feasible**, without changing the 7,200 s
ceiling or scientific comparison. The execution change is two consuming
processes: (1) fresh verification, complete CPU correctness, then the bounded
bridge; (2) a separately frozen production successor with fresh verification,
both production arms and the report. The first process may reuse its own
verified bytes for the subsequent bridge; the second must rehash them.
No coordinator launches a separately hashing child. A third process, extra
scan, or artifact read is never free and must enter the same ledger.

The actual v11 code is not a complete runner. Its CLI supports only preflight;
`verify_historical_closure()` writes a receipt, and `_semantic_inputs()` checks
identity but does not return production data objects. A versioned materializer,
launcher, compatibility map, CPU check and gate integration remain necessary.
Do not call the design implemented, runnable, or cost-passed until those checks
exist. Preserve v9/v10/v11 code, manifests, outputs and receipts unchanged.

V11's successful current-byte closure checked 288 leaves, 1,051 obligations
and 6,493,086,486 bytes in 172.721 s. Use the slower observed 176.727 s scan as
the historical floor. Its two separate CPU initialization/inference probes
took 24.863 and 20.578 s; they were partial probes, not a passed complete check.
V9's per-arm final artifacts total about 753 MB, including a 684,552,140-byte
checkpoint. Two bridge checkpoints therefore add about 1.37 GB to the fresh
production admission closure. At the slower historical whole-pass rate, that
extra read costs approximately 37.3 s. The budget below includes it.

## Unchanged scientific protocol

Inherit the complete [v10 scientific protocol](experiment-016-encoder-motion-replication-v10.md),
[v9 controls](experiment-016-encoder-motion-v9.md) and
[v11 provenance/compatibility requirements](experiment-016-encoder-motion-replication-v11.md).
Only new output identity, verification/materialization, launch organization and
cost guards change. In particular:

- Use all **15,359 clean full-label training ECGs**, unchanged order/labels,
  excluded ECG 12722, and the same 1,306 development ECGs/1,173 patients.
  Raw physical-mV twelve-lead 100 Hz cache, patient partitions and train-only
  fitting stay fixed. Calibration/test endpoint access remains closed.
- M/F start from identical released xECG and the original clean C=.01 affine
  head with fresh optimizer state. Seed **47** fixes both global randomness
  and common complete permutation. No historical/profile/bridge initialization.
- Keep vanilla backend, dropout/stochastic depth Off, FP32 V100, microbatch 16,
  effective batch 64, 240 updates/15,359 exposures and last batch 63. Keep
  `k/240` warmup at k=0,...,239 and update-240-only endpoints.
- M encoder base LR 3e-5, layerwise decay .75, head LR .001. F encoder rates
  are exactly zero while retaining encoder gradients, optimizer groups/state,
  full backward and global norm clipping at 3.0. AdamW betas (.9,.999), epsilon
  1e-8, decay .1 and historical grouping including head bias are unchanged.
- Verify every-update F encoder/buffer identity, M movement at the first
  nonzero step, common initial gradients, native/affine logits and the released
  F feature/refit control. Preserve all per-update diagnostics, fixed 128
  training-record diagnostics at 0/1/60/120/240, RNG/mode restoration,
  checkpoints every 40 updates and on interruption, and sequential replay.
- Fully extract final train/development features for **both** encoders and fit
  separate train-only StandardScaler + C=.01 lbfgs probes, max_iter=3000,
  random_state=42, requiring convergence. No cached-F shortcut.

The sole primary contrast remains `D47 = AUROC(F_joint47)-AUROC(M_joint47)`.
The practical harm screen is D47 >= .005; separately flag a strictly positive
interval. The narrower readout pattern additionally requires
`R47 = AUROC(M_refit47)-AUROC(F_refit47) >= -.002` and
`G47 = (M_refit47-M_joint47)-(F_refit47-F_joint47) >= .005`, with all terms
expressed as AUROC. Preserve directional/nonreplication and uncertain-interval
labels from v10; no combined estimate overrides a failed seed-47 screen.

Report all four readouts and released probe with AUROC/AP/BCE and unchanged
five patient-group threshold folds. Use 2,000 common paired whole-patient
bootstrap draws, seed **16020**, exact ECG/patient/label alignment, preserved
patient multiplicity, percentile 95% intervals and invalid-draw counts.
Fewer than 1,900 valid draws leaves interval inference unresolved. Show both
seed contrasts, their range and arithmetic mean with common-draw intervals;
never average logits. Retain v9's original seed-16019 report; label recomputed
seed-46 intervals as new combined-analysis intervals. Two fixed seeds and
repeated development inspection do not provide independent validation or
seed-population uncertainty. Retain the released probe as practical reference.

## Verification and two-launch execution

Freeze new v12 reusable/CLI/test files and output root
`outputs/experiment016_encoder_motion_replication_v12`. Freeze a bridge-only
manifest/source map before the first real-data process; it permits CPU checks
and bridge only. The separate production manifest must bind the passed CPU,
compatibility, bridge and gate receipts plus all final code/protocol hashes.
Record exact commands, revision, source maps, dependencies and device receipts.
No production manifest exists at design time.

Before real-data execution, synthetic tests must exercise the exact installed
CUDA helper (strict fake API with `device_count`, without `get_device_count`),
preflight-before-hashing failures, full permutation/scheduler logic, provenance
omissions/conflicts/mutations, wrong identity/seed/checkpoint, incomplete pairs,
ledger omissions and gate arithmetic. Charge executable study-check time.

Each launcher performs cheap manifest/source/API validation first, then obtains
the shared GPU lock nonblocking and checks actual host process/device identity
before any large read. Preserve downloader paths/processes and do not restart
the legacy runner. Keep the lock through the consuming process; no unlocked gap
between preflight and use. Record installed Torch/CUDA, V100 16 GB and current
driver; historical driver equality remains unproven. Incompatible device or
environment stops before hashing. No intentional queue wait is allowed.

In the bridge process, hash the complete transitive closure anew. Version v11's
tested verifier into v12 without writing v11 outputs. Retain every historical
hash obligation, semantic assertion and exact pinned nested-v9 fingerprint.
Materialize the exact historical data objects from the same validated graph,
without calling old recursively hashing loaders. Compare runtime bindings and
ASTs with v9/v10, including the materializer's returned cohort, row mapping,
probe and diagnostic inputs. No global monkeypatching or mutable shared cache.
Reject missing/conflicting hashes, path escape, symlink substitution and changed
file identity. Keep input identity under observation during the consuming
process and abort on change; observations never replace a new process's hash.

Complete the real CPU correctness check in that same process: identical M/F
initial tensors/four real training ECG logits, zero optimizer moments, complete
seed-47 permutations equal across M/F and different from 46, group rates and
all 240 warmup multipliers. Verify the full production callable/dependency map.
Free CPU models before GPU work. Recheck the measured cost gate **before the
first bridge update**. Only a passed check and gate permit proceeding.

The bridge retains exactly v10's M-then-F three real effective-batch updates,
same 240-update scheduler, first-two-update gradient/head/control comparisons,
CPU serialization and two sequential next-update replays. Keep exact
order/counters/RNG/scheduler and `atol=1e-8, rtol=1e-6` floating-tensor checks.
Bound the entire GPU bridge, model initialization, I/O, replay, cleanup and
bridge completion work to **300 s**. One active GPU model at a time; no
development metrics. Timeout, OOM or failed correctness closes v12.

The separate production process freshly hashes the complete historical closure
**and new bridge/check/compatibility/executable artifacts**. It starts M then F
from release/probe, never bridge weights, and runs the unchanged full pipelines.
Recheck admission after preparation, before M, after M/before F, at safe
checkpoints and before reporting. Generate the two-seed report in this process;
rehash all artifacts it reads and all completion artifacts. Reusing the same
historical pass does not waive verification of newly generated outputs.

## Frozen cost rule and limited feasibility claim

Keep the v11 equation and historical charges exactly:

```text
H = 2693.7965717150364 s
P = 1098.4113020410005 s
E = all actual v12 executable study-stage wall time so far
n = completed v12 production pipelines (0, 1, 2)
Pstar = max(P, every measured completed v12 pipeline)
Qremaining = all remaining preparation, verification, bridge and other overhead
R >= 300 s until report/completion work finishes
projected = H + E + 1.25*((2-n)*Pstar + Qremaining + R) <= 7200 s
```

The initial **planning allowances**, not newly measured stage timings, are:

| Remaining work beyond production pipelines/report reserve | Seconds |
| --- | ---: |
| Fresh bridge-process historical closure and semantics | 200 |
| Fresh production closure including two new bridge checkpoints | 225 |
| Complete CPU materialization/correctness beyond the closure | 100 |
| Entire bounded GPU bridge | 300 |
| Study-specific synthetic checks, manifest binding/checks, startup, host checks and additional artifact/completion audit overhead | 200 |
| Total Qremaining | 1,025 |

Thus `H + 1.25*(2*P + 1025 + 300) = 7096.074827 s`, leaving
**103.925173 s of inflated headroom**. This is a conservative initial envelope,
not a passing measured admission receipt. The 200/225 s scan allowances round
up whole-pass evidence and the added bridge-checkpoint byte cost; the 100 s CPU
allowance exceeds twice the 45.441 s observed partial probes. The 200 s overhead
allowance remains a reserve whose inventory must be implemented and measured.
Any remaining unmeasured mandatory component without a defensible bound means
the admission gate cannot pass. Do not silently treat that component as zero.

Before launching the real bridge process, the inventory must enumerate all
future scans and checks, artifact bytes and report work. Use the larger of these
allowances and slower applicable observed measurements/whole-pass throughput
forecasts; do not reduce them on a faster warm-cache run. At the CPU admission
boundary replace completed work with actual E and retain all remaining reserves.
Charge new-source/manifest hashing, imports, data materialization, synthetic and
real checks, device checks, failed invocations and any unforeseen extra process.
Nested timings count once, but enclosing startup/cleanup never disappears.
Report actual nonoverlapping monotonic intervals and conservative forecasts.
If the report cannot fit 300 s, increase R and reapply the same gate.

Checkpoint guards must budget remaining work plus checkpoint/stop overhead,
not merely wait for elapsed H+E to reach 7,200. A final completion requires
both a valid complete pair/report and `H + actual total v12 stage time <= 7200`.
Report any breach even if artifacts finish. No shortened epoch, label subset,
weaker verification, alternate precision, dropped control or threshold change
can cure a failed gate.

Keep a separate cumulative investigation ledger showing v10's approximate
**1,086.935835 s** and v11's **579.826784 s lower bound**, totaling at least
approximately **1,666.762619 s** before v12, with their recorded uncertainty
and omitted startup explicitly retained. Add all v12 actual work to this total.
Display H as a historical planning charge, not newly consumed compute. Design
reading/writing time is separate from executable study work. Never claim the
combined investigation, or repeated failed versions, cost only two hours.

## Final-attempt rule

There is **one real-data admission/bridge invocation and one production
invocation**. No automatic rerun of a failed real-data check, bridge, production
or report; no production resume in this final attempt. Preserve interruption
checkpoints as evidence without launching them. Synthetic engineering tests
may be corrected before real-data admission, with all executable study-check
time retained in E and remaining costs rechecked. Any failed real-data identity,
compatibility, correctness, resource or cost gate ends v12 without a completed
replication result. A partial pair is not a scientific outcome.

No v13/reset fallback, changed ceiling, new seed, extra arm, calibration/test
promotion or other experiment follows automatically. Once execution starts,
do not revise the gate, allowances downward, thresholds or decision labels in
response to cost or scientific outcomes. Close the attempt and preserve the
negative/failed record if the honest measured gate cannot fit. Both queue
documents must distinguish design completion from implementation, admission,
bridge, production and final evidence.
