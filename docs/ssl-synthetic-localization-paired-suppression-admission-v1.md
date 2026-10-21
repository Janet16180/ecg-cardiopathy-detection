# Paired suppression identity and accounting admission

19 October 2026. This admission follows the successful independent audit of the fit replication.
The paired suppression protocol and scientific config were committed as `4500999` before the
coordinator first read the replication outcomes. This new record changes no hypothesis, loss,
coefficient, seed, split, comparison, decision threshold, operation count or resource limit.

The companion admission config binds 33 explicit replication artifacts by hashes freshly read from
their own paths. These comprise the main and audit summaries and ledgers, fresh case metadata and
observed arrays for exact duplicate checks, and fit 1–3 initial states, schedules, supervised final
model/Adam/loss files and both arms' fresh metric rows. The 45 fixed generator and original learning
bindings in the frozen scientific config were also checked against their explicit live paths with
zero mismatches. Thus the four new replication receipts join the 45 unchanged parent bindings;
the remaining replication bindings identify the specific replay and closure artifacts.

The audited replication summary is
`a9f9daca25273f23f10dc46faca019bd680eefe24c1c6cc951f33e3fcb46282e`, and its terminal ledger is
`f17f0c112a3eeed8b5f46be389c2a06ff8dcb8c2dd6169b1833a398f20a0d264`.
The latter is the single accounting base for this successor: cumulative current-machine CPU
190.681380118 seconds and GPU-lock wall 178.56192114995793 seconds. Add only this successor's new
main and audit terminal costs. Do not add earlier stages again or convert unknown historical costs
into measured values.

The replication's actual whole-stage wall time was 241.48855727212504 seconds, exceeding its
167.77376635419205-second profile forecast while remaining within its frozen cap. The new study's
already-required recurrent full-path update timing must include batch construction, forward,
complete objective, backward, Adam update, loss/component persistence write and flush, and CUDA
synchronization. Profile and main use the same persistence path. Measure the original steps 2 and 3
for every B, P, K and PK objective, then use the slowest as specified in the frozen forecast. This
clarifies what a complete recurring path includes; it authorizes no additional updates, timing
discard, retries, changed safety factors or larger limits.

Retain the already computed logits from the first original profile forward, before its optimizer
update, for all four arms. Save float32 arrays of shape (8, 12, 5000) as
`fit_1/profile/{arm}/step1_logits.npy`, with int64 shape-(8,) case indices alongside them as
`step1_case_indices.npy`; these indices must equal the first fit-1 training-schedule row. This uses
zero additional model forwards or updates. Add 8 MiB to the forecast for the four payloads totaling
7,680,000 bytes and their small headers/indices; charge capture and persistence once as cold
profile work. The main 1 GiB cap is unchanged.

The independent audit reconstructs the admitted TRAIN targets and evaluates the real first-step
BCE, H, C, K and total objective from these saved logits using float64 arithmetic. Compare each
active component and total with absolute tolerance 1e-6 to allow float32 GPU sigmoid/reduction
rounding; absent terms remain zero. Other metric/metadata comparisons retain absolute tolerance
1e-12. This strengthens first-profile objective evidence without independently replaying the full
training path, which remains source-bound. It changes no observed input, target or objective.

The independent replication audit reports zero mismatches across 264 bindings and independently
reconstructed saved metrics. It uses no additional model forward or optimizer update. Its success
permits predecessor admission; it does not remove the need for source review, exact B-arm replay,
new actual-path profile gates, and the separately counted independent audit of this successor.
