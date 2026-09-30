# Experiment 040: causal Transformer and SimDINOv2-style results

All six real GPU profiles passed. The prescribed all-or-none time gate then rejected the 18-fit
suite: its conservative combined projection was 29,295.03 seconds against the shared 28,800-second
ceiling, a shortfall of 495.03 seconds (8.25 minutes). No full fit, development score or closed-set
evaluation occurred. The frontend and objective questions remain unanswered.

## Executed GPU profiles

Each package used seed 39042, 24 actual float32 optimizer updates at batch 128, exact next-update
checkpoint recovery, and a fresh final-size batch of 16 with complete objective gradients and exact
student/teacher/optimizer/RNG recovery. Every normal and final-batch check passed. Teacher gradients
were absent; its completed-update counter stayed zero for CPC and reached two in the partial-batch
SimDINO/hybrid recovery check. The teacher schedule was 1,954 updates for the active objectives.

| Objective | Context | Seconds/update | Peak allocated GB | Total parameters | Student parameters | Active student parameters |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| cpc | gru | 0.223869 | 1.303 | 3,034,112 | 1,615,360 | 1,615,360 |
| cpc | xlstm | 0.147943 | 2.653 | 3,117,600 | 1,657,104 | 1,657,104 |
| simdino | gru | 0.307570 | 2.167 | 3,034,112 | 1,615,360 | 1,418,752 |
| simdino | xlstm | 0.191258 | 4.850 | 3,117,600 | 1,657,104 | 1,460,496 |
| hybrid | gru | 0.241790 | 3.268 | 3,034,112 | 1,615,360 | 1,615,360 |
| hybrid | xlstm | 0.289729 | 7.307 | 3,117,600 | 1,657,104 | 1,657,104 |

Total parameters include the frozen EMA teacher, which is inactive for CPC-only. SimDINO-only
excludes the unused CPC predictors from active student parameters. Timing is a measured short
profile, not full-training time; the admission rule retains its frozen 1.5 multiplier and reserves.
No memory exhaustion, numerical failure or severe-performance diagnosis was observed. Since
full training never began, these profiles cannot establish convergence or development performance.

## Resource decision

| Admission quantity | Seconds |
| --- | ---: |
| Closed Experiment 039 execution | 10685.795656 |
| Experiment 040 charged before admission | 182.395425 |
| Combined charged work at admission | 10868.191081 |
| Conservative remaining schedule, including reporting reserve | 14826.834148 |
| Diagnostic/correction reserve | 3600.000000 |
| Projected combined total | 29295.025229 |
| Frozen ceiling | 28800.000000 |

The remaining projection covers all three objectives, both contexts and all three seeds, outstanding
preparation/profiles/readout/audit work, checkpoint costs and the 900-second reporting reserve.
The separate 3,600-second diagnostic/correction reserve was retained. The original cap, coefficients,
exposure count and scheduled family were unchanged; no faster replacement profile was selected.
The coordinator stopped at `resource_gate_failed`, with zero of 18 original fits completed.

## Integrity and validation

Preparation exactly reproduced all matched Experiment 039 development metrics before any new
score. The predecessor closed all 18 audited cells / 36 fits, its primary and exploratory analyses
and its final independent audit. Its immutable final charge was 10,685.795655530459 seconds.

The final independent 040 audit verified 134 pinned source/input hashes, all six profile contracts,
exact admission profile/manifest bindings, the resource arithmetic and the six successful stage
attempts in both cell and global ledgers. It read saved metadata/artifacts, without waveforms or
closed data. Its status is `reviewed_partial_receipts`; no scientific completion or family decision
is claimed. All nine scheduled cells lack completed training/readout/audit stages.

Implementation verification: 102 focused CPU tests passed in 11.91 seconds and Ruff passed.
GitHub CI passed 1,167 tests with six skipped in 114.16 seconds, Ruff, wheel and source-distribution
builds. CI now checks out the exact pinned public upstream loss reference. The earlier broad local
worktree run had 1,168 passes and one existing v11 path-guard failure caused by the shared outputs
symlink; that isolated frozen test passed in the main checkout (one pass in 13.14 seconds).

Local evidence: `outputs/experiment040_cpc_simdino/`, including the three first-seed manifests and
profiles, `admission.json`, `status.json`, `final_independent_audit.json`, its retained audit source,
`day_ledger.json` and the unabridged CLI-generated partial report. Original outputs are retained.
Protocol `ac18dac` and frozen scientific source commit `a675030` preceded these profiles.

## Interpretation and next work

This is a resource admission result, with no evidence favoring or rejecting an encoder or objective.
No bootstrap, superiority decision, combined-benefit claim, coefficient tuning or correction cycle
was performed. The original Experiment 040 backlog entry remains blocked. A separate prospective
resource plan is ranked in the backlog; any fresh allocation requires new authorization and an
explicit protocol/source identity before scoring. No automatic restart is scheduled.

The tested frontend is the fixed one-block local causal Transformer over the 039 patch projection:
49-sample raw support, stride 16, independent five-second halves, GRU or mLSTM context. The adapted
SimDINOv2-style path uses two raw-masked student views, one clean EMA teacher, 256-wide global mean
representations and the upstream coding rate with epsilon 0.05 and weight 0.1. Hybrid adds this
fixed loss to clean CPC. The frozen downstream representation remains 512-wide mean/max features.

The 25k v4 manifest is byte-identical to v3 and contains no EchoNext records. No SPH, Challenge
calibration/test, EchoNext test or final frozen test was evaluated. Experiment 008 remains deferred.
The label stays an ECG-annotation proxy; the candidate screening pipeline, referral rule and age
subgroups are unchanged.

## Git handover

The finalized personal Experiment 039 branch (`28aeceef17b809ff26112427ba60c44cd860c97b`)
was merged into this branch. Only the four queue/backlog/priority documentation conflicts needed
resolution; both studies and the predecessor provenance mapping were retained. `origin/main` is
merged. All frozen 040 scientific source/dependency/protocol bytes remain identical to `a675030`;
original receipt commit identifiers remain historical identities. PR #59 targets `main` and remains
unmerged.

## Final accounting

Experiment 040's final charged execution is **274.192796229152 seconds**;
closed Experiment 039 charged **10685.795655530459 seconds**.
The combined ledger is **10959.988451759611 seconds**, against 28,800. This includes
source preflight, every actual preparation/profile stage, the independent audit, report generation
and its measured startup/wrapper time, post-merge receipt/document validation, plus an explicit
30-second conservative allowance for the earlier untimed metadata/ranking startup. The allowance
is not presented as measured GPU or training time. No full-training or bootstrap time was executed.
Accounting writes use the same outer-stage convention as the frozen study ledger; no further
scientific execution is scheduled after this closure.
