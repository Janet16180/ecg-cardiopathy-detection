# NLP-inspired ECG architectures on the fixed 25k subset

**Completed 27 September 2026.** Experiments 011–013 each used the same
Experiment 019 source-stratified 25,000-record training selection (index SHA-256
`43f405bfbddbb415f5072480744cebe6427afbeed2474a7be0e625cf1deac791`),
115,359 waveform exposures and 902 updates per arm. Their within-experiment
controls used the same row stream and objective. Each study had a real V100
full-path profile below its 7,200-second admission gate, a separate verified
full queue, and an independent artifact audit. The reported endpoint is an ECG
diagnostic annotation proxy. Calibration and test patients remained closed.

| Experiment and prespecified contrast | Full-label AUROC, 15,359 labels | Limited-label AUROC, 1,518 labels | Paired patient 95% CI for difference, full / limited |
| --- | ---: | ---: | --- |
| 011 CKDA minus KDA | 0.92660 vs 0.92109; **+0.00551** | 0.90839 vs 0.90360; **+0.00479** | [−0.00225, +0.01326] / [−0.00275, +0.01304] |
| 012 mixed support minus matched local support | 0.91958 vs 0.92954; **−0.00996** | 0.89971 vs 0.90963; **−0.00993** | [−0.01865, −0.00210] / [−0.01914, −0.00061] |
| 013 Mamba-3 SISO minus Mamba-2 | 0.91769 vs 0.92586; **−0.00818** | 0.89577 vs 0.90383; **−0.00806** | [−0.01722, +0.00083] / [−0.01745, +0.00178] |

The primary comparison in 012 disfavors the mixed support architecture at both
label budgets. The 011 and 013 intervals cross zero, so these single-seed
screens do not establish a difference for their primary contrasts. Secondary
reference comparisons are informative but do not change those decisions:
011 CKDA exceeded its fresh GRU by +0.01046 AUROC at full labels (CI +0.00125
to +0.01954), and 013 Mamba-2 exceeded its fresh GRU by +0.01250 (CI +0.00375
to +0.02161). These are separate model initializations and should be compared
within their own experiments. Average precision, the second label budget, and
all arm scores are in the exact result receipts below.

| Experiment | V100 projected complete cost / 7,200 s | Frozen queues | Result and independent audit |
| --- | ---: | --- | --- |
| 011 | 6,194.37 s | [profile](../outputs/experiment_queue_nlp25k_011_profile_v2/queue.json), [full](../outputs/experiment_queue_nlp25k_011_full_v2/queue.json) | [result](../outputs/experiment011_delta_memory_25k_v2/result.json), [audit](../outputs/experiment011_delta_memory_25k_v2/audit.json) |
| 012 | 1,555.26 s training suite; separate 956.16 s readout | [profile](../outputs/experiment_queue_nlp25k_012_profile_v3/queue.json), [train](../outputs/experiment_queue_nlp25k_012_train_v3/queue.json), [readout profile](../outputs/experiment_queue_nlp25k_012_readout_profile_v3/queue.json), [readout](../outputs/experiment_queue_nlp25k_012_readout_full_v3/queue.json) | [result](../outputs/experiment012_temporal_hybrid_25k_v3_readout/result.json), [audit](../outputs/experiment012_temporal_hybrid_25k_v3_readout/audit.json) |
| 013 | 1,374.09 s | [profile](../outputs/experiment_queue_nlp25k_013_profile_v2/queue.json), [full](../outputs/experiment_queue_nlp25k_013_full_v2/queue.json) | [result](../outputs/experiment013_mamba3_25k_v2/result.json), [audit](../outputs/experiment013_mamba3_25k_v2/audit_v2.json) |

The source hashes, exact launch commands and status receipts are in the frozen
queues and [live coordination record](experiment-queue.md). The shared
[cache seal](../outputs/cache_sessions/nlp25k_v1/seal.json) was created by a
fresh full SHA-256 check of both waveform caches; subsequent stages validated
file identity and bounded content blocks rather than rehashing the whole cache
on every launch. The profile-only 011 v1 was interrupted before that seal and
contributed no result. A versioned 013 auditor corrected a missing root-level
`pyproject.toml`/`uv.lock` path mapping after its first audit failed; frozen
experiment sources and outputs were not edited. Its final audit passed.

These results are short-budget, one-seed development screens on a repeatedly
inspected cohort. They do not establish clinical utility, broad failure or
superiority of NLP architectures, or the result of a longer training run.
Any positive follow-up needs a separately frozen seed replication and artifact
checks before calibration or test is considered. No follow-up GPU job is
scheduled by this report.
