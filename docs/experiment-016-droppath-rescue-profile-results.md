# Experiment 016 DropPath rescue: diagnostic and cost-gate result

The frozen three-arm, two-epoch development study **did not start**. Its [v5 profile-only manifest](../outputs/experiment_queue_016_rescue_profile_v5/queue.json) passed source verification, the paired-mode diagnostic, and all three full-path V100 profile correctness checks. The [cost receipt](../outputs/experiment016_droppath_rescue_v5/cost_gate.json) projects **7,624.25 seconds** for the study, exceeding the frozen **7,200-second** ceiling by **424.25 seconds**. The profile stage then deliberately exited nonzero; the coordinator's `failed` status records this gate, not a failed model or checkpoint check. No full training manifest exists. Calibration and test patients were not evaluated.

The [CPU check](../outputs/experiment016_droppath_rescue_v5/check.json) verified the exact 15,359-record clean training cohort, the 1,306-record development split, and pinned released xECG weights, feature cache, input receipts, and source identities. The fixed C=0.01 clean frozen probe had **0.9619404 development AUROC**; this is the starting probe, not a rescue outcome. Initial model logits matched the cached probe within 1.19e-6. Nine block conditional-mean checks and the Off train/eval identity check passed.

The [paired-mode diagnostic](../outputs/experiment016_droppath_rescue_v5/diagnostic.json) used 128 balanced **training ECGs** and eight paired mask seeds. Values below are means across the eight draws; the common eval-mode BCE was 0.23642 and eval AUROC was 0.97534 on this selected training subset.

| DropPath mode | Train-mode BCE | Train-mode AUROC | Mean absolute train/eval logit shift |
| --- | ---: | ---: | ---: |
| Legacy | 1.38697 | 0.80417 | 7.03066 |
| Residual | 1.15972 | 0.74786 | 4.10203 |
| Off | 0.23642 | 0.97534 | 0 |

Residual reduced the mean logit shift versus Legacy, but retained a large train/eval mismatch, and its diagnostic train-mode AUROC was lower. The balanced training subset and untrained mode comparison cannot establish a development performance gain. Off behaved identically in train and eval modes, as required by the control.

Each arm's [full-path profile receipt](../outputs/experiment016_droppath_rescue_v5/profile.json) covers one clean-data epoch, 15,359 ECG exposures, 240 updates, development evaluation, checkpoint write/load, and sequential same-device replay. Checkpoint model/optimizer/scheduler/RNG roundtrips passed; replay kept batch order, masks, RNG, and scheduler exact. Post-update floating tensors passed the [frozen v5 tolerance](experiment-016-droppath-rescue-v5.md) of `atol=1e-8, rtol=1e-6`; the largest absolute difference was 1.86e-9. Peak allocated GPU memory was 15.15 GB (decimal) across arms.

| Arm | Complete profile pass | Peak allocated GPU memory |
| --- | ---: | ---: |
| Legacy | 605.51 s | 15,150,038,016 B |
| Residual | 576.10 s | 15,150,135,808 B |
| Off | 722.35 s | 15,151,050,240 B |

The frozen gate used measured preparation plus profiles, **1,981.65 s**, and a conservative remaining-work allowance: `1,981.65 + 1.25 × (6 × 722.35 + 180) = 7,624.25 s`. The one-epoch profile models were discarded; their development scores are timing-stage observations, not a completed two-epoch comparison or model-selection result. The historical [016 result](../outputs/experiment016_xecg_probe_finetune/report.md) remains the only completed fine-tuning screen.

Versioned [v2](../outputs/experiment_queue_016_rescue_profile_v2/queue.json), [v3](../outputs/experiment_queue_016_rescue_profile_v3/queue.json), and [v4](../outputs/experiment_queue_016_rescue_profile_v4/queue.json) profile manifests and logs were preserved. They documented, respectively, restricted checkpoint metadata loading, V100 memory exhaustion from simultaneous replay copies, and tiny same-device post-update floating differences with exact batch/RNG state. The [v5 source map](../outputs/experiment_queue_016_rescue_profile_v5/sources.json) covers 32 frozen source, input, protocol, and test artifacts; `--check` passed after the profile. No source in that map or historical receipt was changed. A smaller successor requires its own frozen design, verified manifest, and full-path cost gate.
