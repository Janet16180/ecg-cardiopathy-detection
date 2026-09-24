# Experiment 013: compact Mamba-3 for raw ECG

**Status:** user-authorized and queued for implementation, after Experiments 011 and 012. No runner, downloaded Mamba checkpoint, GPU profile or project performance result exists yet. This is a planning record; exact architecture and training settings must be frozen before a comparative run. See the [persistent queue](experiment-queue.md).

## Question and source

Can a compact Mamba-3 sequence encoder improve transfer from unlabeled twelve-lead ECGs? The proposed model learns waveform representations from scratch. Its language-model results motivate an architecture experiment; they do not establish an ECG ranking.

Primary references: [Mamba-3 paper](https://arxiv.org/abs/2603.15569), [official implementation](https://github.com/state-spaces/mamba). The paper introduces a richer state-space discretization, complex-valued state dynamics and a multi-input/multi-output formulation. Pin a specific source revision and verify which variant is implemented before assigning an experiment name.

## Initial comparison

| Arm | Role |
| --- | --- |
| Compact Mamba-3 | Proposed independent ECG encoder |
| Compact Mamba-2 | State-space family reference |
| Original compact GRU | Practical project reference |

Mamba-3 minus Mamba-2 is the primary architecture-family comparison. It changes a collection of mechanisms; it does not isolate complex-valued states or MIMO individually. Specify SISO/MIMO before training. If those mechanisms need attribution, define separate matched ablations without choosing them from test outcomes.

Use the same causal twelve-lead waveform stem, independent five-second halves, token positions, CPC horizons/negative exclusions, normalization and downstream mean/max readout across arms. Start all arms from fresh parameters, copy the common stem/prediction-head initialization, and report actual parameter counts and optimizer budgets. Choose the compact widths/depths before outcomes, with capacity matching where practical. Retain the existing GRU configuration as an explicitly labeled reference if a size-matched variant is also used.

The starting budget is 20 SSL epochs on the fixed 56,875-record pool, followed by the exact 15,360-label and 1,518-label transfer manifests. This budget is provisional until runtime profiling. Equal record exposure is not equal compute; report both. A different masked or bidirectional learning objective would require a separate comparison with matching information access.

## Next implementation steps

1. Pin official source and review the recurrence, precision requirements and V100 kernel support. Do not download a large text model as a substitute for implementing the compact ECG architecture.
2. Select and document the exact Mamba-3 variant and matched Mamba-2 settings. If the official kernel does not support V100, assess a faithful reference implementation; do not silently substitute an older architecture.
3. Verify numerical/gradient agreement against the chosen reference, causality, independent record/half state reset, and checkpoint/optimizer/RNG recovery on CPU. Add new modules without modifying the frozen existing jobs.
4. Prepare a resumable runner with source/data hashes, common splits and finite-update checks. Measure real forward/backward/optimizer memory and speed on the GPU only when scheduled; freeze one feasible batch policy across comparisons.
5. Attach verified commands to a successor executable queue, then update both persistent queue files with the launch and result locations.

Use development data for model selection, separate calibration patients for probability calibration and threshold selection, and paired patient-bootstrap comparisons on the fixed test set. Report actual test sensitivity, specificity, AUROC, AP and calibration. The current test cohort has already been inspected, and one seed supports exploratory conclusions only. Short token sequences and V100 backend costs may eliminate the advantages reported for large language models.
