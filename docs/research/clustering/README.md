# ECG clustering

New clustering code belongs in `ecg_experiment/clustering/`, mirrored by `tests/clustering/`.
Place each new study's protocol and results in its own directory here. Existing study paths are retained.

## Physical morphology

| Study | Protocol | Results |
| --- | --- | --- |
| CLUSTER-01: physical morphology | [Protocol](../../cluster-01-physical-morphology.md) | [Results](../../cluster-01-physical-morphology-results.md) |
| CLUSTER-02: stability selection across representations | [Protocol](../../cluster-02-stability-selection.md) | [Results](../../cluster-02-stability-selection-results.md) |

## Learned representations

- [Experiment 024: CPC geometry and reference prototypes](../../experiment-024-embedding-geometry-results.md)
- [Experiment 024b: geometry across hospitals and encoders](../../experiment-024b-multisource-geometry-results.md)

These links distinguish the recent physical-feature studies from earlier learned-vector studies.
Use their results and available live receipts to establish which findings can actually be reproduced.
Clustering describes similarity and associations; it does not establish disease causes.

Follow the [shared directory guide](../README.md) for new code, commands and aggregate reports.
