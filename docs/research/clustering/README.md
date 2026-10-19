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

## Native signal and reference-conditioned geometry

[CLUSTER-03 protocol](cluster03-native-reference/protocol.md), [profile results](cluster03-native-reference/profile-results.md)
and [results](cluster03-native-reference/results.md) compare native QRS/global ECG-JEPA vectors, the
actual No Time to Train record/regional affinities and physical features on 1,993 matched development
patients. All selected partitions are stable under patient subsampling, but cross-view/reference-bank
persistence fails. Raw QRS clusters are strongly rate-associated; NTT affinities produce mixed,
reference-informed annotation enrichment rather than independently discovered disease categories.

## Reference uncertainty and matched controls

[CLUSTER-04](cluster04-bank-robustness/results.md) tests fifty support-bank draws within the fixed
reference pool. Its conditional robustness gate passes, and five-bank coordinate averaging improves
assignment persistence. This does not reverse CLUSTER-03 or establish clinical disease categories.
[CLUSTER-05](cluster05-matched-controls/results.md) completes the separately frozen dimension,
regional-pooling and scalar-scale comparison. Its matched concordance gate fails, and fixed-GMM
posthoc checks expose numerical-scale dependence. Neither result supplies clinical evidence masks.
