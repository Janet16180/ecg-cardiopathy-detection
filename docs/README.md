# Documentation index

## Working on the project

- [Contributor workflow](../CONTRIBUTING.md): setup, reviews, tests, and experiment discipline.
- [Command guide](../scripts/README.md): script folders and entry points.
- [Environment setup](../environments/README.md): uv and pretrained dependencies.
- [Data versioning](data-versioning.md): DVC, completed datasets, and validation.
- [Experiment tracking](experiment-tracking.md): MLflow and historical results.

## Data

- [Sources and acquisition](data-sources.md)
- [Quality assessment](data-quality-assessment.md)
- [Data processing review](data-processing-astra-review.md)
- [Clean-data audit and cheap rerun plan](clean-data-rerun-review.md)
- [Training dataset v1](training-dataset-v1.md)
- [Processed ECG exploration](processed-ecg-eda.md)
- [Challenge postprocessing](challenge-postprocessing.md)
- [Public-data strategy](public-data-strategy.md)

## Experiments and findings

Start with the [experiment queue](experiment-queue.md) and
[JSON catalog](experiment-queue.json). They link every active, deferred, completed,
and proposed experiment to its protocol and evidence. The 25 September scoped
016 and clean cached-feature reruns, followed by the development-only 017 clean
replication, are complete. The later 016 seed-47 bridge stopped before GPU
updates. No experiment queue is active; new studies proceed one at a time under
the queue's cost and verification gates.

- [Model findings](model-findings-report.md)
- [CPC improvement investigation backlog](cpc-improvement-investigation.md)
- [ECG readout-gap paper investigation](experiment-016-paper-investigation.md)
- [Experiment 016 second-seed stop report](experiment-016-encoder-motion-replication-v10-results.md)
- [Experiment 011 KDA/CKDA implementation](experiment-011-delta-memory.md)
- [Experiment 017 clean replication results](experiment-017-clean-replication-results.md)
- [First experiment protocol](experiment-001.md) and [results](experiment001-results.md)
- [Pretrained encoders](pretrained-notes.md), [JEPA](jepa-notes.md), and [released CPC](released-ecg-cpc.md)
- [Important papers](important-papers.md)
- [Cross-domain architecture candidates](cross-domain-architecture-candidates.md)
- [Astra proposals](astra-next-model-ideas.md)

Historical protocols retain the commands and environment paths used at the time.
Use the current environment and command guides for new work. Reproduce old runs
from their recorded source revision or snapshot; new runs require fresh manifests.
