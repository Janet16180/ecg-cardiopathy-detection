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
- [Seeded 100k-plus-labels training cohort](sampled-100k-plus-labels-v1.md)
- [Processed ECG exploration](processed-ecg-eda.md)
- [Challenge postprocessing](challenge-postprocessing.md)
- [Public-data strategy](public-data-strategy.md)

## Experiments and findings

Start with the [experiment queue](experiment-queue.md) and
[JSON catalog](experiment-queue.json). They link every active, deferred, completed,
and proposed experiment to its protocol and evidence. The 25 September CPC
local-readout study and 016 seed-47 xECG replication are complete on development
patients. Earlier 016 v10–v13 attempts stopped before the replication endpoint;
their evidence remains separate. No experiment queue is active; new studies
proceed one at a time under the queue's cost and verification gates.

- [Model findings](model-findings-report.md)
- [CPC improvement investigation](cpc-improvement-investigation.md), [ranked findings](cpc-improvement-research-2026-09-25.md), [local-readout protocol](cpc-local-readout-v1.md), and [development result](cpc-local-readout-v1-results.md)
- [Experiment 018 compact CPC data-scaling pilot](experiment-018-cpc-data-scaling.md), [cached v2 cost stop](experiment-018-cpc-data-scaling-v2.md), and [v3 training successor](experiment-018-cpc-data-scaling-v3.md)
- [Experiment 018 v3 training results](experiment-018-cpc-data-scaling-v3-results.md)
- [Experiment 018 frozen development readout](experiment-018-cpc-data-scaling-readout.md) and [results](experiment-018-cpc-data-scaling-readout-results.md)
- [ECG readout-gap paper investigation](experiment-016-paper-investigation.md)
- [Experiment 016 second-seed v10 stop](experiment-016-encoder-motion-replication-v10-results.md)
  and [v11 cost stop](experiment-016-encoder-motion-replication-v11-results.md)
- [Experiment 016 seed-47 v14 development replication](experiment-016-encoder-motion-replication-v14-results.md)
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
