# ECG Cardiopathy Detection

Dataset origins, versions, download receipts, and acquisition commands are recorded in [docs/data-sources.md](docs/data-sources.md).
The current public-data quality scores, EDA, and read-only preparation policy are in [docs/data-quality-assessment.md](docs/data-quality-assessment.md).
Derived Challenge views and CODE native preparation are documented in [docs/challenge-postprocessing.md](docs/challenge-postprocessing.md) and [docs/processed-ecg-eda.md](docs/processed-ecg-eda.md).
The independent data-processing review, applied integrity fixes, and remaining scientific gates are in [docs/data-processing-astra-review.md](docs/data-processing-astra-review.md).
The versioned 500 Hz merged training dataset, its source exclusions, loader, and label rules are documented in [docs/training-dataset-v1.md](docs/training-dataset-v1.md).

Time series analysis of electrocardiogram (ECG) recordings for automatic detection of cardiopathies.

## Problem

The planned university dataset contains approximately 20,000 raw 12-lead ECG recordings, of which
about 1% have an expert cardiologist's text report. Access is pending from the partner university
in Italy. This makes
the task a semi-supervised / low-label-regime problem rather than a standard supervised
classification one: the main challenge is learning useful representations of the signal without
depending on labels, and then transferring those representations to a classifier trained on the
small labeled subset.

## First public-data experiment

Start with PTB-XL v1.0.3 and simulate limited annotation by exposing labels for approximately
10% of eligible training patients. The signals are real; only label availability is simulated.
The first target is a **diagnostic ECG abnormality proxy**, not a claim that a patient is healthy
or a validated referral decision. See [the experiment protocol](docs/experiment-001.md) for label
rules, model comparisons, sources, and known pretraining overlap.

The [complete model findings report](docs/model-findings-report.md) covers every evaluated model,
all 37 runs, interpretation, limitations, and the larger-data plan. A shorter set of measured
comparisons and figures is in [the results summary](docs/experiment001-results.md).
The [important papers reading guide](docs/important-papers.md) collects the core models,
recent architectures, ECG/report methods, dataset references, and evaluation literature.
The [experiment queue](docs/experiment-queue.md) and [JSON task catalog](docs/experiment-queue.json)
record execution order, authorized implementation tasks and live-status locations. Start there
after a context reset. The [independent architecture shortlist](docs/cross-domain-architecture-candidates.md)
investigates transferring NLP, genomic and vision model designs to raw ECGs.
The study includes supervised CNN/transformer references, compact masked-autoencoder and latent
prediction models, and released HuBERT-ECG, ECG-FM, and ECG-JEPA encoders. The additional
[custom architecture experiment](docs/custom-architecture.md) tests whether predicting differences
between lead representations improves on ordinary latent prediction.

Public annotation availability is separate from local annotation scarcity. A second experiment
uses **all eligible public training labels**; outputs are under `outputs/experiment002_public_labels/`.
The intended next stage is public-data training followed by adaptation and held-out evaluation
on the university cohort.
See [the public-data strategy](docs/public-data-strategy.md) for the additional Georgia SSL pilot,
label compatibility, duplicate checks, and limitations of external evaluation.

The next [MIMIC adaptation experiment](docs/experiment-003-mimic.md) caps acquisition at
**200,000 additional ECG recordings** (approximately 24–30 GB of raw files). It compares
continued self-supervision with matched update budgets and uses all eligible public training
labels downstream. Data preparation and running jobs are not completed evaluation results.

The [compact CPC experiment](docs/experiment-004-cpc.md) uses an audited subset of the
downloaded MIMIC records plus PTB-XL training ECGs. It compares CPC with the same small
network plus temporal agreement, using both 10% and all eligible labels. A separate
reference evaluates the authors' released HEEDB-pretrained S4 ECG-CPC checkpoint.
The [word2vec-style loss comparison](docs/experiment-005-word2vec.md) tests sampled
InfoNCE against SGNS with the same 16 negative draws. The [next CPC ideas](docs/cpc-next-ideas.md)
describe five proposed improvements and the controls needed to test them.
The [tokenization experiment](docs/experiment-006-tokenization.md) tests refreshed
cluster targets, heartbeat-aligned chunks, and learned chunk boundaries, with
continued-CPC and fixed-chunk controls. It uses the same audited ECG pool and
both label budgets.

The [released xECG fine-tuning experiment](docs/experiment-007-xecg.md) adapts
the authors' pretrained xLSTM ECG encoder to the same binary target, using
all training labels and the fixed 10% subset. Its execution follows the CPC suites.
The [vision-to-ECG research review](docs/vision-to-ecg-research.md) distinguishes
recent image-learning objectives from encoder architectures. The
[xECG local-feature retention experiment](docs/experiment-008-vision-ssl.md)
tests four matched continuation objectives, including temporal Gram anchoring
and retention weighted toward unusual visible patterns.

## Running the experiments

Use Python 3.11 and `uv`. The main project lock pins the PyTorch 2.6.0 CUDA 12.4
wheels used for the measured runs on the Tesla V100 16 GB. Create an isolated
environment from `pyproject.toml` and `uv.lock`:

```bash
UV_PROJECT_ENVIRONMENT=.venv-uv uv sync --locked --python 3.11
UV_PROJECT_ENVIRONMENT=.venv-uv uv run --locked python -m pytest -q tests/test_audit_ecg_selection_bias.py
```

This installs the development group, including pytest. Set the same
`UV_PROJECT_ENVIRONMENT` for subsequent `uv run` commands; use `--no-dev`
with `uv sync` for a run-only environment. `.venv-uv` keeps this setup separate
from the existing experiment environments. The existing `requirements-lock.txt` records the earlier
compact-model environment; the new `uv.lock` also includes h5py for CODE preparation
and pytest for tests. Existing running jobs should keep their current environment.
The pretrained encoder environment has separate source packages and version constraints;
follow [its setup instructions](docs/pretrained-notes.md) and
`requirements-pretrained-lock.txt` for those runs.

Prepare the data and run the compact models:

```bash
.venv/bin/python scripts/download_ptbxl_metadata.py
.venv/bin/python scripts/prepare_ptbxl.py
.venv/bin/python scripts/download_ptbxl_waveforms.py --archive --sampling-rate 100
.venv/bin/python scripts/download_ptbxl_waveforms.py --archive --sampling-rate 500
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/run_compact_suite.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/report_experiment.py
```

The archive is about 1.84 GB; extracted signals, environments, caches, and model checkpoints
need additional disk space. Downloads validate the official file checksums. Raw data and
generated manifests remain under the git-ignored `data/` directory. Per-run configs, checkpoints,
histories, test probabilities, and metrics are under `outputs/`.

See [pretrained encoder instructions](docs/pretrained-notes.md) and
[published ECG-JEPA instructions](docs/jepa-notes.md) for official source packages, weights,
preprocessing, feature extraction, and provenance. Public downloads used here require no
Hugging Face login. Each frozen probe must use its own labeled training sample, even when
features were extracted for a larger union of records.

Run the full-public-label CNN reference with:

```bash
.venv/bin/python scripts/prepare_ptbxl.py --label-fraction 1 --seed 42
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m ecg_experiment.run \
  --model cnn --manifest-dir data/processed/ptbxl/seed42_fraction1 \
  --output-dir outputs/experiment002_public_labels --device cuda
```

## Approach

Workflow:

1. Exploratory analysis of the ECG signals (sampling rate, length, noise, class balance in the
   labeled subset).
2. Preprocessing: filtering, normalization, segmentation into beats or fixed-length windows.
3. Compare released ECG self-supervised encoders, including ECG-FM (wav2vec 2.0), HuBERT-ECG,
   and ECG-JEPA; test further self-supervised adaptation on training signals only.
4. Fine-tuning / linear probing on the labeled subset.
5. Evaluation with metrics suited to imbalanced data (AUROC, AUPRC, per-class recall).

## Stack

- Python 3.10+
- PyTorch
- scikit-learn
- NumPy / pandas / SciPy
- matplotlib

## Interpretation

The evaluation target is an annotation proxy that excludes unresolved cases. Probabilities
calibrated on PTB-XL are not validated disease probabilities for students. The report records
annotation costs, ECG-level metrics with patient-cluster intervals, excluded cases, pretraining
exposure, and the difference between retrospective experiments and local screening validation.

## Data

The ECG dataset is not included in this repository. Place raw recordings under `data/raw/` locally.

## Team

- José Emiliano Luna López
- Andre Nicolai Gutierrez Bautista
- Clara Janet Rivera Medina

Sponsor: Gerardo Jesús Camacho González, TEC, Computer Science Department

## Project context

Master's project at Tecnológico de Monterrey. Status: accepted.

## License

Apache License 2.0. See [LICENSE](LICENSE).
