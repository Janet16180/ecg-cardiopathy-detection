# ECG Cardiopathy Detection

Research on representation learning from raw twelve-lead ECGs when few recordings
have expert annotations. Public-data experiments use a diagnostic ECG annotation
proxy; this does not establish that a person is healthy or validate referral decisions.

**Experiments are paused for repository maintenance.** Do not resume training or
scheduling until the user requests it. Downloads can continue. Start with the
[experiment queue](docs/experiment-queue.md) for current research status and recovery.

## Get started

The verified environment uses **Python 3.11**, `uv`, and the pinned PyTorch 2.6 / CUDA
12.4 stack. Linux x86-64 is the research platform; tests run on CPU.
Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
make setup
make check
```

Use `make mlflow` to view the local experiment register, `make register-runs` to
import completed results, and `make validate-data` to verify the DVC dataset catalog.

Without Make:

```bash
export UV_PROJECT_ENVIRONMENT=.venv-uv
uv sync --locked --group data --group tracking
uv run --locked ruff check ecg_experiment scripts tests
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' \
  uv run --locked --group data --group tracking python -m pytest -q
```

`pyproject.toml` declares dependencies; `uv.lock` fixes their resolved versions.
`.venv-uv` keeps the new setup separate from historical environments used by
downloaders. Follow [the environment guide](environments/README.md) for pretrained
encoders and archived environment receipts.

## Shared workflow

| Need | Where to start |
| --- | --- |
| Contribute code and review changes | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Version completed local datasets with DVC | [Data versioning](docs/data-versioning.md) |
| Search and compare recorded runs in MLflow | [Experiment tracking](docs/experiment-tracking.md) |
| Understand the folder structure | [Repository organization](docs/repository-organization.md) |
| Find protocols, results, and data documentation | [Documentation index](docs/README.md) |
| Recover paused work | [Queue](docs/experiment-queue.md) and [machine-readable catalog](docs/experiment-queue.json) |

Git stores code, protocols, lockfiles, and DVC pointers. DVC identifies completed
datasets. MLflow indexes run parameters, aggregate metrics, and provenance.
Data and large generated artifacts remain outside Git. Local tracking and DVC
storage work without cloud accounts; team storage must be configured before
collaborators can share data or a tracking dashboard.
Training data stays local and is excluded from GitHub and Git LFS uploads.

## Project layout

```text
ecg_experiment/   Reusable data, model, evaluation, and tracking code
scripts/          Commands for acquisition, preparation, checks, and experiments
configs/          Shared configuration
tests/            CPU correctness, leakage, and recovery checks
data/             Local raw and derived datasets; Git tracks only DVC metadata/docs
notebooks/        Optional exploration
reports/          Reviewed aggregate reports for new work
docs/             Scientific protocols, results index, and working guides
environments/     Environment setup and immutable historical receipts
outputs/          Local experiment results, checkpoints, and verification evidence
```

The layout follows [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/using-the-template/)
with existing scientific paths retained for reproducibility. See the
[refactor record](docs/repository-refactor.md) for the migration and verification.

## Research context

The planned university cohort contains approximately 20,000 raw ECGs, with expert
reports for about 1%; access is pending. Current studies use public datasets and
simulate annotation scarcity with frozen patient splits and label subsets.
The shared public pool has 56,875 training recordings, with 15,360 full training
labels or a fixed 1,518-label subset. See [data sources](docs/data-sources.md) and
[the findings report](docs/model-findings-report.md).

This is a master's project at Tecnológico de Monterrey. Team: José Emiliano Luna
López, Andre Nicolai Gutierrez Bautista, and Clara Janet Rivera Medina.
Sponsor: Gerardo Jesús Camacho González, TEC, Computer Science Department.

Code is licensed under [Apache 2.0](LICENSE). Dataset access and reuse follow each
upstream source's terms.
