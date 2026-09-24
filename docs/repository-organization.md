# Repository organization

We adopt [Cookiecutter Data Science's organization](https://cookiecutter-data-science.drivendata.org/using-the-template/)
in this existing repository. The template's top-level Python package, task runner,
data separation, notebooks, reports, and review workflow fit this project.
Generating a new template over existing files would overwrite project decisions;
the existing scientific paths are retained because frozen manifests and active
downloaders refer to them.

| Path | Contents and owner responsibilities |
| --- | --- |
| `ecg_experiment/` | Importable, reusable implementation; test changed behavior |
| `scripts/` | Command modules; use explicit arguments and record run identities |
| `configs/` | Reviewed configuration; no secrets or personal absolute paths |
| `tests/` | Synthetic fixtures and CPU tests; no private datasets/network required |
| `data/raw/` | Source bytes and upstream checksums; preserve acquisition identity |
| `data/processed/` | Derived views and manifests; record source versions and fitting cohort |
| `data/acquisition/` | Acquisition progress and checksum receipts; completion must be verified |
| `notebooks/` | Named exploration; move shared logic into Python |
| `reports/` | Small reviewed aggregate findings and figures for new studies |
| `docs/` | Protocols, decisions, results index, and team guides |
| `environments/` | Separate environment definitions and archived receipts |
| `outputs/` | Local machine artifacts, checkpoints, logs, validation receipts, MLflow storage |

Existing `docs/` reports and `outputs/` result locations stay in place so citations
and frozen artifact hashes continue to resolve. Avoid copying large datasets
just to fit a folder name. New intermediate data may use `data/interim/` when a
pipeline actually needs it.

The Python package remains `ecg_experiment`; installing the project makes it
importable from notebooks and other directories. Run script modules from the
repository root because historical scripts resolve repository data paths.

## Tool responsibilities

**uv** manages Python dependencies and environments. **Git** reviews and versions
source and lightweight metadata. **DVC** versions dataset contents and can record
reproducible pipeline dependencies. **pytest** checks code and data-contract behavior.
**MLflow** indexes experiment parameters, aggregate metrics, and provenance.
These tools complement the existing frozen source maps and patient split controls.

Data checks and tests are separate from model performance evaluation. Passing
CI, a data validation stage, or a GPU profile is not a positive scientific result.

For three people, feature branches with a second reviewer, one shared lockfile,
and documented commands provide a manageable starting workflow. Configure a shared
DVC remote and one MLflow server when the team selects its storage host. Do not
copy SQLite databases between collaborators as a synchronization mechanism.
