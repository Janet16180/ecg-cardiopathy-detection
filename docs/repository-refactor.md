# Repository refactor — 24 September 2026

The repository now has one documented team workflow, a buildable Python package,
locked uv environments, a local MLflow experiment index, and DVC metadata for
completed public data. This is repository maintenance; training remains paused.

Final verification: **169 CPU tests passed**, Ruff and both uv locks passed,
and both package builds succeeded. DVC verified **149,038 waveform/header files**
and patient isolation; its status is up to date. MLflow contains **78 historical
runs**. Git can see only **2,180 bytes** of metadata under `data/`; no dataset
content was uploaded. Full details are in the
[completion record](../reports/repository-refactor.json).

## Changes

- The root dependency definition is `pyproject.toml` with `uv.lock`. Development,
  DVC, and MLflow dependencies have named groups. The pretrained environment has
  a separate uv project because its dependency constraints differ.
- Root `requirements*.txt` files were removed after verbatim archival under
  `environments/archive/`. Their checksums and the earlier lock remain available.
- The structure follows Cookiecutter Data Science's top-level package, scripts,
  tests, configs, notebooks, reports, documentation, and local data directories.
  Historical file paths stay valid; no raw dataset or result tree was relocated.
- `Makefile`, contributor guidance, editor settings, local hooks, a pull request
  template, and GitHub Actions define the shared workflow. CI runs CPU checks
  and builds the package without requiring ECG datasets or a GPU.
- MLflow indexes existing aggregate results with source hashes and distinct
  development/test tags. Import is repeatable and does not rerun experiments.
  Future runs can opt in with explicit frozen code, data, and manifest hashes.
- DVC versions completed source bytes and frozen patient split manifests.
  Its validation stage checks upstream hashes, source completeness, labels,
  fixed cohorts, and patient separation. It contains no training stage.

## Preservation and verification

The pre-refactor archive is
`outputs/refactor_pause/source_before_refactor.tar.gz`, SHA-256
`4911551226c3df2d2297895aef65430132a54199ce0edaac63d97b47d2a2f8ed`.
Its `source_hashes.json` was retained unchanged. The new CPU command
`python -m scripts.verify_refactor` checks the archive and all 109 historical
scientific source and test files against that snapshot. New infrastructure is
implemented in new files. Documentation and dependency definitions changed.

The existing `.venv` and `.venv-pretrained` remain intact. Host process inspection
confirmed that MIMIC and CODE acquisition were active during maintenance; their
source files and destination paths were retained. Data, checkpoints, results,
and historical executable manifests remain in place.

Verification evidence is recorded separately in
`outputs/repository_refactor/verification.json` and the completion record in
`reports/repository-refactor.json`. A new maintenance-only executable manifest
under `outputs/repository_refactor/` freezes the current source map and verification
command; it is checked without starting a scheduler. It does not replace or
resume an experiment manifest.

Use `make check` for Ruff and the complete CPU test suite. Tests cover model and
checkpoint behavior, caching, evaluation, queue recovery, data integrity and
patient leakage, registry provenance, and archive preservation. `uv build
--no-sources` builds a wheel and source archive; explicit inclusion rules keep
local data and outputs out of both. These are software checks, not performance
results or a new GPU qualification.

## Team setup and remaining choices

Run `make setup`, then use [the data guide](data-versioning.md) and
[MLflow guide](experiment-tracking.md). No shared storage host was selected, so
DVC cache and the MLflow database remain local. The guides describe the next
configuration step when the team chooses a remote and tracking server.

The pretrained dependency lock and native wheel build are verified. Disposable
copies of the pinned source commits, constrained build dependencies, and
uv-managed Python headers produced wheels that passed imports outside the
repository. This does not test model inference. See
[environment setup](../environments/README.md) for the fresh-machine commands.

No commit or publication is required to use these local changes. Review and commit
the source, lockfiles, DVC pointers, validation reports, and documentation through
the normal team workflow. Data bytes and experiment output directories remain
ignored. Set up remote backups before relying on either local tool as shared storage.

When training is explicitly resumed, select an experiment from the queue, create
a new frozen executable manifest with its actual environment and inputs, and
perform the required real GPU/data-path profile before training. Existing results
retain their original source receipts.

## Design references

- [Cookiecutter Data Science workflow](https://cookiecutter-data-science.drivendata.org/using-the-template/)
- [uv environment configuration](https://docs.astral.sh/uv/concepts/projects/config/)
- [uv in GitHub Actions](https://docs.astral.sh/uv/guides/integration/github/)
- [MLflow tracking](https://mlflow.org/docs/latest/ml/tracking/)
- [DVC data management](https://dvc.org/doc/user-guide/data-management)
