# Working together

Use Python 3.11 and `uv` from the repository root. Setup and CPU check commands
are in the [README](README.md#get-started); uv uses `.venv` by default.
See the [environment guide](environments/README.md) before syncing an environment
used by an active downloader.

## A normal change

1. Pull the latest work and create a short branch, such as `data/validate-ptbxl`.
2. Agree who owns the affected files before editing them together.
3. Make one focused change. Add a test when behavior or a scientific assumption changes.
4. Run Ruff and CPU tests from the README. For data changes, also run the checks in
   [the data guide](docs/data-versioning.md).
5. Open a pull request and have another team member review it. Explain the
   problem, resulting behavior, validation, and any limitations.

Commit Python source, tests, `pyproject.toml`, `uv.lock`, DVC pointers, protocols,
and small aggregate reports. Keep raw data, local environments, model weights,
per-patient outputs, and tracking databases out of Git. A DVC pointer needs its
data in the team's approved remote before another collaborator can pull it.
The current storage policy is **local data only**: do not upload training data to
GitHub or Git LFS, or configure/push to a DVC remote until the user chooses one.

## Dependencies

Use `uv add PACKAGE` for runtime libraries and `uv add --group dev PACKAGE` for
development tools. Use groups `data` and `tracking` for DVC and MLflow tooling.
Review both `pyproject.toml` and `uv.lock`; use `uv sync --locked` in reproducible
setups. Scientific dependency upgrades require regression checks and a new run
identity. Git history preserves previous dependency definitions.

## Experiments

**Training and scheduling remain paused until the user requests a resume.**
Read [the queue](docs/experiment-queue.md), its JSON catalog, and the selected
protocol before experiment work. Editing the catalog does not schedule a run.

Before running a new experiment, freeze its hypothesis, data and patient splits,
label budget, seeds, controls, objective, selection rule, and resource budget.
Fit transformations on training patients only. Keep development, calibration,
and test patients separate. Compare models with matched initialization and
training budgets. Log failed and negative experiments as well as successful ones.
MLflow records are an index; source receipts and result files remain the evidence.

Changes to scientific code need new source hashes and a successor executable
manifest. Preserve previous receipts and results. Profile the actual data path
on the shared V100 before training, and use the GPU lock. Coordinate with legacy
runners as described in the queue. Never run GPU jobs in CI.

## Code and notebooks

Keep reusable Python in `ecg_experiment/`, command entry points in `scripts/`,
tests in `tests/`, experiment settings in `configs/`, and explanations in `docs/`.
Use `uv run python -m scripts.<area>.<command>` from the project root; see
the [command guide](scripts/README.md). Group commands by purpose. Classes should
encapsulate state and responsibilities; keep stateless operations as functions.
Scripts import shared logic from `ecg_experiment/`, never from other scripts.
Prefer flat control flow with guard clauses, let unexpected errors raise, and
give every function type hints and a NumPy-style docstring. Ruff enforces these
rules, import order, and a complexity limit; run `uv run ruff check ecg_experiment
scripts tests` before pushing. Preserve historical experiment
receipts; code changes need new manifests before execution. Use Git history for
maintenance changes instead of separate refactor journals.

Use notebooks for exploration with names such as `01-jr-signal-quality.ipynb`.
Clear outputs containing patient information and move reusable logic into Python.

Optional local hooks:

```bash
uv run --locked pre-commit install --hook-type pre-commit --hook-type pre-push
```

Hooks are local conveniences; CI provides the shared check. Hook installation
writes to `.git/hooks` and is performed by each collaborator on their own clone.
