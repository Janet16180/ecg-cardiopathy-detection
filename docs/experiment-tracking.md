# Experiment registry

The JSON results and frozen queue receipts under `outputs/` remain the scientific record. MLflow provides a searchable view of their aggregate results. Importing them does not run training, resume a queue, rewrite a receipt, or copy raw ECG data.

## Use locally

```bash
export UV_PROJECT_ENVIRONMENT=.venv-uv
uv sync --locked --group tracking --group data
uv run --locked --group tracking --group data python -m scripts.import_mlflow_history --dry-run
uv run --locked --group tracking --group data python -m scripts.import_mlflow_history
uv run --locked --group tracking --group data mlflow ui --backend-store-uri sqlite:///outputs/mlflow/mlflow.db
```

The default registry is `outputs/mlflow/mlflow.db` and is local to this workstation. It should stay out of Git. The importer records a relative source path, its SHA-256 digest, the result stage, aggregate metrics, and small parameters. From adjacent configuration files it reads an allowlist of label count, model size, training budget, optimizer, and explicit source hashes; it also records the configuration file SHA-256. It never uploads the raw configuration. It does not claim that the current checkout or Git revision produced a historical run. Runs from Experiments 001–009 are tagged `held_out_test`; the completed screens in 014, 015, and corrected 017 v2 are tagged `development_screen`. A development score cannot stand in for calibration or test performance. The importer does not infer clinical health from the diagnostic annotation proxy.

The importer is idempotent: a second run skips each source already registered with the same digest. If a source or configuration file changes, or a matching registry run is incomplete or failed, it stops with an actionable error so the conflict can be investigated. New allowlisted parameters may be added to earlier registry rows when their existing values agree; conflicting values are never overwritten. Available completion receipts are checked against `metrics.json` hashes. Files without a receipt are marked as such. By default it uploads no artifacts. `--log-aggregate-artifacts` uploads a newly generated JSON file containing only the selected numeric metrics and result stage; it never copies source JSON. Predictions, patient rows, waveforms, feature caches, and checkpoints are never uploaded by this command.

## Opt in future verified runs

Call `record_verified_result` after a new runner has written a receipt. It requires the frozen manifest, code, and data SHA-256 values from that execution; it never substitutes the current Git revision. Keep this call outside the scientific runner's critical path if a temporary registry outage must not affect training.

```python
from pathlib import Path
from ecg_experiment.tracking import record_verified_result

root = Path(".").resolve()
record_verified_result(
    root,
    root / "outputs/experiment018/seed42/completion.json",
    "sqlite:///outputs/mlflow/mlflow.db",  # or a team tracking URI
    name="018_seed42",
    stage="development_screen",
    metrics={"auroc": 0.91},
    params={"seed": 42, "objective": "bce", "model_parameters": 1041537},
    manifest_sha256=manifest_hash,
    code_sha256=code_hash,
    data_sha256=data_hash,
)
```

Valid stages are `development_screen`, `calibration`, and `held_out_test`; the helper prefixes metric names with the stage. A failed execution can be recorded using `stage="execution_failed"`, `status="FAILED"`, and `metrics={}` after writing its failure receipt. Calls are idempotent for completed runs and reject changed sources, supplied metrics, provenance hashes, or incomplete registry entries. The helper logs no artifacts and does not enable MLflow autologging. The frozen receipt and queue manifest remain the record of whether calibration and test were authorized.

## Use with a team server

For a shared registry, run an MLflow tracking server with a database backend and an artifact destination accessible to that server. Point each teammate's client at the same HTTPS endpoint, then run the importer once:

```bash
export UV_PROJECT_ENVIRONMENT=.venv-uv
export MLFLOW_TRACKING_URI=https://your-mlflow-host.example
uv run --locked --group tracking --group data python -m scripts.import_mlflow_history
```

The local SQLite file is suitable for one machine. A team server should use a managed PostgreSQL or MySQL backend and appropriate authentication, access controls, and backups. Store connection details outside Git. Keep the existing source receipts and queue documents as the authority for frozen execution. For future runners, record the label budget, seed, objective, model size, development metrics, and eventually calibration/test metrics only after their planned gates.

MLflow documents the distinction between [backend metadata and artifact storage](https://mlflow.org/docs/latest/tracking/backend-stores/), [tracking server configuration](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/), and the [tracking APIs](https://mlflow.org/docs/latest/ml/tracking/).
