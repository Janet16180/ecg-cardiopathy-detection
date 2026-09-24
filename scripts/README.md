# Command guide

Run commands from the repository root with
`uv run --locked python -m scripts.<folder>.<name>` in the appropriate
environment. Use `uv run --locked python -m scripts.<name>` for the few
entrypoints retained at the `scripts/` root. Check the selected protocol and
the [experiment queue](../docs/experiment-queue.md) before any scientific run;
training and scheduling remain paused.

| Folder | Purpose | Example module |
| --- | --- | --- |
| `data/` | Download metadata, prepare sources, build datasets | `scripts.data.prepare_ptbxl` |
| `features/` | Extract and combine cached features | `scripts.features.extract_jepa` |
| `experiments/` | Train, probe, and evaluate models | `scripts.experiments.run_cpc_experiment` |
| `coordination/` | Queue and handoff commands | `scripts.coordination.run_priority_queue` |
| `reports/` | Analysis and report generation | `scripts.reports.eda_processed_ecg` |
| `validation/` | Audits and source checks | `scripts.validation.validate_data_versioning` |
| `tracking/` | MLflow indexing | `scripts.tracking.import_mlflow_history` |

Active download entrypoints remain at the root: `scripts.prepare_mimic_ssl`,
`scripts.download_public_ecg`, `scripts.download_ptbxl_waveforms`, and
`scripts.extract_pretrained`. The active `scripts/download_missing_ecg.sh`
driver remains there too. Use
`uv run --locked --group data dvc repro validate_completed_data` for completed
data checks.
Acquisition can take substantial time, so inspect live coordination files before
starting a copy. Data validation does not acquire data or start training.

Moving a command changes its source hash and module path. Existing checkpoints,
receipts, and executable manifests remain historical evidence; prepare a new
verified manifest before resuming work.
