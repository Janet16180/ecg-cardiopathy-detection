# Command guide

Run commands from the repository root using `python -m scripts.NAME` inside the
appropriate uv environment. Consult the command's protocol before acquisition
or experiment work; some legacy entry points encode a fixed protocol.

| Work | Entry points |
| --- | --- |
| Repository checks | `make check`, `scripts.verify_refactor` |
| Completed data validation | `make validate-data`, `scripts.validate_data_versioning` |
| MLflow register | `make register-runs`, `scripts.import_mlflow_history` |
| PTB acquisition | `download_ptbxl_metadata`, `download_ptbxl_waveforms` |
| Other public acquisition | `download_public_ecg`, `download_missing_ecg.sh` |
| MIMIC selection/acquisition | `prepare_mimic_ssl` |
| Preparation | `prepare_ptbxl`, `prepare_public_ecg`, `prepare_code15`, `build_training_dataset` |
| Quality and overlap | `audit_dataset_quality`, `audit_public_pool_overlap`, `audit_ecg_selection_bias` |
| Cached features | `extract_pretrained`, `extract_jepa`, `extract_ecg_cpc` |
| Scientific runs and reports | Use the selected protocol linked in the [queue](../docs/experiment-queue.md) |

Acquisition commands are resumable but may take substantial time. Check existing
processes and their receipt paths before starting another copy. Data validation
does not perform acquisition or training.

The `run_*`, `handoff_*`, and versioned profile modules belong to historical
experiments with frozen source hashes. Their names and paths are intentional
provenance. Follow the queue's pause and successor-manifest rules before using
them. New reusable logic belongs in `ecg_experiment/`; a script should provide
arguments and call that logic.
