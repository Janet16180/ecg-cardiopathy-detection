# Local data versioning and validation

This repository uses DVC for completed, selected datasets. Tracked sources include the **PTB-XL 1.0.3** waveform directories (21,799 records at each of 500 Hz and 100 Hz) and four complete Challenge releases: Georgia (10,344 records), CPSC 2018 (6,877), CPSC-Extra (3,453), and Chapman/Shaoxing (10,247). The Challenge acquisition receipts report 6,906–20,688 verified waveform files per source. DVC pointers record their content. Tracking does not change the frozen PTB/MIMIC experiment cohort or add Challenge labels to training.

## What is tracked

| Item | Purpose |
| --- | --- |
| `data/raw/ptb-xl/1.0.3/*.dvc` | PTB 500 Hz and 100 Hz waveform directories, diagnostic metadata and official checksum manifest |
| `data/raw/challenge-20*/**/*.dvc` | Four completed Challenge waveform directories and official checksum manifests |
| `data/acquisition/*.json.dvc` | Four completed acquisition receipts; active CODE receipt excluded |
| `data/processed/ptbxl/seed42_fraction*/{all_train_ssl,labeled_train,validation,test}.csv.dvc` and `data/processed/training_union_500hz_v1/heldout_references.csv.dvc` | Exact frozen PTB cohort identities needed for split checks |
| `configs/datasets.json` | Human-readable source selection, receipt, official manifest and split audit paths |
| `dvc.yaml` | Read-only validation stage; no training or download stage |
| `reports/data-validation.json` | Validation counts after a successful run |

The official `SHA256SUMS.txt` and acquisition receipts stay in their original locations. The validator checks Challenge receipt completion and counts, hashes each official checksum manifest, verifies every selected waveform and header against upstream SHA256, rejects extra or missing files, and checks pairing. The official manifests also list directory `RECORDS` index files that the downloader did not fetch; counts here refer to waveform/header pairs. PTB-XL's official manifest is bound to its frozen build receipt by SHA256; both 100 Hz and 500 Hz recordings and the two metadata CSV files are checked against it. It also verifies that PTB training, development, calibration and test patient IDs do not overlap, and that the limited-label selection is a consistent subset of the full-label selection. Exact frozen split file hashes guard against changed or reduced cohorts. DVC itself tracks bytes and pipeline dependencies; it does not establish these properties.

MIMIC downloads and CODE-15% archives are outside DVC tracking while incomplete or active. The 18.4 GB processed training union and existing frozen experiment inputs are also outside these cache entries. Their builders, receipts and hash checks remain the source of truth until a deliberate future migration. Acquisition receipts establish download integrity, not fitness for the clinical endpoint or permission to pool labels. The PTB source ZIP is an unverified archive duplicate and is not tracked; the checked waveform directories and metadata are tracked directly.

## Commands

Install the optional data tools with `UV_PROJECT_ENVIRONMENT=.venv-uv uv sync --locked --group data`. Then, from the repository root:

```bash
UV_PROJECT_ENVIRONMENT=.venv-uv uv run --locked --group data dvc status
UV_PROJECT_ENVIRONMENT=.venv-uv uv run --locked --group data dvc repro validate_completed_data
UV_PROJECT_ENVIRONMENT=.venv-uv uv run --locked --group data python -m pytest -q tests/test_data_versioning.py
```

To track another **completed, stable** dataset, first verify its acquisition receipt and license, then add an entry to `configs/datasets.json` and the appropriate validation checks. Use `dvc add --no-relink PATH` only after confirming the directory is not being written by a downloader. Review the generated pointer and `.gitignore` entry. Never run `dvc add` over an active download path. Do not rerun `dvc init` on an initialized clone.

## Local storage and sharing

DVC has a local cache under `.dvc/cache`; the pointer and config are small Git metadata. No waveform data is uploaded to GitHub, Git LFS or DVC storage as part of this setup. The repository currently has **no shared DVC remote**, so another team member cannot retrieve these bytes with `dvc pull` from a fresh clone yet. Each team member can use their already downloaded verified source or a future approved shared remote. Configure a remote only after choosing a storage location, access controls, retention policy and dataset redistribution terms. Do not publish MIMIC or any data by default.

This host's ext filesystem does not support reflinks. DVC was configured to copy into its cache and `--no-relink` was used so the existing source files remain in place and their link type does not change. The 21 pointers cover 149,053 files and 7,675,037,745 content bytes (7.15 GiB); the local cache occupies about 7.5 GiB including filesystem overhead. Avoid `hardlink` or `symlink` for raw downloader destinations: writing those links could damage cached content. [DVC's cache documentation](https://dvc.org/doc/user-guide/data-management/large-dataset-optimization) explains link choices, and its [remote storage guide](https://dvc.org/doc/user-guide/data-management/remote-storage) explains sharing. Only small metadata can be reconstructed from Git alone until a remote exists.
