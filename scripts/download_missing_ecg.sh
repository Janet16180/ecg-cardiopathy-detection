#!/usr/bin/env bash
# Sequential public-data acquisition; resumable and independent of experiment inputs.
set -euo pipefail

cd "$(dirname "$0")/.."
.venv/bin/python -u -m scripts.download_public_ecg georgia --workers 4
.venv/bin/python -u -m scripts.download_public_ecg cpsc_2018 --workers 4
.venv/bin/python -u -m scripts.download_public_ecg cpsc_2018_extra --workers 4
.venv/bin/python -u -m scripts.download_public_ecg chapman_shaoxing --workers 4
.venv/bin/python -u -m scripts.download_public_ecg code_15pct
