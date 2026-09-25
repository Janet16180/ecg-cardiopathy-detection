"""Write an immutable, CPU-only canonical versus historical CPC input audit."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy

from ecg_experiment.cpc_input_audit import audit_inputs
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.staging import published_directory

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs/data_quality/clean_cpc_input_audit_v1"
CODE_PATHS = ("ecg_experiment/cpc_input_audit.py", "scripts/validation/audit_clean_cpc_inputs.py",
              "ecg_experiment/training_dataset.py", "ecg_experiment/waveforms.py",
              "ecg_experiment/public_sources.py", "scripts/data/prepare_cpc_data.py")


def code_hashes() -> dict[str, str]:
    """Record the exact source inspected for this audit."""
    return {name: sha256_file(ROOT / name) for name in CODE_PATHS}


def small_input_hashes(dataset: Path, cache: Path) -> dict[str, str]:
    """Hash metadata and tables that may be changed by concurrent sessions."""
    paths = (dataset / "metadata.json", dataset / "train_manifest.csv", dataset / "exclusions.csv",
             cache / "complete.json", cache / "rows.csv", cache / "ecg_ids.npy",
             ROOT / "outputs/data_quality/training_union_v1/receipt.json",
             ROOT / "data/processed/ptbxl/seed42_fraction1/all_train_ssl.csv",
             ROOT / "data/processed/mimic_ssl_40k_cpc/ssl_manifest.csv",
             ROOT / "data/processed/mimic_ssl_40k_cpc/audit.sqlite3")
    return {str(path.resolve()): sha256_file(path) for path in paths}


def main() -> None:
    """Run the sampled audit once, refusing to overwrite an existing receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "data/processed/training_union_500hz_v1")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/processed/cpc_pool_40k")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Audit output already exists: {output}")
    before = code_hashes()
    small_before = small_input_hashes(args.dataset_dir, args.cache_dir)
    result = audit_inputs(ROOT, args.dataset_dir, args.cache_dir)
    if code_hashes() != before or small_input_hashes(args.dataset_dir, args.cache_dir) != small_before:
        raise RuntimeError("Audit source or small input changed during execution")
    if result["summary"]["retained_bitwise_equal"] != 64 or result["summary"]["excluded_bitwise_equal"] != 66:
        raise RuntimeError("Historical cache differs from canonical resampling")
    result["code_sha256"] = before
    result["small_input_sha256"] = small_before
    result["created_at_utc"] = datetime.now(UTC).isoformat()
    result["versions"] = {"python": sys.version.split()[0], "numpy": np.__version__,
                          "scipy": scipy.__version__}
    result["command_scope"] = {"dataset_dir": str(args.dataset_dir.resolve()),
                               "cache_dir": str(args.cache_dir.resolve()),
                               "output_dir": str(output)}
    with published_directory(output) as stage:
        write_json_atomic(stage / "receipt.json", result)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
