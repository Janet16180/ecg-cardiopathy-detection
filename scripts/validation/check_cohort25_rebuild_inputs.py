"""Inspect cohort-rebuild prerequisites without opening any patient data."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

CODE = (
    "scripts/data/build_training_dataset.py",
    "scripts/data/prepare_public_ecg.py",
    "scripts/data/build_clean_cohorts.py",
    "scripts/data/build_ningbo_clean.py",
    "scripts/data/build_challenge_splits.py",
    "scripts/data/build_clean_cohorts_v3.py",
    "scripts/data/build_clean_cohorts_v4.py",
    "ecg_experiment/clean_cohorts_v2.py",
    "ecg_experiment/clean_cohorts_v3.py",
    "ecg_experiment/challenge_splits.py",
    "ecg_experiment/ecg_quality.py",
)
GENERATED = (
    "data/processed/ptbxl/seed42_fraction1/all_train_ssl.csv",
    "data/processed/training_union_500hz_v1/metadata.json",
    "data/processed/training_union_500hz_v1/train_manifest.csv",
    "data/processed/training_union_500hz_v1/heldout_references.csv",
    "data/processed/challenge_ecg_views/chapman_strict_10s_v1/manifest.csv",
    "outputs/data_quality/clean_cohorts_v1/record_quality.csv.gz",
    "outputs/data_quality/ptbxl_reference_hashes_v2.json",
    "outputs/eda/features/challenge_headers.parquet",
    "outputs/eda/features/challenge_hashes.parquet",
    "data/processed/ningbo_clean_v1/rows.csv",
    "outputs/data_quality/ningbo_v1/exact_overlap_receipt.json",
    "data/processed/challenge_splits_v1/rows.csv",
    "data/processed/clean_25k_v3/train_manifest.csv",
    "data/processed/clean_25k_v4/train_manifest.csv",
)


def inspect(root: Path) -> dict[str, object]:
    """Return code provenance and file availability, without reading generated files.

    Parameters
    ----------
    root : Path
        Checkout containing the local data directories.

    Returns
    -------
    dict[str, object]
        Tracked code hashes and existence checks, never patient-level values.
    """
    tracked = set(subprocess.check_output(
        ["git", "-C", str(root), "ls-files"], text=True
    ).splitlines())
    code = [{"path": name, "tracked": name in tracked,
             "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()}
            for name in CODE]
    generated = [{"path": name, "exists": (root / name).is_file()}
                 for name in GENERATED]
    return {"kind": "availability_audit_not_experiment", "code": code,
            "generated_inputs": generated, "patient_data_opened": False,
            "training_or_scoring_performed": False}


def main() -> None:
    """Print an inventory for a requested checkout."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(inspect(args.root), indent=2))


if __name__ == "__main__":
    main()
