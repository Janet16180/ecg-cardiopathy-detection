"""Write aggregate EDA for a seeded ECG training cohort."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from ecg_experiment.files import sha256_file, write_json_atomic

ROOT = Path(__file__).resolve().parents[2]
UNION = ROOT / "data/processed/training_union_500hz_v1/train_manifest.csv"
CHAPMAN = ROOT / "data/processed/challenge_ecg_views/chapman_strict_10s_v1/manifest.csv"


def flagged_counts(selected_ids: set[str], source: Path, id_column: str) -> Counter:
    """Count review flags for selected records in one verified source view."""
    rows = pd.read_csv(source, dtype=str, keep_default_na=False, usecols=[id_column, "qc_flags"])
    selected = rows.loc[rows[id_column].isin(selected_ids), "qc_flags"]
    return Counter(flag for flag in selected if flag)


def summarize(directory: Path) -> dict[str, object]:
    """Aggregate source, storage, label, patient and review-flag counts."""
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    rows = pd.read_csv(directory / "train_manifest.csv", dtype=str, keep_default_na=False)
    labels = pd.read_csv(directory / "labels_fraction1.csv", dtype=str, keep_default_na=False)
    selected_ids = set(rows["record_id"])
    flags = flagged_counts(selected_ids, UNION, "record_id")
    flags.update(flagged_counts(selected_ids, CHAPMAN, "ecg_id"))
    identified = rows.loc[rows["source"].isin(["ptbxl", "mimic"])]
    patient_counts = identified.groupby("source")["patient_id"].nunique().to_dict()

    if len(rows) != metadata["record_count"] or len(labels) != metadata["labeled_records"]:
        raise ValueError("Dataset count changed")
    if len(selected_ids) != len(rows) or labels["target"].nunique() != 2:
        raise ValueError("Identity or binary target changed")
    return {
        "status": "aggregate_eda_complete",
        "dataset_metadata_sha256": sha256_file(metadata_path),
        "dataset_manifest_sha256": sha256_file(directory / "train_manifest.csv"),
        "record_count": len(rows),
        "unlabeled_sample_count": metadata["unlabeled_target"],
        "labeled_count": len(labels),
        "label_counts": labels["target"].value_counts().to_dict(),
        "source_counts": rows["source"].value_counts().to_dict(),
        "backend_counts": rows["backend"].value_counts().to_dict(),
        "identified_source_patient_counts": patient_counts,
        "review_flag_counts_in_selected_shards": dict(flags),
        "review_flag_policy": "retained for review; not mapped to target or dropped",
        "unidentified_patient_sources": [
            "georgia", "cpsc_2018", "cpsc_2018_extra", "chapman_shaoxing"
        ],
        "source_sha256": sha256_file(ROOT / "scripts/reports/summarize_sampled_cohort.py"),
    }


def main() -> None:
    """Read the sampled manifest and save aggregate EDA."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.dataset_dir.resolve())
    write_json_atomic(args.output, result, sort_keys=True)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
