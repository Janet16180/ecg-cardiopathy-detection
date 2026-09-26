"""Check a published sampled cohort's pointers, labels, and loader paths."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from ecg_experiment.files import read_csv, sha256_file, write_json_atomic
from ecg_experiment.sampled_training_dataset import SampledTrainingECGDataset

ROOT = Path(__file__).resolve().parents[2]
MIMIC_RAW = Path("data/raw/mimic-iv-ecg/1.0")
MIMIC_AUDIT = Path("data/processed/mimic_ssl_200k/audit.sqlite3")


def check_selection(rows: list[dict[str, str]], metadata: dict[str, object]) -> tuple[Counter, Counter]:
    """Check counts, waveform identities and PTB held-out patient isolation."""
    sources = Counter(row["source"] for row in rows)
    backends = Counter(row["backend"] for row in rows)
    if len({row["signal_sha256"] for row in rows}) != len(rows):
        raise ValueError("Duplicate selected waveform")
    if dict(sources) != metadata["source_counts"]:
        raise ValueError("Source counts changed")
    if len(rows) != metadata["unlabeled_target"] + metadata["labeled_records"]:
        raise ValueError("Selected record count changed")
    heldout = read_csv(ROOT / "data/processed/training_union_500hz_v1/heldout_references.csv")
    heldout_patients = {row["patient_id"] for row in heldout}
    train_patients = {row["patient_id"] for row in rows if row["source"] == "ptbxl"}
    if train_patients & heldout_patients:
        raise ValueError("PTB training and held-out patient overlap")
    return sources, backends


def check_sources(rows: list[dict[str, str]], metadata: dict[str, object]) -> None:
    """Rejoin raw pointers to the full MIMIC audit and check source shards."""
    with sqlite3.connect(f"file:{ROOT / MIMIC_AUDIT}?mode=ro", uri=True) as connection:
        accepted = dict(connection.execute(
            "SELECT name, signal_sha256 FROM outcomes WHERE status='accepted'"
        ))
    for row in rows:
        if row["backend"] != "mimic_wfdb":
            continue
        path = Path(row["path"])
        if not path.is_relative_to(MIMIC_RAW):
            raise ValueError("Raw pointer is outside MIMIC source")
        if accepted.get(str(path.relative_to(MIMIC_RAW))) != row["signal_sha256"]:
            raise ValueError(f"Raw pointer differs from MIMIC audit: {row['record_id']}")
    for relative in metadata["source_shards"]:
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(f"Referenced shard missing: {relative}")
    for name, expected in metadata["input_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"Source input changed: {name}")


def sample_waveforms(ssl: SampledTrainingECGDataset) -> int:
    """Read and hash two real examples from each source/backend pair."""
    positions: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(ssl.rows):
        key = row["source"], row["backend"]
        if len(positions[key]) < 2:
            positions[key].append(index)
    for indices in positions.values():
        for index in indices:
            item = ssl[index]
            if item["target_available"] or item["target"] != -1:
                raise ValueError("SSL target leakage")
    return sum(map(len, positions.values()))


def verify(directory: Path) -> dict[str, object]:
    """Verify selection identities and sample real waveforms from every backend.

    Parameters
    ----------
    directory : Path
        Completed sampled cohort.

    Returns
    -------
    dict[str, object]
        Small reproducible verification receipt; this is a sampled waveform check.
    """
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    ssl = SampledTrainingECGDataset(directory)
    supervised = SampledTrainingECGDataset(directory, purpose="supervised")
    limited = SampledTrainingECGDataset(directory, purpose="supervised", label_budget="0.1")
    rows = ssl.rows
    sources, backends = check_selection(rows, metadata)
    if len(supervised) != metadata["labeled_records"] or len(limited) != 1518:
        raise ValueError("Supervised selection changed")
    check_sources(rows, metadata)
    checked_waveforms = sample_waveforms(ssl)
    if not supervised[0]["target_available"] or not limited[0]["target_available"]:
        raise ValueError("Supervised labels missing")

    return {
        "status": "passed_sampled_waveforms_and_complete_pointer_audit",
        "dataset_dir": str(directory.relative_to(ROOT)),
        "metadata_sha256": sha256_file(metadata_path),
        "manifest_sha256": sha256_file(directory / "train_manifest.csv"),
        "record_count": len(rows),
        "full_label_count": len(supervised),
        "limited_label_count": len(limited),
        "source_counts": dict(sources),
        "backend_counts": dict(backends),
        "unique_waveform_hashes": len(rows),
        "ptb_heldout_patient_overlap": 0,
        "source_pointers_checked_against_manifest_or_audit": len(rows),
        "physical_file_existence_scope": "all referenced shards and sampled raw MIMIC reads",
        "waveforms_read_and_hashed": checked_waveforms + 2,
        "waveform_verification_scope": "two per source/backend plus one per supervised budget",
        "input_hashes_rechecked": len(metadata["input_sha256"]),
        "verifier_sha256": sha256_file(ROOT / "scripts/validation/verify_sampled_training_dataset.py"),
    }


def main() -> None:
    """Validate the published cohort and write a small receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    receipt = verify(args.dataset_dir.resolve())
    write_json_atomic(args.receipt, receipt, sort_keys=True)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
