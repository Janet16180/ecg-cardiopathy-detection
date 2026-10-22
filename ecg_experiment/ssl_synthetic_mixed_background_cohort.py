"""Declare a new supervised PTB training cohort without reading signal payloads."""

import hashlib
import json
from pathlib import Path
from typing import Any

from ecg_experiment import ssl_synthetic_partition as partition
from ecg_experiment.files import sha256_file
from ecg_experiment.paths import to_stored

CONFIG = "configs/ssl_synthetic_mixed_background_v1.json"
SOURCES = (
    CONFIG,
    "configs/ssl_synthetic_partition_v1.json",
    "docs/ssl-synthetic-localization-mixed-background-v1.md",
    "ecg_experiment/ssl_synthetic_mixed_background_cohort.py",
    "scripts/data/prepare_ssl_synthetic_mixed_background_cohort.py",
    "tests/test_ssl_synthetic_mixed_background_cohort.py",
    "ecg_experiment/ssl_synthetic_partition.py",
    "ecg_experiment/files.py",
    "ecg_experiment/paths.py",
    "ecg_experiment/__init__.py",
)


def select_records(
    records: list[dict[str, Any]], namespace: str, count: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Choose fixed patients after patient-wide fold exclusions and original role checks.

    Parameters
    ----------
    records : list[dict[str, Any]]
        Original four-field metadata with frozen patient roles added.
    namespace : str
        Prospective patient hash namespace.
    count : int
        Fixed cohort size, never reduced after waveform access.

    Returns
    -------
    tuple[list[dict[str, Any]], list[dict[str, Any]]
        Ordered selected records and the complete ordered eligible candidate list.
    """
    excluded = {r["patient_id"] for r in records if r["strat_fold"] in (8, 9, 10)}
    candidates = {}
    for row in records:
        if row["role"] != "train" or row["patient_id"] in excluded:
            continue
        patient = row["patient_id"]
        if patient not in candidates or int(row["ecg_id"]) < int(candidates[patient]["ecg_id"]):
            candidates[patient] = row
    ordered = sorted(
        candidates.values(),
        key=lambda r: (
            hashlib.sha256(f"{namespace}:{r['patient_id']}".encode()).hexdigest(),
            int(r["patient_id"]),
        ),
    )
    if count <= 0 or len(ordered) < count:
        raise ValueError("Insufficient eligible patients for the complete prospective cohort")
    return ordered[:count], ordered


def checksum_table(path: Path) -> dict[str, str]:
    """Read the public file-identity table without opening waveform or header files.

    Parameters
    ----------
    path : Path
        Official SHA256SUMS.txt file.

    Returns
    -------
    dict[str, str]
        Unique logical source paths and their declared SHA256 values.
    """
    result = {}
    for line in path.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.removeprefix("./")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest) or name in result:
            raise ValueError("Malformed or duplicated official checksum declaration")
        result[name] = digest
    return result


def prepare(root: Path, destination: Path) -> dict[str, Any]:
    """Save a complete metadata-only selection before any selected signal is decoded.

    Parameters
    ----------
    root : Path
        Isolated source checkout with permitted per-source raw aliases.
    destination : Path
        New baseline-owned local setup directory, never overwritten.

    Returns
    -------
    dict[str, Any]
        Aggregate metadata evidence; individual identities remain in local selection.json.
    """
    config = json.loads((root / CONFIG).read_text())
    source = root / config["raw_alias"]
    metadata, official = source / "ptbxl_database.csv", source / "SHA256SUMS.txt"
    if (
        sha256_file(metadata) != config["metadata_sha256"]
        or sha256_file(official) != config["official_table_sha256"]
    ):
        raise ValueError("Raw metadata or official declaration changed")
    checksums = checksum_table(official)
    if checksums["ptbxl_database.csv"] != config["metadata_sha256"]:
        raise ValueError("Official metadata identity differs")
    policy = json.loads((root / config["partition_policy"]).read_text())
    partition.validate_policy(policy)
    assigned = partition.assign_records(partition.read_metadata(metadata.read_bytes()), policy)
    selected, candidates = select_records(assigned, config["cohort"]["order_namespace"], 40)
    declarations = []
    for slot, row in enumerate(selected, start=40):
        name = str(row["filename_hr"])
        if not all((source / (name + suffix)).is_file() for suffix in (".hea", ".dat")):
            raise ValueError("Selected source header/data pair is missing; no substitution")
        declarations.append(
            {
                **row,
                "geometry_index": slot,
                "header_sha256": checksums[name + ".hea"],
                "signal_sha256": checksums[name + ".dat"],
            }
        )
    sources = {name: sha256_file(root / name) for name in SOURCES}
    cohort = {
        "study_id": config["study_id"],
        "scope": "new supervised PTB-only metadata selection; not a recovered v4 cohort",
        "interpreted_columns": list(partition.FIELDS),
        "metadata_sha256": config["metadata_sha256"],
        "official_table_sha256": config["official_table_sha256"],
        "partition_policy": policy,
        "partition_counts": partition.partition_counts(assigned),
        "cohort_policy": config["cohort"],
        "source_sha256": sources,
        "records": declarations,
        "eligible_candidates": candidates,
        "fold8_9_10_quarantined_patients": sorted(
            {r["patient_id"] for r in assigned if r["strat_fold"] >= 8}, key=int
        ),
        "selected_header_or_signal_decodes": 0,
        "clinical_labels_interpreted": 0,
        "v4_membership_verified": False,
    }
    destination.mkdir(parents=True, exist_ok=False)
    selection = destination / "selection.json"
    selection.write_text(json.dumps(cohort, indent=2) + "\n")
    summary = {
        "status": "PASS",
        "selection_path": to_stored(selection),
        "selection_sha256": sha256_file(selection),
        "metadata_sha256": config["metadata_sha256"],
        "eligible_distinct_train_patients": len(candidates),
        "selected_distinct_patients": len(declarations),
        "selected_waveform_decodes": 0,
        "clinical_labels_interpreted": 0,
        "source_sha256": sources,
    }
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
