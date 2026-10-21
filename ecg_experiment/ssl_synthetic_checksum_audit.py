"""Audit declared exact signal duplicates without opening waveform payloads."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import PurePosixPath
from typing import Any

from ecg_experiment.ssl_synthetic_partition import (
    assign_records,
    partition_counts,
    read_metadata,
    validate_policy,
)


def verify_partition(
    manifest: dict[str, Any], summary: dict[str, Any], manifest_bytes: bytes, metadata_bytes: bytes
) -> None:
    """
    Require unchanged manifest bytes and exact replay from original source metadata.

    Parameters
    ----------
    manifest : dict
        Local record-level partition.
    summary : dict
        Previously written aggregate partition receipt.
    manifest_bytes : bytes
        Exact original bytes, before JSON parsing.
    metadata_bytes : bytes
        Original source metadata, interpreted only by the partition module.
    """
    validate_policy(manifest["policy"])
    if hashlib.sha256(manifest_bytes).hexdigest() != summary["manifest_sha256"]:
        raise ValueError("Partition hash differs from its original receipt")
    if any(summary[key] != value for key, value in manifest.items() if key != "records"):
        raise ValueError("Partition and original aggregate receipt disagree")
    if hashlib.sha256(metadata_bytes).hexdigest() != manifest["metadata_sha256"]:
        raise ValueError("Source metadata differs from the partition input")
    expected = assign_records(read_metadata(metadata_bytes), manifest["policy"])
    if expected != manifest["records"] or partition_counts(expected) != manifest["counts"]:
        raise ValueError("Partition assignments do not reproduce from original metadata")


def read_checksums(text: str) -> dict[str, str]:
    """
    Parse unique, relative file entries in an official SHA-256 checksum table.

    Parameters
    ----------
    text : str
        Entire source checksum table, containing metadata only.

    Returns
    -------
    dict
        Relative POSIX file path to normalized SHA-256 digest.
    """
    entries = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        path = PurePosixPath(name.removeprefix("*"))
        if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
            raise ValueError("Invalid SHA-256 digest")
        if path.is_absolute() or ".." in path.parts or str(path) == "." or "\\" in str(path):
            raise ValueError("Checksum entries must use relative file paths")
        if str(path) in entries:
            raise ValueError("Duplicate checksum entry")
        entries[str(path)] = digest.lower()
    return entries


def audit_duplicates(records: list[dict[str, Any]], checksums: dict[str, str]) -> dict[str, Any]:
    """
    Group declared encoded-signal digests and report aggregate duplication only.

    Parameters
    ----------
    records : list of dict
        Verified original partition records, including excluded folds.
    checksums : dict
        Official relative file names and SHA-256 digests.

    Returns
    -------
    dict
        Gate decision and aggregate duplicate groups, without record/patient identifiers.
    """
    grouped = defaultdict(list)
    for record in records:
        name = record["filename_hr"] + ".dat"
        if name not in checksums:
            raise ValueError("Official checksum missing for a partition signal file")
        grouped[checksums[name]].append(record)
    repeated = [group for group in grouped.values() if len(group) > 1]
    cross_patient = [group for group in repeated if len({r["patient_id"] for r in group}) > 1]
    cross_role = [group for group in repeated if len({r["role"] for r in group}) > 1]
    role_pairs = Counter(
        "|".join(pair) for group in cross_role for pair in combinations(sorted({r["role"] for r in group}), 2)
    )
    return {
        "records_checked": len(records),
        "unique_signal_digests": len(grouped),
        "duplicate_groups": len(repeated),
        "records_in_duplicate_groups": sum(map(len, repeated)),
        "cross_patient_duplicate_groups": len(cross_patient),
        "records_in_cross_patient_duplicate_groups": sum(map(len, cross_patient)),
        "cross_role_duplicate_groups": len(cross_role),
        "cross_role_group_counts_by_role_pair": dict(sorted(role_pairs.items())),
        "duplicate_gate_passed": not cross_patient and not cross_role,
        "training_admitted": False,
        "performance_scored": False,
        "training_started": False,
        "waveform_payload_opened": False,
        "scope": "Exact encoded signal digests declared by official metadata; no fresh payload verification",
        "not_verified": [
            "fresh local waveform integrity, decoding and signal quality",
            "near duplicates and identical physical signals with different encoding",
            "clinical labels, external checkpoint exposure and resource/generator admission",
        ],
    }
