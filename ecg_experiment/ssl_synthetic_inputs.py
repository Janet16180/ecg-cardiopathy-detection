"""Stream permitted waveform integrity and training-quality checks without opening protected roles."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import wfdb

from ecg_experiment.ecg_quality import assess
from ecg_experiment.files import sha256_file
from ecg_experiment.ssl_synthetic_readiness import checked_relative
from ecg_experiment.waveforms import LEADS, read_record

PERMITTED_ROLES = ("train", "validation")
PROTECTED_ROLES = ("engineering_test", "excluded")


def validate_header(path: Path, relative_name: str) -> None:
    """
    Validate metadata and forbid unverified WFDB signal redirection.

    Parameters
    ----------
    path : Path
        Freshly checksum-verified local header.
    relative_name : str
        Partition record name without an extension.
    """
    lines = [
        line for line in path.read_text().splitlines() if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(lines) != 13:
        raise ValueError("invalid_header_layout")
    first = lines[0].split()
    if len(first) < 4 or "/" in first[0]:
        raise ValueError("invalid_header_layout")
    expected_name = Path(relative_name).name
    if first[0] != expected_name or any(line.split()[0] != expected_name + ".dat" for line in lines[1:]):
        raise ValueError("unverified_payload_reference")
    fields = [line.split() for line in lines[1:]]
    if any(
        len(tokens) < 3 or "/" not in tokens[2] or tokens[2].rsplit("/", 1)[1] != "mV" for tokens in fields
    ):
        raise ValueError("invalid_units")
    header = wfdb.rdheader(str(path.with_suffix("")))
    if header.fs != 500 or header.n_sig != 12 or header.sig_len != 5000:
        raise ValueError("invalid_header_geometry")
    names = [name.upper() for name in header.sig_name]
    if len(names) != 12 or set(names) != {name.upper() for name in LEADS}:
        raise ValueError("invalid_leads")
    if header.units != ["mV"] * 12:
        raise ValueError("invalid_units")


def verify_files(raw: Path, name: str, checksums: dict[str, str], result: dict[str, Any]) -> None:
    """
    Hash both payload files freshly and require official byte equality.

    Parameters
    ----------
    raw : Path
        PTB-XL source directory.
    name : str
        Validated relative record stem.
    checksums : dict
        Official path-to-digest mapping.

    result : dict
        Mutable local row retaining observed hashes and verified byte counts even on failure.
    """
    for extension in (".hea", ".dat"):
        relative = name + extension
        if relative not in checksums:
            raise ValueError("missing_official_checksum")
        path = raw / relative
        actual = sha256_file(path)
        result["payload_sha256"][extension] = actual
        if actual != checksums[relative]:
            raise ValueError("checksum_mismatch")
        result["verified_bytes"] += path.stat().st_size


def audit_record(record: dict[str, Any], raw: Path, checksums: dict[str, str]) -> dict[str, Any]:
    """
    Audit one row after role filtering, retaining technical failures locally.

    Parameters
    ----------
    record : dict
        Original immutable partition row.
    raw : Path
        PTB-XL source directory.
    checksums : dict
        Official source checksums.

    Returns
    -------
    dict
        Original row plus integrity, quality and eligibility fields.
    """
    role = record["role"]
    if role not in (*PERMITTED_ROLES, *PROTECTED_ROLES):
        raise ValueError("Unknown partition role")
    result = dict(
        record,
        status="protected_unread",
        train_optimization_eligible=None,
        validation_retained=role == "validation",
        integrity_failures=[],
        quality_exclusions=[],
        review_flags=[],
        payload_sha256={},
        verified_bytes=0,
    )
    if role in PROTECTED_ROLES:
        return result
    result.update(status="integrity_failed", train_optimization_eligible=False if role == "train" else None)
    phase = "integrity"
    try:
        name = str(checked_relative(record["filename_hr"]))
        if "\\" in name or not name.startswith("records500/"):
            raise ValueError("invalid_source_path")
        verify_files(raw, name, checksums, result)
        phase = "header"
        validate_header(raw / (name + ".hea"), name)
        phase = "decode"
        signal = read_record(raw, name)
    except (OSError, ValueError, EOFError) as error:
        result["integrity_failures"] = [phase + ":" + type(error).__name__]
        result["failure_detail"] = str(error)
        return result
    exclusions, flags = assess(signal)
    result.update(status="decoded", quality_exclusions=exclusions, review_flags=flags)
    if role == "train":
        result["train_optimization_eligible"] = not exclusions
    return result


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Summarize all original roles without exposing identifiers.

    Parameters
    ----------
    records : list of dict
        Complete row-level audit output including protected roles.

    Returns
    -------
    dict
        Aggregate counts and input-only admission decision.
    """
    by_role = {}
    for role in (*PERMITTED_ROLES, *PROTECTED_ROLES):
        rows = [row for row in records if row["role"] == role]
        by_role[role] = {
            "records": len(rows),
            "patients": len({row["patient_id"] for row in rows}),
            "status_counts": dict(Counter(row["status"] for row in rows)),
            "integrity_failures": dict(Counter(x for row in rows for x in row["integrity_failures"])),
            "quality_exclusions": dict(Counter(x for row in rows for x in row["quality_exclusions"])),
            "review_flags": dict(Counter(x for row in rows for x in row["review_flags"])),
            "records_with_quality_exclusions": sum(bool(row["quality_exclusions"]) for row in rows),
            "records_with_review_flags": sum(bool(row["review_flags"]) for row in rows),
            "train_optimization_eligible": sum(row["train_optimization_eligible"] is True for row in rows),
            "train_optimization_eligible_patients": len(
                {row["patient_id"] for row in rows if row["train_optimization_eligible"] is True}
            ),
            "validation_retained": sum(row["validation_retained"] for row in rows),
        }
    permitted = [row for row in records if row["role"] in PERMITTED_ROLES]
    return {
        "records": len(records),
        "by_role": by_role,
        "permitted_records": len(permitted),
        "verified_bytes": sum(row["verified_bytes"] for row in permitted),
        "input_integrity_gate_passed": bool(permitted)
        and all(row["status"] == "decoded" for row in permitted),
        "training_admitted": False,
        "training_started": False,
        "performance_scored": False,
        "protected_waveforms_opened": False,
    }


def write_audit(records: list[dict[str, Any]], output: Path, binding: dict[str, Any]) -> dict[str, Any]:
    """
    Create local complete manifests in a new destination only.

    Parameters
    ----------
    records : list of dict
        Every audited original record.
    output : Path
        Already exclusively created directory, empty except progress output.
    binding : dict
        Source bindings retained in the derived eligibility manifest.

    Returns
    -------
    dict
        Aggregate counts plus output hashes.
    """
    eligibility = {
        "schema_version": 1,
        "binding": binding,
        "records": records,
        "base_partition_changed": False,
        "training_admitted": False,
    }
    path = output / "eligibility.json"
    with path.open("x") as stream:
        stream.write(json.dumps(eligibility, indent=2) + "\n")
    report = aggregate_records(records)
    report["output_sha256"] = {
        name: sha256_file(output / name) for name in ("records.jsonl", "eligibility.json")
    }
    return report
