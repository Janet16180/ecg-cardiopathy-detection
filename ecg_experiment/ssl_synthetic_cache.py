"""Prepare the prospectively selected real TRAIN subset in physical mV."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ecg_experiment.files import sha256_file, sha256_json
from ecg_experiment.resample import resample_full
from ecg_experiment.ssl_synthetic_checksum_audit import read_checksums, verify_partition
from ecg_experiment.ssl_synthetic_inputs import validate_header, verify_files
from ecg_experiment.ssl_synthetic_partition import canonical_identifier
from ecg_experiment.ssl_synthetic_readiness import checked_relative
from ecg_experiment.waveforms import LEADS, read_record

AUDIT_SUMMARY_SHA256 = "160bc03cd0d476f1679508635f99c2c2affe31bb0e565f1592adbd9dd504e2fa"


def select_records(records: list[dict[str, Any]], namespace: str, count: int) -> list[dict[str, Any]]:
    """
    Select patients before joining quality and choose their lowest numeric ECG ID.

    Parameters
    ----------
    records : list of dict
        Immutable original partition records, including all roles.
    namespace : str
        Frozen patient-hash namespace.
    count : int
        Exact prospectively selected patient count.

    Returns
    -------
    list of dict
        Original selected rows in patient digest order.
    """
    patients = {}
    for record in records:
        if record["role"] != "train":
            continue
        patient = canonical_identifier(record["patient_id"])
        if patient != record["patient_id"]:
            raise ValueError("Patient ID is not canonical")
        previous = patients.get(patient)
        if previous is None or int(record["ecg_id"]) < int(previous["ecg_id"]):
            patients[patient] = record
    if count < 1 or len(patients) < count:
        raise ValueError("Insufficient original training patients")
    ordered = sorted(
        patients, key=lambda p: (hashlib.sha256(f"{namespace}:{p}".encode()).hexdigest(), int(p))
    )
    return [dict(patients[patient]) for patient in ordered[:count]]


def join_eligibility(selected: list[dict[str, Any]], audited: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Attach audit observations only after the complete original subset is fixed.

    Parameters
    ----------
    selected : list of dict
        Ordered original subset without quality filtering.
    audited : list of dict
        Complete verified input-audit rows.

    Returns
    -------
    list of dict
        Selected rows retaining quality skips without replacement.
    """
    by_id = {row["ecg_id"]: row for row in audited}
    output = []
    for original in selected:
        if original["role"] != "train":
            raise ValueError("Only TRAIN records can enter the cache selection")
        row = by_id[original["ecg_id"]]
        if any(row[key] != value for key, value in original.items()) or row["status"] != "decoded":
            raise ValueError("Selected input audit is inconsistent or failed")
        eligible = row["train_optimization_eligible"]
        if not isinstance(eligible, bool) or eligible != (not row["quality_exclusions"]):
            raise ValueError("Inconsistent training quality eligibility")
        output.append(
            dict(
                original,
                status="pending" if eligible else "quality_skipped",
                cache_index=None,
                quality_exclusions=row["quality_exclusions"],
                review_flags=row["review_flags"],
                audit_status=row["status"],
                audit_train_optimization_eligible=eligible,
                audit_payload_sha256=row["payload_sha256"],
                payload_sha256={},
                verified_bytes=0,
            )
        )
    return output


def load_audited_inputs(
    root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    """
    Bind the completed input audit and replay the immutable partition metadata.

    Parameters
    ----------
    root : Path
        Repository root, preserving symlink paths.

    Returns
    -------
    tuple
        Original partition, audited rows, official checksums and complete audit bindings.
    """
    directory = root / "outputs/ssl_synthetic_localization/input_audit_v1"
    summary_path = directory / "summary.json"
    if sha256_file(summary_path) != AUDIT_SUMMARY_SHA256:
        raise ValueError("Input-audit summary differs from the frozen cache input")
    summary = json.loads(summary_path.read_bytes())
    if not summary["input_integrity_gate_passed"]:
        raise ValueError("Input-audit integrity gate did not pass")
    input_hashes = {"outputs/ssl_synthetic_localization/input_audit_v1/summary.json": AUDIT_SUMMARY_SHA256}
    for name, expected in (summary["input_sha256"] | summary["source_sha256"]).items():
        if sha256_file(root / checked_relative(name)) != expected:
            raise ValueError("An input-audit binding changed")
    for name, expected in summary["output_sha256"].items():
        relative = "outputs/ssl_synthetic_localization/input_audit_v1/" + name
        if sha256_file(root / checked_relative(relative)) != expected:
            raise ValueError("An input-audit output changed")
        input_hashes[relative] = expected
    base_summary = json.loads(
        (root / "outputs/ssl_synthetic_localization/partition_summary.json").read_bytes()
    )
    manifest_bytes = (root / checked_relative(base_summary["manifest"])).read_bytes()
    manifest = json.loads(manifest_bytes)
    metadata = (root / checked_relative(manifest["metadata"])).read_bytes()
    verify_partition(manifest, base_summary, manifest_bytes, metadata)
    eligibility = json.loads((directory / "eligibility.json").read_bytes())
    audited = eligibility["records"]
    retained = [
        {key: row[key] for key in original}
        for row, original in zip(audited, manifest["records"], strict=True)
    ]
    if retained != manifest["records"] or eligibility["binding"]["input_sha256"] != summary["input_sha256"]:
        raise ValueError("Audit does not retain the original complete partition")
    checksums = read_checksums((root / "data/raw/ptb-xl/1.0.3/SHA256SUMS.txt").read_text())
    return (
        manifest,
        audited,
        checksums,
        {
            "input_sha256": input_hashes | summary["input_sha256"],
            "input_audit_source_sha256": summary["source_sha256"],
            "input_audit_gate_passed": True,
        },
    )


def prepare_signal(row: dict[str, Any], raw: Path, checksums: dict[str, str]) -> np.ndarray:
    """
    Verify a selected eligible TRAIN record before reading and whole-record resampling.

    Parameters
    ----------
    row : dict
        Selected eligible TRAIN row, updated with fresh observed hashes.
    raw : Path
        PTB-XL raw source directory.
    checksums : dict
        Verified official path-to-digest mapping.

    Returns
    -------
    np.ndarray
        Canonical float32 physical mV signal, shape ``(12, 2500)``.
    """
    if row["role"] != "train" or row["status"] != "pending" or row["quality_exclusions"]:
        raise ValueError("Only selected eligible TRAIN records may open payloads")
    name = str(checked_relative(row["filename_hr"]))
    if "\\" in name or not name.startswith("records500/"):
        raise ValueError("Invalid source path")
    verify_files(raw, name, checksums, row)
    if row["payload_sha256"] != row["audit_payload_sha256"]:
        raise ValueError("Fresh payload hashes differ from the original input audit")
    validate_header(raw / (name + ".hea"), name)
    signal = resample_full(read_record(raw, name), 1, 2)
    if signal.shape != (len(LEADS), 2500) or signal.dtype != np.float32 or not np.isfinite(signal).all():
        raise ValueError("Invalid physical cache signal geometry or values")
    return signal


def prepare_cache(
    selection: list[dict[str, Any]], raw: Path, checksums: dict[str, str], output: Path, minimum: int
) -> dict[str, Any]:
    """
    Stream selected eligible records into an exclusively created NumPy memmap.

    Parameters
    ----------
    selection : list of dict
        Mutable original selection with eligibility already joined.
    raw : Path
        Raw PTB-XL source directory.
    checksums : dict
        Bound official checksums.
    output : Path
        New destination directory, never overwritten.
    minimum : int
        Frozen minimum usable patient count.

    Returns
    -------
    dict
        Aggregate preparation counts and gate; partial caches never pass.
    """
    if output.exists() or output.is_symlink():
        raise FileExistsError("Real-cache output directory must be new")
    if any(row["role"] != "train" for row in selection):
        raise ValueError("Protected or nontraining record in cache selection")
    output.mkdir(parents=True, exist_ok=False)
    ready = [row for row in selection if row["status"] == "pending"]
    signals = np.lib.format.open_memmap(
        output / "signals.npy", mode="w+", dtype=np.float32, shape=(len(ready), len(LEADS), 2500)
    )
    for index, row in enumerate(ready):
        try:
            signals[index] = prepare_signal(row, raw, checksums)
        except (OSError, ValueError, EOFError) as error:
            row.update(
                status="integrity_failed", failure_type=type(error).__name__, failure_detail=str(error)
            )
            continue
        row.update(status="cached", cache_index=index)
    signals.flush()
    del signals
    cached = sum(row["status"] == "cached" for row in selection)
    failures = sum(row["status"] == "integrity_failed" for row in selection)
    skipped = [row for row in selection if row["status"] == "quality_skipped"]
    return {
        "selected_patients": len(selection),
        "usable_patients": cached,
        "quality_skipped_patients": len(skipped),
        "quality_skip_reasons": dict(
            Counter(reason for row in skipped for reason in row["quality_exclusions"])
        ),
        "integrity_failed_patients": failures,
        "allocated_shape": [len(ready), len(LEADS), 2500],
        "dtype": "float32",
        "sample_rate_hz": 250,
        "units": "mV",
        "lead_order": list(LEADS),
        "physical_scale_applied": False,
        "minimum_usable_patients": minimum,
        "cache_gate_passed": failures == 0 and cached >= minimum,
        "verified_bytes": sum(row["verified_bytes"] for row in selection),
        "training_admitted": False,
        "training_started": False,
        "performance_scored": False,
        "validation_waveforms_opened": False,
        "protected_waveforms_opened": False,
    }


def selection_identity(selected: list[dict[str, Any]]) -> str:
    """
    Hash the ordered original subset before adding eligibility or observations.

    Parameters
    ----------
    selected : list of dict
        Original selected partition rows.

    Returns
    -------
    str
        Canonical ordered selection SHA-256 digest.
    """
    return sha256_json(selected)
