"""Pure contracts for canonical ECG signals, manifests, and label selections."""

from __future__ import annotations

from typing import Any

import numpy as np

from .waveforms import SAMPLE_RATE

SCHEMA_VERSION = 1
SIGNAL_SHAPE = (12, 5000)
IDENTIFIED_SOURCES = ("ptbxl", "mimic")


def validate_signal(signal: np.ndarray) -> None:
    """Require a finite float32 ``[12, 5000]`` waveform without constant leads."""
    if signal.dtype != np.float32 or signal.shape != SIGNAL_SHAPE or not np.isfinite(signal).all():
        raise ValueError("Expected finite float32 [12,5000] waveform")
    if np.any(np.ptp(signal, axis=1) == 0):
        raise ValueError("Full constant lead")


def validate_metadata(metadata: dict[str, Any]) -> None:
    """Require a completed dataset with the canonical waveform metadata."""
    complete = (
        metadata.get("complete")
        and metadata.get("schema_version") == SCHEMA_VERSION
        and metadata.get("shape_per_record") == list(SIGNAL_SHAPE)
        and metadata.get("sampling_rate_hz") == SAMPLE_RATE
        and metadata.get("units") == "mV"
    )
    if not complete:
        raise ValueError("Invalid or incomplete canonical dataset")


def validate_training_rows(rows: list[dict[str, str]], record_count: int) -> None:
    """Require the declared count of unique training records."""
    if len(rows) != record_count:
        raise ValueError("Record count mismatch")
    if any(row["split"] != "train" for row in rows):
        raise ValueError("SSL split or label leakage")
    if len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate training record identity")


def select_labels(
    rows: list[dict[str, str]], labels: list[dict[str, str]]
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Select labeled PTB training rows and return their binary targets."""
    targets = {row["record_id"]: int(row["target"]) for row in labels}
    if len(targets) != len(labels) or not set(targets.values()) <= {0, 1}:
        raise ValueError("Invalid endpoint labels")

    selected = [row for row in rows if row["record_id"] in targets]
    if len(selected) != len(labels) or any(row["source"] != "ptbxl" for row in selected):
        raise ValueError("Labels must belong to training PTB rows")

    patients = {row["record_id"]: row["patient_id"] for row in selected}
    if any(patients[row["record_id"]] != row["patient_id"] for row in labels):
        raise ValueError("Label patient identity mismatch")
    return selected, targets


def identity_semantics_valid(row: dict[str, str]) -> bool:
    """Only PTB-XL and MIMIC have patient IDs; only PTB-XL has proxy labels."""
    known = row["source"] in IDENTIFIED_SOURCES
    if row["patient_identity_known"] != str(known).lower():
        return False
    if not known and row["patient_id"]:
        return False
    expected_scope = "ptbxl_proxy_available_separately" if row["source"] == "ptbxl" else "ssl_only"
    return row["label_scope"] == expected_scope
