"""Explicit role/seed adapters for the matched-control real-background study; frozen math is reused."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_json
from ecg_experiment.resample import resample_full
from ecg_experiment.ssl_synthetic_inputs import validate_header, verify_files
from ecg_experiment.ssl_synthetic_readiness import checked_relative
from ecg_experiment.ssl_synthetic_real_background import (
    _edit_parameters,
    _lowest,
    _nuisance_parameters,
)
from ecg_experiment.ssl_synthetic_real_background import cases_for_background as legacy_cases
from ecg_experiment.ssl_synthetic_real_background import render_case as legacy_render
from ecg_experiment.waveforms import read_record

STUDY_ID = "ssl_synthetic_paired_background_v1"
SEED_NAMESPACE = "ssl_paired_background_v1"
ROLES = ("train", "calibration", "evaluation")


def select_rows(
    records: list[dict[str, Any]], audited: list[dict[str, Any]], previous: dict[str, Any]
) -> dict[str, Any]:
    """
    Reuse all original TRAIN reservations and choose new unfiltered validation patients once.

    Parameters
    ----------
    records, audited : list of dict
        Verified immutable original partition and technical audit rows.
    previous : dict
        Frozen prior real-background selection, with all 500 untouched TRAIN reservations.

    Returns
    -------
    dict
        Exact original TRAIN rows plus new 692-row global mapping; no waveform is accessed.
    """
    reserved = [row for row in previous["records"] if row["study_role"] == "train_reserved"]
    spent = {row["patient_id"] for row in previous["records"] if row["study_role"] != "train_reserved"}
    if len(reserved) != 500 or len(spent) != 192:
        raise ValueError("Original 500/192 selection changed")
    audit = {row["ecg_id"]: row for row in audited}
    original = {row["ecg_id"]: row for row in records}
    for row in reserved:
        source = original[row["ecg_id"]]
        if (
            source["role"] != "train"
            or any(row[key] != value for key, value in source.items())
            or audit[row["ecg_id"]]["train_optimization_eligible"] is not True
        ):
            raise ValueError("An original TRAIN reservation changed; replacement is forbidden")
    remaining = _lowest(
        [row for row in records if row["role"] == "validation" and row["patient_id"] not in spent]
    )
    remaining.sort(
        key=lambda row: (
            hashlib.sha256(f"{SEED_NAMESPACE}:validation:{row['patient_id']}".encode()).hexdigest(),
            int(row["patient_id"]),
        )
    )
    if len(remaining) < 192:
        raise ValueError("Insufficient remaining original validation patients")
    output = []
    groups = (("train", reserved), ("calibration", remaining[:64]), ("evaluation", remaining[64:192]))
    for role, rows in groups:
        for local, row in enumerate(rows):
            index = len(output)
            observed = audit[row["ecg_id"]]
            output.append(
                dict(
                    row,
                    study_role=role,
                    cache_index=index,
                    subject_index=index,
                    role_subject_index=local,
                    source_reservation_row_sha256=(sha256_json(row) if role == "train" else None),
                    audit_payload_sha256=observed["payload_sha256"],
                    quality_exclusions=observed["quality_exclusions"],
                    review_flags=observed["review_flags"],
                )
            )
    patients = {row["patient_id"] for row in output}
    if len(patients) != 692 or patients & spent:
        raise ValueError("New patients overlap each other or the spent real-background cohort")
    return {
        "original_train_reservation_rows": reserved,
        "records": output,
        "excluded_prior_real_background_patients": len(spent),
        "original_train_reservation_sha256": sha256_json(reserved),
        "selection_identity_sha256": sha256_json(output),
        "waveforms_opened_when_written": 0,
    }


def _legacy_role(role: str) -> str:
    if role not in ROLES:
        raise ValueError("Unknown paired-study role")
    return "evaluation" if role == "evaluation" else "calibration"


def cases_for_background(
    row: dict[str, Any], anchors: list[int], background_hash: str, config: dict[str, Any]
) -> list[dict[str, Any]]:
    """
    Adapt frozen case templates to new TRAIN/calibration/evaluation identities and seeds.

    Parameters
    ----------
    row : dict
        Newly frozen global subject mapping, preserving the true original source role.
    anchors : list of int
        Frozen unedited-background algorithmic anchors.
    background_hash : str
        Cached float32 array-byte SHA-256.
    config : dict
        Unchanged authored waveform/anchor qualification configuration.

    Returns
    -------
    list of dict
        Four frozen variant templates; role adaptation is label-only and never reads raw data.
    """
    role = row["study_role"]
    expected_source = "train" if role == "train" else "validation"
    if row["role"] != expected_source:
        raise ValueError("Paired study/source partition roles disagree")
    legacy_role = _legacy_role(role)
    template_row = dict(row, role="validation", study_role=legacy_role)
    cases = legacy_cases(template_row, anchors, background_hash, config)
    seed = int.from_bytes(
        hashlib.sha256(f"{SEED_NAMESPACE}:{role}:{row['patient_id']}".encode()).digest()[:8], "big"
    )
    edit = _edit_parameters(seed, anchors, config)
    nuisance = _nuisance_parameters(seed)[0]
    for case in cases:
        case.update(
            study_id=STUDY_ID,
            role=role,
            source_partition_role=row["role"],
            seed=seed,
            edit_parameters=edit,
            nuisance_parameters=nuisance,
            nuisance=bool(seed & 1) if case["target"] else case["variant"] == "nuisance",
        )
        case["matched_control_case_index"] = 4 * row["cache_index"] + (3 if case["nuisance"] else 2)
    return cases


def render_case(background: np.ndarray, case: dict[str, Any]) -> dict[str, Any]:
    """
    Reuse the unchanged once-cast native renderer through an explicit label-only role adapter.

    Parameters
    ----------
    background : np.ndarray
        Corresponding qualified float32 (12,2500) mV cached background.
    case : dict
        Frozen paired-study parameters with true source and study role.

    Returns
    -------
    dict
        Identical frozen rendering API; only edited is a model input.
    """
    if case.get("study_id") != STUDY_ID:
        raise ValueError("Not a frozen paired-study case")
    role = _legacy_role(case["role"])
    return legacy_render(background, dict(case, role=role))


def prepare_background(row: dict[str, Any], checksums: dict[str, str]) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Freshly verify one newly selected authorized raw record, checking source/global roles first.

    Parameters
    ----------
    row : dict
        Frozen selected row with original partition role and global cache index.
    checksums : dict
        Source-bound official SHA-256 table.

    Returns
    -------
    tuple
        Native physical float32 waveform and fresh integrity observations.
    """
    role, index = row["study_role"], row["cache_index"]
    bounds = {
        "train": ("train", 0, 500),
        "calibration": ("validation", 500, 564),
        "evaluation": ("validation", 564, 692),
    }
    if (
        role not in bounds
        or not isinstance(index, int)
        or row["role"] != bounds[role][0]
        or not bounds[role][1] <= index < bounds[role][2]
    ):
        raise ValueError("Protected/unselected source or global study role")
    name = str(checked_relative(row["filename_hr"]))
    if not name.startswith("records500/") or "\\" in name:
        raise ValueError("Invalid selected source path")
    raw = ROOT / "data/raw/ptb-xl/1.0.3"
    observed = {"payload_sha256": {}, "verified_bytes": 0}
    verify_files(raw, name, checksums, observed)
    if observed["payload_sha256"] != row["audit_payload_sha256"]:
        raise ValueError("Fresh payload bytes differ from original technical audit")
    validate_header(raw / (name + ".hea"), name)
    signal = resample_full(read_record(raw, name), 1, 2)
    if signal.shape != (12, 2500) or signal.dtype != np.float32 or not np.isfinite(signal).all():
        raise ValueError("Invalid native selected background")
    return signal, observed
