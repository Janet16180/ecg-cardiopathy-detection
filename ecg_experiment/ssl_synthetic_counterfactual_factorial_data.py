"""Explicit frozen-TRAIN/new-validation adapters for the localization-loss factorial."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from ecg_experiment.files import sha256_json
from ecg_experiment.ssl_synthetic_paired_background_data import prepare_background
from ecg_experiment.ssl_synthetic_paired_background_data import render_case as train_render
from ecg_experiment.ssl_synthetic_real_background import (
    _edit_parameters,
    _lowest,
    _nuisance_parameters,
)
from ecg_experiment.ssl_synthetic_real_background import cases_for_background as templates
from ecg_experiment.ssl_synthetic_real_background import render_case as native_render

STUDY_ID = "ssl_synthetic_counterfactual_factorial_v1"


def select_validation(
    records: list[dict], audited: list[dict], prior_selections: list[dict], ssl_selection: dict, config: dict
) -> dict:
    """Choose new validation patients without quality filtering after excluding every prior subject."""
    spent = {
        row["patient_id"]
        for selection in prior_selections
        for row in selection["records"]
        if row["role"] == "validation"
    }
    train = {row["patient_id"] for row in prior_selections[-1]["records"] if row["study_role"] == "train"}
    ssl = {row["patient_id"] for row in ssl_selection["records"]}
    if len(spent) != 384 or len(train) != 500 or len(ssl) != 2048 or train & ssl:
        raise ValueError("Frozen predecessor patient identities changed")
    eligible = _lowest(
        [
            row
            for row in records
            if row["role"] == "validation" and row["patient_id"] not in spent | train | ssl
        ]
    )
    eligible.sort(
        key=lambda row: (
            hashlib.sha256(
                f"{config['validation_selection_namespace']}:{row['patient_id']}".encode()
            ).hexdigest(),
            int(row["patient_id"]),
        )
    )
    if len(eligible) < 192:
        raise ValueError("Insufficient remaining original validation patients")
    audit = {row["ecg_id"]: row for row in audited}
    output = []
    for local, source in enumerate(eligible[:192]):
        observed = audit[source["ecg_id"]]
        if observed["status"] != "decoded" or any(observed[key] != value for key, value in source.items()):
            raise ValueError("New validation source/audit identity changed")
        output.append(
            dict(
                source,
                study_role="calibration" if local < 64 else "evaluation",
                cache_index=500 + local,
                subject_index=500 + local,
                background_row=local,
                role_subject_index=local if local < 64 else local - 64,
                audit_payload_sha256=observed["payload_sha256"],
                quality_exclusions=observed["quality_exclusions"],
                review_flags=observed["review_flags"],
            )
        )
    if len({row["patient_id"] for row in output}) != 192:
        raise ValueError("New validation patient duplication")
    return {
        "records": output,
        "selection_identity_sha256": sha256_json(output),
        "excluded_prior_validation_patients": 384,
        "reused_train_patients": 500,
        "waveforms_opened_when_written": 0,
    }


def new_case_row(index: int) -> int:
    """Map only new global case2000..2767 to its local mask/map row0..767."""
    if not isinstance(index, (int, np.integer)) or not 2000 <= index < 2768:
        raise ValueError("Not a new validation global case")
    return int(index) - 2000


def background_row(subject: int) -> int:
    """Map only virtual validation subject500..691 to new background0..191."""
    if not isinstance(subject, (int, np.integer)) or not 500 <= subject < 692:
        raise ValueError("Not a new validation virtual subject")
    return int(subject) - 500


def cases_for_background(row: dict, anchors: list[int], digest: str, config: dict) -> list[dict]:
    """Adapt only the new validation namespace while preserving frozen native rendering math."""
    local = background_row(row["cache_index"])
    expected = "calibration" if local < 64 else "evaluation"
    if row["role"] != "validation" or row["study_role"] != expected or row["background_row"] != local:
        raise ValueError("Protected or inconsistent new validation mapping")
    cases = templates(row, anchors, digest, config)
    seed = int.from_bytes(
        hashlib.sha256(f"{config['case_seed_namespace']}:{expected}:{row['patient_id']}".encode()).digest()[
            :8
        ],
        "big",
    )
    edit, nuisance = _edit_parameters(seed, anchors, config), _nuisance_parameters(seed)[0]
    for case in cases:
        case.update(
            study_id=STUDY_ID,
            source_partition_role="validation",
            seed=seed,
            edit_parameters=edit,
            nuisance_parameters=nuisance,
            nuisance=bool(seed & 1) if case["target"] else case["variant"] == "nuisance",
            background_row=local,
            mask_row=new_case_row(case["case_index"]),
        )
        case["matched_control_case_index"] = 4 * row["cache_index"] + (3 if case["nuisance"] else 2)
    return cases


def render_case(signal: np.ndarray, case: dict) -> dict[str, Any]:
    """Dispatch frozen TRAIN or new validation rendering without changing either case identity."""
    if 0 <= case["case_index"] < 2000:
        if case["role"] != "train" or case["study_id"] != "ssl_synthetic_paired_background_v1":
            raise ValueError("Original TRAIN descriptor changed")
        return train_render(signal, case)
    if case["study_id"] != STUDY_ID or case["mask_row"] != new_case_row(case["case_index"]):
        raise ValueError("New validation descriptor changed")
    if case["background_row"] != background_row(case["cache_index"]):
        raise ValueError("New background row mapping changed")
    expected = "calibration" if case["mask_row"] < 256 else "evaluation"
    if case["role"] != expected:
        raise ValueError("New validation role mapping changed")
    return native_render(signal, case)


def decode_background(row: dict, checksums: dict[str, str]) -> tuple[np.ndarray, dict]:
    """Reject anything outside the new frozen validation mapping before raw-path/file access."""
    local = background_row(row["cache_index"])
    if row["role"] != "validation" or row["background_row"] != local:
        raise ValueError("Only new validation payloads may be decoded")
    expected = "calibration" if local < 64 else "evaluation"
    if row["study_role"] != expected:
        raise ValueError("New validation split mapping changed")
    return prepare_background(row, checksums)
