"""Create a metadata-only patient partition without interpreting diagnostic labels."""

from __future__ import annotations

import csv
import hashlib
import io
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any

FIELDS = ("ecg_id", "patient_id", "strat_fold", "filename_hr")
ROLES = ("train", "validation", "engineering_test", "excluded")


def validate_policy(policy: dict[str, Any]) -> None:
    """
    Require the prospectively committed v1 allocation policy.

    Parameters
    ----------
    policy : dict
        Fixed version, source, folds, hash namespace, seed and bucket boundaries.

    Raises
    ------
    ValueError
        If any policy field differs from the frozen v1 policy.
    """
    expected = {
        "schema_version": 1,
        "partition_id": "ssl_synthetic_localization_v1",
        "dataset": "ptbxl_1.0.3",
        "eligible_folds": list(range(1, 9)),
        "excluded_folds": [9, 10],
        "hash_namespace": "ssl_synthetic_partition_v1",
        "seed": 20261018,
        "bucket_count": 10000,
        "train_stop": 8000,
        "validation_stop": 9000,
    }
    if policy != expected or any(type(policy[k]) is not type(v) for k, v in expected.items()):
        raise ValueError("Policy differs from the frozen v1 allocation")
    if any(type(fold) is not int for key in ("eligible_folds", "excluded_folds") for fold in policy[key]):
        raise ValueError("Fold identifiers must be integers")


def canonical_identifier(value: str) -> str:
    """
    Normalize a positive integral identifier without rounding.

    Parameters
    ----------
    value : str
        Integer or integral decimal text from source metadata.

    Returns
    -------
    str
        Canonical integer text, shared by values such as ``123`` and ``123.0``.
    """
    try:
        number = Decimal(value)
    except InvalidOperation as error:
        raise ValueError("Identifiers and folds must be positive finite integers") from error
    if not number.is_finite() or number <= 0 or number != number.to_integral_value():
        raise ValueError("Identifiers and folds must be positive finite integers")
    return str(int(number))


def read_metadata(payload: bytes) -> list[dict[str, str | int]]:
    """
    Parse only the four permitted identity and source-partition fields.

    Parameters
    ----------
    payload : bytes
        Original metadata CSV bytes, also hashed by the caller.

    Returns
    -------
    list of dict
        Normalized identifiers, folds and checked high-resolution recording paths.
    """
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))
    if reader.fieldnames is None or any(reader.fieldnames.count(field) != 1 for field in FIELDS):
        raise ValueError("Expected each permitted metadata column exactly once")
    rows = []
    seen = set()
    for raw in reader:
        ecg_id = canonical_identifier(raw["ecg_id"])
        patient_id = canonical_identifier(raw["patient_id"])
        fold = int(canonical_identifier(raw["strat_fold"]))
        path = PurePosixPath(raw["filename_hr"])
        expected_name = f"{int(ecg_id):05d}_hr"
        if len(path.parts) != 3 or path.parts[0] != "records500" or path.name != expected_name:
            raise ValueError("Unexpected high-resolution recording path")
        if path.parts[1] in {".", ".."} or "\\" in str(path):
            raise ValueError("Recording path cannot contain parent traversal or backslashes")
        if ecg_id in seen or fold not in range(1, 11):
            raise ValueError("Duplicate ECG identifier or invalid source fold")
        seen.add(ecg_id)
        rows.append(
            {"ecg_id": ecg_id, "patient_id": patient_id, "strat_fold": fold, "filename_hr": str(path)}
        )
    if not rows:
        raise ValueError("Metadata contains no records")
    return sorted(rows, key=lambda row: int(row["ecg_id"]))


def patient_role(patient_id: str, policy: dict[str, Any]) -> str:
    """
    Assign an eligible patient using the committed digest bucket rule.

    Parameters
    ----------
    patient_id : str
        Canonical patient identifier.
    policy : dict
        Validated v1 policy.

    Returns
    -------
    str
        Training, validation or engineering-test role.
    """
    key = f"{policy['hash_namespace']}:{policy['seed']}:{patient_id}"
    bucket = int(hashlib.sha256(key.encode()).hexdigest(), 16) % policy["bucket_count"]
    if bucket < policy["train_stop"]:
        return "train"
    if bucket < policy["validation_stop"]:
        return "validation"
    return "engineering_test"


def assign_records(rows: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Keep all repeat records together and quarantine mixed-fold patients.

    Parameters
    ----------
    rows : list of dict
        Normalized source metadata from ``read_metadata``.
    policy : dict
        Validated v1 allocation policy.

    Returns
    -------
    list of dict
        Source records with a patient-level role and background identity.
    """
    excluded = {row["patient_id"] for row in rows if row["strat_fold"] in policy["excluded_folds"]}
    roles = {row["patient_id"]: patient_role(row["patient_id"], policy) for row in rows}
    roles.update(dict.fromkeys(excluded, "excluded"))
    return [
        {**row, "role": roles[row["patient_id"]], "background_id": f"ptbxl:patient:{row['patient_id']}"}
        for row in rows
    ]


def partition_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Verify nonempty roles and patient separation, returning aggregates only.

    Parameters
    ----------
    records : list of dict
        Assigned source records.

    Returns
    -------
    dict
        Record/patient counts, source-fold counts and overlap assertions.
    """
    patients = {role: {r["patient_id"] for r in records if r["role"] == role} for role in ROLES}
    membership = defaultdict(set)
    for record in records:
        membership[record["patient_id"]].add(record["role"])
    overlap = sum(len(roles) > 1 for roles in membership.values())
    forbidden = sum(r["strat_fold"] in {9, 10} and r["role"] != "excluded" for r in records)
    if overlap or forbidden or any(not patients[role] for role in ROLES):
        raise ValueError("Patient partition has overlap, forbidden records or an empty role")
    counts = Counter(r["role"] for r in records)
    folds = Counter(r["strat_fold"] for r in records)
    return {
        "records": len(records),
        "patients": len(membership),
        "by_role": {role: {"records": counts[role], "patients": len(patients[role])} for role in ROLES},
        "records_by_source_fold": {str(fold): folds[fold] for fold in sorted(folds)},
        "patients_in_multiple_roles": overlap,
        "excluded_fold_records_in_eligible_roles": forbidden,
        "eligible_fold_records_quarantined": sum(
            r["strat_fold"] <= 8 and r["role"] == "excluded" for r in records
        ),
    }


def check_record_paths(records: list[dict[str, Any]], source_root: Path) -> int:
    """
    Check eligible header/data path existence without opening either payload.

    Parameters
    ----------
    records : list of dict
        Assigned source records.
    source_root : Path
        Directory containing the original metadata CSV and records500 folder.

    Returns
    -------
    int
        Number of eligible header/data pairs whose paths exist.
    """
    eligible = [record for record in records if record["role"] != "excluded"]
    missing = sum(
        not (source_root / (record["filename_hr"] + suffix)).is_file()
        for record in eligible
        for suffix in (".hea", ".dat")
    )
    if missing:
        raise ValueError(f"Missing {missing} eligible header/data paths; nothing is silently excluded")
    return len(eligible)


def build_partition(payload: bytes, policy: dict[str, Any], source_root: Path) -> dict[str, Any]:
    """
    Build a deterministic metadata-only manifest with explicit training gates.

    Parameters
    ----------
    payload : bytes
        Original PTB-XL metadata CSV.
    policy : dict
        Fixed v1 allocation policy.
    source_root : Path
        Metadata parent directory, used only for eligible path-existence checks.

    Returns
    -------
    dict
        Local record-level manifest; clinical labels and waveforms are never interpreted.
    """
    validate_policy(policy)
    records = assign_records(read_metadata(payload), policy)
    counts = partition_counts(records)
    checked = check_record_paths(records, source_root)
    return {
        "schema_version": 1,
        "partition_id": policy["partition_id"],
        "policy": policy,
        "metadata_sha256": hashlib.sha256(payload).hexdigest(),
        "interpreted_columns": list(FIELDS),
        "counts": counts,
        "eligible_file_pairs_present": checked,
        "records": records,
        "training_admitted": False,
        "performance_scored": False,
        "training_started": False,
        "not_verified": [
            "cross-patient waveform duplicates and waveform quality",
            "biological patient identity and clinical annotation correctness",
            "external checkpoint pretraining exposure",
            "generator qualification and actual training resource profile",
        ],
    }
