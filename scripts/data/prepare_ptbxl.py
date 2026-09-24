#!/usr/bin/env python3
"""Prepare patient-disjoint PTB-XL manifests for a low-label proxy task."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ecg_experiment.files import sha256_file, write_csv_atomic, write_json_atomic

ROOT = Path(__file__).resolve().parents[2]
WAVEFORM_FIELDS = ("ecg_id", "patient_id", "filename_lr", "filename_hr")
LABELED_FIELDS = (*WAVEFORM_FIELDS, "target")
AUDIT_FIELDS = (*WAVEFORM_FIELDS, "strat_fold", "target", "reason", "scp_codes")
OFFICIAL_FOLDS = {str(number) for number in range(1, 11)}
LAST_TRAIN_FOLD = 8
VALIDATION_FOLD = "9"
TEST_FOLD = "10"


def read_diagnostic_codes(path: Path) -> dict[str, str]:
    """
    Map each diagnostic SCP statement to its diagnostic class.

    Parameters
    ----------
    path : Path
        ``scp_statements.csv``.

    Returns
    -------
    dict[str, str]
        Diagnostic class keyed by SCP code, for diagnostic statements only.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"", "diagnostic", "diagnostic_class"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing SCP columns: {sorted(required - set(reader.fieldnames or []))}")
        result = {}
        for row in reader:
            code = row[""].strip()
            category = row["diagnostic_class"].strip()
            if row["diagnostic"].strip() in ("1.0", "1") and code and category:
                result[code] = category
        return result


def classify(codes: set[str], diagnostic_codes: dict[str, str]) -> tuple[str, str]:
    """
    Assign the proxy target and its audit reason.

    Parameters
    ----------
    codes : set[str]
        SCP codes listed for the record.
    diagnostic_codes : dict[str, str]
        Diagnostic class keyed by SCP code.

    Returns
    -------
    tuple[str, str]
        ``"0"``, ``"1"`` or ``""`` (unresolved), and the reason.
    """
    if "NORM" in codes:
        if codes <= {"NORM", "SR"}:
            return "0", "norm_only_or_sinus_rhythm"
        return "", "norm_with_other_codes"
    abnormal = {diagnostic_codes[code] for code in codes & diagnostic_codes.keys()}
    if any(category != "NORM" for category in abnormal):
        return "1", "non_norm_diagnostic_class"
    return "", "no_non_norm_diagnostic_class"


def parse_codes(value: str, ecg_id: str) -> set[str]:
    """
    Parse the ``scp_codes`` dictionary literal of one record.

    Parameters
    ----------
    value : str
        Python dictionary literal from the metadata table.
    ecg_id : str
        Record identifier for error messages.

    Returns
    -------
    set[str]
        Listed SCP codes.

    Raises
    ------
    ValueError
        If the value is not a dictionary with string keys.
    """
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"Invalid scp_codes for ecg_id {ecg_id}") from exc
    if not isinstance(parsed, dict) or not all(isinstance(code, str) for code in parsed):
        raise ValueError(f"Expected SCP code dictionary for ecg_id {ecg_id}")
    # A listed SCP key is present even when its likelihood is zero (unknown).
    return set(parsed)


def _checked_fold(row: dict[str, str], ecg_id: str, patient_id: str,
                  patient_folds: dict[str, str]) -> str:
    """Return the record's official fold, requiring each patient to stay in one fold."""
    fold = row["strat_fold"].strip()
    if fold not in OFFICIAL_FOLDS:
        raise ValueError(f"Invalid strat_fold {fold!r} for ecg_id {ecg_id}")
    if patient_id in patient_folds and patient_folds[patient_id] != fold:
        raise ValueError(f"Patient {patient_id} spans official folds")
    patient_folds[patient_id] = fold
    return fold


def load_records(path: Path, diagnostic_codes: dict[str, str]) -> list[dict[str, str]]:
    """
    Read and label every PTB-XL record.

    Parameters
    ----------
    path : Path
        ``ptbxl_database.csv``.
    diagnostic_codes : dict[str, str]
        Diagnostic class keyed by SCP code.

    Returns
    -------
    list[dict[str, str]]
        Records with waveform fields, fold, target, reason and raw SCP codes.

    Raises
    ------
    ValueError
        If columns or identifiers are missing, an ID repeats, or a patient
        spans folds.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {*WAVEFORM_FIELDS, "strat_fold", "scp_codes"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing database columns: {sorted(required - set(reader.fieldnames or []))}")
        records = []
        ids = set()
        patient_folds: dict[str, str] = {}
        for row in reader:
            record = {field: row[field].strip() for field in WAVEFORM_FIELDS}
            ecg_id, patient_id = record["ecg_id"], record["patient_id"]
            if not all(record.values()):
                raise ValueError(f"Missing waveform identifier or filename for ecg_id {ecg_id}")
            if ecg_id in ids:
                raise ValueError(f"Duplicate ecg_id {ecg_id}")
            ids.add(ecg_id)
            fold = _checked_fold(row, ecg_id, patient_id, patient_folds)
            target, reason = classify(parse_codes(row["scp_codes"], ecg_id), diagnostic_codes)
            records.append({**record, "strat_fold": fold, "target": target,
                            "reason": reason, "scp_codes": row["scp_codes"]})
    return records


def write_csv(path: Path, fields: tuple[str, ...], rows: Iterable[dict[str, str]]) -> None:
    """
    Write only ``fields`` of each row as CSV.

    Parameters
    ----------
    path : Path
        Destination file.
    fields : tuple[str, ...]
        Columns to keep, in order.
    rows : Iterable[dict[str, str]]
        Rows that may contain extra keys.
    """
    write_csv_atomic(path, ({field: row[field] for field in fields} for row in rows), fields)


def select_patients(eligible: list[dict[str, str]], label_fraction: float, seed: int) -> tuple[set[str], int]:
    """
    Sample whole patients to receive labels.

    Parameters
    ----------
    eligible : list[dict[str, str]]
        Resolved training records.
    label_fraction : float
        Fraction of eligible patients to label.
    seed : int
        ``random.Random`` seed.

    Returns
    -------
    tuple[set[str], int]
        Selected patient IDs, and how many were eligible.
    """
    eligible_patients = sorted({record["patient_id"] for record in eligible})
    selected_count = max(1, round(len(eligible_patients) * label_fraction)) if eligible_patients else 0
    selected = set(random.Random(seed).sample(eligible_patients, selected_count))
    return selected, len(eligible_patients)


def _counts(rows: list[dict[str, str]]) -> dict[str, int]:
    return {"records": len(rows), "patients": len({r["patient_id"] for r in rows}),
            "positive_records": sum(r["target"] == "1" for r in rows)}


def _fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0


def summarize(metadata_dir: Path, records: list[dict[str, str]], splits: dict[str, list[dict[str, str]]],
              eligible: list[dict[str, str]], selected_count: int, eligible_patient_count: int,
              label_fraction: float, seed: int) -> dict[str, Any]:
    """
    Describe the selection, split sizes and unresolved records.

    Parameters
    ----------
    metadata_dir : Path
        PTB-XL metadata directory.
    records : list[dict[str, str]]
        All labeled records.
    splits : dict[str, list[dict[str, str]]]
        Rows of each written manifest.
    eligible : list[dict[str, str]]
        Resolved training records.
    selected_count : int
        Number of labeled patients.
    eligible_patient_count : int
        Number of patients with a resolved training record.
    label_fraction : float
        Requested fraction of eligible patients.
    seed : int
        Selection seed.

    Returns
    -------
    dict[str, Any]
        Summary written to ``summary.json``.
    """
    train, labeled_train = splits["all_train_ssl"], splits["labeled_train"]
    all_train_patients = {record["patient_id"] for record in train}
    unresolved_by_fold = Counter(record["strat_fold"] for record in records if not record["target"])
    return {
        "task": "diagnostic_abnormality",
        "target_meaning": "Proxy of PTB-XL diagnostic findings, not health or referral truth",
        "seed": seed,
        "requested_label_fraction": label_fraction,
        "metadata_source_version": metadata_dir.name,
        "metadata_sha256": {
            "ptbxl_database.csv": sha256_file(metadata_dir / "ptbxl_database.csv"),
            "scp_statements.csv": sha256_file(metadata_dir / "scp_statements.csv"),
        },
        "eligible_training": _counts(eligible),
        "selected_eligible_patients": selected_count,
        "actual_eligible_patient_fraction": _fraction(selected_count, eligible_patient_count),
        "actual_eligible_record_fraction": _fraction(len(labeled_train), len(eligible)),
        "actual_all_training_patient_fraction": _fraction(selected_count, len(all_train_patients)),
        "actual_all_training_record_fraction": _fraction(len(labeled_train), len(train)),
        "splits": {name: _counts(rows) for name, rows in splits.items()},
        "excluded_unresolved_validation_records": sum(
            record["strat_fold"] == VALIDATION_FOLD and not record["target"] for record in records),
        "excluded_unresolved_test_records": sum(
            record["strat_fold"] == TEST_FOLD and not record["target"] for record in records),
        "unresolved_by_fold": dict(sorted(unresolved_by_fold.items(), key=lambda item: int(item[0]))),
        "pretraining_exposure_caveat": "HuBERT and ECG-FM pretrained checkpoints include PTB-XL exposure; "
                                       "their PTB-XL results are not an external untouched test.",
    }


def write_manifests(output_dir: Path, splits: dict[str, list[dict[str, str]]],
                    records: list[dict[str, str]]) -> None:
    """
    Write the split manifests and the full audit table.

    Parameters
    ----------
    output_dir : Path
        Destination directory.
    splits : dict[str, list[dict[str, str]]]
        Rows of each manifest.
    records : list[dict[str, str]]
        All records, written with their target and reason to ``audit/metadata.csv``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit").mkdir(exist_ok=True)
    write_csv(output_dir / "labeled_train.csv", LABELED_FIELDS, splits["labeled_train"])
    write_csv(output_dir / "validation.csv", LABELED_FIELDS, splits["validation"])
    write_csv(output_dir / "test.csv", LABELED_FIELDS, splits["test"])
    write_csv(output_dir / "unlabeled_train.csv", WAVEFORM_FIELDS, splits["unlabeled_train"])
    write_csv(output_dir / "all_train_ssl.csv", WAVEFORM_FIELDS, splits["all_train_ssl"])
    write_csv(output_dir / "audit" / "metadata.csv", AUDIT_FIELDS, records)


def prepare(metadata_dir: Path, output_dir: Path, label_fraction: float = 0.1,
            seed: int = 42) -> dict[str, Any]:
    """
    Write patient-disjoint PTB-XL manifests with a limited labeled subset.

    Parameters
    ----------
    metadata_dir : Path
        Directory with ``ptbxl_database.csv`` and ``scp_statements.csv``.
    output_dir : Path
        New or empty output directory.
    label_fraction : float
        Fraction of eligible training patients to label, in (0, 1].
    seed : int
        Patient selection seed.

    Returns
    -------
    dict[str, Any]
        Summary, also written to ``summary.json``.

    Raises
    ------
    ValueError
        If the fraction is out of range or the metadata is invalid.
    FileExistsError
        If the output directory is not empty.
    """
    if not 0 < label_fraction <= 1:
        raise ValueError("--label-fraction must be greater than 0 and at most 1")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Output directory is nonempty: {output_dir}")

    diagnostic_codes = read_diagnostic_codes(metadata_dir / "scp_statements.csv")
    records = load_records(metadata_dir / "ptbxl_database.csv", diagnostic_codes)
    train = [record for record in records if int(record["strat_fold"]) <= LAST_TRAIN_FOLD]
    eligible = [record for record in train if record["target"]]
    selected, eligible_patient_count = select_patients(eligible, label_fraction, seed)
    splits = {
        "all_train_ssl": train,
        "labeled_train": [record for record in eligible if record["patient_id"] in selected],
        "unlabeled_train": [record for record in train
                            if not record["target"] or record["patient_id"] not in selected],
        "validation": [record for record in records
                       if record["strat_fold"] == VALIDATION_FOLD and record["target"]],
        "test": [record for record in records if record["strat_fold"] == TEST_FOLD and record["target"]],
    }
    write_manifests(output_dir, splits, records)
    summary = summarize(metadata_dir, records, splits, eligible, len(selected),
                        eligible_patient_count, label_fraction, seed)
    write_json_atomic(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    """Parse arguments, prepare the manifests and print the summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=ROOT / "data/raw/ptb-xl/1.0.3")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--label-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    default_output = ROOT / "data/processed/ptbxl" / f"seed{args.seed}_fraction{args.label_fraction:g}"
    output_dir = args.output_dir or default_output
    print(json.dumps(prepare(args.metadata_dir, output_dir, args.label_fraction, args.seed), indent=2))


if __name__ == "__main__":
    main()
