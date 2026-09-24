#!/usr/bin/env python3
"""Prepare patient-disjoint PTB-XL manifests for a low-label proxy task."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


WAVEFORM_FIELDS = ("ecg_id", "patient_id", "filename_lr", "filename_hr")
LABELED_FIELDS = (*WAVEFORM_FIELDS, "target")
AUDIT_FIELDS = (*WAVEFORM_FIELDS, "strat_fold", "target", "reason", "scp_codes")


def read_diagnostic_codes(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"", "diagnostic", "diagnostic_class"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing SCP columns: {sorted(required - set(reader.fieldnames or []))}")
        result = {}
        for row in reader:
            if row["diagnostic"].strip() == "1.0" or row["diagnostic"].strip() == "1":
                code = row[""].strip()
                category = row["diagnostic_class"].strip()
                if code and category:
                    result[code] = category
        return result


def classify(codes: set[str], diagnostic_codes: dict[str, str]) -> tuple[str, str]:
    """Return a proxy target and its audit reason; empty target means unresolved."""
    if "NORM" in codes:
        if codes <= {"NORM", "SR"}:
            return "0", "norm_only_or_sinus_rhythm"
        return "", "norm_with_other_codes"
    abnormal = {diagnostic_codes[code] for code in codes & diagnostic_codes.keys()}
    if any(category != "NORM" for category in abnormal):
        return "1", "non_norm_diagnostic_class"
    return "", "no_non_norm_diagnostic_class"


def parse_codes(value: str, ecg_id: str) -> set[str]:
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"Invalid scp_codes for ecg_id {ecg_id}") from exc
    if not isinstance(parsed, dict) or not all(isinstance(code, str) for code in parsed):
        raise ValueError(f"Expected SCP code dictionary for ecg_id {ecg_id}")
    # A listed SCP key is present even when its likelihood is zero (unknown).
    return set(parsed)


def load_records(path: Path, diagnostic_codes: dict[str, str]) -> list[dict[str, str]]:
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
            fold = row["strat_fold"].strip()
            if fold not in {str(number) for number in range(1, 11)}:
                raise ValueError(f"Invalid strat_fold {fold!r} for ecg_id {ecg_id}")
            if patient_id in patient_folds and patient_folds[patient_id] != fold:
                raise ValueError(f"Patient {patient_id} spans official folds")
            patient_folds[patient_id] = fold
            codes = parse_codes(row["scp_codes"], ecg_id)
            target, reason = classify(codes, diagnostic_codes)
            records.append({**record, "strat_fold": fold, "target": target,
                            "reason": reason, "scp_codes": row["scp_codes"]})
    return records


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(metadata_dir: Path, output_dir: Path, label_fraction: float = 0.1,
            seed: int = 42) -> dict:
    if not 0 < label_fraction <= 1:
        raise ValueError("--label-fraction must be greater than 0 and at most 1")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Output directory is nonempty: {output_dir}")

    diagnostic_codes = read_diagnostic_codes(metadata_dir / "scp_statements.csv")
    records = load_records(metadata_dir / "ptbxl_database.csv", diagnostic_codes)
    train = [record for record in records if int(record["strat_fold"]) <= 8]
    eligible = [record for record in train if record["target"]]
    eligible_patients = sorted({record["patient_id"] for record in eligible})
    selected_count = max(1, round(len(eligible_patients) * label_fraction)) if eligible_patients else 0
    selected = set(random.Random(seed).sample(eligible_patients, selected_count))

    labeled_train = [record for record in eligible if record["patient_id"] in selected]
    unlabeled_train = [record for record in train if not record["target"] or record["patient_id"] not in selected]
    validation = [record for record in records if record["strat_fold"] == "9" and record["target"]]
    test = [record for record in records if record["strat_fold"] == "10" and record["target"]]

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit").mkdir(exist_ok=True)
    write_csv(output_dir / "labeled_train.csv", LABELED_FIELDS, labeled_train)
    write_csv(output_dir / "validation.csv", LABELED_FIELDS, validation)
    write_csv(output_dir / "test.csv", LABELED_FIELDS, test)
    write_csv(output_dir / "unlabeled_train.csv", WAVEFORM_FIELDS, unlabeled_train)
    write_csv(output_dir / "all_train_ssl.csv", WAVEFORM_FIELDS, train)
    write_csv(output_dir / "audit" / "metadata.csv", AUDIT_FIELDS, records)

    def counts(rows: list[dict[str, str]]) -> dict[str, int]:
        return {"records": len(rows), "patients": len({r["patient_id"] for r in rows}),
                "positive_records": sum(r["target"] == "1" for r in rows)}

    all_train_patients = {record["patient_id"] for record in train}
    summary = {
        "task": "diagnostic_abnormality",
        "target_meaning": "Proxy of PTB-XL diagnostic findings, not health or referral truth",
        "seed": seed,
        "requested_label_fraction": label_fraction,
        "metadata_source_version": metadata_dir.name,
        "metadata_sha256": {
            "ptbxl_database.csv": sha256(metadata_dir / "ptbxl_database.csv"),
            "scp_statements.csv": sha256(metadata_dir / "scp_statements.csv"),
        },
        "eligible_training": counts(eligible),
        "selected_eligible_patients": selected_count,
        "actual_eligible_patient_fraction": selected_count / len(eligible_patients) if eligible_patients else 0,
        "actual_eligible_record_fraction": len(labeled_train) / len(eligible) if eligible else 0,
        "actual_all_training_patient_fraction": selected_count / len(all_train_patients) if all_train_patients else 0,
        "actual_all_training_record_fraction": len(labeled_train) / len(train) if train else 0,
        "splits": {"all_train_ssl": counts(train), "labeled_train": counts(labeled_train),
                   "unlabeled_train": counts(unlabeled_train), "validation": counts(validation),
                   "test": counts(test)},
        "excluded_unresolved_validation_records": sum(record["strat_fold"] == "9" and not record["target"] for record in records),
        "excluded_unresolved_test_records": sum(record["strat_fold"] == "10" and not record["target"] for record in records),
        "unresolved_by_fold": dict(sorted(Counter(record["strat_fold"] for record in records
                                               if not record["target"]).items(), key=lambda item: int(item[0]))),
        "pretraining_exposure_caveat": "HuBERT and ECG-FM pretrained checkpoints include PTB-XL exposure; their PTB-XL results are not an external untouched test.",
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=Path("data/raw/ptb-xl/1.0.3"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--label-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output_dir = args.output_dir or Path("data/processed/ptbxl") / f"seed{args.seed}_fraction{args.label_fraction:g}"
    print(json.dumps(prepare(args.metadata_dir, output_dir, args.label_fraction, args.seed), indent=2))


if __name__ == "__main__":
    main()
