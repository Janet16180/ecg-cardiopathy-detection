#!/usr/bin/env python3
"""Read-only review of CODE-15% part-0 exact-waveform duplicate identities and flags."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from scripts.prepare_code15 import (DIAGNOSIS_FIELDS, RAW, RECEIPT, ROOT,
                                    metadata_rows, sha256_file, signal_sha256,
                                    verify_file)


FIELDS = ("excluded_exam_id", "retained_exam_id", "excluded_patient_id",
          "retained_patient_id", "same_patient_id", "diagnosis_set_relation",
          "normal_ecg_agrees", "excluded_positive_codes", "retained_positive_codes",
          "signal_sha256")


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def relation(a: set[str], b: set[str]) -> str:
    if a == b:
        return "same"
    if a < b:
        return "excluded_subset"
    if b < a:
        return "retained_subset"
    return "non_nested"


def audit(prepared_dir: Path, output_dir: Path) -> dict:
    if output_dir.resolve().is_relative_to((ROOT / "data/raw").resolve()):
        raise ValueError("Audit output must be outside data/raw")
    info = json.loads((prepared_dir / "metadata.json").read_text())
    if (info.get("limit_per_archive") is not None or
            [item["archive"] for item in info["archives"]] != ["exams_part0.zip"]):
        raise ValueError("This audit requires the completed CODE part-0 preparation")
    for name in ("manifest.csv", "exclusions.csv", "source_labels.csv"):
        if sha256_file(prepared_dir / name) != info["output_sha256"][name]:
            raise ValueError(f"Prepared CODE file changed: {name}")
    manifest = csv_rows(prepared_dir / "manifest.csv")
    excluded = [row for row in csv_rows(prepared_dir / "exclusions.csv")
                if row["reason"] == "duplicate_native_signal"]
    retained_by_hash = {row["signal_sha256"]: row for row in manifest}
    if len(retained_by_hash) != len(manifest):
        raise ValueError("Prepared manifest contains duplicate hashes")
    receipt = json.loads(RECEIPT.read_text())
    verified = {row["name"]: row for row in receipt["verified_files"]}
    for name in ("exams.csv", "exams_part0.zip"):
        verify_file(RAW / name, verified[name])
    metadata = metadata_rows(RAW / "exams.csv")
    results = []
    counts = Counter()
    archive_path = RAW / "exams_part0.zip"
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if len(members) != 1 or members[0].filename != "exams_part0.hdf5":
            raise ValueError("Unexpected CODE part-0 archive members")
        with tempfile.TemporaryDirectory(prefix="code15_duplicates_") as directory:
            hdf5_path = Path(directory) / "exams_part0.hdf5"
            with archive.open(members[0]) as source, hdf5_path.open("wb") as target:
                shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
            with h5py.File(hdf5_path, "r") as handle:
                ids, traces = handle["exam_id"], handle["tracings"]
                for row in excluded:
                    index = int(row["hdf5_index"])
                    exam_id = int(row["exam_id"])
                    if int(ids[index]) != exam_id:
                        raise ValueError(f"Duplicate index/ID mismatch: {exam_id}")
                    trace = np.asarray(traces[index])
                    digest = signal_sha256(trace)
                    retained = retained_by_hash.get(digest)
                    if retained is None:
                        raise ValueError(f"No retained copy for duplicate {exam_id}")
                    original = metadata[exam_id]
                    kept = metadata[int(retained["exam_id"])]
                    a = {code for code in DIAGNOSIS_FIELDS if original[code] == "True"}
                    b = {code for code in DIAGNOSIS_FIELDS if kept[code] == "True"}
                    label_relation = relation(a, b)
                    same_patient = original["patient_id"] == kept["patient_id"]
                    normal_agrees = original["normal_ecg"] == kept["normal_ecg"]
                    results.append({"excluded_exam_id": exam_id,
                                    "retained_exam_id": retained["exam_id"],
                                    "excluded_patient_id": original["patient_id"],
                                    "retained_patient_id": kept["patient_id"],
                                    "same_patient_id": str(same_patient).lower(),
                                    "diagnosis_set_relation": label_relation,
                                    "normal_ecg_agrees": str(normal_agrees).lower(),
                                    "excluded_positive_codes": ",".join(sorted(a)),
                                    "retained_positive_codes": ",".join(sorted(b)),
                                    "signal_sha256": digest})
                    counts[f"diagnosis_{label_relation}"] += 1
                    counts["same_patient_id"] += same_patient
                    counts["different_patient_id"] += not same_patient
                    counts["normal_ecg_disagrees"] += not normal_agrees
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "code15_duplicate_annotations.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)
    report = {"prepared_metadata_sha256": sha256_file(prepared_dir / "metadata.json"),
              "prepared_manifest_sha256": info["output_sha256"]["manifest.csv"],
              "official_archive_md5": verified["exams_part0.zip"]["checksum"],
              "duplicates_audited": len(results), "counts": dict(counts),
              "detail_sha256": sha256_file(detail_path),
              "interpretation": "Released flag-set differences are not adjudicated clinical contradictions"}
    (output_dir / "code15_duplicate_annotations.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path,
                        default=ROOT / "data/processed/code15_quality/part0_native")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/data_quality/code15_duplicates")
    args = parser.parse_args()
    print(json.dumps(audit(args.prepared_dir.resolve(), args.output_dir.resolve()), indent=2))


if __name__ == "__main__":
    main()
