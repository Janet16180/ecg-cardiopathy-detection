#!/usr/bin/env python3
"""Read-only review of CODE-15% part-0 exact-waveform duplicate identities and flags."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ecg_experiment.code15 import (
    DIAGNOSIS_FIELDS,
    extracted_hdf5,
    metadata_rows,
    verified_files,
    verify_file,
)
from ecg_experiment.files import read_csv, sha256_file, write_csv_atomic
from ecg_experiment.public_sources import signal_sha256

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/code-15pct/zenodo-4916206"
RECEIPT = ROOT / "data/acquisition/code_15pct.json"
FIELDS = ("excluded_exam_id", "retained_exam_id", "excluded_patient_id",
          "retained_patient_id", "same_patient_id", "diagnosis_set_relation",
          "normal_ecg_agrees", "excluded_positive_codes", "retained_positive_codes",
          "signal_sha256")


def relation(a: set[str], b: set[str]) -> str:
    """
    Describe how the excluded copy's code set relates to the retained copy's.

    Parameters
    ----------
    a : set[str]
        Positive codes of the excluded copy.
    b : set[str]
        Positive codes of the retained copy.

    Returns
    -------
    str
        ``same``, ``excluded_subset``, ``retained_subset`` or ``non_nested``.
    """
    result = "non_nested"
    if a == b:
        result = "same"
    elif a < b:
        result = "excluded_subset"
    elif b < a:
        result = "retained_subset"
    return result


def load_prepared(prepared_dir: Path) -> tuple[dict[str, Any], list[dict[str, str]],
                                               dict[str, dict[str, str]]]:
    """
    Load a completed part-0 preparation and check its published hashes.

    Parameters
    ----------
    prepared_dir : Path
        Output of ``prepare_code15`` for part 0 without a limit.

    Returns
    -------
    tuple[dict[str, Any], list[dict[str, str]], dict[str, dict[str, str]]]
        Preparation metadata, duplicate-signal exclusion rows, and retained
        manifest rows keyed by signal hash.

    Raises
    ------
    ValueError
        If the preparation is partial, changed, or has duplicate hashes.
    """
    info = json.loads((prepared_dir / "metadata.json").read_text())
    if (info.get("limit_per_archive") is not None or
            [item["archive"] for item in info["archives"]] != ["exams_part0.zip"]):
        raise ValueError("This audit requires the completed CODE part-0 preparation")
    for name in ("manifest.csv", "exclusions.csv", "source_labels.csv"):
        if sha256_file(prepared_dir / name) != info["output_sha256"][name]:
            raise ValueError(f"Prepared CODE file changed: {name}")
    manifest = read_csv(prepared_dir / "manifest.csv")
    excluded = [row for row in read_csv(prepared_dir / "exclusions.csv")
                if row["reason"] == "duplicate_native_signal"]
    retained_by_hash = {row["signal_sha256"]: row for row in manifest}
    if len(retained_by_hash) != len(manifest):
        raise ValueError("Prepared manifest contains duplicate hashes")
    return info, excluded, retained_by_hash


def compare_duplicate(row: dict[str, str], ids: h5py.Dataset, traces: h5py.Dataset,
                      retained_by_hash: dict[str, dict[str, str]],
                      metadata: dict[int, dict[str, str]]) -> dict[str, Any]:
    """
    Compare one excluded duplicate trace with the retained copy it matches.

    Parameters
    ----------
    row : dict[str, str]
        Exclusion row of a duplicate native signal.
    ids, traces : h5py.Dataset
        Exam IDs and native traces of the raw part-0 HDF5.
    retained_by_hash : dict[str, dict[str, str]]
        Retained manifest rows keyed by signal hash.
    metadata : dict[int, dict[str, str]]
        ``exams.csv`` rows keyed by exam ID.

    Returns
    -------
    dict[str, Any]
        Detail row in ``FIELDS`` order.

    Raises
    ------
    ValueError
        If the index/ID pair or the retained copy cannot be matched.
    """
    index = int(row["hdf5_index"])
    exam_id = int(row["exam_id"])
    if int(ids[index]) != exam_id:
        raise ValueError(f"Duplicate index/ID mismatch: {exam_id}")
    digest = signal_sha256(np.asarray(traces[index]))
    retained = retained_by_hash.get(digest)
    if retained is None:
        raise ValueError(f"No retained copy for duplicate {exam_id}")
    original = metadata[exam_id]
    kept = metadata[int(retained["exam_id"])]
    a = {code for code in DIAGNOSIS_FIELDS if original[code] == "True"}
    b = {code for code in DIAGNOSIS_FIELDS if kept[code] == "True"}
    return {"excluded_exam_id": exam_id,
            "retained_exam_id": retained["exam_id"],
            "excluded_patient_id": original["patient_id"],
            "retained_patient_id": kept["patient_id"],
            "same_patient_id": str(original["patient_id"] == kept["patient_id"]).lower(),
            "diagnosis_set_relation": relation(a, b),
            "normal_ecg_agrees": str(original["normal_ecg"] == kept["normal_ecg"]).lower(),
            "excluded_positive_codes": ",".join(sorted(a)),
            "retained_positive_codes": ",".join(sorted(b)),
            "signal_sha256": digest}


def summarize(results: list[dict[str, Any]]) -> Counter[str]:
    """
    Count label relations and identity agreement over the compared duplicates.

    Parameters
    ----------
    results : list[dict[str, Any]]
        Detail rows from :func:`compare_duplicate`.

    Returns
    -------
    Counter[str]
        Summary counts.
    """
    counts = Counter()
    for result in results:
        same_patient = result["same_patient_id"] == "true"
        counts[f"diagnosis_{result['diagnosis_set_relation']}"] += 1
        counts["same_patient_id"] += same_patient
        counts["different_patient_id"] += not same_patient
        counts["normal_ecg_disagrees"] += result["normal_ecg_agrees"] != "true"
    return counts


def audit(prepared_dir: Path, output_dir: Path) -> dict[str, Any]:
    """
    Compare released flags and patient IDs between exact duplicate traces.

    Parameters
    ----------
    prepared_dir : Path
        Completed part-0 preparation.
    output_dir : Path
        Directory for the detail CSV and summary JSON; outside ``data/raw``.

    Returns
    -------
    dict[str, Any]
        Summary report.

    Raises
    ------
    ValueError
        If the output is inside raw data or any input check fails.
    """
    if output_dir.resolve().is_relative_to((ROOT / "data/raw").resolve()):
        raise ValueError("Audit output must be outside data/raw")
    info, excluded, retained_by_hash = load_prepared(prepared_dir)
    verified = verified_files(RECEIPT)
    for name in ("exams.csv", "exams_part0.zip"):
        verify_file(RAW / name, verified[name])
    metadata = metadata_rows(RAW / "exams.csv")
    with extracted_hdf5(RAW / "exams_part0.zip", "exams_part0.hdf5") as handle:
        ids, traces = handle["exam_id"], handle["tracings"]
        results = [compare_duplicate(row, ids, traces, retained_by_hash, metadata) for row in excluded]
    detail_path = output_dir / "code15_duplicate_annotations.csv"
    write_csv_atomic(detail_path, results, FIELDS)
    report = {"prepared_metadata_sha256": sha256_file(prepared_dir / "metadata.json"),
              "prepared_manifest_sha256": info["output_sha256"]["manifest.csv"],
              "official_archive_md5": verified["exams_part0.zip"]["checksum"],
              "duplicates_audited": len(results), "counts": dict(summarize(results)),
              "detail_sha256": sha256_file(detail_path),
              "interpretation": "Released flag-set differences are not adjudicated clinical contradictions"}
    (output_dir / "code15_duplicate_annotations.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    """Run the duplicate audit and print its summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path,
                        default=ROOT / "data/processed/code15_quality/part0_native")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/data_quality/code15_duplicates")
    args = parser.parse_args()
    print(json.dumps(audit(args.prepared_dir.resolve(), args.output_dir.resolve()), indent=2))


if __name__ == "__main__":
    main()
