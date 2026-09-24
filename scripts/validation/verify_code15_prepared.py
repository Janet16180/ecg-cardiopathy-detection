#!/usr/bin/env python3
"""Verify every materialized CODE-15% trace against its published manifest hash."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from scripts.data.prepare_code15 import inspect_trace, sha256_file, signal_sha256


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify(directory: Path) -> dict:
    metadata = json.loads((directory / "metadata.json").read_text())
    if not metadata["native_materialized"]:
        raise ValueError("No native materialization to verify")
    if metadata.get("canonical_500hz_10s_eligible") is not False:
        raise ValueError("Native CODE data must not be declared canonical 500 Hz/10 s")
    for name, expected in metadata["output_sha256"].items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f"Output file checksum mismatch: {name}")
    manifest = _rows(directory / "manifest.csv")
    demographics = _rows(directory / "demographics.csv")
    labels = _rows(directory / "source_labels.csv")
    exclusions = _rows(directory / "exclusions.csv")
    if (len(manifest) != metadata["counts"]["accepted"] or
            len(exclusions) + len(manifest) != metadata["counts"]["audited"]):
        raise ValueError("Manifest/exclusion counts do not match metadata")
    if len({row["exam_id"] for row in manifest}) != len(manifest):
        raise ValueError("Duplicate accepted exam ID")
    if len({row["signal_sha256"] for row in manifest}) != len(manifest):
        raise ValueError("Duplicate accepted native signal")
    if len({row["patient_id"] for row in manifest}) != metadata["distinct_patients_accepted"]:
        raise ValueError("Distinct patient count does not match metadata")
    if [row["exam_id"] for row in manifest] != [row["exam_id"] for row in demographics]:
        raise ValueError("Demographics not aligned to manifest")
    if [row["exam_id"] for row in manifest] != [row["exam_id"] for row in labels]:
        raise ValueError("Labels not aligned to manifest")
    if ([(row["exam_id"], row["patient_id"]) for row in manifest] !=
            [(row["exam_id"], row["patient_id"]) for row in demographics] or
            [(row["exam_id"], row["patient_id"]) for row in manifest] !=
            [(row["exam_id"], row["patient_id"]) for row in labels]):
        raise ValueError("Patient IDs are not aligned across linked tables")
    groups = defaultdict(list)
    for row in manifest:
        groups[row["prepared_file"]].append(row)
    checked = 0
    for filename, rows in groups.items():
        if not filename or Path(filename).name != filename:
            raise ValueError("Unsafe or missing prepared file reference")
        with h5py.File(directory / filename, "r") as handle:
            ids, signals = handle["exam_id"], handle["tracings"]
            if (len(ids) != len(rows) or signals.shape != (len(rows), 4096, 12)
                    or signals.dtype != np.dtype("<f4") or ids.dtype.kind not in "iu"):
                raise ValueError(f"Prepared HDF5 shape mismatch: {filename}")
            if (handle.attrs.get("sample_rate_hz") != 400 or
                    handle.attrs.get("unit_status") != "unverified_native_stored_values" or
                    handle.attrs.get("duration_status") != "original_duration_unresolved"):
                raise ValueError(f"Prepared HDF5 acquisition contract mismatch: {filename}")
            for expected_index, row in enumerate(rows):
                index = int(row["prepared_index"])
                if index != expected_index or int(ids[index]) != int(row["exam_id"]):
                    raise ValueError(f"Prepared HDF5 index/ID mismatch: {filename}:{index}")
                trace = np.asarray(signals[index])
                digest = signal_sha256(trace)
                if digest != row["signal_sha256"]:
                    raise ValueError(f"Prepared signal mismatch: {filename}:{index}")
                reason, left, right, span, flags = inspect_trace(trace)
                if (reason is not None or left != int(row["edge_zero_left"])
                        or right != int(row["edge_zero_right"])
                        or span != int(row["active_span"]) or flags != row["qc_flags"]):
                    raise ValueError(f"Prepared signal QC mismatch: {filename}:{index}")
                checked += 1
    return {"verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "directory": str(directory), "exam_rows_verified": checked,
            "excluded_rows_verified": len(exclusions), "files_verified": len(metadata["output_sha256"]),
            "metadata_sha256": sha256_file(directory / "metadata.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--receipt", type=Path, help="Write a verification receipt outside the prepared directory")
    args = parser.parse_args()
    result = verify(args.directory)
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
