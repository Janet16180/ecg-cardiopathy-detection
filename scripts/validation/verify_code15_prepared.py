#!/usr/bin/env python3
"""Verify every materialized CODE-15% trace against its published manifest hash."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ecg_experiment.code15 import TRACE_SHAPE, inspect_trace
from ecg_experiment.files import read_csv, sha256_file
from ecg_experiment.provenance import utc_now
from ecg_experiment.public_sources import signal_sha256

SAMPLE_RATE_HZ = 400


def verify_tables(directory: Path, metadata: dict[str, Any]) -> tuple[list[dict[str, str]], int]:
    """
    Check the published tables against their hashes, counts and alignment.

    Parameters
    ----------
    directory : Path
        Prepared CODE-15% directory.
    metadata : dict[str, Any]
        Its ``metadata.json`` content.

    Returns
    -------
    tuple[list[dict[str, str]], int]
        Manifest rows and the number of exclusion rows.

    Raises
    ------
    ValueError
        If a file hash, count, identity or linked-table alignment differs.
    """
    for name, expected in metadata["output_sha256"].items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f"Output file checksum mismatch: {name}")
    manifest = read_csv(directory / "manifest.csv")
    demographics = read_csv(directory / "demographics.csv")
    labels = read_csv(directory / "source_labels.csv")
    exclusions = read_csv(directory / "exclusions.csv")
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
    patients = [row["patient_id"] for row in manifest]
    if (patients != [row["patient_id"] for row in demographics] or
            patients != [row["patient_id"] for row in labels]):
        raise ValueError("Patient IDs are not aligned across linked tables")
    return manifest, len(exclusions)


def verify_native_file(path: Path, rows: list[dict[str, str]]) -> None:
    """
    Check one native HDF5 file row by row against the manifest.

    Parameters
    ----------
    path : Path
        Prepared native HDF5 file.
    rows : list[dict[str, str]]
        Manifest rows that reference it, in file order.

    Raises
    ------
    ValueError
        If the file contract, an index, a signal hash or its QC fields differ.
    """
    filename = path.name
    with h5py.File(path, "r") as handle:
        ids, signals = handle["exam_id"], handle["tracings"]
        if (len(ids) != len(rows) or signals.shape != (len(rows), *TRACE_SHAPE)
                or signals.dtype != np.dtype("<f4") or ids.dtype.kind not in "iu"):
            raise ValueError(f"Prepared HDF5 shape mismatch: {filename}")
        if (handle.attrs.get("sample_rate_hz") != SAMPLE_RATE_HZ or
                handle.attrs.get("unit_status") != "unverified_native_stored_values" or
                handle.attrs.get("duration_status") != "original_duration_unresolved"):
            raise ValueError(f"Prepared HDF5 acquisition contract mismatch: {filename}")
        for expected_index, row in enumerate(rows):
            index = int(row["prepared_index"])
            if index != expected_index or int(ids[index]) != int(row["exam_id"]):
                raise ValueError(f"Prepared HDF5 index/ID mismatch: {filename}:{index}")
            trace = np.asarray(signals[index])
            if signal_sha256(trace) != row["signal_sha256"]:
                raise ValueError(f"Prepared signal mismatch: {filename}:{index}")
            reason, left, right, span, flags = inspect_trace(trace)
            if (reason is not None or left != int(row["edge_zero_left"])
                    or right != int(row["edge_zero_right"])
                    or span != int(row["active_span"]) or flags != row["qc_flags"]):
                raise ValueError(f"Prepared signal QC mismatch: {filename}:{index}")


def verify(directory: Path) -> dict[str, Any]:
    """
    Verify a native CODE-15% preparation end to end.

    Parameters
    ----------
    directory : Path
        Prepared directory made with ``--materialize-native``.

    Returns
    -------
    dict[str, Any]
        Verification receipt.

    Raises
    ------
    ValueError
        If the preparation is not native, claims canonical eligibility, or
        any table or trace check fails.
    """
    metadata = json.loads((directory / "metadata.json").read_text())
    if not metadata["native_materialized"]:
        raise ValueError("No native materialization to verify")
    if metadata.get("canonical_500hz_10s_eligible") is not False:
        raise ValueError("Native CODE data must not be declared canonical 500 Hz/10 s")
    manifest, excluded = verify_tables(directory, metadata)
    groups = defaultdict(list)
    for row in manifest:
        groups[row["prepared_file"]].append(row)
    for filename, rows in groups.items():
        if not filename or Path(filename).name != filename:
            raise ValueError("Unsafe or missing prepared file reference")
        verify_native_file(directory / filename, rows)
    return {"verified_at_utc": utc_now(),
            "directory": str(directory), "exam_rows_verified": len(manifest),
            "excluded_rows_verified": excluded, "files_verified": len(metadata["output_sha256"]),
            "metadata_sha256": sha256_file(directory / "metadata.json")}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse the command line.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or ``None`` for ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--receipt", type=Path,
                        help="Write a verification receipt outside the prepared directory")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """
    Verify a prepared directory and optionally write a receipt.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or ``None`` for ``sys.argv``.
    """
    args = parse_args(argv)
    result = verify(args.directory)
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
