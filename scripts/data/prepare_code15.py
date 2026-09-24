#!/usr/bin/env python3
"""Index and quality-check checksum-verified CODE-15% archives without changing raw data.

This deliberately does not remove zeros, infer original duration, resample, or assert
physical amplitude units. The Zenodo description's example padding does not match
the exact-zero boundaries observed in the local first archive.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ecg_experiment import code15
from ecg_experiment.code15 import (
    DIAGNOSIS_FIELDS,
    TRACE_SHAPE,
    extracted_hdf5,
    inspect_trace,
    metadata_rows,
    verified_files,
    verify_file,
)
from ecg_experiment.files import sha256_file, write_csv_atomic
from ecg_experiment.public_sources import signal_sha256
from ecg_experiment.staging import published_directory

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw/code-15pct/zenodo-4916206"
RECEIPT = ROOT / "data/acquisition/code_15pct.json"
SOURCE_URL = "https://zenodo.org/records/4916206"
ARCHIVE_COUNT = 18
SAMPLE_RATE_HZ = 400
PROGRESS_INTERVAL = 1000
MANIFEST_FIELDS = ("exam_id", "patient_id", "archive", "hdf5_member", "hdf5_index",
                   "prepared_file", "prepared_index",
                   "signal_sha256", "edge_zero_left", "edge_zero_right", "active_span",
                   "qc_flags")
META_FIELDS = ("exam_id", "patient_id", "age", "is_male", "nn_predicted_age")
LABEL_FIELDS = ("exam_id", "patient_id", *DIAGNOSIS_FIELDS, "normal_ecg",
                "diagnosis_label_provenance", "normal_ecg_provenance")
EXCLUSION_FIELDS = ("exam_id", "archive", "hdf5_index", "reason", "detail")
TABLES = {"manifest.csv": MANIFEST_FIELDS, "demographics.csv": META_FIELDS,
          "source_labels.csv": LABEL_FIELDS, "exclusions.csv": EXCLUSION_FIELDS}
NATIVE_ATTRIBUTES = {"sample_rate_hz": SAMPLE_RATE_HZ,
                     "unit_status": "unverified_native_stored_values",
                     "duration_status": "original_duration_unresolved"}

NativeDatasets = tuple[h5py.Dataset, h5py.Dataset]


@dataclass
class Preparation:
    """Accumulated audit rows and identities across every inspected archive."""

    metadata: dict[int, dict[str, str]]
    records: list[dict[str, Any]] = field(default_factory=list)
    demographics: list[dict[str, str]] = field(default_factory=list)
    labels: list[dict[str, str]] = field(default_factory=list)
    excluded: list[dict[str, Any]] = field(default_factory=list)
    counts: Counter[str] = field(default_factory=Counter)
    edge_pairs: Counter[tuple[int, int]] = field(default_factory=Counter)
    seen_ids: set[int] = field(default_factory=set)
    seen_signals: set[str] = field(default_factory=set)


def prepare(parts: list[int], output_dir: Path, limit: int | None = None,
            materialize_native: bool = False,
            raw_dir: Path = RAW, receipt_path: Path = RECEIPT) -> dict[str, Any]:
    """
    Make a manifest of native HDF5 rows; publish only after full completion.

    Parameters
    ----------
    parts : list[int]
        Distinct archive numbers from 0 through 17.
    output_dir : Path
        New directory to publish; it must not exist.
    limit : int | None
        Inspect only the first ``limit`` rows of each archive.
    materialize_native : bool
        Also copy accepted native float32 traces to one HDF5 per archive.
    raw_dir : Path
        Directory with the verified release files.
    receipt_path : Path
        Acquisition receipt with the official checksums.

    Returns
    -------
    dict[str, Any]
        The published ``metadata.json`` content.

    Raises
    ------
    ValueError
        If arguments are invalid or an input fails its contract.
    FileExistsError
        If the output or an interrupted staging directory exists.
    """
    if (not parts or any(part < 0 or part >= ARCHIVE_COUNT for part in parts)
            or len(set(parts)) != len(parts)):
        raise ValueError("parts must be distinct integers from 0 through 17")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if (output_dir.resolve().is_relative_to(raw_dir.resolve()) or
            output_dir.resolve().is_relative_to((ROOT / "data/raw").resolve())):
        raise ValueError("Output directory cannot be inside raw source")
    staging_dir = output_dir.with_name(output_dir.name + ".inprogress")
    if output_dir.exists() or staging_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing or interrupted output: {output_dir}")
    with published_directory(output_dir, staging_dir) as stage:
        result = _prepare_into(sorted(parts), stage, limit, materialize_native, raw_dir, receipt_path)
    return result


def _prepare_into(parts: list[int], output_dir: Path, limit: int | None,
                  materialize_native: bool, raw_dir: Path, receipt_path: Path) -> dict[str, Any]:
    """Build the complete prepared release inside a newly created staging directory."""
    verified = verified_files(receipt_path)
    if "exams.csv" not in verified:
        raise ValueError("exams.csv lacks a checksum receipt")
    verify_file(raw_dir / "exams.csv", verified["exams.csv"])
    state = Preparation(metadata_rows(raw_dir / "exams.csv"))
    checked_archives = []
    prepared_hashes = {}
    for part in parts:
        archive_name = f"exams_part{part}.zip"
        if archive_name not in verified:
            raise ValueError(f"{archive_name} lacks a checksum receipt")
        verify_file(raw_dir / archive_name, verified[archive_name])
        prepared_name = f"exams_part{part}_native.hdf5" if materialize_native else ""
        archive = _prepare_archive(raw_dir / archive_name, f"exams_part{part}.hdf5",
                                   output_dir, prepared_name, limit, state)
        if prepared_name:
            prepared_hashes[prepared_name] = sha256_file(output_dir / prepared_name)
        checked_archives.append({**archive, "official_md5": verified[archive_name]["checksum"],
                                 "local_sha256": sha256_file(raw_dir / archive_name)})
    tables = {"manifest.csv": state.records, "demographics.csv": state.demographics,
              "source_labels.csv": state.labels, "exclusions.csv": state.excluded}
    for name, rows in tables.items():
        write_csv_atomic(output_dir / name, rows, TABLES[name])
    result = _summary(parts, limit, materialize_native, state, checked_archives,
                      verified["exams.csv"]["checksum"], raw_dir)
    result["output_sha256"] = {**{name: sha256_file(output_dir / name) for name in TABLES},
                               **prepared_hashes}
    (output_dir / "metadata.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _prepare_archive(archive_path: Path, member_name: str, output_dir: Path, prepared_name: str,
                     limit: int | None, state: Preparation) -> dict[str, Any]:
    """Audit one archive, optionally writing its accepted traces to ``prepared_name``."""
    with extracted_hdf5(archive_path, member_name) as handle:
        if set(handle.keys()) != {"exam_id", "tracings"}:
            raise ValueError(f"Unexpected HDF5 keys: {archive_path.name}")
        ids, signals = handle["exam_id"], handle["tracings"]
        if (signals.ndim != 3 or signals.shape[1:] != TRACE_SHAPE
                or signals.dtype != np.dtype("<f4") or ids.dtype.kind not in "iu"
                or len(ids) != len(signals)):
            raise ValueError(f"Unexpected HDF5 shape: {archive_path.name}")
        count = min(len(ids), limit) if limit is not None else len(ids)
        staging = output_dir / (prepared_name + ".tmp") if prepared_name else None
        with _native_file(staging, count) as native:
            prepared_count = 0
            for index in range(count):
                trace_source = {"archive": archive_path.name, "hdf5_member": member_name,
                                "hdf5_index": index, "prepared_file": prepared_name}
                accepted = _audit_trace(int(ids[index]), signals, trace_source, native,
                                        prepared_count, state)
                prepared_count += accepted and native is not None
            if native is not None:
                native[0].resize((prepared_count, *TRACE_SHAPE))
                native[1].resize((prepared_count,))
        if staging is not None:
            staging.replace(output_dir / prepared_name)
        return {"archive": archive_path.name, "member": member_name,
                "hdf5_records": len(ids), "audited_records": count}


@contextmanager
def _native_file(path: Path | None, count: int) -> Iterator[NativeDatasets | None]:
    """Create the unscaled native HDF5 copy, or yield ``None`` when not materializing.

    The dataset handles stay open until the file closes; reopening them per
    trace changes the HDF5 metadata layout and thus the published file hash.
    """
    if path is None:
        yield None
        return
    with h5py.File(path, "w") as prepared:
        signals = prepared.create_dataset("tracings", shape=(count, *TRACE_SHAPE),
                                          maxshape=(None, *TRACE_SHAPE), chunks=(1, *TRACE_SHAPE),
                                          dtype="<f4")
        ids = prepared.create_dataset("exam_id", shape=(count,), maxshape=(None,), chunks=True, dtype="i8")
        prepared.attrs.update(NATIVE_ATTRIBUTES)
        yield signals, ids


def _metadata_exclusion(exam_id: int, row: dict[str, str] | None, member_name: str,
                        seen_ids: set[int]) -> tuple[str | None, str]:
    """Return the metadata exclusion reason and detail, if any, before reading the trace."""
    reason, detail = None, ""
    if exam_id in seen_ids:
        reason = "duplicate_exam_id"
    elif row is None:
        reason = "missing_csv_metadata"
    elif row["trace_file"] != member_name:
        reason, detail = "trace_file_mismatch", row["trace_file"]
    elif not row["patient_id"]:
        reason = "missing_patient_id"
    return reason, detail


def _audit_trace(exam_id: int, signals: h5py.Dataset, trace_source: dict[str, Any],
                 native: NativeDatasets | None, prepared_index: int, state: Preparation) -> bool:
    """Accept or exclude one trace, recording its rows; return whether it was accepted."""
    index = trace_source["hdf5_index"]
    row = state.metadata.get(exam_id)
    reason, detail = _metadata_exclusion(exam_id, row, trace_source["hdf5_member"], state.seen_ids)
    state.seen_ids.add(exam_id)
    if reason is None:
        trace = np.asarray(signals[index])
        reason, left, right, span, flags = inspect_trace(trace)
    if reason is None:
        digest = signal_sha256(trace)
        if digest in state.seen_signals:
            reason = "duplicate_native_signal"
    if reason is None:
        state.seen_signals.add(digest)
        state.edge_pairs[(left, right)] += 1
        if native is not None:
            native[0][prepared_index] = trace
            native[1][prepared_index] = exam_id
        state.records.append({"exam_id": exam_id, "patient_id": row["patient_id"], **trace_source,
                              "prepared_index": prepared_index if native is not None else "",
                              "signal_sha256": digest, "edge_zero_left": left,
                              "edge_zero_right": right, "active_span": span, "qc_flags": flags})
        state.demographics.append({key: row[key] for key in META_FIELDS})
        state.labels.append({
            **{key: row[key] for key in ("exam_id", "patient_id", *DIAGNOSIS_FIELDS, "normal_ecg")},
            "diagnosis_label_provenance": "released_flags; subset-specific method unverified",
            "normal_ecg_provenance": "automatic_annotation_per_Zenodo"})
        state.counts["accepted"] += 1
        state.counts["review_flagged"] += bool(flags)
    else:
        state.excluded.append({"exam_id": exam_id, "archive": trace_source["archive"],
                               "hdf5_index": index, "reason": reason, "detail": detail})
        state.counts[f"excluded_{reason}"] += 1
    state.counts["audited"] += 1
    if state.counts["audited"] % PROGRESS_INTERVAL == 0:
        print(f"CODE-15%: audited {state.counts['audited']} traces; accepted {state.counts['accepted']}",
              flush=True)
    return reason is None


def _summary(parts: list[int], limit: int | None, materialize_native: bool, state: Preparation,
             checked_archives: list[dict[str, Any]], csv_md5: str, raw_dir: Path) -> dict[str, Any]:
    """Describe the preparation, its inputs and its unresolved caveats."""
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_url": SOURCE_URL, "csv_records": len(state.metadata),
        "csv_official_md5": csv_md5,
        "csv_local_sha256": sha256_file(raw_dir / "exams.csv"),
        "archives": checked_archives, "limit_per_archive": limit,
        "complete_release": len(parts) == ARCHIVE_COUNT and limit is None,
        "counts": dict(state.counts),
        "distinct_patients_accepted": len({row["patient_id"] for row in state.records}),
        "edge_zero_pairs": {f"{a},{b}": n for (a, b), n in sorted(state.edge_pairs.items())},
        "signal_contract": ("native float32 [4096,12], 400 Hz, release lead order "
                            "DI,DII,DIII,AVR,AVL,AVF,V1-V6"),
        "unit_status": "physical amplitude units not independently verified; no scaling applied",
        "duration_status": "original 7/10-second duration unresolved; exact-zero edges are descriptive only",
        "canonical_500hz_10s_eligible": False,
        "native_materialized": materialize_native,
        "label_status": ("six released diagnosis flags have unverified subset-specific derivation; "
                         "normal_ecg is automatic; no mapping to PTB-XL endpoint"),
        "split_status": "patient IDs retained; no split assigned",
        "source_sha256": sha256_file(Path(__file__)),
        "library_source_sha256": sha256_file(Path(code15.__file__)),
    }


def main() -> None:
    """Prepare the requested CODE-15% archives and print the metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", type=int, nargs="+", default=[0],
                        help="Verified archive numbers 0..17; default: pilot archive 0")
    parser.add_argument("--limit", type=int, help="Inspect only the first N rows in each archive")
    parser.add_argument("--materialize-native", action="store_true",
                        help="Copy accepted native float32 traces to a separate HDF5 without scaling")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.parts, args.output_dir, args.limit,
                             args.materialize_native), indent=2), flush=True)


if __name__ == "__main__":
    main()
