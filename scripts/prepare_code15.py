#!/usr/bin/env python3
"""Index and quality-check checksum-verified CODE-15% archives without changing raw data.

This deliberately does not remove zeros, infer original duration, resample, or assert
physical amplitude units. The Zenodo description's example padding does not match
the exact-zero boundaries observed in the local first archive.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/code-15pct/zenodo-4916206"
RECEIPT = ROOT / "data/acquisition/code_15pct.json"
SOURCE_URL = "https://zenodo.org/records/4916206"
DIAGNOSIS_FIELDS = ("1dAVb", "RBBB", "LBBB", "SB", "ST", "AF")
MANIFEST_FIELDS = ("exam_id", "patient_id", "archive", "hdf5_member", "hdf5_index",
                   "prepared_file", "prepared_index",
                   "signal_sha256", "edge_zero_left", "edge_zero_right", "active_span",
                   "qc_flags")
META_FIELDS = ("exam_id", "patient_id", "age", "is_male", "nn_predicted_age")
LABEL_FIELDS = ("exam_id", "patient_id", *DIAGNOSIS_FIELDS, "normal_ecg",
                "diagnosis_label_provenance", "normal_ecg_provenance")
EXCLUSION_FIELDS = ("exam_id", "archive", "hdf5_index", "reason", "detail")


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signal_sha256(trace: np.ndarray) -> str:
    """Hash the native float32 samples in the manifest's byte order."""
    return hashlib.sha256(np.ascontiguousarray(trace, dtype="<f4").tobytes()).hexdigest()


def verify_file(path: Path, expected: dict) -> None:
    if not path.is_file() or path.stat().st_size != expected["size"]:
        raise ValueError(f"Missing or wrong-size verified input: {path}")
    algorithm, value = expected["checksum"].split(":", 1)
    if algorithm != "md5" or md5_file(path) != value:
        raise ValueError(f"Official checksum mismatch: {path}")


def metadata_rows(path: Path) -> dict[int, dict[str, str]]:
    rows = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"exam_id", "patient_id", "trace_file", "age", "is_male",
                    "nn_predicted_age", "normal_ecg", *DIAGNOSIS_FIELDS}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("exams.csv lacks required columns")
        for row in reader:
            exam_id = int(row["exam_id"])
            if exam_id in rows:
                raise ValueError(f"Duplicate exam_id in exams.csv: {exam_id}")
            rows[exam_id] = row
    return rows


def inspect_trace(trace: np.ndarray) -> tuple[str | None, int, int, int, str]:
    """Return exclusion reason, exact-zero edges, span and review flags."""
    if trace.shape != (4096, 12):
        return "shape_contract", 0, 0, 0, ""
    if not np.isfinite(trace).all():
        return "nonfinite_signal", 0, 0, 0, ""
    active = np.flatnonzero(np.any(trace != 0, axis=1))
    if not len(active):
        return "all_zero_signal", 4096, 4096, 0, ""
    if np.any(np.ptp(trace, axis=0) == 0):
        return "constant_lead", 0, 0, 0, ""
    left, right = int(active[0]), int(4095 - active[-1])
    flags = []
    if abs(left - right) > 2:
        flags.append("asymmetric_zero_edges_review")
    if np.any(np.std(trace, axis=0) < 0.01):
        flags.append("near_flat_native_units_review")
    return None, left, right, int(active[-1] - active[0] + 1), ";".join(flags)


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare(parts: list[int], output_dir: Path, limit: int | None = None,
            materialize_native: bool = False,
            raw_dir: Path = RAW, receipt_path: Path = RECEIPT) -> dict:
    """Make a manifest of native HDF5 rows; publish only after full completion."""
    if not parts or any(part < 0 or part > 17 for part in parts) or len(set(parts)) != len(parts):
        raise ValueError("parts must be distinct integers from 0 through 17")
    parts = sorted(parts)
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if (output_dir.resolve().is_relative_to(raw_dir.resolve()) or
            output_dir.resolve().is_relative_to((ROOT / "data/raw").resolve())):
        raise ValueError("Output directory cannot be inside raw source")
    staging_dir = output_dir.with_name(output_dir.name + ".inprogress")
    if output_dir.exists() or staging_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing or interrupted output: {output_dir}")
    staging_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir()
    try:
        result = _prepare_into(parts, staging_dir, limit, materialize_native, raw_dir, receipt_path)
        staging_dir.rename(output_dir)
        return result
    except Exception:
        shutil.rmtree(staging_dir)
        raise


def _prepare_into(parts: list[int], output_dir: Path, limit: int | None,
                  materialize_native: bool, raw_dir: Path, receipt_path: Path) -> dict:
    """Build the complete prepared release inside a newly created staging directory."""
    receipt = json.loads(receipt_path.read_text())
    verified = {item["name"]: item for item in receipt["verified_files"]}
    if "exams.csv" not in verified:
        raise ValueError("exams.csv lacks a checksum receipt")
    verify_file(raw_dir / "exams.csv", verified["exams.csv"])
    metadata = metadata_rows(raw_dir / "exams.csv")
    records, demographics, labels, excluded = [], [], [], []
    counts = Counter()
    edge_pairs = Counter()
    seen_ids = set()
    seen_signals = set()
    checked_archives = []
    prepared_hashes = {}
    for part in parts:
        archive_name = f"exams_part{part}.zip"
        member_name = f"exams_part{part}.hdf5"
        if archive_name not in verified:
            raise ValueError(f"{archive_name} lacks a checksum receipt")
        archive_path = raw_dir / archive_name
        verify_file(archive_path, verified[archive_name])
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            if len(members) != 1 or members[0].filename != member_name:
                raise ValueError(f"Unexpected ZIP contents: {archive_name}")
            with tempfile.TemporaryDirectory(prefix="code15_prepare_") as temporary_dir:
                temporary = Path(temporary_dir) / member_name
                with archive.open(member_name) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
                with h5py.File(temporary, "r") as handle:
                    if set(handle.keys()) != {"exam_id", "tracings"}:
                        raise ValueError(f"Unexpected HDF5 keys: {archive_name}")
                    ids, signals = handle["exam_id"], handle["tracings"]
                    if (signals.ndim != 3 or signals.shape[1:] != (4096, 12)
                            or signals.dtype != np.dtype("<f4") or ids.dtype.kind not in "iu"
                            or len(ids) != len(signals)):
                        raise ValueError(f"Unexpected HDF5 shape: {archive_name}")
                    count = min(len(ids), limit) if limit is not None else len(ids)
                    prepared_name = f"exams_part{part}_native.hdf5" if materialize_native else ""
                    staging = output_dir / (prepared_name + ".tmp") if materialize_native else None
                    prepared = h5py.File(staging, "w") if staging is not None else None
                    prepared_count = 0
                    try:
                        if prepared is not None:
                            prepared_signals = prepared.create_dataset(
                                "tracings", shape=(count, 4096, 12), maxshape=(None, 4096, 12),
                                chunks=(1, 4096, 12), dtype="<f4")
                            prepared_ids = prepared.create_dataset(
                                "exam_id", shape=(count,), maxshape=(None,), chunks=True, dtype="i8")
                            prepared.attrs["sample_rate_hz"] = 400
                            prepared.attrs["unit_status"] = "unverified_native_stored_values"
                            prepared.attrs["duration_status"] = "original_duration_unresolved"
                        for index in range(count):
                            exam_id = int(ids[index])
                            row = metadata.get(exam_id)
                            reason = None
                            detail = ""
                            if exam_id in seen_ids:
                                reason = "duplicate_exam_id"
                            elif row is None:
                                reason = "missing_csv_metadata"
                            elif row["trace_file"] != member_name:
                                reason = "trace_file_mismatch"
                                detail = row["trace_file"]
                            elif not row["patient_id"]:
                                reason = "missing_patient_id"
                            seen_ids.add(exam_id)
                            if reason is None:
                                trace = np.asarray(signals[index])
                                reason, left, right, span, flags = inspect_trace(trace)
                                if reason is None:
                                    digest = signal_sha256(trace)
                                    if digest in seen_signals:
                                        reason = "duplicate_native_signal"
                                    else:
                                        seen_signals.add(digest)
                                        edge_pairs[(left, right)] += 1
                                        patient_id = row["patient_id"]
                                        if prepared is not None:
                                            prepared_signals[prepared_count] = trace
                                            prepared_ids[prepared_count] = exam_id
                                        records.append({"exam_id": exam_id, "patient_id": patient_id,
                                                        "archive": archive_name, "hdf5_member": member_name,
                                                        "hdf5_index": index, "prepared_file": prepared_name,
                                                        "prepared_index": prepared_count if prepared is not None else "",
                                                        "signal_sha256": digest,
                                                        "edge_zero_left": left, "edge_zero_right": right,
                                                        "active_span": span, "qc_flags": flags})
                                        prepared_count += prepared is not None
                                        demographics.append({key: row[key] for key in META_FIELDS})
                                        labels.append({**{key: row[key] for key in
                                                          ("exam_id", "patient_id", *DIAGNOSIS_FIELDS,
                                                           "normal_ecg")},
                                                       "diagnosis_label_provenance": "released_flags; subset-specific method unverified",
                                                       "normal_ecg_provenance": "automatic_annotation_per_Zenodo"})
                                        counts["accepted"] += 1
                                        counts["review_flagged"] += bool(flags)
                            if reason is not None:
                                excluded.append({"exam_id": exam_id, "archive": archive_name,
                                                 "hdf5_index": index, "reason": reason,
                                                 "detail": detail})
                                counts[f"excluded_{reason}"] += 1
                            counts["audited"] += 1
                            if counts["audited"] % 1000 == 0:
                                print(f"CODE-15%: audited {counts['audited']} traces; accepted {counts['accepted']}",
                                      flush=True)
                        if prepared is not None:
                            prepared_signals.resize((prepared_count, 4096, 12))
                            prepared_ids.resize((prepared_count,))
                    finally:
                        if prepared is not None:
                            prepared.close()
                    if staging is not None:
                        final = output_dir / prepared_name
                        staging.replace(final)
                        prepared_hashes[prepared_name] = sha256_file(final)
                    checked_archives.append({"archive": archive_name, "member": member_name,
                                             "hdf5_records": len(ids), "audited_records": count,
                                             "official_md5": verified[archive_name]["checksum"],
                                             "local_sha256": sha256_file(archive_path)})
    _write_csv(output_dir / "manifest.csv", MANIFEST_FIELDS, records)
    _write_csv(output_dir / "demographics.csv", META_FIELDS, demographics)
    _write_csv(output_dir / "source_labels.csv", LABEL_FIELDS, labels)
    _write_csv(output_dir / "exclusions.csv", EXCLUSION_FIELDS, excluded)
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": SOURCE_URL, "csv_records": len(metadata),
        "csv_official_md5": verified["exams.csv"]["checksum"],
        "csv_local_sha256": sha256_file(raw_dir / "exams.csv"),
        "archives": checked_archives, "limit_per_archive": limit,
        "complete_release": len(parts) == 18 and limit is None,
        "counts": dict(counts), "distinct_patients_accepted": len({row["patient_id"] for row in records}),
        "edge_zero_pairs": {f"{a},{b}": n for (a, b), n in sorted(edge_pairs.items())},
        "signal_contract": "native float32 [4096,12], 400 Hz, release lead order DI,DII,DIII,AVR,AVL,AVF,V1-V6",
        "unit_status": "physical amplitude units not independently verified; no scaling applied",
        "duration_status": "original 7/10-second duration unresolved; exact-zero edges are descriptive only",
        "canonical_500hz_10s_eligible": False,
        "native_materialized": materialize_native,
        "label_status": "six released diagnosis flags have unverified subset-specific derivation; normal_ecg is automatic; no mapping to PTB-XL endpoint",
        "split_status": "patient IDs retained; no split assigned",
        "source_sha256": sha256_file(Path(__file__)),
        "output_sha256": {**{name: sha256_file(output_dir / name) for name in
                              ("manifest.csv", "demographics.csv", "source_labels.csv", "exclusions.csv")},
                          **prepared_hashes},
    }
    (output_dir / "metadata.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
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
