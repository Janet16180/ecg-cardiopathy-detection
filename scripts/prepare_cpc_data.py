#!/usr/bin/env python3
"""Build a locked 40k MIMIC CPC selection and a resumable PTB/MIMIC waveform pool.

This script never downloads or writes raw waveforms. The MIMIC selection is the
whole-patient prefix of the existing seed-42 200k selection. Every selected raw
file is checked against the official SHA256SUMS before any waveform is audited.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import scipy
from scipy.signal import resample_poly

from scripts.extract_pretrained import read_record
from scripts.prepare_mimic_ssl import (
    audit, atomic_text, lock_selection, ptbxl_hashes, read_patients,
    required_checksums, select_patients, selection_hash, sha256, write_outputs,
)

PTB_SPLITS = ("all_train_ssl", "validation", "test")
ROW_FIELDS = ("ecg_id", "patient_id", "source", "split")


def read_locked_prefix(parent: Path, cap: int, seed: int, record_list: Path):
    """Independently recompute selection, then prove it prefixes the locked 200k."""
    parent_config = json.loads((parent / "selection.json").read_text())
    if parent_config["seed"] != seed or parent_config["max_records"] < cap:
        raise ValueError("Parent selection has incompatible seed or cap")
    list_hash = sha256(record_list)
    if parent_config["record_list_sha256"] != list_hash:
        raise ValueError("Official record list changed since parent selection")
    patients = read_patients(record_list)
    rows, subjects = select_patients(patients, seed, cap)
    with (parent / "selected_records.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        if next(reader, None) != ["subject_id", "study_id", "path"]:
            raise ValueError("Parent selection columns changed")
        for expected in rows:
            if tuple(next(reader, ())) != expected:
                raise ValueError("40k selection is not a prefix of locked 200k selection")
    return rows, subjects, patients, list_hash


def verify_selected_files(rows, raw_dir: Path, sums_path: Path, list_path: Path):
    """Fail closed on missing or corrupt pairs; make no changes to raw files."""
    checksums = required_checksums(sums_path, {row[2] for row in rows})
    if sha256(list_path) != checksums["record_list.csv"]:
        raise ValueError("Official record_list.csv SHA256 mismatch")
    license_path = raw_dir / "LICENSE.txt"
    if not license_path.is_file() or sha256(license_path) != checksums["LICENSE.txt"]:
        raise ValueError("Official LICENSE.txt missing or SHA256 mismatch")
    total_bytes = 0
    start = time.monotonic()
    for index, (_, _, name) in enumerate(rows, 1):
        for extension in (".hea", ".dat"):
            relative = name + extension
            path = raw_dir / relative
            if not path.is_file():
                raise FileNotFoundError(f"Selected raw file is missing: {path}")
            if sha256(path) != checksums[relative]:
                raise ValueError(f"Official SHA256 mismatch: {path}")
            total_bytes += path.stat().st_size
        if index % 5000 == 0 or index == len(rows):
            print(f"Official SHA256 verified {index:,}/{len(rows):,} MIMIC pairs "
                  f"in {time.monotonic()-start:.1f}s", flush=True)
    return {"verified_records": len(rows), "files_downloaded_this_run": 0,
            "selected_file_bytes": total_bytes}


def read_ptb_rows(manifest_dir: Path, ptb_dir: Path):
    """Keep public split identity while leaving targets out of the CPC pool."""
    rows, hashes, ids, patients = [], {}, set(), {split: set() for split in PTB_SPLITS}
    for split in PTB_SPLITS:
        path = manifest_dir / f"{split}.csv"
        hashes[path.name] = sha256(path)
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not {"ecg_id", "patient_id", "filename_hr"}.issubset(reader.fieldnames or ()):
                raise ValueError(f"Invalid PTB split columns: {path}")
            for row in reader:
                ecg_id, patient_id = row["ecg_id"], row["patient_id"]
                if ecg_id in ids:
                    raise ValueError(f"PTB ECG appears in multiple splits: {ecg_id}")
                ids.add(ecg_id)
                patients[split].add(patient_id)
                relative = row["filename_hr"]
                if not relative.startswith("records500/"):
                    raise ValueError(f"Unexpected PTB waveform path: {relative}")
                rows.append({"ecg_id": ecg_id, "patient_id": patient_id,
                             "source": "ptbxl", "split": "train" if split == "all_train_ssl" else split,
                             "raw_dir": str(ptb_dir), "filename_hr": relative})
    for i, a in enumerate(PTB_SPLITS):
        for b in PTB_SPLITS[i + 1:]:
            if patients[a] & patients[b]:
                raise ValueError(f"PTB patient leakage between {a} and {b}")
    counts = Counter(row["split"] for row in rows)
    if counts != {"train": 17418, "validation": 1870, "test": 1896}:
        raise ValueError(f"Unexpected PTB split counts: {dict(counts)}")
    return rows, hashes


def write_csv_atomic(path: Path, fieldnames, rows):
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fieldnames} for row in rows)
    os.replace(partial, path)


def resample_halves(raw: np.ndarray, rate: int) -> np.ndarray:
    if raw.shape != (12, 5000) or not np.isfinite(raw).all():
        raise ValueError("Expected finite canonical 12 x 5000 ECG")
    if rate not in (100, 250):
        raise ValueError("CPC rate must be 100 or 250 Hz")
    halves = [resample_poly(raw[:, start:start + 2500], 1, 500 // rate, axis=1)
              for start in (0, 2500)]
    reduced = np.concatenate(halves, axis=1).astype(np.float32, copy=False)
    if reduced.shape != (12, rate * 10) or not np.isfinite(reduced).all():
        raise ValueError("Invalid resampled ECG")
    return reduced


def build_cache(ptb_rows, mimic_rows, ptb_hashes, mimic_metadata: dict,
                output_dir: Path, rate: int, workers: int = 4):
    if rate not in (100, 250):
        raise ValueError("CPC rate must be 100 or 250 Hz")
    if workers < 1 or workers > 8:
        raise ValueError("CPC cache workers must be in [1, 8]")
    rows = ptb_rows + [{"ecg_id": row["ecg_id"], "patient_id": row["patient_id"],
                        "source": "mimic", "split": "train", "raw_dir": row["raw_dir"],
                        "filename_hr": row["filename_hr"]} for row in mimic_rows]
    ids = [row["ecg_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate CPC ECG ID")
    output_dir.mkdir(parents=True, exist_ok=True)
    row_path = output_dir / "rows.csv"
    if row_path.exists():
        with row_path.open(newline="", encoding="utf-8") as stream:
            old = list(csv.DictReader(stream))
        if old != [{field: row[field] for field in ROW_FIELDS} for row in rows]:
            raise ValueError("Existing cache rows differ; use another output directory")
    else:
        if any(path.name != "rows.csv.partial" for path in output_dir.iterdir()):
            raise ValueError("Cache directory contains files without rows.csv")
        write_csv_atomic(row_path, ROW_FIELDS, rows)
    ids_path = output_dir / "ecg_ids.npy"
    if ids_path.exists():
        if np.load(ids_path, allow_pickle=False).tolist() != ids:
            raise ValueError("Existing CPC ID array differs")
    else:
        partial_ids = output_dir / "ecg_ids.partial.npy"
        np.save(partial_ids, np.asarray(ids, dtype=str), allow_pickle=False)
        os.replace(partial_ids, ids_path)
    shape = (len(rows), 12, rate * 10)
    identity = {
        "shape": list(shape), "dtype": "float32", "sample_rate_hz": rate,
        "scipy_version": scipy.__version__,
        "resampling": (f"Each nonoverlapping 5-second 500 Hz half independently resample_poly"
                       f"(up=1, down={500 // rate}, axis=1), then concatenate; "
                       "SciPy default Kaiser beta=5 FIR, length 2*10*down+1 taps; "
                       "canonical lead order, physical mV, no per-record normalization"),
        "rows_sha256": sha256(row_path), "ecg_ids_sha256": sha256(ids_path),
        "mimic_selection_sha256": mimic_metadata["selection_sha256"],
        "mimic_manifest_sha256": mimic_metadata["manifest_sha256"],
        "ptb_manifest_sha256": ptb_hashes,
    }
    complete = output_dir / "complete.json"
    if complete.is_file():
        old = json.loads(complete.read_text())
        if any(old.get(key) != value for key, value in identity.items()):
            raise ValueError("Existing completed cache has different inputs")
        signal_path = output_dir / "signals.npy"
        if not signal_path.is_file() or sha256(signal_path) != old["signals_sha256"]:
            raise ValueError("Completed CPC cache SHA256 mismatch")
        return old
    partial = output_dir / "signals.partial.npy"
    signal_path = output_dir / "signals.npy"
    progress = output_dir / "progress.json"
    if progress.exists():
        state = json.loads(progress.read_text())
        if state["identity"] != identity:
            raise ValueError("CPC cache checkpoint differs from inputs")
        done = state["completed_rows"]
        if signal_path.is_file() and done == len(rows) and not partial.exists():
            info = {**identity, "record_count": len(rows),
                    "split_counts": dict(Counter(row["split"] for row in rows)),
                    "source_counts": dict(Counter(row["source"] for row in rows)),
                    "signals_sha256": sha256(signal_path)}
            atomic_text(complete, json.dumps(info, indent=2) + "\n")
            progress.unlink()
            return info
        if not partial.is_file() or signal_path.exists():
            raise ValueError("CPC cache checkpoint is missing its partial array")
        matrix = np.lib.format.open_memmap(partial, mode="r+")
        if matrix.shape != shape or matrix.dtype != np.float32 or not 0 <= done <= len(rows):
            raise ValueError("Invalid CPC cache checkpoint")
    else:
        if partial.exists() or signal_path.exists():
            raise ValueError("Partial CPC array has no progress checkpoint")
        done = 0
        matrix = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=shape)
        atomic_text(progress, json.dumps({"identity": identity, "completed_rows": 0}) + "\n")
    started = time.monotonic()
    def decode(row):
        raw = read_record(Path(row["raw_dir"]), row["filename_hr"])
        return resample_halves(raw, rate)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(done, len(rows), 100):
            stop = min(start + 100, len(rows))
            futures = [pool.submit(decode, row) for row in rows[start:stop]]
            for index, future in enumerate(futures, start):
                matrix[index] = future.result()
            matrix.flush()
            atomic_text(progress, json.dumps({"identity": identity, "completed_rows": stop}) + "\n")
            if stop % 1000 == 0 or stop == len(rows):
                print(f"CPC cache {stop:,}/{len(rows):,} ECGs; "
                      f"{time.monotonic()-started:.1f}s this run", flush=True)
    matrix.flush()
    del matrix
    os.replace(partial, signal_path)
    info = {**identity, "record_count": len(rows),
            "split_counts": dict(Counter(row["split"] for row in rows)),
            "source_counts": dict(Counter(row["source"] for row in rows)),
            "signals_sha256": sha256(signal_path)}
    atomic_text(complete, json.dumps(info, indent=2) + "\n")
    progress.unlink()
    return info


def prepare(args):
    raw_dir = args.mimic_raw_dir.resolve()
    ptb_dir = args.ptb_raw_dir.resolve()
    if args.cache_only:
        metadata = json.loads((args.mimic_output / "metadata.json").read_text())
        manifest_path = args.mimic_output / "ssl_manifest.csv"
        if sha256(manifest_path) != metadata["manifest_sha256"]:
            raise ValueError("Audited MIMIC manifest SHA256 mismatch")
        if sha256(args.mimic_output / "selected_records.csv") != metadata["selected_records_sha256"]:
            raise ValueError("Locked MIMIC selection SHA256 mismatch")
        if (metadata["selected_records"] != metadata["download"]["verified_records"] or
                metadata["ptbxl_records_checked"] != 21799):
            raise ValueError("MIMIC audit is incomplete")
        with manifest_path.open(newline="", encoding="utf-8") as stream:
            accepted = list(csv.DictReader(stream))
        if len(accepted) != metadata["accepted_records"]:
            raise ValueError("MIMIC manifest count mismatch")
        ptb_rows, ptb_manifest_hashes = read_ptb_rows(args.ptb_manifest_dir, ptb_dir)
        info = build_cache(ptb_rows, accepted, ptb_manifest_hashes, metadata,
                           args.cache_output, args.sample_rate, args.cache_workers)
        print(json.dumps(info, indent=2), flush=True)
        return
    sums_path, list_path = raw_dir / "SHA256SUMS.txt", raw_dir / "record_list.csv"
    rows, subjects, patients, list_hash = read_locked_prefix(
        args.parent_selection, args.max_records, args.seed, list_path)
    digest = selection_hash(rows, args.seed, args.max_records, list_hash)
    lock_selection(args.mimic_output, rows, digest, args.seed, args.max_records, list_hash)
    print(f"Locked {len(rows):,} ECGs from {len(subjects):,} whole patients; "
          f"selection SHA256 {digest}", flush=True)
    stats = verify_selected_files(rows, raw_dir, sums_path, list_path)
    ptb_hashes, ptb_count = ptbxl_hashes(ptb_dir)
    if ptb_count != 21799:
        raise ValueError(f"Expected all 21,799 PTB-XL raw identities, got {ptb_count}")
    accepted, reasons = audit(rows, raw_dir, args.mimic_output, digest, ptb_hashes)
    write_outputs(args.mimic_output, raw_dir, rows, subjects, accepted, reasons,
                  stats, args.seed, args.max_records, digest, list_hash,
                  sha256(sums_path), sha256(raw_dir / "LICENSE.txt"),
                  sum(map(len, patients.values())), len(patients), ptb_count)
    if args.audit_only:
        return
    ptb_rows, ptb_manifest_hashes = read_ptb_rows(args.ptb_manifest_dir, ptb_dir)
    metadata = json.loads((args.mimic_output / "metadata.json").read_text())
    info = build_cache(ptb_rows, accepted, ptb_manifest_hashes, metadata,
                       args.cache_output, args.sample_rate, args.cache_workers)
    print(json.dumps(info, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic-raw-dir", type=Path, default=Path("data/raw/mimic-iv-ecg/1.0"))
    parser.add_argument("--ptb-raw-dir", type=Path, default=Path("data/raw/ptb-xl/1.0.3"))
    parser.add_argument("--parent-selection", type=Path, default=Path("data/processed/mimic_ssl_200k"))
    parser.add_argument("--mimic-output", type=Path, default=Path("data/processed/mimic_ssl_40k_cpc"))
    parser.add_argument("--ptb-manifest-dir", type=Path, default=Path("data/processed/ptbxl/seed42_fraction1"))
    parser.add_argument("--cache-output", type=Path, default=Path("data/processed/cpc_pool_40k"))
    parser.add_argument("--max-records", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-rate", type=int, choices=(100, 250), default=250)
    parser.add_argument("--cache-workers", type=int, default=4)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--cache-only", action="store_true",
                        help="Build cache from a completed, checksum-verified audit")
    args = parser.parse_args()
    if args.max_records < 1:
        parser.error("max-records must be positive")
    if args.cache_workers < 1 or args.cache_workers > 8:
        parser.error("cache-workers must be in [1, 8]")
    if args.audit_only and args.cache_only:
        parser.error("audit-only and cache-only cannot be combined")
    prepare(args)


if __name__ == "__main__":
    main()
