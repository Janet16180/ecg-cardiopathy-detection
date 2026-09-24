#!/usr/bin/env python3
"""Prepare a checksum-verified, patient-sampled MIMIC-IV-ECG 1.0 SSL pool.

Only record_list.csv supplies selection and patient identity. No diagnoses,
machine measurements, ECG times, or cardiologist reports enter the manifest.
The download and waveform audit can be restarted with identical arguments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path, PurePosixPath

import numpy as np
import requests
import wfdb

from scripts.download_ptbxl_waveforms import fetch_file, sha256
from scripts.extract_pretrained import read_record


BASE = "https://physionet.org/files/mimic-iv-ecg/1.0"
FIELDS = ("ecg_id", "patient_id", "raw_dir", "filename_hr", "source")
SHA_LINE = re.compile(r"([0-9a-fA-F]{64})[ \t]+\*?(?:\./)?(.+)")
PATH = re.compile(r"files/p(\d{4})/p(\d{8})/s(\d{8})/(\d{8})")
_thread = threading.local()


def signal_hash(signal: np.ndarray) -> str:
    """Identity of the canonical 12 x 5000 physical-mV float32 signal."""
    return hashlib.sha256(np.ascontiguousarray(signal, dtype="<f4").tobytes()).hexdigest()


def safe_record_path(subject: str, study: str, name: str) -> bool:
    match = PATH.fullmatch(name)
    return bool(match and match[2] == subject and match[1] == subject[:4]
                and match[3] == study and match[4] == study)


def read_patients(path: Path) -> dict[str, list[tuple[str, str]]]:
    patients = defaultdict(list)
    seen = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or []) != {"subject_id", "study_id", "file_name", "ecg_time", "path"}:
            raise ValueError("Unexpected official record_list.csv columns")
        for row in reader:
            subject, study, name = row["subject_id"], row["study_id"], row["path"]
            if not safe_record_path(subject, study, name) or row["file_name"] != study:
                raise ValueError(f"Unsafe or inconsistent MIMIC waveform path: {name!r}")
            if name in seen:
                raise ValueError(f"Duplicate official waveform path: {name}")
            seen.add(name)
            patients[subject].append((study, name))
    if not patients:
        raise ValueError("Official record_list.csv contains no ECGs")
    return patients


def select_patients(patients: dict[str, list[tuple[str, str]]], seed: int,
                    max_records: int) -> tuple[list[tuple[str, str, str]], list[str]]:
    """Choose whole patients by seeded SHA256 rank, stopping before cap overflow."""
    if max_records < 1:
        raise ValueError("max_records must be positive")
    ranking = sorted(patients, key=lambda subject:
                     (hashlib.sha256(f"{seed}:{subject}".encode()).digest(), subject))
    selected, subjects = [], []
    for subject in ranking:
        records = sorted(patients[subject])
        if len(selected) + len(records) > max_records:
            break
        subjects.append(subject)
        selected.extend((subject, study, name) for study, name in records)
    if not selected:
        raise ValueError("No whole patient fits under max_records")
    return selected, subjects


def selection_hash(rows: list[tuple[str, str, str]], seed: int, cap: int,
                   list_hash: str) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps({"seed": seed, "max_records": cap,
                              "record_list_sha256": list_hash}, sort_keys=True).encode() + b"\n")
    for row in rows:
        digest.update("\t".join(row).encode() + b"\n")
    return digest.hexdigest()


def required_checksums(manifest: Path, paths: set[str]) -> dict[str, str]:
    wanted = {"record_list.csv", "LICENSE.txt"} | {name + extension for name in paths for extension in (".hea", ".dat")}
    found = {}
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            match = SHA_LINE.fullmatch(line.rstrip("\r\n"))
            if match is None:
                raise ValueError("Malformed official SHA256SUMS.txt line")
            digest, name = match.groups()
            if name in wanted:
                if name in found:
                    raise ValueError(f"Duplicate official checksum for {name}")
                found[name] = digest.lower()
    missing = wanted - found.keys()
    if missing:
        raise ValueError(f"Official SHA256SUMS.txt lacks {len(missing)} required files, e.g. {min(missing)}")
    return found


def fetch_metadata(raw_dir: Path, timeout: float, retries: int) -> tuple[Path, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    checksums_path = raw_dir / "SHA256SUMS.txt"
    if not checksums_path.is_file():
        fetch_metadata_ranges(f"{BASE}/SHA256SUMS.txt", checksums_path, timeout, retries)
    list_path = raw_dir / "record_list.csv"
    if not list_path.is_file():
        fetch_metadata_ranges(f"{BASE}/record_list.csv", list_path, timeout, retries)
    return checksums_path, list_path


def fetch_metadata_ranges(url: str, destination: Path, timeout: float, retries: int,
                          workers: int = 4, chunk_bytes: int = 4 * 1024 * 1024) -> None:
    """Resume independently verified byte ranges of large public metadata files."""
    with requests.get(url, headers={"Range": "bytes=0-0"}, timeout=timeout) as response:
        response.raise_for_status()
        match = re.fullmatch(r"bytes 0-0/(\d+)", response.headers.get("Content-Range", ""))
        if response.status_code != 206 or not match or len(response.content) != 1:
            raise ValueError(f"Server did not honor byte-range probe for {url}")
        size = int(match[1])
    pieces = [(start, min(start + chunk_bytes, size) - 1)
              for start in range(0, size, chunk_bytes)]
    destination.parent.mkdir(parents=True, exist_ok=True)

    def piece_path(index: int) -> Path:
        return destination.with_name(f"{destination.name}.range.{index:05d}")

    def fetch_piece(index: int, start: int, end: int) -> None:
        target = piece_path(index)
        expected_size = end - start + 1
        if target.is_file() and target.stat().st_size == expected_size:
            return
        partial = target.with_name(target.name + ".part")
        for attempt in range(retries + 1):
            try:
                with requests.get(url, headers={"Range": f"bytes={start}-{end}"},
                                  timeout=timeout, stream=True) as response:
                    response.raise_for_status()
                    expected_range = f"bytes {start}-{end}/{size}"
                    if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
                        raise ValueError(f"Unexpected Content-Range for {url}: {response.headers.get('Content-Range')}")
                    with partial.open("wb") as handle:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                handle.write(chunk)
                if partial.stat().st_size != expected_size:
                    raise ValueError(f"Short range response for {url}: {start}-{end}")
                os.replace(partial, target)
                return
            except (OSError, requests.RequestException, ValueError):
                partial.unlink(missing_ok=True)
                if attempt == retries:
                    raise
                time.sleep(min(2 ** attempt, 30))

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_piece, i, start, end): i
                   for i, (start, end) in enumerate(pieces)}
        for completed, future in enumerate(as_completed(futures), 1):
            future.result()
            if completed % 8 == 0 or completed == len(pieces):
                print(f"Metadata {destination.name}: {completed}/{len(pieces)} ranges, "
                      f"{completed * chunk_bytes / max(1, time.monotonic()-started) / 1e6:.2f} MB/s approximate", flush=True)
    assembled = destination.with_name(destination.name + ".part")
    with assembled.open("wb") as target:
        for i in range(len(pieces)):
            with piece_path(i).open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    target.write(chunk)
    if assembled.stat().st_size != size:
        raise ValueError(f"Assembled metadata size mismatch for {destination}")
    os.replace(assembled, destination)
    for i in range(len(pieces)):
        piece_path(i).unlink()


def ensure_metadata_file(raw_dir: Path, name: str, expected: str,
                         timeout: float, retries: int) -> None:
    path = raw_dir / name
    if not path.is_file() or sha256(path) != expected:
        if name == "record_list.csv":
            path.unlink(missing_ok=True)
            fetch_metadata_ranges(f"{BASE}/{name}", path, timeout, retries)
            if sha256(path) != expected:
                raise ValueError("Official record_list.csv does not match SHA256SUMS.txt")
        else:
            fetch_file(f"{BASE}/{name}", path, timeout, retries,
                       lambda temporary: sha256(temporary) == expected)


def verify_or_fetch(relative: str, raw_dir: Path, checksum: str,
                    timeout: float, retries: int) -> tuple[bool, int]:
    destination = raw_dir / relative
    if not PurePosixPath(relative).is_relative_to("files") or ".." in PurePosixPath(relative).parts:
        raise ValueError(f"Unsafe download path: {relative}")

    def valid(path: Path) -> bool:
        if not path.is_file():
            return False
        return sha256(path) == checksum

    existed = valid(destination)
    if not existed:
        # Each worker retains its own HTTP connection across both files and many
        # records. For hundreds of thousands of small files, this matters more
        # than transfer bandwidth.
        if not hasattr(_thread, "session"):
            _thread.session = requests.Session()
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".part")
        for attempt in range(retries + 1):
            try:
                with _thread.session.get(f"{BASE}/{relative}", timeout=timeout, stream=True) as response:
                    response.raise_for_status()
                    with partial.open("wb") as handle:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                handle.write(chunk)
                if not valid(partial):
                    raise ValueError(f"SHA256 or waveform size mismatch for {relative}")
                os.replace(partial, destination)
                break
            except (OSError, requests.RequestException, ValueError):
                partial.unlink(missing_ok=True)
                if attempt == retries:
                    raise
                time.sleep(min(2 ** attempt, 30))
    return (not existed), destination.stat().st_size


def download_record(row: tuple[str, str, str], raw_dir: Path,
                    checksums: dict[str, str], timeout: float, retries: int) -> tuple[int, int]:
    _, _, name = row
    downloaded, size = 0, 0
    for extension in (".hea", ".dat"):
        got, amount = verify_or_fetch(name + extension, raw_dir,
                                      checksums[name + extension], timeout, retries)
        downloaded += int(got)
        size += amount
    return downloaded, size


def download_selected(rows: list[tuple[str, str, str]], raw_dir: Path,
                      checksums: dict[str, str], workers: int, timeout: float,
                      retries: int) -> dict[str, int]:
    """Keep at most twice the worker count of requests in flight."""
    completed = downloaded = total_bytes = 0
    pending = {}
    iterator = iter(rows)
    last_report = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        def submit_next() -> bool:
            try:
                row = next(iterator)
            except StopIteration:
                return False
            pending[pool.submit(download_record, row, raw_dir, checksums, timeout, retries)] = row
            return True

        for _ in range(min(len(rows), workers * 2)):
            submit_next()
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                row = pending.pop(future)
                try:
                    got, amount = future.result()
                except Exception as error:
                    for other in pending:
                        other.cancel()
                    raise RuntimeError(f"Download failed for {row[2]}: {error}") from error
                completed += 1
                downloaded += got
                total_bytes += amount
                submit_next()
            now = time.monotonic()
            if completed % 500 == 0 or now - last_report >= 30 or completed == len(rows):
                print(f"Verified {completed:,}/{len(rows):,} MIMIC ECGs; "
                      f"downloaded {downloaded:,} files; total {total_bytes / 1e9:.2f} GB", flush=True)
                last_report = now
    return {"verified_records": completed, "files_downloaded_this_run": downloaded,
            "selected_file_bytes": total_bytes}


def ptbxl_hashes(ptbxl_dir: Path) -> tuple[set[str], int]:
    hashes = set()
    with (ptbxl_dir / "ptbxl_database.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "filename_hr" not in (reader.fieldnames or []):
            raise ValueError("PTB-XL metadata lacks filename_hr")
        count = 0
        for row in reader:
            hashes.add(signal_hash(read_record(ptbxl_dir, row["filename_hr"])))
            count += 1
            if count % 5000 == 0:
                print(f"Audited {count:,} PTB-XL ECG identities", flush=True)
    if count == 0:
        raise ValueError("No PTB-XL records for leakage audit")
    return hashes, count


def check_waveform(raw_dir: Path, name: str) -> np.ndarray:
    if (raw_dir / (name + ".dat")).stat().st_size != 12 * 5000 * 2:
        raise ValueError("Expected 120,000-byte 16-bit 12-lead waveform data")
    header = wfdb.rdheader(str(raw_dir / name))
    if len(header.units) != 12 or any(unit != "mV" for unit in header.units):
        raise ValueError(f"Expected 12 physical-mV leads, found {header.units}")
    return read_record(raw_dir, name)


def audit(rows: list[tuple[str, str, str]], raw_dir: Path, output_dir: Path,
          selection_digest: str, ptb_hashes: set[str]) -> tuple[list[dict[str, str]], Counter]:
    """Checkpoint exclusions and identities in SQLite; resume in selection order."""
    db = sqlite3.connect(output_dir / "audit.sqlite3")
    try:
        db.execute("CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS outcomes (name TEXT PRIMARY KEY, status TEXT NOT NULL, "
                   "signal_sha256 TEXT, reason TEXT, detail TEXT)")
        previous = db.execute("SELECT value FROM state WHERE key='selection_sha256'").fetchone()
        if previous and previous[0] != selection_digest:
            raise ValueError("Existing audit has different patient selection; use another output directory")
        db.execute("INSERT OR IGNORE INTO state VALUES ('selection_sha256', ?)", (selection_digest,))
        db.commit()
        seen = set()
        accepted = []
        reasons = Counter()
        started = time.monotonic()
        for index, (subject, study, name) in enumerate(rows, 1):
            outcome = db.execute("SELECT status, signal_sha256, reason, detail FROM outcomes WHERE name=?",
                                 (name,)).fetchone()
            if outcome is None:
                try:
                    digest = signal_hash(check_waveform(raw_dir, name))
                    if digest in ptb_hashes:
                        status, reason = "excluded", "exact_duplicate_ptbxl"
                    elif digest in seen:
                        status, reason = "excluded", "exact_duplicate_mimic"
                    else:
                        status, reason = "accepted", None
                    detail = None
                except (ValueError, OSError, TypeError, IndexError) as error:
                    status, digest, reason, detail = "excluded", None, "input_contract", str(error)
                db.execute("INSERT INTO outcomes VALUES (?, ?, ?, ?, ?)",
                           (name, status, digest, reason, detail))
            else:
                status, digest, reason, detail = outcome
                if status == "accepted" and digest in ptb_hashes:
                    raise ValueError(f"Previously accepted {name} now matches PTB-XL")
                if status == "accepted" and digest in seen:
                    raise ValueError(f"Audit cache has duplicate accepted hash at {name}")
            if status == "accepted":
                seen.add(digest)
                accepted.append({"ecg_id": "mimic:" + study,
                                 "patient_id": "mimic:" + subject,
                                 "raw_dir": str(raw_dir.resolve()),
                                 "filename_hr": name, "source": "mimic"})
            else:
                reasons[reason] += 1
            if index % 500 == 0 or index == len(rows):
                db.commit()
                if index % 5000 == 0 or time.monotonic() - started >= 30 or index == len(rows):
                    print(f"Audited {index:,}/{len(rows):,} MIMIC ECGs; "
                          f"accepted {len(accepted):,}; excluded {sum(reasons.values()):,}", flush=True)
                    started = time.monotonic()
        db.commit()
        return accepted, reasons
    finally:
        db.close()


def atomic_text(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def lock_selection(output_dir: Path, rows: list[tuple[str, str, str]],
                   digest: str, seed: int, cap: int, list_hash: str) -> None:
    """Record the exact chosen subjects/studies before costly waveform fetches."""
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {"selection_sha256": digest, "seed": seed, "max_records": cap,
              "record_list_sha256": list_hash, "selected_records": len(rows)}
    config_path = output_dir / "selection.json"
    if config_path.is_file():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("Existing output directory uses another patient selection")
        selected_path = output_dir / "selected_records.csv"
        if not selected_path.is_file():
            raise ValueError("Existing selection is missing selected_records.csv")
        with selected_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            if next(reader, None) != ["subject_id", "study_id", "path"] or list(map(tuple, reader)) != rows:
                raise ValueError("Existing selected_records.csv differs from deterministic selection")
        return
    if any(output_dir.iterdir()):
        raise ValueError("Output directory contains files but no selection lock")
    selected_path = output_dir / "selected_records.csv"
    temporary = selected_path.with_name(selected_path.name + ".partial")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("subject_id", "study_id", "path"))
        writer.writerows(rows)
    os.replace(temporary, selected_path)
    atomic_text(config_path, json.dumps(config, indent=2) + "\n")


def write_outputs(output_dir: Path, raw_dir: Path, rows: list[tuple[str, str, str]],
                  subjects: list[str], accepted: list[dict[str, str]], reasons: Counter,
                  stats: dict[str, int], seed: int, cap: int, selection_digest: str,
                  list_hash: str, sums_hash: str, license_hash: str,
                  source_records: int, source_patients: int, ptb_count: int) -> None:
    destination = output_dir / "ssl_manifest.csv"
    temporary = destination.with_name(destination.name + ".partial")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(accepted)
    os.replace(temporary, destination)
    with sqlite3.connect(output_dir / "audit.sqlite3") as db:
        exclusions = db.execute("SELECT name, reason, detail FROM outcomes WHERE status='excluded' ORDER BY name")
        exclusion_path = output_dir / "exclusions.csv"
        temporary = exclusion_path.with_name(exclusion_path.name + ".partial")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("filename_hr", "reason", "detail"))
            writer.writerows(exclusions)
        os.replace(temporary, exclusion_path)
    metadata = {
        "source_url": BASE, "license": "PhysioNet MIMIC-IV-ECG 1.0; see official LICENSE.txt",
        "selection": "Seeded SHA256 rank of subject_id; take complete patients until next would exceed max_records",
        "seed": seed, "max_records": cap,
        "source_records": source_records, "source_patients": source_patients,
        "selected_records": len(rows),
        "selected_patients": len(subjects), "accepted_records": len(accepted),
        "accepted_patients": len({r["patient_id"] for r in accepted}),
        "exclusion_counts": dict(sorted(reasons.items())),
        "selection_sha256": selection_digest,
        "record_list_sha256": list_hash, "official_checksums_sha256": sums_hash,
        "license_sha256": license_hash,
        "selected_records_sha256": sha256(output_dir / "selected_records.csv"),
        "manifest_sha256": sha256(destination), "exclusions_sha256": sha256(exclusion_path),
        "ptbxl_records_checked": ptb_count,
        "patient_identity": "subject_id from official MIMIC-IV-ECG record_list.csv; all selected records of each chosen subject kept before waveform audit",
        "signal_identity": "SHA256 of canonical lead-ordered float32 physical-mV 12 x 5000 decoded samples",
        "contract": "12 named leads, physical mV, 500 Hz, 5000 samples, finite decoded signal",
        "labels": "No diagnostic labels, machine measurements, reports, or ECG times used",
        "raw_dir": str(raw_dir.resolve()),
        "download": stats,
    }
    atomic_text(output_dir / "metadata.json", json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"manifest": str(destination), "selected_records": len(rows),
                      "accepted_records": len(accepted), "excluded": dict(reasons)}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/mimic-iv-ecg/1.0"))
    parser.add_argument("--ptbxl-dir", type=Path, default=Path("data/raw/ptb-xl/1.0.3"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/mimic_ssl_200k"))
    parser.add_argument("--max-records", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--selection-only", action="store_true",
                        help="Verify metadata and report deterministic sample without downloading waveforms")
    args = parser.parse_args()
    if args.max_records < 1 or args.workers < 1 or args.workers > 64 or args.timeout <= 0 or args.retries < 0:
        parser.error("max-records/workers/timeout must be positive, workers <=64, retries >=0")
    sums_path, list_path = fetch_metadata(args.raw_dir, args.timeout, args.retries)
    # First get the official hash of record_list.csv without holding 176 MB in memory.
    metadata_checksums = required_checksums(sums_path, set())
    ensure_metadata_file(args.raw_dir, "record_list.csv", metadata_checksums["record_list.csv"],
                         args.timeout, args.retries)
    ensure_metadata_file(args.raw_dir, "LICENSE.txt", metadata_checksums["LICENSE.txt"],
                         args.timeout, args.retries)
    list_digest = sha256(list_path)
    patients = read_patients(list_path)
    rows, subjects = select_patients(patients, args.seed, args.max_records)
    selected_digest = selection_hash(rows, args.seed, args.max_records, list_digest)
    source_records = sum(map(len, patients.values()))
    print(f"Selected {len(rows):,} ECGs from {len(subjects):,} whole patients "
          f"of {source_records:,} official ECGs; selection SHA256 {selected_digest}", flush=True)
    if args.selection_only:
        return
    checksums = required_checksums(sums_path, {row[2] for row in rows})
    lock_selection(args.output_dir, rows, selected_digest, args.seed, args.max_records, list_digest)
    stats = download_selected(rows, args.raw_dir, checksums, args.workers, args.timeout, args.retries)
    ptb_hashes, ptb_count = ptbxl_hashes(args.ptbxl_dir)
    accepted, reasons = audit(rows, args.raw_dir, args.output_dir, selected_digest, ptb_hashes)
    write_outputs(args.output_dir, args.raw_dir, rows, subjects, accepted, reasons,
                  stats, args.seed, args.max_records, selected_digest, list_digest,
                  sha256(sums_path), metadata_checksums["LICENSE.txt"],
                  source_records, len(patients), ptb_count)


if __name__ == "__main__":
    main()
