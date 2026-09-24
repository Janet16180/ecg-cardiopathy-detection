#!/usr/bin/env python3
"""Prepare a checksum-verified, patient-sampled MIMIC-IV-ECG 1.0 SSL pool.

Only record_list.csv supplies selection and patient identity. No diagnoses,
machine measurements, ECG times, or cardiologist reports enter the manifest.
The download and waveform audit can be restarted with identical arguments.
"""

from __future__ import annotations

import argparse
import os
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path, PurePosixPath

import requests

from ecg_experiment.downloads import fetch_file
from ecg_experiment.files import sha256_file
from ecg_experiment.mimic import (
    BASE, audit, lock_selection, ptbxl_hashes, read_patients, required_checksums,
    select_patients, selection_hash, write_outputs,
)


_thread = threading.local()


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
    if not path.is_file() or sha256_file(path) != expected:
        if name == "record_list.csv":
            path.unlink(missing_ok=True)
            fetch_metadata_ranges(f"{BASE}/{name}", path, timeout, retries)
            if sha256_file(path) != expected:
                raise ValueError("Official record_list.csv does not match SHA256SUMS.txt")
        else:
            fetch_file(f"{BASE}/{name}", path, timeout, retries,
                       lambda temporary: sha256_file(temporary) == expected)


def verify_or_fetch(relative: str, raw_dir: Path, checksum: str,
                    timeout: float, retries: int) -> tuple[bool, int]:
    destination = raw_dir / relative
    if not PurePosixPath(relative).is_relative_to("files") or ".." in PurePosixPath(relative).parts:
        raise ValueError(f"Unsafe download path: {relative}")

    def valid(path: Path) -> bool:
        if not path.is_file():
            return False
        return sha256_file(path) == checksum

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
    list_digest = sha256_file(list_path)
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
                  sha256_file(sums_path), metadata_checksums["LICENSE.txt"],
                  source_records, len(patients), ptb_count)


if __name__ == "__main__":
    main()
