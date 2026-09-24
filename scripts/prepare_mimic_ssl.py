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
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from itertools import islice
from pathlib import Path, PurePosixPath

import requests

from ecg_experiment.downloads import fetch_file
from ecg_experiment.files import sha256_file
from ecg_experiment.mimic import (
    BASE,
    audit,
    lock_selection,
    ptbxl_hashes,
    read_patients,
    required_checksums,
    select_patients,
    selection_hash,
    write_outputs,
)

ROOT = Path(__file__).resolve().parents[1]
CHUNK_BYTES = 1024 * 1024
RANGE_BYTES = 4 * 1024 * 1024
MAX_BACKOFF_SECONDS = 30
MAX_WORKERS = 64

SelectionRow = tuple[str, str, str]

_thread = threading.local()


def fetch_metadata(raw_dir: Path, timeout: float, retries: int) -> tuple[Path, Path]:
    """
    Fetch the official checksum manifest and record list if they are missing.

    Parameters
    ----------
    raw_dir : Path
        Local MIMIC-IV-ECG release root.
    timeout : float
        Socket timeout in seconds.
    retries : int
        Additional attempts per byte range.

    Returns
    -------
    tuple[Path, Path]
        Paths of ``SHA256SUMS.txt`` and ``record_list.csv``.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    checksums_path = raw_dir / "SHA256SUMS.txt"
    if not checksums_path.is_file():
        fetch_metadata_ranges(f"{BASE}/SHA256SUMS.txt", checksums_path, timeout, retries)
    list_path = raw_dir / "record_list.csv"
    if not list_path.is_file():
        fetch_metadata_ranges(f"{BASE}/record_list.csv", list_path, timeout, retries)
    return checksums_path, list_path


def _probe_size(url: str, timeout: float) -> int:
    """Return the file size from a one-byte range request."""
    with requests.get(url, headers={"Range": "bytes=0-0"}, timeout=timeout) as response:
        response.raise_for_status()
        match = re.fullmatch(r"bytes 0-0/(\d+)", response.headers.get("Content-Range", ""))
        if response.status_code != 206 or not match or len(response.content) != 1:
            raise ValueError(f"Server did not honor byte-range probe for {url}")
        return int(match[1])


def _range_path(destination: Path, index: int) -> Path:
    return destination.with_name(f"{destination.name}.range.{index:05d}")


def _write_stream(response: requests.Response, path: Path) -> None:
    with path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
            if chunk:
                handle.write(chunk)


def _fetch_range(url: str, target: Path, start: int, end: int, size: int,
                 timeout: float, retries: int) -> None:
    """Download bytes ``start``..``end`` into ``target``, retrying with backoff."""
    expected_size = end - start + 1
    if target.is_file() and target.stat().st_size == expected_size:
        return
    partial = target.with_name(target.name + ".part")
    expected_range = f"bytes {start}-{end}/{size}"
    for attempt in range(retries + 1):
        try:
            with requests.get(url, headers={"Range": f"bytes={start}-{end}"},
                              timeout=timeout, stream=True) as response:
                response.raise_for_status()
                content_range = response.headers.get("Content-Range")
                if response.status_code != 206 or content_range != expected_range:
                    raise ValueError(f"Unexpected Content-Range for {url}: {content_range}")
                _write_stream(response, partial)
            if partial.stat().st_size != expected_size:
                raise ValueError(f"Short range response for {url}: {start}-{end}")
            os.replace(partial, target)
            return
        except (OSError, requests.RequestException, ValueError):
            partial.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, MAX_BACKOFF_SECONDS))


def _assemble(destination: Path, piece_count: int, size: int) -> None:
    """Concatenate the downloaded ranges into ``destination`` and remove them."""
    assembled = destination.with_name(destination.name + ".part")
    with assembled.open("wb") as target:
        for index in range(piece_count):
            with _range_path(destination, index).open("rb") as source:
                while chunk := source.read(CHUNK_BYTES):
                    target.write(chunk)
    if assembled.stat().st_size != size:
        raise ValueError(f"Assembled metadata size mismatch for {destination}")
    os.replace(assembled, destination)
    for index in range(piece_count):
        _range_path(destination, index).unlink()


def fetch_metadata_ranges(url: str, destination: Path, timeout: float, retries: int,
                          workers: int = 4, chunk_bytes: int = RANGE_BYTES) -> None:
    """
    Resume independently verified byte ranges of large public metadata files.

    Parameters
    ----------
    url : str
        Source URL; the server must honor range requests.
    destination : Path
        Final file path.
    timeout : float
        Socket timeout in seconds.
    retries : int
        Additional attempts per range.
    workers : int
        Parallel range downloads.
    chunk_bytes : int
        Size of each range.

    Raises
    ------
    ValueError
        If the server ignores ranges or the assembled size differs.
    """
    size = _probe_size(url, timeout)
    pieces = [(start, min(start + chunk_bytes, size) - 1) for start in range(0, size, chunk_bytes)]
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_fetch_range, url, _range_path(destination, index), start, end, size,
                               timeout, retries)
                   for index, (start, end) in enumerate(pieces)]
        for completed, future in enumerate(as_completed(futures), 1):
            future.result()
            if completed % 8 == 0 or completed == len(pieces):
                rate = completed * chunk_bytes / max(1, time.monotonic() - started) / 1e6
                print(f"Metadata {destination.name}: {completed}/{len(pieces)} ranges, "
                      f"{rate:.2f} MB/s approximate", flush=True)
    _assemble(destination, len(pieces), size)


def ensure_metadata_file(raw_dir: Path, name: str, expected: str,
                         timeout: float, retries: int) -> None:
    """
    Refetch an official metadata file whose SHA-256 does not match.

    Parameters
    ----------
    raw_dir : Path
        Local release root.
    name : str
        ``record_list.csv`` (fetched by ranges) or another small file.
    expected : str
        Official SHA-256.
    timeout : float
        Socket timeout in seconds.
    retries : int
        Additional attempts.

    Raises
    ------
    ValueError
        If a refetched ``record_list.csv`` still does not match.
    """
    path = raw_dir / name
    if path.is_file() and sha256_file(path) == expected:
        return
    if name != "record_list.csv":
        fetch_file(f"{BASE}/{name}", path, timeout, retries,
                   lambda temporary: sha256_file(temporary) == expected)
        return
    path.unlink(missing_ok=True)
    fetch_metadata_ranges(f"{BASE}/{name}", path, timeout, retries)
    if sha256_file(path) != expected:
        raise ValueError("Official record_list.csv does not match SHA256SUMS.txt")


def _session() -> requests.Session:
    # Each worker keeps its own HTTP connection across files and records. For
    # hundreds of thousands of small files this matters more than bandwidth.
    if not hasattr(_thread, "session"):
        _thread.session = requests.Session()
    return _thread.session


def _matches(path: Path, checksum: str) -> bool:
    return path.is_file() and sha256_file(path) == checksum


def verify_or_fetch(relative: str, raw_dir: Path, checksum: str,
                    timeout: float, retries: int) -> tuple[bool, int]:
    """
    Keep a verified local file or download it again, retrying with backoff.

    Parameters
    ----------
    relative : str
        Path under ``files/`` in the release.
    raw_dir : Path
        Local release root.
    checksum : str
        Official SHA-256.
    timeout : float
        Socket timeout in seconds.
    retries : int
        Additional attempts.

    Returns
    -------
    tuple[bool, int]
        Whether the file was downloaded, and its size.

    Raises
    ------
    ValueError
        If the path is unsafe, or the last attempt fails verification.
    """
    destination = raw_dir / relative
    if not PurePosixPath(relative).is_relative_to("files") or ".." in PurePosixPath(relative).parts:
        raise ValueError(f"Unsafe download path: {relative}")
    existed = _matches(destination, checksum)
    if existed:
        return False, destination.stat().st_size
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    for attempt in range(retries + 1):
        try:
            with _session().get(f"{BASE}/{relative}", timeout=timeout, stream=True) as response:
                response.raise_for_status()
                _write_stream(response, partial)
            if not _matches(partial, checksum):
                raise ValueError(f"SHA256 or waveform size mismatch for {relative}")
            os.replace(partial, destination)
            break
        except (OSError, requests.RequestException, ValueError):
            partial.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, MAX_BACKOFF_SECONDS))
    return True, destination.stat().st_size


def download_record(row: SelectionRow, raw_dir: Path,
                    checksums: dict[str, str], timeout: float, retries: int) -> tuple[int, int]:
    """
    Verify or fetch both files of one selected record.

    Parameters
    ----------
    row : SelectionRow
        ``(subject_id, study_id, path)``.
    raw_dir : Path
        Local release root.
    checksums : dict[str, str]
        Official digests keyed by relative path.
    timeout : float
        Socket timeout in seconds.
    retries : int
        Additional attempts per file.

    Returns
    -------
    tuple[int, int]
        Number of files downloaded and their combined size.
    """
    _, _, name = row
    downloaded, size = 0, 0
    for extension in (".hea", ".dat"):
        got, amount = verify_or_fetch(name + extension, raw_dir,
                                      checksums[name + extension], timeout, retries)
        downloaded += int(got)
        size += amount
    return downloaded, size


def download_selected(rows: Sequence[SelectionRow], raw_dir: Path,
                      checksums: dict[str, str], workers: int, timeout: float,
                      retries: int) -> dict[str, int]:
    """
    Download every selected record, keeping at most twice the worker count in flight.

    Parameters
    ----------
    rows : Sequence[SelectionRow]
        Selected records.
    raw_dir : Path
        Local release root.
    checksums : dict[str, str]
        Official digests keyed by relative path.
    workers : int
        Download threads.
    timeout : float
        Socket timeout in seconds.
    retries : int
        Additional attempts per file.

    Returns
    -------
    dict[str, int]
        Verified record count, files downloaded and total bytes.

    Raises
    ------
    RuntimeError
        After cancelling pending work, if any record fails.
    """
    completed = downloaded = total_bytes = 0
    pending: dict[Future[tuple[int, int]], SelectionRow] = {}
    remaining = iter(rows)
    last_report = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        def submit(row: SelectionRow) -> None:
            pending[pool.submit(download_record, row, raw_dir, checksums, timeout, retries)] = row

        for row in islice(remaining, workers * 2):
            submit(row)
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                row = pending.pop(future)
                try:
                    got, amount = future.result()
                except Exception as error:
                    # Cancel queued work, then surface the failing record.
                    for other in pending:
                        other.cancel()
                    raise RuntimeError(f"Download failed for {row[2]}: {error}") from error
                completed += 1
                downloaded += got
                total_bytes += amount
                next_row = next(remaining, None)
                if next_row is not None:
                    submit(next_row)
            now = time.monotonic()
            if completed % 500 == 0 or now - last_report >= 30 or completed == len(rows):
                print(f"Verified {completed:,}/{len(rows):,} MIMIC ECGs; "
                      f"downloaded {downloaded:,} files; total {total_bytes / 1e9:.2f} GB", flush=True)
                last_report = now
    return {"verified_records": completed, "files_downloaded_this_run": downloaded,
            "selected_file_bytes": total_bytes}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/mimic-iv-ecg/1.0")
    parser.add_argument("--ptbxl-dir", type=Path, default=ROOT / "data/raw/ptb-xl/1.0.3")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/mimic_ssl_200k")
    parser.add_argument("--max-records", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--selection-only", action="store_true",
                        help="Verify metadata and report deterministic sample without downloading waveforms")
    args = parser.parse_args()
    if (args.max_records < 1 or args.workers < 1 or args.workers > MAX_WORKERS or args.timeout <= 0
            or args.retries < 0):
        parser.error("max-records/workers/timeout must be positive, workers <=64, retries >=0")
    return args


def _verified_metadata(args: argparse.Namespace) -> tuple[Path, Path, dict[str, str]]:
    """Fetch and verify the checksum manifest, record list and license."""
    sums_path, list_path = fetch_metadata(args.raw_dir, args.timeout, args.retries)
    # First get the official hash of record_list.csv without holding 176 MB in memory.
    metadata_checksums = required_checksums(sums_path, set())
    for name in ("record_list.csv", "LICENSE.txt"):
        ensure_metadata_file(args.raw_dir, name, metadata_checksums[name], args.timeout, args.retries)
    return sums_path, list_path, metadata_checksums


def main() -> None:
    """Select whole patients, download their ECGs and audit the waveforms."""
    args = _parse_args()
    sums_path, list_path, metadata_checksums = _verified_metadata(args)
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
