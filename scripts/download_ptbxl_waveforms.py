#!/usr/bin/env python3
"""Download and verify PTB-XL waveform pairs from the official public mirror."""

from __future__ import annotations

import argparse
import csv
import http.client
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path, PurePosixPath

from ecg_experiment.downloads import fetch_file, parse_checksums
from ecg_experiment.files import sha256_file

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3"
ARCHIVE_URL = "https://physionet-open.s3.amazonaws.com/ptb-xl/ptb-xl-1.0.3.zip"
MANIFEST_NAME = "SHA256SUMS.txt"
LEAD_COUNT = 12
DURATION_SECONDS = 10
BYTES_PER_SAMPLE = 2
COPY_CHUNK_BYTES = 1024 * 1024
REPORT_INTERVAL_SECONDS = 30
EXTENSIONS = (".hea", ".dat")
# Errors from one record that are collected so the remaining records still finish.
RECORD_ERRORS = (OSError, ValueError, http.client.HTTPException)


def _checked_base(base: str, directory: str) -> str:
    path = PurePosixPath(base)
    if (not base or path.is_absolute() or ".." in path.parts or
            path.parts[0] != directory or len(path.parts) != 3 or path.suffix):
        raise ValueError(f"Unsafe or unexpected waveform path: {base!r}")
    return base


def waveform_paths(metadata_path: Path, sampling_rate: int, limit: int | None = None) -> list[str]:
    """
    Read the waveform record paths listed in the PTB-XL metadata table.

    Parameters
    ----------
    metadata_path : Path
        ``ptbxl_database.csv``.
    sampling_rate : int
        100 selects ``filename_lr``; any other value selects ``filename_hr``.
    limit : int | None
        Keep only the first ``limit`` records.

    Returns
    -------
    list[str]
        Record paths without extension, in table order.

    Raises
    ------
    ValueError
        If the column is missing, a path is unsafe, or a path repeats.
    """
    column = "filename_lr" if sampling_rate == 100 else "filename_hr"
    directory = "records100" if sampling_rate == 100 else "records500"
    paths = []
    seen = set()
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if column not in (reader.fieldnames or []):
            raise ValueError(f"Missing metadata column {column}")
        for row in reader:
            base = _checked_base(row[column].strip(), directory)
            if base in seen:
                raise ValueError(f"Duplicate waveform path: {base}")
            seen.add(base)
            paths.append(base)
            if limit is not None and len(paths) >= limit:
                break
    return paths


def fetch_manifest(metadata_dir: Path, base_url: str, timeout: float, retries: int) -> dict[str, str]:
    """
    Load the official checksum manifest, downloading it if absent.

    Parameters
    ----------
    metadata_dir : Path
        Local PTB-XL release root.
    base_url : str
        Mirror URL of the release root.
    timeout : float
        Socket timeout in seconds.
    retries : int
        Additional download attempts.

    Returns
    -------
    dict[str, str]
        SHA-256 digest keyed by relative path.
    """
    path = metadata_dir / MANIFEST_NAME
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        fetch_file(f"{base_url.rstrip('/')}/{MANIFEST_NAME}", path, timeout, retries)
    return parse_checksums(path.read_text(encoding="utf-8"))


def _valid_header(path: Path, relative: str, sampling_rate: int) -> bool:
    try:
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
        record_name, lead_count, frequency, sample_count = first_line.split()[:4]
        return (record_name == Path(relative).stem and int(lead_count) == LEAD_COUNT and
                float(frequency.split("/")[0]) == sampling_rate and
                int(sample_count) == DURATION_SECONDS * sampling_rate)
    except (UnicodeError, IndexError, ValueError):
        return False


def validate_file(path: Path, relative: str, sampling_rate: int, checksum: str) -> bool:
    """
    Check a local WFDB file's shape contract and official checksum.

    Parameters
    ----------
    path : Path
        Local file.
    relative : str
        Release-relative name, which decides between header and data checks.
    sampling_rate : int
        Expected sampling rate in Hz.
    checksum : str
        Official SHA-256 digest.

    Returns
    -------
    bool
        Whether the file exists and passes every check.
    """
    if not path.is_file():
        return False
    if relative.endswith(".dat"):
        expected_size = LEAD_COUNT * DURATION_SECONDS * sampling_rate * BYTES_PER_SAMPLE
        valid_contract = path.stat().st_size == expected_size
    else:
        valid_contract = _valid_header(path, relative, sampling_rate)
    return valid_contract and sha256_file(path) == checksum


def download_pair(base: str, metadata_dir: Path, sampling_rate: int,
                  checksums: dict[str, str], base_url: str, timeout: float,
                  retries: int) -> tuple[int, int]:
    """
    Download the header and data files of one record unless already valid.

    Parameters
    ----------
    base : str
        Record path without extension.
    metadata_dir : Path
        Local PTB-XL release root.
    sampling_rate : int
        Expected sampling rate in Hz.
    checksums : dict[str, str]
        Official SHA-256 digests.
    base_url : str
        Mirror URL of the release root.
    timeout : float
        Socket timeout in seconds.
    retries : int
        Additional download attempts per file.

    Returns
    -------
    tuple[int, int]
        Files downloaded and files already valid.
    """
    downloaded = skipped = 0
    for extension in EXTENSIONS:
        relative = base + extension
        checksum = checksums[relative]
        destination = metadata_dir / relative
        if validate_file(destination, relative, sampling_rate, checksum):
            skipped += 1
            continue
        verify = partial(validate_file, relative=relative, sampling_rate=sampling_rate, checksum=checksum)
        fetch_file(f"{base_url.rstrip('/')}/{relative}", destination, timeout, retries, verify)
        downloaded += 1
    return downloaded, skipped


def _archive_members(archive: zipfile.ZipFile, requested: set[str]) -> dict[str, zipfile.ZipInfo]:
    """Map each requested relative path to its single archive member."""
    members = {}
    for info in archive.infolist():
        name = PurePosixPath(info.filename)
        if len(name.parts) < 3:
            continue
        relative = "/".join(name.parts[-3:])
        if relative not in requested:
            continue
        if relative in members:
            raise ValueError(f"Duplicate archive member for {relative}")
        members[relative] = info
    missing = requested - members.keys()
    if missing:
        raise ValueError(f"Official ZIP lacks {len(missing)} requested files, e.g. {min(missing)}")
    return members


def _extract_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, destination: Path,
                    relative: str, sampling_rate: int, checksum: str) -> None:
    """Extract one member through a partial file and keep it only if it verifies."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".archive.part")
    try:
        with archive.open(info) as source, partial.open("wb") as target:
            while chunk := source.read(COPY_CHUNK_BYTES):
                target.write(chunk)
        if not validate_file(partial, relative, sampling_rate, checksum):
            raise ValueError(f"Integrity check failed for archive member {relative}")
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)


def download_archive(metadata_dir: Path, bases: list[str], sampling_rate: int,
                     checksums: dict[str, str], timeout: float, retries: int,
                     archive_url: str = ARCHIVE_URL) -> dict[str, int]:
    """
    Extract requested records from the official release ZIP.

    Parameters
    ----------
    metadata_dir : Path
        Local PTB-XL release root; the ZIP is kept there.
    bases : list[str]
        Record paths without extension.
    sampling_rate : int
        Expected sampling rate in Hz.
    checksums : dict[str, str]
        Official SHA-256 digests.
    timeout : float
        Socket timeout in seconds for the ZIP download.
    retries : int
        Additional ZIP download attempts.
    archive_url : str
        Official ZIP URL.

    Returns
    -------
    dict[str, int]
        Record count, files extracted and files already valid.

    Raises
    ------
    ValueError
        If the ZIP lacks, duplicates, or corrupts a requested file.
    """
    archive_path = metadata_dir / "ptb-xl-1.0.3.zip"
    if not archive_path.is_file() or not zipfile.is_zipfile(archive_path):
        print(f"Downloading official ZIP ({archive_url})", flush=True)
        fetch_file(archive_url, archive_path, timeout, retries)
    requested = {base + ext for base in bases for ext in EXTENSIONS}
    with zipfile.ZipFile(archive_path) as archive:
        members = _archive_members(archive, requested)
        downloaded = skipped = 0
        last_report = time.monotonic()
        for number, relative in enumerate(sorted(requested), 1):
            destination = metadata_dir / relative
            if validate_file(destination, relative, sampling_rate, checksums[relative]):
                skipped += 1
            else:
                _extract_member(archive, members[relative], destination, relative,
                                sampling_rate, checksums[relative])
                downloaded += 1
            now = time.monotonic()
            due = number % 2000 == 0 or now - last_report >= REPORT_INTERVAL_SECONDS
            if due or number == len(requested):
                print(f"{number}/{len(requested)} files; {downloaded} extracted, "
                      f"{skipped} verified existing", flush=True)
                last_report = now
    return {"records": len(bases), "files_downloaded": downloaded,
            "files_verified_existing": skipped}


def _download_records(metadata_dir: Path, bases: list[str], sampling_rate: int,
                      checksums: dict[str, str], workers: int, timeout: float, retries: int,
                      base_url: str) -> dict[str, int]:
    """Download records in parallel, collecting per-record failures."""
    downloaded = skipped = completed = 0
    last_report = time.monotonic()
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_pair, base, metadata_dir, sampling_rate,
                               checksums, base_url, timeout, retries): base for base in bases}
        for future in as_completed(futures):
            completed += 1
            try:
                got, already = future.result()
                downloaded += got
                skipped += already
            except RECORD_ERRORS as exc:
                errors.append(f"{futures[future]}: {exc}")
            now = time.monotonic()
            due = completed % 1000 == 0 or now - last_report >= REPORT_INTERVAL_SECONDS
            if due or completed == len(bases):
                print(f"{completed}/{len(bases)} records; {downloaded} files downloaded, "
                      f"{skipped} verified existing, {len(errors)} failed", flush=True)
                last_report = now
    if errors:
        raise RuntimeError(f"Failed {len(errors)} records; first errors: {'; '.join(errors[:5])}")
    return {"records": completed, "files_downloaded": downloaded, "files_verified_existing": skipped}


def download(metadata_dir: Path, sampling_rate: int = 100, workers: int = 16,
             limit: int | None = None, timeout: float = 30, retries: int = 3,
             base_url: str = BASE_URL, archive: bool = False,
             archive_url: str = ARCHIVE_URL) -> dict[str, int]:
    """
    Download and verify every waveform pair listed in the PTB-XL metadata.

    Parameters
    ----------
    metadata_dir : Path
        Local PTB-XL release root containing ``ptbxl_database.csv``.
    sampling_rate : int
        100 or 500 Hz records.
    workers : int
        Parallel record downloads.
    limit : int | None
        Keep only the first ``limit`` records.
    timeout : float
        Socket timeout in seconds.
    retries : int
        Additional download attempts per file.
    base_url : str
        Mirror URL of the release root.
    archive : bool
        Extract from the official ZIP instead of per-file downloads.
    archive_url : str
        Official ZIP URL.

    Returns
    -------
    dict[str, int]
        Record count, files downloaded and files already valid.

    Raises
    ------
    ValueError
        If an argument is invalid or the manifest lacks a requested file.
    RuntimeError
        If any record failed; every other record is still attempted.
    """
    if sampling_rate not in (100, 500):
        raise ValueError("--sampling-rate must be 100 or 500")
    if workers < 1 or (limit is not None and limit < 1) or timeout <= 0 or retries < 0:
        raise ValueError("workers, limit and timeout must be positive; retries cannot be negative")
    bases = waveform_paths(metadata_dir / "ptbxl_database.csv", sampling_rate, limit)
    checksums = fetch_manifest(metadata_dir, base_url, timeout, retries)
    missing = [base + ext for base in bases for ext in EXTENSIONS if base + ext not in checksums]
    if missing:
        raise ValueError(f"Official SHA256SUMS.txt lacks {len(missing)} requested files, e.g. {missing[0]}")
    if archive:
        return download_archive(metadata_dir, bases, sampling_rate, checksums,
                                timeout, retries, archive_url)
    return _download_records(metadata_dir, bases, sampling_rate, checksums, workers,
                             timeout, retries, base_url)


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
    parser.add_argument("--metadata-dir", type=Path, default=ROOT / "data/raw/ptb-xl/1.0.3")
    parser.add_argument("--sampling-rate", type=int, choices=(100, 500), default=100)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, help="Download only the first N records for a smoke test")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--archive", action="store_true",
                        help="Use the official 1.7 GB ZIP for faster bulk transfer")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """
    Parse arguments, download, and print the counts.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or ``None`` for ``sys.argv``.
    """
    args = parse_args(argv)
    print(download(args.metadata_dir, args.sampling_rate, args.workers,
                   args.limit, args.timeout, args.retries, archive=args.archive))


if __name__ == "__main__":
    main()
