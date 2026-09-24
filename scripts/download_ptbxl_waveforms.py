#!/usr/bin/env python3
"""Download and verify PTB-XL waveform pairs from the official public mirror."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath


BASE_URL = "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3"
ARCHIVE_URL = "https://physionet-open.s3.amazonaws.com/ptb-xl/ptb-xl-1.0.3.zip"
MANIFEST_NAME = "SHA256SUMS.txt"
LEADS = 12
DURATION_SECONDS = 10


def waveform_paths(metadata_path: Path, sampling_rate: int, limit: int | None = None) -> list[str]:
    column = "filename_lr" if sampling_rate == 100 else "filename_hr"
    directory = "records100" if sampling_rate == 100 else "records500"
    paths = []
    seen = set()
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if column not in (reader.fieldnames or []):
            raise ValueError(f"Missing metadata column {column}")
        for row in reader:
            base = row[column].strip()
            path = PurePosixPath(base)
            if (not base or path.is_absolute() or ".." in path.parts or
                    path.parts[0] != directory or len(path.parts) != 3 or path.suffix):
                raise ValueError(f"Unsafe or unexpected waveform path: {base!r}")
            if base in seen:
                raise ValueError(f"Duplicate waveform path: {base}")
            seen.add(base)
            paths.append(base)
            if limit is not None and len(paths) >= limit:
                break
    return paths


def parse_checksums(contents: str) -> dict[str, str]:
    checksums = {}
    for line in contents.splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64})[ \t]+\*?(?:\./)?(.+)", line)
        if not match:
            raise ValueError(f"Invalid SHA256SUMS line: {line[:100]!r}")
        digest, name = match.groups()
        if name in checksums and checksums[name] != digest.lower():
            raise ValueError(f"Conflicting checksum for {name}")
        checksums[name] = digest.lower()
    return checksums


def fetch_manifest(metadata_dir: Path, base_url: str, timeout: float, retries: int) -> dict[str, str]:
    path = metadata_dir / MANIFEST_NAME
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        fetch_file(f"{base_url.rstrip('/')}/{MANIFEST_NAME}", path, timeout, retries)
    return parse_checksums(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_file(path: Path, relative: str, sampling_rate: int, checksum: str) -> bool:
    if not path.is_file():
        return False
    if relative.endswith(".dat"):
        if path.stat().st_size != LEADS * DURATION_SECONDS * sampling_rate * 2:
            return False
    else:
        try:
            first_line = path.read_text(encoding="utf-8").splitlines()[0]
            fields = first_line.split()
            record_name, lead_count, frequency, sample_count = fields[:4]
            if (record_name != Path(relative).stem or int(lead_count) != LEADS or
                    float(frequency.split("/")[0]) != sampling_rate or
                    int(sample_count) != DURATION_SECONDS * sampling_rate):
                return False
        except (UnicodeError, IndexError, ValueError):
            return False
    return sha256(path) == checksum


def fetch_file(url: str, destination: Path, timeout: float, retries: int,
               verify=None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response, partial.open("wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
            if verify is not None and not verify(partial):
                raise ValueError(f"Integrity check failed for {url}")
            os.replace(partial, destination)
            return
        except (OSError, urllib.error.URLError, ValueError):
            partial.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 30))


def download_pair(base: str, metadata_dir: Path, sampling_rate: int,
                  checksums: dict[str, str], base_url: str, timeout: float,
                  retries: int) -> tuple[int, int]:
    downloaded = skipped = 0
    for extension in (".hea", ".dat"):
        relative = base + extension
        checksum = checksums[relative]
        destination = metadata_dir / relative
        if validate_file(destination, relative, sampling_rate, checksum):
            skipped += 1
            continue
        fetch_file(f"{base_url.rstrip('/')}/{relative}", destination, timeout, retries,
                   lambda path, name=relative, digest=checksum: validate_file(path, name, sampling_rate, digest))
        downloaded += 1
    return downloaded, skipped


def download_archive(metadata_dir: Path, bases: list[str], sampling_rate: int,
                     checksums: dict[str, str], timeout: float, retries: int,
                     archive_url: str = ARCHIVE_URL) -> dict[str, int]:
    archive_path = metadata_dir / "ptb-xl-1.0.3.zip"
    if not archive_path.is_file() or not zipfile.is_zipfile(archive_path):
        print(f"Downloading official ZIP ({archive_url})", flush=True)
        fetch_file(archive_url, archive_path, timeout, retries)
    requested = {base + ext for base in bases for ext in (".hea", ".dat")}
    with zipfile.ZipFile(archive_path) as archive:
        members = {}
        for info in archive.infolist():
            name = PurePosixPath(info.filename)
            if len(name.parts) < 3:
                continue
            relative = "/".join(name.parts[-3:])
            if relative in requested:
                if relative in members:
                    raise ValueError(f"Duplicate archive member for {relative}")
                members[relative] = info
        missing = requested - members.keys()
        if missing:
            raise ValueError(f"Official ZIP lacks {len(missing)} requested files, e.g. {min(missing)}")
        downloaded = skipped = 0
        last_report = time.monotonic()
        for number, relative in enumerate(sorted(requested), 1):
            destination = metadata_dir / relative
            if validate_file(destination, relative, sampling_rate, checksums[relative]):
                skipped += 1
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                partial = destination.with_name(destination.name + ".archive.part")
                try:
                    with archive.open(members[relative]) as source, partial.open("wb") as target:
                        while chunk := source.read(1024 * 1024):
                            target.write(chunk)
                    if not validate_file(partial, relative, sampling_rate, checksums[relative]):
                        raise ValueError(f"Integrity check failed for archive member {relative}")
                    os.replace(partial, destination)
                finally:
                    partial.unlink(missing_ok=True)
                downloaded += 1
            now = time.monotonic()
            if number % 2000 == 0 or now - last_report >= 30 or number == len(requested):
                print(f"{number}/{len(requested)} files; {downloaded} extracted, "
                      f"{skipped} verified existing", flush=True)
                last_report = now
    return {"records": len(bases), "files_downloaded": downloaded,
            "files_verified_existing": skipped}


def download(metadata_dir: Path, sampling_rate: int = 100, workers: int = 16,
             limit: int | None = None, timeout: float = 30, retries: int = 3,
             base_url: str = BASE_URL, archive: bool = False,
             archive_url: str = ARCHIVE_URL) -> dict[str, int]:
    if sampling_rate not in (100, 500):
        raise ValueError("--sampling-rate must be 100 or 500")
    if workers < 1 or (limit is not None and limit < 1) or timeout <= 0 or retries < 0:
        raise ValueError("workers, limit and timeout must be positive; retries cannot be negative")
    bases = waveform_paths(metadata_dir / "ptbxl_database.csv", sampling_rate, limit)
    checksums = fetch_manifest(metadata_dir, base_url, timeout, retries)
    missing = [base + ext for base in bases for ext in (".hea", ".dat") if base + ext not in checksums]
    if missing:
        raise ValueError(f"Official SHA256SUMS.txt lacks {len(missing)} requested files, e.g. {missing[0]}")
    if archive:
        return download_archive(metadata_dir, bases, sampling_rate, checksums,
                                timeout, retries, archive_url)

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
            except Exception as exc:
                errors.append(f"{futures[future]}: {exc}")
            now = time.monotonic()
            if completed % 1000 == 0 or now - last_report >= 30 or completed == len(bases):
                print(f"{completed}/{len(bases)} records; {downloaded} files downloaded, "
                      f"{skipped} verified existing, {len(errors)} failed", flush=True)
                last_report = now
    if errors:
        raise RuntimeError(f"Failed {len(errors)} records; first errors: {'; '.join(errors[:5])}")
    return {"records": completed, "files_downloaded": downloaded, "files_verified_existing": skipped}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=Path("data/raw/ptb-xl/1.0.3"))
    parser.add_argument("--sampling-rate", type=int, choices=(100, 500), default=100)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, help="Download only the first N records for a smoke test")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--archive", action="store_true", help="Use the official 1.7 GB ZIP for faster bulk transfer")
    args = parser.parse_args()
    print(download(args.metadata_dir, args.sampling_rate, args.workers,
                   args.limit, args.timeout, args.retries, archive=args.archive))


if __name__ == "__main__":
    main()
