#!/usr/bin/env python3
"""Resume and verify public ECG waveform downloads without changing experiment pools."""

from __future__ import annotations

import argparse
import hashlib
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any

import requests

from ecg_experiment.downloads import parse_checksums
from ecg_experiment.files import sha256_file, write_json_atomic

ROOT = Path(__file__).resolve().parents[1]
COHORTS = {
    "georgia": ("challenge-2020", "1.0.2", "training/georgia/"),
    "cpsc_2018": ("challenge-2020", "1.0.2", "training/cpsc_2018/"),
    "cpsc_2018_extra": ("challenge-2020", "1.0.2", "training/cpsc_2018_extra/"),
    "chapman_shaoxing": ("challenge-2021", "1.0.3", "training/chapman_shaoxing/"),
}
CODE15_RECORD_ID = "4916206"
CODE15_API = f"https://zenodo.org/api/records/{CODE15_RECORD_ID}"
CODE15_ARCHIVES = 18
REQUEST_TIMEOUT = (15, 90)
DOWNLOAD_RETRIES = 5
MAX_BACKOFF_SECONDS = 30
STREAM_CHUNK_BYTES = 1024 * 1024
HASH_CHUNK_BYTES = 4 * 1024 * 1024
RECEIPT_INTERVAL_FILES = 200
MAX_WORKERS = 16
_thread = threading.local()


def session() -> requests.Session:
    """
    Return this thread's reusable HTTP session.

    Returns
    -------
    requests.Session
        Session created on first use in the calling thread.
    """
    if not hasattr(_thread, "session"):
        _thread.session = requests.Session()
    return _thread.session


def _stream_to_part(url: str, part: Path) -> None:
    """Append to ``part`` with a Range request, restarting if the server ignores it."""
    offset = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with session().get(url, headers=headers, timeout=REQUEST_TIMEOUT, stream=True) as response:
        response.raise_for_status()
        if offset and response.status_code != 206:
            offset = 0
        mode = "ab" if offset else "wb"
        with part.open(mode) as handle:
            for block in response.iter_content(chunk_size=STREAM_CHUNK_BYTES):
                if block:
                    handle.write(block)


def fetch(url: str, target: Path, algorithm: str, digest: str, retries: int = DOWNLOAD_RETRIES) -> bool:
    """
    Download a file unless a verified copy exists, resuming a partial transfer.

    The partial file is kept across interruptions and network errors.

    Parameters
    ----------
    url : str
        Source URL.
    target : Path
        Final path.
    algorithm : str
        ``hashlib`` algorithm name of ``digest``.
    digest : str
        Expected hexadecimal digest.
    retries : int
        Additional attempts after a failure.

    Returns
    -------
    bool
        Whether bytes were fetched.

    Raises
    ------
    OSError, requests.RequestException, ValueError
        The last error once every attempt has failed.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and file_hash(target, algorithm) == digest:
        return False
    part = target.with_name(target.name + ".part")
    for attempt in range(retries + 1):
        try:
            _stream_to_part(url, part)
            if file_hash(part, algorithm) != digest:
                # A complete transfer with a wrong digest cannot be safely resumed.
                part.unlink(missing_ok=True)
                raise ValueError(f"Checksum mismatch for {url}")
            os.replace(part, target)
            return True
        except (OSError, requests.RequestException, ValueError):
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, MAX_BACKOFF_SECONDS))
    raise RuntimeError("Unreachable download retry state")


def file_hash(path: Path, algorithm: str) -> str:
    """
    Hash a file with any ``hashlib`` algorithm.

    Parameters
    ----------
    path : Path
        File to hash.
    algorithm : str
        Algorithm name such as ``"md5"`` or ``"sha256"``.

    Returns
    -------
    str
        Hexadecimal digest.
    """
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while block := handle.read(HASH_CHUNK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: dict[str, Any]) -> None:
    """
    Write a receipt atomically as indented, key-sorted JSON.

    Parameters
    ----------
    path : Path
        Destination file; parent directories are created.
    value : dict[str, Any]
        Receipt contents.
    """
    write_json_atomic(path, value, sort_keys=True)


def _official_manifest(base: str, root: Path) -> Path:
    """Return the local official manifest, saving it first as acquisition evidence."""
    manifest = root / "SHA256SUMS.txt"
    if not manifest.is_file():
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with session().get(base + "/SHA256SUMS.txt", timeout=REQUEST_TIMEOUT) as response:
            response.raise_for_status()
            manifest.write_bytes(response.content)
    return manifest


def _cohort_files(checksums: dict[str, str], cohort: str, prefix: str,
                  limit: int | None) -> tuple[list[str], list[str]]:
    """Select sorted record stems and their header/waveform file names."""
    stems = sorted({name[:-4] for name in checksums
                    if name.startswith(prefix) and name.endswith(".hea")})
    if limit:
        stems = stems[:limit]
    files = [stem + extension for stem in stems for extension in (".hea", ".mat")]
    if not stems or any(name not in checksums for name in files):
        raise ValueError(f"Missing official records or checksums for {cohort}")
    if any(PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
           for name in files):
        raise ValueError("Unsafe manifest path")
    return stems, files


def physionet(cohort: str, workers: int, limit: int | None) -> None:
    """
    Download and verify one PhysioNet Challenge cohort, updating its receipt.

    Parameters
    ----------
    cohort : str
        Key of ``COHORTS``.
    workers : int
        Parallel file downloads.
    limit : int | None
        Keep only the first ``limit`` sorted records; ``None`` or 0 keeps all.
    """
    project, version, prefix = COHORTS[cohort]
    base = f"https://physionet.org/files/{project}/{version}"
    root = ROOT / "data/raw" / project / version
    manifest = _official_manifest(base, root)
    checksums = parse_checksums(manifest.read_text())
    stems, files = _cohort_files(checksums, cohort, prefix, limit)
    receipt_path = ROOT / "data/acquisition" / f"{cohort}.json"
    receipt = {"dataset": cohort, "source": base, "source_version": version,
               "source_manifest": str(manifest.relative_to(ROOT)),
               "source_manifest_sha256": sha256_file(manifest),
               "local_root": str(root.relative_to(ROOT)),
               "expected_records": len(stems), "expected_files": len(files),
               "verified_files": 0, "state": "running",
               "selection": "all official records" if not limit else f"first {limit} sorted records"}
    save_json(receipt_path, receipt)

    def fetch_official(name: str) -> bool:
        return fetch(base + "/" + name, root / name, "sha256", checksums[name])

    transferred = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_official, name): name for name in files}
        for number, future in enumerate(as_completed(futures), 1):
            transferred += int(future.result())
            if number % RECEIPT_INTERVAL_FILES == 0 or number == len(files):
                receipt.update(verified_files=number, transferred_files=transferred)
                save_json(receipt_path, receipt)
                print(f"{cohort}: verified {number}/{len(files)} files", flush=True)
    receipt["state"] = "complete"
    save_json(receipt_path, receipt)


def _archive_number(entry: dict[str, Any]) -> int:
    return int(entry["key"].removeprefix("exams_part").removesuffix(".zip"))


def _code15_entries(metadata: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return the ``exams.csv`` entry and the numbered archives of the Zenodo record."""
    if str(metadata.get("id")) != CODE15_RECORD_ID:
        raise ValueError("Unexpected Zenodo record")
    files = metadata["files"]
    archives = sorted((f for f in files if f["key"].startswith("exams_part")
                       and f["key"].endswith(".zip")), key=_archive_number)
    if len(archives) != CODE15_ARCHIVES:
        raise ValueError(f"Expected 18 CODE-15% archives, found {len(archives)}")
    metadata_files = [f for f in files if f["key"] == "exams.csv"]
    if len(metadata_files) != 1:
        raise ValueError("CODE-15% exams.csv missing")
    return metadata_files, archives


def _checked_zenodo_entry(entry: dict[str, Any]) -> tuple[str, str, str, str]:
    """Return a Zenodo file's name, checksum algorithm, digest and URL after safety checks."""
    name = entry["key"]
    if PurePosixPath(name).name != name:
        raise ValueError("Unsafe Zenodo file name")
    algorithm, digest = entry["checksum"].split(":", 1)
    if algorithm not in ("md5", "sha256"):
        raise ValueError(f"Unsupported Zenodo checksum: {algorithm}")
    url = entry["links"]["self"]
    if not url.startswith("https://zenodo.org/"):
        raise ValueError("Unexpected Zenodo download URL")
    return name, algorithm, digest, url


def code15(limit: int | None) -> None:
    """
    Download and verify CODE-15% metadata and archives, updating the receipt.

    Parameters
    ----------
    limit : int | None
        Download only the first ``limit`` numbered archives; 0 means metadata only.
    """
    with session().get(CODE15_API, timeout=REQUEST_TIMEOUT) as response:
        response.raise_for_status()
        metadata = response.json()
    metadata_files, archives = _code15_entries(metadata)
    selected = metadata_files + archives[:limit] if limit is not None else metadata_files + archives
    root = ROOT / "data/raw/code-15pct/zenodo-4916206"
    receipt_path = ROOT / "data/acquisition/code_15pct.json"
    receipt = {"dataset": "CODE-15%", "source": "https://zenodo.org/records/4916206",
               "source_api": CODE15_API, "record_id": int(CODE15_RECORD_ID),
               "local_root": str(root.relative_to(ROOT)),
               "expected_archives": len(archives), "selected_archives": len(selected) - 1,
               "verified_files": [], "state": "running",
               "selection": "all archives" if limit is None else f"first {limit} numbered archives"}
    save_json(receipt_path, receipt)
    for entry in selected:
        name, algorithm, digest, url = _checked_zenodo_entry(entry)
        fetch(url, root / name, algorithm, digest)
        receipt["verified_files"].append({"name": name, "size": entry["size"],
                                          "checksum": entry["checksum"], "url": url})
        save_json(receipt_path, receipt)
        print(f"CODE-15%: verified {name} ({len(receipt['verified_files'])}/{len(selected)})", flush=True)
    receipt["state"] = "complete" if limit is None else "partial_verified"
    save_json(receipt_path, receipt)


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
    parser.add_argument("dataset", choices=(*COHORTS, "code_15pct"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int,
                        help="First N records/archives for a pilot; CODE permits zero for metadata only")
    args = parser.parse_args(argv)
    minimum_limit = int(args.dataset != "code_15pct")
    if not 1 <= args.workers <= MAX_WORKERS or (args.limit is not None and args.limit < minimum_limit):
        parser.error("workers must be 1–16; limit must be nonnegative for CODE or positive otherwise")
    return args


def main(argv: list[str] | None = None) -> None:
    """
    Parse arguments and download the requested dataset.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or ``None`` for ``sys.argv``.
    """
    args = parse_args(argv)
    if args.dataset == "code_15pct":
        code15(args.limit)
    else:
        physionet(args.dataset, args.workers, args.limit)


if __name__ == "__main__":
    main()
