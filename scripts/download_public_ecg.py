#!/usr/bin/env python3
"""Resume and verify public ECG waveform downloads without changing experiment pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import requests

from scripts.download_ptbxl_waveforms import parse_checksums, sha256


ROOT = Path(__file__).resolve().parents[1]
COHORTS = {
    "georgia": ("challenge-2020", "1.0.2", "training/georgia/"),
    "cpsc_2018": ("challenge-2020", "1.0.2", "training/cpsc_2018/"),
    "cpsc_2018_extra": ("challenge-2020", "1.0.2", "training/cpsc_2018_extra/"),
    "chapman_shaoxing": ("challenge-2021", "1.0.3", "training/chapman_shaoxing/"),
}
_thread = threading.local()


def session() -> requests.Session:
    if not hasattr(_thread, "session"):
        _thread.session = requests.Session()
    return _thread.session


def fetch(url: str, target: Path, algorithm: str, digest: str, retries: int = 5) -> bool:
    """Return whether bytes were fetched; retain a partial file across interruptions."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and file_hash(target, algorithm) == digest:
        return False
    part = target.with_name(target.name + ".part")
    for attempt in range(retries + 1):
        try:
            offset = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with session().get(url, headers=headers, timeout=(15, 90), stream=True) as response:
                response.raise_for_status()
                if offset and response.status_code != 206:
                    offset = 0
                mode = "ab" if offset else "wb"
                with part.open(mode) as handle:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            handle.write(block)
            if file_hash(part, algorithm) != digest:
                # A complete transfer with a wrong digest cannot be safely resumed.
                part.unlink(missing_ok=True)
                raise ValueError(f"Checksum mismatch for {url}")
            os.replace(part, target)
            return True
        except (OSError, requests.RequestException, ValueError):
            if attempt == retries:
                raise
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError("Unreachable download retry state")


def file_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def physionet(cohort: str, workers: int, limit: int | None) -> None:
    project, version, prefix = COHORTS[cohort]
    base = f"https://physionet.org/files/{project}/{version}"
    root = ROOT / "data/raw" / project / version
    manifest = root / "SHA256SUMS.txt"
    if not manifest.is_file():
        # The official manifest is itself saved as acquisition evidence.
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with session().get(base + "/SHA256SUMS.txt", timeout=(15, 90)) as response:
            response.raise_for_status()
            manifest.write_bytes(response.content)
    checksums = parse_checksums(manifest.read_text())
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
    receipt_path = ROOT / "data/acquisition" / f"{cohort}.json"
    receipt = {"dataset": cohort, "source": base, "source_version": version,
               "source_manifest": str(manifest.relative_to(ROOT)),
               "source_manifest_sha256": sha256(manifest),
               "local_root": str(root.relative_to(ROOT)),
               "expected_records": len(stems), "expected_files": len(files),
               "verified_files": 0, "state": "running",
               "selection": "all official records" if not limit else f"first {limit} sorted records"}
    save_json(receipt_path, receipt)

    def one(name: str) -> bool:
        return fetch(base + "/" + name, root / name, "sha256", checksums[name])

    transferred = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, name): name for name in files}
        for number, future in enumerate(as_completed(futures), 1):
            transferred += int(future.result())
            if number % 200 == 0 or number == len(files):
                receipt.update(verified_files=number, transferred_files=transferred)
                save_json(receipt_path, receipt)
                print(f"{cohort}: verified {number}/{len(files)} files", flush=True)
    receipt["state"] = "complete"
    save_json(receipt_path, receipt)


def code15(limit: int | None) -> None:
    api = "https://zenodo.org/api/records/4916206"
    with session().get(api, timeout=(15, 90)) as response:
        response.raise_for_status()
        metadata = response.json()
    if str(metadata.get("id")) != "4916206":
        raise ValueError("Unexpected Zenodo record")
    files = metadata["files"]
    archives = sorted((f for f in files if f["key"].startswith("exams_part")
                       and f["key"].endswith(".zip")),
                      key=lambda f: int(f["key"].removeprefix("exams_part").removesuffix(".zip")))
    if len(archives) != 18:
        raise ValueError(f"Expected 18 CODE-15% archives, found {len(archives)}")
    metadata_files = [f for f in files if f["key"] == "exams.csv"]
    if len(metadata_files) != 1:
        raise ValueError("CODE-15% exams.csv missing")
    selected = metadata_files + archives[:limit] if limit is not None else metadata_files + archives
    root = ROOT / "data/raw/code-15pct/zenodo-4916206"
    receipt_path = ROOT / "data/acquisition/code_15pct.json"
    receipt = {"dataset": "CODE-15%", "source": "https://zenodo.org/records/4916206",
               "source_api": api, "record_id": 4916206,
               "local_root": str(root.relative_to(ROOT)),
               "expected_archives": len(archives), "selected_archives": len(selected) - 1,
               "verified_files": [], "state": "running",
               "selection": "all archives" if limit is None else f"first {limit} numbered archives"}
    save_json(receipt_path, receipt)
    for entry in selected:
        name = entry["key"]
        if PurePosixPath(name).name != name:
            raise ValueError("Unsafe Zenodo file name")
        algorithm, digest = entry["checksum"].split(":", 1)
        if algorithm not in ("md5", "sha256"):
            raise ValueError(f"Unsupported Zenodo checksum: {algorithm}")
        url = entry["links"]["self"]
        if not url.startswith("https://zenodo.org/"):
            raise ValueError("Unexpected Zenodo download URL")
        fetch(url, root / name, algorithm, digest)
        receipt["verified_files"].append({"name": name, "size": entry["size"],
                                          "checksum": entry["checksum"], "url": url})
        save_json(receipt_path, receipt)
        print(f"CODE-15%: verified {name} ({len(receipt['verified_files'])}/{len(selected)})", flush=True)
    receipt["state"] = "complete" if limit is None else "partial_verified"
    save_json(receipt_path, receipt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=(*COHORTS, "code_15pct"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, help="First N records/archives for a pilot; CODE permits zero for metadata only")
    args = parser.parse_args()
    if not 1 <= args.workers <= 16 or (args.limit is not None and args.limit < int(args.dataset != "code_15pct")):
        parser.error("workers must be 1–16; limit must be nonnegative for CODE or positive otherwise")
    if args.dataset == "code_15pct":
        code15(args.limit)
    else:
        physionet(args.dataset, args.workers, args.limit)


if __name__ == "__main__":
    main()
