"""Download a checksum-verified Georgia pilot pool for unlabeled adaptation."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath
from typing import Any

import wfdb

from ecg_experiment.downloads import fetch_file, parse_checksums
from ecg_experiment.files import read_csv, sha256_file, write_csv_atomic, write_json_atomic
from ecg_experiment.public_sources import signal_sha256
from ecg_experiment.waveforms import read_record

ROOT = Path(__file__).resolve().parents[2]
BASE = "https://physionet.org/files/challenge-2020/1.0.2"
GEORGIA_PREFIX = "training/georgia/g1/"
MANIFEST_FIELDS = ["ecg_id", "patient_id", "raw_dir", "filename_hr", "source"]
TIMEOUT_SECONDS = 60
RETRIES = 3


def fetch_checksums(raw_dir: Path) -> tuple[Path, dict[str, str]]:
    """
    Download the official checksum manifest once and parse it.

    Parameters
    ----------
    raw_dir : Path
        Local Challenge 2020 release root.

    Returns
    -------
    tuple[Path, dict[str, str]]
        Manifest path and digests keyed by relative path.
    """
    checksum_path = raw_dir / "SHA256SUMS.txt"
    if not checksum_path.exists():
        fetch_file(BASE + "/SHA256SUMS.txt", checksum_path, TIMEOUT_SECONDS, RETRIES)
    return checksum_path, parse_checksums(checksum_path.read_text())


def georgia_stems(checksums: dict[str, str]) -> tuple[list[str], list[str]]:
    """
    List the Georgia g1 records and their header/waveform files.

    Parameters
    ----------
    checksums : dict[str, str]
        Official digests keyed by relative path.

    Returns
    -------
    tuple[list[str], list[str]]
        Sorted record stems and the ``.hea``/``.mat`` paths to download.

    Raises
    ------
    ValueError
        If no records are listed, or a file lacks a checksum or has an unsafe path.
    """
    stems = sorted(name[:-4] for name in checksums
                   if name.startswith(GEORGIA_PREFIX) and name.endswith(".hea"))
    if not stems:
        raise ValueError("Official checksums contain no Georgia g1 headers")
    paths = [stem + suffix for stem in stems for suffix in (".hea", ".mat")]
    for path in paths:
        if path not in checksums or ".." in PurePosixPath(path).parts:
            raise ValueError(f"Missing checksum or unsafe path: {path}")
    return stems, paths


def _download(name: str, raw_dir: Path, checksums: dict[str, str]) -> int:
    """Fetch one file unless a verified copy exists; return its size."""
    destination = raw_dir / name

    def verified(file: Path) -> bool:
        return sha256_file(file) == checksums[name]

    if not destination.exists() or not verified(destination):
        fetch_file(BASE + "/" + name, destination, TIMEOUT_SECONDS, RETRIES, verified)
    return destination.stat().st_size


def download_files(raw_dir: Path, paths: list[str], checksums: dict[str, str], workers: int) -> int:
    """
    Download and verify every listed file in parallel.

    Parameters
    ----------
    raw_dir : Path
        Local release root.
    paths : list[str]
        Relative paths to fetch.
    checksums : dict[str, str]
        Official digests keyed by relative path.
    workers : int
        Download threads.

    Returns
    -------
    int
        Total bytes of the verified files.
    """
    total_bytes = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_download, name, raw_dir, checksums) for name in paths]
        for number, future in enumerate(as_completed(futures), 1):
            total_bytes += future.result()
            if number % 200 == 0 or number == len(paths):
                print(f"Verified {number}/{len(paths)} Georgia files", flush=True)
    return total_bytes


def ptb_signal_hashes(ptbxl_dir: Path) -> tuple[set[str], int]:
    """
    Hash every decoded PTB-XL waveform, including held-out folds.

    This is an identity/integrity audit; no targets or reports are read.

    Parameters
    ----------
    ptbxl_dir : Path
        PTB-XL release root.

    Returns
    -------
    tuple[set[str], int]
        Signal hashes and the number of records checked.
    """
    rows = read_csv(ptbxl_dir / "ptbxl_database.csv")
    hashes = set()
    for number, row in enumerate(rows, 1):
        hashes.add(signal_sha256(read_record(ptbxl_dir, row["filename_hr"])))
        if number % 5000 == 0:
            print(f"Audited {number}/{len(rows)} PTB-XL waveform identities", flush=True)
    return hashes, len(rows)


def audit_records(raw_dir: Path, stems: list[str], ptb_hashes: set[str]
                  ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """
    Accept Georgia records that decode in mV and match no PTB-XL or earlier waveform.

    Parameters
    ----------
    raw_dir : Path
        Local release root.
    stems : list[str]
        Record stems in manifest order.
    ptb_hashes : set[str]
        Decoded PTB-XL waveform hashes.

    Returns
    -------
    tuple[list[dict[str, str]], list[dict[str, str]]]
        Accepted manifest rows and excluded records with reasons.
    """
    accepted, excluded, seen = [], [], set()
    for stem in stems:
        try:
            header = wfdb.rdheader(str(raw_dir / stem))
            if header.units != ["mV"] * 12:
                raise ValueError(f"Expected all leads in mV; found {header.units}")
            signal = read_record(raw_dir, stem)
        except (ValueError, OSError) as error:
            excluded.append({"record": stem, "reason": "input_contract", "detail": str(error)})
            continue
        digest = signal_sha256(signal)
        if digest in ptb_hashes or digest in seen:
            excluded.append({"record": stem, "reason": "exact_decoded_waveform_duplicate"})
            continue
        seen.add(digest)
        name = PurePosixPath(stem).name
        accepted.append({"ecg_id": "georgia:" + name, "patient_id": "georgia:record_" + name,
                         "raw_dir": str(raw_dir.resolve()), "filename_hr": stem, "source": "georgia"})
    return accepted, excluded


def _metadata(stems: list[str], accepted: list[dict[str, str]], excluded: list[dict[str, str]],
              total_bytes: int, checksum_path: Path, manifest: Path, ptb_count: int) -> dict[str, Any]:
    return {
        "source_url": BASE, "license": "CC BY 4.0",
        "selection": "all records in g1 folder; convenience pilot, not random/full cohort",
        "downloaded_records": len(stems), "accepted_records": len(accepted), "excluded": excluded,
        "downloaded_bytes": total_bytes, "checksums_sha256": sha256_file(checksum_path),
        "manifest_sha256": sha256_file(manifest),
        "signal_identity": "SHA256 of canonical lead-ordered float32 physical mV",
        "ptbxl_records_checked": ptb_count,
        "patient_identity": ("Not available; patient_id denotes record grouping only. "
                             "Same-person records may remain."),
        "labels": "No Georgia diagnoses or reports used for training or evaluation.",
        "pretraining_exposure": ("Released ECG-FM and HuBERT already include Georgia "
                                 "through PhysioNet pretraining."),
    }


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
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/challenge-2020/1.0.2")
    parser.add_argument("--ptbxl-dir", type=Path, default=ROOT / "data/raw/ptb-xl/1.0.3")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/georgia_ssl_g1")
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """
    Download, audit and write the Georgia SSL manifest and metadata.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or ``None`` for ``sys.argv``.
    """
    args = parse_args(argv)
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checksum_path, checksums = fetch_checksums(args.raw_dir)
    stems, paths = georgia_stems(checksums)
    total_bytes = download_files(args.raw_dir, paths, checksums, args.workers)
    # Exclude exact decoded-waveform matches to ANY PTB-XL fold, including held-out ECGs.
    ptb_hashes, ptb_count = ptb_signal_hashes(args.ptbxl_dir)
    accepted, excluded = audit_records(args.raw_dir, stems, ptb_hashes)
    manifest = args.output_dir / "ssl_manifest.csv"
    write_csv_atomic(manifest, accepted, MANIFEST_FIELDS)
    metadata = _metadata(stems, accepted, excluded, total_bytes, checksum_path, manifest, ptb_count)
    write_json_atomic(args.output_dir / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
