"""Recover frozen PTB-XL training waveform quality without opening held-out signals."""

from __future__ import annotations

import csv
import hashlib
import resource
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import ROOT
from .downloads import parse_checksums
from .ecg_quality import assess
from .files import sha256_file, write_json_atomic
from .paths import to_stored
from .public_sources import signal_sha256
from .waveforms import read_record

RAW = Path("data/raw/ptb-xl/1.0.3")
RECOVERY = Path("outputs/cohort25_recovery_20261028/ptb")
MANIFEST = RECOVERY / "seed42_fraction1/all_train_ssl.csv"
EXPECTED_MANIFEST_MD5 = "82d391124fdf28d5ee2a55d29e5dcf2c"
SOURCE_FILES = (
    "ecg_experiment/ptb_training_recovery.py", "ecg_experiment/waveforms.py",
    "ecg_experiment/ecg_quality.py", "ecg_experiment/public_sources.py",
    "ecg_experiment/downloads.py", "ecg_experiment/paths.py",
    "scripts/data/recover_ptb_training_quality.py",
)
FIELDS = ("record_id", "patient_id", "path", "strat_fold", "signal_sha256", "exclusion_reasons",
          "review_flags", "header_sha256", "data_sha256")


def stored(path: Path) -> str:
    """Return the shared input root's relative name through the receipt path helper."""
    return to_stored(ROOT / path)


def training_rows(input_root: Path) -> tuple[list[dict[str, str]], dict[str, str]]:
    """
    Verify the frozen training manifest and every selected record's official training fold.

    Parameters
    ----------
    input_root : Path
        Checkout holding local raw data and recovered manifests.

    Returns
    -------
    tuple[list[dict[str, str]], dict[str, str]]
        Training rows and official raw-file checksums.
    """
    manifest = input_root / MANIFEST
    if hashlib.md5(manifest.read_bytes()).hexdigest() != EXPECTED_MANIFEST_MD5:
        raise ValueError("Frozen PTB training manifest MD5 mismatch")
    with manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 17418 or len({row["ecg_id"] for row in rows}) != 17418:
        raise ValueError("Expected exactly 17,418 unique training ECGs")
    with (input_root / RAW / "ptbxl_database.csv").open(newline="") as handle:
        metadata = {row["ecg_id"]: row for row in csv.DictReader(handle)}
    for row in rows:
        source = metadata[row["ecg_id"]]
        if not 1 <= int(source["strat_fold"]) <= 8:
            raise ValueError("Held-out PTB fold in training selection")
        for key in ("patient_id", "filename_hr"):
            if row[key] != source[key]:
                raise ValueError(f"PTB source metadata mismatch: {key}")
        relative = Path(row["filename_hr"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "records500":
            raise ValueError("Unsafe PTB training waveform path")
        row["strat_fold"] = source["strat_fold"]
    checksums = parse_checksums((input_root / RAW / "SHA256SUMS.txt").read_text())
    return rows, checksums


def assess_training_record(item: tuple[str, dict[str, str], str, str]) -> dict[str, str | float]:
    """
    Verify official raw bytes before and after decoding one approved training ECG.

    Parameters
    ----------
    item : tuple[str, dict[str, str], str, str]
        Raw directory, validated manifest row, expected header hash and expected data hash.

    Returns
    -------
    dict[str, str | float]
        Waveform identity, unchanged quality decisions and worker peak memory.
    """
    raw_dir, row, header_hash, data_hash = item
    root = Path(raw_dir)
    expected = {".hea": header_hash, ".dat": data_hash}
    for suffix, digest in expected.items():
        if sha256_file(root / (row["filename_hr"] + suffix)) != digest:
            raise ValueError(f"Official checksum mismatch before decoding: {row['ecg_id']}{suffix}")
    signal = read_record(root, row["filename_hr"])
    for suffix, digest in expected.items():
        if sha256_file(root / (row["filename_hr"] + suffix)) != digest:
            raise ValueError(f"Official checksum mismatch after decoding: {row['ecg_id']}{suffix}")
    exclusion, flags = assess(signal)
    return {
        "record_id": "ptbxl:" + row["ecg_id"], "patient_id": "ptbxl:" + row["patient_id"],
        "path": stored(RAW / row["filename_hr"]), "strat_fold": row["strat_fold"],
        "signal_sha256": signal_sha256(signal), "exclusion_reasons": ";".join(exclusion),
        "review_flags": ";".join(flags), "header_sha256": header_hash, "data_sha256": data_hash,
        "worker_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def recover(input_root: Path, output: Path, workers: int, limit: int | None) -> dict[str, object]:
    """
    Assess a profile or the full verified training selection and write a new local receipt.

    Parameters
    ----------
    input_root : Path
        Checkout holding local data.
    output : Path
        New output directory, relative to input_root.
    workers : int
        Number of CPU workers, from one to four.
    limit : int or None
        First training records for a resource profile; None assesses all training records.

    Returns
    -------
    dict[str, object]
        Receipt of actual quality counts, input/code hashes and resources.
    """
    if workers not in range(1, 5) or (limit is not None and limit < 1):
        raise ValueError("Use one to four workers and a positive profile limit")
    if output.is_absolute() or ".." in output.parts:
        raise ValueError("Output must stay relative to the input root")
    destination = input_root / output
    if destination.exists():
        raise FileExistsError(destination)
    start = time.monotonic()
    rows, checksums = training_rows(input_root)
    selected = rows if limit is None else rows[:limit]
    inputs = (MANIFEST, RAW / "ptbxl_database.csv", RAW / "SHA256SUMS.txt")
    input_hashes = {stored(path): sha256_file(input_root / path) for path in inputs}
    code_hashes = {stored(Path(name)): sha256_file(ROOT / name) for name in SOURCE_FILES}
    items = [(str(input_root / RAW), row, checksums[row["filename_hr"] + ".hea"],
              checksums[row["filename_hr"] + ".dat"]) for row in selected]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, result in enumerate(pool.map(assess_training_record, items, chunksize=32), 1):
            results.append(result)
            if index % 2048 == 0:
                print(f"Verified and assessed {index}/{len(selected)} training ECGs", flush=True)
    if input_hashes != {stored(path): sha256_file(input_root / path) for path in inputs}:
        raise ValueError("PTB recovery inputs changed during processing")
    destination.mkdir(parents=True)
    quality = destination / "record_quality.csv"
    with quality.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    exclusion_counts = Counter(reason for row in results for reason in
                               str(row["exclusion_reasons"]).split(";") if reason)
    review_counts = Counter(flag for row in results for flag in str(row["review_flags"]).split(";") if flag)
    passed = sum(row["exclusion_reasons"] == "" for row in results)
    receipt = {
        "complete": True, "scope": "training_quality" if limit is None else "resource_profile",
        "records": len(results), "passed": passed, "excluded": len(results) - passed,
        "expected_historical_passed": 17405,
        "historical_count_matches": passed == 17405 if limit is None else None,
        "exclusion_counts": dict(exclusion_counts), "review_counts": dict(review_counts),
        "input_sha256": input_hashes, "source_sha256": code_hashes,
        "manifest_md5": EXPECTED_MANIFEST_MD5,
        "quality_path": stored(output / quality.name), "quality_sha256": sha256_file(quality),
        "workers": workers, "wall_seconds": time.monotonic() - start,
        "parent_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "worker_peak_rss_kib": max(row["worker_peak_rss_kib"] for row in results),
        "child_user_seconds": resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime,
        "child_system_seconds": resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,
        "protected_waveforms_read": 0,
        "note": "Original training selection, official checksums, decoder and quality policy; "
                "no model scores.",
    }
    write_json_atomic(destination / "receipt.json", receipt, sort_keys=True)
    return receipt
