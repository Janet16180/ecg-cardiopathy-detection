#!/usr/bin/env python3
"""Audit and index public WFDB cohorts without modifying raw ECG files.

The output is a manifest of canonical ten-second views plus explicit exclusions.
`ecg_experiment.public_sources.load_view` implements the same read-only
transformation for downstream consumers.
No labels are mapped to the PTB-XL abnormality proxy here.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import wfdb

from ecg_experiment import public_sources
from ecg_experiment.downloads import parse_checksums
from ecg_experiment.files import read_csv, sha256_file, write_csv_atomic
from ecg_experiment.public_sources import SOURCE, load_view, signal_sha256
from ecg_experiment.staging import published_directory
from ecg_experiment.waveforms import read_record
from ecg_experiment.wfdb_records import first_unverified_file, header_label_codes

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_FIELDS = ("ecg_id", "patient_id", "patient_identity_known", "source", "raw_dir",
                   "filename_hr", "source_samples", "window_start", "label_codes", "signal_sha256",
                   "qc_flags")
EXCLUSION_FIELDS = ("ecg_id", "source", "reason", "detail")
POLICIES = ("strict_10s", "ssl_center_crop")
PROGRESS_INTERVAL_RECORDS = 1000
PROGRESS_INTERVAL_REFERENCE = 5000


def _verified_ptb_rows(raw: Path, checksums: dict[str, str]) -> tuple[list[dict[str, str]], str]:
    """Check PTB-XL metadata and every waveform pair against the official checksums."""
    metadata = raw / "ptbxl_database.csv"
    fingerprint = sha256_file(metadata)
    if checksums.get("ptbxl_database.csv") != fingerprint:
        raise ValueError("PTB-XL metadata official checksum mismatch")
    rows = read_csv(metadata)
    if not rows or len({row["filename_hr"] for row in rows}) != len(rows):
        raise ValueError("Empty or duplicate PTB-XL reference records")
    for row in rows:
        for suffix in (".hea", ".dat"):
            name = row["filename_hr"] + suffix
            path = (raw / name).resolve()
            if not path.is_relative_to(raw.resolve()) or checksums.get(name) != sha256_file(path):
                raise ValueError(f"PTB-XL waveform official checksum mismatch: {name}")
    return rows, fingerprint


def _hashes_digest(hashes: list[str]) -> str:
    payload = json.dumps(hashes, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _cached_hashes(cache: Path, identity: dict[str, Any]) -> set[str] | None:
    """Return cached hashes only when the cache identity and digest both match."""
    if not cache.is_file():
        return None
    previous = json.loads(cache.read_text())
    if any(previous.get(key) != value for key, value in identity.items()):
        return None
    hashes = previous["hashes"]
    if len(hashes) != len(set(hashes)) or previous.get("hashes_sha256") != _hashes_digest(hashes):
        return None
    return set(hashes)


def _write_reference_cache(cache: Path, identity: dict[str, Any], hashes: set[str]) -> None:
    ordered = sorted(hashes)
    result = {**identity, "hashes": ordered, "hashes_sha256": _hashes_digest(ordered)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_name(f".{cache.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(result) + "\n")
    os.replace(temporary, cache)


def ptb_reference_hashes(cache: Path) -> set[str]:
    """
    Return decoded-signal hashes of every PTB-XL record for leakage checks.

    Current raw bytes are verified before a cached reference is trusted:
    metadata alone cannot detect a replaced waveform. The cache saves decoding,
    not official-checksum verification. This intentionally checks every PTB-XL
    reference, including held-out records, without accessing their labels.

    Parameters
    ----------
    cache : Path
        JSON cache of the reference, rebuilt when its identity differs.

    Returns
    -------
    set[str]
        Signal SHA-256 digests.

    Raises
    ------
    ValueError
        If any official checksum does not match.
    """
    raw = ROOT / "data/raw/ptb-xl/1.0.3"
    checksum_path = raw / "SHA256SUMS.txt"
    checksums = parse_checksums(checksum_path.read_text())
    rows, fingerprint = _verified_ptb_rows(raw, checksums)
    decoder_source = inspect.getsource(read_record) + inspect.getsource(signal_sha256)
    identity = {
        "schema_version": 2,
        "ptbxl_database_sha256": fingerprint,
        "official_checksums_sha256": sha256_file(checksum_path),
        "records_checked": len(rows),
        "decoder_sha256": hashlib.sha256(decoder_source.encode()).hexdigest(),
        "wfdb_version": wfdb.__version__, "numpy_version": np.__version__,
    }
    cached = _cached_hashes(cache, identity)
    if cached is not None:
        return cached
    hashes = set()
    for number, row in enumerate(rows, 1):
        hashes.add(signal_sha256(read_record(raw, row["filename_hr"])))
        if number % PROGRESS_INTERVAL_REFERENCE == 0:
            print(f"PTB-XL identity reference: {number}/{len(rows)}", flush=True)
    _write_reference_cache(cache, identity, hashes)
    return hashes


def selected_files(sources: list[str], limit: int | None,
                   allow_incomplete: bool) -> tuple[list[tuple[str, Path, str]], dict[str, Any]]:
    """
    List candidate records of each source from its official checksum manifest.

    Parameters
    ----------
    sources : list[str]
        Keys of ``SOURCE``.
    limit : int | None
        Keep only the first ``limit`` sorted records per source.
    allow_incomplete : bool
        Permit a source whose acquisition receipt is not complete.

    Returns
    -------
    tuple[list[tuple[str, Path, str]], dict[str, Any]]
        ``(source, raw_dir, stem)`` candidates and per-source provenance.

    Raises
    ------
    ValueError
        If a source download is incomplete and that is not allowed.
    """
    selected = []
    manifests = {}
    for source in sources:
        project, version, prefix = SOURCE[source]
        raw_dir = ROOT / "data/raw" / project / version
        receipt = ROOT / "data/acquisition" / f"{source}.json"
        if not allow_incomplete and json.loads(receipt.read_text())["state"] != "complete":
            raise ValueError(f"{source} download is not complete")
        checksum_file = raw_dir / "SHA256SUMS.txt"
        checksums = parse_checksums(checksum_file.read_text())
        stems = sorted(name[:-4] for name in checksums
                       if name.startswith(prefix + "/") and name.endswith(".hea"))
        if limit is not None:
            stems = stems[:limit]
        manifests[source] = {"source_url": f"https://physionet.org/files/{project}/{version}",
                             "checksum_file": str(checksum_file.relative_to(ROOT)),
                             "checksum_sha256": sha256_file(checksum_file),
                             "candidate_records": len(stems)}
        selected.extend((source, raw_dir, stem) for stem in stems)
    return selected, manifests


def prepare(sources: list[str], policy: str, output_dir: Path,
            limit: int | None = None, allow_incomplete: bool = False) -> dict[str, Any]:
    """
    Audit public cohorts and publish the manifest directory atomically.

    Parameters
    ----------
    sources : list[str]
        Distinct keys of ``SOURCE``.
    policy : str
        ``"strict_10s"`` or ``"ssl_center_crop"``.
    output_dir : Path
        New directory outside ``data/raw``.
    limit : int | None
        First ``limit`` records per source, for a pilot.
    allow_incomplete : bool
        Permit sources still downloading.

    Returns
    -------
    dict[str, Any]
        Audit metadata, also written to ``metadata.json``.

    Raises
    ------
    ValueError
        If an argument is invalid or the output is inside ``data/raw``.
    FileExistsError
        If the output already exists.
    """
    if not sources or len(sources) != len(set(sources)) or set(sources) - set(SOURCE):
        raise ValueError("sources must be distinct known datasets")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if policy not in POLICIES:
        raise ValueError("Unknown policy")
    if output_dir.resolve().is_relative_to((ROOT / "data/raw").resolve()):
        raise ValueError("Processed outputs must be outside data/raw")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite published audit: {output_dir}")
    with published_directory(output_dir) as stage:
        return _prepare(sources, policy, stage, limit, allow_incomplete)


def _audit_record(source: str, raw_dir: Path, stem: str, policy: str, checksums: dict[str, str],
                  ptb_hashes: set[str], seen: set[str]) -> dict[str, Any]:
    """
    Return the manifest row of one accepted record.

    Raises
    ------
    ValueError
        Whose message starts with the exclusion reason code; the caller
        records the text before the first colon as the reason.
    """
    ecg_id = f"{source}:{Path(stem).stem}"
    if first_unverified_file(raw_dir, stem, (".hea", ".mat"), checksums) is not None:
        raise ValueError("official_checksum_or_missing_file")
    signal, start, samples, flags = load_view(raw_dir, stem, policy)
    digest = signal_sha256(signal)
    if digest in ptb_hashes:
        raise ValueError("exact_ptbxl_duplicate")
    if digest in seen:
        raise ValueError("exact_pool_duplicate")
    seen.add(digest)
    return {"ecg_id": ecg_id, "patient_id": ecg_id,
            "patient_identity_known": "false", "source": source,
            "raw_dir": str(raw_dir.resolve()), "filename_hr": stem,
            "source_samples": samples, "window_start": start,
            "label_codes": header_label_codes(raw_dir / (stem + ".hea")),
            "signal_sha256": digest, "qc_flags": flags}


def _audit_candidates(candidates: list[tuple[str, Path, str]], policy: str,
                      source_checksums: dict[str, dict[str, str]],
                      ptb_hashes: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, str]], Counter]:
    """Split candidates into accepted rows and exclusions, counting outcomes."""
    seen: set[str] = set()
    accepted = []
    excluded = []
    counts = Counter()
    for number, (source, raw_dir, stem) in enumerate(candidates, 1):
        try:
            row = _audit_record(source, raw_dir, stem, policy, source_checksums[source],
                                ptb_hashes, seen)
        except (ValueError, OSError, IndexError) as error:
            # Reason codes are published data: keep the text before the first colon.
            reason = str(error).split(":", 1)[0]
            excluded.append({"ecg_id": f"{source}:{Path(stem).stem}", "source": source,
                             "reason": reason, "detail": str(error)})
            counts[f"excluded_{reason}"] += 1
        else:
            accepted.append(row)
            counts[f"accepted_{source}"] += 1
            counts["manual_review_flags"] += bool(row["qc_flags"])
        if number % PROGRESS_INTERVAL_RECORDS == 0 or number == len(candidates):
            print(f"{policy}: audited {number}/{len(candidates)} records; accepted {len(accepted)}",
                  flush=True)
    return accepted, excluded, counts


def _prepare(sources: list[str], policy: str, output_dir: Path,
             limit: int | None, allow_incomplete: bool) -> dict[str, Any]:
    """Write the manifest, exclusions and metadata into a staging directory."""
    candidates, provenance = selected_files(sources, limit, allow_incomplete)
    if not candidates:
        raise ValueError("No candidates")
    reference_cache = ROOT / "outputs/data_quality/ptbxl_reference_hashes_v2.json"
    ptb_hashes = ptb_reference_hashes(reference_cache)
    source_checksums = {source: parse_checksums((ROOT / details["checksum_file"]).read_text())
                        for source, details in provenance.items()}
    accepted, excluded, counts = _audit_candidates(candidates, policy, source_checksums, ptb_hashes)
    manifest = output_dir / "manifest.csv"
    write_csv_atomic(manifest, accepted, MANIFEST_FIELDS)
    exclusions = output_dir / "exclusions.csv"
    write_csv_atomic(exclusions, excluded, EXCLUSION_FIELDS)
    result = {"schema_version": 2, "complete": True,
              "limit_per_source": limit, "allow_incomplete_acquisition": allow_incomplete,
              "ptb_reference_receipt_sha256": sha256_file(reference_cache),
              "policy": policy, "sources": sources, "provenance": provenance,
              "candidate_records": len(candidates), "accepted_records": len(accepted),
              "excluded_records": len(excluded), "counts": dict(counts),
              "ptbxl_reference_count": len(ptb_hashes),
              "patient_grouping": "Record-only surrogate for Challenge sources; not a verified patient ID",
              "label_rule": "Original Challenge SNOMED codes preserved, not mapped to PTB-XL proxy",
              "strict_rule": "500Hz, twelve named leads, physical mV, finite, 5000 samples, no constant lead",
              "ssl_crop_rule": "long records center-cropped to 5000 samples; "
                               "labels not valid for a cropped window without review",
              "manifest_sha256": sha256_file(manifest), "exclusions_sha256": sha256_file(exclusions),
              "preparation_source_sha256": sha256_file(Path(__file__)),
              "canonicalization_source_sha256": sha256_file(Path(public_sources.__file__))}
    (output_dir / "metadata.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    """Parse arguments, run the audit and print its metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=SOURCE, action="append", required=True)
    parser.add_argument("--policy", choices=POLICIES, default="strict_10s")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="First N records per source for a pilot")
    parser.add_argument("--allow-incomplete", action="store_true", help="Pilot a source still downloading")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    print(json.dumps(prepare(args.source, args.policy, args.output_dir,
                             args.limit, args.allow_incomplete), indent=2), flush=True)


if __name__ == "__main__":
    main()
