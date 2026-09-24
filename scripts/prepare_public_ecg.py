#!/usr/bin/env python3
"""Audit and index public WFDB cohorts without modifying raw ECG files.

The output is a manifest of canonical ten-second views plus explicit exclusions.
`load_view` implements the same read-only transformation for downstream consumers.
No labels are mapped to the PTB-XL abnormality proxy here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import shutil
import uuid
from collections import Counter
from pathlib import Path

import numpy as np
import wfdb

from scripts.download_ptbxl_waveforms import parse_checksums, sha256
from scripts.extract_pretrained import LEADS, read_record


ROOT = Path(__file__).resolve().parents[1]
SOURCE = {
    "georgia": ("challenge-2020", "1.0.2", "training/georgia"),
    "cpsc_2018": ("challenge-2020", "1.0.2", "training/cpsc_2018"),
    "cpsc_2018_extra": ("challenge-2020", "1.0.2", "training/cpsc_2018_extra"),
    "chapman_shaoxing": ("challenge-2021", "1.0.3", "training/chapman_shaoxing"),
}
MANIFEST_FIELDS = ("ecg_id", "patient_id", "patient_identity_known", "source", "raw_dir",
                   "filename_hr", "source_samples", "window_start", "label_codes", "signal_sha256",
                   "qc_flags")
EXCLUSION_FIELDS = ("ecg_id", "source", "reason", "detail")


def signal_sha256(signal: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(signal, dtype="<f4").tobytes()).hexdigest()


def load_view(raw_dir: Path, stem: str, policy: str) -> tuple[np.ndarray, int, int, str]:
    """Return canonical [12,5000] physical-mV float32, start, full length, QC flags."""
    if policy not in {"strict_10s", "ssl_center_crop"}:
        raise ValueError("unknown_policy")
    path = (raw_dir / stem).resolve()
    if not path.is_relative_to(raw_dir.resolve()):
        raise ValueError("unsafe_path")
    record = wfdb.rdrecord(str(path))
    names = [name.upper() for name in record.sig_name]
    if record.fs != 500 or len(names) != 12 or set(names) != {x.upper() for x in LEADS}:
        raise ValueError("lead_or_rate_contract")
    if record.units != ["mV"] * 12:
        raise ValueError("units_contract")
    waveform = np.asarray(record.p_signal, dtype=np.float32)
    if waveform.ndim != 2 or waveform.shape[1] != 12:
        raise ValueError("nonfinite_or_shape")
    samples = waveform.shape[0]
    if samples < 5000 or (policy == "strict_10s" and samples != 5000):
        raise ValueError("duration_contract")
    start = (samples - 5000) // 2 if policy == "ssl_center_crop" else 0
    signal = waveform[start:start + 5000, [names.index(x.upper()) for x in LEADS]].T.copy()
    if signal.shape != (12, 5000) or not np.isfinite(signal).all():
        raise ValueError("nonfinite_or_shape")
    spread = np.ptp(signal, axis=1)
    if np.any(spread == 0):
        raise ValueError("constant_lead")
    flags = []
    if np.any(signal.std(axis=1) < 0.01):
        flags.append("near_flat_lead_review")
    if np.max(np.abs(signal)) > 10:
        flags.append("amplitude_over_10mV_review")
    return signal, start, samples, ";".join(flags)


def ptb_reference_hashes(cache: Path) -> set[str]:
    """Verify *current* raw bytes before trusting a cached leakage reference.

    Metadata alone cannot detect a replaced waveform. The cache saves decoding,
    not official-checksum verification. This intentionally checks every PTB-XL
    reference, including held-out records, without accessing their labels.
    """
    raw = ROOT / "data/raw/ptb-xl/1.0.3"
    metadata = raw / "ptbxl_database.csv"
    checksum_path = raw / "SHA256SUMS.txt"
    checksums = parse_checksums(checksum_path.read_text())
    fingerprint = sha256(metadata)
    if checksums.get("ptbxl_database.csv") != fingerprint:
        raise ValueError("PTB-XL metadata official checksum mismatch")
    with metadata.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len({row["filename_hr"] for row in rows}) != len(rows):
        raise ValueError("Empty or duplicate PTB-XL reference records")
    for row in rows:
        for suffix in (".hea", ".dat"):
            name = row["filename_hr"] + suffix
            path = (raw / name).resolve()
            if not path.is_relative_to(raw.resolve()) or checksums.get(name) != sha256(path):
                raise ValueError(f"PTB-XL waveform official checksum mismatch: {name}")
    decoder_source = inspect.getsource(read_record) + inspect.getsource(signal_sha256)
    identity = {
        "schema_version": 2,
        "ptbxl_database_sha256": fingerprint,
        "official_checksums_sha256": sha256(checksum_path),
        "records_checked": len(rows),
        "decoder_sha256": hashlib.sha256(decoder_source.encode()).hexdigest(),
        "wfdb_version": wfdb.__version__, "numpy_version": np.__version__,
    }
    if cache.is_file():
        previous = json.loads(cache.read_text())
        if all(previous.get(key) == value for key, value in identity.items()):
            hashes = previous["hashes"]
            payload = json.dumps(hashes, separators=(",", ":")).encode()
            if (len(hashes) == len(set(hashes)) and
                    previous.get("hashes_sha256") == hashlib.sha256(payload).hexdigest()):
                return set(hashes)
    hashes = set()
    for number, row in enumerate(rows, 1):
        hashes.add(signal_sha256(read_record(raw, row["filename_hr"])))
        if number % 5000 == 0:
            print(f"PTB-XL identity reference: {number}/{len(rows)}", flush=True)
    ordered = sorted(hashes)
    hashes_payload = json.dumps(ordered, separators=(",", ":")).encode()
    result = {**identity, "hashes": ordered,
              "hashes_sha256": hashlib.sha256(hashes_payload).hexdigest()}
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_name(f".{cache.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(result) + "\n")
    os.replace(temporary, cache)
    return hashes


def selected_files(sources: list[str], limit: int | None,
                   allow_incomplete: bool) -> tuple[list[tuple[str, Path, str]], dict]:
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
                             "checksum_sha256": sha256(checksum_file),
                             "candidate_records": len(stems)}
        for stem in stems:
            selected.append((source, raw_dir, stem))
    return selected, manifests


def prepare(sources: list[str], policy: str, output_dir: Path,
            limit: int | None = None, allow_incomplete: bool = False) -> dict:
    if not sources or len(sources) != len(set(sources)) or set(sources) - set(SOURCE):
        raise ValueError("sources must be distinct known datasets")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    if policy not in ("strict_10s", "ssl_center_crop"):
        raise ValueError("Unknown policy")
    if output_dir.resolve().is_relative_to((ROOT / "data/raw").resolve()):
        raise ValueError("Processed outputs must be outside data/raw")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite published audit: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    stage.mkdir()
    try:
        result = _prepare(sources, policy, stage, limit, allow_incomplete)
        if output_dir.exists():
            raise FileExistsError(f"Output appeared during preparation: {output_dir}")
        os.rename(stage, output_dir)
        return result
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _prepare(sources: list[str], policy: str, output_dir: Path,
             limit: int | None, allow_incomplete: bool) -> dict:
    candidates, provenance = selected_files(sources, limit, allow_incomplete)
    if not candidates:
        raise ValueError("No candidates")
    reference_cache = ROOT / "outputs/data_quality/ptbxl_reference_hashes_v2.json"
    ptb_hashes = ptb_reference_hashes(reference_cache)
    source_checksums = {source: parse_checksums((ROOT / details["checksum_file"]).read_text())
                        for source, details in provenance.items()}
    seen = set()
    accepted = []
    excluded = []
    counts = Counter()
    for number, (source, raw_dir, stem) in enumerate(candidates, 1):
        ecg_id = f"{source}:{Path(stem).stem}"
        checksums = source_checksums[source]
        try:
            for extension in (".hea", ".mat"):
                name = stem + extension
                file = raw_dir / name
                if name not in checksums or not file.is_file() or sha256(file) != checksums[name]:
                    raise ValueError("official_checksum_or_missing_file")
            signal, start, samples, flags = load_view(raw_dir, stem, policy)
            digest = signal_sha256(signal)
            if digest in ptb_hashes:
                raise ValueError("exact_ptbxl_duplicate")
            if digest in seen:
                raise ValueError("exact_pool_duplicate")
            seen.add(digest)
            header = (raw_dir / (stem + ".hea")).read_text(encoding="utf-8")
            label = next((line.split(":", 1)[1].strip() for line in header.splitlines()
                          if line.startswith("# Dx:")), "")
            accepted.append({"ecg_id": ecg_id, "patient_id": ecg_id,
                             "patient_identity_known": "false", "source": source,
                             "raw_dir": str(raw_dir.resolve()), "filename_hr": stem,
                             "source_samples": samples, "window_start": start,
                             "label_codes": label, "signal_sha256": digest, "qc_flags": flags})
            counts[f"accepted_{source}"] += 1
            counts["manual_review_flags"] += bool(flags)
        except (ValueError, OSError, IndexError) as error:
            reason = str(error).split(":", 1)[0]
            excluded.append({"ecg_id": ecg_id, "source": source,
                             "reason": reason, "detail": str(error)})
            counts[f"excluded_{reason}"] += 1
        if number % 1000 == 0 or number == len(candidates):
            print(f"{policy}: audited {number}/{len(candidates)} records; accepted {len(accepted)}", flush=True)
    manifest = output_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(accepted)
    exclusions = output_dir / "exclusions.csv"
    with exclusions.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXCLUSION_FIELDS)
        writer.writeheader()
        writer.writerows(excluded)
    result = {"schema_version": 2, "complete": True,
              "limit_per_source": limit, "allow_incomplete_acquisition": allow_incomplete,
              "ptb_reference_receipt_sha256": sha256(reference_cache),
              "policy": policy, "sources": sources, "provenance": provenance,
              "candidate_records": len(candidates), "accepted_records": len(accepted),
              "excluded_records": len(excluded), "counts": dict(counts),
              "ptbxl_reference_count": len(ptb_hashes),
              "patient_grouping": "Record-only surrogate for Challenge sources; not a verified patient ID",
              "label_rule": "Original Challenge SNOMED codes preserved, not mapped to PTB-XL proxy",
              "strict_rule": "500Hz, twelve named leads, physical mV, finite, 5000 samples, no constant lead",
              "ssl_crop_rule": "long records center-cropped to 5000 samples; labels not valid for a cropped window without review",
              "manifest_sha256": sha256(manifest), "exclusions_sha256": sha256(exclusions),
              "preparation_source_sha256": sha256(Path(__file__))}
    (output_dir / "metadata.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=SOURCE, action="append", required=True)
    parser.add_argument("--policy", choices=("strict_10s", "ssl_center_crop"), default="strict_10s")
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
