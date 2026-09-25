#!/usr/bin/env python3
"""Materialize audited Challenge ECG views into immutable, hashed NumPy shards.

This consumes a completed `prepare_public_ecg` manifest. Raw WFDB files are read
and checked against the official release checksums again; nothing under data/raw
is written. The output manifest describes each [12, 5000] float32 mV array and
keeps source annotations separate from permission to use them as window labels.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ecg_experiment.downloads import parse_checksums
from ecg_experiment.files import read_csv, sha256_file, write_csv_atomic
from ecg_experiment.public_sources import SOURCE, load_view, signal_sha256
from ecg_experiment.staging import published_directory
from ecg_experiment.waveforms import LEADS
from ecg_experiment.wfdb_records import first_unverified_file, header_label_codes

ROOT = Path(__file__).resolve().parents[2]
FIELDS = (
    "ecg_id", "source", "patient_id", "patient_identity_known", "filename_hr",
    "source_samples", "window_start", "window_samples", "sampling_rate_hz",
    "units", "lead_order", "original_label_codes", "label_scope",
    "record_annotation_available", "endpoint_supervised_eligible",
    "patient_independent_eval_eligible",
    "duplicate_label_conflict", "overlap_by_record", "overlap_by_signal",
    "qc_flags", "min_lead_std_mv", "max_abs_mv", "shard", "shard_index",
    "signal_sha256",
)
# Official SHA-256 digests keyed by source, then by release-relative file name.
Checksums = dict[str, dict[str, str]]
POLICIES = {"strict_10s", "ssl_center_crop"}
WINDOW_SAMPLES = 5000
SAMPLING_RATE_HZ = 500
CANONICAL_SHAPE = (12, WINDOW_SAMPLES)
PROGRESS_INTERVAL = 1000
SIGNAL_CONTRACT = {"canonical_shape": list(CANONICAL_SHAPE), "dtype": "float32", "units": "mV",
                   "sampling_rate_hz": SAMPLING_RATE_HZ, "lead_order": list(LEADS),
                   "filtering": "none", "amplitude_scaling": "none"}


def _within(path: Path, directory: Path) -> bool:
    return path.resolve().is_relative_to(directory.resolve())


def check_output(output_dir: Path, input_dir: Path) -> None:
    """
    Refuse outputs inside raw data, overlapping the input, or already present.

    Parameters
    ----------
    output_dir : Path
        Planned materialization directory.
    input_dir : Path
        Audited ``prepare_public_ecg`` directory.

    Raises
    ------
    ValueError
        If the output is inside ``data/raw`` or overlaps the input.
    FileExistsError
        If the output already exists.
    """
    raw = ROOT / "data/raw"
    resolved = output_dir.resolve()
    if resolved == raw.resolve() or _within(resolved, raw):
        raise ValueError("Output must be outside data/raw")
    if _within(resolved, input_dir) or _within(input_dir, resolved):
        raise ValueError("Output must not contain or overwrite the input manifest")
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")


def _check_audit_tables(meta: dict[str, Any], rows: list[dict[str, str]],
                        exclusions: list[dict[str, str]]) -> None:
    """Reconcile the audited manifest and exclusions with their metadata."""
    if len(rows) != meta["accepted_records"] or len(exclusions) != meta["excluded_records"]:
        raise ValueError("Audited row counts mismatch")
    if len(rows) + len(exclusions) != meta["candidate_records"]:
        raise ValueError("Accepted plus excluded does not equal candidate count")
    if len({row["ecg_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate accepted ECG ID")
    if len({row["signal_sha256"] for row in rows}) != len(rows):
        raise ValueError("Duplicate accepted waveform hash")
    if {row["source"] for row in rows} - set(meta["sources"]):
        raise ValueError("Manifest source not in audit metadata")


def _official_checksums(meta: dict[str, Any], allow_incomplete: bool) -> Checksums:
    """Load each source's official checksum manifest after checking its provenance."""
    checksums = {}
    for source in meta["sources"]:
        if source not in SOURCE:
            raise ValueError(f"Unknown source: {source}")
        project, version, _ = SOURCE[source]
        provenance = meta["provenance"][source]
        checksum_path = (ROOT / provenance["checksum_file"]).resolve()
        expected_path = (ROOT / "data/raw" / project / version / "SHA256SUMS.txt").resolve()
        if checksum_path != expected_path or sha256_file(checksum_path) != provenance["checksum_sha256"]:
            raise ValueError(f"Official checksum manifest mismatch: {source}")
        receipt = ROOT / "data/acquisition" / f"{source}.json"
        if not allow_incomplete and json.loads(receipt.read_text())["state"] != "complete":
            raise ValueError(f"Source acquisition is incomplete: {source}")
        checksums[source] = parse_checksums(checksum_path.read_text())
    return checksums


def load_input(prepared_dir: Path, allow_incomplete: bool
               ) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]], Checksums]:
    """
    Load and verify a published ``prepare_public_ecg`` audit.

    Parameters
    ----------
    prepared_dir : Path
        Audit directory with ``metadata.json``, ``manifest.csv`` and ``exclusions.csv``.
    allow_incomplete : bool
        Permit sources whose acquisition receipt is not complete.

    Returns
    -------
    tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]], Checksums]
        Metadata, accepted rows, exclusions, and official checksums per source.

    Raises
    ------
    ValueError
        If any hash, count, identity or provenance check fails.
    """
    meta = json.loads((prepared_dir / "metadata.json").read_text())
    manifest_file = prepared_dir / "manifest.csv"
    exclusions_file = prepared_dir / "exclusions.csv"
    if meta["policy"] not in POLICIES:
        raise ValueError("Unsupported audited view policy")
    if sha256_file(manifest_file) != meta["manifest_sha256"]:
        raise ValueError("Audited manifest hash mismatch")
    if sha256_file(exclusions_file) != meta["exclusions_sha256"]:
        raise ValueError("Audited exclusions hash mismatch")
    rows = read_csv(manifest_file)
    exclusions = read_csv(exclusions_file)
    _check_audit_tables(meta, rows, exclusions)
    return meta, rows, exclusions, _official_checksums(meta, allow_incomplete)


def _is_conflict(item: dict[str, str], excluded: dict[str, str], retained: dict[str, dict[str, str]],
                 checksums: Checksums) -> bool:
    """Verify one duplicate comparison against raw data; return whether labels differ."""
    kept_id = item["retained_ecg_id"]
    if kept_id not in retained or item["retained_label_codes"] != retained[kept_id]["label_codes"]:
        raise ValueError("Duplicate comparison retained record or labels mismatch")
    source = excluded["source"]
    project, version, prefix = SOURCE[source]
    raw_dir = ROOT / "data/raw" / project / version
    record_id = item["excluded_ecg_id"].split(":", 1)[1]
    headers = list((raw_dir / prefix).glob(f"*/{record_id}.hea"))
    if len(headers) != 1:
        raise ValueError(f"Ambiguous duplicate raw header: {item['excluded_ecg_id']}")
    stem = str(headers[0].relative_to(raw_dir).with_suffix(""))
    unverified = first_unverified_file(raw_dir, stem, (".hea", ".mat"), checksums[source])
    if unverified is not None:
        raise ValueError(f"Official duplicate file checksum mismatch: {unverified}")
    if item["excluded_label_codes"] != header_label_codes(headers[0]):
        raise ValueError(f"Duplicate comparison source annotation mismatch: {record_id}")
    duplicate_signal, _, _, _ = load_view(raw_dir, stem, "strict_10s")
    if signal_sha256(duplicate_signal) != retained[kept_id]["signal_sha256"]:
        raise ValueError(f"Duplicate comparison waveform mismatch: {record_id}")
    # Empty codes are dropped here, unlike the comparison audit; this is deliberate.
    excluded_codes = set(filter(None, item["excluded_label_codes"].split(",")))
    retained_codes = set(filter(None, item["retained_label_codes"].split(",")))
    if item["same_label_set"] != str(excluded_codes == retained_codes).lower():
        raise ValueError("Duplicate comparison label-set flag mismatch")
    return excluded_codes != retained_codes


def conflicted_ids(rows: list[dict[str, str]], exclusions: list[dict[str, str]], comparisons: Path | None,
                   checksums: Checksums) -> set[str]:
    """
    Find retained records whose exact duplicates carry different annotations.

    Parameters
    ----------
    rows : list[dict[str, str]]
        Accepted audit rows.
    exclusions : list[dict[str, str]]
        Audit exclusions.
    comparisons : Path | None
        Duplicate-label comparison table from ``audit_duplicate_labels``.
    checksums : Checksums
        Official checksums per source.

    Returns
    -------
    set[str]
        ECG IDs of retained records with a conflicting duplicate.

    Raises
    ------
    ValueError
        If the comparison table is missing, incomplete, or disagrees with raw data.
    """
    duplicate_rows = {row["ecg_id"]: row for row in exclusions
                      if row["reason"] == "exact_pool_duplicate"}
    if not duplicate_rows:
        return set()
    if comparisons is None or not comparisons.is_file():
        raise ValueError("Duplicate comparison table required for strict label safety")
    comparison_rows = read_csv(comparisons)
    if {row["excluded_ecg_id"] for row in comparison_rows} != set(duplicate_rows):
        raise ValueError("Duplicate comparison table does not cover every excluded copy")
    retained = {row["ecg_id"]: row for row in rows}
    return {item["retained_ecg_id"] for item in comparison_rows
            if _is_conflict(item, duplicate_rows[item["excluded_ecg_id"]], retained, checksums)}


def overlap_sets(other_manifest: Path | None) -> tuple[set[str], set[str]]:
    """
    Read record IDs and signal hashes of another view's manifest.

    Parameters
    ----------
    other_manifest : Path | None
        Manifest to compare with; ``None`` gives empty sets.

    Returns
    -------
    tuple[set[str], set[str]]
        ECG IDs and signal hashes.
    """
    if other_manifest is None:
        return set(), set()
    rows = read_csv(other_manifest)
    return {row["ecg_id"] for row in rows}, {row["signal_sha256"] for row in rows}


def _write_shard(stage: Path, number: int, batch: list[np.ndarray]) -> dict[str, Any]:
    name = f"shard_{number:05d}.npy"
    destination = stage / name
    temporary = stage / f".{name}.tmp"
    with temporary.open("wb") as handle:
        np.save(handle, np.stack(batch).astype("<f4", copy=False), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return {"file": name, "records": len(batch), "sha256": sha256_file(destination),
            "bytes": destination.stat().st_size}


def _label_scope(policy: str, samples: int, conflict: bool, label_codes: str) -> str:
    """Return how a record's source annotation may be used for this view."""
    if policy == "ssl_center_crop":
        scope = "ssl_only_crop" if samples > WINDOW_SAMPLES else "ssl_only_view"
    elif conflict:
        scope = "conflicting_duplicate_annotations"
    elif label_codes:
        scope = "original_record_annotation_unmapped"
    else:
        scope = "missing_record_annotation"
    return scope


def _verify_row_contract(row: dict[str, str], signal: np.ndarray, policy: str) -> None:
    """Check the row's waveform, window, annotation and patient claims."""
    if (row["sampling_rate_hz"] != str(SAMPLING_RATE_HZ) or row["window_samples"] != str(WINDOW_SAMPLES) or
            row["units"] != "mV" or row["lead_order"] != ",".join(LEADS)):
        raise ValueError("Materialized row signal contract mismatch")
    samples, start = int(row["source_samples"]), int(row["window_start"])
    expected_start = (samples - WINDOW_SAMPLES) // 2 if policy == "ssl_center_crop" else 0
    if (samples < WINDOW_SAMPLES or start != expected_start or
            (policy == "strict_10s" and samples != WINDOW_SAMPLES)):
        raise ValueError("Materialized row duration contract mismatch")
    if (row["endpoint_supervised_eligible"] != "false" or
            row["patient_independent_eval_eligible"] != "false" or
            row["patient_identity_known"] != "false"):
        raise ValueError("Unsupported supervised or patient identity eligibility")
    expected_scope = _label_scope(policy, samples, row["duplicate_label_conflict"] == "true",
                                  row["original_label_codes"])
    if (row["label_scope"] != expected_scope or row["record_annotation_available"] !=
            str(expected_scope == "original_record_annotation_unmapped").lower()):
        raise ValueError("Materialized annotation eligibility mismatch")
    if np.any(np.ptp(signal, axis=1) == 0):
        raise ValueError("Constant lead in materialized waveform")
    if (not np.isclose(float(row["min_lead_std_mv"]), signal.std(axis=1).min(), rtol=1e-6, atol=1e-8) or
            not np.isclose(float(row["max_abs_mv"]), np.abs(signal).max(), rtol=1e-6, atol=1e-8)):
        raise ValueError("Materialized waveform QC statistics mismatch")


def _check_published_rows(output_dir: Path, metadata: dict[str, Any], rows: list[dict[str, str]]) -> None:
    """Reconcile manifest rows and shard references with the published metadata."""
    if len(rows) != metadata["materialized_records"]:
        raise ValueError("Materialized record count mismatch")
    if (len(rows) != metadata["accepted_records"] or
            len(rows) + metadata["excluded_records"] != metadata["candidate_records"]):
        raise ValueError("Materialized accepted/excluded counts mismatch")
    if (len({row["ecg_id"] for row in rows}) != len(rows) or
            len({row["signal_sha256"] for row in rows}) != len(rows)):
        raise ValueError("Duplicate materialized record or waveform")
    names = [part["file"] for part in metadata["shards"]]
    if (len(set(names)) != len(names) or
            any(Path(name).name != name or not _within(output_dir / name, output_dir) for name in names)):
        raise ValueError("Unsafe or duplicate materialized shard reference")
    if sum(part["records"] for part in metadata["shards"]) != len(rows):
        raise ValueError("Materialized shard total mismatch")


def _verify_shard(output_dir: Path, shard: dict[str, Any], rows: list[dict[str, str]], policy: str) -> None:
    """Check one shard's bytes and every signal against its manifest rows."""
    path = output_dir / shard["file"]
    if sha256_file(path) != shard["sha256"] or path.stat().st_size != shard["bytes"]:
        raise ValueError(f"Materialized shard hash mismatch: {path}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.dtype != np.float32 or array.shape != (shard["records"], *CANONICAL_SHAPE):
        raise ValueError(f"Materialized shard contract mismatch: {path}")
    for index, row in enumerate(rows):
        if row["shard"] != shard["file"] or int(row["shard_index"]) != index:
            raise ValueError("Materialized manifest shard index mismatch")
        if not np.isfinite(array[index]).all() or signal_sha256(array[index]) != row["signal_sha256"]:
            raise ValueError(f"Materialized signal hash mismatch: {row['ecg_id']}")
        _verify_row_contract(row, array[index], policy)


def verify_materialized(output_dir: Path) -> dict[str, Any]:
    """
    Recheck the published receipt, every shard, and each canonical signal hash.

    Parameters
    ----------
    output_dir : Path
        Materialized directory, published or staged.

    Returns
    -------
    dict[str, Any]
        Verified record and shard counts and the manifest hash.

    Raises
    ------
    ValueError
        If any contract, count, hash or row claim fails.
    """
    metadata = json.loads((output_dir / "metadata.json").read_text())
    if metadata.get("complete") is not True or metadata.get("schema_version") != 1:
        raise ValueError("Unpublished or unsupported materialization")
    if any(metadata.get(key) != value for key, value in SIGNAL_CONTRACT.items()):
        raise ValueError("Materialized metadata signal contract mismatch")
    if metadata.get("policy") not in POLICIES:
        raise ValueError("Unsupported materialized policy")
    manifest = output_dir / "manifest.csv"
    if sha256_file(manifest) != metadata["manifest_sha256"]:
        raise ValueError("Materialized manifest hash mismatch")
    rows = read_csv(manifest)
    _check_published_rows(output_dir, metadata, rows)
    cursor = 0
    for shard in metadata["shards"]:
        _verify_shard(output_dir, shard, rows[cursor:cursor + shard["records"]], metadata["policy"])
        cursor += shard["records"]
    return {"verified_records": cursor, "verified_shards": len(metadata["shards"]),
            "manifest_sha256": metadata["manifest_sha256"]}


def _verified_view(row: dict[str, str], policy: str,
                   checksums: Checksums) -> tuple[np.ndarray, int, int, str]:
    """Re-read one audited record from checksum-verified raw files and match the audit."""
    source = row["source"]
    project, version, prefix = SOURCE[source]
    raw_dir = (ROOT / "data/raw" / project / version).resolve()
    if Path(row["raw_dir"]).resolve() != raw_dir:
        raise ValueError(f"Raw directory mismatch: {row['ecg_id']}")
    stem = row["filename_hr"]
    if not stem.startswith(prefix + "/") or not (raw_dir / stem).resolve().is_relative_to(raw_dir):
        raise ValueError(f"Raw stem outside expected source: {row['ecg_id']}")
    unverified = first_unverified_file(raw_dir, stem, (".hea", ".mat"), checksums[source])
    if unverified is not None:
        raise ValueError(f"Official raw file checksum mismatch: {unverified}")
    signal, start, samples, flags = load_view(raw_dir, stem, policy)
    if signal.dtype != np.float32 or signal.shape != CANONICAL_SHAPE or not np.isfinite(signal).all():
        raise ValueError(f"Invalid canonical array: {row['ecg_id']}")
    if (signal_sha256(signal) != row["signal_sha256"] or start != int(row["window_start"])
            or samples != int(row["source_samples"]) or flags != row["qc_flags"]):
        raise ValueError(f"Materialized view differs from audited manifest: {row['ecg_id']}")
    return signal, start, samples, flags


def _output_row(row: dict[str, str], signal: np.ndarray, start: int, samples: int, flags: str,
                label_scope: str, conflict: bool, overlap_ids: set[str], overlap_hashes: set[str],
                shard_number: int, shard_index: int) -> dict[str, Any]:
    """Build the published manifest row of one materialized record."""
    return {
        "ecg_id": row["ecg_id"], "source": row["source"], "patient_id": row["patient_id"],
        "patient_identity_known": row["patient_identity_known"], "filename_hr": row["filename_hr"],
        "source_samples": samples, "window_start": start, "window_samples": WINDOW_SAMPLES,
        "sampling_rate_hz": SAMPLING_RATE_HZ, "units": "mV", "lead_order": ",".join(LEADS),
        "original_label_codes": row["label_codes"], "label_scope": label_scope,
        "record_annotation_available": str(label_scope == "original_record_annotation_unmapped").lower(),
        "endpoint_supervised_eligible": "false",
        "patient_independent_eval_eligible": "false",
        "duplicate_label_conflict": str(conflict).lower(),
        "overlap_by_record": str(row["ecg_id"] in overlap_ids).lower(),
        "overlap_by_signal": str(row["signal_sha256"] in overlap_hashes).lower(),
        "qc_flags": flags, "min_lead_std_mv": float(signal.std(axis=1).min()),
        "max_abs_mv": float(np.max(np.abs(signal))),
        "shard": f"shard_{shard_number:05d}.npy", "shard_index": shard_index,
        "signal_sha256": row["signal_sha256"],
    }


def _materialize_rows(stage: Path, rows: list[dict[str, str]], policy: str,
                      checksums: Checksums, conflicts: set[str],
                      overlap: tuple[set[str], set[str]], shard_size: int
                      ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
    """Write every record into shards in manifest order; return rows, shards and counts."""
    overlap_ids, overlap_hashes = overlap
    counts = Counter()
    output_rows: list[dict[str, Any]] = []
    shards: list[dict[str, Any]] = []
    batch: list[np.ndarray] = []
    for index, row in enumerate(rows):
        signal, start, samples, flags = _verified_view(row, policy, checksums)
        conflict = row["ecg_id"] in conflicts
        label_scope = _label_scope(policy, samples, conflict, row["label_codes"])
        batch.append(signal)
        output_rows.append(_output_row(row, signal, start, samples, flags, label_scope, conflict,
                                       overlap_ids, overlap_hashes, index // shard_size, index % shard_size))
        counts[f"source_{row['source']}"] += 1
        counts[f"label_scope_{label_scope}"] += 1
        counts["label_conflict_retained"] += conflict
        counts["overlap_by_record"] += row["ecg_id"] in overlap_ids
        counts["overlap_by_signal"] += row["signal_sha256"] in overlap_hashes
        counts["qc_flagged"] += bool(flags)
        if len(batch) == shard_size or index + 1 == len(rows):
            shards.append(_write_shard(stage, index // shard_size, batch))
            batch.clear()
        if (index + 1) % PROGRESS_INTERVAL == 0 or index + 1 == len(rows):
            print(f"{policy}: materialized {index + 1}/{len(rows)}", flush=True)
    return output_rows, shards, counts


def materialize(prepared_dir: Path, output_dir: Path, *, shard_size: int = 128,
                comparisons: Path | None = None, overlap_manifest: Path | None = None,
                allow_incomplete: bool = False) -> dict[str, Any]:
    """
    Materialize an audited view as verified shards and publish it atomically.

    Parameters
    ----------
    prepared_dir : Path
        Published ``prepare_public_ecg`` audit.
    output_dir : Path
        New directory outside ``data/raw`` and the input.
    shard_size : int
        Records per shard.
    comparisons : Path | None
        Duplicate-label comparison table; required for a strict view with duplicates.
    overlap_manifest : Path | None
        Manifest of another view whose exact overlaps are flagged.
    allow_incomplete : bool
        Permit sources whose acquisition is not complete.

    Returns
    -------
    dict[str, Any]
        Metadata, also written to ``metadata.json``.

    Raises
    ------
    ValueError
        If an input check fails or a re-read view differs from the audit.
    FileExistsError
        If the output already exists.
    """
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    prepared_dir = prepared_dir.resolve()
    output_dir = output_dir.resolve()
    check_output(output_dir, prepared_dir)
    meta, rows, exclusions, checksums = load_input(prepared_dir, allow_incomplete)
    policy = meta["policy"]
    if policy == "strict_10s":
        conflicts = conflicted_ids(rows, exclusions, comparisons, checksums)
    else:
        conflicts = set()  # The entire centered-window view is SSL-only.
    overlap = overlap_sets(overlap_manifest)
    if overlap_manifest is not None and overlap_manifest.resolve() == prepared_dir / "manifest.csv":
        raise ValueError("Overlap manifest must describe a different view")

    with published_directory(output_dir) as stage:
        output_rows, shards, counts = _materialize_rows(stage, rows, policy, checksums, conflicts,
                                                        overlap, shard_size)
        manifest = stage / "manifest.csv"
        write_csv_atomic(manifest, output_rows, FIELDS)
        canonicalization_source = ROOT / "ecg_experiment/public_sources.py"
        uses_comparisons = policy == "strict_10s" and comparisons and comparisons.is_file()
        result = {
            "schema_version": 1, "policy": policy, "complete": True,
            **SIGNAL_CONTRACT,
            "input_prepared_dir": str(prepared_dir),
            "input_metadata_sha256": sha256_file(prepared_dir / "metadata.json"),
            "input_manifest_sha256": meta["manifest_sha256"],
            "input_exclusions_sha256": meta["exclusions_sha256"],
            "input_preparation_source_sha256": meta["preparation_source_sha256"],
            "official_provenance": meta["provenance"],
            "duplicate_comparisons_sha256": sha256_file(comparisons) if uses_comparisons else None,
            "overlap_manifest_sha256": sha256_file(overlap_manifest) if overlap_manifest else None,
            "candidate_records": meta["candidate_records"], "accepted_records": len(rows),
            "excluded_records": len(exclusions), "materialized_records": len(output_rows),
            "counts": dict(counts), "shard_size": shard_size, "shards": shards,
            "manifest_sha256": sha256_file(manifest),
            "materialization_source_sha256": sha256_file(Path(__file__)),
            "canonicalization_source_sha256": (sha256_file(canonicalization_source)
                                               if canonicalization_source.exists() else None),
            "label_policy": "Strict nonconflicting records retain original annotations for later "
                            "adjudication; none are mapped to the PTB-XL endpoint or marked "
                            "endpoint-supervised eligible. All centered-window views are SSL-only.",
            "patient_policy": "Challenge record IDs are not verified patient IDs; "
                              "no patient-independent evaluation claim.",
            "overlap_policy": "Exact canonical signal and ECG-ID overlap with specified other view; "
                              "near-duplicates and patient overlap remain unknown.",
        }
        (stage / "metadata.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        verify_materialized(stage)
    return result


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
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--duplicate-comparisons", type=Path)
    parser.add_argument("--overlap-manifest", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Explicitly permit an acquisition pilot")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """
    Parse arguments, materialize the view and print a short summary.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or ``None`` for ``sys.argv``.
    """
    args = parse_args(argv)
    result = materialize(args.prepared_dir, args.output_dir, shard_size=args.shard_size,
                         comparisons=args.duplicate_comparisons,
                         overlap_manifest=args.overlap_manifest,
                         allow_incomplete=args.allow_incomplete)
    summary = {"output_dir": str(args.output_dir), "materialized_records": result["materialized_records"],
               "counts": result["counts"]}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
