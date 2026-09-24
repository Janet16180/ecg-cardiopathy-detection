#!/usr/bin/env python3
"""Materialize audited Challenge ECG views into immutable, hashed NumPy shards.

This consumes a completed `prepare_public_ecg` manifest. Raw WFDB files are read
and checked against the official release checksums again; nothing under data/raw
is written. The output manifest describes each [12, 5000] float32 mV array and
keeps source annotations separate from permission to use them as window labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import uuid
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.download_ptbxl_waveforms import parse_checksums, sha256
from scripts.extract_pretrained import LEADS
from scripts.data.prepare_public_ecg import ROOT, SOURCE, load_view, signal_sha256


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _within(path: Path, directory: Path) -> bool:
    return path.resolve().is_relative_to(directory.resolve())


def check_output(output_dir: Path, input_dir: Path) -> None:
    raw = ROOT / "data/raw"
    resolved = output_dir.resolve()
    if resolved == raw.resolve() or _within(resolved, raw):
        raise ValueError("Output must be outside data/raw")
    if _within(resolved, input_dir) or _within(input_dir, resolved):
        raise ValueError("Output must not contain or overwrite the input manifest")
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")


def load_input(prepared_dir: Path, allow_incomplete: bool) -> tuple[dict, list[dict], list[dict], dict]:
    meta_file = prepared_dir / "metadata.json"
    manifest_file = prepared_dir / "manifest.csv"
    exclusions_file = prepared_dir / "exclusions.csv"
    meta = json.loads(meta_file.read_text())
    if meta["policy"] not in {"strict_10s", "ssl_center_crop"}:
        raise ValueError("Unsupported audited view policy")
    if sha256(manifest_file) != meta["manifest_sha256"]:
        raise ValueError("Audited manifest hash mismatch")
    if sha256(exclusions_file) != meta["exclusions_sha256"]:
        raise ValueError("Audited exclusions hash mismatch")
    rows = read_csv(manifest_file)
    exclusions = read_csv(exclusions_file)
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
    checksums = {}
    for source in meta["sources"]:
        if source not in SOURCE:
            raise ValueError(f"Unknown source: {source}")
        project, version, _ = SOURCE[source]
        provenance = meta["provenance"][source]
        checksum_path = (ROOT / provenance["checksum_file"]).resolve()
        expected_path = (ROOT / "data/raw" / project / version / "SHA256SUMS.txt").resolve()
        if checksum_path != expected_path or sha256(checksum_path) != provenance["checksum_sha256"]:
            raise ValueError(f"Official checksum manifest mismatch: {source}")
        receipt = ROOT / "data/acquisition" / f"{source}.json"
        if not allow_incomplete and json.loads(receipt.read_text())["state"] != "complete":
            raise ValueError(f"Source acquisition is incomplete: {source}")
        checksums[source] = parse_checksums(checksum_path.read_text())
    return meta, rows, exclusions, checksums


def conflicted_ids(rows: list[dict], exclusions: list[dict], comparisons: Path | None,
                   checksums: dict) -> set[str]:
    duplicate_rows = {row["ecg_id"]: row for row in exclusions
                      if row["reason"] == "exact_pool_duplicate"}
    duplicate_ids = set(duplicate_rows)
    if not duplicate_ids:
        return set()
    if comparisons is None or not comparisons.is_file():
        raise ValueError("Duplicate comparison table required for strict label safety")
    comparisons_rows = read_csv(comparisons)
    if {row["excluded_ecg_id"] for row in comparisons_rows} != duplicate_ids:
        raise ValueError("Duplicate comparison table does not cover every excluded copy")
    retained = {row["ecg_id"]: row for row in rows}
    conflicts = set()
    for item in comparisons_rows:
        kept_id = item["retained_ecg_id"]
        if kept_id not in retained or item["retained_label_codes"] != retained[kept_id]["label_codes"]:
            raise ValueError("Duplicate comparison retained record or labels mismatch")
        excluded = duplicate_rows[item["excluded_ecg_id"]]
        source = excluded["source"]
        project, version, prefix = SOURCE[source]
        raw_dir = ROOT / "data/raw" / project / version
        record_id = item["excluded_ecg_id"].split(":", 1)[1]
        headers = list((raw_dir / prefix).glob(f"*/{record_id}.hea"))
        if len(headers) != 1:
            raise ValueError(f"Ambiguous duplicate raw header: {item['excluded_ecg_id']}")
        header = headers[0]
        stem = str(header.relative_to(raw_dir).with_suffix(""))
        for suffix in (".hea", ".mat"):
            name = stem + suffix
            if checksums[source].get(name) != sha256(raw_dir / name):
                raise ValueError(f"Official duplicate file checksum mismatch: {name}")
        actual_label = next((line.split(":", 1)[1].strip()
                             for line in header.read_text().splitlines()
                             if line.startswith("# Dx:")), "")
        if item["excluded_label_codes"] != actual_label:
            raise ValueError(f"Duplicate comparison source annotation mismatch: {record_id}")
        duplicate_signal, _, _, _ = load_view(raw_dir, stem, "strict_10s")
        if signal_sha256(duplicate_signal) != retained[kept_id]["signal_sha256"]:
            raise ValueError(f"Duplicate comparison waveform mismatch: {record_id}")
        a = set(filter(None, item["excluded_label_codes"].split(",")))
        b = set(filter(None, item["retained_label_codes"].split(",")))
        if item["same_label_set"] != str(a == b).lower():
            raise ValueError("Duplicate comparison label-set flag mismatch")
        if a != b:
            conflicts.add(kept_id)
    return conflicts


def overlap_sets(other_manifest: Path | None) -> tuple[set[str], set[str]]:
    if other_manifest is None:
        return set(), set()
    rows = read_csv(other_manifest)
    return {row["ecg_id"] for row in rows}, {row["signal_sha256"] for row in rows}


def _write_shard(stage: Path, number: int, batch: list[np.ndarray]) -> dict:
    name = f"shard_{number:05d}.npy"
    destination = stage / name
    temporary = stage / f".{name}.tmp"
    with temporary.open("wb") as handle:
        np.save(handle, np.stack(batch).astype("<f4", copy=False), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return {"file": name, "records": len(batch), "sha256": sha256(destination),
            "bytes": destination.stat().st_size}


def _verify_row_contract(row: dict[str, str], signal: np.ndarray, policy: str) -> None:
    """Check the row's waveform, window, annotation and patient claims."""
    if (row["sampling_rate_hz"] != "500" or row["window_samples"] != "5000" or
            row["units"] != "mV" or row["lead_order"] != ",".join(LEADS)):
        raise ValueError("Materialized row signal contract mismatch")
    samples, start = int(row["source_samples"]), int(row["window_start"])
    expected_start = (samples - 5000) // 2 if policy == "ssl_center_crop" else 0
    if (samples < 5000 or start != expected_start or
            (policy == "strict_10s" and samples != 5000)):
        raise ValueError("Materialized row duration contract mismatch")
    if (row["endpoint_supervised_eligible"] != "false" or
            row["patient_independent_eval_eligible"] != "false" or
            row["patient_identity_known"] != "false"):
        raise ValueError("Unsupported supervised or patient identity eligibility")
    if policy == "ssl_center_crop":
        expected_scope = "ssl_only_crop" if samples > 5000 else "ssl_only_view"
    elif row["duplicate_label_conflict"] == "true":
        expected_scope = "conflicting_duplicate_annotations"
    else:
        expected_scope = ("original_record_annotation_unmapped" if row["original_label_codes"]
                          else "missing_record_annotation")
    if (row["label_scope"] != expected_scope or row["record_annotation_available"] !=
            str(expected_scope == "original_record_annotation_unmapped").lower()):
        raise ValueError("Materialized annotation eligibility mismatch")
    if np.any(np.ptp(signal, axis=1) == 0):
        raise ValueError("Constant lead in materialized waveform")
    if (not np.isclose(float(row["min_lead_std_mv"]), signal.std(axis=1).min(), rtol=1e-6, atol=1e-8) or
            not np.isclose(float(row["max_abs_mv"]), np.abs(signal).max(), rtol=1e-6, atol=1e-8)):
        raise ValueError("Materialized waveform QC statistics mismatch")


def verify_materialized(output_dir: Path) -> dict:
    """Recheck the published receipt, every shard, and each canonical signal hash."""
    metadata = json.loads((output_dir / "metadata.json").read_text())
    if metadata.get("complete") is not True or metadata.get("schema_version") != 1:
        raise ValueError("Unpublished or unsupported materialization")
    contract = {"canonical_shape": [12, 5000], "dtype": "float32", "units": "mV",
                "sampling_rate_hz": 500, "lead_order": list(LEADS),
                "filtering": "none", "amplitude_scaling": "none"}
    if any(metadata.get(key) != value for key, value in contract.items()):
        raise ValueError("Materialized metadata signal contract mismatch")
    if metadata.get("policy") not in {"strict_10s", "ssl_center_crop"}:
        raise ValueError("Unsupported materialized policy")
    manifest = output_dir / "manifest.csv"
    if sha256(manifest) != metadata["manifest_sha256"]:
        raise ValueError("Materialized manifest hash mismatch")
    rows = read_csv(manifest)
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
            any(Path(name).name != name or
                not (output_dir / name).resolve().is_relative_to(output_dir.resolve()) for name in names)):
        raise ValueError("Unsafe or duplicate materialized shard reference")
    if sum(part["records"] for part in metadata["shards"]) != len(rows):
        raise ValueError("Materialized shard total mismatch")
    cursor = 0
    for shard in metadata["shards"]:
        path = output_dir / shard["file"]
        if sha256(path) != shard["sha256"] or path.stat().st_size != shard["bytes"]:
            raise ValueError(f"Materialized shard hash mismatch: {path}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.dtype != np.float32 or array.shape != (shard["records"], 12, 5000):
            raise ValueError(f"Materialized shard contract mismatch: {path}")
        for index in range(shard["records"]):
            row = rows[cursor]
            if row["shard"] != shard["file"] or int(row["shard_index"]) != index:
                raise ValueError("Materialized manifest shard index mismatch")
            if not np.isfinite(array[index]).all() or signal_sha256(array[index]) != row["signal_sha256"]:
                raise ValueError(f"Materialized signal hash mismatch: {row['ecg_id']}")
            _verify_row_contract(row, array[index], metadata["policy"])
            cursor += 1
    if cursor != len(rows):
        raise ValueError("Materialized shard total mismatch")
    return {"verified_records": cursor, "verified_shards": len(metadata["shards"]),
            "manifest_sha256": metadata["manifest_sha256"]}


def materialize(prepared_dir: Path, output_dir: Path, *, shard_size: int = 128,
                comparisons: Path | None = None, overlap_manifest: Path | None = None,
                allow_incomplete: bool = False) -> dict:
    if shard_size < 1:
        raise ValueError("shard_size must be positive")
    prepared_dir = prepared_dir.resolve()
    output_dir = output_dir.resolve()
    check_output(output_dir, prepared_dir)
    meta, rows, exclusions, checksums = load_input(prepared_dir, allow_incomplete)
    if meta["policy"] == "strict_10s":
        conflicts = conflicted_ids(rows, exclusions, comparisons, checksums)
    else:
        conflicts = set()  # The entire centered-window view is SSL-only.
    overlap_ids, overlap_hashes = overlap_sets(overlap_manifest)
    if overlap_manifest is not None and overlap_manifest.resolve() == prepared_dir / "manifest.csv":
        raise ValueError("Overlap manifest must describe a different view")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = output_dir.parent / f".{output_dir.name}.staging-{uuid.uuid4().hex}"
    stage.mkdir()
    counts = Counter()
    output_rows: list[dict] = []
    shards: list[dict] = []
    batch: list[np.ndarray] = []
    try:
        for index, row in enumerate(rows):
            source = row["source"]
            project, version, prefix = SOURCE[source]
            raw_dir = (ROOT / "data/raw" / project / version).resolve()
            if Path(row["raw_dir"]).resolve() != raw_dir:
                raise ValueError(f"Raw directory mismatch: {row['ecg_id']}")
            stem = row["filename_hr"]
            if not stem.startswith(prefix + "/") or not (raw_dir / stem).resolve().is_relative_to(raw_dir):
                raise ValueError(f"Raw stem outside expected source: {row['ecg_id']}")
            for suffix in (".hea", ".mat"):
                name = stem + suffix
                file = raw_dir / name
                if checksums[source].get(name) != sha256(file):
                    raise ValueError(f"Official raw file checksum mismatch: {name}")
            signal, start, samples, flags = load_view(raw_dir, stem, meta["policy"])
            if signal.dtype != np.float32 or signal.shape != (12, 5000) or not np.isfinite(signal).all():
                raise ValueError(f"Invalid canonical array: {row['ecg_id']}")
            digest = signal_sha256(signal)
            if (digest != row["signal_sha256"] or start != int(row["window_start"])
                    or samples != int(row["source_samples"]) or flags != row["qc_flags"]):
                raise ValueError(f"Materialized view differs from audited manifest: {row['ecg_id']}")
            long_crop = samples > 5000
            conflict = row["ecg_id"] in conflicts
            if meta["policy"] == "ssl_center_crop":
                label_scope = "ssl_only_crop" if long_crop else "ssl_only_view"
            elif conflict:
                label_scope = "conflicting_duplicate_annotations"
            elif row["label_codes"]:
                label_scope = "original_record_annotation_unmapped"
            else:
                label_scope = "missing_record_annotation"
            annotation_available = label_scope == "original_record_annotation_unmapped"
            shard_number = index // shard_size
            shard_index = index % shard_size
            batch.append(signal)
            output_rows.append({
                "ecg_id": row["ecg_id"], "source": source, "patient_id": row["patient_id"],
                "patient_identity_known": row["patient_identity_known"], "filename_hr": stem,
                "source_samples": samples, "window_start": start, "window_samples": 5000,
                "sampling_rate_hz": 500, "units": "mV", "lead_order": ",".join(LEADS),
                "original_label_codes": row["label_codes"], "label_scope": label_scope,
                "record_annotation_available": str(annotation_available).lower(),
                "endpoint_supervised_eligible": "false",
                "patient_independent_eval_eligible": "false",
                "duplicate_label_conflict": str(conflict).lower(),
                "overlap_by_record": str(row["ecg_id"] in overlap_ids).lower(),
                "overlap_by_signal": str(digest in overlap_hashes).lower(),
                "qc_flags": flags, "min_lead_std_mv": float(signal.std(axis=1).min()),
                "max_abs_mv": float(np.max(np.abs(signal))),
                "shard": f"shard_{shard_number:05d}.npy", "shard_index": shard_index,
                "signal_sha256": digest,
            })
            counts[f"source_{source}"] += 1
            counts[f"label_scope_{label_scope}"] += 1
            counts["label_conflict_retained"] += conflict
            counts["overlap_by_record"] += row["ecg_id"] in overlap_ids
            counts["overlap_by_signal"] += digest in overlap_hashes
            counts["qc_flagged"] += bool(flags)
            if len(batch) == shard_size or index + 1 == len(rows):
                shards.append(_write_shard(stage, shard_number, batch))
                batch.clear()
            if (index + 1) % 1000 == 0 or index + 1 == len(rows):
                print(f"{meta['policy']}: materialized {index + 1}/{len(rows)}", flush=True)
        manifest = stage / "manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(output_rows)
        result = {
            "schema_version": 1, "policy": meta["policy"], "complete": True,
            "canonical_shape": [12, 5000], "dtype": "float32", "units": "mV",
            "sampling_rate_hz": 500, "lead_order": list(LEADS),
            "filtering": "none", "amplitude_scaling": "none",
            "input_prepared_dir": str(prepared_dir),
            "input_metadata_sha256": sha256(prepared_dir / "metadata.json"),
            "input_manifest_sha256": meta["manifest_sha256"],
            "input_exclusions_sha256": meta["exclusions_sha256"],
            "input_preparation_source_sha256": meta["preparation_source_sha256"],
            "official_provenance": meta["provenance"],
            "duplicate_comparisons_sha256": sha256(comparisons) if meta["policy"] == "strict_10s" and comparisons and comparisons.is_file() else None,
            "overlap_manifest_sha256": sha256(overlap_manifest) if overlap_manifest else None,
            "candidate_records": meta["candidate_records"], "accepted_records": len(rows),
            "excluded_records": len(exclusions), "materialized_records": len(output_rows),
            "counts": dict(counts), "shard_size": shard_size, "shards": shards,
            "manifest_sha256": sha256(manifest),
            "materialization_source_sha256": sha256(Path(__file__)),
            "canonicalization_source_sha256": sha256(ROOT / "scripts/data/prepare_public_ecg.py") if (ROOT / "scripts/data/prepare_public_ecg.py").exists() else None,
            "label_policy": "Strict nonconflicting records retain original annotations for later adjudication; none are mapped to the PTB-XL endpoint or marked endpoint-supervised eligible. All centered-window views are SSL-only.",
            "patient_policy": "Challenge record IDs are not verified patient IDs; no patient-independent evaluation claim.",
            "overlap_policy": "Exact canonical signal and ECG-ID overlap with specified other view; near-duplicates and patient overlap remain unknown.",
        }
        (stage / "metadata.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        verify_materialized(stage)
        if output_dir.exists():
            raise FileExistsError(f"Output appeared during materialization: {output_dir}")
        os.rename(stage, output_dir)
        return result
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--duplicate-comparisons", type=Path)
    parser.add_argument("--overlap-manifest", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true", help="Explicitly permit an acquisition pilot")
    args = parser.parse_args()
    result = materialize(args.prepared_dir, args.output_dir, shard_size=args.shard_size,
                         comparisons=args.duplicate_comparisons,
                         overlap_manifest=args.overlap_manifest,
                         allow_incomplete=args.allow_incomplete)
    print(json.dumps({"output_dir": str(args.output_dir), "materialized_records": result["materialized_records"],
                      "counts": result["counts"]}, indent=2))


if __name__ == "__main__":
    main()
