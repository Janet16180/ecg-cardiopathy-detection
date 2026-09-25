"""Read-only validation of completed data sources and frozen PTB splits."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from itertools import combinations
from pathlib import Path
from typing import Any

from .files import SHA256_HEX_LENGTH, sha256_file

CONFIG_SCHEMA_VERSION = 1
PTB_METADATA_MANIFEST = "data/raw/ptb-xl/1.0.3/SHA256SUMS.txt"
LABELED_SPLITS = ("full_labeled", "limited_labeled", "validation", "test")
DISJOINT_SPLITS = ("full_train", "validation", "test")
HELDOUT_SPLITS = ("development", "calibration", "test")


def _official_hashes(path: Path, prefix: str) -> dict[str, str]:
    """
    Read the official checksums of files under one release prefix.

    Parameters
    ----------
    path : Path
        Official ``SHA256SUMS.txt``.
    prefix : str
        Release-relative directory prefix; it is removed from the keys.

    Returns
    -------
    dict[str, str]
        Lowercase digest keyed by path relative to ``prefix``.

    Raises
    ------
    ValueError
        If a line is malformed, a path is unsafe or repeated, or nothing is
        listed under ``prefix``.
    """
    selected: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition(" ")
        relative = relative.lstrip(" *")
        if not separator or len(digest) != SHA256_HEX_LENGTH or not relative:
            raise ValueError(f"Invalid official checksum line: {line[:100]}")
        if not relative.startswith(prefix):
            continue
        local = relative.removeprefix(prefix)
        if not local or local in selected or Path(local).is_absolute() or ".." in Path(local).parts:
            raise ValueError(f"Invalid or duplicate source path: {relative}")
        selected[local] = digest.lower()
    if not selected:
        raise ValueError(f"No official checksum entries under {prefix}")
    return selected


def _file_inventory(source: Path) -> set[str]:
    """List files relative to their source directory."""
    return {str(path.relative_to(source)) for path in source.rglob("*") if path.is_file()}


def _reject_symlinks(source: Path, source_id: str) -> None:
    """Require source files to live in the verified source tree."""
    if any(path.is_symlink() for path in source.rglob("*")):
        raise ValueError(f"{source_id}: source contains symlinks")


def _record_stems(paths: Iterable[str], suffix: str) -> set[str]:
    """Identify records by removing a header or waveform file extension."""
    return {path.removesuffix(suffix) for path in paths if path.endswith(suffix)}


def _verify_checksums(
    source: Path, relatives: Iterable[str], expected: Mapping[str, str], source_id: str
) -> None:
    """Compare each source file with its official checksum."""
    for relative in relatives:
        if sha256_file(source / relative) != expected[relative]:
            raise ValueError(f"{source_id}: checksum mismatch: {relative}")


def _validate_ptb_source(root: Path, source: Path, spec: dict[str, Any]) -> dict[str, Any]:
    """Check PTB-XL record files against the frozen official manifest."""
    manifest = root / spec["official_manifest"]
    manifest_digest = sha256_file(manifest)
    if manifest_digest != spec["official_manifest_sha256"]:
        raise ValueError(f"{spec['id']}: PTB official manifest hash differs from frozen reference")

    expected = _official_hashes(manifest, spec["official_prefix"])
    if len(expected) != spec["expected_files"] or _file_inventory(source) != expected.keys():
        raise ValueError(f"{spec['id']}: PTB file inventory differs from official manifest")
    _reject_symlinks(source, spec["id"])

    headers = _record_stems(expected, ".hea")
    if headers != _record_stems(expected, ".dat") or len(headers) != spec["expected_records"]:
        raise ValueError(f"{spec['id']}: PTB waveform/header pair count differs from config")
    _verify_checksums(source, expected, expected, spec["id"])

    return {
        "id": spec["id"],
        "source_path": spec["path"],
        "state": "verified",
        "records": len(headers),
        "files": len(expected),
        "official_manifest_sha256": manifest_digest,
    }


def _check_challenge_receipt(receipt: dict[str, Any], manifest: Path, spec: dict[str, Any]) -> None:
    """Require a completed acquisition with matching counts and manifest provenance."""
    if receipt.get("state") != "complete":
        raise ValueError(f"{spec['id']}: acquisition is not complete")
    expected_counts = {
        "expected_files": spec["expected_files"],
        "verified_files": spec["expected_files"],
        "expected_records": spec["expected_records"],
    }
    for key, expected in expected_counts.items():
        if receipt.get(key) != expected:
            raise ValueError(f"{spec['id']}: receipt {key} differs from config")

    if receipt.get("source_manifest_sha256") != sha256_file(manifest):
        raise ValueError(f"{spec['id']}: official manifest hash differs from receipt")
    if receipt.get("source_manifest") != spec["official_manifest"]:
        raise ValueError(f"{spec['id']}: official manifest path differs from receipt")


def _validate_challenge_source(root: Path, source: Path, spec: dict[str, Any]) -> dict[str, Any]:
    """Check a completed Challenge acquisition against its receipt and official manifest."""
    receipt = json.loads((root / spec["receipt"]).read_text(encoding="utf-8"))
    manifest = root / spec["official_manifest"]
    _check_challenge_receipt(receipt, manifest, spec)

    expected = _official_hashes(manifest, spec["official_prefix"])
    waveform_entries = {path for path in expected if path.endswith((".hea", ".mat"))}
    if len(waveform_entries) != spec["expected_files"]:
        raise ValueError(f"{spec['id']}: official waveform file count differs from config")

    actual = _file_inventory(source)
    if actual != waveform_entries:
        raise ValueError(
            f"{spec['id']}: missing {len(waveform_entries - actual)}, "
            f"unexpected {len(actual - waveform_entries)} files"
        )
    _reject_symlinks(source, spec["id"])

    headers = _record_stems(waveform_entries, ".hea")
    waveforms = _record_stems(waveform_entries, ".mat")
    if headers != waveforms or len(headers) != spec["expected_records"]:
        raise ValueError(f"{spec['id']}: waveform/header pair count differs from receipt")
    _verify_checksums(source, sorted(waveform_entries), expected, spec["id"])

    return {
        "id": spec["id"],
        "source_path": spec["path"],
        "state": "verified",
        "records": len(headers),
        "waveform_files": len(waveform_entries),
        "files": len(actual),
        "official_manifest_sha256": receipt["source_manifest_sha256"],
    }


def validate_source(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    """
    Verify every file of one completed source against its official checksums.

    Parameters
    ----------
    root : Path
        Project root that the spec paths are relative to.
    spec : dict[str, Any]
        Source entry from the dataset config; ``kind`` is ``"ptbxl_records"``,
        ``"challenge_pairs"`` or absent (Challenge pairs).

    Returns
    -------
    dict[str, Any]
        Verification summary with record and file counts.

    Raises
    ------
    ValueError
        If the source is missing, symlinked, incomplete, differs from its
        manifest or receipt, or has an unsupported kind.
    """
    source = root / spec["path"]
    if not source.is_dir() or source.is_symlink():
        raise ValueError(f"{spec['id']}: source directory missing or symlinked")
    kind = spec.get("kind")
    if kind not in (None, "challenge_pairs", "ptbxl_records"):
        raise ValueError(f"Unsupported source kind: {kind}")
    validate = _validate_ptb_source if kind == "ptbxl_records" else _validate_challenge_source
    return validate(root, source, spec)


def load_split(path: Path) -> dict[str, tuple[str, str | None]]:
    """
    Read a frozen split manifest.

    Parameters
    ----------
    path : Path
        CSV with ``ecg_id``, ``patient_id`` and optionally ``target``.

    Returns
    -------
    dict[str, tuple[str, str | None]]
        ``(patient_id, target)`` keyed by ECG ID.

    Raises
    ------
    ValueError
        If identity columns are missing, an identity is empty or repeated, or
        the split is empty.
    """
    rows: dict[str, tuple[str, str | None]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"ecg_id", "patient_id"} <= set(reader.fieldnames):
            raise ValueError(f"Missing ECG or patient ID columns: {path}")

        for row in reader:
            record = row["ecg_id"].strip()
            patient = row["patient_id"].strip()
            if not record or not patient or record in rows:
                raise ValueError(f"Missing or duplicate identity in {path}")
            rows[record] = (patient, row.get("target"))

    if not rows:
        raise ValueError(f"Empty split: {path}")
    return rows


def _check_disjoint(sets: Mapping[str, Iterable[str]], names: Iterable[str], problem: str) -> None:
    """Reject overlap between any pair of the named record or patient groups."""
    for left, right in combinations(names, 2):
        if set(sets[left]) & set(sets[right]):
            raise ValueError(f"{problem} between {left} and {right}")


def _check_labels(splits: dict[str, dict[str, tuple[str, str | None]]]) -> None:
    """Require binary targets and label selections nested in the training split."""
    for name in LABELED_SPLITS:
        if any(target not in {"0", "1"} for _, target in splits[name].values()):
            raise ValueError(f"{name}: target must be binary and present")

    train = splits["full_train"]
    for name in ("full_labeled", "limited_labeled"):
        for record, identity in splits[name].items():
            if record not in train or train[record][0] != identity[0]:
                raise ValueError(
                    f"{name}: record missing from full training split or patient mismatch"
                )

    for record, identity in splits["limited_labeled"].items():
        if splits["full_labeled"].get(record) != identity:
            raise ValueError("Limited-label selection differs from full-label selection")


def _load_heldout(path: Path) -> dict[str, dict[str, str]]:
    """Read held-out references as patient keyed by record for each partition."""
    heldout: dict[str, dict[str, str]] = {name: {} for name in HELDOUT_SPLITS}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required_fields = {"record_id", "patient_id", "split"}
        if not reader.fieldnames or not required_fields <= set(reader.fieldnames):
            raise ValueError("Held-out reference columns are incomplete")

        for row in reader:
            name = row["split"]
            if name not in heldout:
                raise ValueError(f"Unknown held-out split: {name}")
            record = row["record_id"].removeprefix("ptbxl:")
            patient = row["patient_id"].removeprefix("ptbxl:")
            if not record or not patient or record in heldout[name]:
                raise ValueError("Missing or duplicate held-out identity")
            heldout[name][record] = patient
    return heldout


def _validate_heldout(
    path: Path, splits: dict[str, dict[str, tuple[str, str | None]]], train_patients: set[str]
) -> dict[str, dict[str, int]]:
    """Check that held-out references partition PTB validation/test without leakage."""
    heldout = _load_heldout(path)
    validation_records = set(heldout["development"]) | set(heldout["calibration"])
    if validation_records != set(splits["validation"]):
        raise ValueError("Development/calibration do not partition PTB validation")
    if set(heldout["test"]) != set(splits["test"]):
        raise ValueError("Held-out test differs from PTB test")
    _check_disjoint(heldout, HELDOUT_SPLITS, "Record overlap")
    if any(not rows for rows in heldout.values()):
        raise ValueError("An expected held-out partition is empty")

    for name, rows in heldout.items():
        reference = splits["test"] if name == "test" else splits["validation"]
        if any(reference[record][0] != patient for record, patient in rows.items()):
            raise ValueError(f"{name}: held-out patient mapping differs from PTB split")

    heldout_patients = {name: set(rows.values()) for name, rows in heldout.items()}
    _check_disjoint(heldout_patients, HELDOUT_SPLITS, "Patient leakage")
    if train_patients & set().union(*heldout_patients.values()):
        raise ValueError("Patient leakage between train and held-out references")

    return {
        "record_counts": {name: len(rows) for name, rows in heldout.items()},
        "patient_counts": {name: len(rows) for name, rows in heldout_patients.items()},
    }


def validate_splits(root: Path, paths: dict[str, str]) -> dict[str, Any]:
    """
    Audit frozen PTB splits for label validity, nesting and patient leakage.

    Parameters
    ----------
    root : Path
        Project root that the split paths are relative to.
    paths : dict[str, str]
        Split CSVs keyed by ``full_train``, ``full_labeled``,
        ``limited_labeled``, ``validation``, ``test`` and optionally
        ``heldout_references``.

    Returns
    -------
    dict[str, Any]
        Record and patient counts, the held-out partition summary (or
        ``None``) and zero patient leakage.

    Raises
    ------
    ValueError
        If any label, nesting, partition or disjointness check fails.
    """
    splits = {
        name: load_split(root / relative)
        for name, relative in paths.items()
        if name != "heldout_references"
    }
    _check_labels(splits)

    patients_by_split = {
        name: {patient for patient, _ in splits[name].values()}
        for name in DISJOINT_SPLITS
    }
    _check_disjoint(patients_by_split, DISJOINT_SPLITS, "Patient leakage")
    _check_disjoint(splits, DISJOINT_SPLITS, "Record overlap")

    heldout_details = None
    if "heldout_references" in paths:
        heldout_path = root / paths["heldout_references"]
        heldout_details = _validate_heldout(heldout_path, splits, patients_by_split["full_train"])

    return {
        "state": "verified",
        "record_counts": {key: len(value) for key, value in splits.items()},
        "patient_counts": {name: len(patients) for name, patients in patients_by_split.items()},
        "heldout_partition": heldout_details,
        "patient_leakage": 0,
    }


def _check_frozen_hashes(root: Path, settings: dict[str, Any]) -> None:
    """Compare PTB metadata and frozen split files with their recorded digests."""
    ptb_hashes = _official_hashes(root / PTB_METADATA_MANIFEST, "")
    for relative in settings["ptb_metadata"]:
        path = root / relative
        if path.name not in ptb_hashes or sha256_file(path) != ptb_hashes[path.name]:
            raise ValueError(f"PTB metadata checksum mismatch: {relative}")

    split_paths = settings["ptb_split_audit"]
    if settings["ptb_split_sha256"].keys() != split_paths.keys():
        raise ValueError("PTB split hash inventory differs from split paths")
    for name, relative in split_paths.items():
        if sha256_file(root / relative) != settings["ptb_split_sha256"][name]:
            raise ValueError(f"Frozen PTB split checksum mismatch: {name}")


def validate_datasets(root: Path, config: Path) -> dict[str, Any]:
    """
    Verify PTB metadata, frozen splits and every source listed in the dataset config.

    Parameters
    ----------
    root : Path
        Project root that config paths are relative to.
    config : Path
        Dataset config JSON.

    Returns
    -------
    dict[str, Any]
        Validation report with per-source summaries and the split audit.

    Raises
    ------
    ValueError
        If the schema is unsupported or any check fails.
    """
    settings = json.loads(config.read_text(encoding="utf-8"))
    if settings.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("Unsupported dataset config schema")

    _check_frozen_hashes(root, settings)
    sources = [validate_source(root, spec) for spec in settings["datasets"]]
    split_audit = validate_splits(root, settings["ptb_split_audit"])

    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "sources": sources,
        "ptb_metadata_files": len(settings["ptb_metadata"]),
        "ptb_split_audit": split_audit,
    }
