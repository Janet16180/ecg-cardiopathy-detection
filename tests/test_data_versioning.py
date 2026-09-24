"""Meaningful failure checks for the read-only data validation gate."""

import csv
import hashlib
import json
from pathlib import Path

import pytest

from ecg_experiment.data_validation import DatasetValidator, validate_source, validate_splits


def _split(path: Path, rows: list[tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("ecg_id", "patient_id", "target"))
        writer.writerows(rows)


def _split_fixture(root: Path) -> dict[str, str]:
    paths = {name: f"{name}.csv" for name in
             ("full_train", "full_labeled", "limited_labeled", "validation", "test")}
    _split(root / paths["full_train"], [("1", "p1", ""), ("2", "p2", "")])
    _split(root / paths["full_labeled"], [("1", "p1", "1"), ("2", "p2", "0")])
    _split(root / paths["limited_labeled"], [("1", "p1", "1")])
    _split(root / paths["validation"], [("3", "p3", "0")])
    _split(root / paths["test"], [("4", "p4", "1")])
    return paths


def _source_fixture(root: Path) -> dict:
    source = root / "raw/training/cpsc_extra/g1"
    source.mkdir(parents=True)
    files = {"Q1.hea": b"header", "Q1.mat": b"waveform"}
    lines = []
    for name, content in files.items():
        (source / name).write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        lines.append(f"{digest} training/cpsc_extra/g1/{name}\n")
    manifest = root / "raw/SHA256SUMS.txt"
    manifest.write_text("".join(lines), encoding="utf-8")
    receipt = {"state": "complete", "expected_files": 2, "verified_files": 2,
               "expected_records": 1, "source_manifest": "raw/SHA256SUMS.txt",
               "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    (root / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    return {"id": "synthetic", "path": "raw/training/cpsc_extra", "receipt": "receipt.json",
            "official_manifest": "raw/SHA256SUMS.txt", "official_prefix": "training/cpsc_extra/",
            "expected_files": 2, "expected_records": 1}


def test_completed_source_checks_actual_upstream_bytes(tmp_path: Path) -> None:
    spec = _source_fixture(tmp_path)
    assert validate_source(tmp_path, spec)["files"] == 2
    (tmp_path / spec["path"] / "g1/Q1.mat").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_source(tmp_path, spec)


def test_ptb_record_pair_checks_frozen_manifest(tmp_path: Path) -> None:
    source = tmp_path / "ptb/records500/00000"
    source.mkdir(parents=True)
    lines = []
    for name, content in (("00001_hr.hea", b"header"), ("00001_hr.dat", b"waveform")):
        (source / name).write_bytes(content)
        lines.append(f"{hashlib.sha256(content).hexdigest()} records500/00000/{name}\n")
    manifest = tmp_path / "ptb/SHA256SUMS.txt"
    manifest.write_text("".join(lines), encoding="utf-8")
    spec = {"id": "ptb_synthetic", "kind": "ptbxl_records", "path": "ptb/records500",
            "official_manifest": "ptb/SHA256SUMS.txt", "official_prefix": "records500/",
            "official_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "expected_files": 2, "expected_records": 1}
    assert validate_source(tmp_path, spec)["records"] == 1
    (source / "00001_hr.dat").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_source(tmp_path, spec)


def test_incomplete_receipt_and_missing_pair_fail(tmp_path: Path) -> None:
    spec = _source_fixture(tmp_path)
    receipt_path = tmp_path / spec["receipt"]
    receipt = json.loads(receipt_path.read_text())
    receipt["state"] = "running"
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="not complete"):
        validate_source(tmp_path, spec)
    receipt["state"] = "complete"
    receipt_path.write_text(json.dumps(receipt))
    (tmp_path / spec["path"] / "g1/Q1.hea").unlink()
    with pytest.raises(ValueError, match="missing 1"):
        validate_source(tmp_path, spec)


def test_patient_leakage_and_label_subset_fail(tmp_path: Path) -> None:
    paths = _split_fixture(tmp_path)
    assert validate_splits(tmp_path, paths)["patient_leakage"] == 0
    _split(tmp_path / paths["validation"], [("3", "p1", "0")])
    with pytest.raises(ValueError, match="Patient leakage"):
        validate_splits(tmp_path, paths)
    _split(tmp_path / paths["validation"], [("3", "p3", "0")])
    _split(tmp_path / paths["limited_labeled"], [("1", "p1", "0")])
    with pytest.raises(ValueError, match="Limited-label selection differs"):
        validate_splits(tmp_path, paths)
    _split(tmp_path / paths["limited_labeled"], [("1", "p1", "")])
    with pytest.raises(ValueError, match="target must be binary"):
        validate_splits(tmp_path, paths)


def test_development_calibration_partition_and_patient_isolation(tmp_path: Path) -> None:
    paths = _split_fixture(tmp_path)
    _split(tmp_path / paths["validation"], [("3", "p3", "0"), ("5", "p5", "1")])
    heldout = tmp_path / "heldout.csv"
    with heldout.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("record_id", "patient_id", "split"))
        writer.writerows((("ptbxl:3", "ptbxl:p3", "development"),
                         ("ptbxl:5", "ptbxl:p5", "calibration"),
                         ("ptbxl:4", "ptbxl:p4", "test")))
    paths["heldout_references"] = "heldout.csv"
    result = validate_splits(tmp_path, paths)
    assert result["heldout_partition"]["record_counts"] == {
        "development": 1, "calibration": 1, "test": 1}
    heldout.write_text("record_id,patient_id,split\nptbxl:3,ptbxl:p3,development\n"
                       "ptbxl:5,ptbxl:p3,calibration\nptbxl:4,ptbxl:p4,test\n")
    with pytest.raises(ValueError, match="patient mapping differs"):
        validate_splits(tmp_path, paths)


def test_validator_orchestrates_frozen_hashes_and_source_checks(tmp_path: Path) -> None:
    source = _source_fixture(tmp_path)
    splits = _split_fixture(tmp_path)
    metadata = tmp_path / "ptb_metadata.csv"
    metadata.write_bytes(b"metadata")
    manifest = tmp_path / "data/raw/ptb-xl/1.0.3/SHA256SUMS.txt"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(f"{hashlib.sha256(metadata.read_bytes()).hexdigest()} ptb_metadata.csv\n")
    config = tmp_path / "datasets.json"
    settings = {
        "schema_version": 1,
        "datasets": [source],
        "ptb_metadata": [metadata.name],
        "ptb_split_audit": splits,
        "ptb_split_sha256": {
            name: hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest()
            for name, relative in splits.items()
        },
    }
    config.write_text(json.dumps(settings), encoding="utf-8")
    validator = DatasetValidator(tmp_path, config)
    report = validator.validate()
    assert report["sources"][0]["files"] == 2
    assert report["ptb_split_audit"]["record_counts"]["full_train"] == 2

    (tmp_path / splits["test"]).write_text("ecg_id,patient_id,target\n4,p4,0\n")
    with pytest.raises(ValueError, match="Frozen PTB split checksum mismatch: test"):
        validator.validate()

    _split(tmp_path / splits["test"], [("4", "p4", "1")])
    (tmp_path / source["path"] / "g1/Q1.mat").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch: g1/Q1.mat"):
        validator.validate()
