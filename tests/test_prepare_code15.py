"""CODE-15% preparation must preserve native traces and patient/label provenance."""

from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.data.prepare_code15 import inspect_trace, prepare, sha256_file
from scripts.validation.verify_code15_prepared import verify


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def test_native_materialization_and_explicit_exclusions(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    csv_path = raw / "exams.csv"
    fields = ("exam_id", "age", "is_male", "nn_predicted_age", "1dAVb", "RBBB",
              "LBBB", "SB", "ST", "AF", "patient_id", "normal_ecg", "trace_file")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for exam_id, patient in ((101, 5), (102, 6), (103, 7)):
            writer.writerow({"exam_id": exam_id, "age": 20, "is_male": "True",
                             "nn_predicted_age": 21, "1dAVb": "False", "RBBB": "False",
                             "LBBB": "False", "SB": "False", "ST": "False", "AF": "False",
                             "patient_id": patient, "normal_ecg": "True",
                             "trace_file": "exams_part0.hdf5"})
    time = np.arange(4096, dtype=np.float32)
    base = np.tile(np.sin(time / 20)[:, None], (1, 12)).astype(np.float32)
    base[:581] = 0
    base[-581:] = 0
    bad = base.copy()
    bad[:, 4] = 0
    hdf5_path = tmp_path / "exams_part0.hdf5"
    with h5py.File(hdf5_path, "w") as handle:
        handle.create_dataset("exam_id", data=np.array([101, 102, 103]))
        handle.create_dataset("tracings", data=np.stack((base, bad, base)))
    archive_path = raw / "exams_part0.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(hdf5_path, "exams_part0.hdf5")
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"verified_files": [
        {"name": path.name, "checksum": f"md5:{_md5(path)}", "size": path.stat().st_size}
        for path in (csv_path, archive_path)]}))
    raw_before = {path.name: sha256_file(path) for path in (csv_path, archive_path)}

    out = tmp_path / "processed"
    result = prepare([0], out, materialize_native=True, raw_dir=raw, receipt_path=receipt)

    assert result["counts"]["audited"] == 3
    assert result["counts"]["accepted"] == 1
    assert result["counts"]["excluded_constant_lead"] == 1
    assert result["counts"]["excluded_duplicate_native_signal"] == 1
    assert result["edge_zero_pairs"] == {"581,581": 1}
    assert result["canonical_500hz_10s_eligible"] is False
    with (out / "manifest.csv").open(newline="") as handle:
        manifest = list(csv.DictReader(handle))
    assert manifest[0]["patient_id"] == "5"
    assert manifest[0]["prepared_index"] == "0"
    with h5py.File(out / "exams_part0_native.hdf5") as handle:
        np.testing.assert_array_equal(handle["tracings"][0], base)
        assert list(handle["exam_id"][:]) == [101]
    with (out / "source_labels.csv").open(newline="") as handle:
        labels = list(csv.DictReader(handle))
    assert labels[0]["normal_ecg_provenance"] == "automatic_annotation_per_Zenodo"
    assert labels[0]["diagnosis_label_provenance"].startswith("released_flags")
    assert {path.name: sha256_file(path) for path in (csv_path, archive_path)} == raw_before
    assert verify(out)["exam_rows_verified"] == 1
    assert not (tmp_path / "processed.inprogress").exists()
    with pytest.raises(FileExistsError):
        prepare([0], out, materialize_native=True, raw_dir=raw, receipt_path=receipt)

    failed = tmp_path / "failed"
    with pytest.raises(ValueError, match="lacks a checksum receipt"):
        prepare([0, 1], failed, materialize_native=True, raw_dir=raw, receipt_path=receipt)
    assert not failed.exists()
    assert not (tmp_path / "failed.inprogress").exists()

    # Matching file hashes alone must not make a false sampling rate valid.
    with h5py.File(out / "exams_part0_native.hdf5", "r+") as handle:
        handle.attrs["sample_rate_hz"] = 500
    metadata = json.loads((out / "metadata.json").read_text())
    metadata["output_sha256"]["exams_part0_native.hdf5"] = sha256_file(out / "exams_part0_native.hdf5")
    (out / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="acquisition contract"):
        verify(out)


def test_signal_qc_exclusions_and_edge_measurement() -> None:
    zeros = np.zeros((4096, 12), dtype=np.float32)
    assert inspect_trace(zeros)[0] == "all_zero_signal"
    nonfinite = zeros.copy()
    nonfinite[100, 0] = np.nan
    assert inspect_trace(nonfinite)[0] == "nonfinite_signal"
    time = np.arange(4096, dtype=np.float32)
    trace = np.tile(np.sin(time / 10)[:, None], (1, 12))
    trace[:48] = 0
    trace[-65:] = 0
    reason, left, right, span, flags = inspect_trace(trace)
    assert reason is None
    assert (left, right, span) == (48, 65, 3983)
    assert "asymmetric_zero_edges_review" in flags
