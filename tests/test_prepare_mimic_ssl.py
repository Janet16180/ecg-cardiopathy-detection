"""MIMIC SSL preparation: whole-patient selection, official hashes and resumable audit."""

import csv
import hashlib
import json
from unittest.mock import patch

import numpy as np
import pytest
import wfdb

from ecg_experiment.mimic import (
    audit,
    check_waveform,
    lock_selection,
    read_patients,
    required_checksums,
    select_patients,
    selection_hash,
)
from ecg_experiment.public_sources import signal_sha256
from scripts.prepare_mimic_ssl import verify_or_fetch


def test_patient_selection_stable_whole_and_locked(tmp_path):
    path = tmp_path / "record_list.csv"
    rows = []
    for patient, count in (("10000001", 2), ("10000002", 3), ("10000003", 1)):
        for number in range(count):
            study = f"{40000000 + int(patient[-1]) * 10 + number:08d}"
            rows.append({"subject_id": patient, "study_id": study, "file_name": study,
                         "ecg_time": "ignored", "path": f"files/p1000/p{patient}/s{study}/{study}"})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(reversed(rows))
    patients = read_patients(path)
    selected, subjects = select_patients(patients, seed=42, max_records=4)
    assert len(selected) <= 4
    assert {row[0] for row in selected} == set(subjects)
    for subject in subjects:
        assert sum(row[0] == subject for row in selected) == len(patients[subject])
    digest = selection_hash(selected, 42, 4, "a" * 64)
    assert digest == selection_hash(selected, 42, 4, "a" * 64)
    assert digest != selection_hash(selected, 43, 4, "a" * 64)
    output = tmp_path / "out"
    lock_selection(output, selected, digest, 42, 4, "a" * 64)
    lock_selection(output, selected, digest, 42, 4, "a" * 64)
    assert json.loads((output / "selection.json").read_text())["selection_sha256"] == digest
    with pytest.raises(ValueError, match="another patient selection"):
        lock_selection(output, selected, digest, 43, 4, "a" * 64)


def test_manifest_requires_official_hashes(tmp_path):
    name = "files/p1000/p10000001/s40000001/40000001"
    sums = tmp_path / "SHA256SUMS.txt"
    sums.write_text("".join(f"{'a' * 64} {file}\n" for file in
                            ("LICENSE.txt", "record_list.csv", name + ".hea", name + ".dat")))
    assert len(required_checksums(sums, {name})) == 4
    sums.write_text(sums.read_text().replace(name + ".dat", "other.dat"))
    with pytest.raises(ValueError, match="lacks 1 required"):
        required_checksums(sums, {name})


def test_waveform_contract_reorders_leads_and_rejects_bad_rate(tmp_path):
    directory = tmp_path / "waveforms"
    directory.mkdir()
    names = ["I", "II", "III", "aVR", "aVF", "aVL", "V1", "V2", "V3", "V4", "V5", "V6"]
    data = np.tile(np.arange(12, dtype=np.float64), (5000, 1)) / 200
    wfdb.wrsamp("sample", fs=500, units=["mV"] * 12, sig_name=names,
                p_signal=data, fmt=["16"] * 12, adc_gain=[200] * 12,
                baseline=[0] * 12, write_dir=str(directory))
    canonical = check_waveform(directory, "sample")
    assert canonical.shape == (12, 5000)
    assert float(canonical[4, 0]) == pytest.approx(5 / 200)
    assert float(canonical[5, 0]) == pytest.approx(4 / 200)
    assert np.isfinite(canonical).all()
    original = (directory / "sample.hea").read_text()
    (directory / "sample.hea").write_text(original.replace("sample 12 500 5000", "sample 12 250 5000"))
    with pytest.raises(ValueError, match="Expected 500 Hz"):
        check_waveform(directory, "sample")


def test_exact_dedup_and_resume(tmp_path):
    rows = [("10000001", "40000001", "files/p1000/p10000001/s40000001/40000001"),
            ("10000002", "40000002", "files/p1000/p10000002/s40000002/40000002"),
            ("10000003", "40000003", "files/p1000/p10000003/s40000003/40000003")]
    output = tmp_path / "out"
    output.mkdir()
    first = np.ones((12, 5000), dtype=np.float32)
    other = np.zeros((12, 5000), dtype=np.float32)
    with patch("ecg_experiment.mimic.check_waveform", side_effect=[first, first, other]) as read:
        accepted, reasons = audit(rows, tmp_path, output, "selection", {signal_sha256(other)})
        assert read.call_count == 3
    assert len(accepted) == 1
    assert reasons["exact_duplicate_mimic"] == 1
    assert reasons["exact_duplicate_ptbxl"] == 1
    with patch("ecg_experiment.mimic.check_waveform", side_effect=AssertionError("redecoded")):
        resumed, again = audit(rows, tmp_path, output, "selection", {signal_sha256(other)})
    assert resumed == accepted
    assert again == reasons
    with pytest.raises(ValueError, match="different patient selection"):
        audit(rows, tmp_path, output, "other", {signal_sha256(other)})


def test_verified_file_is_kept_and_unsafe_path_rejected(tmp_path):
    relative = "files/p1000/record.hea"
    (tmp_path / relative).parent.mkdir(parents=True)
    (tmp_path / relative).write_bytes(b"header")
    checksum = hashlib.sha256(b"header").hexdigest()
    with patch("scripts.prepare_mimic_ssl._session", side_effect=AssertionError("downloaded")):
        assert verify_or_fetch(relative, tmp_path, checksum, 1, 0) == (False, 6)
    with pytest.raises(ValueError, match="Unsafe download path"):
        verify_or_fetch("files/../escape", tmp_path, checksum, 1, 0)
    with pytest.raises(ValueError, match="Unsafe download path"):
        verify_or_fetch("other/record.hea", tmp_path, checksum, 1, 0)
