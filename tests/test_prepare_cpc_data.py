"""CPC pool preparation: locked selection, official hashes and resumable cache."""

import csv
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from scipy.signal import resample_poly

from ecg_experiment.mimic import read_patients, select_patients, selection_hash
from scripts.data import prepare_cpc_data
from scripts.data.prepare_cpc_data import (
    build_cache,
    read_locked_prefix,
    read_ptb_rows,
    resample_halves,
    verify_selected_files,
)

METADATA = {"selection_sha256": "selection", "manifest_sha256": "manifest"}


def test_half_resampling_keeps_boundary_independent():
    raw = np.zeros((12, 5000), dtype=np.float32)
    raw[:, 2499] = 1
    actual = resample_halves(raw, 250)
    assert actual.shape == (12, 2500)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual[:, 1250:], 0)
    np.testing.assert_allclose(actual[:, :1250], resample_poly(raw[:, :2500], 1, 2, axis=1))
    assert resample_halves(raw, 100).shape == (12, 1000)
    with pytest.raises(ValueError, match="finite"):
        resample_halves(np.full((12, 5000), np.nan, dtype=np.float32), 250)
    with pytest.raises(ValueError, match="100 or 250"):
        resample_halves(raw, 200)


def test_locked_prefix_stops_before_partial_patient(tmp_path):
    official = tmp_path / "record_list.csv"
    rows = []
    for patient, count in (("10000001", 3), ("10000002", 2), ("10000003", 2)):
        for offset in range(count):
            study = str(40000000 + int(patient[-1]) * 10 + offset)
            rows.append({"subject_id": patient, "study_id": study, "file_name": study,
                         "ecg_time": "unused", "path": f"files/p1000/p{patient}/s{study}/{study}"})
    with official.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    patients = read_patients(official)
    parent_rows, _ = select_patients(patients, 42, 7)
    list_hash = hashlib.sha256(official.read_bytes()).hexdigest()
    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "selection.json").write_text(json.dumps({
        "seed": 42, "max_records": 7, "record_list_sha256": list_hash,
        "selection_sha256": selection_hash(parent_rows, 42, 7, list_hash)}))
    with (parent / "selected_records.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("subject_id", "study_id", "path"))
        writer.writerows(parent_rows)

    chosen, subjects, _, _ = read_locked_prefix(parent, 4, 42, official)

    assert len(chosen) <= 4
    assert len({row[0] for row in chosen}) == len(subjects)
    for subject in subjects:
        assert sum(row[0] == subject for row in chosen) == len(patients[subject])
    with pytest.raises(ValueError, match="incompatible seed"):
        read_locked_prefix(parent, 4, 43, official)
    (parent / "selected_records.csv").write_text("subject_id,study_id,path\nwrong,wrong,wrong\n")
    with pytest.raises(ValueError, match="prefix"):
        read_locked_prefix(parent, 4, 42, official)


def test_official_hash_verification_fails_closed(tmp_path):
    raw = tmp_path / "raw"
    name = "files/p1000/p10000001/s40000001/40000001"
    for relative, content in (("record_list.csv", b"list"), ("LICENSE.txt", b"license"),
                              (name + ".hea", b"header"), (name + ".dat", b"data")):
        path = raw / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    sums = raw / "SHA256SUMS.txt"
    with sums.open("w") as stream:
        for relative in ("record_list.csv", "LICENSE.txt", name + ".hea", name + ".dat"):
            stream.write(f"{hashlib.sha256((raw / relative).read_bytes()).hexdigest()} {relative}\n")
    selected = [("10000001", "40000001", name)]
    assert verify_selected_files(selected, raw, sums, raw / "record_list.csv")["verified_records"] == 1
    (raw / (name + ".dat")).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_selected_files(selected, raw, sums, raw / "record_list.csv")


def _write_split(path: Path, rows: list[tuple[str, str, str]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("ecg_id", "patient_id", "filename_hr"))
        writer.writerows(rows)


def test_ptb_rows_reject_patient_leakage(tmp_path):
    _write_split(tmp_path / "all_train_ssl.csv", [("1", "p1", "records500/00000/1")])
    _write_split(tmp_path / "validation.csv", [("2", "p1", "records500/00000/2")])
    _write_split(tmp_path / "test.csv", [("3", "p3", "records500/00000/3")])
    with pytest.raises(ValueError, match="leakage between all_train_ssl and validation"):
        read_ptb_rows(tmp_path, tmp_path)


def test_ptb_rows_require_frozen_split_counts(tmp_path):
    _write_split(tmp_path / "all_train_ssl.csv", [("1", "p1", "records500/00000/1")])
    _write_split(tmp_path / "validation.csv", [("2", "p2", "records500/00000/2")])
    _write_split(tmp_path / "test.csv", [("3", "p3", "records500/00000/3")])
    with pytest.raises(ValueError, match="Unexpected PTB split counts"):
        read_ptb_rows(tmp_path, tmp_path)


def test_cache_rows_ids_and_resume_validation(tmp_path):
    row = {"ecg_id": "1", "patient_id": "10.0", "source": "ptbxl", "split": "train",
           "raw_dir": str(tmp_path), "filename_hr": "record"}
    mimic = {"ecg_id": "mimic:40000001", "patient_id": "mimic:10000001",
             "raw_dir": str(tmp_path), "filename_hr": "mimic"}
    raw = np.zeros((12, 5000), dtype=np.float32)
    cache = tmp_path / "cache"
    with patch("scripts.data.prepare_cpc_data.read_record", return_value=raw) as reader:
        info = build_cache([row], [mimic], {"all_train_ssl.csv": "hash"}, METADATA, cache, 250)
        assert reader.call_count == 2
    assert info["shape"] == [2, 12, 2500]
    assert np.load(cache / "signals.npy", mmap_mode="r").shape == (2, 12, 2500)
    assert np.load(cache / "ecg_ids.npy").tolist() == ["1", "mimic:40000001"]
    assert not (cache / "progress.json").exists()
    with patch("scripts.data.prepare_cpc_data.read_record", side_effect=AssertionError("redecoded")):
        assert build_cache([row], [mimic], {"all_train_ssl.csv": "hash"}, METADATA, cache, 250) == info
    with pytest.raises(ValueError, match="different inputs"):
        build_cache([row], [mimic], {"all_train_ssl.csv": "changed"}, METADATA, cache, 250)


def test_cache_resumes_at_flushed_checkpoint(tmp_path):
    rows = [{"ecg_id": str(index), "patient_id": str(index), "source": "ptbxl",
             "split": "train", "raw_dir": str(tmp_path), "filename_hr": str(index)}
            for index in range(101)]
    raw = np.zeros((12, 5000), dtype=np.float32)
    count = 0

    def fail_after_checkpoint(*_):
        nonlocal count
        count += 1
        if count == 101:
            raise RuntimeError("interrupted")
        return raw

    cache = tmp_path / "cache"
    with patch("scripts.data.prepare_cpc_data.read_record", side_effect=fail_after_checkpoint), \
            pytest.raises(RuntimeError, match="interrupted"):
        build_cache(rows, [], {}, METADATA, cache, 250)
    assert json.loads((cache / "progress.json").read_text())["completed_rows"] == 100
    with patch("scripts.data.prepare_cpc_data.read_record", return_value=raw) as reader:
        build_cache(rows, [], {}, METADATA, cache, 250)
        assert reader.call_count == 1
    assert np.load(cache / "signals.npy", mmap_mode="r").shape == (101, 12, 2500)


def test_cache_completes_after_rename_before_completion_record(tmp_path, monkeypatch):
    rows = [{"ecg_id": "1", "patient_id": "1", "source": "ptbxl", "split": "train",
             "raw_dir": str(tmp_path), "filename_hr": "1"}]
    raw = np.ones((12, 5000), dtype=np.float32)
    cache = tmp_path / "cache"
    monkeypatch.setattr(prepare_cpc_data, "read_record", lambda *args: raw)
    info = build_cache(rows, [], {}, METADATA, cache, 100)
    progress = {key: info[key] for key in info if key not in
                ("record_count", "split_counts", "source_counts", "signals_sha256")}
    (cache / "complete.json").unlink()
    (cache / "progress.json").write_text(json.dumps({"identity": progress, "completed_rows": 1}) + "\n")
    def redecode(*args):
        raise AssertionError("redecoded")

    monkeypatch.setattr(prepare_cpc_data, "read_record", redecode)
    assert build_cache(rows, [], {}, METADATA, cache, 100) == info
    assert not (cache / "progress.json").exists()


def test_cache_rejects_files_without_row_identities(tmp_path):
    rows = [{"ecg_id": "1", "patient_id": "1", "source": "ptbxl", "split": "train",
             "raw_dir": str(tmp_path), "filename_hr": "1"}]
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "rows.csv.partial").write_text("interrupted")
    (cache / "signals.partial.npy").write_bytes(b"")
    with pytest.raises(ValueError, match="files without rows.csv"):
        build_cache(rows, [], {}, METADATA, cache, 250)
