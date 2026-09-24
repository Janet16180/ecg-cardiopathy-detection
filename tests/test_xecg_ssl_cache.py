"""Verify interrupted cache preparation preserves identities and waveform rows."""

from unittest.mock import patch

import numpy as np
import pytest

from scripts.data.prepare_xecg_ssl import prepare


def test_interrupted_prefix_resumes_and_rejects_changed_sources(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = []
    for index in range(130):
        for suffix in (".hea", ".dat"):
            (raw / f"{index}{suffix}").write_bytes(str(index).encode())
        rows.append({"ecg_id": str(index), "patient_id": str(index), "source": "ptbxl",
                     "split": "train", "raw_dir": str(raw), "filename_hr": str(index)})
    requested = (rows, {"fixture": "stable"})

    def read(_, relative):
        return np.full((12, 5000), int(relative) + 1, dtype=np.float32)

    def interrupted(directory, relative):
        if relative == "128":
            raise RuntimeError("simulated interruption")
        return read(directory, relative)

    with patch("scripts.data.prepare_xecg_ssl.selected_rows", return_value=requested), \
            patch("scripts.data.prepare_xecg_ssl.read_record_float64", side_effect=interrupted), \
            pytest.raises(RuntimeError, match="simulated"):
        prepare(tmp_path / "resumed", workers=1)
    with patch("scripts.data.prepare_xecg_ssl.selected_rows", return_value=requested), \
            patch("scripts.data.prepare_xecg_ssl.read_record_float64", side_effect=read):
        actual = prepare(tmp_path / "resumed", workers=1)
        expected = prepare(tmp_path / "fresh", workers=1)
        assert prepare(tmp_path / "fresh", workers=1) == expected
    np.testing.assert_array_equal(np.load(tmp_path / "resumed/views.npy"),
                                  np.load(tmp_path / "fresh/views.npy"))
    resumed_hashes = (tmp_path / "resumed/raw_sha256.npy").read_bytes()
    assert resumed_hashes == (tmp_path / "fresh/raw_sha256.npy").read_bytes()
    assert actual["views_sha256"] == expected["views_sha256"]
    assert actual["ecg_ids"] == [str(index) for index in range(130)]
    assert actual["all_train_only"]
    assert not (tmp_path / "resumed/progress.json").exists()
    with patch("scripts.data.prepare_xecg_ssl.selected_rows", return_value=(rows, {"fixture": "changed"})), \
            pytest.raises(ValueError, match="identity differs"):
        prepare(tmp_path / "resumed", workers=1)


def test_partial_arrays_without_progress_are_rejected(tmp_path):
    rows = [{"ecg_id": "1", "patient_id": "1", "source": "ptbxl", "split": "train",
             "raw_dir": str(tmp_path), "filename_hr": "1"}]
    output = tmp_path / "cache"
    output.mkdir()
    (output / "views.partial.npy").write_bytes(b"")
    with patch("scripts.data.prepare_xecg_ssl.selected_rows", return_value=(rows, {})), \
            pytest.raises(ValueError, match="lacks progress record"):
        prepare(output, workers=1)
