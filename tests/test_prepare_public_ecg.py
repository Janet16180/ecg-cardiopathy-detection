"""Checks for read-only canonicalization and duration rules of new public cohorts."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from scripts.prepare_public_ecg import load_view, prepare


LEADS = ["V6", "V5", "V4", "V3", "V2", "V1", "aVF", "aVL", "aVR", "III", "II", "I"]


def fake_record(samples: int = 6000) -> SimpleNamespace:
    time = np.arange(samples, dtype=np.float32) / 1000
    signal = np.stack([time + index / 10 for index in range(12)], axis=1)
    return SimpleNamespace(fs=500, sig_name=LEADS, units=["mV"] * 12, p_signal=signal)


def test_ssl_crop_preserves_raw_record_and_reorders_leads(tmp_path: Path) -> None:
    raw = fake_record()
    original = raw.p_signal.copy()
    with patch("scripts.prepare_public_ecg.wfdb.rdrecord", return_value=raw):
        view, start, samples, flags = load_view(tmp_path, "record", "ssl_center_crop")
    assert (start, samples, flags) == (500, 6000, "")
    assert view.shape == (12, 5000)
    np.testing.assert_allclose(view[0], original[500:5500, 11])
    np.testing.assert_array_equal(raw.p_signal, original)


def test_strict_rejects_non_ten_second_record(tmp_path: Path) -> None:
    with patch("scripts.prepare_public_ecg.wfdb.rdrecord", return_value=fake_record()):
        with pytest.raises(ValueError, match="duration_contract"):
            load_view(tmp_path, "record", "strict_10s")


def test_constant_lead_is_excluded(tmp_path: Path) -> None:
    raw = fake_record(5000)
    raw.p_signal[:, 0] = 0
    with patch("scripts.prepare_public_ecg.wfdb.rdrecord", return_value=raw):
        with pytest.raises(ValueError, match="constant_lead"):
            load_view(tmp_path, "record", "strict_10s")


def test_preparation_refuses_output_inside_raw(tmp_path: Path) -> None:
    with patch("scripts.prepare_public_ecg.ROOT", tmp_path):
        with pytest.raises(ValueError, match="outside data/raw"):
            prepare(["georgia"], "strict_10s", tmp_path / "data/raw/derived")
    assert not (tmp_path / "data/raw/derived").exists()


def test_unknown_policy_cannot_silently_select_first_window(tmp_path: Path) -> None:
    with patch("scripts.prepare_public_ecg.wfdb.rdrecord") as reader:
        with pytest.raises(ValueError, match="unknown_policy"):
            load_view(tmp_path, "record", "typo")
    reader.assert_not_called()


def test_preparation_preserves_published_directory(tmp_path: Path) -> None:
    output = tmp_path / "published"
    output.mkdir()
    (output / "manifest.csv").write_text("immutable")
    with pytest.raises(FileExistsError, match="overwrite"):
        prepare(["georgia"], "strict_10s", output)
    assert (output / "manifest.csv").read_text() == "immutable"


def test_failed_preparation_is_not_published(tmp_path: Path) -> None:
    output = tmp_path / "candidate"
    with patch("scripts.prepare_public_ecg._prepare", side_effect=ValueError("bad input")):
        with pytest.raises(ValueError, match="bad input"):
            prepare(["georgia"], "strict_10s", output)
    assert not output.exists()
    assert not list(tmp_path.glob(".candidate.staging-*"))


def test_reference_cache_rejects_changed_raw_even_with_unchanged_metadata(tmp_path: Path) -> None:
    import hashlib
    import json
    from scripts.prepare_public_ecg import ptb_reference_hashes
    raw = tmp_path / "data/raw/ptb-xl/1.0.3"
    raw.mkdir(parents=True)
    (raw / "ptbxl_database.csv").write_text("filename_hr\nrecord\n")
    (raw / "record.hea").write_bytes(b"header")
    (raw / "record.dat").write_bytes(b"waveform")
    sums = "".join(f"{hashlib.sha256((raw / name).read_bytes()).hexdigest()}  {name}\n"
                   for name in ("ptbxl_database.csv", "record.hea", "record.dat"))
    (raw / "SHA256SUMS.txt").write_text(sums)
    cache = tmp_path / "cache.json"
    with patch("scripts.prepare_public_ecg.ROOT", tmp_path), patch(
            "scripts.prepare_public_ecg.read_record", return_value=np.ones((12, 5000), dtype=np.float32)), patch(
            "scripts.prepare_public_ecg.inspect.getsource", return_value="decoder"):
        ptb_reference_hashes(cache)
        assert json.loads(cache.read_text())["schema_version"] == 2
        (raw / "record.dat").write_bytes(b"corrupt!")
        with pytest.raises(ValueError, match="waveform official checksum mismatch"):
            ptb_reference_hashes(cache)
