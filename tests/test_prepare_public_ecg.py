"""Checks for read-only canonicalization and duration rules of new public cohorts."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import scripts.data.prepare_public_ecg as preparation
from ecg_experiment import public_sources
from ecg_experiment.files import read_csv, sha256_file
from ecg_experiment.public_sources import load_view, signal_sha256

LEADS = ["V6", "V5", "V4", "V3", "V2", "V1", "aVF", "aVL", "aVR", "III", "II", "I"]


def fake_record(samples: int = 6000) -> SimpleNamespace:
    time = np.arange(samples, dtype=np.float32) / 1000
    signal = np.stack([time + index / 10 for index in range(12)], axis=1)
    return SimpleNamespace(fs=500, sig_name=LEADS, units=["mV"] * 12, p_signal=signal)


def test_ssl_crop_preserves_raw_record_and_reorders_leads(tmp_path: Path, monkeypatch) -> None:
    raw = fake_record()
    original = raw.p_signal.copy()
    monkeypatch.setattr(public_sources.wfdb, "rdrecord", Mock(return_value=raw))
    view, start, samples, flags = load_view(tmp_path, "record", "ssl_center_crop")
    assert (start, samples, flags) == (500, 6000, "")
    assert view.shape == (12, 5000)
    np.testing.assert_allclose(view[0], original[500:5500, 11])
    np.testing.assert_array_equal(raw.p_signal, original)


def test_strict_rejects_non_ten_second_record(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(public_sources.wfdb, "rdrecord", Mock(return_value=fake_record()))
    with pytest.raises(ValueError, match="duration_contract"):
        load_view(tmp_path, "record", "strict_10s")


def test_constant_lead_is_excluded(tmp_path: Path, monkeypatch) -> None:
    raw = fake_record(5000)
    raw.p_signal[:, 0] = 0
    monkeypatch.setattr(public_sources.wfdb, "rdrecord", Mock(return_value=raw))
    with pytest.raises(ValueError, match="constant_lead"):
        load_view(tmp_path, "record", "strict_10s")


def test_unknown_policy_cannot_silently_select_first_window(tmp_path: Path, monkeypatch) -> None:
    reader = Mock()
    monkeypatch.setattr(public_sources.wfdb, "rdrecord", reader)
    with pytest.raises(ValueError, match="unknown_policy"):
        load_view(tmp_path, "record", "typo")
    reader.assert_not_called()


def test_preparation_refuses_output_inside_raw(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(preparation, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="outside data/raw"):
        preparation.prepare(["georgia"], "strict_10s", tmp_path / "data/raw/derived")
    assert not (tmp_path / "data/raw/derived").exists()


def test_preparation_preserves_published_directory(tmp_path: Path) -> None:
    output = tmp_path / "published"
    output.mkdir()
    (output / "manifest.csv").write_text("immutable")
    with pytest.raises(FileExistsError, match="overwrite"):
        preparation.prepare(["georgia"], "strict_10s", output)
    assert (output / "manifest.csv").read_text() == "immutable"


def test_failed_preparation_is_not_published(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "candidate"
    monkeypatch.setattr(preparation, "_prepare", Mock(side_effect=ValueError("bad input")))
    with pytest.raises(ValueError, match="bad input"):
        preparation.prepare(["georgia"], "strict_10s", output)
    assert not output.exists()
    assert not list(tmp_path.glob(".candidate.staging-*"))


def test_reference_cache_rejects_changed_raw_with_unchanged_metadata(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "data/raw/ptb-xl/1.0.3"
    raw.mkdir(parents=True)
    (raw / "ptbxl_database.csv").write_text("filename_hr\nrecord\n")
    (raw / "record.hea").write_bytes(b"header")
    (raw / "record.dat").write_bytes(b"waveform")
    sums = "".join(f"{hashlib.sha256((raw / name).read_bytes()).hexdigest()}  {name}\n"
                   for name in ("ptbxl_database.csv", "record.hea", "record.dat"))
    (raw / "SHA256SUMS.txt").write_text(sums)
    cache = tmp_path / "cache.json"
    reader = Mock(return_value=np.ones((12, 5000), dtype=np.float32))
    monkeypatch.setattr(preparation, "ROOT", tmp_path)
    monkeypatch.setattr(preparation, "read_record", reader)
    monkeypatch.setattr(preparation.inspect, "getsource", Mock(return_value="decoder"))
    first = preparation.ptb_reference_hashes(cache)
    assert json.loads(cache.read_text())["schema_version"] == 2
    assert preparation.ptb_reference_hashes(cache) == first
    assert reader.call_count == 1
    (raw / "record.dat").write_bytes(b"corrupt!")
    with pytest.raises(ValueError, match="waveform official checksum mismatch"):
        preparation.ptb_reference_hashes(cache)


def _signal(key: int) -> np.ndarray:
    return np.stack([np.arange(5000, dtype=np.float32) * (lead + 1) + key for lead in range(12)])


def test_exclusion_reasons_and_accepted_rows(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "data/raw/challenge-2020/1.0.2"
    stems = [f"training/georgia/g1/E0000{number}" for number in range(1, 7)]
    lines = []
    for number, stem in enumerate(stems, 1):
        path = raw / stem
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".hea").write_text(f"E0000{number} 12 500 5000\n# Dx: {number}\n")
        path.with_suffix(".mat").write_bytes(stem.encode())
        suffixes = (".hea",) if number == 6 else (".hea", ".mat")
        lines += [f"{sha256_file(path.with_suffix(suffix))}  {stem}{suffix}" for suffix in suffixes]
    (raw / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n")
    (tmp_path / "data/acquisition").mkdir(parents=True)
    (tmp_path / "data/acquisition/georgia.json").write_text('{"state": "complete"}')
    cache = tmp_path / "outputs/data_quality/ptbxl_reference_hashes_v2.json"
    cache.parent.mkdir(parents=True)
    cache.write_text("{}")
    views = {stems[0]: _signal(1), stems[1]: _signal(1), stems[2]: _signal(99),
             stems[3]: ValueError("duration_contract"), stems[4]: OSError("disk: unreadable")}

    def fake_view(_raw: Path, stem: str, _policy: str) -> tuple[np.ndarray, int, int, str]:
        if isinstance(views[stem], Exception):
            raise views[stem]
        return views[stem], 0, 5000, ""

    monkeypatch.setattr(preparation, "ROOT", tmp_path)
    monkeypatch.setattr(preparation, "load_view", fake_view)
    monkeypatch.setattr(preparation, "ptb_reference_hashes", Mock(return_value={signal_sha256(_signal(99))}))
    output = tmp_path / "prepared"
    result = preparation.prepare(["georgia"], "strict_10s", output)

    assert result["counts"] == {"accepted_georgia": 1, "manual_review_flags": 0,
                                "excluded_exact_pool_duplicate": 1, "excluded_exact_ptbxl_duplicate": 1,
                                "excluded_duration_contract": 1, "excluded_disk": 1,
                                "excluded_official_checksum_or_missing_file": 1}
    accepted = read_csv(output / "manifest.csv")
    assert [(row["ecg_id"], row["label_codes"]) for row in accepted] == [("georgia:E00001", "1")]
    excluded = {row["ecg_id"]: (row["reason"], row["detail"]) for row in read_csv(output / "exclusions.csv")}
    assert excluded["georgia:E00005"] == ("disk", "disk: unreadable")
    assert excluded["georgia:E00002"] == ("exact_pool_duplicate", "exact_pool_duplicate")
    assert result["manifest_sha256"] == sha256_file(output / "manifest.csv")
