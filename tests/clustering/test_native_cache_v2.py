"""Synthetic checks for the forward canonical-order reader correction and stage reuse."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from ecg_experiment.clustering import native_cache_v2 as cache
from ecg_experiment.eda.signals import LEADS
from ecg_experiment.files import sha256_file


@pytest.fixture
def waveform_reader(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DATABASE", tmp_path / "ptbxl_database.csv")
    monkeypatch.setattr(cache, "to_stored", str)
    path = tmp_path / "records500/00001_hr"
    path.parent.mkdir()
    path.with_suffix(".hea").write_text("synthetic header")
    path.with_suffix(".dat").write_bytes(b"synthetic waveform bytes")
    expected = np.tile(np.arange(12, dtype=np.float64), (5000, 1))
    fields = {"fs": 500, "units": ["mV"] * 12, "sig_name": list(LEADS)}
    signal = expected.copy()
    hashes = {}

    def read(_):
        assert len(hashes) == 2
        return signal, fields

    monkeypatch.setattr(cache.wfdb, "rdsamp", read)
    return {"filename_hr": "records500/00001_hr"}, hashes, signal, fields, expected, path


@pytest.mark.parametrize("style", ["uppercase", "permuted", "di_aliases", "permuted_aliases"])
def test_canonical_names_and_aliases_restore_identical_physical_columns(waveform_reader, style):
    row, hashes, signal, fields, expected, _ = waveform_reader
    names = [name.upper() for name in LEADS]
    if "aliases" in style:
        names[:3] = ["DI", "DII", "DIII"]
    if "permuted" in style:
        order = [11, 2, 0, 9, 6, 3, 1, 8, 4, 10, 5, 7]
        signal[:] = expected[:, order]
        names = [names[index] for index in order]
    fields["sig_name"] = names
    actual = cache.read_waveform(row, hashes)
    np.testing.assert_array_equal(actual, expected)
    assert len(hashes) == 2
    assert actual.dtype == np.float64


@pytest.mark.parametrize("kind", ["duplicate", "missing", "unknown"])
def test_invalid_lead_metadata_aborts_without_quality_reclassification(waveform_reader, kind):
    row, hashes, _, fields, _, _ = waveform_reader
    if kind == "duplicate":
        fields["sig_name"][1] = "DI"
    elif kind == "missing":
        fields["sig_name"] = fields["sig_name"][:-1]
    else:
        fields["sig_name"][1] = "unexpected"
    with pytest.raises(ValueError, match="Unexpected lead names") as caught:
        cache.read_waveform(row, hashes)
    assert not isinstance(caught.value, cache.QualityExclusionError)


def test_v2_hash_before_reader_and_changed_bytes_abort(waveform_reader):
    row, hashes, _, _, _, path = waveform_reader
    cache.read_waveform(row, hashes)
    path.with_suffix(".dat").write_bytes(b"changed synthetic bytes")
    with pytest.raises(ValueError, match="Previously bound waveform changed"):
        cache.read_waveform(row, hashes)


@pytest.mark.parametrize("kind", ["units", "sampling_rate", "nonfinite"])
def test_existing_prospective_quality_exclusions_are_preserved(waveform_reader, kind):
    row, hashes, signal, fields, _, _ = waveform_reader
    if kind == "units":
        fields["units"] = ["uV"] * 12
    elif kind == "sampling_rate":
        fields["fs"] = 250
    else:
        signal[0, 0] = np.nan
    with pytest.raises(cache.QualityExclusionError):
        cache.read_waveform(row, hashes)


def test_v2_source_freeze_preserves_original_hashes_and_requires_committed_successors(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "ROOT", tmp_path)
    monkeypatch.setattr(cache, "ADDITIONAL_SOURCES", ("successor.py",))
    monkeypatch.setattr(cache, "original_frozen_sources", lambda: {"original.py": "frozen-hash"})
    source = b"committed successor bytes"
    (tmp_path / "successor.py").write_bytes(source)
    monkeypatch.setattr(cache.subprocess, "check_output", lambda *_args, **_kwargs: source)
    hashes = cache.frozen_sources()
    assert hashes["original.py"] == "frozen-hash"
    assert hashes["successor.py"] == hashlib.sha256(source).hexdigest()
    (tmp_path / "successor.py").write_bytes(b"dirty successor")
    with pytest.raises(ValueError, match="Uncommitted successor"):
        cache.frozen_sources()


def test_v2_extract_uses_corrected_reader_without_script_imports(tmp_path, monkeypatch):
    frame = pd.DataFrame(
        {
            "ecg_id": [1],
            "patient_id": [11],
            "strat_fold": [1],
            "filename_hr": ["records500/1_hr"],
        }
    )
    monkeypatch.setattr(cache, "verified_cohort", lambda: (frame, None, None, {}))
    monkeypatch.setattr(cache, "validate_profile", lambda *_: {"waveform_sha256": {}})
    monkeypatch.setattr(cache, "load_encoder", lambda *_: None)
    monkeypatch.setattr(cache, "to_stored", str)
    profile = tmp_path / "profile.json"
    profile.write_text("{}")
    profile.with_name("status.json").write_text("{}")
    output = tmp_path / "cache"
    output.mkdir()
    calls = []

    def read(row, _):
        calls.append(row["ecg_id"])
        return np.ones((5000, 12))

    vector = np.ones(768) / np.sqrt(768)
    feature = {
        "qrs": vector,
        "global_pair": vector,
        "region_vectors": np.stack([vector] * 3),
        "qrs_bounds": np.array([[1.0, 1.3], [2.0, 2.3], [3.0, 3.3]]),
    }
    monkeypatch.setattr(cache, "read_waveform", read)
    monkeypatch.setattr(cache, "record_features", lambda *_: feature)
    result = cache.extract(output, {}, profile)
    assert calls == [1]
    assert result["stage"] == "native_cache_extraction_v2"
    assert result["retention_gate_passed"] is True
    assert result["extraction_forward_count"] == 1
    with np.load(output / "native_features_local.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["qrs"][0], vector)
        assert saved["qrs"].dtype == np.float64


def test_v2_profile_admission_requires_v2_source_and_exact_status(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "runtime_fingerprint", lambda: {"threads": 2})
    path = tmp_path / "result.json"
    times = [0.5, 0.6, 0.5, 0.5]
    value = {
        "stage": "native_cache_cpu_profile_v2",
        "status": "completed",
        "source_sha256": {"v2": "hash"},
        "input_sha256": {},
        "runtime": {"threads": 2},
        "checkpoint_sha256": cache.CHECKPOINT_SHA256,
        "profile_ecg_ids": [1, 2, 3, 4],
        "candidate_patients": 1993,
        "profile_gate_passed": True,
        "gates": dict.fromkeys(
            [
                "finite_dense_shape",
                "all_parameters_frozen",
                "four_fixed_records_passed",
                "real_record_budget",
                "rss_budget",
                "profile_budget",
                "projected_extraction_budget",
            ],
            True,
        ),
        "real_record_seconds": times,
        "load_seconds": 2,
        "seconds": 10,
        "peak_rss_bytes": 10000,
        "projected_extraction_seconds": 2 + 1.5 * max(times) * 1993,
        "waveform_sha256": {},
    }
    path.write_text(json.dumps(value))
    path.with_name("status.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "result_sha256": sha256_file(path),
            }
        )
    )
    cache.validate_profile(path, {"v2": "hash"}, {}, [1, 2, 3, 4])
    with pytest.raises(ValueError, match="source, input, model, runtime or cohort"):
        cache.validate_profile(path, {"v1": "hash"}, {}, [1, 2, 3, 4])
    path.with_name("status.json").write_text(json.dumps({"status": "failed"}))
    with pytest.raises(ValueError, match="final status"):
        cache.validate_profile(path, {"v2": "hash"}, {}, [1, 2, 3, 4])
