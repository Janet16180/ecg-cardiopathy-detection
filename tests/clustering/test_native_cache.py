"""Synthetic integrity and pooling checks without local models or real ECG access."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ecg_experiment.clustering import native_cache as cache
from ecg_experiment.files import sha256_file
from ecg_experiment.native_reference_ntt05 import SignalRegion, lead_patch_tokens
from scripts.experiments.clustering import cache_native as runner


@pytest.fixture
def identities(monkeypatch):
    monkeypatch.setattr(cache, "train_patient", lambda _: True)
    frame = pd.DataFrame(
        {
            "ecg_id": np.arange(1, 2000),
            "patient_id": np.arange(1, 2000),
            "strat_fold": np.ones(1999, dtype=int),
            "filename_hr": [f"records500/{index:05d}_hr" for index in range(1, 2000)],
        }
    )
    manifest = {
        "records": [{"ecg_id": index, "patient_id": index, "role": "train"} for index in range(1, 2000)]
    }
    return frame, manifest


def test_roles_quarantine_whole_closed_fold_patient(identities):
    selected, manifest = identities
    other = selected.iloc[[0]].copy()
    other["ecg_id"] = 90000
    other["strat_fold"] = 9
    all_ids = pd.concat([selected, other], ignore_index=True)
    with pytest.raises(ValueError, match="whole-patient training"):
        cache.verify_identity_roles(selected, all_ids, manifest)


def test_roles_require_exact_source_and_manifest(identities):
    selected, manifest = identities
    cache.verify_identity_roles(selected, selected, manifest)
    changed = selected.copy()
    changed.loc[0, "filename_hr"] = "records500/wrong_hr"
    with pytest.raises(ValueError, match="differs from source"):
        cache.verify_identity_roles(changed, selected, manifest)
    manifest["records"][0]["role"] = "validation"
    with pytest.raises(ValueError, match="training role"):
        cache.verify_identity_roles(selected, selected, manifest)


def test_waveform_hashes_precede_reader_and_changes_abort(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DATABASE", tmp_path / "ptbxl_database.csv")
    monkeypatch.setattr(cache, "to_stored", lambda path: str(path))
    path = tmp_path / "records500/00001_hr"
    path.parent.mkdir()
    path.with_suffix(".hea").write_text("synthetic header")
    path.with_suffix(".dat").write_bytes(b"synthetic data")
    hashes = {}

    def read(_):
        assert len(hashes) == 2
        return np.ones((5000, 12)), {"fs": 500, "units": ["mV"] * 12, "sig_name": cache.LEADS}

    monkeypatch.setattr(cache.wfdb, "rdsamp", read)
    row = {"filename_hr": "records500/00001_hr"}
    assert cache.read_waveform(row, hashes).shape == (5000, 12)
    path.with_suffix(".dat").write_bytes(b"changed bytes")
    with pytest.raises(ValueError, match="waveform changed"):
        cache.read_waveform(row, hashes)


def test_missing_files_abort_instead_of_quality_exclusion(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DATABASE", tmp_path / "ptbxl_database.csv")
    with pytest.raises(FileNotFoundError):
        cache.read_waveform({"filename_hr": "records500/missing_hr"}, {})


def test_only_known_detector_quality_error_is_excluded(monkeypatch):
    def detect(_):
        raise ValueError("Fewer than three complete automatic beats")

    monkeypatch.setattr(cache, "automatic_regions", detect)
    with pytest.raises(cache.QualityExclusionError, match="three complete"):
        cache.record_features(None, np.ones((5000, 12)), 1)

    def broken(_):
        raise ValueError("unexpected detector error")

    monkeypatch.setattr(cache, "automatic_regions", broken)
    with pytest.raises(ValueError, match="unexpected detector") as caught:
        cache.record_features(None, np.ones((5000, 12)), 1)
    assert not isinstance(caught.value, cache.QualityExclusionError)


def test_region_cache_preserves_frozen_patient_pooling_and_precision(monkeypatch):
    rng = np.random.default_rng(10)
    tokens = lead_patch_tokens(
        rng.normal(size=(8, 50, 768)).astype(np.float32),
        ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6"),
        250,
        50,
        2500,
        "test-space",
        "synthetic context",
        "synthetic-record",
    )
    qrs = [SignalRegion(start, start + 0.3, ("V1", "V6")) for start in (1.0, 2.0, 3.0)]
    late = [SignalRegion(start, start + 0.3, ("V1", "V6")) for start in (1.4, 2.4, 3.4)]
    monkeypatch.setattr(cache, "automatic_regions", lambda _: (qrs, late))
    monkeypatch.setattr(cache, "encode_signal", lambda *_: tokens)
    feature = cache.record_features(None, np.ones((5000, 12)), 12)
    expected = cache.descriptors(tokens, qrs, late)
    np.testing.assert_array_equal(feature["qrs"], expected["native_qrs"])
    np.testing.assert_array_equal(feature["global_pair"], expected["native_global_pair"])
    assert feature["region_vectors"].dtype == np.float64
    np.testing.assert_allclose(np.linalg.norm(feature["region_vectors"], axis=1), 1, atol=1e-14)
    arrays = cache.pack_features([{"ecg_id": 12, "patient_id": 44}] * 2, [feature, feature])
    assert arrays["qrs"].shape == (2, 768)
    assert arrays["region_vectors"].shape == (6, 768)
    np.testing.assert_array_equal(arrays["region_offsets"], [0, 3, 6])
    np.testing.assert_array_equal(arrays["qrs_bounds"][:3], [[1, 1.3], [2, 2.3], [3, 3.3]])


def test_model_value_error_is_never_a_quality_exclusion(monkeypatch):
    region = SignalRegion(1, 1.3, ("V1", "V6"))
    monkeypatch.setattr(cache, "automatic_regions", lambda _: ([region] * 3, [region] * 3))

    def broken(*_):
        raise ValueError("checkpoint or model failure")

    monkeypatch.setattr(cache, "encode_signal", broken)
    with pytest.raises(ValueError, match="model failure") as caught:
        cache.record_features(None, np.ones((5000, 12)), 12)
    assert not isinstance(caught.value, cache.QualityExclusionError)


def profile_receipt():
    times = [0.5, 0.6, 0.5, 0.5]
    return {
        "stage": "native_cache_cpu_profile",
        "status": "completed",
        "source_sha256": {"source": "hash"},
        "input_sha256": {"input": "hash"},
        "runtime": {"threads": 2},
        "checkpoint_sha256": runner.CHECKPOINT_SHA256,
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
        "projected_extraction_seconds": 2 + 1.5 * max(times) * 1993,
        "seconds": 10,
        "peak_rss_bytes": 10000,
        "waveform_sha256": {},
    }


def test_profile_admission_binds_status_runtime_and_measured_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "runtime_fingerprint", lambda: {"threads": 2})
    path = tmp_path / "result.json"
    value = profile_receipt()

    def write():
        path.write_text(json.dumps(value))
        path.with_name("status.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "result_sha256": sha256_file(path),
                }
            )
        )

    write()
    runner.validate_profile(path, {"source": "hash"}, {"input": "hash"}, [1, 2, 3, 4])
    value["runtime"] = {"threads": 3}
    write()
    with pytest.raises(ValueError, match="runtime or cohort"):
        runner.validate_profile(path, {"source": "hash"}, {"input": "hash"}, [1, 2, 3, 4])
    value["runtime"] = {"threads": 2}
    value["real_record_seconds"] = [3.1] * 4
    write()
    with pytest.raises(ValueError, match="real-record measurements"):
        runner.validate_profile(path, {"source": "hash"}, {"input": "hash"}, [1, 2, 3, 4])


def test_timeout_stops_without_model_or_data_access(monkeypatch):
    monkeypatch.setattr(runner.time, "perf_counter", lambda: 2.0)
    with pytest.raises(TimeoutError, match="deadline"):
        runner.budget(1.0)


@pytest.mark.parametrize("failure", [ValueError("model failed"), OSError("read failed")])
def test_extraction_aborts_on_model_or_file_error(tmp_path, monkeypatch, failure):
    frame = pd.DataFrame(
        {
            "ecg_id": [1, 2],
            "patient_id": [11, 22],
            "strat_fold": [1, 2],
            "filename_hr": ["records500/1_hr", "records500/2_hr"],
        }
    )
    monkeypatch.setattr(runner, "verified_cohort", lambda: (frame, None, None, {}))
    monkeypatch.setattr(runner, "validate_profile", lambda *_: {"waveform_sha256": {}})
    monkeypatch.setattr(runner, "load_encoder", lambda *_: None)
    monkeypatch.setattr(runner, "to_stored", str)
    profile = tmp_path / "profile.json"
    profile.write_text("{}")
    profile.with_name("status.json").write_text("{}")
    output = tmp_path / "extraction"
    output.mkdir()

    def broken(*_):
        raise failure

    monkeypatch.setattr(runner, "read_waveform", broken)
    with pytest.raises(type(failure), match=str(failure)):
        runner.extract(output, {}, profile)
    assert not (output / "native_features_local.npz").exists()


def test_extraction_quality_exclusions_retain_identity_and_float64_vectors(tmp_path, monkeypatch):
    frame = pd.DataFrame(
        {
            "ecg_id": [1, 2],
            "patient_id": [11, 22],
            "strat_fold": [1, 2],
            "filename_hr": ["records500/1_hr", "records500/2_hr"],
        }
    )
    monkeypatch.setattr(runner, "verified_cohort", lambda: (frame, None, None, {}))
    monkeypatch.setattr(runner, "validate_profile", lambda *_: {"waveform_sha256": {}})
    monkeypatch.setattr(runner, "load_encoder", lambda *_: None)
    monkeypatch.setattr(runner, "to_stored", str)
    profile = tmp_path / "profile.json"
    profile.write_text("{}")
    profile.with_name("status.json").write_text("{}")
    output = tmp_path / "extraction"
    output.mkdir()

    def read(row, _):
        if row["ecg_id"] == 2:
            raise cache.QualityExclusionError("Expected finite ten-second twelve-lead waveform")
        return np.ones((5000, 12))

    vector = np.ones(768) / np.sqrt(768)
    feature = {
        "qrs": vector,
        "global_pair": vector,
        "region_vectors": np.stack([vector] * 3),
        "qrs_bounds": np.array([[1.0, 1.3], [2.0, 2.3], [3.0, 3.3]]),
    }
    monkeypatch.setattr(runner, "read_waveform", read)
    monkeypatch.setattr(runner, "record_features", lambda *_: feature)
    result = runner.extract(output, {}, profile)
    assert result["accepted_patients"] == 1
    assert result["excluded_patients"] == 1
    assert result["retention_gate_passed"] is False
    assert result["extraction_forward_count"] == 1
    assert "status.json" not in " ".join(result["artifact_sha256"])
    exclusions = json.loads((output / "exclusions_local.json").read_text())
    assert exclusions[0]["patient_id"] == 22
    with np.load(output / "native_features_local.npz", allow_pickle=False) as saved:
        assert saved["qrs"].dtype == np.float64
        np.testing.assert_array_equal(saved["patient_ids"], [11])
        np.testing.assert_array_equal(saved["region_offsets"], [0, 3])
