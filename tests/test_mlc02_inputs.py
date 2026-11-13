"""Tests of the MLC-02 Stage 1 readers, cache index and deterministic input pipelines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import wfdb

from ecg_experiment import mlc02_waveforms as waves
from ecg_experiment.cpu_screening041 import RAW_FEATURES
from ecg_experiment.eda.signals import LEADS
from ecg_experiment.external_encoders import jepa_input, xecg_input
from ecg_experiment.mlc01_dataset import array_sha256, microvolts
from ecg_experiment.mlc02_features import (
    measure,
    model_input,
    pending_shards,
    prepare,
    profile_positions,
    save_shard,
    shard_bounds,
)
from ecg_experiment.mlc02_integrity import compare_measurements, compare_vectors

SHUFFLED = ["V6", "I", "AVR", "II", "V1", "III", "aVL", "V2", "aVF", "V3", "V5", "V4"]


def identity_frame(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Small identity table with record key, role and source per row."""
    return pd.DataFrame(
        {
            "record_key": [key for key, _, _ in rows],
            "source": [source for _, _, source in rows],
            "family": [source for _, _, source in rows],
            "record": [key.split(":")[1] for key, _, _ in rows],
            "role": [role for _, role, _ in rows],
            "raw_path": [f"data/raw/{key}" for key, _, _ in rows],
            "included": True,
            "fs": 500,
            "n_samples": 5000,
            "sha256_10s": [f"h{i}" for i in range(len(rows))],
            "manifest_row": np.arange(len(rows)),
        }
    )


def digital_record(seed: int) -> np.ndarray:
    """Integer microvolt samples (5000, 12) in canonical lead order."""
    rng = np.random.default_rng(seed)
    return np.round(rng.normal(0, 400, size=(5000, 12))).astype(np.int64)


def write_wfdb(directory: Path, name: str, digital: np.ndarray) -> Path:
    """Write a 500 Hz format-16 record whose columns are stored in the ``SHUFFLED`` lead order."""
    order = [[lead.upper() for lead in LEADS].index(lead.upper()) for lead in SHUFFLED]
    wfdb.wrsamp(
        name,
        fs=500,
        units=["mV"] * 12,
        sig_name=SHUFFLED,
        d_signal=digital[:, order].astype(np.int16),
        fmt=["16"] * 12,
        adc_gain=[1000.0] * 12,
        baseline=[0] * 12,
        write_dir=str(directory),
    )
    return directory / name


def test_read_identity_reads_identity_columns_only(tmp_path: Path) -> None:
    frame = identity_frame([("ptbxl:1", "train", "ptbxl"), ("sph:A1", "test_external", "sph")])
    frame = frame.drop(columns="manifest_row").assign(b_lvh=[1, 0], a_MI=[0, 1], normal=[1, 0])
    frame.loc[len(frame)] = {**frame.iloc[0].to_dict(), "record_key": "ptbxl:2", "included": False}
    path = tmp_path / "manifest.csv"
    frame.to_csv(path, index=False)
    identity = waves.read_identity(path, expected=2)
    assert list(identity.columns) == waves.IDENTITY_COLUMNS + ["manifest_row"]
    assert identity["record_key"].tolist() == ["ptbxl:1", "sph:A1"]
    with pytest.raises(ValueError, match="Expected 3 included"):
        waves.read_identity(path, expected=3)
    frame.loc[0, "source"] = "echonext"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Unknown source"):
        waves.read_identity(path, expected=2)


def test_compare_manifests_reports_identity_and_inclusion(tmp_path: Path) -> None:
    frame = identity_frame(
        [("ptbxl:1", "train", "ptbxl"), ("cpsc_2018_extra:Q1", "train", "cpsc_2018_extra")]
    )
    frame = frame.drop(columns="manifest_row")
    frame.to_csv(tmp_path / "used.csv", index=False)
    frame.assign(included=[True, False]).to_csv(tmp_path / "other.csv", index=False)
    used, other = waves.read_rows(tmp_path / "used.csv"), waves.read_rows(tmp_path / "other.csv")
    assert used["included"].dtype == bool
    record = waves.compare_manifests(used, other)
    assert record["identity_equal"]
    assert record["only_used"] == {"cpsc_2018_extra/train": 1}
    assert record["only_other"] == {}
    changed = other.copy()
    changed.loc[1, "sha256_10s"] = "other"
    assert not waves.compare_manifests(used, changed)["identity_equal"]


def test_cache_index_numbers_rows_within_blocks_in_manifest_order() -> None:
    rows = [
        ("ptbxl:1", "train", "ptbxl"),
        ("ningbo:A", "train", "ningbo"),
        ("ptbxl:2", "train", "ptbxl"),
        ("ptbxl:3", "validation", "ptbxl"),
        ("ningbo:B", "train", "ningbo"),
    ]
    index = waves.cache_index(identity_frame(rows))
    assert index["position"].tolist() == [0, 1, 2, 3, 4]
    assert index["block"].tolist() == [
        "train__ptbxl",
        "train__ningbo",
        "train__ptbxl",
        "validation__ptbxl",
        "train__ningbo",
    ]
    assert index["row"].tolist() == [0, 0, 1, 0, 1]
    assert waves.block_sizes(index) == {"train__ptbxl": 2, "train__ningbo": 2, "validation__ptbxl": 1}
    tasks = waves.chunk_tasks(index, Path("/cache"), 1)
    assert [task["name"] for task in tasks] == [
        "train__ptbxl__000000",
        "train__ptbxl__000001",
        "train__ningbo__000000",
        "train__ningbo__000001",
        "validation__ptbxl__000000",
    ]


def test_guard_refuses_closed_and_unlisted_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(waves, "GUARD", {"closed": frozenset(), "allowed": frozenset()})
    with pytest.raises(PermissionError):
        waves.guard_path("data/raw/challenge-2020/1.0.2/training/georgia/g1/E00001", "georgia")
    closed = frozenset({"georgia:training/georgia/g1/E00002"})
    allowed = frozenset(
        {
            "data/raw/challenge-2020/1.0.2/training/georgia/g1/E00001",
            "data/raw/challenge-2020/1.0.2/training/georgia/g1/E00002",
        }
    )
    waves.init_worker(closed, allowed)
    assert (
        waves.guard_path("data/raw/challenge-2020/1.0.2/training/georgia/g1/E00001", "georgia").name
        == "E00001"
    )
    with pytest.raises(PermissionError):
        waves.guard_path("data/raw/challenge-2020/1.0.2/training/georgia/g1/E00002", "georgia")
    with pytest.raises(PermissionError):
        waves.guard_path("data/raw/challenge-2020/1.0.2/training/georgia/g1/E00003", "georgia")


def test_read_window_returns_canonical_lead_order(tmp_path: Path) -> None:
    digital = digital_record(0)
    path = write_wfdb(tmp_path, "rec", digital)
    window = waves.read_window(path, "ptbxl")
    assert window.shape == (5000, 12)
    np.testing.assert_array_equal(np.round(window * 1000).astype(np.int64), digital)


def test_float32_storage_keeps_the_manifest_hash() -> None:
    digital = digital_record(1)
    window = digital / 1000.0
    expected = array_sha256(microvolts(window))
    stored = np.ascontiguousarray(window.T, dtype=np.float32)
    assert array_sha256(microvolts(stored.T)) == expected
    half = np.random.default_rng(2).normal(0, 0.5, size=(5000, 12)).astype(np.float16).astype(np.float64)
    assert array_sha256(microvolts(np.ascontiguousarray(half.T, dtype=np.float32).T)) == array_sha256(
        microvolts(half)
    )


def test_cache_chunks_write_and_verify_with_missing_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = {}
    for number in range(3):
        digital = digital_record(10 + number)
        if number == 1:
            digital[100:104, 4] = waves.MISSING_DIGITAL
        path = write_wfdb(tmp_path, f"r{number}", digital)
        window = digital / 1000.0
        window[digital == waves.MISSING_DIGITAL] = np.nan
        records[str(path)] = (digital, array_sha256(microvolts(window)))
    identity = identity_frame([(f"ptbxl:{n}", "train", "ptbxl") for n in range(3)])
    identity["raw_path"] = list(records)
    identity["sha256_10s"] = [value[1] for value in records.values()]
    index = waves.cache_index(identity)
    cache = tmp_path / "cache"
    cache.mkdir()
    for name, size in waves.block_sizes(index).items():
        waves.open_block(cache, name, size, "w+")
    monkeypatch.setattr(waves, "GUARD", {"closed": frozenset({"x"}), "allowed": frozenset(records)})
    results = [row for task in waves.chunk_tasks(index, cache, 2) for row in waves.write_chunk(task)]
    assert all(row["sha256_read_ok"] and row["sha256_float32_ok"] for row in results)
    assert [row["missing_samples_10s"] for row in results] == [0, 4, 0]
    checks = [row for task in waves.chunk_tasks(index, cache, 2) for row in waves.verify_chunk(task)]
    assert all(row["sha256_cache_ok"] for row in checks)
    assert [row["sha256_f32"] for row in checks] == [row["sha256_f32"] for row in results]
    array = waves.open_block(cache, "train__ptbxl", 3, "r")
    assert array.shape == (3, 12, 5000)
    assert array.dtype == np.float32
    assert np.isnan(array[1, 4, 100:104]).all()
    assert np.isfinite(array[0]).all()
    filled = model_input(np.asarray(array[1]))
    assert np.isfinite(filled).all()
    assert (filled[4, 100:104] == np.float32(-32.768)).all()
    np.testing.assert_array_equal(filled[:, :100], array[1][:, :100])


def test_missing_samples_must_be_the_negative_rail(tmp_path: Path) -> None:
    digital = digital_record(3)
    digital[10, 0] = waves.MISSING_DIGITAL
    path = write_wfdb(tmp_path, "rail", digital)
    window = waves.read_window(path, "ptbxl")
    assert waves.check_missing_rail(path, window) == 1
    window[20, 1] = np.nan
    with pytest.raises(ValueError, match="negative rail"):
        waves.check_missing_rail(path, window)


def constant_leads() -> np.ndarray:
    """Float32 [12, 5000] signal whose lead j is the constant j."""
    return np.repeat(np.arange(12, dtype=np.float32)[:, None], 5000, axis=1)


def test_jepa_input_selects_leads_i_ii_and_v1_to_v6() -> None:
    output = jepa_input(constant_leads())
    assert output.shape == (8, 2500)
    assert output.dtype == np.float32
    np.testing.assert_allclose(output.mean(axis=1), [0, 1, 6, 7, 8, 9, 10, 11], atol=1e-5)


def test_xecg_input_is_time_major_at_100_hz() -> None:
    output = xecg_input(constant_leads())
    assert output.shape == (1000, 12)
    assert output.dtype == np.float32
    np.testing.assert_allclose(output.mean(axis=0), np.arange(12), atol=1e-5)


def test_cpc_input_crops_then_resamples_like_the_released_script() -> None:
    from ecg_experiment import mlc02_cpc

    mlc02_cpc.add_paths()
    pytest.importorskip("resampy")
    assert mlc02_cpc.crop_starts() == [0, 1250, 2500, 3750]
    ramp = np.tile(np.arange(5000, dtype=np.float32) / 5000, (12, 1))
    views = mlc02_cpc.cpc_input(ramp)
    assert views.shape == (4, 12, 600)
    assert views.dtype == np.float32
    for crop, start in enumerate(mlc02_cpc.crop_starts()):
        assert abs(views[crop, 0, 300] - (start + 625) / 5000) < 1e-3
    from scripts.features.extract_ecg_cpc import preprocess

    signal = np.random.default_rng(4).normal(0, 0.3, size=(12, 5000)).astype(np.float32)
    np.testing.assert_array_equal(mlc02_cpc.cpc_input(signal), preprocess(signal))


def test_measure_keeps_status_and_missing_values_for_a_flat_lead() -> None:
    window = np.random.default_rng(5).normal(0, 0.2, size=(2, 12, 5000)).astype(np.float32)
    window[0, 3] = 0.0
    values, status = measure(window)
    assert values.shape == (2, len(RAW_FEATURES))
    assert values.dtype == np.float64
    assert status.loc[0, "status"] == "derivacion_plana"
    assert np.isnan(values[0]).all()
    assert list(status.columns) == ["status", "implausible", "flat_leads"]


def test_shards_cover_every_record_once_and_resume(tmp_path: Path) -> None:
    bounds = shard_bounds(10, 4)
    assert bounds == [(0, 4), (4, 8), (8, 10)]
    prepare(tmp_path, {"plan": 1})
    with pytest.raises(ValueError, match="another plan"):
        prepare(tmp_path, {"plan": 2})
    save_shard(tmp_path, 1, np.zeros((4, 3), np.float32), None, {"start": 4, "stop": 8, "seconds": 1.0})
    assert pending_shards(tmp_path, bounds) == [0, 2]
    assert json.loads((tmp_path / "shard_00001.json").read_text())["stop"] == 8


def test_profile_positions_take_the_first_records_of_each_source() -> None:
    index = pd.DataFrame({"position": np.arange(6), "source": ["a", "a", "b", "a", "b", "b"]})
    assert profile_positions(index, 2).tolist() == [0, 1, 2, 4]


def test_compare_vectors_reports_row_relative_differences() -> None:
    old = np.array([[1.0, -2.0], [0.5, 0.5]])
    new = old + np.array([[0.0, 2e-4], [0.0, 0.0]])
    stats = compare_vectors(new, old)
    assert stats["bit_identical_records"] == 1
    assert stats["records_rel_above_tolerance"] == 0
    assert stats["max_abs"] == pytest.approx(2e-4)
    assert stats["max_rel"] == pytest.approx(1e-4)


def test_compare_measurements_counts_missing_and_zero_mismatches() -> None:
    old = np.full((2, len(RAW_FEATURES)), 1.5)
    new = old.copy()
    old[0, 0], new[0, 1] = np.nan, np.nan
    old[1, 2], new[1, 2] = 0.0, 0.3
    stats = compare_measurements(new, old)
    assert stats["missing_pattern_mismatches"] == 2
    assert stats["zero_mismatches"] == 1
    assert stats["records_rel_above_tolerance"] == 2
    assert stats["records_equal_after_six_digits"] == 0
