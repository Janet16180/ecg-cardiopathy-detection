"""Artificial checks of numerical intervention truth and failure accounting."""

from contextlib import nullcontext

import numpy as np
import pytest

from ecg_experiment import ssl_synthetic_generated_quality as quality
from ecg_experiment import ssl_synthetic_generated_quality_run as runner


def test_float64_standardization_matches_declared_quantiles():
    raw = np.random.default_rng(3).normal(size=(8, 5000)).astype(np.float32)
    values, parameters = quality.standardize(raw)
    center = np.median(raw.astype(np.float64), axis=1)
    quantiles = np.quantile(raw.astype(np.float64), [0.05, 0.95], axis=1, method="linear")
    expected = ((raw.astype(np.float64) - center[:, None]) / np.diff(quantiles, axis=0).T).astype(np.float32)
    assert np.array_equal(values, expected)
    assert parameters["median"] == center.tolist()


def test_all_five_affected_limb_leads_and_exact_hull():
    raw = np.random.default_rng(4).normal(size=(8, 5000)).astype(np.float32)
    values, _ = quality.standardize(raw)
    variants, strict, support = quality.intervention_bank(values, 0)
    result = quality.intervention_metrics(variants, strict, support)
    assert np.array_equal(np.flatnonzero(strict[0].any(axis=1)), [1, 2, 3, 4, 5])
    assert np.all(~support | strict)
    for mode in ["focal", "repeated"]:
        assert result[mode]["outside_intended_hull_cells"] == 0
        assert result[mode]["retained_energy_fraction"] >= 0.99
        assert result[mode]["paired_difference_oracle_ap"] == 1
    variants[2, 6, 4000] += 1
    strict = variants[2:].astype(np.float64) != variants[1].astype(np.float64)
    support = np.abs(variants[2:].astype(np.float64) - variants[1].astype(np.float64)) > 0.01
    assert (
        quality.intervention_metrics(variants, strict, support)["focal"]["outside_intended_hull_cells"] == 1
    )


def test_flat_lead_admission_rejects_without_altering_input():
    raw = np.zeros((8, 5000), dtype=np.float32)
    original = raw.copy()
    with pytest.raises(ValueError, match="small robust range"):
        quality.standardize(raw)
    assert np.array_equal(raw, original)


def test_activity_coincidence_sensitive_to_alignment():
    waveform = np.zeros(5000)
    waveform[::500] = 1
    aligned = np.tile(waveform, (8, 1)).astype(np.float32)
    scores = quality.coherence(aligned, 0)
    assert scores["native_jaccard"] == 1
    assert scores["difference"] > 0.8
    assert quality.coherence(aligned, 0) == scores


def test_spectral_zero_denominator_is_explicit():
    raw = np.zeros((8, 5000), dtype=np.float32)
    result = quality.morphology(raw, raw)
    assert result["zero_power_denominator"] == [True] * 8
    assert result["spectral_power_ratios"]["25_to_100"] == [None] * 8


def test_failed_gpu_stage_still_charges_lock_wall(monkeypatch):
    monkeypatch.setattr(runner, "gpu_lock", lambda *args, **kwargs: nullcontext())
    ticks = iter([10.0, 13.5])
    monkeypatch.setattr(runner.time, "perf_counter", lambda: next(ticks))
    clock = {}
    with pytest.raises(RuntimeError, match="profile failed"), runner.measured_gpu_lock(clock):
        raise RuntimeError("profile failed")
    assert clock["gpu_cost"] == 3.5


def test_failed_predecessor_replay_counts_only_actual_calls(tmp_path, monkeypatch):
    quality.save_array(tmp_path / "noise.npy", np.zeros((4, 1, 8, 5000), dtype=np.float32))
    counts = {"cpu_predecessor_replay": 0}
    attempts = dict(counts)

    def fail(model, noise):
        raise ValueError("actual attempted forward failed")

    monkeypatch.setattr(runner.shared, "generate", fail)
    with pytest.raises(ValueError, match="actual attempted forward"):
        runner.predecessor_closure(tmp_path, {"case_files": [{"raw": "unused"}]}, None, counts, attempts)
    assert counts["cpu_predecessor_replay"] == 0
    assert attempts["cpu_predecessor_replay"] == 1


def test_partial_bank_bytes_survive_a_later_failure(tmp_path):
    noise = np.zeros((128, 1, 8, 5000), dtype=np.float32)
    quality.save_array(tmp_path / "noise.npy", noise)
    banks = runner.create_banks(tmp_path, noise)
    raw = np.full((8, 5000), 0.125, dtype=np.float32)
    banks["raw"][0] = raw
    banks["raw"].flush()
    with pytest.raises(RuntimeError, match="subsequent quality rejection"):
        raise RuntimeError("subsequent quality rejection")
    retained = np.load(tmp_path / "raw.npy", mmap_mode="r", allow_pickle=False)
    assert np.array_equal(retained[0], raw)
    assert retained.shape == (128, 8, 5000)
    assert not np.any(retained[1:])
