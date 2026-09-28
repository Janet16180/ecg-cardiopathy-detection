"""Checks of the EchoNext quality rules and CPC input scaling."""

import numpy as np
import pytest

from ecg_experiment.echonext import assess, to_cpc_scale


def ecg_like(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    time = np.arange(2500) / 250
    beats = np.sin(2 * np.pi * 1.2 * time) ** 21
    return beats[None, :] * rng.uniform(0.5, 2, (12, 1)) + rng.normal(0, 0.01, (12, 2500))


def test_a_clean_waveform_passes():
    assert assess(ecg_like()) == []


def test_a_lead_constant_for_part_of_the_recording_is_flagged():
    signal = ecg_like()
    signal[6:, :1250] = -0.1
    assert assess(signal) == ["flat_segment"]


def test_nonfinite_and_constant_leads_are_flagged():
    signal = ecg_like()
    signal[0, 10] = np.nan
    assert assess(signal) == ["nonfinite"]
    signal = ecg_like()
    signal[3] = 0.2
    assert "constant_lead" in assess(signal)


def test_wrong_shape_raises():
    with pytest.raises(ValueError, match="Expected"):
        assess(np.zeros((12, 5000)))


def test_scaling_matches_historical_per_lead_statistics():
    rng = np.random.default_rng(1)
    signals = rng.normal(3, 2, (50, 12, 2500))
    lead_mean, lead_std = signals.mean(axis=(0, 2)), signals.std(axis=(0, 2))
    target_mean, target_std = np.linspace(-0.1, 0.1, 12), np.linspace(0.1, 0.3, 12)
    scaled = to_cpc_scale(signals, lead_mean, lead_std, target_mean, target_std)
    assert np.allclose(scaled.mean(axis=(0, 2)), target_mean, atol=1e-5)
    assert np.allclose(scaled.std(axis=(0, 2)), target_std, atol=1e-5)
