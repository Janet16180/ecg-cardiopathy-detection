"""Synthetic checks of the record-level ECG quality policy."""

import numpy as np
import pytest

from ecg_experiment.ecg_quality import assess, longest_constant_runs


def _ecg(seed=0):
    """Beat-like spikes at 75 bpm on a small noise floor, consistent limb leads."""
    rng = np.random.default_rng(seed)
    time = np.arange(5000) / 500
    beats = np.exp(-(((time % 0.8) - 0.4) ** 2) / (2 * 0.01**2))
    signal = beats * rng.uniform(0.5, 1.5, (12, 1)) + rng.normal(0, 0.02, (12, 5000))
    signal[2] = signal[1] - signal[0]
    return signal.astype(np.float32)


def test_clean_ecg_passes_without_flags():
    assert assess(_ecg()) == ([], [])


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (lambda s: s.__setitem__((4, 10), np.nan), "nonfinite"),
        (lambda s: s.__setitem__(7, 0.3), "constant_lead"),
        (lambda s: s.__setitem__((7, slice(1000, 1600)), 0.1), "flat_segment"),
        (lambda s: s.__setitem__((slice(None), slice(2000, 2080)), 32.767), "adc_rail"),
        (lambda s: s.__setitem__((5, 300), 25.0), "extreme_amplitude"),
    ],
)
def test_each_exclusion_rule(change, reason):
    signal = _ecg()
    change(signal)
    reasons, _ = assess(signal)
    assert reason in reasons


def test_rail_just_below_the_exact_16_bit_limit_is_caught():
    signal = _ecg()
    signal[:, 2000:2080] = 32.69
    assert "adc_rail" in assess(signal)[0]


def test_near_flat_and_noise_dominated_records():
    rng = np.random.default_rng(1)
    quiet = rng.normal(0, 0.005, (12, 5000)).astype(np.float32)
    assert "near_flat_record" in assess(quiet)[0]
    muscle = np.sin(2 * np.pi * 80 * np.arange(5000) / 500)
    noise = (_ecg() + muscle).astype(np.float32)
    assert "noise_dominated" in assess(noise)[0]


def test_review_flags_do_not_exclude():
    signal = _ecg()
    signal[3, 100] = 12.0
    signal[2] += 0.2
    reasons, flags = assess(signal)
    assert reasons == []
    assert flags == ["amplitude_over_10mv", "limb_identity_violated"]


def test_longest_constant_runs_counts_samples():
    signal = np.array([[0, 1, 1, 1, 2], [3, 3, 3, 3, 3]], dtype=np.float32)
    assert longest_constant_runs(signal).tolist() == [3, 5]


def test_wrong_shape_raises():
    with pytest.raises(ValueError, match="Expected"):
        assess(np.zeros((5000, 12), dtype=np.float32))
