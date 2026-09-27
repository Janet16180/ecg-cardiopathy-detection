"""Checks of the CODE-15 to CPC conversion."""

import numpy as np
import pandas as pd

from ecg_experiment.code15_cpc import (
    active_bounds,
    amplitude_factor,
    central_window,
    patient_split,
    to_cpc,
    window_quality,
)


def _tracing(active, total=4096):
    tracing = np.zeros((total, 12), dtype=np.float32)
    left = (total - active) // 2
    time = np.arange(active) / 400
    tracing[left:left + active] = np.sin(2 * np.pi * 5 * time)[:, None] + 0.1
    return tracing, left


def test_padding_is_found_on_both_sides():
    tracing, left = _tracing(2934)
    assert active_bounds(tracing) == (left, left + 2934)
    assert active_bounds(np.zeros((4096, 12))) == (0, 0)


def test_short_recordings_have_no_window():
    assert central_window(_tracing(2934)[0]) is None
    assert central_window(_tracing(4096)[0]).shape == (12, 4000)


def test_resampling_keeps_duration_and_frequency():
    window = central_window(_tracing(4096)[0])
    signal = to_cpc(window)
    assert signal.shape == (12, 2500)
    spectrum = np.abs(np.fft.rfft(signal[0, :1250] - signal[0, :1250].mean()))
    assert np.fft.rfftfreq(1250, 1 / 250)[spectrum.argmax()] == 5


def test_quality_flags_flat_leads():
    window = central_window(_tracing(4096)[0])
    assert not window_quality(window)["flat_segment"]
    window[3, 1000:1400] = 0.2
    assert window_quality(window)["flat_segment"]
    window[5] = 0.0
    assert window_quality(window)["constant_lead"]


def test_patient_split_keeps_patients_together():
    patients = pd.Series(np.repeat(np.arange(200), 3))
    split = patient_split(patients, 0.05, seed=1)
    assert (split.groupby(patients).nunique() == 1).all()
    assert split.eq("monitoring").sum() == 30


def test_amplitude_factor_recovers_a_known_scale():
    rng = np.random.default_rng(0)
    ptb = rng.normal(0, 0.15, (50, 12, 500))
    assert abs(amplitude_factor(ptb, ptb * 2.2) - 1 / 2.2) < 1e-9
