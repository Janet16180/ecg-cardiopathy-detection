"""Checks of the Experiment 023 tabular inputs, 250 Hz encoder inputs and component masks."""

import numpy as np
import pandas as pd
import pytest

from ecg_experiment.echonext_readout import (
    age_sex_inputs,
    cosine_summary,
    downsample_250,
    jepa_input_250,
    measured,
    pooled_lead_statistics,
    tabular_inputs,
    train_medians,
    xecg_input_250,
)
from ecg_experiment.xecg import preprocess_xecg


def rows() -> pd.DataFrame:
    return pd.DataFrame({
        "age_at_ecg": [40, 60, 80], "sex": ["male", "female", "male"],
        "ventricular_rate": [60.0, 80.0, 100.0], "atrial_rate": [60.0, np.nan, 100.0],
        "pr_interval": [np.nan, 160.0, 200.0], "qrs_duration": [90.0, 100.0, 110.0],
        "qt_corrected": [400.0, np.nan, 440.0],
        "lvef_lte_45_flag": [0, 1, 0], "lvef_value": [60.0, 30.0, np.nan],
    })


def slow_waves(samples: int, rate: int) -> np.ndarray:
    time = np.arange(samples) / rate
    return np.stack([np.sin(2 * np.pi * (1 + lead) * time) * (lead + 1) for lead in range(12)])


def test_tabular_inputs_impute_training_medians_and_flag_atrial_rate_and_pr():
    frame = rows()
    medians = train_medians(frame)
    inputs = tabular_inputs(frame, medians)
    assert inputs.shape == (3, 9)
    assert np.array_equal(inputs[:, :2], [[4, 1], [6, 0], [8, 1]])
    assert inputs[1, 3] == 80.0
    assert inputs[0, 4] == 180.0
    assert inputs[1, 6] == 420.0
    assert np.array_equal(inputs[:, 7:], [[0, 1], [1, 0], [0, 0]])


def test_an_unknown_sex_value_raises():
    frame = rows()
    frame.loc[0, "sex"] = "unknown"
    with pytest.raises(ValueError, match="sex"):
        age_sex_inputs(frame)


def test_downsampling_keeps_the_first_ten_seconds_of_a_slow_waveform():
    signal = np.concatenate([slow_waves(5000, 500), np.zeros((12, 500))], axis=1)
    result = downsample_250(signal)
    assert result.shape == (12, 2500)
    assert np.allclose(result[:, 100:-100], slow_waves(2500, 250)[:, 100:-100], rtol=5e-3, atol=1e-3)


def test_downsampling_rejects_short_waveforms():
    with pytest.raises(ValueError, match="10 s"):
        downsample_250(np.zeros((12, 2500)))


def test_pooled_statistics_equal_those_of_the_concatenated_samples():
    rng = np.random.default_rng(0)
    signals = [rng.normal(lead, 2, (12, 100)) for lead in range(5)]
    mean, std = pooled_lead_statistics(signals)
    joined = np.concatenate(signals, axis=1)
    assert np.allclose(mean, joined.mean(axis=1))
    assert np.allclose(std, joined.std(axis=1))


def test_jepa_input_keeps_leads_i_ii_and_the_precordial_leads():
    signal = np.arange(12, dtype=np.float64)[:, None] * np.ones((12, 2500))
    result = jepa_input_250(signal)
    assert result.dtype == np.float32
    assert np.array_equal(result[:, 0], [0, 1, 6, 7, 8, 9, 10, 11])


def test_xecg_input_from_250_hz_matches_the_500_hz_path_for_a_bandlimited_waveform():
    from_250 = xecg_input_250(slow_waves(2500, 250))
    from_500 = preprocess_xecg(slow_waves(5000, 500))
    assert from_250.shape == (1000, 12)
    assert np.allclose(from_250, from_500, atol=1e-4)


def test_cosine_summary_reports_mean_and_minimum():
    first = np.array([[1.0, 0.0], [1.0, 1.0]])
    second = np.array([[1.0, 0.0], [1.0, 0.0]])
    summary = cosine_summary(first, second)
    assert summary["min_cosine"] == pytest.approx(np.sqrt(0.5))
    assert summary["mean_cosine"] == pytest.approx((1 + np.sqrt(0.5)) / 2)


def test_component_mask_keeps_rows_with_a_measured_echo_value():
    assert np.array_equal(measured(rows(), "lvef_lte_45_flag"), [True, True, False])
