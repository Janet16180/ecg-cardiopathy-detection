"""Scientific coverage and failure checks for waveform-only ST localization."""

import numpy as np
import pytest

from ecg_experiment import waveform_st060


def test_missing_windows_remain_in_candidate_domain(monkeypatch):
    anchors = np.array([250, 500, 750, 8000, 8200, 8400, 15500, 16000])
    monkeypatch.setattr(waveform_st060, "r_peaks", lambda signal, fs: anchors)
    features = waveform_st060.waveform_features(np.zeros((22500, 2)))
    np.testing.assert_array_equal(features["centers"], [45, 75])
    assert not features["inference_failed"]
    np.testing.assert_array_equal(features["valid_windows"], [True, False])
    np.testing.assert_array_equal(features["st"], np.zeros((2, 2)))


def test_failed_reference_does_not_receive_chance_hit(monkeypatch):
    monkeypatch.setattr(waveform_st060, "r_peaks", lambda signal, fs: np.array([8000, 8500, 9000]))
    features = waveform_st060.waveform_features(np.zeros((22500, 2)))
    gold = {
        "centers": np.array([45, 75]),
        "evaluation_mask": np.array([True, True]),
        "targets": np.array([[True, True], [False, False]]),
        "axis_targets": np.array([[True, True], [False, False]]),
    }
    prior = {
        "record": "failure",
        "patient": "failure",
        "eligible_st": True,
        "eligible_axis": True,
        "st": {"hit": 1.0},
        "channels": ["first", "second"],
    }
    row = waveform_st060.record_metrics(features, gold, prior)
    assert row["inference_failed"]
    assert row["st"]["hit"] == row["whole"]["hit"] == 0
    assert row["st"]["axis_hit"] == 0


def test_candidate_domain_cannot_silently_drop_windows():
    features = {"centers": np.array([45])}
    gold = {"centers": np.array([45, 75])}
    with pytest.raises(ValueError, match="candidate domain changed"):
        waveform_st060.record_metrics(features, gold, {})


def test_detector_matching_is_one_to_one():
    metrics = waveform_st060.detector_coverage(np.array([250, 260, 500]), np.array([250, 500]))
    assert metrics["matched"] == 2
    assert metrics["sensitivity"] == 1
    assert metrics["precision"] == pytest.approx(2 / 3)
