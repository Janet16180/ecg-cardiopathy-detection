"""Metric tests use explicit supports and ties, not a generator or model's own predictions."""

import numpy as np
import pytest

from ecg_experiment.ssl_synthetic_metrics import (
    control_threshold,
    paired_subject_difference,
    positive_map_metrics,
)


def test_uniform_ap_and_fractional_ties_equal_prevalence():
    support = np.zeros((12, 2500), dtype=bool)
    support[0, :300] = True
    metrics = positive_map_metrics(support, np.ones_like(support, dtype=float))
    assert metrics["average_precision"] == pytest.approx(0.01)
    assert metrics["top_one_percent_precision"] == pytest.approx(0.01)
    assert metrics["top_one_percent_recall"] == pytest.approx(0.01)
    perfect = positive_map_metrics(support, support.astype(float))
    assert perfect["average_precision"] == 1.0
    assert perfect["top_one_percent_precision"] == 1.0
    assert perfect["top_one_percent_recall"] == 1.0


def test_whole_subject_macro_prevents_variant_count_from_changing_weight():
    ids = np.array(["a", "a", "a", "b"])
    first = np.array([0.2, 0.2, 0.2, 0.6])
    second = np.zeros(4)
    result = paired_subject_difference(ids, first, second, draws=100, seed=4)
    assert result["difference"] == pytest.approx(0.4)
    assert result["subjects"] == 2
    assert result["cases"] == 4
    assert result == paired_subject_difference(ids, first, second, draws=100, seed=4)
    assert result["skipped_draws"] == 0


def test_threshold_is_joint_higher_quantile_with_strict_ties_negative():
    sham = np.arange(128) / 128
    nuisance = sham[::-1]
    threshold = control_threshold(sham, nuisance)
    assert threshold == np.quantile(np.maximum(sham, nuisance), 0.95, method="higher")
    assert not np.any(np.array([threshold, threshold]) > threshold)


def test_invalid_cases_cannot_disappear_from_denominator():
    with pytest.raises(ValueError, match="both support and background"):
        positive_map_metrics(np.zeros((2, 2)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="finite scores"):
        positive_map_metrics(np.eye(2), np.full((2, 2), np.nan))
    with pytest.raises(ValueError, match="silently removed"):
        paired_subject_difference(np.array([1]), np.array([np.nan]), np.array([0]))
