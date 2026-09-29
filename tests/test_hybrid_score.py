"""Checks of the Experiment 031 hybrid-score helpers."""

import numpy as np
import pytest

from ecg_experiment.hybrid_score import (
    either_thresholds,
    fit_stack,
    normal_standardizer,
    referred,
    rule_thresholds,
    standardize,
)
from ecg_experiment.referral_budget import budget_threshold


def test_standardize_centres_and_scales_on_the_normals():
    normals = np.array([1.0, 2.0, 3.0, 4.0])
    stats = normal_standardizer(normals)
    z = standardize(normals, stats)
    assert z.mean() == pytest.approx(0.0)
    assert z.std() == pytest.approx(1.0)


def test_standardizer_rejects_constant_scores():
    with pytest.raises(ValueError, match="no spread"):
        normal_standardizer(np.ones(5))


def test_either_refers_at_most_the_budget_of_normals():
    rng = np.random.default_rng(0)
    for correlation in (0.0, 0.5, 0.95):
        base = rng.standard_normal(200)
        readout = base
        distance = correlation * base + np.sqrt(1 - correlation**2) * rng.standard_normal(200)
        for per_mille in (10, 20, 50, 100):
            thresholds = np.array(either_thresholds(readout, distance, per_mille))
            limit = per_mille * 200 // 1000
            count = referred(np.column_stack([readout, distance]), thresholds).sum()
            assert count <= limit
            assert count >= limit // 2


def test_either_equals_the_single_rule_when_the_scores_agree():
    scores = np.random.default_rng(1).permutation(np.arange(1.0, 201.0))
    thresholds = either_thresholds(scores, 2 * scores, 50)
    assert thresholds == (budget_threshold(scores, 50), 2 * budget_threshold(scores, 50))


def test_either_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="differ in length"):
        either_thresholds(np.arange(5.0), np.arange(4.0), 50)


def test_rule_thresholds_dispatch_on_columns():
    rng = np.random.default_rng(2)
    single = rng.random(100)
    pair = rng.random((100, 2))
    assert rule_thresholds(single, 50)[0] == budget_threshold(single, 50)
    assert referred(single, rule_thresholds(single, 50)).sum() == 5
    assert len(rule_thresholds(pair, 50)) == 2


def test_stack_weights_the_informative_score():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 2000)
    informative = y + 0.5 * rng.standard_normal(2000)
    noise = rng.standard_normal(2000)
    model = fit_stack(np.column_stack([informative, noise]), y)
    assert model.coef_[0, 0] > 1.0
    assert abs(model.coef_[0, 1]) < 0.2
