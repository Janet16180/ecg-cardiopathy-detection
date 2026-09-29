"""Checks of the Experiment 030 referral-budget helpers."""

import numpy as np
import pytest

from ecg_experiment.referral_budget import (
    bootstrap_counts,
    budget_threshold,
    count_above,
    group_sums,
    referral_share,
    referrals_per_1000,
    wilson_interval,
)


def test_budget_threshold_leaves_floor_of_budget_times_m_normals_above():
    scores = np.random.default_rng(0).permutation(np.arange(1.0, 201.0))
    threshold = budget_threshold(scores, 50)
    assert threshold == 190.0
    assert (scores > threshold).sum() == 10
    assert (scores > budget_threshold(scores, 10)).sum() == 2
    assert budget_threshold(scores[:50], 10) == scores[:50].max()


def test_budget_threshold_rejects_a_budget_outside_the_range():
    with pytest.raises(ValueError, match="Budget out of range"):
        budget_threshold(np.arange(10.0), 1000)
    with pytest.raises(ValueError, match="No normal scores"):
        budget_threshold(np.array([]), 50)


def test_new_normals_exceed_the_threshold_at_the_expected_beta_mean():
    rng = np.random.default_rng(1)
    rates = [1 - budget_threshold(rng.random(100), 50) for _ in range(4000)]
    assert np.mean(rates) == pytest.approx(6 / 101, abs=0.002)


def test_referral_share_averages_the_per_draw_rates():
    rng = np.random.default_rng(2)
    scores, thresholds = rng.normal(size=500), rng.normal(size=37)
    per_draw = count_above(np.sort(scores), thresholds) / len(scores)
    for threshold, rate in zip(thresholds, per_draw, strict=True):
        assert rate == (scores > threshold).mean()
    assert referral_share(scores, thresholds).mean() == pytest.approx(per_draw.mean(), abs=1e-12)


def test_bootstrap_counts_resample_every_group_with_replacement_reproducibly():
    counts = bootstrap_counts(30, 5, seed=4)
    assert counts.shape == (5, 30)
    assert (counts.sum(axis=1) == 30).all()
    assert np.array_equal(counts, bootstrap_counts(30, 5, seed=4))


def test_group_sums_add_rows_by_group():
    sums = group_sums(np.array([0, 2, 0]), np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), 3)
    assert np.array_equal(sums, [[6.0, 8.0], [0.0, 0.0], [3.0, 4.0]])


def test_wilson_interval_matches_a_known_value():
    low, high = wilson_interval(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-4)
    assert high == pytest.approx(0.7634, abs=1e-4)


def test_referrals_per_1000_combine_abnormal_and_normal_referrals():
    assert referrals_per_1000(np.array([0.8]), np.array([0.05]), 0.05)[0] == pytest.approx(87.5)
