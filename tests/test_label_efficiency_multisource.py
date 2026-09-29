"""Checks of the Experiment 025b pooled draws, weighted AUROC and decision rule."""

import numpy as np
from sklearn.metrics import roc_auc_score

from ecg_experiment.label_efficiency import paired_summary
from ecg_experiment.label_efficiency_multisource import (
    bootstrap_mean_auroc,
    challenge_units,
    decision,
    interval,
    plus_positions,
    resample_counts,
    sort_scores,
    verdict,
    weighted_auroc,
)


def test_challenge_units_name_each_record():
    units = challenge_units(np.array(["ningbo", "georgia"]), np.array(["JS1", "E1"]))
    assert units.tolist() == ["challenge:ningbo:JS1", "challenge:georgia:E1"]


def test_plus_positions_add_every_challenge_row():
    assert plus_positions(np.array([3, 1]), 5, 8).tolist() == [1, 3, 5, 6, 7]


def test_weighted_auroc_equals_roc_auc_on_repeated_rows():
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.4).astype(int)
    scores = np.round(rng.normal(size=300) + y, 1)
    counts = rng.integers(0, 4, size=300)
    rows = np.repeat(np.arange(300), counts)
    expected = roc_auc_score(y[rows], scores[rows])
    assert abs(weighted_auroc(sort_scores(y, scores), counts) - expected) < 1e-12


def test_bootstrap_mean_auroc_averages_draws_on_each_resample():
    rng = np.random.default_rng(1)
    y = (rng.random(200) < 0.5).astype(int)
    scores = rng.normal(size=(3, 200)) + y
    units = np.arange(200) // 2
    counts, invalid = resample_counts(units, y, 5, 7)
    assert invalid == 0
    assert counts.shape == (5, 200)
    assert (counts.sum(axis=1) == 200).all()
    values = bootstrap_mean_auroc(y, scores, counts)
    rows = np.repeat(np.arange(200), counts[2])
    expected = np.mean([roc_auc_score(y[rows], vector[rows]) for vector in scores])
    assert abs(values[2] - expected) < 1e-12


def test_verdict_needs_the_draw_rule_and_the_interval():
    above = paired_summary(np.full(20, 0.9), np.full(20, 0.8))
    below = paired_summary(np.full(20, 0.8), np.full(20, 0.9))
    positive = {"difference": 0.1, "ci_low": 0.01, "ci_high": 0.2}
    spanning = {"difference": 0.1, "ci_low": -0.01, "ci_high": 0.2}
    negative = {"difference": -0.1, "ci_low": -0.2, "ci_high": -0.01}
    assert verdict("pooled", "ptbxl", above, positive) == "pooled better"
    assert verdict("pooled", "ptbxl", above, spanning) == "not distinguished"
    assert verdict("pooled", "ptbxl", below, negative) == "ptbxl better"
    assert verdict("pooled", "ptbxl", below, positive) == "not distinguished"


def test_decision_follows_the_protocol():
    assert decision(["pooled better", "pooled better"]) == "adopt_pooled"
    assert decision(["ptbxl better", "ptbxl better"]) == "pooling_hurts"
    assert decision(["pooled better", "not distinguished"]) == "mixed"


def test_interval_is_paired():
    first = np.array([0.1, 0.2, 0.3, 0.4])
    bounds = interval(0.05, first, first - 0.05)
    assert abs(bounds["ci_low"] - 0.05) < 1e-12
    assert abs(bounds["ci_high"] - 0.05) < 1e-12
