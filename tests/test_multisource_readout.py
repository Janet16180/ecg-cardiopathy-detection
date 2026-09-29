"""Checks of the Experiment 022b arm membership and weighted readout."""

import numpy as np
import pytest
from scipy.special import expit

from ecg_experiment.external_readout import fit_logistic_c
from ecg_experiment.full_development import fit_logistic, predict
from ecg_experiment.multisource_calibration import family_weights
from ecg_experiment.multisource_readout import (
    ARMS,
    arm_members,
    fit_readout,
    fit_weighted_logistic,
    probability_quality,
)


def simulated(records: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(records, 6)) * np.array([1.0, 2.0, 0.5, 3.0, 1.0, 0.1]) + 0.3
    y = (rng.random(records) < expit(x[:, 0] - 0.5 * x[:, 1] + 0.2)).astype(int)
    return x, y


def test_arm_members_follow_the_protocol():
    families = np.array(["ptbxl", "ptbxl", "chapman_ningbo", "chapman_ningbo", "georgia", "cpsc", "cpsc"])
    sources = np.array(["ptbxl", "ptbxl", "ningbo", "chapman_shaoxing", "georgia", "cpsc_2018",
                        "cpsc_2018_extra"])
    members = arm_members(families, sources)
    assert set(members) == set(ARMS)
    rows = {arm: np.flatnonzero(selected).tolist() for arm, selected in members.items()}
    assert rows["ptbxl"] == [0, 1]
    assert rows["pooled"] == rows["balanced"] == list(range(7))
    assert rows["ptbxl_chapman_ningbo"] == [0, 1, 2, 3]
    assert rows["ptbxl_ningbo"] == [0, 1, 2]
    assert rows["loso_chapman_ningbo"] == [0, 1, 4, 5, 6]
    assert rows["loso_georgia"] == [0, 1, 2, 3, 5, 6]
    assert rows["loso_cpsc"] == [0, 1, 2, 3, 4]


def test_unweighted_readout_is_the_022_head():
    x, y = simulated(400, 3)
    reference = predict(fit_logistic(x, y), x)
    assert np.array_equal(predict(fit_readout(x, y), x), reference)
    assert np.array_equal(predict(fit_logistic_c(x, y, 0.01), x), reference)


def test_unit_weights_reproduce_the_unweighted_head():
    x, y = simulated(500, 4)
    weighted = predict(fit_weighted_logistic(x, y, np.ones(len(y))), x)
    assert np.abs(weighted - predict(fit_logistic(x, y), x)).max() < 1e-8


def test_integer_weights_equal_repeated_rows():
    x, y = simulated(300, 5)
    counts = np.random.default_rng(6).integers(1, 4, len(y))
    weighted = fit_weighted_logistic(x, y, counts.astype(float))
    repeated = fit_logistic(np.repeat(x, counts, axis=0), np.repeat(y, counts))
    assert np.abs(predict(weighted, x) - predict(repeated, x)).max() < 1e-6


def test_family_weights_move_the_balanced_head():
    x, y = simulated(600, 7)
    families = np.array(["ptbxl"] * 500 + ["georgia"] * 100)
    weights = family_weights(families)
    assert weights[families == "georgia"].sum() == pytest.approx(weights[families == "ptbxl"].sum())
    assert not np.allclose(predict(fit_readout(x, y, weights), x), predict(fit_readout(x, y), x))


def test_probability_quality_of_perfectly_calibrated_scores():
    rng = np.random.default_rng(8)
    probabilities = rng.uniform(0.05, 0.95, 20000)
    y = (rng.random(len(probabilities)) < probabilities).astype(int)
    quality = probability_quality(y, probabilities)
    assert quality["brier"] == pytest.approx(np.mean((probabilities - y) ** 2))
    assert abs(quality["intercept"]) < 0.05
    assert quality["slope"] == pytest.approx(1.0, abs=0.05)
