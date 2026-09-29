"""Checks of the Experiment 027b weighted calibration and bootstrap helpers."""

import numpy as np
import pytest
from scipy.special import expit

from ecg_experiment.evaluation import select_threshold
from ecg_experiment.multisource_calibration import (
    bootstrap_rates,
    family_weights,
    fit_arm,
    fit_weighted_platt,
    summarize,
    verdict,
    weighted_rates,
    weighted_threshold,
)
from ecg_experiment.screening_threshold import calibrate, fit_platt, operating_point, patient_rates


def simulated(records: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 2, records)
    y = (rng.random(records) < expit(0.8 * logits - 0.3)).astype(int)
    return logits, y


def test_family_weights_equalize_families_and_average_one():
    families = np.array(["a"] * 6 + ["b"] * 2 + ["c"] * 4)
    weights = family_weights(families)
    assert weights.mean() == pytest.approx(1.0)
    totals = {name: weights[families == name].sum() for name in "abc"}
    assert totals["a"] == pytest.approx(totals["b"]) == pytest.approx(totals["c"])


def test_weighted_threshold_with_unit_weights_matches_select_threshold():
    rng = np.random.default_rng(1)
    for records in (20, 57, 348, 1001):
        y = np.r_[np.ones(records), np.zeros(30)].astype(int)
        probabilities = np.round(rng.random(len(y)), 2)
        assert weighted_threshold(y, probabilities, np.ones(len(y))) == select_threshold(y, probabilities)


def test_weighted_threshold_follows_the_weights():
    y = np.array([1, 1, 1, 1, 0])
    probabilities = np.array([0.1, 0.4, 0.6, 0.9, 0.5])
    assert weighted_threshold(y, probabilities, np.array([1.0, 1, 1, 1, 1]), target=0.75) == 0.4
    assert weighted_threshold(y, probabilities, np.array([0.0, 1, 1, 2, 1]), target=0.75) == 0.6
    assert weighted_threshold(y, probabilities, np.array([10.0, 1, 1, 1, 1]), target=0.75) == 0.1


def test_weighted_platt_with_unit_weights_matches_fit_platt():
    logits, y = simulated(3000, 2)
    weighted = fit_weighted_platt(logits, y, np.ones(len(y)))
    plain = fit_platt(logits, y)
    assert weighted.coef_[0, 0] == pytest.approx(plain.coef_[0, 0], abs=1e-6)
    assert weighted.intercept_[0] == pytest.approx(plain.intercept_[0], abs=1e-6)


def test_weighted_platt_equals_repeating_records():
    logits, y = simulated(500, 3)
    counts = np.random.default_rng(4).integers(1, 4, len(y))
    weighted = fit_weighted_platt(logits, y, counts.astype(float))
    repeated = fit_platt(np.repeat(logits, counts), np.repeat(y, counts))
    assert weighted.coef_[0, 0] == pytest.approx(repeated.coef_[0, 0], abs=1e-4)


def test_fit_arm_unweighted_is_the_027_rule():
    logits, y = simulated(800, 5)
    calibrator, threshold = fit_arm(logits, y)
    reference = fit_platt(logits, y)
    assert calibrator.coef_[0, 0] == reference.coef_[0, 0]
    assert threshold == select_threshold(y, calibrate(reference, logits))


def test_weighted_rates_with_unit_weights_match_operating_point():
    logits, y = simulated(2000, 6)
    first = expit(logits)
    second = expit(0.5 * logits + 0.4)
    probabilities = np.column_stack([first, second])
    thresholds = np.array([0.3, 0.55])
    rates = weighted_rates(np.ones(len(y)), y, probabilities, thresholds)
    for column in range(2):
        point = operating_point(y, probabilities[:, column], thresholds[column])
        for name in ("sensitivity", "specificity", "brier", "ece_10_bins"):
            assert rates[name][column] == pytest.approx(point[name], abs=1e-12)
        assert rates["deviation"][column] == pytest.approx(abs(point["sensitivity"] - 0.95), abs=1e-12)


def test_bootstrap_rates_use_the_same_draws_as_patient_rates():
    logits, y = simulated(600, 7)
    patients = np.repeat(np.arange(300), 2)
    probabilities = expit(logits)[:, None]
    threshold = np.array([0.4])
    drawn = bootstrap_rates(patients, y, probabilities, threshold, draws=50, seed=11)
    reference = patient_rates(patients, y, {"head": probabilities[:, 0] >= 0.4}, draws=50, seed=11)
    assert np.allclose(drawn["sensitivity"][:, 0], reference["sensitivity"]["head"])
    assert np.allclose(drawn["specificity"][:, 0], reference["specificity"]["head"])


def test_summarize_gives_differences_from_the_reference_arm():
    columns = [("ref", "h"), ("new", "h")]
    observed = {name: np.array([0.1, 0.3]) for name in
                ("sensitivity", "specificity", "brier", "ece_10_bins", "deviation")}
    drawn = {name: np.array([[0.1, 0.2], [0.1, 0.4], [0.2, 0.3]]) for name in observed}
    summary = summarize(columns, observed, drawn, "ref")
    assert summary["valid_draws"] == 3
    assert summary["differences"]["new"]["h"]["deviation"]["difference"] == pytest.approx(0.2)
    assert "ref" not in summary["differences"]
    assert summary["rates"]["ref"]["h"]["brier"]["value"] == pytest.approx(0.1)


def test_verdict_reads_the_interval():
    assert verdict({"ci_low": -0.03, "ci_high": -0.01}) == "better"
    assert verdict({"ci_low": 0.01, "ci_high": 0.03}) == "worse"
    assert verdict({"ci_low": -0.01, "ci_high": 0.02}) == "not_distinguished"

