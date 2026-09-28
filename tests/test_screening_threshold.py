"""Checks of the Experiment 027 calibration, threshold and screening helpers."""

import numpy as np
import pytest
from scipy.special import expit

from ecg_experiment.screening_threshold import (
    bootstrap_summary,
    calibrate,
    calibration_fit,
    fit_platt,
    head_logits,
    operating_point,
    patient_rates,
    screening_threshold,
)
from scripts.experiments.run_calibrated_threshold027 import CONTRASTS, reading


def simulated(records: int, slope: float, intercept: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 2, records)
    y = (rng.random(records) < expit(intercept + slope * logits)).astype(int)
    return logits, y


def test_head_logits_invert_the_sigmoid():
    values = np.array([-3.0, 0.0, 2.5])
    assert np.allclose(head_logits(expit(values)), values)


def test_head_logits_reject_certain_probabilities():
    with pytest.raises(ValueError, match="strictly"):
        head_logits(np.array([0.5, 1.0]))


def test_fit_platt_recovers_the_generating_mapping():
    logits, y = simulated(20000, 0.5, -1.0, 0)
    calibrator = fit_platt(logits, y)
    assert calibrator.coef_[0, 0] == pytest.approx(0.5, abs=0.03)
    assert calibrator.intercept_[0] == pytest.approx(-1.0, abs=0.05)


def test_fit_platt_rejects_a_reversed_score():
    logits, y = simulated(2000, 1.0, 0.0, 1)
    with pytest.raises(RuntimeError, match="Nonpositive"):
        fit_platt(-logits, y)


def test_calibrate_is_monotone():
    logits, y = simulated(500, 1.0, 0.0, 2)
    calibrator = fit_platt(logits, y)
    grid = np.linspace(-5, 5, 11)
    assert np.all(np.diff(calibrate(calibrator, grid)) > 0)


def test_screening_threshold_is_the_highest_that_refers_95_percent():
    y = np.array([1] * 20 + [0] * 5)
    probabilities = np.concatenate([np.linspace(0.05, 0.95, 20), np.linspace(0.01, 0.5, 5)])
    threshold = screening_threshold(y, probabilities)
    assert np.mean(probabilities[y == 1] >= threshold) >= 0.95
    higher = np.sort(probabilities[y == 1])[np.sort(probabilities[y == 1]) > threshold][0]
    assert np.mean(probabilities[y == 1] >= higher) < 0.95


def test_calibration_fit_is_ideal_for_calibrated_probabilities():
    logits, y = simulated(40000, 1.0, 0.0, 3)
    fit = calibration_fit(y, expit(logits))
    assert fit["intercept"] == pytest.approx(0.0, abs=0.03)
    assert fit["slope"] == pytest.approx(1.0, abs=0.03)


def test_calibration_fit_detects_overconfident_and_shifted_probabilities():
    logits, y = simulated(40000, 0.5, -1.0, 4)
    fit = calibration_fit(y, expit(logits))
    assert fit["slope"] == pytest.approx(0.5, abs=0.03)
    assert fit["intercept"] < -0.5


def test_operating_point_counts_and_scenarios():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    probabilities = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.3, 0.2, 0.1, 0.1, 0.05])
    point = operating_point(y, probabilities, 0.5)
    assert (point["tp"], point["fn"], point["fp"], point["tn"]) == (3, 1, 1, 5)
    assert point["sensitivity"] == 0.75
    assert point["specificity"] == pytest.approx(5 / 6)
    assert point["ppv"] == 0.75
    assert point["npv"] == pytest.approx(5 / 6)
    assert point["false_referrals_per_1000_negatives"] == pytest.approx(1000 / 6)
    scenario = point["assumed_prevalence"]["0.01"]
    assert scenario["expected_fp_per_1000"] == pytest.approx(990 / 6)
    assert scenario["ppv"] == pytest.approx(7.5 / (7.5 + 165))


def test_patient_rates_are_reproducible_and_bounded():
    rng = np.random.default_rng(5)
    patients = np.repeat(np.arange(100), 2).astype(str)
    y = rng.integers(0, 2, 200)
    referred = {"a": rng.random(200) < 0.7, "b": rng.random(200) < 0.5}
    first = patient_rates(patients, y, referred, 50, 27027)
    second = patient_rates(patients, y, referred, 50, 27027)
    assert np.array_equal(first["specificity"]["a"], second["specificity"]["a"])
    assert len(first["sensitivity"]["b"]) == 50
    assert np.all((first["sensitivity"]["a"] >= 0) & (first["sensitivity"]["a"] <= 1))


def test_patient_rates_keep_a_patients_ecgs_together():
    patients = np.array(["p", "p", "q", "q"])
    y = np.array([1, 0, 1, 0])
    referred = {"a": np.array([True, True, False, False])}
    rates = patient_rates(patients, y, referred, 200, 0)
    # Each patient's ECGs are all referred or all passed, so sensitivity = 1 - specificity in every draw.
    assert np.allclose(rates["sensitivity"]["a"], 1 - rates["specificity"]["a"])


def test_bootstrap_summary_of_identical_heads_has_zero_difference():
    rng = np.random.default_rng(6)
    patients = np.arange(300).astype(str)
    y = rng.integers(0, 2, 300)
    flags = rng.random(300) < 0.6
    summary = bootstrap_summary(patients, y, {"first": flags, "second": flags.copy()},
                                {"first_minus_second": ("first", "second")}, 100, 1)
    contrast = summary["specificity_contrasts"]["first_minus_second"]
    assert contrast == {"difference": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    assert summary["heads"]["first"]["sensitivity"]["value"] == pytest.approx(np.mean(flags[y == 1]))
    assert summary["valid_draws"] == 100


def rates(sensitivity: tuple[float, float, float], difference: tuple[float, float]) -> dict[str, object]:
    value, low, high = sensitivity
    heads = {head: {"sensitivity": {"value": value, "ci_low": low, "ci_high": high}}
             for head in ("cpc_standard", "jepa_standard", "xecg_standard")}
    contrasts = {name: {"difference": 0.0, "ci_low": difference[0], "ci_high": difference[1]}
                 for name in CONTRASTS}
    return {"heads": heads, "specificity_contrasts": contrasts}


def test_reading_applies_the_prespecified_rules():
    bootstrap = {"sph": rates((0.93, 0.92, 0.96), (0.01, 0.03)),
                 "development_full": rates((0.96, 0.95, 0.97), (-0.02, 0.01))}
    result = reading(bootstrap)
    assert result["threshold_transfers_to_sph"]["cpc_standard"] == {
        "transfers": True, "point_at_or_above_target": False}
    verdicts = result["specificity_contrasts"]
    assert verdicts["sph"]["xecg_minus_cpc"]["better_for_screening"] == "xecg_standard"
    assert verdicts["development_full"]["xecg_minus_jepa"]["better_for_screening"] is None


def test_reading_flags_a_threshold_that_does_not_transfer():
    bootstrap = {"sph": rates((0.90, 0.89, 0.91), (-0.03, -0.01)),
                 "development_full": rates((0.96, 0.95, 0.97), (-0.03, -0.01))}
    result = reading(bootstrap)
    assert not result["threshold_transfers_to_sph"]["jepa_standard"]["transfers"]
    assert result["specificity_contrasts"]["sph"]["jepa_minus_cpc"]["better_for_screening"] == "cpc_standard"
