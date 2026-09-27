"""Checks of the Experiment 020 baseline and bootstrap helpers."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from ecg_experiment.full_development import demographics, fit_logistic, patient_bootstrap, predict


def test_demographics_fill_missing_age_with_the_training_median():
    frame = pd.DataFrame({"age": [50.0, np.nan], "male": [1.0, 0.0]})
    assert demographics(frame, median_age=60.0).tolist() == [[5.0, 1.0, 0.0], [6.0, 0.0, 1.0]]


def test_logistic_head_learns_a_separable_signal():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 3))
    y = (x[:, 0] > 0).astype(int)
    head = fit_logistic(x, y)
    assert roc_auc_score(y, predict(head, x)) > 0.99


def test_bootstrap_of_identical_scores_is_zero():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    scores = rng.random(200)
    patients = np.repeat(np.arange(100), 2)
    result = patient_bootstrap(patients, y, scores, scores, draws=50)
    assert result["difference"] == 0
    assert result["ci_low"] == result["ci_high"] == 0


def test_bootstrap_detects_a_better_model():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 400)
    good = y + rng.normal(0, 0.5, 400)
    poor = y + rng.normal(0, 3, 400)
    result = patient_bootstrap(np.arange(400), y, good, poor, draws=200)
    assert result["difference"] > 0
    assert result["ci_low"] > 0


def test_bootstrap_keeps_a_patients_ecgs_together():
    y = np.array([0, 0, 1, 1])
    patients = np.array(["a", "a", "b", "b"])
    result = patient_bootstrap(patients, y, np.array([0.1, 0.2, 0.8, 0.9]), np.array([0.9, 0.8, 0.2, 0.1]),
                               draws=100)
    assert result["invalid_draws"] > 0
    assert result["difference"] == pytest.approx(1.0)
