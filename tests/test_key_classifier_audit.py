"""Independent prediction, scaler and multiclass arithmetic verification checks."""

import copy

import numpy as np
import pytest

from ecg_experiment.key_classifier_audit import (
    manual_prediction,
    multiclass_check,
    validate_reference_draw,
    validate_scaler,
)
from ecg_experiment.key_classifier_models import (
    STATE_BITS,
    fit_package,
    joint_metrics,
    predict_package,
    resample_references,
)


@pytest.mark.parametrize("family", ["joint_logistic", "independent_logistic"])
def test_parameter_arithmetic_matches_model_predictions_without_query_fitting(family):
    rng = np.random.default_rng(57)
    states = np.tile(np.arange(16), 4)
    raw = rng.normal(size=(64, 8))
    package = fit_package(raw, STATE_BITS[states], family, 1.0)
    query = rng.normal(size=(9, 8))
    actual, _ = predict_package(package, query)
    manual, _ = manual_prediction(package, query)
    np.testing.assert_allclose(manual, actual, atol=1e-12, rtol=0)
    validate_scaler(package, raw, 1.0)
    with pytest.raises(AssertionError):
        validate_scaler(package, raw + 30, 1.0)


def test_independent_group_draw_reproduces_all_original_row_multiplicities():
    groups = np.array(["p1", "p1", "p2", "r1", "r2", "r3"])
    family = np.array(["A", "A", "A", "B", "B", "B"])
    rows = resample_references(groups, family, 20261206)
    validate_reference_draw(groups, family, rows, 20261206)
    malformed = np.array([0, 2, 3, 4, 5])
    with pytest.raises(ValueError, match="patient group was split"):
        validate_reference_draw(groups, family, malformed, 20261206)


def test_multiclass_audit_checks_counts_not_only_rounded_metrics():
    rng = np.random.default_rng(5)
    probability = rng.uniform(0.01, 1, size=(32, 16))
    probability /= probability.sum(axis=1, keepdims=True)
    flags = STATE_BITS[np.tile(np.arange(16), 2)]
    metrics = joint_metrics(flags, probability, majority=0)
    assert multiclass_check(flags, probability, metrics, flags) == 58
    changed = copy.deepcopy(metrics)
    changed["confusion"][0][0] += 1
    with pytest.raises(AssertionError):
        multiclass_check(flags, probability, changed, flags)
