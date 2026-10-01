"""Scientific regression checks for the factorial analysis and collapse gate."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from ecg_experiment.encoder_context_analysis039 import (
    comparisons,
    factorial_intervals,
    severe_failures,
)
from ecg_experiment.encoder_context_diagnostics039 import feature_health


def _arrays():
    rng = np.random.default_rng(42)
    y = np.tile([0, 1], 12)
    arrays = {"targets": y, "patient_ids": np.repeat(np.arange(12), 2)}
    names = {name for first, second, _ in comparisons().values() for name in (first, second)}
    for name in sorted(names):
        arrays[name] = rng.random((3, len(y))) + y * 0.3
    return arrays


def test_joint_bootstrap_averages_seed_differences_and_is_reproducible():
    arrays = _arrays()
    result = factorial_intervals(arrays, draws=25)
    assert result == factorial_intervals(arrays, draws=25)
    assert result["primary_family_size"] == 8
    assert result["skipped_draws"] == 0
    name = "25k_limited_patch_minus_cnn_gru"
    delta = result["contrasts"][name]
    first, second, _ = comparisons()[name]
    expected = [roc_auc_score(arrays["targets"], a) - roc_auc_score(arrays["targets"], b)
                for a, b in zip(arrays[first], arrays[second], strict=True)]
    np.testing.assert_array_equal(delta["seed_differences"], expected)
    assert delta["difference"] == np.mean(expected)
    ensemble_difference = (roc_auc_score(arrays["targets"], arrays[first].mean(axis=0))
                           - roc_auc_score(arrays["targets"], arrays[second].mean(axis=0)))
    assert delta["difference"] != ensemble_difference
    assert delta["simultaneous_high"] - delta["difference"] == pytest.approx(
        result["simultaneous_radius"])
    for comparison in result["contrasts"].values():
        assert comparison["promising"] == bool(comparison["primary"] and
                                               comparison["difference"] >= 0.005 and
                                               comparison["simultaneous_low"] > 0)


def test_failure_rule_compares_corresponding_seed_and_context():
    arrays = _arrays()
    for name in arrays:
        if name not in ("targets", "patient_ids"):
            arrays[name] = np.tile(arrays["targets"], (3, 1)).astype(float)
    arrays["25_patch_limited_xlstm"][1] = 1 - arrays["targets"]
    failures = severe_failures(arrays)
    assert len(failures) == 1
    assert failures[0]["encoder"] == "patch"
    assert failures[0]["context"] == "xlstm"
    assert failures[0]["seed"] == 39043
    assert failures[0]["difference"] == -1


def test_feature_collapse_is_not_confused_with_small_but_valid_variance():
    features = np.random.default_rng(7).normal(size=(64, 512))
    assert feature_health(features)["collapsed"] is False
    features[:, :461] = 1
    assert feature_health(features)["collapsed"] is True
    features[0, 0] = np.nan
    assert feature_health(features)["finite"] is False


def test_single_class_bootstrap_cannot_produce_a_study_decision():
    arrays = _arrays()
    arrays["targets"][:] = 0
    with pytest.raises(ValueError, match="Both target classes"):
        factorial_intervals(arrays, draws=2)
