"""Synthetic scientific controls for fixed bank covariance scaling."""

import time

import numpy as np
import pytest

from ecg_experiment.clustering.bank_scale import (
    averaging_decision,
    equivalence,
    fit_condition,
    group_agreement,
)


@pytest.mark.parametrize(("covariance", "clusters"), [("tied", 2), ("diag", 2), ("full", 3)])
def test_transformed_covariance_floor_preserves_small_scale_geometry(covariance, clusters):
    rng = np.random.default_rng(7)
    values = np.concatenate([rng.normal(i * 0.035, 0.006, (70, 3)) for i in range(clusters)])
    fitting = np.arange(0, len(values), 2)
    anchors = np.arange(1, len(values), 2)
    raw = fit_condition(values, fitting, anchors, covariance, clusters, "raw", time.perf_counter() + 30)
    scaled = fit_condition(
        values, fitting, anchors, covariance, clusters, "equivalent", time.perf_counter() + 30
    )
    result = equivalence(raw, scaled, fitting, anchors, covariance, clusters)
    assert result["passed"], result
    divisor = scaled[0]["divisor"]
    assert np.isclose(scaled[0]["reg_covar"], 1e-3 / divisor**2)


def test_anchor_extremes_do_not_fit_the_coordinate_scalar():
    rng = np.random.default_rng(8)
    values = rng.normal(size=(90, 3))
    fitting, anchors = np.arange(60), np.arange(60, 90)
    changed = values.copy()
    changed[anchors] *= 100
    first = fit_condition(values, fitting, anchors, "diag", 2, "equivalent", time.perf_counter() + 30)
    second = fit_condition(changed, fitting, anchors, "diag", 2, "equivalent", time.perf_counter() + 30)
    assert first[0]["divisor"] == second[0]["divisor"]
    np.testing.assert_array_equal(first[2]["means"], second[2]["means"])
    np.testing.assert_array_equal(first[2]["covariances"], second[2]["covariances"])


def test_per_group_persistence_exposes_minority_loss():
    reference = np.array([0] * 12 + [1] * 4)
    fitting, anchors = np.array([0, 1, 12, 13]), np.array([2, 3, 4, 14, 15])
    labels = np.stack([reference.copy(), reference.copy()])
    labels[:, 14:] = 0
    result = group_agreement(labels, reference, fitting, anchors, 2)
    assert result["0"]["agreement"]["median"] == 1
    assert result["1"]["anchors"] == 2
    assert result["1"]["agreement"]["median"] == 0


def test_empty_finite_reference_component_is_reported_without_dropping_fits():
    reference = np.zeros(8, dtype=int)
    result = group_agreement(np.stack([reference, reference]), reference, np.arange(4), np.arange(4, 8), 2)
    assert result["1"]["anchors"] == 0
    assert result["1"]["agreement"] is None
    assert result["1"]["fraction_agreement_at_least_0.8"] is None


def test_averaging_admission_and_signed_effect_are_both_required():
    single = {"admissible_banks": 45, "pairwise_anchor_ari": {"p10": 0.4}}
    ensemble = {"admissible_banks": 9, "pairwise_anchor_ari": {"p10": 0.55}}
    assert averaging_decision(single, ensemble)["meaningful_conditional_averaging_benefit"]
    ensemble["admissible_banks"] = 8
    assert not averaging_decision(single, ensemble)["meaningful_conditional_averaging_benefit"]
    ensemble["pairwise_anchor_ari"]["p10"] = 0.3
    assert averaging_decision(single, ensemble)["p10_ari_difference"] < 0


def test_equivalence_detects_incorrect_covariance_backtransform():
    rng = np.random.default_rng(9)
    values = rng.normal(size=(80, 3)) * 0.01
    fitting, anchors = np.arange(60), np.arange(60, 80)
    raw = fit_condition(values, fitting, anchors, "tied", 2, "raw", time.perf_counter() + 30)
    corrupted = (raw[0], raw[1], {**raw[2], "covariances": raw[2]["covariances"] * 2})
    result = equivalence(raw, corrupted, fitting, anchors, "tied", 2)
    assert not result["passed"]
    assert not result["conditions"]["covariances_within_1e-6"]
