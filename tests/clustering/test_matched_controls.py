"""Synthetic checks of fitting-only matched geometry and the prospective decision."""

from __future__ import annotations

import time

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from ecg_experiment.clustering.matched_controls import (
    VIEWS,
    center_scale,
    decision,
    matched_transforms,
    orient_components,
    pool_regions,
    region_variance,
)
from ecg_experiment.clustering.reference_geometry import fit_model


def test_component_sign_is_deterministic_before_max_pooling():
    components = np.array([[-2.0, 2.0, 0], [0, -1.0, -3.0]])
    first = orient_components(components)
    second = orient_components(-components)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, [[2, -2, 0], [0, 1, 3]])
    values = np.array([[1.0, 4], [2, -1], [-3, 2]])
    pooled = pool_regions(values, np.array([0, 3]))
    np.testing.assert_allclose(pooled[0], np.r_[values.mean(0), [2, 4], values.std(0)])
    with pytest.raises(ValueError, match="boundaries"):
        pool_regions(values, np.array([0, 2]))


def test_center_and_rms_do_not_adapt_to_validation_anchors():
    rng = np.random.default_rng(2)
    values = rng.normal(size=(30, 9))
    fitting = np.arange(20)
    changed = values.copy()
    changed[20:] += 1e6
    first_state, first = center_scale(values, fitting, True)
    second_state, second = center_scale(changed, fitting, True)
    assert first_state == second_state
    np.testing.assert_allclose(first[fitting], second[fitting])
    np.testing.assert_allclose(first[fitting].mean(0), 0, atol=1e-15)
    np.testing.assert_allclose(np.mean(first[fitting] ** 2), 1, atol=1e-15)
    with pytest.raises(ValueError, match="Degenerate"):
        center_scale(np.ones((30, 3)), fitting, True)


def test_region_variance_weights_patients_equally_and_uses_fitting_rows_only():
    cache = {
        "region_vectors": np.array([[0.0, 0]] * 3 + [[2.0, 2]] * 6 + [[1e7, 0]] * 3),
        "region_offsets": np.array([0, 3, 9, 12]),
    }
    actual = region_variance(cache, np.array([0, 1]), np.array([[1.0, 0]]))
    assert actual == 0.5
    duplicated = {
        "region_vectors": np.array([[0.0, 0]] * 3 + [[2.0, 2]] * 12),
        "region_offsets": np.array([0, 3, 15]),
    }
    assert region_variance(duplicated, np.array([0, 1]), np.array([[1.0, 0]])) == actual


def test_complete_matched_transform_refits_without_anchor_leakage():
    rng = np.random.default_rng(14)
    qrs = rng.normal(size=(35, 12))
    regions = rng.normal(size=(105, 12))
    fitting = np.arange(25)
    data = {
        "cache": {"qrs": qrs, "region_vectors": regions, "region_offsets": np.arange(0, 106, 3)},
        "samples": [fitting],
        "ntt_record3": rng.normal(size=(35, 3)),
        "ntt_region9": rng.normal(size=(35, 9)),
    }
    first = matched_transforms(data, time.perf_counter() + 20)
    qrs[25:] += 1e5
    regions[75:] += 1e5
    data["ntt_record3"][25:] += 1e5
    data["ntt_region9"][25:] += 1e5
    second = matched_transforms(data, time.perf_counter() + 20)
    for view in VIEWS:
        np.testing.assert_allclose(first[view][0]["coordinates"][:25], second[view][0]["coordinates"][:25])
        first_state = {k: v for k, v in first[view][0].items() if k != "coordinates"}
        second_state = {k: v for k, v in second[view][0].items() if k != "coordinates"}
        assert first_state == second_state


def test_secondary_agreement_cannot_rescue_primary_ineligibility():
    labels = np.array([0, 0, 1, 1])
    selected = {view: {"key": view, "eligible": True} for view in VIEWS}
    assignments = {view: {"anchor": labels} for view in VIEWS}
    selected["native_region9_rms"]["eligible"] = False
    result = decision(selected, assignments)
    assert result["primary_anchor_ari"] == 1
    assert not result["primary_matched_geometric_concordance_passed"]
    selected["native_region9_rms"]["eligible"] = True
    assignments["native_region9_rms"]["anchor"] = np.array([0, 1, 0, 1])
    assert not decision(selected, assignments)["primary_matched_geometric_concordance_passed"]


def test_global_scalar_keeps_fixed_kmeans_partition():
    rng = np.random.default_rng(22)
    values = np.r_[rng.normal(-3, 0.2, (30, 3)), rng.normal(3, 0.2, (30, 3))]
    model_raw, _ = fit_model(values, "kmeans", "none", 2, 20261018)
    model_scaled, _ = fit_model(values / 3.7, "kmeans", "none", 2, 20261018)
    assert adjusted_rand_score(model_raw.predict(values), model_scaled.predict(values / 3.7)) == 1
