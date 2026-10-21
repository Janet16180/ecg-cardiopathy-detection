"""Scientific invariants for equal-patient regional covariance fitting."""

import numpy as np
import pytest

from ecg_experiment.clustering.regional_projection import (
    VIEWS,
    decision,
    patient_variance,
    weighted_projection,
)


def _cache(repetitions=1):
    first = np.array([[-3.0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [3, 0, 0, 0, 0]])
    second = np.array([[0, -2.0, 1, 0, 0], [0, 0, -1, 1, 0], [0, 2, 0, -1, 0]])
    anchor = np.full((3, 5), 1e8)
    regions = np.r_[first, np.tile(second, (repetitions, 1)), anchor]
    return {
        "region_vectors": regions,
        "region_offsets": np.array([0, 3, 3 + 3 * repetitions, 6 + 3 * repetitions]),
    }


def test_covariance_matches_explicit_patient_weighted_formula():
    cache = _cache(4)
    result = weighted_projection(cache, np.array([0, 1]))
    values = cache["region_vectors"][:15]
    weights = np.r_[np.full(3, 1 / 6), np.full(12, 1 / 24)]
    center = weights @ values
    centered = values - center
    covariance = (centered * weights[:, None]).T @ centered
    np.testing.assert_allclose(result["center"], center, atol=1e-15)
    np.testing.assert_allclose(result["eigenvalues"], np.linalg.eigvalsh(covariance)[::-1], atol=1e-14)
    assert result["total_region_variance"] == pytest.approx(np.trace(covariance))
    assert result["weighted_region_variance_retained"] == pytest.approx(
        result["eigenvalues"][:3].sum() / np.trace(covariance)
    )
    ordinary_center = values.mean(axis=0)
    ordinary_covariance = (values - ordinary_center).T @ (values - ordinary_center) / len(values)
    assert not np.allclose(covariance, ordinary_covariance)


def test_duplicating_one_patients_regions_preserves_fit():
    first = weighted_projection(_cache(1), np.array([0, 1]))
    second = weighted_projection(_cache(7), np.array([0, 1]))
    for key in ("center", "components", "eigenvalues"):
        np.testing.assert_allclose(first[key], second[key], atol=1e-13)


def test_anchor_changes_do_not_adapt_projection_and_signs_are_fixed():
    cache = _cache(2)
    first = weighted_projection(cache, np.array([0, 1]))
    cache["region_vectors"][-3:] *= -100
    second = weighted_projection(cache, np.array([0, 1]))
    np.testing.assert_array_equal(first["center"], second["center"])
    np.testing.assert_array_equal(first["components"], second["components"])
    pivots = np.argmax(np.abs(first["components"]), axis=1)
    assert np.all(first["components"][np.arange(3), pivots] > 0)
    np.testing.assert_allclose(first["components"] @ first["components"].T, np.eye(3), atol=1e-14)


def test_invalid_offsets_and_duplicate_fitting_patients_raise():
    with pytest.raises(ValueError, match="distinct"):
        weighted_projection(_cache(), np.array([0, 0]))
    cache = _cache()
    cache["region_offsets"][1] = 2
    with pytest.raises(ValueError, match="boundaries"):
        weighted_projection(cache, np.array([0, 1]))


def test_patient_retention_excludes_anchors_and_centers_fitting_population():
    values = np.array([[-2.0, 0], [2, 0], [0, 1e9]])
    assert patient_variance(values, np.array([0, 1]), np.array([[1.0, 0]])) == 1
    assert patient_variance(values, np.array([0, 1]), np.array([[0.0, 1]])) == 0


def test_secondary_agreement_cannot_rescue_ineligible_primary():
    labels = np.array([0, 0, 1, 1])
    selected = {view: {"key": view, "eligible": True} for view in VIEWS}
    assignments = {view: {"anchor": labels} for view in VIEWS}
    selected[VIEWS[0]]["eligible"] = False
    assert not decision(selected, assignments)["primary_regional_concordance_passed"]
    selected[VIEWS[0]]["eligible"] = True
    assignments[VIEWS[0]]["anchor"] = np.array([0, 1, 0, 1])
    assert not decision(selected, assignments)["primary_regional_concordance_passed"]
