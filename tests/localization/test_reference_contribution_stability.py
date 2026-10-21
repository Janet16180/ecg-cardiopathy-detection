"""Synthetic fixed-contrast, occupancy, empty-set and patient-coverage controls."""

import numpy as np
import pytest

from ecg_experiment.localization.contextual_margin import rank_correlation
from ecg_experiment.localization.reference_contribution_stability import (
    fixed_contributions,
    nested_error,
    normalized_ensemble_prototypes,
    pair_metrics,
    primary_decision,
    rank_pairs,
    top_sets,
)


def geometry_fixture():
    tokens = np.array([[1.0, 0.1, 0], [0.5, 0.4, 0.1], [0.2, 0.3, 0.8]])
    weights = np.array([[0.5, 0.5, 0], [0, 0.5, 0.5]])
    means = weights @ tokens
    norms = np.linalg.norm(means, axis=1)
    beats = means / norms[:, None]
    denominator = np.linalg.norm(beats.sum(axis=0))
    return (
        {
            "beats": beats,
            "denominator": denominator,
            "patient": beats.sum(axis=0) / denominator,
            "tokens": tokens,
            "token_coefficients": (weights / norms[:, None]).sum(axis=0),
        },
        weights,
        norms,
    )


def test_fixed_negative_contrast_and_unique_overlapping_token_sums():
    geometry, weights, norms = geometry_fixture()
    bank = np.eye(3)
    prototypes = np.stack([bank, bank[[1, 0, 2]]])
    result = fixed_contributions(geometry, prototypes, (0, 1))
    assert result["margins"][0] > 0
    assert result["margins"][1] < 0
    assert np.argmax(result["scores"][1]) != 0
    np.testing.assert_allclose(result["beat"].sum(axis=1), result["margins"], atol=1e-14)
    np.testing.assert_allclose(result["token"].sum(axis=1), result["margins"], atol=1e-14)
    delta = bank[0] - bank[1]
    explicit = weights * (geometry["tokens"] @ delta)[None, :] / (norms[:, None] * geometry["denominator"])
    np.testing.assert_allclose(explicit.sum(axis=0), result["token"][0], atol=1e-14)
    assert np.count_nonzero(weights[:, 1]) == 2


def test_positive_quarter_retains_index_ties_and_empty_set_denominators():
    values = np.array([[1.0, 1, 1, 1], [-1.0, -2, -3, -4], [-2.0, -3, -4, -5]])
    np.testing.assert_array_equal(top_sets(values, True)[0], [True, False, False, False])
    summaries, arrays = pair_metrics(values)
    assert arrays["positive_quarter_jaccard"][0] == 0
    assert arrays["positive_quarter_jaccard"][1] == 0
    assert np.isnan(arrays["positive_quarter_jaccard"][2])
    assert summaries["positive_quarter_jaccard"]["count"] == 3
    assert summaries["positive_quarter_jaccard"]["defined"] == 2


def test_vectorized_spearman_matches_average_rank_helper_with_constants():
    values = np.array([[1.0, 1, 3, 2], [2.0, 1, 1, 3], [1.0, 1, 1, 1]])
    pairs = np.triu_indices(3, 1)
    actual = rank_pairs(values, pairs)
    assert actual[0] == pytest.approx(rank_correlation(values[0], values[1]), abs=1e-14)
    assert np.isnan(actual[1:]).all()


def patient_rows():
    return [
        {
            "single": {"pairs": {"positive_quarter_jaccard": {"mean": 0.4, "defined": 1225}}},
            "ensemble": {"pairs": {"positive_quarter_jaccard": {"mean": 0.55, "defined": 45}}},
        }
        for _ in range(60)
    ]


def test_primary_weights_each_patient_and_requires_coverage_for_each_family():
    rows = patient_rows()
    assert primary_decision(rows)["conditional_highlight_stability_benefit"]
    rows[0]["single"]["pairs"]["positive_quarter_jaccard"]["defined"] = 1163
    assert not primary_decision(rows)["conditional_highlight_stability_benefit"]
    rows[0]["single"]["pairs"]["positive_quarter_jaccard"]["defined"] = 1164
    rows[0]["ensemble"]["pairs"]["positive_quarter_jaccard"]["defined"] = 42
    assert not primary_decision(rows)["conditional_highlight_stability_benefit"]


def test_missing_patient_difference_stays_null_instead_of_averaging_favorable_subset():
    rows = patient_rows()
    rows[0]["single"]["pairs"]["positive_quarter_jaccard"]["mean"] = None
    decision = primary_decision(rows)
    assert decision["mean_patient_jaccard_difference"] is None
    assert decision["patient_difference_distribution"]["undefined"] == 1
    assert not decision["conditional_highlight_stability_benefit"]


def test_nonunit_prototype_is_rejected():
    geometry, _, _ = geometry_fixture()
    with pytest.raises(ValueError, match="unit"):
        fixed_contributions(geometry, np.stack([np.eye(3) * 2]), (0, 1))


def test_ensemble_normalizes_each_finding_instead_of_averaging_scores():
    single = np.stack([np.eye(3)] * 5)
    single[3:, 0] = [0, 1, 0]
    ensemble = normalized_ensemble_prototypes(single)
    np.testing.assert_allclose(np.linalg.norm(ensemble, axis=2), 1, atol=1e-14)
    np.testing.assert_allclose(ensemble[0, 0], np.array([0.6, 0.4, 0]) / np.hypot(0.6, 0.4))
    assert not np.allclose(ensemble[0, 0], single[:, 0].mean(axis=0))


def test_nonfinite_replay_cannot_evade_integrity_comparison():
    with pytest.raises(ValueError, match="Nonfinite"):
        nested_error({"score": float("nan")}, {"score": 0.5})
