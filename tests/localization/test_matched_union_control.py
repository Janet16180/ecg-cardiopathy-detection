"""Synthetic matched-patient weighting, exact identities and symmetric decision controls."""

import numpy as np
import pytest

from ecg_experiment.localization.matched_union_control import (
    aggregate_methods,
    array_error,
    between_methods,
    primary_decision,
    union_diagnostics,
    union_prototypes,
)


def varied_fixture() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(10)
    vectors = rng.normal(size=(6, 5))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    banks = np.array([[0, 1, 2], [0, 3, 4], [1, 3, 5], [0, 2, 5], [2, 4, 5]])
    return vectors, banks


def test_analytic_coefficients_include_each_bank_norm() -> None:
    vectors, banks = varied_fixture()
    result = union_prototypes(vectors, banks)
    norms = np.linalg.norm(vectors[banks].sum(axis=1), axis=1)
    coefficients = np.array(
        [sum(1 / norms[j] for j, bank in enumerate(banks) if i in bank) for i in range(len(vectors))]
    )
    np.testing.assert_allclose(result["weights"]["A"], coefficients / coefficients.sum(), atol=1e-14)
    np.testing.assert_allclose(result["prototypes"]["A"], result["analytic_A"], atol=1e-14)
    assert not np.allclose(result["weights"]["A"], result["weights"]["C"])
    for weights in result["weights"].values():
        assert np.all(weights > 0)
        assert weights.sum() == pytest.approx(1)


def test_equal_bank_norms_make_a_and_c_identical() -> None:
    vectors = np.eye(4)
    banks = np.array([[0, 1, 2], [0, 1, 3], [0, 1, 3], [1, 2, 3], [0, 2, 3]])
    result = union_prototypes(vectors, banks)
    np.testing.assert_allclose(result["prototypes"]["A"], result["prototypes"]["C"], atol=1e-14)
    assert not np.allclose(result["prototypes"]["C"], result["prototypes"]["B"])


def test_equal_occurrence_counts_make_c_and_b_identical() -> None:
    rng = np.random.default_rng(11)
    vectors = rng.normal(size=(5, 4))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    banks = np.array([[(i + offset) % 5 for offset in range(3)] for i in range(5)])
    result = union_prototypes(vectors, banks)
    np.testing.assert_allclose(result["weights"]["B"], result["weights"]["C"], atol=1e-14)
    np.testing.assert_allclose(result["prototypes"]["B"], result["prototypes"]["C"], atol=1e-14)


def test_identical_patient_vectors_make_all_prototypes_identical() -> None:
    vectors, banks = varied_fixture()
    vectors[:] = np.array([1, 0, 0, 0, 0])
    result = union_prototypes(vectors, banks)
    for method in ("A", "B", "C"):
        np.testing.assert_allclose(result["prototypes"][method], vectors[0], atol=1e-14)


def test_undefined_bank_normalization_is_not_an_exclusion() -> None:
    vectors = np.array([[1.0, 0], [-0.5, np.sqrt(0.75)], [-0.5, -np.sqrt(0.75)]])
    with pytest.raises(ValueError, match="bank normalization"):
        union_prototypes(vectors, np.tile(np.arange(3), (5, 1)))


def patient_rows(delta: float) -> list[dict]:
    return [
        {
            method: {
                "pairs": {
                    "positive_quarter_jaccard": {
                        "count": 45,
                        "defined": 45,
                        "mean": 0.5 + (delta if method == "A" else 0),
                    }
                }
            }
            for method in ("A", "B", "C")
        }
        for _ in range(60)
    ]


@pytest.mark.parametrize(
    ("delta", "category"),
    [
        (0.1, "A_material_conditional_consistency_advantage"),
        (-0.1, "B_material_conditional_consistency_advantage"),
        (0.01, "no_material_mean_difference_at_operational_resolution"),
    ],
)
def test_primary_is_symmetric_and_prespecified(delta: float, category: str) -> None:
    result = primary_decision(patient_rows(delta))
    assert result["assessable"]
    assert result["category"] == category


@pytest.mark.parametrize(
    ("value", "method", "category"),
    [
        (0.05, "A", "A_material_conditional_consistency_advantage"),
        (0.05, "B", "B_material_conditional_consistency_advantage"),
        (np.nextafter(0.05, 0), "A", "no_material_mean_difference_at_operational_resolution"),
        (np.nextafter(0.05, 0), "B", "no_material_mean_difference_at_operational_resolution"),
    ],
)
def test_declared_resolution_boundaries_are_inclusive(value: float, method: str, category: str) -> None:
    rows = patient_rows(0)
    for row in rows:
        row["A"]["pairs"]["positive_quarter_jaccard"]["mean"] = 0.0
        row["B"]["pairs"]["positive_quarter_jaccard"]["mean"] = 0.0
        row[method]["pairs"]["positive_quarter_jaccard"]["mean"] = value
    assert primary_decision(rows)["category"] == category


def test_opposite_patient_effects_remain_visible_when_mean_cancels() -> None:
    rows = patient_rows(0.1)
    rows[30:] = patient_rows(-0.1)[30:]
    result = primary_decision(rows)
    assert result["category"] == "no_material_mean_difference_at_operational_resolution"
    assert result["effect"]["patient_counts"]["at_least_plus_0.05"] == 30
    assert result["effect"]["patient_counts"]["at_most_minus_0.05"] == 30


def test_every_patient_coverage_and_missing_difference_are_required() -> None:
    rows = patient_rows(0.1)
    rows[0]["B"]["pairs"]["positive_quarter_jaccard"]["defined"] = 42
    assert primary_decision(rows)["category"] == "unassessable"
    rows[0]["B"]["pairs"]["positive_quarter_jaccard"]["mean"] = None
    result = primary_decision(rows)
    assert result["effect"]["mean_patient_difference"] is None
    assert result["effect"]["patient_counts"]["undefined"] == 1


def test_same_block_events_and_zero_mass_maps_preserve_nulls() -> None:
    left = np.array([1.0, -1, 0, 0, 1, 1, 1, 1, 1, 1])
    right = np.array([-1.0, 1, 1, 0, 1, 1, 1, 1, 1, 1])
    first = {"beat": left[:, None] * [1, 0, 0, 0], "token": left[:, None] * [1, 0, 0, 0], "margins": left}
    second = {"beat": right[:, None] * [1, 0, 0, 0], "token": right[:, None] * [1, 0, 0, 0], "margins": right}
    stats, arrays = between_methods(first, second, np.ones(4, dtype=bool))
    assert stats["strict_margin_sign_reversal"]["event_count"] == 2
    assert stats["zero_margin_transition"]["event_count"] == 1
    assert stats["both_margins_zero"]["event_count"] == 1
    assert stats["beat_normalized_absolute_TV"]["undefined"] == 2
    assert np.isnan(arrays["positive_quarter_jaccard"][3])


def test_duplicate_union_counts_require_exact_uniform_prototypes() -> None:
    signature = ((1, 2, 3), (4, 5, 6), (7, 8, 9))
    prototypes = np.tile(np.eye(3), (10, 1, 1))
    result = union_diagnostics([signature] * 10, prototypes)
    assert result["unique_three_finding_union_triples"] == 1
    assert result["identical_union_triple_block_pairs"] == 45
    prototypes[-1, 0, 0] -= 0.01
    with pytest.raises(ValueError, match="canonical unions"):
        union_diagnostics([signature] * 10, prototypes)


def test_predecessor_null_mask_cannot_be_changed() -> None:
    with pytest.raises(ValueError, match="null mask"):
        array_error(np.array([0.0, np.nan]), np.array([0.0, 0.0]))


def test_aggregation_declares_45_pairs_for_every_method() -> None:
    row = patient_rows(0.01)[0]
    for method in ("A", "B", "C"):
        row[method]["metrics"] = {"synthetic": 0.1}
    events = {
        name: {"mean": 0.0, "undefined": 0, "event_count": 0}
        for name in ("strict_margin_sign_reversal", "zero_margin_transition", "both_margins_zero")
    }
    row["between"] = dict.fromkeys(("A:B", "A:C", "B:C"), events)
    prototypes = {method: np.tile(np.eye(3), (10, 1, 1)) for method in ("A", "B", "C")}
    result = aggregate_methods([row], prototypes)
    for method in ("A", "B", "C"):
        assert (
            result["methods"][method]["pair_metrics"]["positive_quarter_jaccard"]["all_pairs_per_patient"]
            == 45
        )
