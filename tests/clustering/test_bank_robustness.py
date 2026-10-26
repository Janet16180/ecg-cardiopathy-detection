"""Scientific invariants for prospective support-bank sensitivity."""

import time

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from ecg_experiment.clustering.bank_robustness import (
    align_fitting,
    bank_affinities,
    bank_fit,
    bank_prototypes,
    primary_decision,
    sensitivity,
    support_draws,
)
from ecg_experiment.clustering.reference_geometry import affinity_views
from ecg_experiment.same_wave_ntt04 import FINDINGS


def selected_supports():
    """Create independent class pools without any real patient data."""
    groups = []
    for label, finding in enumerate(FINDINGS):
        for draw in range(3):
            groups.append(
                {
                    "finding": finding,
                    "supports": [
                        {"patient_id": label * 100 + draw * 3 + i, "ecg_id": label * 100 + draw * 3 + i}
                        for i in range(3)
                    ],
                }
            )
    return {"support_draws": groups}


def test_banks_are_unique_patient_equal_and_order_independent():
    """Selection depends on sorted identities and the frozen RNG, never insertion order."""
    selection = selected_supports()
    first = support_draws(selection)
    selection["support_draws"].reverse()
    assert support_draws(selection) == first
    assert len(first) == 50
    signatures = {
        tuple(row["patient_id"] for finding in FINDINGS for row in bank["supports"][finding])
        for bank in first
    }
    assert len(signatures) == 50
    for bank in first:
        assert all(len({r["patient_id"] for r in bank["supports"][finding]}) == 3 for finding in FINDINGS)
    features = {
        str(i): {"native_qrs": np.eye(3)[i // 100]}
        for bank in first
        for finding in FINDINGS
        for row in bank["supports"][finding]
        for i in [row["ecg_id"]]
    }
    np.testing.assert_allclose(bank_prototypes(first[0], features), np.eye(3))


def test_single_bank_coordinates_equal_frozen_multi_bank_adapter():
    """A real single bank reproduces the historical region aggregation exactly."""
    rng = np.random.default_rng(13)
    qrs = rng.normal(size=(3, 8))
    qrs /= np.linalg.norm(qrs, axis=1, keepdims=True)
    regions = rng.normal(size=(12, 8))
    regions /= np.linalg.norm(regions, axis=1, keepdims=True)
    prototype = rng.normal(size=(3, 8))
    prototype /= np.linalg.norm(prototype, axis=1, keepdims=True)
    cache = {"qrs": qrs, "region_vectors": regions, "region_offsets": np.array([0, 3, 7, 12])}
    record, regional = bank_affinities(cache, prototype)
    historical_record, historical_regional, _ = affinity_views(cache, np.stack([prototype] * 3))
    np.testing.assert_allclose(record, historical_record, atol=1e-14)
    np.testing.assert_allclose(regional, historical_regional, atol=1e-14)
    cache["region_offsets"][1] = 1
    with pytest.raises(ValueError, match="boundaries"):
        bank_affinities(cache, prototype)


def test_alignment_never_uses_anchor_labels_and_ties_are_lexicographic():
    """Flipping anchor labels cannot change a fitting-only cluster-name mapping."""
    reference = np.array([0, 1, 0, 1, 0, 1])
    labels = np.array([1, 0, 1, 0, 1, 0])
    fitting = np.arange(4)
    aligned, mapping = align_fitting(labels, reference, fitting, 2)
    assert mapping == [1, 0]
    np.testing.assert_array_equal(aligned, reference)
    reference[4:] = 1 - reference[4:]
    assert align_fitting(labels, reference, fitting, 2)[1] == mapping
    assert align_fitting(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]), np.arange(4), 2)[1] == [0, 1]


def test_invalid_banks_stay_in_all_summary_denominators():
    """Admissibility failure cannot disappear from otherwise perfectly agreeing partitions."""
    reference = np.array([0, 0, 1, 1, 0, 1])
    assignments = np.stack([reference, 1 - reference, reference])
    checks = [{"admissible": True}, {"admissible": False}, {"admissible": True}]
    summary, details = sensitivity(assignments, checks, reference, np.arange(4), np.array([4, 5]), 2)
    assert summary["banks"] == 3
    assert summary["admissible_banks"] == 2
    assert summary["pairwise_comparisons"] == 3
    assert summary["reference_agreement"]["minimum"] == 1
    np.testing.assert_array_equal(details["anchor_coassignment"], np.eye(2))
    np.testing.assert_array_equal(details["entropy_bits"], [0, 0])


def test_bank_fit_centering_is_fit_only():
    """An anchor translation cannot affect training predictions or centering."""
    rng = np.random.default_rng(8)
    x = np.r_[rng.normal(-3, 0.1, (30, 2)), rng.normal(3, 0.1, (30, 2)), rng.normal(size=(10, 2))]
    config = {"method": "kmeans", "covariance": "none", "clusters": 2}
    _, labels = bank_fit(x, np.arange(60), np.arange(60, 70), config, time.perf_counter() + 20)
    x[60:] += 1000
    _, changed = bank_fit(x, np.arange(60), np.arange(60, 70), config, time.perf_counter() + 20)
    assert adjusted_rand_score(labels[:60], changed[:60]) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("admissible_banks", 44),
        ("pairwise_anchor_ari", {"p10": 0.49}),
        ("fraction_anchors_agreement_at_least_0.80", 0.79),
    ],
)
def test_each_primary_gate_is_necessary(field, value):
    """No secondary result can override a failed operational condition."""
    summary = {
        "banks": 50,
        "anchor_denominator": 411,
        "admissible_banks": 50,
        "pairwise_anchor_ari": {"p10": 0.9},
        "fraction_anchors_agreement_at_least_0.80": 0.99,
    }
    assert primary_decision(summary)["primary_robustness_gate_passed"]
    summary[field] = value
    assert not primary_decision(summary)["primary_robustness_gate_passed"]
