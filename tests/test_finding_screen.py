"""Checks of the Experiment 033 finding-screen helpers."""

import numpy as np
import pytest

from ecg_experiment.finding_screen import (
    clipped_logits,
    extend_split,
    finding_z,
    referred_matrix,
    split_thresholds,
)
from ecg_experiment.referral_budget import budget_threshold


def test_clipped_logits_count_the_clipped_values():
    values, clipped = clipped_logits(np.array([0.0, 0.5, 1.0]))
    assert np.isfinite(values).all()
    assert values[1] == 0.0
    assert clipped == 2


def test_clipped_logits_reject_values_outside_the_unit_interval():
    with pytest.raises(ValueError, match="within"):
        clipped_logits(np.array([0.2, 1.5]))


def test_finding_z_is_standard_on_the_source_normals():
    source = np.array([0.1, 0.2, 0.3, 0.4])
    z = finding_z(source, source)
    assert z.mean() == pytest.approx(0.0)
    assert z.std() == pytest.approx(1.0)


def test_no_finding_score_gives_the_030_threshold():
    scores = np.random.default_rng(0).random(200)
    assert split_thresholds(scores[:, None], 50, ())[0] == budget_threshold(scores, 50)


def test_split_rule_refers_at_most_k_minus_the_extra_thresholds():
    rng = np.random.default_rng(1)
    for correlation in (0.0, 0.5, 0.95):
        base = rng.standard_normal(200)
        other = correlation * base + np.sqrt(1 - correlation**2) * rng.standard_normal(200)
        third = rng.standard_normal(200)
        for per_mille in (20, 50, 100):
            limit = per_mille * 200 // 1000
            for shares in ((100,), (300,), (100, 100)):
                normals = np.column_stack([base, other, third][:1 + len(shares)])
                thresholds = split_thresholds(normals, per_mille, shares)
                count = referred_matrix(normals, thresholds[None, :])[:, 0].sum()
                assert count <= limit - len(shares)
                assert count >= limit - len(shares) - sum(share * limit // 1000 for share in shares)


def test_finding_threshold_sits_at_its_share_of_the_budget():
    normals = np.column_stack([np.arange(200.0), np.random.default_rng(2).permutation(np.arange(200.0))])
    thresholds = split_thresholds(normals, 50, (200,))
    assert (normals[:, 1] > thresholds[1]).sum() == 2


def test_overlap_is_recycled_to_the_binary_readout():
    scores = np.arange(200.0)
    normals = np.column_stack([scores, scores])
    thresholds = split_thresholds(normals, 50, (200,))
    assert (normals[:, 0] > thresholds[0]).sum() == 9


def test_split_rule_rejects_ranks_beyond_the_budget():
    normals = np.random.default_rng(3).random((200, 2))
    with pytest.raises(ValueError, match="exceed"):
        split_thresholds(normals, 50, (1000,))


def test_split_rule_rejects_a_shape_mismatch():
    with pytest.raises(ValueError, match="one column per finding"):
        split_thresholds(np.zeros((10, 2)), 50, (100, 100))


def test_extend_split_keeps_known_halves_and_follows_patients():
    patients = np.array(["a", "a", "b", "c", "d", "e", "f"])
    known = np.array([True, False, True, False, False, False, False])
    evaluation = np.array([True, False, False, False, False, False, False])
    positive = np.array([0, 0, 0, 1, 1, 0, 0])
    result = extend_split(patients, known, evaluation, positive, 7)
    assert result[:3].tolist() == [True, True, False]
    assert result[3:5].sum() == 1
    assert result[5:].sum() == 1


def test_extend_split_rejects_a_patient_in_both_halves():
    patients = np.array(["a", "a"])
    with pytest.raises(ValueError, match="both halves"):
        extend_split(patients, np.array([True, True]), np.array([True, False]), np.zeros(2), 1)
