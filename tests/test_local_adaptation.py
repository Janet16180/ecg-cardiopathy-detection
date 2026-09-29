"""Checks of the Experiment 029 local adaptation helpers."""

import numpy as np
import pytest
from scipy.special import expit

from ecg_experiment.local_adaptation import (
    adapt,
    blend_weights,
    conservative_threshold,
    draw_records,
    local_components,
    recentered_scores,
    screening_rates,
    split_site,
    tolerance_rank,
)
from ecg_experiment.normal_manifold import fit_mahalanobis, mahalanobis_scores
from ecg_experiment.screening_threshold import calibrate, fit_platt, screening_threshold


def simulated(records: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 2, records)
    y = (rng.random(records) < expit(0.8 * logits - 0.3)).astype(int)
    return logits, y


def test_split_site_keeps_patients_together_and_halves_each_stratum():
    patients = np.repeat(np.arange(400), 2)
    y = np.repeat(np.arange(400) % 3 == 0, 2).astype(int)
    evaluation = split_site(patients, y, seed=5)
    assert not set(patients[evaluation]) & set(patients[~evaluation])
    positive_patients = np.unique(patients[y == 1])
    in_evaluation = np.unique(patients[evaluation & (y == 1)])
    assert len(in_evaluation) == len(positive_patients) // 2
    assert np.array_equal(evaluation, split_site(patients, y, seed=5))


def test_split_site_puts_a_patient_with_any_positive_in_the_positive_stratum():
    patients = np.array([0, 0, 1, 2, 3])
    y = np.array([0, 1, 1, 0, 0])
    evaluation = split_site(patients, y, seed=0)
    assert evaluation[0] == evaluation[1]
    assert evaluation[[0, 2]].sum() == 1


def test_draw_records_at_an_assumed_prevalence_draws_each_class_without_replacement():
    y = np.r_[np.ones(300), np.zeros(3000)].astype(int)
    rng = np.random.default_rng(0)
    counts = []
    for _ in range(200):
        chosen = draw_records(y, 400, rng, prevalence=0.05)
        assert len(np.unique(chosen)) == 400
        counts.append(y[chosen].sum())
    assert np.mean(counts) == pytest.approx(20, abs=1.5)
    plain = draw_records(y, 50, np.random.default_rng(1))
    assert len(np.unique(plain)) == 50


def test_tolerance_rank_matches_known_values():
    assert tolerance_rank(44) == 0
    assert tolerance_rank(45) == 1
    assert tolerance_rank(77) == 2
    assert tolerance_rank(340) == 12


def test_conservative_threshold_reaches_the_target_in_most_simulated_draws():
    rng = np.random.default_rng(3)
    reached = []
    for _ in range(2000):
        scores = rng.random(200)
        threshold, confident = conservative_threshold(np.ones(200, dtype=int), scores)
        assert confident
        reached.append(1 - threshold >= 0.95)
    assert np.mean(reached) >= 0.88


def test_conservative_threshold_without_enough_positives_uses_the_lowest():
    scores = np.array([0.9, 0.2, 0.7, 0.1])
    threshold, confident = conservative_threshold(np.array([1, 1, 1, 0]), scores)
    assert threshold == 0.2
    assert not confident


def test_platt_threshold_refers_the_same_ecgs_as_threshold_only():
    source_logits, source_y = simulated(500, 1)
    source = fit_platt(source_logits, source_y)
    source_point = (source, screening_threshold(source_y, calibrate(source, source_logits)))
    local_logits, local_y = simulated(120, 2)
    evaluation_logits, _ = simulated(3000, 3)
    decisions = []
    for option in ("threshold_only", "platt_threshold"):
        calibrator, threshold, fallback, _ = adapt(option, local_logits, local_y, source_logits, source_y,
                                                   source_point)
        assert not fallback
        decisions.append(calibrate(calibrator, evaluation_logits) >= threshold)
    assert np.array_equal(*decisions)


def test_adapt_falls_back_to_the_source_point_without_local_positives():
    source_logits, source_y = simulated(500, 4)
    source = fit_platt(source_logits, source_y)
    source_point = (source, 0.4)
    local_logits = np.array([-1.0, 0.5, 2.0])
    for option in ("threshold_only", "platt_threshold", "conservative"):
        calibrator, threshold, fallback, _ = adapt(option, local_logits, np.zeros(3, dtype=int),
                                                   source_logits, source_y, source_point)
        assert fallback
        assert calibrator is source
        assert threshold == 0.4
    _, _, fallback, _ = adapt("blend", local_logits, np.zeros(3, dtype=int), source_logits, source_y,
                              source_point)
    assert not fallback
    with pytest.raises(ValueError, match="Unknown option"):
        adapt("other", local_logits, np.zeros(3, dtype=int), source_logits, source_y, source_point)


def test_blend_weights_give_equal_totals():
    weights = blend_weights(800, 50)
    assert weights[:800].sum() == pytest.approx(weights[800:].sum())


def test_screening_rates():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    referred = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0], dtype=bool)
    rates = screening_rates(y, referred, 0.05)
    assert rates["sensitivity"] == 0.75
    assert rates["specificity"] == pytest.approx(5 / 6)
    assert rates["referrals_per_1000"] == 400
    assert rates["referrals_per_1000_assumed"] == pytest.approx(1000 * (0.05 * 0.75 + 0.95 / 6))


def test_local_components():
    assert [local_components(m) for m in (50, 100, 200, 500, 1000)] == [12, 25, 50, 64, 64]


def test_recentered_scores_with_the_reference_mean_equal_plain_scores():
    rng = np.random.default_rng(6)
    normals = rng.normal(size=(300, 80))
    model = fit_mahalanobis(normals)
    x = rng.normal(size=(20, 80))
    assert np.allclose(recentered_scores(model, x, model[0].mean_), mahalanobis_scores(model, x))
    shifted = recentered_scores(model, x + 3.0, model[0].mean_ + 3.0)
    assert np.allclose(shifted, mahalanobis_scores(model, x))
