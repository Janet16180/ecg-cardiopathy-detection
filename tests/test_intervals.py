import time

import numpy as np
import pytest

from ecg_experiment import evaluation, full_development, normal_manifold, screening_threshold
from ecg_experiment.intervals import (
    METRICS,
    metric_intervals,
    paired_auroc_difference,
    patient_groups,
    patient_resample,
    threshold_metrics,
    two_class_resamples,
)


def _cohort(records=600, patients=180, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.permutation(np.array([f"p{index:04d}" for index in rng.integers(0, patients, records)]))
    y = (rng.random(records) < 0.3).astype(int)
    first = 1 / (1 + np.exp(-(1.5 * y + rng.normal(0, 1, records))))
    second = 1 / (1 + np.exp(-(0.8 * y + rng.normal(0, 1, records))))
    return ids, y, first, second


def test_groups_match_brute_force_on_shuffled_ids():
    ids, _, _, _ = _cohort()
    expected = [np.flatnonzero(ids == patient) for patient in np.unique(ids)]
    groups = patient_groups(ids)
    assert len(groups) == len(expected)
    assert all(np.array_equal(got, want) for got, want in zip(groups, expected, strict=True))
    assert np.array_equal(np.sort(np.concatenate(groups)), np.arange(len(ids)))


def test_groups_accept_integer_ids_and_reject_two_dimensional_input():
    groups = patient_groups(np.array([7, 3, 7, 3, 9]))
    assert [group.tolist() for group in groups] == [[1, 3], [0, 2], [4]]
    with pytest.raises(ValueError, match="one-dimensional"):
        patient_groups(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="nonempty"):
        patient_groups(np.array([]))


def test_grouping_one_million_records_is_fast():
    rng = np.random.default_rng(1)
    ids = rng.integers(0, 300_000, 1_000_000)
    start = time.perf_counter()
    groups = patient_groups(ids)
    elapsed = time.perf_counter() - start
    assert sum(len(group) for group in groups) == len(ids)
    assert elapsed < 5


def test_resample_keeps_whole_patients():
    ids, _, _, _ = _cohort()
    groups = patient_groups(ids)
    rows = patient_resample(groups, np.random.default_rng(3))
    drawn = ids[rows]
    for patient in np.unique(drawn):
        copies = np.count_nonzero(drawn == patient) / np.count_nonzero(ids == patient)
        assert copies == int(copies)


def test_resamples_match_normal_manifold():
    ids, y, _, _ = _cohort()
    expected, invalid = normal_manifold.patient_resamples(ids, y, 200, 11)
    got = list(two_class_resamples(patient_groups(ids), y, 200, np.random.default_rng(11)))
    assert len(got) == len(expected)
    assert 200 - len(got) == invalid
    assert all(np.array_equal(a, b) for a, b in zip(got, expected, strict=True))


# The small cohort has single-class draws, so the skip counts are compared too.
COHORTS = [{"records": 600, "patients": 180}, {"records": 12, "patients": 8, "seed": 2}]


@pytest.mark.parametrize("cohort", COHORTS)
def test_paired_difference_matches_full_development_exactly(cohort):
    ids, y, first, second = _cohort(**cohort)
    expected = full_development.patient_bootstrap(ids, y, first, second, draws=200, seed=20020)
    got = paired_auroc_difference(ids, y, first, second, draws=200, seed=20020)
    assert got["difference"] == expected["difference"]
    assert got["ci_low"] == expected["ci_low"]
    assert got["ci_high"] == expected["ci_high"]
    assert got["skipped_draws"] == expected["invalid_draws"]
    assert (got["skipped_draws"] > 0) == (len(ids) < 100)
    assert got["draws"] == 200
    assert got["seed"] == 20020


@pytest.mark.parametrize("cohort", COHORTS)
def test_metric_intervals_match_evaluation_exactly(cohort):
    ids, y, prob, _ = _cohort(**cohort)
    expected = evaluation.patient_bootstrap(y, prob, ids, 0.4, repeats=200, seed=7)
    got = metric_intervals(y, prob, ids, 0.4, draws=200, seed=7)
    for name in METRICS:
        assert [got["metrics"][name]["ci_low"], got["metrics"][name]["ci_high"]] == expected[name]
    point = evaluation.metrics(y, prob, 0.4)
    assert all(got["metrics"][name]["value"] == point[name] for name in METRICS)


def test_threshold_rates_match_screening_threshold_exactly():
    ids, y, prob, _ = _cohort()
    expected = screening_threshold.patient_rates(ids, y, {"head": prob >= 0.5}, 200, 9)
    rng = np.random.default_rng(9)
    draws = [
        threshold_metrics(y[rows], prob[rows], 0.5)
        for rows in two_class_resamples(patient_groups(ids), y, 200, rng)
    ]
    for name in ("sensitivity", "specificity"):
        assert np.array_equal([values[name] for values in draws], expected[name]["head"])


def test_intervals_are_reproducible_and_seed_dependent():
    ids, y, first, second = _cohort()
    same = [paired_auroc_difference(ids, y, first, second, draws=100, seed=5) for _ in range(2)]
    other = paired_auroc_difference(ids, y, first, second, draws=100, seed=6)
    assert same[0] == same[1]
    assert other["ci_low"] != same[0]["ci_low"]
    assert metric_intervals(y, first, ids, 0.5, 100, 5) == metric_intervals(y, first, ids, 0.5, 100, 5)


def test_intervals_contain_observed_values_and_order():
    ids, y, first, second = _cohort(records=1500, patients=600)
    paired = paired_auroc_difference(ids, y, first, second, draws=200, seed=2)
    assert paired["ci_low"] < paired["difference"] < paired["ci_high"]
    assert paired["difference"] > 0
    summary = metric_intervals(y, first, ids, 0.5, draws=200, seed=2)
    for name in METRICS:
        values = summary["metrics"][name]
        assert values["ci_low"] <= values["value"] <= values["ci_high"]


def test_single_class_draws_are_skipped_and_counted():
    ids = np.array(["a", "b", "c", "d"])
    y = np.array([1, 0, 0, 0])
    scores = np.array([0.9, 0.2, 0.3, 0.1])
    other = np.array([0.4, 0.5, 0.3, 0.1])
    paired = paired_auroc_difference(ids, y, scores, other, draws=200, seed=0)
    summary = metric_intervals(y, scores, ids, 0.5, draws=200, seed=0)
    # A draw of 4 patients misses the only positive with probability (3/4)^4, about 32%.
    assert 30 < paired["skipped_draws"] < 100
    assert summary["skipped_draws"] == paired["skipped_draws"]


def test_all_single_class_draws_raise():
    ids = np.array(["a", "b", "c", "d"])
    y = np.array([1, 0, 0, 0])
    scores = np.array([0.9, 0.2, 0.3, 0.1])
    # With seed 33, none of the first three draws contains patient "a".
    with pytest.raises(ValueError, match="all 3 bootstrap draws had a single class"):
        paired_auroc_difference(ids, y, scores, scores[::-1], draws=3, seed=33)
    with pytest.raises(ValueError, match="all 3 bootstrap draws had a single class"):
        metric_intervals(y, scores, ids, 0.5, draws=3, seed=33)


def test_misaligned_inputs_raise():
    ids = np.array(["a", "b", "c"])
    with pytest.raises(ValueError, match="equal length"):
        metric_intervals(np.array([1, 0, 0]), np.full(2, 0.5), ids, 0.5, draws=5, seed=0)
