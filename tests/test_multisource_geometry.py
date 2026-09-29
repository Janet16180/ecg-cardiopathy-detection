"""Checks of the Experiment 024b partition bootstraps, neighbor shares and source probes on synthetic data."""

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, homogeneity_score
from sklearn.preprocessing import normalize

from ecg_experiment.embedding_geometry import neighbor_indices
from ecg_experiment.multisource_geometry import (
    bootstrap_partition,
    bootstrap_source_probe,
    challenge_combination,
    chunked_neighbors,
    equal_family_draw,
    explained_entropy,
    neighbor_shares,
    one_vs_rest_probabilities,
    paired_interval,
    source_probe_metrics,
)


def test_challenge_combination_matches_the_ptbxl_format():
    frame = pd.DataFrame({"primary": [0, 1, 1, 1], "MI": [0, 1, 1, 0], "STTC": [0, 0, 1, 0],
                          "CD": [0, 1, 0, 0], "HYP": [0, 0, 0, 1]})
    assert challenge_combination(frame).tolist() == ["NORM", "CD+MI", "MI+STTC", "HYP"]


def test_equal_family_draw_takes_the_same_count_from_each_family_reproducibly():
    families = np.array(["b"] * 30 + ["a"] * 20 + ["c"] * 10)
    first = equal_family_draw(families, 5, seed=3)
    assert np.array_equal(first, equal_family_draw(families, 5, seed=3))
    assert np.array_equal(first, np.sort(first))
    assert pd.Series(families[first]).value_counts().to_dict() == {"a": 5, "b": 5, "c": 5}


def test_explained_entropy_is_one_for_a_factor_the_partition_determines():
    partition = np.repeat([0, 1, 2, 3], 25)
    factors = {"coarse": partition // 2, "unrelated": np.tile([0, 1], 50)}
    entropy = explained_entropy(partition, factors)
    assert entropy["coarse"] == 1.0
    assert entropy["unrelated"] < 0.01


def test_bootstrap_partition_is_seeded_and_brackets_the_observed_ami():
    rng = np.random.default_rng(4)
    factor = rng.integers(0, 3, 600)
    partition = np.where(rng.random(600) < 0.7, factor, rng.integers(0, 3, 600))
    first = bootstrap_partition(partition, {"factor": factor}, 200, seed=9)
    second = bootstrap_partition(partition, {"factor": factor}, 200, seed=9)
    assert np.array_equal(first["factor"], second["factor"])
    observed = adjusted_mutual_info_score(factor, partition)
    low, high = np.percentile(first["factor"], [2.5, 97.5])
    assert low < observed < high
    entropy = bootstrap_partition(partition, {"factor": factor}, 5, seed=9, measure=homogeneity_score)
    assert entropy["factor"].shape == (5,)


def test_paired_interval_uses_paired_differences():
    first, second = np.arange(100.0), np.arange(100.0) - 1.0
    assert paired_interval(0.5, first, second) == {"difference": 0.5, "ci_low": 1.0, "ci_high": 1.0}


def test_chunked_neighbors_equals_neighbor_indices_with_and_without_groups():
    rng = np.random.default_rng(5)
    train, query = normalize(rng.normal(size=(300, 6))), normalize(rng.normal(size=(1100, 6)))
    train_groups, query_groups = rng.integers(0, 3, 300), rng.integers(0, 3, 1100)
    assert np.array_equal(chunked_neighbors(train, query), neighbor_indices(train, query))
    assert np.array_equal(chunked_neighbors(train, query, train_groups, query_groups),
                          neighbor_indices(train, query, 25, train_groups, query_groups))


def test_neighbor_shares_average_over_queries():
    neighbors = np.array([[0, 1], [2, 2]])
    groups = np.array(["a", "b", "c"])
    assert neighbor_shares(neighbors, groups, ["a", "b", "c"]) == {"a": 0.25, "b": 0.25, "c": 0.5}


def test_source_probe_identifies_separated_sources():
    rng = np.random.default_rng(6)
    names = ["a", "b", "c"]
    classes = np.repeat(names, 200)
    centers = {"a": [3, 0], "b": [0, 3], "c": [-3, -3]}
    x = np.array([centers[name] for name in classes]) + rng.normal(size=(600, 2))
    probabilities = one_vs_rest_probabilities(x[::2], classes[::2], x[1::2], names)
    assert probabilities.shape == (300, 3)
    result = source_probe_metrics(classes[1::2], probabilities, names)
    assert result["balanced_accuracy"] > 0.9
    assert result["mean_auroc"] > 0.95
    drawn = bootstrap_source_probe(classes[1::2], probabilities, names, 50, seed=1)["balanced_accuracy"]
    assert drawn["ci_low"] <= result["balanced_accuracy"] <= drawn["ci_high"]
    assert bootstrap_source_probe(classes[1::2], probabilities, names, 5, seed=1)["invalid_draws"] == {
        "value": 0.0}


def test_source_probe_balanced_accuracy_weights_classes_equally():
    classes = np.array(["a"] * 8 + ["b"] * 2)
    probabilities = np.column_stack([np.ones(10), np.zeros(10)])
    assert source_probe_metrics(classes, probabilities, ["a", "b"])["balanced_accuracy"] == 0.5
