"""Checks of the Experiment 024 prototypes, neighbors, partitions and audits on synthetic vectors."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, normalize

from ecg_experiment.embedding_geometry import (
    age_decade,
    ami_summary,
    audit_candidates,
    device_balanced_prototype,
    heart_rate_band,
    label_disagreement,
    medoids,
    method_scores,
    multi_prototype_scores,
    neighbor_indices,
    patient_subset,
    prototype,
    prototype_scores,
    ridge_r2,
    same_device_summary,
    unit_space,
)


def two_classes(rows: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = np.arange(rows) % 2
    x = rng.normal(size=(rows, 8))
    x[:, 0] += 3 * (2 * y - 1)
    return normalize(x), y


def test_unit_space_uses_the_training_scaler_and_unit_length():
    rng = np.random.default_rng(1)
    train = rng.normal(5, 2, size=(50, 4))
    scaler = StandardScaler().fit(train)
    vectors = unit_space(scaler, rng.normal(5, 2, size=(10, 4)).astype(np.float32))
    assert vectors.dtype == np.float64
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)


def test_prototype_is_the_normalized_class_mean():
    x = normalize(np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]))
    assert np.allclose(prototype(x, np.array([True, True, False])), [2 ** -0.5, 2 ** -0.5])


def test_prototype_scores_separate_well_separated_classes():
    x, y = two_classes()
    assert roc_auc_score(y, prototype_scores(x, y, x)) > 0.95


def test_multi_prototype_scores_capture_a_two_mode_class():
    rng = np.random.default_rng(2)
    positive = np.vstack([rng.normal([4, 0], 0.3, (100, 2)), rng.normal([-4, 0], 0.3, (100, 2))])
    negative = rng.normal([0, 4], 0.3, (200, 2))
    x = normalize(np.vstack([positive, negative]))
    y = np.repeat([1, 0], 200)
    assert roc_auc_score(y, multi_prototype_scores(x, y, x, k=2)) > 0.99


def test_device_balanced_prototype_weights_devices_equally_and_skips_small_ones():
    x = normalize(np.vstack([np.tile([1.0, 0.0], (90, 1)), np.tile([0.0, 1.0], (30, 1)),
                             np.tile([-1.0, 0.0], (5, 1))]))
    devices = np.array(["a"] * 90 + ["b"] * 30 + ["c"] * 5)
    balanced = device_balanced_prototype(x, np.ones(len(x), bool), devices, minimum=20)
    assert np.allclose(balanced, [2 ** -0.5, 2 ** -0.5])
    with pytest.raises(ValueError, match="No device"):
        device_balanced_prototype(x, devices == "c", devices, minimum=20)


def test_neighbor_indices_are_sorted_and_respect_groups():
    x, _ = two_classes(rows=60)
    brute = np.argsort(-(x[:5] @ x.T), axis=1)[:, :4]
    assert np.array_equal(neighbor_indices(x, x[:5], k=4), brute)
    own = np.arange(len(x))
    excluded = neighbor_indices(x, x, k=4, train_groups=own, query_groups=own)
    assert not (excluded == own[:, None]).any()
    groups = np.arange(len(x)) % 3
    other = neighbor_indices(x, x, k=4, train_groups=groups, query_groups=groups)
    assert (groups[other] != groups[:, None]).all()


def test_method_scores_returns_every_protocol_method():
    x, y = two_classes()
    devices = np.where(np.arange(len(y)) % 4 < 2, "a", "b")
    scores = method_scores(x, y, devices, x[:40], devices[:40])
    assert set(scores) == {"prototype", "prototype_device_balanced", "knn", "knn_other_device",
                           "multi_prototype_2", "multi_prototype_4", "multi_prototype_8"}
    assert all(len(values) == 40 for values in scores.values())
    assert scores["knn"].min() >= 0
    assert scores["knn"].max() <= 1


def test_same_device_summary_compares_neighbors_with_training_share():
    train_devices = np.array(["a", "a", "a", "b"])
    neighbors = np.array([[0, 1], [3, 0]])
    summary = same_device_summary(neighbors, train_devices, np.array(["a", "b"]))
    assert summary["same_device_fraction"] == pytest.approx(0.75)
    assert summary["expected_share"] == pytest.approx(0.5)
    assert summary["per_device"]["b"]["training_share"] == pytest.approx(0.25)


def test_label_disagreement_and_audit_candidates():
    y = np.array([0, 1, 1, 1])
    neighbors = np.array([[1, 2], [2, 3], [0, 3], [0, 1]])
    disagreement = label_disagreement(neighbors, y)
    assert np.allclose(disagreement, [1.0, 0.0, 0.5, 0.5])
    assert np.array_equal(audit_candidates(disagreement, np.array([0.9, 1.0, 1.0, 0.1])),
                          [True, False, False, False])


def test_medoids_are_group_members_closest_to_the_prototype():
    x = normalize(np.array([[1.0, 0.1], [1.0, 0.0], [0.0, 1.0], [1.0, -0.1]]))
    mask = np.array([True, True, False, True])
    positions, similarity = medoids(x, mask, count=2)
    assert positions[0] == 1
    assert 2 not in positions
    assert similarity[0] >= similarity[1]


def test_ridge_r2_recovers_a_linear_target():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(300, 3))
    target = x @ np.array([1.0, -2.0, 0.5])
    assert ridge_r2(x[:200], target[:200], x[200:], target[200:]) > 0.99


def test_age_decade_and_heart_rate_band():
    assert age_decade(np.array([5.0, 64.0, np.nan])).tolist() == ["0", "60", "missing"]
    bands = heart_rate_band(np.array([59.9, 60.0, 100.0, 100.1])).tolist()
    assert bands == ["<60", "60-100", "60-100", ">100"]


def test_ami_summary_is_one_for_identical_partitions():
    partition = np.array([0, 0, 1, 1, 2, 2])
    summary = ami_summary(partition, {"same": partition.astype(str), "other": np.array(list("ababab"))})
    assert summary["same"] == pytest.approx(1.0)
    assert summary["other"] < 0.5


def test_patient_subset_takes_one_row_per_patient_reproducibly():
    patients = np.repeat(np.arange(50), 3)
    chosen = patient_subset(patients, 20, seed=24024)
    assert len(set(patients[chosen])) == 20
    assert np.array_equal(chosen, patient_subset(patients, 20, seed=24024))
    with pytest.raises(ValueError, match="Fewer"):
        patient_subset(patients, 60)
