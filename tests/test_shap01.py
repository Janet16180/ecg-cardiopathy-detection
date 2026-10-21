"""Exact coalition identities, grouping and training-only background contracts."""

import hashlib
import itertools
import math
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from ecg_experiment.clustering01 import FEATURES
from ecg_experiment.shap01 import (
    background_indices,
    fit_fixed_model,
    group_matrix,
    linear_shap,
    read_labels,
    reconstruction,
    sensitivity,
    training_role,
)


def enumerate_shap(sample, beta, intercept, background, feature_groups):
    mean = background.mean(axis=0)
    count = len(feature_groups)
    result = []
    for group, indices in enumerate(feature_groups):
        total = 0
        other = [i for i in range(count) if i != group]
        for size in range(count):
            for coalition in itertools.combinations(other, size):
                first = mean.copy()
                for member in coalition:
                    first[feature_groups[member]] = sample[feature_groups[member]]
                second = first.copy()
                second[indices] = sample[indices]
                weight = math.factorial(size) * math.factorial(count - size - 1) / math.factorial(count)
                total += weight * ((intercept + beta @ second) - (intercept + beta @ first))
        result.append(total)
    return np.array(result)


def test_linear_formula_matches_all_coalitions_and_merged_group_game():
    beta = np.array([2.0, -3.0, 0.25])
    sample = np.array([1.5, 2.0, -4.0])
    background = np.array([[0.0, 1.0, 0.5], [2.0, -1.0, -0.5]])
    base, phi = linear_shap(sample[None], beta, -0.7, background)
    exact = enumerate_shap(sample, beta, -0.7, background, [[0], [1], [2]])
    np.testing.assert_allclose(phi[0], exact, atol=1e-14)
    np.testing.assert_allclose(base + phi.sum(), -0.7 + beta @ sample, atol=1e-14)
    merged = enumerate_shap(sample, beta, -0.7, background, [[0, 1], [2]])
    np.testing.assert_allclose(merged, [phi[0, :2].sum(), phi[0, 2]], atol=1e-14)


def test_unused_correlated_feature_has_zero_interventional_contribution():
    background = np.array([[0.0, 0.0], [2.0, 2.0]])
    _, phi = linear_shap(np.array([[3.0, 3.0]]), np.array([4.0, 0.0]), 0, background)
    np.testing.assert_array_equal(phi, [[8.0, 0.0]])


def test_background_shift_is_patient_independent_and_base_cancels_it():
    values = np.array([[1, 2, 3], [-5, 8, 6]])
    beta = np.array([0.2, -0.4, 2.0])
    first = np.zeros((3, 3))
    second = np.ones((2, 3))
    base1, phi1 = linear_shap(values, beta, 0.3, first)
    base2, phi2 = linear_shap(values, beta, 0.3, second)
    np.testing.assert_allclose(phi2 - phi1, np.tile(-beta, (2, 1)), atol=1e-14)
    np.testing.assert_allclose(base1 + phi1.sum(axis=1), base2 + phi2.sum(axis=1), atol=1e-14)


def test_group_membership_covers_each_feature_once_without_cancellation_loss():
    groups, matrix = group_matrix(FEATURES)
    assert len(groups) == 13
    np.testing.assert_array_equal(matrix.sum(axis=1), np.ones(53))
    np.testing.assert_array_equal(matrix.sum(axis=0), [4] * 12 + [5])
    phi = np.arange(106).reshape(2, 53) - 50
    np.testing.assert_allclose((phi @ matrix).sum(axis=1), phi.sum(axis=1))
    with pytest.raises(ValueError, match="53"):
        group_matrix(FEATURES[:-1] + [FEATURES[0]])


def test_backgrounds_never_use_evaluation_or_undefined_labels():
    train = np.array([True, True, False, True])
    labels = np.array([0.0, 1.0, 0.0, np.nan])
    strict = np.array([0.0, np.nan, 0.0, 0.0])
    selected = background_indices(train, labels, strict)
    np.testing.assert_array_equal(selected["all_training"], [0, 1])
    np.testing.assert_array_equal(selected["strict_normal_training"], [0])


def test_empty_clean_background_stops_instead_of_substituting_test_normals():
    with pytest.raises(ValueError, match="nonempty"):
        background_indices(np.array([True, False]), np.array([1, 0]), np.array([1, 0]))


def test_label_parse_is_gated_before_opening_poisoned_reserved_payload(tmp_path):
    identities = pd.DataFrame({"ecg_id": [1], "patient_id": [1]})
    with pytest.raises(ValueError, match="verified"):
        read_labels(identities, set(), tmp_path / "must_not_open.csv")


def test_split_matches_exact_hash_namespace():
    for patient in (1, 5, 123, 12345):
        bucket = int(hashlib.sha256(f"shap01_split_v1:20261018:{patient}".encode()).hexdigest(), 16) % 10000
        assert training_role(patient) == (bucket < 8000)


def test_fit_contract_sends_only_training_rows_to_scaler_and_single_model(monkeypatch):
    training = [p for p in range(1, 100) if training_role(p)][:4]
    evaluation = [p for p in range(1, 100) if not training_role(p)][:4]
    patients = np.array(training + evaluation)
    features = np.arange(16).reshape(8, 2)
    target = np.tile([0, 1], 4)
    strict = target.astype(float)
    scaler = MagicMock()
    scaler.fit.return_value = scaler
    scaler.transform.side_effect = lambda values: values
    model = MagicMock()
    model.n_iter_, model.max_iter = np.array([1]), 2000
    constructor = MagicMock(return_value=model)
    monkeypatch.setattr("ecg_experiment.shap01.StandardScaler", lambda: scaler)
    monkeypatch.setattr("ecg_experiment.shap01.LogisticRegression", constructor)
    _, _, train_rows, eval_rows, backgrounds = fit_fixed_model(features, target, patients, strict)
    np.testing.assert_array_equal(scaler.fit.call_args.args[0], features[:4])
    np.testing.assert_array_equal(model.fit.call_args.args[0], features[:4])
    assert constructor.call_count == model.fit.call_count == scaler.fit.call_count == 1
    assert constructor.call_args.kwargs == {
        "C": 1,
        "l1_ratio": 0.0,
        "class_weight": "balanced",
        "solver": "lbfgs",
        "max_iter": 2000,
        "tol": 1e-8,
        "random_state": 20261018,
    }
    assert not set(train_rows) & set(eval_rows)
    assert set(backgrounds["strict_normal_training"]) <= set(train_rows)


def test_top_positive_reconstruction_keeps_negative_and_other_positive_groups():
    values = np.array([[9, -10, 8, 7, 6, 5, 4], [-1, -2, -3, -4, -5, -6, -7]])
    rebuilt, positive, remaining = reconstruction(-0.5, values)
    np.testing.assert_array_equal(positive, [35, 0])
    np.testing.assert_allclose(rebuilt, -0.5 + values.sum(axis=1))
    np.testing.assert_array_equal(remaining, [-6, -28])


def test_rank_changes_and_no_positive_groups_are_explicit():
    first = np.array([[2, 1, -0.5], [0, 0, 0]])
    second = np.array([[0.1, 3, -0.5], [0, 0, 0]])
    summary, frame = sensitivity(first, second, ["I", "II", "global"])
    assert summary["top_absolute_changed"] == 0.5
    assert summary["top_positive_changed"] == 0.5
    assert summary["undefined_rank_correlations"] == 1
    assert frame.top_positive_all.iloc[1] == frame.top_positive_normal.iloc[1] == "none"
