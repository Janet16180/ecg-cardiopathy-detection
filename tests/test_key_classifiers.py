"""Scientific label, grouping, marginalization and model-selection boundary checks."""

import numpy as np
import pandas as pd
import pytest

from ecg_experiment.key_classifier_models import (
    STATE_BITS,
    encode_states,
    fit_package,
    folds,
    joint_metrics,
    marginal_scores,
    predict_package,
    record_groups,
    resample_references,
)


def test_joint_states_preserve_all_sixteen_membership_patterns():
    np.testing.assert_array_equal(encode_states(STATE_BITS), np.arange(16))
    with pytest.raises(ValueError, match="binary"):
        encode_states(np.full((2, 4), 0.5))


def test_marginalization_preserves_overlapping_probabilities():
    probability = np.zeros((1, 16))
    probability[0, [0, 15]] = [0.4, 0.6]
    np.testing.assert_allclose(marginal_scores(probability), [[0.6, 0.6, 0.6, 0.6]])
    with pytest.raises(ValueError, match="sum to one"):
        marginal_scores(probability / 2)


def test_patient_groups_do_not_merge_unknown_records_or_split_known_patients():
    frame = pd.DataFrame(
        {
            "patient_identity_known": ["True", "True", "False", "False", "True", "True"],
            "source": ["a"] * 6,
            "patient_id": ["1", "1", "", "", "2", "3"],
            "record_id": ["a", "b", "c", "d", "e", "f"],
        }
    )
    group = record_groups(frame)
    assert group[0] == group[1]
    assert group[2] != group[3]
    fold = folds(group)
    assert fold[0] == fold[1]
    for f in range(3):
        assert not set(group[fold == f]) & set(group[fold != f])


def test_reference_resampling_preserves_whole_group_multiplicities():
    group = np.array(["a", "a", "b", "c", "c", "d"])
    family = np.array(["A", "A", "A", "B", "B", "B"])
    rows = resample_references(group, family, 20261206)
    count = np.bincount(rows, minlength=6)
    assert count[0] == count[1]
    assert count[3] == count[4]
    assert count[0] + count[2] == 2
    assert count[3] + count[5] == 2
    np.testing.assert_array_equal(rows, resample_references(group, family, 20261206))


def test_prediction_keeps_training_scaler_fixed_and_aligns_absent_joint_states():
    rng = np.random.default_rng(42)
    states = np.repeat([0, 1, 3, 8], 20)
    raw = rng.normal(size=(80, 6)) + STATE_BITS[states].sum(axis=1)[:, None]
    model = fit_package(raw, STATE_BITS[states], "joint_logistic", 1.0)
    mean = model["scaler"].mean_.copy()
    scores, probability = predict_package(model, rng.normal(size=(10, 6)) + 50)
    np.testing.assert_array_equal(model["scaler"].mean_, mean)
    np.testing.assert_array_equal(probability[:, [2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]], 0)
    np.testing.assert_allclose(scores, probability @ STATE_BITS)


def test_multiclass_metrics_exclude_absent_true_states_from_macro_recall():
    flags = STATE_BITS[[0, 0, 3, 3]]
    probability = np.zeros((4, 16))
    probability[:2, 0] = 1
    probability[2:, 3] = 1
    result = joint_metrics(flags, probability, majority=0)
    assert result["supported_states"] == [0, 3]
    assert result["macro_recall_supported_states"] == 1
    assert result["recall"][1] is None
    assert result["precision"][1] is None
    assert result["f1"][1] is None
    assert result["training_majority_accuracy"] == 0.5


def test_top_two_ties_include_the_same_argmax_as_top_one():
    probability = np.ones((3, 16)) / 16
    result = joint_metrics(STATE_BITS[[0, 0, 0]], probability, majority=0)
    assert result["top_two_accuracy"] >= result["exact_state_accuracy"]
