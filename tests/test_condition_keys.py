"""Scientific invariants for cached condition-key construction and scoring."""

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from ecg_experiment.condition_key_inputs import pooling_check, positions
from ecg_experiment.condition_keys import (
    macro_interval,
    region_scores,
    region_vectors,
    selected_vectors,
    spans,
    unit,
    whole_bank,
)


def test_spans_are_complete_and_in_bounds():
    windows = spans()
    assert windows.shape == (109, 2)
    assert len(set(map(tuple, windows))) == 109
    assert windows.min() == 0
    assert windows.max() == 40
    assert np.array_equal(spans(True), np.column_stack([np.arange(0, 40, 4), np.arange(4, 41, 4)]))


def test_regions_pool_before_normalizing_and_use_training_scaler():
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(3, 40, 6))
    scaler = StandardScaler().fit(rng.normal(size=(20, 6)))
    actual = region_vectors(raw, scaler, np.array([[0, 4], [7, 15]]))
    expected = np.stack([unit(scaler.transform(raw[:, a:b].mean(axis=1)))
                         for a, b in [(0, 4), (7, 15)]], axis=1)
    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_positive_and_normal_support_select_opposite_margin_extrema():
    z = np.array([[[1., 0.], [-1., 0.]], [[1., 0.], [-1., 0.]]])
    bank = {"positive": np.tile([1., 0.], (4, 1)), "normal": np.tile([-1., 0.], (4, 1))}
    selected, choices = selected_vectors(z, np.array([[1, 1, 1, 1], [0, 0, 0, 0]]), bank)
    np.testing.assert_array_equal(choices, [[0] * 4, [1] * 4])
    np.testing.assert_array_equal(selected[:, 0], [[1., 0.], [-1., 0.]])


def test_query_selection_needs_no_labels_and_uses_first_tie():
    z = np.array([[[1., 0.], [1., 0.], [-1., 0.]]])
    bank = {"positive": np.tile([1., 0.], (4, 1)), "normal": np.tile([-1., 0.], (4, 1))}
    scores, choices = region_scores(z, bank)
    np.testing.assert_array_equal(scores, [[2.] * 4])
    np.testing.assert_array_equal(choices, [[0] * 4])


def test_multilabel_support_includes_each_group_without_becoming_normal():
    rng = np.random.default_rng(9)
    vectors = rng.normal(size=(8, 6))
    flags = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0],
                      [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    scaler, bank = whole_bank(vectors, flags)
    z = unit(scaler.transform(vectors))
    np.testing.assert_allclose(bank["positive"][0], unit(z[[2, 5]].mean(axis=0)))
    np.testing.assert_allclose(bank["normal"][0], unit(z[:2].mean(axis=0)))


def test_pooling_mismatch_stops_instead_of_dropping_records():
    tokens = np.ones((3, 40, 4), dtype=np.float16)
    assert pooling_check(tokens, np.arange(3), np.ones((3, 4)))["rows"] == 3
    with pytest.raises(ValueError, match="pooling/padding"):
        pooling_check(tokens, np.arange(3), np.full((3, 4), 2.))


def test_alignment_rejects_missing_and_duplicate_identifiers():
    import pandas as pd

    np.testing.assert_array_equal(positions(pd.Series(["b", "a"]), pd.Series(["a", "b"])), [1, 0])
    with pytest.raises(ValueError, match="missing"):
        positions(pd.Series(["a"]), pd.Series(["c"]))
    with pytest.raises(ValueError, match="Duplicate"):
        positions(pd.Series(["a", "a"]), pd.Series(["a"]))


def test_patient_bootstrap_identical_scores_have_exact_zero_difference():
    rng = np.random.default_rng(2)
    flags = np.tile(np.eye(4, dtype=int), (8, 1))
    scores = rng.normal(size=flags.shape)
    result = macro_interval(np.repeat(np.arange(16), 2), flags, scores, scores, draws=30)
    assert result["difference"] == result["ci_low"] == result["ci_high"] == 0
    assert result["draws"] - result["skipped_draws"] > 0
