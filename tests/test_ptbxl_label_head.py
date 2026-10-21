import numpy as np
import pandas as pd
import pytest

from ecg_experiment.ptbxl_label_head import (
    choose_regularization, diagnosis_names, fit_head, head_scores, label_matrix, load_embeddings,
    normal_bank_scores, only_statements, reference_rows,
)


def test_labels_and_normal_only_records_come_from_scp_codes():
    metadata = pd.DataFrame({"scp_codes": ["{'NORM': 100.0, 'SR': 0.0}", "{'AFIB': 0.0}", "{'NORM': 80.0, 'PVC': 0.0}"]})
    assert label_matrix(metadata, ("NORM", "PVC")).tolist() == [[True, False], [False, False], [True, True]]
    assert only_statements(metadata, ("NORM", "SR"), "NORM").tolist() == [True, False, False]
    assert diagnosis_names(("AFIB", "XYZ")) == ["Fibrilación auricular", "XYZ"]


def test_reference_rows_reproduce_the_seeded_draw_and_keep_all_without_size():
    train = np.arange(100, 200)
    assert np.array_equal(reference_rows(train, 10, 47), np.sort(np.random.default_rng(47).choice(train, 10, replace=False)))
    assert np.array_equal(reference_rows(train, None, 47), train)


def test_load_embeddings_refuses_another_order(tmp_path):
    path = tmp_path / "embeddings.npz"
    np.savez(path, ecg_id=np.array([1, 2, 3]), encoder=np.zeros((3, 4)))
    assert load_embeddings(path, np.array([1, 2, 3]), ("encoder",))["encoder"].shape == (3, 4)
    with pytest.raises(ValueError):
        load_embeddings(path, np.array([3, 2, 1]), ("encoder",))


def test_head_learns_a_separable_label_and_regularization_table_has_every_candidate():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(400, 5))
    labels = np.column_stack([features[:, 0] > 0, features[:, 1] > 0])
    scores = head_scores(fit_head(features[:300], labels[:300], 1.0), features[300:])
    assert scores.shape == (100, 2) and ((scores[:, 0] > 0.5) == labels[300:, 0]).mean() > 0.9
    table = choose_regularization(features[:300], labels[:300], features[300:], labels[300:], (0.01, 1.0))
    assert list(table.index) == [0.01, 1.0] and table.max() > 0.9


def test_normal_bank_scores_are_zero_for_a_normal_reference_and_grow_away_from_it():
    normal = np.array([[1.0, 0.0], [0.9, 0.1]])
    query = np.array([[1.0, 0.0], [-1.0, 0.0]])
    scores = normal_bank_scores(query, normal, np.zeros(2), k=1)
    assert scores[0] == pytest.approx(0) and scores[1] > 1.5
