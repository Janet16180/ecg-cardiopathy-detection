import numpy as np
import pytest

from scripts.experiments.run_jepa_cpc_fusion import (check_alignment, dev_score, development_folds, gate,
                                          normalized_train_logits, select)


def test_join_checks_patient_and_split_not_row_order():
    train = {"ecg_id": "2", "patient_id": "B", "target": "1"}
    dev = {"ecg_id": "1", "patient_id": "A", "target": "0"}
    rows = [{"ecg_id": "1", "patient_id": "A", "split": "validation"},
            {"ecg_id": "2", "patient_id": "B", "split": "train"}]
    index = check_alignment({"train": [train], "development": [dev]}, {1: 1, 2: 0}, rows, "synthetic")
    assert index[2][0] == 1
    with pytest.raises(ValueError, match="patient/split"):
        check_alignment({"train": [train], "development": [dev]}, {1: 1, 2: 0},
                        [{**rows[0], "patient_id": "wrong"}, rows[1]], "synthetic")
    with pytest.raises(ValueError, match="Patient overlap"):
        check_alignment({"train": [train], "development": [{**dev, "patient_id": "B"}]},
                        {1: 1, 2: 0}, rows, "synthetic")


def test_train_only_logit_statistics():
    norm, stats = normalized_train_logits([1., 3., 100.], 2)
    assert stats == {"mean": 2., "std": 1., "training_records": 2}
    assert np.allclose(norm, [-1., 1., 98.])


def test_dev_threshold_inclusive_ties_and_selection_gate():
    score = dev_score(np.array([1, 1, 0, 0]), np.array([.4, .4, .4, .1]), np.array([0, 1, 0, 1]))
    assert set(score["crossfit_thresholds"].values()) == {.4}
    assert score["sensitivity"] == 1.
    assert score["specificity"] == .5
    scores = {a: {"specificity": (.5 if a in (0., 1.) else .54),
                  "auroc": .9, "threshold": 0., "sensitivity": .95} for a in (0., .25, .5, .75, 1.)}
    assert select(scores) == .5
    boot = {"interior_selection_fraction": .9, "point_weight_positive_gain_fraction": .9}
    assert gate(scores, .5, boot)["pass"]
    assert not gate(scores, 0., boot)["pass"]


def test_patient_folds_keep_groups_together():
    y = np.array([0, 0, 1, 1] * 10)
    patients = [f"p{i // 2}" for i in range(len(y))]
    fold = development_folds(y, patients)
    assert len(set(fold)) == 5
    for patient in set(patients):
        assert len(set(fold[np.array(patients) == patient])) == 1
