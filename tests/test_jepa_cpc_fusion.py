"""Patient-aligned JEPA/CPC fusion: joins, train-only normalization, cross-fitting and the gate."""

import json

import numpy as np
import pytest

from scripts.experiments import run_jepa_cpc_fusion
from scripts.experiments.run_jepa_cpc_fusion import (
    bootstrap_screen,
    check_alignment,
    crossfit_has_both_classes,
    dev_score,
    development_folds,
    gate,
    normalized_train_logits,
    select,
)


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


def test_bootstrap_skips_draws_where_a_fold_lacks_a_class():
    rng = np.random.default_rng(1)
    y = (rng.random(60) < 0.3).astype(int)
    y[:2] = [0, 1]
    patients = [f"p{i // 2}" for i in range(60)]
    fold = development_folds(y, patients)
    je, cp = rng.normal(size=60) + y, rng.normal(size=60)
    scores = {a: dev_score(y, a * je + (1 - a) * cp, fold) for a in (0., .25, .5, .75, 1.)}
    boot = bootstrap_screen(y, je, cp, patients, fold, select(scores))
    assert 0 < boot["valid_draws"] < 300
    assert sum(boot["selected_weight_counts"].values()) == boot["valid_draws"]


def test_crossfit_class_check_matches_dev_score_requirements():
    y = np.array([0, 1, 0, 1, 0, 1])
    assert crossfit_has_both_classes(y, np.array([0, 0, 1, 1, 2, 2]))
    assert not crossfit_has_both_classes(y, np.array([0, 1, 0, 1, 0, 1]))
    assert not crossfit_has_both_classes(np.zeros(4, dtype=int), np.array([0, 0, 1, 1]))
    with pytest.raises(ValueError, match="lacks a class"):
        dev_score(y, np.arange(6.), np.array([0, 1, 0, 1, 0, 1]))


def test_failed_run_is_recorded_in_run_status(tmp_path, monkeypatch):
    def fail(output_dir, start):
        raise ValueError("synthetic failure")

    monkeypatch.setattr(run_jepa_cpc_fusion, "screen_budgets", fail)
    with pytest.raises(ValueError, match="synthetic failure"):
        run_jepa_cpc_fusion.main(["--output-dir", str(tmp_path)])
    status = json.loads((tmp_path / "run_status.json").read_text())
    assert (status["state"], status["reason"]) == ("failed", "synthetic failure")
    assert not (tmp_path / "coordination.json").exists()
