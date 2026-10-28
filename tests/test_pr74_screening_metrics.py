"""Protect strict budget ties, confusion totals, role boundaries and immutable receipt completion."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecg_experiment import pr74_screening_io as artifact_io
from ecg_experiment.files import sha256_file
from ecg_experiment.pr74_screening_features import read_training_signal
from ecg_experiment.pr74_screening_metrics import budget_threshold, matrix, point_metrics, threshold_sweep


def test_strict_budget_threshold_uses_only_normals_and_does_not_refer_ties():
    normals = np.array([0.0] * 90 + [0.5] * 10)
    first = budget_threshold(np.r_[np.zeros(100), 1], np.r_[normals, 0.1])
    second = budget_threshold(np.r_[np.zeros(100), 1], np.r_[normals, 1.0])
    assert first == second
    assert first["threshold"] == 0.5
    assert first["flagged_normals"] == 0
    assert first["allowed_normals"] == 5
    assert first["tied_normals"] == 10


def test_budget_boundary_is_exact_order_statistic_not_evaluation_sensitivity():
    result = budget_threshold(np.zeros(100), np.arange(100) / 100)
    assert result["threshold"] == 0.94
    assert result["flagged_normals"] == 5
    with pytest.raises(ValueError, match="calibration"):
        budget_threshold(np.ones(3), np.ones(3))


def test_confusion_matrix_counts_every_record_and_zero_precision_denominator_is_null():
    y, score = np.array([1, 1, 0, 0]), np.array([0.9, 0.5, 0.5, 0.1])
    result = matrix(y, score, 0.5)
    assert {key: result[key] for key in ("tp", "tn", "fp", "fn")} == {"tp": 1, "tn": 2, "fp": 0, "fn": 1}
    assert sum(result[key] for key in ("tp", "tn", "fp", "fn")) == len(y)
    assert matrix(y, score, 1.0)["precision"] is None
    assert point_metrics(np.zeros(2), np.zeros(2), 0.0)["auroc"] is None


def test_ordered_sweep_is_monotone_preserves_ties_and_includes_frozen_point():
    y, score = np.array([1, 0, 1, 0, 1]), np.array([0.1, 0.2, 0.2, 0.7, 0.9])
    sweep = threshold_sweep(y, score, 0.5)
    assert np.all(np.diff(sweep["threshold"]) > 0)
    assert 0.5 in sweep["threshold"]
    assert np.all(np.diff(sweep["tp"]) <= 0)
    assert np.all(np.diff(sweep["fp"]) <= 0)
    for i in range(len(sweep["threshold"])):
        assert sweep["tp"][i] + sweep["fn"][i] == 3
        assert sweep["tn"][i] + sweep["fp"][i] == 2
    assert sweep["tp"][0] == 3
    assert sweep["fp"][0] == 2


@pytest.mark.parametrize(("y", "scores"), [([0, 1], [0.0]), ([0, 2], [0.1, 0.2]), ([0, 1], [0, np.nan])])
def test_malformed_scores_cannot_produce_a_confusion_matrix(y, scores):
    with pytest.raises(ValueError, match="aligned|Undefined|nonfinite"):
        matrix(np.array(y), np.array(scores), 0.5)


@pytest.mark.parametrize("split", ["test", "calibration", "val"])
def test_no_waveform_is_opened_for_a_nontraining_feature_pointer(split, monkeypatch):
    from ecg_experiment import pr74_screening_features as features

    def forbidden(*args):
        pytest.fail("A nontraining row reached raw decoding")

    monkeypatch.setattr(features, "raw_pair", forbidden)
    row = pd.Series({"split": split, "samples": "5000", "window_start": "0"})
    with pytest.raises(ValueError, match="training windows"):
        read_training_signal(Path("/unused"), row)


def test_mutated_scientific_source_prevents_completed_receipt(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.py").write_text("initial recipe\n")
    monkeypatch.setattr(artifact_io, "ROOT", source)
    receipt = {"source_sha256": {"model.py": sha256_file(source / "model.py")}}
    stage = tmp_path / "outputs/experiment/profile"
    stage.mkdir(parents=True)
    (stage / "profile.json").write_text("{}")
    (source / "model.py").write_text("changed recipe\n")
    with pytest.raises(ValueError, match="Artifact changed"):
        artifact_io.finish(tmp_path, Path("outputs/experiment"), "profile", receipt, {}, {})
    assert not (stage / "completion.json").exists()
