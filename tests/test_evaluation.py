import csv
import json

import numpy as np
import pytest

from ecg_experiment.evaluation import (
    evaluate_predictions,
    metrics,
    paired_comparison,
    partition_validation,
    patient_bootstrap,
    scenario_ppv,
    select_threshold,
)


def _rows(count, seed=0):
    rng = np.random.default_rng(seed)
    targets = rng.integers(0, 2, count)
    targets[:2] = (0, 1)
    return [{"ecg_id": str(100 + index), "patient_id": f"p{index // 2}", "target": str(target)}
            for index, target in enumerate(targets)]


def _logits(rows, seed=1):
    rng = np.random.default_rng(seed)
    return np.array([2.0 * int(row["target"]) - 1 for row in rows]) + rng.normal(0, 1.5, len(rows))


def test_threshold_inclusive_ties_and_impossible_selection():
    assert select_threshold([1, 1, 1, 0], [0.9, 0.5, 0.5, 0.8], 2 / 3) == 0.5
    assert select_threshold([1, 1, 0], [0.9, 0.2, 0.8], 0.5) == 0.9
    with pytest.raises(ValueError, match="both classes"):
        select_threshold([0, 0], [0.1, 0.2])
    with pytest.raises(ValueError, match="finite probabilities"):
        select_threshold([0, 1], [0.1, float("nan")])


def test_metrics_counts_calibration_and_json_safety():
    result = metrics([1, 0, 1, 0], [0.8, 0.7, 0.3, 0.2], 0.5)
    assert (result["tp"], result["tn"], result["fp"], result["fn"]) == (1, 1, 1, 1)
    assert result["sensitivity"] == 0.5
    assert result["specificity"] == 0.5
    assert result["precision"] == 0.5
    assert result["f1"] == 0.5
    assert result["accuracy"] == 0.5
    assert result["brier"] == pytest.approx(0.265)
    assert result["ece_10_bins"] == pytest.approx(0.45)
    json.dumps(result, allow_nan=False)

    none_denominators = metrics([0, 0], [0.0, 0.0], 1.0)
    assert none_denominators["sensitivity"] is None
    assert none_denominators["precision"] is None
    assert none_denominators["auroc"] is None
    assert none_denominators["average_precision"] is None
    assert none_denominators["f1"] is None
    json.dumps(none_denominators, allow_nan=False)


def test_metrics_places_probability_one_in_last_calibration_bin():
    assert metrics([1, 0], [1.0, 0.0], 0.5)["ece_10_bins"] == 0.0


def test_metrics_rejects_threshold_outside_unit_interval():
    with pytest.raises(ValueError, match="threshold must be finite"):
        metrics([0, 1], [0.2, 0.8], 1.5)


def test_patient_bootstrap_deterministic_and_clustered():
    y = [1, 1, 0, 0, 1, 0]
    prob = [0.8, 0.7, 0.2, 0.1, 0.6, 0.3]
    patients = ["a", "a", "b", "b", "c", "c"]
    first = patient_bootstrap(y, prob, patients, 0.5, repeats=100, seed=7)
    second = patient_bootstrap(y, prob, patients, 0.5, repeats=100, seed=7)
    assert first == second
    assert set(first) == {"auroc", "average_precision", "sensitivity", "specificity", "precision", "brier"}
    assert first["sensitivity"] == [1.0, 1.0]
    json.dumps(first, allow_nan=False)

    # Every cluster is one class; single-class resamples are skipped, not scored.
    sparse = patient_bootstrap([1, 0], [0.9, 0.1], ["a", "b"], 0.5, repeats=20, seed=1)
    assert sparse["auroc"] == [1.0, 1.0]


def test_scenario():
    result = scenario_ppv(0.9, 0.95, 0.01)
    assert result["expected_tp_per_1000"] == pytest.approx(9)
    assert result["expected_fp_per_1000"] == pytest.approx(49.5)
    assert result["expected_referrals_per_1000"] == pytest.approx(58.5)
    assert result["ppv"] == pytest.approx(9 / 58.5)
    assert scenario_ppv(0, 1, 0)["ppv"] is None
    with pytest.raises(ValueError, match="prevalence must be finite"):
        scenario_ppv(0.9, 0.9, 2)


def test_partition_validation_is_fixed_and_keeps_patients_together():
    rows = _rows(60)
    development, calibration = partition_validation(rows)
    assert (development, calibration) == partition_validation(rows)
    assert len(development) + len(calibration) == len(rows)
    assert not {r["patient_id"] for r in development} & {r["patient_id"] for r in calibration}


def test_partition_validation_requires_both_classes_in_each_partition():
    rows = [{"ecg_id": str(index), "patient_id": f"p{index}", "target": "0"} for index in range(20)]
    with pytest.raises(ValueError, match="each contain both classes"):
        partition_validation(rows)


def test_evaluate_predictions_writes_consistent_artifacts(tmp_path):
    calibration, test = _rows(80, seed=2), _rows(40, seed=3)
    calibration_logits, test_logits = _logits(calibration), _logits(test, seed=4)
    result = evaluate_predictions("model", calibration_logits, test_logits, calibration, test,
                                  tmp_path / "out", seed=42, bootstrap=20)

    written = (tmp_path / "out" / "metrics.json").read_text()
    assert written == json.dumps(result, indent=2, allow_nan=False) + "\n"
    assert result["model"] == "model"
    assert result["label_seed"] == 42
    assert result["calibration"]["records"] == 80
    assert result["calibration"]["slope"] > 0

    with (tmp_path / "out" / "test_predictions.csv").open(newline="") as handle:
        predictions = list(csv.DictReader(handle))
    assert [row["ecg_id"] for row in predictions] == [row["ecg_id"] for row in test]
    assert [float(row["raw_logit"]) for row in predictions] == test_logits.tolist()
    predicted = [int(row["prediction"]) for row in predictions]
    expected = [int(float(row["probability"]) >= result["threshold"]) for row in predictions]
    assert predicted == expected
    assert sum(predicted) == result["test"]["tp"] + result["test"]["fp"]

    arrays = np.load(tmp_path / "out" / "calibration_predictions.npz")
    assert np.array_equal(arrays["logits"], calibration_logits)
    assert arrays["ecg_ids"].tolist() == [int(row["ecg_id"]) for row in calibration]
    assert select_threshold(arrays["targets"], arrays["probabilities"]) == result["threshold"]


def test_evaluate_predictions_rejects_inverted_scores_before_writing(tmp_path):
    calibration, test = _rows(80, seed=2), _rows(40, seed=3)
    inverted = -10 * np.array([int(row["target"]) for row in calibration], dtype=float)
    with pytest.raises(RuntimeError, match="Nonpositive calibration slope"):
        evaluate_predictions("model", inverted, _logits(test), calibration, test, tmp_path / "out", 42)
    assert not (tmp_path / "out").exists()


def test_paired_comparison_of_a_model_with_itself_is_zero(tmp_path):
    calibration, test = _rows(80, seed=2), _rows(40, seed=3)
    for name in ("left", "right"):
        evaluate_predictions(name, _logits(calibration), _logits(test, seed=4), calibration, test,
                             tmp_path / name, seed=42, bootstrap=5)
    result = paired_comparison(tmp_path / "left", tmp_path / "right", repeats=20)
    assert set(result) == {"auroc", "average_precision", "sensitivity", "specificity"}
    assert all(value == {"difference": 0.0, "ci95": [0.0, 0.0]} for value in result.values())
