"""Verify the saved development-only CPC scaling readout artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ecg_experiment.files import sha256_file, write_json_atomic

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/experiment018_cpc_data_scaling_readout_v1"


def check(condition: bool, message: str) -> None:
    """Reject a failed artifact invariant with a useful message."""
    if not condition:
        raise ValueError(message)


def main() -> None:
    """Recompute saved metrics and check local artifact identity."""
    result_path = OUTPUT / "result.json"
    result = json.loads(result_path.read_text())
    check(result["status"] == "complete_development_only", "Incomplete readout")
    check(result["calibration_test_evaluated"] is False, "Unexpected held-out evaluation")
    check(sha256_file(OUTPUT / "profile.json") == result["profile_sha256"],
          "Profile receipt changed")

    for arm, expected in result["feature_sha256"].items():
        path = OUTPUT / f"{arm}_features.npy"
        check(sha256_file(path) == expected, f"{arm} features changed")
        features = np.load(path, mmap_mode="r")
        check(features.shape == (15359 + 1306, 512), f"{arm} feature shape")
        check(bool(np.isfinite(features).all()), f"{arm} nonfinite features")

    predictions_path = OUTPUT / "development_predictions.npz"
    check(sha256_file(predictions_path) == result["development_predictions_sha256"],
          "Predictions changed")
    with np.load(predictions_path) as predictions:
        labels = predictions["targets"]
        check(labels.shape == (1306,), "Development label shape")
        check(set(np.unique(labels)) == {0, 1}, "Development classes")
        check(len(set(predictions["record_ids"])) == 1306, "Development record count")
        check(len(set(predictions["patient_ids"])) == 1173, "Development patient count")
        expected_keys = {
            "record_ids", "patient_ids", "targets",
            *(f"{budget}_{arm}" for budget in ("full", "limited")
              for arm in ("initial", "old", "new")),
        }
        check(set(predictions.files) == expected_keys, "Prediction array keys")
        for budget, arms in result["scores"].items():
            for arm, score in arms.items():
                probabilities = predictions[f"{budget}_{arm}"]
                check(probabilities.shape == labels.shape, f"{budget}/{arm} prediction shape")
                check(bool(np.isfinite(probabilities).all()), f"{budget}/{arm} nonfinite score")
                check(bool(((probabilities >= 0) & (probabilities <= 1)).all()),
                      f"{budget}/{arm} score range")
                check(abs(roc_auc_score(labels, probabilities) - score["auroc"]) < 1e-12,
                      f"{budget}/{arm} AUROC mismatch")
                check(abs(average_precision_score(labels, probabilities)
                          - score["average_precision"]) < 1e-12,
                      f"{budget}/{arm} average precision mismatch")
            for name, contrast in result["contrasts"][budget].items():
                other = "old" if name == "new_minus_old" else "initial"
                difference = arms["new"]["auroc"] - arms[other]["auroc"]
                check(abs(difference - contrast["difference"]) < 1e-12,
                      f"{budget}/{name} AUROC difference mismatch")
                check(contrast["valid_draws"] == 2000, f"{budget}/{name} bootstrap count")
                check(contrast["invalid_draws"] == 0, f"{budget}/{name} invalid draws")

    audit = {
        "status": "passed_development_only",
        "result_sha256": sha256_file(result_path),
        "predictions_sha256": result["development_predictions_sha256"],
        "feature_sha256": result["feature_sha256"],
        "checked_budgets": ["full", "limited"],
        "checked_encoders": ["initial", "old", "new"],
        "calibration_test_evaluated": False,
    }
    write_json_atomic(OUTPUT / "readout_audit.json", audit)
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
