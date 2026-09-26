"""Independently verify the saved 25k development-only readout."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ecg_experiment.files import sha256_file, write_json_atomic

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/experiment019_cpc_25k_readout_v1"
PRIOR = ROOT / "outputs/experiment018_cpc_data_scaling_readout_v1"


def check(condition: bool, message: str) -> None:
    """Reject inconsistent persisted readout artifacts."""
    if not condition:
        raise ValueError(message)


def main() -> None:
    """Rehash features/predictions and reproduce saved development metrics."""
    result_path = OUTPUT / "result.json"
    result = json.loads(result_path.read_text())
    check(result["status"] == "complete_development_only", "Incomplete readout")
    check(result["calibration_test_evaluated"] is False, "Unexpected held-out evaluation")
    check(sha256_file(OUTPUT / "profile.json") == result["profile_sha256"],
          "Profile changed")
    features_path = OUTPUT / "features.npy"
    check(sha256_file(features_path) == result["features_sha256"], "Features changed")
    features = np.load(features_path, mmap_mode="r")
    check(features.shape == (15359 + 1306, 512), "Feature dimensions changed")
    check(bool(np.isfinite(features).all()), "Nonfinite features")

    predictions_path = OUTPUT / "development_predictions.npz"
    check(sha256_file(predictions_path) == result["predictions_sha256"],
          "Predictions changed")
    with np.load(predictions_path) as saved, np.load(PRIOR / "development_predictions.npz") as prior:
        for key in ("record_ids", "patient_ids", "targets"):
            check(np.array_equal(saved[key], prior[key]), f"{key} differs from comparator")
        targets = saved["targets"]
        check(len(set(saved["record_ids"])) == 1306, "Development record count")
        check(len(set(saved["patient_ids"])) == 1173, "Development patient count")
        for budget, score in result["scores_25k"].items():
            probabilities = saved[f"{budget}_25k"]
            check(probabilities.shape == targets.shape, f"{budget} prediction shape")
            check(bool(np.isfinite(probabilities).all()), f"{budget} nonfinite predictions")
            check(abs(roc_auc_score(targets, probabilities) - score["auroc"]) < 1e-12,
                  f"{budget} AUROC mismatch")
            check(abs(average_precision_score(targets, probabilities)
                      - score["average_precision"]) < 1e-12,
                  f"{budget} average-precision mismatch")
            for label, comparator in (("25k_minus_115k", "new"),
                                      ("25k_minus_initial", "initial")):
                contrast = result["contrasts"][budget][label]
                difference = (score["auroc"]
                              - roc_auc_score(targets, prior[f"{budget}_{comparator}"]))
                check(abs(difference - contrast["difference"]) < 1e-12,
                      f"{budget}/{label} contrast mismatch")
                check(contrast["valid_draws"] == 2000 and contrast["invalid_draws"] == 0,
                      f"{budget}/{label} bootstrap draws")

    audit = {
        "status": "passed_development_only",
        "result_sha256": sha256_file(result_path),
        "features_sha256": result["features_sha256"],
        "predictions_sha256": result["predictions_sha256"],
        "calibration_test_evaluated": False,
    }
    write_json_atomic(OUTPUT / "readout_audit.json", audit)
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
