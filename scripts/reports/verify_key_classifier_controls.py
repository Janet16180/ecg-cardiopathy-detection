"""Verify secondary control directions and paired prototype/scaler consistency."""

import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from ecg_experiment import ROOT
from ecg_experiment.condition_key_inputs_v2 import load_inputs
from ecg_experiment.files import sha256_file
from ecg_experiment.key_classifier_audit import metric_check, transformed
from ecg_experiment.pr74_screening_io import verify_hashes


def main() -> None:
    """Verify registered secondary controls from stored parameters without any new fit."""
    directory = ROOT / "outputs/condition_classifiers_20261106"
    control = ROOT / "outputs/condition_classifiers_negative_control_20261106"
    parent = json.loads((directory / "result.json").read_text())
    result = json.loads((control / "result.json").read_text())
    if result["parent_result_sha256"] != sha256_file(directory / "result.json"):
        raise ValueError("Negative-reference control does not bind the completed parent")
    verify_hashes(ROOT, result["source_sha256"])
    verify_hashes(ROOT, result["artifact_sha256"])
    verify_hashes(ROOT, parent["artifact_sha256"])
    arrays = np.load(control / "artifacts.npz", allow_pickle=False)
    data = load_inputs()
    np.testing.assert_array_equal(arrays["flags"], data["query_flags"])
    final = joblib.load(directory / "final_models.joblib")
    draws = joblib.load(directory / "reference_models.joblib")
    checked = 0
    for value in draws.values():
        for model in value["models"].values():
            for name in ("mean_", "scale_", "var_", "n_samples_seen_"):
                np.testing.assert_array_equal(
                    getattr(value["prototype_scaler"], name), getattr(model["scaler"], name)
                )
                checked += 1
    with threadpool_limits(2):
        scaler = final["joint_logistic"]["scaler"]
        train = transformed(data["train_pooled"], scaler)
        query = transformed(data["query_pooled"], scaler)
        positives, negatives = [], []
        for c in range(4):
            yes = data["train_flags"][:, c] == 1
            positive = train[yes].mean(axis=0)
            negative = train[~yes].mean(axis=0)
            positives.append(positive / np.linalg.norm(positive))
            negatives.append(negative / np.linalg.norm(negative))
        positive, negative = np.stack(positives), np.stack(negatives)
        np.testing.assert_allclose(positive, arrays["positive"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(negative, arrays["negative"], atol=1e-12, rtol=0)
        scores = query @ (positive - negative).T
        error = float(np.max(np.abs(scores - arrays["scores"])))
        np.testing.assert_allclose(scores, arrays["scores"], atol=1e-12, rtol=0)
        checked += metric_check(arrays["flags"], arrays["scores"], result["metrics"])
        checked += metric_check(arrays["flags"], arrays["scores"], result["condition_vs_normal"], True)
    verification = {
        "status": "passed",
        "scaler_and_metric_fields": checked,
        "reference_draws_with_identical_classifier_prototype_scalers": len(draws),
        "negative_control_max_manual_score_error": error,
        "negative_control_sha256": sha256_file(control / "result.json"),
        "script_sha256": sha256_file(Path(__file__)),
        "new_fits": 0,
    }
    path = ROOT / "reports/condition-classifiers-2026-11-06/control-verification.json"
    path.write_text(json.dumps(verification, indent=2) + "\n")
    print(verification)


if __name__ == "__main__":
    main()
