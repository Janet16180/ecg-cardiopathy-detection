"""Independently verify the completed full-backbone comparison without model deserialization."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file, write_json_atomic


def check_point(row: dict[str, Any], predictions: Any) -> int:
    """Reconstruct a strict calibration cutoff, integer matrix and ranking metrics independently."""
    y, held = predictions["target"], predictions["evaluation"]
    scores = predictions[row["model"]]
    normal = np.sort(scores[~held & (y == 0)])
    allowed = int(np.floor(len(normal) * 0.05))
    cutoff = float(normal[-allowed - 1])
    if cutoff != row["threshold"]:
        raise ValueError("Fine-tuning cutoff is not its frozen normal-only budget cutoff")
    flagged, positive = scores[held] > cutoff, y[held] == 1
    tp, fp = int(np.count_nonzero(flagged & positive)), int(np.count_nonzero(flagged & ~positive))
    fn, tn = int(np.count_nonzero(~flagged & positive)), int(np.count_nonzero(~flagged & ~positive))
    points = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "ecgs": len(positive),
              "positives": tp + fn, "negatives": tn + fp, "sensitivity": tp / (tp + fn),
              "specificity": tn / (tn + fp), "precision": tp / (tp + fp),
              "referral_fraction": (tp + fp) / len(positive), "auroc": roc_auc_score(y[held], scores[held]),
              "average_precision": average_precision_score(y[held], scores[held])}
    for key, value in points.items():
        if abs(value - row[key]) > 1e-12:
            raise ValueError(f"Independent fine-tuning mismatch: {row['site']}/{row['model']}/{key}")
    return len(points) + 1


def check_contrast(row: dict[str, Any], contrast: dict[str, Any]) -> int:
    """Verify point differences and the exact predeclared practical decision rule."""
    reference = contrast["control_metrics"]
    difference = row["sensitivity"] - reference["sensitivity"]
    delta = contrast["sensitivity_difference"]
    increase = reference["specificity"] - row["specificity"]
    if abs(delta["difference"] - difference) > 1e-12:
        raise ValueError("Paired contrast's point value differs from its integer matrices")
    if abs(contrast["normal_referral_increase"] - increase) > 1e-12:
        raise ValueError("Normal-referral change differs from its integer matrices")
    passed = difference >= 0.02 and delta["ci_low"] > 0 and increase <= 0.005
    if passed != contrast["practical_improvement"]:
        raise ValueError("Fine-tuning decision disagrees with its prespecified gate")
    return 3


def main() -> None:
    """Audit all completed points, captured source identities and the exact saved tensor artifact."""
    directory = ROOT / "reports/avance5_finetune_v2"
    result = json.loads((directory / "results.json").read_text())
    if result.get("complete") is not True or result["trained_records"] != 25000:
        raise ValueError("The full-cohort study has not completed")
    checks = 1
    for name, expected in result["source_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError("Scientific source changed after training")
        checks += 1
    if sha256_file(ROOT / "outputs/avance5_finetune_v2/final.safetensors") != result["checkpoint_sha256"]:
        raise ValueError("Saved fine-tuned checkpoint changed")
    checks += 1
    for site in ("ptb", "sph"):
        path = ROOT / "outputs/avance5_finetune_v2" / f"{site}_predictions.npz"
        with np.load(path, allow_pickle=False) as predictions:
            rows = [row for row in result["records"] if row["site"] == site]
            for row in rows:
                checks += check_point(row, predictions)
                checks += check_contrast(row, result["contrasts"][site + "/" + row["model"]])
    verification = {"passed": True, "independent_checks": checks, "model_site_comparisons": 4,
                    "weights_deserialized": False, "predictions_recomputed": False,
                    "patient_bootstrap_repeated": False, "waveform_read": False}
    write_json_atomic(directory / "verification.json", verification)
    print(verification)


if __name__ == "__main__":
    main()
