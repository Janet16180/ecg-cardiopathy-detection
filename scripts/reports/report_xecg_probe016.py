"""Summarize the completed Experiment 016 development-only screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from ecg_experiment.evaluation import partition_validation, select_threshold
from ecg_experiment.files import read_csv, sha256_file, write_json_atomic, write_text_atomic

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs/experiment016_xecg_probe_finetune"
DEFAULT_MANIFEST = ROOT / "data/processed/ptbxl/seed42_fraction1"


def _development_rows(manifest: Path) -> list[dict[str, str]]:
    """Load the fixed development partition without reading held-out labels."""
    development, _ = partition_validation(read_csv(manifest / "validation.csv"))
    if len(development) != 1306:
        raise ValueError("Experiment 016 development partition changed")
    return development


def _load_arm_logits(directory: Path, expected_ids: np.ndarray) -> np.ndarray:
    """Verify completed arm artifacts and return ordered development logits."""
    completion = json.loads((directory / "complete.json").read_text())
    for name, expected in completion["sha256"].items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f"Altered Experiment 016 arm artifact: {directory / name}")
    with np.load(directory / "development_logits.npz") as saved:
        if not np.array_equal(saved["ecg_ids"], expected_ids):
            raise ValueError("Experiment 016 arm development order differs")
        logits = saved["logits"].astype(np.float64)
    if logits.shape != expected_ids.shape or not np.isfinite(logits).all():
        raise ValueError("Experiment 016 arm logits are malformed")
    return logits


def _load_probe_logits(output: Path, train_records: int, expected_ids: np.ndarray) -> np.ndarray:
    """Apply the selected frozen probe to the saved development features."""
    receipt = json.loads((output / "probe.json").read_text())
    if sha256_file(output / "probe.npz") != receipt["probe_sha256"]:
        raise ValueError("Experiment 016 probe changed")
    ids = np.load(output / "features/ecg_ids.npy")
    if not np.array_equal(ids[train_records:], expected_ids):
        raise ValueError("Experiment 016 probe development identities differ")
    features = np.load(output / "features/features.npy", mmap_mode="r")[train_records:]
    with np.load(output / "probe.npz") as probe:
        logits = ((features.astype(np.float64) - probe["mean"]) / probe["scale"])
        logits = logits @ probe["coefficient"] + float(probe["intercept"])
    if logits.shape != expected_ids.shape or not np.isfinite(logits).all():
        raise ValueError("Experiment 016 probe logits are malformed")
    return logits


def _fold_operating_point(y: np.ndarray, logits: np.ndarray, patients: np.ndarray) -> dict[str, float]:
    """Estimate a 95%-sensitivity threshold on other development patient folds."""
    probabilities = expit(logits)
    predictions = np.zeros(len(y), dtype=bool)
    for fit_indices, held_indices in GroupKFold(n_splits=5).split(logits, y, groups=patients):
        threshold = select_threshold(y[fit_indices], probabilities[fit_indices], 0.95)
        predictions[held_indices] = probabilities[held_indices] >= threshold
    positive, negative = y == 1, y == 0
    return {"cross_fold_sensitivity": float(predictions[positive].mean()),
            "cross_fold_specificity": float((~predictions[negative]).mean())}


def _paired_auc_interval(y: np.ndarray, left: np.ndarray, right: np.ndarray,
                         patients: np.ndarray) -> list[float]:
    """Bootstrap paired patient clusters for an exploratory AUROC difference."""
    unique = np.unique(patients)
    groups = [np.flatnonzero(patients == patient) for patient in unique]
    rng = np.random.default_rng(2026)
    differences = []
    for _ in range(500):
        indices = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        if len(np.unique(y[indices])) == 2:
            differences.append(roc_auc_score(y[indices], left[indices])
                               - roc_auc_score(y[indices], right[indices]))
    return [float(value) for value in np.quantile(differences, [0.025, 0.975])]


def summarize(output: Path, manifest: Path) -> dict[str, Any]:
    """Compute fixed development metrics and write a small aggregate report."""
    rows = _development_rows(manifest)
    ids = np.array([int(row["ecg_id"]) for row in rows])
    y = np.array([int(row["target"]) for row in rows])
    patients = np.array([row["patient_id"] for row in rows])
    train_records = len(read_csv(manifest / "labeled_train.csv"))
    logits = {"frozen_probe": _load_probe_logits(output, train_records, ids),
              "random_head": _load_arm_logits(output / "random_head", ids),
              "probe_head": _load_arm_logits(output / "probe_head", ids)}
    metrics = {}
    for name, scores in logits.items():
        metrics[name] = {"development_auroc": float(roc_auc_score(y, scores)),
                         "development_average_precision": float(average_precision_score(y, scores)),
                         **_fold_operating_point(y, scores, patients)}
    difference = metrics["probe_head"]["development_auroc"] - metrics["random_head"]["development_auroc"]
    comparison = {"probe_minus_random_auroc": difference,
                  "paired_patient_bootstrap_95pct": _paired_auc_interval(
                      y, logits["probe_head"], logits["random_head"], patients)}
    profile = json.loads((output / "profile.json").read_text())
    probe = json.loads((output / "probe.json").read_text())
    extraction = json.loads((output / "features/receipt.json").read_text())
    diagnostic_path = output / "initial_mode_diagnostic.json"
    diagnostic = json.loads(diagnostic_path.read_text()) if diagnostic_path.is_file() else None
    cost_gate = json.loads((output / "cost_gate.json").read_text())
    arm_configs = {name: json.loads((output / name / "config.json").read_text())
                   for name in ("random_head", "probe_head")}
    result = {"records": len(rows), "selected_C": probe["selected_C"],
              "metrics": metrics, "comparison": comparison,
              "extraction_seconds": extraction["seconds"], "probe_fit_seconds": probe["seconds"],
              "profile_pass_seconds": [row["seconds"] for row in profile["passes"]],
              "projected_total_seconds": cost_gate["projected_total_seconds"],
              "arm_configs": arm_configs,
              "initial_mode_diagnostic": diagnostic,
              "runner_sha256": sha256_file(ROOT / "scripts/experiments/run_xecg_probe_finetune016.py"),
              "analysis_source_sha256": sha256_file(Path(__file__)),
              "manifest_sha256": sha256_file(ROOT / "outputs/experiment_queue_016/queue.json")}
    write_json_atomic(output / "analysis.json", result)
    lines = ["# Experiment 016: xECG probe-initialized fine-tuning",
             "", "Full-label, seed-42 development screen; 15,360 training labels and 1,306 development"
             " records.",
             "No calibration or test labels were used.", "",
             "| Arm | Development AUROC | Average precision | Cross-fold sensitivity |"
             " Cross-fold specificity |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name in ("frozen_probe", "random_head", "probe_head"):
        row = metrics[name]
        lines.append(f"| {name.replace('_', ' ')} | {row['development_auroc']:.4f} | "
                     f"{row['development_average_precision']:.4f} | "
                     f"{row['cross_fold_sensitivity']:.3f} | {row['cross_fold_specificity']:.3f} |")
    interval = comparison["paired_patient_bootstrap_95pct"]
    lines += ["", f"Probe-head minus random-head AUROC: {difference:+.4f} "
              f"(paired patient-bootstrap 95% interval {interval[0]:+.4f} to {interval[1]:+.4f}).",
              f"The frozen probe selected C={probe['selected_C']}. Feature extraction took "
              f"{extraction['seconds']:.1f} s; probe fitting took {probe['seconds']:.1f} s.",
              f"Complete GPU profile passes took {profile['passes'][0]['seconds']:.1f} and "
              f"{profile['passes'][1]['seconds']:.1f} s. The two-hour gate projected "
              f"{cost_gate['projected_total_seconds']:.1f} s for the full suite.",
              f"Random-head fine-tuning selected epoch {arm_configs['random_head']['best_epoch']} "
              f"of 2 in {arm_configs['random_head']['elapsed_seconds']:.1f} s; probe-head "
              f"selected epoch {arm_configs['probe_head']['best_epoch']} of 2 in "
              f"{arm_configs['probe_head']['elapsed_seconds']:.1f} s.",
              "This one-seed development comparison tests the transfer recipe only.",
              "The endpoint is an ECG annotation proxy, not confirmed health or a referral decision.", ""]
    if diagnostic is not None:
        lines += ["On a fixed balanced sample of 128 labeled training ECGs, the unmodified probe head "
                  f"had BCE {diagnostic['probe_eval_bce']:.3f} in evaluation mode versus "
                  f"{diagnostic['probe_train_mode_bce']:.3f} in training mode with drop path 0.5. "
                  "This large initial mode shift is consistent with the observed fine-tuning decline; "
                  "it does not isolate drop path from later optimizer effects.", ""]
    write_text_atomic(output / "report.md", "\n".join(lines))
    return result


def main() -> None:
    """Read completed Experiment 016 artifacts and emit aggregate analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    result = summarize(args.output_dir, args.manifest_dir)
    print(json.dumps({"comparison": result["comparison"], "metrics": result["metrics"]}), flush=True)


if __name__ == "__main__":
    main()
