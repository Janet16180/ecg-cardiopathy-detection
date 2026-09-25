"""Development-only repeat of the Experiment 014 fusion grid on clean probes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from .clean_cached_probes import (
    COUNTS,
    FRACTIONS,
    PREFLIGHT,
    PTB,
    ROOT,
    aligned_features,
    input_hashes,
    validate_selection,
    verify_preflight,
)
from .evaluation import partition_validation
from .files import read_csv, sha256_file, write_json_atomic

GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
FOLD_SEED = 14042
FOLDS = 5
TARGET_SENSITIVITY = 0.95
BOOTSTRAP_DRAWS = 300


def linear_logits(features: np.ndarray, probe_path: Path) -> np.ndarray:
    """Apply a checked standardized linear probe to cache features."""
    with np.load(probe_path) as saved:
        mean, scale = saved["mean"], saved["scale"]
        coefficient = saved["coefficient"].reshape(-1)
        intercept = float(saved["intercept"].reshape(-1)[0])
    if len(mean) != features.shape[1] or len(scale) != len(mean) or len(coefficient) != len(mean):
        raise ValueError("Probe dimension differs from cached features")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Invalid probe standardization")
    return ((features - mean) / scale) @ coefficient + intercept


def normalized_logits(logits: np.ndarray, train_count: int) -> tuple[np.ndarray, dict[str, float]]:
    """Normalize train and development logits with labeled training statistics."""
    mean, std = float(np.mean(logits[:train_count])), float(np.std(logits[:train_count], ddof=0))
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("Invalid labeled-training logit distribution")
    return (logits - mean) / std, {"mean": mean, "std": std}


def development_folds(y: np.ndarray, patients: list[str]) -> np.ndarray:
    """Assign five stratified patient folds using the frozen 014 seed."""
    groups = np.asarray(patients)
    splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=FOLD_SEED)
    folds = np.full(len(y), -1, dtype=int)
    for index, (training, heldout) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
        if set(y[training]) != {0, 1} or set(y[heldout]) != {0, 1}:
            raise ValueError("Development fold lacks a class")
        folds[heldout] = index
    if np.any(folds < 0) or any(len(set(folds[groups == p])) != 1 for p in set(groups)):
        raise ValueError("Development patient spans folds")
    return folds


def dev_score(y: np.ndarray, logits: np.ndarray, folds: np.ndarray) -> dict[str, Any]:
    """Compute AUROC and a cross-fitted 95%-sensitivity operating point."""
    predicted = np.zeros(len(y), dtype=bool)
    thresholds = {}
    for index in sorted(set(folds)):
        training, heldout = folds != index, folds == index
        if set(y[training]) != {0, 1} or set(y[heldout]) != {0, 1}:
            raise ValueError("Cross-fitted fold lacks a class")
        positive = np.sort(logits[training & (y == 1)])
        required = int(np.ceil(TARGET_SENSITIVITY * len(positive) - 1e-12))
        threshold = float(positive[len(positive) - required])
        predicted[heldout] = logits[heldout] >= threshold
        thresholds[str(index)] = threshold
    return {"auroc": float(roc_auc_score(y, logits)),
            "specificity": float(np.mean(~predicted[y == 0])),
            "sensitivity": float(np.mean(predicted[y == 1])),
            "crossfit_thresholds": thresholds}


def select(scores: dict[float, dict[str, Any]]) -> float:
    """Apply the original specificity, AUROC and weight tie rule."""
    return max(GRID, key=lambda alpha: (scores[alpha]["specificity"], scores[alpha]["auroc"],
                                         -abs(alpha - .5), -alpha))


def bootstrap_screen(y: np.ndarray, jepa: np.ndarray, cpc: np.ndarray,
                     patients: list[str], folds: np.ndarray, chosen: float) -> dict[str, Any]:
    """Measure grid selection stability by resampling development patients."""
    groups: dict[str, list[int]] = {}
    for index, patient in enumerate(patients):
        groups.setdefault(str(patient), []).append(index)
    blocks = list(groups.values())
    rng = np.random.default_rng(FOLD_SEED)
    selected, gains = [], []
    for _ in range(BOOTSTRAP_DRAWS):
        sample = np.concatenate([blocks[i] for i in rng.integers(len(blocks), size=len(blocks))])
        sampled_y, sampled_folds = y[sample], folds[sample]
        if (set(sampled_y) != {0, 1}
                or any(set(sampled_y[sampled_folds == i]) != {0, 1}
                       or set(sampled_y[sampled_folds != i]) != {0, 1} for i in set(sampled_folds))):
            continue
        scores = {alpha: dev_score(sampled_y, alpha * jepa[sample] + (1 - alpha) * cpc[sample],
                                   sampled_folds) for alpha in GRID}
        selected.append(select(scores))
        endpoint = max(scores[0.0]["specificity"], scores[1.0]["specificity"])
        gains.append(scores[chosen]["specificity"] - endpoint)
    if not selected:
        raise ValueError("No valid patient bootstrap draw")
    return {"seed": FOLD_SEED, "valid_draws": len(selected),
            "selected_weight_counts": {str(alpha): selected.count(alpha) for alpha in GRID},
            "interior_selection_fraction": float(np.mean(np.isin(selected, GRID[1:-1]))),
            "point_weight_positive_gain_fraction": float(np.mean(np.asarray(gains) > 0)),
            "point_weight_gain_ci95": [float(value) for value in np.quantile(gains, [.025, .975])]}


def gate(scores: dict[float, dict[str, Any]], chosen: float, bootstrap: dict[str, Any]) -> dict[str, Any]:
    """Apply the original 014 development advance criteria."""
    endpoint = max(scores[0.0]["specificity"], scores[1.0]["specificity"])
    endpoint_auc = max(scores[0.0]["auroc"], scores[1.0]["auroc"])
    adjacent = [alpha for alpha in GRID[1:-1] if abs(alpha - chosen) == .25]
    checks = {"interior_weight": chosen in GRID[1:-1],
              "specificity_gain_ge_0.02": scores[chosen]["specificity"] - endpoint >= .02,
              "auroc_loss_le_0.002": scores[chosen]["auroc"] >= endpoint_auc - .002,
              "adjacent_interior_gain": any(scores[alpha]["specificity"] > endpoint for alpha in adjacent),
              "bootstrap_interior_ge_half": bootstrap["interior_selection_fraction"] >= .5,
              "bootstrap_gain_fraction_ge_0.75": bootstrap["point_weight_positive_gain_fraction"] >= .75}
    return {"pass": all(checks.values()), "checks": checks,
            "specificity_gain_vs_best_endpoint": scores[chosen]["specificity"] - endpoint,
            "auroc_difference_vs_best_endpoint": scores[chosen]["auroc"] - endpoint_auc}


def run_budget(budget: str, probes: Path, output: Path) -> dict[str, Any]:
    """Screen one clean-budget fusion grid without opening test predictions."""
    fraction = FRACTIONS[budget]
    train = validate_selection(read_csv(PREFLIGHT / f"labels_fraction{fraction}.csv"),
                               read_csv(PTB / f"seed42_fraction{fraction}" / "labeled_train.csv"))
    if len(train) != COUNTS[budget]:
        raise ValueError("Clean label count differs")
    development, _ = partition_validation(read_csv(PTB / f"seed42_fraction{fraction}" / "validation.csv"))
    digests = input_hashes(budget)
    features = aligned_features(budget, train, development, digests)
    identity = {"probe_input_sha256": digests,
                "screen_source_sha256": sha256_file(Path(__file__)),
                "protocol_sha256": sha256_file(ROOT / "docs/clean-probe-fusion-v1.md"),
                "probe_sha256": {}}
    logits, stats = {}, {}
    for model in ("jepa", "cpc"):
        path = probes / f"{budget}_{model}.npz"
        receipt = json.loads((probes / f"{budget}_{model}.json").read_text())
        digest = sha256_file(path)
        if receipt["input_sha256"] != digests or receipt["model_sha256"] != digest:
            raise ValueError(f"Clean {budget} {model} probe provenance differs")
        identity["probe_sha256"][model] = digest
        train_x, dev_x = features[model]
        raw = linear_logits(np.concatenate((train_x, dev_x)), path)
        normalized, stats[model] = normalized_logits(raw, len(train))
        logits[model] = normalized[len(train):]
    y = np.asarray([int(row["target"]) for row in development])
    patients = [row["patient_id"] for row in development]
    folds = development_folds(y, patients)
    scores = {alpha: dev_score(y, alpha * logits["jepa"] + (1 - alpha) * logits["cpc"], folds)
              for alpha in GRID}
    chosen = select(scores)
    bootstrap = bootstrap_screen(y, logits["jepa"], logits["cpc"], patients, folds, chosen)
    decision = gate(scores, chosen, bootstrap)
    result = {"budget": budget, "cohort": "clean_original_ptbxl_labels", "train_count": len(train),
              "development_count": len(development), "fingerprint": identity,
              "train_logit_normalization": stats, "grid": {str(a): scores[a] for a in GRID},
              "selected_alpha": chosen, "fold_assignment_sha256": hashlib.sha256(
                  np.asarray(folds, dtype=np.int8).tobytes()).hexdigest(),
              "patient_bootstrap": bootstrap, "gate": decision, "calibration_test_opened": False}
    destination = output / f"{budget}.json"
    if destination.exists():
        if json.loads(destination.read_text()) != result:
            raise ValueError(f"Existing clean {budget} fusion result differs")
    else:
        write_json_atomic(destination, result, sort_keys=True)
    return result


def run(probes: Path, output: Path) -> dict[str, Any]:
    """Verify preflight and screen both clean budgets on development patients."""
    verify_preflight()
    output.mkdir(parents=True, exist_ok=True)
    results = {budget: run_budget(budget, probes, output) for budget in FRACTIONS}
    summary = {budget: {"selected_alpha": result["selected_alpha"],
                        "gate_pass": result["gate"]["pass"],
                        "grid": {alpha: {key: value for key, value in scores.items()
                                         if key in ("auroc", "specificity", "sensitivity")}
                                 for alpha, scores in result["grid"].items()}}
               for budget, result in results.items()}
    write_json_atomic(output / "summary.json", summary, sort_keys=True)
    return summary
