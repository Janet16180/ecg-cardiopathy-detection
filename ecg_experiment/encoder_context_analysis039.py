"""Joint patient-bootstrap analysis of the fixed CPC encoder/context factorial."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.intervals import patient_groups, two_class_resamples
from ecg_experiment.paths import to_stored

ENCODERS = ("cnn", "multiscale", "patch")
CONTEXTS = ("gru", "xlstm")
SEEDS = (39042, 39043, 39044)
TIERS = (25, 50)
BUDGETS = ("limited", "full")
OUTPUT = "outputs/experiment039_encoder_context"


def cell_path(root: Path, tier: int, encoder: str, seed: int) -> Path:
    """Locate a two-context study cell.

    Parameters
    ----------
    root : Path
        Repository root.
    tier : int
        Cohort size in thousands.
    encoder : str
        Waveform encoder name.
    seed : int
        Fixed initialization seed.

    Returns
    -------
    Path
        Directory containing the cell's original immutable artifacts.
    """
    return root / OUTPUT / encoder / f"seed{seed}" / f"{tier}k"


def _cell_result(path: Path) -> dict[str, Any]:
    """Require unchanged audited development-only probabilities."""
    result = json.loads((path / "result.json").read_text())
    audit = json.loads((path / "audit.json").read_text())
    if (audit["status"] != "passed_development_only"
            or result["status"] != "complete_development_only"
            or audit["result_sha256"] != sha256_file(path / "result.json")
            or result["predictions_sha256"] != sha256_file(path / "development_predictions.npz")
            or result["calibration_or_test_scored"]):
        raise ValueError(f"Cell lacks unchanged development-only audit: {path}")
    return result


def _check_probability(y: np.ndarray, probability: np.ndarray, observed: dict[str, Any]) -> None:
    """Verify finite probabilities and exact saved metric replay."""
    if not np.isfinite(probability).all():
        raise ValueError("Nonfinite saved probabilities")
    if (roc_auc_score(y, probability) != observed["auroc"]
            or average_precision_score(y, probability) != observed["average_precision"]):
        raise ValueError("Factorial score disagrees with saved probabilities")


def load_cells(root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read only completed audited cells and enforce identical development alignment.

    Parameters
    ----------
    root : Path
        Repository root with all eighteen audited cells.

    Returns
    -------
    tuple[dict[str, np.ndarray], dict[str, Any]]
        Aligned identities and seed-by-record score arrays, plus artifact hashes.
    """
    arrays: dict[str, np.ndarray] = {}
    hashes = {}
    for tier in TIERS:
        for encoder in ENCODERS:
            seed_arrays: dict[str, list[np.ndarray]] = {}
            for seed in SEEDS:
                path = cell_path(root, tier, encoder, seed)
                result = _cell_result(path)
                for name in ("manifest.json", "result.json", "audit.json", "development_predictions.npz"):
                    hashes[to_stored(path / name)] = sha256_file(path / name)
                with np.load(path / "development_predictions.npz", allow_pickle=False) as saved:
                    for name in ("record_ids", "patient_ids", "targets"):
                        if name in arrays and not np.array_equal(arrays[name], saved[name]):
                            raise ValueError("Development rows changed between factorial cells")
                        arrays[name] = saved[name].copy()
                    for budget in BUDGETS:
                        for context in CONTEXTS:
                            name = f"{budget}_{context}"
                            probability = saved[name].copy()
                            observed = result["scores"][budget][context]
                            _check_probability(arrays["targets"], probability, observed)
                            seed_arrays.setdefault(name, []).append(probability)
            for name, probabilities in seed_arrays.items():
                arrays[f"{tier}_{encoder}_{name}"] = np.stack(probabilities)
    return arrays, hashes


def comparisons() -> dict[str, tuple[str, str, bool]]:
    """Define encoder, context and scaling comparisons before examining outcomes.

    Returns
    -------
    dict[str, tuple[str, str, bool]]
        Comparison names mapped to first/second score keys and primary membership.
    """
    pairs = {}
    for budget in BUDGETS:
        for tier in TIERS:
            for context in CONTEXTS:
                baseline = f"{tier}_cnn_{budget}_{context}"
                for encoder in ENCODERS[1:]:
                    name = f"{tier}k_{budget}_{encoder}_minus_cnn_{context}"
                    pairs[name] = (f"{tier}_{encoder}_{budget}_{context}", baseline, budget == "limited")
            for encoder in ENCODERS:
                name = f"{tier}k_{budget}_{encoder}_xlstm_minus_gru"
                pairs[name] = (f"{tier}_{encoder}_{budget}_xlstm", f"{tier}_{encoder}_{budget}_gru", False)
        for encoder in ENCODERS:
            for context in CONTEXTS:
                name = f"{budget}_{encoder}_{context}_50k_minus_25k"
                pairs[name] = (f"50_{encoder}_{budget}_{context}", f"25_{encoder}_{budget}_{context}", False)
    return pairs


def _auroc_by_seed(y: np.ndarray, probabilities: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Compute separate seed scores without forming an ensemble."""
    return np.asarray([roc_auc_score(y[rows], seed[rows]) for seed in probabilities])


def factorial_intervals(arrays: dict[str, np.ndarray], draws: int = 2000,
                        seed: int = 39045) -> dict[str, Any]:
    """Bootstrap all comparisons on shared whole-patient draws.

    Parameters
    ----------
    arrays : dict[str, np.ndarray]
        Identical development identities and three-seed probability matrices.
    draws : int
        Whole-patient draws, including single-class draws subsequently skipped.
    seed : int
        Bootstrap generator seed.

    Returns
    -------
    dict[str, Any]
        Seed differences, ordinary intervals and approximate simultaneous primary bands.
    """
    y = arrays["targets"]
    if len(np.unique(y)) != 2:
        raise ValueError("Both target classes are required for factorial intervals")
    pairs = comparisons()
    score_names = sorted({name for first, second, _ in pairs.values() for name in (first, second)})
    observed = {name: _auroc_by_seed(y, arrays[name], np.arange(len(y))) for name in score_names}
    differences = {name: observed[first] - observed[second] for name, (first, second, _) in pairs.items()}
    draws_by_pair: dict[str, list[float]] = {name: [] for name in pairs}
    rng = np.random.default_rng(seed)
    for rows in two_class_resamples(patient_groups(arrays["patient_ids"]), y, draws, rng):
        scores = {name: _auroc_by_seed(y, arrays[name], rows) for name in score_names}
        for name, (first, second, _) in pairs.items():
            draws_by_pair[name].append(float((scores[first] - scores[second]).mean()))
    accepted = len(next(iter(draws_by_pair.values())))
    if not accepted:
        raise ValueError("Every factorial bootstrap draw had a single class")
    primary_names = [name for name, (_, _, primary) in pairs.items() if primary]
    centered = np.asarray([np.asarray(draws_by_pair[name]) - differences[name].mean()
                           for name in primary_names])
    radius = float(np.percentile(np.max(np.abs(centered), axis=0), 95))
    result = {}
    for name, (_, _, primary) in pairs.items():
        delta = differences[name]
        mean = float(delta.mean())
        lower, upper = np.percentile(draws_by_pair[name], [2.5, 97.5])
        result[name] = {
            "difference": mean, "seed_differences": delta.tolist(),
            "seed_sd": float(delta.std(ddof=1)), "seed_range": [float(delta.min()), float(delta.max())],
            "ci_low": float(lower), "ci_high": float(upper), "primary": primary,
            "simultaneous_low": mean - radius if primary else None,
            "simultaneous_high": mean + radius if primary else None,
            "promising": bool(primary and mean >= 0.005 and mean - radius > 0),
        }
    return {"contrasts": result, "draws": draws, "skipped_draws": draws - accepted, "seed": seed,
            "primary_family_size": len(primary_names), "simultaneous_radius": radius,
            "uncertainty_scope": "patient sampling conditional on three realized training seeds"}


def severe_failures(arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """Identify original severe underperformance using the committed rule.

    Parameters
    ----------
    arrays : dict[str, np.ndarray]
        Aligned seed-by-record probabilities.

    Returns
    -------
    list[dict[str, Any]]
        Each triggered novel encoder/context/tier/seed comparison.
    """
    failures = []
    y = arrays["targets"]
    rows = np.arange(len(y))
    for tier in TIERS:
        for context in CONTEXTS:
            baseline = _auroc_by_seed(y, arrays[f"{tier}_cnn_limited_{context}"], rows)
            for encoder in ENCODERS[1:]:
                scores = _auroc_by_seed(y, arrays[f"{tier}_{encoder}_limited_{context}"], rows)
                for index, init_seed in enumerate(SEEDS):
                    if scores[index] <= 0.60 or scores[index] - baseline[index] <= -0.03:
                        failures.append({"tier": tier, "encoder": encoder, "context": context,
                                         "seed": init_seed, "auroc": float(scores[index]),
                                         "baseline_auroc": float(baseline[index]),
                                         "difference": float(scores[index] - baseline[index])})
    return failures


def aggregate(root: Path) -> dict[str, Any]:
    """Write the complete original factorial analysis without replacing cell results.

    Parameters
    ----------
    root : Path
        Repository root.

    Returns
    -------
    dict[str, Any]
        Hash-bound aggregate of executed original development results.
    """
    arrays, hashes = load_cells(root)
    destination = root / OUTPUT / "aggregate.json"
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved["input_sha256"] != hashes:
            raise ValueError("Completed factorial aggregate inputs changed")
        return saved
    y = arrays["targets"]
    scores = {}
    for name, probabilities in arrays.items():
        if name in ("targets", "record_ids", "patient_ids"):
            continue
        values = _auroc_by_seed(y, probabilities, np.arange(len(y)))
        aps = [float(average_precision_score(y, probability)) for probability in probabilities]
        scores[name] = {"auroc": values.tolist(), "mean_auroc": float(values.mean()),
                        "sd_auroc": float(values.std(ddof=1)), "average_precision": aps,
                        "mean_average_precision": float(np.mean(aps))}
    result = {"status": "complete_original_factorial_development_only", "input_sha256": hashes,
              "seeds": list(SEEDS), "tiers": list(TIERS), "encoders": list(ENCODERS),
              "contexts": list(CONTEXTS), "completed_fits": 36, "scores": scores,
              "inference": factorial_intervals(arrays), "severe_failures": severe_failures(arrays),
              "development_records": len(y), "development_patients": len(np.unique(arrays["patient_ids"])),
              "calibration_or_test_scored": False}
    write_json_atomic(destination, result, sort_keys=True)
    return result
