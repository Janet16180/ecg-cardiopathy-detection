"""Hash-bound paired analysis of the prospective causal Transformer objective study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ecg_experiment import encoder_context_analysis039 as paired
from ecg_experiment import encoder_context_interactions039 as interaction_bootstrap
from ecg_experiment import encoder_context_study039 as predecessor
from ecg_experiment import simdino_study040 as study
from ecg_experiment.files import sha256_file, sha256_json, write_json_atomic
from ecg_experiment.paths import to_stored

OBJECTIVES = study.OBJECTIVES
GROUPS = ("patch", *OBJECTIVES)
CONTEXTS = paired.CONTEXTS
BUDGETS = paired.BUDGETS
SEEDS = study.SEEDS
OUTPUT = f"outputs/{study.NAME}"
ARTIFACTS = ("manifest.json", "profile.json", "training.json", "features.npz", "head_parameters.npz",
             "development_predictions.npz", "result.json", "audit.json", "stage_walltime.json")


def cell_path(root: Path, group: str, seed: int) -> Path:
    """Locate an original objective cell or its historical patch control.

    Parameters
    ----------
    root : Path
        Repository containing local outputs.
    group : str
        Patch control or one of the three objectives.
    seed : int
        Matched initialization seed.

    Returns
    -------
    Path
        Directory containing the original immutable artifacts.
    """
    if group == "patch":
        return paired.cell_path(root, 25, "patch", seed)
    return study.directory(root, group, seed)


def _validated_cell(root: Path, path: Path) -> dict[str, Any]:
    """Require successful stages, immutable artifacts and unchanged scientific sources."""
    for stage in study.STAGES:
        study.require_success(path, stage)
    result = predecessor.audited_result(path)
    if result["calibration_or_test_scored"]:
        raise ValueError("Only development results are permitted")
    required = {"features.npz": "features_sha256", "head_parameters.npz": "head_parameters_sha256",
                "training.json": "training_sha256"}
    for filename, key in required.items():
        if sha256_file(path / filename) != result[key]:
            raise ValueError(f"Changed audited artifact: {filename}")
    manifest = json.loads((path / "manifest.json").read_text())
    identity = sha256_json(manifest)
    receipts = {"result": result,
                "training": json.loads((path / "training.json").read_text()),
                "profile": json.loads((path / "profile.json").read_text())}
    for name, receipt in receipts.items():
        if receipt["identity_sha256"] != identity:
            raise ValueError(f"The {name} receipt has a different manifest identity")
    for filename, expected in manifest["files_sha256"].items():
        scientific = filename.startswith(("ecg_experiment/", "scripts/", "tests/", "docs/", "third_party/"))
        if ((scientific or filename in ("pyproject.toml", "uv.lock"))
                and sha256_file(root / filename) != expected):
            raise ValueError(f"Changed pinned scientific source: {filename}")
    return result


def _probabilities(path: Path, result: dict[str, Any],
                   arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Load identically aligned probabilities and reproduce both saved metrics exactly."""
    values = {}
    with np.load(path / "development_predictions.npz", allow_pickle=False) as saved:
        for name in ("record_ids", "patient_ids", "targets"):
            if name in arrays and not np.array_equal(arrays[name], saved[name]):
                raise ValueError("Development identities or targets changed between study cells")
            arrays[name] = saved[name].copy()
        for budget in BUDGETS:
            for context in CONTEXTS:
                name = f"{budget}_{context}"
                probability = saved[name].copy()
                if (probability.shape != arrays["targets"].shape or not np.isfinite(probability).all()
                        or np.any((probability < 0) | (probability > 1))):
                    raise ValueError("Malformed development probabilities")
                paired._check_probability(arrays["targets"], probability, result["scores"][budget][context])
                values[name] = probability
    return values


def load_cells(root: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Load all nine new audited cells and three matched historical controls.

    Parameters
    ----------
    root : Path
        Repository containing complete original outputs.

    Returns
    -------
    tuple[dict[str, np.ndarray], dict[str, str]]
        Aligned three-seed probability matrices and hashes of every consumed artifact.
    """
    arrays: dict[str, np.ndarray] = {}
    hashes = {}
    for group in GROUPS:
        seed_arrays: dict[str, list[np.ndarray]] = {}
        for seed in SEEDS:
            path = cell_path(root, group, seed)
            result = _validated_cell(root, path)
            for filename in ARTIFACTS:
                hashes[to_stored(path / filename)] = sha256_file(path / filename)
            for name, probability in _probabilities(path, result, arrays).items():
                seed_arrays.setdefault(name, []).append(probability)
        arrays.update({f"{group}_{name}": np.stack(values) for name, values in seed_arrays.items()})
    return arrays, hashes


def comparisons() -> dict[str, tuple[str, str, bool]]:
    """Define the six primary contrasts and fixed secondary objective/context comparisons.

    Returns
    -------
    dict[str, tuple[str, str, bool]]
        Names mapped to first/second probability keys and primary membership.
    """
    pairs = {}
    for budget in BUDGETS:
        for context in CONTEXTS:
            for first, second in (("cpc", "patch"), ("hybrid", "cpc"),
                                  ("hybrid", "simdino"), ("simdino", "cpc")):
                name = f"{budget}_{first}_minus_{second}_{context}"
                primary = budget == "limited" and (first, second) != ("simdino", "cpc")
                pairs[name] = (f"{first}_{budget}_{context}", f"{second}_{budget}_{context}", primary)
        for group in GROUPS:
            pairs[f"{budget}_{group}_xlstm_minus_gru"] = (
                f"{group}_{budget}_xlstm", f"{group}_{budget}_gru", False)
    return pairs


def _interaction_keys() -> dict[str, tuple[str, str, str, str]]:
    """Specify objective gains under mLSTM minus the corresponding GRU gains."""
    return {
        f"{budget}_{first}_minus_{second}_by_context": (
            f"{first}_{budget}_xlstm", f"{second}_{budget}_xlstm",
            f"{first}_{budget}_gru", f"{second}_{budget}_gru")
        for budget in BUDGETS
        for first, second in (("cpc", "patch"), ("simdino", "cpc"),
                              ("hybrid", "cpc"), ("hybrid", "simdino"))
    }


def combined_benefit(inference: dict[str, Any]) -> dict[str, bool]:
    """Require a hybrid to beat both component objectives under each context.

    Parameters
    ----------
    inference : dict[str, Any]
        Complete six-contrast simultaneous analysis.

    Returns
    -------
    dict[str, bool]
        Context-specific decisions, false for an incomplete primary family.
    """
    complete = inference.get("primary_family_size") == 6
    return {
        context: bool(complete and all(
            inference["contrasts"][f"limited_hybrid_minus_{control}_{context}"]["promising"]
            for control in ("cpc", "simdino")))
        for context in CONTEXTS
    }


def joint_intervals(arrays: dict[str, np.ndarray], draws: int = 2000,
                    seed: int = 40045) -> dict[str, Any]:
    """Reuse the frozen paired patient bootstrap with this study's prospective contrast map.

    Parameters
    ----------
    arrays : dict[str, np.ndarray]
        Aligned identities and three-seed probability matrices.
    draws : int
        Number of whole-patient draws, including skipped single-class draws.
    seed : int
        One shared bootstrap generator seed.

    Returns
    -------
    dict[str, Any]
        Paired intervals, six-comparison simultaneous bands and combined-benefit decisions.
    """
    if not isinstance(draws, (int, np.integer)) or isinstance(draws, bool) or draws <= 0:
        raise ValueError("draws must be a positive integer")
    names = sorted({name for first, second, _ in comparisons().values() for name in (first, second)})
    interaction_bootstrap._validated(arrays, names)
    original = paired.comparisons
    try:
        paired.comparisons = comparisons
        result = paired.factorial_intervals(arrays, draws=draws, seed=seed)
    finally:
        paired.comparisons = original
    result["combined_benefit"] = combined_benefit(result)
    return result


def interaction_intervals(arrays: dict[str, np.ndarray], draws: int = 2000,
                          seed: int = 40045) -> dict[str, Any]:
    """Reuse the paired interaction bootstrap for exploratory difference-of-differences.

    Parameters
    ----------
    arrays : dict[str, np.ndarray]
        Aligned identities and three-seed probability matrices.
    draws : int
        Whole-patient bootstrap draws.
    seed : int
        Same generator seed used for the primary patient draws.

    Returns
    -------
    dict[str, Any]
        Exploratory objective-by-context AUROC interaction intervals.
    """
    original = interaction_bootstrap._interaction_keys
    try:
        interaction_bootstrap._interaction_keys = _interaction_keys
        result = interaction_bootstrap.analyze_interactions(arrays, draws=draws, seed=seed)
    finally:
        interaction_bootstrap._interaction_keys = original
    result["formula"] = "mean_seed[(first_xlstm - second_xlstm) - (first_gru - second_gru)]"
    return result


def _scores(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Compute descriptive per-seed AUROC and AP without ensembling probabilities."""
    y = arrays["targets"]
    scores = {}
    for name, probabilities in arrays.items():
        if name in ("record_ids", "patient_ids", "targets"):
            continue
        auc = np.asarray([roc_auc_score(y, probability) for probability in probabilities])
        ap = np.asarray([average_precision_score(y, probability) for probability in probabilities])
        scores[name] = {"auroc": auc.tolist(), "mean_auroc": float(auc.mean()),
                        "sd_auroc": float(auc.std(ddof=1)),
                        "range_auroc": [float(auc.min()), float(auc.max())],
                        "average_precision": ap.tolist(), "mean_average_precision": float(ap.mean())}
    return scores


def severe_failures(scores: dict[str, Any]) -> list[dict[str, Any]]:
    """Compare original fits with their prespecified same-seed controls.

    Parameters
    ----------
    scores : dict[str, Any]
        Exact per-seed score summaries for all original packages and patch controls.

    Returns
    -------
    list[dict[str, Any]]
        Severe-development triggers; numerical and collapse diagnoses are separate receipts.
    """
    failures = []
    for objective in OBJECTIVES:
        control = "patch" if objective == "cpc" else "cpc"
        for context in CONTEXTS:
            observed = scores[f"{objective}_limited_{context}"]["auroc"]
            baseline = scores[f"{control}_limited_{context}"]["auroc"]
            for seed, score, reference in zip(SEEDS, observed, baseline, strict=True):
                if score <= 0.60 or score - reference <= -0.03:
                    failures.append({"objective": objective, "context": context, "seed": seed,
                                     "auroc": score, "baseline_auroc": reference,
                                     "difference": score - reference, "control": control})
    return failures


def progress(root: Path) -> dict[str, Any]:
    """Describe completed and unfinished cells without granting partial-family decisions.

    Parameters
    ----------
    root : Path
        Repository containing prospective or partially executed study outputs.

    Returns
    -------
    dict[str, Any]
        Audited completed cells, pending cells and durable coordinator/resource status.
    """
    completed, unfinished = [], []
    for seed in SEEDS:
        for objective in OBJECTIVES:
            path = study.directory(root, objective, seed)
            cell: dict[str, Any] = {"objective": objective, "seed": seed}
            if (path / "audit.json").exists():
                result = _validated_cell(root, path)
                cell["scores"] = result["scores"]
                completed.append(cell)
            else:
                cell["latest_stages"] = {stage: study.latest_stage(path, stage) for stage in study.STAGES}
                unfinished.append(cell)
    state = {}
    for name in ("status", "admission"):
        path = root / OUTPUT / f"{name}.json"
        state[f"{name}_receipt"] = json.loads(path.read_text()) if path.exists() else None
    return {"status": "incomplete_original_study", "completed_cells": completed,
            "unfinished_cells": unfinished, "completed_fits": len(completed) * 2,
            "scheduled_fits": 18, "primary_family_complete": not unfinished,
            "primary_decisions": None, **state}


def aggregate(root: Path) -> dict[str, Any]:
    """Write an immutable complete aggregate or return an explicit incomplete study snapshot.

    Parameters
    ----------
    root : Path
        Repository containing local original results.

    Returns
    -------
    dict[str, Any]
        Complete hash-bound inference, or progress with no incomplete-family decisions.
    """
    snapshot = progress(root)
    destination = root / OUTPUT / "aggregate.json"
    if snapshot["unfinished_cells"]:
        if destination.exists():
            raise ValueError("An input to the completed aggregate disappeared")
        return snapshot
    arrays, hashes = load_cells(root)
    if destination.exists():
        saved = json.loads(destination.read_text())
        if saved["input_sha256"] != hashes:
            raise ValueError("Completed study aggregate inputs changed")
        return saved
    scores = _scores(arrays)
    ap = {name: float(np.mean(np.asarray(scores[first]["average_precision"])
                             - np.asarray(scores[second]["average_precision"])))
          for name, (first, second, _) in comparisons().items()}
    result = {"status": "complete_original_study_development_only", "input_sha256": hashes,
              "seeds": list(SEEDS), "objectives": list(OBJECTIVES), "contexts": list(CONTEXTS),
              "completed_fits": 18, "primary_family_complete": True, "scores": scores,
              "inference": joint_intervals(arrays), "interactions": interaction_intervals(arrays),
              "average_precision_differences": ap, "severe_failures": severe_failures(scores),
              "development_records": len(arrays["targets"]),
              "development_patients": len(np.unique(arrays["patient_ids"])),
              "calibration_or_test_scored": False}
    write_json_atomic(destination, result, sort_keys=True)
    return result
