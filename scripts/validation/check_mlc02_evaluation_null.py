"""
Null check of the MLC-02 evaluation code on the real MLC-01 labels with RANDOM scores.

No model is fitted and no model score is computed. Every probability is drawn uniformly at random
(seeded), so every macro AUROC must be near 0.5, validation-fixed thresholds must give a test specificity
near 0.90, and paired differences between two random models must be compatible with zero. The run also
counts single-condition records under both readings and times the bootstrap. Only label and structural
columns of the manifest are read.

Run from the repository root:
``OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m scripts.validation.check_mlc02_evaluation_null``.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ecg_experiment import ROOT, mlc02_bootstrap, mlc02_data, mlc02_evaluation
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.mlc01_build import git_state
from ecg_experiment.paths import to_stored

OUTPUT = ROOT / "outputs/mlc02_evaluation_null_check_20261110"
TEST_SETS = ("test_same_source", "test_external", "sph_calibration")
SPACES = {"B": ("labels", mlc02_data.CONDITIONS), "B_lik50": ("labels_lik50", mlc02_data.CONDITIONS),
          "A": ("labels_a", mlc02_data.GROUPS_A)}
MODULES = ("ecg_experiment/mlc02_data.py", "ecg_experiment/mlc02_evaluation.py",
           "ecg_experiment/mlc02_bootstrap.py", "ecg_experiment/intervals.py",
           "scripts/validation/check_mlc02_evaluation_null.py")


def evaluation_masks(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Masks of the evaluation sets: each validation source and the three test-like roles.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Output of ``mlc02_data.load_manifest``.

    Returns
    -------
    dict[str, np.ndarray]
        Set name to record mask.
    """
    masks = {f"validation:{source}": (data["role"] == "validation") & (data["source"] == source)
             for source in mlc02_data.VALIDATION_SOURCES}
    return masks | {role: data["role"] == role for role in TEST_SETS}


def random_scores(shape: tuple[int, int], seed: int, model: int) -> np.ndarray:
    """
    Uniform random probabilities of one random "model".

    Parameters
    ----------
    shape : tuple[int, int]
        ``[records, conditions]``.
    seed : int
        Base seed.
    model : int
        Index of the random model.

    Returns
    -------
    np.ndarray
        Probabilities in [0, 1).
    """
    return np.random.default_rng([seed, model]).random(shape)


def null_sd(labels: np.ndarray) -> float:
    """
    Return the standard deviation of the macro AUROC of independent random scores.

    Parameters
    ----------
    labels : np.ndarray
        ``[records, conditions]`` states of one set.

    Returns
    -------
    float
        ``sqrt(sum_c (P_c + N_c + 1) / (12 P_c N_c)) / C`` over the C entering conditions; NaN if none
        enters.
    """
    entering = mlc02_evaluation.enters_macro(labels)
    counts = mlc02_evaluation.support(labels)
    positives, negatives = counts["positives"][entering], counts["negatives"][entering]
    if not entering.any():
        return math.nan
    variance = np.sum((positives + negatives + 1) / (12 * positives * negatives))
    return float(math.sqrt(variance) / entering.sum())


def set_rows(data: dict[str, np.ndarray], scores: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """
    Macro AUROC of random scores in every evaluation set and label space.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Output of ``mlc02_data.load_manifest``.
    scores : dict[str, np.ndarray]
        Label space to random probabilities of all records.

    Returns
    -------
    list[dict[str, Any]]
        One row per set and space.
    """
    rows = []
    for name, mask in evaluation_masks(data).items():
        for space, (column, conditions) in SPACES.items():
            labels = data[column][mask]
            table = mlc02_evaluation.metrics_table(labels, scores[space][mask], conditions=conditions)
            aurocs = [row["auroc"] for row in table["conditions"] if row["enters_macro"]]
            spread = null_sd(labels)
            rows.append({"set": name, "space": space, "records": int(mask.sum()),
                         "entering": len(table["entering"]), "macro_auroc": table["macro_auroc"],
                         "null_sd": spread, "z": (table["macro_auroc"] - 0.5) / spread,
                         "min_condition_auroc": min(aurocs, default=math.nan),
                         "max_condition_auroc": max(aurocs, default=math.nan)})
    return rows


def null_distribution(data: dict[str, np.ndarray], seed: int, repeats: int) -> dict[str, dict[str, float]]:
    """
    Spread of the space-B macro AUROC of random scores over many seeds, in units of ``null_sd``.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Output of ``mlc02_data.load_manifest``.
    seed : int
        Base seed; repeat ``r`` uses random model ``1000 + r``.
    repeats : int
        Number of random score matrices.

    Returns
    -------
    dict[str, dict[str, float]]
        Per set: ``mean_z``, ``sd_z`` and ``max_abs_z`` of ``(macro - 0.5) / null_sd``.
    """
    masks = evaluation_masks(data)
    spread = {name: null_sd(data["labels"][mask]) for name, mask in masks.items()}
    z: dict[str, list[float]] = {name: [] for name in masks}
    for repeat in range(repeats):
        scores = random_scores(data["labels"].shape, seed, 1000 + repeat)
        for name, mask in masks.items():
            macro = mlc02_evaluation.macro_auroc(data["labels"][mask], scores[mask])
            z[name].append((macro - 0.5) / spread[name])
    return {name: {"mean_z": float(np.mean(values)), "sd_z": float(np.std(values)),
                   "max_abs_z": float(np.max(np.abs(values))), "repeats": repeats}
            for name, values in z.items()}


def operating_points(data: dict[str, np.ndarray], scores: np.ndarray) -> dict[str, Any]:
    """
    Validation-fixed thresholds and Platt calibration of random scores, applied to the test sets.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Output of ``mlc02_data.load_manifest``.
    scores : np.ndarray
        Random probabilities of all records (space B).

    Returns
    -------
    dict[str, Any]
        Selection metric, thresholds, Platt parameters, and per test set the range over conditions of
        specificity and sensitivity at the thresholds.
    """
    validation = data["role"] == "validation"
    labels = data["labels"][validation]
    thresholds = mlc02_evaluation.specificity_thresholds(labels, scores[validation])
    platt = mlc02_evaluation.fit_platt(labels, scores[validation])
    result: dict[str, Any] = {
        "selection": mlc02_evaluation.selection_metric(labels, scores[validation],
                                                       data["source"][validation]),
        "thresholds": dict(zip(mlc02_data.CONDITIONS, thresholds.tolist(), strict=True)),
        "platt_intercept_range": [float(platt[:, 0].min()), float(platt[:, 0].max())],
        "platt_slope_range": [float(platt[:, 1].min()), float(platt[:, 1].max())]}
    for name in ("validation", *TEST_SETS):
        mask = data["role"] == name
        table = mlc02_evaluation.metrics_table(data["labels"][mask], scores[mask], thresholds)
        specificity = [row["specificity"] for row in table["conditions"]]
        sensitivity = [row["sensitivity"] for row in table["conditions"]]
        result[name] = {"specificity_range": [min(specificity), max(specificity)],
                        "sensitivity_range": [min(sensitivity), max(sensitivity)]}
    return result


def multiclass_rows(data: dict[str, np.ndarray], scores: np.ndarray,
                    thresholds: np.ndarray) -> dict[str, Any]:
    """
    Single-condition record counts under both readings, and the derived view of random scores.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Output of ``mlc02_data.load_manifest``.
    scores : np.ndarray
        Random probabilities of all records (space B).
    thresholds : np.ndarray
        Validation-fixed thresholds of the same scores.

    Returns
    -------
    dict[str, Any]
        Per set: ``multiclass_view`` output and subgroup record counts.
    """
    result = {}
    for name, mask in evaluation_masks(data).items():
        labels = data["labels"][mask]
        view = mlc02_evaluation.multiclass_view(labels, scores[mask], thresholds)
        positives = (labels == 1).sum(axis=1)
        result[name] = {**view, "records": int(mask.sum()),
                        "records_no_positive": int((positives == 0).sum()),
                        "records_two_or_more_positives": int((positives >= 2).sum()),
                        "sex": {key: int(value.sum()) for key, value in
                                mlc02_evaluation.sex_masks(data["sex"][mask]).items()},
                        "flat_or_missing_lead": int(data["lead_flag"][mask].sum())}
    return result


def timed(function: Any, *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """
    Call a function and measure its wall time.

    Parameters
    ----------
    function : Any
        Callable.
    *args, **kwargs : Any
        Its arguments.

    Returns
    -------
    tuple[Any, float]
        Result and seconds.
    """
    started = time.perf_counter()
    result = function(*args, **kwargs)
    return result, time.perf_counter() - started


def bootstrap_check(data: dict[str, np.ndarray], role: str, models: int, seed: int, draws: int,
                    workers: int) -> dict[str, Any]:
    """
    Bootstrap timing, intervals and paired differences of random models on one test set.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        Output of ``mlc02_data.load_manifest``.
    role : str
        Test role.
    models : int
        Number of random models.
    seed : int
        Base seed of the scores.
    draws : int
        Bootstrap resamples.
    workers : int
        Threads of ``bootstrap_models``.

    Returns
    -------
    dict[str, Any]
        Timings, resampling identity, model 0 interval report summary and paired comparisons with model 0.
    """
    validation = data["role"] == "validation"
    mask = data["role"] == role
    labels = data["labels"][mask]
    probabilities, thresholds = {}, {}
    for model in range(models):
        scores = random_scores(data["labels"].shape, seed, model)
        probabilities[f"random_{model:02d}"] = scores[mask]
        thresholds[f"random_{model:02d}"] = mlc02_evaluation.specificity_thresholds(
            data["labels"][validation], scores[validation])
    resamples, resample_seconds = timed(mlc02_bootstrap.cluster_resamples, data["group"][mask], labels, draws)
    first = next(iter(probabilities))
    _, single_seconds = timed(mlc02_bootstrap.bootstrap_metrics, labels, probabilities[first], resamples,
                              thresholds[first], True)
    ranking, ranking_seconds = timed(mlc02_bootstrap.bootstrap_models, labels, probabilities, resamples,
                                     thresholds, False, workers)
    _, full_seconds = timed(mlc02_bootstrap.bootstrap_models, labels, probabilities, resamples, thresholds,
                            True, workers)
    report, _ = mlc02_bootstrap.interval_report(labels, probabilities[first], resamples, thresholds[first])
    pairs, pair_seconds = timed(lambda: [mlc02_bootstrap.paired_comparison(ranking[name], ranking[first])
                                         for name in list(probabilities)[1:]])
    return {"role": role, "records": int(mask.sum()), "groups": resamples["groups"], "draws": draws,
            "redraws": resamples["redraws"], "fingerprint": resamples["fingerprint"], "models": models,
            "workers": workers,
            "seconds": {"resamples": resample_seconds, "one_model_all_metrics_one_thread": single_seconds,
                        "all_models_without_calibration": ranking_seconds,
                        "all_models_with_calibration": full_seconds, "paired_comparisons": pair_seconds},
            "model_0": model_summary(report), "paired_with_model_0": pair_summary(pairs)}


def model_summary(report: dict[str, Any]) -> dict[str, Any]:
    """
    Summarize the interval report of one random model.

    Parameters
    ----------
    report : dict[str, Any]
        Output of ``mlc02_bootstrap.interval_report``.

    Returns
    -------
    dict[str, Any]
        Macro AUROC with its interval, conditions whose interval contains 0.5, calibration NaN draws and
        conditions called usable.
    """
    rows = report["conditions"]
    return {"macro_auroc": report["macro_auroc"],
            "macro_auroc_ci": [report["macro_auroc_ci_low"], report["macro_auroc_ci_high"]],
            "conditions_whose_auroc_interval_contains_0.5": sum(
                row["auroc_ci_low"] <= 0.5 <= row["auroc_ci_high"] for row in rows),
            "calibration_nan_draws": sum(row["calibration_slope_nan_draws"] for row in rows),
            "usable_conditions": sum(row["usable"] for row in rows)}


def pair_summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Summarize paired comparisons of random models with model 0.

    Parameters
    ----------
    pairs : list[dict[str, Any]]
        Outputs of ``mlc02_bootstrap.paired_comparison``.

    Returns
    -------
    dict[str, Any]
        Macro intervals containing zero, the macro differences, and the Holm-adjusted condition tests.
    """
    holm = [row["p_holm"] for pair in pairs for row in pair["conditions"]]
    return {"comparisons": len(pairs),
            "macro_intervals_containing_0": sum(pair["macro"]["ci_low"] <= 0 <= pair["macro"]["ci_high"]
                                                for pair in pairs),
            "macro_differences": [pair["macro"]["difference"] for pair in pairs],
            "condition_tests": len(holm), "holm_below_0.05": sum(p < 0.05 for p in holm),
            "min_p_holm": min(holm)}


def clean(value: Any) -> Any:
    """
    Replace NaN and infinity by None, recursively, for strict JSON.

    Parameters
    ----------
    value : Any
        JSON-like value.

    Returns
    -------
    Any
        The same value with non-finite floats replaced.
    """
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [clean(item) for item in value]
    if isinstance(value, float | np.floating):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer | np.bool_):
        return value.item()
    return value


def main() -> None:
    """Run the null check and write ``null_check.json``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=mlc02_data.MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--models", type=int, default=12)
    parser.add_argument("--draws", type=int, default=mlc02_bootstrap.DRAWS)
    parser.add_argument("--workers", type=int, default=mlc02_bootstrap.WORKERS)
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()
    started = time.perf_counter()
    data, load_seconds = timed(mlc02_data.load_manifest, args.manifest)
    scores = {space: random_scores(data[column].shape, args.seed, 0) for space, (column, _) in SPACES.items()}
    points = operating_points(data, scores["B"])
    thresholds = np.array([points["thresholds"][name] for name in mlc02_data.CONDITIONS])
    result = {
        "what": "RANDOM scores on the real MLC-01 v4 labels; no model is fitted or scored",
        "git": git_state(), "command": sys.argv, "manifest": to_stored(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "code_sha256": {path: sha256_file(ROOT / path) for path in MODULES},
        "seed": args.seed, "threads": {name: os.environ.get(name) for name in
                                       ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_CORETYPE")},
        "load_seconds": load_seconds,
        "sets": set_rows(data, scores), "null_distribution": null_distribution(data, args.seed, args.repeats),
        "operating_points": points,
        "multiclass": multiclass_rows(data, scores["B"], thresholds),
        "bootstrap": [bootstrap_check(data, role, args.models, args.seed, args.draws, args.workers)
                      for role in ("test_external", "test_same_source")]}
    result["total_seconds"] = time.perf_counter() - started
    write_json_atomic(args.output_dir / "null_check.json", clean(result))


if __name__ == "__main__":
    main()
