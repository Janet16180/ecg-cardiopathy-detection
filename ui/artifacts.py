"""Load only allowlisted aggregate artifacts bound to completed local receipts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from ecg_experiment.files import sha256_file

from .config import CACHED, ArtifactMode

EVALUATION = CACHED.evaluation_dir
COHORTS = CACHED.secondary_cohorts
PRIMARY_COHORTS = CACHED.primary_cohorts
COUNT_KEYS = ("tp", "tn", "fp", "fn")
RATE_KEYS = ("sensitivity", "specificity", "precision", "referral_fraction")


class ArtifactUnavailableError(ValueError):
    """An expected executed aggregate is missing, incomplete or inconsistent."""


def read_json(path: Path) -> dict[str, Any]:
    """Read one bounded local JSON object without deserializing executable artifacts."""
    if not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
        raise ArtifactUnavailableError("The completed local artifact is not available.")
    try:
        value = json.loads(path.read_text())
    except (ValueError, UnicodeError) as error:
        raise ArtifactUnavailableError("The local artifact is not valid JSON.") from error
    if not isinstance(value, dict):
        raise ArtifactUnavailableError("The local artifact must contain a JSON object.")
    return value


def number(value: object, *, nullable: bool = False) -> float | None:
    """Require finite numerical evidence; represent undefined rates as null."""
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ArtifactUnavailableError("Nonfinite or invalid numerical evidence.")
    return float(value)


def count(value: object) -> int:
    """Require a nonnegative exact integer count."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ArtifactUnavailableError("Invalid confusion-matrix count.")
    return value


def rates(matrix: dict[str, int]) -> dict[str, float | None]:
    """Compute rates from counts, preserving undefined zero-denominator quantities."""
    tp, tn, fp, fn = (matrix[key] for key in COUNT_KEYS)
    pairs = ((tp, tp + fn), (tn, tn + fp), (tp, tp + fp), (tp + fp, tp + tn + fp + fn))
    return {key: numerator / denominator if denominator else None
            for key, (numerator, denominator) in zip(RATE_KEYS, pairs, strict=True)}


def matrix_metrics(value: dict[str, Any]) -> dict[str, Any]:
    """Verify saved matrix denominators and rates before presenting aggregate evidence."""
    matrix = {key: count(value[key]) for key in COUNT_KEYS}
    denominators = {"positives": matrix["tp"] + matrix["fn"], "negatives": matrix["tn"] + matrix["fp"],
                    "ecgs": sum(matrix.values())}
    for key, expected in denominators.items():
        if key in value and count(value[key]) != expected:
            raise ArtifactUnavailableError("Confusion-matrix denominators disagree.")
    computed = rates(matrix)
    for key, expected in computed.items():
        actual = number(value.get(key), nullable=True)
        if actual != expected and (actual is None or expected is None or abs(actual - expected) > 1e-9):
            raise ArtifactUnavailableError("Saved rates disagree with the confusion matrix.")
    return {**matrix, **denominators, **computed}


def sweep_rows(value: dict[str, Any], positives: int, negatives: int) -> dict[str, list]:
    """Require an exact ordered cutoff grid with invariant reference-class denominators."""
    keys = ("threshold", *COUNT_KEYS, *RATE_KEYS)
    lengths = {len(value[key]) for key in keys}
    if len(lengths) != 1 or not lengths or not 1 <= next(iter(lengths)) <= 100_000:
        raise ArtifactUnavailableError("Incomplete threshold sweep.")
    result: dict[str, list] = {key: [] for key in keys}
    previous = -math.inf
    previous_flagged = positives + negatives
    for index, cutoff in enumerate(value["threshold"]):
        threshold = number(cutoff)
        if threshold is None or threshold <= previous:
            raise ArtifactUnavailableError("Saved score cutoffs must be finite and strictly increasing.")
        matrix = matrix_metrics({key: value[key][index] for key in (*COUNT_KEYS, *RATE_KEYS)})
        if matrix["positives"] != positives or matrix["negatives"] != negatives:
            raise ArtifactUnavailableError("Threshold sweep changes its evaluation population.")
        flagged = matrix["tp"] + matrix["fp"]
        if flagged > previous_flagged:
            raise ArtifactUnavailableError("Referral counts must not increase as the cutoff rises.")
        for key in keys:
            result[key].append(threshold if key == "threshold" else matrix[key])
        previous, previous_flagged = threshold, flagged
    return result


def evaluation_pool(value: dict[str, Any]) -> dict[str, Any]:
    """Project a development pool onto the supported aggregate schema."""
    if value["pool"] not in {"ptb", "sph"} or value["comparison"] != ">":
        raise ArtifactUnavailableError("Unsupported evaluation scope or threshold comparator.")
    metrics = matrix_metrics(value["metrics"])
    for key in ("auroc", "average_precision"):
        metric = number(value["metrics"].get(key), nullable=True)
        if metric is not None and not 0 <= metric <= 1:
            raise ArtifactUnavailableError("Invalid ranking metric.")
        metrics[key] = metric
    calibration = {key: count(value["calibration"][key])
                   for key in ("normals", "flagged_normals", "allowed_normals", "tied_normals")}
    if (calibration["flagged_normals"] > calibration["allowed_normals"] or
            calibration["allowed_normals"] > calibration["normals"] or
            calibration["tied_normals"] > calibration["normals"]):
        raise ArtifactUnavailableError("Calibration counts contradict the frozen normal budget.")
    sweep = sweep_rows(value["sweep"], metrics["positives"], metrics["negatives"])
    if value["threshold"] not in sweep["threshold"]:
        raise ArtifactUnavailableError("The saved threshold grid must include the exact calibrated cutoff.")
    return {"pool": value["pool"], "label": str(value["label"]),
            "threshold": number(value["threshold"]), "comparison": ">", "calibration": calibration,
            "metrics": metrics,
            "sweep": sweep}


def evaluation_models(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expose aggregate model descriptions without record predictions or paths."""
    if not values or len(values) > 20:
        raise ArtifactUnavailableError("No supported executed model results.")
    models = []
    for value in values:
        model_id = value["id"]
        if not isinstance(model_id, str) or not model_id.replace("_", "").isalnum():
            raise ArtifactUnavailableError("Invalid public model identity.")
        pools = [evaluation_pool(pool) for pool in value["evaluations"]]
        if not pools or len({pool["pool"] for pool in pools}) != len(pools):
            raise ArtifactUnavailableError("Missing or repeated evaluation population.")
        models.append({"id": model_id, "label": str(value["label"]),
                       "cohort": str(value["cohort"]), "evaluations": pools})
    if len({value["id"] for value in models}) != len(models):
        raise ArtifactUnavailableError("Repeated public model identity.")
    return models


def primary_comparison(value: dict[str, Any], model_ids: set[str]) -> dict[str, Any]:
    """Expose the prespecified paired development comparison without selecting a metric winner."""
    if (value["first"] not in model_ids or value["second"] not in model_ids or
            value["site"] not in {"sph", "ptb"}):
        raise ArtifactUnavailableError("The primary comparison does not match executed models.")
    interval = value["sensitivity_difference"]
    difference = {key: number(interval[key]) for key in ("difference", "ci_low", "ci_high")}
    if difference["ci_low"] > difference["ci_high"]:
        raise ArtifactUnavailableError("Invalid primary comparison interval.")
    difference.update({key: count(interval[key]) for key in ("draws", "skipped_draws", "seed")})
    gates = value["gates"]
    flags = (value["development_advantage"], value["pipeline_promoted"], *gates.values())
    if not all(isinstance(flag, bool) for flag in flags):
        raise ArtifactUnavailableError("Primary decision gates must be explicit booleans.")
    return {"first": value["first"], "second": value["second"], "site": value["site"],
            "sensitivity_difference": difference, "development_advantage": value["development_advantage"],
            "pipeline_promoted": value["pipeline_promoted"],
            "interpretation": str(value["interpretation"]),
            "uncertainty_scope": str(value["uncertainty_scope"])}


def load_evaluation(root: Path, mode: ArtifactMode = CACHED) -> dict[str, Any]:
    """Require a completed corrective evaluation receipt and return only verified aggregates."""
    receipt = read_json(root / mode.evaluation_dir / "completion.json")
    artifact = mode.evaluation_dir / "screening_evaluation.json"
    if (receipt.get("status") != "completed" or receipt.get("stage") != "evaluate" or
            receipt.get("artifacts", {}).get(artifact.as_posix()) != sha256_file(root / artifact)):
        raise ArtifactUnavailableError("The corrective evaluation has no completed artifact receipt.")
    value = read_json(root / artifact)
    if (value.get("schema_version") != 1 or value.get("complete") is not True or
            value.get("scope") != "reused_development" or
            value.get("score_interpretation") != "uncalibrated"):
        raise ArtifactUnavailableError("Only completed corrective development evaluations are supported.")
    operating = value["operating_point"]
    if (operating["normal_budget"] != 0.05 or operating["comparison"] != ">" or
            operating["selection"] != "calibration_normals_only"):
        raise ArtifactUnavailableError("The frozen operating-point contract differs.")
    models = evaluation_models(value["models"])
    default_model = value["default_model_id"]
    if default_model not in {model["id"] for model in models}:
        raise ArtifactUnavailableError("The frozen default model is missing from executed results.")
    return {"schema_version": 1, "complete": True, "scope": "reused_development", "artifact_mode": mode.name,
            "score_interpretation": "uncalibrated", "operating_point": {
                "normal_budget": 0.05, "comparison": ">", "selection": "calibration_normals_only"},
            "models": models, "default_model_id": default_model,
            "primary": primary_comparison(value["primary"], {model["id"] for model in models}),
            "artifact_sha256": sha256_file(root / artifact)}



def cohort_summary(value: dict[str, Any]) -> dict[str, Any]:
    """Project only executed demographic, source and label-evidence aggregates."""
    counts = ("records", "known_patients", "records_without_patient_identity")
    groups = ("target", "sources", "families", "label_evidence_rank", "label_evidence")
    age_counts = ("exact_count", "missing_count", "right_censored_over_89_count",
                  "exact_18_25_count", "exact_under_18_count")
    ages = {key: count(value["age"][key]) for key in age_counts}
    ages.update({key: number(value["age"][key], nullable=True)
                 for key in ("exact_min", "exact_max", "exact_mean", "exact_median")})
    return {**{key: count(value[key]) for key in counts},
            **{key: {str(name): count(total) for name, total in value[key].items()} for key in groups},
            "age": ages}


def cohort_selection(root: Path, directory: Path, scope: str,
                     names: tuple[str, str]) -> dict[str, Any]:
    """Require a completed cohort receipt and its fixed manifest/summary artifact bindings."""
    receipt = read_json(root / directory / "receipt.json")
    if receipt.get("complete") is not True or receipt.get("scope") != scope:
        raise ArtifactUnavailableError("The new cohorts have no completed assembly receipt.")
    for name in ("summary.json", *(name + ".csv" for name in names)):
        artifact = directory / name
        if receipt.get("artifacts", {}).get(artifact.as_posix()) != sha256_file(root / artifact):
            raise ArtifactUnavailableError("A cohort artifact differs from its completion receipt.")
    values = read_json(root / directory / "summary.json")
    return {"complete": True, "scope": scope, "seed": count(receipt["seed"]),
            "cohorts": {key: cohort_summary(values[key]) for key in names},
            "overlap_between_training_cohorts": {key: count(value) for key, value in
                                                 values["overlap_between_training_cohorts"].items()},
            "receipt_sha256": sha256_file(root / directory / "receipt.json")}


def load_cohorts(root: Path, mode: ArtifactMode = CACHED) -> dict[str, Any]:
    """Expose primary strictly young matched cohorts and separate secondary expanded-age cohorts."""
    primary = cohort_selection(root, mode.primary_cohorts, "strict_young_matched_cohorts",
                               ("strict_18_25", "general_matched_size"))
    young, control = (primary["cohorts"][name] for name in ("strict_18_25", "general_matched_size"))
    age = young["age"]
    if (young["records"] != control["records"] or age["exact_18_25_count"] != young["records"] or
            age["missing_count"] or age["right_censored_over_89_count"]):
        raise ArtifactUnavailableError("The primary strictly-young cohort contract differs.")
    if any(young[key] != control[key] for key in ("sources", "target", "label_evidence_rank")):
        raise ArtifactUnavailableError("The primary control is not matched on declared aggregate strata.")
    try:
        secondary = cohort_selection(root, mode.secondary_cohorts, "new_supervised_age_label_cohorts",
                                     ("younger", "all_age"))
    except (ArtifactUnavailableError, FileNotFoundError):
        secondary = None
    return {"complete": True, "artifact_mode": mode.name, "primary": primary, "secondary": secondary}
