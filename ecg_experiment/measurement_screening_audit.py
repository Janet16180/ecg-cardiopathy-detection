"""Independent arithmetic audit of saved measurement-screening predictions.

This module never fits a model, performs inference or reads a waveform. It does
not import experiment metric or calibration helpers. Arithmetic agreement cannot
validate waveform truth, native pathology annotations or clinical usefulness.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
import traceback
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.paths import to_stored

SEED = 20261018
DRAWS = 2000
TOLERANCE = 1e-12
ARMS = ("logistic", "hgb")
CLASSES = ("MI", "STTC", "CD", "HYP")
SAFE_SUFFIXES = {".py", ".md", ".json", ".csv", ".npz", ".joblib", ".toml", ".lock", ".ipynb", ".pdf"}


def compare(expected: Any, actual: Any, name: str, checks: list[dict[str, Any]]) -> None:
    """Append exact structural and tolerance-limited numeric comparisons.

    Parameters
    ----------
    expected, actual : Any
        Independently derived and saved values.
    name : str
        Dot-separated scientific field name.
    checks : list[dict[str, Any]]
        Mutable collection of audit checks.
    """
    if isinstance(expected, dict) and isinstance(actual, dict):
        compare(sorted(expected), sorted(actual), f"{name}.keys", checks)
        for key in expected.keys() & actual.keys():
            compare(expected[key], actual[key], f"{name}.{key}", checks)
        return
    if isinstance(expected, (float, np.floating)) and isinstance(actual, (int, float, np.number)):
        passed = bool(np.isfinite(expected) and np.isfinite(actual) and abs(expected - actual) <= TOLERANCE)
    else:
        passed = bool(expected == actual)
    check = {"name": name, "passed": passed}
    if not passed:
        check.update(expected=expected, actual=actual)
    checks.append(check)


def metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    """Calculate five record-level metrics using strict referral decisions.

    Parameters
    ----------
    y : np.ndarray
        Binary targets containing both classes.
    probability : np.ndarray
        Aligned finite predicted probabilities.
    threshold : float
        Refer scores strictly above this value.

    Returns
    -------
    dict[str, float]
        AUROC, average precision, sensitivity, specificity and Brier score.
    """
    positive = y == 1
    referred = probability > threshold
    return {
        "auroc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "sensitivity": float(np.mean(referred[positive])),
        "specificity": float(np.mean(~referred[~positive])),
        "brier": float(np.mean((probability - y) ** 2)),
    }


def patient_samples(patients: np.ndarray, y: np.ndarray, draws: int) -> Iterator[np.ndarray]:
    """Generate sorted-patient bootstrap samples, skipping single-class draws.

    Parameters
    ----------
    patients : np.ndarray
        Aligned patient IDs, preserving their saved numeric or string type.
    y : np.ndarray
        Binary targets.
    draws : int
        Total draws, including skipped samples.

    Yields
    ------
    np.ndarray
        All record positions for each sampled whole patient, in draw order.
    """
    unique, inverse = np.unique(patients, return_inverse=True)
    ordered = np.argsort(inverse, kind="stable")
    members = np.split(ordered, np.cumsum(np.bincount(inverse))[:-1])
    random = np.random.default_rng(SEED)
    for _ in range(draws):
        sample = np.concatenate([members[i] for i in random.integers(0, len(unique), len(unique))])
        if len(np.unique(y[sample])) == 2:
            yield sample


def threshold_record(scores: np.ndarray) -> dict[str, Any]:
    """Derive the frozen empirical five-percent normal referral budget.

    Parameters
    ----------
    scores : np.ndarray
        Normal calibration probabilities.

    Returns
    -------
    dict[str, Any]
        Exact order statistic, comparator and achieved calibration counts.
    """
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Normal calibration probabilities must be nonempty and finite")
    allowed = int(np.floor(len(scores) * 0.05))
    threshold = float(np.sort(scores)[len(scores) - allowed - 1])
    referred = int(np.sum(scores > threshold))
    return {
        "threshold": threshold,
        "normal_count": len(scores),
        "allowed_count": allowed,
        "referred_normal_count": referred,
        "normal_referral_rate": referred / len(scores),
        "comparator": ">",
        "order_index_descending": allowed,
    }


def miss_counts(table: pd.DataFrame, probability: np.ndarray, threshold: float) -> dict[str, Any]:
    """Count overlapping existing superclass positives and missed records.

    Parameters
    ----------
    table : pd.DataFrame
        Aligned targets and superclass flags.
    probability : np.ndarray
        Saved probabilities.
    threshold : float
        Strict referral threshold.

    Returns
    -------
    dict[str, Any]
        Positive and missed counts for each existing abnormal superclass.
    """
    positive = table.target.to_numpy() == 1
    missed = positive & (probability <= threshold)
    return {
        name: {
            "positive_count": int(np.sum(positive & table[name].eq(1).to_numpy())),
            "missed_count": int(np.sum(missed & table[name].eq(1).to_numpy())),
        }
        for name in CLASSES
    }


def interval_summary(observed: dict[str, float], sampled: list[dict[str, float]]) -> dict[str, Any]:
    """Build percentile intervals from independent bootstrap calculations.

    Parameters
    ----------
    observed : dict[str, float]
        Metrics on the full saved prediction table.
    sampled : list[dict[str, float]]
        Metrics of valid whole-patient samples.

    Returns
    -------
    dict[str, Any]
        Metric point values and linear percentile intervals.
    """
    return {
        name: {
            "value": value,
            "ci_low": float(np.percentile([row[name] for row in sampled], 2.5)),
            "ci_high": float(np.percentile([row[name] for row in sampled], 97.5)),
        }
        for name, value in observed.items()
    }


def paired_summary(observed: float, samples: list[float], draws: int) -> dict[str, float | int]:
    """Summarize paired patient-sampled metric differences.

    Parameters
    ----------
    observed : float
        Difference on the full saved evaluation table.
    samples : list[float]
        Differences from valid paired samples.
    draws : int
        Attempted bootstrap draws.

    Returns
    -------
    dict[str, float | int]
        Point difference, interval and exact sampling convention.
    """
    return {
        "difference": observed,
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
        "draws": draws,
        "skipped_draws": draws - len(samples),
        "seed": SEED,
    }


def analyze_predictions(
    table: pd.DataFrame, scores: dict[str, np.ndarray], thresholds: dict[str, float], draws: int = DRAWS
) -> dict[str, Any]:
    """Recompute all saved evaluation statistics without production helpers.

    Parameters
    ----------
    table : pd.DataFrame
        Saved aligned patient IDs, binary targets and superclass flags.
    scores : dict[str, np.ndarray]
        Saved probabilities for the logistic and boosting arms.
    thresholds : dict[str, float]
        Frozen strict referral thresholds.
    draws : int
        Whole-patient draws; production audits always use 2,000.

    Returns
    -------
    dict[str, Any]
        Both arm summaries and paired AUROC and sensitivity differences.
    """
    targets = table.target.to_numpy()
    if set(targets) != {0, 1} or table.patient_id.isna().any():
        raise ValueError("Saved evaluation requires both binary targets and complete patients")
    y = targets.astype(int)
    if any(
        len(p) != len(y) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) for p in scores.values()
    ):
        raise ValueError("Saved probabilities must align and lie in [0,1]")
    observed = {name: metrics(y, scores[name], thresholds[name]) for name in ARMS}
    samples: dict[str, list[dict[str, float]]] = {name: [] for name in ARMS}
    for index in patient_samples(table.patient_id.to_numpy(), y, draws):
        for name in ARMS:
            samples[name].append(metrics(y[index], scores[name][index], thresholds[name]))
    if not samples["hgb"]:
        raise ValueError("Every independent bootstrap sample was single-class")
    arms = {
        name: arm_result(table, scores[name], thresholds[name], observed[name], samples[name], draws)
        for name in ARMS
    }
    paired = {
        metric: paired_summary(
            observed["hgb"][metric] - observed["logistic"][metric],
            [
                first[metric] - second[metric]
                for first, second in zip(samples["hgb"], samples["logistic"], strict=True)
            ],
            draws,
        )
        for metric in ("auroc", "sensitivity")
    }
    return {"arms": arms, **paired}


def arm_result(
    table: pd.DataFrame,
    probability: np.ndarray,
    threshold: float,
    observed: dict[str, float],
    sampled: list[dict[str, float]],
    draws: int,
) -> dict[str, Any]:
    """Assemble one independent arm result in the saved scientific schema.

    Parameters
    ----------
    table : pd.DataFrame
        Evaluation identities, labels and class flags.
    probability : np.ndarray
        Saved model probabilities.
    threshold : float
        Strict referral threshold.
    observed : dict[str, float]
        Independently computed point metrics.
    sampled : list[dict[str, float]]
        Independently computed valid bootstrap metrics.
    draws : int
        Number of attempted patient draws.

    Returns
    -------
    dict[str, Any]
        Complete arm summary including referral intervals and class misses.
    """
    intervals = interval_summary(observed, sampled)
    specificity = intervals["specificity"]
    return {
        "metrics": intervals,
        "threshold": threshold,
        "comparator": ">",
        "draws": draws,
        "skipped_draws": draws - len(sampled),
        "seed": SEED,
        "normal_referral_rate": {
            "value": float(np.mean(probability[table.target.to_numpy() == 0] > threshold)),
            "ci_low": 1 - specificity["ci_high"],
            "ci_high": 1 - specificity["ci_low"],
        },
        "exploratory_superclass_misses": miss_counts(table, probability, threshold),
    }


def decision_gates(analysis: dict[str, Any]) -> list[bool]:
    """Apply the four unrounded frozen development criteria.

    Parameters
    ----------
    analysis : dict[str, Any]
        Independently reconstructed arm summaries and differences.

    Returns
    -------
    list[bool]
        AUROC gain, interval, sensitivity and normal-workload checks.
    """
    return [
        analysis["auroc"]["difference"] >= 0.01,
        analysis["auroc"]["ci_low"] > 0,
        analysis["sensitivity"]["difference"] >= -0.02,
        analysis["arms"]["hgb"]["normal_referral_rate"]["value"] <= 0.08,
    ]


def declared_file(path: str) -> Path:
    """Restrict receipt lookups to declared repository non-waveform artifacts.

    Parameters
    ----------
    path : str
        Repository-relative artifact path from a production receipt.

    Returns
    -------
    Path
        Safe local path, without following or loading any waveform payload.

    Raises
    ------
    ValueError
        If a path escapes the repository or has a waveform/unknown suffix.
    """
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix not in SAFE_SUFFIXES:
        raise ValueError(f"Receipt path is not an allowed non-waveform artifact: {path}")
    return ROOT / relative


def check_bindings(bindings: dict[str, str], label: str, checks: list[dict[str, Any]]) -> None:
    """Rehash only explicitly declared safe source and non-waveform input paths.

    Parameters
    ----------
    bindings : dict[str, str]
        Declared repository paths and SHA-256 values.
    label : str
        Audit namespace.
    checks : list[dict[str, Any]]
        Mutable check collection.
    """
    for name, expected in bindings.items():
        path = declared_file(name)
        compare(expected, sha256_file(path), f"{label}.{name}", checks)


def check_stage(path: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    """Verify one saved 042 receipt while excluding waveform raw-input hashes.

    Parameters
    ----------
    path : Path
        Existing stage directory.
    checks : list[dict[str, Any]]
        Mutable check collection.

    Returns
    -------
    dict[str, Any]
        Parsed completion receipt.
    """
    receipt = json.loads((path / "completion.json").read_text())
    check_bindings(receipt["source_sha256"], path.name, checks)
    check_bindings({receipt["protocol"]["path"]: receipt["protocol"]["sha256"]}, path.name, checks)
    for name, expected in receipt["artifacts"].items():
        if Path(name).name != name or Path(name).suffix not in SAFE_SUFFIXES:
            raise ValueError("Stage artifact name must be an immediate non-waveform file")
        compare(expected, sha256_file(path / name), f"{path.name}.artifact.{name}", checks)
    check_stage_inputs(path, receipt, checks)
    return receipt


def check_stage_inputs(path: Path, receipt: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    """Verify declared metadata and small-file hashes while excluding waveform bindings.

    Parameters
    ----------
    path : Path
        Stage directory, distinguishing PTB qualification from SPH preparation.
    receipt : dict[str, Any]
        Existing stage receipt with explicitly declared input hashes.
    checks : list[dict[str, Any]]
        Mutable check collection.
    """
    if "metadata_sha256" in receipt:
        name = (
            "data/raw/ptb-xl/1.0.3/ptbxl_database.csv"
            if path.name == "qualification"
            else "data/raw/sph/sph/17912441/metadata.csv"
        )
        check_bindings({name: receipt["metadata_sha256"]}, f"042.{path.name}.metadata", checks)
    if "cache_sha256" in receipt:
        check_bindings(
            {"outputs/avance2_fusion/caracteristicas_primarias_ee39c8f6f7.csv": receipt["cache_sha256"]},
            f"042.{path.name}.cache",
            checks,
        )
    for name, expected in receipt.get("small_files_md5", {}).items():
        source = declared_file(f"data/raw/sph/sph/{name}")
        with source.open("rb") as handle:
            actual = hashlib.file_digest(handle, "md5").hexdigest()
        compare(expected, actual, f"042.{path.name}.small_file.{name}", checks)


def check_042_gates(stages: dict[str, dict[str, Any]], checks: list[dict[str, Any]]) -> None:
    """Verify technical qualification and the frozen resource admission gates.

    Parameters
    ----------
    stages : dict[str, dict[str, Any]]
        Parsed qualification, profile, preparation and evaluation receipts.
    checks : list[dict[str, Any]]
        Mutable check collection.
    """
    qualification = stages["qualification"]
    expected = {
        "passed": True,
        "records": 32,
        "features": 130,
        "serialized_max_abs_difference": 0.0,
        "missingness_equal": True,
        "status_plausibility_equal": True,
        "flat_leads_equal": True,
    }
    for name, value in expected.items():
        compare(value, qualification[name], f"042.qualification.{name}", checks)
    compare(
        "ee39c8f6f7",
        qualification["notebook_extractor_sha256"][:10],
        "042.qualification.extractor_lineage",
        checks,
    )
    profile = stages["profile"]
    compare(64, profile["records"], "042.profile.records", checks)
    compare(True, profile["resource_gate_passed"], "042.profile.admitted", checks)
    compare(True, 120 <= profile["forecast_wall_seconds"] <= 3600, "042.profile.wall_ceiling", checks)
    compare(
        True, 0 < profile["forecast_storage_bytes"] <= 512 * 1024**2, "042.profile.storage_ceiling", checks
    )
    compare(
        True,
        sum(stage["wall_seconds"] for stage in stages.values()) <= 7200,
        "042.full_stage_wall_budget",
        checks,
    )


def audit_041(path: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit saved 041 v2 calibration, statistics, decisions and bindings.

    Parameters
    ----------
    path : Path
        Completed Experiment 041 v2 output directory.
    checks : list[dict[str, Any]]
        Mutable check collection.

    Returns
    -------
    dict[str, Any]
        Audited counts and reconstructed analysis.
    """
    result = json.loads((path / "result.json").read_text())
    expected_fields = [
        "status",
        "experiment",
        "execution_version",
        "provenance",
        "cohort",
        "integrity",
        "config",
        "counts",
        "thresholds",
        "development",
        "paired_auroc_hgb_minus_logistic",
        "paired_sensitivity_hgb_minus_logistic",
        "decision",
        "model_fit_status",
        "timing",
        "artifact_sha256",
        "artifacts",
        "total_seconds",
    ]
    compare(sorted(expected_fields), sorted(result), "041.result_fields", checks)
    compare("completed", result["status"], "041.status", checks)
    compare(2, result["execution_version"], "041.execution_version", checks)
    check_bindings(result["provenance"]["files_sha256"], "041.input_sources", checks)
    check_bindings(result["artifact_sha256"], "041.artifacts", checks)
    with np.load(path / "predictions.npz", allow_pickle=False) as payload:
        table = pd.DataFrame({key: payload[key] for key in payload.files})
    compare(True, bool(table.ecg_id.is_unique), "041.unique_ecg_ids", checks)
    compare([8, 9], sorted(table.strat_fold.unique().tolist()), "041.saved_folds", checks)
    compare(1, int(table.groupby("patient_id").strat_fold.nunique().max()), "041.patient_roles", checks)
    calibration, evaluation = (table.loc[table.strat_fold == fold] for fold in (8, 9))
    thresholds = {}
    for name in ARMS:
        derived = threshold_record(calibration.loc[calibration.target == 0, name].to_numpy())
        compare(derived, result["thresholds"][name], f"041.thresholds.{name}", checks)
        thresholds[name] = derived["threshold"]
    scores = {name: evaluation[name].to_numpy() for name in ARMS}
    analysis = analyze_predictions(evaluation, scores, thresholds)
    compare(analysis["arms"], result["development"], "041.development", checks)
    compare(analysis["auroc"], result["paired_auroc_hgb_minus_logistic"], "041.auroc_difference", checks)
    compare(
        analysis["sensitivity"],
        result["paired_sensitivity_hgb_minus_logistic"],
        "041.sensitivity_difference",
        checks,
    )
    gates = dict(
        zip(
            (
                "auroc_gain_at_least_0.01",
                "auroc_ci_low_positive",
                "sensitivity_difference_at_least_minus_0.02",
                "hgb_normal_referral_at_most_0.08",
            ),
            decision_gates(analysis),
            strict=True,
        )
    )
    compare(
        {"gates": gates, "outcome": "advance_development" if all(gates.values()) else "do_not_advance"},
        result["decision"],
        "041.decision",
        checks,
    )
    check_041_counts(result, table, checks)
    integrity = json.loads((path / "integrity.json").read_text())
    compare("passed", integrity["status"], "041.integrity_status", checks)
    historical = json.loads(declared_file("outputs/avance3_baseline/baseline_metricas.json").read_text())
    expected_metrics = {name: historical["validacion"][name] for name in ("AUROC", "AUPRC")}
    compare(expected_metrics, integrity["validation"], "041.historical_metric_binding", checks)
    compare(
        True,
        integrity["prediction_max_absolute_difference"] <= TOLERANCE,
        "041.historical_prediction_tolerance",
        checks,
    )
    return {"saved_rows": len(table), "evaluation_rows": len(evaluation), "analysis": analysis}


def count_role(table: pd.DataFrame) -> dict[str, int]:
    """Count saved ECGs, patients and binary labels for one role.

    Parameters
    ----------
    table : pd.DataFrame
        One aligned binary-label role.

    Returns
    -------
    dict[str, int]
        ECG, patient, positive and normal counts.
    """
    return {
        "ecgs": len(table),
        "patients": int(table.patient_id.nunique()),
        "positive": int(table.target.sum()),
        "normal": int((table.target == 0).sum()),
    }


def check_041_counts(result: dict[str, Any], predictions: pd.DataFrame, checks: list[dict[str, Any]]) -> None:
    """Verify role accounting and exact prediction identities against the declared CSV.

    Parameters
    ----------
    result : dict[str, Any]
        Saved 041 result.
    predictions : pd.DataFrame
        Saved calibration and development payload.
    checks : list[dict[str, Any]]
        Mutable check collection.
    """
    source = declared_file("outputs/avance2_fusion/ptbxl_caracteristicas_seleccionadas.csv")
    columns = ["ecg_id", "patient_id", "strat_fold", "target", *CLASSES]
    selected = pd.read_csv(source, usecols=columns)
    selected = selected.loc[selected.strat_fold <= 9]
    for name, mask in {
        "fit": selected.strat_fold <= 7,
        "calibration": selected.strat_fold == 8,
        "development": selected.strat_fold == 9,
    }.items():
        compare(count_role(selected.loc[mask]), result["counts"][name], f"041.counts.{name}", checks)
    expected = selected.loc[selected.strat_fold.isin([8, 9]), columns].reset_index(drop=True)
    compare(
        True,
        bool(np.array_equal(expected.to_numpy(), predictions[columns].to_numpy())),
        "041.saved_identity_labels_match_source",
        checks,
    )


def expected_role(patient: str) -> str:
    """Reconstruct the frozen SHA-256 patient role without production helpers.

    Parameters
    ----------
    patient : str
        Original SPH patient string.

    Returns
    -------
    str
        Calibration or evaluation from the first digest byte.
    """
    digest = hashlib.sha256(("experiment042:20261018:" + patient).encode("utf-8")).digest()
    return "calibration" if digest[0] < 128 else "evaluation"


def read_042_tables(path: Path, checks: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Load saved predictions and normal calibration tables with exact decimal parsing.

    Parameters
    ----------
    path : Path
        Completed 042 output root.
    checks : list[dict[str, Any]]
        Mutable check collection.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, pd.DataFrame]]
        Evaluation and per-arm normal calibration tables.
    """
    options = {"index_col": "ecg_id", "dtype": {"patient_id": str}, "float_precision": "round_trip"}
    evaluation = pd.read_csv(path / "evaluation/predictions.csv", **options)
    normal = {
        name: pd.read_csv(path / f"evaluation/{name}_normal_calibration.csv", **options) for name in ARMS
    }
    compare(True, bool(evaluation.index.is_unique), "042.unique_ecg_ids", checks)
    for name, frame in normal.items():
        compare(
            True,
            bool(frame.index.is_unique and frame.target.eq(0).all()),
            f"042.normal_schema.{name}",
            checks,
        )
        compare(
            0, len(set(frame.patient_id) & set(evaluation.patient_id)), f"042.patient_disjoint.{name}", checks
        )
    return evaluation, normal


def audit_042(path: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit external predictions, local calibration, transport, decisions and receipts.

    Parameters
    ----------
    path : Path
        Completed Experiment 042 output root.
    checks : list[dict[str, Any]]
        Mutable check collection.

    Returns
    -------
    dict[str, Any]
        Saved counts and independently reconstructed analysis.
    """
    stages = {
        stage: check_stage(path / stage, checks)
        for stage in ("qualification", "profile", "preparation", "evaluation")
    }
    check_042_gates(stages, checks)
    receipt = stages["evaluation"]
    check_bindings(receipt["predecessor_bindings"], "042.predecessor", checks)
    compare(
        receipt["models"]["sha256"],
        sha256_file(declared_file(receipt["models"]["path"])),
        "042.models",
        checks,
    )
    compare(
        receipt["preparation_receipt_sha256"],
        sha256_file(path / "preparation/completion.json"),
        "042.preparation_binding",
        checks,
    )
    result = json.loads((path / "evaluation/results.json").read_text())
    expected_fields = [
        "primary",
        "sensitivity_difference",
        "arms",
        "local_thresholds",
        "local_calibration_ties",
        "ptb_threshold_transport",
        "decision_checks",
        "decision",
        "counts",
        "misses_by_superclass",
    ]
    compare(sorted(expected_fields), sorted(result), "042.result_fields", checks)
    predecessor = json.loads((declared_file(receipt["models"]["path"]).parent / "result.json").read_text())
    for name in ARMS:
        compare(
            predecessor["thresholds"][name]["threshold"],
            result["ptb_threshold_transport"][name]["ptb_threshold"],
            f"042.original_ptb_threshold.{name}",
            checks,
        )
    evaluation, normal = read_042_tables(path, checks)
    thresholds = {}
    for name in ARMS:
        derived = threshold_record(normal[name].probability.to_numpy())
        compare(derived, result["local_thresholds"][name], f"042.thresholds.{name}", checks)
        thresholds[name] = derived["threshold"]
        compare(
            int(np.sum(normal[name].probability == thresholds[name])),
            result["local_calibration_ties"][name],
            f"042.ties.{name}",
            checks,
        )
        compare(
            True,
            bool(
                np.array_equal(
                    evaluation[f"{name}_referred"], evaluation[f"{name}_probability"] > thresholds[name]
                )
            ),
            f"042.decisions.{name}",
            checks,
        )
    scores = {name: evaluation[f"{name}_probability"].to_numpy() for name in ARMS}
    analysis = analyze_predictions(evaluation, scores, thresholds)
    compare(analysis["arms"], result["arms"], "042.arms", checks)
    compare(analysis["auroc"], result["primary"], "042.primary", checks)
    compare(analysis["sensitivity"], result["sensitivity_difference"], "042.sensitivity_difference", checks)
    check_042_decisions(result, analysis, checks)
    check_042_transport(result, evaluation, scores, checks)
    check_042_counts(path, result, evaluation, normal, checks)
    return {
        "evaluation_rows": len(evaluation),
        "analysis": analysis,
        "qualification_waveform_hashes_intentionally_not_rehashed": True,
    }


def check_042_decisions(
    result: dict[str, Any], analysis: dict[str, Any], checks: list[dict[str, Any]]
) -> None:
    """Verify every external gate and overlapping missed-superclass aggregate.

    Parameters
    ----------
    result : dict[str, Any]
        Saved external result.
    analysis : dict[str, Any]
        Independently computed external statistics.
    checks : list[dict[str, Any]]
        Mutable check collection.
    """
    gates = dict(
        zip(
            (
                "auroc_delta_at_least_0_01",
                "auroc_lower_bound_positive",
                "sensitivity_delta_at_least_minus_0_02",
                "hgb_normal_referral_at_most_0_08",
            ),
            decision_gates(analysis),
            strict=True,
        )
    )
    compare(gates, result["decision_checks"], "042.decision_checks", checks)
    compare(
        "support_external_development" if all(gates.values()) else "do_not_advance_external",
        result["decision"],
        "042.decision",
        checks,
    )
    missed = {
        name: {
            code: analysis["arms"][name]["exploratory_superclass_misses"][code]["missed_count"]
            for code in CLASSES
        }
        for name in ARMS
    }
    compare(missed, result["misses_by_superclass"], "042.misses_by_superclass", checks)


def check_042_transport(
    result: dict[str, Any], table: pd.DataFrame, scores: dict[str, np.ndarray], checks: list[dict[str, Any]]
) -> None:
    """Verify descriptive transport metrics at the original saved PTB thresholds.

    Parameters
    ----------
    result : dict[str, Any]
        Saved external result.
    table : pd.DataFrame
        Evaluation targets and patient identities.
    scores : dict[str, np.ndarray]
        Saved probabilities of both arms.
    checks : list[dict[str, Any]]
        Mutable check collection.
    """
    for name in ARMS:
        saved = result["ptb_threshold_transport"][name]
        observed = metrics(table.target.to_numpy(), scores[name], saved["ptb_threshold"])
        expected = {
            "ptb_threshold": saved["ptb_threshold"],
            "comparator": ">",
            "sensitivity": observed["sensitivity"],
            "normal_referral_rate": 1 - observed["specificity"],
        }
        compare(expected, saved, f"042.transport.{name}", checks)


def check_042_counts(
    path: Path,
    result: dict[str, Any],
    evaluation: pd.DataFrame,
    normals: dict[str, pd.DataFrame],
    checks: list[dict[str, Any]],
) -> None:
    """Validate saved patient roles, duplicate retention, identities and all external counts.

    Parameters
    ----------
    path : Path
        External experiment output root.
    result : dict[str, Any]
        Saved external result.
    evaluation : pd.DataFrame
        Saved evaluation predictions.
    normals : dict[str, pd.DataFrame]
        Saved normal calibration predictions.
    checks : list[dict[str, Any]]
        Mutable check collection.
    """
    columns = [
        "ecg_id",
        "patient_id",
        "role",
        "primary",
        "use_evaluation",
        "quality_fail",
        "duplicate_status",
        "signal_sha256",
        "aha_codes",
        *CLASSES,
    ]
    prepared = pd.read_csv(
        path / "preparation/measurements.csv", usecols=columns, index_col="ecg_id", dtype={"patient_id": str}
    )
    compare(
        True, bool(prepared.role.equals(prepared.patient_id.map(expected_role))), "042.role_hashes", checks
    )
    usable = prepared.loc[prepared.use_evaluation & prepared.primary.notna()]
    calibration = usable.loc[usable.role == "calibration"]
    expected_eval = usable.loc[usable.role == "evaluation"].rename(columns={"primary": "target"})
    identity = ["patient_id", "target", *CLASSES, "quality_fail"]
    compare(True, bool(np.array_equal(expected_eval.index, evaluation.index)), "042.evaluation_ids", checks)
    compare(
        True,
        bool(np.array_equal(expected_eval[identity].to_numpy(), evaluation[identity].to_numpy())),
        "042.evaluation_labels",
        checks,
    )
    check_042_retention(path, prepared, checks)
    expected_normal = calibration.loc[calibration.primary == 0]
    for name, frame in normals.items():
        compare(
            True,
            bool(
                np.array_equal(expected_normal.index, frame.index)
                and np.array_equal(expected_normal.patient_id, frame.patient_id)
            ),
            f"042.normal_ids.{name}",
            checks,
        )
    expected = {
        "all_rows": len(prepared),
        "calibration_labelled": len(calibration),
        "calibration_normals": int((calibration.primary == 0).sum()),
        "calibration_positive": int((calibration.primary == 1).sum()),
        "calibration_patients": int(calibration.patient_id.nunique()),
        "retained_unique_or_kept": int(prepared.use_evaluation.sum()),
        "quality_failure_evaluation_rows": int(expected_eval.quality_fail.sum()),
        "evaluation_labelled": len(expected_eval),
        "evaluation_patients": int(expected_eval.patient_id.nunique()),
        "evaluation_positive": int(expected_eval.target.sum()),
        "evaluation_negative": int((expected_eval.target == 0).sum()),
        "undefined_primary": int(prepared.primary.isna().sum()),
    }
    compare(expected, result["counts"], "042.counts", checks)


def check_042_retention(path: Path, prepared: pd.DataFrame, checks: list[dict[str, Any]]) -> None:
    """Reconstruct saved-window deduplication and verify the pre-inference role table.

    Parameters
    ----------
    path : Path
        External output root containing its prepared manifest and frozen roles.
    prepared : pd.DataFrame
        Saved per-window hashes, original AHA strings, roles and retention flags.
    checks : list[dict[str, Any]]
        Mutable check collection.
    """
    ordered = prepared.sort_index()
    grouped = ordered.groupby("signal_sha256")
    count = grouped.aha_codes.transform("size")
    conflicting = grouped.aha_codes.transform("nunique") > 1
    first = ~ordered.signal_sha256.duplicated()
    expected = np.where(
        count == 1,
        "unique",
        np.where(conflicting, "dropped_conflict", np.where(first, "kept", "dropped_copy")),
    )
    compare(
        True,
        bool(np.array_equal(expected, ordered.duplicate_status.to_numpy())),
        "042.duplicate_policy_from_saved_hashes_and_aha",
        checks,
    )
    compare(
        True,
        bool(np.array_equal(np.isin(expected, ["unique", "kept"]), ordered.use_evaluation)),
        "042.retention_policy",
        checks,
    )
    roles = pd.read_csv(path / "preparation/patient_roles.csv", index_col="ecg_id", dtype={"patient_id": str})
    compare(
        True,
        bool(
            np.array_equal(prepared.index, roles.index)
            and np.array_equal(prepared[["patient_id", "role"]].to_numpy(), roles.to_numpy())
        ),
        "042.patient_roles_bound_before_inference",
        checks,
    )


def run_audit(experiment041: Path, experiment042: Path, output: Path) -> dict[str, Any]:
    """Write an exclusive independent saved-output audit, preserving every mismatch.

    Parameters
    ----------
    experiment041, experiment042 : Path
        Completed production experiment directories.
    output : Path
        New audit directory; existing directories are rejected.

    Returns
    -------
    dict[str, Any]
        Audit status, all checks, mismatches, source identities and timing.
    """
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    checks: list[dict[str, Any]] = []
    receipt: dict[str, Any] = {
        "scope": (
            "Saved-output arithmetic verification only; no model training, inference, "
            "waveform truth or native pathology validation."
        ),
        "draws": DRAWS,
        "seed": SEED,
        "numeric_absolute_tolerance": TOLERANCE,
    }
    try:
        with threadpool_limits(limits=1):
            receipt["041"] = audit_041(experiment041, checks)
            receipt["042"] = audit_042(experiment042, checks)
    except Exception:
        checks.append(
            {
                "name": "audit_execution",
                "passed": False,
                "traceback": traceback.format_exc().replace(str(ROOT), "."),
            }
        )
    receipt.update(
        {
            "status": "passed" if checks and all(item["passed"] for item in checks) else "failed",
            "checks": checks,
            "check_count": len(checks),
            "mismatches": [item for item in checks if not item["passed"]],
            "wall_seconds": time.monotonic() - started,
            "git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "source_sha256": {
                to_stored(Path(__file__)): sha256_file(Path(__file__)),
                "scripts/reports/audit_measurement_screening.py": sha256_file(
                    ROOT / "scripts/reports/audit_measurement_screening.py"
                ),
            },
            "input_sha256": {
                to_stored(path): sha256_file(path)
                for path in (
                    experiment041 / "result.json",
                    experiment041 / "predictions.npz",
                    experiment042 / "evaluation/results.json",
                    experiment042 / "evaluation/predictions.csv",
                )
                if path.exists()
            },
        }
    )
    write_json_atomic(output / "audit.json", receipt)
    write_json_atomic(
        output / "completion.json",
        {
            "status": receipt["status"],
            "check_count": len(checks),
            "mismatch_count": len(receipt["mismatches"]),
            "artifacts": {to_stored(output / "audit.json"): sha256_file(output / "audit.json")},
        },
    )
    return receipt
