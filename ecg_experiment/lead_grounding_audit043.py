"""Independent arithmetic verification of saved Experiment 043 outputs.

This audit is conditional on the saved targets, offered masks and reviewed source.
It does not parse annotation payloads, unpickle models, fit models, run inference,
read waveforms or validate native pathology truth or classifier explanations.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
import traceback
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.paths import to_stored

SEED = 20261018
DRAWS = 2000
TOLERANCE = 1e-12
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
MODELS = ("context_hgb", "wave_hgb", "context_lookup")
METRICS = ("average_precision", "brier", "exact_set_accuracy")
CANDIDATE_MEASUREMENTS = ("std", "ptp", "kurt", "r_amp", "s_depth", "q_depth", "qrs_area", "st", "t_amp")
RAW_FEATURES = (
    "noise_drift_frac",
    "noise_hf_frac",
    "noise_powerline_frac",
    "sp_low_rel",
    "sp_mid_rel",
    "sp_high_rel",
    "sp_entropy",
    "sp_centroid_hz",
    *(f"{measurement}_{lead}" for lead in LEADS for measurement in ("std", "ptp", "kurt")),
    "n_beats",
    "beat_similarity",
    "rr_mean_ms",
    "rr_sdnn_ms",
    "rr_rmssd_ms",
    "rr_range_rel",
    "hr_bpm",
    "qrs_dur_ms",
    "qt_ms",
    "pq_peak_ms",
    "p_rms_mv",
    "t_peak_rms_mv",
    "pr_ms",
    "p_amp_II",
    *(
        f"{measurement}_{lead}"
        for lead in LEADS
        for measurement in ("r_amp", "s_depth", "q_depth", "qrs_area", "st", "t_amp")
    ),
)
SCOPE = (
    "Arithmetic verification conditional on saved targets/masks and reviewed source only; "
    "no independent model fitting, inference, annotation regeneration, waveform or native pathology "
    "validation, or classifier explanation."
)
SAFE_SUFFIXES = {".json", ".jsonl", ".csv", ".npz", ".joblib", ".py", ".md", ".toml", ".lock", ".ipynb"}


def compare(expected: Any, actual: Any, name: str, checks: list[dict[str, Any]]) -> None:
    """Record structural and recursive numeric agreement with fixed absolute tolerance.

    Parameters
    ----------
    expected, actual : Any
        Independently calculated and saved scientific values.
    name : str
        Stable name of the compared field.
    checks : list[dict[str, Any]]
        Mutable collection of checks and explicit mismatches.
    """
    if isinstance(expected, dict) and isinstance(actual, dict):
        compare(sorted(expected), sorted(actual), f"{name}.keys", checks)
        for key in sorted(expected.keys() & actual.keys()):
            compare(expected[key], actual[key], f"{name}.{key}", checks)
        return
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        compare(len(expected), len(actual), f"{name}.length", checks)
        for index, (first, second) in enumerate(zip(expected, actual, strict=False)):
            compare(first, second, f"{name}[{index}]", checks)
        return
    if isinstance(expected, (float, np.floating)) and isinstance(actual, (int, float, np.number)):
        passed = bool(np.isfinite(expected) and np.isfinite(actual) and abs(expected - actual) <= TOLERANCE)
    else:
        passed = bool(expected == actual)
    check = {"name": name, "passed": passed}
    if not passed:
        check.update(expected=expected, actual=actual)
    checks.append(check)


def validate_payload(payload: dict[str, np.ndarray], frozen_counts: bool = True) -> dict[str, int]:
    """Require canonical offered-only targets and predictions with complete saved identities.

    Parameters
    ----------
    payload : dict[str, np.ndarray]
        Saved NPZ arrays, loaded without pickle support.
    frozen_counts : bool
        Require the protocol's exact development accounting in production audits.

    Returns
    -------
    dict[str, int]
        Verified query, ECG, patient and informative-subset counts.

    Raises
    ------
    ValueError
        If the schema, unknown mask, identities or frozen accounting differs.
    """
    identities = ("ecg_id", "patient_id", "strat_fold", "target_dx", "question", "informative")
    expected = {*identities, "offered", "target", "lead_order"}
    expected.update(f"{name}__{metric}" for name in MODELS for metric in (*METRICS, "probability"))
    if set(payload) != expected:
        raise ValueError("Saved prediction payload has unexpected or missing fields")
    count = len(payload["ecg_id"])
    if not count or any(payload[name].shape != (count,) for name in identities):
        raise ValueError("Saved query identities must be nonempty aligned vectors")
    if payload["lead_order"].tolist() != list(LEADS):
        raise ValueError("Saved lead order differs from the frozen canonical order")
    validate_masks(payload, count)
    validate_identity(payload)
    counts = {
        "query_count": count,
        "ecg_count": len(np.unique(payload["ecg_id"])),
        "patient_count": len(np.unique(payload["patient_id"])),
        "informative_query_count": int(payload["informative"].sum()),
        "informative_patient_count": len(np.unique(payload["patient_id"][payload["informative"]])),
    }
    expected_counts = {
        "query_count": 137,
        "ecg_count": 100,
        "patient_count": 97,
        "informative_query_count": 69,
        "informative_patient_count": 56,
    }
    if frozen_counts and counts != expected_counts:
        raise ValueError("Saved development accounting differs from the frozen 137/100/97 and 69/56 counts")
    return counts


def validate_masks(payload: dict[str, np.ndarray], count: int) -> None:
    """Reject unknown targets treated as negatives and predictions on unoffered leads.

    Parameters
    ----------
    payload : dict[str, np.ndarray]
        Saved offered masks, targets and per-arm probabilities.
    count : int
        Number of saved queries.

    Raises
    ------
    ValueError
        If masks, target cardinalities or finite probability ranges are invalid.
    """
    offered, target = payload["offered"], payload["target"]
    if offered.shape != (count, 12) or target.shape != offered.shape or offered.dtype != bool:
        raise ValueError("Offered and target arrays must have aligned query-by-12 shapes")
    if payload["informative"].dtype != bool or target.dtype != np.int8:
        raise ValueError("Informative masks must be boolean and target storage must be int8")
    if not np.all(target[~offered] == -1) or not np.isin(target[offered], [0, 1]).all():
        raise ValueError("Unoffered targets must be unknown -1; offered targets must be known binary values")
    positives = np.sum(target == 1, axis=1)
    if np.any(positives == 0) or np.any(positives >= offered.sum(axis=1)):
        raise ValueError("Empty or all-positive offered targets violate frozen qualification")
    for name in MODELS:
        validate_arm_arrays(payload, name, count)


def validate_arm_arrays(payload: dict[str, np.ndarray], name: str, count: int) -> None:
    """Validate one saved model's offered probabilities and query-metric vectors.

    Parameters
    ----------
    payload : dict[str, np.ndarray]
        Saved prediction arrays.
    name : str
        One frozen model name.
    count : int
        Number of queries.
    """
    offered = payload["offered"]
    probability = payload[f"{name}__probability"]
    if probability.shape != offered.shape or not np.isnan(probability[~offered]).all():
        raise ValueError("Unoffered model probabilities must be NaN with aligned query-by-12 shape")
    known = probability[offered]
    if not np.isfinite(known).all() or np.any((known < 0) | (known > 1)):
        raise ValueError("Offered probabilities must be finite and within [0,1]")
    for metric in METRICS:
        values = payload[f"{name}__{metric}"]
        if values.shape != (count,) or not np.isfinite(values).all():
            raise ValueError("Saved query metrics must be aligned finite vectors")


def validate_identity(payload: dict[str, np.ndarray]) -> None:
    """Check fold-nine-only identities and unique exact-context/offered-set query keys.

    Parameters
    ----------
    payload : dict[str, np.ndarray]
        Saved query identities, contexts and offered masks.

    Raises
    ------
    ValueError
        If identities are malformed, repeated inconsistently or include another fold.
    """
    if not np.all(payload["strat_fold"] == 9):
        raise ValueError("Only fold-nine queries may appear in saved evaluation predictions")
    for name in ("ecg_id", "patient_id"):
        values = payload[name]
        if not np.isfinite(values).all() or np.any(values <= 0) or np.any(values != np.floor(values)):
            raise ValueError("Saved ECG and patient identities must be finite positive integers")
    keys, patients = [], {}
    for index, ecg_id in enumerate(payload["ecg_id"]):
        patient = int(payload["patient_id"][index])
        if ecg_id in patients and patients[ecg_id] != patient:
            raise ValueError("An ECG has inconsistent saved patient identities")
        patients[ecg_id] = patient
        context = (payload["target_dx"][index], payload["question"][index])
        if not all(isinstance(value, (str, np.str_)) for value in context):
            raise ValueError("Saved contexts must preserve diagnosis and question strings")
        keys.append((int(ecg_id), *context, tuple(np.flatnonzero(payload["offered"][index]))))
    if len(set(keys)) != len(keys):
        raise ValueError("Saved ECG/context/canonical-offered-set query keys are not unique")


def query_statistics(payload: dict[str, np.ndarray], name: str) -> dict[str, np.ndarray]:
    """Independently compute tie-aware AP, Brier and strict-threshold exact-set accuracy.

    Parameters
    ----------
    payload : dict[str, np.ndarray]
        Validated saved targets, offered masks and probabilities.
    name : str
        Model whose saved probabilities are audited.

    Returns
    -------
    dict[str, np.ndarray]
        One metric value per query, excluding every unoffered lead.
    """
    result: dict[str, list[float]] = {metric: [] for metric in METRICS}
    for offered, target, probability in zip(
        payload["offered"], payload["target"], payload[f"{name}__probability"], strict=True
    ):
        y, scores = target[offered], probability[offered]
        result["average_precision"].append(float(average_precision_score(y, scores)))
        result["brier"].append(float(np.mean((scores - y) ** 2)))
        result["exact_set_accuracy"].append(float(np.array_equal(scores > 0.5, y.astype(bool))))
    return {metric: np.asarray(values) for metric, values in result.items()}


def patient_statistics(patients: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average each patient's queries before assigning equal weight to patients.

    Parameters
    ----------
    patients : np.ndarray
        Patient identity of each included query.
    values : np.ndarray
        Aligned per-query metric values.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Sorted unique patients and their query means.
    """
    if patients.ndim != 1 or values.shape != patients.shape or not len(patients):
        raise ValueError("Patient means require nonempty aligned query vectors")
    unique = np.unique(patients)
    means = np.array([np.mean(values[patients == patient]) for patient in unique])
    return unique, means


def paired_difference(
    patients: np.ndarray, first: np.ndarray, second: np.ndarray, draws: int = DRAWS
) -> dict[str, Any]:
    """Bootstrap sorted patients' paired query means with the frozen independent RNG.

    Parameters
    ----------
    patients : np.ndarray
        Query patient identities.
    first, second : np.ndarray
        Paired query-level AP arrays.
    draws : int
        Number of patient draws; fixed at 2,000 for real audits.

    Returns
    -------
    dict[str, Any]
        Unrounded paired difference and linear percentile interval.
    """
    ids, first_means = patient_statistics(patients, first)
    _, second_means = patient_statistics(patients, second)
    difference = first_means - second_means
    random = np.random.default_rng(SEED)
    sampled = np.array([np.mean(difference[random.integers(0, len(ids), len(ids))]) for _ in range(draws)])
    low, high = np.percentile(sampled, [2.5, 97.5])
    return {
        "difference": float(np.mean(difference)),
        "ci_low": float(low),
        "ci_high": float(high),
        "draws": draws,
        "skipped_draws": 0,
        "seed": SEED,
        "patients": len(ids),
    }


def decision(primary: dict[str, Any], informative: dict[str, Any], lookup: dict[str, Any]) -> dict[str, Any]:
    """Apply the exact four frozen grounding criteria without control selection.

    Parameters
    ----------
    primary, informative, lookup : dict[str, Any]
        Independent wave-minus-context, informative and wave-minus-lookup contrasts.

    Returns
    -------
    dict[str, Any]
        Gate booleans and their fixed conjunction.
    """
    gates = {
        "primary_gain_at_least_0.05": primary["difference"] >= 0.05,
        "primary_ci_low_positive": primary["ci_low"] > 0,
        "informative_primary_gain_positive": informative["difference"] > 0,
        "wave_lookup_gain_nonnegative": lookup["difference"] >= 0,
    }
    return {
        "outcome": "advance_benchmark_grounding" if all(gates.values()) else "do_not_advance_grounding",
        "gates": gates,
    }


def reconstruct(payload: dict[str, np.ndarray], draws: int = DRAWS) -> dict[str, Any]:
    """Reconstruct every saved scientific endpoint and patient aggregate independently.

    Parameters
    ----------
    payload : dict[str, np.ndarray]
        Validated predictions and identity arrays.
    draws : int
        Paired patient bootstrap draws.

    Returns
    -------
    dict[str, Any]
        Summary, query metrics and patient means for complete saved-output comparison.
    """
    patients, informative = payload["patient_id"], payload["informative"]
    query = {name: query_statistics(payload, name) for name in MODELS}
    saved_means, models = {"patient_id": np.unique(patients)}, {}
    for name, metrics in query.items():
        models[name] = {}
        for metric, values in metrics.items():
            _, means = patient_statistics(patients, values)
            saved_means[f"{name}__{metric}"] = means
            models[name][metric] = float(np.mean(means))
        _, means = patient_statistics(patients[informative], metrics["average_precision"][informative])
        models[name]["informative_average_precision"] = float(np.mean(means))
    ap = {name: metrics["average_precision"] for name, metrics in query.items()}
    primary = paired_difference(patients, ap["wave_hgb"], ap["context_hgb"], draws)
    guard = paired_difference(
        patients[informative], ap["wave_hgb"][informative], ap["context_hgb"][informative], draws
    )
    lookup = {
        name: paired_difference(patients, ap[name], ap["context_lookup"], draws)
        for name in ("context_hgb", "wave_hgb")
    }
    summary = {
        "models": models,
        "primary_wave_minus_context": primary,
        "informative_wave_minus_context": guard,
        "contrasts_to_lookup": lookup,
        "contexts": context_statistics(payload, query),
        "query_count": len(patients),
        "patient_count": len(np.unique(patients)),
        "informative_query_count": int(informative.sum()),
        "informative_patient_count": len(np.unique(patients[informative])),
        "decision": decision(primary, guard, lookup["wave_hgb"]),
    }
    return {"summary": summary, "query_metrics": query, "patient_means": saved_means}


def context_statistics(payload: dict[str, np.ndarray], query: dict[str, Any]) -> list[dict[str, Any]]:
    """Reconstruct descriptive patient-macro AP within each exact diagnosis/question pair.

    Parameters
    ----------
    payload : dict[str, np.ndarray]
        Query identities and exact condition/question strings.
    query : dict[str, Any]
        Independently calculated query metrics for all arms.

    Returns
    -------
    list[dict[str, Any]]
        Canonically sorted descriptive context summaries.
    """
    contexts = list(zip(payload["target_dx"].tolist(), payload["question"].tolist(), strict=True))
    summaries = []
    for diagnosis, question in sorted(set(contexts)):
        mask = np.array([context == (diagnosis, question) for context in contexts])
        patients = payload["patient_id"][mask]
        scores = {
            name: float(np.mean(patient_statistics(patients, metrics["average_precision"][mask])[1]))
            for name, metrics in query.items()
        }
        summaries.append(
            {
                "target_dx": diagnosis,
                "question": question,
                "queries": int(mask.sum()),
                "patients": len(np.unique(patients)),
                "aggregation": "patient_macro_within_context",
                "average_precision": scores,
            }
        )
    return summaries


def verify_qualification(qualification: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    """Check the frozen setup accounting without reading annotation answers or fitting data.

    Parameters
    ----------
    qualification : dict[str, Any]
        Saved prefit qualification receipt, conditional on reviewed source.
    checks : list[dict[str, Any]]
        Mutable comparison collection.
    """
    roles = {"fit": (1117, 823, 776), "unused": (154, 110, 99), "development": (137, 100, 97)}
    for role, values in roles.items():
        for name, expected in zip(("queries", "ecgs", "patients"), values, strict=True):
            compare(expected, qualification["roles"][role][name], f"qualification.{role}.{name}", checks)
        histograms = qualification["roles"][role]
        for name in ("answer_cardinality", "offered_cardinality"):
            histogram = histograms[name]
            compare(values[0], sum(histogram.values()), f"qualification.{role}.{name}.total", checks)
            compare(
                True,
                all(0 < int(key) <= 12 and count > 0 for key, count in histogram.items()),
                f"qualification.{role}.{name}.valid",
                checks,
            )
    fields = {
        "training_contexts": 23,
        "constant_training_contexts": 0,
        "informative_development_queries": 69,
        "informative_development_patients": 56,
    }
    for name, expected in fields.items():
        compare(expected, qualification[name], f"qualification.{name}", checks)
    for name in (
        "duplicate_queries",
        "conflicting_queries",
        "missing_joins",
        "empty_targets",
        "all_positive_targets",
    ):
        compare(0, qualification["parser"][name], f"qualification.parser.{name}", checks)


def verify_bindings(bindings: dict[str, str], checks: list[dict[str, Any]], namespace: str) -> None:
    """Rehash declared local artifacts without decoding models, waveforms or annotation payloads.

    Parameters
    ----------
    bindings : dict[str, str]
        Declared repository-relative paths and SHA-256 values.
    checks : list[dict[str, Any]]
        Mutable comparison collection.
    namespace : str
        Source/input or output artifact namespace.
    """
    for name, expected in bindings.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or relative.suffix not in SAFE_SUFFIXES:
            raise ValueError("Only declared repository non-waveform artifact bindings may be hashed")
        compare(expected, sha256_file(ROOT / relative), f"{namespace}.{name}", checks)


def compare_saved(
    result: dict[str, Any],
    payload: dict[str, np.ndarray],
    means: dict[str, np.ndarray],
    reconstructed: dict[str, Any],
    checks: list[dict[str, Any]],
) -> None:
    """Compare every saved metric, patient mean, contrast, gate and context summary.

    Parameters
    ----------
    result : dict[str, Any]
        Saved complete production result.
    payload, means : dict[str, np.ndarray]
        Saved query payload and sorted patient means.
    reconstructed : dict[str, Any]
        Independent scientific calculations.
    checks : list[dict[str, Any]]
        Mutable comparison collection.
    """
    for name, expected in reconstructed["summary"].items():
        compare(expected, result[name], f"result.{name}", checks)
    for name, metrics in reconstructed["query_metrics"].items():
        for metric, expected in metrics.items():
            compare(
                expected.tolist(), payload[f"{name}__{metric}"].tolist(), f"query.{name}.{metric}", checks
            )
    expected_means = reconstructed["patient_means"]
    compare(sorted(expected_means), sorted(means), "patient_means.fields", checks)
    for name in expected_means.keys() & means.keys():
        compare(expected_means[name].tolist(), means[name].tolist(), f"patient_means.{name}", checks)


def verify_result_schema(result: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    """Reject unreviewed scientific fields and verify the frozen result's numerical settings.

    Parameters
    ----------
    result : dict[str, Any]
        Saved complete result.
    checks : list[dict[str, Any]]
        Mutable comparison collection.
    """
    fields = [
        "status",
        "experiment",
        "provenance",
        "qualification",
        "missingness",
        "models",
        "primary_wave_minus_context",
        "informative_wave_minus_context",
        "contrasts_to_lookup",
        "contexts",
        "query_count",
        "patient_count",
        "informative_query_count",
        "informative_patient_count",
        "decision",
        "config",
        "timing",
        "model_fit_status",
        "artifact_sha256",
        "artifacts",
        "total_seconds",
    ]
    compare(sorted(fields), sorted(result), "result.fields", checks)
    compare("completed", result["status"], "result.status", checks)
    compare("043", result["experiment"], "result.experiment", checks)
    for name, expected in {
        "bootstrap_draws": DRAWS,
        "seed": SEED,
        "exact_set_comparator": ">",
        "exact_set_threshold": 0.5,
        "fit_folds": list(range(1, 8)),
        "unused_fold": 8,
        "development_fold": 9,
        "test_scoring": False,
    }.items():
        compare(expected, result["config"][name], f"config.{name}", checks)
    model = {
        "max_iter": 200,
        "learning_rate": 0.05,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 30,
        "l2_regularization": 10,
        "early_stopping": False,
        "random_state": SEED,
    }
    compare(model, result["config"]["model"], "config.model", checks)
    compare(
        ["context_code", "lead_code"],
        result["config"]["categorical_features"],
        "config.categorical_features",
        checks,
    )
    compare([1, 1], result["config"]["lookup_beta_prior"], "config.lookup_beta_prior", checks)
    compare(10, result["config"]["lookup_shrinkage"], "config.lookup_shrinkage", checks)
    compare("1/number_of_offered_leads", result["config"]["query_weight"], "config.query_weight", checks)
    verify_feature_manifest(result["config"], checks)


def verify_feature_manifest(config: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    """Verify the exact declared input difference and frozen numeric/resource policies.

    Parameters
    ----------
    config : dict[str, Any]
        Saved feature lists and numerical execution settings.
    checks : list[dict[str, Any]]
        Mutable comparison collection.
    """
    shared = ["context_code", "lead_code", *(f"offered_{lead}" for lead in LEADS)]
    waveform = [
        *shared,
        *(f"raw__{name}" for name in RAW_FEATURES),
        *(f"candidate__{name}" for name in CANDIDATE_MEASUREMENTS),
    ]
    compare(
        {"context_hgb": shared, "wave_hgb": waveform},
        config["feature_columns"],
        "config.feature_columns",
        checks,
    )
    compare(list(RAW_FEATURES), config["raw_feature_columns"], "config.raw_feature_columns", checks)
    compare(
        list(CANDIDATE_MEASUREMENTS),
        config["candidate_measurements"],
        "config.candidate_measurements",
        checks,
    )
    compare("NaN; no imputation or clipping", config["nonfinite_policy"], "config.nonfinite_policy", checks)
    compare(True, type(config["threads"]) is int and 1 <= config["threads"] <= 4, "config.thread_cap", checks)


def audit_artifacts(experiment: Path, checks: list[dict[str, Any]]) -> dict[str, Any]:
    """Read saved payloads only and independently compare their complete scientific arithmetic.

    Parameters
    ----------
    experiment : Path
        Completed production output directory.
    checks : list[dict[str, Any]]
        Mutable comparison collection.

    Returns
    -------
    dict[str, Any]
        Verified schema counts and independent reconstructed summary.
    """
    result = json.loads((experiment / "result.json").read_text())
    verify_result_schema(result, checks)
    verify_bindings(result["provenance"]["files_sha256"], checks, "source_input")
    expected_artifacts = (
        "qualification.json",
        "missingness.json",
        "fitted_models.joblib",
        "predictions.npz",
        "patient_means.npz",
    )
    compare(
        sorted(to_stored(experiment / name) for name in expected_artifacts),
        sorted(result["artifact_sha256"]),
        "artifact.complete_binding_set",
        checks,
    )
    verify_bindings(result["artifact_sha256"], checks, "artifact")
    qualification = json.loads((experiment / "qualification.json").read_text())
    compare(qualification, result["qualification"], "qualification.result_copy", checks)
    missingness = json.loads((experiment / "missingness.json").read_text())
    compare(missingness, result["missingness"], "missingness.result_copy", checks)
    verify_qualification(qualification, checks)
    with np.load(experiment / "predictions.npz", allow_pickle=False) as saved:
        payload = {name: saved[name] for name in saved.files}
    with np.load(experiment / "patient_means.npz", allow_pickle=False) as saved:
        means = {name: saved[name] for name in saved.files}
    counts = validate_payload(payload)
    reconstructed = reconstruct(payload)
    compare_saved(result, payload, means, reconstructed, checks)
    return {
        "counts": counts,
        "summary": reconstructed["summary"],
        "target_identity_limit": (
            "Saved target/mask identity accepted conditionally; source JSONL hashed only, not parsed."
        ),
    }


def run_audit(experiment: Path, output: Path) -> dict[str, Any]:
    """Persist an exclusive arithmetic audit and its output checksum, including failures.

    Parameters
    ----------
    experiment : Path
        Completed Experiment 043 output directory.
    output : Path
        New audit output directory; existing directories are refused.

    Returns
    -------
    dict[str, Any]
        Status, checks, mismatches, source/input identities and timing.
    """
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    checks: list[dict[str, Any]] = []
    receipt: dict[str, Any] = {
        "scope": SCOPE,
        "absolute_tolerance": TOLERANCE,
        "draws": DRAWS,
        "seed": SEED,
        "threads": 1,
    }
    try:
        with threadpool_limits(limits=1):
            receipt["verification"] = audit_artifacts(experiment, checks)
        compare(True, time.monotonic() - started <= 300, "audit.wall_budget_300_seconds", checks)
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
            "status": "passed" if checks and all(check["passed"] for check in checks) else "failed",
            "checks": checks,
            "check_count": len(checks),
            "mismatches": [check for check in checks if not check["passed"]],
            "wall_seconds": time.monotonic() - started,
            "git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "versions": {
                "python": platform.python_version(),
                **{name: version(name) for name in ("numpy", "scikit-learn", "threadpoolctl")},
            },
            "source_sha256": {
                to_stored(path): sha256_file(path)
                for path in (
                    Path(__file__),
                    ROOT / "scripts/reports/audit_lead_grounding043.py",
                    ROOT / "ecg_experiment/files.py",
                    ROOT / "ecg_experiment/paths.py",
                )
            },
            "input_sha256": {
                to_stored(experiment / name): sha256_file(experiment / name)
                for name in (
                    "result.json",
                    "predictions.npz",
                    "patient_means.npz",
                    "qualification.json",
                    "missingness.json",
                )
                if (experiment / name).exists()
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
