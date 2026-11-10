"""Training-only selection and reference-resampling comparison for KEY-04."""

from __future__ import annotations

import platform
import subprocess
import time
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from . import ROOT
from .condition_key_inputs import json_read, verify_inputs
from .condition_key_inputs_v2 import load_inputs
from .condition_keys import GROUPS, macro_interval, ranking_metrics, unit, whole_bank
from .files import sha256_file, write_json_atomic, write_npz_atomic
from .key_classifier_models import (
    CS,
    FAMILIES,
    encode_states,
    fit_package,
    folds,
    joint_metrics,
    predict_package,
    record_groups,
    resample_references,
)
from .paths import to_stored
from .pr74_screening_io import verify_hashes

OUT = ROOT / "outputs/condition_classifiers_20261106"
REPORT = ROOT / "reports/condition-classifiers-2026-11-06/aggregate.json"
PROTOCOL = "docs/condition-classifier-comparison-protocol-2026-11-06.md"
OLD = ROOT / "outputs/condition_keys_20261103_v2"
LONG = ROOT / "outputs/long_section_keys_20261106"


def binding() -> dict[str, Any]:
    """
    Require committed implementation and bind predecessor sources and artifacts.

    Returns
    -------
    dict[str, Any]
        Scientific source identity and predecessor artifact hashes.
    """
    sources = {}
    prior = {}
    for directory in (OLD, LONG):
        result = json_read(directory / "result.json")
        verify_hashes(ROOT, result["source_sha256"])
        verify_hashes(ROOT, result["artifact_sha256"])
        sources.update(result["source_sha256"])
        prior.update(result["artifact_sha256"])
        prior[to_stored(directory / "result.json")] = sha256_file(directory / "result.json")
    own = [
        PROTOCOL,
        "ecg_experiment/key_classifier_models.py",
        "ecg_experiment/key_classifier_study.py",
        "scripts/experiments/run_key_classifiers.py",
        "tests/test_key_classifiers.py",
    ]
    for name in own:
        if subprocess.check_output(["git", "show", "HEAD:" + name], cwd=ROOT) != (ROOT / name).read_bytes():
            raise ValueError(f"Uncommitted scientific source: {name}")
        sources[name] = sha256_file(ROOT / name)
    return {
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "protocol": PROTOCOL,
        "source_sha256": sources,
        "predecessor_sha256": prior,
    }


def check_time(started: float) -> None:
    """
    Enforce the registered whole-run resource ceiling at stage boundaries.

    Parameters
    ----------
    started : float
        Monotonic start time.
    """
    if time.perf_counter() - started > 10800:
        raise TimeoutError("KEY-04 exceeded its 10800-second stage-boundary ceiling")


def replay() -> dict[str, Any]:
    """
    Recompute complete predecessor metrics and the KEY-01 primary interval exactly.

    Returns
    -------
    dict[str, Any]
        Counts and exact interval verification.
    """
    count = 0
    for directory, flag in ((OLD, "query_flags"), (LONG, "flags")):
        result = json_read(directory / "result.json")
        arrays = np.load(directory / "artifacts.npz", allow_pickle=False)
        for arm in result["metrics"]:
            for normal, key in ((False, "metrics"), (True, "condition_vs_normal")):
                if ranking_metrics(arrays[flag], arrays["score_" + arm], normal) != result[key][arm]:
                    raise ValueError("Predecessor metrics do not reproduce")
                count += 1
    arrays = np.load(OLD / "artifacts.npz", allow_pickle=False)
    interval = macro_interval(
        arrays["query_patient"],
        arrays["query_flags"],
        arrays["score_window_selected_keys"],
        arrays["score_whole"],
    )
    if interval != json_read(OLD / "result.json")["primary"]:
        raise ValueError("Predecessor primary interval does not reproduce")
    return {"metric_dictionaries_exact": count, "primary_interval_exact": True}


def qualify() -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Verify feature/label order and reproduce the whole-record prototype baseline.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        Aligned cached inputs and qualification receipt.
    """
    hashes = verify_inputs()
    integrity = replay()
    data = load_inputs()
    saved = np.load(OLD / "artifacts.npz", allow_pickle=False)
    for role, prefix in (("train", "train"), ("query", "query")):
        if not np.array_equal(data[role]["record_id"].to_numpy(dtype=str), saved[prefix + "_record_id"]):
            raise ValueError("Ordered cohort identity mismatch")
        if not np.array_equal(data[role + "_flags"], saved[prefix + "_flags"]):
            raise ValueError("Ordered membership mismatch")
    scaler, bank = whole_bank(data["train_pooled"], data["train_flags"])
    z = unit(scaler.transform(data["query_pooled"].astype(float)))
    baseline = z @ (bank["positive"] - bank["normal"]).T
    error = float(np.max(np.abs(baseline - saved["score_whole"])))
    if error > 1e-12:
        raise ValueError("Whole-record baseline score mismatch")
    if ranking_metrics(data["query_flags"], baseline) != json_read(OLD / "result.json")["metrics"]["whole"]:
        raise ValueError("Whole-record baseline metrics mismatch")
    data["baseline"] = baseline
    data["train_pooled"] = data["train_pooled"].astype(np.float64)
    data["query_pooled"] = data["query_pooled"].astype(np.float64)
    groups = record_groups(data["train"])
    data["groups"] = groups
    data["folds"] = folds(groups)
    states = encode_states(data["train_flags"])
    counts = np.bincount(states, minlength=16)
    if (counts == 0).any():
        raise ValueError("Training joint-state coverage differs from feasibility check")
    return data, {
        "input_sha256": hashes,
        **integrity,
        "baseline_max_score_error": error,
        "training_state_counts": counts.tolist(),
        "training_groups": len(np.unique(groups)),
        "known_identity_records": int(data["train"]["patient_identity_known"].eq("True").sum()),
        "folds": [
            {
                "fold": f,
                "records": int((data["folds"] == f).sum()),
                "groups": len(np.unique(groups[data["folds"] == f])),
                "state_counts": np.bincount(states[data["folds"] == f], minlength=16).tolist(),
            }
            for f in range(3)
        ],
    }


def select_models(data: dict[str, Any], started: float) -> tuple[dict, dict, dict]:
    """
    Select regularization using only patient-grouped training folds.

    Parameters
    ----------
    data : dict[str, Any]
        Qualified feature arrays and training-only fold assignments.
    started : float
        Whole-run monotonic start.

    Returns
    -------
    tuple[dict, dict, dict]
        Public CV summary, private prediction arrays and fitted fold packages.
    """
    raw, flags, fold_id = data["train_pooled"], data["train_flags"], data["folds"]
    tasks = [(family, 1.0, 0) for family in FAMILIES]
    tasks += [
        (family, c, fold)
        for family in FAMILIES
        for c in CS
        for fold in range(3)
        if not (c == 1.0 and fold == 0)
    ]
    rows, packages, scores = [], {}, {}
    profile_seconds = 0.0
    for number, (family, c, fold) in enumerate(tasks):
        fitting, validation = fold_id != fold, fold_id == fold
        package = fit_package(raw[fitting], flags[fitting], family, c)
        prediction, _ = predict_package(package, raw[validation])
        key = f"{family}_C{c:g}"
        if key not in scores:
            scores[key] = np.full((len(flags), 4), np.nan)
        scores[key][validation] = prediction
        metrics = ranking_metrics(flags[validation], prediction)
        rows.append(
            {
                "family": family,
                "C": c,
                "fold": fold,
                "metrics": metrics,
                "fit_seconds": package["fit_seconds"],
                "iterations": package["iterations"],
            }
        )
        packages[f"{key}_fold{fold}"] = package
        print(
            f"CV {number + 1}/{len(tasks)} {key} fold {fold}: {metrics['macro_auroc']:.6f}; "
            f"{package['fit_seconds']:.1f}s fit",
            flush=True,
        )
        if number < 2:
            profile_seconds += package["fit_seconds"]
        if number == 1:
            forecast = profile_seconds * (9 + 11 * 1.5) * 2 + 300
            profile = {
                "first_fold_both_families_fit_seconds": profile_seconds,
                "forecast_seconds": forecast,
                "ceiling_seconds": 7200,
                "safety_factor": 2,
                "fixed_overhead_seconds": 300,
                "admitted": forecast <= 7200,
            }
            write_json_atomic(OUT / "profile.json", profile)
            if not profile["admitted"]:
                raise TimeoutError("Classifier profile failed the 7200-second forecast gate")
        check_time(started)
    summary, chosen = [], {}
    for family in FAMILIES:
        candidates = []
        for c in CS:
            mean = float(
                np.mean(
                    [
                        row["metrics"]["macro_auroc"]
                        for row in rows
                        if row["family"] == family and row["C"] == c
                    ]
                )
            )
            candidates.append((mean, c))
            summary.append({"family": family, "C": c, "mean_fold_macro_auroc": mean})
        chosen[family] = max(candidates, key=lambda x: (x[0], -x[1]))[1]
    if any(not np.isfinite(value).all() for value in scores.values()):
        raise ValueError("Missing or nonfinite out-of-fold predictions")
    return {"fold_metrics": rows, "selection_table": summary, "chosen_C": chosen}, scores, packages


def error_subsets(flags: np.ndarray, scores: dict[str, np.ndarray]) -> dict:
    """
    Describe exclusive and co-occurring positive groups against the same normal negatives.

    Parameters
    ----------
    flags : np.ndarray
        Evaluation memberships.
    scores : dict[str, np.ndarray]
        Fixed four-group scores by arm.

    Returns
    -------
    dict
        Denominators and ranking metrics for prespecified descriptive subsets.
    """
    result = {}
    normal = ~flags.any(axis=1)
    for column, group in enumerate(GROUPS):
        result[group] = {}
        for name, membership in (
            ("exclusive", flags.sum(axis=1) == 1),
            ("cooccurring", flags.sum(axis=1) > 1),
        ):
            positive = (flags[:, column] == 1) & membership
            keep = normal | positive
            result[group][name] = {
                "positives": int(positive.sum()),
                "normals": int(normal.sum()),
                "models": {
                    arm: {
                        "auroc": float(roc_auc_score(flags[keep, column], value[keep, column])),
                        "average_precision": float(
                            average_precision_score(flags[keep, column], value[keep, column])
                        ),
                    }
                    for arm, value in scores.items()
                }
                if positive.any()
                else {},
            }
    return result


def reference_sensitivity(data: dict[str, Any], chosen: dict, started: float) -> tuple[list, dict, dict]:
    """
    Execute every registered reference-group bootstrap refit with fixed selected C.

    Parameters
    ----------
    data : dict[str, Any]
        Qualified original training and evaluation inputs.
    chosen : dict
        Training-selected regularization strengths, never reselected here.
    started : float
        Whole-run monotonic start.

    Returns
    -------
    tuple[list, dict, dict]
        All draw summaries, private row/score arrays and fitted packages.
    """
    summary, arrays, fitted = [], {}, {}
    for seed in range(20261206, 20261216):
        rows = resample_references(data["groups"], data["train"]["family"].to_numpy(), seed)
        x, y = data["train_pooled"][rows], data["train_flags"][rows]
        scaler, keys = whole_bank(x, y)
        z = unit(scaler.transform(data["query_pooled"]))
        scores = {"whole": z @ (keys["positive"] - keys["normal"]).T}
        packages = {}
        for family in FAMILIES:
            packages[family] = fit_package(x, y, family, chosen[family])
            scores[family], _ = predict_package(packages[family], data["query_pooled"])
        metrics = {arm: ranking_metrics(data["query_flags"], value) for arm, value in scores.items()}
        contrasts = {
            family: {
                metric: metrics[family][metric] - metrics["whole"][metric]
                for metric in ("macro_auroc", "macro_average_precision")
            }
            for family in FAMILIES
        }
        summary.append(
            {
                "seed": seed,
                "training_rows": len(rows),
                "training_state_counts": np.bincount(encode_states(y), minlength=16).tolist(),
                "unique_training_groups": len(np.unique(data["groups"][rows])),
                "metrics": metrics,
                "differences": contrasts,
                "fit_seconds": {family: packages[family]["fit_seconds"] for family in FAMILIES},
            }
        )
        arrays[f"rows_{seed}"] = rows
        arrays.update({f"score_{seed}_{arm}": value for arm, value in scores.items()})
        fitted[str(seed)] = {"models": packages, "prototype_scaler": scaler, "prototype_keys": keys}
        print(
            f"Reference draw {seed}: joint minus prototype {contrasts['joint_logistic']['macro_auroc']:+.6f}",
            flush=True,
        )
        check_time(started)
    return summary, arrays, fitted


def execute(started: float, identity: dict) -> dict:
    """
    Complete qualification, model selection, held-out scoring and fixed sensitivity refits.

    Parameters
    ----------
    started : float
        Complete-run start.
    identity : dict
        Frozen source/protocol identity.

    Returns
    -------
    dict
        Completed aggregate result; all patient/model artifacts remain local.
    """
    data, integrity = qualify()
    write_json_atomic(OUT / "qualification.json", integrity)
    print("Exact predecessor replay and cohort qualification passed", flush=True)
    cv, cv_scores, cv_packages = select_models(data, started)
    write_json_atomic(OUT / "cross_validation.json", cv)
    joblib.dump(cv_packages, OUT / "cv_models.joblib", compress=3)
    write_npz_atomic(
        OUT / "cv_predictions.npz",
        folds=data["folds"],
        groups=data["groups"],
        flags=data["train_flags"],
        **cv_scores,
    )
    models = {
        family: fit_package(data["train_pooled"], data["train_flags"], family, cv["chosen_C"][family])
        for family in FAMILIES
    }
    joblib.dump(models, OUT / "final_models.joblib", compress=3)
    scores = {"whole": data["baseline"]}
    joint_probability = None
    for family, model in models.items():
        scores[family], probability = predict_package(model, data["query_pooled"])
        if probability is not None:
            joint_probability = probability
    if joint_probability is None:
        raise ValueError("Missing joint-state probabilities")
    metrics = {name: ranking_metrics(data["query_flags"], value) for name, value in scores.items()}
    patients = data["query"]["patient_key"].to_numpy(dtype=str)
    intervals = {
        family: macro_interval(patients, data["query_flags"], scores[family], scores["whole"])
        for family in FAMILIES
    }
    primary = intervals["joint_logistic"]
    majority = int(np.bincount(encode_states(data["train_flags"]), minlength=16).argmax())
    write_npz_atomic(
        OUT / "predictions.npz",
        flags=data["query_flags"],
        patients=patients,
        record_id=data["query"]["record_id"].to_numpy(dtype=str),
        joint_probability=joint_probability,
        **{"score_" + key: value for key, value in scores.items()},
    )
    print("Main held-out comparison:", primary, flush=True)
    stability, draw_arrays, draw_models = reference_sensitivity(data, cv["chosen_C"], started)
    write_npz_atomic(OUT / "reference_draws.npz", **draw_arrays)
    joblib.dump(draw_models, OUT / "reference_models.joblib", compress=3)
    result = {
        "study": "KEY-04",
        "status": "completed",
        **identity,
        "qualification": integrity,
        "cross_validation": cv,
        "metrics": metrics,
        "condition_vs_normal": {
            arm: ranking_metrics(data["query_flags"], value, True) for arm, value in scores.items()
        },
        "primary": primary,
        "exploratory_independent_contrast": intervals["independent_logistic"],
        "decision": {
            "gain_at_least_0.01": primary["difference"] >= 0.01,
            "lower_95_above_zero": primary["ci_low"] > 0,
            "ranking_improvement": primary["difference"] >= 0.01 and primary["ci_low"] > 0,
            "clinical_pipeline_promoted": False,
        },
        "multiclass": joint_metrics(data["query_flags"], joint_probability, majority),
        "error_subsets": error_subsets(data["query_flags"], scores),
        "reference_sensitivity": stability,
        "final_fit_seconds": {k: v["fit_seconds"] for k, v in models.items()},
        "wall_seconds": time.perf_counter() - started,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "gpu_used": False,
        "encoder_fitted": False,
        "scope": "reused_external_development",
    }
    check_time(started)
    verify_hashes(ROOT, identity["source_sha256"])
    result["artifact_sha256"] = {to_stored(p): sha256_file(p) for p in sorted(OUT.iterdir()) if p.is_file()}
    write_json_atomic(OUT / "result.json", result)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(REPORT, result)
    return result


def run() -> dict:
    """
    Run the frozen study once and preserve any failed attempt.

    Returns
    -------
    dict
        Completed result.
    """
    identity = binding()
    OUT.mkdir(exist_ok=False)
    write_json_atomic(OUT / "started.json", identity)
    started = time.perf_counter()
    try:
        with threadpool_limits(2):
            return execute(started, identity)
    except Exception as error:
        write_json_atomic(
            OUT / "failure.json", {"error": str(error), "wall_seconds": time.perf_counter() - started}
        )
        raise
