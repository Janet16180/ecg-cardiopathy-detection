"""Independent parameter arithmetic and split verification for completed KEY-04 outputs."""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

from . import ROOT
from .condition_key_inputs import json_read
from .condition_key_inputs_v2 import load_inputs
from .condition_keys import macro_interval
from .files import sha256_file, write_json_atomic
from .pr74_screening_io import verify_hashes

OUT = ROOT / "outputs/condition_classifiers_20261106"
REPORT = ROOT / "reports/condition-classifiers-2026-11-06/verification.json"
BITS = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(int)


def transformed(raw: np.ndarray, scaler: Any) -> np.ndarray:
    """
    Reconstruct standardized unit features directly from saved scalar parameters.

    Parameters
    ----------
    raw : np.ndarray
        Original cached pooled vectors.
    scaler : Any
        Saved training-fitted transformation.

    Returns
    -------
    np.ndarray
        Manual float64 standardized unit vectors.
    """
    z = (np.asarray(raw, dtype=np.float64) - scaler.mean_) / scaler.scale_
    return z / np.sqrt(np.sum(z * z, axis=1, keepdims=True))


def manual_prediction(package: dict, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Compute predictions from coefficients without calling model prediction methods.

    Parameters
    ----------
    package : dict
        Saved classifier package.
    raw : np.ndarray
        Pooled feature inputs.

    Returns
    -------
    tuple[np.ndarray, np.ndarray or None]
        Manually reconstructed marginal and joint-state probabilities.
    """
    z = transformed(raw, package["scaler"])
    if package["family"] == "independent_logistic":
        scores = np.column_stack(
            [expit(z @ model.coef_[0] + model.intercept_[0]) for model in package["models"]]
        )
        return scores, None
    model = package["models"][0]
    logits = z @ model.coef_.T + model.intercept_
    if logits.shape[1] == 1:
        p = expit(logits[:, 0])
        probability = np.column_stack([1 - p, p])
    else:
        probability = softmax(logits, axis=1)
    complete = np.zeros((len(raw), 16))
    complete[:, model.classes_.astype(int)] = probability
    return complete @ BITS, complete


def validate_scaler(package: dict, raw: np.ndarray, expected_c: float) -> None:
    """
    Verify the saved transformation and logistic recipe against its exact training rows.

    Parameters
    ----------
    package : dict
        Fitted family package.
    raw : np.ndarray
        Exact original fitting rows, including registered bootstrap multiplicities.
    expected_c : float
        Frozen selected or cross-validation C.
    """
    scaler = package["scaler"]
    np.testing.assert_allclose(scaler.mean_, raw.mean(axis=0), rtol=1e-10, atol=1e-12)
    scale = raw.std(axis=0)
    scale[scale == 0] = 1
    np.testing.assert_allclose(scaler.scale_, scale, rtol=1e-10, atol=1e-12)
    if scaler.n_samples_seen_ != len(raw):
        raise ValueError("Scaler was fitted on a different number of rows")
    for model in package["models"]:
        if (
            expected_c != model.C
            or model.solver != "lbfgs"
            or model.max_iter != 5000
            or model.tol != 1e-8
            or model.class_weight is not None
            or model.random_state != 42
        ):
            raise ValueError("Logistic recipe differs")
        if np.max(model.n_iter_) >= 5000:
            raise ValueError("Iteration-cap model admitted")


def metric_check(flags: np.ndarray, scores: np.ndarray, expected: dict, normal: bool = False) -> int:
    """
    Independently recompute every per-group and macro ranking metric.

    Parameters
    ----------
    flags, scores : np.ndarray
        Binary annotation memberships and fixed model scores.
    expected : dict
        Saved complete ranking dictionary.
    normal : bool
        Restrict each group's negatives to the frozen normal pool.

    Returns
    -------
    int
        Number of compared metric and denominator fields.
    """
    auc, ap = [], []
    for c, group in enumerate(("MI", "STTC", "CD", "HYP")):
        mask = ((flags[:, c] == 1) | (flags.sum(axis=1) == 0)) if normal else np.ones(len(flags), bool)
        y, s = flags[mask, c], scores[mask, c]
        point = expected["groups"][group]
        a, p = float(roc_auc_score(y, s)), float(average_precision_score(y, s))
        if point != {
            "records": len(y),
            "positives": int(y.sum()),
            "negatives": int(len(y) - y.sum()),
            "auroc": a,
            "average_precision": p,
        }:
            raise ValueError("Ranking metric or denominator mismatch")
        auc.append(a)
        ap.append(p)
    if np.mean(auc) != expected["macro_auroc"] or np.mean(ap) != expected["macro_average_precision"]:
        raise ValueError("Macro ranking mismatch")
    return 22


def validate_reference_draw(groups: np.ndarray, families: np.ndarray, rows: np.ndarray, seed: int) -> None:
    """
    Verify whole-group multiplicities and seeded family draw counts independently.

    Parameters
    ----------
    groups, families : np.ndarray
        Original training identity groups and source families.
    rows : np.ndarray
        Saved original-row indices, including sampled multiplicities.
    seed : int
        Registered resampling seed.
    """
    if rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= len(groups)):
        raise ValueError("Invalid reference draw positions")
    record_counts = np.bincount(rows, minlength=len(groups))
    counts = pd.Series(record_counts, index=groups).groupby(level=0).agg(["min", "max"])
    if not np.array_equal(counts["min"], counts["max"]):
        raise ValueError("A sampled patient group was split")
    rng = np.random.default_rng(seed)
    for family in sorted(set(families)):
        unique = np.unique(groups[families == family])
        choices = rng.integers(0, len(unique), size=len(unique))
        expected_counts = np.bincount(choices, minlength=len(unique))
        np.testing.assert_array_equal(counts.loc[unique, "min"].to_numpy(), expected_counts)


def multiclass_check(
    flags: np.ndarray, probability: np.ndarray, expected: dict, train_flags: np.ndarray
) -> int:
    """
    Reconstruct joint-state counts and scores without the study's metric helper.

    Parameters
    ----------
    flags, train_flags : np.ndarray
        Evaluation and original training memberships.
    probability : np.ndarray
        Saved sixteen-state probabilities.
    expected : dict
        Registered multiclass metric dictionary.

    Returns
    -------
    int
        Number of checked count/statistic fields.
    """
    truth = flags @ (1 << np.arange(4))
    predicted = probability.argmax(axis=1)
    matrix = np.bincount(truth * 16 + predicted, minlength=256).reshape(16, 16)
    np.testing.assert_array_equal(matrix, expected["confusion"])
    support, called = matrix.sum(axis=1), matrix.sum(axis=0)
    np.testing.assert_array_equal(support, expected["support"])
    np.testing.assert_array_equal(called, expected["predicted_count"])
    np.testing.assert_array_equal(np.flatnonzero(support), expected["supported_states"])
    for state in range(16):
        precision = matrix[state, state] / called[state] if called[state] else None
        recall = matrix[state, state] / support[state] if support[state] else None
        f1 = 2 * matrix[state, state] / (support[state] + called[state]) if support[state] else None
        for name, value in [("precision", precision), ("recall", recall), ("f1", f1)]:
            if value is None:
                if expected[name][state] is not None:
                    raise ValueError("Undefined multiclass metric was presented as a number")
            else:
                np.testing.assert_allclose(value, expected[name][state], atol=1e-12, rtol=0)
    top_two = np.argsort(-probability, axis=1, kind="stable")[:, :2]
    most_common = int(np.bincount(train_flags @ (1 << np.arange(4)), minlength=16).argmax())
    values = {
        "exact_state_accuracy": float(np.mean(truth == predicted)),
        "macro_recall_supported_states": float((np.diag(matrix)[support > 0] / support[support > 0]).mean()),
        "top_two_accuracy": float((top_two == truth[:, None]).any(axis=1).mean()),
        "training_majority_state": most_common,
        "training_majority_accuracy": float(np.mean(truth == most_common)),
        "log_loss": float(
            -np.log(np.clip(probability[np.arange(len(truth)), truth], np.finfo(float).eps, 1)).mean()
        ),
    }
    for name, value in values.items():
        np.testing.assert_allclose(value, expected[name], atol=1e-12, rtol=0)
    return 58


def subset_check(flags: np.ndarray, predictions: Any, expected: dict) -> int:
    """
    Verify exclusive/co-occurring annotation subset denominators and ranking results.

    Parameters
    ----------
    flags : np.ndarray
        Existing evaluation memberships.
    predictions : Any
        Saved score arrays by model family.
    expected : dict
        Frozen descriptive error-subset summaries.

    Returns
    -------
    int
        Number of verified metrics and denominators.
    """
    checked = 0
    normal = flags.sum(axis=1) == 0
    for c, group in enumerate(("MI", "STTC", "CD", "HYP")):
        for name, count_mask in [
            ("exclusive", flags.sum(axis=1) == 1),
            ("cooccurring", flags.sum(axis=1) > 1),
        ]:
            positive = (flags[:, c] == 1) & count_mask
            value = expected[group][name]
            if value["positives"] != int(positive.sum()) or value["normals"] != int(normal.sum()):
                raise ValueError("Error-subset denominator mismatch")
            mask = normal | positive
            checked += 2
            for arm, metrics in value["models"].items():
                scores = predictions["score_" + arm][mask, c]
                np.testing.assert_allclose(
                    roc_auc_score(flags[mask, c], scores), metrics["auroc"], atol=1e-12
                )
                np.testing.assert_allclose(
                    average_precision_score(flags[mask, c], scores), metrics["average_precision"], atol=1e-12
                )
                checked += 2
    return checked


def audit_cv(r: dict, data: dict, raw: np.ndarray, cv: Any) -> tuple[int, float, int]:
    """
    Audit every out-of-fold package and the training-only regularization selection.

    Parameters
    ----------
    r, data : dict
        Completed aggregate and verified original inputs.
    raw : np.ndarray
        Original training features.
    cv : Any
        Saved out-of-fold scores and fold assignments.

    Returns
    -------
    tuple[int, float, int]
        Metric count, maximum score error and verified package count.
    """
    count, worst = 0, 0.0
    cv_models = joblib.load(OUT / "cv_models.joblib")
    for row in r["cross_validation"]["fold_metrics"]:
        family, c, fold = row["family"], row["C"], row["fold"]
        key = f"{family}_C{c:g}"
        package = cv_models[f"{key}_fold{fold}"]
        fitting, held = cv["folds"] != fold, cv["folds"] == fold
        validate_scaler(package, raw[fitting], c)
        prediction, _ = manual_prediction(package, raw[held])
        error = float(np.max(np.abs(prediction - cv[key][held])))
        worst = max(worst, error)
        np.testing.assert_allclose(prediction, cv[key][held], atol=1e-12, rtol=0)
        count += metric_check(data["train_flags"][held], prediction, row["metrics"])
    table = pd.DataFrame(r["cross_validation"]["selection_table"])
    for family in ("joint_logistic", "independent_logistic"):
        part = table.loc[table.family == family]
        for row in part.itertuples():
            values = [
                f["metrics"]["macro_auroc"]
                for f in r["cross_validation"]["fold_metrics"]
                if f["family"] == family and f["C"] == row.C
            ]
            if np.mean(values) != row.mean_fold_macro_auroc:
                raise ValueError("Training-only selection mean mismatch")
        selected = part.sort_values(["mean_fold_macro_auroc", "C"], ascending=[False, True]).iloc[0]["C"]
        if selected != r["cross_validation"]["chosen_C"][family]:
            raise ValueError("Hyperparameter selection mismatch")
    return count, worst, len(cv_models)


def audit_final(r: dict, data: dict, raw: np.ndarray, query: np.ndarray) -> tuple[int, float, int]:
    """
    Audit final saved parameters and the primary patient interval.

    Parameters
    ----------
    r, data : dict
        Completed aggregate and verified original inputs.
    raw, query : np.ndarray
        Original training and evaluation features.

    Returns
    -------
    tuple[int, float, int]
        Metric count, maximum score error and final family count.
    """
    count, worst = 0, 0.0
    predictions = np.load(OUT / "predictions.npz", allow_pickle=False)
    np.testing.assert_array_equal(predictions["record_id"], data["query"]["record_id"].to_numpy(dtype=str))
    np.testing.assert_array_equal(predictions["flags"], data["query_flags"])
    models = joblib.load(OUT / "final_models.joblib")
    for family, package in models.items():
        validate_scaler(package, raw, r["cross_validation"]["chosen_C"][family])
        scores, joint = manual_prediction(package, query)
        worst = max(worst, float(np.max(np.abs(scores - predictions["score_" + family]))))
        np.testing.assert_allclose(scores, predictions["score_" + family], atol=1e-12, rtol=0)
        if joint is not None:
            np.testing.assert_allclose(joint, predictions["joint_probability"], atol=1e-12, rtol=0)
    for name in r["metrics"]:
        count += metric_check(data["query_flags"], predictions["score_" + name], r["metrics"][name])
        count += metric_check(
            data["query_flags"], predictions["score_" + name], r["condition_vs_normal"][name], True
        )
    interval = macro_interval(
        predictions["patients"],
        predictions["flags"],
        predictions["score_joint_logistic"],
        predictions["score_whole"],
    )
    if interval != r["primary"]:
        raise ValueError("Primary interval mismatch")
    count += multiclass_check(
        predictions["flags"], predictions["joint_probability"], r["multiclass"], data["train_flags"]
    )
    count += subset_check(predictions["flags"], predictions, r["error_subsets"])
    return count, worst, len(models)


def audit_draws(
    r: dict, data: dict, raw: np.ndarray, query: np.ndarray, expected_groups: np.ndarray, frame: Any
) -> tuple[int, float, int]:
    """
    Audit all sampled rows, scaler fits, classifier scores and prototype reconstructions.

    Parameters
    ----------
    r, data : dict
        Completed aggregate and verified original inputs.
    raw, query : np.ndarray
        Original training and evaluation features.
    expected_groups : np.ndarray
        Verified training identity groups.
    frame : Any
        Original training source-family metadata.

    Returns
    -------
    tuple[int, float, int]
        Metric count, maximum score error and verified reference-draw count.
    """
    count, worst = 0, 0.0
    draw_arrays = np.load(OUT / "reference_draws.npz", allow_pickle=False)
    draw_models = joblib.load(OUT / "reference_models.joblib")
    families = frame["family"].to_numpy()
    for draw in r["reference_sensitivity"]:
        seed = draw["seed"]
        rows = draw_arrays[f"rows_{seed}"]
        validate_reference_draw(expected_groups, families, rows, seed)
        fitted = draw_models[str(seed)]
        for family, package in fitted["models"].items():
            validate_scaler(package, raw[rows], r["cross_validation"]["chosen_C"][family])
            scores, _ = manual_prediction(package, query)
            expected = draw_arrays[f"score_{seed}_{family}"]
            worst = max(worst, float(np.max(np.abs(scores - expected))))
            np.testing.assert_allclose(scores, expected, atol=1e-12, rtol=0)
            count += metric_check(data["query_flags"], expected, draw["metrics"][family])
        z = transformed(raw[rows], fitted["prototype_scaler"])
        normal = data["train_flags"][rows].sum(axis=1) == 0
        keys = []
        for c in range(4):
            positive = z[data["train_flags"][rows, c] == 1].mean(axis=0)
            negative = z[normal].mean(axis=0)
            keys.append(positive / np.linalg.norm(positive) - negative / np.linalg.norm(negative))
        scores = transformed(query, fitted["prototype_scaler"]) @ np.stack(keys).T
        np.testing.assert_allclose(scores, draw_arrays[f"score_{seed}_whole"], atol=1e-12, rtol=0)
        count += metric_check(
            data["query_flags"], draw_arrays[f"score_{seed}_whole"], draw["metrics"]["whole"]
        )
        for family in ("joint_logistic", "independent_logistic"):
            for metric in ("macro_auroc", "macro_average_precision"):
                if (
                    draw["differences"][family][metric]
                    != draw["metrics"][family][metric] - draw["metrics"]["whole"][metric]
                ):
                    raise ValueError("Reference sensitivity contrast mismatch")
        print(f"Audited reference draw {seed}", flush=True)
    return count, worst, len(draw_models)


def audit() -> dict[str, Any]:
    """
    Verify saved folds, CV selection, model predictions, intervals and all ten refits.

    Returns
    -------
    dict[str, Any]
        Audit receipt with parameter reconstruction errors and comparison counts.
    """
    r = json_read(OUT / "result.json")
    verify_hashes(ROOT, r["source_sha256"])
    verify_hashes(ROOT, r["predecessor_sha256"])
    verify_hashes(ROOT, r["qualification"]["input_sha256"])
    verify_hashes(ROOT, r["artifact_sha256"])
    data = load_inputs()
    raw = data["train_pooled"].astype(float)
    query = data["query_pooled"].astype(float)
    frame = data["train"]
    expected_groups = np.array(
        [
            f"known:{row.source}:{row.patient_id}"
            if row.patient_identity_known == "True"
            else f"unknown:{row.record_id}"
            for row in frame.itertuples()
        ],
        dtype=str,
    )
    cv = np.load(OUT / "cv_predictions.npz", allow_pickle=False)
    np.testing.assert_array_equal(cv["groups"], expected_groups)
    np.testing.assert_array_equal(cv["flags"], data["train_flags"])
    splitter = GroupKFold(n_splits=3, shuffle=True, random_state=20261106)
    for fold, (fit, held) in enumerate(splitter.split(raw, groups=expected_groups)):
        np.testing.assert_array_equal(np.flatnonzero(cv["folds"] == fold), held)
        if set(expected_groups[fit]) & set(expected_groups[held]):
            raise ValueError("Known-patient fold overlap")
    cv_count, cv_error, cv_n = audit_cv(r, data, raw, cv)
    final_count, final_error, _ = audit_final(r, data, raw, query)
    draw_count, draw_error, draw_n = audit_draws(r, data, raw, query, expected_groups, frame)
    count = cv_count + final_count + draw_count
    worst = max(cv_error, final_error, draw_error)
    result = {
        "status": "passed",
        "metric_and_denominator_fields": count,
        "cv_packages_verified": cv_n,
        "reference_draws_verified": draw_n,
        "max_parameter_prediction_error": worst,
        "primary_interval_exact": True,
        "known_patient_fold_separation_verified": True,
        "bootstrap_group_multiplicities_exact": True,
        "result_sha256": sha256_file(OUT / "result.json"),
        "audit_source_sha256": sha256_file(ROOT / "ecg_experiment/key_classifier_audit.py"),
        "new_fits": 0,
    }
    write_json_atomic(REPORT, result)
    return result


def run() -> dict[str, Any]:
    """
    Run audit arithmetic with the experiment's numerical thread limit.

    Returns
    -------
    dict[str, Any]
        Verification receipt.
    """
    with threadpool_limits(2):
        return audit()
