"""
Validation analyses of KEY-05.

Construction table, ladder, pairs, source, reference counts, stability and agreement with the defining
measurements.

Scores of keys are cosines or dot products, not probabilities. AUROC and average precision depend only on the
order of the scores, so ``rank_unit`` maps each column to its ranks divided by ``n + 1`` (ties kept as ties)
before the MLC-02 metric and bootstrap code, which expects values in [0, 1], is applied. Two macro AUROCs are
reported:

- ``selection``: MLC-02's selection metric, the within-source macro AUROC (conditions with at least 20
  positives and 20 negatives in the source) averaged over the validation sources, with within-source
  patient-cluster resamples combined draw by draw (``mlc03_effect.source_bootstrap``);
- ``pooled``: the macro AUROC over the 14 conditions on all validation records together, with patient-cluster
  resamples of the whole set (``mlc02_bootstrap.cluster_resamples``).

Both use 2,000 draws through ``intervals.patient_groups`` and ``intervals.patient_resample``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from ecg_experiment import key05_keys as kk
from ecg_experiment.intervals import patient_groups, patient_resample
from ecg_experiment.mlc02_bootstrap import (
    DRAWS,
    SEED,
    bootstrap_metrics,
    bootstrap_p_value,
    cluster_resamples,
    holm_adjust,
    percentile_interval,
)
from ecg_experiment.mlc02_data import CONDITIONS, VALIDATION_SOURCES
from ecg_experiment.mlc02_evaluation import MIN_SUPPORT, auroc_by_condition, enters_macro, selection_metric
from ecg_experiment.mlc03_effect import condition_draws, selection_draws, source_bootstrap

PAIR_MINIMUM = 100
REFERENCE_COUNTS = (5, 10, 20, 50, 100, 200, 500)
REFERENCE_DRAWS = 50
REFERENCE_SEED = 20261112
STABILITY_DRAWS = 200
STABILITY_SEED = 20261113
LADDER = ("K1", "K2", "K4", "K6", "K7", "P0")


def rank_unit(scores: np.ndarray) -> np.ndarray:
    """
    Order-preserving map of each score column into (0, 1).

    Parameters
    ----------
    scores : np.ndarray
        ``[records, conditions]`` finite scores.

    Returns
    -------
    np.ndarray
        Average ranks divided by ``records + 1``.

    Raises
    ------
    ValueError
        If a score is not finite.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    return rankdata(scores, method="average", axis=0) / (len(scores) + 1)


def average_precision_by_condition(labels: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """
    Average precision of each condition on its defined entries.

    Parameters
    ----------
    labels : np.ndarray
        ``[records, conditions]`` states.
    scores : np.ndarray
        Aligned scores.

    Returns
    -------
    np.ndarray
        One value per condition; NaN where the defined entries hold a single class.
    """
    out = np.full(labels.shape[1], np.nan)
    for column in range(labels.shape[1]):
        defined = ~np.isnan(labels[:, column])
        y = labels[defined, column]
        if 0 < y.sum() < len(y):
            out[column] = average_precision_score(y, scores[defined, column])
    return out


def per_source_table(labels: np.ndarray, scores: dict[str, np.ndarray], sources: np.ndarray) -> pd.DataFrame:
    """
    AUROC and average precision of every model, condition and source, and pooled over the sources.

    Parameters
    ----------
    labels : np.ndarray
        Validation states.
    scores : dict[str, np.ndarray]
        Model to ``[records, conditions]`` scores.
    sources : np.ndarray
        Source of each record.

    Returns
    -------
    pd.DataFrame
        ``model``, ``source`` (``pooled`` for all records), ``condition``, ``positives``, ``negatives``,
        ``enters_macro``, ``auroc`` and ``average_precision``.
    """
    rows = []
    units = [("pooled", np.ones(len(labels), dtype=bool))]
    units += [(name, sources == name) for name in VALIDATION_SOURCES if (sources == name).any()]
    for unit, mask in units:
        entering = enters_macro(labels[mask])
        for model, values in scores.items():
            auroc = auroc_by_condition(labels[mask], rank_unit(values[mask]))
            precision = average_precision_by_condition(labels[mask], values[mask])
            for column, condition in enumerate(CONDITIONS):
                rows.append(
                    {
                        "model": model,
                        "source": unit,
                        "condition": condition,
                        "positives": int((labels[mask, column] == 1).sum()),
                        "negatives": int((labels[mask, column] == 0).sum()),
                        "enters_macro": bool(entering[column]),
                        "auroc": float(auroc[column]),
                        "average_precision": float(precision[column]),
                    }
                )
    return pd.DataFrame(rows)


def bootstrap_draws(
    labels: np.ndarray, groups: np.ndarray, sources: np.ndarray, scores: dict[str, np.ndarray]
) -> dict[str, Any]:
    """
    Observed values and resampled draws of both macro AUROCs for every model, on shared resamples.

    Parameters
    ----------
    labels : np.ndarray
        Validation states.
    groups : np.ndarray
        Bootstrap group of each record.
    sources : np.ndarray
        Source of each record.
    scores : dict[str, np.ndarray]
        Model to scores.

    Returns
    -------
    dict[str, Any]
        Per model: ``selection`` (observed), ``selection_draws`` ``[draws]``, ``condition`` (observed
        within-source per-condition metric, ``[14]``), ``condition_draws`` ``[draws, 14]``, ``pooled``
        (observed macro), ``pooled_draws``, ``pooled_condition`` and ``pooled_condition_draws``; and
        ``resamples`` (the identity of every resample set).
    """
    ranked = {name: rank_unit(values) for name, values in scores.items()}
    within = source_bootstrap(labels, groups, sources, ranked)
    pooled_resamples = cluster_resamples(groups, labels, DRAWS, SEED)
    out: dict[str, Any] = {"models": {}}
    for name, values in ranked.items():
        observed = selection_metric(labels, values, sources)
        pooled = bootstrap_metrics(labels, values, pooled_resamples, None, calibration=False)
        out["models"][name] = {
            "selection": observed["selection_metric"],
            "selection_draws": selection_draws(within, name),
            "condition": np.array([observed["per_condition"][c] for c in CONDITIONS]),
            "condition_draws": condition_draws(within, name),
            "pooled": pooled["observed_macro_auroc"],
            "pooled_draws": pooled["macro_auroc"],
            "pooled_condition": pooled["observed_auroc"],
            "pooled_condition_draws": pooled["samples"]["auroc"],
        }
    identity = ("draws", "redraws", "seed", "groups", "records", "fingerprint")
    out["resamples"] = {
        "pooled": {key: pooled_resamples[key] for key in identity},
        **{source: part["resamples"] for source, part in within.items()},
    }
    return out


def interval(values: np.ndarray) -> tuple[float, float]:
    """
    Percentile 95% interval of resampled values, ignoring NaN.

    Parameters
    ----------
    values : np.ndarray
        ``[draws]``.

    Returns
    -------
    tuple[float, float]
        Lower and upper bound.
    """
    low, high, _ = percentile_interval(values)
    return float(low), float(high)


def construction_table(draws: dict[str, Any], reference: str = "P0") -> pd.DataFrame:
    """
    Both macro AUROCs of every model with intervals, and the paired difference from the reference.

    Parameters
    ----------
    draws : dict[str, Any]
        Output of ``bootstrap_draws``.
    reference : str
        Model the differences are taken from.

    Returns
    -------
    pd.DataFrame
        One row per model.
    """
    base = draws["models"][reference]
    rows = []
    for name, model in draws["models"].items():
        row = {"model": name}
        for metric in ("selection", "pooled"):
            low, high = interval(model[f"{metric}_draws"])
            difference = model[f"{metric}_draws"] - base[f"{metric}_draws"]
            d_low, d_high = interval(difference)
            row |= {
                metric: model[metric],
                f"{metric}_ci_low": low,
                f"{metric}_ci_high": high,
                f"{metric}_minus_{reference}": model[metric] - base[metric],
                f"{metric}_minus_{reference}_ci_low": d_low,
                f"{metric}_minus_{reference}_ci_high": d_high,
            }
        rows.append(row)
    return pd.DataFrame(rows)


def condition_table(draws: dict[str, Any], reference: str = "P0") -> pd.DataFrame:
    """
    Per-condition AUROC of every model (within-source mean and pooled) and the paired difference from P0.

    Parameters
    ----------
    draws : dict[str, Any]
        Output of ``bootstrap_draws``.
    reference : str
        Model the differences are taken from.

    Returns
    -------
    pd.DataFrame
        One row per model and condition, with intervals and Holm-adjusted bootstrap p-values over the 14
        conditions.
    """
    base = draws["models"][reference]
    rows = []
    for name, model in draws["models"].items():
        parts = {}
        for metric, key in (("within", "condition"), ("pooled", "pooled_condition")):
            difference = model[f"{key}_draws"] - base[f"{key}_draws"]
            p = np.array([bootstrap_p_value(difference[:, c]) for c in range(len(CONDITIONS))])
            parts[metric] = (model[key], model[f"{key}_draws"], difference, p, holm_adjust(p))
        for column, condition in enumerate(CONDITIONS):
            row = {"model": name, "condition": condition}
            for metric, (observed, values, difference, _, holm) in parts.items():
                low, high = interval(values[:, column])
                d_low, d_high = interval(difference[:, column])
                row |= {
                    metric: float(observed[column]),
                    f"{metric}_ci_low": low,
                    f"{metric}_ci_high": high,
                    f"{metric}_minus_{reference}": float(observed[column] - base_value(base, metric, column)),
                    f"{metric}_minus_{reference}_ci_low": d_low,
                    f"{metric}_minus_{reference}_ci_high": d_high,
                    f"{metric}_p_holm": float(holm[column]) if name != reference else math.nan,
                }
            rows.append(row)
    return pd.DataFrame(rows)


def base_value(base: dict[str, Any], metric: str, column: int) -> float:
    """
    Observed per-condition value of the reference model.

    Parameters
    ----------
    base : dict[str, Any]
        Reference entry of ``bootstrap_draws``.
    metric : str
        ``within`` or ``pooled``.
    column : int
        Condition column.

    Returns
    -------
    float
        The value.
    """
    return float(base["condition" if metric == "within" else "pooled_condition"][column])


def ladder_table(draws: dict[str, Any], steps: tuple[str, ...]) -> pd.DataFrame:
    """
    Paired per-condition gain of each step of the ladder over the previous one.

    Parameters
    ----------
    draws : dict[str, Any]
        Output of ``bootstrap_draws``.
    steps : tuple[str, ...]
        Model ids in ladder order.

    Returns
    -------
    pd.DataFrame
        One row per step, condition and metric (``within``, ``pooled``), plus a ``macro`` row per step.
    """
    rows = []
    for previous, current in zip(steps[:-1], steps[1:], strict=True):
        a, b = draws["models"][previous], draws["models"][current]
        for metric, key, macro in (
            ("within", "condition", "selection"),
            ("pooled", "pooled_condition", "pooled"),
        ):
            difference = b[f"{key}_draws"] - a[f"{key}_draws"]
            for column, condition in enumerate(CONDITIONS):
                low, high = interval(difference[:, column])
                rows.append(
                    {
                        "step": f"{previous} to {current}",
                        "metric": metric,
                        "condition": condition,
                        "gain": float(b[key][column] - a[key][column]),
                        "ci_low": low,
                        "ci_high": high,
                    }
                )
            low, high = interval(b[f"{macro}_draws"] - a[f"{macro}_draws"])
            rows.append(
                {
                    "step": f"{previous} to {current}",
                    "metric": metric,
                    "condition": "macro",
                    "gain": float(b[macro] - a[macro]),
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    return pd.DataFrame(rows)


def stratified_auroc(
    y: np.ndarray, score: np.ndarray, sources: np.ndarray, minimum: int = MIN_SUPPORT
) -> float:
    """
    Mean within-source AUROC over the sources with enough positives and negatives.

    Parameters
    ----------
    y : np.ndarray
        Binary targets.
    score : np.ndarray
        Scores.
    sources : np.ndarray
        Source of each record.
    minimum : int
        Required positives and negatives in a source.

    Returns
    -------
    float
        The mean; NaN if no source qualifies.
    """
    values = []
    for name in np.unique(sources):
        inside = sources == name
        positives = int(y[inside].sum())
        if positives >= minimum and inside.sum() - positives >= minimum:
            values.append(roc_auc_score(y[inside], score[inside]))
    return float(np.mean(values)) if values else math.nan


def pair_rows(
    labels: np.ndarray, sources: np.ndarray, scores: np.ndarray, model: str, minimum: int = PAIR_MINIMUM
) -> list[dict[str, Any]]:
    """
    For each ordered pair (A, B): AUROC of A's score for B among records negative for A.

    Parameters
    ----------
    labels : np.ndarray
        Validation states.
    sources : np.ndarray
        Source of each record.
    scores : np.ndarray
        ``[records, conditions]`` scores; column A is key A.
    model : str
        Model id written in each row.
    minimum : int
        Pairs with fewer B-positives among A-negatives are left out.

    Returns
    -------
    list[dict[str, Any]]
        ``model``, ``key`` (A), ``target`` (B), ``positives``, ``negatives``, ``auroc`` (pooled) and
        ``auroc_within`` (mean within-source AUROC).
    """
    rows = []
    for a, key in enumerate(CONDITIONS):
        for b, target in enumerate(CONDITIONS):
            if a == b:
                continue
            mask = (labels[:, a] == 0) & ~np.isnan(labels[:, b])
            y = labels[mask, b]
            positives = int(y.sum())
            if positives < minimum or len(y) - positives < minimum:
                continue
            rows.append(
                {
                    "model": model,
                    "key": key,
                    "target": target,
                    "positives": positives,
                    "negatives": int(len(y) - positives),
                    "auroc": float(roc_auc_score(y, scores[mask, a])),
                    "auroc_within": stratified_auroc(y, scores[mask, a], sources[mask]),
                }
            )
    return rows


def projected_keys(keys: np.ndarray) -> np.ndarray:
    """
    Each key with the span of the other keys projected out.

    Parameters
    ----------
    keys : np.ndarray
        ``[conditions, features]``.

    Returns
    -------
    np.ndarray
        ``[conditions, features]``.
    """
    return np.stack([kk.project_out(keys[c], np.delete(keys, c, axis=0)) for c in range(len(keys))])


def source_directions(x: np.ndarray, sources: np.ndarray) -> dict[str, Any]:
    """
    Directions that separate the train sources: multinomial logistic regression on standardized features.

    Parameters
    ----------
    x : np.ndarray
        ``[train records, features]``.
    sources : np.ndarray
        Source of each record.

    Returns
    -------
    dict[str, Any]
        ``model`` (fitted), ``classes`` and ``coef`` (``[sources, features]``).
    """
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=1.0, max_iter=2000).fit(x, sources)
    return {"model": model, "classes": [str(name) for name in model.classes_], "coef": model.coef_.copy()}


def between_source_share(score: np.ndarray, sources: np.ndarray) -> float:
    """
    Share of a score's variance that lies between source means.

    Parameters
    ----------
    score : np.ndarray
        ``[records]``.
    sources : np.ndarray
        Source of each record.

    Returns
    -------
    float
        Between-source sum of squares over the total.
    """
    total = float(((score - score.mean()) ** 2).sum())
    between = sum(
        float((sources == name).sum()) * (score[sources == name].mean() - score.mean()) ** 2
        for name in np.unique(sources)
    )
    return between / total


def condition_metrics(labels: np.ndarray, scores: np.ndarray, sources: np.ndarray) -> dict[str, np.ndarray]:
    """
    Within-source mean and pooled AUROC per condition, from raw scores.

    Parameters
    ----------
    labels : np.ndarray
        Validation states.
    scores : np.ndarray
        ``[records, conditions]`` scores.
    sources : np.ndarray
        Source of each record.

    Returns
    -------
    dict[str, np.ndarray]
        ``within`` and ``pooled``, ``[14]`` each.
    """
    ranked = rank_unit(scores)
    observed = selection_metric(labels, ranked, sources)
    return {
        "within": np.array([observed["per_condition"][c] for c in CONDITIONS]),
        "pooled": auroc_by_condition(labels, ranked),
    }


def draw_references(y: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """
    Draw ``count`` positives and ``count`` negatives without replacement.

    Parameters
    ----------
    y : np.ndarray
        States of one condition.
    count : int
        Records per class.
    rng : np.random.Generator
        Generator.

    Returns
    -------
    np.ndarray
        Drawn record indices, positives first.
    """
    positives, negatives = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return np.concatenate(
        [rng.choice(positives, count, replace=False), rng.choice(negatives, count, replace=False)]
    )


def reference_keys(
    x: np.ndarray,
    labels: np.ndarray,
    count: int | None,
    rng: np.random.Generator,
    total_covariance: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    K2 and both K4 variants for every condition from ``count`` drawn positives and negatives.

    Parameters
    ----------
    x : np.ndarray
        Standardized train features.
    labels : np.ndarray
        Train states.
    count : int | None
        Records per class; None uses all defined records.
    rng : np.random.Generator
        Generator of the draws.
    total_covariance : np.ndarray
        Shrunk covariance of all train records (no labels used).

    Returns
    -------
    dict[str, np.ndarray]
        ``K2``, ``K4-all`` (total covariance) and ``K4-drawn`` (within-class covariance of the drawn
        records), each ``[conditions, features]``.
    """
    out = {name: np.empty((labels.shape[1], x.shape[1])) for name in ("K2", "K4-all", "K4-drawn")}
    for column in range(labels.shape[1]):
        y = labels[:, column]
        if count is None:
            rows = np.flatnonzero(~np.isnan(y))
        else:
            rows = draw_references(y, count, rng)
        fit = kk.within_class_covariance(x[rows], y[rows])
        difference = fit["positive_mean"] - fit["negative_mean"]
        out["K2"][column] = difference
        out["K4-all"][column] = np.linalg.solve(total_covariance, difference)
        out["K4-drawn"][column] = np.linalg.solve(fit["covariance"], difference)
    return out


def reference_curve(
    train_x: np.ndarray,
    train_labels: np.ndarray,
    valid_x: np.ndarray,
    valid_labels: np.ndarray,
    valid_sources: np.ndarray,
    total_covariance: np.ndarray,
    counts: tuple[int, ...] = REFERENCE_COUNTS,
    draws: int = REFERENCE_DRAWS,
    seed: int = REFERENCE_SEED,
) -> pd.DataFrame:
    """
    Score keys built from few reference records on validation, every draw.

    Parameters
    ----------
    train_x, train_labels : np.ndarray
        Standardized train features and states.
    valid_x, valid_labels, valid_sources : np.ndarray
        Validation features, states and sources.
    total_covariance : np.ndarray
        Shrunk covariance of all train records.
    counts : tuple[int, ...]
        Records per class; ``all`` (every defined record) is added once.
    draws : int
        Draws per count.
    seed : int
        Seed of the draws.

    Returns
    -------
    pd.DataFrame
        One row per variant, count, draw and condition with ``within`` and ``pooled`` AUROC.
    """
    rng = np.random.default_rng(seed)
    rows = []
    plan = [(count, draw) for count in counts for draw in range(draws)] + [(None, 0)]
    for count, draw in plan:
        keys = reference_keys(train_x, train_labels, count, rng, total_covariance)
        for variant, values in keys.items():
            metrics = condition_metrics(valid_labels, valid_x @ values.T, valid_sources)
            for column, condition in enumerate(CONDITIONS):
                rows.append(
                    {
                        "variant": variant,
                        "count": "all" if count is None else count,
                        "draw": draw,
                        "condition": condition,
                        "within": float(metrics["within"][column]),
                        "pooled": float(metrics["pooled"][column]),
                    }
                )
    return pd.DataFrame(rows)


def bootstrap_weights(
    groups: np.ndarray, draws: int = STABILITY_DRAWS, seed: int = STABILITY_SEED
) -> np.ndarray:
    """
    Record weights of whole-group bootstrap resamples of train.

    Parameters
    ----------
    groups : np.ndarray
        Group of each train record (patient for PTB-XL, record otherwise).
    draws : int
        Number of resamples.
    seed : int
        Seed of ``numpy.random.default_rng``.

    Returns
    -------
    np.ndarray
        ``[draws, records]`` integer weights.
    """
    members = patient_groups(groups)
    rng = np.random.default_rng(seed)
    return np.stack(
        [np.bincount(patient_resample(members, rng), minlength=len(groups)) for _ in range(draws)]
    )


def key_cosines(replicates: np.ndarray, full: np.ndarray) -> dict[str, float]:
    """
    Cosine summaries of resampled keys of one condition.

    Parameters
    ----------
    replicates : np.ndarray
        ``[draws, features]`` keys built on resamples.
    full : np.ndarray
        ``[features]`` key built on all train records.

    Returns
    -------
    dict[str, float]
        Median and 5th percentile of the pairwise cosines among replicates and of the cosine with the full
        key.
    """
    unit = kk.unit_rows(replicates)
    pairwise = (unit @ unit.T)[np.triu_indices(len(unit), 1)]
    to_full = unit @ (full / np.linalg.norm(full))
    return {
        "pairwise_median": float(np.median(pairwise)),
        "pairwise_p05": float(np.percentile(pairwise, 5)),
        "to_full_median": float(np.median(to_full)),
        "to_full_p05": float(np.percentile(to_full, 5)),
    }


def spearman_rows(
    score: np.ndarray, measurement: np.ndarray, sources: np.ndarray, minimum: int = 30
) -> list[dict[str, Any]]:
    """
    Spearman correlation of a score with a measurement, within each source with enough measured records.

    Parameters
    ----------
    score : np.ndarray
        ``[records]``.
    measurement : np.ndarray
        ``[records]``, NaN where unreliable or missing.
    sources : np.ndarray
        Source of each record.
    minimum : int
        Required measured records in a source.

    Returns
    -------
    list[dict[str, Any]]
        ``source``, ``records`` and ``spearman``.
    """
    rows = []
    for name in VALIDATION_SOURCES:
        inside = (sources == name) & np.isfinite(measurement)
        if inside.sum() < minimum or np.ptp(measurement[inside]) == 0:
            continue
        rho = spearmanr(score[inside], measurement[inside]).statistic
        rows.append({"source": name, "records": int(inside.sum()), "spearman": float(rho)})
    return rows


def binned_curve(
    score: np.ndarray, measurement: np.ndarray, sources: np.ndarray, bins: int = 10, minimum: int = 30
) -> list[dict[str, Any]]:
    """
    Median score per decile of the measurement, within each source.

    Parameters
    ----------
    score : np.ndarray
        ``[records]``.
    measurement : np.ndarray
        ``[records]``, NaN where unreliable.
    sources : np.ndarray
        Source of each record.
    bins : int
        Quantile bins.
    minimum : int
        Required measured records in a source.

    Returns
    -------
    list[dict[str, Any]]
        ``source``, ``bin``, ``records``, ``measurement_median`` and ``score_median``.
    """
    rows = []
    for name in VALIDATION_SOURCES:
        inside = (sources == name) & np.isfinite(measurement)
        if inside.sum() < minimum:
            continue
        values, scores = measurement[inside], score[inside]
        edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
        index = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)
        for position in np.unique(index):
            members = index == position
            rows.append(
                {
                    "source": name,
                    "bin": int(position),
                    "records": int(members.sum()),
                    "measurement_median": float(np.median(values[members])),
                    "score_median": float(np.median(scores[members])),
                }
            )
    return rows
