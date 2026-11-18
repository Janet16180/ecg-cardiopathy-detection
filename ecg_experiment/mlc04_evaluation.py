"""
Metrics, intervals, comparisons and readings of MLC-04.

``docs/mlc04-final-test-protocol-2026-11-11.md`` fixes them:

- The metric is the MLC-02 selection metric: the macro AUROC within each Challenge source over the conditions
  with at least ``PRIMARY_MINIMUM`` positives and as many negatives in that source, averaged over the sources
  where any condition enters. The same metric with the validation entry rule (``VALIDATION_MINIMUM``) is
  descriptive.
- Intervals: 2,000 record resamples within each source (``mlc02_bootstrap.cluster_resamples``, built on
  ``intervals.patient_groups`` and ``patient_resample``; each record is its own group, a draw is redrawn if
  an entering condition loses its positives or negatives), shared by every model; the metric's resample is
  the mean over sources of each draw's within-source macro AUROC, and its interval the 2.5 and 97.5
  percentiles.
- Primary reading (protocol Amendment 1): confirmed if the lower bound is above the like-for-like validation
  value (``LIKE_FOR_LIKE_METRIC``, the same five-source, entry-100 metric of P0-v5 on the validation groups in
  the dry run) minus ``CONFIRMATION_MARGIN``, not confirmed if the upper bound is below that bound, undecided
  otherwise. The six-source validation value (``VALIDATION_METRIC``) and its bound are printed beside it
  without a reading.
- Secondary comparisons, paired on the same resamples, Holm-adjusted over the three: the two-sided bootstrap
  p-value of each difference against its null value (0, 0 and ``KEY_MARGIN``) is
  ``mlc02_bootstrap.bootstrap_p_value`` of the resampled differences minus the null.

K6 scores are not probabilities; they are mapped into (0, 1) by their ranks within the evaluated set
(``key05_analysis.rank_unit``), which leaves every AUROC and average precision unchanged.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ecg_experiment.key05_analysis import rank_unit
from ecg_experiment.mlc02_bootstrap import (
    DRAWS,
    SEED,
    bootstrap_metrics,
    bootstrap_p_value,
    cluster_resamples,
    holm_adjust,
    percentile_interval,
)
from ecg_experiment.mlc02_data import CHALLENGE_SOURCES, CONDITIONS
from ecg_experiment.mlc02_evaluation import auroc_by_condition, check_scores, condition_metrics, enters_macro

PRIMARY_MINIMUM = 100
VALIDATION_MINIMUM = 20
VALIDATION_METRIC = 0.9697697420282113
LIKE_FOR_LIKE_METRIC = 0.9610687901468651
CONFIRMATION_MARGIN = 0.010
KEY_MARGIN = -0.010
ALPHA = 0.05
RELIABILITY_BINS = 10
MIN_BIN_RECORDS = 20
INTERVAL_METRICS = ("auroc", "average_precision", "brier", "sensitivity", "specificity")


def as_probability(name: str, values: np.ndarray) -> np.ndarray:
    """
    Model output as a probability matrix: K6 scores through ``rank_unit``, others unchanged.

    Parameters
    ----------
    name : str
        Model id.
    values : np.ndarray
        ``[records, 14]`` output of the model on one evaluated set.

    Returns
    -------
    np.ndarray
        Values in [0, 1].
    """
    return rank_unit(values) if name == "K6" else np.asarray(values, dtype=np.float64)


def source_metric(
    labels: np.ndarray, probability: np.ndarray, sources: np.ndarray, minimum: int
) -> dict[str, Any]:
    """
    Mean over Challenge sources of the within-source macro AUROC, with a given entry rule.

    Parameters
    ----------
    labels : np.ndarray
        ``[records, 14]`` states.
    probability : np.ndarray
        Aligned probabilities.
    sources : np.ndarray
        Source of each record.
    minimum : int
        Positives and negatives a condition needs in a source to enter its macro.

    Returns
    -------
    dict[str, Any]
        ``metric``, ``macro_by_source``, ``entering_by_source`` and ``excluded_sources`` (no condition
        enters).

    Raises
    ------
    ValueError
        If a record has another source or no source has an entering condition.
    """
    labels, probability = check_scores(labels, probability)
    sources = np.asarray(sources, dtype=str)
    if not set(sources) <= set(CHALLENGE_SOURCES):
        raise ValueError(f"sources outside the Challenge: {sorted(set(sources) - set(CHALLENGE_SOURCES))}")
    by_source, entering = {}, {}
    for source in [name for name in CHALLENGE_SOURCES if name in set(sources)]:
        mask = sources == source
        enter = enters_macro(labels[mask], minimum)
        entering[source] = [CONDITIONS[c] for c in np.flatnonzero(enter)]
        if enter.any():
            by_source[source] = float(auroc_by_condition(labels[mask], probability[mask])[enter].mean())
    if not by_source:
        raise ValueError("no source has a condition that enters the macro")
    return {
        "metric": float(np.mean(list(by_source.values()))),
        "macro_by_source": by_source,
        "entering_by_source": entering,
        "excluded_sources": sorted(set(entering) - set(by_source)),
    }


def source_resamples(
    labels: np.ndarray,
    groups: np.ndarray,
    sources: np.ndarray,
    minimum: int,
    draws: int = DRAWS,
    seed: int = SEED,
) -> dict[str, dict[str, Any]]:
    """
    Record resamples within every source where a condition enters, shared by every model.

    Parameters
    ----------
    labels : np.ndarray
        ``[records, 14]`` states.
    groups : np.ndarray
        Bootstrap group of each record (the record key: the Challenge has no patient identifier).
    sources : np.ndarray
        Source of each record.
    minimum : int
        Entry rule.
    draws : int
        Resamples per source.
    seed : int
        Seed of each source's generator.

    Returns
    -------
    dict[str, dict[str, Any]]
        Source to ``cluster_resamples`` output.
    """
    out = {}
    for source in CHALLENGE_SOURCES:
        mask = np.asarray(sources) == source
        if mask.any() and enters_macro(labels[mask], minimum).any():
            out[source] = cluster_resamples(np.asarray(groups)[mask], labels[mask], draws, seed, minimum)
    return out


def source_boot(
    labels: np.ndarray,
    probability: np.ndarray,
    sources: np.ndarray,
    resamples: dict[str, dict[str, Any]],
    thresholds: np.ndarray | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Resampled metrics of one model within every resampled source.

    Parameters
    ----------
    labels : np.ndarray
        ``[records, 14]`` states.
    probability : np.ndarray
        Aligned probabilities.
    sources : np.ndarray
        Source of each record.
    resamples : dict[str, dict[str, Any]]
        Output of ``source_resamples`` for the same records and labels.
    thresholds : np.ndarray | None
        Frozen thresholds, for sensitivity and specificity.

    Returns
    -------
    dict[str, dict[str, Any]]
        Source to ``bootstrap_metrics`` output (no calibration fits).
    """
    out = {}
    for source, part in resamples.items():
        mask = np.asarray(sources) == source
        out[source] = bootstrap_metrics(labels[mask], probability[mask], part, thresholds, calibration=False)
    return out


def metric_draws(boot: dict[str, dict[str, Any]]) -> np.ndarray:
    """
    Resampled metric: the mean over sources of each draw's within-source macro AUROC.

    Parameters
    ----------
    boot : dict[str, dict[str, Any]]
        Output of ``source_boot``.

    Returns
    -------
    np.ndarray
        ``[draws]``.
    """
    return np.mean([part["macro_auroc"] for part in boot.values()], axis=0)


def interval(values: np.ndarray) -> list[float]:
    """
    Percentile 95% interval of resampled values.

    Parameters
    ----------
    values : np.ndarray
        ``[draws]``.

    Returns
    -------
    list[float]
        Lower and upper bound.
    """
    low, high, _ = percentile_interval(values)
    return [float(low), float(high)]


def metric_summary(boot: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Observed metric, its interval and the per-source macro AUROC of one model.

    Parameters
    ----------
    boot : dict[str, dict[str, Any]]
        Output of ``source_boot``.

    Returns
    -------
    dict[str, Any]
        ``value``, ``ci``, ``macro_by_source`` (value and interval per source) and ``sources``.
    """
    return {
        "value": float(np.mean([part["observed_macro_auroc"] for part in boot.values()])),
        "ci": interval(metric_draws(boot)),
        "macro_by_source": {
            source: {
                "value": part["observed_macro_auroc"],
                "ci": interval(part["macro_auroc"]),
                "entering": [CONDITIONS[c] for c in np.flatnonzero(part["entering"])],
            }
            for source, part in boot.items()
        },
        "sources": list(boot),
    }


def compare(
    first: dict[str, dict[str, Any]], second: dict[str, dict[str, Any]], null: float = 0.0
) -> dict[str, Any]:
    """
    Paired difference of the metric, first minus second, on the same resamples.

    Parameters
    ----------
    first, second : dict[str, dict[str, Any]]
        ``source_boot`` outputs of two models on the same records, labels and resamples.
    null : float
        Null value of the difference for the bootstrap p-value.

    Returns
    -------
    dict[str, Any]
        ``difference``, ``ci``, ``p_value`` (two-sided, against ``null``), ``null`` and ``by_source``.

    Raises
    ------
    ValueError
        If the two outputs come from different resamples.
    """
    if list(first) != list(second) or any(first[s]["fingerprint"] != second[s]["fingerprint"] for s in first):
        raise ValueError("paired comparisons need the same resamples")
    draws = metric_draws(first) - metric_draws(second)
    observed = float(
        np.mean([first[s]["observed_macro_auroc"] - second[s]["observed_macro_auroc"] for s in first])
    )
    by_source = {
        s: {
            "difference": float(first[s]["observed_macro_auroc"] - second[s]["observed_macro_auroc"]),
            "ci": interval(first[s]["macro_auroc"] - second[s]["macro_auroc"]),
        }
        for s in first
    }
    return {
        "difference": observed,
        "ci": interval(draws),
        "p_value": bootstrap_p_value(draws - null),
        "null": null,
        "by_source": by_source,
    }


def primary_reading(
    low: float, high: float, validation: float = LIKE_FOR_LIKE_METRIC, margin: float = CONFIRMATION_MARGIN
) -> str:
    """
    Read the test interval of the metric against the validation value.

    Parameters
    ----------
    low, high : float
        95% interval of the test metric.
    validation : float
        Validation value of the metric.
    margin : float
        Allowed shortfall.

    Returns
    -------
    str
        ``confirmed`` (lower bound above ``validation - margin``), ``not confirmed`` (upper bound below it) or
        ``undecided``.
    """
    bound = validation - margin
    if low > bound:
        return "confirmed"
    if high < bound:
        return "not confirmed"
    return "undecided"


def direction(adjusted: float, difference: float, null: float, names: tuple[str, str, str]) -> str:
    """
    Read one Holm-adjusted comparison: above, below or not different from its null value.

    Parameters
    ----------
    adjusted : float
        Holm-adjusted p-value.
    difference : float
        Observed difference.
    null : float
        Null value.
    names : tuple[str, str, str]
        Readings for above, below and neither.

    Returns
    -------
    str
        One of ``names``.
    """
    if adjusted < ALPHA and difference > null:
        return names[0]
    if adjusted < ALPHA and difference < null:
        return names[1]
    return names[2]


def secondary_readings(
    cleaning: dict[str, Any], cleaning_v4: dict[str, Any], larger: dict[str, Any], key: dict[str, Any]
) -> dict[str, Any]:
    """
    Holm-adjust the three secondary comparisons and read each by the protocol's rule.

    The cleaning comparison enters the Holm family with its v5-label p-value; its v4-label interval is a
    condition read at the unadjusted 95% level (cleaning helps only if that interval is not below zero).

    Parameters
    ----------
    cleaning : dict[str, Any]
        ``compare`` of P0-v5 and P0-v4 on v5-rule labels.
    cleaning_v4 : dict[str, Any]
        The same on v4 labels.
    larger : dict[str, Any]
        ``compare`` of H-All-v5 and P0-v5.
    key : dict[str, Any]
        ``compare`` of K6 and P0-v5 against ``KEY_MARGIN``.

    Returns
    -------
    dict[str, Any]
        ``holm`` (adjusted p-values) and a reading per comparison.
    """
    adjusted = holm_adjust(np.array([cleaning["p_value"], larger["p_value"], key["p_value"]]))
    v5_reading = direction(adjusted[0], cleaning["difference"], 0.0, ("above", "below", "not shown"))
    helps = v5_reading == "above" and cleaning_v4["ci"][1] >= 0.0
    return {
        "holm": {
            "cleaning": float(adjusted[0]),
            "larger_model": float(adjusted[1]),
            "key": float(adjusted[2]),
        },
        "cleaning": {
            "v5_labels": v5_reading,
            "v4_labels_not_below_zero": bool(cleaning_v4["ci"][1] >= 0.0),
            "reading": "cleaning helps"
            if helps
            else ("cleaning lowers the metric" if v5_reading == "below" else "no gain from cleaning shown"),
        },
        "larger_model": direction(
            adjusted[1],
            larger["difference"],
            0.0,
            ("H-All above P0", "H-All below P0", "no difference shown"),
        ),
        "key": direction(
            adjusted[2], key["difference"], KEY_MARGIN, ("equivalent", "clearly worse", "undecided")
        ),
    }


def resampled_cells(boot: dict[str, Any], column: int) -> dict[str, float]:
    """
    Interval bounds of one condition's resampled metrics within one source.

    Parameters
    ----------
    boot : dict[str, Any]
        One source's ``bootstrap_metrics`` output.
    column : int
        Condition column.

    Returns
    -------
    dict[str, float]
        ``<metric>_ci_low`` and ``<metric>_ci_high`` for every resampled metric of ``INTERVAL_METRICS``.
    """
    out = {}
    for name in INTERVAL_METRICS:
        if name in boot["samples"]:
            low, high, _ = percentile_interval(boot["samples"][name][:, column])
            out[f"{name}_ci_low"], out[f"{name}_ci_high"] = float(low), float(high)
    return out


def condition_rows(
    labels: np.ndarray,
    probabilities: dict[str, np.ndarray],
    sources: np.ndarray,
    thresholds: np.ndarray,
    boot: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """
    Per source and condition: support, every model's AUROC, and P0-v5's metrics with intervals.

    Parameters
    ----------
    labels : np.ndarray
        ``[records, 14]`` states.
    probabilities : dict[str, np.ndarray]
        Model id to aligned probabilities (K6 already rank-mapped); must hold ``P0-v5``.
    sources : np.ndarray
        Source of each record.
    thresholds : np.ndarray
        Frozen thresholds of P0-v5.
    boot : dict[str, dict[str, Any]]
        ``source_boot`` of P0-v5 with the thresholds, on resamples of the validation entry rule.

    Returns
    -------
    pd.DataFrame
        One row per source and condition; intervals only where the condition enters the source's resamples.
    """
    rows = []
    for source in [name for name in CHALLENGE_SOURCES if name in set(sources)]:
        mask = np.asarray(sources) == source
        aurocs = {
            name: auroc_by_condition(labels[mask], values[mask]) for name, values in probabilities.items()
        }
        for column, condition in enumerate(CONDITIONS):
            point = condition_metrics(
                labels[mask, column], probabilities["P0-v5"][mask, column], float(thresholds[column])
            )
            row = {
                "source": source,
                "condition": condition,
                **point,
                "enters_validation_rule": bool(
                    point["positives"] >= VALIDATION_MINIMUM and point["negatives"] >= VALIDATION_MINIMUM
                ),
                "enters_primary_rule": bool(
                    point["positives"] >= PRIMARY_MINIMUM and point["negatives"] >= PRIMARY_MINIMUM
                ),
            }
            row |= {f"auroc_{name}": float(values[column]) for name, values in aurocs.items()}
            if source in boot and boot[source]["entering"][column]:
                row |= resampled_cells(boot[source], column)
            rows.append(row)
    calibration = ["enters_macro", "calibration_intercept", "calibration_slope", "calibration_in_the_large"]
    return pd.DataFrame(rows).drop(columns=calibration)


def reliability(
    labels: np.ndarray, probability: np.ndarray, sources: np.ndarray, bins: int = RELIABILITY_BINS
) -> pd.DataFrame:
    """
    Reliability table of one model: equal-width probability bins within each source and condition.

    Parameters
    ----------
    labels : np.ndarray
        ``[records, 14]`` states.
    probability : np.ndarray
        Aligned probabilities.
    sources : np.ndarray
        Source of each record.
    bins : int
        Number of bins over [0, 1].

    Returns
    -------
    pd.DataFrame
        ``source``, ``condition``, ``bin_low``, ``bin_high``, ``records``, ``mean_probability`` and
        ``observed_rate`` for every non-empty bin of the defined entries.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for source in [name for name in CHALLENGE_SOURCES if name in set(sources)]:
        mask = np.asarray(sources) == source
        for column, condition in enumerate(CONDITIONS):
            y, p = labels[mask, column], probability[mask, column]
            defined = ~np.isnan(y)
            which = np.clip(np.digitize(p[defined], edges[1:-1]), 0, bins - 1)
            for b in np.unique(which):
                inside = which == b
                rows.append(
                    {
                        "source": source,
                        "condition": condition,
                        "bin_low": float(edges[b]),
                        "bin_high": float(edges[b + 1]),
                        "records": int(inside.sum()),
                        "mean_probability": float(p[defined][inside].mean()),
                        "observed_rate": float(y[defined][inside].mean()),
                    }
                )
    return pd.DataFrame(rows)


def support_rows(labels: np.ndarray, sources: np.ndarray) -> pd.DataFrame:
    """
    Positives, negatives and undefined entries per source and condition.

    Parameters
    ----------
    labels : np.ndarray
        ``[records, 14]`` states.
    sources : np.ndarray
        Source of each record.

    Returns
    -------
    pd.DataFrame
        One row per source and condition, with the records of the source.
    """
    rows = []
    for source in [name for name in CHALLENGE_SOURCES if name in set(sources)]:
        part = labels[np.asarray(sources) == source]
        for column, condition in enumerate(CONDITIONS):
            rows.append(
                {
                    "source": source,
                    "condition": condition,
                    "records": len(part),
                    "positives": int((part[:, column] == 1).sum()),
                    "negatives": int((part[:, column] == 0).sum()),
                    "undefined": int(np.isnan(part[:, column]).sum()),
                }
            )
    return pd.DataFrame(rows)


def evaluate_set(
    data: dict[str, np.ndarray], outputs: dict[str, np.ndarray], minimum: int, names: tuple[str, ...]
) -> dict[str, Any]:
    """
    Resamples and per-model metric summaries of one labelled set.

    Parameters
    ----------
    data : dict[str, np.ndarray]
        ``record_key``, ``source`` and ``labels`` of the set.
    outputs : dict[str, np.ndarray]
        Model id to raw output aligned with the set.
    minimum : int
        Entry rule.
    names : tuple[str, ...]
        Models to resample.

    Returns
    -------
    dict[str, Any]
        ``resamples`` (source to ``cluster_resamples`` output), ``identity`` (their identity per source),
        ``boot`` (model to ``source_boot``) and ``summary`` (model to ``metric_summary``).
    """
    resamples = source_resamples(data["labels"], data["record_key"], data["source"], minimum)
    boot = {
        name: source_boot(data["labels"], as_probability(name, outputs[name]), data["source"], resamples)
        for name in names
    }
    identity = {
        source: {
            key: part[key]
            for key in ("draws", "redraws", "seed", "groups", "records", "fingerprint", "labels_sha256")
        }
        for source, part in resamples.items()
    }
    return {
        "resamples": resamples,
        "identity": identity,
        "boot": boot,
        "summary": {name: metric_summary(boot[name]) for name in names},
    }


def evaluate(
    v5: dict[str, np.ndarray],
    v4: dict[str, np.ndarray],
    outputs_v5: dict[str, np.ndarray],
    outputs_v4: dict[str, np.ndarray],
    thresholds: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Every number of the protocol for one evaluated role.

    Parameters
    ----------
    v5, v4 : dict[str, np.ndarray]
        ``record_key``, ``source`` and ``labels`` of the v5-included records with v5-rule labels, and of the
        v4-included records with v4 labels.
    outputs_v5, outputs_v4 : dict[str, np.ndarray]
        Model id to raw output aligned with ``v5`` and ``v4``.
    thresholds : np.ndarray
        Frozen thresholds of P0-v5, in ``CONDITIONS`` order.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        The result (JSON-ready) and the tables (``conditions``, ``reliability``, ``support_v5``,
        ``support_v4`` as DataFrames).
    """
    models = tuple(outputs_v5)
    primary = evaluate_set(v5, outputs_v5, PRIMARY_MINIMUM, models)
    on_v4 = evaluate_set(v4, outputs_v4, PRIMARY_MINIMUM, ("P0-v5", "P0-v4"))
    descriptive = evaluate_set(v5, outputs_v5, VALIDATION_MINIMUM, models)
    boot = primary["boot"]
    comparisons = {
        "cleaning_v5_labels": compare(boot["P0-v5"], boot["P0-v4"]),
        "cleaning_v4_labels": compare(on_v4["boot"]["P0-v5"], on_v4["boot"]["P0-v4"]),
        "larger_model": compare(boot["H-All-v5"], boot["P0-v5"]),
        "key": compare(boot["K6"], boot["P0-v5"], KEY_MARGIN),
    }
    value = primary["summary"]["P0-v5"]
    result = {
        "records": {"v5": len(v5["record_key"]), "v4": len(v4["record_key"])},
        "primary": {
            "model": "P0-v5",
            "value": value["value"],
            "ci": value["ci"],
            "validation_value": LIKE_FOR_LIKE_METRIC,
            "bound": LIKE_FOR_LIKE_METRIC - CONFIRMATION_MARGIN,
            "reading": primary_reading(*value["ci"]),
            "six_source_validation_value": VALIDATION_METRIC,
            "six_source_bound": VALIDATION_METRIC - CONFIRMATION_MARGIN,
            "entering_by_source": {s: v["entering"] for s, v in value["macro_by_source"].items()},
        },
        "models_primary_rule": primary["summary"],
        "models_v4_labels": on_v4["summary"],
        "models_validation_rule": descriptive["summary"],
        "comparisons": comparisons,
        "readings": secondary_readings(
            comparisons["cleaning_v5_labels"],
            comparisons["cleaning_v4_labels"],
            comparisons["larger_model"],
            comparisons["key"],
        ),
        "resamples": {
            "primary_rule": primary["identity"],
            "v4_labels": on_v4["identity"],
            "validation_rule": descriptive["identity"],
        },
    }
    probabilities = {name: as_probability(name, values) for name, values in outputs_v5.items()}
    p0_boot = source_boot(
        v5["labels"], probabilities["P0-v5"], v5["source"], descriptive["resamples"], thresholds
    )
    bins = reliability(v5["labels"], probabilities["P0-v5"], v5["source"])
    conditions = condition_rows(v5["labels"], probabilities, v5["source"], thresholds, p0_boot)
    tables = {
        "conditions": conditions.merge(calibration_gaps(bins), on=["source", "condition"], how="left"),
        "reliability": bins,
        "support_v5": support_rows(v5["labels"], v5["source"]),
        "support_v4": support_rows(v4["labels"], v4["source"]),
    }
    return result, tables


def calibration_gaps(bins: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise a reliability table by the gaps between mean probability and observed rate of its bins.

    The record-weighted mean of the absolute gaps, and the largest gap among bins with enough records.

    Parameters
    ----------
    bins : pd.DataFrame
        Output of ``reliability``.

    Returns
    -------
    pd.DataFrame
        ``source``, ``condition``, ``calibration_gap_mean`` and ``calibration_gap_max`` (bins of at least
        ``MIN_BIN_RECORDS`` records for the largest gap).
    """
    frame = bins.assign(gap=(bins["mean_probability"] - bins["observed_rate"]).abs())
    rows = []
    for (source, condition), part in frame.groupby(["source", "condition"], sort=False):
        large = part[part["records"] >= MIN_BIN_RECORDS]
        rows.append(
            {
                "source": source,
                "condition": condition,
                "calibration_gap_mean": float((part["gap"] * part["records"]).sum() / part["records"].sum()),
                "calibration_gap_max": float(large["gap"].max()) if len(large) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def json_ready(value: Any) -> Any:
    """
    Convert NumPy values inside a nested structure to plain Python, NaN to None.

    Parameters
    ----------
    value : Any
        Nested dictionaries, lists and scalars.

    Returns
    -------
    Any
        The same structure, JSON-serializable.
    """
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
