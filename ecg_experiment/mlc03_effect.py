"""
Part D of MLC-03: what training and validating on dataset v5 changes for P0 and H-All (validation only).

``docs/mlc03-data-quality-protocol-2026-11-11.md`` (Part D) is the contract: P0 and H-All are refitted on
v5 train with their MLC-02 recipes and scored on v5 validation, beside the same models trained on v4 (their
stored MLC-02 validation predictions) scored on the same v5 validation records and labels. Nothing is scored
on a test role: ``mlc02_feature_data.load_split`` refuses them, and v5 labels are read for train and
validation records only. The result does not decide the rules.

Recipes (``mlc02_feature_models``, ``mlc02_feature_runs``): P0 is one L2 logistic regression per condition on
standardized, row-normalized pooled xECG, C per condition from ``LOGISTIC_C`` by the per-condition validation
metric; H-All is the one-hidden-layer network on all features at MLC-02's chosen learning rate and weight
decay, three seeds, each seed's epoch chosen on the validation selection metric, the candidate being the mean
of the seeds. Intervals are paired patient-cluster bootstraps within each validation source
(``mlc02_bootstrap``), 2,000 draws; the selection metric's interval combines the sources draw by draw.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ecg_experiment import mlc03_dataset_v5 as mv
from ecg_experiment.mlc02_bootstrap import (
    DRAWS,
    SEED,
    bootstrap_metrics,
    bootstrap_p_value,
    cluster_resamples,
    holm_adjust,
    percentile_interval,
)
from ecg_experiment.mlc02_data import CONDITIONS, VALIDATION_SOURCES, check_states, load_predictions, subset
from ecg_experiment.mlc02_evaluation import auroc_by_condition, enters_macro, selection_metric
from ecg_experiment.mlc02_feature_data import OPEN_ROLES, load_split
from ecg_experiment.mlc02_feature_models import CANDIDATES

EFFECT = mv.QUALITY / "effect"
STORED = {
    "P0": CANDIDATES / "P0" / "validation__seeds-42.npz",
    "H-All": CANDIDATES / "H-All" / "validation__seeds-20261110-20261111-20261112.npz",
}
STORED_METRIC = {"P0": 0.9642697112976489, "H-All": 0.9705630596869196}
H_ALL_RATE = 3e-4
H_ALL_DECAY = 1e-4


def v5_states(keys: np.ndarray, manifest: Path = mv.V5 / "manifest.csv") -> tuple[np.ndarray, np.ndarray]:
    """
    Inclusion and space-B states of v4 train and validation records in dataset v5.

    Parameters
    ----------
    keys : np.ndarray
        Record keys of v4-included train and validation records.
    manifest : Path
        v5 ``manifest.csv``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``included`` (bool per key) and ``labels`` (``[keys, 14]`` states 1.0, 0.0 or NaN).

    Raises
    ------
    ValueError
        If a key is missing or belongs to a test role in v5.
    """
    columns = [f"b_{name}" for name in CONDITIONS]
    frame = pd.read_csv(
        manifest, usecols=["record_key", "role", "included", *columns], dtype={"record_key": str}
    )
    frame = frame[frame["role"].isin(OPEN_ROLES)].set_index("record_key")
    missing = set(keys) - set(frame.index)
    if missing:
        raise ValueError(f"{len(missing)} records are not open records of v5")
    rows = frame.loc[np.asarray(keys)]
    labels = rows[columns].to_numpy(dtype=np.float64)
    check_states(labels, "v5 space B")
    return rows["included"].to_numpy(dtype=bool), labels


def load_v5(names: tuple[str, ...]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """
    Train and validation records of v5 with their features, from the guarded v4 loader.

    Parameters
    ----------
    names : tuple[str, ...]
        Feature sets.

    Returns
    -------
    tuple
        v5 manifest arrays (v4 arrays of the records still included, ``labels`` replaced by the v5 states),
        their features, and a check record (records kept, label cells changed, no label created or flipped).

    Raises
    ------
    ValueError
        If a v5 label is defined where v4's is not, or a defined label differs.
    """
    data, features = load_split(OPEN_ROLES, names)
    included, labels = v5_states(data["record_key"])
    old = data["labels"]
    if (np.isnan(old) & ~np.isnan(labels)).any() or (~np.isnan(labels) & (labels != old)).any():
        raise ValueError("v5 creates or flips a train or validation label")
    data5 = subset({**data, "labels": labels}, included)
    features5 = {name: values[included] for name, values in features.items()}
    check = {
        "v4_records": len(included),
        "v5_records": int(included.sum()),
        "excluded": int((~included).sum()),
        "labels_made_undefined_on_kept_records": int((~np.isnan(old) & np.isnan(labels))[included].sum()),
    }
    return data5, features5, check


def stored_on(model: str, data: dict[str, np.ndarray], keep: np.ndarray | None = None) -> np.ndarray:
    """
    Read the stored MLC-02 validation probabilities of a v4-trained model, on all or some validation records.

    Parameters
    ----------
    model : str
        ``P0`` or ``H-All``.
    data : dict[str, np.ndarray]
        v4 validation manifest arrays, in manifest order.
    keep : np.ndarray | None
        Mask of the records to return; None returns all.

    Returns
    -------
    np.ndarray
        ``[records, 14]`` probabilities.
    """
    probability = load_predictions(STORED[model], data["record_key"])["probability"]
    return probability if keep is None else probability[keep]


def source_bootstrap(
    labels: np.ndarray, groups: np.ndarray, sources: np.ndarray, probabilities: dict[str, np.ndarray]
) -> dict[str, dict[str, Any]]:
    """
    Patient-cluster bootstrap of several models within each validation source, on shared resamples.

    Parameters
    ----------
    labels : np.ndarray
        Validation states.
    groups : np.ndarray
        Bootstrap group of each record.
    sources : np.ndarray
        Validation source of each record.
    probabilities : dict[str, np.ndarray]
        Model id to aligned probabilities.

    Returns
    -------
    dict[str, dict[str, Any]]
        Per source with an entering condition: ``resamples`` identity and ``models`` (``bootstrap_metrics``
        output per model, without calibration).
    """
    out = {}
    for source in VALIDATION_SOURCES:
        mask = sources == source
        if not mask.any() or not enters_macro(labels[mask]).any():
            continue
        resamples = cluster_resamples(groups[mask], labels[mask], DRAWS, SEED)
        models = {
            name: bootstrap_metrics(labels[mask], probability[mask], resamples, None, calibration=False)
            for name, probability in probabilities.items()
        }
        identity = {
            key: resamples[key] for key in ("draws", "redraws", "seed", "groups", "records", "fingerprint")
        }
        out[source] = {"resamples": identity, "models": models}
    return out


def interval(values: np.ndarray) -> tuple[float, float]:
    """
    Percentile 95% interval of resampled values.

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


def selection_draws(boot: dict[str, dict[str, Any]], model: str) -> np.ndarray:
    """
    Resampled selection metric: the mean over sources of each draw's within-source macro AUROC.

    Parameters
    ----------
    boot : dict[str, dict[str, Any]]
        Output of ``source_bootstrap``.
    model : str
        Model id.

    Returns
    -------
    np.ndarray
        ``[draws]``.
    """
    return np.mean([part["models"][model]["macro_auroc"] for part in boot.values()], axis=0)


def condition_draws(boot: dict[str, dict[str, Any]], model: str) -> np.ndarray:
    """
    Resampled per-condition metric: a condition's within-source AUROC averaged over the sources it enters.

    Parameters
    ----------
    boot : dict[str, dict[str, Any]]
        Output of ``source_bootstrap``.
    model : str
        Model id.

    Returns
    -------
    np.ndarray
        ``[draws, 14]``, NaN for a condition that enters no source.
    """
    stacked = np.stack([part["models"][model]["samples"]["auroc"] for part in boot.values()])
    entering = np.stack([part["models"][model]["entering"] for part in boot.values()])
    values = np.where(entering[:, None, :], stacked, np.nan)
    with np.errstate(all="ignore"):
        counts = entering.sum(axis=0)
        means = np.nansum(values, axis=0) / np.where(counts > 0, counts, np.nan)
    return means


def compare(
    labels: np.ndarray,
    groups: np.ndarray,
    sources: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    names: tuple[str, str],
) -> dict[str, Any]:
    """
    Paired comparison of two models on one validation set: first minus second.

    Parameters
    ----------
    labels : np.ndarray
        Validation states.
    groups : np.ndarray
        Bootstrap groups.
    sources : np.ndarray
        Validation sources.
    first, second : np.ndarray
        Aligned probabilities.
    names : tuple[str, str]
        Model ids of ``first`` and ``second``.

    Returns
    -------
    dict[str, Any]
        ``selection`` (both values, their intervals, the difference and its interval), ``conditions`` (per
        condition: the metric averaged over entering sources for both, difference, interval, bootstrap p and
        Holm-adjusted p over the conditions), ``by_source`` (per source and entering condition: both AUROCs,
        difference, interval, p and Holm-adjusted p within the source, with positives and negatives) and the
        resampling identity per source.
    """
    boot = source_bootstrap(labels, groups, sources, dict(zip(names, (first, second), strict=True)))
    observed = {
        name: selection_metric(labels, values, sources)
        for name, values in zip(names, (first, second), strict=True)
    }
    draws = {name: selection_draws(boot, name) for name in names}
    difference = draws[names[0]] - draws[names[1]]
    selection = {
        name: {"value": observed[name]["selection_metric"], "ci": interval(draws[name])} for name in names
    }
    selection["difference"] = {
        "value": observed[names[0]]["selection_metric"] - observed[names[1]]["selection_metric"],
        "ci": interval(difference),
    }
    per_condition = {name: condition_draws(boot, name) for name in names}
    differences = per_condition[names[0]] - per_condition[names[1]]
    p_values = np.array(
        [
            bootstrap_p_value(differences[:, c]) if np.isfinite(differences[:, c]).any() else np.nan
            for c in range(len(CONDITIONS))
        ]
    )
    holm = holm_adjust(p_values)
    conditions = []
    for column, condition in enumerate(CONDITIONS):
        values = [observed[name]["per_condition"][condition] for name in names]
        if not math.isfinite(values[0]):
            continue
        conditions.append(
            {
                "condition": condition,
                "sources": observed[names[0]]["per_condition_sources"][condition],
                names[0]: values[0],
                names[1]: values[1],
                "difference": values[0] - values[1],
                "ci_low": interval(differences[:, column])[0],
                "ci_high": interval(differences[:, column])[1],
                "p_value": float(p_values[column]),
                "p_holm": float(holm[column]),
            }
        )
    by_source = []
    for source, part in boot.items():
        mask = sources == source
        a, b = (part["models"][name] for name in names)
        source_differences = a["samples"]["auroc"] - b["samples"]["auroc"]
        source_p = np.array(
            [
                bootstrap_p_value(source_differences[:, c]) if a["entering"][c] else np.nan
                for c in range(len(CONDITIONS))
            ]
        )
        source_holm = holm_adjust(source_p)
        lows, highs, _ = percentile_interval(source_differences)
        positives = (labels[mask] == 1).sum(axis=0)
        negatives = (labels[mask] == 0).sum(axis=0)
        for column, condition in enumerate(CONDITIONS):
            if not a["entering"][column]:
                continue
            by_source.append(
                {
                    "source": source,
                    "condition": condition,
                    "positives": int(positives[column]),
                    "negatives": int(negatives[column]),
                    names[0]: float(a["observed_auroc"][column]),
                    names[1]: float(b["observed_auroc"][column]),
                    "difference": float(a["observed_auroc"][column] - b["observed_auroc"][column]),
                    "ci_low": float(lows[column]),
                    "ci_high": float(highs[column]),
                    "p_value": float(source_p[column]),
                    "p_holm_within_source": float(source_holm[column]),
                }
            )
    return {
        "selection": selection,
        "macro_by_source": {name: observed[name]["macro_auroc_by_source"] for name in names},
        "conditions": conditions,
        "by_source": by_source,
        "resamples": {source: part["resamples"] for source, part in boot.items()},
    }


def per_source_auroc(labels: np.ndarray, probability: np.ndarray, sources: np.ndarray) -> pd.DataFrame:
    """
    Within-source AUROC of every condition, with support and whether it enters the source's macro.

    Parameters
    ----------
    labels : np.ndarray
        Validation states.
    probability : np.ndarray
        Aligned probabilities.
    sources : np.ndarray
        Validation sources.

    Returns
    -------
    pd.DataFrame
        One row per source and condition.
    """
    rows = []
    for source in VALIDATION_SOURCES:
        mask = sources == source
        if not mask.any():
            continue
        auroc = auroc_by_condition(labels[mask], probability[mask])
        entering = enters_macro(labels[mask])
        for column, condition in enumerate(CONDITIONS):
            rows.append(
                {
                    "source": source,
                    "condition": condition,
                    "positives": int((labels[mask, column] == 1).sum()),
                    "negatives": int((labels[mask, column] == 0).sum()),
                    "undefined": int(np.isnan(labels[mask, column]).sum()),
                    "enters_macro": bool(entering[column]),
                    "auroc": float(auroc[column]),
                }
            )
    return pd.DataFrame(rows)


WORKER: dict[str, Any] = {}


def init_fit_worker(directory: str, threads: int) -> None:
    """
    Memory-map one split's logistic inputs in a worker and limit its BLAS threads.

    Parameters
    ----------
    directory : str
        Folder with ``z_fit.npy``, ``y_fit.npy`` and ``z_score.npy``.
    threads : int
        BLAS and OpenMP threads of this worker.
    """
    from threadpoolctl import threadpool_limits

    WORKER["limits"] = threadpool_limits(threads)
    folder = Path(directory)
    WORKER.update(
        z_fit=np.load(folder / "z_fit.npy", mmap_mode="r"),
        y_fit=np.load(folder / "y_fit.npy"),
        z_score=np.load(folder / "z_score.npy", mmap_mode="r"),
    )


def fit_task(task: tuple[int, float]) -> dict[str, Any]:
    """
    Fit P0's logistic regression for one condition at one C and score the split's scored records.

    The same steps as ``mlc02_feature_runs.logistic_task``.

    Parameters
    ----------
    task : tuple[int, float]
        Condition column and C.

    Returns
    -------
    dict[str, Any]
        ``fit_logistic`` output with ``column``, ``c`` and ``probability``.
    """
    from scipy.special import expit

    from ecg_experiment.mlc02_feature_models import fit_logistic

    column, c = task
    fit = fit_logistic(np.asarray(WORKER["z_fit"]), WORKER["y_fit"][:, column], c)
    probability = expit(np.asarray(WORKER["z_score"]) @ fit["coef"] + fit["intercept"])
    return {**fit, "column": column, "c": c, "probability": probability}
