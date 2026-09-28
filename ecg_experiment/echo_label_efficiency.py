"""Rows, paired draws and summaries of the Experiment 028 EchoNext label-efficiency study.

EchoNext is credentialed: these functions return arrays and aggregates for local use, and callers write only
aggregates to documents.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .eda.echonext import COMPOSITE
from .label_efficiency import draw_subset, paired_summary, reading, summarize

Plan = tuple[str, int, int, np.ndarray]


def split_rows(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Usable training rows, all validation rows and usable validation rows.

    Parameters
    ----------
    rows : pd.DataFrame
        EchoNext cache rows indexed by ``ecg_key``, with ``split``, ``use`` and the composite label.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Training pool, all validation rows and the usable validation rows, in cache order.

    Raises
    ------
    ValueError
        If the composite label is not binary or a usable validation patient has several ECGs.
    """
    if not rows[COMPOSITE].isin((0, 1)).all():
        raise ValueError("EchoNext composite label malformed")
    train = rows[(rows["split"] == "train") & rows["use"]]
    val_all = rows[rows["split"] == "val"]
    val = val_all[val_all["use"]]
    if val["patient_key"].duplicated().any():
        raise ValueError("A validation patient has several usable ECGs")
    return train, val_all, val


def check_keys(train_keys: np.ndarray, val_keys: np.ndarray, train: pd.DataFrame,
               val_all: pd.DataFrame) -> None:
    """
    Require the cached feature keys to equal the row keys, in order.

    Parameters
    ----------
    train_keys, val_keys : np.ndarray
        ``train_ecg_keys`` and ``val_all_ecg_keys`` of the feature file.
    train, val_all : pd.DataFrame
        Usable training rows and all validation rows.

    Raises
    ------
    ValueError
        If either key array differs from its rows.
    """
    if not np.array_equal(train_keys, train.index.to_numpy()):
        raise ValueError("Training feature keys differ from the usable training rows")
    if not np.array_equal(val_keys, val_all.index.to_numpy()):
        raise ValueError("Validation feature keys differ from the validation rows")


def draw_plans(patients: np.ndarray, y: np.ndarray, budgets: tuple[int, ...], draws: int,
               seed: int) -> list[Plan]:
    """
    Paired draws of every budget, then the whole pool as budget ``"all"``.

    Parameters
    ----------
    patients : np.ndarray
        Patient ID of each pool ECG.
    y : np.ndarray
        Binary label of each pool ECG.
    budgets : tuple[int, ...]
        Label budgets below all.
    draws : int
        Draws per budget.
    seed : int
        Base seed; draw ``d`` of every budget uses ``seed + d``.

    Returns
    -------
    list[Plan]
        Budget label, draw, seed and sorted pool positions of every draw.
    """
    plans = [(str(size), draw, seed + draw, draw_subset(patients, y, size, seed + draw))
             for size in budgets for draw in range(draws)]
    plans.append(("all", 0, seed, np.arange(len(y))))
    return plans


def gain_target(auroc: float, fraction: float) -> float:
    """
    AUROC that keeps ``fraction`` of the gain over chance.

    Parameters
    ----------
    auroc : float
        Reference AUROC.
    fraction : float
        Share of ``auroc - 0.5`` to keep.

    Returns
    -------
    float
        ``0.5 + fraction * (auroc - 0.5)``.
    """
    return 0.5 + fraction * (auroc - 0.5)


def smallest_budget_by_draws(values: dict[int, np.ndarray], target: float,
                             threshold: float = 0.9) -> int | None:
    """
    Smallest budget at which enough draws reach a target.

    Parameters
    ----------
    values : dict[int, np.ndarray]
        Per-draw AUROC per budget.
    target : float
        AUROC to reach.
    threshold : float
        Minimum fraction of draws at or above ``target``.

    Returns
    -------
    int | None
        The smallest qualifying budget, or None.
    """
    reached = [size for size, draws in values.items() if np.mean(np.asarray(draws) >= target) >= threshold]
    return min(reached) if reached else None


def wide_metric(table: pd.DataFrame, budget: str, metric: str) -> pd.DataFrame:
    """
    One metric of one budget with draws as rows and encoders as columns.

    Parameters
    ----------
    table : pd.DataFrame
        Per-draw rows of one readout, with ``budget``, ``draw``, ``encoder`` and the metric.
    budget : str
        Budget label.
    metric : str
        ``auroc`` or ``average_precision``.

    Returns
    -------
    pd.DataFrame
        The metric per draw and encoder.
    """
    frame = table[table["budget"] == budget]
    return frame.pivot(index="draw", columns="encoder", values=metric)


def budget_summary(table: pd.DataFrame, budget: str, encoders: tuple[str, ...],
                   contrasts: tuple[tuple[str, str], ...]) -> dict[str, dict]:
    """
    Per-encoder summaries, paired contrasts and 90% readings of one budget.

    Parameters
    ----------
    table : pd.DataFrame
        Per-draw rows of one readout.
    budget : str
        Budget label below all.
    encoders : tuple[str, ...]
        Encoder names.
    contrasts : tuple[tuple[str, str], ...]
        Pairs ``(first, second)`` for ``first - second``.

    Returns
    -------
    dict[str, dict]
        ``encoders`` (metric summaries), ``contrasts`` (paired summaries) and ``reading`` (AUROC verdicts).
    """
    wide = {metric: wide_metric(table, budget, metric) for metric in ("auroc", "average_precision")}
    by_encoder = {name: {metric: summarize(values[name].to_numpy()) for metric, values in wide.items()}
                  for name in encoders}
    paired = {f"{first}_minus_{second}": {
        metric: paired_summary(values[first].to_numpy(), values[second].to_numpy())
        for metric, values in wide.items()} for first, second in contrasts}
    verdicts = {f"{first}_minus_{second}": reading(first, second, paired[f"{first}_minus_{second}"]["auroc"])
                for first, second in contrasts}
    return {"encoders": by_encoder, "contrasts": paired, "reading": verdicts}
