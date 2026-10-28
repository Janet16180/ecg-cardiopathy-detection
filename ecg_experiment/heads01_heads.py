"""Fixed HEADS-01 readouts: superclass mixture heads, token-pooled inputs and paired SPH contrasts."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import expit, logsumexp
from sklearn.ensemble import HistGradientBoostingClassifier

from .cpu_screening041 import HGB_CONFIG
from .full_development import fit_logistic
from .intervals import patient_groups, two_class_resamples

SUPERCLASSES = ("MI", "STTC", "CD", "HYP")
DRAWS = 2000
SEED = 20261028


def fit_hgb(x: Any, y: np.ndarray) -> HistGradientBoostingClassifier:
    """
    Fit the frozen predecessor boosting recipe.

    Parameters
    ----------
    x : Any
        Training measurements (DataFrame or array, missing values kept).
    y : np.ndarray
        Binary targets.

    Returns
    -------
    HistGradientBoostingClassifier
        Fitted model.
    """
    return HistGradientBoostingClassifier(**HGB_CONFIG).fit(x, y)


def decision(model: Any, x: Any) -> np.ndarray:
    """
    Log-odds of a fitted HGB model or ``fit_logistic`` head.

    Parameters
    ----------
    model : Any
        ``HistGradientBoostingClassifier`` or a ``(scaler, logistic)`` tuple.
    x : Any
        Inputs.

    Returns
    -------
    np.ndarray
        One float64 log-odds per row.
    """
    if isinstance(model, tuple):
        scaler, logistic = model
        return logistic.decision_function(scaler.transform(np.asarray(x, dtype=np.float64)))
    return np.asarray(model.decision_function(x), dtype=np.float64)


def fit_superclass(x: Any, y: np.ndarray, flags: np.ndarray, kind: str) -> dict[str, Any]:
    """
    Fit one superclass-versus-negative head per superclass with a frozen recipe.

    Parameters
    ----------
    x : Any
        Training inputs aligned with ``y``.
    y : np.ndarray
        Binary cohort targets.
    flags : np.ndarray
        ``[records, 4]`` 0/1 superclass membership in ``SUPERCLASSES`` order.
    kind : str
        ``"hgb"`` or ``"logistic"``.

    Returns
    -------
    dict[str, Any]
        ``heads`` (one model per superclass), ``counts`` per superclass and the cohort counts
        ``negatives`` and ``positives``.

    Raises
    ------
    ValueError
        If flags disagree with the binary target or a superclass has no positive.
    """
    y, flags = np.asarray(y, dtype=int), np.asarray(flags, dtype=int)
    if flags.shape != (len(y), len(SUPERCLASSES)) or not np.array_equal(flags.max(axis=1), y):
        raise ValueError("Superclass flags must reproduce the binary target exactly")
    rows = np.asarray(x) if kind == "logistic" else x
    negatives = y == 0
    heads, counts = {}, {}
    for column, name in enumerate(SUPERCLASSES):
        keep = negatives | (flags[:, column] == 1)
        target = flags[keep, column]
        if not target.any():
            raise ValueError(f"No positive for superclass {name}")
        subset = rows[keep] if kind == "logistic" else rows.iloc[np.flatnonzero(keep)]
        heads[name] = fit_logistic(subset, target) if kind == "logistic" else fit_hgb(subset, target)
        counts[name] = int(target.sum())
    return {"heads": heads, "counts": counts, "negatives": int(negatives.sum()),
            "positives": int((~negatives).sum())}


def mixture_score(fitted: dict[str, Any], x: Any) -> np.ndarray:
    """
    Mixture likelihood-ratio score on the binary heads' probability scale.

    Each head's log-likelihood ratio is its log-odds minus its training log prior odds. The
    mixture weights are the superclass training counts; the result is mapped back with the
    cohort's binary prior odds.

    Parameters
    ----------
    fitted : dict[str, Any]
        Output of ``fit_superclass``.
    x : Any
        Inputs.

    Returns
    -------
    np.ndarray
        Scores in (0, 1).
    """
    counts = np.array([fitted["counts"][name] for name in SUPERCLASSES], dtype=np.float64)
    negatives = float(fitted["negatives"])
    ratios = np.column_stack([
        decision(fitted["heads"][name], x) - np.log(count / negatives)
        for name, count in zip(SUPERCLASSES, counts, strict=True)
    ])
    mixture = logsumexp(ratios + np.log(counts / counts.sum()), axis=1)
    return expit(mixture + np.log(fitted["positives"] / negatives))


def paired_sensitivity(patients: np.ndarray, y: np.ndarray, first: np.ndarray, second: np.ndarray,
                       levels: tuple[float, ...] = (0.95, 0.975), draws: int = DRAWS,
                       seed: int = SEED) -> dict[str, Any]:
    """
    Paired whole-patient bootstrap of a sensitivity difference of frozen referral decisions.

    Parameters
    ----------
    patients : np.ndarray
        Patient ID per evaluated ECG.
    y : np.ndarray
        Binary targets.
    first, second : np.ndarray
        Boolean referral decisions of the two models.
    levels : tuple[float, ...]
        Two-sided percentile interval levels computed from the same draws.
    draws : int
        Number of bootstrap draws, including skipped single-class draws.
    seed : int
        Seed of ``numpy.random.default_rng``.

    Returns
    -------
    dict[str, Any]
        Observed ``difference``, ``intervals`` keyed by level, ``draws``, ``skipped_draws`` and ``seed``.

    Raises
    ------
    ValueError
        If every draw is skipped.
    """
    y = np.asarray(y, dtype=int)
    first, second = np.asarray(first, dtype=bool), np.asarray(second, dtype=bool)
    rng = np.random.default_rng(seed)
    values = [float(first[rows][y[rows] == 1].mean() - second[rows][y[rows] == 1].mean())
              for rows in two_class_resamples(patient_groups(np.asarray(patients)), y, draws, rng)]
    if not values:
        raise ValueError("Every bootstrap draw had a single class")
    intervals = {}
    for level in levels:
        tail = 100 * (1 - level) / 2
        low, high = np.percentile(values, [tail, 100 - tail])
        intervals[str(level)] = [float(low), float(high)]
    return {"difference": float(first[y == 1].mean() - second[y == 1].mean()), "intervals": intervals,
            "draws": draws, "skipped_draws": draws - len(values), "seed": seed}


def gates(sensitivity: dict[str, Any], specificity_change: float) -> dict[str, Any]:
    """
    Apply the three prespecified HEADS-01 gates without rounding.

    Parameters
    ----------
    sensitivity : dict[str, Any]
        Output of ``paired_sensitivity`` with a 0.975 interval.
    specificity_change : float
        Candidate minus reference SPH evaluation specificity.

    Returns
    -------
    dict[str, Any]
        Each gate, ``established`` (all three) and ``statistically_supported`` (gates 2 and 3).
    """
    result = {"gain_at_least_0.02": sensitivity["difference"] >= 0.02,
              "lower_97.5_above_zero": sensitivity["intervals"]["0.975"][0] > 0,
              "specificity_drop_at_most_0.005": specificity_change >= -0.005}
    result["established"] = all(result.values())
    result["statistically_supported"] = (result["lower_97.5_above_zero"]
                                         and result["specificity_drop_at_most_0.005"])
    return result
