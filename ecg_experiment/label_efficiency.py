"""Label draws, cross-validated readout and summaries for the Experiment 025 label-efficiency study."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

C_GRID = (0.001, 0.01, 0.1, 1.0)
FOLDS = 5


def draw_subset(patients: np.ndarray, y: np.ndarray, size: int, seed: int) -> np.ndarray:
    """
    Draw ``size`` ECGs from distinct patients at the pool's label prevalence.

    The pool is permuted, the first ECG of each patient in that order is kept, and the first
    ``round(size * prevalence)`` positives and the remaining negatives are taken in the same order.

    Parameters
    ----------
    patients : np.ndarray
        Patient ID of each pool ECG.
    y : np.ndarray
        Binary label of each pool ECG.
    size : int
        Number of ECGs to draw.
    seed : int
        Seed of ``numpy.random.default_rng``.

    Returns
    -------
    np.ndarray
        Sorted positions of the drawn ECGs in the pool.

    Raises
    ------
    ValueError
        If the pool has too few patients of either class.
    """
    order = np.random.default_rng(seed).permutation(len(y))
    _, first = np.unique(patients[order], return_index=True)
    candidates = order[np.sort(first)]
    positives = round(size * float(np.mean(y)))
    chosen_positive = candidates[y[candidates] == 1][:positives]
    chosen_negative = candidates[y[candidates] == 0][:size - positives]
    if len(chosen_positive) + len(chosen_negative) != size:
        raise ValueError(f"The pool cannot supply {size} ECGs from distinct patients")
    return np.sort(np.concatenate([chosen_positive, chosen_negative]))


def fit_logistic_c(x: np.ndarray, y: np.ndarray, c: float) -> tuple[StandardScaler, LogisticRegression]:
    """
    Fit Experiment 020's scaler and L2 logistic head with a chosen C, in float64.

    Parameters
    ----------
    x : np.ndarray
        Training inputs.
    y : np.ndarray
        Binary targets.
    c : float
        Inverse L2 regularization strength.

    Returns
    -------
    tuple[StandardScaler, LogisticRegression]
        Fitted scaler and classifier.

    Raises
    ------
    RuntimeError
        If the solver does not converge.
    """
    x = np.asarray(x, dtype=np.float64)
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(C=c, penalty="l2", fit_intercept=True, solver="lbfgs", max_iter=5000,
                               tol=1e-8, class_weight=None, random_state=42)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(scaler.transform(x), y)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught) or model.n_iter_[0] >= 5000:
        raise RuntimeError(f"Logistic fit did not converge at C={c}")
    return scaler, model


def select_c(x: np.ndarray, y: np.ndarray, patients: np.ndarray, seed: int) -> tuple[float, list[float]]:
    """
    Choose C by patient-grouped, stratified 5-fold cross-validation on training rows.

    Parameters
    ----------
    x : np.ndarray
        Training inputs.
    y : np.ndarray
        Binary targets.
    patients : np.ndarray
        Patient ID of each row, used as the fold group.
    seed : int
        Fold shuffling seed.

    Returns
    -------
    tuple[float, list[float]]
        The C with the highest mean fold AUROC (ties go to the smaller C) and the mean AUROC of every
        C in ``C_GRID``.
    """
    folds = list(StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=seed).split(x, y, patients))
    means = []
    for c in C_GRID:
        aurocs = []
        for inner, held in folds:
            scaler, model = fit_logistic_c(x[inner], y[inner], c)
            scores = model.decision_function(scaler.transform(np.asarray(x[held], dtype=np.float64)))
            aurocs.append(roc_auc_score(y[held], scores))
        means.append(float(np.mean(aurocs)))
    return C_GRID[int(np.argmax(means))], means


def summarize(values: np.ndarray) -> dict[str, float]:
    """
    Mean, sample SD and 2.5/97.5 percentiles of per-draw values.

    Parameters
    ----------
    values : np.ndarray
        One value per draw.

    Returns
    -------
    dict[str, float]
        ``mean``, ``sd``, ``p2_5`` and ``p97_5``.
    """
    low, high = np.percentile(values, [2.5, 97.5])
    return {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)),
            "p2_5": float(low), "p97_5": float(high)}


def paired_summary(first: np.ndarray, second: np.ndarray) -> dict[str, float | int]:
    """
    Summarize paired per-draw differences, first minus second.

    Parameters
    ----------
    first, second : np.ndarray
        Metric of two encoders on the same draws.

    Returns
    -------
    dict[str, float | int]
        Mean difference, fraction of draws above 0, and the counts of positive and negative draws.
    """
    difference = np.asarray(first) - np.asarray(second)
    return {"mean": float(np.mean(difference)), "fraction_above_zero": float(np.mean(difference > 0)),
            "positive_draws": int(np.sum(difference > 0)), "negative_draws": int(np.sum(difference < 0)),
            "draws": len(difference)}


def reading(first: str, second: str, summary: dict[str, float | int], threshold: float = 0.9) -> str:
    """
    Apply the prespecified 90% rule to a paired contrast.

    Parameters
    ----------
    first, second : str
        Encoder names of the contrast ``first - second``.
    summary : dict[str, float | int]
        Output of ``paired_summary``.
    threshold : float
        Minimum fraction of draws favoring one encoder.

    Returns
    -------
    str
        ``"<first> better"``, ``"<second> better"`` or ``"no clear difference"``.
    """
    verdict = "no clear difference"
    if summary["positive_draws"] >= threshold * summary["draws"]:
        verdict = f"{first} better"
    elif summary["negative_draws"] >= threshold * summary["draws"]:
        verdict = f"{second} better"
    return verdict


def smallest_budget(means: dict[int, float], target: float) -> int | None:
    """
    Smallest budget whose mean AUROC reaches a target.

    Parameters
    ----------
    means : dict[int, float]
        Mean AUROC per budget.
    target : float
        AUROC to reach.

    Returns
    -------
    int | None
        The smallest budget with a mean of at least ``target``, or None.
    """
    reached = [size for size, value in means.items() if value >= target]
    return min(reached) if reached else None

