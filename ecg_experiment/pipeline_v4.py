"""The ensemble readout, retrain agreement, saved network weights and screen summaries of Experiment 046.

Pipeline v4 is pipeline v3 with its binary readout replaced by the unfitted mean of two logits (Experiment
045's ensemble E). The screen summaries generalize Experiment 044's to any list of paired pipeline contrasts,
and the matched-rate screen is Experiment 044's post hoc oracle check.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .finding_screen import split_thresholds
from .pipeline_v2 import percentile_interval
from .referral_budget import referrals_per_1000


def ensemble_logit(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """
    Return the unfitted mean of two readouts' logits.

    Parameters
    ----------
    first, second : np.ndarray
        Logits of the same rows.

    Returns
    -------
    np.ndarray
        ``(first + second) / 2`` in float64.

    Raises
    ------
    ValueError
        If the shapes differ or a logit is not finite.
    """
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or not (np.isfinite(first).all() and np.isfinite(second).all()):
        raise ValueError("Logits must be finite and of the same shape")
    return (first + second) / 2


def candidate_reading(interval: list[float], margin: float, candidate: str, current: str) -> str:
    """
    Read a candidate minus current difference by Experiment 044's rule.

    Parameters
    ----------
    interval : list[float]
        Lower and upper bounds of the difference.
    margin : float
        Positive no-worse margin.
    candidate, current : str
        Pipeline names.

    Returns
    -------
    str
        ``adopt_{candidate}`` when the lower bound is above 0, ``no_worse_keep_{current}`` when it is above
        ``-margin``, otherwise ``keep_{current}``.
    """
    if interval[0] > 0:
        return f"adopt_{candidate}"
    if interval[0] > -margin:
        return f"no_worse_keep_{current}"
    return f"keep_{current}"


def retrain_agreement(new: np.ndarray, old: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """
    Compare a retrained network's logits with the saved ones of the same rows.

    Parameters
    ----------
    new, old : np.ndarray
        Retrained and saved logits.
    y : np.ndarray
        Binary labels of the rows.

    Returns
    -------
    dict[str, Any]
        Pearson ``r``, both AUROCs, their difference (new minus old), the largest absolute logit difference
        and whether the logits are bit-identical.

    Raises
    ------
    ValueError
        If the shapes differ.
    """
    new, old = np.asarray(new, dtype=np.float64), np.asarray(old, dtype=np.float64)
    if new.shape != old.shape or new.shape != np.shape(y):
        raise ValueError("Logits and labels must have the same shape")
    auroc_new, auroc_old = float(roc_auc_score(y, new)), float(roc_auc_score(y, old))
    return {"rows": len(new), "r": float(np.corrcoef(new, old)[0, 1]), "auroc_new": auroc_new,
            "auroc_old": auroc_old, "auroc_difference": auroc_new - auroc_old,
            "max_abs_difference": float(np.abs(new - old).max()), "identical": bool(np.array_equal(new, old))}


def state_arrays(states: Mapping[int, Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """
    Flatten per-seed network states into named arrays for one ``.npz`` file.

    Parameters
    ----------
    states : Mapping[int, Mapping[str, np.ndarray]]
        Per seed, the ``state_dict`` of a network as NumPy arrays.

    Returns
    -------
    dict[str, np.ndarray]
        ``seed{seed}.{key}`` arrays.
    """
    return {f"seed{seed}.{key}": np.asarray(value) for seed, state in states.items()
            for key, value in state.items()}


def load_state(model: torch.nn.Module, arrays: Mapping[str, np.ndarray], seed: int) -> torch.nn.Module:
    """
    Load one seed's weights written by ``state_arrays`` into a network, strictly.

    Parameters
    ----------
    model : torch.nn.Module
        Network of the same architecture.
    arrays : Mapping[str, np.ndarray]
        Arrays written by ``state_arrays``.
    seed : int
        Seed whose weights to load.

    Returns
    -------
    torch.nn.Module
        The same network with the weights loaded.
    """
    prefix = f"seed{seed}."
    state = {key[len(prefix):]: torch.from_numpy(np.array(value)) for key, value in arrays.items()
             if key.startswith(prefix)}
    model.load_state_dict(state, strict=True)
    return model


def matched_rate_screen(matrix: np.ndarray, counts: np.ndarray, normal: np.ndarray,
                        masks: Mapping[str, np.ndarray], per_mille: int, shares: tuple[int, ...]
                        ) -> dict[str, float]:
    """
    Set thresholds on the (resampled) evaluation normals and return each outcome's referral rate.

    This is Experiment 044's post hoc oracle: the normals that set the thresholds are the evaluation normals
    themselves, so every pipeline refers the same share of them.

    Parameters
    ----------
    matrix : np.ndarray
        ``(n, 1 + F)`` scores of the evaluation ECGs.
    counts : np.ndarray
        How often each ECG enters (1 for the observed data).
    normal : np.ndarray
        Mask of the evaluation normals.
    masks : Mapping[str, np.ndarray]
        Outcome masks.
    per_mille : int
        Budget in thousandths.
    shares : tuple[int, ...]
        Finding shares of the rule.

    Returns
    -------
    dict[str, float]
        Referral rate per outcome, weighted by ``counts``.
    """
    normals = np.repeat(matrix[normal], counts[normal].astype(np.int64), axis=0)
    thresholds = split_thresholds(normals, per_mille, shares)
    referred = (matrix > thresholds).any(axis=1)
    return {name: float((counts * referred * mask).sum() / (counts * mask).sum())
            for name, mask in masks.items()}


def summarize_screens(draws: pd.DataFrame, boot: Mapping[str, np.ndarray],
                      keys: list[tuple[str, str, int, int]], contrasts: tuple[tuple[str, str, str], ...],
                      prevalence: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Summarize every screen and its paired pipeline and rule-minus-``binary`` contrasts, as Experiment 044.

    Parameters
    ----------
    draws : pd.DataFrame
        One row per pipeline, rule, m, budget and draw, with the outcome columns of ``boot``.
    boot : Mapping[str, np.ndarray]
        ``(resamples, keys)`` referral rates per outcome.
    keys : list[tuple[str, str, int, int]]
        ``(pipeline, rule, m, budget per mille)`` in the column order of ``boot``.
    contrasts : tuple[tuple[str, str, str], ...]
        ``(label, first, second)`` pipeline pairs, each read as first minus second.
    prevalence : float
        Assumed share of abnormal ECGs for the per-1,000 numbers.

    Returns
    -------
    tuple[list[dict[str, Any]], list[dict[str, Any]]]
        Summary rows and contrast rows.
    """
    groups = draws.groupby(["pipeline", "rule", "m", "budget"], sort=False)
    column = {key: index for index, key in enumerate(keys)}
    outcomes = list(boot)
    summary, rows = [], []
    for key in keys:
        pipeline, rule, m, budget = key
        group = groups.get_group((pipeline, rule, m, budget / 1000))
        row: dict[str, Any] = {"pipeline": pipeline, "rule": rule, "m": m, "budget": budget / 1000,
                               "threshold_0_mean": float(group["threshold_0"].mean()),
                               "share_rate_within_1pp": float(((group["normal"] - budget / 1000).abs()
                                                               <= 0.01 + 1e-12).mean())}
        for name in outcomes:
            values = group[name]
            row[name] = {"mean": float(values.mean()), "p5": float(values.quantile(0.05)),
                         "p95": float(values.quantile(0.95)),
                         "ci": percentile_interval(boot[name][:, column[key]])}
        for label in ("composite", "binary_positive"):
            per_draw = referrals_per_1000(group[label], group["normal"], prevalence)
            resampled = referrals_per_1000(boot[label][:, column[key]], boot["normal"][:, column[key]],
                                           prevalence)
            row[f"per_1000_{label}"] = {
                "referrals": float(per_draw.mean()), "referrals_ci": percentile_interval(resampled),
                "caught": float(1000 * prevalence * group[label].mean()),
                "caught_ci": percentile_interval(1000 * prevalence * boot[label][:, column[key]])}
        summary.append(row)
        references = [(label, (second, rule, m, budget)) for label, first, second in contrasts
                      if first == pipeline and (second, rule, m, budget) in column]
        if rule != "binary":
            references.append((f"{pipeline}_{rule}_minus_binary", (pipeline, "binary", m, budget)))
        for label, reference in references:
            base = groups.get_group((reference[0], reference[1], m, budget / 1000))
            for name in outcomes:
                difference = boot[name][:, column[key]] - boot[name][:, column[reference]]
                rows.append({"contrast": label, "rule": rule, "m": m, "budget": budget / 1000,
                             "outcome": name, "difference": float(group[name].mean() - base[name].mean()),
                             "ci": percentile_interval(difference)})
    return summary, rows
