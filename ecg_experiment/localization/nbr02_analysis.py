"""NBR-02 arm tables, selectivity, paired arm contrasts, template diagnostics and decision."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..intervals import patient_groups, patient_resample
from . import nbr01_analysis as na
from .nbr01_features import KINDS, MODELS
from .nbr02_templates import ARMS, NEW_ARMS

COUNTERPART = slice(1, 53)
SCORES = tuple(f"{model}_{kind}" for model in MODELS for kind in KINDS)
CONTRASTS = (("C6", "P0"), ("C0", "P0"), ("C6", "P6"), ("P6", "P0"), ("C6", "C0"))
CONTRAST_NAMES = ("selectivity", "territory", "territory_minus_energy", "swap_territory", "wave",
                  "wave_minus_energy", "qrs_bbb", "stt_stt", "bbb_lead", "top1_hit", "top3_hit",
                  "abnormal/lead_abs", "normal/lead_abs", "abnormal/grid_abs", "normal/grid_abs",
                  "abnormal/full_drop", "normal/full_drop", "abnormal/full_score", "abnormal/grid_flip",
                  "abnormal/lead_flip", "abnormal/component_flip", "abnormal/full_flip", "normal/lead_flip")
ENERGY_NAMES = ("territory", "wave", "qrs_bbb", "stt_stt", "bbb_lead", "top1_hit", "top3_hit")
TEMPLATE_PAIRS = ("P6_P0", "C6_C0", "C0_P0", "C6_P0")
GROUPS = ("inferior", "anterior", "lbbb", "rbbb", "stt", "normal")
MARGIN = -0.05


def arm_arrays(base_scores: np.ndarray, base_energy: np.ndarray, arm_scores: np.ndarray,
               arm_energy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    NBR-01-layout scores and energies with the 52 normal-counterpart variants taken from one arm.

    Parameters
    ----------
    base_scores, base_energy : np.ndarray
        NBR-01 (records, 105, models, kinds) scores and (records, 105) energies.
    arm_scores, arm_energy : np.ndarray
        The arm's (records, 52, models, kinds) scores and (records, 52) energies.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Copies with the original and self-swap variants of NBR-01 and the arm's counterpart variants.
    """
    scores, energy = base_scores.copy(), base_energy.copy()
    scores[:, COUNTERPART], energy[:, COUNTERPART] = arm_scores, arm_energy
    return scores, energy


def arm_tables(base: dict[str, np.ndarray], new: dict[str, np.ndarray],
               thresholds: dict[str, float]) -> dict[str, dict[str, Any]]:
    """
    One ``nbr01_analysis.record_table`` per arm; P0 is NBR-01's saved arrays unchanged.

    Parameters
    ----------
    base : dict[str, np.ndarray]
        NBR-01 ``scores.npz``.
    new : dict[str, np.ndarray]
        NBR-02 ``scores`` (records, 3, 52, models, kinds) and ``energy`` (records, 3, 52).
    thresholds : dict[str, float]
        Frozen thresholds.

    Returns
    -------
    dict[str, dict[str, Any]]
        Table of each arm in ``ARMS`` order.
    """
    tables = {"P0": na.record_table(base["scores"], base["energy"], thresholds)}
    for position, arm in enumerate(NEW_ARMS):
        scores, energy = arm_arrays(base["scores"], base["energy"], new["scores"][:, position],
                                    new["energy"][:, position])
        tables[arm] = na.record_table(scores, energy, thresholds)
    return {arm: tables[arm] for arm in ARMS}


def selectivity(statistics: dict[str, float], prefix: str, score: str) -> float:
    """Mean absolute whole-lead score change on abnormal ECGs divided by the same on normals."""
    return statistics[f"{prefix}{score}/abnormal/lead_abs"] / statistics[f"{prefix}{score}/normal/lead_abs"]


def contrasts(result: dict[str, float]) -> dict[str, float]:
    """Paired arm differences of the selected statistics, from one set of arm statistics."""
    found = {}
    for first, second in CONTRASTS:
        keys = [f"{score}/{name}" for score in SCORES for name in CONTRAST_NAMES]
        keys += [f"energy/{name}" for name in ENERGY_NAMES]
        for key in keys:
            if f"{first}/{key}" in result:
                found[f"{first}-{second}/{key}"] = result[f"{first}/{key}"] - result[f"{second}/{key}"]
    return found


def template_diagnostics(group: np.ndarray, diagnostics: dict[str, np.ndarray],
                         rows: np.ndarray) -> dict[str, float]:
    """Mean template distances and clip rates of each group on one set of record positions."""
    found = {}
    for name in GROUPS:
        mask = group == name
        if not mask.any():
            continue
        for pair in TEMPLATE_PAIRS:
            found[f"template/{pair}/{name}"] = float(diagnostics[f"template_{pair}"][rows][mask].mean())
        for arm in ("P6", "C6"):
            clipped = diagnostics[f"{arm}_clipped"][rows][mask]
            found[f"clip/{arm}/{name}/any"] = float((clipped > 0).mean())
            found[f"clip/{arm}/{name}/mean"] = float(clipped.mean())
    return found


def arm_statistics(tables: dict[str, dict[str, Any]], groups: np.ndarray, diagnostics: dict[str, np.ndarray],
                   rows: np.ndarray) -> dict[str, float]:
    """
    Compute every NBR-02 statistic on one set of record positions.

    Parameters
    ----------
    tables : dict[str, dict[str, Any]]
        Output of ``arm_tables``.
    groups : np.ndarray
        Group of each record.
    diagnostics : dict[str, np.ndarray]
        Per-record template distances (``template_<A>_<B>``) and clip counts (``<arm>_clipped``).
    rows : np.ndarray
        Record positions.

    Returns
    -------
    dict[str, float]
        NBR-01 statistics prefixed by arm, selectivities, paired arm contrasts and template diagnostics.
    """
    result: dict[str, float] = {}
    for arm, table in tables.items():
        for key, value in na.statistics(table, groups, rows).items():
            result[f"{arm}/{key}"] = value
        for score in SCORES:
            result[f"{arm}/{score}/selectivity"] = selectivity(result, f"{arm}/", score)
    result.update(contrasts(result))
    result.update(template_diagnostics(groups[rows], diagnostics, rows))
    return result


def bootstrap(tables: dict[str, dict[str, Any]], groups: np.ndarray, patients: np.ndarray,
              diagnostics: dict[str, np.ndarray], draws: int = na.DRAWS,
              seed: int = na.SEED) -> dict[str, dict[str, float]]:
    """
    Point values and 95% whole-patient bootstrap intervals, all statistics on the same draws.

    Parameters
    ----------
    tables : dict[str, dict[str, Any]]
        Output of ``arm_tables``.
    groups, patients : np.ndarray
        Group and patient of each record.
    diagnostics : dict[str, np.ndarray]
        Per-record template diagnostics.
    draws : int
        Bootstrap draws.
    seed : int
        Generator seed.

    Returns
    -------
    dict[str, dict[str, float]]
        Per statistic: ``point``, ``low``, ``high`` and ``valid_draws``.
    """
    point = arm_statistics(tables, groups, diagnostics, np.arange(len(groups)))
    rng = np.random.default_rng(seed)
    resample = patient_groups(np.asarray(patients))
    values: dict[str, list[float]] = {key: [] for key in point}
    for _ in range(draws):
        drawn = arm_statistics(tables, groups, diagnostics, patient_resample(resample, rng))
        for key in point:
            value = drawn.get(key, float("nan"))
            if np.isfinite(value):
                values[key].append(value)
    result = {}
    for key, value in point.items():
        low, high = np.percentile(values[key], [2.5, 97.5]) if values[key] else (np.nan, np.nan)
        result[key] = {"point": value, "low": float(low), "high": float(high),
                       "valid_draws": len(values[key])}
    return result


def decision(statistics: dict[str, dict[str, float]]) -> dict[str, Any]:
    """
    Apply the frozen NBR-02 decision rule.

    Parameters
    ----------
    statistics : dict[str, dict[str, float]]
        Output of ``bootstrap``.

    Returns
    -------
    dict[str, Any]
        The three parts, their statistics and whether C6 replaces P0.
    """
    parts = {
        "A_selectivity": ("C6-P0/all_age_blend/selectivity", 0.0),
        "B_territory_noninferior": ("C6-P0/all_age_blend/territory", MARGIN),
        "C_operator_check": ("C6/all_age_blend/abnormal/full_drop", 0.0),
    }
    result: dict[str, Any] = {}
    for name, (key, bound) in parts.items():
        value = statistics[key]
        result[name] = {"statistic": key, "point": value["point"], "low": value["low"],
                        "high": value["high"], "bound": bound, "passed": bool(value["low"] > bound)}
    result["C6_replaces_P0"] = all(part["passed"] for part in result.values())
    return result
