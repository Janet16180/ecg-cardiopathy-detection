"""Explanation rule for Experiment 047: refer by the readout, then explain with ``U_B`` or attention."""

from __future__ import annotations

from typing import Any

import numpy as np

from .fragment_localization import bootstrap_mean
from .lead_wave_maps import (
    ANTERIOR,
    INFERIOR,
    UnitMap,
    premature_hit,
    red_marks,
    top_lead,
    two_group_difference,
)
from .two_layer_map import Explanation

REFERRAL_QUANTILE = 0.95
LOCALIZATION_MARGIN = -0.10
REGIONS = {"anterior": (ANTERIOR, "anterior", "inferior"), "inferior": (INFERIOR, "inferior", "anterior")}


def referral_threshold(normal_logits: np.ndarray, quantile: float = REFERRAL_QUANTILE) -> float:
    """
    Return the referral threshold, a quantile of the readout logits of normal ECGs.

    Parameters
    ----------
    normal_logits : np.ndarray
        Readout logit of each normal ECG.
    quantile : float
        Quantile, with NumPy's default linear interpolation.

    Returns
    -------
    float
        The threshold; an ECG is referred when its logit is strictly above it.
    """
    return float(np.quantile(np.asarray(normal_logits, dtype=np.float64), quantile))


def referred_ids(ids: list[int], logits: np.ndarray, threshold: float) -> list[int]:
    """Return the IDs whose logit is strictly above the threshold, in the given order."""
    return [i for i, value in zip(ids, np.asarray(logits), strict=True) if value > threshold]


def group_rates(referred: set[int], groups: dict[str, list[int]]) -> dict[str, float]:
    """Return the share of each group's IDs that are referred."""
    return {name: float(np.mean([i in referred for i in members])) for name, members in groups.items()}


def layer_shares(layers: dict[int, int], groups: dict[str, list[int]]) -> dict[str, dict[str, float | int]]:
    """
    Return, per group, the number of explained ECGs and the share explained by each layer.

    Parameters
    ----------
    layers : dict[int, int]
        Supplying layer (1 or 2) of each explained ECG.
    groups : dict[str, list[int]]
        Group members; members without an explanation are ignored.

    Returns
    -------
    dict[str, dict[str, float | int]]
        ``n``, ``layer1`` and ``layer2`` per group; the shares are NaN for an empty group.
    """
    found: dict[str, dict[str, float | int]] = {}
    for name, members in groups.items():
        values = np.array([layers[i] for i in members if i in layers])
        found[name] = {"n": int(len(values)),
                       "layer1": float(np.mean(values == 1)) if len(values) else float("nan"),
                       "layer2": float(np.mean(values == 2)) if len(values) else float("nan")}
    return found


def explanation_metrics(units: dict[int, UnitMap], patients: dict[int, Any],
                        windows: dict[int, list[tuple[float, float]]], anterior: list[int],
                        inferior: list[int], draws: int, seed: int) -> dict[str, Any]:
    """
    Score one explanation's top units: premature-beat hit - chance and the two lead contrasts.

    Parameters
    ----------
    units : dict[int, UnitMap]
        The units that supply each explained ECG's top unit; only these ECGs are scored.
    patients : dict[int, Any]
        Patient ID by ECG ID.
    windows : dict[int, list[tuple[float, float]]]
        Premature-beat windows by ECG ID.
    anterior, inferior : list[int]
        Anterior-only and inferior-only infarct ECG IDs.
    draws : int
        Bootstrap draws.
    seed : int
        Bootstrap seed.

    Returns
    -------
    dict[str, Any]
        ``metrics`` (``pvc_ecgs``, ``hit_rate``, ``chance_rate``, ``hit_minus_chance`` and ``lead_contrast``)
        and ``per_ecg`` (``excess`` by PVC ECG and the lead indicator by region and group).
    """
    included = [i for i in windows if i in units]
    hit, chance = np.array([premature_hit(units[i], windows[i]) for i in included]).reshape(-1, 2).T
    metrics: dict[str, Any] = {"pvc_ecgs": len(included), "hit_rate": float(hit.mean()),
                               "chance_rate": float(chance.mean()),
                               "hit_minus_chance": bootstrap_mean(np.array([patients[i] for i in included]),
                                                                  hit - chance, draws, seed)}
    members = {"anterior": [i for i in anterior if i in units],
               "inferior": [i for i in inferior if i in units]}
    indicators: dict[str, dict[str, dict[int, float]]] = {}
    contrasts = {}
    for region, (leads, first, second) in REGIONS.items():
        indicators[region] = {group: {i: float(top_lead(units[i]) in leads) for i in members[group]}
                              for group in (first, second)}
        a, b = indicators[region][first], indicators[region][second]
        contrasts[region] = two_group_difference(
            np.array([patients[i] for i in a]), np.array(list(a.values())),
            np.array([patients[i] for i in b]), np.array(list(b.values())), draws, seed)
    metrics["lead_contrast"] = contrasts
    metrics["infarct_ecgs"] = {group: len(ids) for group, ids in members.items()}
    return {"metrics": metrics,
            "per_ecg": {"excess": dict(zip(included, hit - chance, strict=True)), "lead": indicators}}


def paired_difference(mine: dict[str, Any], theirs: dict[str, Any], patients: dict[int, Any], draws: int,
                      seed: int) -> dict[str, Any]:
    """
    Return the paired differences of two explanations on the same ECGs: hit - chance and the lead contrasts.

    Parameters
    ----------
    mine, theirs : dict[str, Any]
        ``explanation_metrics`` outputs on the same explained ECGs.
    patients : dict[int, Any]
        Patient ID by ECG ID.
    draws : int
        Bootstrap draws.
    seed : int
        Bootstrap seed.

    Returns
    -------
    dict[str, Any]
        ``hit_minus_chance`` (per-ECG bootstrap of the difference) and ``lead_contrast`` per region (the
        per-ECG indicator difference through ``two_group_difference``).

    Raises
    ------
    ValueError
        If the two explanations cover different ECGs.
    """
    a, b = mine["per_ecg"], theirs["per_ecg"]
    if list(a["excess"]) != list(b["excess"]) or any(
            list(a["lead"][r][g]) != list(b["lead"][r][g]) for r in a["lead"] for g in a["lead"][r]):
        raise ValueError("The two explanations cover different ECGs")
    pvc = list(a["excess"])
    found: dict[str, Any] = {"hit_minus_chance": bootstrap_mean(
        np.array([patients[i] for i in pvc]), np.array([a["excess"][i] - b["excess"][i] for i in pvc]),
        draws, seed)}
    contrasts = {}
    for region, (_, first, second) in REGIONS.items():
        sides = []
        for group in (first, second):
            ids = list(a["lead"][region][group])
            sides += [np.array([patients[i] for i in ids]),
                      np.array([a["lead"][region][group][i] - b["lead"][region][group][i] for i in ids])]
        contrasts[region] = two_group_difference(*sides, draws, seed)
    found["lead_contrast"] = contrasts
    return found


def rule_reading(hit_difference_low: float, anterior_low: float,
                 margin: float = LOCALIZATION_MARGIN) -> dict[str, bool]:
    """
    Apply the prespecified reading: keep premature-beat localization and gain the anterior contrast.

    Parameters
    ----------
    hit_difference_low : float
        Lower bound of the rule's hit - chance minus the comparator's.
    anterior_low : float
        Lower bound of the rule's anterior contrast.
    margin : float
        Non-inferiority margin of the hit - chance difference.

    Returns
    -------
    dict[str, bool]
        ``keeps_premature_localization``, ``anterior_above_zero`` and ``improves``.
    """
    keeps, anterior = bool(hit_difference_low > margin), bool(anterior_low > 0)
    return {"keeps_premature_localization": keeps, "anterior_above_zero": anterior,
            "improves": keeps and anterior}


def rule_marks(found: Explanation, threshold: float) -> list[tuple[int, float, float]]:
    """
    Return the marks of an explanation: the supplying layer's units above its threshold, plus its top unit.

    Parameters
    ----------
    found : Explanation
        The ECG's explanation.
    threshold : float
        The supplying layer's own red threshold.

    Returns
    -------
    list[tuple[int, float, float]]
        Lead index, start and end of each mark, the top unit included once.
    """
    marks = red_marks(found.units, threshold)
    top = int(found.units.scores.argmax())
    unit = (int(found.units.leads[top]), float(found.units.starts[top]), float(found.units.ends[top]))
    return marks if unit in marks else [*marks, unit]
