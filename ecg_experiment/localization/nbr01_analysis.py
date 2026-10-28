"""NBR-01 attribution shares, label-derived contrasts, flip rates and whole-patient bootstrap."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..intervals import patient_groups, patient_resample
from . import nbr01_counterpart as nc
from .nbr01_features import KINDS, MODELS
from .nbr01_selection import BBB_LEADS, LEAD_NAMES, TERRITORY_LEADS

DRAWS = 2000
SEED = 20261028
ABNORMAL = ("inferior", "anterior", "lbbb", "rbbb", "stt")
TERRITORIES = ("inferior", "anterior", "lateral")
QRS, STT_COMPONENT = nc.COMPONENTS.index("QRS"), nc.COMPONENTS.index("ST-T")


def positions() -> dict[str, np.ndarray]:
    """
    Variant positions of each attribution family.

    Returns
    -------
    dict[str, np.ndarray]
        ``lead`` (12), ``component`` (3), ``grid`` (12, 3), ``full`` (scalar) and the same with a
        ``swap_`` prefix.
    """
    index = {item.name: position for position, item in enumerate(nc.variants())}
    result = {}
    for prefix in ("", "swap_"):
        result[prefix + "lead"] = np.array([index[f"{prefix}lead_{lead}"] for lead in range(nc.LEADS)])
        result[prefix + "component"] = np.array([index[f"{prefix}component_{part}"]
                                                 for part in range(len(nc.COMPONENTS))])
        result[prefix + "grid"] = np.array([[index[f"{prefix}grid_{lead}_{part}"]
                                             for part in range(len(nc.COMPONENTS))]
                                            for lead in range(nc.LEADS)])
        result[prefix + "full"] = np.array(index[f"{prefix}full"])
    return result


def shares(values: np.ndarray) -> np.ndarray:
    """
    Positive parts normalized to sum to one per row; uniform where no value is positive.

    Parameters
    ----------
    values : np.ndarray
        (records, regions) drops or energies.

    Returns
    -------
    np.ndarray
        (records, regions) shares.
    """
    positive = np.clip(values, 0.0, None)
    total = positive.sum(axis=1, keepdims=True)
    uniform = np.full_like(positive, 1.0 / positive.shape[1])
    return np.where(total > 0, positive / np.where(total > 0, total, 1.0), uniform)


def lead_mask(names: tuple[str, ...]) -> np.ndarray:
    """Boolean mask over the 12 leads."""
    return np.isin(np.array(LEAD_NAMES), names)


def group_contrast(values: np.ndarray, groups: np.ndarray, own: tuple[str, ...],
                   other: tuple[str, ...]) -> float:
    """Mean of ``values`` over ECGs of ``own`` groups minus the mean over ECGs of ``other`` groups."""
    first, second = np.isin(groups, own), np.isin(groups, other)
    if not first.any() or not second.any():
        return float("nan")
    return float(values[first].mean() - values[second].mean())


def territory_contrast(lead_shares: np.ndarray, groups: np.ndarray) -> float:
    """
    Average, over the territory groups present, of the expected-lead share contrast.

    Each territory group's share in its expected leads minus that share in the other territory groups.

    Parameters
    ----------
    lead_shares : np.ndarray
        (records, 12) lead shares.
    groups : np.ndarray
        Group of each record.

    Returns
    -------
    float
        Mean contrast, NaN if fewer than two territory groups are present.
    """
    present = tuple(name for name in TERRITORIES if np.isin(groups, name).any())
    if len(present) < 2:
        return float("nan")
    values = [group_contrast(lead_shares[:, lead_mask(TERRITORY_LEADS[name])].sum(axis=1), groups, (name,),
                             tuple(other for other in present if other != name)) for name in present]
    return float(np.mean(values))


def wave_contrasts(component_shares: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    """
    QRS share of BBB ECGs and ST-T share of ST-T ECGs, each against the other abnormal groups.

    Parameters
    ----------
    component_shares : np.ndarray
        (records, 3) component shares.
    groups : np.ndarray
        Group of each record.

    Returns
    -------
    dict[str, float]
        ``qrs_bbb``, ``stt_stt`` and their mean ``wave``.
    """
    qrs = group_contrast(component_shares[:, QRS], groups, ("lbbb", "rbbb"), ("inferior", "anterior", "stt"))
    stt = group_contrast(component_shares[:, STT_COMPONENT], groups, ("stt",),
                         ("inferior", "anterior", "lbbb", "rbbb"))
    return {"qrs_bbb": qrs, "stt_stt": stt, "wave": float(np.mean([qrs, stt]))}


def bbb_lead_contrast(lead_shares: np.ndarray, groups: np.ndarray) -> float:
    """RBBB V1-V2 share and LBBB I, aVL, V5-V6 share, each against the other BBB group, averaged."""
    right = group_contrast(lead_shares[:, lead_mask(BBB_LEADS["rbbb"])].sum(axis=1), groups, ("rbbb",),
                           ("lbbb",))
    left = group_contrast(lead_shares[:, lead_mask(BBB_LEADS["lbbb"])].sum(axis=1), groups, ("lbbb",),
                          ("rbbb",))
    return float(np.mean([right, left]))


def top_hits(values: np.ndarray, groups: np.ndarray, top: int) -> np.ndarray:
    """
    Whether one of a territory ECG's ``top`` largest positive lead values lies in its expected leads.

    Parameters
    ----------
    values : np.ndarray
        (records, 12) drops or energies.
    groups : np.ndarray
        Group of each record.
    top : int
        Number of leads.

    Returns
    -------
    np.ndarray
        Float 0/1 per record, NaN for records outside the territory groups. Ties keep lead order.
    """
    order = np.argsort(-values, axis=1, kind="stable")[:, :top]
    chosen = np.take_along_axis(values, order, axis=1)
    hits = np.full(len(values), np.nan)
    for name in TERRITORIES:
        rows = np.flatnonzero(groups == name)
        if len(rows):
            expected = np.flatnonzero(lead_mask(TERRITORY_LEADS[name]))
            hits[rows] = (np.isin(order[rows], expected) & (chosen[rows] > 0)).any(axis=1)
    return hits


def record_table(scores: np.ndarray, energy: np.ndarray, thresholds: dict[str, float]) -> dict[str, Any]:
    """
    Per-record quantities that every bootstrap statistic averages.

    Parameters
    ----------
    scores : np.ndarray
        (records, variants, models, kinds) scores after replacement.
    energy : np.ndarray
        (records, variants) replaced energy.
    thresholds : dict[str, float]
        Frozen thresholds keyed ``<model>_<kind>``.

    Returns
    -------
    dict[str, Any]
        Per model and kind: drops, shares, flags and flips; per map: lead and component shares.
    """
    where = positions()
    table: dict[str, Any] = {"energy_lead": shares(energy[:, where["lead"]]),
                             "energy_component": shares(energy[:, where["component"]]),
                             "energy_lead_values": energy[:, where["lead"]]}
    for model_position, model in enumerate(MODELS):
        for kind_position, kind in enumerate(KINDS):
            value = scores[:, :, model_position, kind_position]
            original = value[:, 0]
            threshold = thresholds[f"{model}_{kind}"]
            flagged = original > threshold
            entry = {"original": original, "flagged": flagged}
            for family in ("lead", "component", "grid", "swap_lead", "swap_component", "swap_grid"):
                after = value[:, where[family]]
                drop = original.reshape(-1, *([1] * (after.ndim - 1))) - after
                entry[f"{family}_drop"] = drop
                entry[f"{family}_flip"] = flagged & (after.reshape(len(value), -1) <= threshold).any(axis=1)
            for family in ("full", "swap_full"):
                after = value[:, where[family]]
                entry[f"{family}_score"] = after
                entry[f"{family}_drop"] = original - after
                entry[f"{family}_flip"] = flagged & (after <= threshold)
            entry["lead_shares"] = shares(entry["lead_drop"])
            entry["component_shares"] = shares(entry["component_drop"])
            entry["swap_lead_shares"] = shares(entry["swap_lead_drop"])
            entry["grid_abs"] = np.abs(entry["grid_drop"]).reshape(len(value), -1).mean(axis=1)
            entry["swap_grid_abs"] = np.abs(entry["swap_grid_drop"]).reshape(len(value), -1).mean(axis=1)
            entry["lead_abs"] = np.abs(entry["lead_drop"]).mean(axis=1)
            entry["swap_lead_abs"] = np.abs(entry["swap_lead_drop"]).mean(axis=1)
            table[f"{model}_{kind}"] = entry
    return table


def flip_rate(flips: np.ndarray, flagged: np.ndarray, mask: np.ndarray) -> float:
    """Share of flagged records in ``mask`` whose flag is removed."""
    chosen = flagged & mask
    return float(flips[chosen].mean()) if chosen.any() else float("nan")


def statistics(table: dict[str, Any], groups: np.ndarray, rows: np.ndarray) -> dict[str, float]:
    """
    Every reported statistic on one set of record positions (the sample or a bootstrap draw).

    Parameters
    ----------
    table : dict[str, Any]
        Output of ``record_table``.
    groups : np.ndarray
        Group of each record.
    rows : np.ndarray
        Record positions, with repeats in a bootstrap draw.

    Returns
    -------
    dict[str, float]
        Flat statistics keyed ``<model>_<kind>/<name>``, plus energy-map values under ``energy/``.
    """
    group = groups[rows]
    abnormal = np.isin(group, ABNORMAL)
    normal = group == "normal"
    result = {}
    energy_lead, energy_component = table["energy_lead"][rows], table["energy_component"][rows]
    energy_values = table["energy_lead_values"][rows]
    energy_territory = territory_contrast(energy_lead, group)
    energy_wave = wave_contrasts(energy_component, group)
    result["energy/territory"] = energy_territory
    result["energy/bbb_lead"] = bbb_lead_contrast(energy_lead, group)
    result.update({f"energy/{key}": value for key, value in energy_wave.items()})
    for top in (1, 3):
        result[f"energy/top{top}_hit"] = float(np.nanmean(top_hits(energy_values, group, top)))
    for key in [f"{model}_{kind}" for model in MODELS for kind in KINDS]:
        entry = {name: value[rows] for name, value in table[key].items()}
        territory = territory_contrast(entry["lead_shares"], group)
        result[f"{key}/territory"] = territory
        result[f"{key}/territory_minus_energy"] = territory - energy_territory
        result[f"{key}/swap_territory"] = territory_contrast(entry["swap_lead_shares"], group)
        result[f"{key}/bbb_lead"] = bbb_lead_contrast(entry["lead_shares"], group)
        result[f"{key}/bbb_lead_minus_energy"] = result[f"{key}/bbb_lead"] - result["energy/bbb_lead"]
        for name, value in wave_contrasts(entry["component_shares"], group).items():
            result[f"{key}/{name}"] = value
            result[f"{key}/{name}_minus_energy"] = value - energy_wave[name]
        for top in (1, 3):
            hits = top_hits(entry["lead_drop"], group, top)
            result[f"{key}/top{top}_hit"] = float(np.nanmean(hits))
            result[f"{key}/top{top}_hit_minus_energy"] = result[f"{key}/top{top}_hit"] - result[
                f"energy/top{top}_hit"]
        for subset, mask in (("abnormal", abnormal), ("normal", normal),
                             *((name, group == name) for name in (*ABNORMAL,))):
            if not mask.any():
                continue
            result[f"{key}/{subset}/flagged"] = float(entry["flagged"][mask].mean())
            result[f"{key}/{subset}/original"] = float(entry["original"][mask].mean())
            result[f"{key}/{subset}/full_score"] = float(entry["full_score"][mask].mean())
            result[f"{key}/{subset}/full_drop"] = float(entry["full_drop"][mask].mean())
            result[f"{key}/{subset}/swap_full_drop"] = float(entry["swap_full_drop"][mask].mean())
            result[f"{key}/{subset}/grid_abs"] = float(entry["grid_abs"][mask].mean())
            result[f"{key}/{subset}/swap_grid_abs"] = float(entry["swap_grid_abs"][mask].mean())
            result[f"{key}/{subset}/lead_abs"] = float(entry["lead_abs"][mask].mean())
            result[f"{key}/{subset}/swap_lead_abs"] = float(entry["swap_lead_abs"][mask].mean())
            for family in ("grid", "lead", "component", "full", "swap_grid", "swap_lead", "swap_full"):
                result[f"{key}/{subset}/{family}_flip"] = flip_rate(entry[f"{family}_flip"], entry["flagged"],
                                                                     mask)
    return result


def bootstrap(table: dict[str, Any], groups: np.ndarray, patients: np.ndarray,
              draws: int = DRAWS, seed: int = SEED) -> dict[str, dict[str, float]]:
    """
    Point values and 95% whole-patient bootstrap intervals of every statistic.

    Parameters
    ----------
    table : dict[str, Any]
        Output of ``record_table``.
    groups, patients : np.ndarray
        Group and patient of each record.
    draws : int
        Bootstrap draws.
    seed : int
        Generator seed.

    Returns
    -------
    dict[str, dict[str, float]]
        Per statistic: ``point``, ``low``, ``high`` and ``valid_draws`` (draws where it was defined).
    """
    point = statistics(table, groups, np.arange(len(groups)))
    rng = np.random.default_rng(seed)
    resample_groups = patient_groups(np.asarray(patients))
    values: dict[str, list[float]] = {key: [] for key in point}
    for _ in range(draws):
        drawn = statistics(table, groups, patient_resample(resample_groups, rng))
        for key in point:
            value = drawn.get(key, float("nan"))
            if np.isfinite(value):
                values[key].append(value)
    result = {}
    for key, value in point.items():
        low, high = (np.percentile(values[key], [2.5, 97.5]) if values[key] else (np.nan, np.nan))
        result[key] = {"point": value, "low": float(low), "high": float(high),
                       "valid_draws": len(values[key])}
    return result


def counts(groups: np.ndarray, table: dict[str, Any]) -> pd.DataFrame:
    """Count records and flagged records of each group under both blends."""
    frame = pd.DataFrame({"group": groups})
    for model in MODELS:
        frame[f"{model}_flagged"] = table[f"{model}_blend"]["flagged"]
    return frame.groupby("group").agg(records=("group", "size"),
                                      **{f"{model}_flagged": (f"{model}_flagged", "sum") for model in MODELS})
