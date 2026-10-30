"""
NBR-01 normal and self-swap counterparts, region masks and replaced variants.

The normal counterpart is the frozen Experiment 070 operator, ``replacement070.counterpart_difference``,
driven by the 062 rank-0 population model refitted on the LocBench ``fit`` clean normals. The self-swap
counterpart uses the same warp, trapezoid blend and taper, but each beat receives the record's own next
beat instead of the population template.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from .. import normative_map062 as nm
from .. import replacement070 as rp
from ..locbench import START, STOP
from ..locbench_cache import open_cache
from ..locbench_data import load_partition

LEADS = 12
COMPONENTS = ("P", "QRS", "ST-T")
COMPONENT_LABELS = ((0, 1), (2,), (3, 4))
WORK_070 = Path("/tmp/ecg070-work/full")
SET_SIZES_070 = {"A": 1500, "N": 400}

_STATE: dict[str, Any] = {}


@dataclass(frozen=True)
class Variant:
    """
    One replaced version of a record.

    Attributes
    ----------
    name : str
        Stable identifier, such as ``grid_6_2`` or ``swap_lead_1``.
    source : str
        ``"none"``, ``"counterpart"`` or ``"swap"``.
    leads : tuple[int, ...]
        Replaced leads.
    components : tuple[int, ...]
        Replaced components, indices into ``COMPONENTS``.
    """

    name: str
    source: str
    leads: tuple[int, ...]
    components: tuple[int, ...]


def variants() -> list[Variant]:
    """
    Return the fixed list of 105 variants scored for every record.

    Returns
    -------
    list[Variant]
        The original, then for the normal counterpart and the self-swap in turn: 36 lead-by-component
        cells, 12 whole leads, 3 whole components and the full replacement.
    """
    every, all_components = tuple(range(LEADS)), tuple(range(len(COMPONENTS)))
    items = [Variant("original", "none", (), ())]
    for source, prefix in (("counterpart", ""), ("swap", "swap_")):
        items += [Variant(f"{prefix}grid_{lead}_{part}", source, (lead,), (part,))
                  for lead in range(LEADS) for part in range(len(COMPONENTS))]
        items += [Variant(f"{prefix}lead_{lead}", source, (lead,), all_components) for lead in range(LEADS)]
        items += [Variant(f"{prefix}component_{part}", source, every, (part,))
                  for part in range(len(COMPONENTS))]
        items.append(Variant(f"{prefix}full", source, every, all_components))
    return items


def normative(model: nm.PopulationModel) -> SimpleNamespace:
    """Wrap a rank-0 population model in the attribute layout that ``replacement070`` reads."""
    return SimpleNamespace(models={"rank0": model})


def _cache_features(ecg_id: int) -> nm.RecordFeatures | None:
    return nm.record_features(np.asarray(_STATE["signals"][_STATE["row"][ecg_id]], dtype=float))


def _limited(function: Any, value: Any) -> Any:
    with threadpool_limits(limits=1):
        return function(value)


def _cache_features_limited(ecg_id: int) -> nm.RecordFeatures | None:
    return _limited(_cache_features, ecg_id)


def fit_rank0(workers: int = 16) -> tuple[nm.PopulationModel, dict[str, Any]]:
    """
    Refit the 062 rank-0 population model exactly as ``normative_map062.fit_normative`` does.

    Parameters
    ----------
    workers : int
        Forked processes for the per-record features, one thread each.

    Returns
    -------
    tuple[nm.PopulationModel, dict[str, Any]]
        The model and a summary of the fit records.
    """
    started = time.perf_counter()
    partition = load_partition()
    ids = partition[(partition.partition == "fit") & partition.clean_normal].ecg_id.to_numpy().astype(int)
    signals, index = open_cache()
    _STATE.update(signals=signals, row={int(e): i for i, e in enumerate(index.ecg_id)})
    with get_context("fork").Pool(workers) as pool:
        features = pool.map(_cache_features_limited, ids.tolist(), chunksize=16)
    kept = [position for position, value in enumerate(features) if value is not None]
    meta = index.set_index("ecg_id").loc[ids[kept]]
    intervals = np.stack([features[position].intervals for position in kept])
    covariates = nm.design(intervals[:, 0], meta.age.to_numpy(), meta.sex.to_numpy())
    medians = np.stack([features[position].median.ravel() for position in kept])
    model = nm.fit_population(medians, covariates, 0)
    summary = {"fit_records": len(ids), "fit_used": len(kept), "fit_patients": int(meta.patient_id.nunique()),
               "seconds": time.perf_counter() - started}
    return model, summary


def records_070() -> dict[str, pd.DataFrame]:
    """Redraw Experiment 070's sets A and N with its own selection functions and seeds."""
    partition = load_partition()
    evaluation = partition[partition["partition"] == "evaluation"].reset_index(drop=True)
    positive = rp.detection_positive(evaluation["ecg_id"].to_numpy())
    return {"A": rp.one_per_patient(evaluation[positive], SET_SIZES_070["A"], rp.SEED),
            "N": rp.one_per_patient(evaluation[evaluation["clean_normal"]], SET_SIZES_070["N"],
                                    rp.NORMAL_SEED)}


def _difference_070(row: tuple[int, int, str]) -> tuple[str, int, float]:
    position, saved_row, name = row
    with threadpool_limits(limits=1):
        raw = np.asarray(_STATE["signals"][position], np.float64)
        meta = _STATE["index"].iloc[position]
        built = rp.counterpart_difference(raw, float(meta.age), float(meta.sex), _STATE["normative"])
    saved = np.asarray(np.load(WORK_070 / name / "difference.npy", mmap_mode="r")[saved_row], np.float64)
    return name, saved_row, float(np.max(np.abs(built.astype(np.float64) - saved)))


def reproduce_070(model: nm.PopulationModel, count: int, workers: int = 16) -> dict[str, Any]:
    """
    Rebuild Experiment 070 counterparts and compare them with its saved ``difference.npy`` rows.

    Parameters
    ----------
    model : nm.PopulationModel
        Refitted rank-0 model.
    count : int
        Prepared records checked per set, the first ones in 070's draw order.
    workers : int
        Forked processes, one thread each.

    Returns
    -------
    dict[str, Any]
        Per set: records checked and the largest absolute difference in mV.

    Raises
    ------
    ValueError
        If a saved work array does not match the redrawn set.
    """
    signals, index = open_cache()
    row_of = {int(e): i for i, e in enumerate(index.ecg_id)}
    jobs = []
    for name, frame in records_070().items():
        status = np.load(WORK_070 / name / "status.npy")
        if len(status) != len(frame):
            raise ValueError(f"070 set {name} size differs from its saved work arrays")
        for saved_row in np.flatnonzero(status == 1)[:count]:
            jobs.append((row_of[int(frame.ecg_id.iloc[saved_row])], int(saved_row), name))
    _STATE.update(signals=signals, index=index, normative=normative(model))
    with get_context("fork").Pool(workers) as pool:
        results = pool.map(_difference_070, jobs, chunksize=2)
    summary: dict[str, Any] = {}
    for name in SET_SIZES_070:
        values = [value for set_name, _, value in results if set_name == name]
        summary[name] = {"records": len(values), "max_abs_difference_mv": max(values)}
    return summary


def beat_timeline(templates: np.ndarray, knots: np.ndarray,
                  samples: int = rp.SAMPLES) -> tuple[np.ndarray, np.ndarray]:
    """
    ``replacement070.template_timeline`` with one template per beat, also returning the summed weight.

    Parameters
    ----------
    templates : np.ndarray
        (beats, 12, BINS) beats on the 062 grid.
    knots : np.ndarray
        (beats, 6) aligned warp knots of the record.
    samples : int
        Record length.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (12, samples) blended timeline and (samples,) summed trapezoid weight.
    """
    total = np.zeros((templates.shape[1], samples))
    weight = np.zeros(samples)
    grid = np.arange(nm.BINS, dtype=float)
    for template, row in zip(templates, knots, strict=True):
        start, end = max(math.ceil(row[0]), 0), min(math.floor(row[-1]), samples - 1)
        if end < start:
            continue
        times = np.arange(start, end + 1)
        positions = np.interp(times, row, nm.GRID_KNOTS)
        ramp = np.clip(np.minimum(times - row[0], row[-1] - times) / rp.RAMP, 0.0, 1.0)
        for lead in range(template.shape[0]):
            total[lead, times] += ramp * np.interp(positions, grid, template[lead])
        weight[times] += ramp
    return total / np.maximum(weight, 1.0), weight


def record_plan(raw: np.ndarray, age: float, sex: float, model: nm.PopulationModel) -> dict[str, Any] | None:
    """
    Build the normal and self-swap differences and component masks of one record.

    Parameters
    ----------
    raw : np.ndarray
        (12, 5000) signal in mV at 500 Hz.
    age : float
        Years, NaN treated as 90 by the 062 design.
    sex : float
        PTB-XL coding: 0 male, 1 female.
    model : nm.PopulationModel
        Refitted rank-0 model.

    Returns
    -------
    dict[str, Any] or None
        ``difference`` (normal counterpart minus record) and ``swap`` (self-swap counterpart minus record),
        float32 (12, 5000); ``components`` (3, 5000) boolean time masks restricted to samples 100-4899 where
        complete-beat windows carry full weight; ``labels``, ``knots`` and ``heart_rate``. None when 062
        finds fewer than two complete beats or a component mask is empty.
    """
    raw = np.asarray(raw, np.float64)
    difference = rp.counterpart_difference(raw, age, sex, normative(model))
    if difference is None:
        return None
    features = nm.record_features(raw)
    clean = nm.preprocess(raw, nm.delineate(raw)["qon"])
    partner = (np.arange(len(features.knots)) + 1) % len(features.knots)
    swapped, weight = beat_timeline(np.nan_to_num(features.beats[partner]), features.knots, raw.shape[1])
    labels = rp.composition_labels(raw)
    scored = np.zeros(raw.shape[1], bool)
    scored[START:STOP] = True
    allowed = scored & (weight >= 1.0)
    components = np.stack([np.isin(labels, ids) & allowed for ids in COMPONENT_LABELS])
    if not components.any(axis=1).all():
        return None
    return {"difference": difference, "swap": (swapped - clean).astype(np.float32), "components": components,
            "labels": labels, "knots": features.knots, "heart_rate": float(features.intervals[0])}


def variant_masks(components: np.ndarray, items: list[Variant]) -> np.ndarray:
    """
    Binary lead-by-sample masks of the variants.

    Parameters
    ----------
    components : np.ndarray
        (3, samples) component time masks from ``record_plan``.
    items : list[Variant]
        Variants.

    Returns
    -------
    np.ndarray
        (variants, 12, samples) float64 masks.
    """
    masks = np.zeros((len(items), LEADS, components.shape[1]))
    for position, item in enumerate(items):
        if item.source != "none":
            masks[position, list(item.leads)] = components[list(item.components)].any(axis=0)
    return masks


def variant_signals(raw: np.ndarray, plan: dict[str, Any],
                    items: list[Variant]) -> tuple[np.ndarray, np.ndarray]:
    """
    Build replaced signals ``x + w * (c - x)`` with 070's 22 ms Hann taper, and the energy of each change.

    Parameters
    ----------
    raw : np.ndarray
        (12, samples) signal in mV.
    plan : dict[str, Any]
        Output of ``record_plan``.
    items : list[Variant]
        Variants.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (variants, 12, samples) float64 signals, and (variants,) summed squared change in mV^2.
    """
    raw = np.asarray(raw, np.float64)
    weights = rp.taper(torch.from_numpy(variant_masks(plan["components"], items))).numpy()
    changes = {"counterpart": plan["difference"].astype(np.float64), "swap": plan["swap"].astype(np.float64)}
    signals = np.repeat(raw[None], len(items), axis=0)
    energy = np.zeros(len(items))
    for position, item in enumerate(items):
        if item.source == "none":
            continue
        change = weights[position] * changes[item.source]
        signals[position] = raw + change
        energy[position] = float(np.sum(change**2))
    return signals, energy
