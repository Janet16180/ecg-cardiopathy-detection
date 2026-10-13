"""Attention finding heads on frozen ECG-JEPA tokens (Experiment 050): token lookup, training and readings.

Tokens of one experiment can live in several row caches. A locator maps each virtual row (a training,
development or SPH ECG) to its cache and its row there, and a batch function reads any set of virtual rows
from the right caches in order.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch

from .ann_heads import Recipe, RowFile, predict, read_rows, train
from .lead_wave_maps import UnitMap
from .waveforms import LEADS

MATCH_MARGIN = 0.005


def token_locator(parts: Sequence[tuple[int, np.ndarray, np.ndarray]], total: int) -> np.ndarray:
    """
    Build the cache and row of every virtual row.

    Parameters
    ----------
    parts : Sequence[tuple[int, np.ndarray, np.ndarray]]
        ``(cache, virtual rows, cache rows)`` triples.
    total : int
        Number of virtual rows.

    Returns
    -------
    np.ndarray
        ``(total, 2)`` integer array of cache index and cache row, ``-1`` where a row has no tokens.

    Raises
    ------
    ValueError
        If a virtual row is located twice or the lengths of a part differ.
    """
    found = np.full((total, 2), -1, dtype=np.int64)
    for cache, virtual, rows in parts:
        virtual, rows = np.asarray(virtual, dtype=np.int64), np.asarray(rows, dtype=np.int64)
        if len(virtual) != len(rows) or (found[virtual, 0] >= 0).any():
            raise ValueError("Each virtual row needs exactly one cache row")
        found[virtual] = np.column_stack([np.full(len(rows), cache), rows])
    return found


def located_tokens(stores: Sequence[RowFile], locator: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """
    Read the tokens of virtual rows from their caches, in the given order.

    Parameters
    ----------
    stores : Sequence[RowFile]
        Open caches, indexed as in the locator.
    locator : np.ndarray
        Output of ``token_locator``.
    rows : np.ndarray
        Virtual rows.

    Returns
    -------
    np.ndarray
        ``[len(rows), *token shape]`` tokens in the caches' dtype.

    Raises
    ------
    ValueError
        If a row has no tokens or the caches differ in shape or dtype.
    """
    rows = np.asarray(rows, dtype=np.int64)
    where = locator[rows]
    if (where[:, 0] < 0).any():
        raise ValueError("A requested row has no tokens")
    shapes = {(store.shape[1:], store.dtype) for store in stores}
    if len(shapes) != 1:
        raise ValueError("Token caches differ in shape or dtype")
    values = np.empty((len(rows), *stores[0].shape[1:]), dtype=stores[0].dtype)
    for cache in np.unique(where[:, 0]):
        chosen = where[:, 0] == cache
        values[chosen] = read_rows(stores[cache], where[chosen, 1])
    return values


def located_batch(stores: Sequence[RowFile], locator: np.ndarray, device: str) -> Callable[[np.ndarray], Any]:
    """Return a batch function reading virtual rows as float32 tensors on the device."""
    return lambda rows: torch.from_numpy(located_tokens(stores, locator, rows)).to(device).float()


def train_seeds(build: Callable[[], torch.nn.Module], batch: Callable[[np.ndarray], Any], fit: dict[str, Any],
                score: dict[str, np.ndarray], recipe: Recipe, seeds: Sequence[int], device: str,
                keep: tuple[str, ...] = (), log: Callable[[str], None] = print) -> dict[str, Any]:
    """
    Train one network per seed (seeded before it is built), keep its best weights and score the parts.

    Parameters
    ----------
    build : Callable[[], torch.nn.Module]
        Builds the network on the CPU.
    batch : Callable[[np.ndarray], Any]
        Maps rows to an input tensor on the device.
    fit : dict[str, Any]
        ``rows``, ``y``, ``weights``, ``check_rows`` and ``check_y``.
    score : dict[str, np.ndarray]
        Rows per scored part.
    recipe : Recipe
        Training recipe.
    seeds : Sequence[int]
        Seeds.
    device : str
        Torch device.
    keep : tuple[str, ...]
        Parts whose per-unit contributions are kept.
    log : Callable[[str], None]
        Receives the training lines.

    Returns
    -------
    dict[str, Any]
        ``logits`` and ``contributions`` (seed means per part), ``seed_logits``, ``seed_contributions``,
        ``states`` (best weights per seed as arrays) and ``seeds`` (best epoch, validation AUROC, history).
    """
    seed_logits, seed_contributions, states, info = {}, {}, {}, {}
    for seed in seeds:
        began = time.perf_counter()
        torch.manual_seed(seed)
        model = build().to(device)
        result = train(model, batch, fit["rows"], fit["y"], fit["weights"], fit["check_rows"], fit["check_y"],
                       recipe, seed, log=log)
        states[seed] = {key: value.detach().cpu().numpy().copy() for key, value in model.state_dict().items()}
        seed_logits[seed], seed_contributions[seed] = {}, {}
        for part, rows in score.items():
            logits, units = predict(model, batch, rows, recipe.batch_size * 4,
                                    keep_contributions=part in keep)
            seed_logits[seed][part] = logits
            if part in keep:
                seed_contributions[seed][part] = units
        info[str(seed)] = {"best_epoch": result.best_epoch, "validation_auroc": result.best_auroc,
                           "epochs_run": len(result.history), "seconds": time.perf_counter() - began,
                           "history": result.history}
        del model
    return {"logits": {part: np.mean([seed_logits[s][part] for s in seeds], axis=0) for part in score},
            "contributions": {part: np.mean([seed_contributions[s][part] for s in seeds], axis=0)
                              for part in keep},
            "seed_logits": seed_logits, "seed_contributions": seed_contributions, "states": states,
            "seeds": info}


def finding_reading(ci_low: float, margin: float = MATCH_MARGIN) -> str:
    """
    Read a candidate minus comparator AUROC difference.

    Parameters
    ----------
    ci_low : float
        Lower bound of the difference.
    margin : float
        Positive matching margin.

    Returns
    -------
    str
        ``beats`` when the lower bound is above 0, ``matches`` when it is above ``-margin``, else ``below``.
    """
    if ci_low > 0:
        return "beats"
    if ci_low > -margin:
        return "matches"
    return "below"


def top_unit(unit_map: UnitMap) -> dict[str, Any]:
    """Return the lead name, time span and score of a map's top unit."""
    index = int(np.argmax(unit_map.scores))
    return {"lead": LEADS[int(unit_map.leads[index])], "start": float(unit_map.starts[index]),
            "end": float(unit_map.ends[index]), "score": float(unit_map.scores[index])}


def unlabeled_sph_table(sph: pd.DataFrame, labeled: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Return Experiment 043's reader rows of the SPH ECGs outside the labeled positions, and their positions.

    Parameters
    ----------
    sph : pd.DataFrame
        SPH evaluation rows with ``ecg_id`` and ``signal_sha256``.
    labeled : np.ndarray
        Positions of the labeled rows.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray]
        Reader rows (``kind``, ``key``, ``ecg_id``, ``signal_sha256``) and their positions.
    """
    mask = np.ones(len(sph), dtype=bool)
    mask[labeled] = False
    positions = np.flatnonzero(mask)
    rows = sph.iloc[positions]
    table = pd.DataFrame({"kind": "sph", "key": ("sph:" + rows["ecg_id"]).to_numpy(dtype=str),
                          "ecg_id": rows["ecg_id"].to_numpy(dtype=str),
                          "signal_sha256": rows["signal_sha256"].to_numpy(dtype=str)})
    return table, positions
