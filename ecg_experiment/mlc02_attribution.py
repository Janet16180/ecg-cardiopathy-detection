"""Attribute the PTB-XL differences of MLC-02 Stage 1 to the float32 cache.

The earlier xECG and measurement caches read PTB-XL as float64 (``int16 / 1000``), while MLC-02 reads the
float32 cache. Recomputing the same records from the raw float64 samples, with the MLC-02 batches, separates
the effect of the storage dtype from everything else. Only included PTB-XL records of the manifest are opened.
"""

from __future__ import annotations

import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from ecg_experiment.cpu_screening041 import RAW_FEATURES
from ecg_experiment.measurement_extractor042 import extract_array, read_ecg
from ecg_experiment.mlc02_integrity import (
    CACHES,
    cache_keys,
    cache_values,
    compare_measurements,
    compare_vectors,
)
from ecg_experiment.mlc02_waveforms import closed_keys, guard_path, init_worker

PTB_MEASUREMENTS = next(
    spec for spec in CACHES if spec["cache"] == "PR-74 PTB-XL" and spec["extractor"] == "measurements"
)
PTB_XECG = [spec for spec in CACHES if spec["extractor"] == "xecg" and "PTB-XL" in spec["cache"]]


def raw_ptb_window(raw_path: str) -> np.ndarray:
    """
    Read one PTB-XL record as the earlier caches did: raw int16 divided by 1000, float64.

    Parameters
    ----------
    raw_path : str
        Manifest ``raw_path`` of an included PTB-XL record.

    Returns
    -------
    np.ndarray
        Float64 ``(5000, 12)`` in the release's lead order (I, II, III, aVR, aVL, aVF, V1-V6).

    Raises
    ------
    ValueError
        If the record is not 10 s of twelve leads.
    """
    signal = read_ecg(guard_path(raw_path, "ptbxl"))
    if signal.shape != (5000, 12):
        raise ValueError(f"Unexpected PTB-XL shape {signal.shape}: {raw_path}")
    return signal


def raw_measurements(raw_path: str) -> list[float]:
    """
    Extract the 130 measurements of one PTB-XL record from its raw float64 samples.

    Parameters
    ----------
    raw_path : str
        Manifest ``raw_path``.

    Returns
    -------
    list[float]
        Values in ``RAW_FEATURES`` order, NaN where the extractor returned none.
    """
    row = extract_array(raw_ptb_window(raw_path))
    return [row.get(name, np.nan) for name in RAW_FEATURES]


def shared_positions(spec: dict[str, str], index: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Cache rows and MLC-02 positions of the records a cache shares with the index.

    Parameters
    ----------
    spec : dict[str, str]
        One entry of ``CACHES``.
    index : pd.DataFrame
        MLC-02 cache index.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Row numbers in the cache and the matching index positions.
    """
    keys, _ = cache_keys(spec)
    position = pd.Series(index.index.to_numpy(), index=index["record_key"].to_numpy())
    rows = np.flatnonzero(pd.Index(position.index).get_indexer(keys) >= 0)
    return rows, position.loc[keys[rows]].to_numpy()


def measurement_attribution(index: pd.DataFrame, new: np.ndarray, workers: int) -> dict[str, object]:
    """
    Compare the PR-74 PTB-XL measurements with a raw float64 recomputation and with the float32 run.

    Parameters
    ----------
    index : pd.DataFrame
        MLC-02 cache index.
    new : np.ndarray
        MLC-02 measurements in index order.
    workers : int
        Processes.

    Returns
    -------
    dict[str, object]
        ``float32_vs_cache``, ``raw_float64_vs_cache`` and ``float32_vs_raw_float64`` from
        ``compare_measurements``.
    """
    rows, where = shared_positions(PTB_MEASUREMENTS, index)
    old = cache_values(PTB_MEASUREMENTS)[rows]
    paths = index["raw_path"].to_numpy()[where].tolist()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with Pool(workers, initializer=init_worker, initargs=(closed_keys(), frozenset(paths))) as pool:
        raw = np.array(pool.map(raw_measurements, paths, chunksize=64), dtype=np.float64)
    return {
        "records": len(where),
        "float32_vs_cache": compare_measurements(new[where], old),
        "raw_float64_vs_cache": compare_measurements(raw, old),
        "float32_vs_raw_float64": compare_measurements(new[where], raw),
    }


def xecg_attribution(
    index: pd.DataFrame, new: np.ndarray, positions: np.ndarray, model: object
) -> dict[str, object]:
    """
    Recompute xECG on PTB-XL records from raw float64 samples in the MLC-02 microbatches and compare.

    Parameters
    ----------
    index : pd.DataFrame
        MLC-02 cache index.
    new : np.ndarray
        MLC-02 xECG features in index order.
    positions : np.ndarray
        Consecutive index positions of PTB-XL records starting a shard, so the microbatches equal the run's.
    model : object
        Frozen xECG from ``mlc02_features.load_model``.

    Returns
    -------
    dict[str, object]
        ``float32_vs_raw_float64`` on the selected records, and per PTB-XL cache the comparison of the raw
        recomputation and of the float32 run with the cache on the records they share.
    """
    from ecg_experiment.external_encoders import xecg_features, xecg_input

    rows = index.iloc[positions]
    if not (rows["source"] == "ptbxl").all():
        raise ValueError("Attribution positions must be PTB-XL records")
    init_worker(closed_keys(), frozenset(rows["raw_path"]))
    raw = xecg_features(model, np.stack([xecg_input(raw_ptb_window(path).T) for path in rows["raw_path"]]))
    record: dict[str, object] = {
        "records": len(positions),
        "first_position": int(positions[0]),
        "float32_vs_raw_float64": compare_vectors(new[positions], raw),
    }
    for spec in PTB_XECG:
        cache_rows, where = shared_positions(spec, index)
        inside = np.isin(where, positions)
        old = cache_values(spec)[cache_rows[inside]]
        local = np.searchsorted(positions, where[inside])
        record[spec["cache"]] = {
            "shared": int(inside.sum()),
            "raw_float64_vs_cache": compare_vectors(raw[local], old),
            "float32_vs_cache": compare_vectors(new[where[inside]], old),
        }
    return record
