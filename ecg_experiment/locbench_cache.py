"""
Local waveform cache for LocBench v1: PTB-XL folds 1-8 and the 399 ECGAD records.

Raw 500 Hz signals in mV, canonical lead order, float32, shape (n, 12, 5000), stored as a NumPy memmap
on local disk because ``/home`` is a slow network mount. The cache is rebuildable and never committed.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ecg_experiment.ecgad import load_split
from ecg_experiment.eda.ptbxl import load_metadata, read_signal

CACHE_DIR = Path(os.environ.get("ECG_LOCBENCH_CACHE", "/tmp/ecg-locbench-cache"))
SAMPLES = 5000


def build_cache(cache_dir: Path = CACHE_DIR, workers: int = 16) -> None:
    """
    Read every folds 1-8 record and every ECGAD record once and write the memmap and its index.

    Parameters
    ----------
    cache_dir : Path
        Output directory, created if missing.
    workers : int
        Reader threads.
    """
    meta = load_metadata()
    ecgad = set(load_split().ecg_id)
    rows = meta[(meta.strat_fold <= 8) | meta.index.isin(ecgad)]
    cache_dir.mkdir(parents=True, exist_ok=True)
    signals = np.lib.format.open_memmap(
        cache_dir / "signals.npy.part", mode="w+", dtype=np.float32, shape=(len(rows), 12, SAMPLES)
    )

    def read(position: int) -> None:
        signal, fs = read_signal(rows.filename_hr.iloc[position])
        if fs != 500 or signal.shape != (SAMPLES, 12):
            raise ValueError(f"Unexpected record {rows.index[position]}: fs={fs}, shape={signal.shape}")
        signals[position] = signal.T.astype(np.float32)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(read, range(len(rows))))
    signals.flush()
    index = pd.DataFrame(
        {
            "ecg_id": rows.index.to_numpy(),
            "patient_id": rows.patient_id.to_numpy().astype(int),
            "strat_fold": rows.strat_fold.to_numpy(),
            "age": rows.age_capped.to_numpy(),
            "sex": rows.sex.to_numpy(),
        }
    )
    index.to_json(cache_dir / "index.json", orient="split", index=False)
    os.replace(cache_dir / "signals.npy.part", cache_dir / "signals.npy")
    (cache_dir / "complete.json").write_text(json.dumps({"records": len(rows)}))


def open_cache(cache_dir: Path = CACHE_DIR) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Memory-mapped signals and their index.

    Parameters
    ----------
    cache_dir : Path
        Directory written by ``build_cache``.

    Returns
    -------
    tuple[np.ndarray, pd.DataFrame]
        Read-only float32 array (n, 12, 5000) in mV, and a table with ``ecg_id``, ``patient_id``,
        ``strat_fold``, ``age`` (NaN above 89) and ``sex`` in the same row order.
    """
    if not (cache_dir / "complete.json").exists():
        raise FileNotFoundError(f"No complete LocBench cache in {cache_dir}; run build_cache first")
    signals = np.load(cache_dir / "signals.npy", mmap_mode="r")
    index = pd.read_json(cache_dir / "index.json", orient="split")
    return signals, index
