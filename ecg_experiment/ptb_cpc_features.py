"""PTB-XL CPC inputs and frozen features for readouts after Experiment 020.

These functions repeat the input path of the frozen Experiment 020 runner so that later readouts use the
same PTB-XL signals without importing a script.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .cpc import CPCEncoder
from .cpc_input_audit import historical_resample
from .cpc_pool import Pool
from .waveforms import read_record

ROOT = Path(__file__).resolve().parents[1]
POOL_DIR = ROOT / "data/processed/cpc_pool_40k"
PTB_RAW = ROOT / "data/raw/ptb-xl/1.0.3"
NORMALIZATION = ROOT / "outputs/experiment004_cpc_40k/normalization.json"


def ptb_signals(pool: Pool, frame: pd.DataFrame) -> np.ndarray:
    """
    Read 250 Hz PTB-XL CPC inputs from the historical cache or the raw files.

    Parameters
    ----------
    pool : Pool
        Historical 250 Hz cache.
    frame : pd.DataFrame
        Rows with ``ecg_id`` and ``filename_hr``, as from ``full_development.ptb_table``.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(rows, 12, 2500)`` in mV.
    """
    rows = []
    for row in frame.itertuples():
        index = pool.index.get(str(row.ecg_id))
        cached = index is not None and pool.rows[index]["source"] == "ptbxl"
        rows.append(np.array(pool.signals[index]) if cached
                    else historical_resample(read_record(PTB_RAW, row.filename_hr)))
    return np.stack(rows)


@torch.inference_mode()
def pooled_features(model: CPCEncoder, signals: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """
    Normalize with the historical statistics and return the 512 pooled features.

    Parameters
    ----------
    model : CPCEncoder
        Encoder on the GPU, in evaluation mode.
    signals : np.ndarray
        Array of shape ``(records, 12, 2500)`` in mV.
    factor : float
        Amplitude factor applied before normalization.

    Returns
    -------
    np.ndarray
        Float32 array of shape ``(records, 512)``.

    Raises
    ------
    ValueError
        If a feature is not finite.
    """
    statistics = json.loads(NORMALIZATION.read_text())
    mean = np.asarray(statistics["mean"], dtype=np.float32)[:, None]
    std = np.asarray(statistics["std"], dtype=np.float32)[:, None]
    chunks = []
    for start in range(0, len(signals), 128):
        batch = torch.from_numpy((signals[start:start + 128] * factor - mean) / std).cuda()
        _, contexts = model(batch)
        chunks.append(CPCEncoder.pooled(contexts).float().cpu().numpy())
    features = np.concatenate(chunks)
    if features.shape != (len(signals), 512) or not np.isfinite(features).all():
        raise ValueError("Invalid CPC features")
    return features
