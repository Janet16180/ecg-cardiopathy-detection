"""Frozen CPC pool and normalization inputs for new ECG architecture profiles."""

from __future__ import annotations

import numpy as np

from . import ROOT, cpc_pool
from .files import sha256_file

NORMALIZATION = ROOT / "outputs/experiment004_cpc_40k"
EXPECTED_TRAINING_RECORDS = 56875
BATCH_SIZE = 128
SEED = 9001
LR = 1e-3
WEIGHT_DECAY = 0.01


def prepare_inputs() -> tuple[cpc_pool.Pool, np.ndarray, np.ndarray, dict[str, str]]:
    """Verify exact pool bytes and training-only normalization once per profile."""
    pool = cpc_pool.Pool(cpc_pool.DEFAULT_CACHE)
    if len(pool.train_rows) != EXPECTED_TRAINING_RECORDS:
        raise ValueError("Frozen CPC training pool count changed")

    signal_path = pool.directory / "signals.npy"
    signal_hash = sha256_file(signal_path)
    if signal_hash != pool.metadata["signals_sha256"]:
        raise ValueError("Frozen CPC signal cache changed")
    mean, std = pool.normalization(
        NORMALIZATION, {str(signal_path.resolve()): signal_hash}
    )
    sources = {
        str((pool.directory / name).resolve()): sha256_file(pool.directory / name)
        for name in ("complete.json", "rows.csv", "ecg_ids.npy")
    }
    sources[str(signal_path.resolve())] = signal_hash
    normalization_path = NORMALIZATION / "normalization.json"
    sources[str(normalization_path.resolve())] = sha256_file(normalization_path)
    return pool, mean, std, sources
