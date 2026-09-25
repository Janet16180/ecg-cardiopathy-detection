"""Optional exact-float32 GPU staging for the frozen CPC training waveforms."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch

from .cpc_pool import Pool


def stage_training_signals(
    pool: Pool,
    mean: np.ndarray,
    std: np.ndarray,
    *,
    device: str = "cuda",
    chunk_size: int = 128,
) -> torch.Tensor:
    """Copy normalized training signals to one device without modifying the cache.

    The copy retains float32 precision and the exact training-row order. Only
    training records are staged; held-out waveforms remain outside this tensor.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    indices = pool.indices(pool.train_rows)
    shape = (len(indices), *pool.signals.shape[1:])
    staged = torch.empty(shape, dtype=torch.float32, device=device)
    lead_mean = mean[:, None]
    lead_std = std[:, None]
    for start in range(0, len(indices), chunk_size):
        selected = indices[start : start + chunk_size]
        block = np.array(pool.signals[selected], dtype=np.float32, copy=True)
        block -= lead_mean
        block /= lead_std
        staged[start : start + len(selected)] = torch.from_numpy(block).to(device)
    return staged


def shuffled_batches(
    signals: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
) -> Iterator[torch.Tensor]:
    """Yield every staged recording once in a seeded random order."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    order = torch.randperm(len(signals), generator=generator, device="cpu")
    for start in range(0, len(signals), batch_size):
        indices = order[start : start + batch_size].to(signals.device)
        yield signals[indices]
