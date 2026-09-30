"""Stage a frozen normalized ECG subset once for repeated GPU exposures."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch

from .cpc_scaling_cached import CachedCPCDataset


def stage_selected(
    dataset: CachedCPCDataset,
    order: np.ndarray,
    *,
    subset_size: int,
    exposures: int,
    device: str = "cuda",
    chunk_size: int = 128,
) -> tuple[torch.Tensor, np.ndarray]:
    """Copy each selected row once in normalized float32 and map exposure order."""
    if chunk_size < 1 or len(order) != exposures or order.ndim != 1:
        raise ValueError("Invalid staging chunk or exposure count")
    selected = np.unique(order)
    if (len(selected) != subset_size or selected[0] < 0
            or selected[-1] >= len(dataset)):
        raise ValueError("Frozen selected cache rows changed")
    positions = np.full(len(dataset), -1, dtype=np.int32)
    positions[selected] = np.arange(subset_size, dtype=np.int32)
    staged_order = positions[order]
    if np.any(staged_order < 0):
        raise ValueError("Exposure order includes an unstaged cache row")
    shape = (subset_size, *dataset.signals.shape[1:])
    staged = torch.empty(shape, dtype=torch.float32, device=device)
    for start in range(0, subset_size, chunk_size):
        indices = selected[start:start + chunk_size]
        block = np.array(dataset.signals[indices], dtype=np.float32, copy=True)
        block -= dataset.mean
        block /= dataset.std
        if not np.isfinite(block).all():
            raise ValueError("Nonfinite normalized waveform in selected subset")
        staged[start:start + len(indices)] = torch.from_numpy(block).to(device)
    return staged, staged_order


def staged_batches(
    staged: torch.Tensor, order: np.ndarray, *, batch_size: int,
) -> Iterator[torch.Tensor]:
    """Yield the exact frozen exposure suffix from staged rows."""
    if batch_size < 1:
        raise ValueError("Invalid batch size")
    for start in range(0, len(order), batch_size):
        indices = torch.as_tensor(order[start:start + batch_size],
                                  dtype=torch.long, device=staged.device)
        yield staged[indices]
