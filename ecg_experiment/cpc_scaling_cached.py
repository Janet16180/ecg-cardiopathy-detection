"""Read a completed local 250 Hz cache for the CPC scaling successor."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .files import sha256_file


class CachedCPCDataset(Dataset[tuple[torch.Tensor, float, str]]):
    """Serve manifest-aligned CPC signals with the historical normalization."""

    def __init__(self, cache: Path, normalization: Path, *, verify_hash: bool = True) -> None:
        """Verify cache completion and open its read-only waveform array."""
        complete = json.loads((cache / "complete.json").read_text())
        path = cache / "signals.npy"
        if verify_hash and sha256_file(path) != complete["signals_sha256"]:
            raise ValueError("CPC scaling cache hash mismatch")
        self.signals = np.load(path, mmap_mode="r", allow_pickle=False)
        if (self.signals.shape != tuple(complete["signals_shape"])
                or self.signals.dtype != np.float32):
            raise ValueError("CPC scaling cache contract mismatch")
        values = json.loads(normalization.read_text())
        self.mean = np.asarray(values["mean"], dtype=np.float32)[:, None]
        self.std = np.asarray(values["std"], dtype=np.float32)[:, None]
        if self.mean.shape != (12, 1) or self.std.shape != (12, 1) or np.any(self.std <= 0):
            raise ValueError("Invalid CPC normalization")

    def __len__(self) -> int:
        """Return the verified number of cached ECGs."""
        return len(self.signals)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, float, str]:
        """Normalize one cached signal without changing the cache."""
        signal = np.array(self.signals[index], copy=True)
        signal -= self.mean
        signal /= self.std
        return torch.from_numpy(signal), -1.0, str(index)
