"""Verified 100k-cohort input adapter for a compact CPC scaling pilot."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .cpc_input_audit import historical_resample
from .sampled_training_dataset import SampledTrainingECGDataset


class SampledCPCDataset(Dataset[tuple[torch.Tensor, float, str]]):
    """Convert hash-checked 500 Hz ECGs to the frozen 250 Hz CPC input."""

    def __init__(self, directory: Path, normalization: Path) -> None:
        """Keep the historical train-only normalizer fixed across comparison arms."""
        self.source = SampledTrainingECGDataset(directory, purpose="ssl")
        values = json.loads(normalization.read_text())
        self.mean = np.asarray(values["mean"], dtype=np.float32)[:, None]
        self.std = np.asarray(values["std"], dtype=np.float32)[:, None]
        if self.mean.shape != (12, 1) or self.std.shape != (12, 1) or np.any(self.std <= 0):
            raise ValueError("Invalid historical CPC normalization")

    def __len__(self) -> int:
        """Return the frozen sampled-cohort size."""
        return len(self.source)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, float, str]:
        """Read, hash-check, resample, and normalize one ECG."""
        row = self.source[index]
        signal = historical_resample(row["signal"])
        signal = (signal - self.mean) / self.std
        return torch.from_numpy(signal), -1.0, row["patient_id"] or row["record_id"]
