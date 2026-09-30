"""CPU checks for the frozen selected-row GPU staging path."""

import numpy as np
import pytest
import torch

from ecg_experiment.selected_waveform_stage import stage_selected, staged_batches


class SmallDataset:
    """Minimal cache view for exact CPU staging checks."""

    def __init__(self, signals: np.ndarray, mean: np.ndarray, std: np.ndarray) -> None:
        """Expose the cache fields used by the staging function."""
        self.signals = signals
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        """Return the number of cache rows."""
        return len(self.signals)


def test_staging_preserves_normalization_order_and_partial_batch() -> None:
    """Repeated exposures use exactly one normalized copy of each selected row."""
    signals = np.arange(5 * 12 * 7, dtype=np.float32).reshape(5, 12, 7)
    mean = np.arange(12, dtype=np.float32)[:, None]
    std = np.full((12, 1), 2.0, dtype=np.float32)
    dataset = SmallDataset(signals, mean, std)
    order = np.asarray([3, 1, 4, 1, 3], dtype=np.int64)
    staged, positions = stage_selected(dataset, order, subset_size=3, exposures=5,
                                       device="cpu", chunk_size=2)
    actual = torch.cat(list(staged_batches(staged, positions, batch_size=2))).numpy()
    expected = (signals[order] - mean) / std
    np.testing.assert_array_equal(actual, expected)
    assert [len(batch) for batch in staged_batches(staged, positions, batch_size=2)] == [2, 2, 1]


def test_staging_rejects_changed_selection_and_nonfinite_rows() -> None:
    """A malformed stream or invalid normalized selected row fails closed."""
    signals = np.zeros((4, 12, 7), dtype=np.float32)
    dataset = SmallDataset(signals, np.zeros((12, 1), dtype=np.float32),
                           np.ones((12, 1), dtype=np.float32))
    with pytest.raises(ValueError, match="selected cache rows"):
        stage_selected(dataset, np.asarray([0, 1]), subset_size=3,
                       exposures=2, device="cpu")
    signals[1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite normalized"):
        stage_selected(dataset, np.asarray([0, 1]), subset_size=2,
                       exposures=2, device="cpu")
