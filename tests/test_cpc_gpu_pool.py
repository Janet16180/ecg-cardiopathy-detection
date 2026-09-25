"""Exact staging and deterministic batch-order checks for the 011 GPU cache."""

import numpy as np
import torch

from ecg_experiment.cpc_gpu_pool import shuffled_batches, stage_training_signals


class TinyPool:
    """Minimal cache adapter with train and held-out rows interleaved."""

    def __init__(self) -> None:
        self.signals = np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3)
        self.train_rows = [{"ecg_id": "2"}, {"ecg_id": "0"}, {"ecg_id": "3"}]

    def indices(self, rows):
        return [int(row["ecg_id"]) for row in rows]


def test_staging_keeps_exact_float32_values_and_source_bytes() -> None:
    pool = TinyPool()
    original = pool.signals.copy()
    mean = np.array([2, 3], dtype=np.float32)
    std = np.array([4, 5], dtype=np.float32)
    result = stage_training_signals(pool, mean, std, device="cpu", chunk_size=2)

    expected = original[[2, 0, 3]].copy()
    expected -= mean[:, None]
    expected /= std[:, None]
    torch.testing.assert_close(result, torch.from_numpy(expected), atol=0, rtol=0)
    np.testing.assert_array_equal(pool.signals, original)


def test_seeded_batches_expose_each_training_record_once() -> None:
    signals = torch.arange(15).reshape(5, 3)
    first = list(shuffled_batches(signals, 2, torch.Generator().manual_seed(42)))
    second = list(shuffled_batches(signals, 2, torch.Generator().manual_seed(42)))
    assert [len(batch) for batch in first] == [2, 2, 1]
    torch.testing.assert_close(torch.cat(first), torch.cat(second), atol=0, rtol=0)
    assert sorted(torch.cat(first)[:, 0].tolist()) == [0, 3, 6, 9, 12]
