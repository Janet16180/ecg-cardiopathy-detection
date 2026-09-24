"""Checks for the RAM subset used by affordable waveform pilots."""

import numpy as np
import pytest
import torch

from ecg_experiment import bounded_waveform_cache as bounded
from ecg_experiment.cpc_pool import Pool, loader


def test_subset_preserves_waveforms_rows_and_seeded_batches(synthetic_cpc_pool):
    pool = Pool(synthetic_cpc_pool)
    rows = [pool.train_rows[i] for i in (103, 2, 507, 31, 6, 88, 13, 0)]
    cache = bounded.BoundedWaveformCache(pool, rows, max_bytes=1_000_000,
                                        reserve_bytes=0, chunk_records=3,
                                        expected_source="ptbxl")
    assert cache.signals.shape == (8, 12, 2500)
    assert cache.signals.flags.writeable is False
    assert cache.indices(rows[::2]) == [0, 2, 4, 6]
    assert cache.bytes_loaded == 8 * 12 * 2500 * 4
    for i, row in enumerate(rows):
        np.testing.assert_array_equal(cache.signals[i], pool.signals[pool.index[row["ecg_id"]]])

    mean = np.linspace(-0.01, 0.01, 12, dtype=np.float32)
    std = np.linspace(0.1, 0.3, 12, dtype=np.float32)
    old_data = loader(pool, rows, mean, std, 3, True,
                      torch.Generator().manual_seed(42), "cpu")
    new_data = loader(cache, rows, mean, std, 3, True,
                      torch.Generator().manual_seed(42), "cpu")
    for old_batch, new_batch in zip(old_data, new_data, strict=True):
        torch.testing.assert_close(old_batch[0], new_batch[0], atol=0, rtol=0)
        assert old_batch[1].tolist() == new_batch[1].tolist()
        assert list(old_batch[2]) == list(new_batch[2])


def test_rejects_budget_duplicate_or_patient_mismatch(synthetic_cpc_pool, monkeypatch):
    pool = Pool(synthetic_cpc_pool)
    rows = pool.train_rows[:2]
    monkeypatch.setattr(bounded, "_available_memory_bytes", lambda: 1_000_000)
    with pytest.raises(MemoryError, match="max_bytes"):
        bounded.BoundedWaveformCache(pool, rows, max_bytes=1)
    with pytest.raises(MemoryError, match="available"):
        bounded.BoundedWaveformCache(pool, rows, max_bytes=1_000_000,
                                    reserve_bytes=1_000_000)
    with pytest.raises(ValueError, match="Duplicate"):
        bounded.BoundedWaveformCache(pool, [rows[0], rows[0]])
    changed = dict(rows[0], patient_id="wrong-patient")
    with pytest.raises(ValueError, match="patient_id"):
        bounded.BoundedWaveformCache(pool, [changed])
    with pytest.raises(ValueError, match="source"):
        bounded.BoundedWaveformCache(pool, [rows[0]], expected_source="mimic")


def test_cgroup_limit_uses_tightest_finite_cap_up_to_root(tmp_path, monkeypatch):
    monkeypatch.setattr(bounded, "CGROUP_ROOT", tmp_path)
    leaf = tmp_path / "user.slice" / "job"
    leaf.mkdir(parents=True)
    (leaf / "memory.max").write_text("max\n")
    (leaf / "memory.current").write_text("5\n")
    (leaf.parent / "memory.max").write_text("1000\n")
    (leaf.parent / "memory.current").write_text("400\n")
    (tmp_path / "memory.max").write_text("5000\n")
    (tmp_path / "memory.current").write_text("4700\n")
    assert bounded._cgroup_limit("/user.slice/job", 10_000) == 300
    assert bounded._cgroup_limit("/user.slice/job", 100) == 100
    (tmp_path / "memory.current").write_text("6000\n")
    assert bounded._cgroup_limit("/user.slice/job", 10_000) == 0
