"""Bounded RAM view of selected rows from the existing CPC waveform cache.

The source cache and its scientific preprocessing remain unchanged.  A subset is
copied in source-file order to reduce random disk reads, then exposed in the
caller's exact row order so existing samplers and normalization still apply.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np


def _available_memory_bytes() -> int:
    """Conservative allocation headroom from Linux host and finite cgroup caps."""
    info = Path("/proc/meminfo").read_text()
    available = next(
        int(line.split()[1]) * 1024
        for line in info.splitlines()
        if line.startswith("MemAvailable:")
    )
    cgroup = Path("/proc/self/cgroup")
    if cgroup.exists():
        for line in cgroup.read_text().splitlines():
            hierarchy, controllers, group = line.split(":", 2)
            if hierarchy != "0" or controllers:
                continue
            path = Path("/sys/fs/cgroup") / group.lstrip("/")
            root = Path("/sys/fs/cgroup")
            while path == root or root in path.parents:
                maximum_path = path / "memory.max"
                current_path = path / "memory.current"
                if maximum_path.exists() and current_path.exists():
                    maximum = maximum_path.read_text().strip()
                    if maximum != "max":
                        available = min(available, max(0, int(maximum) - int(current_path.read_text())))
                if path == root:
                    break
                path = path.parent
            break
    return available


class BoundedWaveformCache:
    """Float32, read-only subset in ``rows`` order with explicit RAM limits.

    ``pool`` is the verified CPC ``Pool``.  Each row needs an ``ecg_id``.
    ``signals[i]`` is exactly the source waveform for ``rows[i]``.  A cache may
    cover several splits; ``indices(subset_rows)`` supports the existing
    ``PoolDataset`` without changing its sampling or normalization behavior.
    """

    def __init__(
        self,
        pool,
        rows,
        *,
        max_bytes: int = 2_400_000_000,
        reserve_bytes: int = 1_000_000_000,
        chunk_records: int = 128,
        expected_source: str | None = None,
    ):
        rows = list(rows)
        if not rows:
            raise ValueError("At least one waveform row is required")
        if max_bytes <= 0 or reserve_bytes < 0 or chunk_records <= 0:
            raise ValueError("Invalid memory or chunk limit")
        source = pool.signals
        if source.dtype != np.float32 or source.ndim != 3 or source.shape[1:] != (12, 2500):
            raise ValueError("Expected float32 CPC waveforms with shape [N,12,2500]")
        row_ids = [str(row["ecg_id"]) for row in rows]
        if len(set(row_ids)) != len(row_ids):
            raise ValueError("Duplicate ECG IDs in requested rows")
        source_indices = np.asarray(pool.indices(rows), dtype=np.int64)
        if np.any(source_indices < 0) or np.any(source_indices >= len(source)):
            raise ValueError("Source index outside waveform cache")
        for row, source_index in zip(rows, source_indices):
            source_row = pool.rows[int(source_index)]
            if expected_source is not None and source_row.get("source") != expected_source:
                raise ValueError(f"Wrong waveform source for ECG {row['ecg_id']}")
            for key in ("ecg_id", "patient_id", "source", "split"):
                if key in row and key in source_row and str(row[key]) != str(source_row[key]):
                    raise ValueError(f"Cache row mismatch for {key}: {row['ecg_id']}")

        required = len(rows) * int(np.prod(source.shape[1:])) * source.dtype.itemsize
        if required > max_bytes:
            raise MemoryError(f"Waveform subset needs {required} bytes; max_bytes={max_bytes}")
        available = _available_memory_bytes()
        if required + reserve_bytes > available:
            raise MemoryError(
                f"Waveform subset needs {required} bytes plus {reserve_bytes} reserve; "
                f"available={available}"
            )

        started = time.monotonic()
        signals = np.empty((len(rows), *source.shape[1:]), dtype=np.float32)
        source_order = np.argsort(source_indices, kind="stable")
        for start in range(0, len(rows), chunk_records):
            destinations = source_order[start:start + chunk_records]
            signals[destinations] = source[source_indices[destinations]]
        signals.flags.writeable = False

        self.signals = signals
        self.rows = rows
        self.index = {ecg_id: i for i, ecg_id in enumerate(row_ids)}
        self.bytes_loaded = required
        self.load_seconds = time.monotonic() - started

    def indices(self, rows):
        return [self.index[str(row["ecg_id"])] for row in rows]
