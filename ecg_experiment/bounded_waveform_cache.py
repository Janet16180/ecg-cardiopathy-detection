"""Bounded RAM view of selected rows from the existing CPC waveform cache.

The source cache and its scientific preprocessing remain unchanged.  A subset is
copied in source-file order to reduce random disk reads, then exposed in the
caller's exact row order so existing samplers and normalization still apply.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

CGROUP_ROOT = Path("/sys/fs/cgroup")
WAVEFORM_SHAPE = (12, 2500)
ROW_IDENTITY_KEYS = ("ecg_id", "patient_id", "source", "split")


class WaveformPool(Protocol):
    """The parts of the verified CPC ``Pool`` that the bounded cache reads."""

    signals: np.ndarray
    rows: Sequence[Mapping[str, Any]]

    def indices(self, rows: Iterable[Mapping[str, Any]]) -> list[int]:
        """Return source-cache positions of ``rows``."""
        ...


def _cgroup_limit(group: str, available: int) -> int:
    """Tighten ``available`` by the remaining headroom of each finite cgroup v2 cap up to the root."""
    path = CGROUP_ROOT / group.lstrip("/")
    while path == CGROUP_ROOT or CGROUP_ROOT in path.parents:
        maximum_path = path / "memory.max"
        current_path = path / "memory.current"
        if maximum_path.exists() and current_path.exists():
            maximum = maximum_path.read_text().strip()
            if maximum != "max":
                available = min(available, max(0, int(maximum) - int(current_path.read_text())))
        if path == CGROUP_ROOT:
            break
        path = path.parent
    return available


def _available_memory_bytes() -> int:
    """Conservative allocation headroom from Linux host and finite cgroup caps."""
    info = Path("/proc/meminfo").read_text()
    available = next(
        int(line.split()[1]) * 1024
        for line in info.splitlines()
        if line.startswith("MemAvailable:")
    )
    cgroup = Path("/proc/self/cgroup")
    if not cgroup.exists():
        return available
    for line in cgroup.read_text().splitlines():
        hierarchy, controllers, group = line.split(":", 2)
        if hierarchy == "0" and not controllers:
            available = _cgroup_limit(group, available)
            break
    return available


def _check_rows(pool: WaveformPool, rows: list[Mapping[str, Any]], source_indices: np.ndarray,
                expected_source: str | None) -> None:
    """Require each requested row to match its source-cache row and expected source."""
    for row, source_index in zip(rows, source_indices, strict=True):
        source_row = pool.rows[int(source_index)]
        if expected_source is not None and source_row.get("source") != expected_source:
            raise ValueError(f"Wrong waveform source for ECG {row['ecg_id']}")
        for key in ROW_IDENTITY_KEYS:
            if key in row and key in source_row and str(row[key]) != str(source_row[key]):
                raise ValueError(f"Cache row mismatch for {key}: {row['ecg_id']}")


def _check_memory(required: int, max_bytes: int, reserve_bytes: int) -> None:
    """Refuse a subset larger than ``max_bytes`` or than current headroom minus the reserve."""
    if required > max_bytes:
        raise MemoryError(f"Waveform subset needs {required} bytes; max_bytes={max_bytes}")
    available = _available_memory_bytes()
    if required + reserve_bytes > available:
        raise MemoryError(
            f"Waveform subset needs {required} bytes plus {reserve_bytes} reserve; "
            f"available={available}"
        )


class BoundedWaveformCache:
    """
    Float32, read-only subset in ``rows`` order with explicit RAM limits.

    ``pool`` is the verified CPC ``Pool``.  Each row needs an ``ecg_id``.
    ``signals[i]`` is exactly the source waveform for ``rows[i]``.  A cache may
    cover several splits; ``indices(subset_rows)`` supports the existing
    ``PoolDataset`` without changing its sampling or normalization behavior.
    """

    def __init__(
        self,
        pool: WaveformPool,
        rows: Iterable[Mapping[str, Any]],
        *,
        max_bytes: int = 2_400_000_000,
        reserve_bytes: int = 1_000_000_000,
        chunk_records: int = 128,
        expected_source: str | None = None,
    ) -> None:
        """
        Validate the rows and copy their waveforms into RAM.

        Parameters
        ----------
        pool : WaveformPool
            Verified CPC waveform pool.
        rows : Iterable[Mapping[str, Any]]
            Rows to cache, in the order exposed by ``signals``.
        max_bytes : int
            Upper bound on the copied waveform bytes.
        reserve_bytes : int
            Memory that must remain available after the copy.
        chunk_records : int
            Records copied per read.
        expected_source : str | None
            Required ``source`` of every row, if given.

        Raises
        ------
        ValueError
            If the rows or limits are invalid or disagree with the pool.
        MemoryError
            If the subset exceeds ``max_bytes`` or available memory.
        """
        rows = list(rows)
        if not rows:
            raise ValueError("At least one waveform row is required")
        if max_bytes <= 0 or reserve_bytes < 0 or chunk_records <= 0:
            raise ValueError("Invalid memory or chunk limit")
        source = pool.signals
        if source.dtype != np.float32 or source.ndim != 3 or source.shape[1:] != WAVEFORM_SHAPE:
            raise ValueError("Expected float32 CPC waveforms with shape [N,12,2500]")
        row_ids = [str(row["ecg_id"]) for row in rows]
        if len(set(row_ids)) != len(row_ids):
            raise ValueError("Duplicate ECG IDs in requested rows")
        source_indices = np.asarray(pool.indices(rows), dtype=np.int64)
        if np.any(source_indices < 0) or np.any(source_indices >= len(source)):
            raise ValueError("Source index outside waveform cache")
        _check_rows(pool, rows, source_indices, expected_source)
        required = len(rows) * int(np.prod(source.shape[1:])) * source.dtype.itemsize
        _check_memory(required, max_bytes, reserve_bytes)

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

    def indices(self, rows: Iterable[Mapping[str, Any]]) -> list[int]:
        """
        Map rows to their positions in ``signals``.

        Parameters
        ----------
        rows : Iterable[Mapping[str, Any]]
            Rows that are part of this cache.

        Returns
        -------
        list[int]
            Cache positions in ``rows`` order.
        """
        return [self.index[str(row["ecg_id"])] for row in rows]
