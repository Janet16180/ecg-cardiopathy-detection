"""Waveform caching and manifest-only access; no diagnostic metadata is loaded."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .files import read_csv, write_text_atomic
from .waveforms import LEADS as WAVEFORM_LEADS

# Uppercase names, in canonical order, as recorded in the 100 Hz cache metadata.
LEADS = [lead.upper() for lead in WAVEFORM_LEADS]
CACHE_MANIFESTS = ("all_train_ssl.csv", "validation.csv", "test.csv")
SAMPLE_RATE = 100
SAMPLES = 1000
PROGRESS_INTERVAL = 2000
# Fixed chunking keeps the float64 accumulation order, and so the scale, reproducible.
SCALE_CHUNK_RECORDS = 256
MIN_SCALE = 1e-6
CLIP_LIMIT = 20


def _cached_records(manifest_dir: Path) -> dict[int, dict[str, str]]:
    """Collect the requested records, keyed by their numeric ECG identity."""
    records = {}
    for name in CACHE_MANIFESTS:
        for row in read_csv(manifest_dir / name):
            records[int(row["ecg_id"])] = row
    return records


def _read_cache_signal(raw_dir: Path, row: dict[str, str], ecg_id: int) -> np.ndarray:
    """Read one 100 Hz WFDB record as lead-ordered ``[12, 1000]`` physical mV."""
    import wfdb

    signal, header = wfdb.rdsamp(str(raw_dir / row["filename_lr"]))
    names = [name.upper() for name in header["sig_name"]]

    if header["fs"] != SAMPLE_RATE or signal.shape != (SAMPLES, len(LEADS)):
        raise ValueError(f"Unexpected waveform dimensions for ECG {ecg_id}")
    if any(unit.lower() != "mv" for unit in header["units"]):
        raise ValueError(f"Unexpected waveform units for ECG {ecg_id}")
    lead_indices = [names.index(lead) for lead in LEADS]
    ordered = signal[:, lead_indices].T
    if not np.isfinite(ordered).all():
        raise ValueError(f"Nonfinite waveform for ECG {ecg_id}")
    return ordered


def build_cache(raw_dir: str | Path, manifest_dir: str | Path, cache_dir: str | Path) -> None:
    """
    Read all training and eligible evaluation ECGs at 100 Hz into a physical-mV cache.

    An existing complete cache is reused when it holds exactly the requested
    records.

    Parameters
    ----------
    raw_dir : str | Path
        PTB-XL release root.
    manifest_dir : str | Path
        Directory holding the SSL training, validation and test manifests.
    cache_dir : str | Path
        Destination of ``signals.npy``, ``ecg_ids.npy`` and ``complete.json``.

    Raises
    ------
    ValueError
        If an existing cache holds other records, or a waveform breaks the
        rate, shape, unit or finiteness contract.
    """
    raw_dir = Path(raw_dir)
    manifest_dir = Path(manifest_dir)
    cache_dir = Path(cache_dir)

    records = _cached_records(manifest_dir)
    ids = np.array(sorted(records), dtype=np.int64)

    if (cache_dir / "complete.json").exists():
        existing = np.load(cache_dir / "ecg_ids.npy")
        if not np.array_equal(existing, ids):
            raise ValueError("Cached records differ from the requested manifests")
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    signals = np.lib.format.open_memmap(
        cache_dir / "signals.npy", mode="w+", dtype="float32", shape=(len(ids), len(LEADS), SAMPLES)
    )
    for index, ecg_id in enumerate(ids):
        signals[index] = _read_cache_signal(raw_dir, records[int(ecg_id)], ecg_id)
        if (index + 1) % PROGRESS_INTERVAL == 0:
            print(f"Cached {index + 1}/{len(ids)} ECGs", flush=True)
    signals.flush()
    np.save(cache_dir / "ecg_ids.npy", ids)

    metadata = {
        "records": len(ids),
        "shape": list(signals.shape),
        "sampling_rate": SAMPLE_RATE,
        "lead_order": LEADS,
        "units": "mV",
        "dtype": "float32",
    }
    write_text_atomic(cache_dir / "complete.json", json.dumps(metadata, indent=2))


class Waveforms:
    """
    Memory-mapped 100 Hz waveform cache addressed by ECG ID.

    Parameters
    ----------
    cache_dir : str | Path
        Cache written by ``build_cache``.
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self.x = np.load(Path(cache_dir) / "signals.npy", mmap_mode="r")
        ids = np.load(Path(cache_dir) / "ecg_ids.npy")
        self.index = {int(ecg_id): i for i, ecg_id in enumerate(ids)}

    def indices(self, rows: list[dict[str, str]]) -> np.ndarray:
        """
        Map manifest rows to cache positions.

        Parameters
        ----------
        rows : list[dict[str, str]]
            Rows whose ``ecg_id`` is in the cache.

        Returns
        -------
        np.ndarray
            Positions in ``x``, in row order.
        """
        return np.array([self.index[int(row["ecg_id"])] for row in rows])

    def training_scale(self, train_rows: list[dict[str, str]]) -> np.ndarray:
        """
        Compute per-lead RMS amplitude over demeaned training recordings.

        Only training signals are used. Each recording is demeaned, but
        relative amplitudes between patients are retained.

        Parameters
        ----------
        train_rows : list[dict[str, str]]
            Training manifest rows.

        Returns
        -------
        np.ndarray
            Float32 per-lead scale, at least ``MIN_SCALE``.
        """
        total = np.zeros(len(LEADS), dtype=np.float64)
        indices = self.indices(train_rows)
        for start in range(0, len(indices), SCALE_CHUNK_RECORDS):
            batch_indices = indices[start : start + SCALE_CHUNK_RECORDS]
            batch = np.array(self.x[batch_indices], dtype=np.float64)
            batch -= batch.mean(axis=-1, keepdims=True)
            total += np.square(batch).sum(axis=(0, 2))

        rms = np.sqrt(total / (len(indices) * self.x.shape[-1]))
        return np.maximum(rms, MIN_SCALE).astype("float32")


class ECGDataset(Dataset):
    """
    Demeaned, scaled and clipped cache waveforms with binary targets.

    Parameters
    ----------
    waveforms : Waveforms
        Source cache.
    rows : list[dict[str, str]]
        Rows to serve; ``target`` defaults to -1 when absent.
    scale : np.ndarray
        Per-lead scale from ``Waveforms.training_scale``.
    """

    def __init__(self, waveforms: Waveforms, rows: list[dict[str, str]], scale: np.ndarray) -> None:
        self.waveforms = waveforms
        self.rows = rows
        self.scale = scale
        self.indices = waveforms.indices(rows)

    def __len__(self) -> int:
        """Return the number of records."""
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the normalized ``[12, 1000]`` signal and float32 target of record ``i``."""
        signal = np.array(self.waveforms.x[self.indices[i]], dtype=np.float32, copy=True)
        signal -= signal.mean(axis=-1, keepdims=True)
        signal /= self.scale[:, None]
        signal = np.clip(signal, -CLIP_LIMIT, CLIP_LIMIT)

        target = float(self.rows[i].get("target", -1))
        return torch.from_numpy(signal), torch.tensor(target, dtype=torch.float32)
