"""Shared pytest fixtures with synthetic data; no private ECG data is required."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

POOL_FIELDS = ("ecg_id", "patient_id", "source", "split", "target")


def write_synthetic_cpc_pool(directory: Path, records: int = 600, seed: int = 0) -> Path:
    """
    Write a small cache with the layout expected by the CPC ``Pool``.

    Parameters
    ----------
    directory : Path
        Destination directory; it is created if missing.
    records : int
        Number of ECG records to generate.
    seed : int
        Seed for the synthetic waveforms.

    Returns
    -------
    Path
        The cache directory.
    """
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    signals = rng.standard_normal((records, 12, 2500), dtype=np.float32)
    ecg_ids = np.arange(1000, 1000 + records, dtype=np.int64)
    splits = ["train"] * (records - 40) + ["validation"] * 20 + ["test"] * 20
    np.save(directory / "signals.npy", signals)
    np.save(directory / "ecg_ids.npy", ecg_ids)
    with (directory / "rows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=POOL_FIELDS)
        writer.writeheader()
        for index, (ecg_id, split) in enumerate(zip(ecg_ids, splits, strict=True)):
            writer.writerow({"ecg_id": str(ecg_id), "patient_id": f"p{index // 2}",
                             "source": "ptbxl", "split": split, "target": index % 2})
    (directory / "complete.json").write_text(json.dumps({"records": records}))
    return directory


@pytest.fixture
def synthetic_cpc_pool(tmp_path: Path) -> Path:
    """Directory holding a synthetic CPC waveform cache."""
    return write_synthetic_cpc_pool(tmp_path / "cpc_pool")
