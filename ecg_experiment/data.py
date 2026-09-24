"""Waveform caching and manifest-only access; no diagnostic metadata is loaded."""

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def read_manifest(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def build_cache(raw_dir, manifest_dir, cache_dir):
    """Read all training and eligible evaluation ECGs at 100 Hz into a physical-mV cache."""
    import wfdb

    raw_dir, manifest_dir, cache_dir = map(Path, (raw_dir, manifest_dir, cache_dir))
    records = {}
    for name in ("all_train_ssl.csv", "validation.csv", "test.csv"):
        for row in read_manifest(manifest_dir / name):
            records[int(row["ecg_id"])] = row
    ids = np.array(sorted(records), dtype=np.int64)
    if (cache_dir / "complete.json").exists():
        existing = np.load(cache_dir / "ecg_ids.npy")
        if not np.array_equal(existing, ids):
            raise ValueError("Cached records differ from the requested manifests")
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    x = np.lib.format.open_memmap(cache_dir / "signals.npy", mode="w+", dtype="float32",
                                 shape=(len(ids), 12, 1000))
    for index, ecg_id in enumerate(ids):
        signal, header = wfdb.rdsamp(str(raw_dir / records[int(ecg_id)]["filename_lr"]))
        names = [name.upper() for name in header["sig_name"]]
        if header["fs"] != 100 or signal.shape != (1000, 12):
            raise ValueError(f"Unexpected waveform dimensions for ECG {ecg_id}")
        if any(unit.lower() != "mv" for unit in header["units"]):
            raise ValueError(f"Unexpected waveform units for ECG {ecg_id}")
        ordered = signal[:, [names.index(lead) for lead in LEADS]].T
        if not np.isfinite(ordered).all():
            raise ValueError(f"Nonfinite waveform for ECG {ecg_id}")
        x[index] = ordered
        if (index + 1) % 2000 == 0:
            print(f"Cached {index + 1}/{len(ids)} ECGs", flush=True)
    x.flush()
    np.save(cache_dir / "ecg_ids.npy", ids)
    (cache_dir / "complete.json").write_text(json.dumps({
        "records": len(ids), "shape": list(x.shape), "sampling_rate": 100,
        "lead_order": LEADS, "units": "mV", "dtype": "float32",
    }, indent=2))


class Waveforms:
    def __init__(self, cache_dir):
        self.x = np.load(Path(cache_dir) / "signals.npy", mmap_mode="r")
        ids = np.load(Path(cache_dir) / "ecg_ids.npy")
        self.index = {int(ecg_id): i for i, ecg_id in enumerate(ids)}

    def indices(self, rows):
        return np.array([self.index[int(row["ecg_id"])] for row in rows])

    def training_scale(self, train_rows):
        # Fit only on training signals. Demean each recording; retain relative patient amplitudes.
        total = np.zeros(12, dtype=np.float64)
        indices = self.indices(train_rows)
        for start in range(0, len(indices), 256):
            batch = np.array(self.x[indices[start:start + 256]], dtype=np.float64)
            batch -= batch.mean(axis=-1, keepdims=True)
            total += np.square(batch).sum(axis=(0, 2))
        return np.maximum(np.sqrt(total / (len(indices) * self.x.shape[-1])), 1e-6).astype("float32")


class ECGDataset(Dataset):
    def __init__(self, waveforms, rows, scale):
        self.waveforms, self.rows, self.scale = waveforms, rows, scale
        self.indices = waveforms.indices(rows)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        signal = np.array(self.waveforms.x[self.indices[i]], dtype=np.float32, copy=True)
        signal -= signal.mean(axis=-1, keepdims=True)
        signal /= self.scale[:, None]
        signal = np.clip(signal, -20, 20)
        target = float(self.rows[i].get("target", -1))
        return torch.from_numpy(signal), torch.tensor(target, dtype=torch.float32)
