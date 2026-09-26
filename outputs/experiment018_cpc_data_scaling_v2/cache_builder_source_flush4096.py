"""Materialize the verified sampled cohort as a local 250 Hz CPC read cache."""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ecg_experiment.cpc_input_audit import historical_resample
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.sampled_training_dataset import SampledTrainingECGDataset

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/processed/sampled_100k_plus_labels_v1"
OUTPUT = ROOT / "data/processed/sampled_100k_cpc_cache_v1"
SIGNAL_SHAPE = (12, 2500)
CHECKPOINT_RECORDS = 4096


def build(output: Path, workers: int) -> None:
    """Write source-verified resampled signals in manifest row order."""
    dataset = SampledTrainingECGDataset(SOURCE, purpose="ssl")
    output.mkdir(parents=True, exist_ok=True)
    final = output / "signals.npy"
    partial = output / "signals.partial.npy"
    progress_path = output / "progress.json"
    complete_path = output / "complete.json"
    if complete_path.exists():
        raise FileExistsError(f"Cache already complete: {complete_path}")
    identity = {
        "source_metadata_sha256": sha256_file(SOURCE / "metadata.json"),
        "source_manifest_sha256": sha256_file(SOURCE / "train_manifest.csv"),
        "resample_source_sha256": sha256_file(ROOT / "ecg_experiment/cpc_input_audit.py"),
        "loader_source_sha256": sha256_file(ROOT / "ecg_experiment/sampled_training_dataset.py"),
        "signal_contract_source_sha256": sha256_file(ROOT / "ecg_experiment/training_contracts.py"),
        "record_count": len(dataset),
    }
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress["identity"] != identity:
            raise ValueError("Cache progress identity mismatch")
        done = progress["done"]
        constant_lead_mimic = progress["constant_lead_mimic"]
        previous_elapsed = progress.get(
            "elapsed_total_seconds", progress.get("elapsed_seconds_since_resume", 0),
        )
        signals = np.load(partial, mmap_mode="r+")
    else:
        done = 0
        constant_lead_mimic = 0
        previous_elapsed = 0.0
        signals = np.lib.format.open_memmap(
            partial, mode="w+", dtype=np.float32,
            shape=(len(dataset), *SIGNAL_SHAPE),
        )
        write_json_atomic(progress_path, {
            "identity": identity, "done": 0, "constant_lead_mimic": 0,
        })
    if signals.shape != (len(dataset), *SIGNAL_SHAPE) or signals.dtype != np.float32:
        raise ValueError("Cache shape or dtype mismatch")
    start = time.monotonic()
    # Group shard reads to avoid repeatedly opening the same large source file.
    order = sorted(range(len(dataset)),
                   key=lambda index: (dataset.rows[index]["backend"],
                                      dataset.rows[index]["path"]))
    def converted(index: int) -> tuple[int, np.ndarray, bool]:
        """Read and transform one source-verified ECG."""
        row = dataset[index]
        constant = row["source"] == "mimic" and np.any(np.ptp(row["signal"], axis=1) == 0)
        return index, historical_resample(row["signal"]), constant

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for chunk_start in range(done, len(dataset), 256):
            positions = range(chunk_start, min(chunk_start + 256, len(dataset)))
            indices = [order[position] for position in positions]
            # WFDB files are independent; shards use the dataset's mutable
            # verified-mmap cache and therefore stay sequential.
            is_mimic = all(dataset.rows[index]["backend"] == "mimic_wfdb"
                           for index in indices)
            rows = executor.map(converted, indices) if is_mimic else map(converted, indices)
            for position, (index, waveform, constant) in zip(positions, rows, strict=True):
                constant_lead_mimic += int(constant)
                signals[index] = waveform
                if (position + 1) % CHECKPOINT_RECORDS == 0 or position + 1 == len(dataset):
                    signals.flush()
                    write_json_atomic(progress_path, {
                        "identity": identity, "done": position + 1,
                        "constant_lead_mimic": constant_lead_mimic,
                        "elapsed_total_seconds": previous_elapsed + time.monotonic() - start,
                    })
                    print(f"cached {position + 1}/{len(dataset)} ECGs", flush=True)
    del signals
    os.replace(partial, final)
    write_json_atomic(complete_path, {
        "identity": identity,
        "signals_sha256": sha256_file(final),
        "signals_shape": [len(dataset), *SIGNAL_SHAPE],
        "dtype": "float32",
        "constant_lead_mimic": constant_lead_mimic,
        "elapsed_total_seconds": previous_elapsed + time.monotonic() - start,
    })
    progress_path.unlink()


def main() -> None:
    """Build a local derived cache without changing raw or source manifests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    build(args.output, args.workers)


if __name__ == "__main__":
    main()
