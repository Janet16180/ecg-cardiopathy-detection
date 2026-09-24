#!/usr/bin/env python3
"""Prepare resumable, label-free causal beat boundaries for the CPC pool.

The detector emits only after a finite confirmation delay. Its thresholds use
previous samples, and long intervals are split by an online max-gap rule.
Output rows retain exact alignment with cpc_pool_40k/ecg_ids.npy.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ecg_experiment.ecg_tokenizers import (
    GRID_STEPS, MAX_EVENTS, beat_metadata, _sha256,
)
from scripts.prepare_mimic_ssl import atomic_text


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "data/processed/cpc_pool_40k"
DEFAULT_OUTPUT = ROOT / "data/processed/cpc_beats_40k"
ARRAYS = {
    "boundaries": (np.bool_, (2, GRID_STEPS)),
    "forced": (np.bool_, (2, GRID_STEPS)),
    "peaks": (np.int16, (2, MAX_EVENTS)),
    "confirmations": (np.int16, (2, MAX_EVENTS)),
    "counts": (np.uint8, (2,)),
}


def prepare(cache_dir: Path, output_dir: Path, workers: int = 2):
    if not 1 <= workers <= 8:
        raise ValueError("workers must be in [1,8]")
    cache_dir = cache_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source = json.loads((cache_dir / "complete.json").read_text())
    signals = np.load(cache_dir / "signals.npy", mmap_mode="r", allow_pickle=False)
    cache_ids = np.load(cache_dir / "ecg_ids.npy", allow_pickle=False)
    count = len(cache_ids)
    if signals.shape != (count, 12, 2500) or signals.dtype != np.float32:
        raise ValueError("CPC cache must contain [N,12,2500] float32 signals")
    if source["record_count"] != count or source["shape"] != list(signals.shape):
        raise ValueError("CPC cache completion metadata differs from waveform array")
    if _sha256(cache_dir / "ecg_ids.npy") != source["ecg_ids_sha256"]:
        raise ValueError("CPC cache ECG ID hash mismatch")
    identity = {
        "cache_complete_sha256": _sha256(cache_dir / "complete.json"),
        "cache_signals_sha256": source["signals_sha256"],
        "cache_ecg_ids_sha256": source["ecg_ids_sha256"],
        "detector_code_sha256": _sha256(ROOT / "ecg_experiment/ecg_tokenizers.py"),
        "record_count": count,
        "sample_rate_hz": 250,
        "grid_steps_per_half": GRID_STEPS,
        "detector": {
            "leads": ["II", "V2"], "causal_bandpass_hz": [5, 18],
            "bandpass_order": 2, "envelope": "maximum absolute two-lead filter output",
            "threshold_mV": "max(0.045, 2.0 * prior EWMA envelope)",
            "ewma_alpha": 0.01, "confirmation_delay_samples": 16,
            "refractory_samples": 75, "minimum_chunk_tokens": 4,
            "maximum_chunk_tokens": 24,
            "mapping": "ceil(confirmation_sample/16); never backdate to R peak",
            "fallback": "online forced boundary at 24 tokens, then half-end; no whole-half quality switch",
        },
    }
    complete_path = output_dir / "complete.json"
    if complete_path.exists():
        info = json.loads(complete_path.read_text())
        if info["identity"] != identity:
            raise ValueError("Existing beat metadata uses different source or detector")
        for filename, expected in info["sha256"].items():
            if _sha256(output_dir / filename) != expected:
                raise ValueError(f"Completed beat metadata hash mismatch: {filename}")
        return info
    id_path = output_dir / "ecg_ids.npy"
    if not id_path.exists():
        temporary = output_dir / "ecg_ids.partial.npy"
        shutil.copyfile(cache_dir / "ecg_ids.npy", temporary)
        os.replace(temporary, id_path)
    if _sha256(id_path) != source["ecg_ids_sha256"]:
        raise ValueError("Beat metadata ECG IDs differ from cache")
    progress_path = output_dir / "progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress["identity"] != identity:
            raise ValueError("Beat metadata checkpoint differs from inputs")
        done = progress["completed_rows"]
        if not 0 <= done <= count:
            raise ValueError("Invalid beat metadata checkpoint position")
    else:
        done = 0
        for name in ARRAYS:
            (output_dir / f"{name}.partial.npy").unlink(missing_ok=True)
        atomic_text(progress_path, json.dumps({"identity": identity, "completed_rows": 0}) + "\n")
    matrices = {}
    if done < count:
        for name, (dtype, suffix) in ARRAYS.items():
            partial = output_dir / f"{name}.partial.npy"
            if done == 0:
                matrices[name] = np.lib.format.open_memmap(
                    partial, mode="w+", dtype=dtype, shape=(count, *suffix))
            else:
                matrices[name] = np.lib.format.open_memmap(partial, mode="r+")
                if matrices[name].shape != (count, *suffix) or matrices[name].dtype != dtype:
                    raise ValueError(f"Invalid beat metadata checkpoint array: {name}")
    started = time.monotonic()
    if done < count:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for start in range(done, count, 100):
                stop = min(start + 100, count)
                futures = [pool.submit(beat_metadata, np.asarray(signals[index]))
                           for index in range(start, stop)]
                for index, future in enumerate(futures, start):
                    values = future.result()
                    for name, value in zip(ARRAYS, values):
                        matrices[name][index] = value
                for matrix in matrices.values():
                    matrix.flush()
                atomic_text(progress_path, json.dumps({"identity": identity,
                                                       "completed_rows": stop}) + "\n")
                if stop % 1000 == 0 or stop == count:
                    print(f"Beat metadata {stop:,}/{count:,} ECGs; "
                          f"{time.monotonic() - started:.1f}s this run", flush=True)
    for matrix in matrices.values():
        matrix.flush()
    matrices.clear()
    # A crash during these renames is recoverable: completed_rows=count and
    # each file is accepted in either partial or final form on the next run.
    for name in ARRAYS:
        partial = output_dir / f"{name}.partial.npy"
        final = output_dir / f"{name}.npy"
        if partial.exists():
            os.replace(partial, final)
        if not final.exists():
            raise ValueError(f"Missing completed beat metadata array: {name}")
    boundaries = np.load(output_dir / "boundaries.npy", mmap_mode="r")
    forced = np.load(output_dir / "forced.npy", mmap_mode="r")
    counts = np.load(output_dir / "counts.npy", mmap_mode="r")
    if not np.all(boundaries[:, :, -1]) or np.any(forced & ~boundaries):
        raise ValueError("Beat metadata lacks half-end or has impossible forced flags")
    beat_histogram = Counter(int(value) for value in counts.reshape(-1))
    if _sha256(ROOT / "ecg_experiment/ecg_tokenizers.py") != identity["detector_code_sha256"]:
        raise ValueError("Detector code changed during beat metadata preparation")
    info = {
        "identity": identity, "record_count": count,
        "beat_count_histogram_per_half": {str(key): value for key, value in sorted(beat_histogram.items())},
        "half_count": count * 2,
        "mean_boundaries_per_half": float(boundaries.sum() / (count * 2)),
        "mean_forced_per_half": float(forced.sum() / (count * 2)),
        "low_event_halves_lt2": int((counts < 2).sum()),
        "sha256": {name: _sha256(output_dir / name) for name in
                   ("ecg_ids.npy", *(f"{name}.npy" for name in ARRAYS))},
    }
    atomic_text(complete_path, json.dumps(info, indent=2, allow_nan=False) + "\n")
    progress_path.unlink(missing_ok=True)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    result = prepare(args.cache_dir, args.output_dir, args.workers)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
