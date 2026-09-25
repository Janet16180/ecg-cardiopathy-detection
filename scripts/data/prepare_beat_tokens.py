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
from typing import Any

import numpy as np

from ecg_experiment.ecg_tokenizers import GRID_STEPS, MAX_EVENTS, beat_metadata
from ecg_experiment.files import sha256_file, write_text_atomic

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / "data/processed/cpc_pool_40k"
DEFAULT_OUTPUT = ROOT / "data/processed/cpc_beats_40k"
DETECTOR_CODE = ROOT / "ecg_experiment/ecg_tokenizers.py"
CPC_SAMPLES = 2500
BATCH_ROWS = 100
PROGRESS_INTERVAL = 1000
MAX_WORKERS = 8
ARRAYS = {
    "boundaries": (np.bool_, (2, GRID_STEPS)),
    "forced": (np.bool_, (2, GRID_STEPS)),
    "peaks": (np.int16, (2, MAX_EVENTS)),
    "confirmations": (np.int16, (2, MAX_EVENTS)),
    "counts": (np.uint8, (2,)),
}


def _load_cache(cache_dir: Path) -> tuple[dict[str, Any], np.ndarray, int]:
    """Open the CPC waveform cache and check it against its completion record."""
    source = json.loads((cache_dir / "complete.json").read_text())
    signals = np.load(cache_dir / "signals.npy", mmap_mode="r", allow_pickle=False)
    count = len(np.load(cache_dir / "ecg_ids.npy", allow_pickle=False))
    if signals.shape != (count, 12, CPC_SAMPLES) or signals.dtype != np.float32:
        raise ValueError("CPC cache must contain [N,12,2500] float32 signals")
    if source["record_count"] != count or source["shape"] != list(signals.shape):
        raise ValueError("CPC cache completion metadata differs from waveform array")
    if sha256_file(cache_dir / "ecg_ids.npy") != source["ecg_ids_sha256"]:
        raise ValueError("CPC cache ECG ID hash mismatch")
    return source, signals, count


def _identity(cache_dir: Path, source: dict[str, Any], count: int) -> dict[str, Any]:
    """Describe the inputs and detector settings that the output depends on."""
    return {
        "cache_complete_sha256": sha256_file(cache_dir / "complete.json"),
        "cache_signals_sha256": source["signals_sha256"],
        "cache_ecg_ids_sha256": source["ecg_ids_sha256"],
        "detector_code_sha256": sha256_file(DETECTOR_CODE),
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


def _verified_completion(output_dir: Path, identity: dict[str, Any]) -> dict[str, Any]:
    """Return an existing completion record after checking its identity and hashes."""
    info = json.loads((output_dir / "complete.json").read_text())
    if info["identity"] != identity:
        raise ValueError("Existing beat metadata uses different source or detector")
    for filename, expected in info["sha256"].items():
        if sha256_file(output_dir / filename) != expected:
            raise ValueError(f"Completed beat metadata hash mismatch: {filename}")
    return info


def _copy_ids(cache_dir: Path, output_dir: Path, expected_sha256: str) -> None:
    """Place the cache's ECG ID array next to the beat arrays."""
    id_path = output_dir / "ecg_ids.npy"
    if not id_path.exists():
        temporary = output_dir / "ecg_ids.partial.npy"
        shutil.copyfile(cache_dir / "ecg_ids.npy", temporary)
        os.replace(temporary, id_path)
    if sha256_file(id_path) != expected_sha256:
        raise ValueError("Beat metadata ECG IDs differ from cache")


def _write_progress(path: Path, identity: dict[str, Any], completed_rows: int) -> None:
    write_text_atomic(path, json.dumps({"identity": identity, "completed_rows": completed_rows}) + "\n")


def _resume_position(output_dir: Path, identity: dict[str, Any], count: int) -> int:
    """Return completed rows from a matching checkpoint, or start a new one."""
    progress_path = output_dir / "progress.json"
    if not progress_path.exists():
        for name in ARRAYS:
            (output_dir / f"{name}.partial.npy").unlink(missing_ok=True)
        _write_progress(progress_path, identity, 0)
        return 0
    progress = json.loads(progress_path.read_text())
    if progress["identity"] != identity:
        raise ValueError("Beat metadata checkpoint differs from inputs")
    done = progress["completed_rows"]
    if not 0 <= done <= count:
        raise ValueError("Invalid beat metadata checkpoint position")
    return done


def _open_partial_arrays(output_dir: Path, count: int, done: int) -> dict[str, np.memmap]:
    """Create fresh partial arrays, or reopen checkpointed ones for writing."""
    matrices = {}
    for name, (dtype, suffix) in ARRAYS.items():
        partial = output_dir / f"{name}.partial.npy"
        if done == 0:
            matrices[name] = np.lib.format.open_memmap(partial, mode="w+", dtype=dtype,
                                                       shape=(count, *suffix))
            continue
        matrices[name] = np.lib.format.open_memmap(partial, mode="r+")
        if matrices[name].shape != (count, *suffix) or matrices[name].dtype != dtype:
            raise ValueError(f"Invalid beat metadata checkpoint array: {name}")
    return matrices


def _compute(signals: np.ndarray, output_dir: Path, identity: dict[str, Any], done: int,
             workers: int) -> None:
    """Fill the partial arrays from ``done`` onward, checkpointing every batch."""
    count = len(signals)
    matrices = _open_partial_arrays(output_dir, count, done)
    progress_path = output_dir / "progress.json"
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(done, count, BATCH_ROWS):
            stop = min(start + BATCH_ROWS, count)
            futures = [pool.submit(beat_metadata, np.asarray(signals[index]))
                       for index in range(start, stop)]
            for index, future in enumerate(futures, start):
                for name, value in zip(ARRAYS, future.result(), strict=True):
                    matrices[name][index] = value
            for matrix in matrices.values():
                matrix.flush()
            _write_progress(progress_path, identity, stop)
            if stop % PROGRESS_INTERVAL == 0 or stop == count:
                print(f"Beat metadata {stop:,}/{count:,} ECGs; "
                      f"{time.monotonic() - started:.1f}s this run", flush=True)
    for matrix in matrices.values():
        matrix.flush()


def _publish_arrays(output_dir: Path) -> None:
    """Rename partial arrays into place.

    A crash during these renames is recoverable: completed_rows=count and each
    file is accepted in either partial or final form on the next run.
    """
    for name in ARRAYS:
        partial = output_dir / f"{name}.partial.npy"
        final = output_dir / f"{name}.npy"
        if partial.exists():
            os.replace(partial, final)
        if not final.exists():
            raise ValueError(f"Missing completed beat metadata array: {name}")


def _finalize(output_dir: Path, identity: dict[str, Any], count: int) -> dict[str, Any]:
    """Check the completed arrays, then write the completion record."""
    boundaries = np.load(output_dir / "boundaries.npy", mmap_mode="r")
    forced = np.load(output_dir / "forced.npy", mmap_mode="r")
    counts = np.load(output_dir / "counts.npy", mmap_mode="r")
    if not np.all(boundaries[:, :, -1]) or np.any(forced & ~boundaries):
        raise ValueError("Beat metadata lacks half-end or has impossible forced flags")
    beat_histogram = Counter(int(value) for value in counts.reshape(-1))
    if sha256_file(DETECTOR_CODE) != identity["detector_code_sha256"]:
        raise ValueError("Detector code changed during beat metadata preparation")
    info = {
        "identity": identity, "record_count": count,
        "beat_count_histogram_per_half": {str(key): value for key, value in sorted(beat_histogram.items())},
        "half_count": count * 2,
        "mean_boundaries_per_half": float(boundaries.sum() / (count * 2)),
        "mean_forced_per_half": float(forced.sum() / (count * 2)),
        "low_event_halves_lt2": int((counts < 2).sum()),
        "sha256": {name: sha256_file(output_dir / name) for name in
                   ("ecg_ids.npy", *(f"{name}.npy" for name in ARRAYS))},
    }
    write_text_atomic(output_dir / "complete.json", json.dumps(info, indent=2, allow_nan=False) + "\n")
    (output_dir / "progress.json").unlink(missing_ok=True)
    return info


def prepare(cache_dir: Path, output_dir: Path, workers: int = 2) -> dict[str, Any]:
    """
    Compute causal beat boundaries for every CPC pool ECG, resuming if interrupted.

    Parameters
    ----------
    cache_dir : Path
        Completed CPC waveform cache with ``signals.npy`` and ``ecg_ids.npy``.
    output_dir : Path
        Destination; a completed output is verified and returned unchanged.
    workers : int
        Detector threads, from 1 to 8.

    Returns
    -------
    dict[str, Any]
        Completion record with identity, summary statistics and file hashes.

    Raises
    ------
    ValueError
        If inputs, a checkpoint or a completed output do not match.
    """
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError("workers must be in [1,8]")
    cache_dir = cache_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source, signals, count = _load_cache(cache_dir)
    identity = _identity(cache_dir, source, count)
    if (output_dir / "complete.json").exists():
        return _verified_completion(output_dir, identity)
    _copy_ids(cache_dir, output_dir, source["ecg_ids_sha256"])
    done = _resume_position(output_dir, identity, count)
    if done < count:
        _compute(signals, output_dir, identity, done, workers)
    _publish_arrays(output_dir)
    return _finalize(output_dir, identity, count)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse the command line.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or ``None`` for ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """
    Prepare beat metadata from the command line and print the completion record.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or ``None`` for ``sys.argv``.
    """
    args = parse_args(argv)
    result = prepare(args.cache_dir, args.output_dir, args.workers)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
