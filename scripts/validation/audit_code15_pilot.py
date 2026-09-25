#!/usr/bin/env python3
"""Inspect a fixed sample from the checksum-verified CODE-15% part 0 archive."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from ecg_experiment.code15 import TRACE_SHAPE, extracted_hdf5, verified_files, verify_file
from ecg_experiment.provenance import utc_now

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "data/raw/code-15pct/zenodo-4916206/exams_part0.zip"
RECEIPT = ROOT / "data/acquisition/code_15pct.json"
OUTPUT = ROOT / "outputs/data_quality/code15_part0.json"
SEED = 42
SAMPLE_SIZE = 64
NEAR_FLAT_STD = 0.01


def sample_counts(signal: h5py.Dataset, selected: list[int]) -> tuple[Counter[str], Counter[tuple[int, int]]]:
    """
    Tally shape, finiteness, zero padding and near-flat leads over sampled traces.

    Parameters
    ----------
    signal : h5py.Dataset
        Native ``[N, 4096, 12]`` traces.
    selected : list[int]
        Trace indices to inspect.

    Returns
    -------
    tuple[Counter[str], Counter[tuple[int, int]]]
        Screening counts and exact-zero ``(left, right)`` edge pairs.
    """
    counts = Counter()
    pads = Counter()
    for index in selected:
        ecg = np.asarray(signal[index])
        counts["read"] += 1
        counts["shape_4096x12"] += ecg.shape == TRACE_SHAPE
        finite = bool(np.isfinite(ecg).all())
        counts["finite"] += finite
        if not finite or ecg.shape != TRACE_SHAPE:
            continue
        active = np.flatnonzero(np.any(ecg != 0, axis=1))
        if len(active):
            left, right = int(active[0]), int(ecg.shape[0] - 1 - active[-1])
            pads[(left, right)] += 1
        else:
            counts["all_zero"] += 1
        counts["any_near_flat_lead"] += bool(np.any(ecg.std(axis=0) < NEAR_FLAT_STD))
    return counts, pads


def audit() -> dict[str, Any]:
    """
    Verify the archive checksum and screen a seeded sample of its traces.

    Returns
    -------
    dict[str, Any]
        Report written to ``OUTPUT``.

    Raises
    ------
    ValueError
        If the archive lacks a receipt or fails its official checksum.
    """
    expected = verified_files(RECEIPT).get(ARCHIVE.name)
    if expected is None:
        raise ValueError("CODE part 0 lacks an official checksum receipt")
    verify_file(ARCHIVE, expected)
    with extracted_hdf5(ARCHIVE, "exams_part0.hdf5") as handle:
        keys = list(handle.keys())
        signal_key = "tracings" if "tracings" in handle else "signal"
        signal = handle[signal_key]
        selected = random.Random(SEED).sample(range(signal.shape[0]), min(SAMPLE_SIZE, signal.shape[0]))
        counts, pads = sample_counts(signal, selected)
        return {"generated_at_utc": utc_now(),
                "source_archive": str(ARCHIVE.relative_to(ROOT)),
                "hdf5_keys": keys, "signal_key": signal_key,
                "signal_shape": list(signal.shape), "signal_dtype": str(signal.dtype),
                "sample_size": len(selected), "seed": SEED,
                "counts": dict(counts),
                "zero_padding_counts": {f"{left},{right}": count
                                        for (left, right), count in sorted(pads.items())},
                "near_flat_rule": "screening flag only; std < 0.01 in native stored units"}


def main() -> None:
    """Write the pilot report and print its path."""
    result = audit()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
