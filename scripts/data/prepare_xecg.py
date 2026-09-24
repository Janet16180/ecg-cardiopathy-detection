#!/usr/bin/env python3
"""Cache exact official xECG 100 Hz PTB-XL views for downstream training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from ecg_experiment.files import sha256_file
from ecg_experiment.xecg import LEADS, preprocess_xecg

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_FILES = ("labeled_train.csv", "validation.csv", "test.csv")
VIEW_SHAPE = (1000, 12)
PROGRESS_INTERVAL = 500


def read_record_float64(raw_dir: Path, relative_name: str) -> np.ndarray:
    """
    Read a PTB-XL record in float64 physical mV, in official xECG lead order.

    Unlike ``ecg_experiment.waveforms.read_record`` this keeps WFDB's float64
    samples, which the xECG preprocessing was validated against.

    Parameters
    ----------
    raw_dir : Path
        PTB-XL release root that the record must stay inside.
    relative_name : str
        Record path relative to ``raw_dir``, without extension.

    Returns
    -------
    np.ndarray
        Float64 ``[12, 5000]`` signal.

    Raises
    ------
    ValueError
        If the path escapes ``raw_dir`` or the record is not ten seconds of
        twelve-lead 500 Hz physical mV.
    """
    import wfdb

    base = (Path(raw_dir) / relative_name).resolve()
    if not base.is_relative_to(Path(raw_dir).resolve()):
        raise ValueError(f"Record escapes raw directory: {relative_name}")
    signal, fields = wfdb.rdsamp(str(base))
    names = tuple(name.upper() for name in fields["sig_name"])
    expected = tuple(name.upper() for name in LEADS)
    if len(names) != 12 or len(set(names)) != 12 or set(names) != set(expected):
        raise ValueError(f"Unexpected WFDB leads for {relative_name}: {fields['sig_name']}")
    if fields["fs"] != 500 or signal.shape != (5000, 12):
        raise ValueError(f"Expected 10 seconds at 500 Hz for {relative_name}")
    if fields.get("units") != ["mV"] * 12:
        raise ValueError(f"Expected physical mV in all leads for {relative_name}: {fields.get('units')}")
    return signal[:, [names.index(name) for name in expected]].T


def _manifest_rows(manifest_dir: Path) -> list[dict[str, str]]:
    rows = []
    for name in MANIFEST_FILES:
        with (manifest_dir / name).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not {"ecg_id", "filename_hr", "target"}.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing fields in {name}")
            rows.extend(reader)
    ids = [row["ecg_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate ECG IDs across xECG cache manifests")
    return rows


def _raw_source_digest(raw_dir: Path, rows: list[dict[str, str]]) -> str:
    """Fingerprint every WFDB header and waveform used by the cache."""
    digest = hashlib.sha256()
    root = raw_dir.resolve()
    for row in rows:
        relative = Path(row["filename_hr"])
        for suffix in (".hea", ".dat"):
            path = (root / relative).with_suffix(suffix).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise FileNotFoundError(f"Missing or unsafe WFDB source: {path}")
            digest.update(str(path.relative_to(root)).encode())
            digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _check_existing(cache_dir: Path, requested: dict[str, Any]) -> None:
    """Accept an existing cache only when its fingerprint and array hash match."""
    array_path = cache_dir / "views.npy"
    meta_path = cache_dir / "metadata.json"
    if not array_path.is_file() or not meta_path.is_file():
        raise ValueError(f"Incomplete xECG cache in {cache_dir}")
    actual = json.loads(meta_path.read_text())
    if any(actual.get(key) != value for key, value in requested.items()):
        raise ValueError(f"xECG cache fingerprint differs: {cache_dir}")
    if actual.get("views_sha256") != sha256_file(array_path):
        raise ValueError(f"xECG cache array checksum differs: {array_path}")
    cached = np.load(array_path, mmap_mode="r")
    if cached.shape != tuple(requested["shape"]) or cached.dtype != np.float32:
        raise ValueError(f"Malformed xECG cache array: {array_path}")


def _fill_views(partial: Path, raw_dir: Path, rows: list[dict[str, str]]) -> None:
    """Write every preprocessed view into a new partial array."""
    matrix = np.lib.format.open_memmap(partial, mode="w+", dtype="float32", shape=(len(rows), *VIEW_SHAPE))
    for index, row in enumerate(rows):
        matrix[index] = preprocess_xecg(read_record_float64(raw_dir, row["filename_hr"]))
        if (index + 1) % PROGRESS_INTERVAL == 0 or index + 1 == len(rows):
            print(f"Cached xECG {index + 1}/{len(rows)}", flush=True)
    matrix.flush()


def cache_xecg_views(raw_dir: Path, manifest_dir: Path, cache_dir: Path) -> Path:
    """
    Write ``views.npy`` [N,1000,12] with fingerprinted metadata atomically.

    The metadata's ``ecg_ids`` list provides the row index mapping. Existing
    caches are accepted only when source and manifest fingerprints match.

    Parameters
    ----------
    raw_dir : Path
        PTB-XL release root.
    manifest_dir : Path
        Directory with the labeled, validation and test manifests.
    cache_dir : Path
        Cache directory.

    Returns
    -------
    Path
        Path of ``views.npy``.

    Raises
    ------
    ValueError
        If an existing cache is incomplete or does not match.
    FileExistsError
        If an interrupted partial array remains.
    """
    raw_dir, manifest_dir, cache_dir = map(Path, (raw_dir, manifest_dir, cache_dir))
    rows = _manifest_rows(manifest_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    array_path = cache_dir / "views.npy"
    meta_path = cache_dir / "metadata.json"
    partial = cache_dir / "views.partial.npy"
    requested = {
        "format_version": 1,
        "raw_dir": str(raw_dir.resolve()),
        "raw_source_sha256": _raw_source_digest(raw_dir, rows),
        "manifest_sha256": {name: sha256_file(manifest_dir / name) for name in MANIFEST_FILES},
        "preparation_sha256": sha256_file(Path(__file__)),
        "adapter_sha256": sha256_file(ROOT / "ecg_experiment/xecg.py"),
        "ecg_ids": [row["ecg_id"] for row in rows],
        "shape": [len(rows), *VIEW_SHAPE],
        "dtype": "float32",
        "preprocessing": ("physical mV; canonical 12 leads; 500-to-100 Hz scipy FFT resampling; "
                          "no cleaning or normalization"),
    }
    if array_path.exists() or meta_path.exists():
        _check_existing(cache_dir, requested)
        return array_path
    if partial.exists():
        raise FileExistsError(f"Incomplete prior cache: {partial}")
    start = time.monotonic()
    _fill_views(partial, raw_dir, rows)
    os.replace(partial, array_path)
    meta_path.write_text(json.dumps({**requested, "views_sha256": sha256_file(array_path),
                                     "seconds": time.monotonic() - start}, indent=2) + "\n")
    return array_path


def main() -> None:
    """Build or verify an xECG view cache from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/ptb-xl/1.0.3")
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    print(cache_xecg_views(args.raw_dir, args.manifest_dir, args.cache_dir))


if __name__ == "__main__":
    main()
