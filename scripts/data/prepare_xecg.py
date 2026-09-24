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

import numpy as np

from ecg_experiment.xecg import LEADS, preprocess_xecg, sha256


def read_record(raw_dir: Path, relative_name: str) -> np.ndarray:
    """Read physical mV, reorder WFDB channels to official xECG lead order."""
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


def _manifest_rows(manifest_dir: Path):
    files = ("labeled_train.csv", "validation.csv", "test.csv")
    rows = []
    for name in files:
        with (manifest_dir / name).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not {"ecg_id", "filename_hr", "target"}.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing fields in {name}")
            rows.extend(reader)
    ids = [row["ecg_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate ECG IDs across xECG cache manifests")
    return files, rows


def _raw_source_digest(raw_dir: Path, rows: list[dict]) -> str:
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
            digest.update(bytes.fromhex(sha256(path)))
    return digest.hexdigest()


def cache_xecg_views(raw_dir: Path, manifest_dir: Path, cache_dir: Path) -> Path:
    """Write ``views.npy`` [N,1000,12] with fingerprinted metadata atomically.

    The metadata's ``ecg_ids`` list provides the row index mapping. Existing
    caches are accepted only when source and manifest fingerprints match.
    """
    raw_dir, manifest_dir, cache_dir = map(Path, (raw_dir, manifest_dir, cache_dir))
    files, rows = _manifest_rows(manifest_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    array_path = cache_dir / "views.npy"
    meta_path = cache_dir / "metadata.json"
    partial = cache_dir / "views.partial.npy"
    requested = {
        "format_version": 1,
        "raw_dir": str(raw_dir.resolve()),
        "raw_source_sha256": _raw_source_digest(raw_dir, rows),
        "manifest_sha256": {name: sha256(manifest_dir / name) for name in files},
        "preparation_sha256": sha256(Path(__file__)),
        "adapter_sha256": sha256(Path(__file__).resolve().parents[2] / "ecg_experiment/xecg.py"),
        "ecg_ids": [row["ecg_id"] for row in rows],
        "shape": [len(rows), 1000, 12],
        "dtype": "float32",
        "preprocessing": "physical mV; canonical 12 leads; 500-to-100 Hz scipy FFT resampling; no cleaning or normalization",
    }
    if array_path.exists() or meta_path.exists():
        if not array_path.is_file() or not meta_path.is_file():
            raise ValueError(f"Incomplete xECG cache in {cache_dir}")
        actual = json.loads(meta_path.read_text())
        if any(actual.get(key) != value for key, value in requested.items()):
            raise ValueError(f"xECG cache fingerprint differs: {cache_dir}")
        if actual.get("views_sha256") != sha256(array_path):
            raise ValueError(f"xECG cache array checksum differs: {array_path}")
        cached = np.load(array_path, mmap_mode="r")
        if cached.shape != tuple(requested["shape"]) or cached.dtype != np.float32:
            raise ValueError(f"Malformed xECG cache array: {array_path}")
        return array_path
    if partial.exists():
        raise FileExistsError(f"Incomplete prior cache: {partial}")
    start = time.monotonic()
    matrix = np.lib.format.open_memmap(partial, mode="w+", dtype="float32", shape=(len(rows), 1000, 12))
    try:
        for i, row in enumerate(rows):
            matrix[i] = preprocess_xecg(read_record(raw_dir, row["filename_hr"]))
            if (i + 1) % 500 == 0 or i + 1 == len(rows):
                print(f"Cached xECG {i + 1}/{len(rows)}", flush=True)
        matrix.flush()
        del matrix
        os.replace(partial, array_path)
        meta_path.write_text(json.dumps({**requested, "views_sha256": sha256(array_path),
                                         "seconds": time.monotonic() - start}, indent=2) + "\n")
    finally:
        if "matrix" in locals():
            del matrix
    return array_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/ptb-xl/1.0.3"))
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    print(cache_xecg_views(args.raw_dir, args.manifest_dir, args.cache_dir))


if __name__ == "__main__":
    main()
