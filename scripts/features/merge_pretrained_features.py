#!/usr/bin/env python3
"""Merge disjoint frozen-feature extractions in the order of a union manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ecg_experiment.files import sha256_file, write_json_atomic

SPLITS = ("labeled_train", "validation", "test")
COMPATIBILITY_KEYS = ("model", "feature_dimension", "checkpoint", "input", "preprocessing", "pooling",
                      "source_commit")


def read_source(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any], Path]:
    """
    Open one feature extraction and check it.

    Parameters
    ----------
    path : Path
        Extraction directory with ``metadata.json``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict[str, Any], Path]
        ECG identifiers, memory-mapped features, metadata, and the metadata path.

    Raises
    ------
    ValueError
        If the features are misaligned, nonfinite, or have duplicate identifiers.
    """
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    ids = np.load(path / metadata["ecg_ids_file"], allow_pickle=False)
    features = np.load(path / metadata["features_file"], allow_pickle=False, mmap_mode="r")
    if features.shape != (len(ids), metadata["feature_dimension"]):
        raise ValueError(f"Feature shape mismatch in {path}")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError(f"Duplicate ECG ID within {path}")
    if not np.isfinite(features).all():
        raise ValueError(f"Nonfinite features in {path}")
    return ids, features, metadata, metadata_path


def union_ids(manifest_dir: Path) -> tuple[list[str], dict[str, str]]:
    """
    Read the union ECG identifiers in split order.

    Parameters
    ----------
    manifest_dir : Path
        Directory with the union split manifests.

    Returns
    -------
    tuple[list[str], dict[str, str]]
        Identifiers and the digest of each manifest.

    Raises
    ------
    ValueError
        If an identifier repeats.
    """
    ids: list[str] = []
    hashes = {}
    for split in SPLITS:
        path = manifest_dir / f"{split}.csv"
        hashes[path.name] = sha256_file(path)
        with path.open(newline="", encoding="utf-8") as stream:
            ids.extend(row["ecg_id"].strip() for row in csv.DictReader(stream))
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate ECG IDs in union manifests")
    return ids, hashes


def write_merged(path: Path, ids: list[str], locations: dict[str, tuple[np.ndarray, int]],
                 dimension: int) -> None:
    """
    Write features in union order into a new ``.npy`` file.

    Parameters
    ----------
    path : Path
        Destination file.
    ids : list[str]
        Union identifiers in output order.
    locations : dict[str, tuple[np.ndarray, int]]
        Source array and row of each identifier.
    dimension : int
        Feature dimension.
    """
    merged = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(ids), dimension))
    for index, ecg_id in enumerate(ids):
        features, source_index = locations[ecg_id]
        merged[index] = features[source_index]
    merged.flush()


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
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--extra", type=Path, required=True)
    parser.add_argument("--union-manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """
    Merge a base and an extra extraction and record their provenance.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or ``None`` for ``sys.argv``.

    Raises
    ------
    ValueError
        If the sources are incompatible, overlap, or do not cover the union.
    FileExistsError
        If merged outputs already exist.
    """
    args = parse_args(argv)

    base_ids, base_features, base_metadata, base_metadata_path = read_source(args.base)
    extra_ids, extra_features, extra_metadata, extra_metadata_path = read_source(args.extra)
    for key in COMPATIBILITY_KEYS:
        if base_metadata[key] != extra_metadata[key]:
            raise ValueError(f"Source metadata mismatch for {key}")
    if set(base_ids.tolist()) & set(extra_ids.tolist()):
        raise ValueError("Base and extra ECG IDs overlap")
    ids, manifest_hashes = union_ids(args.union_manifest_dir)

    locations = {ecg_id: (base_features, index) for index, ecg_id in enumerate(base_ids.tolist())}
    locations.update({ecg_id: (extra_features, index) for index, ecg_id in enumerate(extra_ids.tolist())})
    if set(ids) != set(locations):
        raise ValueError(f"Union IDs and source IDs differ: missing {len(set(ids) - set(locations))}, "
                         f"unexpected {len(set(locations) - set(ids))}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_features = args.output_dir / "features.npy"
    output_ids = args.output_dir / "ecg_ids.npy"
    output_metadata = args.output_dir / "metadata.json"
    if any(path.exists() for path in (output_features, output_ids, output_metadata)):
        raise FileExistsError(f"Merge output already exists in {args.output_dir}")
    write_merged(output_features, ids, locations, base_metadata["feature_dimension"])
    np.save(output_ids, np.asarray(ids, dtype=str), allow_pickle=False)
    metadata = {key: base_metadata[key] for key in COMPATIBILITY_KEYS}
    metadata.update({
        "record_count": len(ids),
        "ecg_ids_file": output_ids.name,
        "features_file": output_features.name,
        "manifest_sha256": manifest_hashes,
        "sources": [
            {"directory": str(args.base.resolve()), "record_count": len(base_ids),
             "metadata_sha256": sha256_file(base_metadata_path)},
            {"directory": str(args.extra.resolve()), "record_count": len(extra_ids),
             "metadata_sha256": sha256_file(extra_metadata_path)},
        ],
        "features_sha256": sha256_file(output_features),
        "ecg_ids_sha256": sha256_file(output_ids),
    })
    write_json_atomic(output_metadata, metadata, allow_nan=True)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
