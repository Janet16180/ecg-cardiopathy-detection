#!/usr/bin/env python3
"""Merge disjoint frozen-feature extractions in the order of a union manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


SPLITS = ("labeled_train", "validation", "test")
COMPATIBILITY_KEYS = ("model", "feature_dimension", "checkpoint", "input", "preprocessing", "pooling", "source_commit")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_source(path: Path):
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--extra", type=Path, required=True)
    parser.add_argument("--union-manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    base_ids, base_features, base_metadata, base_metadata_path = read_source(args.base)
    extra_ids, extra_features, extra_metadata, extra_metadata_path = read_source(args.extra)
    for key in COMPATIBILITY_KEYS:
        if base_metadata[key] != extra_metadata[key]:
            raise ValueError(f"Source metadata mismatch for {key}")
    if set(base_ids.tolist()) & set(extra_ids.tolist()):
        raise ValueError("Base and extra ECG IDs overlap")

    union_ids = []
    manifest_hashes = {}
    for split in SPLITS:
        path = args.union_manifest_dir / f"{split}.csv"
        manifest_hashes[path.name] = sha256(path)
        with path.open(newline="", encoding="utf-8") as stream:
            union_ids.extend(row["ecg_id"].strip() for row in csv.DictReader(stream))
    if len(set(union_ids)) != len(union_ids):
        raise ValueError("Duplicate ECG IDs in union manifests")

    locations = {ecg_id: (base_features, index) for index, ecg_id in enumerate(base_ids.tolist())}
    locations.update({ecg_id: (extra_features, index) for index, ecg_id in enumerate(extra_ids.tolist())})
    if set(union_ids) != set(locations):
        raise ValueError(f"Union IDs and source IDs differ: missing {len(set(union_ids) - set(locations))}, "
                         f"unexpected {len(set(locations) - set(union_ids))}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_features = args.output_dir / "features.npy"
    output_ids = args.output_dir / "ecg_ids.npy"
    output_metadata = args.output_dir / "metadata.json"
    if any(path.exists() for path in (output_features, output_ids, output_metadata)):
        raise FileExistsError(f"Merge output already exists in {args.output_dir}")
    merged = np.lib.format.open_memmap(output_features, mode="w+", dtype=np.float32,
                                       shape=(len(union_ids), base_metadata["feature_dimension"]))
    for index, ecg_id in enumerate(union_ids):
        features, source_index = locations[ecg_id]
        merged[index] = features[source_index]
    merged.flush()
    del merged
    np.save(output_ids, np.asarray(union_ids, dtype=str), allow_pickle=False)
    metadata = {key: base_metadata[key] for key in COMPATIBILITY_KEYS}
    metadata.update({
        "record_count": len(union_ids),
        "ecg_ids_file": output_ids.name,
        "features_file": output_features.name,
        "manifest_sha256": manifest_hashes,
        "sources": [
            {"directory": str(args.base.resolve()), "record_count": len(base_ids),
             "metadata_sha256": sha256(base_metadata_path)},
            {"directory": str(args.extra.resolve()), "record_count": len(extra_ids),
             "metadata_sha256": sha256(extra_metadata_path)},
        ],
        "features_sha256": sha256(output_features),
        "ecg_ids_sha256": sha256(output_ids),
    })
    output_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
