#!/usr/bin/env python3
"""Extract frozen ECG-JEPA and xECG features for every ECG of the Avance 4 notebook.

The manifest step writes ``labeled_train.csv`` (folds 1-8), ``validation.csv`` (fold 9) and
``test.csv`` (fold 10) from the Avance 2 table, so the same files also feed
``scripts/features/extract_ecg_cpc.py``. Features are written as ``features.npy``,
``ecg_ids.npy`` and ``metadata.json``, the layout of the other extractors.

Examples
--------
.venv/bin/python -m scripts.features.extract_avance4_encoders manifest
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=third_party/jepa-deps \
    .venv/bin/python -m scripts.features.extract_avance4_encoders extract --encoder ecg_jepa
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ecg_experiment import external_encoders as ext
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.paths import to_stored
from ecg_experiment.provenance import git_head
from ecg_experiment.waveforms import manifest_rows

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "outputs/avance2_fusion/ptbxl_caracteristicas_seleccionadas.csv"
DATABASE = ROOT / "data/raw/ptb-xl/1.0.3/ptbxl_database.csv"
OUT = ROOT / "outputs/avance4_modelos"
MANIFEST_DIR = OUT / "manifest"
SPLIT_FILES = {"train": "labeled_train.csv", "val": "validation.csv", "test": "test.csv"}
CHUNK = 256


def write_manifest(table: Path, database: Path, manifest_dir: Path) -> dict[str, int]:
    """
    Write one manifest per Avance split with the ECG ID and 500 Hz file name.

    Parameters
    ----------
    table : Path
        Avance 2 feature table, indexed by ``ecg_id`` with a ``split`` column.
    database : Path
        Official ``ptbxl_database.csv``.
    manifest_dir : Path
        Output directory.

    Returns
    -------
    dict[str, int]
        Number of ECGs written per split.
    """
    splits = pd.read_csv(table, index_col="ecg_id", usecols=["ecg_id", "split"]).split
    names = pd.read_csv(database, index_col="ecg_id", usecols=["ecg_id", "filename_hr"]).filename_hr
    manifest_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, file_name in SPLIT_FILES.items():
        ids = splits.index[splits == split].sort_values()
        with (manifest_dir / file_name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["ecg_id", "filename_hr"])
            writer.writerows((int(i), names.loc[i]) for i in ids)
        counts[split] = len(ids)
    return counts


def encoder_parts(encoder: str) -> tuple:
    """
    Loader, input builder, feature function, checkpoint files and source of one encoder.

    Parameters
    ----------
    encoder : str
        ``"ecg_jepa"`` or ``"xecg"``.

    Returns
    -------
    tuple
        ``(load, make_input, features, checkpoints, source_dir)``.
    """
    if encoder == "ecg_jepa":
        return ext.load_jepa, ext.ptb_jepa_input, ext.jepa_features, [ext.JEPA_CHECKPOINT], ext.JEPA_SOURCE
    checkpoints = sorted(p for p in ext.DEFAULT_CHECKPOINT_DIR.glob("*") if p.is_file())
    source = ext.DEFAULT_CHECKPOINT_DIR
    return ext.load_xecg_backbone, ext.ptb_xecg_input, ext.xecg_features, checkpoints, source


def extract(encoder: str, manifest_dir: Path, output_dir: Path, limit: int | None) -> dict:
    """
    Embed every manifest record with a frozen encoder and save the features.

    Parameters
    ----------
    encoder : str
        ``"ecg_jepa"`` or ``"xecg"``.
    manifest_dir : Path
        Directory written by ``write_manifest``.
    output_dir : Path
        Where ``features.npy``, ``ecg_ids.npy`` and ``metadata.json`` go.
    limit : int | None
        First records only, for a smoke test.

    Returns
    -------
    dict
        The metadata written next to the features.
    """
    load, make_input, features_of, checkpoints, source = encoder_parts(encoder)
    rows, manifest_hashes = manifest_rows(manifest_dir, limit)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with gpu_lock("cuda"):
        model = load()
        chunks = []
        for first in range(0, len(rows), CHUNK):
            batch = rows[first:first + CHUNK]
            inputs = np.stack([make_input(row["filename_hr"].removesuffix(".dat")) for row in batch])
            chunks.append(features_of(model, inputs))
            print(f"{encoder}: {first + len(batch)}/{len(rows)} ECG", flush=True)
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
    features = np.concatenate(chunks)
    np.save(output_dir / "features.npy", features)
    np.save(output_dir / "ecg_ids.npy", np.asarray([int(row["ecg_id"]) for row in rows]))
    metadata = {
        "encoder": encoder, "records": len(rows), "dimension": int(features.shape[1]),
        "splits": {s: sum(r["split"] == s for r in rows) for s in ("labeled_train", "validation", "test")},
        "manifest_sha256": manifest_hashes,
        "checkpoint_sha256": {to_stored(p): sha256_file(p) for p in checkpoints},
        "source": to_stored(source), "git_head": git_head(ROOT),
        "seconds": round(time.perf_counter() - start, 1),
        "peak_gpu_gb": round(peak_gb, 2), "gpu": torch.cuda.get_device_name(),
    }
    write_json_atomic(output_dir / "metadata.json", metadata)
    return metadata


def main() -> None:
    """Parse arguments and run one step."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("manifest", "extract"))
    parser.add_argument("--encoder", choices=("ecg_jepa", "xecg"))
    parser.add_argument("--manifest-dir", type=Path, default=MANIFEST_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.stage == "manifest":
        print(write_manifest(TABLE, DATABASE, args.manifest_dir))
        return
    if args.encoder is None:
        parser.error("extract needs --encoder")
    output_dir = args.output_dir or OUT / "features" / args.encoder
    print(extract(args.encoder, args.manifest_dir, output_dir, args.limit))


if __name__ == "__main__":
    main()
