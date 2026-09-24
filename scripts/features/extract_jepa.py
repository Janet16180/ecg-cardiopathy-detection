#!/usr/bin/env python3
"""Extract frozen features from the published ECG-JEPA multiblock encoder.

Uses the official author's encoder and checkpoint, with the same PTB-XL waveform
processing as their linear evaluation: 500 Hz, Fourier resampling to 2,500
samples, and leads I, II, V1–V6. Run from the repository's main .venv.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

from scripts.extract_pretrained import git_head, manifest_rows, read_record, sha256


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "third_party" / "ECG_JEPA"
DEFAULT_CHECKPOINT = ROOT / "third_party" / "checkpoints" / "ecg-jepa" / "multiblock_epoch100.pth"
LEAD_INDICES = (0, 1, 6, 7, 8, 9, 10, 11)
SPLITS = ("labeled_train", "validation", "test")
CHECKPOINT_URL = "https://drive.google.com/file/d/1gMOT4xjQQg0GZkY1iE6NuDzua4ALw00l/view"


def preprocess(signal: np.ndarray) -> np.ndarray:
    from scipy.signal import resample

    if signal.shape != (12, 5000):
        raise ValueError(f"Expected 12 leads x 5,000 samples, got {signal.shape}")
    # Lead selection and Fourier resampling commute; this matches waves_ptbxl.
    reduced = resample(signal[list(LEAD_INDICES)], 2500, axis=1).astype(np.float32)
    if not np.isfinite(reduced).all():
        raise ValueError("Nonfinite resampled waveform")
    return reduced


def load_encoder(checkpoint: Path, device: str):
    if not SOURCE_DIR.is_dir():
        raise FileNotFoundError(f"Official ECG-JEPA source missing: {SOURCE_DIR}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Official ECG-JEPA checkpoint missing: {checkpoint}")
    sys.path.insert(0, str(SOURCE_DIR))
    from models import load_encoder as official_load_encoder

    encoder, dimension = official_load_encoder(str(checkpoint))
    encoder.eval().to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder, dimension


def extract(args: argparse.Namespace) -> dict:
    import timm
    import torch

    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see a CUDA device")
    rows, manifest_hashes = manifest_rows(args.manifest_dir, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_features = args.output_dir / "features.npy"
    final_ids = args.output_dir / "ecg_ids.npy"
    final_meta = args.output_dir / "metadata.json"
    partial = args.output_dir / "features.partial.npy"
    if any(path.exists() for path in (final_features, final_ids, final_meta, partial)):
        raise FileExistsError(f"Output files already exist in {args.output_dir}")

    checkpoint_hash = sha256(args.checkpoint)
    encoder, dimension = load_encoder(args.checkpoint, args.device)
    started = time.monotonic()
    matrix = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32,
                                       shape=(len(rows), dimension))
    try:
        for index in range(0, len(rows), args.batch_size):
            batch = rows[index:index + args.batch_size]
            signals = np.stack([preprocess(read_record(args.raw_dir, row["filename_hr"]))
                                for row in batch])
            tensor = torch.from_numpy(signals).to(args.device)
            with torch.inference_mode():
                vectors = encoder.representation(tensor).cpu().numpy()
            if vectors.shape != (len(batch), dimension) or not np.isfinite(vectors).all():
                raise ValueError(f"Invalid encoder output at record {index}: {vectors.shape}")
            matrix[index:index + len(batch)] = vectors
            completed = index + len(batch)
            if completed % 100 < args.batch_size or completed == len(rows):
                print(f"ecg-jepa-multiblock: {completed}/{len(rows)} records, "
                      f"{time.monotonic() - started:.1f}s", flush=True)
        matrix.flush()
        del matrix
        matrix = None
        np.save(final_ids, np.asarray([row["ecg_id"] for row in rows], dtype=str), allow_pickle=False)
        os.replace(partial, final_features)
        elapsed = time.monotonic() - started
        metadata = {
            "model": "ecg-jepa-multiblock",
            "record_count": len(rows),
            "feature_dimension": dimension,
            "features_file": final_features.name,
            "ecg_ids_file": final_ids.name,
            "manifest_sha256": manifest_hashes,
            "checkpoint": {"source_url": CHECKPOINT_URL, "filename": args.checkpoint.name,
                           "sha256": checkpoint_hash, "epoch": 99},
            "input": "standard 12-lead PTB-XL, 500 Hz, 10 seconds",
            "preprocessing": "official PTB-XL path: scipy.signal.resample to 2500 samples, then leads I, II, V1-V6; no amplitude normalization",
            "pooling": "official encoder.representation mean across 400 lead-patch tokens",
            "elapsed_seconds": elapsed,
            "records_per_second": len(rows) / elapsed,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "torch_version": torch.__version__,
            "timm_version": timm.__version__,
            "device": args.device,
            "cuda_device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
            "source_commit": git_head(SOURCE_DIR),
            "split_counts": {split: sum(row["split"] == split for row in rows) for split in SPLITS},
        }
        final_meta.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return metadata
    finally:
        if matrix is not None:
            del matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--limit", type=int, help="First N records for a smoke test")
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1 or (args.limit is not None and args.limit < 1):
        parser.error("batch-size, threads, and limit must be positive")
    print(json.dumps(extract(args), indent=2))


if __name__ == "__main__":
    main()
