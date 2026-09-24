#!/usr/bin/env python3
"""Extract frozen ECG foundation-model features for the labeled PTB-XL splits.

Run this from the separate Python 3.11 ``.venv-pretrained`` environment. Each
record contributes two nonoverlapping five-second views; their embeddings are
averaged to one row. Input files are streamed, never held as one ECG array.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import numpy as np

from ecg_experiment.foundation_models import HF_MODELS, checkpoint_info, load_model
from ecg_experiment.provenance import git_head
from ecg_experiment.waveforms import SPLITS, manifest_rows, read_record


def embed_views(model_name: str, model, views: np.ndarray, device: str) -> np.ndarray:
    import torch

    source = torch.from_numpy(np.ascontiguousarray(views)).float().to(device)
    with torch.inference_mode():
        if model_name == "hubert-small":
            tokens = model(input_values=source).last_hidden_state
        else:
            result = model.extract_features(source=source, padding_mask=None, mask=False)
            tokens = result["x"]
        if tokens.ndim != 3 or tokens.shape[0] != views.shape[0]:
            raise ValueError(f"Unexpected token shape: {tuple(tokens.shape)}")
        embedded = tokens.mean(dim=1).cpu().numpy()
    if not np.isfinite(embedded).all():
        raise ValueError("Model returned nonfinite embeddings")
    return embedded


def extract(args: argparse.Namespace) -> dict:
    import torch

    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot see a CUDA device")
    rows, manifest_hashes = manifest_rows(args.manifest_dir, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_features = args.output_dir / "features.npy"
    final_ids = args.output_dir / "ecg_ids.npy"
    final_meta = args.output_dir / "metadata.json"
    if any(path.exists() for path in (final_features, final_ids, final_meta)):
        raise FileExistsError(f"Existing features, IDs, or metadata in {args.output_dir}")
    checkpoint, checkpoint_metadata = checkpoint_info(args.model)
    model, preprocess = load_model(args.model, checkpoint, args.device)
    started = time.monotonic()
    partial = args.output_dir / "features.partial.npy"
    if partial.exists():
        raise FileExistsError(f"Previous partial extraction exists: {partial}")
    matrix = None
    try:
        for index in range(0, len(rows), args.batch_size):
            batch = rows[index : index + args.batch_size]
            views = np.concatenate([preprocess(read_record(args.raw_dir, row["filename_hr"]))
                                    for row in batch], axis=0)
            embeddings = embed_views(args.model, model, views, args.device).reshape(len(batch), 2, -1).mean(axis=1)
            if matrix is None:
                matrix = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32,
                                                   shape=(len(rows), embeddings.shape[1]))
            matrix[index : index + len(batch)] = embeddings
            if (index + len(batch)) % 100 < args.batch_size or index + len(batch) == len(rows):
                print(f"{args.model}: {index + len(batch)}/{len(rows)} records, "
                      f"{time.monotonic() - started:.1f}s", flush=True)
        assert matrix is not None
        matrix.flush()
        del matrix
        matrix = None
        ids = np.asarray([row["ecg_id"] for row in rows], dtype=str)
        np.save(final_ids, ids, allow_pickle=False)
        os.replace(partial, final_features)
        elapsed = time.monotonic() - started
        metadata = {
            "model": args.model,
            "record_count": len(rows),
            "feature_dimension": int(embeddings.shape[1]),
            "ecg_ids_file": final_ids.name,
            "features_file": final_features.name,
            "manifest_sha256": manifest_hashes,
            "checkpoint": checkpoint_metadata,
            "input": "standard 12-lead PTB-XL, 500 Hz, 10 seconds",
            "preprocessing": ("official HuBERT-ECG FIR 0.05-47 Hz, per-lead min-max [-1,1], "
                              "two 5-second views flattened lead-major then decimated by 5"
                              if args.model == "hubert-small" else
                              "ECG-FM per-lead z-score across 10 seconds, then two 5-second views"),
            "pooling": "mean across model tokens per view, then mean across two views",
            "elapsed_seconds": elapsed,
            "records_per_second": len(rows) / elapsed,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "torch_version": torch.__version__,
            "device": args.device,
            "cuda_device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
            "source_commit": git_head(Path(__file__).resolve().parents[1] / "third_party" /
                                      ("HuBERT-ECG" if args.model == "hubert-small" else "fairseq-signals")),
            "split_counts": {split: sum(row["split"] == split for row in rows) for split in SPLITS},
        }
        final_meta.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return metadata
    finally:
        if matrix is not None:
            del matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(HF_MODELS), required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2, help="ECG records per forward pass")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--limit", type=int, help="First N records for a timing/shape smoke test")
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1 or (args.limit is not None and args.limit < 1):
        parser.error("batch-size, threads, and limit must be positive")
    print(json.dumps(extract(args), indent=2))


if __name__ == "__main__":
    main()
