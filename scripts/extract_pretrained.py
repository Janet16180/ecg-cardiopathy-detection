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
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ecg_experiment.foundation_models import HF_MODELS, checkpoint_info, load_model
from ecg_experiment.provenance import git_head
from ecg_experiment.waveforms import SPLITS, manifest_rows, read_record

ROOT = Path(__file__).resolve().parents[1]
VIEWS_PER_RECORD = 2
PROGRESS_INTERVAL = 100


def embed_views(model_name: str, model: Any, views: np.ndarray, device: str) -> np.ndarray:
    """
    Mean-pool the model's token embeddings for each view.

    Parameters
    ----------
    model_name : str
        ``"hubert-small"`` or an ECG-FM model name.
    model : Any
        Loaded backbone in evaluation mode.
    views : np.ndarray
        Preprocessed views, one per row.
    device : str
        Torch device for the forward pass.

    Returns
    -------
    np.ndarray
        One embedding per view.

    Raises
    ------
    ValueError
        If the tokens have an unexpected shape or the embeddings are not finite.
    """
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


def _write_features(args: argparse.Namespace, rows: list[dict[str, str]], model: Any,
                    preprocess: Callable[[np.ndarray], np.ndarray], partial: Path) -> int:
    """Stream every record through the model into ``partial``; return the feature size."""
    if not rows:
        raise ValueError("No records to extract")
    started = time.monotonic()
    matrix = None
    for index in range(0, len(rows), args.batch_size):
        batch = rows[index : index + args.batch_size]
        views = np.concatenate([preprocess(read_record(args.raw_dir, row["filename_hr"]))
                                for row in batch], axis=0)
        embeddings = embed_views(args.model, model, views, args.device)
        embeddings = embeddings.reshape(len(batch), VIEWS_PER_RECORD, -1).mean(axis=1)
        if matrix is None:
            # The feature size is known only after the first forward pass.
            matrix = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32,
                                               shape=(len(rows), embeddings.shape[1]))
        matrix[index : index + len(batch)] = embeddings
        if (index + len(batch)) % PROGRESS_INTERVAL < args.batch_size or index + len(batch) == len(rows):
            print(f"{args.model}: {index + len(batch)}/{len(rows)} records, "
                  f"{time.monotonic() - started:.1f}s", flush=True)
    matrix.flush()
    return int(matrix.shape[1])


def _preprocessing_description(model_name: str) -> str:
    if model_name == "hubert-small":
        return ("official HuBERT-ECG FIR 0.05-47 Hz, per-lead min-max [-1,1], "
                "two 5-second views flattened lead-major then decimated by 5")
    return "ECG-FM per-lead z-score across 10 seconds, then two 5-second views"


def extract(args: argparse.Namespace) -> dict[str, Any]:
    """
    Extract and save pooled features for every manifest record.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    dict[str, Any]
        Metadata written next to the features.

    Raises
    ------
    RuntimeError
        If CUDA was requested but is unavailable.
    FileExistsError
        If earlier outputs or a partial extraction exist.
    """
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
    feature_dimension = _write_features(args, rows, model, preprocess, partial)
    np.save(final_ids, np.asarray([row["ecg_id"] for row in rows], dtype=str), allow_pickle=False)
    os.replace(partial, final_features)
    elapsed = time.monotonic() - started
    cuda = args.device == "cuda"
    metadata = {
        "model": args.model,
        "record_count": len(rows),
        "feature_dimension": feature_dimension,
        "ecg_ids_file": final_ids.name,
        "features_file": final_features.name,
        "manifest_sha256": manifest_hashes,
        "checkpoint": checkpoint_metadata,
        "input": "standard 12-lead PTB-XL, 500 Hz, 10 seconds",
        "preprocessing": _preprocessing_description(args.model),
        "pooling": "mean across model tokens per view, then mean across two views",
        "elapsed_seconds": elapsed,
        "records_per_second": len(rows) / elapsed,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "torch_version": torch.__version__,
        "device": args.device,
        "cuda_device_name": torch.cuda.get_device_name(0) if cuda else None,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if cuda else None,
        "source_commit": git_head(ROOT / "third_party" /
                                  ("HuBERT-ECG" if args.model == "hubert-small" else "fairseq-signals")),
        "split_counts": {split: sum(row["split"] == split for row in rows) for split in SPLITS},
    }
    final_meta.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


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
    parser.add_argument("--model", choices=tuple(HF_MODELS), required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2, help="ECG records per forward pass")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--limit", type=int, help="First N records for a timing/shape smoke test")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.threads < 1 or (args.limit is not None and args.limit < 1):
        parser.error("batch-size, threads, and limit must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    """
    Parse arguments, extract features and print the metadata.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or ``None`` for ``sys.argv``.
    """
    args = parse_args(argv)
    print(json.dumps(extract(args), indent=2))


if __name__ == "__main__":
    main()
