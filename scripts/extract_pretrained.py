#!/usr/bin/env python3
"""Extract frozen ECG foundation-model features for the labeled PTB-XL splits.

Run this from the separate Python 3.11 ``.venv-pretrained`` environment. Each
record contributes two nonoverlapping five-second views; their embeddings are
averaged to one row. Input files are streamed, never held as one ECG array.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import resource
import subprocess
import time
from pathlib import Path

import numpy as np


LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
SPLITS = ("labeled_train", "validation", "test")
SAMPLE_RATE = 500
SAMPLES_PER_VIEW = 2500
HF_MODELS = {
    "hubert-small": ("Edoardo-Coppola/hubert-ecg-small", "model.safetensors"),
    "ecg-fm": ("wanglab/ecg-fm", "mimic_iv_ecg_physionet_pretrained.pt"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_rows(manifest_dir: Path, limit: int | None) -> tuple[list[dict[str, str]], dict[str, str]]:
    rows: list[dict[str, str]] = []
    hashes = {}
    seen = set()
    for split in SPLITS:
        path = manifest_dir / f"{split}.csv"
        hashes[path.name] = sha256(path)
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not {"ecg_id", "filename_hr"}.issubset(reader.fieldnames or []):
                raise ValueError(f"{path} needs ecg_id and filename_hr columns")
            for row in reader:
                ecg_id = row["ecg_id"].strip()
                if ecg_id in seen:
                    raise ValueError(f"Duplicate ecg_id across manifests: {ecg_id}")
                seen.add(ecg_id)
                rows.append({"ecg_id": ecg_id, "filename_hr": row["filename_hr"].strip(), "split": split})
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("No records selected")
    return rows, hashes


def read_record(raw_dir: Path, relative_name: str) -> np.ndarray:
    import wfdb

    path = (raw_dir / relative_name).resolve()
    if not path.is_relative_to(raw_dir.resolve()):
        raise ValueError(f"Waveform path escapes raw directory: {relative_name}")
    record = wfdb.rdrecord(str(path))
    if record.fs != SAMPLE_RATE:
        raise ValueError(f"Expected 500 Hz, got {record.fs} at {path}")
    names = tuple(record.sig_name)
    upper_names = tuple(name.upper() for name in names)
    upper_leads = tuple(name.upper() for name in LEADS)
    if set(upper_names) != set(upper_leads) or len(names) != 12:
        raise ValueError(f"Unexpected lead names at {path}: {names}")
    signal = np.asarray(record.p_signal, dtype=np.float32)
    if signal.shape[0] != 5000:
        raise ValueError(f"Expected a 10-second ECG at {path}; got {signal.shape}")
    signal = signal[:, [upper_names.index(name) for name in upper_leads]].T
    if not np.isfinite(signal).all():
        raise ValueError(f"Nonfinite waveform at {path}")
    return signal


def preprocess_hubert(signal: np.ndarray) -> np.ndarray:
    """Official FIR/min-max pipeline followed by the official flattened decimation."""
    from scipy import signal as scipy_signal
    from hubert_ecg.utils import ecg_preprocessing

    normalized = ecg_preprocessing(signal, original_frequency=SAMPLE_RATE)
    views = []
    for start in (0, SAMPLES_PER_VIEW):
        view = normalized[:, start : start + SAMPLES_PER_VIEW]
        # ECGDataset flattens the 12 leads before decimating by five.
        views.append(scipy_signal.decimate(view.reshape(-1), 5).astype(np.float32))
    return np.stack(views)


def preprocess_ecg_fm(signal: np.ndarray) -> np.ndarray:
    """Official 500 Hz lead-wise z-score, then nonoverlapping 5-second views."""
    mean = signal.mean(axis=1, keepdims=True)
    std = signal.std(axis=1, keepdims=True)
    standardized = np.divide(signal - mean, std, out=np.zeros_like(signal), where=std > 0)
    return np.stack((standardized[:, :SAMPLES_PER_VIEW], standardized[:, SAMPLES_PER_VIEW:]))


def checkpoint_info(model_name: str) -> tuple[Path | None, dict]:
    from huggingface_hub import HfApi, hf_hub_download

    repo, filename = HF_MODELS[model_name]
    revision = HfApi().model_info(repo).sha
    local_dir = Path(__file__).resolve().parents[1] / "third_party" / "checkpoints" / model_name
    checkpoint = hf_hub_download(repo_id=repo, filename=filename, revision=revision,
                                 local_dir=local_dir)
    if model_name == "hubert-small":
        hf_hub_download(repo_id=repo, filename="config.json", revision=revision,
                        local_dir=local_dir)
    return Path(checkpoint), {"repo": repo, "revision": revision, "filename": filename,
                              "sha256": sha256(Path(checkpoint))}


def load_model(model_name: str, checkpoint: Path | None, device: str):
    import torch

    if model_name == "hubert-small":
        import hubert_ecg  # noqa: F401 - registers the Hugging Face model type
        from transformers import AutoModel

        # Use the downloaded snapshot cache. The official model implementation
        # registers its custom architecture when hubert_ecg is imported.
        assert checkpoint is not None
        model = AutoModel.from_pretrained(str(checkpoint.parent), local_files_only=True)
        preprocess = preprocess_hubert
    else:
        from fairseq_signals.models import build_model_from_checkpoint

        assert checkpoint is not None
        model = build_model_from_checkpoint(str(checkpoint))
        preprocess = preprocess_ecg_fm
    model.eval().to(device)
    return model, preprocess


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


def git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


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
