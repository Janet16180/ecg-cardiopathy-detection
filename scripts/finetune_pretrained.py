#!/usr/bin/env python3
"""Fine-tune the official HuBERT-small or ECG-FM SSL backbone on PTB-XL.

Run with ``.venv-pretrained/bin/python -m scripts.finetune_pretrained``. The
published checkpoint is used as an encoder only: no disease-trained head is
loaded. Waveform preprocessing matches ``scripts.extract_pretrained``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ecg_experiment.run import evaluate_predictions, partition_validation, read_manifest
from scripts.extract_pretrained import (
    checkpoint_info, git_head, load_model, preprocess_ecg_fm, preprocess_hubert,
    read_record, sha256,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def manifest_data(manifest_dir: Path):
    names = ("labeled_train", "validation", "test")
    rows = {name: read_manifest(manifest_dir / f"{name}.csv") for name in names}
    hashes = {f"{name}.csv": sha256(manifest_dir / f"{name}.csv") for name in names}
    ids = [row["ecg_id"] for name in names for row in rows[name]]
    if len(ids) != len(set(ids)):
        raise ValueError("ECG IDs overlap across labeled train, validation, and test")
    for name in names:
        if not rows[name]:
            raise ValueError(f"{name} has no records")
    # Validate the training labels here; validation is checked only after its
    # development/calibration split. Leave locked test labels untouched.
    if set(row["target"] for row in rows["labeled_train"]) != {"0", "1"}:
        raise ValueError("labeled_train must contain both binary classes")
    return rows, hashes


def cache_views(model_name: str, raw_dir: Path, cache_dir: Path, rows, manifest_hashes):
    """Cache deterministic official preprocessing, keyed to source and manifests."""
    all_rows = [row for name in ("labeled_train", "validation", "test") for row in rows[name]]
    ids = [row["ecg_id"] for row in all_rows]
    preprocess = preprocess_hubert if model_name == "hubert-small" else preprocess_ecg_fm
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "metadata.json"
    array_path = cache_dir / "views.npy"
    source_hash = sha256(Path(__file__).with_name("extract_pretrained.py"))
    requested = {
        "model": model_name,
        "raw_dir": str(raw_dir.resolve()),
        "extractor_sha256": source_hash,
        "manifest_sha256": manifest_hashes,
        "ecg_ids": ids,
    }
    if meta_path.exists() or array_path.exists():
        if not meta_path.exists() or not array_path.exists():
            raise ValueError(f"Incomplete cache in {cache_dir}")
        saved = json.loads(meta_path.read_text())
        if any(saved.get(key) != value for key, value in requested.items()):
            raise ValueError(f"Preprocessing cache does not match the requested inputs: {cache_dir}")
        array = np.load(array_path, mmap_mode="r")
        expected = ((len(ids), 2, 6000) if model_name == "hubert-small"
                    else (len(ids), 2, 12, 2500))
        if array.shape != expected or array.dtype != np.float32:
            raise ValueError(f"Malformed preprocessing cache in {cache_dir}: {array.shape}")
        return array, {ecg_id: i for i, ecg_id in enumerate(ids)}, saved

    partial = cache_dir / "views.partial.npy"
    if partial.exists():
        raise FileExistsError(f"Previous incomplete cache exists: {partial}")
    shape = ((len(ids), 2, 6000) if model_name == "hubert-small"
             else (len(ids), 2, 12, 2500))
    matrix = np.lib.format.open_memmap(partial, mode="w+", dtype="float32", shape=shape)
    start = time.monotonic()
    try:
        for i, row in enumerate(all_rows):
            views = preprocess(read_record(raw_dir, row["filename_hr"]))
            if views.shape != shape[1:] or not np.isfinite(views).all():
                raise ValueError(f"Unexpected preprocessed views for ECG {row['ecg_id']}")
            matrix[i] = views
            if (i + 1) % 500 == 0 or i + 1 == len(ids):
                print(f"Cached {model_name}: {i + 1}/{len(ids)} ECGs", flush=True)
        matrix.flush()
        del matrix
        os.replace(partial, array_path)
        metadata = {**requested, "shape": list(shape), "dtype": "float32",
                    "seconds": time.monotonic() - start,
                    "preprocessing": ("Official HuBERT FIR/min-max, two 5-second flattened/decimated views"
                                      if model_name == "hubert-small" else
                                      "Official ECG-FM lead-wise z-score, two 5-second views")}
        meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    finally:
        if "matrix" in locals():
            del matrix
    return np.load(array_path, mmap_mode="r"), {ecg_id: i for i, ecg_id in enumerate(ids)}, metadata


class ViewDataset(Dataset):
    def __init__(self, views, index, rows):
        self.views, self.rows = views, rows
        self.indices = [index[row["ecg_id"]] for row in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        views = np.array(self.views[self.indices[i]], dtype=np.float32, copy=True)
        return torch.from_numpy(views), torch.tensor(float(self.rows[i]["target"]), dtype=torch.float32)


class FineTunedECG(nn.Module):
    def __init__(self, model_name: str, backbone: nn.Module, feature_dim: int):
        super().__init__()
        self.model_name = model_name
        self.backbone = backbone
        self.head = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, 1))

    def forward(self, views):
        batch, number_of_views = views.shape[:2]
        source = views.reshape(batch * number_of_views, *views.shape[2:])
        # Same token extraction and pooling as extract_pretrained.embed_views,
        # deliberately without torch.inference_mode() so gradients reach the backbone.
        if self.model_name == "hubert-small":
            tokens = self.backbone(input_values=source).last_hidden_state
        else:
            tokens = self.backbone.extract_features(source=source, padding_mask=None, mask=False)["x"]
        if tokens.ndim != 3 or tokens.shape[0] != batch * number_of_views:
            raise ValueError(f"Unexpected token shape: {tuple(tokens.shape)}")
        features = tokens.mean(dim=1).reshape(batch, number_of_views, -1).mean(dim=1)
        return self.head(features).squeeze(-1)


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    result = []
    for views, _ in loader:
        result.append(model(views.to(device)).cpu().numpy())
    return np.concatenate(result)


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def save_resume_checkpoint(path, fingerprint, model, optimizer, best_state,
                           best_epoch, best_auc, history, elapsed_seconds):
    """Commit a complete epoch boundary without exposing a partial checkpoint."""
    numpy_state = np.random.get_state()
    checkpoint = {
        "version": 1, "fingerprint": fingerprint,
        "model": cpu_state(model), "optimizer": optimizer.state_dict(),
        "best_model": best_state, "best_epoch": best_epoch, "best_auc": best_auc,
        "history": history, "elapsed_seconds": elapsed_seconds,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python_rng": random.getstate(),
        "numpy_rng": (numpy_state[0], numpy_state[1].tolist(),
                      numpy_state[2], numpy_state[3], numpy_state[4]),
    }
    temporary = path.with_name(path.name + ".partial")
    try:
        torch.save(checkpoint, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_resume_checkpoint(path, fingerprint, model, optimizer):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("version") != 1 or checkpoint.get("fingerprint") != fingerprint:
        raise ValueError(f"Fine-tuning resume checkpoint configuration differs: {path}")
    history = checkpoint["history"]
    if (not isinstance(history, list) or not history
            or any(record.get("epoch") != index + 1 for index, record in enumerate(history))
            or checkpoint["best_epoch"] not in range(1, len(history) + 1)
            or checkpoint["best_model"] is None):
        raise ValueError(f"Malformed fine-tuning resume checkpoint: {path}")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["torch_rng"])
    if torch.cuda.is_available():
        if len(checkpoint["cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count changed since fine-tuning checkpoint")
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    random.setstate(checkpoint["python_rng"])
    numpy_state = checkpoint["numpy_rng"]
    np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32),
                         numpy_state[2], numpy_state[3], numpy_state[4]))
    return checkpoint


def check_adaptation_budget(metadata, history):
    """Accept complete fixed-epoch runs and exact-update bounded runs."""
    if "max_updates" in metadata:
        final = history[-1] if history else {}
        if (final.get("updates") != metadata["max_updates"]
                or final.get("seen_examples") != metadata["max_updates"] * metadata["batch_size"]):
            raise ValueError("Adaptation did not finish its exact optimizer-update budget")
    elif len(history) != metadata["epochs"]:
        raise ValueError("Adaptation did not finish its fixed epoch budget")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("hubert-small", "ecg-fm"), required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path,
                        help="Preprocessed views; default: data/processed/ptbxl/pretrained_views/<model>/<manifest hash>")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--adapted-backbone", type=Path,
                        help="Optional training-only ECG-FM continued-SSL checkpoint")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from the last complete fine-tuning epoch")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.threads) < 1:
        parser.error("epochs, patience, batch-size, and threads must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.adapted_backbone and args.model != "ecg-fm":
        parser.error("--adapted-backbone is supported for ECG-FM only")
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    rows, manifest_hashes = manifest_data(args.manifest_dir)
    development, calibration = partition_validation(rows["validation"])
    for name, subset in (("development", development), ("calibration", calibration)):
        if set(row["target"] for row in subset) != {"0", "1"}:
            raise ValueError(f"{name} split has only one class")
    cache_dir = args.cache_dir or (Path("data/processed/ptbxl/pretrained_views") / args.model /
                                   hashlib.sha256(json.dumps(manifest_hashes, sort_keys=True).encode()).hexdigest()[:16])
    views, index, cache_meta = cache_views(args.model, args.raw_dir, cache_dir, rows, manifest_hashes)
    checkpoint, checkpoint_meta = checkpoint_info(args.model)
    backbone, _ = load_model(args.model, checkpoint, args.device)
    adaptation_meta = None
    if args.adapted_backbone:
        adapted = torch.load(args.adapted_backbone, map_location="cpu", weights_only=True)
        adaptation_meta = adapted["metadata"]
        expected_ssl = read_manifest(args.manifest_dir / "all_train_ssl.csv")
        if (adaptation_meta["official_checkpoint"]["sha256"] != checkpoint_meta["sha256"]
                or adaptation_meta["training_ecg_ids"] != [r["ecg_id"] for r in expected_ssl]
                or adaptation_meta["manifest_sha256"]["all_train_ssl.csv"]
                != sha256(args.manifest_dir / "all_train_ssl.csv")):
            raise ValueError("Adaptation checkpoint source or training manifest differs")
        heldout_patients = {r["patient_id"] for name in ("validation", "test") for r in rows[name]}
        if heldout_patients & set(adaptation_meta["training_patient_ids"]):
            raise ValueError("Adaptation checkpoint encountered held-out patients")
        check_adaptation_budget(adaptation_meta, adapted["history"])
        backbone.load_state_dict(adapted["backbone"], strict=True)
        adaptation_meta = {**adaptation_meta, "checkpoint_sha256": sha256(args.adapted_backbone)}
        del adapted
    if args.model == "hubert-small":
        # The SSL model's time/feature masking is an upstream objective, not
        # supervised fine-tuning augmentation. Disable it explicitly.
        backbone.config.mask_time_prob = 0.0
        backbone.config.mask_feature_prob = 0.0
        backbone.config.apply_spec_augment = False
    first_view = torch.from_numpy(np.array(views[0], copy=True)).to(args.device)
    with torch.no_grad():
        if args.model == "hubert-small":
            dimension = backbone(input_values=first_view).last_hidden_state.shape[-1]
        else:
            dimension = backbone.extract_features(source=first_view, padding_mask=None, mask=False)["x"].shape[-1]
    model = FineTunedECG(args.model, backbone, int(dimension)).to(args.device)
    train_loader = DataLoader(ViewDataset(views, index, rows["labeled_train"]),
                              batch_size=args.batch_size, shuffle=True, num_workers=0)
    dev_loader = DataLoader(ViewDataset(views, index, development),
                            batch_size=args.batch_size, num_workers=0)
    calibration_loader = DataLoader(ViewDataset(views, index, calibration),
                                    batch_size=args.batch_size, num_workers=0)
    test_loader = DataLoader(ViewDataset(views, index, rows["test"]),
                             batch_size=args.batch_size, num_workers=0)
    development_y = np.array([int(row["target"]) for row in development])
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 1e-5},
        {"params": model.head.parameters(), "lr": 1e-3},
    ], weight_decay=0.01)
    if adaptation_meta and adaptation_meta.get("external_ssl"):
        run_name = f"{args.model}_pooled_adapted_finetuned"
    else:
        run_name = f"{args.model}_adapted_finetuned" if args.adapted_backbone else f"{args.model}_finetuned"
    directory = args.output_dir / f"{run_name}_seed{args.seed}"
    resume_path = directory / "resume.pt"
    if args.resume:
        if (directory / "config.json").is_file() and (directory / "metrics.json").is_file():
            raise ValueError(f"Fine-tuning output is already complete: {directory}")
        if not resume_path.is_file():
            raise FileNotFoundError(f"Fine-tuning resume checkpoint is absent: {resume_path}")
    elif directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    fingerprint = {
        "model": args.model, "seed": args.seed, "epochs": args.epochs,
        "patience": args.patience, "batch_size": args.batch_size,
        "bootstrap": args.bootstrap, "device": args.device,
        "manifest_sha256": manifest_hashes,
        "raw_dir": str(args.raw_dir.resolve()),
        "cache_dir": str(cache_dir.resolve()),
        "official_checkpoint_sha256": checkpoint_meta["sha256"],
        "adaptation_checkpoint_sha256": (adaptation_meta["checkpoint_sha256"]
                                         if adaptation_meta else None),
        "extractor_source_sha256": cache_meta["extractor_sha256"],
        "finetune_source_sha256": sha256(Path(__file__)),
    }
    best_auc, best_epoch, best_state = -1.0, 0, None
    history = []
    started = time.monotonic()
    if args.resume:
        saved = load_resume_checkpoint(resume_path, fingerprint, model, optimizer)
        best_auc, best_epoch, best_state = (saved["best_auc"], saved["best_epoch"],
                                             saved["best_model"])
        history = saved["history"]
        started -= saved["elapsed_seconds"]
        del saved
        print(json.dumps({"stage": "pretrained_finetune_resumed", "model": args.model,
                          "completed_epochs": len(history), "best_epoch": best_epoch}), flush=True)
    for epoch in range(len(history), args.epochs):
        if epoch - best_epoch >= args.patience:
            break
        model.train()
        total_loss = 0.0
        for batch_views, target in train_loader:
            batch_views, target = batch_views.to(args.device), target.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_views)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite supervised loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(target)
        dev_auc = float(roc_auc_score(development_y, predict(model, dev_loader, args.device)))
        if dev_auc > best_auc:
            best_auc, best_epoch, best_state = dev_auc, epoch + 1, cpu_state(model)
        record = {"epoch": epoch + 1, "loss": total_loss / len(rows["labeled_train"]),
                  "development_auroc": dev_auc, "seconds": time.monotonic() - started}
        history.append(record)
        (directory / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")
        save_resume_checkpoint(resume_path, fingerprint, model, optimizer, best_state,
                               best_epoch, best_auc, history, record["seconds"])
        print(json.dumps({"stage": "pretrained_finetune", "model": args.model, **record}), flush=True)
        if epoch + 1 - best_epoch >= args.patience:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    torch.save({"model": best_state, "model_name": args.model, "feature_dim": dimension,
                "best_epoch": best_epoch, "checkpoint": checkpoint_meta,
                "adaptation": adaptation_meta}, directory / "model.pt")
    # The locked test set is touched only after all training and checkpoint selection.
    evaluate_predictions(run_name,
        predict(model, calibration_loader, args.device), predict(model, test_loader, args.device),
        calibration, rows["test"], directory, args.seed, args.bootstrap)
    config = {
        "model": args.model, "label_seed": args.seed, "official_checkpoint": checkpoint_meta,
        "official_source_commit": git_head(Path(__file__).resolve().parents[1] / "third_party" /
            ("HuBERT-ECG" if args.model == "hubert-small" else "fairseq-signals")),
        "extractor_source_sha256": cache_meta["extractor_sha256"],
        "finetune_source_sha256": sha256(Path(__file__)),
        "manifest_sha256": manifest_hashes, "cache_dir": str(cache_dir.resolve()),
        "cache_shape": cache_meta["shape"], "preprocessing": cache_meta["preprocessing"],
        "views": "two nonoverlapping 5-second views, mean of token means",
        "spec_augment": "disabled for HuBERT supervised fine-tuning",
        "augmentation": "none", "optimizer": "AdamW(weight_decay=0.01)",
        "backbone_lr": 1e-5, "head_lr": 1e-3, "epochs_budget": args.epochs,
        "patience": args.patience, "best_epoch": best_epoch,
        "best_development_auroc": best_auc, "batch_size": args.batch_size,
        "records": {"train": len(rows["labeled_train"]), "development": len(development),
                    "calibration": len(calibration), "test": len(rows["test"])},
        "device": args.device, "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "torch_version": torch.__version__, "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
        "adaptation": adaptation_meta,
    }
    (directory / "config.json").write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
    resume_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
