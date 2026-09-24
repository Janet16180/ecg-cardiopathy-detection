#!/usr/bin/env python3
"""Fine-tune the official HuBERT-small or ECG-FM SSL backbone on PTB-XL.

Run with ``.venv-pretrained/bin/python -m scripts.experiments.finetune_pretrained``. The
published checkpoint is used as an encoder only: no disease-trained head is
loaded. Waveform preprocessing matches ``ecg_experiment.foundation_models``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ecg_experiment.data import read_manifest
from ecg_experiment.evaluation import evaluate_predictions, partition_validation
from ecg_experiment.files import sha256_file, write_json_atomic, write_torch_atomic
from ecg_experiment.finetuning import flat_rng_state, peak_cuda_memory, require_cuda, restore_flat_rng_state
from ecg_experiment.foundation_models import (
    checkpoint_info,
    load_model,
    preprocess_ecg_fm,
    preprocess_hubert,
    preprocessing_source_sha256,
)
from ecg_experiment.provenance import git_head
from ecg_experiment.reproducibility import cpu_state, seed_everything
from ecg_experiment.waveforms import read_record

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_ROOT = ROOT / "data/processed/ptbxl/pretrained_views"
SPLITS = ("labeled_train", "validation", "test")
HUBERT = "hubert-small"
HUBERT_VIEW_SHAPE = (2, 6000)
ECG_FM_VIEW_SHAPE = (2, 12, 2500)
PROGRESS_INTERVAL = 500
BACKBONE_LR = 1e-5
HEAD_LR = 1e-3
WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 1.0

Rows = list[dict[str, str]]


def manifest_data(manifest_dir: Path) -> tuple[dict[str, Rows], dict[str, str]]:
    """
    Read and check the labeled train, validation, and test manifests.

    Parameters
    ----------
    manifest_dir : Path
        Directory holding the three split manifests.

    Returns
    -------
    tuple[dict[str, list[dict[str, str]]], dict[str, str]]
        Rows per split and the SHA-256 of each manifest.

    Raises
    ------
    ValueError
        If ECG IDs overlap, a split is empty, or training lacks a class.
    """
    rows = {name: read_manifest(manifest_dir / f"{name}.csv") for name in SPLITS}
    hashes = {f"{name}.csv": sha256_file(manifest_dir / f"{name}.csv") for name in SPLITS}
    ids = [row["ecg_id"] for name in SPLITS for row in rows[name]]
    if len(ids) != len(set(ids)):
        raise ValueError("ECG IDs overlap across labeled train, validation, and test")
    for name in SPLITS:
        if not rows[name]:
            raise ValueError(f"{name} has no records")
    # Validate the training labels here; validation is checked only after its
    # development/calibration split. Leave locked test labels untouched.
    if {row["target"] for row in rows["labeled_train"]} != {"0", "1"}:
        raise ValueError("labeled_train must contain both binary classes")
    return rows, hashes


def cache_shape(model_name: str, records: int) -> tuple[int, ...]:
    """
    Return the preprocessed view array shape of a backbone.

    Parameters
    ----------
    model_name : str
        ``"hubert-small"`` or ``"ecg-fm"``.
    records : int
        Number of cached ECGs.

    Returns
    -------
    tuple[int, ...]
        ``[records, 2, 6000]`` for HuBERT or ``[records, 2, 12, 2500]`` for ECG-FM.
    """
    return (records, *(HUBERT_VIEW_SHAPE if model_name == HUBERT else ECG_FM_VIEW_SHAPE))


def write_view_cache(model_name: str, raw_dir: Path, rows: Rows, partial: Path,
                     shape: tuple[int, ...]) -> None:
    """
    Preprocess every ECG into a memory-mapped partial cache file.

    Parameters
    ----------
    model_name : str
        Backbone whose official preprocessing is applied.
    raw_dir : Path
        PTB-XL waveform directory.
    rows : list[dict[str, str]]
        Records in cache order.
    partial : Path
        Destination ``.npy`` file.
    shape : tuple[int, ...]
        Array shape from ``cache_shape``.

    Raises
    ------
    ValueError
        If a preprocessed record has an unexpected shape or nonfinite values.
    """
    preprocess = preprocess_hubert if model_name == HUBERT else preprocess_ecg_fm
    matrix = np.lib.format.open_memmap(partial, mode="w+", dtype="float32", shape=shape)
    for i, row in enumerate(rows):
        views = preprocess(read_record(raw_dir, row["filename_hr"]))
        if views.shape != shape[1:] or not np.isfinite(views).all():
            raise ValueError(f"Unexpected preprocessed views for ECG {row['ecg_id']}")
        matrix[i] = views
        if (i + 1) % PROGRESS_INTERVAL == 0 or i + 1 == len(rows):
            print(f"Cached {model_name}: {i + 1}/{len(rows)} ECGs", flush=True)
    matrix.flush()


def _load_existing_cache(model_name: str, cache_dir: Path, requested: dict[str, Any],
                         records: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Open a complete cache after checking that it was made from the requested inputs."""
    meta_path, array_path = cache_dir / "metadata.json", cache_dir / "views.npy"
    if not meta_path.exists() or not array_path.exists():
        raise ValueError(f"Incomplete cache in {cache_dir}")
    saved = json.loads(meta_path.read_text())
    if any(saved.get(key) != value for key, value in requested.items()):
        raise ValueError(f"Preprocessing cache does not match the requested inputs: {cache_dir}")
    array = np.load(array_path, mmap_mode="r")
    if array.shape != cache_shape(model_name, records) or array.dtype != np.float32:
        raise ValueError(f"Malformed preprocessing cache in {cache_dir}: {array.shape}")
    return array, saved


def cache_views(model_name: str, raw_dir: Path, cache_dir: Path, rows: dict[str, Rows],
                manifest_hashes: dict[str, str]) -> tuple[np.ndarray, dict[str, int], dict[str, Any]]:
    """
    Cache deterministic official preprocessing, keyed to source and manifests.

    Parameters
    ----------
    model_name : str
        ``"hubert-small"`` or ``"ecg-fm"``.
    raw_dir : Path
        PTB-XL waveform directory.
    cache_dir : Path
        Cache directory.
    rows : dict[str, list[dict[str, str]]]
        Split rows; the cache holds all splits in order.
    manifest_hashes : dict[str, str]
        Manifest digests recorded in the cache metadata.

    Returns
    -------
    tuple[np.ndarray, dict[str, int], dict[str, Any]]
        Memory-mapped views, cache row per ECG identifier, and metadata.

    Raises
    ------
    ValueError
        If an existing cache is incomplete, malformed, or made from other inputs.
    FileExistsError
        If an interrupted cache file is present.
    """
    all_rows = [row for name in SPLITS for row in rows[name]]
    ids = [row["ecg_id"] for row in all_rows]
    index = {ecg_id: i for i, ecg_id in enumerate(ids)}
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "metadata.json"
    array_path = cache_dir / "views.npy"
    requested = {
        "model": model_name,
        "raw_dir": str(raw_dir.resolve()),
        "extractor_sha256": preprocessing_source_sha256(),
        "manifest_sha256": manifest_hashes,
        "ecg_ids": ids,
    }
    if meta_path.exists() or array_path.exists():
        array, saved = _load_existing_cache(model_name, cache_dir, requested, len(ids))
        return array, index, saved

    partial = cache_dir / "views.partial.npy"
    if partial.exists():
        raise FileExistsError(f"Previous incomplete cache exists: {partial}")
    shape = cache_shape(model_name, len(ids))
    start = time.monotonic()
    write_view_cache(model_name, raw_dir, all_rows, partial, shape)
    os.replace(partial, array_path)
    metadata = {**requested, "shape": list(shape), "dtype": "float32",
                "seconds": time.monotonic() - start,
                "preprocessing": ("Official HuBERT FIR/min-max, two 5-second flattened/decimated views"
                                  if model_name == HUBERT else
                                  "Official ECG-FM lead-wise z-score, two 5-second views")}
    write_json_atomic(meta_path, metadata, allow_nan=True)
    return np.load(array_path, mmap_mode="r"), index, metadata


class ViewDataset(Dataset):
    """
    Labeled preprocessed views read from the cache.

    Parameters
    ----------
    views : np.ndarray
        Memory-mapped view cache.
    index : dict[str, int]
        Cache row of each string ECG identifier.
    rows : list[dict[str, str]]
        Manifest rows with ``ecg_id`` and ``target``.
    """

    def __init__(self, views: np.ndarray, index: dict[str, int], rows: Rows) -> None:
        self.views, self.rows = views, rows
        self.indices = [index[row["ecg_id"]] for row in rows]

    def __len__(self) -> int:
        """Return the number of records."""
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the views of one ECG and its binary target.

        Parameters
        ----------
        i : int
            Record position.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Float32 views and scalar float32 target.
        """
        views = np.array(self.views[self.indices[i]], dtype=np.float32, copy=True)
        return torch.from_numpy(views), torch.tensor(float(self.rows[i]["target"]), dtype=torch.float32)


def backbone_tokens(model_name: str, backbone: nn.Module, source: torch.Tensor) -> torch.Tensor:
    """
    Return final-layer tokens of a HuBERT or ECG-FM backbone.

    Parameters
    ----------
    model_name : str
        ``"hubert-small"`` or ``"ecg-fm"``.
    backbone : nn.Module
        Loaded backbone.
    source : torch.Tensor
        Flattened views, one per batch row.

    Returns
    -------
    torch.Tensor
        ``[views, tokens, features]`` tokens.
    """
    if model_name == HUBERT:
        return backbone(input_values=source).last_hidden_state
    return backbone.extract_features(source=source, padding_mask=None, mask=False)["x"]


class FineTunedECG(nn.Module):
    """
    Pretrained backbone with a LayerNorm and linear binary head.

    Parameters
    ----------
    model_name : str
        ``"hubert-small"`` or ``"ecg-fm"``.
    backbone : nn.Module
        Loaded backbone.
    feature_dim : int
        Token feature dimension.
    """

    def __init__(self, model_name: str, backbone: nn.Module, feature_dim: int) -> None:
        super().__init__()
        self.model_name = model_name
        self.backbone = backbone
        self.head = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, 1))

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        """
        Predict one logit per ECG from the mean of its view token means.

        Parameters
        ----------
        views : torch.Tensor
            ``[batch, views, ...]`` preprocessed views.

        Returns
        -------
        torch.Tensor
            Logits of shape ``[batch]``.

        Raises
        ------
        ValueError
            If the backbone returns tokens of an unexpected shape.
        """
        batch, number_of_views = views.shape[:2]
        source = views.reshape(batch * number_of_views, *views.shape[2:])
        # Same token extraction and pooling as extract_pretrained.embed_views,
        # deliberately without torch.inference_mode() so gradients reach the backbone.
        tokens = backbone_tokens(self.model_name, self.backbone, source)
        if tokens.ndim != 3 or tokens.shape[0] != batch * number_of_views:
            raise ValueError(f"Unexpected token shape: {tuple(tokens.shape)}")
        features = tokens.mean(dim=1).reshape(batch, number_of_views, -1).mean(dim=1)
        return self.head(features).squeeze(-1)


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    """
    Return logits for every batch of a loader.

    Parameters
    ----------
    model : nn.Module
        Classifier, switched to evaluation mode.
    loader : DataLoader
        Loader yielding ``(views, target)`` batches.
    device : str
        Torch device of ``model``.

    Returns
    -------
    np.ndarray
        Concatenated logits.
    """
    model.eval()
    return np.concatenate([model(views.to(device)).cpu().numpy() for views, _ in loader])


def save_resume_checkpoint(path: Path, fingerprint: dict[str, Any], model: nn.Module,
                           optimizer: torch.optim.Optimizer, best_state: dict[str, torch.Tensor] | None,
                           best_epoch: int, best_auc: float, history: list[dict[str, Any]],
                           elapsed_seconds: float) -> None:
    """
    Commit a complete epoch boundary without exposing a partial checkpoint.

    Parameters
    ----------
    path : Path
        Destination ``resume.pt``.
    fingerprint : dict[str, Any]
        Inputs that a resumed run must match.
    model : nn.Module
        Current classifier.
    optimizer : torch.optim.Optimizer
        Current optimizer.
    best_state : dict[str, torch.Tensor] | None
        CPU weights of the best development epoch.
    best_epoch : int
        One-based best development epoch.
    best_auc : float
        Best development AUROC.
    history : list[dict[str, Any]]
        One row per completed epoch.
    elapsed_seconds : float
        Training seconds so far.
    """
    write_torch_atomic(path, {
        "version": 1, "fingerprint": fingerprint,
        "model": cpu_state(model), "optimizer": optimizer.state_dict(),
        "best_model": best_state, "best_epoch": best_epoch, "best_auc": best_auc,
        "history": history, "elapsed_seconds": elapsed_seconds, **flat_rng_state(),
    })


def load_resume_checkpoint(path: Path, fingerprint: dict[str, Any], model: nn.Module,
                           optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """
    Restore training state from an epoch-boundary checkpoint.

    Parameters
    ----------
    path : Path
        Checkpoint written by ``save_resume_checkpoint``.
    fingerprint : dict[str, Any]
        Inputs of the current run.
    model : nn.Module
        Classifier to restore in place.
    optimizer : torch.optim.Optimizer
        Optimizer to restore in place.

    Returns
    -------
    dict[str, Any]
        The loaded checkpoint.

    Raises
    ------
    ValueError
        If the checkpoint belongs to other inputs or is malformed.
    """
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
    restore_flat_rng_state(checkpoint)
    return checkpoint


def check_adaptation_budget(metadata: dict[str, Any], history: list[dict[str, Any]]) -> None:
    """
    Accept complete fixed-epoch runs and exact-update bounded runs.

    Parameters
    ----------
    metadata : dict[str, Any]
        Adaptation configuration.
    history : list[dict[str, Any]]
        Adaptation history, one row per epoch.

    Raises
    ------
    ValueError
        If the adaptation stopped before its update or epoch budget.
    """
    if "max_updates" in metadata:
        final = history[-1] if history else {}
        if (final.get("updates") != metadata["max_updates"]
                or final.get("seen_examples") != metadata["max_updates"] * metadata["batch_size"]):
            raise ValueError("Adaptation did not finish its exact optimizer-update budget")
    elif len(history) != metadata["epochs"]:
        raise ValueError("Adaptation did not finish its fixed epoch budget")


def parse_args() -> argparse.Namespace:
    """
    Parse and validate command-line arguments.

    Returns
    -------
    argparse.Namespace
        Validated arguments.

    Raises
    ------
    RuntimeError
        If CUDA is requested but unavailable.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(HUBERT, "ecg-fm"), required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path,
                        help="Preprocessed views; default: "
                             "data/processed/ptbxl/pretrained_views/<model>/<manifest hash>")
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
    require_cuda(args.device)
    if args.adapted_backbone and args.model != "ecg-fm":
        parser.error("--adapted-backbone is supported for ECG-FM only")
    return args


def load_adapted_backbone(args: argparse.Namespace, backbone: nn.Module, checkpoint_meta: dict[str, Any],
                          rows: dict[str, Rows]) -> dict[str, Any]:
    """
    Load a verified training-only continued-SSL ECG-FM checkpoint into a backbone.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments with ``adapted_backbone`` and ``manifest_dir``.
    backbone : nn.Module
        Official ECG-FM backbone, updated in place.
    checkpoint_meta : dict[str, Any]
        Official checkpoint metadata.
    rows : dict[str, list[dict[str, str]]]
        Split rows used to check held-out patient isolation.

    Returns
    -------
    dict[str, Any]
        Adaptation metadata with the adapted checkpoint digest.

    Raises
    ------
    ValueError
        If the adaptation used another source, manifest, held-out patients,
        or an incomplete budget.
    """
    adapted = torch.load(args.adapted_backbone, map_location="cpu", weights_only=True)
    metadata = adapted["metadata"]
    expected_ssl = read_manifest(args.manifest_dir / "all_train_ssl.csv")
    if (metadata["official_checkpoint"]["sha256"] != checkpoint_meta["sha256"]
            or metadata["training_ecg_ids"] != [r["ecg_id"] for r in expected_ssl]
            or metadata["manifest_sha256"]["all_train_ssl.csv"]
            != sha256_file(args.manifest_dir / "all_train_ssl.csv")):
        raise ValueError("Adaptation checkpoint source or training manifest differs")
    heldout_patients = {r["patient_id"] for name in ("validation", "test") for r in rows[name]}
    if heldout_patients & set(metadata["training_patient_ids"]):
        raise ValueError("Adaptation checkpoint encountered held-out patients")
    check_adaptation_budget(metadata, adapted["history"])
    backbone.load_state_dict(adapted["backbone"], strict=True)
    return {**metadata, "checkpoint_sha256": sha256_file(args.adapted_backbone)}


def disable_spec_augment(backbone: nn.Module) -> None:
    """
    Turn off HuBERT's SSL time and feature masking.

    The masking is an upstream objective, not supervised fine-tuning
    augmentation.

    Parameters
    ----------
    backbone : nn.Module
        HuBERT backbone, updated in place.
    """
    backbone.config.mask_time_prob = 0.0
    backbone.config.mask_feature_prob = 0.0
    backbone.config.apply_spec_augment = False


def run_name(args: argparse.Namespace, adaptation_meta: dict[str, Any] | None) -> str:
    """
    Name the fine-tuning run after its backbone and adaptation.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    adaptation_meta : dict[str, Any] | None
        Adaptation metadata, if an adapted backbone is used.

    Returns
    -------
    str
        Run name without the seed suffix.
    """
    suffix = "finetuned"
    if adaptation_meta and adaptation_meta.get("external_ssl"):
        suffix = "pooled_adapted_finetuned"
    elif args.adapted_backbone:
        suffix = "adapted_finetuned"
    return f"{args.model}_{suffix}"


def prepare_directory(args: argparse.Namespace, directory: Path) -> None:
    """
    Create the run directory, refusing to overwrite or resume inconsistently.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    directory : Path
        Run directory.

    Raises
    ------
    ValueError
        If ``--resume`` targets a completed run.
    FileNotFoundError
        If ``--resume`` is given without a resume checkpoint.
    FileExistsError
        If a fresh run targets a nonempty directory.
    """
    if args.resume:
        if (directory / "config.json").is_file() and (directory / "metrics.json").is_file():
            raise ValueError(f"Fine-tuning output is already complete: {directory}")
        if not (directory / "resume.pt").is_file():
            raise FileNotFoundError(f"Fine-tuning resume checkpoint is absent: {directory / 'resume.pt'}")
    elif directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: str) -> float:
    """
    Train one epoch with one optimizer update per batch.

    Parameters
    ----------
    model : nn.Module
        Classifier to train in place.
    loader : DataLoader
        Shuffled training loader.
    optimizer : torch.optim.Optimizer
        Optimizer of ``model``.
    device : str
        Torch device of ``model``.

    Returns
    -------
    float
        Sum of per-batch mean losses weighted by batch size.

    Raises
    ------
    RuntimeError
        If a loss is nonfinite.
    """
    model.train()
    total_loss = 0.0
    for batch_views, target in loader:
        batch_views, target = batch_views.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_views)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite supervised loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
        total_loss += float(loss.detach()) * len(target)
    return total_loss


def fit(args: argparse.Namespace, model: nn.Module, optimizer: torch.optim.Optimizer,
        train_loader: DataLoader, dev_loader: DataLoader, development_y: np.ndarray,
        directory: Path, fingerprint: dict[str, Any]) -> tuple[dict[str, torch.Tensor], int, float, float]:
    """
    Train with development-AUROC early stopping and epoch-boundary resume.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    model : nn.Module
        Classifier to train in place.
    optimizer : torch.optim.Optimizer
        Optimizer of ``model``.
    train_loader : DataLoader
        Shuffled training loader.
    dev_loader : DataLoader
        Development loader.
    development_y : np.ndarray
        Development targets in loader order.
    directory : Path
        Run directory for ``history.json`` and ``resume.pt``.
    fingerprint : dict[str, Any]
        Inputs recorded in the resume checkpoint.

    Returns
    -------
    tuple[dict[str, torch.Tensor], int, float, float]
        Best CPU weights, best epoch, best development AUROC, and the
        monotonic start time adjusted for resumed seconds.

    Raises
    ------
    RuntimeError
        If no epoch was selected.
    """
    resume_path = directory / "resume.pt"
    best_auc, best_epoch, best_state = -1.0, 0, None
    history = []
    started = time.monotonic()
    if args.resume:
        saved = load_resume_checkpoint(resume_path, fingerprint, model, optimizer)
        best_auc, best_epoch, best_state = saved["best_auc"], saved["best_epoch"], saved["best_model"]
        history = saved["history"]
        started -= saved["elapsed_seconds"]
        del saved
        print(json.dumps({"stage": "pretrained_finetune_resumed", "model": args.model,
                          "completed_epochs": len(history), "best_epoch": best_epoch}), flush=True)
    for epoch in range(len(history), args.epochs):
        if epoch - best_epoch >= args.patience:
            break
        total_loss = train_epoch(model, train_loader, optimizer, args.device)
        dev_auc = float(roc_auc_score(development_y, predict(model, dev_loader, args.device)))
        if dev_auc > best_auc:
            best_auc, best_epoch, best_state = dev_auc, epoch + 1, cpu_state(model)
        record = {"epoch": epoch + 1, "loss": total_loss / len(train_loader.dataset),
                  "development_auroc": dev_auc, "seconds": time.monotonic() - started}
        history.append(record)
        write_json_atomic(directory / "history.json", history)
        save_resume_checkpoint(resume_path, fingerprint, model, optimizer, best_state,
                               best_epoch, best_auc, history, record["seconds"])
        print(json.dumps({"stage": "pretrained_finetune", "model": args.model, **record}), flush=True)
        if epoch + 1 - best_epoch >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("No fine-tuning checkpoint selected")
    return best_state, best_epoch, best_auc, started


def main() -> None:
    """Fine-tune a pretrained backbone, select by development AUROC, and evaluate once."""
    args = parse_args()
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    rows, manifest_hashes = manifest_data(args.manifest_dir)
    development, calibration = partition_validation(rows["validation"])
    for name, subset in (("development", development), ("calibration", calibration)):
        if {row["target"] for row in subset} != {"0", "1"}:
            raise ValueError(f"{name} split has only one class")
    manifest_digest = hashlib.sha256(json.dumps(manifest_hashes, sort_keys=True).encode()).hexdigest()[:16]
    cache_dir = args.cache_dir or (DEFAULT_CACHE_ROOT / args.model / manifest_digest)
    views, index, cache_meta = cache_views(args.model, args.raw_dir, cache_dir, rows, manifest_hashes)
    checkpoint, checkpoint_meta = checkpoint_info(args.model)
    backbone, _ = load_model(args.model, checkpoint, args.device)
    adaptation_meta = None
    if args.adapted_backbone:
        adaptation_meta = load_adapted_backbone(args, backbone, checkpoint_meta, rows)
    if args.model == HUBERT:
        disable_spec_augment(backbone)
    first_view = torch.from_numpy(np.array(views[0], copy=True)).to(args.device)
    with torch.no_grad():
        dimension = backbone_tokens(args.model, backbone, first_view).shape[-1]
    model = FineTunedECG(args.model, backbone, int(dimension)).to(args.device)
    # Unseeded shuffling draws from the global torch generator seeded above.
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
        {"params": model.backbone.parameters(), "lr": BACKBONE_LR},
        {"params": model.head.parameters(), "lr": HEAD_LR},
    ], weight_decay=WEIGHT_DECAY)
    name = run_name(args, adaptation_meta)
    directory = args.output_dir / f"{name}_seed{args.seed}"
    prepare_directory(args, directory)
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
        "finetune_source_sha256": sha256_file(Path(__file__)),
        "reproducibility_source_sha256": sha256_file(ROOT / "ecg_experiment/reproducibility.py"),
        "finetuning_source_sha256": sha256_file(ROOT / "ecg_experiment/finetuning.py"),
    }
    best_state, best_epoch, best_auc, started = fit(args, model, optimizer, train_loader, dev_loader,
                                                    development_y, directory, fingerprint)
    model.load_state_dict(best_state)
    # Plain torch.save keeps the archive name, and so the file bytes, of earlier runs.
    torch.save({"model": best_state, "model_name": args.model, "feature_dim": dimension,
                "best_epoch": best_epoch, "checkpoint": checkpoint_meta,
                "adaptation": adaptation_meta}, directory / "model.pt")
    # The locked test set is touched only after all training and checkpoint selection.
    evaluate_predictions(name,
                         predict(model, calibration_loader, args.device),
                         predict(model, test_loader, args.device),
                         calibration, rows["test"], directory, args.seed, args.bootstrap)
    config = {
        "model": args.model, "label_seed": args.seed, "official_checkpoint": checkpoint_meta,
        "official_source_commit": git_head(ROOT / "third_party" /
                                           ("HuBERT-ECG" if args.model == HUBERT else "fairseq-signals")),
        "extractor_source_sha256": cache_meta["extractor_sha256"],
        "finetune_source_sha256": sha256_file(Path(__file__)),
        "manifest_sha256": manifest_hashes, "cache_dir": str(cache_dir.resolve()),
        "cache_shape": cache_meta["shape"], "preprocessing": cache_meta["preprocessing"],
        "views": "two nonoverlapping 5-second views, mean of token means",
        "spec_augment": "disabled for HuBERT supervised fine-tuning",
        "augmentation": "none", "optimizer": f"AdamW(weight_decay={WEIGHT_DECAY})",
        "backbone_lr": BACKBONE_LR, "head_lr": HEAD_LR, "epochs_budget": args.epochs,
        "patience": args.patience, "best_epoch": best_epoch,
        "best_development_auroc": best_auc, "batch_size": args.batch_size,
        "records": {"train": len(rows["labeled_train"]), "development": len(development),
                    "calibration": len(calibration), "test": len(rows["test"])},
        "device": args.device,
        "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "torch_version": torch.__version__, "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_memory_bytes": peak_cuda_memory(args.device),
        "adaptation": adaptation_meta,
    }
    write_json_atomic(directory / "config.json", config)
    (directory / "resume.pt").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
