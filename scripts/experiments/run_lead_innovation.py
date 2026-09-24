#!/usr/bin/env python3
"""Run the prespecified eight-lead multiscale SSL ablation and classifiers."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ecg_experiment.data import ECGDataset, Waveforms, read_manifest
from ecg_experiment.evaluation import evaluate_predictions, partition_validation
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.lead_innovation import LEAD_INDICES, LeadClassifier, LeadMultiscaleEncoder, LeadSSL
from ecg_experiment.reproducibility import cpu_state, seed_everything
from ecg_experiment.training import checked_step, parameter_count, warmup_cosine_lr

ROOT = Path(__file__).resolve().parents[2]
NAMES = {"supervised": "lead_multiscale_supervised",
         "ordinary": "lead_multiscale_latent",
         "innovation": "lead_multiscale_innovation"}
SSL_SEED = 42
SSL_LR = 3e-4
ENCODER_LR = 3e-4
HEAD_LR = 1e-3
WEIGHT_DECAY = 0.01
INNOVATION_WEIGHT = 0.5
TEACHER_MOMENTUM_START = 0.99
TEACHER_MOMENTUM_RISE = 0.009
GAIN_RANGE = (0.9, 1.1)
NOISE_STD = 0.01


class EightLeadDataset(Dataset):
    """
    Scaled PTB-XL waveforms restricted to leads I, II and V1-V6.

    Parameters
    ----------
    waveforms : Waveforms
        Waveform cache.
    rows : list[dict[str, str]]
        Manifest rows.
    scale : np.ndarray
        Training-only per-lead RMS scale.
    """

    def __init__(self, waveforms: Waveforms, rows: list[dict[str, str]], scale: np.ndarray) -> None:
        self.base = ECGDataset(waveforms, rows, scale)

    def __len__(self) -> int:
        """Return the number of records."""
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Any]:
        """Return the eight-lead signal and target of record ``index``."""
        signal, target = self.base[index]
        return signal[list(LEAD_INDICES)], target


def make_loader(waveforms: Waveforms, rows: list[dict[str, str]], scale: np.ndarray, batch_size: int,
                shuffle: bool = False) -> DataLoader:
    """
    Build a single-process eight-lead loader.

    Parameters
    ----------
    waveforms : Waveforms
        Waveform cache.
    rows : list[dict[str, str]]
        Manifest rows.
    scale : np.ndarray
        Training-only per-lead RMS scale.
    batch_size : int
        Records per batch.
    shuffle : bool
        Shuffle each epoch with the global torch generator.

    Returns
    -------
    DataLoader
        Loader yielding ``(signal, target)`` batches.
    """
    return DataLoader(EightLeadDataset(waveforms, rows, scale), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0, drop_last=False)


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    """
    Compute logits for every batch in evaluation mode.

    Parameters
    ----------
    model : nn.Module
        Classifier returning one logit per record.
    loader : DataLoader
        Loader from ``make_loader``.
    device : str
        Device holding ``model``.

    Returns
    -------
    np.ndarray
        Logits in loader order.
    """
    model.eval()
    return np.concatenate([model(signal.to(device)).cpu().numpy() for signal, _ in loader])


def ssl_checkpoint(args: argparse.Namespace) -> Path:
    """
    Location of the variant's completed SSL encoder.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``output_dir`` and ``variant``.

    Returns
    -------
    Path
        ``encoder.pt`` path.
    """
    return args.output_dir / f"lead_multiscale_{args.variant}_ssl" / "encoder.pt"


def matches_ssl(state: dict[str, Any], args: argparse.Namespace, training_ids: list[str],
                scale: np.ndarray) -> bool:
    """
    Check that a saved SSL encoder was trained with these settings and waveforms.

    Parameters
    ----------
    state : dict[str, Any]
        Saved ``encoder.pt`` contents.
    args : argparse.Namespace
        Needs ``variant`` and ``ssl_epochs``.
    training_ids : list[str]
        SSL training ECG IDs in order.
    scale : np.ndarray
        Training-only per-lead RMS scale.

    Returns
    -------
    bool
        True when every recorded setting matches.
    """
    return (state["variant"] == args.variant and state["seed"] == SSL_SEED
            and state["epochs"] == args.ssl_epochs and state["training_ecg_ids"] == training_ids
            and np.allclose(state["scale"], scale))


def ssl_epoch(model: LeadSSL, optimizer: torch.optim.Optimizer, loader: DataLoader, args: argparse.Namespace,
              epoch: int) -> dict[str, float]:
    """
    Train one SSL epoch, updating the EMA teacher after every step.

    Parameters
    ----------
    model : LeadSSL
        Model in training mode.
    optimizer : torch.optim.Optimizer
        Its optimizer, with the epoch's learning rate set.
    loader : DataLoader
        Shuffled training loader.
    args : argparse.Namespace
        Needs ``device``, ``variant`` and ``ssl_epochs``.
    epoch : int
        Zero-based epoch, which sets the teacher momentum.

    Returns
    -------
    dict[str, float]
        Record-weighted mean losses.
    """
    totals, observations = {}, 0
    momentum = TEACHER_MOMENTUM_START + TEACHER_MOMENTUM_RISE * epoch / max(1, args.ssl_epochs - 1)
    for signal, _ in loader:
        signal = signal.to(args.device)
        loss, details = model(signal)
        checked_step(loss, model, optimizer, f"{args.variant} SSL")
        model.update_teacher(momentum)
        for key, value in {"loss": float(loss.detach()), **details}.items():
            totals[key] = totals.get(key, 0.0) + value * len(signal)
        observations += len(signal)
    return {key: value / observations for key, value in totals.items()}


def pretrain(args: argparse.Namespace, waveforms: Waveforms, all_train: list[dict[str, str]],
             scale: np.ndarray) -> Path:
    """
    Pretrain the variant's SSL encoder once, or verify the completed checkpoint.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    waveforms : Waveforms
        Waveform cache.
    all_train : list[dict[str, str]]
        Training rows of every label budget.
    scale : np.ndarray
        Training-only per-lead RMS scale.

    Returns
    -------
    Path
        The completed ``encoder.pt``.

    Raises
    ------
    ValueError
        If an existing checkpoint used other settings or waveforms.
    FileExistsError
        If an incomplete run exists.
    """
    seed_everything(SSL_SEED)
    checkpoint = ssl_checkpoint(args)
    directory = checkpoint.parent
    directory.mkdir(parents=True, exist_ok=True)
    expected_ids = [row["ecg_id"] for row in all_train]
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not matches_ssl(state, args, expected_ids, scale):
            raise ValueError(f"SSL checkpoint does not match inputs/settings: {checkpoint}")
        print(f"Using completed SSL checkpoint {checkpoint}", flush=True)
        return checkpoint
    if (directory / "history.json").exists():
        raise FileExistsError(f"Incomplete SSL run in {directory}; use a new output directory")
    model = LeadSSL(0.0 if args.variant == "ordinary" else INNOVATION_WEIGHT).to(args.device)
    loader = make_loader(waveforms, all_train, scale, args.ssl_batch_size, shuffle=True)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                                  lr=SSL_LR, weight_decay=WEIGHT_DECAY)
    history = []
    started = time.monotonic()
    for epoch in range(args.ssl_epochs):
        model.train()
        lr = warmup_cosine_lr(SSL_LR, epoch, args.ssl_epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr
        means = ssl_epoch(model, optimizer, loader, args, epoch)
        record = {"epoch": epoch + 1, "lr": lr, "seconds": time.monotonic() - started, **means}
        history.append(record)
        write_json_atomic(directory / "history.json", history)
        print(json.dumps({"stage": f"lead_{args.variant}_ssl", **record}), flush=True)
    torch.save({"encoder": cpu_state(model.encoder), "scale": scale.tolist(),
                "variant": args.variant, "seed": SSL_SEED, "epochs": args.ssl_epochs,
                "training_ecg_ids": expected_ids}, checkpoint)
    write_json_atomic(directory / "config.json", {
        "architecture": repr(model), "variant": args.variant,
        "ordinary_weight": 1.0, "innovation_weight": model.innovation_weight,
        "variance_weight": 0.1, "ssl_epochs": args.ssl_epochs,
        "ssl_batch_size": args.ssl_batch_size, "seed": SSL_SEED,
        "training_records": len(all_train), "seconds": time.monotonic() - started,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "encoder_parameters": parameter_count(model.encoder),
        "mask": "One whole lead and two distinct leads with contiguous 2-second spans; raw samples hidden "
                "before both branches",
        "target": "Stop-gradient teacher tokens batch-centered per lead/time coordinate before LayerNorm; "
                  "innovation subtracts the seven other centered leads at the same time",
        "target_batch_dependence": "Teacher content targets depend on the current training batch; batches "
                                   "require at least two records",
        "checkpoint": str(checkpoint)})
    return checkpoint


def build_classifier(args: argparse.Namespace, all_train: list[dict[str, str]],
                     scale: np.ndarray) -> tuple[LeadClassifier, Path | None]:
    """
    Build the classifier from scratch or from the variant's verified SSL encoder.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``variant``, ``ssl_epochs``, ``output_dir`` and ``device``.
    all_train : list[dict[str, str]]
        SSL training rows.
    scale : np.ndarray
        Training-only per-lead RMS scale.

    Returns
    -------
    tuple[LeadClassifier, Path | None]
        Classifier on the device and the SSL checkpoint used, if any.

    Raises
    ------
    ValueError
        If the SSL checkpoint does not match this experiment.
    """
    encoder = LeadMultiscaleEncoder()
    checkpoint = None
    if args.variant != "supervised":
        checkpoint = ssl_checkpoint(args)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not matches_ssl(state, args, [row["ecg_id"] for row in all_train], scale):
            raise ValueError(f"SSL checkpoint does not match this experiment: {checkpoint}")
        encoder.load_state_dict(state["encoder"])
    return LeadClassifier(encoder).to(args.device), checkpoint


def supervised_epoch(model: LeadClassifier, optimizer: torch.optim.Optimizer, loader: DataLoader,
                     device: str, name: str) -> float:
    """
    Train one supervised epoch with random gain and additive noise.

    Parameters
    ----------
    model : LeadClassifier
        Classifier in training mode.
    optimizer : torch.optim.Optimizer
        Its optimizer.
    loader : DataLoader
        Shuffled labeled loader.
    device : str
        Device holding ``model``.
    name : str
        Arm name, used in error messages.

    Returns
    -------
    float
        Record-weighted loss sum.
    """
    total_loss = 0.0
    for signal, target in loader:
        signal, target = signal.to(device), target.to(device)
        gain = torch.empty(len(signal), 1, 1, device=device).uniform_(*GAIN_RANGE)
        signal = signal * gain + torch.randn_like(signal) * NOISE_STD
        loss = nn.functional.binary_cross_entropy_with_logits(model(signal), target)
        checked_step(loss, model, optimizer, f"{name} supervised")
        total_loss += float(loss.detach()) * len(signal)
    return total_loss


def supervised(args: argparse.Namespace, waveforms: Waveforms, train: list[dict[str, str]],
               validation: list[dict[str, str]], test: list[dict[str, str]], all_train: list[dict[str, str]],
               scale: np.ndarray) -> None:
    """
    Train a classifier with development early stopping, then evaluate it.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    waveforms : Waveforms
        Waveform cache.
    train : list[dict[str, str]]
        Labeled training rows.
    validation : list[dict[str, str]]
        Validation rows, split into development and calibration patients.
    test : list[dict[str, str]]
        Test rows.
    all_train : list[dict[str, str]]
        SSL training rows.
    scale : np.ndarray
        Training-only per-lead RMS scale.

    Raises
    ------
    FileExistsError
        If the output directory already holds results.
    """
    seed_everything(args.seed)
    name = NAMES[args.variant]
    directory = args.output_dir / f"{name}_seed{args.seed}"
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    model, checkpoint = build_classifier(args, all_train, scale)
    development, calibration = partition_validation(validation)
    train_loader = make_loader(waveforms, train, scale, args.batch_size, shuffle=True)
    dev_loader = make_loader(waveforms, development, scale, args.batch_size)
    development_y = np.array([int(row["target"]) for row in development])
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": ENCODER_LR},
        {"params": model.head.parameters(), "lr": HEAD_LR}], weight_decay=WEIGHT_DECAY)
    best_auc, best_epoch, best_state, history = -1.0, 0, None, []
    started = time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        total_loss = supervised_epoch(model, optimizer, train_loader, args.device, name)
        dev_auc = float(roc_auc_score(development_y, predict(model, dev_loader, args.device)))
        if dev_auc > best_auc:
            best_auc, best_epoch, best_state = dev_auc, epoch + 1, cpu_state(model)
        record = {"epoch": epoch + 1, "loss": total_loss / len(train),
                  "development_auroc": dev_auc, "seconds": time.monotonic() - started}
        history.append(record)
        write_json_atomic(directory / "history.json", history)
        print(json.dumps({"stage": name, "label_seed": args.seed, **record}), flush=True)
        if epoch + 1 - best_epoch >= args.patience:
            break
    model.load_state_dict(best_state)
    torch.save({"model": best_state, "scale": scale.tolist(), "best_epoch": best_epoch,
                "label_seed": args.seed, "architecture": name,
                "ssl_checkpoint": str(checkpoint) if checkpoint else None}, directory / "model.pt")
    calibration_loader = make_loader(waveforms, calibration, scale, args.batch_size)
    calibration_logits = predict(model, calibration_loader, args.device)
    test_logits = predict(model, make_loader(waveforms, test, scale, args.batch_size), args.device)
    evaluate_predictions(name, calibration_logits, test_logits, calibration, test, directory, args.seed,
                         args.bootstrap)
    write_json_atomic(directory / "config.json", {
        "architecture": repr(model), "variant": args.variant,
        "parameters": parameter_count(model),
        "encoder_parameters": parameter_count(model.encoder),
        "label_seed": args.seed, "labeled_training_records": len(train),
        "development_records": len(development), "calibration_records": len(calibration),
        "test_records": len(test), "epochs_budget": args.epochs,
        "patience": args.patience, "best_epoch": best_epoch,
        "best_development_auroc": best_auc, "batch_size": args.batch_size,
        "ssl_checkpoint": str(checkpoint) if checkpoint else None,
        "ssl_seed": SSL_SEED if checkpoint else None,
        "ssl_epochs": args.ssl_epochs if checkpoint else None,
        "label_manifest_sha256": sha256_file(args.manifest_dir / "labeled_train.csv"),
        "normalization": "Per-record per-lead demeaning, training-only per-lead RMS scaling, clipping +/-20; "
                         "then I, II, V1-V6",
        "pretraining_exposure": "Own SSL uses official PTB-XL folds 1-8 only; no validation/test waveforms",
        "hypothesis": "Lead-residual latent prediction may preserve lead-specific morphology beyond shared "
                      "rhythm; this is not physiological proof",
        "device": args.device, "torch_version": torch.__version__,
        "seconds": time.monotonic() - started})


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """
    Parse and validate the command line.

    Parameters
    ----------
    argv : Sequence[str] | None
        Arguments, or ``None`` for ``sys.argv``.

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
    parser.add_argument("--stage", choices=("ssl", "train"), required=True)
    parser.add_argument("--variant", choices=tuple(NAMES), required=True)
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "data/processed/ptbxl/seed42_fraction0.1")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data/processed/ptbxl/waveforms100")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/experiment001")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ssl-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--ssl-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=500)
    args = parser.parse_args(argv)
    if min(args.threads, args.batch_size, args.ssl_batch_size, args.epochs,
           args.ssl_epochs, args.patience, args.bootstrap) < 1:
        parser.error("threads, batch sizes, epochs, patience, and bootstrap must be positive")
    if args.stage == "ssl" and args.ssl_batch_size < 2:
        parser.error("SSL batch size must be at least two for batch-centered teacher targets")
    if args.stage == "ssl" and args.variant == "supervised":
        parser.error("supervised variant has no SSL stage")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """
    Run the SSL or supervised stage while holding the shared GPU lock.

    Parameters
    ----------
    argv : Sequence[str] | None
        Arguments, or ``None`` for ``sys.argv``.
    """
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    with gpu_lock(args.device):
        waveforms = Waveforms(args.cache_dir)
        all_train = read_manifest(args.manifest_dir / "all_train_ssl.csv")
        scale = waveforms.training_scale(all_train)
        if args.stage == "ssl":
            pretrain(args, waveforms, all_train, scale)
            return
        supervised(args, waveforms, read_manifest(args.manifest_dir / "labeled_train.csv"),
                   read_manifest(args.manifest_dir / "validation.csv"),
                   read_manifest(args.manifest_dir / "test.csv"), all_train, scale)


if __name__ == "__main__":
    main()
