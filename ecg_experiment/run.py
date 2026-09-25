"""Train and evaluate the compact PTB-XL experiment on fixed patient manifests."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from .data import ECGDataset, Waveforms, build_cache
from .evaluation import evaluate_predictions, partition_validation
from .files import read_csv, sha256_file, write_json_atomic, write_torch_atomic
from .models import CNN, JEPA, Classifier, MaskedAutoencoder, PatchTransformer
from .reproducibility import cpu_state, seed_everything
from .training import checked_step, require_cuda, warmup_cosine_lr

SSL_MODELS = frozenset({"mae", "jepa"})
RESULT_NAMES = {"cnn": "cnn_supervised", "transformer": "transformer_supervised",
                "mae": "mae_finetuned", "jepa": "jepa_finetuned"}
SSL_LR = 3e-4
# Match the optimizer across transformer scratch/MAE/JEPA to isolate SSL initialization.
ENCODER_LR = 3e-4
HEAD_LR = 1e-3
WEIGHT_DECAY = 0.01
TEACHER_MOMENTUM_START = 0.99
TEACHER_MOMENTUM_RANGE = 0.009
GAIN_RANGE = (0.9, 1.1)
NOISE_STD = 0.01


def make_loader(waveforms: Waveforms, rows: list[dict[str, str]], scale: np.ndarray,
                batch_size: int, shuffle: bool = False) -> DataLoader:
    """
    Build a single-process loader over cache rows.

    Parameters
    ----------
    waveforms : Waveforms
        Source cache.
    rows : list[dict[str, str]]
        Rows to serve.
    scale : np.ndarray
        Per-lead training scale.
    batch_size : int
        Records per batch; the last batch may be smaller.
    shuffle : bool
        Shuffle each epoch with the global torch generator.

    Returns
    -------
    DataLoader
        Loader yielding ``(signal, target)`` batches.
    """
    return DataLoader(ECGDataset(waveforms, rows, scale), batch_size=batch_size,
                      shuffle=shuffle, num_workers=0, drop_last=False)


def _check_ssl_checkpoint(state: dict[str, Any], args: argparse.Namespace,
                          train: list[dict[str, str]], scale: np.ndarray) -> None:
    if (state["model"] != args.model or state["seed"] != args.seed
            or state["epochs"] != args.ssl_epochs
            or state["training_ecg_ids"] != [r["ecg_id"] for r in train]
            or not np.allclose(scale, state["scale"])):
        raise ValueError("Existing SSL checkpoint differs from this experiment's inputs/settings")


def _ssl_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
               args: argparse.Namespace, epoch: int) -> dict[str, float]:
    """Train one SSL epoch and return record-weighted mean losses."""
    totals = {}
    observations = 0
    for signal, _ in loader:
        signal = signal.to(args.device)
        loss, details = model(signal)
        checked_step(loss, model, optimizer, f"{args.model} SSL")
        if isinstance(model, JEPA):
            model.update_teacher(TEACHER_MOMENTUM_START
                                 + TEACHER_MOMENTUM_RANGE * epoch / max(1, args.ssl_epochs - 1))
        totals["loss"] = totals.get("loss", 0) + float(loss.detach()) * len(signal)
        for key, value in details.items():
            totals[key] = totals.get(key, 0) + value * len(signal)
        observations += len(signal)
    return {key: value / observations for key, value in totals.items()}


def pretrain(args: argparse.Namespace, waveforms: Waveforms, train: list[dict[str, str]],
             scale: np.ndarray) -> Path:
    """
    Pretrain an MAE or JEPA encoder on all training waveforms, without labels.

    A completed checkpoint is reused only when its settings and training
    records match.

    Parameters
    ----------
    args : argparse.Namespace
        Runner settings from ``main``.
    waveforms : Waveforms
        Source cache.
    train : list[dict[str, str]]
        SSL training rows.
    scale : np.ndarray
        Per-lead training scale.

    Returns
    -------
    Path
        The ``encoder.pt`` checkpoint.

    Raises
    ------
    ValueError
        If an existing checkpoint was made with other inputs or settings.
    RuntimeError
        If the SSL loss or gradients become nonfinite.
    """
    seed_everything(args.seed)
    directory = args.output_dir / f"{args.model}_ssl"
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / "encoder.pt"
    if checkpoint.exists():
        _check_ssl_checkpoint(torch.load(checkpoint, map_location="cpu", weights_only=True),
                              args, train, scale)
        print(f"Using completed SSL checkpoint {checkpoint}", flush=True)
        return checkpoint
    model = (MaskedAutoencoder() if args.model == "mae" else JEPA()).to(args.device)
    loader = make_loader(waveforms, train, scale, args.ssl_batch_size, shuffle=True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=SSL_LR, weight_decay=WEIGHT_DECAY)
    history = []
    start = time.monotonic()
    for epoch in range(args.ssl_epochs):
        model.train()
        lr = warmup_cosine_lr(SSL_LR, epoch, args.ssl_epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr
        losses = _ssl_epoch(model, loader, optimizer, args, epoch)
        row = {"epoch": epoch + 1, "lr": lr, "seconds": time.monotonic() - start, **losses}
        history.append(row)
        print(json.dumps({"stage": args.model + "_ssl", **row}), flush=True)
        write_json_atomic(directory / "history.json", history)
    training_ids = [r["ecg_id"] for r in train]
    write_torch_atomic(checkpoint, {"encoder": cpu_state(model.encoder), "scale": scale.tolist(),
                                    "model": args.model, "seed": args.seed, "epochs": args.ssl_epochs,
                                    "training_records": len(train), "training_ecg_ids": training_ids})
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    write_json_atomic(directory / "config.json", {
        "architecture": repr(model), "ssl_epochs": args.ssl_epochs,
        "ssl_batch_size": args.ssl_batch_size, "seed": args.seed,
        "seconds": time.monotonic() - start, "trainable_parameters": trainable})
    return checkpoint


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader) -> np.ndarray:
    """
    Compute logits for every batch in evaluation mode.

    Parameters
    ----------
    model : nn.Module
        Classifier returning one logit per record.
    loader : DataLoader
        Loader from ``make_loader``.

    Returns
    -------
    np.ndarray
        Logits in loader order.
    """
    model.eval()
    device = next(model.parameters()).device
    return np.concatenate([model(signal.to(device)).cpu().numpy() for signal, _ in loader])


def _load_ssl_encoder(encoder: nn.Module, ssl_checkpoint: Path, args: argparse.Namespace,
                      scale: np.ndarray) -> None:
    state = torch.load(ssl_checkpoint, map_location="cpu", weights_only=True)
    expected_training = [r["ecg_id"] for r in read_csv(args.manifest_dir / "all_train_ssl.csv")]
    if state["model"] != args.model or state["training_ecg_ids"] != expected_training:
        raise ValueError("SSL checkpoint has a different model or training patient manifest")
    encoder.load_state_dict(state["encoder"])
    if not np.allclose(scale, state["scale"]):
        raise ValueError("SSL and supervised normalization do not match")


def _supervised_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                      device: str, name: str) -> float:
    """Train one supervised epoch and return the summed record loss."""
    total_loss = 0.0
    # Gentle noise/gain augmentation preserves lead relationships and rhythm timing.
    for signal, target in loader:
        signal, target = signal.to(device), target.to(device)
        gain = torch.empty(len(signal), 1, 1, device=device).uniform_(*GAIN_RANGE)
        signal = signal * gain + torch.randn_like(signal) * NOISE_STD
        loss = nn.functional.binary_cross_entropy_with_logits(model(signal), target)
        checked_step(loss, model, optimizer, f"{name} supervised")
        total_loss += float(loss.detach()) * len(signal)
    return total_loss


def _fit_supervised(args: argparse.Namespace, model: nn.Module, name: str, directory: Path,
                    train_loader: DataLoader, dev_loader: DataLoader, dev_y: np.ndarray,
                    train_records: int) -> tuple[dict[str, torch.Tensor], int, float]:
    """Train with development early stopping; return the best state, epoch and AUROC."""
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": ENCODER_LR},
        {"params": model.head.parameters(), "lr": HEAD_LR}], weight_decay=WEIGHT_DECAY)
    best_auc, best_epoch, best_state, history = -1, 0, None, []
    start = time.monotonic()
    for epoch in range(args.epochs):
        model.train()
        total_loss = _supervised_epoch(model, train_loader, optimizer, args.device, name)
        dev_auc = float(roc_auc_score(dev_y, predict(model, dev_loader)))
        if dev_auc > best_auc:
            best_auc, best_epoch, best_state = dev_auc, epoch + 1, cpu_state(model)
        row = {"epoch": epoch + 1, "loss": total_loss / train_records, "development_auroc": dev_auc,
               "seconds": time.monotonic() - start}
        history.append(row)
        print(json.dumps({"stage": name, "label_seed": args.seed, **row}), flush=True)
        write_json_atomic(directory / "history.json", history)
        if epoch + 1 - best_epoch >= args.patience:
            break
    return best_state, best_epoch, best_auc


def supervised(args: argparse.Namespace, waveforms: Waveforms, train: list[dict[str, str]],
               validation: list[dict[str, str]], test: list[dict[str, str]], scale: np.ndarray,
               ssl_checkpoint: Path | None = None) -> dict[str, Any]:
    """
    Train a classifier, select its epoch on development patients, then evaluate it.

    Validation patients are split into development (early stopping) and
    calibration (Platt scaling and threshold); test labels are used only in
    the final evaluation.

    Parameters
    ----------
    args : argparse.Namespace
        Runner settings from ``main``.
    waveforms : Waveforms
        Source cache.
    train : list[dict[str, str]]
        Labeled training rows.
    validation : list[dict[str, str]]
        Validation rows, partitioned by ``partition_validation``.
    test : list[dict[str, str]]
        Test rows.
    scale : np.ndarray
        Per-lead training scale.
    ssl_checkpoint : Path | None
        Pretrained encoder for ``mae``/``jepa`` models.

    Returns
    -------
    dict[str, Any]
        Evaluation result written to ``metrics.json``.

    Raises
    ------
    FileExistsError
        If this model and seed already have results in ``output_dir``.
    ValueError
        If the SSL checkpoint does not match the model, manifest or scale.
    RuntimeError
        If the supervised loss or gradients become nonfinite.
    """
    seed_everything(args.seed)
    name = RESULT_NAMES[args.model]
    directory = args.output_dir / f"{name}_seed{args.seed}"
    if directory.exists() and (directory / "metrics.json").exists():
        raise FileExistsError(
            f"Completed results already exist at {directory}; choose a new output directory")
    directory.mkdir(parents=True, exist_ok=True)
    encoder = CNN() if args.model == "cnn" else PatchTransformer()
    if ssl_checkpoint:
        _load_ssl_encoder(encoder, ssl_checkpoint, args, scale)
    model = Classifier(encoder).to(args.device)
    development, calibration = partition_validation(validation)
    train_loader = make_loader(waveforms, train, scale, args.batch_size, shuffle=True)
    dev_loader = make_loader(waveforms, development, scale, args.batch_size)
    dev_y = np.array([int(row["target"]) for row in development])
    start = time.monotonic()
    best_state, best_epoch, best_auc = _fit_supervised(args, model, name, directory, train_loader,
                                                       dev_loader, dev_y, len(train))
    model.load_state_dict(best_state)
    write_torch_atomic(directory / "model.pt", {"model": best_state, "scale": scale.tolist(),
                                                "best_epoch": best_epoch, "label_seed": args.seed,
                                                "architecture": name})
    # No test labels have been loaded into model fitting or checkpoint selection.
    result = evaluate_predictions(
        name, predict(model, make_loader(waveforms, calibration, scale, args.batch_size)),
        predict(model, make_loader(waveforms, test, scale, args.batch_size)),
        calibration, test, directory, args.seed, args.bootstrap)
    config = {"architecture": repr(model), "parameters": sum(p.numel() for p in model.parameters()),
              "epochs_budget": args.epochs, "patience": args.patience, "best_epoch": best_epoch,
              "best_development_auroc": best_auc, "labeled_training_records": len(train),
              "development_records": len(development), "calibration_records": len(calibration),
              "test_records": len(test), "seed": args.seed, "torch_threads": args.threads,
              "device": args.device,
              "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
              "torch_version": str(torch.__version__), "seconds": time.monotonic() - start,
              "ssl_checkpoint": str(ssl_checkpoint) if ssl_checkpoint else None,
              "label_manifest_sha256": sha256_file(args.manifest_dir / "labeled_train.csv"),
              "normalization": ("Per-record per-lead demeaning, training-only per-lead RMS scaling, "
                                "clipping +/-20"),
              "pretraining_exposure": "Own SSL uses official folds1-8 only; no validation/test waveforms"}
    write_json_atomic(directory / "config.json", config)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["cache", "ssl", "train"], default="train")
    parser.add_argument("--model", choices=["cnn", "transformer", "mae", "jepa"], default="cnn")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/ptb-xl/1.0.3"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/processed/ptbxl/seed42_fraction0.1"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/processed/ptbxl/waveforms100"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiment001"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ssl-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--ssl-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--ssl-checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    """
    Run the requested stage: build the cache, pretrain, or train and evaluate.

    Raises
    ------
    RuntimeError
        If CUDA is requested but unavailable.
    ValueError
        If the SSL stage is requested for a supervised-only model.
    """
    args = _parse_args()
    if args.stage == "ssl" and args.model not in SSL_MODELS:
        raise ValueError("SSL requires --model mae or jepa")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.stage == "cache":
        build_cache(args.raw_dir, args.manifest_dir, args.cache_dir)
        return
    require_cuda(args.device)
    waveforms = Waveforms(args.cache_dir)
    all_train = read_csv(args.manifest_dir / "all_train_ssl.csv")
    scale = waveforms.training_scale(all_train)
    ssl_checkpoint = args.ssl_checkpoint
    if args.model in SSL_MODELS and not ssl_checkpoint:
        ssl_checkpoint = pretrain(args, waveforms, all_train, scale)
    if args.stage == "ssl":
        return
    supervised(args, waveforms, read_csv(args.manifest_dir / "labeled_train.csv"),
               read_csv(args.manifest_dir / "validation.csv"),
               read_csv(args.manifest_dir / "test.csv"), scale, ssl_checkpoint)


if __name__ == "__main__":
    main()
