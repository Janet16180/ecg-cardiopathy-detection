#!/usr/bin/env python3
"""Run matched local CPC/CPC+CMSC pretraining and PTB-XL fine-tuning."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ecg_experiment.cpc import CPCPretrainer
from ecg_experiment.cpc_pool import (
    DEFAULT_CACHE,
    DEFAULT_MANIFEST,
    ROOT,
    SSL_SEED,
    Pool,
    fine_tune,
    fingerprint,
    loader,
    make_source_hashes,
    resume_or_new,
    save_epoch,
)
from ecg_experiment.evaluation import paired_comparison
from ecg_experiment.files import write_json_atomic, write_torch_atomic
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.reproducibility import cpu_state, seed_everything
from ecg_experiment.training import (
    COSINE_FLOOR,
    WARMUP_EPOCHS,
    checked_step,
    parameter_count,
    peak_gpu_gb,
    time_updates,
    warmup_cosine_lr,
)

DEFAULT_OUTPUT = ROOT / "outputs/experiment004_cpc_40k"
SSL_LR = 1e-3
WEIGHT_DECAY = 0.01
HYBRID_CMSC_WEIGHT = 0.1
PROFILE_WARMUP = 5
PRETRAINED_VARIANTS = ("cpc", "hybrid")
ALL_VARIANTS = ("scratch", "cpc", "hybrid")
COMPARISONS = (("hybrid", "cpc"), ("cpc", "scratch"), ("hybrid", "scratch"))


def pretrained_variants(variant: str) -> tuple[str, ...]:
    """
    Resolve the ``--variant`` choice to the arms that have an SSL stage.

    Parameters
    ----------
    variant : str
        ``--variant`` value.

    Returns
    -------
    tuple[str, ...]
        Pretrained arms to run.
    """
    return PRETRAINED_VARIANTS if variant == "all" else (variant,)


def ssl_settings(args: argparse.Namespace, variant: str) -> dict[str, Any]:
    """
    Fingerprinted SSL hyperparameters of one arm.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``ssl_epochs`` and ``ssl_batch_size``.
    variant : str
        ``"cpc"`` or ``"hybrid"``.

    Returns
    -------
    dict[str, Any]
        Settings entering the stage fingerprint.
    """
    return {"stage": "pretrain", "variant": variant, "seed": SSL_SEED,
            "epochs": args.ssl_epochs, "batch_size": args.ssl_batch_size,
            "optimizer": "AdamW", "lr": SSL_LR, "weight_decay": WEIGHT_DECAY,
            "warmup_epochs": WARMUP_EPOCHS, "cosine_floor": COSINE_FLOOR,
            "horizons": [4, 8, 12], "temperature": 0.1,
            "negative_exclusion_tokens": 3, "cmsc_weight": HYBRID_CMSC_WEIGHT if variant == "hybrid" else 0}


def ssl_epoch(model: CPCPretrainer, optimizer: torch.optim.Optimizer, data: DataLoader,
              device: str) -> dict[str, Any]:
    """
    Train one SSL epoch and summarize its losses.

    Parameters
    ----------
    model : CPCPretrainer
        Model in training mode.
    optimizer : torch.optim.Optimizer
        Its optimizer, with the epoch's learning rate set.
    data : DataLoader
        Shuffled training loader.
    device : str
        Device holding ``model``.

    Returns
    -------
    dict[str, Any]
        Update and record counts, then record-weighted mean losses.
    """
    totals = {"loss": 0.0, "cpc": 0.0, "cmsc": 0.0}
    count = 0
    updates = 0
    for signal, _, patients in data:
        signal = signal.to(device, non_blocking=True)
        loss, details = model(signal, patients)
        checked_step(loss, model, optimizer, "SSL")
        for name, value in {"loss": float(loss.detach()), **details}.items():
            totals[name] += value * len(signal)
        count += len(signal)
        updates += 1
    return {"optimizer_updates": updates, "record_exposures": count,
            **{key: value / count for key, value in totals.items()}}


def pretrain(args: argparse.Namespace, pool: Pool, mean: np.ndarray, std: np.ndarray,
             source_hashes: dict[str, str], variant: str) -> Path:
    """
    Pretrain one CPC arm resumably, or verify its completed encoder.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    pool : Pool
        Frozen CPC waveform cache.
    mean : np.ndarray
        Per-lead mean.
    std : np.ndarray
        Per-lead standard deviation.
    source_hashes : dict[str, str]
        Input digests from ``make_source_hashes``.
    variant : str
        ``"cpc"`` or ``"hybrid"``.

    Returns
    -------
    Path
        The completed ``encoder.pt``.

    Raises
    ------
    ValueError
        If saved configuration or encoder fingerprints differ.
    """
    seed_everything(SSL_SEED)
    directory = args.output_dir / f"{variant}_ssl"
    directory.mkdir(parents=True, exist_ok=True)
    fp, inputs = fingerprint(source_hashes, mean, std, ssl_settings(args, variant))
    config = directory / "config.json"
    if config.exists() and json.loads(config.read_text())["fingerprint"] != fp:
        raise ValueError(f"Existing pretraining config fingerprint mismatch: {directory}")
    write_json_atomic(config, {"fingerprint": fp, "inputs": inputs,
                      "description": "Local compact causal CNN/GRU CPC; not published S4 ECG-CPC"})
    complete = directory / "encoder.pt"
    if complete.exists():
        saved = torch.load(complete, map_location="cpu", weights_only=True)
        if saved["fingerprint"] != fp:
            raise ValueError(f"Completed SSL fingerprint mismatch: {complete}")
        return complete
    model = CPCPretrainer(hybrid=(variant == "hybrid")).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SSL_LR, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(SSL_SEED)
    data = loader(pool, pool.train_rows, mean, std, args.ssl_batch_size, True, generator, args.device)
    start_epoch, history, _, _, _ = resume_or_new(directory, fp, model, optimizer, generator)
    started = time.monotonic()
    previous_elapsed = history[-1]["elapsed_seconds"] if history else 0.0
    for epoch in range(start_epoch, args.ssl_epochs):
        model.train()
        lr = warmup_cosine_lr(SSL_LR, epoch, args.ssl_epochs)
        optimizer.param_groups[0]["lr"] = lr
        summary = ssl_epoch(model, optimizer, data, args.device)
        record = {"epoch": epoch + 1, "lr": lr,
                  "elapsed_seconds": previous_elapsed + time.monotonic() - started, **summary}
        history.append(record)
        save_epoch(directory, fp, epoch + 1, model, optimizer, generator, history)
        print(json.dumps({"stage": "pretrain", "variant": variant, **record}), flush=True)
    write_torch_atomic(complete, {"fingerprint": fp, "encoder": cpu_state(model.encoder),
                                  "variant": variant, "epochs": args.ssl_epochs, "seed": SSL_SEED,
                                  "training_records": len(pool.train_rows),
                                  "parameters": parameter_count(model)})
    return complete


def report(args: argparse.Namespace) -> None:
    """
    Write test metrics and paired comparisons for every completed label budget.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``output_dir``, ``ssl_epochs`` and ``bootstrap``.
    """
    comparisons = {}
    lines = ["# Local CPC experiment on the 40k ECG training pool", "",
             "Compact causal CNN/GRU CPC experiment; this is not an S4 ECG-CPC reproduction.", "",
             f"All models use the same architecture and PTB-XL label manifests. CPC and CPC+CMSC use "
             f"{args.ssl_epochs} fixed SSL epochs. The CMSC arm adds a 0.1-weighted cross-half contrastive "
             "loss. Thresholds use held-out calibration patients; CIs resample test patients.", ""]
    for budget in ("0.1", "1"):
        paths = {variant: args.output_dir / f"{variant}_fraction{budget}_seed42" for variant in ALL_VARIANTS}
        if not all((path / "completion.json").exists() for path in paths.values()):
            continue
        lines += [f"## {budget} label fraction", "", "| Arm | AUROC | AP | Sensitivity | Specificity |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for variant, path in paths.items():
            result = json.loads((path / "metrics.json").read_text())["test"]
            lines.append(f"| {variant} | {result['auroc']:.3f} | {result['average_precision']:.3f} | "
                         f"{result['sensitivity']:.3f} | {result['specificity']:.3f} |")
        lines.append("")
        for left, right in COMPARISONS:
            comparison = paired_comparison(paths[left], paths[right], args.bootstrap)
            comparisons[f"fraction{budget}_{left}_minus_{right}"] = comparison
            auc = comparison["auroc"]
            lines.append(f"{left} minus {right}: AUROC {auc['difference']:+.3f} "
                         f"(paired patient 95% CI {auc['ci95'][0]:+.3f} to {auc['ci95'][1]:+.3f}).")
        lines.append("")
    write_json_atomic(args.output_dir / "paired_comparisons.json", comparisons)
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")


def profile_step(model: CPCPretrainer, optimizer: torch.optim.Optimizer,
                 device: str) -> Callable[[tuple[Any, ...]], tuple[Any, int]]:
    """
    Build the unclipped SSL update used by the profile stage.

    Parameters
    ----------
    model : CPCPretrainer
        Model to update.
    optimizer : torch.optim.Optimizer
        Its optimizer.
    device : str
        Device holding ``model``.

    Returns
    -------
    Callable[[tuple[Any, ...]], tuple[Any, int]]
        Update returning ``(loss, details)`` and the batch size.
    """
    def step(batch: tuple[Any, ...]) -> tuple[Any, int]:
        signal, _, patients = batch
        signal = signal.to(device, non_blocking=True)
        loss, details = model(signal, patients)
        # The frozen profile measured unclipped updates.
        checked_step(loss, model, optimizer, "profile", clip=False)
        return (loss, details), len(signal)

    return step


def profile_updates(args: argparse.Namespace, pool: Pool, mean: np.ndarray, std: np.ndarray) -> None:
    """
    Measure five warmup and N real optimizer updates with no saved checkpoint.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    pool : Pool
        Frozen CPC waveform cache.
    mean : np.ndarray
        Per-lead mean.
    std : np.ndarray
        Per-lead standard deviation.
    """
    for variant in pretrained_variants(args.variant):
        seed_everything(SSL_SEED)
        model = CPCPretrainer(hybrid=(variant == "hybrid")).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=SSL_LR, weight_decay=WEIGHT_DECAY)
        generator = torch.Generator().manual_seed(SSL_SEED)
        data = loader(pool, pool.train_rows, mean, std, args.ssl_batch_size, True, generator, args.device)
        timing = time_updates(data, profile_step(model, optimizer, args.device), PROFILE_WARMUP,
                              args.profile_updates, args.device,
                              "Pool is too small for the requested profile")
        loss, details = timing.last
        seconds = timing.seconds
        measured_updates = timing.updates - PROFILE_WARMUP
        print(json.dumps({"stage": "profile", "variant": variant,
                          "model_parameters": parameter_count(model),
                          "encoder_parameters": parameter_count(model.encoder),
                          "warmup_updates": PROFILE_WARMUP, "measured_updates": measured_updates,
                          "measured_records": timing.measured_records, "measured_seconds": seconds,
                          "seconds_per_update": seconds / measured_updates,
                          "records_per_second": timing.measured_records / seconds,
                          "estimated_ssl_epoch_seconds":
                              len(pool.train_rows) / (timing.measured_records / seconds),
                          "peak_gpu_gb": peak_gpu_gb(args.device),
                          "loss": float(loss.detach()), **details}), flush=True)


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
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("profile", "pretrain", "train", "all"), default="all")
    parser.add_argument("--variant", choices=("cpc", "hybrid", "scratch", "all"), default="all")
    parser.add_argument("--labels", choices=("0.1", "1", "all"), default="all")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ssl-batch-size", type=int, default=128)
    parser.add_argument("--ssl-epochs", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--profile-updates", type=int, default=20,
                        help="Measured SSL updates after five warmup updates in profile stage")
    args = parser.parse_args(argv)
    if min(args.threads, args.batch_size, args.ssl_batch_size, args.ssl_epochs, args.epochs, args.patience,
           args.bootstrap) < 1:
        parser.error("Thread, batch, epoch, patience, and bootstrap counts must be positive")
    if args.stage in ("profile", "pretrain") and args.variant == "scratch":
        parser.error("Scratch has no profile or pretraining stage")
    if args.profile_updates < 1:
        parser.error("profile-updates must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    return args


def run_stages(args: argparse.Namespace, pool: Pool, source_hashes: dict[str, str]) -> None:
    """
    Pretrain, fine-tune and report the requested arms.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    pool : Pool
        Frozen CPC waveform cache.
    source_hashes : dict[str, str]
        Input digests from ``make_source_hashes``.
    """
    mean, std = pool.normalization(args.output_dir, source_hashes)
    if args.stage in ("pretrain", "all"):
        for variant in pretrained_variants(args.variant):
            if variant != "scratch":
                pretrain(args, pool, mean, std, source_hashes, variant)
    if args.stage in ("train", "all"):
        for budget in (("1", "0.1") if args.labels == "all" else (args.labels,)):
            for variant in (ALL_VARIANTS if args.variant == "all" else (args.variant,)):
                fine_tune(args, pool, mean, std, source_hashes, variant, budget)
        report(args)


def main(argv: Sequence[str] | None = None) -> None:
    """
    Run the requested CPC stages while holding the shared GPU lock.

    Parameters
    ----------
    argv : Sequence[str] | None
        Arguments, or ``None`` for ``sys.argv``.
    """
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    if args.stage != "profile":
        args.output_dir.mkdir(parents=True, exist_ok=True)
    with gpu_lock(args.device):
        pool = Pool(args.cache_dir)
        source_hashes = make_source_hashes(pool, args.manifest_dir)
        if args.stage != "profile":
            run_stages(args, pool, source_hashes)
            return
        with tempfile.TemporaryDirectory(prefix="cpc_profile_") as temporary:
            mean, std = pool.normalization(temporary, source_hashes)
            profile_updates(args, pool, mean, std)


if __name__ == "__main__":
    main()
