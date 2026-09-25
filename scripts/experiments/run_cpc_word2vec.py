#!/usr/bin/env python3
"""Run matched sampled InfoNCE-16 and word2vec-style SGNS-16 ECG CPC arms."""

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
from torch import nn
from torch.utils.data import DataLoader

from ecg_experiment import cpc_pool
from ecg_experiment.cpc_word2vec import SampledCPCPretrainer
from ecg_experiment.evaluation import paired_comparison
from ecg_experiment.files import sha256_file, write_json_atomic, write_torch_atomic
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.reproducibility import capture_rng_state, cpu_state, restore_rng_state, seed_everything
from ecg_experiment.training import (
    COSINE_FLOOR,
    GRADIENT_CLIP_NORM,
    WARMUP_EPOCHS,
    checked_step,
    parameter_count,
    peak_gpu_gb,
    time_updates,
    warmup_cosine_lr,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs/experiment005_cpc_word2vec"
VARIANTS = ("sampled_info", "sgns")
SAMPLER_SEED = 424242
SSL_LR = 1e-3
WEIGHT_DECAY = 0.01
PROFILE_WARMUP = 5
# Code outside cpc_pool.CODE_FILES that determines this experiment's results.
EXTRA_CODE = ("ecg_experiment/cpc_word2vec.py", "scripts/experiments/run_cpc_word2vec.py",
              "ecg_experiment/training.py", "ecg_experiment/cpc.py")
LOSS_TERMS = ("loss", "positive_score", "negative_score", "token_variance", "mean_pair_cosine")


def source_hashes_with_new_code(pool: cpc_pool.Pool, manifest_dir: Path) -> dict[str, str]:
    """
    Hash the pool, the label manifests and this experiment's own code.

    Parameters
    ----------
    pool : cpc_pool.Pool
        Frozen CPC waveform cache.
    manifest_dir : Path
        PTB-XL manifest root.

    Returns
    -------
    dict[str, str]
        SHA-256 keyed by resolved path.
    """
    hashes = cpc_pool.make_source_hashes(pool, manifest_dir)
    for name in EXTRA_CODE:
        path = ROOT / name
        hashes[str(path.resolve())] = sha256_file(path)
    return hashes


def settings(args: argparse.Namespace, variant: str) -> dict[str, Any]:
    """
    Fingerprinted SSL hyperparameters of one arm.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``ssl_epochs`` and ``ssl_batch_size``.
    variant : str
        ``"sampled_info"`` or ``"sgns"``.

    Returns
    -------
    dict[str, Any]
        Settings entering the stage fingerprint.
    """
    return {"stage": "pretrain", "variant": variant, "seed": cpc_pool.SSL_SEED,
            "sampler_seed": SAMPLER_SEED, "epochs": args.ssl_epochs,
            "batch_size": args.ssl_batch_size, "optimizer": "AdamW",
            "lr": SSL_LR, "weight_decay": WEIGHT_DECAY, "warmup_epochs": WARMUP_EPOCHS,
            "cosine_floor": COSINE_FLOOR, "horizons": [4, 8, 12],
            "temperature": 0.1, "negative_count": 16,
            "negative_sampling": ("Uniform with replacement, same half, exclude target and positions "
                                  "within +/-3 tokens"),
            "objective": ("-log(exp(pos)/(exp(pos)+sum_16 exp(neg)))" if variant == "sampled_info"
                          else "softplus(-pos)+sum_16 softplus(neg); mean over queries and horizons"),
            "normalization": "Global train-only per-lead mean/std; no augmentation"}


def resume_or_new(directory: Path, fingerprint: str, model: nn.Module, optimizer: torch.optim.Optimizer,
                  loader_generator: torch.Generator,
                  sampler_generator: torch.Generator) -> tuple[int, list[dict[str, Any]]]:
    """
    Restore the last completed SSL epoch, including the private negative sampler.

    Parameters
    ----------
    directory : Path
        Stage directory that may hold ``epoch_state.pt``.
    fingerprint : str
        Digest the saved state must match.
    model : nn.Module
        Model restored in place.
    optimizer : torch.optim.Optimizer
        Optimizer restored in place.
    loader_generator : torch.Generator
        Shuffling generator restored with the global random states.
    sampler_generator : torch.Generator
        Negative-sampling generator restored in place.

    Returns
    -------
    tuple[int, list[dict[str, Any]]]
        Completed epochs and history.

    Raises
    ------
    ValueError
        If history exists without state, or the fingerprint differs.
    """
    state_path = directory / "epoch_state.pt"
    if not state_path.exists():
        if (directory / "history.json").exists():
            raise ValueError(f"History exists without resumable state: {directory}")
        return 0, []
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if state["fingerprint"] != fingerprint:
        raise ValueError(f"Resume fingerprint mismatch: {directory}")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    restore_rng_state(state["rng"], loader_generator)
    sampler_generator.set_state(state["sampler_rng"])
    return state["epoch"], state["history"]


def save_epoch(directory: Path, fingerprint: str, epoch: int, model: nn.Module,
               optimizer: torch.optim.Optimizer, loader_generator: torch.Generator,
               sampler_generator: torch.Generator, history: list[dict[str, Any]]) -> None:
    """
    Atomically save an SSL epoch boundary and its history.

    Parameters
    ----------
    directory : Path
        Stage directory.
    fingerprint : str
        Stage digest stored with the state.
    epoch : int
        Number of completed epochs.
    model : nn.Module
        Model to save.
    optimizer : torch.optim.Optimizer
        Optimizer to save.
    loader_generator : torch.Generator
        Shuffling generator saved with the global random states.
    sampler_generator : torch.Generator
        Negative-sampling generator.
    history : list[dict[str, Any]]
        Per-epoch records, also written to ``history.json``.
    """
    write_torch_atomic(directory / "epoch_state.pt", {
        "fingerprint": fingerprint, "epoch": epoch,
        "model": cpu_state(model), "optimizer": optimizer.state_dict(),
        "rng": capture_rng_state(loader_generator),
        "sampler_rng": sampler_generator.get_state(), "history": history})
    write_json_atomic(directory / "history.json", history)


def ssl_epoch(model: SampledCPCPretrainer, optimizer: torch.optim.Optimizer, data: DataLoader,
              sampler_generator: torch.Generator, device: str) -> dict[str, Any]:
    """
    Train one SSL epoch and summarize losses and gradient clipping.

    Parameters
    ----------
    model : SampledCPCPretrainer
        Model in training mode.
    optimizer : torch.optim.Optimizer
        Its optimizer, with the epoch's learning rate set.
    data : DataLoader
        Shuffled training loader.
    sampler_generator : torch.Generator
        Negative-sampling generator.
    device : str
        Device holding ``model``.

    Returns
    -------
    dict[str, Any]
        Update counts, gradient-norm statistics and record-weighted means.
    """
    totals = dict.fromkeys(LOSS_TERMS, 0.0)
    count = updates = clipped = 0
    norm_total = 0.0
    for signal, _, _ in data:
        signal = signal.to(device, non_blocking=True)
        loss, details = model(signal, sampler_generator)
        norm = checked_step(loss, model, optimizer, "sampled CPC")
        clipped += int(float(norm) > GRADIENT_CLIP_NORM)
        norm_total += float(norm)
        updates += 1
        count += len(signal)
        for name, value in {"loss": float(loss.detach()), **details}.items():
            totals[name] += value * len(signal)
    return {"optimizer_updates": updates, "record_exposures": count,
            "mean_gradient_norm_before_clip": norm_total / updates,
            "gradient_clip_fraction": clipped / updates,
            **{name: value / count for name, value in totals.items()}}


def pretrain(args: argparse.Namespace, pool: cpc_pool.Pool, mean: np.ndarray, std: np.ndarray,
             source_hashes: dict[str, str], variant: str) -> Path:
    """
    Pretrain one sampled-objective arm resumably, or verify its completed encoder.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    pool : cpc_pool.Pool
        Frozen CPC waveform cache.
    mean : np.ndarray
        Per-lead mean.
    std : np.ndarray
        Per-lead standard deviation.
    source_hashes : dict[str, str]
        Input digests from ``source_hashes_with_new_code``.
    variant : str
        ``"sampled_info"`` or ``"sgns"``.

    Returns
    -------
    Path
        The completed ``encoder.pt``.

    Raises
    ------
    ValueError
        If saved configuration or encoder fingerprints differ.
    """
    seed_everything(cpc_pool.SSL_SEED)
    directory = args.output_dir / f"{variant}_ssl"
    directory.mkdir(parents=True, exist_ok=True)
    fp, inputs = cpc_pool.fingerprint(source_hashes, mean, std, settings(args, variant))
    config_path = directory / "config.json"
    if config_path.exists() and json.loads(config_path.read_text())["fingerprint"] != fp:
        raise ValueError(f"Existing SSL config fingerprint mismatch: {directory}")
    write_json_atomic(config_path, {"fingerprint": fp, "inputs": inputs,
                      "description": "Same compact causal CNN/GRU encoder and heads as experiment 004; "
                                     "sampled losses only",
                      "loss_scale_note": "SGNS sums sixteen negative terms; raw loss is not comparable "
                                         "to InfoNCE"})
    complete = directory / "encoder.pt"
    if complete.exists():
        saved = torch.load(complete, map_location="cpu", weights_only=True)
        if saved["fingerprint"] != fp:
            raise ValueError(f"Completed SSL fingerprint mismatch: {complete}")
        return complete
    model = SampledCPCPretrainer(variant).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SSL_LR, weight_decay=WEIGHT_DECAY)
    loader_generator = torch.Generator().manual_seed(cpc_pool.SSL_SEED)
    sampler_generator = torch.Generator().manual_seed(SAMPLER_SEED)
    data = cpc_pool.loader(pool, pool.train_rows, mean, std, args.ssl_batch_size,
                           True, loader_generator, args.device)
    start_epoch, history = resume_or_new(directory, fp, model, optimizer, loader_generator, sampler_generator)
    started = time.monotonic()
    previous_elapsed = history[-1]["elapsed_seconds"] if history else 0.0
    for epoch in range(start_epoch, args.ssl_epochs):
        model.train()
        lr = warmup_cosine_lr(SSL_LR, epoch, args.ssl_epochs)
        optimizer.param_groups[0]["lr"] = lr
        summary = ssl_epoch(model, optimizer, data, sampler_generator, args.device)
        record = {"epoch": epoch + 1, "lr": lr,
                  "elapsed_seconds": previous_elapsed + time.monotonic() - started, **summary}
        history.append(record)
        save_epoch(directory, fp, epoch + 1, model, optimizer, loader_generator, sampler_generator, history)
        print(json.dumps({"stage": "pretrain", "variant": variant, **record}), flush=True)
    write_torch_atomic(complete, {"fingerprint": fp,
                                  "encoder": cpu_state(model.encoder), "variant": variant,
                                  "epochs": args.ssl_epochs, "seed": cpc_pool.SSL_SEED,
                                  "sampler_seed": SAMPLER_SEED, "training_records": len(pool.train_rows),
                                  "parameters": parameter_count(model)})
    return complete


def profile_step(model: SampledCPCPretrainer, optimizer: torch.optim.Optimizer,
                 sampler_generator: torch.Generator, device: str,
                 ) -> Callable[[tuple[Any, ...]], tuple[Any, int]]:
    """
    Build the clipped SSL update used by the profile stage.

    Parameters
    ----------
    model : SampledCPCPretrainer
        Model to update.
    optimizer : torch.optim.Optimizer
        Its optimizer.
    sampler_generator : torch.Generator
        Negative-sampling generator.
    device : str
        Device holding ``model``.

    Returns
    -------
    Callable[[tuple[Any, ...]], tuple[Any, int]]
        Update returning ``(loss, details)`` and the batch size.
    """
    def step(batch: tuple[Any, ...]) -> tuple[Any, int]:
        signal = batch[0].to(device, non_blocking=True)
        loss, details = model(signal, sampler_generator)
        checked_step(loss, model, optimizer, "profile")
        return (loss, details), len(signal)

    return step


def profile(args: argparse.Namespace, pool: cpc_pool.Pool, mean: np.ndarray, std: np.ndarray) -> None:
    """
    Measure five warmup and N real optimizer updates per arm with no saved checkpoint.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    pool : cpc_pool.Pool
        Frozen CPC waveform cache.
    mean : np.ndarray
        Per-lead mean.
    std : np.ndarray
        Per-lead standard deviation.
    """
    for variant in (VARIANTS if args.variant == "all" else (args.variant,)):
        seed_everything(cpc_pool.SSL_SEED)
        model = SampledCPCPretrainer(variant).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=SSL_LR, weight_decay=WEIGHT_DECAY)
        loader_generator = torch.Generator().manual_seed(cpc_pool.SSL_SEED)
        sampler_generator = torch.Generator().manual_seed(SAMPLER_SEED)
        data = cpc_pool.loader(pool, pool.train_rows, mean, std, args.ssl_batch_size,
                               True, loader_generator, args.device)
        timing = time_updates(data, profile_step(model, optimizer, sampler_generator, args.device),
                              PROFILE_WARMUP, args.profile_updates, args.device,
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


def report(args: argparse.Namespace) -> None:
    """
    Write test metrics and the paired SGNS-minus-InfoNCE comparison.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``cache_dir``, ``output_dir``, ``ssl_epochs`` and ``bootstrap``.
    """
    comparisons = {}
    cache_metadata = json.loads((args.cache_dir / "complete.json").read_text())
    train_count = cache_metadata.get("split_counts", {}).get("train")
    cohort = (f"{train_count:,} combined MIMIC and PTB-XL training ECGs" if train_count is not None
              else "MIMIC plus PTB-XL training ECGs")
    lines = [f"# Sampled CPC objectives on {cohort}", "",
             "Both arms use the same compact causal CNN/GRU, temporal predictions, and sixteen sampled "
             "same-half negatives per query. Sampling permits replacement; positions within three tokens of "
             "the positive are excluded.", "",
             f"Both arms use {args.ssl_epochs} fixed SSL epochs. InfoNCE uses positive-versus-sixteen "
             "softmax classification. SGNS uses softplus(-positive) plus the sum of sixteen negative "
             "softplus terms. Their raw losses have different scales and class-prior optima; compare "
             "downstream metrics.", ""]
    lines += ["This is a one-seed exploratory comparison on the PTB-XL test set already used by earlier "
              "project experiments; the test set is not a fresh external confirmation cohort.", ""]
    for budget in ("1", "0.1"):
        paths = {variant: args.output_dir / f"{variant}_fraction{budget}_seed42" for variant in VARIANTS}
        if not all((path / "completion.json").exists() for path in paths.values()):
            continue
        lines += [f"## {budget} label fraction", "",
                  "| Arm | AUROC | AP | Sensitivity | Specificity |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for variant, path in paths.items():
            result = json.loads((path / "metrics.json").read_text())["test"]
            lines.append(f"| {variant} | {result['auroc']:.3f} | {result['average_precision']:.3f} | "
                         f"{result['sensitivity']:.3f} | {result['specificity']:.3f} |")
        comparison = paired_comparison(paths["sgns"], paths["sampled_info"], args.bootstrap)
        comparisons[f"fraction{budget}_sgns_minus_sampled_info"] = comparison
        lines += ["", "SGNS minus sampled InfoNCE (paired test-patient bootstrap):", ""]
        for name, result in comparison.items():
            lines.append(f"- {name}: {result['difference']:+.3f} "
                         f"(95% CI {result['ci95'][0]:+.3f} to {result['ci95'][1]:+.3f})")
        lines.append("")
    write_json_atomic(args.output_dir / "paired_comparisons.json", comparisons)
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")


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
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), default="all")
    parser.add_argument("--labels", choices=("0.1", "1", "all"), default="all")
    parser.add_argument("--cache-dir", type=Path, default=cpc_pool.DEFAULT_CACHE)
    parser.add_argument("--manifest-dir", type=Path, default=cpc_pool.DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ssl-batch-size", type=int, default=128)
    parser.add_argument("--ssl-epochs", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--profile-updates", type=int, default=20)
    args = parser.parse_args(argv)
    if min(args.threads, args.batch_size, args.ssl_batch_size, args.ssl_epochs,
           args.epochs, args.patience, args.bootstrap, args.profile_updates) < 1:
        parser.error("Thread, batch, epoch, patience, bootstrap, and profile counts must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    return args


def run_stages(args: argparse.Namespace, pool: cpc_pool.Pool, source_hashes: dict[str, str]) -> None:
    """
    Pretrain, fine-tune and report the requested arms.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    pool : cpc_pool.Pool
        Frozen CPC waveform cache.
    source_hashes : dict[str, str]
        Input digests from ``source_hashes_with_new_code``.
    """
    variants = VARIANTS if args.variant == "all" else (args.variant,)
    mean, std = pool.normalization(args.output_dir, source_hashes)
    if args.stage in ("pretrain", "all"):
        for variant in variants:
            pretrain(args, pool, mean, std, source_hashes, variant)
    if args.stage in ("train", "all"):
        for budget in (("1", "0.1") if args.labels == "all" else (args.labels,)):
            for variant in variants:
                cpc_pool.fine_tune(args, pool, mean, std, source_hashes, variant, budget)
        report(args)


def main(argv: Sequence[str] | None = None) -> None:
    """
    Run the requested stages while holding the shared GPU lock.

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
        pool = cpc_pool.Pool(args.cache_dir)
        source_hashes = source_hashes_with_new_code(pool, args.manifest_dir)
        if args.stage != "profile":
            run_stages(args, pool, source_hashes)
            return
        with tempfile.TemporaryDirectory(prefix="cpc_word2vec_profile_") as temporary:
            mean, std = pool.normalization(temporary, source_hashes)
            profile(args, pool, mean, std)


if __name__ == "__main__":
    main()
