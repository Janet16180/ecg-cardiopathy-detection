#!/usr/bin/env python3
"""Experiment010: matched full, within-lead, and cross-lead CPC continuation."""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ecg_experiment import cpc_pool
from ecg_experiment.cpc_crosslead import AUX_WEIGHT, GROUP_A, GROUP_B, VARIANTS, CrossLeadPretrainer
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
from scripts.experiments.run_cpc_tokenization import bootstrap

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs/experiment010_cpc_crosslead"
DEFAULT_BOOTSTRAP = ROOT / "outputs/experiment004_cpc_40k/cpc_ssl"
PROFILE_WARMUP = 2
MIN_PROFILE_UPDATES = 5
PROTOCOL_SSL_EPOCHS = 10
SSL_LR = 1e-3
WEIGHT_DECAY = 0.01
# Code outside cpc_pool.CODE_FILES that determines this experiment's results;
# bootstrap() is imported from the tokenization runner.
EXTRA_CODE = ("ecg_experiment/cpc_crosslead.py", "scripts/experiments/run_cpc_crosslead.py",
              "scripts/experiments/run_cpc_tokenization.py", "ecg_experiment/training.py",
              "ecg_experiment/cpc.py")
LOSS_TERMS = ("loss", "full_cpc", "aux_a", "aux_b", "token_variance",
              "adjacent_token_cosine", "view_a_token_variance", "view_b_token_variance")
COMPARISONS = (("crosslead", "withinlead"), ("withinlead", "native"))
Roundtrip = Callable[[nn.Module, torch.optim.Optimizer, torch.Tensor, str, torch.Generator], bool]


def source_hashes(args: argparse.Namespace, pool: cpc_pool.Pool) -> dict[str, str]:
    """
    Hash the pool, label manifests, bootstrap checkpoint and this experiment's code.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``manifest_dir`` and ``bootstrap_dir``.
    pool : cpc_pool.Pool
        Frozen CPC waveform cache.

    Returns
    -------
    dict[str, str]
        SHA-256 keyed by resolved path.
    """
    hashes = cpc_pool.make_source_hashes(pool, args.manifest_dir)
    paths = [ROOT / name for name in EXTRA_CODE]
    paths += [args.bootstrap_dir / name for name in ("encoder.pt", "epoch_state.pt", "config.json")]
    paths.append(args.bootstrap_dir.parent / "normalization.json")
    hashes.update({str(path.resolve()): sha256_file(path) for path in paths})
    return hashes


def fixed_normalization(pool: cpc_pool.Pool, args: argparse.Namespace, hashes: dict[str, str],
                        bootstrap_config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """
    Read the exact 004 normalization; ``Pool.normalization`` validates its source.

    Parameters
    ----------
    pool : cpc_pool.Pool
        Frozen CPC waveform cache.
    args : argparse.Namespace
        Needs ``bootstrap_dir``.
    hashes : dict[str, str]
        Input digests from ``source_hashes``.
    bootstrap_config : dict[str, Any]
        Experiment 004 SSL configuration.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Per-lead mean and standard deviation.

    Raises
    ------
    FileNotFoundError
        If the 004 normalization is missing.
    ValueError
        If it differs from the statistics fingerprinted by 004.
    """
    path = args.bootstrap_dir.parent / "normalization.json"
    if not path.exists():
        raise FileNotFoundError(f"Experiment004 normalization is missing: {path}")
    mean, std = pool.normalization(path.parent, hashes)
    original = bootstrap_config["inputs"]["normalization"]
    if mean.tolist() != original["mean"] or std.tolist() != original["std"]:
        # 004 fingerprints float32 arrays; do not silently use a changed file.
        raise ValueError("Experiment004 normalization differs from bootstrap config")
    return mean, std


def ssl_settings(args: argparse.Namespace, variant: str) -> dict[str, Any]:
    """
    Fingerprinted SSL hyperparameters of one arm.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``ssl_epochs`` and ``ssl_batch_size``.
    variant : str
        One of ``VARIANTS``.

    Returns
    -------
    dict[str, Any]
        Settings entering the stage fingerprint.
    """
    return {"stage": "pretrain", "variant": variant, "seed": cpc_pool.SSL_SEED,
            "epochs": args.ssl_epochs, "batch_size": args.ssl_batch_size,
            "bootstrap": "Experiment004 CPC completed epoch20 encoder and all three heads",
            "optimizer": "AdamW", "lr": SSL_LR, "weight_decay": WEIGHT_DECAY,
            "warmup_epochs": WARMUP_EPOCHS, "cosine_floor": COSINE_FLOOR, "horizons": [4, 8, 12],
            "temperature": 0.1, "negative_exclusion_tokens": 3,
            "views": {"full": list(range(12)), "A": list(GROUP_A), "B": list(GROUP_B)},
            "pass_order": ["full", "A", "B"],
            "auxiliary_weight": 0.0 if variant == "native" else AUX_WEIGHT,
            "auxiliary_target": "opposite lead group" if variant == "crosslead" else "same lead group",
            "normalization": ("exact Experiment004 train-only normalization; zero absent leads after "
                              "normalization")}


def ssl_fingerprint(args: argparse.Namespace, pool: cpc_pool.Pool, mean: np.ndarray, std: np.ndarray,
                    hashes: dict[str, str], variant: str) -> tuple[str, dict[str, Any]]:
    """
    Fingerprint one arm's SSL stage.

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
    hashes : dict[str, str]
        Input digests from ``source_hashes``.
    variant : str
        One of ``VARIANTS``.

    Returns
    -------
    tuple[str, dict[str, Any]]
        Digest and fingerprinted inputs.
    """
    return cpc_pool.fingerprint(hashes, mean, std, ssl_settings(args, variant))


def profile_path(args: argparse.Namespace, variant: str) -> Path:
    """
    Location of one arm's CUDA profile receipt.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``output_dir``.
    variant : str
        One of ``VARIANTS``.

    Returns
    -------
    Path
        ``profile_<variant>.json`` in the output directory.
    """
    return args.output_dir / f"profile_{variant}.json"


def require_profile(args: argparse.Namespace, variant: str, fingerprint: str) -> None:
    """
    Require a completed CUDA profile, with resume check, of these exact inputs.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``output_dir``.
    variant : str
        One of ``VARIANTS``.
    fingerprint : str
        Current SSL fingerprint of the arm.

    Raises
    ------
    RuntimeError
        If no profile exists.
    ValueError
        If the profile is for other inputs, another device or is incomplete.
    """
    path = profile_path(args, variant)
    if not path.exists():
        raise RuntimeError(f"CUDA SSL requires a matching completed profile: {path}")
    record = json.loads(path.read_text())
    if (record.get("fingerprint") != fingerprint or record.get("device") != "cuda"
            or record.get("measured_updates", 0) < MIN_PROFILE_UPDATES or not record.get("resume_roundtrip")):
        raise ValueError(f"CUDA profile gate does not match inputs or is incomplete: {path}")


def new_model(variant: str, encoder: dict[str, torch.Tensor], heads: dict[str, torch.Tensor],
              device: str) -> CrossLeadPretrainer:
    """
    Build one arm from the Experiment 004 encoder and prediction heads.

    Parameters
    ----------
    variant : str
        One of ``VARIANTS``.
    encoder : dict[str, torch.Tensor]
        Bootstrap encoder state.
    heads : dict[str, torch.Tensor]
        Bootstrap prediction-head state.
    device : str
        Target device.

    Returns
    -------
    CrossLeadPretrainer
        Model on ``device``.
    """
    model = CrossLeadPretrainer(variant)
    model.load_bootstrap(encoder, heads)
    return model.to(device)


def crosslead_step(model: nn.Module, optimizer: torch.optim.Optimizer,
                   signal: torch.Tensor) -> tuple[float, dict[str, float], float]:
    """
    Take one checked, clipped update and require finite parameters afterwards.

    Parameters
    ----------
    model : nn.Module
        Pretrainer returning ``(loss, details)``.
    optimizer : torch.optim.Optimizer
        Its optimizer.
    signal : torch.Tensor
        Normalized batch on the model's device.

    Returns
    -------
    tuple[float, dict[str, float], float]
        Loss, loss details and the gradient norm before clipping.

    Raises
    ------
    RuntimeError
        If the loss, gradient norm or updated parameters are nonfinite.
    """
    loss, details = model(signal)
    grad_norm = checked_step(loss, model, optimizer, "cross-lead CPC")
    if not torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all():
        raise RuntimeError("Nonfinite cross-lead CPC parameters")
    return float(loss.detach()), details, float(grad_norm)


def profile_roundtrip(model: nn.Module, optimizer: torch.optim.Optimizer, signal: torch.Tensor, variant: str,
                      generator: torch.Generator) -> bool:
    """
    Verify a saved model/optimizer/RNG state reproduces the next dropout update bitwise.

    Parameters
    ----------
    model : nn.Module
        Profiled model.
    optimizer : torch.optim.Optimizer
        Its optimizer, with populated moments.
    signal : torch.Tensor
        Batch used for the compared update.
    variant : str
        One of ``VARIANTS``.
    generator : torch.Generator
        Loader generator saved with the random states.

    Returns
    -------
    bool
        True when both updates agree; the random states are restored afterwards.

    Raises
    ------
    AssertionError
        If any resumed tensor differs.
    """
    snapshot = {"model": cpu_state(model), "optimizer": copy.deepcopy(optimizer.state_dict()),
                "rng": capture_rng_state(generator)}
    duplicate = CrossLeadPretrainer(variant).to(signal.device)
    duplicate.load_state_dict(snapshot["model"])
    copy_optimizer = torch.optim.AdamW(duplicate.parameters(), lr=SSL_LR, weight_decay=WEIGHT_DECAY)
    copy_optimizer.load_state_dict(snapshot["optimizer"])
    restore_rng_state(snapshot["rng"], generator)
    crosslead_step(model, optimizer, signal)
    expected = cpu_state(model)
    restore_rng_state(snapshot["rng"], generator)
    crosslead_step(duplicate, copy_optimizer, signal)
    for key, tensor in expected.items():
        torch.testing.assert_close(tensor, duplicate.state_dict()[key].cpu(), atol=0, rtol=0)
    # Restore so the profile itself has no hidden difference after the check.
    restore_rng_state(snapshot["rng"], generator)
    return True


def profile_step(model: nn.Module, optimizer: torch.optim.Optimizer,
                 device: str) -> Callable[[tuple[Any, ...]], tuple[Any, int]]:
    """
    Build the profiled update, which also keeps its batch for the resume check.

    Parameters
    ----------
    model : nn.Module
        Model to update.
    optimizer : torch.optim.Optimizer
        Its optimizer.
    device : str
        Device holding ``model``.

    Returns
    -------
    Callable[[tuple[Any, ...]], tuple[Any, int]]
        Update returning ``(loss, details, grad_norm, signal)`` and the batch size.
    """
    def step(batch: tuple[Any, ...]) -> tuple[Any, int]:
        signal = batch[0].to(device, non_blocking=True)
        return (*crosslead_step(model, optimizer, signal), signal), len(signal)

    return step


def profile(args: argparse.Namespace, pool: cpc_pool.Pool, mean: np.ndarray, std: np.ndarray,
            hashes: dict[str, str], bootstrap_state: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
            variants: Sequence[str], roundtrip: Roundtrip = profile_roundtrip) -> None:
    """
    Time real updates per arm, check resume reproducibility and write the profile gate.

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
    hashes : dict[str, str]
        Input digests from ``source_hashes``.
    bootstrap_state : tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
        Bootstrap encoder and prediction-head states.
    variants : Sequence[str]
        Arms to profile.
    roundtrip : Roundtrip
        Resume check run after the measured updates.

    Raises
    ------
    RuntimeError
        If the pool has fewer batches than the requested profile.
    """
    encoder, heads = bootstrap_state
    for variant in variants:
        seed_everything(cpc_pool.SSL_SEED)
        model = new_model(variant, encoder, heads, args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=SSL_LR, weight_decay=WEIGHT_DECAY)
        generator = torch.Generator().manual_seed(cpc_pool.SSL_SEED)
        data = cpc_pool.loader(pool, pool.train_rows, mean, std, args.ssl_batch_size,
                               True, generator, args.device)
        model.train()
        too_small = "Pool is too small for the requested profile"
        timing = time_updates(data, profile_step(model, optimizer, args.device), PROFILE_WARMUP,
                              args.profile_updates, args.device, too_small)
        if timing.updates < PROFILE_WARMUP + args.profile_updates:
            raise RuntimeError(too_small)
        loss, details, grad_norm, last_signal = timing.last
        seconds = timing.seconds
        resumed = roundtrip(model, optimizer, last_signal, variant, generator)
        fp, inputs = ssl_fingerprint(args, pool, mean, std, hashes, variant)
        record = {"stage": "profile", "variant": variant, "fingerprint": fp,
                  "inputs": inputs, "device": args.device,
                  "model_parameters": parameter_count(model),
                  "encoder_parameters": parameter_count(model.encoder),
                  "warmup_updates": PROFILE_WARMUP, "measured_updates": args.profile_updates,
                  "measured_records": timing.measured_records, "measured_seconds": seconds,
                  "seconds_per_update": seconds / args.profile_updates,
                  "estimated_ssl_epoch_seconds": len(pool.train_rows) * seconds / timing.measured_records,
                  "peak_gpu_gb": peak_gpu_gb(args.device),
                  "resume_roundtrip": resumed, "loss": loss, "gradient_norm": grad_norm, **details}
        write_json_atomic(profile_path(args, variant), record)
        print(json.dumps({k: v for k, v in record.items() if k != "inputs"}), flush=True)


def resume_or_new(directory: Path, fingerprint: str, model: nn.Module, optimizer: torch.optim.Optimizer,
                  generator: torch.Generator) -> tuple[int, list[dict[str, Any]]]:
    """
    Restore the last completed SSL epoch, or start fresh.

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
    generator : torch.Generator
        Loader generator restored with the global random states.

    Returns
    -------
    tuple[int, list[dict[str, Any]]]
        Completed epochs and history.

    Raises
    ------
    ValueError
        If history exists without state, or the fingerprint differs.
    """
    path = directory / "epoch_state.pt"
    if not path.exists():
        if (directory / "history.json").exists():
            raise ValueError(f"History exists without resumable state: {directory}")
        return 0, []
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["fingerprint"] != fingerprint:
        raise ValueError(f"Resume fingerprint mismatch: {directory}")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    restore_rng_state(state["rng"], generator)
    return state["epoch"], state["history"]


def save_epoch(directory: Path, fingerprint: str, epoch: int, model: nn.Module,
               optimizer: torch.optim.Optimizer, generator: torch.Generator,
               history: list[dict[str, Any]]) -> None:
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
    generator : torch.Generator
        Loader generator saved with the global random states.
    history : list[dict[str, Any]]
        Per-epoch records, also written to ``history.json``.
    """
    write_torch_atomic(directory / "epoch_state.pt", {
        "fingerprint": fingerprint, "epoch": epoch, "model": cpu_state(model),
        "optimizer": optimizer.state_dict(), "rng": capture_rng_state(generator),
        "history": history})
    write_json_atomic(directory / "history.json", history)


def ssl_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, data: DataLoader | Sequence[Any],
              device: str) -> dict[str, Any]:
    """
    Train one SSL epoch and summarize losses and gradient clipping.

    Parameters
    ----------
    model : nn.Module
        Pretrainer in training mode.
    optimizer : torch.optim.Optimizer
        Its optimizer, with the epoch's learning rate set.
    data : DataLoader | Sequence[Any]
        Shuffled training batches.
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
        loss, details, norm = crosslead_step(model, optimizer, signal)
        clipped += int(norm > GRADIENT_CLIP_NORM)
        norm_total += norm
        updates += 1
        count += len(signal)
        for name, value in {"loss": loss, **details}.items():
            totals[name] += value * len(signal)
    return {"optimizer_updates": updates, "record_exposures": count,
            "mean_gradient_norm_before_clip": norm_total / updates,
            "gradient_clip_fraction": clipped / updates,
            **{name: value / count for name, value in totals.items()}}


def pretrain(args: argparse.Namespace, pool: cpc_pool.Pool, mean: np.ndarray, std: np.ndarray,
             hashes: dict[str, str], bootstrap_state: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]],
             variant: str) -> Path:
    """
    Continue one arm from the 004 bootstrap resumably, or verify its completed encoder.

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
    hashes : dict[str, str]
        Input digests from ``source_hashes``.
    bootstrap_state : tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]
        Bootstrap encoder and prediction-head states.
    variant : str
        One of ``VARIANTS``.

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
    fp, inputs = ssl_fingerprint(args, pool, mean, std, hashes, variant)
    if args.device == "cuda":
        require_profile(args, variant, fp)
    config_path = directory / "config.json"
    if config_path.exists() and json.loads(config_path.read_text())["fingerprint"] != fp:
        raise ValueError(f"Existing SSL config fingerprint mismatch: {directory}")
    write_json_atomic(config_path, {"fingerprint": fp, "inputs": inputs})
    complete = directory / "encoder.pt"
    if complete.exists():
        saved = torch.load(complete, map_location="cpu", weights_only=True)
        if saved["fingerprint"] != fp or saved["variant"] != variant or saved["epochs"] != args.ssl_epochs:
            raise ValueError(f"Completed SSL checkpoint mismatch: {complete}")
        return complete
    model = new_model(variant, *bootstrap_state, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SSL_LR, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(cpc_pool.SSL_SEED)
    data = cpc_pool.loader(pool, pool.train_rows, mean, std, args.ssl_batch_size,
                           True, generator, args.device)
    start_epoch, history = resume_or_new(directory, fp, model, optimizer, generator)
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
                                  "variant": variant, "epochs": args.ssl_epochs, "seed": cpc_pool.SSL_SEED,
                                  "training_records": len(pool.train_rows),
                                  "parameters": parameter_count(model)})
    return complete


def require_all_ssl(args: argparse.Namespace, pool: cpc_pool.Pool, mean: np.ndarray, std: np.ndarray,
                    hashes: dict[str, str]) -> None:
    """
    Require every arm's SSL stage to be complete before any transfer starts.

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
    hashes : dict[str, str]
        Input digests from ``source_hashes``.

    Raises
    ------
    ValueError
        If an arm is incomplete, changed, or its encoder disagrees with its final state.
    """
    for variant in VARIANTS:
        fp, _ = ssl_fingerprint(args, pool, mean, std, hashes, variant)
        directory = args.output_dir / f"{variant}_ssl"
        config = json.loads((directory / "config.json").read_text())
        saved = torch.load(directory / "encoder.pt", map_location="cpu", weights_only=True)
        state = torch.load(directory / "epoch_state.pt", map_location="cpu", weights_only=False)
        if (config["fingerprint"] != fp or saved["fingerprint"] != fp
                or state["fingerprint"] != fp or state["epoch"] != args.ssl_epochs
                or saved["epochs"] != args.ssl_epochs or saved["variant"] != variant):
            raise ValueError(f"Incomplete or changed SSL arm: {variant}")
        encoder = {key.removeprefix("encoder."): value for key, value in state["model"].items()
                   if key.startswith("encoder.")}
        if encoder.keys() != saved["encoder"].keys() or any(
                not torch.equal(value, saved["encoder"][key]) for key, value in encoder.items()):
            raise ValueError(f"SSL final encoder and epoch state disagree: {variant}")


def report(args: argparse.Namespace) -> None:
    """
    Write test metrics and paired comparisons for every completed label budget.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``output_dir`` and ``bootstrap``.
    """
    comparisons = {}
    lines = ["# Experiment010: cross-lead CPC", "",
             "Three arms continue the same completed Experiment004 epoch20 CPC encoder and three prediction "
             "heads for ten SSL epochs. All use full twelve-lead CPC. Within-lead and cross-lead add "
             "0.1-weighted CPC on masked A and B views; cross-lead uses the opposite view's future "
             "targets.", "",
             "This is a one-seed exploratory comparison on the previously used PTB-XL test set, not a fresh "
             "external confirmation cohort. Intervals resample test patients in matched pairs.", ""]
    for budget in ("1", "0.1"):
        paths = {variant: args.output_dir / f"{variant}_fraction{budget}_seed42" for variant in VARIANTS}
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
            lines += [f"{left} minus {right} (paired patient bootstrap):", ""]
            for name, result in comparison.items():
                lines.append(f"- {name}: {result['difference']:+.3f} "
                             f"(95% CI {result['ci95'][0]:+.3f} to {result['ci95'][1]:+.3f})")
            lines.append("")
    write_json_atomic(args.output_dir / "paired_comparisons.json", comparisons)
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")


def check_stage(args: argparse.Namespace, pool: cpc_pool.Pool, mean: np.ndarray, std: np.ndarray,
                hashes: dict[str, str], variants: Sequence[str]) -> None:
    """
    Print each arm's fingerprint and readiness without training.

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
    hashes : dict[str, str]
        Input digests from ``source_hashes``.
    variants : Sequence[str]
        Arms to check.
    """
    for variant in variants:
        fp, _ = ssl_fingerprint(args, pool, mean, std, hashes, variant)
        print(json.dumps({"stage": "check", "variant": variant, "fingerprint": fp,
                          "profile_ready": profile_path(args, variant).exists(),
                          "ssl_complete": (args.output_dir / f"{variant}_ssl/encoder.pt").exists()}),
              flush=True)


def check_protocol(args: argparse.Namespace) -> None:
    """
    Reject settings that break the fixed Experiment 010 protocol.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Raises
    ------
    ValueError
        If the SSL duration differs or ``all`` omits arms.
    RuntimeError
        If CUDA is requested but unavailable.
    """
    if args.ssl_epochs != PROTOCOL_SSL_EPOCHS:
        raise ValueError("Experiment010 protocol requires exactly ten continuation epochs")
    if args.stage == "all" and args.variant != "all":
        raise ValueError("The all stage requires all three SSL arms before transfer")
    if args.device == "cuda" and args.stage != "check" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")


def run(args: argparse.Namespace, roundtrip: Roundtrip = profile_roundtrip) -> None:
    """
    Run the requested stage under the protocol's fixed settings.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    roundtrip : Roundtrip
        Resume check used by the profile stage.

    """
    check_protocol(args)
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with gpu_lock("cpu" if args.stage == "check" else args.device):
        pool = cpc_pool.Pool(args.cache_dir)
        hashes = source_hashes(args, pool)
        encoder, heads, bootstrap_config = bootstrap(args, hashes)
        mean, std = fixed_normalization(pool, args, hashes, bootstrap_config)
        variants = VARIANTS if args.variant == "all" else (args.variant,)
        if args.stage == "check":
            check_stage(args, pool, mean, std, hashes, variants)
            return
        if args.stage == "profile":
            profile(args, pool, mean, std, hashes, (encoder, heads), variants, roundtrip)
            return
        if args.stage in ("pretrain", "all"):
            for variant in variants:
                pretrain(args, pool, mean, std, hashes, (encoder, heads), variant)
        if args.stage in ("train", "all"):
            require_all_ssl(args, pool, mean, std, hashes)
            for budget in (("1", "0.1") if args.labels == "all" else (args.labels,)):
                for variant in variants:
                    cpc_pool.fine_tune(args, pool, mean, std, hashes, variant, budget)
            report(args)


def main(argv: Sequence[str] | None = None, roundtrip: Roundtrip = profile_roundtrip) -> None:
    """
    Parse the command line and run the requested stage.

    Parameters
    ----------
    argv : Sequence[str] | None
        Arguments, or ``None`` for ``sys.argv``.
    roundtrip : Roundtrip
        Resume check used by the profile stage.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("check", "profile", "pretrain", "train", "all"), default="all")
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), default="all")
    parser.add_argument("--labels", choices=("0.1", "1", "all"), default="all")
    parser.add_argument("--cache-dir", type=Path, default=cpc_pool.DEFAULT_CACHE)
    parser.add_argument("--manifest-dir", type=Path, default=cpc_pool.DEFAULT_MANIFEST)
    parser.add_argument("--bootstrap-dir", type=Path, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ssl-batch-size", type=int, default=128)
    parser.add_argument("--ssl-epochs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--profile-updates", type=int, default=5)
    args = parser.parse_args(argv)
    if min(args.threads, args.batch_size, args.ssl_batch_size, args.ssl_epochs,
           args.epochs, args.patience, args.bootstrap) < 1 or args.profile_updates < MIN_PROFILE_UPDATES:
        parser.error("Counts must be positive and profile-updates must be at least five")
    run(args, roundtrip)


if __name__ == "__main__":
    main()
