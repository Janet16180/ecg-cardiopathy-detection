#!/usr/bin/env python3
"""Experiment 017: matched causal morphology-template CPC development pilot."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F  # noqa: N812 - conventional alias

from ecg_experiment.bounded_waveform_cache import BoundedWaveformCache
from ecg_experiment.cpc import CPCClassifier
from ecg_experiment.cpc_morphology import SUPPORT, TEMPLATES, MorphologyCPCClassifier
from ecg_experiment.cpc_pool import Pool
from ecg_experiment.evaluation import FULL_LABELS, LIMITED_LABELS
from ecg_experiment.files import sha256_file, sha256_json, write_json_atomic, write_torch_atomic
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.pilot import (
    BATCH,
    BUDGETS,
    DEFAULT_MAX_CACHE_BYTES,
    DEFAULT_RESERVE_BYTES,
    INTERRUPTED_EXIT,
    MIN_CACHE_BYTES,
    SAVE_EVERY,
    Partitions,
    Progress,
    check_final_ssl,
    check_roundtrip,
    check_waveform_sample,
    clipped_sigmoid,
    development_batches,
    exposed_labels,
    file_identity,
    fixed_batches,
    fold_operating_point,
    install_stop_handler,
    interrupted,
    labels_and_patients,
    load_partitions,
    load_state,
    normalized_batch,
    profile_arms,
    run_pilot,
    save_state,
)
from ecg_experiment.receipts import check_completion, check_existing_config, require_receipt, write_completion
from ecg_experiment.reproducibility import capture_rng_state, cpu_state, seed_everything
from ecg_experiment.training import checked_step, parameter_count

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data/processed/cpc_pool_40k"
MANIFEST = ROOT / "data/processed/ptbxl"
SSL = ROOT / "outputs/experiment004_cpc_40k/cpc_ssl/encoder.pt"
NORMALIZATION = ROOT / "outputs/experiment004_cpc_40k/normalization.json"
OUTPUT = ROOT / "outputs/experiment017_morphology_templates"
POOL_VERIFICATION = ROOT / "outputs/experiment015_jepa_cpc_distillation/provenance/verification.json"
POOL_VERIFIER = "scripts/experiments/run_jepa_cpc_distillation.py"
POOL_FILES = (("signals.npy", "signals_sha256"), ("rows.csv", "rows_sha256"),
              ("ecg_ids.npy", "ecg_ids_sha256"))
SEED = 42
# Offset of the common first-dropout stream; see make_model.
DROPOUT_SEED_OFFSET = 12345
TEMPLATE_SEED_OFFSET = 17017
EPOCHS = 5
ENCODER_LR = 3e-4
HEAD_LR = 1e-3
WEIGHT_DECAY = 0.01
KINDS = ("none", "conv", "template")
HALF_SAMPLES = 1250
INITIAL_BEST = {"auc": -1.0, "epoch": 0}
ARTIFACTS = {"history.json": "history_sha256", "best_model.pt": "best_model_sha256"}
PLANNING_GATE_SECONDS = 7200
REQUIRED_GAIN = 0.002
SENSITIVITY_TOLERANCE = 0.005
CODE_PATHS = ("scripts/experiments/run_cpc_morphology017.py", "ecg_experiment/cpc_morphology.py",
              "ecg_experiment/cpc.py", "ecg_experiment/bounded_waveform_cache.py",
              "scripts/experiments/run_cpc_experiment.py", "ecg_experiment/data.py",
              "ecg_experiment/cpc_pool.py", "ecg_experiment/files.py",
              "ecg_experiment/reproducibility.py", "ecg_experiment/pilot.py", "ecg_experiment/receipts.py",
              "ecg_experiment/training.py",
              "ecg_experiment/run.py", "ecg_experiment/evaluation.py",
              "docs/experiment-017-morphology.md")


@dataclass
class PilotData:
    """
    Verified inputs of the pilot, plus the preloaded waveforms once training starts.

    Attributes
    ----------
    pool : Pool
        Frozen CPC waveform cache.
    partitions : Partitions
        Frozen PTB-XL partitions.
    mean : np.ndarray
        Train-only per-lead mean shaped ``[12, 1]``.
    std : np.ndarray
        Train-only per-lead standard deviation shaped ``[12, 1]``.
    provenance : dict[str, Any]
        Source, code and settings identity of the experiment.
    pool_receipt_sha256 : str
        Digest of the Experiment 015 pool receipt, recorded for audit.
    precheck_seconds : float
        Wall time of input verification.
    waveforms : BoundedWaveformCache | None
        Training then development waveforms in RAM.
    bank : torch.Tensor | None
        Initial template bank.
    template_receipt : list[dict[str, Any]] | None
        Source window of each initial template.
    preload_seconds : float
        Wall time of the preload.
    """

    pool: Pool
    partitions: Partitions
    mean: np.ndarray
    std: np.ndarray
    provenance: dict[str, Any]
    pool_receipt_sha256: str
    precheck_seconds: float
    waveforms: BoundedWaveformCache | None = None
    bank: torch.Tensor | None = None
    template_receipt: list[dict[str, Any]] | None = None
    preload_seconds: float = 0.0


def verified_pool_hashes(args: argparse.Namespace,
                         pool: Pool) -> tuple[dict[str, str], dict[str, dict[str, int]], str, str]:
    """
    Reuse Experiment 015's complete pool hash check while every source file is unchanged.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``cache_dir`` and ``pool_verification``.
    pool : Pool
        Frozen CPC waveform cache.

    Returns
    -------
    tuple[dict[str, str], dict[str, dict[str, int]], str, str]
        Pool content digests, pool file identities, the verifier source digest
        and the receipt digest.

    Raises
    ------
    FileNotFoundError
        If the receipt is missing.
    ValueError
        If the receipt, its verifier source or any pool file changed.
    """
    if not args.pool_verification.exists():
        raise FileNotFoundError(f"Verified immutable CPC pool receipt is missing: {args.pool_verification}")
    before = {name: file_identity(args.cache_dir / name) for name, _ in POOL_FILES}
    receipt = json.loads(args.pool_verification.read_text())
    if (receipt.get("stage") != "check"
            or receipt.get("fingerprint") != sha256_json(receipt.get("provenance"))):
        raise ValueError("Invalid pool verification receipt")
    source_hash = receipt.get("provenance", {}).get("code", {}).get(POOL_VERIFIER)
    if source_hash != sha256_file(ROOT / POOL_VERIFIER):
        raise ValueError("Frozen pool verifier source changed")
    hashes = receipt["provenance"]["pool_content_sha256"]
    for filename, key in POOL_FILES:
        if (receipt["pool_file_stats"].get(filename) != before[filename]
                or hashes.get(filename) != pool.metadata[key]):
            raise ValueError(f"CPC pool differs from verified receipt: {filename}")
    if any(file_identity(args.cache_dir / name) != before[name] for name in before):
        raise ValueError("CPC pool changed while checking receipt")
    return hashes, before, source_hash, sha256_file(args.pool_verification)


def template_windows(cache: Any, train_count: int, mean: np.ndarray,
                     std: np.ndarray) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """
    Draw seed-fixed, label-blind template windows from training rows in normalized units.

    Parameters
    ----------
    cache : Any
        Cache whose first ``train_count`` rows are training records.
    train_count : int
        Number of training rows eligible for templates.
    mean : np.ndarray
        Per-lead mean shaped ``[12, 1]``.
    std : np.ndarray
        Per-lead standard deviation shaped ``[12, 1]``.

    Returns
    -------
    tuple[torch.Tensor, list[dict[str, Any]]]
        ``[TEMPLATES, 12, SUPPORT]`` bank and the source window of each template.

    Raises
    ------
    ValueError
        If a window has nonfinite samples.
    """
    rng = np.random.default_rng(SEED + TEMPLATE_SEED_OFFSET)
    rows = rng.choice(train_count, size=TEMPLATES, replace=False)
    halves = rng.integers(0, 2, size=TEMPLATES)
    ends = rng.integers(SUPPORT - 1, HALF_SAMPLES, size=TEMPLATES)
    bank = np.empty((TEMPLATES, 12, SUPPORT), dtype=np.float32)
    receipt = []
    for i, (row, half, end) in enumerate(zip(rows, halves, ends, strict=True)):
        start = int(half) * HALF_SAMPLES + int(end) - SUPPORT + 1
        window = np.array(cache.signals[int(row), :, start:start + SUPPORT], copy=True)
        window -= mean
        window /= std
        if not np.isfinite(window).all():
            raise ValueError("Nonfinite initial template")
        bank[i] = window
        receipt.append({"ecg_id": cache.rows[int(row)]["ecg_id"], "half": int(half), "end_in_half": int(end)})
    return torch.from_numpy(bank), receipt


def load_normalization(path: Path, pool: Pool, signals_sha256: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the Experiment 004 train-only normalization and check its source.

    Parameters
    ----------
    path : Path
        ``normalization.json``.
    pool : Pool
        Frozen CPC waveform cache.
    signals_sha256 : str
        Verified digest of the pool's ``signals.npy``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Per-lead mean and standard deviation shaped ``[12, 1]``.

    Raises
    ------
    ValueError
        If the normalization was fitted elsewhere or is invalid.
    """
    norm = json.loads(path.read_text())
    if (norm["source"]["signals_sha256"] != signals_sha256
            or norm["source"]["train_ids_sha256"] != sha256_json([row["ecg_id"] for row in pool.train_rows])):
        raise ValueError("Normalization not fitted on frozen CPC training pool")
    mean = np.asarray(norm["mean"], dtype=np.float32)[:, None]
    std = np.asarray(norm["std"], dtype=np.float32)[:, None]
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Invalid train-only normalization")
    return mean, std


def check_pool_manifests(pool: Pool, manifest_dir: Path) -> None:
    """
    Check the PTB-XL manifests recorded when the pool was built.

    Parameters
    ----------
    pool : Pool
        Frozen CPC waveform cache.
    manifest_dir : Path
        PTB-XL manifest root.

    Raises
    ------
    ValueError
        If a manifest differs from the pool's record.
    """
    for name, expected in pool.metadata["ptb_manifest_sha256"].items():
        if sha256_file(manifest_dir / "seed42_fraction1" / name) != expected:
            raise ValueError(f"Frozen pool PTB manifest changed: {name}")


def code_hashes(extra_code: Sequence[str] = ()) -> dict[str, str]:
    """
    Digest every repository file that determines the pilot's results.

    Parameters
    ----------
    extra_code : Sequence[str]
        Files added by a versioned entry point.

    Returns
    -------
    dict[str, str]
        SHA-256 keyed by repository-relative path.
    """
    return {name: sha256_file(ROOT / name) for name in (*CODE_PATHS, *extra_code)}


def load_inputs(args: argparse.Namespace, extra_code: Sequence[str] = ()) -> PilotData:
    """
    Verify every pilot input and build the experiment provenance.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    extra_code : Sequence[str]
        Additional repository files whose digests enter the provenance.

    Returns
    -------
    PilotData
        Verified inputs, without preloaded waveforms.
    """
    began = time.monotonic()
    pool = Pool(args.cache_dir)
    partitions = load_partitions(pool, args.manifest_dir)
    ssl_config = check_final_ssl(args.ssl)
    hashes, pool_stats, verifier_source_hash, pool_receipt_hash = verified_pool_hashes(args, pool)
    mean, std = load_normalization(args.normalization, pool, hashes["signals.npy"])
    manifest_paths = [args.manifest_dir / f"seed42_fraction{budget}" / f"{name}.csv"
                      for budget in BUDGETS for name in ("labeled_train", "validation", "test")]
    manifest_paths.append(args.manifest_dir / "seed42_fraction1/all_train_ssl.csv")
    check_pool_manifests(pool, args.manifest_dir)
    sources = [args.ssl, args.ssl.parent / "epoch_state.pt", args.ssl.parent / "config.json",
               args.normalization, *manifest_paths]
    provenance = {"sources": {str(path.resolve()): sha256_file(path) for path in sources},
                  "code": code_hashes(extra_code),
                  "pool_complete_sha256": sha256_file(args.cache_dir / "complete.json"),
                  "pool_content_sha256": hashes,
                  "pool_file_stats": pool_stats,
                  "pool_verifier_source_sha256": verifier_source_hash,
                  "ssl_fingerprint": ssl_config["fingerprint"],
                  "settings": {"seed": SEED, "epochs": EPOCHS, "batch": BATCH,
                               "encoder_lr": ENCODER_LR, "head_and_branch_lr": HEAD_LR,
                               "weight_decay": WEIGHT_DECAY, "templates": TEMPLATES,
                               "support_samples": SUPPORT, "save_every": SAVE_EVERY}}
    return PilotData(pool, partitions, mean, std, provenance, pool_receipt_hash, time.monotonic() - began)


def preload(args: argparse.Namespace, data: PilotData) -> None:
    """
    Load training and development waveforms into RAM and draw the template bank.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``max_cache_bytes`` and ``reserve_bytes``.
    data : PilotData
        Verified inputs, updated in place.
    """
    started = time.monotonic()
    rows = data.partitions.full + data.partitions.development
    data.waveforms = BoundedWaveformCache(data.pool, rows, max_bytes=args.max_cache_bytes,
                                          reserve_bytes=args.reserve_bytes, expected_source="ptbxl")
    data.bank, data.template_receipt = template_windows(data.waveforms, len(data.partitions.full),
                                                        data.mean, data.std)
    data.preload_seconds = time.monotonic() - started


def _prefixed(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    return {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)}


def make_model(ssl: Path, kind: str, bank: torch.Tensor | None, device: str) -> MorphologyCPCClassifier:
    """
    Build one arm from the Experiment 004 encoder with a head shared across arms.

    Parameters
    ----------
    ssl : Path
        Experiment 004 ``encoder.pt``.
    kind : str
        ``"none"``, ``"conv"`` or ``"template"``.
    bank : torch.Tensor | None
        Initial template bank, unused by ``"none"``.
    device : str
        Target device.

    Returns
    -------
    MorphologyCPCClassifier
        Model whose first training dropout mask is common to every arm.

    Raises
    ------
    ValueError
        If the checkpoint's encoder keys differ from the original CPC encoder.
    """
    seed_everything(SEED)
    # Construct the common original classifier first: adding a branch must not
    # shift the classifier-head RNG stream across comparison arms.
    reference = CPCClassifier()
    reference_head = cpu_state(reference.head)
    encoder_keys = set(reference.encoder.state_dict())
    model = MorphologyCPCClassifier(kind, bank if kind != "none" else None)
    model.head.load_state_dict(reference_head)
    saved = torch.load(ssl, map_location="cpu", weights_only=True)
    model.encoder.convs.load_state_dict(_prefixed(saved["encoder"], "convs."), strict=True)
    model.encoder.context.load_state_dict(_prefixed(saved["encoder"], "context."), strict=True)
    if set(saved["encoder"]) != encoder_keys:
        raise ValueError("Bootstrap encoder source keys changed")
    model = model.to(device)
    # The first training dropout mask is also common across arms. Resume state
    # overrides this with its exact saved RNG state.
    seed_everything(SEED + DROPOUT_SEED_OFFSET)
    return model


def optimizer_for(model: MorphologyCPCClassifier) -> torch.optim.AdamW:
    """
    AdamW with a lower rate for the pretrained encoder than for new parameters.

    Parameters
    ----------
    model : MorphologyCPCClassifier
        Arm to optimize.

    Returns
    -------
    torch.optim.AdamW
        Optimizer over encoder, head and any morphology branch.
    """
    encoder = model.encoder
    base = list(encoder.convs.parameters()) + list(encoder.context.parameters())
    groups = [{"params": base, "lr": ENCODER_LR}, {"params": model.head.parameters(), "lr": HEAD_LR}]
    if encoder.kind != "none":
        groups.append({"params": [encoder.bank, *encoder.branch_hidden.parameters(),
                                  *encoder.branch_final.parameters()], "lr": HEAD_LR})
    return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


def rng_matches(saved: dict[str, Any]) -> bool:
    """
    Compare the active global random states with a saved checkpoint's.

    Parameters
    ----------
    saved : dict[str, Any]
        States from ``capture_rng_state``.

    Returns
    -------
    bool
        True when every generator state is identical.
    """
    active = capture_rng_state()
    return (active["python"] == saved["python"]
            and active["numpy"][0] == saved["numpy"][0]
            and np.array_equal(active["numpy"][1], saved["numpy"][1])
            and active["numpy"][2:] == saved["numpy"][2:]
            and torch.equal(active["torch"], saved["torch"])
            and len(active["cuda"]) == len(saved["cuda"])
            and all(torch.equal(a, b) for a, b in zip(active["cuda"], saved["cuda"], strict=True)))


@torch.inference_mode()
def predict_development(model: MorphologyCPCClassifier, data: PilotData, device: str) -> np.ndarray:
    """
    Compute development logits in manifest order.

    Parameters
    ----------
    model : MorphologyCPCClassifier
        Arm to evaluate.
    data : PilotData
        Preloaded inputs; development rows follow training rows.
    device : str
        Device holding ``model``.

    Returns
    -------
    np.ndarray
        Float64 development logits.
    """
    model.eval()
    logits = []
    for x in development_batches(data.waveforms, len(data.partitions.full), len(data.partitions.development),
                                 data.mean, data.std, device):
        logits.extend(model(x).cpu().numpy().tolist())
    return np.asarray(logits, dtype=np.float64)


def development_screen(rows: list[dict[str, str]], logits: np.ndarray) -> dict[str, Any]:
    """
    Development AUROC and cross-fitted patient-fold operating point.

    Parameters
    ----------
    rows : list[dict[str, str]]
        Development rows.
    logits : np.ndarray
        Development logits.

    Returns
    -------
    dict[str, Any]
        AUROC, pooled fold sensitivity and specificity, and confusion counts.
    """
    labels, groups = labels_and_patients(rows)
    probabilities = clipped_sigmoid(logits)
    return {"auroc": float(roc_auc_score(labels, probabilities)),
            **fold_operating_point(labels, groups, probabilities, SEED)}


def identity(data: PilotData, budget: str, kind: str) -> dict[str, Any]:
    """
    Everything that determines one arm's result.

    Parameters
    ----------
    data : PilotData
        Preloaded inputs.
    budget : str
        Label budget.
    kind : str
        Arm.

    Returns
    -------
    dict[str, Any]
        Identity whose digest fingerprints the arm.
    """
    return {"provenance": data.provenance, "budget": budget, "arm": kind,
            "train_ids": [row["ecg_id"] for row in data.partitions.full],
            "development_ids": [row["ecg_id"] for row in data.partitions.development],
            "template_selection": data.template_receipt}


def train_epoch(args: argparse.Namespace, data: PilotData, model: MorphologyCPCClassifier,
                optimizer: torch.optim.Optimizer, progress: Progress, labels: tuple[np.ndarray, np.ndarray],
                checkpoint: tuple[Path, str], clock: tuple[float, float | None]) -> None:
    """
    Train the remaining batches of the current epoch, checkpointing periodically.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``device``.
    data : PilotData
        Preloaded inputs.
    model : MorphologyCPCClassifier
        Arm being trained.
    optimizer : torch.optim.Optimizer
        Its optimizer.
    progress : Progress
        Position, updated in place.
    labels : tuple[np.ndarray, np.ndarray]
        Exposure mask and targets from ``exposed_labels``.
    checkpoint : tuple[Path, str]
        Arm directory and fingerprint.
    clock : tuple[float, float | None]
        ``time.monotonic`` start of this invocation and the deadline.

    Raises
    ------
    SystemExit
        With ``INTERRUPTED_EXIT`` after checkpointing on SIGTERM or the deadline.
    """
    exposed, targets = labels
    directory, fingerprint = checkpoint
    started, deadline = clock
    model.train()
    batches = fixed_batches(len(data.partitions.full), exposed, progress.epoch, SEED)
    for position in range(progress.batch, len(batches)):
        indices = batches[position]
        x = normalized_batch(data.waveforms, indices, data.mean, data.std, args.device)
        y = torch.from_numpy(targets[indices]).to(args.device)
        mask = torch.from_numpy(exposed[indices]).to(args.device)
        loss = F.binary_cross_entropy_with_logits(model(x)[mask], y[mask])
        checked_step(loss, model, optimizer, "training")
        totals = progress.totals
        totals["updates"] = totals.get("updates", 0) + 1
        totals["record_exposures"] = totals.get("record_exposures", 0) + len(indices)
        totals["label_exposures"] = totals.get("label_exposures", 0) + int(mask.sum())
        totals["loss_sum"] = totals.get("loss_sum", 0.0) + float(loss.detach())
        progress.batch = position + 1
        stop = interrupted(deadline)
        if progress.batch % SAVE_EVERY == 0 or stop:
            save_state(directory, fingerprint, model, optimizer, progress,
                       progress.elapsed_seconds + time.monotonic() - started)
        if stop:
            raise SystemExit(INTERRUPTED_EXIT)


def finish_epoch(args: argparse.Namespace, data: PilotData, model: MorphologyCPCClassifier,
                 optimizer: torch.optim.Optimizer, progress: Progress, arm: tuple[str, str],
                 checkpoint: tuple[Path, str], started: float) -> None:
    """
    Screen development data, keep the best model and save the epoch boundary.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``device``.
    data : PilotData
        Preloaded inputs.
    model : MorphologyCPCClassifier
        Arm being trained.
    optimizer : torch.optim.Optimizer
        Its optimizer.
    progress : Progress
        Position, updated in place.
    arm : tuple[str, str]
        Label budget and arm kind.
    checkpoint : tuple[Path, str]
        Arm directory and fingerprint.
    started : float
        ``time.monotonic`` start of this invocation.

    Raises
    ------
    RuntimeError
        If the weights became nonfinite.
    """
    budget, kind = arm
    directory, fingerprint = checkpoint
    screen = development_screen(data.partitions.development, predict_development(model, data, args.device))
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise RuntimeError("Nonfinite model weights")
    if screen["auroc"] > progress.best["auc"]:
        progress.best = {"auc": screen["auroc"], "epoch": progress.epoch + 1}
        write_torch_atomic(directory / "best_model.pt", {"fingerprint": fingerprint,
                           "epoch": progress.epoch + 1, "model": cpu_state(model)})
    totals = progress.totals
    row = {"epoch": progress.epoch + 1, "budget": budget, "arm": kind,
           "mean_batch_loss": totals["loss_sum"] / totals["updates"],
           "optimizer_updates": totals["updates"], "record_exposures": totals["record_exposures"],
           "label_exposures": totals["label_exposures"], "development": screen,
           "best_epoch": progress.best["epoch"],
           "elapsed_seconds": progress.elapsed_seconds + time.monotonic() - started}
    progress.history.append(row)
    print(json.dumps(row), flush=True)
    progress.epoch += 1
    progress.batch, progress.totals = 0, {}
    save_state(directory, fingerprint, model, optimizer, progress,
               progress.elapsed_seconds + time.monotonic() - started)


def run_arm(args: argparse.Namespace, data: PilotData, budget: str, kind: str, directory: Path,
            max_epochs: int, deadline: float | None) -> list[dict[str, Any]]:
    """
    Train one arm resumably, or verify it when already complete.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``ssl`` and ``device``.
    data : PilotData
        Preloaded inputs.
    budget : str
        Label budget.
    kind : str
        Arm.
    directory : Path
        Arm output directory.
    max_epochs : int
        Stop after this many completed epochs; only ``EPOCHS`` completes the arm.
    deadline : float | None
        ``time.monotonic`` deadline for checkpoint-and-exit.

    Returns
    -------
    list[dict[str, Any]]
        Per-epoch history.
    """
    directory.mkdir(parents=True, exist_ok=True)
    fingerprint = sha256_json(identity(data, budget, kind))
    if (directory / "completion.json").exists():
        return check_completion(directory, fingerprint, ARTIFACTS)
    config = directory / "config.json"
    check_existing_config(config, fingerprint)
    model = make_model(args.ssl, kind, data.bank, args.device)
    optimizer = optimizer_for(model)
    progress = load_state(directory, fingerprint, model, optimizer, dict(INITIAL_BEST))
    write_json_atomic(config, {"fingerprint": fingerprint, "identity": identity(data, budget, kind),
                               "parameter_count": parameter_count(model),
                               "exposed_labels": FULL_LABELS if budget == "1" else LIMITED_LABELS})
    labels = exposed_labels(data.partitions.full, data.partitions.limited, budget)
    checkpoint = (directory, fingerprint)
    started = time.monotonic()
    while progress.epoch < max_epochs:
        train_epoch(args, data, model, optimizer, progress, labels, checkpoint, (started, deadline))
        finish_epoch(args, data, model, optimizer, progress, (budget, kind), checkpoint, started)
    if progress.epoch == EPOCHS and max_epochs == EPOCHS:
        write_completion(directory, fingerprint, progress.best, ARTIFACTS)
    return progress.history


def verify_roundtrip(args: argparse.Namespace, data: PilotData, kind: str, directory: Path,
                     device: str) -> dict[str, Any]:
    """
    Reload a one-epoch profile checkpoint into a fresh model and compare every state.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``ssl``.
    data : PilotData
        Preloaded inputs.
    kind : str
        Profiled arm.
    directory : Path
        Profile arm directory holding ``resume.pt``.
    device : str
        Device of the fresh model.

    Returns
    -------
    dict[str, Any]
        Roundtrip receipt.

    Raises
    ------
    ValueError
        If the model, optimizer or random states differ after reloading.
    """
    original = torch.load(directory / "resume.pt", map_location="cpu", weights_only=False)
    probe = make_model(args.ssl, kind, data.bank, device)
    optimizer = optimizer_for(probe)
    progress = load_state(directory, sha256_json(identity(data, "1", kind)), probe, optimizer,
                          dict(INITIAL_BEST))
    receipt = check_roundtrip(original, progress, probe, optimizer)
    if not rng_matches(original["rng"]):
        raise ValueError("Profile RNG roundtrip differs")
    return {**receipt, "rng_roundtrip": True}


def best_rows(output: Path) -> tuple[list[str], dict[tuple[str, str], dict[str, Any]]] | None:
    """
    Summarize each completed arm's best development epoch.

    Parameters
    ----------
    output : Path
        Pilot output directory.

    Returns
    -------
    tuple[list[str], dict[tuple[str, str], dict[str, Any]]] | None
        Report table rows and best development screens keyed by
        ``(budget, kind)``, or ``None`` while an arm is incomplete.

    Raises
    ------
    ValueError
        If a completed arm lacks epochs.
    """
    lines, selected = [], {}
    for budget in BUDGETS:
        for kind in KINDS:
            directory = output / f"{kind}_fraction{budget}_seed42"
            if not (directory / "completion.json").exists():
                return None
            history = json.loads((directory / "history.json").read_text())
            if len(history) != EPOCHS:
                raise ValueError("Completed arm lacks five epochs")
            epoch = json.loads((directory / "completion.json").read_text())["best_epoch"]
            dev = history[epoch - 1]["development"]
            selected[(budget, kind)] = dev
            lines.append(f"| {budget} | {kind} | {epoch} | {dev['auroc']:.4f} | "
                         f"{dev['fold_specificity']:.4f} | "
                         f"{dev['fold_sensitivity']:.4f} | {sum(r['optimizer_updates'] for r in history)} | "
                         f"{sum(r['label_exposures'] for r in history)} | "
                         f"{history[-1]['elapsed_seconds']:.1f} |")
    return lines, selected


def report_pilot(output: Path) -> None:
    """
    Write the development-only report once all six arms are complete.

    Parameters
    ----------
    output : Path
        Pilot output directory.
    """
    summary = best_rows(output)
    if summary is None:
        return
    rows, selected = summary
    lines = ["# Experiment 017 development-only morphology pilot", "",
             "Five fixed epochs per arm. No calibration or test predictions were used.", "",
             "| Label budget | Arm | Best epoch | Development AUROC | Patient-fold specificity | "
             "Patient-fold sensitivity | Updates | Labeled exposures | Wall seconds |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |", *rows,
             "", "## Prespecified comparison", ""]
    advance = False
    safe = True
    for budget in BUDGETS:
        template = selected[(budget, "template")]
        conv = selected[(budget, "conv")]
        plain = selected[(budget, "none")]
        gain_conv = template["auroc"] - conv["auroc"]
        gain_plain = template["auroc"] - plain["auroc"]
        sens_delta = template["fold_sensitivity"] - conv["fold_sensitivity"]
        passed = (gain_conv >= REQUIRED_GAIN and gain_plain >= REQUIRED_GAIN
                  and sens_delta >= -SENSITIVITY_TOLERANCE)
        advance |= passed
        safe &= gain_conv >= -REQUIRED_GAIN and sens_delta >= -SENSITIVITY_TOLERANCE
        lines.append(f"Budget {budget}: template minus convolution AUROC {gain_conv:+.4f}, "
                     f"template minus no-branch AUROC {gain_plain:+.4f}, "
                     f"template minus convolution patient-fold sensitivity {sens_delta:+.4f}; "
                     f"positive criteria {'met' if passed else 'not met'}.")
    advance &= safe
    decision = ("A second matched seed is warranted before calibration or test." if advance else
                "The prespecified development screen did not pass; retain results as exploratory evidence.")
    lines.extend(["", decision, "",
                  "Template matches are not validated explanations. The binary endpoint is an ECG diagnostic "
                  "annotation proxy, not verified health or referral need.", ""])
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.md").write_text("\n".join(lines))
    write_json_atomic(output / "completion.json", {
        "status": "development_pilot_complete", "report_sha256": sha256_file(output / "report.md"),
        "advance_to_second_seed": bool(advance),
        "arm_completions_sha256": {
            f"{kind}_fraction{budget}_seed42":
                sha256_file(output / f"{kind}_fraction{budget}_seed42/completion.json")
            for budget in BUDGETS for kind in KINDS}})


def write_check(args: argparse.Namespace, data: PilotData) -> None:
    """
    Record the CPU input check that must precede any GPU stage.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``output_dir`` and ``pool_verification``.
    data : PilotData
        Verified inputs.
    """
    check_waveform_sample(data.pool, data.partitions.full[0])
    fingerprint = sha256_json(data.provenance)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_dir / "provenance/verification.json", {
        "stage": "check", "fingerprint": fingerprint, "provenance": data.provenance,
        "source_015_receipt_sha256_for_audit": data.pool_receipt_sha256,
        "source_015_receipt": str(args.pool_verification.resolve())})
    partitions = data.partitions
    print(json.dumps({"stage": "check", "train": len(partitions.full), "limited": len(partitions.limited),
                      "development": len(partitions.development), "calibration": len(partitions.calibration),
                      "test": len(partitions.test), "precheck_seconds": data.precheck_seconds,
                      "fingerprint": fingerprint}), flush=True)


def profile(args: argparse.Namespace, data: PilotData, deadline: float, roundtrip_device: str) -> None:
    """
    Time one real full-data epoch per arm and write the two-hour planning gate.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    data : PilotData
        Preloaded inputs.
    deadline : float
        ``time.monotonic`` deadline.
    roundtrip_device : str
        Device on which each profile checkpoint is reloaded and compared.
    """
    def profile_arm(kind: str, directory: Path) -> dict[str, Any]:
        run_arm(args, data, "1", kind, directory, 1, deadline)
        return verify_roundtrip(args, data, kind, directory, roundtrip_device)

    durations, roundtrips = profile_arms("experiment017_profile_", KINDS, profile_arm)
    overhead = data.precheck_seconds + data.preload_seconds
    training_estimate = overhead + 2 * EPOCHS * sum(durations.values())
    estimate = overhead + sum(durations.values()) + training_estimate
    receipt = {"stage": "profile", "device": args.device, "full_epoch_seconds": durations,
               "checkpoint_roundtrips": roundtrips,
               "precheck_seconds": data.precheck_seconds, "preload_seconds": data.preload_seconds,
               "projected_six_run_training_seconds": training_estimate,
               "conservative_profile_plus_six_run_seconds": estimate,
               "planning_gate_seconds": PLANNING_GATE_SECONDS,
               "gate_passed": estimate <= PLANNING_GATE_SECONDS,
               "peak_gpu_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
               "fingerprint": sha256_json(data.provenance)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_dir / "profile.json", receipt)
    print(json.dumps(receipt), flush=True)


def require_profile(output_dir: Path, fingerprint: str) -> None:
    """
    Require a passing real-GPU profile of these exact inputs before training.

    Parameters
    ----------
    output_dir : Path
        Pilot output directory holding ``profile.json``.
    fingerprint : str
        Current provenance digest.

    Raises
    ------
    ValueError
        If the profile is missing, from another device or inputs, or failed its gate.
    """
    path = output_dir / "profile.json"
    if not path.exists():
        raise ValueError("Real complete-pass GPU profile required before training")
    receipt = json.loads(path.read_text())
    if (receipt.get("stage") != "profile" or receipt.get("device") != "cuda"
            or receipt["fingerprint"] != fingerprint or not receipt["gate_passed"]):
        raise ValueError("GPU profile fingerprint or two-hour planning gate failed")


def parse_args(argv: list[str] | None, output: Path) -> argparse.Namespace:
    """
    Parse and validate the command line.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or ``None`` for ``sys.argv``.
    output : Path
        Default output directory.

    Returns
    -------
    argparse.Namespace
        Validated arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("check", "profile", "train"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--manifest-dir", type=Path, default=MANIFEST)
    parser.add_argument("--ssl", type=Path, default=SSL)
    parser.add_argument("--normalization", type=Path, default=NORMALIZATION)
    parser.add_argument("--output-dir", type=Path, default=output)
    parser.add_argument("--pool-verification", type=Path, default=POOL_VERIFICATION)
    parser.add_argument("--max-cache-bytes", type=int, default=DEFAULT_MAX_CACHE_BYTES)
    parser.add_argument("--reserve-bytes", type=int, default=DEFAULT_RESERVE_BYTES)
    parser.add_argument("--max-wall-seconds", type=int, default=PLANNING_GATE_SECONDS)
    args = parser.parse_args(argv)
    if (args.threads < 1 or args.max_cache_bytes < MIN_CACHE_BYTES or args.reserve_bytes < 0
            or args.max_wall_seconds < 1):
        parser.error("Invalid thread, memory, or time limits")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    if args.stage == "profile" and args.device != "cuda":
        parser.error("Profile must measure the real V100 GPU data path")
    return args


def main(argv: list[str] | None = None, *, output: Path = OUTPUT, extra_code: Sequence[str] = (),
         roundtrip_device: str | None = None) -> None:
    """
    Run the check, profile or training stage.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or ``None`` for ``sys.argv``.
    output : Path
        Default output directory.
    extra_code : Sequence[str]
        Additional repository files whose digests enter the provenance.
    roundtrip_device : str | None
        Device for profile checkpoint comparisons; defaults to ``--device``.
    """
    args = parse_args(argv, output)
    torch.set_num_threads(args.threads)
    install_stop_handler()
    deadline = time.monotonic() + args.max_wall_seconds
    with gpu_lock(args.device):
        data = load_inputs(args, extra_code)
        if args.stage == "check":
            write_check(args, data)
            return
        preload(args, data)
        fingerprint = sha256_json(data.provenance)
        if args.device == "cuda":
            require_receipt(args.output_dir / "provenance/verification.json", fingerprint,
                            "Verified Experiment 017 CPU check is missing or changed")
        print(json.dumps({"stage": args.stage, "precheck_seconds": data.precheck_seconds,
                          "preload_seconds": data.preload_seconds,
                          "cache_bytes": int(data.waveforms.signals.nbytes),
                          "template_selection": data.template_receipt}), flush=True)
        if args.stage == "profile":
            profile(args, data, deadline, roundtrip_device or args.device)
            return
        if args.device == "cuda":
            require_profile(args.output_dir, fingerprint)
        run_pilot(KINDS, deadline, lambda budget, kind: run_arm(
            args, data, budget, kind, args.output_dir / f"{kind}_fraction{budget}_seed42", EPOCHS, deadline))
        report_pilot(args.output_dir)


if __name__ == "__main__":
    main()
