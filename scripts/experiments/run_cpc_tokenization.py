#!/usr/bin/env python3
"""Continue the same CPC with cluster targets or causal chunk tokenization."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from ecg_experiment import cpc_pool
from ecg_experiment.cpc_tokenization import (
    CLUSTER_WEIGHT,
    CLUSTERS,
    FIT_RECORDS,
    TOKENS_PER_RECORD,
    TokenizationClassifier,
    TokenizationPretrainer,
    cnn_tokens,
    fit_kmeans,
    snapshot_teacher_convs,
)
from ecg_experiment.ecg_tokenizers import load_beat_metadata
from ecg_experiment.evaluation import evaluate_predictions, paired_comparison, partition_validation
from ecg_experiment.files import sha256_file, sha256_json, write_json_atomic, write_torch_atomic
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
DEFAULT_OUTPUT = ROOT / "outputs/experiment006_cpc_tokenization"
DEFAULT_BEATS = ROOT / "data/processed/cpc_beats_40k"
DEFAULT_BOOTSTRAP = ROOT / "outputs/experiment004_cpc_40k/cpc_ssl"
VARIANTS = ("continuation", "clusteraux", "fixedchunk", "beatchunk", "learnedchunk")
CHUNK_VARIANTS = ("fixedchunk", "beatchunk", "learnedchunk")
SEED = 42
REFIT_SEED = 43
BOOTSTRAP_EPOCHS = 20
PREDICTION_HEADS = 3
REFRESH_EPOCH = 5
SSL_LR = 1e-3
ENCODER_LR = 3e-4
HEAD_LR = 1e-3
WEIGHT_DECAY = 0.01
PROFILE_WARMUP = 5
CONTEXT_WIDTH = 512
TOKEN_WIDTH = 256
HALF_TOKENS = 79
RECORD_TOKENS = 2 * HALF_TOKENS
FIXED_CHUNK_TOKENS = 16
FEATURE_BATCH = 64
FEATURE_LOADER_SEED = 4242
# Code outside cpc_pool.CODE_FILES that determines this experiment's results.
EXTRA_CODE = ("ecg_experiment/cpc_tokenization.py", "scripts/experiments/run_cpc_tokenization.py",
              "ecg_experiment/ecg_tokenizers.py", "scripts/data/prepare_beat_tokens.py",
              "ecg_experiment/training.py", "ecg_experiment/cpc.py")
BEAT_FILES = ("complete.json", "ecg_ids.npy", "boundaries.npy", "peaks.npy", "confirmations.npy",
              "counts.npy", "forced.npy")
COMPLETED_ARTIFACTS = ("config.json", "history.json", "model.pt", "metrics.json",
                       "test_predictions.csv", "calibration_predictions.npz")
Bootstrap = tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]
COMPARISONS = (("clusteraux", "continuation"), ("fixedchunk", "continuation"),
               ("beatchunk", "fixedchunk"), ("learnedchunk", "fixedchunk"))


class BeatPoolDataset(cpc_pool.PoolDataset):
    """
    Pool waveforms with their row-aligned causal beat boundaries.

    Parameters
    ----------
    pool : cpc_pool.Pool
        Source cache.
    rows : list[dict[str, str]]
        Rows to serve.
    mean : np.ndarray
        Per-lead mean.
    std : np.ndarray
        Per-lead standard deviation.
    boundaries : np.ndarray
        Boolean boundary grid aligned to pool rows.
    """

    def __init__(self, pool: cpc_pool.Pool, rows: list[dict[str, str]], mean: np.ndarray, std: np.ndarray,
                 boundaries: np.ndarray) -> None:
        super().__init__(pool, rows, mean, std)
        self.boundaries = boundaries

    def __getitem__(self, index: int) -> tuple[torch.Tensor, float, str, torch.Tensor]:
        """Return the normalized signal, target, patient ID and beat boundaries of record ``index``."""
        signal, target, patient = super().__getitem__(index)
        return signal, target, patient, torch.from_numpy(
            np.asarray(self.boundaries[self.indices[index]], dtype=np.bool_).copy())


def loader(pool: cpc_pool.Pool, rows: list[dict[str, str]], mean: np.ndarray, std: np.ndarray,
           boundaries: np.ndarray, batch_size: int, shuffle: bool, generator: torch.Generator | None,
           device: str) -> DataLoader:
    """
    Build a single-process loader over pool rows and their beat boundaries.

    Parameters
    ----------
    pool : cpc_pool.Pool
        Source cache.
    rows : list[dict[str, str]]
        Rows to serve.
    mean : np.ndarray
        Per-lead mean.
    std : np.ndarray
        Per-lead standard deviation.
    boundaries : np.ndarray
        Boolean boundary grid aligned to pool rows.
    batch_size : int
        Records per batch.
    shuffle : bool
        Shuffle each epoch with ``generator``.
    generator : torch.Generator | None
        Shuffling generator.
    device : str
        Target device; CUDA enables pinned memory.

    Returns
    -------
    DataLoader
        Loader yielding ``(signal, target, patient_id, boundaries)`` batches.
    """
    return DataLoader(BeatPoolDataset(pool, rows, mean, std, boundaries),
                      batch_size=batch_size, shuffle=shuffle, generator=generator,
                      num_workers=0, pin_memory=(device == "cuda"), drop_last=False)


def load_beats(path: Path, pool: cpc_pool.Pool) -> np.ndarray:
    """
    Load beat boundaries prepared from these exact waveforms and detector code.

    Parameters
    ----------
    path : Path
        Beat metadata directory.
    pool : cpc_pool.Pool
        Frozen CPC waveform cache.

    Returns
    -------
    np.ndarray
        Row-aligned boundary grid.

    Raises
    ------
    ValueError
        If the metadata came from other waveforms or detector code.
    """
    info = json.loads((path / "complete.json").read_text())
    identity = info["identity"]
    if (identity["cache_signals_sha256"] != pool.metadata["signals_sha256"]
            or identity["cache_complete_sha256"] != sha256_file(pool.directory / "complete.json")
            or identity["detector_code_sha256"] != sha256_file(ROOT / "ecg_experiment/ecg_tokenizers.py")):
        raise ValueError("Beat metadata uses different waveforms or detector code")
    return load_beat_metadata(path, pool.ids)


def source_hashes(args: argparse.Namespace, pool: cpc_pool.Pool) -> dict[str, str]:
    """
    Hash the pool, manifests, beat metadata, bootstrap checkpoint and this experiment's code.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``manifest_dir``, ``beat_dir`` and ``bootstrap_dir``.
    pool : cpc_pool.Pool
        Frozen CPC waveform cache.

    Returns
    -------
    dict[str, str]
        SHA-256 keyed by resolved path.
    """
    hashes = cpc_pool.make_source_hashes(pool, args.manifest_dir)
    paths = [ROOT / name for name in EXTRA_CODE]
    paths += [args.beat_dir / name for name in BEAT_FILES]
    paths += [args.bootstrap_dir / name for name in ("encoder.pt", "epoch_state.pt", "config.json")]
    hashes.update({str(path.resolve()): sha256_file(path) for path in paths})
    return hashes


def bootstrap(args: argparse.Namespace, source_hashes: dict[str, str]) -> Bootstrap:
    """
    Load the completed Experiment 004 encoder and prediction heads, checking their provenance.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``bootstrap_dir``.
    source_hashes : dict[str, str]
        Current input digests; any path shared with 004's inputs must match.

    Returns
    -------
    Bootstrap
        Encoder state, prediction-head state and the 004 configuration.

    Raises
    ------
    ValueError
        If the checkpoint is incomplete, inconsistent, or used changed inputs.
    """
    final = torch.load(args.bootstrap_dir / "encoder.pt", map_location="cpu", weights_only=True)
    state = torch.load(args.bootstrap_dir / "epoch_state.pt", map_location="cpu", weights_only=False)
    if final["variant"] != "cpc" or final["epochs"] != BOOTSTRAP_EPOCHS or state["epoch"] != BOOTSTRAP_EPOCHS:
        raise ValueError("Experiment 004 CPC bootstrap is not its completed 20-epoch state")
    if final["fingerprint"] != state["fingerprint"]:
        raise ValueError("Experiment 004 final encoder and epoch state disagree")
    encoder = {name.removeprefix("encoder."): value for name, value in state["model"].items()
               if name.startswith("encoder.")}
    if encoder.keys() != final["encoder"].keys() or any(
            not torch.equal(encoder[name], final["encoder"][name]) for name in encoder):
        raise ValueError("Experiment 004 final encoder differs from full epoch state")
    heads = {name.removeprefix("heads."): value for name, value in state["model"].items()
             if name.startswith("heads.")}
    if len(heads) != PREDICTION_HEADS:
        raise ValueError("Expected all three pretrained CPC prediction heads")
    config = json.loads((args.bootstrap_dir / "config.json").read_text())
    if config["fingerprint"] != final["fingerprint"]:
        raise ValueError("Experiment 004 bootstrap config fingerprint mismatch")
    for path, declared in config["inputs"]["cache"].items():
        if path in source_hashes and source_hashes[path] != declared:
            raise ValueError(f"Experiment 004 bootstrap input changed: {path}")
    return encoder, heads, config


def load_bootstrap_weights(model: TokenizationPretrainer, encoder_state: dict[str, torch.Tensor],
                           head_state: dict[str, torch.Tensor]) -> None:
    """
    Load the 004 encoder and prediction heads into any arm.

    Parameters
    ----------
    model : TokenizationPretrainer
        Arm to initialize in place.
    encoder_state : dict[str, torch.Tensor]
        Bootstrap encoder state.
    head_state : dict[str, torch.Tensor]
        Bootstrap prediction-head state.
    """
    if model.variant in CHUNK_VARIANTS:
        model.encoder.load_from_cpc_state_dict(encoder_state)
    else:
        model.encoder.load_state_dict(encoder_state)
    model.heads.load_state_dict(head_state)


def classifier_with_matched_head(variant: str, device: str) -> TokenizationClassifier:
    """
    Use identical binary-head weights despite variant constructor draw counts.

    Parameters
    ----------
    variant : str
        One of ``VARIANTS``.
    device : str
        Target device.

    Returns
    -------
    TokenizationClassifier
        Classifier whose head is shared by every arm; the global RNG is unchanged.
    """
    cpu_rng = torch.random.get_rng_state()
    try:
        torch.random.manual_seed(SEED)
        shared_head = cpu_state(nn.Linear(CONTEXT_WIDTH, 1))
    finally:
        torch.random.set_rng_state(cpu_rng)
    model = TokenizationClassifier(variant)
    model.head.load_state_dict(shared_head)
    return model.to(device)


def reset_cluster_heads(model: TokenizationPretrainer, optimizer: torch.optim.Optimizer,
                        generator: torch.Generator, seed: int = REFIT_SEED) -> None:
    """
    Give new K-means IDs new class heads, leaving CPC weights and RNG untouched.

    Parameters
    ----------
    model : TokenizationPretrainer
        Cluster-auxiliary arm.
    optimizer : torch.optim.Optimizer
        Optimizer whose state for the reset heads is dropped.
    generator : torch.Generator
        Loader generator restored with the global random states.
    seed : int
        Seed of the new head weights.
    """
    snapshot = capture_rng_state(generator)
    try:
        torch.manual_seed(seed)
        for head in model.cluster_heads:
            head.reset_parameters()
            for parameter in head.parameters():
                optimizer.state.pop(parameter, None)
    finally:
        restore_rng_state(snapshot, generator)


def boundary_disagreement(encoder: nn.Module) -> float:
    """
    Fraction of emitted chunk boundaries that differ from the fixed 16-token grid.

    Parameters
    ----------
    encoder : nn.Module
        Chunk encoder after a forward pass, exposing ``last_gates``.

    Returns
    -------
    float
        Mean disagreement over records, halves and tokens.
    """
    gates = encoder.last_gates.detach() >= 0.5
    fixed = torch.zeros(HALF_TOKENS, dtype=torch.bool, device=gates.device)
    fixed[FIXED_CHUNK_TOKENS - 1::FIXED_CHUNK_TOKENS] = True
    fixed[-1] = True
    return float((gates != fixed.reshape(1, 1, HALF_TOKENS)).float().mean())


def codebook_selection(pool: cpc_pool.Pool) -> tuple[np.ndarray, np.ndarray]:
    """
    Seed-fixed training records and token positions used for every codebook fit.

    Parameters
    ----------
    pool : cpc_pool.Pool
        Frozen CPC waveform cache.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Training-row indices and ``[records, tokens]`` token positions.
    """
    rng = np.random.default_rng(SEED)
    count = min(FIT_RECORDS, len(pool.train_rows))
    rows = rng.choice(len(pool.train_rows), size=count, replace=False)
    positions = np.stack([rng.choice(RECORD_TOKENS, size=min(TOKENS_PER_RECORD, RECORD_TOKENS), replace=False)
                          for _ in range(count)])
    return rows.astype(np.int64), positions.astype(np.int64)


@torch.inference_mode()
def selected_features(args: argparse.Namespace, pool: cpc_pool.Pool, mean: np.ndarray, std: np.ndarray,
                      teacher_convs: nn.Module, row_indices: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """
    Extract the same fixed train-record/token sample for each codebook round.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``device``.
    pool : cpc_pool.Pool
        Frozen CPC waveform cache.
    mean : np.ndarray
        Per-lead mean.
    std : np.ndarray
        Per-lead standard deviation.
    teacher_convs : nn.Module
        Frozen CNN snapshot.
    row_indices : np.ndarray
        Training-row indices from ``codebook_selection``.
    positions : np.ndarray
        Token positions from ``codebook_selection``.

    Returns
    -------
    np.ndarray
        ``[records * tokens, 256]`` float32 features.

    Raises
    ------
    RuntimeError
        If not every selected record was extracted.
    """
    teacher_convs.eval().to(args.device)
    selected_rows = [pool.train_rows[int(i)] for i in row_indices]
    data = cpc_pool.loader(pool, selected_rows, mean, std, FEATURE_BATCH, False,
                           torch.Generator().manual_seed(FEATURE_LOADER_SEED), args.device)
    features = np.empty((len(row_indices) * positions.shape[1], TOKEN_WIDTH), dtype=np.float32)
    offset = 0
    for signal, _, _ in data:
        batch = len(signal)
        tokens = cnn_tokens(teacher_convs, signal.to(args.device, non_blocking=True))
        flat = tokens.reshape(batch, RECORD_TOKENS, TOKEN_WIDTH)
        indices = torch.from_numpy(positions[offset:offset + batch]).to(args.device)
        chosen = flat.gather(1, indices.unsqueeze(-1).expand(-1, -1, TOKEN_WIDTH))
        start = offset * positions.shape[1]
        features[start:start + batch * positions.shape[1]] = chosen.cpu().reshape(-1, TOKEN_WIDTH).numpy()
        offset += batch
    if offset != len(row_indices):
        raise RuntimeError("Incomplete codebook feature extraction")
    return features


def fit_round(args: argparse.Namespace, pool: cpc_pool.Pool, mean: np.ndarray, std: np.ndarray,
              teacher: nn.Module, row_indices: np.ndarray, positions: np.ndarray,
              seed: int) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Fit one K-means codebook without leaking RNG into dropout or data order.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``device``.
    pool : cpc_pool.Pool
        Frozen CPC waveform cache.
    mean : np.ndarray
        Per-lead mean.
    std : np.ndarray
        Per-lead standard deviation.
    teacher : nn.Module
        Frozen CNN snapshot.
    row_indices : np.ndarray
        Training-row indices from ``codebook_selection``.
    positions : np.ndarray
        Token positions from ``codebook_selection``.
    seed : int
        K-means seed.

    Returns
    -------
    tuple[torch.Tensor, dict[str, Any]]
        Cluster centers on the device and the fit summary.
    """
    dummy_loader_generator = torch.Generator().manual_seed(0)
    snapshot = capture_rng_state(dummy_loader_generator)
    try:
        features = selected_features(args, pool, mean, std, teacher, row_indices, positions)
        centers, fit = fit_kmeans(features, seed=seed)
    finally:
        restore_rng_state(snapshot, dummy_loader_generator)
    return torch.from_numpy(centers).to(args.device), fit


def initial_codebook(args: argparse.Namespace, pool: cpc_pool.Pool, mean: np.ndarray, std: np.ndarray,
                     source_hashes: dict[str, str], encoder_state: dict[str, torch.Tensor]) -> dict[str, Any]:
    """
    Fit, or reload, the round-0 codebook from the bootstrap encoder's CNN.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``output_dir`` and ``device``.
    pool : cpc_pool.Pool
        Frozen CPC waveform cache.
    mean : np.ndarray
        Per-lead mean.
    std : np.ndarray
        Per-lead standard deviation.
    source_hashes : dict[str, str]
        Input digests from ``source_hashes``.
    encoder_state : dict[str, torch.Tensor]
        Bootstrap encoder state.

    Returns
    -------
    dict[str, Any]
        Centers, teacher state, fixed selection and fit summary.

    Raises
    ------
    ValueError
        If a saved codebook used different inputs.
    """
    path = args.output_dir / "cluster_codebook_round0.pt"
    input_fp = sha256_json({"source_hashes": source_hashes,
                            "normalization": [mean.tolist(), std.tolist()],
                            "fit_records": FIT_RECORDS,
                            "tokens_per_record": TOKENS_PER_RECORD,
                            "clusters": CLUSTERS, "seed": SEED})
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["input_fingerprint"] != input_fp:
            raise ValueError("Existing initial codebook has different inputs")
        return saved
    teacher = TokenizationPretrainer("continuation").encoder
    teacher.load_state_dict(encoder_state)
    teacher_convs = snapshot_teacher_convs(teacher)
    rows, positions = codebook_selection(pool)
    centers, fit = fit_round(args, pool, mean, std, teacher_convs, rows, positions, SEED)
    saved = {"input_fingerprint": input_fp, "centers": centers.cpu(),
             "teacher_state": cpu_state(teacher_convs),
             "row_indices": torch.from_numpy(rows),
             "positions": torch.from_numpy(positions), "fit": fit,
             "training_ecg_ids": [pool.train_rows[int(i)]["ecg_id"] for i in rows]}
    write_torch_atomic(path, saved)
    return saved


def ssl_settings(args: argparse.Namespace, variant: str, codebook_hash: str | None = None) -> dict[str, Any]:
    """
    Fingerprinted continued-pretraining hyperparameters of one arm.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``ssl_epochs`` and ``ssl_batch_size``.
    variant : str
        One of ``VARIANTS``.
    codebook_hash : str | None
        Digest of the round-0 codebook for the cluster arm.

    Returns
    -------
    dict[str, Any]
        Settings entering the stage fingerprint.
    """
    clustered = variant == "clusteraux"
    return {"stage": "continued_pretrain", "variant": variant,
            "seed": SEED, "epochs": args.ssl_epochs, "batch_size": args.ssl_batch_size,
            "bootstrap": "Experiment 004 CPC completed encoder plus all three prediction heads",
            "learning_rate": SSL_LR, "weight_decay": WEIGHT_DECAY, "warmup_epochs": WARMUP_EPOCHS,
            "cosine_floor": COSINE_FLOOR, "horizons": [4, 8, 12],
            "first_query": 24, "negative_exclusion_tokens": 3,
            "cluster_codebook_sha256": codebook_hash,
            "cluster_k": CLUSTERS if clustered else None,
            "cluster_ce_weight": CLUSTER_WEIGHT if clustered else None,
            "cluster_refresh_after_epoch": REFRESH_EPOCH if clustered else None,
            "augmentation": "none"}


@dataclass
class ClusterRound:
    """
    Cluster-auxiliary targets of the current codebook round.

    Attributes
    ----------
    centers : torch.Tensor | None
        Cluster centers, or ``None`` for arms without cluster targets.
    teacher : nn.Module | None
        Frozen CNN snapshot that assigns target IDs.
    index : int
        Codebook round.
    fits : list[dict[str, Any]]
        Fit summary of every round so far.
    """

    centers: torch.Tensor | None = None
    teacher: nn.Module | None = None
    index: int = 0
    fits: list[dict[str, Any]] = field(default_factory=list)


def save_ssl_epoch(directory: Path, fingerprint: str, epoch: int, model: nn.Module,
                   optimizer: torch.optim.Optimizer, generator: torch.Generator,
                   history: list[dict[str, Any]], clusters: ClusterRound) -> None:
    """
    Atomically save a continued-pretraining epoch boundary and its history.

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
    clusters : ClusterRound
        Current codebook round.
    """
    write_torch_atomic(directory / "epoch_state.pt", {
        "fingerprint": fingerprint, "epoch": epoch,
        "model": cpu_state(model), "optimizer": optimizer.state_dict(),
        "rng": capture_rng_state(generator), "history": history,
        "centers": clusters.centers.detach().cpu() if clusters.centers is not None else None,
        "teacher_state": cpu_state(clusters.teacher) if clusters.teacher is not None else None,
        "codebook_round": clusters.index, "round_fits": clusters.fits or []})
    write_json_atomic(directory / "history.json", history)


def initial_clusters(model: TokenizationPretrainer, codebook: dict[str, Any] | None,
                     device: str) -> ClusterRound:
    """
    Round-0 cluster targets from the saved codebook, or none for other arms.

    Parameters
    ----------
    model : TokenizationPretrainer
        Arm being trained.
    codebook : dict[str, Any] | None
        Output of ``initial_codebook`` for the cluster arm.
    device : str
        Target device.

    Returns
    -------
    ClusterRound
        Round-0 targets.
    """
    if not codebook:
        return ClusterRound()
    teacher = snapshot_teacher_convs(model.encoder).to(device)
    teacher.load_state_dict(codebook["teacher_state"])
    return ClusterRound(codebook["centers"].to(device), teacher, 0, [codebook["fit"]])


def resume_ssl(directory: Path, fingerprint: str, model: nn.Module, optimizer: torch.optim.Optimizer,
               generator: torch.Generator, clusters: ClusterRound,
               clustered: bool, device: str) -> tuple[int, list[dict[str, Any]]]:
    """
    Restore the last completed continued-pretraining epoch, or start fresh.

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
    clusters : ClusterRound
        Cluster targets, restored in place.
    clustered : bool
        Whether this arm has cluster targets.
    device : str
        Device holding ``model``.

    Returns
    -------
    tuple[int, list[dict[str, Any]]]
        Completed epochs and history.

    Raises
    ------
    ValueError
        If the fingerprint differs or history exists without state.
    """
    state_path = directory / "epoch_state.pt"
    if not state_path.exists():
        if (directory / "history.json").exists():
            raise ValueError("History exists without resumable SSL state")
        # This reset makes dropout sequences identical despite variant-specific construction.
        seed_everything(SEED)
        return 0, []
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if state["fingerprint"] != fingerprint:
        raise ValueError(f"SSL resume fingerprint mismatch: {directory}")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    restore_rng_state(state["rng"], generator)
    clusters.centers = state["centers"].to(device) if clustered else None
    if clustered:
        clusters.teacher.load_state_dict(state["teacher_state"])
    clusters.index, clusters.fits = state["codebook_round"], state["round_fits"]
    return state["epoch"], state["history"]


def ssl_epoch(model: TokenizationPretrainer, optimizer: torch.optim.Optimizer, data: DataLoader,
              clusters: ClusterRound, device: str) -> tuple[dict[str, Any], dict[str, float]]:
    """
    Train one continued-pretraining epoch and summarize losses and clipping.

    Parameters
    ----------
    model : TokenizationPretrainer
        Arm in training mode.
    optimizer : torch.optim.Optimizer
        Its optimizer, with the epoch's learning rate set.
    data : DataLoader
        Shuffled training loader with beat boundaries.
    clusters : ClusterRound
        Current cluster targets.
    device : str
        Device holding ``model``.

    Returns
    -------
    tuple[dict[str, Any], dict[str, float]]
        Update counts with gradient-norm statistics, and record-weighted loss means.
    """
    totals = {}
    count = updates = clipped = 0
    norm_total = 0.0
    for signal, _, _, beat_boundaries in data:
        signal = signal.to(device, non_blocking=True)
        beat_boundaries = beat_boundaries.to(device, non_blocking=True)
        loss, details = model(signal, beat_boundaries, clusters.centers, clusters.teacher)
        if model.variant in CHUNK_VARIANTS:
            details["boundary_disagreement_with_fixed"] = boundary_disagreement(model.encoder)
        norm = checked_step(loss, model, optimizer, f"{model.variant} continued CPC")
        clipped += int(float(norm) > GRADIENT_CLIP_NORM)
        norm_total += float(norm)
        updates += 1
        count += len(signal)
        for name, value in {"loss": float(loss.detach()), **details}.items():
            totals[name] = totals.get(name, 0.0) + float(value) * len(signal)
    stats = {"optimizer_updates": updates, "record_exposures": count,
             "mean_gradient_norm_before_clip": norm_total / updates,
             "gradient_clip_fraction": clipped / updates}
    return stats, {name: value / count for name, value in totals.items()}


def pretrain(args: argparse.Namespace, pool: cpc_pool.Pool, mean: np.ndarray, std: np.ndarray,
             boundaries: np.ndarray, source_hashes: dict[str, str],
             bootstrap_weights: Bootstrap, variant: str, codebook: dict[str, Any] | None = None) -> Path:
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
    boundaries : np.ndarray
        Row-aligned beat boundaries.
    source_hashes : dict[str, str]
        Input digests from ``source_hashes``.
    bootstrap_weights : Bootstrap
        Output of ``bootstrap``.
    variant : str
        One of ``VARIANTS``.
    codebook : dict[str, Any] | None
        Round-0 codebook for the cluster arm.

    Returns
    -------
    Path
        The completed ``encoder.pt``.

    Raises
    ------
    ValueError
        If saved configuration or encoder fingerprints differ.
    """
    encoder_state, head_state, _ = bootstrap_weights
    seed_everything(SEED)
    directory = args.output_dir / f"{variant}_ssl"
    directory.mkdir(parents=True, exist_ok=True)
    codebook_hash = sha256_file(args.output_dir / "cluster_codebook_round0.pt") if codebook else None
    settings = ssl_settings(args, variant, codebook_hash)
    fp, inputs = cpc_pool.fingerprint(pool, source_hashes, mean, std, settings)
    config_path = directory / "config.json"
    if config_path.exists() and json.loads(config_path.read_text())["fingerprint"] != fp:
        raise ValueError(f"Existing SSL config differs: {directory}")
    teacher_note = ("Frozen CNN snapshot per five-epoch round; refit codebook after epoch five and reset "
                    "auxiliary class heads")
    write_json_atomic(config_path, {"fingerprint": fp, "inputs": inputs,
                      "description": ("All arms continue completed Experiment 004 CPC encoder and "
                                      "future heads"),
                      "cluster_teacher": teacher_note if codebook else None})
    complete = directory / "encoder.pt"
    if complete.exists():
        saved = torch.load(complete, map_location="cpu", weights_only=True)
        if saved["fingerprint"] != fp:
            raise ValueError(f"Completed SSL fingerprint mismatch: {complete}")
        return complete
    model = TokenizationPretrainer(variant).to(args.device)
    load_bootstrap_weights(model, encoder_state, head_state)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=SSL_LR, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(SEED)
    data = loader(pool, pool.train_rows, mean, std, boundaries, args.ssl_batch_size, True, generator,
                  args.device)
    clusters = initial_clusters(model, codebook, args.device)
    start_epoch, history = resume_ssl(directory, fp, model, optimizer, generator, clusters, bool(codebook),
                                      args.device)
    started = time.monotonic()
    previous_elapsed = history[-1]["elapsed_seconds"] if history else 0.0
    for epoch in range(start_epoch, args.ssl_epochs):
        if codebook and epoch == REFRESH_EPOCH and clusters.index == 0:
            clusters.teacher = snapshot_teacher_convs(model.encoder).to(args.device)
            clusters.centers, fit = fit_round(args, pool, mean, std, clusters.teacher,
                                              codebook["row_indices"].numpy(), codebook["positions"].numpy(),
                                              REFIT_SEED)
            reset_cluster_heads(model, optimizer, generator, seed=REFIT_SEED)
            clusters.fits.append(fit)
            clusters.index = 1
            save_ssl_epoch(directory, fp, epoch, model, optimizer, generator, history, clusters)
        model.train()
        lr = warmup_cosine_lr(SSL_LR, epoch, args.ssl_epochs)
        optimizer.param_groups[0]["lr"] = lr
        stats, means = ssl_epoch(model, optimizer, data, clusters, args.device)
        record = {"epoch": epoch + 1, "lr": lr,
                  "elapsed_seconds": previous_elapsed + time.monotonic() - started, **stats,
                  "codebook_round": clusters.index if codebook else None, **means}
        history.append(record)
        save_ssl_epoch(directory, fp, epoch + 1, model, optimizer, generator, history, clusters)
        print(json.dumps({"stage": "pretrain", "variant": variant, **record}), flush=True)
    write_torch_atomic(complete, {"fingerprint": fp, "encoder": cpu_state(model.encoder),
                                  "variant": variant, "epochs": args.ssl_epochs, "seed": SEED,
                                  "bootstrap_epoch": BOOTSTRAP_EPOCHS,
                                  "training_records": len(pool.train_rows),
                                  "codebook_round": clusters.index if codebook else None,
                                  "round_fits": clusters.fits, "parameters": parameter_count(model)})
    return complete


@torch.inference_mode()
def predict(model: TokenizationClassifier, data: DataLoader, device: str) -> np.ndarray:
    """
    Compute logits for every batch in evaluation mode.

    Parameters
    ----------
    model : TokenizationClassifier
        Classifier returning one logit per record.
    data : DataLoader
        Loader from ``loader``.
    device : str
        Device holding ``model``.

    Returns
    -------
    np.ndarray
        Logits in loader order.
    """
    model.eval()
    return np.concatenate([model(signal.to(device, non_blocking=True),
                                 beat_boundaries.to(device, non_blocking=True)).cpu().numpy()
                           for signal, _, _, beat_boundaries in data])


def completed_fine_tune(directory: Path, fingerprint: str) -> bool:
    """
    Verify a completed fine-tune against its recorded artifact digests.

    Parameters
    ----------
    directory : Path
        Fine-tune directory.
    fingerprint : str
        Identity the completed run must match.

    Returns
    -------
    bool
        True when a verified completion exists.

    Raises
    ------
    ValueError
        If the fingerprint or any artifact changed.
    """
    completion = directory / "completion.json"
    if not completion.exists():
        return False
    saved = json.loads(completion.read_text())
    if saved["fingerprint"] != fingerprint:
        raise ValueError(f"Completed fine-tune fingerprint mismatch: {directory}")
    for name, declared in saved["artifacts"].items():
        if sha256_file(directory / name) != declared:
            raise ValueError(f"Completed fine-tune artifact changed: {directory / name}")
    return True


def continued_classifier(variant: str, ssl_path: Path, ssl_epochs: int,
                         device: str) -> TokenizationClassifier:
    """
    Classifier with the shared head and this arm's continued encoder.

    Parameters
    ----------
    variant : str
        One of ``VARIANTS``.
    ssl_path : Path
        The arm's completed ``encoder.pt``.
    ssl_epochs : int
        Required continued-pretraining duration.
    device : str
        Target device.

    Returns
    -------
    TokenizationClassifier
        Classifier ready for fine-tuning.

    Raises
    ------
    ValueError
        If the encoder is from another arm or duration.
    """
    model = classifier_with_matched_head(variant, device)
    ssl = torch.load(ssl_path, map_location="cpu", weights_only=True)
    if ssl["variant"] != variant or ssl["epochs"] != ssl_epochs:
        raise ValueError("SSL encoder variant or duration mismatch")
    model.encoder.load_state_dict(ssl["encoder"])
    return model


def fine_tune_epoch(model: TokenizationClassifier, optimizer: torch.optim.Optimizer, data: DataLoader,
                    device: str) -> tuple[float, int, int, float]:
    """
    Train one supervised epoch.

    Parameters
    ----------
    model : TokenizationClassifier
        Classifier in training mode.
    optimizer : torch.optim.Optimizer
        Its optimizer.
    data : DataLoader
        Shuffled labeled loader with beat boundaries.
    device : str
        Device holding ``model``.

    Returns
    -------
    tuple[float, int, int, float]
        Record-weighted loss sum, updates, record exposures and the
        record-weighted boundary-disagreement sum.
    """
    total = 0.0
    updates = exposures = 0
    disagreement_total = 0.0
    for signal, target, _, beat_boundaries in data:
        signal = signal.to(device, non_blocking=True)
        target = target.to(device, dtype=torch.float32, non_blocking=True)
        beat_boundaries = beat_boundaries.to(device, non_blocking=True)
        loss = nn.functional.binary_cross_entropy_with_logits(model(signal, beat_boundaries), target)
        if model.variant in CHUNK_VARIANTS:
            disagreement_total += boundary_disagreement(model.encoder) * len(signal)
        checked_step(loss, model, optimizer, f"{model.variant} fine-tune")
        total += float(loss.detach()) * len(signal)
        updates += 1
        exposures += len(signal)
    return total, updates, exposures, disagreement_total


def fine_tune(args: argparse.Namespace, pool: cpc_pool.Pool, mean: np.ndarray, std: np.ndarray,
              boundaries: np.ndarray, source_hashes: dict[str, str], variant: str, budget: str) -> None:
    """
    Fine-tune one continued encoder on a label budget, resumably, then evaluate it.

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
    boundaries : np.ndarray
        Row-aligned beat boundaries.
    source_hashes : dict[str, str]
        Input digests from ``source_hashes``.
    variant : str
        One of ``VARIANTS``.
    budget : str
        Label fraction, ``"0.1"`` or ``"1"``.

    Raises
    ------
    ValueError
        If the SSL encoder, saved state or completed artifacts do not match.
    RuntimeError
        If no epoch completes.
    """
    rows, manifest_hashes = cpc_pool.manifest_rows(pool, args.manifest_dir, budget)
    seed_everything(SEED)
    directory = args.output_dir / f"{variant}_fraction{budget}_seed42"
    directory.mkdir(parents=True, exist_ok=True)
    ssl_path = args.output_dir / f"{variant}_ssl" / "encoder.pt"
    settings = {"stage": "train", "variant": variant, "budget": budget,
                "seed": SEED, "epochs": args.epochs, "patience": args.patience,
                "batch_size": args.batch_size, "encoder_lr": ENCODER_LR,
                "head_lr": HEAD_LR, "weight_decay": WEIGHT_DECAY,
                "augmentation": "none", "manifest_sha256": manifest_hashes,
                "ssl_checkpoint_sha256": sha256_file(ssl_path)}
    fp, inputs = cpc_pool.fingerprint(pool, source_hashes, mean, std, settings)
    if completed_fine_tune(directory, fp):
        return
    model = continued_classifier(variant, ssl_path, args.ssl_epochs, args.device)
    optimizer = torch.optim.AdamW([{"params": (p for p in model.encoder.parameters() if p.requires_grad),
                                    "lr": ENCODER_LR}, {"params": model.head.parameters(), "lr": HEAD_LR}],
                                  weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(SEED)
    development, calibration = partition_validation(rows["validation"])

    def rows_loader(selected: list[dict[str, str]], shuffle: bool = False) -> DataLoader:
        return loader(pool, selected, mean, std, boundaries, args.batch_size, shuffle,
                      generator if shuffle else None, args.device)

    train_data = rows_loader(rows["labeled_train"], shuffle=True)
    dev_data = rows_loader(development)
    start_epoch, history, best_model, best_auc, best_epoch = cpc_pool.resume_or_new(
        directory, fp, model, optimizer, generator)
    if start_epoch == 0:
        seed_everything(SEED)
    config_path = directory / "config.json"
    if config_path.exists() and json.loads(config_path.read_text())["fingerprint"] != fp:
        raise ValueError(f"Existing fine-tune config differs: {directory}")
    write_json_atomic(config_path, {"fingerprint": fp, "inputs": inputs,
                      "architecture": variant, "model_parameters": parameter_count(model),
                      "encoder_parameters": parameter_count(model.encoder),
                      "labeled_training_records": len(rows["labeled_train"]),
                      "development_records": len(development), "calibration_records": len(calibration),
                      "test_records": len(rows["test"])})
    dev_y = np.array([int(row["target"]) for row in development])
    started = time.monotonic()
    previous_elapsed = history[-1]["elapsed_seconds"] if history else 0.0
    for epoch in range(start_epoch, args.epochs):
        if epoch - best_epoch >= args.patience:
            break
        model.train()
        total, updates, exposures, disagreement = fine_tune_epoch(model, optimizer, train_data, args.device)
        dev_auc = float(roc_auc_score(dev_y, predict(model, dev_data, args.device)))
        if dev_auc > best_auc:
            best_auc, best_epoch, best_model = dev_auc, epoch + 1, cpu_state(model)
        record = {"epoch": epoch + 1, "train_loss": total / len(rows["labeled_train"]),
                  "development_auroc": dev_auc, "best_epoch": best_epoch,
                  "optimizer_updates": updates, "record_exposures": exposures,
                  "elapsed_seconds": previous_elapsed + time.monotonic() - started}
        if variant in CHUNK_VARIANTS:
            record["boundary_disagreement_with_fixed"] = disagreement / exposures
        history.append(record)
        cpc_pool.save_epoch(directory, fp, epoch + 1, model, optimizer, generator,
                            history, best_model, best_auc, best_epoch)
        print(json.dumps({"stage": "train", "variant": variant, "budget": budget, **record}), flush=True)
    if best_model is None:
        raise RuntimeError("No fine-tune epoch completed")
    model.load_state_dict(best_model)
    write_torch_atomic(directory / "model.pt", {"fingerprint": fp, "model": best_model,
                                                "best_epoch": best_epoch, "best_development_auroc": best_auc})
    calibration_logits = predict(model, rows_loader(calibration), args.device)
    test_logits = predict(model, rows_loader(rows["test"]), args.device)
    evaluate_predictions(f"{variant}_fraction{budget}", calibration_logits,
                         test_logits, calibration, rows["test"], directory, SEED, args.bootstrap)
    artifacts = {name: sha256_file(directory / name) for name in COMPLETED_ARTIFACTS}
    write_json_atomic(directory / "completion.json", {"fingerprint": fp, "artifacts": artifacts})


def profile_step(model: TokenizationPretrainer, optimizer: torch.optim.Optimizer, centers: torch.Tensor,
                 teacher: nn.Module | None, device: str) -> Callable[[tuple[Any, ...]], tuple[Any, int]]:
    """
    Build the clipped continued-pretraining update used by the profile stage.

    Parameters
    ----------
    model : TokenizationPretrainer
        Arm to update.
    optimizer : torch.optim.Optimizer
        Its optimizer.
    centers : torch.Tensor
        Placeholder cluster centers.
    teacher : nn.Module | None
        Frozen CNN snapshot for the cluster arm.
    device : str
        Device holding ``model``.

    Returns
    -------
    Callable[[tuple[Any, ...]], tuple[Any, int]]
        Update returning ``(loss, details)`` and the batch size.
    """
    def step(batch: tuple[Any, ...]) -> tuple[Any, int]:
        signal, _, _, beat_boundaries = batch
        signal = signal.to(device, non_blocking=True)
        beat_boundaries = beat_boundaries.to(device, non_blocking=True)
        loss, details = model(signal, beat_boundaries, centers, teacher)
        if model.variant in CHUNK_VARIANTS:
            details["boundary_disagreement_with_fixed"] = boundary_disagreement(model.encoder)
        checked_step(loss, model, optimizer, "profile")
        return (loss, details), len(signal)

    return step


def profile(args: argparse.Namespace, pool: cpc_pool.Pool, mean: np.ndarray, std: np.ndarray,
            boundaries: np.ndarray,
            bootstrap_weights: Bootstrap) -> None:
    """
    Measure five warmup and N real updates per arm with no saved checkpoint.

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
    boundaries : np.ndarray
        Row-aligned beat boundaries.
    bootstrap_weights : Bootstrap
        Output of ``bootstrap``.
    """
    encoder_state, head_state, _ = bootstrap_weights
    for variant in (VARIANTS if args.variant == "all" else (args.variant,)):
        seed_everything(SEED)
        model = TokenizationPretrainer(variant).to(args.device)
        load_bootstrap_weights(model, encoder_state, head_state)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                      lr=SSL_LR, weight_decay=WEIGHT_DECAY)
        generator = torch.Generator().manual_seed(SEED)
        data = loader(pool, pool.train_rows, mean, std, boundaries, args.ssl_batch_size, True, generator,
                      args.device)
        teacher = snapshot_teacher_convs(model.encoder).to(args.device) if variant == "clusteraux" else None
        centers = torch.randn(CLUSTERS, TOKEN_WIDTH, generator=torch.Generator().manual_seed(SEED))
        centers = centers.to(args.device)
        timing = time_updates(data, profile_step(model, optimizer, centers, teacher, args.device),
                              PROFILE_WARMUP, args.profile_updates, args.device,
                              "Insufficient ECGs for profile")
        loss, details = timing.last
        seconds = timing.seconds
        measured_updates = timing.updates - PROFILE_WARMUP
        print(json.dumps({"stage": "profile", "variant": variant,
                          "warmup_updates": PROFILE_WARMUP, "measured_updates": measured_updates,
                          "seconds_per_update": seconds / measured_updates,
                          "records_per_second": timing.measured_records / seconds,
                          "estimated_ssl_epoch_seconds":
                              len(pool.train_rows) * seconds / timing.measured_records,
                          "peak_gpu_gb": peak_gpu_gb(args.device),
                          "model_parameters": parameter_count(model),
                          "loss": float(loss.detach()), **details}), flush=True)


def report(args: argparse.Namespace) -> None:
    """
    Write test metrics and paired comparisons for every completed label budget.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``cache_dir``, ``output_dir`` and ``bootstrap``.
    """
    comparisons = {}
    metadata = json.loads((args.cache_dir / "complete.json").read_text())
    count = metadata.get("split_counts", {}).get("train")
    title = (f"{count:,} combined MIMIC and PTB-XL training ECGs" if count is not None
             else "combined MIMIC and PTB-XL training ECGs")
    lines = [f"# Continued CPC tokenization on {title}", "",
             "All five arms begin from the same completed 20-epoch local CPC encoder and future-prediction "
             "heads, then continue for the same fixed SSL budget. Every arm uses query positions 24 onward "
             "in each five-second half, horizons 4/8/12, and same-half temporal negatives outside ±3 "
             "tokens of the target.", "",
             "The cluster arm predicts K=64 train-only CNN code IDs from a frozen teacher snapshot, "
             "refreshing that snapshot and codebook after epoch five. Chunk arms update context at emitted "
             "boundaries and pool emitted contexts for classification while preserving original causal CNN "
             "tokens as CPC targets. Fixed chunks provide the matched boundary control for beat and learned "
             "chunks; the native continuation uses its original grid-context readout.", "",
             "This is a one-seed exploratory comparison on a PTB-XL test set already used in earlier project "
             "experiments, not a fresh external confirmation cohort.", ""]
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
    parser.add_argument("--beat-dir", type=Path, default=DEFAULT_BEATS)
    parser.add_argument("--bootstrap-dir", type=Path, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--manifest-dir", type=Path, default=cpc_pool.DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--ssl-batch-size", type=int, default=128)
    parser.add_argument("--ssl-epochs", type=int, default=10)
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


def run_stages(args: argparse.Namespace, pool: cpc_pool.Pool, hashes: dict[str, str],
               beats: np.ndarray) -> None:
    """
    Continue, fine-tune and report the requested arms.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    pool : cpc_pool.Pool
        Frozen CPC waveform cache.
    hashes : dict[str, str]
        Input digests from ``source_hashes``.
    beats : np.ndarray
        Row-aligned beat boundaries.
    """
    mean, std = pool.normalization(args.output_dir, hashes)
    boot = bootstrap(args, hashes)
    selected = VARIANTS if args.variant == "all" else (args.variant,)
    codebook = initial_codebook(args, pool, mean, std, hashes, boot[0]) if (
        args.stage in ("pretrain", "all") and "clusteraux" in selected) else None
    if args.stage in ("pretrain", "all"):
        for variant in selected:
            pretrain(args, pool, mean, std, beats, hashes, boot, variant,
                     codebook if variant == "clusteraux" else None)
    if args.stage in ("train", "all"):
        for budget in (("1", "0.1") if args.labels == "all" else (args.labels,)):
            for variant in selected:
                fine_tune(args, pool, mean, std, beats, hashes, variant, budget)
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
        hashes = source_hashes(args, pool)
        beats = load_beats(args.beat_dir, pool)
        if args.stage != "profile":
            run_stages(args, pool, hashes, beats)
            return
        with tempfile.TemporaryDirectory(prefix="cpc_tokenization_profile_") as temporary:
            mean, std = pool.normalization(temporary, hashes)
            profile(args, pool, mean, std, beats, bootstrap(args, hashes))


if __name__ == "__main__":
    main()
