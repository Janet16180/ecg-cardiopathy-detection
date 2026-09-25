#!/usr/bin/env python3
"""Experiment 015: matched cached-JEPA representation distillation into local CPC."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F  # noqa: N812 - conventional alias

from ecg_experiment.bounded_waveform_cache import BoundedWaveformCache
from ecg_experiment.cpc import CPCEncoder
from ecg_experiment.cpc_pool import Pool
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
from ecg_experiment.provenance import utc_now
from ecg_experiment.receipts import check_completion, check_existing_config, require_receipt, write_completion
from ecg_experiment.reproducibility import cpu_state, seed_everything
from ecg_experiment.training import checked_step, parameter_count

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data/processed/cpc_pool_40k"
MANIFEST = ROOT / "data/processed/ptbxl"
TEACHER = ROOT / "data/processed/pretrained/ecg-jepa-full-public"
SSL = ROOT / "outputs/experiment004_cpc_40k/cpc_ssl/encoder.pt"
NORMALIZATION = ROOT / "outputs/experiment004_cpc_40k/normalization.json"
OUTPUT = ROOT / "outputs/experiment015_jepa_cpc_distillation"
SEED = 42
EPOCHS = 5
ARMS = ("control", "distill")
TEACHER_WEIGHT = 0.1
TEACHER_DIMENSION = 768
TEACHER_RECORDS = 19126
CPC_WIDTH = 512
ENCODER_LR = 3e-4
HEAD_LR = 1e-3
WEIGHT_DECAY = 0.01
MIN_TEACHER_NORM = 1e-12
INITIAL_BEST = {"auc": -1.0, "epoch": 0, "model": None}
ARTIFACTS = {"history.json": "history_sha256", "best_student.pt": "best_student_sha256"}
POOL_FILES = (("signals.npy", "signals_sha256"), ("rows.csv", "rows_sha256"),
              ("ecg_ids.npy", "ecg_ids_sha256"))
PLANNING_GATE_SECONDS = 7200
REQUIRED_AUROC_GAIN = 0.002
REQUIRED_SPECIFICITY_GAIN = 0.02
SENSITIVITY_TOLERANCE = 0.005
MIN_VARIANCE_RATIO = 0.1
CODE_PATHS = ("scripts/experiments/run_jepa_cpc_distillation.py", "ecg_experiment/cpc.py",
              "ecg_experiment/bounded_waveform_cache.py",
              "scripts/experiments/run_cpc_experiment.py", "ecg_experiment/data.py",
              "ecg_experiment/cpc_pool.py", "ecg_experiment/files.py",
              "ecg_experiment/reproducibility.py", "ecg_experiment/pilot.py", "ecg_experiment/receipts.py",
              "ecg_experiment/training.py",
              "ecg_experiment/run.py", "ecg_experiment/evaluation.py",
              "docs/astra-next-model-ideas.md", "docs/experiment-015-distillation.md")


class Student(nn.Module):
    """CPC encoder with a binary head and a projection onto the JEPA teacher space."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = CPCEncoder()
        self.head = nn.Linear(CPC_WIDTH, 1)
        self.projector = nn.Linear(CPC_WIDTH, TEACHER_DIMENSION)

    def forward(self, waveforms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Classify waveforms and project their pooled context.

        Parameters
        ----------
        waveforms : torch.Tensor
            Normalized ``[batch, 12, 2500]`` waveforms.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Logits and teacher-space projections.
        """
        _, contexts = self.encoder(waveforms)
        pooled = self.encoder.pooled(contexts)
        return self.head(pooled).squeeze(-1), self.projector(pooled)

    def classifier_state(self) -> dict[str, torch.Tensor]:
        """
        CPU copy of the inference weights, without the training-only projector.

        Returns
        -------
        dict[str, torch.Tensor]
            Encoder and head parameters.
        """
        return {key: value for key, value in cpu_state(self).items() if key.startswith(("encoder.", "head."))}


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
    vectors : np.ndarray
        Cached JEPA teacher vectors aligned to the training rows.
    mean : np.ndarray
        Train-only per-lead mean shaped ``[12, 1]``.
    std : np.ndarray
        Train-only per-lead standard deviation shaped ``[12, 1]``.
    provenance : dict[str, Any]
        Source, code and settings identity of the experiment.
    pool_hash_mode : str
        Whether pool digests were reused from the previous receipt or recomputed.
    pool_stats : dict[str, dict[str, int]]
        File identities of the pool files.
    precheck_seconds : float
        Wall time of input verification.
    waveforms : BoundedWaveformCache | None
        Training then development waveforms in RAM.
    preload_seconds : float
        Wall time of the preload.
    """

    pool: Pool
    partitions: Partitions
    vectors: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    provenance: dict[str, Any]
    pool_hash_mode: str
    pool_stats: dict[str, dict[str, int]]
    precheck_seconds: float
    waveforms: BoundedWaveformCache | None = None
    preload_seconds: float = 0.0


def masked_loss(logits: torch.Tensor, projected: torch.Tensor, targets: torch.Tensor, exposed: torch.Tensor,
                teacher: torch.Tensor, weight: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Binary cross-entropy on exposed labels plus weighted cosine distance to the teacher.

    Both arms run both graphs; the control arm uses zero weight.

    Parameters
    ----------
    logits : torch.Tensor
        Student logits.
    projected : torch.Tensor
        Student teacher-space projections.
    targets : torch.Tensor
        Binary targets; hidden entries are ignored.
    exposed : torch.Tensor
        Boolean mask of exposed labels, shaped like ``logits``.
    teacher : torch.Tensor
        Teacher vectors; no gradient flows into them.
    weight : float
        Weight of the cosine term.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Total loss, cross-entropy and mean cosine distance.

    Raises
    ------
    ValueError
        If the mask is not boolean, has the wrong shape, or exposes nothing.
    """
    if exposed.dtype != torch.bool or exposed.shape != logits.shape:
        raise ValueError("Exposed-label mask must be boolean and match logits")
    if not exposed.any():
        raise ValueError("Every fixed batch must contain exposed labels")
    bce = F.binary_cross_entropy_with_logits(logits[exposed], targets[exposed])
    cosine = 1 - F.cosine_similarity(F.normalize(projected, dim=-1),
                                     F.normalize(teacher.detach(), dim=-1), dim=-1).mean()
    loss = bce + weight * cosine
    return loss, bce, cosine


def load_teacher(teacher_dir: Path, manifest_dir: Path,
                 train_ids: list[str]) -> tuple[dict[str, Any], np.ndarray]:
    """
    Load cached JEPA vectors for the training rows and check their provenance.

    Parameters
    ----------
    teacher_dir : Path
        Feature cache with ``metadata.json``, ``ecg_ids.npy`` and ``features.npy``.
    manifest_dir : Path
        PTB-XL manifest root.
    train_ids : list[str]
        Training ECG IDs in order.

    Returns
    -------
    tuple[dict[str, Any], np.ndarray]
        Teacher metadata and ``[len(train_ids), 768]`` float32 vectors.

    Raises
    ------
    ValueError
        If the cache is from another manifest or its vectors are invalid.
    """
    metadata = json.loads((teacher_dir / "metadata.json").read_text())
    if (metadata.get("feature_dimension") != TEACHER_DIMENSION
            or metadata.get("record_count") != TEACHER_RECORDS):
        raise ValueError("Unexpected JEPA teacher cache metadata")
    full_manifest_hash = sha256_file(manifest_dir / "seed42_fraction1/labeled_train.csv")
    if metadata.get("manifest_sha256", {}).get("labeled_train.csv") != full_manifest_hash:
        raise ValueError("Teacher training manifest differs from this label pool")
    teacher_ids = [str(item) for item in np.load(teacher_dir / "ecg_ids.npy", allow_pickle=False)]
    if len(set(teacher_ids)) != len(teacher_ids):
        raise ValueError("Duplicate teacher ECG IDs")
    teacher_index = {ecg_id: i for i, ecg_id in enumerate(teacher_ids)}
    if not set(train_ids) <= set(teacher_index):
        raise ValueError("Teacher misses eligible training ECGs")
    features = np.load(teacher_dir / "features.npy", mmap_mode="r")
    if features.shape != (TEACHER_RECORDS, TEACHER_DIMENSION) or features.dtype != np.float32:
        raise ValueError("Unexpected teacher feature shape/dtype")
    vectors = np.array(features[[teacher_index[ecg_id] for ecg_id in train_ids]], copy=True)
    if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) < MIN_TEACHER_NORM):
        raise ValueError("Invalid teacher vectors")
    return metadata, vectors


def load_normalization(path: Path, pool: Pool, cache_dir: Path,
                       ssl_cache: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the Experiment 004 normalization and check it came from the SSL pool.

    Parameters
    ----------
    path : Path
        ``normalization.json``.
    pool : Pool
        Frozen CPC waveform cache.
    cache_dir : Path
        Pool directory.
    ssl_cache : dict[str, str]
        Input digests recorded by the Experiment 004 SSL configuration.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Per-lead mean and standard deviation shaped ``[12, 1]``.

    Raises
    ------
    ValueError
        If the normalization is from other data or invalid.
    """
    norm = json.loads(path.read_text())
    if norm["source"]["signals_sha256"] != ssl_cache[str((cache_dir / "signals.npy").resolve())]:
        raise ValueError("Normalization is not from the CPC SSL pool")
    if norm["source"]["train_ids_sha256"] != sha256_json([row["ecg_id"] for row in pool.train_rows]):
        raise ValueError("Normalization training identities changed")
    mean = np.asarray(norm["mean"], np.float32)[:, None]
    std = np.asarray(norm["std"], np.float32)[:, None]
    if np.any(std <= 0) or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Invalid normalization")
    return mean, std


def hash_pool(pool: Pool, cache_dir: Path,
              previous: dict[str, Any]) -> tuple[dict[str, str], dict[str, dict[str, int]], str]:
    """
    Verify the pool files, reusing a previous receipt's digests while files are unchanged.

    Parameters
    ----------
    pool : Pool
        Frozen CPC waveform cache with declared digests.
    cache_dir : Path
        Pool directory.
    previous : dict[str, Any]
        Previous check receipt, or an empty dictionary.

    Returns
    -------
    tuple[dict[str, str], dict[str, dict[str, int]], str]
        Content digests, file identities and the hash mode.

    Raises
    ------
    ValueError
        If a file changed during hashing or differs from its declared digest.
    """
    hashes, stats = {}, {}
    mode = "verified_receipt_reuse"
    for filename, metadata_key in POOL_FILES:
        path = cache_dir / filename
        before = file_identity(path)
        prior_hash = previous.get("provenance", {}).get("pool_content_sha256", {}).get(filename)
        prior_stat = previous.get("pool_file_stats", {}).get(filename)
        if prior_hash == pool.metadata[metadata_key] and prior_stat == before:
            actual = prior_hash
        else:
            actual = sha256_file(path)
            if file_identity(path) != before:
                raise ValueError(f"Waveform pool {filename} changed during hashing")
            mode = "full_content_sha256"
        if actual != pool.metadata[metadata_key]:
            raise ValueError(f"Waveform pool {filename} changed")
        hashes[filename] = actual
        stats[filename] = before
    return hashes, stats, mode


def load_inputs(args: argparse.Namespace) -> PilotData:
    """
    Verify every pilot input and build the experiment provenance.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    PilotData
        Verified inputs, without preloaded waveforms.
    """
    started = time.monotonic()
    pool = Pool(args.cache_dir)
    partitions = load_partitions(pool, args.manifest_dir)
    teacher_meta, vectors = load_teacher(args.teacher_dir, args.manifest_dir,
                                         [row["ecg_id"] for row in partitions.full])
    base_config = check_final_ssl(args.ssl)
    mean, std = load_normalization(args.normalization, pool, args.cache_dir, base_config["inputs"]["cache"])
    manifest_paths = [args.manifest_dir / f"seed42_fraction{budget}" / f"{name}.csv"
                      for budget in BUDGETS for name in ("labeled_train", "validation", "test")]
    prior_path = args.output_dir / "provenance/verification.json"
    previous = json.loads(prior_path.read_text()) if prior_path.exists() else {}
    pool_hashes, pool_stats, pool_hash_mode = hash_pool(pool, args.cache_dir, previous)
    sources = [args.ssl, args.ssl.parent / "epoch_state.pt", args.ssl.parent / "config.json",
               args.normalization,
               args.teacher_dir / "metadata.json", args.teacher_dir / "ecg_ids.npy",
               args.teacher_dir / "features.npy", *manifest_paths]
    provenance = {"sources": {str(path.resolve()): sha256_file(path) for path in sources},
                  "code": {name: sha256_file(ROOT / name) for name in CODE_PATHS},
                  "pool_complete_sha256": sha256_file(args.cache_dir / "complete.json"),
                  "pool_content_sha256": pool_hashes,
                  "base_cpc_config_fingerprint": base_config["fingerprint"],
                  "teacher_metadata": teacher_meta,
                  "settings": {"seed": SEED, "epochs": EPOCHS, "batch_size": BATCH,
                               "save_every_updates": SAVE_EVERY, "teacher_weight": TEACHER_WEIGHT,
                               "encoder_lr": ENCODER_LR, "head_and_projection_lr": HEAD_LR,
                               "weight_decay": WEIGHT_DECAY}}
    return PilotData(pool, partitions, vectors, mean, std, provenance, pool_hash_mode, pool_stats,
                     time.monotonic() - started)


def preload(args: argparse.Namespace, data: PilotData) -> None:
    """
    Load training and development waveforms into RAM.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``max_cache_bytes`` and ``reserve_bytes``.
    data : PilotData
        Verified inputs, updated in place.

    Raises
    ------
    ValueError
        If the cache does not hold every requested row.
    """
    started = time.monotonic()
    combined = data.partitions.full + data.partitions.development
    data.waveforms = BoundedWaveformCache(data.pool, combined, max_bytes=args.max_cache_bytes,
                                          reserve_bytes=args.reserve_bytes, expected_source="ptbxl")
    if len(data.waveforms.signals) != len(combined):
        raise ValueError("Bounded cache row mismatch")
    data.preload_seconds = time.monotonic() - started


def make_model(ssl_path: Path, device: str) -> Student:
    """
    Build a student from the ordinary Experiment 004 CPC encoder.

    Parameters
    ----------
    ssl_path : Path
        Experiment 004 ``encoder.pt``.
    device : str
        Target device.

    Returns
    -------
    Student
        Student with seeded new head and projector.

    Raises
    ------
    ValueError
        If the checkpoint is not the 20-epoch ordinary CPC run.
    """
    seed_everything(SEED)
    model = Student()
    checkpoint = torch.load(ssl_path, map_location="cpu", weights_only=True)
    if checkpoint.get("variant") != "cpc" or checkpoint.get("epochs") != 20:
        raise ValueError("Expected Experiment 004 ordinary 20-epoch CPC checkpoint")
    model.encoder.load_state_dict(checkpoint["encoder"], strict=True)
    return model.to(device)


def make_optimizer(model: Student) -> torch.optim.AdamW:
    """
    AdamW with a lower rate for the pretrained encoder than for new layers.

    Parameters
    ----------
    model : Student
        Student to optimize.

    Returns
    -------
    torch.optim.AdamW
        Optimizer over encoder, head and projector.
    """
    return torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": ENCODER_LR},
                              {"params": model.head.parameters(), "lr": HEAD_LR},
                              {"params": model.projector.parameters(), "lr": HEAD_LR}],
                             weight_decay=WEIGHT_DECAY)


def identity(data: PilotData, budget: str, arm: str) -> dict[str, Any]:
    """
    Everything that determines one arm's result.

    Parameters
    ----------
    data : PilotData
        Verified inputs.
    budget : str
        Label budget.
    arm : str
        ``"control"`` or ``"distill"``.

    Returns
    -------
    dict[str, Any]
        Identity whose digest fingerprints the arm.
    """
    return {"provenance": data.provenance, "budget": budget, "arm": arm,
            "train_ids": [row["ecg_id"] for row in data.partitions.full],
            "development_ids": [row["ecg_id"] for row in data.partitions.development]}


def verify_roundtrip(args: argparse.Namespace, data: PilotData, budget: str, arm: str,
                     directory: Path) -> dict[str, int]:
    """
    Reload a one-epoch profile checkpoint into a fresh CPU model and compare its tensors.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``ssl``.
    data : PilotData
        Verified inputs.
    budget : str
        Profiled label budget.
    arm : str
        Profiled arm.
    directory : Path
        Profile arm directory holding ``resume.pt``.

    Returns
    -------
    dict[str, int]
        Roundtrip receipt.
    """
    original = torch.load(directory / "resume.pt", map_location="cpu", weights_only=False)
    probe = make_model(args.ssl, "cpu")
    optimizer = make_optimizer(probe)
    progress = load_state(directory, sha256_json(identity(data, budget, arm)), probe, optimizer,
                          dict(INITIAL_BEST))
    return check_roundtrip(original, progress, probe, optimizer)


@torch.inference_mode()
def predict_development(model: Student, data: PilotData, device: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute development logits and pooled representations in manifest order.

    Parameters
    ----------
    model : Student
        Student to evaluate.
    data : PilotData
        Preloaded inputs; development rows follow training rows.
    device : str
        Device holding ``model``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Logits and ``[records, 512]`` pooled contexts.
    """
    model.eval()
    values = []
    representations = []
    for x in development_batches(data.waveforms, len(data.partitions.full), len(data.partitions.development),
                                 data.mean, data.std, device):
        _, contexts = model.encoder(x)
        pooled = model.encoder.pooled(contexts)
        values.extend(model.head(pooled).squeeze(-1).cpu().numpy().tolist())
        representations.append(pooled.cpu().numpy())
    return np.asarray(values), np.concatenate(representations)


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
    return {"auroc": float(roc_auc_score(labels, logits)),
            **fold_operating_point(labels, groups, clipped_sigmoid(logits), SEED)}


def train_epoch(args: argparse.Namespace, data: PilotData, model: Student, optimizer: torch.optim.Optimizer,
                progress: Progress, labels: tuple[np.ndarray, np.ndarray, float],
                checkpoint: tuple[Path, str], clock: tuple[float, float | None]) -> None:
    """
    Train the remaining batches of the current epoch, checkpointing periodically.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``device``.
    data : PilotData
        Preloaded inputs.
    model : Student
        Student being trained.
    optimizer : torch.optim.Optimizer
        Its optimizer.
    progress : Progress
        Position, updated in place.
    labels : tuple[np.ndarray, np.ndarray, float]
        Exposure mask, targets and the teacher-loss weight.
    checkpoint : tuple[Path, str]
        Arm directory and fingerprint.
    clock : tuple[float, float | None]
        ``time.monotonic`` start of this invocation and the deadline.

    Raises
    ------
    SystemExit
        With ``INTERRUPTED_EXIT`` after checkpointing on SIGTERM or the deadline.
    """
    exposed, targets, weight = labels
    directory, fingerprint = checkpoint
    started, deadline = clock
    model.train()
    batches = fixed_batches(len(data.partitions.full), exposed, progress.epoch, SEED)
    for position in range(progress.batch, len(batches)):
        indices = batches[position]
        x = normalized_batch(data.waveforms, indices, data.mean, data.std, args.device)
        target = torch.from_numpy(targets[indices]).to(args.device)
        mask = torch.from_numpy(exposed[indices]).to(args.device)
        teacher = torch.from_numpy(data.vectors[indices]).to(args.device)
        logits, projection = model(x)
        loss, bce, cosine = masked_loss(logits, projection, target, mask, teacher, weight)
        checked_step(loss, model, optimizer, "training")
        totals = progress.totals
        totals["updates"] = totals.get("updates", 0) + 1
        totals["record_exposures"] = totals.get("record_exposures", 0) + len(indices)
        totals["label_exposures"] = totals.get("label_exposures", 0) + int(mask.sum())
        totals["bce_sum"] = totals.get("bce_sum", 0.0) + float(bce.detach())
        totals["cosine_sum"] = totals.get("cosine_sum", 0.0) + float(cosine.detach())
        progress.batch = position + 1
        stop = interrupted(deadline)
        if progress.batch % SAVE_EVERY == 0 or stop:
            save_state(directory, fingerprint, model, optimizer, progress,
                       progress.elapsed_seconds + time.monotonic() - started)
        if stop:
            raise SystemExit(INTERRUPTED_EXIT)


def finish_epoch(args: argparse.Namespace, data: PilotData, model: Student, optimizer: torch.optim.Optimizer,
                 progress: Progress, arm: tuple[str, str], checkpoint: tuple[Path, str],
                 started: float) -> None:
    """
    Screen development data, keep the best student and save the epoch boundary.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``device``.
    data : PilotData
        Preloaded inputs.
    model : Student
        Student being trained.
    optimizer : torch.optim.Optimizer
        Its optimizer.
    progress : Progress
        Position, updated in place.
    arm : tuple[str, str]
        Label budget and arm name.
    checkpoint : tuple[Path, str]
        Arm directory and fingerprint.
    started : float
        ``time.monotonic`` start of this invocation.

    Raises
    ------
    RuntimeError
        If development representations or weights became nonfinite.
    """
    budget, name = arm
    directory, fingerprint = checkpoint
    logits, representations = predict_development(model, data, args.device)
    screen = development_screen(data.partitions.development, logits)
    screen["mean_feature_variance"] = float(np.var(representations, axis=0).mean())
    if not np.isfinite(screen["mean_feature_variance"]):
        raise RuntimeError("Nonfinite development representations")
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise RuntimeError("Nonfinite model weights")
    if screen["auroc"] > progress.best["auc"]:
        progress.best = {"auc": screen["auroc"], "epoch": progress.epoch + 1,
                         "model": model.classifier_state()}
        write_torch_atomic(directory / "best_student.pt", {"fingerprint": fingerprint,
                           "epoch": progress.epoch + 1, "model": progress.best["model"]})
    totals = progress.totals
    record = {"epoch": progress.epoch + 1, "arm": name, "budget": budget,
              "train_bce_mean_batch": totals["bce_sum"] / totals["updates"],
              "train_cosine_mean_batch": totals["cosine_sum"] / totals["updates"],
              "optimizer_updates": totals["updates"], "record_exposures": totals["record_exposures"],
              "label_exposures": totals["label_exposures"], "development": screen,
              "best_epoch": progress.best["epoch"],
              "elapsed_seconds": progress.elapsed_seconds + time.monotonic() - started}
    progress.history.append(record)
    print(json.dumps(record), flush=True)
    progress.epoch += 1
    progress.batch, progress.totals = 0, {}
    save_state(directory, fingerprint, model, optimizer, progress,
               progress.elapsed_seconds + time.monotonic() - started)


def run_arm(args: argparse.Namespace, data: PilotData, budget: str, arm: str, directory: Path,
            max_epochs: int, deadline: float | None = None) -> list[dict[str, Any]]:
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
    arm : str
        ``"control"`` or ``"distill"``.
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
    weight = TEACHER_WEIGHT if arm == "distill" else 0.0
    exposed, targets = exposed_labels(data.partitions.full, data.partitions.limited, budget)
    arm_identity = identity(data, budget, arm)
    fingerprint = sha256_json(arm_identity)
    if (directory / "completion.json").exists():
        return check_completion(directory, fingerprint, ARTIFACTS)
    config_path = directory / "config.json"
    check_existing_config(config_path, fingerprint)
    model = make_model(args.ssl, args.device)
    optimizer = make_optimizer(model)
    progress = load_state(directory, fingerprint, model, optimizer, dict(INITIAL_BEST))
    write_json_atomic(config_path, {"fingerprint": fingerprint, "identity": arm_identity,
                                    "model_parameters": parameter_count(model),
                                    "inference_parameters": parameter_count(model.encoder)
                                    + parameter_count(model.head),
                                    "exposed_training_labels": int(exposed.sum())})
    checkpoint = (directory, fingerprint)
    started = time.monotonic()
    while progress.epoch < max_epochs:
        train_epoch(args, data, model, optimizer, progress, (exposed, targets, weight), checkpoint,
                    (started, deadline))
        finish_epoch(args, data, model, optimizer, progress, (budget, arm), checkpoint, started)
    if progress.epoch == EPOCHS and max_epochs == EPOCHS:
        write_completion(directory, fingerprint, progress.best, ARTIFACTS)
    return progress.history


def best_rows(output_dir: Path) -> tuple[list[str], dict[tuple[str, str], dict[str, Any]]] | None:
    """
    Summarize each completed arm's best development epoch.

    Parameters
    ----------
    output_dir : Path
        Pilot output directory.

    Returns
    -------
    tuple[list[str], dict[tuple[str, str], dict[str, Any]]] | None
        Report table rows and best-epoch history records keyed by
        ``(budget, arm)``, or ``None`` while an arm is incomplete.

    Raises
    ------
    ValueError
        If a completed arm lacks epochs.
    """
    lines, results = [], {}
    for budget in BUDGETS:
        for arm in ARMS:
            directory = output_dir / f"{arm}_fraction{budget}_seed42"
            if not (directory / "completion.json").exists():
                return None
            history = json.loads((directory / "history.json").read_text())
            if len(history) != EPOCHS:
                raise ValueError("Completed pilot lacks five epochs")
            best_epoch = json.loads((directory / "completion.json").read_text())["best_epoch"]
            row = history[best_epoch - 1]
            results[(budget, arm)] = row
            dev = row["development"]
            lines.append(f"| {budget} | {arm} | {best_epoch} | {dev['auroc']:.4f} | "
                         f"{dev['fold_specificity']:.4f} | {dev['fold_sensitivity']:.4f} | "
                         f"{dev['mean_feature_variance']:.6g} | "
                         f"{sum(r['optimizer_updates'] for r in history)} | "
                         f"{sum(r['label_exposures'] for r in history)} | "
                         f"{history[-1]['elapsed_seconds']:.1f} |")
    return lines, results


def screen_lines(results: dict[tuple[str, str], dict[str, Any]]) -> tuple[list[str], bool]:
    """
    Apply the prespecified development go/no-go screen.

    Parameters
    ----------
    results : dict[tuple[str, str], dict[str, Any]]
        Best-epoch history records keyed by ``(budget, arm)``.

    Returns
    -------
    tuple[list[str], bool]
        One report line per budget, and whether a second seed is warranted.
    """
    lines = []
    positive = False
    sensitivity_safe = True
    for budget in BUDGETS:
        a, b = results[(budget, "distill")]["development"], results[(budget, "control")]["development"]
        delta_auc = a["auroc"] - b["auroc"]
        delta_spec = a["fold_specificity"] - b["fold_specificity"]
        delta_sens = a["fold_sensitivity"] - b["fold_sensitivity"]
        variance_ratio = a["mean_feature_variance"] / max(b["mean_feature_variance"], 1e-20)
        positive_here = (delta_auc >= REQUIRED_AUROC_GAIN and delta_spec >= REQUIRED_SPECIFICITY_GAIN
                         and delta_sens >= -SENSITIVITY_TOLERANCE and variance_ratio >= MIN_VARIANCE_RATIO)
        positive |= positive_here
        sensitivity_safe &= delta_sens >= -SENSITIVITY_TOLERANCE
        lines.append(f"Budget {budget}: distill minus control AUROC {delta_auc:+.4f}, "
                     f"patient-fold specificity {delta_spec:+.4f}, sensitivity {delta_sens:+.4f}, "
                     f"feature-variance ratio {variance_ratio:.3f}; "
                     f"positive-screen criteria {'met' if positive_here else 'not met'}.")
    other_budget_safe = all(results[(budget, "distill")]["development"]["auroc"]
                            - results[(budget, "control")]["development"]["auroc"] >= -REQUIRED_AUROC_GAIN
                            for budget in BUDGETS)
    return lines, positive and other_budget_safe and sensitivity_safe


def report_pilot(output_dir: Path) -> None:
    """
    Write the development-only report once all four arms are complete.

    Parameters
    ----------
    output_dir : Path
        Pilot output directory.
    """
    summary = best_rows(output_dir)
    if summary is None:
        return
    rows, results = summary
    screen, advance = screen_lines(results)
    lines = ["# Experiment 015 development-only distillation pilot", "",
             "Five fixed epochs per arm; no calibration or test predictions were used.", "",
             "| Label budget | Arm | Best epoch | Development AUROC | Patient-fold specificity | "
             "Patient-fold sensitivity | Mean feature variance | Updates | Labeled exposures | "
             "Wall seconds |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |", *rows,
             "", "## Prespecified development screen", "", *screen,
             "", ("A second matched seed is warranted before calibration/test." if advance else
                  "The prespecified development go/no-go screen did not pass; "
                  "retain all runs as exploratory evidence."),
             "", "The teacher has external pretraining exposure and uncertain overlap. "
             "The endpoint is a diagnostic annotation proxy, not verified health or referral need.", ""]
    (output_dir / "report.md").write_text("\n".join(lines))
    write_json_atomic(output_dir / "completion.json", {
        "status": "development_pilot_complete", "report_sha256": sha256_file(output_dir / "report.md"),
        "advance_to_second_seed": bool(advance),
        "arm_completions_sha256": {
            f"{arm}_fraction{budget}_seed42":
                sha256_file(output_dir / f"{arm}_fraction{budget}_seed42/completion.json")
            for budget in BUDGETS for arm in ARMS}})


def write_check(args: argparse.Namespace, data: PilotData, argv: Sequence[str]) -> None:
    """
    Record the CPU input check that must precede any GPU stage.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``output_dir``.
    data : PilotData
        Verified inputs.
    argv : Sequence[str]
        Command-line arguments, recorded in the receipt.
    """
    check_waveform_sample(data.pool, data.partitions.full[0])
    partitions = data.partitions
    receipt = {"stage": "check", "checked_at_utc": utc_now(),
               "command": [sys.executable, "-m", "scripts.experiments.run_jepa_cpc_distillation", *argv],
               "train": len(partitions.full), "limited": len(partitions.limited),
               "development": len(partitions.development),
               "calibration": len(partitions.calibration), "test": len(partitions.test),
               "precheck_seconds": data.precheck_seconds,
               "pool_hash_mode": data.pool_hash_mode,
               "pool_file_stats": data.pool_stats,
               "fingerprint": sha256_json(data.provenance),
               "provenance": data.provenance}
    write_json_atomic(args.output_dir / "provenance/verification.json", receipt)
    print(json.dumps({key: value for key, value in receipt.items() if key != "provenance"}), flush=True)


def profile(args: argparse.Namespace, data: PilotData, deadline: float) -> None:
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
    """
    def profile_arm(arm: str, directory: Path) -> dict[str, int]:
        run_arm(args, data, "1", arm, directory, 1, deadline)
        return verify_roundtrip(args, data, "1", arm, directory)

    durations, roundtrips = profile_arms("experiment015_profile_", ARMS, profile_arm)
    estimate = data.precheck_seconds + data.preload_seconds + 2 * EPOCHS * sum(durations.values())
    receipt = {"stage": "profile", "device": args.device, "full_epoch_seconds": durations,
               "checkpoint_roundtrips": roundtrips,
               "precheck_seconds": data.precheck_seconds,
               "preload_seconds": data.preload_seconds,
               "conservative_four_run_seconds": estimate,
               "planning_gate_seconds": PLANNING_GATE_SECONDS,
               "gate_passed": estimate <= PLANNING_GATE_SECONDS,
               "peak_gpu_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
               "fingerprint": sha256_json(data.provenance)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output_dir / "profile.json", receipt)
    print(json.dumps(receipt), flush=True)


def require_profile(output_dir: Path, fingerprint: str) -> None:
    """
    Require a passing real-data GPU profile of these exact inputs before training.

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
        raise ValueError("A complete real-data GPU profile is required before training")
    receipt = json.loads(path.read_text())
    if (receipt.get("stage") != "profile" or receipt.get("device") != "cuda"
            or receipt["fingerprint"] != fingerprint or not receipt["gate_passed"]):
        raise ValueError("GPU profile fingerprint or two-hour planning gate failed")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse and validate the command line.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or ``None`` for ``sys.argv``.

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
    parser.add_argument("--teacher-dir", type=Path, default=TEACHER)
    parser.add_argument("--ssl", type=Path, default=SSL)
    parser.add_argument("--normalization", type=Path, default=NORMALIZATION)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--max-cache-bytes", type=int, default=DEFAULT_MAX_CACHE_BYTES)
    parser.add_argument("--reserve-bytes", type=int, default=DEFAULT_RESERVE_BYTES)
    parser.add_argument("--max-wall-seconds", type=int, default=PLANNING_GATE_SECONDS)
    args = parser.parse_args(argv)
    if (args.threads < 1 or args.max_cache_bytes < MIN_CACHE_BYTES or args.reserve_bytes < 0
            or args.max_wall_seconds < 1):
        parser.error("Invalid threads or cache memory limits")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    return args


def main(argv: list[str] | None = None) -> None:
    """
    Run the check, profile or training stage.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or ``None`` for ``sys.argv``.
    """
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    install_stop_handler()
    deadline = time.monotonic() + args.max_wall_seconds
    with gpu_lock(args.device):
        data = load_inputs(args)
        if args.stage == "check":
            write_check(args, data, sys.argv[1:] if argv is None else list(argv))
            return
        preload(args, data)
        fingerprint = sha256_json(data.provenance)
        if args.device == "cuda":
            require_receipt(args.output_dir / "provenance/verification.json", fingerprint,
                            "Verified CPU input/source receipt is missing or changed")
        print(json.dumps({"stage": args.stage, "precheck_seconds": data.precheck_seconds,
                          "preload_seconds": data.preload_seconds,
                          "pool_hash_mode": data.pool_hash_mode,
                          "cache_bytes": int(data.waveforms.signals.nbytes)}), flush=True)
        if args.stage == "profile":
            profile(args, data, deadline)
            return
        if args.device == "cuda":
            require_profile(args.output_dir, fingerprint)
        run_pilot(ARMS, deadline, lambda budget, arm: run_arm(
            args, data, budget, arm, args.output_dir / f"{arm}_fraction{budget}_seed42", EPOCHS, deadline))
        report_pilot(args.output_dir)


if __name__ == "__main__":
    main()
