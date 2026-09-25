#!/usr/bin/env python3
"""Reproducible xECG fine-tuning on the fixed PTB-XL proxy task.

The official pretrained encoder sees a full ten-second, twelve-lead ECG in
physical mV at 100 Hz. Development patients select the epoch; calibration
patients set the probability threshold; test patients are evaluated once.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ecg_experiment.evaluation import evaluate_predictions, partition_validation
from ecg_experiment.files import read_csv, sha256_file, write_json_atomic, write_torch_atomic
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.provenance import git_head
from ecg_experiment.receipts import artifact_hashes, verified_completion
from ecg_experiment.reproducibility import cpu_state, flat_rng_state, restore_flat_rng_state, seed_everything
from ecg_experiment.training import optimizer_state_bytes, peak_gpu_bytes, require_cuda

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_ROOT = ROOT / "data/processed/ptbxl"
DEFAULT_RAW_DIR = ROOT / "data/raw/ptb-xl/1.0.3"
DEFAULT_CHECKPOINT_DIR = ROOT / "third_party/checkpoints/xecg"
DEFAULT_CACHE_DIR = ROOT / "data/processed/ptbxl/xecg_views"
DEFAULT_OUTPUT_DIR = ROOT / "outputs/experiment007_xecg"
EXPECTED_COUNTS = {"full": 15360, "ten_percent": 1518}
VALIDATION_RECORDS = 1870
TEST_RECORDS = 1896
DEVELOPMENT_RECORDS = 1306
CALIBRATION_RECORDS = 564
CACHE_RECORDS = 19126
SPLITS = ("labeled_train", "validation", "test")
XECG_VIEW_SHAPE = (1000, 12)
BACKBONE_BLOCKS = 9
CORE_LR = 3e-5
HEAD_LR = 1e-3
WEIGHT_DECAY = 0.1
LAYERWISE_LR_DECAY = 0.75
GRADIENT_CLIP = 3.0
DROP_PATH_PROB = 0.5
PROFILE_STEPS = 2
COMPLETION_FILES = ("model.pt", "metrics.json", "test_predictions.csv", "calibration_predictions.npz",
                    "config.json", "history.json")
CHECKPOINT_FILES = ("model.safetensors", "config.json", "xECG.py", "download_provenance.json")
TRAINING_STAGES = ("profile", "train", "all")

Rows = list[dict[str, str]]
Manifests = tuple[dict[str, Rows], Rows, Rows, dict[str, str]]


def budgets(value: str) -> tuple[str, ...]:
    """
    Expand a ``--budget`` choice into label budgets.

    Parameters
    ----------
    value : str
        ``"all"``, ``"full"`` or ``"ten_percent"``.

    Returns
    -------
    tuple[str, ...]
        Selected label budgets in run order.

    Raises
    ------
    ValueError
        If ``value`` names no known budget.
    """
    if value == "all":
        return ("full", "ten_percent")
    if value not in EXPECTED_COUNTS:
        raise ValueError(f"Unknown label budget: {value}")
    return (value,)


def manifest_dir(root: Path, budget: str) -> Path:
    """
    Return the manifest directory of a label budget.

    Parameters
    ----------
    root : Path
        Directory holding the seed-42 PTB-XL manifests.
    budget : str
        ``"full"`` or ``"ten_percent"``.

    Returns
    -------
    Path
        Manifest directory for ``budget``.
    """
    return root / ("seed42_fraction1" if budget == "full" else "seed42_fraction0.1")


def _identities(rows: Rows) -> dict[str, tuple[str, str]]:
    return {row["ecg_id"]: (row["patient_id"], row["filename_hr"]) for row in rows}


def _check_label_budgets(full_train: Rows, small_train: Rows) -> None:
    """Require the fixed budget sizes and a ten-percent subset of the full budget."""
    if len(full_train) != EXPECTED_COUNTS["full"] or len(small_train) != EXPECTED_COUNTS["ten_percent"]:
        raise ValueError("Labeled manifests do not match the fixed PTB-XL budgets")
    full_by_id = {row["ecg_id"]: row for row in full_train}
    if len(full_by_id) != len(full_train) or len(_identities(small_train)) != len(small_train):
        raise ValueError("Duplicate training ECG identifiers")
    if any(full_by_id.get(row["ecg_id"]) != row for row in small_train):
        raise ValueError("Ten-percent training manifest differs from full training manifest")


def _check_partitions(rows: dict[str, Rows]) -> None:
    """Require the official evaluation sizes and disjoint records and patients."""
    if len(rows["validation"]) != VALIDATION_RECORDS or len(rows["test"]) != TEST_RECORDS:
        raise ValueError("Unexpected official validation or test budget")
    ids = [row["ecg_id"] for name in SPLITS for row in rows[name]]
    if len(ids) != len(set(ids)):
        raise ValueError("Training, validation, and test ECG identifiers overlap")
    patients = [{row["patient_id"] for row in rows[name]} for name in SPLITS]
    if any(patients[i] & patients[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Training, validation, and test patients overlap")


def load_manifests(root: Path, budget: str) -> Manifests:
    """
    Check both budgets and all patient boundaries before fitting either.

    Parameters
    ----------
    root : Path
        Directory holding the seed-42 PTB-XL manifests.
    budget : str
        Label budget to return.

    Returns
    -------
    tuple
        Split rows, development rows, calibration rows, and manifest digests.

    Raises
    ------
    ValueError
        If any budget, split size, overlap, or label check fails.
    """
    full, small = manifest_dir(root, "full"), manifest_dir(root, "ten_percent")
    rows = {name: read_csv(manifest_dir(root, budget) / f"{name}.csv") for name in SPLITS}
    _check_label_budgets(read_csv(full / "labeled_train.csv"),
                         read_csv(small / "labeled_train.csv"))
    for name in ("validation", "test"):
        if rows[name] != read_csv(full / f"{name}.csv"):
            raise ValueError(f"{name} differs across label budgets")
    _check_partitions(rows)
    if len(rows["labeled_train"]) != EXPECTED_COUNTS[budget]:
        raise ValueError("Selected training budget has wrong size")
    if {row["target"] for row in rows["labeled_train"]} != {"0", "1"}:
        raise ValueError("Training labels must contain both classes")
    development, calibration = partition_validation(rows["validation"])
    if len(development) != DEVELOPMENT_RECORDS or len(calibration) != CALIBRATION_RECORDS:
        raise ValueError("Development/calibration split differs from the fixed protocol")
    hashes = {f"{name}.csv": sha256_file(manifest_dir(root, budget) / f"{name}.csv") for name in SPLITS}
    hashes["full_labeled_train.csv"] = sha256_file(full / "labeled_train.csv")
    hashes["ten_percent_labeled_train.csv"] = sha256_file(small / "labeled_train.csv")
    return rows, development, calibration, hashes


class CachedECGs(Dataset):
    """
    Labeled xECG inputs read from the 100 Hz waveform cache.

    Parameters
    ----------
    cache : np.ndarray
        Memory-mapped ``[records, 1000, 12]`` physical-mV cache.
    index : dict[int, int]
        Cache row of each integer ECG identifier.
    rows : list[dict[str, str]]
        Manifest rows with ``ecg_id`` and ``target``.
    """

    def __init__(self, cache: np.ndarray, index: dict[int, int], rows: Rows) -> None:
        self.cache = cache
        self.indices = [index[int(row["ecg_id"])] for row in rows]
        self.targets = [float(row["target"]) for row in rows]

    def __len__(self) -> int:
        """Return the number of records."""
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return one ECG and its binary target.

        Parameters
        ----------
        i : int
            Record position.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``[1000, 12]`` float32 signal and scalar float32 target.

        Raises
        ------
        ValueError
            If the cached signal is malformed or nonfinite.
        """
        signal = np.array(self.cache[self.indices[i]], dtype=np.float32, copy=True)
        if signal.shape != XECG_VIEW_SHAPE or not np.isfinite(signal).all():
            raise ValueError("xECG cache must contain finite [1000,12] physical-mV ECGs")
        return torch.from_numpy(signal), torch.tensor(self.targets[i], dtype=torch.float32)


def _add_group(groups: list[dict[str, Any]], used: set[int], name: str,
               params: Iterable[nn.Parameter], lr: float, weight_decay: float) -> None:
    """Append the trainable ``params`` as one group, rejecting any reused parameter."""
    selected = [param for param in params if param.requires_grad]
    for param in selected:
        if id(param) in used:
            raise ValueError(f"Overlapping optimizer parameter group: {name}")
        used.add(id(param))
    if selected:
        groups.append({"params": selected, "lr": lr, "weight_decay": weight_decay, "name": name})


def layerwise_parameter_groups(model: nn.Module, core_lr: float = CORE_LR, head_lr: float = HEAD_LR,
                               weight_decay: float = WEIGHT_DECAY,
                               decay: float = LAYERWISE_LR_DECAY) -> list[dict[str, Any]]:
    """
    Assign nine xLSTM blocks monotonically increasing LRs, as official PTB-XL.

    Parameters
    ----------
    model : nn.Module
        xECG classifier with ``backbone`` and ``head``.
    core_lr : float
        Learning rate of the last backbone layer.
    head_lr : float
        Learning rate of the binary head.
    weight_decay : float
        AdamW weight decay of every group.
    decay : float
        Per-layer learning-rate multiplier toward the input.

    Returns
    -------
    list[dict[str, Any]]
        Named AdamW parameter groups covering every trainable parameter once.

    Raises
    ------
    ValueError
        If the backbone does not have nine blocks or the groups do not cover
        the classifier exactly.
    """
    blocks = model.backbone.core.model.blocks
    if len(blocks) != BACKBONE_BLOCKS:
        raise ValueError(f"Expected nine xECG backbone blocks, found {len(blocks)}")
    groups: list[dict[str, Any]] = []
    used: set[int] = set()
    _add_group(groups, used, "patch_embedding", model.backbone.patch_embedding.parameters(),
               core_lr * decay ** (BACKBONE_BLOCKS + 1), weight_decay)
    for index, block in enumerate(blocks):
        _add_group(groups, used, f"core_block_{index}", block.parameters(),
                   core_lr * decay ** (BACKBONE_BLOCKS - index), weight_decay)
    remaining = [param for param in model.backbone.parameters()
                 if param.requires_grad and id(param) not in used]
    _add_group(groups, used, "core_remaining", remaining, core_lr, weight_decay)
    _add_group(groups, used, "binary_head", model.head.parameters(), head_lr, weight_decay)
    expected = {id(param) for param in model.parameters() if param.requires_grad}
    if expected != used:
        raise ValueError("Optimizer groups do not cover the full xECG classifier")
    return groups


def make_scheduler(optimizer: torch.optim.Optimizer, steps_per_epoch: int,
                   epochs: int) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Build a one-epoch linear warmup followed by a half-cosine decay.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer whose group learning rates are scaled.
    steps_per_epoch : int
        Optimizer updates per epoch; also the warmup length.
    epochs : int
        Epoch ceiling that sets the decay length.

    Returns
    -------
    torch.optim.lr_scheduler.LambdaLR
        Scheduler stepped once per optimizer update.
    """
    warmup, total = steps_per_epoch, steps_per_epoch * epochs

    def factor(step: int) -> float:
        if step < warmup:
            return float(step) / max(1, warmup)
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def save_resume(path: Path, fingerprint: dict[str, Any], model: nn.Module,
                optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LambdaLR,
                generator: torch.Generator, best_state: dict[str, torch.Tensor] | None,
                best_epoch: int, best_auc: float, history: list[dict[str, Any]], elapsed: float) -> None:
    """
    Atomically save an epoch-boundary resume checkpoint.

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
    scheduler : torch.optim.lr_scheduler.LambdaLR
        Current scheduler.
    generator : torch.Generator
        Training loader shuffle generator.
    best_state : dict[str, torch.Tensor] | None
        CPU weights of the best development epoch.
    best_epoch : int
        One-based best development epoch.
    best_auc : float
        Best development AUROC.
    history : list[dict[str, Any]]
        One row per completed epoch.
    elapsed : float
        Training seconds so far.
    """
    write_torch_atomic(path, {
        "version": 1, "fingerprint": fingerprint, "model": cpu_state(model),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "loader_rng": generator.get_state(), "best_model": best_state,
        "best_epoch": best_epoch, "best_auc": best_auc, "history": history,
        "elapsed_seconds": elapsed, **flat_rng_state(),
    })


def load_resume(path: Path, fingerprint: dict[str, Any], model: nn.Module,
                optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LambdaLR,
                generator: torch.Generator) -> dict[str, Any]:
    """
    Restore training state from an epoch-boundary resume checkpoint.

    Parameters
    ----------
    path : Path
        Checkpoint written by ``save_resume``.
    fingerprint : dict[str, Any]
        Inputs of the current run.
    model : nn.Module
        Classifier to restore in place.
    optimizer : torch.optim.Optimizer
        Optimizer to restore in place.
    scheduler : torch.optim.lr_scheduler.LambdaLR
        Scheduler to restore in place.
    generator : torch.Generator
        Loader generator to restore in place.

    Returns
    -------
    dict[str, Any]
        The loaded checkpoint.

    Raises
    ------
    ValueError
        If the checkpoint belongs to other inputs or is malformed.
    """
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("version") != 1 or saved.get("fingerprint") != fingerprint:
        raise ValueError("xECG resume configuration differs from current inputs")
    history = saved["history"]
    if (not history or any(row.get("epoch") != index + 1 for index, row in enumerate(history))
            or saved["best_epoch"] not in range(1, len(history) + 1)
            or saved["best_model"] is None):
        raise ValueError("Malformed xECG resume checkpoint")
    model.load_state_dict(saved["model"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    scheduler.load_state_dict(saved["scheduler"])
    generator.set_state(saved["loader_rng"])
    restore_flat_rng_state(saved)
    return saved


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
    """
    Return float32 logits for every batch of a loader.

    Parameters
    ----------
    model : nn.Module
        Classifier, switched to evaluation mode.
    loader : DataLoader
        Loader yielding ``(signal, target)`` batches.
    device : str
        Torch device of ``model``.

    Returns
    -------
    np.ndarray
        Concatenated logits.
    """
    model.eval()
    return np.concatenate([model(signal.to(device)).float().cpu().numpy() for signal, _ in loader])


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler.LambdaLR, device: str, effective_batch_size: int,
                grad_clip: float = GRADIENT_CLIP) -> tuple[float, int]:
    """
    Train one epoch with microbatch gradient accumulation.

    Each effective batch's loss is the mean over its records, including a
    shorter final batch.

    Parameters
    ----------
    model : nn.Module
        Classifier to train in place.
    loader : DataLoader
        Training loader of microbatches.
    optimizer : torch.optim.Optimizer
        Optimizer stepped once per effective batch.
    scheduler : torch.optim.lr_scheduler.LambdaLR
        Scheduler stepped after every optimizer update.
    device : str
        Torch device of ``model``.
    effective_batch_size : int
        Records per optimizer update.
    grad_clip : float
        Maximum total gradient norm.

    Returns
    -------
    tuple[float, int]
        Mean training loss per record and number of optimizer updates.

    Raises
    ------
    RuntimeError
        If the loss or gradient norm is nonfinite, or the update count
        differs from the planned budget.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, seen, steps = 0.0, 0, 0
    for signal, target in loader:
        signal, target = signal.to(device), target.to(device)
        group_start = (seen // effective_batch_size) * effective_batch_size
        group_size = min(effective_batch_size, len(loader.dataset) - group_start)
        logits = model(signal)
        summed = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="sum")
        if not torch.isfinite(summed):
            raise RuntimeError("Nonfinite xECG fine-tuning loss")
        (summed / group_size).backward()
        total_loss += float(summed.detach())
        seen += len(target)
        if seen - group_start == group_size:
            norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if not torch.isfinite(norm):
                raise RuntimeError("Nonfinite xECG gradient norm")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            steps += 1
    if seen != len(loader.dataset) or steps != math.ceil(seen / effective_batch_size):
        raise RuntimeError("xECG optimizer update count differs from planned budget")
    return total_loss / seen, steps


def fit(model: nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LambdaLR,
        train_loader: DataLoader, dev_loader: DataLoader, development_y: np.ndarray,
        output_dir: Path, fingerprint: dict[str, Any], generator: torch.Generator, device: str,
        epochs: int, patience: int, effective_batch_size: int, resume: bool = False) -> dict[str, Any]:
    """
    Train with development-AUROC early stopping and epoch-boundary resume.

    The best development epoch is loaded into ``model`` on return; the first
    epoch wins ties.

    Parameters
    ----------
    model : nn.Module
        Classifier to train in place.
    optimizer : torch.optim.Optimizer
        Optimizer of ``model``.
    scheduler : torch.optim.lr_scheduler.LambdaLR
        Per-update scheduler.
    train_loader : DataLoader
        Shuffled training loader driven by ``generator``.
    dev_loader : DataLoader
        Development loader.
    development_y : np.ndarray
        Development targets in loader order.
    output_dir : Path
        Run directory for ``history.json`` and ``resume.pt``.
    fingerprint : dict[str, Any]
        Inputs recorded in the resume checkpoint.
    generator : torch.Generator
        Training shuffle generator.
    device : str
        Torch device of ``model``.
    epochs : int
        Epoch ceiling.
    patience : int
        Epochs without improvement before stopping.
    effective_batch_size : int
        Records per optimizer update.
    resume : bool
        Continue from ``resume.pt``.

    Returns
    -------
    dict[str, Any]
        History, best epoch, best development AUROC, and elapsed seconds.

    Raises
    ------
    FileNotFoundError
        If ``resume`` is set without a resume checkpoint.
    FileExistsError
        If a resume checkpoint exists but ``resume`` is not set.
    RuntimeError
        If development logits are nonfinite or no epoch was selected.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_path = output_dir / "resume.pt"
    history, best_epoch, best_auc, best_state = [], 0, -1.0, None
    started = time.monotonic()
    if resume:
        if not resume_path.is_file():
            raise FileNotFoundError(f"No complete xECG epoch to resume: {resume_path}")
        saved = load_resume(resume_path, fingerprint, model, optimizer, scheduler, generator)
        history, best_epoch, best_auc, best_state = (saved["history"], saved["best_epoch"],
                                                      saved["best_auc"], saved["best_model"])
        started -= saved["elapsed_seconds"]
    elif resume_path.exists():
        raise FileExistsError(f"Incomplete xECG run exists; use --resume: {resume_path}")
    for epoch in range(len(history), epochs):
        if epoch - best_epoch >= patience:
            break
        loss, updates = train_epoch(model, train_loader, optimizer, scheduler, device, effective_batch_size)
        dev_logits = predict(model, dev_loader, device)
        if not np.isfinite(dev_logits).all():
            raise RuntimeError("Nonfinite development logits")
        auc = float(roc_auc_score(development_y, dev_logits))
        if auc > best_auc:
            best_auc, best_epoch, best_state = auc, epoch + 1, cpu_state(model)
        row = {"epoch": epoch + 1, "loss": loss, "development_auroc": auc,
               "optimizer_updates": updates, "seconds": time.monotonic() - started}
        history.append(row)
        write_json_atomic(output_dir / "history.json", history)
        save_resume(resume_path, fingerprint, model, optimizer, scheduler, generator,
                    best_state, best_epoch, best_auc, history, row["seconds"])
        print(json.dumps({"stage": "xecg_train", **row}), flush=True)
        if epoch + 1 - best_epoch >= patience:
            break
    if best_state is None:
        raise RuntimeError("No xECG checkpoint selected")
    model.load_state_dict(best_state, strict=True)
    return {"history": history, "best_epoch": best_epoch, "best_development_auroc": best_auc,
            "elapsed_seconds": time.monotonic() - started}


def completion_hashes(output_dir: Path) -> dict[str, str]:
    """
    Hash the artifacts that a completed fine-tuning run must keep unchanged.

    Parameters
    ----------
    output_dir : Path
        Completed run directory.

    Returns
    -------
    dict[str, str]
        SHA-256 digest keyed by artifact name.
    """
    return artifact_hashes(output_dir, COMPLETION_FILES)


def valid_completion(output_dir: Path, fingerprint: dict[str, Any]) -> dict[str, Any] | None:
    """
    Return the verified completion receipt of a fine-tuning run, if any.

    Parameters
    ----------
    output_dir : Path
        Run directory.
    fingerprint : dict[str, Any]
        Inputs the completed run must have used.

    Returns
    -------
    dict[str, Any] | None
        The receipt, or None when the run has not completed.

    Raises
    ------
    ValueError
        If the completed run used other inputs or an artifact changed.
    """
    return verified_completion(output_dir, fingerprint, COMPLETION_FILES)


def checkpoint_fingerprint(checkpoint_dir: Path) -> dict[str, str]:
    """
    Hash every component of the official xECG release.

    Parameters
    ----------
    checkpoint_dir : Path
        Directory of the downloaded release.

    Returns
    -------
    dict[str, str]
        SHA-256 digest keyed by component name.

    Raises
    ------
    FileNotFoundError
        If a release component is missing.
    """
    for name in CHECKPOINT_FILES:
        if not (checkpoint_dir / name).is_file():
            raise FileNotFoundError(f"Missing official xECG checkpoint component: {checkpoint_dir / name}")
    return {name: sha256_file(checkpoint_dir / name) for name in CHECKPOINT_FILES}


def source_tree_sha256(root: Path) -> str:
    """
    Digest every Python file under a directory, including relative paths.

    Parameters
    ----------
    root : Path
        Source tree to hash.

    Returns
    -------
    str
        Hexadecimal SHA-256 over sorted paths and file digests.

    Raises
    ------
    FileNotFoundError
        If the tree holds no Python files.
    """
    digest = hashlib.sha256()
    files = sorted(root.rglob("*.py"))
    if not files:
        raise FileNotFoundError(f"No Python source files in {root}")
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def resume_for_budget(directory: Path, requested: bool) -> bool:
    """
    Decide whether a budget resumes; a completed first budget may precede an untouched second.

    Parameters
    ----------
    directory : Path
        Budget run directory.
    requested : bool
        Whether ``--resume`` was given.

    Returns
    -------
    bool
        True when the budget continues from ``resume.pt``.

    Raises
    ------
    FileExistsError
        If a resume checkpoint exists without ``--resume``, or the directory
        holds output without an epoch checkpoint.
    """
    resume_path = directory / "resume.pt"
    has_checkpoint = resume_path.is_file()
    if has_checkpoint and not requested:
        raise FileExistsError(f"Incomplete xECG run exists; use --resume: {resume_path}")
    if not has_checkpoint and directory.is_dir() and any(directory.iterdir()):
        raise FileExistsError(f"Partial xECG run lacks an epoch checkpoint: {directory}")
    return has_checkpoint


def cache_fingerprint(cache_dir: Path, raw_dir: Path,
                      full_manifest_dir: Path) -> tuple[np.ndarray, dict[int, int], dict[str, Any]]:
    """
    Open the 100 Hz xECG cache after checking it against its sources.

    Parameters
    ----------
    cache_dir : Path
        Directory with ``views.npy`` and ``metadata.json``.
    raw_dir : Path
        PTB-XL waveform directory the cache must come from.
    full_manifest_dir : Path
        Full-label manifest directory the cache must come from.

    Returns
    -------
    tuple[np.ndarray, dict[int, int], dict[str, Any]]
        Memory-mapped views, cache row per ECG identifier, and cache digests
        with the parsed metadata.

    Raises
    ------
    FileNotFoundError
        If the cache is missing.
    ValueError
        If the cache shape, identities, sources, or preprocessing differ.
    """
    metadata_path = cache_dir / "metadata.json"
    views_path = cache_dir / "views.npy"
    if not all(path.is_file() for path in (metadata_path, views_path)):
        raise FileNotFoundError(f"xECG 500 Hz to 100 Hz cache is missing from {cache_dir}; run --stage prep")
    metadata = json.loads(metadata_path.read_text())
    ids = [int(ecg_id) for ecg_id in metadata["ecg_ids"]]
    views = np.load(views_path, mmap_mode="r")
    if views.shape != (len(ids), *XECG_VIEW_SHAPE) or views.dtype != np.float32:
        raise ValueError("Malformed xECG FFT cache")
    if len(ids) != CACHE_RECORDS or len(set(ids)) != len(ids):
        raise ValueError("xECG FFT cache lacks the exact PTB-XL train/validation/test ECG union")
    if metadata.get("shape") != list(views.shape) or metadata.get("dtype") != "float32":
        raise ValueError("xECG cache metadata does not match waveform array")
    if metadata.get("raw_dir") != str(raw_dir.resolve()):
        raise ValueError("xECG cache was made from another PTB-XL waveform directory")
    requested = {f"{name}.csv": sha256_file(full_manifest_dir / f"{name}.csv") for name in SPLITS}
    if metadata.get("manifest_sha256") != requested:
        raise ValueError("xECG cache was made from another full-label manifest")
    if (metadata.get("preparation_sha256") != sha256_file(ROOT / "scripts/data/prepare_xecg.py")
            or metadata.get("adapter_sha256") != sha256_file(ROOT / "ecg_experiment/xecg.py")):
        raise ValueError("xECG cache preprocessing source differs from current source")
    return views, {ecg_id: index for index, ecg_id in enumerate(ids)}, {
        "metadata_sha256": sha256_file(metadata_path),
        "views_sha256": sha256_file(views_path), "metadata": metadata,
    }


def cache_digests(cache_hashes: dict[str, Any]) -> dict[str, Any]:
    """
    Drop the parsed metadata from cache digests before recording them.

    Parameters
    ----------
    cache_hashes : dict[str, Any]
        Third value returned by ``cache_fingerprint``.

    Returns
    -------
    dict[str, Any]
        The digests only.
    """
    return {key: value for key, value in cache_hashes.items() if key != "metadata"}


def build_classifier(args: argparse.Namespace, train_records: int) -> tuple[
        nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """
    Seed, load the released encoder, and build the classifier and its optimizer.

    The construction order fixes the random head initialization.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed runner arguments.
    train_records : int
        Training records, which set optimizer updates per epoch.

    Returns
    -------
    tuple[nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]
        Classifier, AdamW optimizer, and scheduler.
    """
    from ecg_experiment.xecg import XECGBinaryClassifier, load_xecg

    seed_everything(args.seed)
    backbone = load_xecg(args.checkpoint_dir, backend="vanilla", device=args.device,
                         drop_path_prob=DROP_PATH_PROB)
    model = XECGBinaryClassifier(backbone).to(args.device)
    optimizer = torch.optim.AdamW(layerwise_parameter_groups(model), weight_decay=WEIGHT_DECAY)
    scheduler = make_scheduler(optimizer, math.ceil(train_records / args.effective_batch_size), args.epochs)
    return model, optimizer, scheduler


def _profile_steps(model: nn.Module, optimizer: torch.optim.Optimizer,
                   scheduler: torch.optim.lr_scheduler.LambdaLR, signal: torch.Tensor,
                   target: torch.Tensor, device: str) -> tuple[list[float], torch.Tensor, torch.Tensor]:
    """Time full-batch optimizer steps on one microbatch, checking every value is finite."""
    step_seconds = []
    for _ in range(PROFILE_STEPS):
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        logits = model(signal)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
        if logits.shape != target.shape or not torch.isfinite(loss):
            raise RuntimeError("xECG profile produced invalid logits/loss")
        loss.backward()
        if not all(param.grad is None or torch.isfinite(param.grad).all() for param in model.parameters()):
            raise RuntimeError("xECG profile produced nonfinite gradients")
        norm = nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        if not torch.isfinite(norm):
            raise RuntimeError("xECG profile produced nonfinite gradient norm")
        optimizer.step()
        scheduler.step()
        if not all(torch.isfinite(param).all() for param in model.parameters()):
            raise RuntimeError("xECG profile produced nonfinite parameters after optimizer step")
        if device == "cuda":
            torch.cuda.synchronize()
        step_seconds.append(time.monotonic() - started)
    return step_seconds, logits, loss


def run_profile(args: argparse.Namespace, rows: dict[str, Rows], development: Rows, views: np.ndarray,
                index: dict[int, int], checkpoint_hashes: dict[str, str],
                cache_hashes: dict[str, Any]) -> None:
    """
    Profile two optimizer steps and one development batch, then save the receipt.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed runner arguments.
    rows : dict[str, list[dict[str, str]]]
        Split rows of the first selected budget.
    development : list[dict[str, str]]
        Development rows.
    views : np.ndarray
        xECG waveform cache.
    index : dict[int, int]
        Cache row per ECG identifier.
    checkpoint_hashes : dict[str, str]
        Release component digests.
    cache_hashes : dict[str, Any]
        Cache digests and metadata.

    Raises
    ------
    RuntimeError
        If any profiled value is nonfinite or malformed.
    """
    model, optimizer, scheduler = build_classifier(args, len(rows["labeled_train"]))
    train_loader = DataLoader(CachedECGs(views, index, rows["labeled_train"]),
                              batch_size=args.microbatch_size, shuffle=False, num_workers=0)
    dev_loader = DataLoader(CachedECGs(views, index, development),
                            batch_size=args.microbatch_size, shuffle=False, num_workers=0)
    signal, target = next(iter(train_loader))
    signal, target = signal.to(args.device), target.to(args.device)
    model.train()
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    step_seconds, logits, loss = _profile_steps(model, optimizer, scheduler, signal, target, args.device)
    if args.device == "cuda":
        torch.cuda.synchronize()
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    with torch.inference_mode():
        dev_signal, _ = next(iter(dev_loader))
        dev_logits = model(dev_signal.to(args.device))
    if not torch.isfinite(dev_logits).all():
        raise RuntimeError("xECG profile produced nonfinite development logits")
    result = {
        "device": args.device,
        "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "microbatch_size": args.microbatch_size, "effective_batch_size": args.effective_batch_size,
        "first_optimizer_step_seconds": step_seconds[0],
        "steady_state_optimizer_step_seconds": step_seconds[1],
        "optimizer_state_bytes": optimizer_state_bytes(optimizer),
        "peak_cuda_memory_bytes": peak_gpu_bytes(args.device),
        "logit_shape": list(logits.shape), "loss": float(loss.detach()),
        "checkpoint_sha256": checkpoint_hashes, "cache_sha256": cache_digests(cache_hashes),
    }
    write_json_atomic(args.output_dir / f"profile_{args.device}.json", result)
    print(json.dumps({"stage": "xecg_profile", **result}), flush=True)


def budget_fingerprint(args: argparse.Namespace, budget: str, hashes: dict[str, str],
                       checkpoint_hashes: dict[str, str], cache_hashes: dict[str, Any]) -> dict[str, Any]:
    """
    Describe every input and source file that determines one budget's run.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed runner arguments.
    budget : str
        Label budget.
    hashes : dict[str, str]
        Manifest digests.
    checkpoint_hashes : dict[str, str]
        Release component digests.
    cache_hashes : dict[str, Any]
        Cache digests and metadata.

    Returns
    -------
    dict[str, Any]
        Fingerprint that completion and resume receipts must match.
    """
    return {
        "budget": budget, "seed": args.seed, "epochs": args.epochs, "patience": args.patience,
        "microbatch_size": args.microbatch_size, "effective_batch_size": args.effective_batch_size,
        "bootstrap": args.bootstrap, "device": args.device, "threads": args.threads,
        "raw_dir": str(args.raw_dir.resolve()), "cache_dir": str(args.cache_dir.resolve()),
        "checkpoint_dir": str(args.checkpoint_dir.resolve()), "checkpoint_sha256": checkpoint_hashes,
        "cache_sha256": cache_digests(cache_hashes),
        "manifest_sha256": hashes,
        "runner_source_sha256": sha256_file(Path(__file__)),
        "adapter_source_sha256": sha256_file(ROOT / "ecg_experiment/xecg.py"),
        "xlstm_python_source_sha256": source_tree_sha256(ROOT / "third_party/xecg-deps/xlstm"),
        "official_ptbxl_config_sha256": sha256_file(
            ROOT / "third_party/bench-xecg/configs/ptb-xl/xlstm_ft.yaml"),
        "official_ptbxl_defaults_sha256": sha256_file(
            ROOT / "third_party/bench-xecg/config_defaults/train_ptb_xl_defaults.yaml"),
        "evaluation_source_sha256": sha256_file(ROOT / "ecg_experiment/evaluation.py"),
        "run_source_sha256": sha256_file(ROOT / "ecg_experiment/run.py"),
        "reproducibility_source_sha256": sha256_file(ROOT / "ecg_experiment/reproducibility.py"),
        "training_source_sha256": sha256_file(ROOT / "ecg_experiment/training.py"),
        "receipts_source_sha256": sha256_file(ROOT / "ecg_experiment/receipts.py"),
        "pretrained_checkpoint_source_commit": git_head(ROOT / "third_party/bench-xecg"),
    }


def budget_config(fingerprint: dict[str, Any], rows: dict[str, Rows], development: Rows,
                  calibration: Rows) -> dict[str, Any]:
    """
    Describe the fine-tuning protocol recorded in ``config.json``.

    Parameters
    ----------
    fingerprint : dict[str, Any]
        Run fingerprint.
    rows : dict[str, list[dict[str, str]]]
        Split rows.
    development : list[dict[str, str]]
        Development rows.
    calibration : list[dict[str, str]]
        Calibration rows.

    Returns
    -------
    dict[str, Any]
        Protocol description.
    """
    return {"fingerprint": fingerprint, "model": "xECG", "task": "PTB-XL diagnostic abnormality proxy",
            "preprocessing": "Official 500 Hz physical-mV ECG FFT-resampled to 100 Hz; "
                             "full 10 seconds, canonical 12 leads; no normalization",
            "augmentation": "none", "head": "Identity + Linear(1024,1)",
            "optimizer": "AdamW", "core_lr": CORE_LR, "head_lr": HEAD_LR,
            "weight_decay": WEIGHT_DECAY, "layerwise_lr_decay": LAYERWISE_LR_DECAY,
            "gradient_clip": GRADIENT_CLIP,
            "scheduler": "one-epoch linear warmup then half-cosine per optimizer step",
            "drop_path_prob": DROP_PATH_PROB, "precision": "float32",
            "checkpoint_selection": "best development AUROC; first epoch wins ties; patience 8",
            "records": {"train": len(rows["labeled_train"]), "development": len(development),
                        "calibration": len(calibration), "test": len(rows["test"])},
            "pretraining_exposure_caveat": ("Released xECG sources do not report PTB-XL or MIMIC as "
                                            "pretraining sources. Cross-source overlap is not independently "
                                            "audited; our PTB-XL test cohort has already been used in "
                                            "earlier experiments."),
            "protocol_deviations": "Binary proxy task; 40-epoch ceiling; effective batch 64 by float32 "
                                   "microbatch accumulation; no augmentation or class weighting."}


def train_budget(args: argparse.Namespace, budget: str, rows: dict[str, Rows], development: Rows,
                 calibration: Rows, hashes: dict[str, str], views: np.ndarray, index: dict[int, int],
                 checkpoint_hashes: dict[str, str], cache_hashes: dict[str, Any]) -> None:
    """
    Fine-tune, select, and evaluate one label budget, reusing a verified completed run.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed runner arguments.
    budget : str
        Label budget.
    rows : dict[str, list[dict[str, str]]]
        Split rows.
    development : list[dict[str, str]]
        Development rows.
    calibration : list[dict[str, str]]
        Calibration rows.
    hashes : dict[str, str]
        Manifest digests.
    views : np.ndarray
        xECG waveform cache.
    index : dict[int, int]
        Cache row per ECG identifier.
    checkpoint_hashes : dict[str, str]
        Release component digests.
    cache_hashes : dict[str, Any]
        Cache digests and metadata.

    Raises
    ------
    ValueError
        If an existing run was made with other inputs.
    """
    directory = args.output_dir / f"xecg_{budget}_seed{args.seed}"
    fingerprint = budget_fingerprint(args, budget, hashes, checkpoint_hashes, cache_hashes)
    if valid_completion(directory, fingerprint):
        print(json.dumps({"stage": "xecg_reuse", "budget": budget, "directory": str(directory)}), flush=True)
        return
    resume_this_budget = resume_for_budget(directory, args.resume)
    if args.resume and (directory / "config.json").is_file():
        existing = json.loads((directory / "config.json").read_text())
        if existing.get("fingerprint") != fingerprint:
            raise ValueError(f"xECG resume configuration differs: {directory}")
    model, optimizer, scheduler = build_classifier(args, len(rows["labeled_train"]))
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(CachedECGs(views, index, rows["labeled_train"]),
                              batch_size=args.microbatch_size, shuffle=True,
                              num_workers=0, generator=generator)
    dev_loader = DataLoader(CachedECGs(views, index, development),
                            batch_size=args.microbatch_size, num_workers=0)
    development_y = np.array([int(row["target"]) for row in development])
    directory.mkdir(parents=True, exist_ok=True)
    config = budget_config(fingerprint, rows, development, calibration)
    write_json_atomic(directory / "config.json", config)
    result = fit(model, optimizer, scheduler, train_loader, dev_loader, development_y,
                 directory, fingerprint, generator, args.device, args.epochs, args.patience,
                 args.effective_batch_size, resume=resume_this_budget)
    write_torch_atomic(directory / "model.pt", {"model": cpu_state(model), "fingerprint": fingerprint,
                                                "best_epoch": result["best_epoch"]})
    # Calibration and test labels enter only after the best development epoch is fixed.
    calibration_loader = DataLoader(CachedECGs(views, index, calibration),
                                    batch_size=args.microbatch_size, num_workers=0)
    test_loader = DataLoader(CachedECGs(views, index, rows["test"]),
                             batch_size=args.microbatch_size, num_workers=0)
    evaluate_predictions("xecg_finetuned_" + budget,
                         predict(model, calibration_loader, args.device),
                         predict(model, test_loader, args.device),
                         calibration, rows["test"], directory, args.seed, args.bootstrap)
    config.update({"best_epoch": result["best_epoch"],
                   "best_development_auroc": result["best_development_auroc"],
                   "elapsed_seconds": result["elapsed_seconds"],
                   "peak_cuda_memory_bytes": peak_gpu_bytes(args.device)})
    write_json_atomic(directory / "config.json", config)
    write_json_atomic(directory / "complete.json", {"fingerprint": fingerprint,
                                                    "sha256": completion_hashes(directory)})
    (directory / "resume.pt").unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """
    Parse and validate runner arguments.

    Parameters
    ----------
    argv : Sequence[str] | None
        Command-line arguments; None reads ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Validated arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("check", "prep", "profile", "train", "all"), default="all")
    parser.add_argument("--budget", choices=("all", "full", "ten_percent"), default="all")
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--microbatch-size", type=int, default=16)
    parser.add_argument("--effective-batch-size", type=int, default=64)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.seed != 42:
        parser.error("This fixed patient and label protocol requires --seed 42")
    if min(args.epochs, args.patience, args.microbatch_size, args.effective_batch_size,
           args.bootstrap, args.threads) < 1:
        parser.error("Epochs, patience, batch sizes, bootstrap, and threads must be positive")
    if args.effective_batch_size % args.microbatch_size:
        parser.error("Effective batch size must be divisible by microbatch size")
    return args


def check_cache_coverage(selected_manifests: dict[str, Manifests], index: dict[int, int]) -> None:
    """
    Require every selected record to be present in the waveform cache.

    Parameters
    ----------
    selected_manifests : dict[str, tuple]
        Output of ``load_manifests`` per selected budget.
    index : dict[int, int]
        Cache row per ECG identifier.

    Raises
    ------
    ValueError
        If a record is missing from the cache.
    """
    for rows, development, calibration, _ in selected_manifests.values():
        for row in rows["labeled_train"] + development + calibration + rows["test"]:
            if int(row["ecg_id"]) not in index:
                raise ValueError(f"xECG FFT cache missing ECG {row['ecg_id']}")


def run_gpu_stages(args: argparse.Namespace, selected: tuple[str, ...],
                   selected_manifests: dict[str, Manifests], views: np.ndarray, index: dict[int, int],
                   checkpoint_hashes: dict[str, str], cache_hashes: dict[str, Any]) -> None:
    """
    Run the profile and training stages while holding the shared GPU lock.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed runner arguments.
    selected : tuple[str, ...]
        Label budgets in run order.
    selected_manifests : dict[str, tuple]
        Output of ``load_manifests`` per selected budget.
    views : np.ndarray
        xECG waveform cache.
    index : dict[int, int]
        Cache row per ECG identifier.
    checkpoint_hashes : dict[str, str]
        Release component digests.
    cache_hashes : dict[str, Any]
        Cache digests and metadata.
    """
    with gpu_lock(args.device, blocking=False):
        if args.stage in ("profile", "all"):
            rows, development, _, _ = selected_manifests[selected[0]]
            run_profile(args, rows, development, views, index, checkpoint_hashes, cache_hashes)
        if args.stage in ("train", "all"):
            for budget in selected:
                train_budget(args, budget, *selected_manifests[budget], views, index,
                             checkpoint_hashes, cache_hashes)


def main(argv: Sequence[str] | None = None) -> None:
    """
    Check inputs, prepare the cache, and profile or fine-tune the selected budgets.

    Parameters
    ----------
    argv : Sequence[str] | None
        Command-line arguments; None reads ``sys.argv``.
    """
    args = parse_args(argv)
    if args.stage in TRAINING_STAGES:
        require_cuda(args.device)
    torch.set_num_threads(args.threads)
    selected = budgets(args.budget)
    selected_manifests = {budget: load_manifests(args.manifest_root, budget) for budget in selected}
    checkpoint_hashes = checkpoint_fingerprint(args.checkpoint_dir)
    print(json.dumps({"stage": "xecg_check", "budgets": list(selected),
                      "checkpoint_sha256": checkpoint_hashes,
                      "records": {budget: len(selected_manifests[budget][0]["labeled_train"])
                                  for budget in selected}}), flush=True)
    if args.stage == "check":
        return
    if args.stage in ("prep", "all"):
        from scripts.data.prepare_xecg import cache_xecg_views
        cache_xecg_views(args.raw_dir, manifest_dir(args.manifest_root, "full"), args.cache_dir)
    views, index, cache_hashes = cache_fingerprint(
        args.cache_dir, args.raw_dir, manifest_dir(args.manifest_root, "full"))
    check_cache_coverage(selected_manifests, index)
    if args.stage == "prep":
        return
    run_gpu_stages(args, selected, selected_manifests, views, index, checkpoint_hashes, cache_hashes)


if __name__ == "__main__":
    main()
