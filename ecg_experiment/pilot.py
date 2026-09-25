"""Frozen PTB-XL partitions, fixed label schedules, and resumable development pilots.

Experiments 015 and 017 share these pieces: the same frozen manifests, the same
exposed-label batch schedule, mid-epoch checkpoints that survive SIGTERM or a
wall-clock deadline, and a development-only patient-fold screen.
"""

from __future__ import annotations

import json
import signal
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

from .cpc import LEADS, SIGNAL_SAMPLES
from .cpc_pool import Pool
from .evaluation import TARGET_SENSITIVITY, partition_validation, select_threshold
from .files import read_csv, write_json_atomic, write_torch_atomic
from .reproducibility import capture_rng_state, cpu_state, restore_rng_state

FULL_LABELS = 15360
LIMITED_LABELS = 1518
PARTITION_COUNTS = (FULL_LABELS, LIMITED_LABELS, 1306, 564, 1896)
BUDGETS = ("1", "0.1")
BATCH = 128
SAVE_EVERY = 20
# Exit status the coordinators treat as "interrupted, resume later".
INTERRUPTED_EXIT = 75
FOLDS = 5
LOGIT_CLIP = 80

_STOP_REQUESTED = threading.Event()


def install_stop_handler() -> None:
    """Checkpoint and exit at the next update when SIGTERM arrives."""
    signal.signal(signal.SIGTERM, lambda _signum, _frame: _STOP_REQUESTED.set())


def file_identity(path: Path) -> dict[str, int]:
    """
    File-system identity recorded by the Experiment 015 pool verifier.

    A receipt's content digests are reused only while every field is unchanged.

    Parameters
    ----------
    path : Path
        File to describe.

    Returns
    -------
    dict[str, int]
        Device, inode, size and modification/change times in nanoseconds.
    """
    stat = Path(path).stat()
    return {"device": stat.st_dev, "inode": stat.st_ino, "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}


@dataclass
class Partitions:
    """
    The frozen PTB-XL training, development, calibration and test rows.

    Attributes
    ----------
    full : list[dict[str, str]]
        All 15,360 labeled training rows.
    limited : list[dict[str, str]]
        The fixed 1,518-label subset of ``full``.
    development : list[dict[str, str]]
        Development patients from the validation manifest.
    calibration : list[dict[str, str]]
        Calibration patients from the validation manifest.
    test : list[dict[str, str]]
        Test rows.
    """

    full: list[dict[str, str]]
    limited: list[dict[str, str]]
    development: list[dict[str, str]]
    calibration: list[dict[str, str]]
    test: list[dict[str, str]]


def load_partitions(pool: Pool, manifest_dir: Path) -> Partitions:
    """
    Read both label budgets and check them against the frozen protocol and pool.

    Parameters
    ----------
    pool : Pool
        CPC waveform cache holding every PTB-XL record.
    manifest_dir : Path
        Directory holding ``seed42_fraction1`` and ``seed42_fraction0.1``.

    Returns
    -------
    Partitions
        Rows of every partition.

    Raises
    ------
    ValueError
        If counts, the limited subset, budget manifests, cache rows or
        patient partitions differ from the frozen protocol.
    """
    full = read_csv(manifest_dir / "seed42_fraction1/labeled_train.csv")
    limited = read_csv(manifest_dir / "seed42_fraction0.1/labeled_train.csv")
    validation = read_csv(manifest_dir / "seed42_fraction1/validation.csv")
    development, calibration = partition_validation(validation)
    test = read_csv(manifest_dir / "seed42_fraction1/test.csv")
    if tuple(map(len, (full, limited, development, calibration, test))) != PARTITION_COUNTS:
        raise ValueError("Frozen partition counts changed")
    by_id = {row["ecg_id"]: row for row in full}
    if len(by_id) != len(full) or any(
            row["ecg_id"] not in by_id
            or (row["patient_id"], row["target"]) != (by_id[row["ecg_id"]]["patient_id"],
                                                      by_id[row["ecg_id"]]["target"])
            for row in limited):
        raise ValueError("Limited budget differs from frozen full training set")
    for name, expected in (("validation", validation), ("test", test)):
        if read_csv(manifest_dir / f"seed42_fraction0.1/{name}.csv") != expected:
            raise ValueError(f"Budget-specific {name} manifest changed")
    for rows, split in ((full, "train"), (development, "validation"),
                        (calibration, "validation"), (test, "test")):
        for row in rows:
            cached = pool.rows[pool.index[row["ecg_id"]]]
            if (cached["source"], cached["split"], cached["patient_id"]) != (
                    "ptbxl", split, row["patient_id"]):
                raise ValueError(f"Manifest/cache mismatch: {row['ecg_id']}")
    patients = [{row["patient_id"] for row in rows} for rows in (full, development, calibration, test)]
    if any(patients[i] & patients[j] for i in range(4) for j in range(i + 1, 4)):
        raise ValueError("Patient partitions overlap")
    return Partitions(full, limited, development, calibration, test)


def check_final_ssl(ssl: Path) -> dict[str, Any]:
    """
    Check that a CPC encoder is the final state of the ordinary Experiment 004 run.

    Parameters
    ----------
    ssl : Path
        ``encoder.pt`` beside its ``config.json`` and ``epoch_state.pt``.

    Returns
    -------
    dict[str, Any]
        The SSL run's ``config.json``.

    Raises
    ------
    ValueError
        If the encoder is not the completed 20-epoch ordinary CPC state.
    """
    config = json.loads((ssl.parent / "config.json").read_text())
    saved = torch.load(ssl, map_location="cpu", weights_only=True)
    final = torch.load(ssl.parent / "epoch_state.pt", map_location="cpu", weights_only=False)
    if (saved.get("variant") != "cpc" or saved.get("epochs") != 20
            or saved.get("fingerprint") != config["fingerprint"]
            or final["epoch"] != 20 or final["fingerprint"] != config["fingerprint"]):
        raise ValueError("Not the final ordinary Experiment 004 CPC encoder")
    if any(not torch.equal(value, final["model"][f"encoder.{key}"])
           for key, value in saved["encoder"].items()):
        raise ValueError("CPC checkpoint differs from final SSL state")
    return config


def check_waveform_sample(pool: Pool, row: dict[str, str]) -> None:
    """
    Read one training waveform to prove the cache is readable and finite.

    Parameters
    ----------
    pool : Pool
        CPC waveform cache.
    row : dict[str, str]
        Manifest row to read.

    Raises
    ------
    ValueError
        If the waveform has the wrong shape or nonfinite samples.
    """
    sample = np.array(pool.signals[pool.indices([row])[0]], copy=True)
    if sample.shape != (LEADS, SIGNAL_SAMPLES) or not np.isfinite(sample).all():
        raise ValueError("Invalid waveform sample")


def exposed_labels(full: list[dict[str, str]], limited: list[dict[str, str]],
                   budget: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Mark which training rows reveal their label under a budget.

    Hidden rows still pass through the model; their targets are zeroed and
    masked out of the loss.

    Parameters
    ----------
    full : list[dict[str, str]]
        All labeled training rows.
    limited : list[dict[str, str]]
        The limited-budget subset.
    budget : str
        ``"1"`` exposes every label; ``"0.1"`` exposes ``limited`` only.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Boolean exposure mask and float32 targets aligned to ``full``.

    Raises
    ------
    ValueError
        If the number of exposed labels differs from the frozen budget.
    """
    limited_ids = {row["ecg_id"] for row in limited}
    exposed = np.asarray([budget == "1" or row["ecg_id"] in limited_ids for row in full], dtype=bool)
    targets = np.asarray([float(row["target"]) if mask else 0.0
                          for row, mask in zip(full, exposed, strict=True)], dtype=np.float32)
    if int(exposed.sum()) != (FULL_LABELS if budget == "1" else LIMITED_LABELS):
        raise ValueError("Exposed label count changed")
    return exposed, targets


def fixed_batches(size: int, exposed: np.ndarray, epoch: int, seed: int,
                  batch_size: int = BATCH) -> list[list[int]]:
    """
    Seeded batches that use every row once and spread exposed labels across all of them.

    Parameters
    ----------
    size : int
        Number of training rows.
    exposed : np.ndarray
        Boolean exposure mask of length ``size``.
    epoch : int
        Zero-based epoch; each epoch has its own order.
    seed : int
        Experiment seed.
    batch_size : int
        Rows per batch; the last batch may be smaller.

    Returns
    -------
    list[list[int]]
        Row indices of each batch.

    Raises
    ------
    ValueError
        If there are fewer exposed labels than batches.
    RuntimeError
        If the schedule loses or duplicates rows or leaves a batch unlabeled.
    """
    rng = np.random.default_rng(seed + 1009 * epoch)
    count = (size + batch_size - 1) // batch_size
    labeled = rng.permutation(np.flatnonzero(exposed))
    hidden = rng.permutation(np.flatnonzero(~exposed))
    if len(labeled) < count:
        raise ValueError("Too few exposed labels for one labeled row per batch")
    batches = []
    offset = 0
    for index, labels in enumerate(np.array_split(labeled, count)):
        capacity = min(batch_size, size - index * batch_size) - len(labels)
        batches.append(rng.permutation(np.concatenate((labels, hidden[offset:offset + capacity]))).tolist())
        offset += capacity
    flat = [row for batch in batches for row in batch]
    if (len(flat) != size or len(set(flat)) != size or set(flat) != set(range(size))
            or not all(exposed[batch].any() for batch in batches)):
        raise RuntimeError("Training schedule lost rows or labels")
    return batches


def normalized_batch(cache: Any, indices: Sequence[int], mean: np.ndarray, std: np.ndarray,
                     device: str) -> torch.Tensor:
    """
    Normalize selected cached waveforms and move them to a device.

    Parameters
    ----------
    cache : Any
        Object whose ``signals`` array is indexed by row.
    indices : Sequence[int]
        Rows to read.
    mean : np.ndarray
        Per-lead mean shaped ``[12, 1]``.
    std : np.ndarray
        Per-lead standard deviation shaped ``[12, 1]``.
    device : str
        Target device.

    Returns
    -------
    torch.Tensor
        Normalized ``[batch, 12, 2500]`` waveforms.
    """
    values = np.array(cache.signals[indices], copy=True)
    values -= mean
    values /= std
    return torch.from_numpy(values).to(device, non_blocking=True)


def development_batches(cache: Any, start: int, count: int, mean: np.ndarray, std: np.ndarray,
                        device: str) -> Iterator[torch.Tensor]:
    """
    Yield normalized development waveforms in cache order.

    Parameters
    ----------
    cache : Any
        Cache whose rows ``start`` to ``start + count`` hold development records.
    start : int
        First development row.
    count : int
        Number of development rows.
    mean : np.ndarray
        Per-lead mean shaped ``[12, 1]``.
    std : np.ndarray
        Per-lead standard deviation shaped ``[12, 1]``.
    device : str
        Target device.

    Yields
    ------
    torch.Tensor
        Batches of at most ``BATCH`` records.
    """
    for first in range(start, start + count, BATCH):
        stop = min(first + BATCH, start + count)
        yield normalized_batch(cache, list(range(first, stop)), mean, std, device)


def clipped_sigmoid(logits: np.ndarray) -> np.ndarray:
    """
    Convert logits to probabilities after clipping them to avoid overflow.

    Parameters
    ----------
    logits : np.ndarray
        Model logits.

    Returns
    -------
    np.ndarray
        Probabilities.
    """
    return 1 / (1 + np.exp(-np.clip(logits, -LOGIT_CLIP, LOGIT_CLIP)))


def fold_operating_point(labels: np.ndarray, groups: np.ndarray, probabilities: np.ndarray,
                         seed: int) -> dict[str, Any]:
    """
    Cross-fit the 95%-sensitivity threshold over patient folds of development data.

    Each fold's threshold is chosen on the other folds and applied to the
    held-out fold; confusion counts are pooled over folds.

    Parameters
    ----------
    labels : np.ndarray
        Binary development labels.
    groups : np.ndarray
        Patient identifiers; a patient never spans folds.
    probabilities : np.ndarray
        Predicted probabilities.
    seed : int
        Fold shuffling seed.

    Returns
    -------
    dict[str, Any]
        Pooled ``fold_sensitivity``, ``fold_specificity`` and ``fold_confusion``.
    """
    splitter = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    counts = {"tp": 0, "fn": 0, "tn": 0, "fp": 0}
    for train, held in splitter.split(probabilities, labels, groups):
        threshold = select_threshold(labels[train], probabilities[train], TARGET_SENSITIVITY)
        predicted = probabilities[held] >= threshold
        counts["tp"] += int(np.sum((labels[held] == 1) & predicted))
        counts["fn"] += int(np.sum((labels[held] == 1) & ~predicted))
        counts["tn"] += int(np.sum((labels[held] == 0) & ~predicted))
        counts["fp"] += int(np.sum((labels[held] == 0) & predicted))
    return {"fold_sensitivity": counts["tp"] / (counts["tp"] + counts["fn"]),
            "fold_specificity": counts["tn"] / (counts["tn"] + counts["fp"]),
            "fold_confusion": counts}


def labels_and_patients(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract binary targets and patient identifiers from manifest rows.

    Parameters
    ----------
    rows : list[dict[str, str]]
        Rows with ``target`` and ``patient_id``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Integer labels and patient identifiers.
    """
    return np.asarray([int(row["target"]) for row in rows]), np.asarray([row["patient_id"] for row in rows])


@dataclass
class Progress:
    """
    Resumable position inside a pilot arm.

    Attributes
    ----------
    epoch : int
        Completed epochs.
    batch : int
        Completed batches of the current epoch.
    history : list[dict[str, Any]]
        One record per completed epoch.
    best : dict[str, Any]
        Best development AUROC so far and its epoch.
    totals : dict[str, Any]
        Running sums for the current epoch.
    elapsed_seconds : float
        Training wall time before this invocation.
    """

    epoch: int
    batch: int
    history: list[dict[str, Any]]
    best: dict[str, Any]
    totals: dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


def save_state(directory: Path, fingerprint: str, model: nn.Module, optimizer: torch.optim.Optimizer,
               progress: Progress, elapsed_seconds: float) -> None:
    """
    Atomically save model, optimizer, random states and progress.

    Parameters
    ----------
    directory : Path
        Arm directory; ``resume.pt`` and ``history.json`` are written there.
    fingerprint : str
        Arm identity the checkpoint belongs to.
    model : nn.Module
        Model to save.
    optimizer : torch.optim.Optimizer
        Optimizer to save.
    progress : Progress
        Current position; its ``elapsed_seconds`` is replaced by the argument.
    elapsed_seconds : float
        Total training wall time up to this checkpoint.
    """
    write_torch_atomic(directory / "resume.pt", {
        "fingerprint": fingerprint, "model": cpu_state(model), "optimizer": optimizer.state_dict(),
        "rng": capture_rng_state(), "epoch": progress.epoch, "batch": progress.batch,
        "history": progress.history, "best": progress.best, "totals": progress.totals,
        "elapsed_seconds": elapsed_seconds})
    write_json_atomic(directory / "history.json", progress.history)


def load_state(directory: Path, fingerprint: str, model: nn.Module, optimizer: torch.optim.Optimizer,
               initial_best: dict[str, Any]) -> Progress:
    """
    Restore a saved arm in place, or report a fresh start.

    Parameters
    ----------
    directory : Path
        Arm directory that may hold ``resume.pt``.
    fingerprint : str
        Identity the checkpoint must match.
    model : nn.Module
        Model restored in place.
    optimizer : torch.optim.Optimizer
        Optimizer restored in place.
    initial_best : dict[str, Any]
        ``best`` value of a fresh start.

    Returns
    -------
    Progress
        Saved progress; the global random states are restored too.

    Raises
    ------
    ValueError
        If history exists without a checkpoint or the fingerprint differs.
    """
    path = directory / "resume.pt"
    if not path.exists():
        if (directory / "history.json").exists():
            raise ValueError("History exists without checkpoint")
        return Progress(0, 0, [], initial_best)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["fingerprint"] != fingerprint:
        raise ValueError("Resume fingerprint mismatch")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    restore_rng_state(state["rng"])
    return Progress(state["epoch"], state["batch"], state["history"], state["best"],
                    state["totals"], state["elapsed_seconds"])


def interrupted(deadline: float | None) -> bool:
    """
    Check for SIGTERM or an expired wall-clock deadline.

    Parameters
    ----------
    deadline : float | None
        ``time.monotonic`` deadline, or ``None`` for no limit.

    Returns
    -------
    bool
        True when the arm must checkpoint and exit.
    """
    return _STOP_REQUESTED.is_set() or (deadline is not None and time.monotonic() >= deadline)


def check_roundtrip(original: dict[str, Any], progress: Progress, probe: nn.Module,
                    optimizer: torch.optim.Optimizer) -> dict[str, int]:
    """
    Compare a reloaded one-epoch profile checkpoint with its saved tensors.

    Parameters
    ----------
    original : dict[str, Any]
        The saved ``resume.pt`` contents.
    progress : Progress
        Progress returned when ``probe`` and ``optimizer`` were loaded.
    probe : nn.Module
        Fresh model loaded from the checkpoint.
    optimizer : torch.optim.Optimizer
        Fresh optimizer loaded from the checkpoint.

    Returns
    -------
    dict[str, int]
        Checked epoch, batch, model tensor count and optimizer slot count.

    Raises
    ------
    ValueError
        If the checkpoint is not at an epoch boundary or any tensor differs.
    """
    if (progress.epoch, progress.batch) != (1, 0):
        raise ValueError("Profile did not complete one epoch")
    loaded_model = probe.state_dict()
    if any(not torch.equal(value, loaded_model[key]) for key, value in original["model"].items()):
        raise ValueError("Profile model roundtrip differs")
    loaded_optimizer = optimizer.state_dict()["state"]
    for key, state in original["optimizer"]["state"].items():
        for name, value in state.items():
            if torch.is_tensor(value) and not torch.equal(value, loaded_optimizer[key][name]):
                raise ValueError("Profile optimizer roundtrip differs")
    return {"epoch": progress.epoch, "batch": progress.batch, "model_tensors": len(original["model"]),
            "optimizer_slots": len(original["optimizer"]["state"])}


def profile_arms(prefix: str, arms: Sequence[str], profile_arm: Callable[[str, Path], dict[str, Any]],
                 ) -> tuple[dict[str, float], dict[str, Any]]:
    """
    Time one full training epoch and a checkpoint roundtrip for each arm.

    Parameters
    ----------
    prefix : str
        Prefix of the discarded temporary output directory.
    arms : Sequence[str]
        Arms to profile in order.
    profile_arm : Callable[[str, Path], dict[str, Any]]
        Trains one epoch of an arm in the given directory and returns its
        roundtrip receipt.

    Returns
    -------
    tuple[dict[str, float], dict[str, Any]]
        Wall seconds and roundtrip receipt of each arm.
    """
    durations, roundtrips = {}, {}
    with tempfile.TemporaryDirectory(prefix=prefix) as temporary:
        for arm in arms:
            started = time.monotonic()
            roundtrips[arm] = profile_arm(arm, Path(temporary) / arm)
            durations[arm] = time.monotonic() - started
    return durations, roundtrips


def run_pilot(arms: Sequence[str], deadline: float, run_arm: Callable[[str, str], Any]) -> None:
    """
    Train every arm at both label budgets, stopping cleanly at the deadline.

    Parameters
    ----------
    arms : Sequence[str]
        Arms to train at each budget.
    deadline : float
        ``time.monotonic`` deadline.
    run_arm : Callable[[str, str], Any]
        Trains one ``(budget, arm)`` pair to completion.

    Raises
    ------
    SystemExit
        With ``INTERRUPTED_EXIT`` when the deadline passes between arms.
    """
    for budget in BUDGETS:
        for arm in arms:
            if time.monotonic() >= deadline:
                raise SystemExit(INTERRUPTED_EXIT)
            run_arm(budget, arm)
