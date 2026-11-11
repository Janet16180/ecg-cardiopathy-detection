"""
Candidate S-R of MLC-02: an xresnet1d101 trained from scratch on 100 Hz waveforms.

The recipe is row S-R of ``docs/mlc02-condition-models-protocol-2026-11-10.md``, the PTB-XL benchmark
recipe of Strodthoff et al. (2021). Each choice is traced to its source in
``docs/mlc02-scratch-network-validation-2026-11-10.md``.

- Input: the cached first 10 s at 500 Hz with missing samples at the stored rail value
  (``mlc02_features.model_input``), resampled to 100 Hz in one polyphase pass over the whole 10 s
  (``resample.resample_full``, factor 1/5), then standardized with one mean and one standard deviation over
  every sample and lead of the training records.
- Model: ``xresnet1d101`` of the vendored benchmark code with ``MODEL_KWARGS``, re-initialized with the
  benchmark's ``weight_init`` (Kaiming-normal convolutions and linear layers, zero biases, batch-norm
  weights 1).
- Training: one random 2.5 s crop per record and epoch (start in ``[0, 749]``, the benchmark's
  ``random.randint(0, timesteps - output_size - 1)``), batches of 128 with the last partial batch dropped,
  binary cross-entropy averaged over the defined label entries (undefined entries masked), AdamW with a
  one-cycle schedule (``ONE_CYCLE``, ``ADAM``), 50 epochs. The epoch is chosen on the validation selection
  metric.
- Evaluation: the mean probability over the four consecutive 2.5 s windows of the 10 s.

Only the train and validation roles can be loaded: ``check_roles`` raises on a test role before any cache
file is opened, and ``read_windows`` checks the role of every record and every block file it opens.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # noqa: N812

from ecg_experiment import ROOT
from ecg_experiment.files import read_json
from ecg_experiment.mlc01_dataset import array_sha256, microvolts
from ecg_experiment.mlc02_data import CONDITIONS, MANIFEST
from ecg_experiment.mlc02_evaluation import PROBABILITY_CLIP
from ecg_experiment.mlc02_features import model_input
from ecg_experiment.mlc02_waveforms import CACHE, SAMPLES
from ecg_experiment.resample import resample_full

OUTPUT = ROOT / "outputs/mlc02_condition_models_20261110"
ROWS = OUTPUT / "features/rows.csv"
INDEX = CACHE / "index_all.csv"
CANDIDATES = OUTPUT / "candidates"
MODEL_ID = "S-R"
CANDIDATE_DIR = CANDIDATES / MODEL_ID
LOFO_DIR = OUTPUT / "lofo" / MODEL_ID
WORK = Path("/var/tmp/janetrivera/mlc02/scratch_network")
VENDORED = ROOT / "third_party/ecg-fm-benchmarking/code"
SEEDS = (20261110, 20261111, 20261112)
ALLOWED_ROLES = ("train", "validation")
CLOSED_ROLES = ("test_same_source", "test_external", "sph_calibration")
FAMILIES = ("ptbxl", "chapman_ningbo", "georgia", "cpsc")
LEADS = 12
DECIMATION = 5
RECORD_SAMPLES = SAMPLES // DECIMATION
CROP = 250
WINDOW_STARTS = (0, 250, 500, 750)
EPOCHS = 50
BATCH = 128
PEAK_LR = 1e-2
WEIGHT_DECAY = 1e-2
ADAM = {"betas": (0.9, 0.99), "eps": 1e-8}
ONE_CYCLE = {
    "pct_start": 0.3,
    "anneal_strategy": "cos",
    "cycle_momentum": True,
    "base_momentum": 0.85,
    "max_momentum": 0.95,
    "div_factor": 25.0,
    "final_div_factor": 1e4,
    "three_phase": False,
}
MODEL_KWARGS = {
    "input_channels": LEADS,
    "num_classes": len(CONDITIONS),
    "kernel_size": 5,
    "ps_head": 0.5,
    "lin_ftrs_head": [128],
}
EVAL_BATCH = 256


def check_roles(roles: Iterable[str]) -> tuple[str, ...]:
    """
    Allow only the train and validation roles.

    Parameters
    ----------
    roles : Iterable[str]
        Manifest roles a caller wants to load.

    Returns
    -------
    tuple[str, ...]
        The distinct roles, sorted.

    Raises
    ------
    PermissionError
        If any role is a test role or the SPH calibration half.
    ValueError
        If no role is given or a role is unknown.
    """
    roles = tuple(sorted(set(roles)))
    closed = [role for role in roles if role in CLOSED_ROLES]
    if closed:
        raise PermissionError(f"S-R loads train and validation records only, not {closed}")
    unknown = [role for role in roles if role not in ALLOWED_ROLES]
    if unknown or not roles:
        raise ValueError(f"unknown or missing roles: {unknown}")
    return roles


def read_records(manifest: Path = MANIFEST, rows: Path = ROWS, index: Path = INDEX) -> pd.DataFrame:
    """
    Identity and cache position of the included v4 records, in manifest order.

    Three tables must agree: the manifest (included records), the feature rows (``included_v4``) and the cache
    index. No label column and no waveform is read.

    Parameters
    ----------
    manifest : Path
        MLC-01 v4 ``manifest.csv``.
    rows : Path
        MLC-02 ``features/rows.csv``.
    index : Path
        Cache ``index_all.csv``.

    Returns
    -------
    pd.DataFrame
        ``record_key``, ``source``, ``family``, ``role``, ``sha256_10s``, ``block`` and ``row``.

    Raises
    ------
    ValueError
        If the tables differ in record order, source, role, hash or cache position.
    """
    identity = pd.read_csv(
        manifest,
        usecols=["record_key", "source", "family", "role", "included", "sha256_10s"],
        dtype=str,
        keep_default_na=False,
    )
    identity = identity[identity["included"] == "True"].drop(columns="included").reset_index(drop=True)
    features = pd.read_csv(
        rows,
        usecols=["record_key", "source", "role", "sha256_10s", "cache_block", "cache_row", "included_v4"],
        dtype={"record_key": str, "included_v4": str},
    )
    features = features[features["included_v4"] == "True"].reset_index(drop=True)
    for column in ("record_key", "source", "role", "sha256_10s"):
        if not features[column].astype(str).equals(identity[column]):
            raise ValueError(f"feature rows and manifest differ in {column}")
    cache = pd.read_csv(
        index, usecols=["record_key", "block", "row", "sha256_10s"], dtype={"record_key": str}
    )
    cache = cache.set_index("record_key").loc[identity["record_key"]].reset_index()
    if not (
        cache["block"].equals(features["cache_block"])
        and cache["row"].equals(features["cache_row"])
        and cache["sha256_10s"].equals(identity["sha256_10s"])
    ):
        raise ValueError("cache index and feature rows differ in block, row or hash")
    return identity.assign(block=cache["block"].to_numpy(), row=cache["row"].to_numpy())


def select_roles(records: pd.DataFrame, roles: Iterable[str]) -> pd.DataFrame:
    """
    Keep the records of the allowed roles, in manifest order.

    Parameters
    ----------
    records : pd.DataFrame
        Output of ``read_records``.
    roles : Iterable[str]
        Roles to keep; checked by ``check_roles``.

    Returns
    -------
    pd.DataFrame
        The selected rows with their original index.
    """
    return records[records["role"].isin(check_roles(roles))]


def block_role(block: str) -> str:
    """
    Role of a cache block from its file stem ``<role>__<source>[__supplement]``.

    Parameters
    ----------
    block : str
        Block name.

    Returns
    -------
    str
        The role part.
    """
    return block.split("__")[0]


def read_windows(records: pd.DataFrame, cache: Path = CACHE) -> np.ndarray:
    """
    Read cached 500 Hz windows of train or validation records and check them against the manifest hash.

    Parameters
    ----------
    records : pd.DataFrame
        Rows of ``read_records``; every role must be allowed.
    cache : Path
        Cache directory.

    Returns
    -------
    np.ndarray
        Float32 ``[records, 12, 5000]``, with the cache's NaN for missing samples.

    Raises
    ------
    PermissionError
        If a record or a block belongs to a test role.
    ValueError
        If a block has another shape or dtype, or a window does not reproduce its ``sha256_10s``.
    """
    check_roles(records["role"])
    check_roles(block_role(block) for block in records["block"])
    result = np.empty((len(records), LEADS, SAMPLES), dtype=np.float32)
    places = np.arange(len(records))
    for block, rows in records.groupby("block", sort=False):
        array = np.load(cache / f"{block}.npy", mmap_mode="r")
        if array.dtype != np.float32 or array.shape[1:] != (LEADS, SAMPLES):
            raise ValueError(f"unexpected cache block {block}: {array.shape} {array.dtype}")
        result[places[records["block"].to_numpy() == block]] = array[rows["row"].to_numpy()]
        del array
    for window, expected, key in zip(result, records["sha256_10s"], records["record_key"], strict=True):
        if array_sha256(microvolts(window.T)) != expected:
            raise ValueError(f"cached window of {key} does not reproduce its manifest hash")
    return result


def to_model_rate(windows: np.ndarray) -> np.ndarray:
    """
    Model-rate waveforms: rail value for missing samples, then 500 Hz to 100 Hz over the whole 10 s.

    Parameters
    ----------
    windows : np.ndarray
        Float32 ``[records, 12, 5000]`` cached windows.

    Returns
    -------
    np.ndarray
        Float32 ``[records, 12, 1000]``.
    """
    result = np.empty((len(windows), LEADS, RECORD_SAMPLES), dtype=np.float32)
    for position, window in enumerate(windows):
        result[position] = resample_full(model_input(window).astype(np.float64), 1, DECIMATION)
    return result


def fit_standardizer(signals: np.ndarray, chunk: int = 2048) -> dict[str, float]:
    """
    One mean and one population standard deviation over every sample and lead of the training records.

    Parameters
    ----------
    signals : np.ndarray
        ``[records, 12, 1000]`` training waveforms at 100 Hz.
    chunk : int
        Records per float64 pass.

    Returns
    -------
    dict[str, float]
        ``mean``, ``std`` and ``values`` (number of samples used).
    """
    values = signals.size
    mean = (
        sum(
            float(signals[start : start + chunk].astype(np.float64).sum())
            for start in range(0, len(signals), chunk)
        )
        / values
    )
    squares = sum(
        float(((signals[start : start + chunk].astype(np.float64) - mean) ** 2).sum())
        for start in range(0, len(signals), chunk)
    )
    return {"mean": mean, "std": math.sqrt(squares / values), "values": int(values)}


def standardize(signals: np.ndarray, stats: dict[str, float]) -> np.ndarray:
    """
    Standardize waveforms with training statistics, in float32.

    Parameters
    ----------
    signals : np.ndarray
        Waveforms at 100 Hz.
    stats : dict[str, float]
        Output of ``fit_standardizer``.

    Returns
    -------
    np.ndarray
        Float32 copy, ``(x - mean) / std``.
    """
    result = np.array(signals, dtype=np.float32, copy=True)
    result -= np.float32(stats["mean"])
    result /= np.float32(stats["std"])
    return result


def build_model() -> torch.nn.Module:
    """
    Build the vendored benchmark ``xresnet1d101`` with ``MODEL_KWARGS``.

    Returns
    -------
    torch.nn.Module
        Network mapping ``[batch, 12, samples]`` to ``[batch, 14]`` logits, initialized as the benchmark
        initializes a model trained from scratch (``model.apply(weight_init)`` after construction).
    """
    if str(VENDORED) not in sys.path:
        sys.path.insert(0, str(VENDORED))
    from clinical_ts.models.basic_conv1d import weight_init
    from clinical_ts.models.xresnet1d import xresnet1d101

    return xresnet1d101(**MODEL_KWARGS).apply(weight_init)


def masked_bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Binary cross-entropy averaged over the defined label entries.

    Parameters
    ----------
    logits : torch.Tensor
        ``[batch, conditions]`` logits.
    labels : torch.Tensor
        Aligned states 1, 0 or NaN (undefined).

    Returns
    -------
    torch.Tensor
        Scalar loss; an undefined entry contributes nothing to the value or the gradient.
    """
    defined = ~torch.isnan(labels)
    losses = F.binary_cross_entropy_with_logits(logits, torch.nan_to_num(labels, nan=0.0), reduction="none")
    return torch.where(defined, losses, torch.zeros_like(losses)).sum() / defined.sum().clamp(min=1)


def log_loss_defined(labels: np.ndarray, probability: np.ndarray) -> float:
    """
    Binary cross-entropy of probabilities over the defined label entries.

    Parameters
    ----------
    labels : np.ndarray
        ``[records, conditions]`` states 1, 0 or NaN.
    probability : np.ndarray
        Aligned probabilities.

    Returns
    -------
    float
        Mean loss, probabilities clipped to ``[PROBABILITY_CLIP, 1 - PROBABILITY_CLIP]``.
    """
    defined = ~np.isnan(labels)
    y, p = labels[defined], np.clip(probability[defined], PROBABILITY_CLIP, 1 - PROBABILITY_CLIP)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def epoch_batches(records: int, seed: int, epoch: int, batch: int = BATCH) -> tuple[np.ndarray, np.ndarray]:
    """
    Shuffled full batches and one random crop start per record for one epoch.

    Parameters
    ----------
    records : int
        Training records.
    seed : int
        Run seed.
    epoch : int
        Epoch number; the order depends only on ``(seed, epoch)``, so a resumed run sees the same batches.
    batch : int
        Batch size; the last partial batch is dropped.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``[batches, batch]`` record rows and their crop starts in ``[0, RECORD_SAMPLES - CROP - 1]``.
    """
    rng = np.random.default_rng([seed, epoch])
    order = rng.permutation(records)
    starts = rng.integers(0, RECORD_SAMPLES - CROP, size=records)
    full = records // batch * batch
    return order[:full].reshape(-1, batch), starts[order[:full]].reshape(-1, batch)


def crops(signals: np.ndarray, rows: np.ndarray, starts: np.ndarray, length: int = CROP) -> np.ndarray:
    """
    Cut one window per selected record.

    Parameters
    ----------
    signals : np.ndarray
        ``[records, 12, samples]`` waveforms.
    rows : np.ndarray
        Selected records.
    starts : np.ndarray
        Start sample of each selected record's window.
    length : int
        Window length in samples.

    Returns
    -------
    np.ndarray
        Contiguous ``[len(rows), 12, length]`` copy.
    """
    view = np.lib.stride_tricks.sliding_window_view(signals, length, axis=2)
    return np.ascontiguousarray(view[rows, :, starts])


@torch.no_grad()
def window_probabilities(
    model: torch.nn.Module, signals: np.ndarray, device: str, batch: int = EVAL_BATCH
) -> np.ndarray:
    """
    Mean probability over the four consecutive 2.5 s windows of each 10 s record.

    Parameters
    ----------
    model : torch.nn.Module
        Trained network; put in evaluation mode here.
    signals : np.ndarray
        Standardized float32 ``[records, 12, 1000]``.
    device : str
        Torch device.
    batch : int
        Records per forward pass (four windows each).

    Returns
    -------
    np.ndarray
        Float64 ``[records, 14]``; the sigmoid is taken in float64.
    """
    model.eval()
    result = np.empty((len(signals), len(CONDITIONS)), dtype=np.float64)
    for start in range(0, len(signals), batch):
        part = torch.from_numpy(np.ascontiguousarray(signals[start : start + batch])).to(device)
        windows = torch.stack([part[:, :, first : first + CROP] for first in WINDOW_STARTS], dim=1)
        logits = model(windows.reshape(-1, LEADS, CROP)).reshape(len(part), len(WINDOW_STARTS), -1)
        result[start : start + len(part)] = torch.sigmoid(logits.double()).mean(dim=1).cpu().numpy()
    return result


def make_optimizer(
    model: torch.nn.Module, steps: int, epochs: int = EPOCHS
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """
    AdamW on every parameter with the one-cycle schedule.

    Parameters
    ----------
    model : torch.nn.Module
        Network.
    steps : int
        Updates per epoch.
    epochs : int
        Epochs.

    Returns
    -------
    tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]
        Optimizer and scheduler, stepped once per update.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=WEIGHT_DECAY, **ADAM)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=PEAK_LR, total_steps=steps * epochs, **ONE_CYCLE
    )
    return optimizer, scheduler


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    signals: np.ndarray,
    labels: np.ndarray,
    rows: np.ndarray,
    starts: np.ndarray,
    device: str,
) -> torch.Tensor:
    """
    One update on one batch of crops.

    Parameters
    ----------
    model : torch.nn.Module
        Network in training mode.
    optimizer : torch.optim.Optimizer
        Output of ``make_optimizer``.
    scheduler : torch.optim.lr_scheduler.LRScheduler
        Output of ``make_optimizer``; stepped after the update.
    signals : np.ndarray
        Standardized float32 ``[records, 12, 1000]``.
    labels : np.ndarray
        Float32 ``[records, 14]`` states with NaN undefined.
    rows, starts : np.ndarray
        Batch records and their crop starts.
    device : str
        Torch device.

    Returns
    -------
    torch.Tensor
        Detached loss on the device.
    """
    x = torch.from_numpy(crops(signals, rows, starts)).to(device, non_blocking=True)
    y = torch.from_numpy(labels[rows]).to(device, non_blocking=True)
    loss = masked_bce(model(x), y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.detach()


def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    signals: np.ndarray,
    labels: np.ndarray,
    seed: int,
    epoch: int,
    device: str,
) -> dict[str, float]:
    """
    One pass over the training records, one random crop each.

    Parameters
    ----------
    model : torch.nn.Module
        Network.
    optimizer : torch.optim.Optimizer
        Output of ``make_optimizer``.
    scheduler : torch.optim.lr_scheduler.LRScheduler
        Output of ``make_optimizer``.
    signals : np.ndarray
        Standardized float32 ``[records, 12, 1000]``.
    labels : np.ndarray
        Float32 ``[records, 14]`` states with NaN undefined.
    seed : int
        Run seed.
    epoch : int
        Epoch number.
    device : str
        Torch device.

    Returns
    -------
    dict[str, float]
        ``train_loss`` (mean over updates), ``updates``, ``learning_rate`` and ``momentum`` after the epoch.

    Raises
    ------
    FloatingPointError
        If the mean loss is not finite.
    """
    model.train()
    rows, starts = epoch_batches(len(signals), seed, epoch)
    losses = [
        train_step(model, optimizer, scheduler, signals, labels, batch_rows, batch_starts, device)
        for batch_rows, batch_starts in zip(rows, starts, strict=True)
    ]
    mean = float(torch.stack(losses).mean())
    if not math.isfinite(mean):
        raise FloatingPointError(f"training loss is {mean} in epoch {epoch}")
    group = optimizer.param_groups[0]
    return {
        "train_loss": mean,
        "updates": len(losses),
        "learning_rate": float(group["lr"]),
        "momentum": float(group["betas"][0]),
    }


def choose_epoch(history: list[dict[str, Any]]) -> int:
    """
    Epoch with the highest validation selection metric, the earliest on equality.

    Parameters
    ----------
    history : list[dict[str, Any]]
        Per-epoch records with ``epoch`` and ``selection_metric``.

    Returns
    -------
    int
        Chosen epoch.
    """
    best = max(row["selection_metric"] for row in history)
    return min(row["epoch"] for row in history if row["selection_metric"] == best)


def load_network(path: Path, device: str) -> torch.nn.Module:
    """
    Network with saved weights, in evaluation mode.

    Parameters
    ----------
    path : Path
        ``state_dict`` file.
    device : str
        Torch device.

    Returns
    -------
    torch.nn.Module
        The network.
    """
    model = build_model()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.to(device).eval()


def predict(
    waveforms: np.ndarray, directory: Path = CANDIDATE_DIR, device: str = "cuda", batch: int = EVAL_BATCH
) -> np.ndarray:
    """
    Three-seed S-R probabilities of raw cached windows.

    The input path is the training one: rail value for missing samples, 100 Hz, the saved training
    standardizer, four 2.5 s windows, mean probability over windows, then mean over seeds.

    Parameters
    ----------
    waveforms : np.ndarray
        Float32 ``[records, 12, 5000]`` at 500 Hz in mV, leads I, II, III, aVR, aVL, aVF, V1-V6.
    directory : Path
        Candidate directory holding ``settings.json`` (normalization and checkpoint names).
    device : str
        Torch device.
    batch : int
        Records per forward pass.

    Returns
    -------
    np.ndarray
        Float64 ``[records, 14]`` probabilities in ``CONDITIONS`` order.

    Raises
    ------
    ValueError
        If the waveforms have another shape.
    """
    waveforms = np.asarray(waveforms, dtype=np.float32)
    if waveforms.ndim != 3 or waveforms.shape[1:] != (LEADS, SAMPLES):
        raise ValueError(f"expected [records, 12, 5000] waveforms, got {waveforms.shape}")
    settings = read_json(Path(directory) / "settings.json")
    signals = standardize(to_model_rate(waveforms), settings["normalization"])
    probabilities = [
        window_probabilities(
            load_network(Path(directory) / run["checkpoint"], device), signals, device, batch
        )
        for run in settings["seeds"]
    ]
    return np.mean(probabilities, axis=0)
