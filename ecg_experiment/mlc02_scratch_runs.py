"""
Runs of candidate S-R (``mlc02_scratch_network``): inputs, profile, overfit check, training, publication.

Every step reads train and validation records only. The 100 Hz inputs are prepared once on local disk
(``WORK/inputs``). Each training run fits its own standardizer on its training records, keeps per-epoch
weights and validation probabilities on local disk, and resumes from ``state.pt`` after an interruption.
Published files go to ``CANDIDATE_DIR`` and ``LOFO_DIR``.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import resource
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ecg_experiment.files import (
    read_json,
    sha256_file,
    write_json_atomic,
    write_text_atomic,
    write_torch_atomic,
)
from ecg_experiment.mlc02_cache_build import code_sha256, git_state
from ecg_experiment.mlc02_data import (
    CONDITIONS,
    MANIFEST,
    load_manifest,
    load_predictions,
    prediction_path,
    role_data,
    save_predictions,
)
from ecg_experiment.mlc02_evaluation import auroc_by_condition, enters_macro, selection_metric, support
from ecg_experiment.mlc02_features import torch_flags
from ecg_experiment.mlc02_scratch_network import (
    ALLOWED_ROLES,
    BATCH,
    CACHE,
    CANDIDATE_DIR,
    CROP,
    EPOCHS,
    FAMILIES,
    INDEX,
    LOFO_DIR,
    MODEL_ID,
    RECORD_SAMPLES,
    ROWS,
    SEEDS,
    WINDOW_STARTS,
    WORK,
    build_model,
    choose_epoch,
    epoch_batches,
    fit_standardizer,
    load_network,
    log_loss_defined,
    make_optimizer,
    predict,
    read_records,
    read_windows,
    select_roles,
    standardize,
    to_model_rate,
    train_epoch,
    train_step,
    window_probabilities,
)
from ecg_experiment.paths import to_stored

CHUNK = 2048
HEADROOM_MIB = 1024
EDAC = Path("/sys/devices/system/edac/mc")
TOLERANCE = 1e-5
CODE = (
    "ecg_experiment/mlc02_scratch_network.py",
    "ecg_experiment/mlc02_scratch_runs.py",
    "scripts/experiments/run_mlc02_scratch_network.py",
    "ecg_experiment/mlc02_data.py",
    "ecg_experiment/mlc02_evaluation.py",
    "ecg_experiment/mlc02_features.py",
    "ecg_experiment/mlc02_waveforms.py",
    "ecg_experiment/mlc01_dataset.py",
    "ecg_experiment/resample.py",
    "ecg_experiment/files.py",
    "ecg_experiment/gpu.py",
    "third_party/ecg-fm-benchmarking/code/clinical_ts/models/xresnet1d.py",
    "third_party/ecg-fm-benchmarking/code/clinical_ts/models/basic_conv1d.py",
)
PACKAGES = ("torch", "numpy", "scipy", "pandas", "scikit-learn")
ENVIRONMENT = (
    "CUDA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_CORETYPE",
    "PYTHONPATH",
    "CUBLAS_WORKSPACE_CONFIG",
)


def log(path: Path, message: str) -> None:
    """
    Print a line and append it to a log file.

    Parameters
    ----------
    path : Path
        Log file.
    message : str
        Text.
    """
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(line + "\n")


def clean(value: Any) -> Any:
    """
    JSON-safe copy: NumPy scalars to Python, NaN and infinity to None.

    Parameters
    ----------
    value : Any
        Nested dicts, lists, tuples and scalars.

    Returns
    -------
    Any
        The cleaned value.
    """
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [clean(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def edac_counts() -> dict[str, int]:
    """
    Read the kernel's corrected and uncorrected memory error counters.

    Returns
    -------
    dict[str, int]
        ``mc<N>/ce_count`` and ``mc<N>/ue_count`` of every memory controller; empty without EDAC.
    """
    return {
        f"{path.parent.name}/{path.name}": int(path.read_text())
        for path in sorted(EDAC.glob("mc*/[cu]e_count"))
    }


def environment() -> dict[str, Any]:
    """
    Describe the packages, devices, torch flags and environment variables of this process.

    Returns
    -------
    dict[str, Any]
        Receipt section.
    """
    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in PACKAGES},
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "devices": devices,
        "torch_threads": torch.get_num_threads(),
        "torch_flags": torch_flags(),
        "variables": {name: os.environ.get(name) for name in ENVIRONMENT},
        "edac": edac_counts(),
    }


def provenance() -> dict[str, Any]:
    """
    Git state and code digests; the run refuses a modified tracked file.

    Returns
    -------
    dict[str, Any]
        ``git`` and ``code_sha256``.

    Raises
    ------
    RuntimeError
        If a tracked file differs from the commit.
    """
    state = git_state()
    if state["modified_tracked"]:
        raise RuntimeError(f"commit the code first; modified: {state['modified_tracked']}")
    return {"git": state, "code_sha256": code_sha256(CODE)}


def peak_rss_mib() -> float:
    """
    Peak resident memory of this process.

    Returns
    -------
    float
        MiB.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def require_free_gpu(needed_mib: float) -> dict[str, float]:
    """
    Check that the current CUDA device has the memory a job needs plus ``HEADROOM_MIB``.

    Parameters
    ----------
    needed_mib : float
        Expected peak reserved memory of the job.

    Returns
    -------
    dict[str, float]
        ``free_mib`` and ``total_mib`` before the job.

    Raises
    ------
    RuntimeError
        If less is free.
    """
    free, total = torch.cuda.mem_get_info()
    record = {"free_mib": free / 2**20, "total_mib": total / 2**20}
    if record["free_mib"] < needed_mib + HEADROOM_MIB:
        raise RuntimeError(
            f"GPU has {record['free_mib']:.0f} MiB free, needs {needed_mib + HEADROOM_MIB:.0f}"
        )
    return record


def prepare(work: Path = WORK) -> dict[str, Any]:
    """
    Read, check and resample every train and validation record to 100 Hz on local disk.

    Parameters
    ----------
    work : Path
        Local work directory; writes ``inputs/signals_100hz.npy``, ``inputs/records.csv`` and
        ``inputs/prepare.json``.

    Returns
    -------
    dict[str, Any]
        The prepare record.
    """
    origin = provenance()
    started = time.time()
    records = select_roles(read_records(), ALLOWED_ROLES).reset_index(drop=True)
    directory = work / "inputs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "signals_100hz.npy"
    signals = np.lib.format.open_memmap(
        path.with_suffix(".tmp.npy"), mode="w+", dtype=np.float32, shape=(len(records), 12, RECORD_SAMPLES)
    )
    missing = 0
    for start in range(0, len(records), CHUNK):
        windows = read_windows(records.iloc[start : start + CHUNK], CACHE)
        missing += int(np.isnan(windows).sum())
        signals[start : start + len(windows)] = to_model_rate(windows)
    signals.flush()
    finite = bool(np.isfinite(signals).all())
    del signals
    os.replace(path.with_suffix(".tmp.npy"), path)
    records.to_csv(directory / "records.csv", index=False)
    record = {
        "records": len(records),
        "by_role": records["role"].value_counts().to_dict(),
        "blocks_opened": sorted(records["block"].unique().tolist()),
        "missing_samples_500hz": missing,
        "finite_100hz": finite,
        "hash_checked": len(records),
        "manifest": to_stored(MANIFEST),
        "manifest_sha256": sha256_file(MANIFEST),
        "rows_sha256": sha256_file(ROWS),
        "index_all_sha256": sha256_file(INDEX),
        "cache_receipts_sha256": {
            name: sha256_file(CACHE / name) for name in ("receipt.json", "receipt_supplement.json")
        },
        "signals_sha256": sha256_file(path),
        "records_sha256": sha256_file(directory / "records.csv"),
        "seconds": time.time() - started,
        "peak_rss_mib": peak_rss_mib(),
        **origin,
    }
    write_json_atomic(directory / "prepare.json", clean(record))
    return record


def load_inputs(work: Path = WORK) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Load the prepared 100 Hz inputs with their manifest labels.

    Parameters
    ----------
    work : Path
        Local work directory.

    Returns
    -------
    tuple[pd.DataFrame, np.ndarray, np.ndarray]
        Records (``record_key``, ``source``, ``family``, ``role``), memory-mapped ``[records, 12, 1000]``
        signals and float64 ``[records, 14]`` labels.

    Raises
    ------
    ValueError
        If the prepared records are not the manifest's train and validation records in order, or the signal
        file changed since ``prepare``.
    """
    directory = work / "inputs"
    receipt = read_json(directory / "prepare.json")
    if sha256_file(directory / "records.csv") != receipt["records_sha256"]:
        raise ValueError("prepared records changed since prepare")
    records = pd.read_csv(directory / "records.csv", dtype=str)
    check = select_roles(pd.DataFrame({"role": records["role"]}), ALLOWED_ROLES)
    data = load_manifest(MANIFEST)
    keep = np.isin(data["role"], ALLOWED_ROLES)
    if len(check) != len(records) or not np.array_equal(data["record_key"][keep], records["record_key"]):
        raise ValueError("prepared records are not the manifest's train and validation records")
    signals = np.load(directory / "signals_100hz.npy", mmap_mode="r")
    if signals.shape != (len(records), 12, RECORD_SAMPLES):
        raise ValueError(f"prepared signals have shape {signals.shape}")
    return records, signals, data["labels"][keep]


def run_data(
    records: pd.DataFrame, signals: np.ndarray, labels: np.ndarray, train: np.ndarray, validation: np.ndarray
) -> dict[str, Any]:
    """
    Standardized arrays of one run, with the standardizer fitted on its training records.

    Parameters
    ----------
    records : pd.DataFrame
        Output of ``load_inputs``.
    signals : np.ndarray
        Output of ``load_inputs``.
    labels : np.ndarray
        Output of ``load_inputs``.
    train, validation : np.ndarray
        Boolean masks of the run's training and epoch-selection records.

    Returns
    -------
    dict[str, Any]
        ``train_x``, ``train_y`` (float32, NaN undefined), ``validation_x``, ``validation_y`` (float64),
        ``validation_sources``, ``validation_keys``, ``normalization`` and ``train_records``.
    """
    train_signals = np.asarray(signals[train])
    stats = fit_standardizer(train_signals)
    return {
        "train_x": standardize(train_signals, stats),
        "train_y": labels[train].astype(np.float32),
        "validation_x": standardize(np.asarray(signals[validation]), stats),
        "validation_y": labels[validation],
        "validation_sources": records["source"].to_numpy()[validation],
        "validation_keys": records["record_key"].to_numpy()[validation],
        "normalization": stats,
        "train_records": int(train.sum()),
    }


def masks(records: pd.DataFrame, family: str | None) -> dict[str, np.ndarray]:
    """
    Training, selection and held-out masks of the main runs (``family`` None) or a leave-one-family-out run.

    Parameters
    ----------
    records : pd.DataFrame
        Output of ``load_inputs``.
    family : str | None
        Held-out family.

    Returns
    -------
    dict[str, np.ndarray]
        ``train``, ``validation`` and, for a family, ``held_out`` (its train and validation records).
    """
    role, families = records["role"].to_numpy(), records["family"].to_numpy()
    if family is None:
        return {"train": role == "train", "validation": role == "validation"}
    inside = families != family
    return {
        "train": (role == "train") & inside,
        "validation": (role == "validation") & inside,
        "held_out": ~inside,
    }


def validate(model: torch.nn.Module, data: dict[str, Any], device: str) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Window-mean validation probabilities and the selection metric.

    Parameters
    ----------
    model : torch.nn.Module
        Network.
    data : dict[str, Any]
        Output of ``run_data``.
    device : str
        Torch device.

    Returns
    -------
    tuple[np.ndarray, dict[str, Any]]
        Probabilities and ``validation_loss``, ``selection_metric`` and ``macro_auroc_by_source``.
    """
    probability = window_probabilities(model, data["validation_x"], device)
    selection = selection_metric(data["validation_y"], probability, data["validation_sources"])
    return probability, {
        "validation_loss": log_loss_defined(data["validation_y"], probability),
        "selection_metric": selection["selection_metric"],
        "macro_auroc_by_source": selection["macro_auroc_by_source"],
    }


def resume(
    directory: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> tuple[int, list[dict[str, Any]]]:
    """
    Restore a run from its last finished epoch, if any.

    Parameters
    ----------
    directory : Path
        Run directory.
    model, optimizer, scheduler
        Freshly built objects, restored in place.

    Returns
    -------
    tuple[int, list[dict[str, Any]]]
        Next epoch and the history so far.
    """
    path = directory / "state.pt"
    if not path.exists():
        return 1, []
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["cpu_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"])
    return state["epoch"] + 1, state["history"]


def train_run(
    directory: Path,
    seed: int,
    data: dict[str, Any],
    device: str,
    epochs: int = EPOCHS,
    report: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """
    Train one run for ``epochs`` epochs, validating and checkpointing after each; resumable.

    Parameters
    ----------
    directory : Path
        Local run directory (``epoch_NNN.pt``, ``validation_epoch_NNN.npy``, ``state.pt``).
    seed : int
        Initialization, shuffling, crop and dropout seed.
    data : dict[str, Any]
        Output of ``run_data``.
    device : str
        Torch device.
    epochs : int
        Epochs.
    report : Callable[[str], None]
        Logger.

    Returns
    -------
    list[dict[str, Any]]
        One record per epoch.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = build_model().to(device)
    optimizer, scheduler = make_optimizer(model, len(data["train_x"]) // BATCH, epochs)
    first, history = resume(directory, model, optimizer, scheduler)
    for epoch in range(first, epochs + 1):
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        row = {
            "epoch": epoch,
            **train_epoch(model, optimizer, scheduler, data["train_x"], data["train_y"], seed, epoch, device),
        }
        trained = time.time()
        probability, scores = validate(model, data, device)
        row |= scores | {
            "seconds_train": trained - started,
            "seconds_validation": time.time() - trained,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        }
        write_torch_atomic(directory / f"epoch_{epoch:03d}.pt", model.state_dict())
        np.save(directory / f"validation_epoch_{epoch:03d}.npy", probability)
        history.append(clean(row))
        write_json_atomic(directory / "history.json", history)
        write_torch_atomic(
            directory / "state.pt",
            {
                "epoch": epoch,
                "history": history,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(),
            },
        )
        report(
            f"{directory.name} epoch {epoch}: loss {row['train_loss']:.4f}, validation loss "
            f"{row['validation_loss']:.4f}, selection {row['selection_metric']:.4f}, "
            f"{row['seconds_train']:.1f}+{row['seconds_validation']:.1f} s"
        )
    return history


def run_record(
    directory: Path, seed: int, data: dict[str, Any], history: list[dict[str, Any]], family: str | None
) -> dict[str, Any]:
    """
    Summary of a finished run: chosen epoch, its score and the run's settings.

    Parameters
    ----------
    directory : Path
        Local run directory.
    seed : int
        Run seed.
    data : dict[str, Any]
        Output of ``run_data``.
    history : list[dict[str, Any]]
        Output of ``train_run``.
    family : str | None
        Held-out family, or None for a main run.

    Returns
    -------
    dict[str, Any]
        The record written to ``run.json``.
    """
    epoch = choose_epoch(history)
    return clean(
        {
            "run": directory.name,
            "seed": seed,
            "held_out_family": family,
            "chosen_epoch": epoch,
            "chosen": history[epoch - 1],
            "epochs": len(history),
            "train_records": data["train_records"],
            "validation_records": len(data["validation_x"]),
            "normalization": data["normalization"],
            "updates_per_epoch": data["train_records"] // BATCH,
            "seconds_total": sum(row["seconds_train"] + row["seconds_validation"] for row in history),
            "peak_allocated_mib": max(row["peak_allocated_mib"] for row in history),
            "peak_reserved_mib": max(row["peak_reserved_mib"] for row in history),
            "checkpoint_sha256": sha256_file(directory / f"epoch_{epoch:03d}.pt"),
        }
    )


def held_out_scores(
    directory: Path,
    epoch: int,
    records: pd.DataFrame,
    signals: np.ndarray,
    labels: np.ndarray,
    held_out: np.ndarray,
    stats: dict[str, float],
    device: str,
) -> dict[str, Any]:
    """
    Score the held-out family's train and validation records with a leave-one-family-out run's chosen epoch.

    Parameters
    ----------
    directory : Path
        Local run directory.
    epoch : int
        Chosen epoch.
    records, signals, labels
        Output of ``load_inputs``.
    held_out : np.ndarray
        Mask of the held-out family's records.
    stats : dict[str, float]
        The run's standardizer.
    device : str
        Torch device.

    Returns
    -------
    dict[str, Any]
        ``probability`` (float64), ``record_key`` and the per-source table of ``source_tables``.
    """
    model = build_model().to(device)
    model.load_state_dict(
        torch.load(directory / f"epoch_{epoch:03d}.pt", map_location=device, weights_only=True)
    )
    probability = window_probabilities(model, standardize(np.asarray(signals[held_out]), stats), device)
    sources = records["source"].to_numpy()[held_out]
    return {
        "probability": probability,
        "record_key": records["record_key"].to_numpy()[held_out],
        "table": source_tables(labels[held_out], probability, sources),
    }


def source_tables(labels: np.ndarray, probability: np.ndarray, sources: np.ndarray) -> dict[str, Any]:
    """
    Within-source macro AUROC and per-condition AUROC with support, per source.

    Parameters
    ----------
    labels : np.ndarray
        ``[records, 14]`` states.
    probability : np.ndarray
        Aligned probabilities.
    sources : np.ndarray
        Source of each record.

    Returns
    -------
    dict[str, Any]
        Per source: ``records``, ``macro_auroc`` (NaN if no condition enters), ``entering`` and ``conditions``
        (AUROC, positives, negatives, enters_macro).
    """
    tables = {}
    for source in dict.fromkeys(sources):
        mask = sources == source
        counts, entering = support(labels[mask]), enters_macro(labels[mask])
        auroc = auroc_by_condition(labels[mask], probability[mask])
        tables[source] = {
            "records": int(mask.sum()),
            "macro_auroc": float(auroc[entering].mean()) if entering.any() else float("nan"),
            "entering": [name for name, flag in zip(CONDITIONS, entering, strict=True) if flag],
            "conditions": {
                name: {
                    "auroc": float(auroc[c]),
                    "positives": int(counts["positives"][c]),
                    "negatives": int(counts["negatives"][c]),
                    "enters_macro": bool(entering[c]),
                }
                for c, name in enumerate(CONDITIONS)
            },
        }
    return tables


def overfit(
    records: pd.DataFrame,
    signals: np.ndarray,
    labels: np.ndarray,
    device: str,
    size: int = 256,
    epochs: int = 150,
    seed: int = SEEDS[0],
) -> dict[str, Any]:
    """
    Overfit check: the recipe on a fixed subset of training records must drive the loss near zero.

    Parameters
    ----------
    records, signals, labels
        Output of ``load_inputs``.
    device : str
        Torch device.
    size : int
        Training records used.
    epochs : int
        Epochs over the subset (two updates each at batch 128).
    seed : int
        Seed of the subset, initialization and crops.

    Returns
    -------
    dict[str, Any]
        Per-epoch training loss, and the evaluation-mode loss and macro AUROC of the four-window mean on the
        same records before and after.
    """
    rows = np.sort(
        np.random.default_rng(seed).choice(np.flatnonzero(records["role"] == "train"), size, False)
    )
    raw = np.asarray(signals[rows])
    x, y = standardize(raw, fit_standardizer(raw)), labels[rows]
    torch.manual_seed(seed)
    model = build_model().to(device)
    optimizer, scheduler = make_optimizer(model, size // BATCH, epochs)
    before = window_probabilities(model, x, device)
    losses = [
        train_epoch(model, optimizer, scheduler, x, y.astype(np.float32), seed, epoch, device)["train_loss"]
        for epoch in range(1, epochs + 1)
    ]
    after = window_probabilities(model, x, device)
    entering = enters_macro(y)
    return clean(
        {
            "records": size,
            "epochs": epochs,
            "updates": epochs * (size // BATCH),
            "train_loss": losses,
            "evaluation_loss_before": log_loss_defined(y, before),
            "evaluation_loss_after": log_loss_defined(y, after),
            "macro_auroc_after": float(auroc_by_condition(y, after)[entering].mean()),
            "entering_conditions": int(entering.sum()),
            "record_keys": records["record_key"].iloc[rows].tolist(),
        }
    )


def profile_updates(data: dict[str, Any], device: str, updates: int, warmup: int = 20) -> dict[str, Any]:
    """
    Time training updates and one validation pass of the full recipe.

    Parameters
    ----------
    data : dict[str, Any]
        Output of ``run_data`` for the main runs.
    device : str
        Torch device.
    updates : int
        Timed updates after ``warmup``.
    warmup : int
        Untimed updates first.

    Returns
    -------
    dict[str, Any]
        Seconds per update and per validation pass, losses, and peak memory.
    """
    torch.manual_seed(SEEDS[0])
    model = build_model().to(device).train()
    optimizer, scheduler = make_optimizer(model, len(data["train_x"]) // BATCH)
    rows, starts = epoch_batches(len(data["train_x"]), SEEDS[0], 1)
    torch.cuda.reset_peak_memory_stats()
    losses = [
        train_step(
            model, optimizer, scheduler, data["train_x"], data["train_y"], rows[step], starts[step], device
        )
        for step in range(warmup)
    ]
    torch.cuda.synchronize()
    began = time.time()
    losses += [
        train_step(
            model, optimizer, scheduler, data["train_x"], data["train_y"], rows[step], starts[step], device
        )
        for step in range(warmup, warmup + updates)
    ]
    torch.cuda.synchronize()
    per_update = (time.time() - began) / updates
    began = time.time()
    _, scores = validate(model, data, device)
    return {
        "updates": updates,
        "warmup": warmup,
        "seconds_per_update": per_update,
        "seconds_validation_pass": time.time() - began,
        "validation_records": len(data["validation_x"]),
        "loss_first": float(losses[0]),
        "loss_last_20": float(torch.stack(losses[-20:]).mean()),
        "selection_metric_after": scores["selection_metric"],
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }


def forecast(timing: dict[str, Any], records: pd.DataFrame) -> dict[str, Any]:
    """
    Forecast of the whole plan from a profile: three main seeds and four leave-one-family-out runs.

    Parameters
    ----------
    timing : dict[str, Any]
        Output of ``profile_updates``.
    records : pd.DataFrame
        Output of ``load_inputs``.

    Returns
    -------
    dict[str, Any]
        Seconds per run and in total.
    """
    validation_records = int((records["role"] == "validation").sum())
    per_record = timing["seconds_validation_pass"] / validation_records
    runs = {f"main_seed-{seed}": masks(records, None) for seed in SEEDS}
    runs |= {f"lofo_{family}": masks(records, family) for family in sorted(set(records["family"]))}
    seconds = {}
    for name, mask in runs.items():
        steps = int(mask["train"].sum()) // BATCH
        held = int(mask.get("held_out", np.zeros(1, bool)).sum())
        seconds[name] = EPOCHS * (
            steps * timing["seconds_per_update"] + int(mask["validation"].sum()) * per_record
        )
        seconds[name] += held * per_record
    return {
        "seconds_by_run": seconds,
        "seconds_total": sum(seconds.values()),
        "hours_total": sum(seconds.values()) / 3600,
    }


def copy_checkpoint(source: Path, target: Path) -> str:
    """
    Copy a checkpoint and return its SHA-256, checked on both sides.

    Parameters
    ----------
    source, target : Path
        Files.

    Returns
    -------
    str
        Digest.

    Raises
    ------
    ValueError
        If the copy differs.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    digest = sha256_file(source)
    if sha256_file(target) != digest:
        raise ValueError(f"copy of {source} differs")
    return digest


def epoch_table(histories: dict[str, list[dict[str, Any]]]) -> str:
    """
    CSV of every run's per-epoch log.

    Parameters
    ----------
    histories : dict[str, list[dict[str, Any]]]
        Run name to history.

    Returns
    -------
    str
        CSV text with one row per run and epoch; per-source macros as ``macro_<source>`` columns.
    """
    rows = []
    for name, history in histories.items():
        for row in history:
            flat = {key: value for key, value in row.items() if key != "macro_auroc_by_source"}
            rows.append(
                {
                    "run": name,
                    **flat,
                    **{f"macro_{key}": value for key, value in row["macro_auroc_by_source"].items()},
                }
            )
    return pd.DataFrame(rows).to_csv(index=False, lineterminator="\n")


def recipe() -> dict[str, Any]:
    """
    Describe the settings of S-R as run, for ``settings.json``.

    Returns
    -------
    dict[str, Any]
        Constants of ``mlc02_scratch_network``.
    """
    from ecg_experiment import mlc02_scratch_network as network

    model = build_model()
    return {
        "model": "xresnet1d101 (third_party/ecg-fm-benchmarking/code/clinical_ts/models/xresnet1d.py)",
        "model_kwargs": network.MODEL_KWARGS,
        "parameters": sum(p.numel() for p in model.parameters()),
        "input": "cached 500 Hz first 10 s; missing samples at -32.768 mV; resample_poly 1/5 over 10 s; "
        "one training mean and standard deviation over all samples and leads",
        "model_rate_hz": 100,
        "crop_samples": CROP,
        "window_starts": list(WINDOW_STARTS),
        "epochs": EPOCHS,
        "batch": BATCH,
        "drop_last": True,
        "peak_lr": network.PEAK_LR,
        "weight_decay": network.WEIGHT_DECAY,
        "optimizer": "torch.optim.AdamW on all parameters",
        "adam": network.ADAM,
        "one_cycle": network.ONE_CYCLE,
        "loss": "binary cross-entropy with logits, mean over defined entries",
        "selection": "epoch with the highest validation selection metric, earliest on equality",
        "evaluation": "mean of float64 sigmoid probabilities over the four consecutive 2.5 s windows",
        "seeds": list(SEEDS),
        "augmentation": "random 2.5 s crop only",
        "class_weights": None,
    }


def run_name(seed: int, family: str | None) -> str:
    """
    Local directory name of a run.

    Parameters
    ----------
    seed : int
        Run seed.
    family : str | None
        Held-out family, or None for a main run.

    Returns
    -------
    str
        ``main_seed-<seed>`` or ``lofo_<family>_seed-<seed>``.
    """
    return f"main_seed-{seed}" if family is None else f"lofo_{family}_seed-{seed}"


def train_and_record(seed: int, family: str | None, device: str, work: Path = WORK) -> dict[str, Any]:
    """
    Train one main or leave-one-family-out run (resumable) and write its ``run.json``.

    A leave-one-family-out run also scores the held-out family's train and validation records with its chosen
    epoch (``held_out.npz``).

    Parameters
    ----------
    seed : int
        Run seed.
    family : str | None
        Held-out family, or None for a main run.
    device : str
        Torch device.
    work : Path
        Local work directory.

    Returns
    -------
    dict[str, Any]
        The run record.
    """
    origin = provenance()
    started = time.time()
    records, signals, labels = load_inputs(work)
    selected = masks(records, family)
    data = run_data(records, signals, labels, selected["train"], selected["validation"])
    directory = work / "runs" / run_name(seed, family)
    directory.mkdir(parents=True, exist_ok=True)
    write_text_atomic(directory / "validation_keys.txt", "\n".join(data["validation_keys"]) + "\n")
    history = train_run(
        directory, seed, data, device, report=lambda message: log(directory / "train.log", message)
    )
    record = run_record(directory, seed, data, history, family)
    if family is not None:
        scores = held_out_scores(
            directory,
            record["chosen_epoch"],
            records,
            signals,
            labels,
            selected["held_out"],
            data["normalization"],
            device,
        )
        np.savez(
            directory / "held_out.npz",
            record_key=np.asarray(scores["record_key"], dtype=str),
            probability=scores["probability"],
        )
        record |= {"held_out_records": int(selected["held_out"].sum()), "held_out": scores["table"]}
    record |= {
        "wall_seconds_last_call": time.time() - started,
        "peak_rss_mib": peak_rss_mib(),
        "environment": environment(),
        **origin,
    }
    write_json_atomic(directory / "run.json", clean(record))
    return record


def validation_arrays(work: Path = WORK) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Keys, labels and sources of the validation records, checked against the scoring library's role order.

    Parameters
    ----------
    work : Path
        Local work directory.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Record keys, float64 labels and sources.

    Raises
    ------
    ValueError
        If the order differs from ``mlc02_data.role_data(..., "validation")``.
    """
    records, _, labels = load_inputs(work)
    mask = (records["role"] == "validation").to_numpy()
    reference = role_data(load_manifest(MANIFEST), "validation")
    keys = records["record_key"].to_numpy()[mask]
    if not np.array_equal(keys, reference["record_key"]) or not np.array_equal(
        labels[mask], reference["labels"], equal_nan=True
    ):
        raise ValueError("validation records differ from the scoring library's")
    return keys, labels[mask], records["source"].to_numpy()[mask]


def publish_seed(seed: int, keys: np.ndarray, work: Path, output: Path) -> tuple[dict[str, Any], np.ndarray]:
    """
    Publish one main seed: chosen checkpoint and validation predictions.

    Parameters
    ----------
    seed : int
        Run seed.
    keys : np.ndarray
        Validation record keys.
    work : Path
        Local work directory.
    output : Path
        Candidate directory.

    Returns
    -------
    tuple[dict[str, Any], np.ndarray]
        The seed entry of ``settings.json`` and its validation probabilities.

    Raises
    ------
    ValueError
        If the run's validation keys differ from ``keys``.
    """
    directory = work / "runs" / run_name(seed, None)
    if (directory / "validation_keys.txt").read_text().split() != keys.tolist():
        raise ValueError(f"run {directory.name} was validated on other records")
    run = read_json(directory / "run.json")
    epoch = run["chosen_epoch"]
    probability = np.load(directory / f"validation_epoch_{epoch:03d}.npy")
    checkpoint = f"seed-{seed}/model_epoch-{epoch:03d}.pt"
    digest = copy_checkpoint(directory / f"epoch_{epoch:03d}.pt", output / checkpoint)
    save_predictions(
        prediction_path(output.parent, MODEL_ID, "validation", (seed,)), keys, probability, MODEL_ID, (seed,)
    )
    return {
        "seed": seed,
        "chosen_epoch": epoch,
        "checkpoint": checkpoint,
        "sha256": digest,
        "selection_metric": run["chosen"]["selection_metric"],
        "normalization": run["normalization"],
        "seconds_total": run["seconds_total"],
        "peak_allocated_mib": run["peak_allocated_mib"],
        "peak_reserved_mib": run["peak_reserved_mib"],
        "git": run["git"],
    }, probability


def publish_main(work: Path = WORK, output: Path = CANDIDATE_DIR) -> dict[str, Any]:
    """
    Publish the three main seeds and their mean: checkpoints, settings, predictions and validation tables.

    Parameters
    ----------
    work : Path
        Local work directory.
    output : Path
        Candidate directory (``CANDIDATE_DIR``).

    Returns
    -------
    dict[str, Any]
        ``results.json`` content.

    Raises
    ------
    ValueError
        If the seeds used different standardizers.
    """
    keys, labels, sources = validation_arrays(work)
    entries, probabilities = zip(*(publish_seed(seed, keys, work, output) for seed in SEEDS), strict=True)
    if len({repr(entry["normalization"]) for entry in entries}) != 1:
        raise ValueError("the seeds were standardized differently")
    mean = np.mean(probabilities, axis=0)
    path = prediction_path(output.parent, MODEL_ID, "validation", SEEDS)
    save_predictions(path, keys, mean, MODEL_ID, SEEDS)
    load_predictions(path, role_data(load_manifest(MANIFEST), "validation")["record_key"])
    settings = {
        "model_id": MODEL_ID,
        "recipe": recipe(),
        "normalization": entries[0]["normalization"],
        "seeds": [
            {key: value for key, value in entry.items() if key != "normalization"} for entry in entries
        ],
    }
    write_json_atomic(output / "settings.json", clean(settings))
    write_json_atomic(output / "normalization.json", clean(entries[0]["normalization"]))
    histories = {
        run_name(seed, None): read_json(work / "runs" / run_name(seed, None) / "history.json")
        for seed in SEEDS
    }
    write_text_atomic(output / "epochs.csv", epoch_table(histories))
    named = {str(seed): probability for seed, probability in zip(SEEDS, probabilities, strict=True)}
    named["mean"] = mean
    results = {
        "selection": {name: selection_metric(labels, p, sources) for name, p in named.items()},
        "per_source": {name: source_tables(labels, p, sources) for name, p in named.items()},
        "chosen_epochs": {str(entry["seed"]): entry["chosen_epoch"] for entry in entries},
        "predictions": {
            name: to_stored(prediction_path(output.parent, MODEL_ID, "validation", seeds))
            for name, seeds in [*((str(s), (s,)) for s in SEEDS), ("mean", SEEDS)]
        },
        "test_roles_scored": False,
    }
    write_json_atomic(output / "results.json", clean(results))
    write_receipt(output, [run_name(seed, None) for seed in SEEDS], work)
    return results


def publish_lofo(work: Path = WORK, output: Path = LOFO_DIR, seed: int = SEEDS[0]) -> dict[str, Any]:
    """
    Publish the four leave-one-family-out runs: chosen checkpoints, held-out predictions and tables.

    Parameters
    ----------
    work : Path
        Local work directory.
    output : Path
        ``LOFO_DIR``.
    seed : int
        The runs' seed.

    Returns
    -------
    dict[str, Any]
        ``lofo.json`` content.
    """
    summary, histories = {}, {}
    for family in FAMILIES:
        directory = work / "runs" / run_name(seed, family)
        run = read_json(directory / "run.json")
        epoch = run["chosen_epoch"]
        digest = copy_checkpoint(
            directory / f"epoch_{epoch:03d}.pt", output / family / f"model_epoch-{epoch:03d}.pt"
        )
        with np.load(directory / "held_out.npz") as archive:
            save_predictions(
                output / family / f"held_out__seeds-{seed}.npz",
                archive["record_key"],
                archive["probability"],
                MODEL_ID,
                (seed,),
            )
        histories[directory.name] = read_json(directory / "history.json")
        summary[family] = {
            "chosen_epoch": epoch,
            "checkpoint_sha256": digest,
            "train_records": run["train_records"],
            "validation_records": run["validation_records"],
            "held_out_records": run["held_out_records"],
            "normalization": run["normalization"],
            "selection_metric_inside": run["chosen"]["selection_metric"],
            "macro_auroc_inside_by_source": run["chosen"]["macro_auroc_by_source"],
            "held_out": run["held_out"],
            "seconds_total": run["seconds_total"],
            "peak_reserved_mib": run["peak_reserved_mib"],
            "git": run["git"],
        }
    write_text_atomic(output / "epochs.csv", epoch_table(histories))
    record = {"seed": seed, "families": summary, "test_roles_scored": False}
    write_json_atomic(output / "lofo.json", clean(record))
    write_receipt(output, [run_name(seed, family) for family in FAMILIES], work)
    return record


def check_predict(device: str, output: Path = CANDIDATE_DIR, tolerance: float = 1e-5) -> dict[str, Any]:
    """
    Recompute the three-seed validation predictions from the 500 Hz cache with ``predict``.

    Parameters
    ----------
    device : str
        Torch device.
    output : Path
        Candidate directory.
    tolerance : float
        Largest allowed absolute difference.

    Returns
    -------
    dict[str, Any]
        Records, largest and mean absolute difference; written to ``predict_check.json``.

    Raises
    ------
    ValueError
        If a difference exceeds ``tolerance``.
    """
    records = select_roles(read_records(), ["validation"])
    saved = load_predictions(
        prediction_path(output.parent, MODEL_ID, "validation", SEEDS), records["record_key"].to_numpy()
    )["probability"]
    started = time.time()
    probability = predict(read_windows(records), output, device)
    difference = np.abs(probability - saved)
    record = {
        "records": len(records),
        "max_abs_difference": float(difference.max()),
        "mean_abs_difference": float(difference.mean()),
        "tolerance": tolerance,
        "device": device,
        "seconds": time.time() - started,
        **provenance(),
    }
    write_json_atomic(output / "predict_check.json", clean(record))
    if record["max_abs_difference"] > tolerance:
        raise ValueError(f"predict differs from the saved predictions by {record['max_abs_difference']}")
    return record


def profile(
    device: str, updates: int = 300, work: Path = WORK, output: Path = CANDIDATE_DIR
) -> dict[str, Any]:
    """
    Profile the recipe on the main training records and forecast the whole plan.

    Parameters
    ----------
    device : str
        Torch device.
    updates : int
        Timed updates.
    work : Path
        Local work directory.
    output : Path
        Candidate directory; writes ``profile.json``.

    Returns
    -------
    dict[str, Any]
        The profile record.
    """
    origin = provenance()
    records, signals, labels = load_inputs(work)
    selected = masks(records, None)
    data = run_data(records, signals, labels, selected["train"], selected["validation"])
    before = require_free_gpu(2048)
    timing = profile_updates(data, device, updates)
    record = {
        "timing": timing,
        "forecast": forecast(timing, records),
        "gpu_before": before,
        "peak_rss_mib": peak_rss_mib(),
        "environment": environment(),
        **origin,
    }
    write_json_atomic(output / "profile.json", clean(record))
    return record


def overfit_check(device: str, work: Path = WORK, output: Path = CANDIDATE_DIR) -> dict[str, Any]:
    """
    Run ``overfit`` and write ``overfit_check.json``.

    Parameters
    ----------
    device : str
        Torch device.
    work : Path
        Local work directory.
    output : Path
        Candidate directory.

    Returns
    -------
    dict[str, Any]
        The check record.
    """
    origin = provenance()
    started = time.time()
    records, signals, labels = load_inputs(work)
    record = overfit(records, signals, labels, device) | {"seconds": time.time() - started, **origin}
    write_json_atomic(output / "overfit_check.json", clean(record))
    return record


def write_receipt(directory: Path, runs: list[str], work: Path = WORK) -> dict[str, Any]:
    """
    Write ``receipt.json``: input hashes, each run's commit, code digests, environment, timings and memory.

    Parameters
    ----------
    directory : Path
        Published directory.
    runs : list[str]
        Local run names.
    work : Path
        Local work directory.

    Returns
    -------
    dict[str, Any]
        The receipt.
    """
    tables = ("held_out", "chosen")
    receipt = {
        "inputs": read_json(work / "inputs" / "prepare.json"),
        "runs": {
            name: {
                key: value
                for key, value in read_json(work / "runs" / name / "run.json").items()
                if key not in tables
            }
            for name in runs
        },
        "publication": {**provenance(), "environment": environment()},
        "test_roles": "no test-role waveform was loaded and no test-role prediction or metric computed",
    }
    write_json_atomic(directory / "receipt.json", clean(receipt))
    return receipt


def check_main_files(records: pd.DataFrame, signals: np.ndarray, device: str, output: Path) -> dict[str, Any]:
    """
    Reload each published main checkpoint and recompute its validation predictions from the prepared inputs.

    Parameters
    ----------
    records, signals
        Output of ``load_inputs``.
    device : str
        Torch device.
    output : Path
        Candidate directory.

    Returns
    -------
    dict[str, Any]
        Per seed: SHA-256 match and largest absolute difference to its prediction file; and the largest
        difference between the mean file and the mean of the seed files.
    """
    settings = read_json(output / "settings.json")
    mask = masks(records, None)["validation"]
    keys = records["record_key"].to_numpy()[mask]
    signals = standardize(np.asarray(signals[mask]), settings["normalization"])
    seeds, saved = [], []
    for entry in settings["seeds"]:
        path = output / entry["checkpoint"]
        stored = load_predictions(
            prediction_path(output.parent, MODEL_ID, "validation", (entry["seed"],)), keys
        )
        recomputed = window_probabilities(load_network(path, device), signals, device)
        saved.append(stored["probability"])
        seeds.append(
            {
                "seed": entry["seed"],
                "checkpoint": entry["checkpoint"],
                "sha256_matches": sha256_file(path) == entry["sha256"],
                "max_abs_difference": float(np.abs(recomputed - stored["probability"]).max()),
            }
        )
    mean = load_predictions(prediction_path(output.parent, MODEL_ID, "validation", SEEDS), keys)[
        "probability"
    ]
    return {
        "records": len(keys),
        "seeds": seeds,
        "mean_file_vs_seed_mean": float(np.abs(mean - np.mean(saved, axis=0)).max()),
    }


def check_lofo_files(records: pd.DataFrame, signals: np.ndarray, device: str, lofo: Path) -> dict[str, Any]:
    """
    Reload each published leave-one-family-out checkpoint and recompute its held-out predictions.

    Parameters
    ----------
    records, signals
        Output of ``load_inputs``.
    device : str
        Torch device.
    lofo : Path
        ``LOFO_DIR``.

    Returns
    -------
    dict[str, Any]
        Per family: records, SHA-256 match and largest absolute difference to the held-out prediction file.
    """
    summary = read_json(lofo / "lofo.json")
    families = {}
    for family, row in summary["families"].items():
        held = masks(records, family)["held_out"]
        path = lofo / family / f"model_epoch-{row['chosen_epoch']:03d}.pt"
        stored = load_predictions(
            lofo / family / f"held_out__seeds-{summary['seed']}.npz", records["record_key"].to_numpy()[held]
        )
        recomputed = window_probabilities(
            load_network(path, device), standardize(np.asarray(signals[held]), row["normalization"]), device
        )
        families[family] = {
            "records": int(held.sum()),
            "sha256_matches": sha256_file(path) == row["checkpoint_sha256"],
            "max_abs_difference": float(np.abs(recomputed - stored["probability"]).max()),
        }
    return families


def check_files(
    device: str, work: Path = WORK, output: Path = CANDIDATE_DIR, lofo: Path = LOFO_DIR
) -> dict[str, Any]:
    """
    Check that every published checkpoint and prediction file loads and reproduces its predictions.

    Parameters
    ----------
    device : str
        Torch device.
    work : Path
        Local work directory.
    output : Path
        Candidate directory.
    lofo : Path
        ``LOFO_DIR``.

    Returns
    -------
    dict[str, Any]
        The check, written to ``files_check.json`` in both directories, with the memory error counters.

    Raises
    ------
    ValueError
        If a digest differs or a recomputed prediction differs by more than ``TOLERANCE``.
    """
    origin = provenance()
    started = time.time()
    records, signals, _ = load_inputs(work)
    record = {
        "main": check_main_files(records, signals, device, output),
        "lofo": check_lofo_files(records, signals, device, lofo),
        "tolerance": TOLERANCE,
        "edac": edac_counts(),
        "seconds": time.time() - started,
        **origin,
    }
    write_json_atomic(output / "files_check.json", clean(record))
    write_json_atomic(lofo / "files_check.json", clean(record))
    checks = [*record["main"]["seeds"], *record["lofo"].values()]
    if not all(item["sha256_matches"] and item["max_abs_difference"] <= TOLERANCE for item in checks):
        raise ValueError("a published file does not reproduce its predictions")
    if record["main"]["mean_file_vs_seed_mean"] > TOLERANCE:
        raise ValueError("the mean prediction file is not the mean of the seed files")
    return record
