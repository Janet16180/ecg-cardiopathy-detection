"""Frozen 40k CPC waveform pool, its PTB-XL label manifests, and resumable fine-tuning."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from . import ROOT
from .cpc import LEADS, SIGNAL_SAMPLES, CPCClassifier
from .evaluation import evaluate_predictions, partition_validation
from .files import read_csv, sha256_file, sha256_json, write_json_atomic, write_torch_atomic
from .receipts import check_completed_stage
from .reproducibility import capture_rng_state, cpu_state, restore_rng_state, seed_everything
from .training import checked_step

DEFAULT_CACHE = ROOT / "data/processed/cpc_pool_40k"
DEFAULT_MANIFEST = ROOT / "data/processed/ptbxl"
SSL_SEED = 42
NORMALIZATION_CHUNK_RECORDS = 128
MIN_STD = 1e-6
ENCODER_LR = 3e-4
HEAD_LR = 1e-3
WEIGHT_DECAY = 0.01
LABEL_MANIFESTS = ("all_train_ssl", "labeled_train", "validation", "test")
COMPLETED_ARTIFACTS = ("config.json", "history.json", "model.pt", "metrics.json",
                       "test_predictions.csv", "calibration_predictions.npz")
# Every file whose code determines CPC pretraining or fine-tuning results.
CODE_FILES = ("ecg_experiment/cpc.py", "ecg_experiment/cpc_pool.py", "ecg_experiment/evaluation.py",
              "ecg_experiment/files.py", "ecg_experiment/reproducibility.py", "ecg_experiment/receipts.py",
              "ecg_experiment/training.py", "scripts/experiments/run_cpc_experiment.py")


class Pool:
    """
    Memory-mapped ``[N, 12, 2500]`` float32 waveform cache with aligned rows.

    Parameters
    ----------
    directory : str | Path
        Completed cache holding ``complete.json``, ``signals.npy``,
        ``ecg_ids.npy`` and ``rows.csv``.

    Raises
    ------
    FileNotFoundError
        If the cache has no completion marker.
    ValueError
        If the arrays and rows are malformed, misaligned or duplicated.
    """

    def __init__(self, directory: str | Path) -> None:
        directory = Path(directory)
        self.directory = directory
        if not (directory / "complete.json").exists():
            raise FileNotFoundError(f"CPC cache is incomplete: {directory}")
        self.metadata = json.loads((directory / "complete.json").read_text())
        self.signals = np.load(directory / "signals.npy", mmap_mode="r")
        self.ids = np.load(directory / "ecg_ids.npy", allow_pickle=False)
        self.rows = read_csv(directory / "rows.csv")
        if (self.signals.dtype != np.float32 or self.signals.ndim != 3
                or self.signals.shape[1:] != (LEADS, SIGNAL_SAMPLES)):
            raise ValueError("Expected float32 cache [N,12,2500]")
        if len(self.ids) != len(self.rows) or len(self.ids) != len(self.signals):
            raise ValueError("Cache ID, row, and signal counts differ")
        self.index = {str(ecg_id): i for i, ecg_id in enumerate(self.ids)}
        if len(self.index) != len(self.ids):
            raise ValueError("Duplicate cache ECG IDs")
        if any(str(self.ids[i]) != row["ecg_id"] for i, row in enumerate(self.rows)):
            raise ValueError("Cache rows are not aligned to signals")
        if any(row["split"] not in ("train", "validation", "test") for row in self.rows):
            raise ValueError("Unknown cache split")
        self.train_rows = [row for row in self.rows if row["split"] == "train"]

    def indices(self, rows: list[dict[str, str]]) -> list[int]:
        """
        Map manifest rows to signal indices.

        Parameters
        ----------
        rows : list[dict[str, str]]
            Rows with an ``ecg_id`` present in the cache.

        Returns
        -------
        list[int]
            Positions in ``signals``, in row order.
        """
        return [self.index[row["ecg_id"]] for row in rows]

    def normalization(self, output_dir: str | Path,
                      source_hashes: dict[str, str] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Global per-lead mV mean/std over training records and every sample.

        The result is saved to ``normalization.json`` and reused when its
        training inputs match.

        Parameters
        ----------
        output_dir : str | Path
            Directory holding ``normalization.json``.
        source_hashes : dict[str, str] | None
            Precomputed digests keyed by resolved path, to avoid rehashing
            ``signals.npy``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Float32 per-lead mean and standard deviation.

        Raises
        ------
        ValueError
            If a saved normalization used different training data.
        """
        path = Path(output_dir) / "normalization.json"
        signal_path = str((self.directory / "signals.npy").resolve())
        signal_hash = source_hashes[signal_path] if source_hashes is not None else sha256_file(signal_path)
        source = {"train_ids_sha256": sha256_json([r["ecg_id"] for r in self.train_rows]),
                  "rows_sha256": sha256_file(self.directory / "rows.csv"),
                  "signals_sha256": signal_hash,
                  "method": "global per-lead mean and population std, training waveforms only"}
        if path.exists():
            info = json.loads(path.read_text())
            if info["source"] != source:
                raise ValueError("Existing normalization uses different training data")
            return np.asarray(info["mean"], dtype=np.float32), np.asarray(info["std"], dtype=np.float32)
        totals = np.zeros(LEADS, dtype=np.float64)
        squares = np.zeros(LEADS, dtype=np.float64)
        indices = self.indices(self.train_rows)
        # Fixed chunking keeps the float64 accumulation order, and so the statistics, reproducible.
        for start in range(0, len(indices), NORMALIZATION_CHUNK_RECORDS):
            block = np.asarray(self.signals[indices[start:start + NORMALIZATION_CHUNK_RECORDS]],
                               dtype=np.float64)
            totals += block.sum(axis=(0, 2))
            squares += np.square(block).sum(axis=(0, 2))
        count = len(indices) * SIGNAL_SAMPLES
        mean = totals / count
        std = np.maximum(np.sqrt(np.maximum(squares / count - mean ** 2, 0)), MIN_STD)
        write_json_atomic(path, {"source": source, "count": count,
                                 "mean": mean.tolist(), "std": std.tolist()})
        return mean.astype(np.float32), std.astype(np.float32)


class PoolDataset(Dataset):
    """
    Normalized pool waveforms with their targets and patient IDs.

    Parameters
    ----------
    pool : Pool
        Source cache.
    rows : list[dict[str, str]]
        Rows to serve; ``target`` defaults to -1 when absent.
    mean : np.ndarray
        Per-lead mean from ``Pool.normalization``.
    std : np.ndarray
        Per-lead standard deviation from ``Pool.normalization``.
    """

    def __init__(self, pool: Pool, rows: list[dict[str, str]], mean: np.ndarray, std: np.ndarray) -> None:
        self.pool, self.rows = pool, rows
        self.indices = pool.indices(rows)
        self.mean = mean[:, None]
        self.std = std[:, None]

    def __len__(self) -> int:
        """Return the number of records."""
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, float, str]:
        """Return the normalized signal, target and patient ID of record ``index``."""
        signal = np.array(self.pool.signals[self.indices[index]], copy=True)
        signal -= self.mean
        signal /= self.std
        row = self.rows[index]
        return torch.from_numpy(signal), float(row.get("target", -1)), row["patient_id"]


def loader(pool: Pool, rows: list[dict[str, str]], mean: np.ndarray, std: np.ndarray,
           batch_size: int, shuffle: bool, generator: torch.Generator | None, device: str) -> DataLoader:
    """
    Build a single-process loader over pool rows.

    Parameters
    ----------
    pool : Pool
        Source cache.
    rows : list[dict[str, str]]
        Rows to serve.
    mean : np.ndarray
        Per-lead mean.
    std : np.ndarray
        Per-lead standard deviation.
    batch_size : int
        Records per batch; the last batch may be smaller.
    shuffle : bool
        Shuffle each epoch with ``generator``.
    generator : torch.Generator | None
        Shuffling generator, saved with resumable checkpoints.
    device : str
        Target device; CUDA enables pinned memory.

    Returns
    -------
    DataLoader
        Loader yielding ``(signal, target, patient_id)`` batches.
    """
    return DataLoader(PoolDataset(pool, rows, mean, std), batch_size=batch_size,
                      shuffle=shuffle, generator=generator, num_workers=0,
                      pin_memory=(device == "cuda"), drop_last=False)


def manifest_rows(pool: Pool, manifest_dir: str | Path,
                  budget: str) -> tuple[dict[str, list[dict[str, str]]], dict[str, str]]:
    """
    Read one PTB-XL label budget and check it against the pool.

    Parameters
    ----------
    pool : Pool
        Cache that must hold every manifest record in the matching split.
    manifest_dir : str | Path
        Directory holding ``seed42_fraction<budget>`` manifests.
    budget : str
        Label fraction, ``"0.1"`` or ``"1"``.

    Returns
    -------
    tuple[dict[str, list[dict[str, str]]], dict[str, str]]
        Rows keyed by manifest name, and each manifest's SHA-256.

    Raises
    ------
    ValueError
        If a manifest record's source, split or patient differs from the pool.
    """
    directory = Path(manifest_dir) / f"seed42_fraction{budget}"
    files = {name: directory / f"{name}.csv" for name in LABEL_MANIFESTS}
    rows = {name: read_csv(path) for name, path in files.items()}
    for name, expected_split in (("all_train_ssl", "train"), ("labeled_train", "train"),
                                 ("validation", "validation"), ("test", "test")):
        for row in rows[name]:
            cache_row = pool.rows[pool.index[row["ecg_id"]]]
            identity = (cache_row["source"], cache_row["split"], cache_row["patient_id"])
            if identity != ("ptbxl", expected_split, row["patient_id"]):
                raise ValueError(f"PTB manifest/cache mismatch for {row['ecg_id']}")
    return rows, {name: sha256_file(path) for name, path in files.items()}


def fingerprint(source_hashes: dict[str, str], mean: np.ndarray, std: np.ndarray,
                settings: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """
    Identify a CPC stage by its data, normalization, settings and code.

    Parameters
    ----------
    source_hashes : dict[str, str]
        Input file digests from ``make_source_hashes``.
    mean : np.ndarray
        Per-lead mean.
    std : np.ndarray
        Per-lead standard deviation.
    settings : dict[str, Any]
        Stage hyperparameters.

    Returns
    -------
    tuple[str, dict[str, Any]]
        Canonical digest and the fingerprinted inputs.
    """
    data = {"cache": source_hashes, "normalization": {"mean": mean.tolist(), "std": std.tolist()},
            "settings": settings, "code": {name: sha256_file(ROOT / name) for name in CODE_FILES}}
    return sha256_json(data), data


def make_source_hashes(pool: Pool, manifest_dir: str | Path) -> dict[str, str]:
    """
    Hash the pool files and both PTB-XL label budgets, checking declared digests.

    Parameters
    ----------
    pool : Pool
        Source cache.
    manifest_dir : str | Path
        Directory holding ``seed42_fraction0.1`` and ``seed42_fraction1``.

    Returns
    -------
    dict[str, str]
        SHA-256 keyed by resolved path.

    Raises
    ------
    ValueError
        If a file differs from the digest recorded in ``complete.json``.
    """
    paths = [pool.directory / name for name in ("complete.json", "rows.csv", "ecg_ids.npy", "signals.npy")]
    paths += [Path(manifest_dir) / f"seed42_fraction{budget}" / f"{name}.csv"
              for budget in ("0.1", "1") for name in LABEL_MANIFESTS]
    hashes = {str(path.resolve()): sha256_file(path) for path in paths}
    for filename, key in (("signals.npy", "signals_sha256"), ("rows.csv", "rows_sha256"),
                          ("ecg_ids.npy", "ecg_ids_sha256")):
        if hashes[str((pool.directory / filename).resolve())] != pool.metadata[key]:
            raise ValueError(f"CPC cache hash mismatch for {filename}")
    for filename, expected in pool.metadata["ptb_manifest_sha256"].items():
        path = Path(manifest_dir) / "seed42_fraction1" / filename
        if str(path.resolve()) in hashes and hashes[str(path.resolve())] != expected:
            raise ValueError(f"PTB manifest hash mismatch for {filename}")
    return hashes


def resume_or_new(
    directory: Path, fingerprint_value: str, model: nn.Module, optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
) -> tuple[int, list[dict[str, Any]], dict[str, torch.Tensor] | None, float, int]:
    """
    Restore the last completed epoch, or start fresh.

    Parameters
    ----------
    directory : Path
        Stage directory that may hold ``epoch_state.pt``.
    fingerprint_value : str
        Digest the saved state must match.
    model : nn.Module
        Model restored in place.
    optimizer : torch.optim.Optimizer
        Optimizer restored in place.
    generator : torch.Generator
        Loader generator restored with the global random states.

    Returns
    -------
    tuple[int, list[dict[str, Any]], dict[str, torch.Tensor] | None, float, int]
        Completed epochs, history, best model state, best development AUROC
        and best epoch.

    Raises
    ------
    ValueError
        If history exists without resumable state, or the fingerprint differs.
    """
    state_path = directory / "epoch_state.pt"
    if not state_path.exists():
        if (directory / "history.json").exists():
            raise ValueError(f"History exists without resumable state in {directory}")
        return 0, [], None, -1.0, 0
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if state["fingerprint"] != fingerprint_value:
        raise ValueError(f"Resume fingerprint mismatch: {directory}")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    restore_rng_state(state["rng"], generator)
    return (state["epoch"], state["history"], state.get("best_model"), state.get("best_auc", -1.0),
            state.get("best_epoch", 0))


def save_epoch(directory: Path, fingerprint_value: str, epoch: int, model: nn.Module,
               optimizer: torch.optim.Optimizer, generator: torch.Generator,
               history: list[dict[str, Any]], best_model: dict[str, torch.Tensor] | None = None,
               best_auc: float = -1.0, best_epoch: int = 0) -> None:
    """
    Atomically save an epoch boundary and its history.

    Parameters
    ----------
    directory : Path
        Stage directory.
    fingerprint_value : str
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
    best_model : dict[str, torch.Tensor] | None
        Best model state so far.
    best_auc : float
        Best development AUROC so far.
    best_epoch : int
        Epoch of ``best_model``.
    """
    write_torch_atomic(directory / "epoch_state.pt", {"fingerprint": fingerprint_value, "epoch": epoch,
                       "model": cpu_state(model), "optimizer": optimizer.state_dict(),
                       "rng": capture_rng_state(generator), "history": history, "best_model": best_model,
                       "best_auc": best_auc, "best_epoch": best_epoch})
    write_json_atomic(directory / "history.json", history)


@torch.inference_mode()
def predict(model: nn.Module, data: DataLoader, device: str) -> np.ndarray:
    """
    Compute logits for every batch in evaluation mode.

    Parameters
    ----------
    model : nn.Module
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
    return np.concatenate([model(signal.to(device, non_blocking=True)).cpu().numpy()
                           for signal, _, _ in data])


def _classifier(args: argparse.Namespace, checkpoint: Path | None, variant: str) -> CPCClassifier:
    """Build the classifier, loading the pretrained encoder when a checkpoint is given."""
    model = CPCClassifier().to(args.device)
    if checkpoint:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["variant"] != variant or saved["epochs"] != args.ssl_epochs:
            raise ValueError("SSL checkpoint variant or duration mismatch")
        model.encoder.load_state_dict(saved["encoder"])
    return model


def _train_epoch(model: nn.Module, train_data: DataLoader, optimizer: torch.optim.Optimizer,
                 device: str) -> tuple[float, int, int]:
    """Train one supervised epoch; return summed record loss, updates and exposures."""
    total = 0.0
    updates = 0
    exposures = 0
    for signal, target, _ in train_data:
        signal = signal.to(device, non_blocking=True)
        target = target.to(device, dtype=torch.float32, non_blocking=True)
        loss = nn.functional.binary_cross_entropy_with_logits(model(signal), target)
        checked_step(loss, model, optimizer, "supervised")
        total += float(loss.detach()) * len(signal)
        updates += 1
        exposures += len(signal)
    return total, updates, exposures


def fine_tune(args: argparse.Namespace, pool: Pool, mean: np.ndarray, std: np.ndarray,
              source_hashes: dict[str, str], variant: str, budget: str) -> None:
    """
    Fine-tune a CPC classifier on one label budget, resumably, then evaluate it.

    Completed runs are verified against their recorded artifact digests
    instead of being retrained.

    Parameters
    ----------
    args : argparse.Namespace
        Runner settings: ``output_dir``, ``manifest_dir``, ``device``,
        ``epochs``, ``patience``, ``batch_size``, ``ssl_epochs`` and
        ``bootstrap``.
    pool : Pool
        Source cache.
    mean : np.ndarray
        Per-lead mean.
    std : np.ndarray
        Per-lead standard deviation.
    source_hashes : dict[str, str]
        Input file digests from ``make_source_hashes``.
    variant : str
        ``"scratch"``, or the name of the pretraining arm whose encoder is
        loaded from ``<output_dir>/<variant>_ssl/encoder.pt``.
    budget : str
        Label fraction, ``"0.1"`` or ``"1"``.

    Raises
    ------
    ValueError
        If saved state, checkpoints or completed artifacts do not match.
    RuntimeError
        If training diverges or no epoch completes.
    """
    rows, manifest_hashes = manifest_rows(pool, args.manifest_dir, budget)
    seed_everything(SSL_SEED)
    directory = args.output_dir / f"{variant}_fraction{budget}_seed42"
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / f"{variant}_ssl" / "encoder.pt" if variant != "scratch" else None
    checkpoint_hash = sha256_file(checkpoint) if checkpoint else None
    settings = {"stage": "train", "variant": variant, "budget": budget, "seed": SSL_SEED,
                "epochs": args.epochs, "patience": args.patience, "batch_size": args.batch_size,
                "encoder_lr": ENCODER_LR, "head_lr": HEAD_LR, "weight_decay": WEIGHT_DECAY,
                "augmentation": "none", "ssl_checkpoint_sha256": checkpoint_hash,
                "manifest_sha256": manifest_hashes}
    fp, inputs = fingerprint(source_hashes, mean, std, settings)
    completion = directory / "completion.json"
    if completion.exists():
        check_completed_stage(directory, fp)
        return
    model = _classifier(args, checkpoint, variant)
    optimizer = torch.optim.AdamW([{"params": model.encoder.parameters(), "lr": ENCODER_LR},
                                   {"params": model.head.parameters(), "lr": HEAD_LR}],
                                  weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(SSL_SEED)
    development, calibration = partition_validation(rows["validation"])
    train_data = loader(pool, rows["labeled_train"], mean, std, args.batch_size, True, generator, args.device)
    dev_data = loader(pool, development, mean, std, args.batch_size, False, None, args.device)
    start_epoch, history, best_model, best_auc, best_epoch = resume_or_new(
        directory, fp, model, optimizer, generator)
    write_json_atomic(directory / "config.json", {
        "fingerprint": fp, "inputs": inputs,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "encoder_parameters": sum(p.numel() for p in model.encoder.parameters()),
        "labeled_training_records": len(rows["labeled_train"]),
        "development_records": len(development), "calibration_records": len(calibration),
        "test_records": len(rows["test"]),
        "description": "Mean/max context pooling over each half, then average halves"})
    dev_y = np.array([int(row["target"]) for row in development])
    started = time.monotonic()
    previous_elapsed = history[-1]["elapsed_seconds"] if history else 0.0
    for epoch in range(start_epoch, args.epochs):
        if epoch - best_epoch >= args.patience:
            break
        model.train()
        total, updates, exposures = _train_epoch(model, train_data, optimizer, args.device)
        dev_auc = float(roc_auc_score(dev_y, predict(model, dev_data, args.device)))
        if dev_auc > best_auc:
            best_auc, best_epoch, best_model = dev_auc, epoch + 1, cpu_state(model)
        record = {"epoch": epoch + 1, "train_loss": total / len(rows["labeled_train"]),
                  "development_auroc": dev_auc, "best_epoch": best_epoch,
                  "optimizer_updates": updates, "record_exposures": exposures,
                  "elapsed_seconds": previous_elapsed + time.monotonic() - started}
        history.append(record)
        save_epoch(directory, fp, epoch + 1, model, optimizer, generator, history,
                   best_model, best_auc, best_epoch)
        print(json.dumps({"stage": "train", "variant": variant, "budget": budget, **record}), flush=True)
    if best_model is None:
        raise RuntimeError("No supervised epoch completed")
    model.load_state_dict(best_model)
    write_torch_atomic(directory / "model.pt", {"fingerprint": fp, "model": best_model,
                       "best_epoch": best_epoch, "best_development_auroc": best_auc})
    calibration_data = loader(pool, calibration, mean, std, args.batch_size, False, None, args.device)
    calibration_logits = predict(model, calibration_data, args.device)
    test_data = loader(pool, rows["test"], mean, std, args.batch_size, False, None, args.device)
    test_logits = predict(model, test_data, args.device)
    evaluate_predictions(f"{variant}_fraction{budget}", calibration_logits, test_logits,
                         calibration, rows["test"], directory, SSL_SEED, args.bootstrap)
    write_json_atomic(completion, {"fingerprint": fp,
                                   "artifacts": {name: sha256_file(directory / name)
                                                 for name in COMPLETED_ARTIFACTS}})
