#!/usr/bin/env python3
"""Continue ECG-FM with a bounded patient-aware temporal contrastive objective.

This CMSC-style adaptation is not a reproduction of ECG-FM's full WCR objective.
Only official PTB-XL folds 1--8 enter optimization; diagnostic annotations are
never returned by the streaming dataset. The final fixed-epoch state is used
unless --max-updates requests an exact optimizer-update budget.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812 - conventional PyTorch alias
from torch.utils.data import DataLoader, Dataset

from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.foundation_models import load_model, preprocess_ecg_fm, preprocessing_source_sha256
from ecg_experiment.provenance import git_head
from ecg_experiment.reproducibility import cpu_state, seed_everything
from ecg_experiment.training import peak_gpu_bytes, require_cuda
from ecg_experiment.waveforms import read_record

ROOT = Path(__file__).resolve().parents[2]
LAST_TRAINING_FOLD = 8
SSL_FIELDS = {"ecg_id", "patient_id", "filename_lr", "filename_hr"}
EXTERNAL_FIELDS = {"ecg_id", "patient_id", "raw_dir", "filename_hr", "source"}
WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 1.0
PROGRESS_INTERVAL = 100

Rows = list[dict[str, str]]


def training_rows(manifest_dir: Path, raw_dir: Path) -> tuple[Rows, dict[str, str]]:
    """
    Check exact official training membership and patient isolation without labels.

    Parameters
    ----------
    manifest_dir : Path
        Directory holding ``all_train_ssl.csv``.
    raw_dir : Path
        PTB-XL directory holding ``ptbxl_database.csv``.

    Returns
    -------
    tuple[list[dict[str, str]], dict[str, str]]
        SSL rows and the digests of the manifest and official metadata.

    Raises
    ------
    ValueError
        If the manifest has label columns, duplicates, other records than
        folds 1--8, other identities than the official metadata, or held-out
        patients.
    """
    path = manifest_dir / "all_train_ssl.csv"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or []) != SSL_FIELDS:
            raise ValueError("SSL manifest must contain waveform identifiers only")
        rows = list(reader)
    ids = [row["ecg_id"] for row in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError("Empty or duplicate SSL ECG identifiers")
    metadata_path = raw_dir / "ptbxl_database.csv"
    with metadata_path.open(newline="") as handle:
        # Retain identity/fold fields only; SCP statements/reports are not inspected.
        official = {r["ecg_id"]: {k: r[k] for k in ("patient_id", "strat_fold", "filename_hr")}
                    for r in csv.DictReader(handle)}
    expected_ids = {key for key, row in official.items() if 1 <= int(row["strat_fold"]) <= LAST_TRAINING_FOLD}
    if set(ids) != expected_ids:
        raise ValueError("SSL manifest must contain exactly all official fold 1--8 ECGs")
    for row in rows:
        source = official[row["ecg_id"]]
        if any(row[key] != source[key] for key in ("patient_id", "filename_hr")):
            raise ValueError("SSL patient identity or waveform differs from official metadata")
    heldout_patients = {r["patient_id"] for r in official.values()
                        if int(r["strat_fold"]) > LAST_TRAINING_FOLD}
    if heldout_patients & {r["patient_id"] for r in rows}:
        raise ValueError("Held-out patient present in SSL data")
    hashes = {path.name: sha256_file(path), "ptbxl_database.csv": sha256_file(metadata_path)}
    return rows, hashes


def _check_external_row(row: dict[str, str], seen: set[str]) -> None:
    """Require a complete, namespaced, unique external row whose waveform stays in its root."""
    if not all(row.values()):
        raise ValueError("Missing external waveform identity or path")
    prefix = row["source"] + ":"
    if not row["ecg_id"].startswith(prefix) or not row["patient_id"].startswith(prefix):
        raise ValueError("External ECG and grouping identifiers must use a source: namespace")
    if row["ecg_id"] in seen:
        raise ValueError("Duplicate external ECG identifier")
    root = Path(row["raw_dir"]).resolve()
    waveform = (root / row["filename_hr"]).resolve()
    if not waveform.is_relative_to(root) or not waveform.with_suffix(".hea").is_file():
        raise ValueError("External waveform is absent or escapes its dataset root")


def extra_training_rows(path: Path | None, existing_ids: Iterable[str]) -> Rows:
    """
    Read an explicitly unlabeled, namespaced external waveform manifest.

    Parameters
    ----------
    path : Path | None
        External manifest, or None for PTB-XL only.
    existing_ids : Iterable[str]
        PTB-XL ECG identifiers already in the pool.

    Returns
    -------
    list[dict[str, str]]
        External rows with ``raw_dir`` resolved; empty when ``path`` is None.

    Raises
    ------
    ValueError
        If the manifest has other columns, an invalid row, or no rows.
    """
    if path is None:
        return []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or []) != EXTERNAL_FIELDS:
            raise ValueError("External SSL manifest must contain only " + ", ".join(sorted(EXTERNAL_FIELDS)))
        rows = list(reader)
    seen = set(existing_ids)
    for row in rows:
        _check_external_row(row, seen)
        seen.add(row["ecg_id"])
        row["raw_dir"] = str(Path(row["raw_dir"]).resolve())
    if not rows:
        raise ValueError("External manifest is empty")
    return rows


class TrainingWaveforms(Dataset):
    """
    Unlabeled ECG-FM views with an integer patient index for positive pairs.

    Parameters
    ----------
    rows : list[dict[str, str]]
        SSL rows; external rows carry their own ``raw_dir``.
    raw_dir : Path
        PTB-XL waveform directory.
    """

    def __init__(self, rows: Rows, raw_dir: Path) -> None:
        self.rows, self.raw_dir = rows, raw_dir
        patients = {value: i for i, value in enumerate(sorted({r["patient_id"] for r in rows}))}
        self.patient_index = [patients[r["patient_id"]] for r in rows]

    def __len__(self) -> int:
        """Return the number of records."""
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """
        Return the two preprocessed views of one ECG and its patient index.

        Parameters
        ----------
        index : int
            Record position.

        Returns
        -------
        tuple[torch.Tensor, int]
            ``[2, 12, 2500]`` views and the patient index.
        """
        row = self.rows[index]
        waveform = read_record(Path(row.get("raw_dir", self.raw_dir)), row["filename_hr"])
        return torch.from_numpy(preprocess_ecg_fm(waveform)), self.patient_index[index]


def temporal_contrastive_loss(features: torch.Tensor, patient_ids: torch.Tensor,
                              temperature: float = 0.1) -> torch.Tensor:
    """
    Symmetric cross-view SupCon; all same-patient cross-view pairs are positive.

    Each half-record anchors against every opposite half-record in the batch.
    Repeated ECGs from one patient are never treated as negative pairs. We average
    log probabilities over a patient's positives, then over both anchor directions.

    Parameters
    ----------
    features : torch.Tensor
        ``[batch, 2, dimension]`` view features.
    patient_ids : torch.Tensor
        ``[batch]`` patient indices.
    temperature : float
        Softmax temperature.

    Returns
    -------
    torch.Tensor
        Scalar loss.

    Raises
    ------
    ValueError
        If the features, temperature, or patient identifiers are invalid.
    """
    if features.ndim != 3 or features.shape[1] != 2 or len(features) < 2:
        raise ValueError("Need at least two ECGs, each with two feature vectors")
    if not temperature > 0 or len(patient_ids) != len(features):
        raise ValueError("Invalid temperature or patient identifiers")
    embedding = F.normalize(features.float(), dim=-1)
    similarities = embedding[:, 0] @ embedding[:, 1].T / temperature
    positives = patient_ids[:, None].eq(patient_ids[None, :]).to(similarities.dtype)
    forward = -(F.log_softmax(similarities, dim=1) * positives).sum(1) / positives.sum(1)
    backward = -(F.log_softmax(similarities.T, dim=1) * positives.T).sum(1) / positives.T.sum(1)
    return (forward.mean() + backward.mean()) / 2


def update_budget(records: int, batch_size: int, epochs: int, max_updates: int | None) -> tuple[int, bool]:
    """
    Return steps per epoch and whether the final batch must be omitted.

    Bounded runs use full batches so every update sees the same number of ECGs.
    Legacy epoch runs retain their original partial final batch behavior.

    Parameters
    ----------
    records : int
        Training records.
    batch_size : int
        Records per optimizer update.
    epochs : int
        Epoch ceiling.
    max_updates : int | None
        Exact optimizer-update budget, or None for full epochs.

    Returns
    -------
    tuple[int, bool]
        Optimizer updates per epoch and whether to drop the last batch.

    Raises
    ------
    ValueError
        If the bounded budget is not positive or cannot be met with full batches.
    """
    if max_updates is None:
        return math.ceil(records / batch_size), False
    if max_updates < 1:
        raise ValueError("--max-updates must be positive")
    steps_per_epoch = records // batch_size
    if steps_per_epoch < 1:
        raise ValueError("--max-updates requires at least one full batch per epoch")
    if max_updates > epochs * steps_per_epoch:
        raise ValueError("--epochs cannot supply --max-updates full-batch optimizer updates")
    return steps_per_epoch, True


def build_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser of the adaptation arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-metadata", type=Path, required=True,
                        help="Existing extraction metadata identifying the official checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-updates", type=int,
                        help="Stop after exactly this many optimizer updates, even within an epoch; "
                             "uses full batches only and resumes from completed epochs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--extra-ssl-manifest", type=Path,
                        help="Unlabeled external CSV: ecg_id,patient_id,raw_dir,filename_hr,source")
    parser.add_argument("--extra-grouping-description",
                        default="Source-specific grouping; patient identity not independently verified")
    parser.add_argument("--protocol-note", default="")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse the command line, rejecting invalid sizes and rates and CUDA requests without CUDA.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or ``None`` for ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Validated arguments.

    Raises
    ------
    RuntimeError
        If CUDA is requested but unavailable.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.batch_size < 2 or args.workers < 0:
        parser.error("Epochs/threads must be positive; batch >=2 and workers >=0")
    rates = (args.learning_rate, args.temperature)
    if not all(np.isfinite(rate) and rate > 0 for rate in rates):
        parser.error("Learning rate and temperature must be finite and positive")
    require_cuda(args.device)
    return args


def adaptation_config(args: argparse.Namespace, ptbxl_rows: Rows, extra_rows: Rows, hashes: dict[str, str],
                      original_meta: dict[str, Any], steps_per_epoch: int) -> dict[str, Any]:
    """
    Describe the adaptation inputs and protocol recorded in ``config.json``.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    ptbxl_rows : list[dict[str, str]]
        PTB-XL SSL rows.
    extra_rows : list[dict[str, str]]
        External SSL rows.
    hashes : dict[str, str]
        PTB-XL manifest digests.
    original_meta : dict[str, Any]
        Official checkpoint metadata.
    steps_per_epoch : int
        Optimizer updates per epoch.

    Returns
    -------
    dict[str, Any]
        Configuration that a resumed run must match exactly.
    """
    rows = ptbxl_rows + extra_rows
    config = {
        "model": "ecg-fm",
        "objective": "CMSC-style symmetric cross-view patient-aware contrastive adaptation; not full WCR",
        "official_checkpoint": original_meta, "manifest_sha256": hashes,
        "training_ecg_ids": [r["ecg_id"] for r in ptbxl_rows],
        "training_patient_ids": sorted({r["patient_id"] for r in rows}),
        "training_records": len(rows), "heldout_patient_overlap": None if extra_rows else 0,
        "ptbxl_heldout_patient_overlap": 0,
        "ptbxl_training_records": len(ptbxl_rows),
        "external_ssl": ({"manifest_sha256": sha256_file(args.extra_ssl_manifest),
                          "records": len(extra_rows),
                          "source_counts": dict(Counter(r["source"] for r in extra_rows)),
                          "grouping_description": args.extra_grouping_description,
                          "waveforms": extra_rows,
                          "limitation": "PTB-XL patient isolation is verified; cross-source patient overlap "
                                        "is not independently verifiable"}
                         if extra_rows else None),
        "preprocessing": "Official 500 Hz 12-lead per-lead z-score over 10 seconds; two nonoverlapping "
                         "5-second views",
        "representation": "Mean final-layer tokens per view, L2 normalized for loss; no projection head",
        "positive_pairs": "All same-patient pairs across the two temporal views; "
                          "averaged positive log probabilities",
        "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "temperature": args.temperature, "weight_decay": WEIGHT_DECAY, "gradient_clip": GRADIENT_CLIP,
        "seed": args.seed, "workers": args.workers, "threads": args.threads,
        "checkpoint_selection": "Final state after fixed epoch budget; no validation/test selection",
        "protocol_note": args.protocol_note,
        "source_sha256": sha256_file(Path(__file__)),
        "training_source_sha256": sha256_file(ROOT / "ecg_experiment/training.py"),
        "extractor_source_sha256": preprocessing_source_sha256(),
        "official_source_commit": git_head(ROOT / "third_party" / "fairseq-signals"),
    }
    if args.max_updates is not None:
        config.update({"max_updates": args.max_updates, "updates_per_epoch": steps_per_epoch,
                       "drop_last": True,
                       "checkpoint_selection": "Final state after exact optimizer-update budget; "
                                               "no validation/test selection"})
    return config


def prepare_output(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """
    Create the output directory, refusing to overwrite or resume inconsistently.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    config : dict[str, Any]
        Configuration of this run.

    Raises
    ------
    FileExistsError
        If the adaptation completed, or a fresh run targets a nonempty directory.
    ValueError
        If ``--resume`` targets another configuration.
    FileNotFoundError
        If ``--resume`` is given without a completed-epoch checkpoint.
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "adapted_backbone.pt").exists():
        raise FileExistsError("Completed adaptation exists")
    if any(args.output_dir.iterdir()) and not args.resume:
        raise FileExistsError("Output directory is not empty; use --resume for identical settings")
    if args.resume and (json.loads((args.output_dir / "config.json").read_text()) != config):
        raise ValueError("Resume configuration differs")
    if args.resume and not (args.output_dir / "resume.pt").is_file():
        raise FileNotFoundError("No completed-epoch resume checkpoint exists; "
                                "restart this run in a clean output directory")


def save_resume(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                history: list[dict[str, Any]], generator: torch.Generator, device: str) -> None:
    """
    Atomically save a completed-epoch resume checkpoint.

    Parameters
    ----------
    path : Path
        Destination ``resume.pt``.
    model : nn.Module
        Backbone being adapted.
    optimizer : torch.optim.Optimizer
        Its optimizer.
    history : list[dict[str, Any]]
        One row per finished epoch.
    generator : torch.Generator
        Loader shuffle generator.
    device : str
        Torch device; CUDA random states are stored only for CUDA runs.
    """
    # The temporary name fixes torch's internal archive name, so the file bytes match earlier runs.
    temporary = path.with_name("resume.partial.pt")
    torch.save({"backbone": cpu_state(model),
                "optimizer": optimizer.state_dict(), "history": history,
                "loader_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else [],
                "python_rng": random.getstate()}, temporary)
    os.replace(temporary, path)


def load_resume(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, generator: torch.Generator,
                args: argparse.Namespace) -> list[dict[str, Any]]:
    """
    Restore a completed-epoch resume checkpoint.

    Parameters
    ----------
    path : Path
        Checkpoint written by ``save_resume``.
    model : nn.Module
        Backbone to restore in place.
    optimizer : torch.optim.Optimizer
        Optimizer to restore in place.
    generator : torch.Generator
        Loader generator to restore in place.
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    list[dict[str, Any]]
        History of the finished epochs.

    Raises
    ------
    ValueError
        If a bounded run would resume after a partial final epoch.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["backbone"])
    optimizer.load_state_dict(state["optimizer"])
    history = state["history"]
    if args.max_updates is not None and any(not item.get("complete_epoch", True) for item in history):
        raise ValueError("Cannot resume after a partial final epoch")
    generator.set_state(state["loader_rng"])
    torch.set_rng_state(state["torch_rng"])
    if args.device == "cuda":
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    random.setstate(state["python_rng"])
    return history


def train_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, loader: DataLoader,
                args: argparse.Namespace, epoch: int, progress: dict[str, int] | None,
                elapsed_before: float, started: float) -> tuple[float, int, float]:
    """
    Run one contrastive epoch, stopping early when a bounded run meets its budget.

    Parameters
    ----------
    model : nn.Module
        Backbone to adapt in place.
    optimizer : torch.optim.Optimizer
        Its optimizer.
    loader : DataLoader
        Shuffled training loader.
    args : argparse.Namespace
        Parsed command-line arguments.
    epoch : int
        Zero-based epoch.
    progress : dict[str, int] | None
        Running ``updates`` and ``seen_examples`` of a bounded run, updated
        in place; None for fixed-epoch runs.
    elapsed_before : float
        Seconds of earlier invocations.
    started : float
        Monotonic start time of this invocation.

    Returns
    -------
    tuple[float, int, float]
        Loss summed over records, records seen, and the largest gradient
        norm before clipping.

    Raises
    ------
    ValueError
        If a batch holds a single ECG.
    RuntimeError
        If the loss or gradient norm is nonfinite.
    """
    model.train()
    total_loss, observations, maximum_grad = 0.0, 0, 0.0
    for step, (views, patients) in enumerate(loader):
        if len(views) < 2:
            raise ValueError("Singleton final contrastive batch; choose another batch size")
        views = views.to(args.device, non_blocking=True)
        patients = patients.to(args.device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        tokens = model.extract_features(source=views.flatten(0, 1), padding_mask=None, mask=False)["x"]
        features = tokens.mean(dim=1).reshape(len(views), 2, -1)
        loss = temporal_contrastive_loss(features, patients, args.temperature)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite contrastive loss")
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP, error_if_nonfinite=True)
        maximum_grad = max(maximum_grad, float(gradient_norm))
        optimizer.step()
        if progress is not None:
            progress["updates"] += 1
            progress["seen_examples"] += len(views)
        total_loss += float(loss.detach()) * len(views)
        observations += len(views)
        if (step + 1) % PROGRESS_INTERVAL == 0:
            print(json.dumps({"epoch": epoch + 1, "records": observations,
                              "running_loss": total_loss / observations, **(progress or {}),
                              "seconds": elapsed_before + time.monotonic() - started}), flush=True)
        if progress is not None and progress["updates"] == args.max_updates:
            break
    return total_loss, observations, maximum_grad


def adapt(args: argparse.Namespace, model: nn.Module, optimizer: torch.optim.Optimizer, loader: DataLoader,
          generator: torch.Generator, history: list[dict[str, Any]], steps_per_epoch: int,
          drop_last: bool, records: int, elapsed_before: float, started: float) -> dict[str, int] | None:
    """
    Train the remaining epochs, saving history and completed-epoch resume checkpoints.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    model : nn.Module
        Backbone to adapt in place.
    optimizer : torch.optim.Optimizer
        Its optimizer.
    loader : DataLoader
        Shuffled training loader.
    generator : torch.Generator
        Loader shuffle generator.
    history : list[dict[str, Any]]
        Finished epochs, extended in place.
    steps_per_epoch : int
        Optimizer updates per full epoch.
    drop_last : bool
        Whether the loader omits partial batches.
    records : int
        Training records.
    elapsed_before : float
        Seconds of earlier invocations.
    started : float
        Monotonic start time of this invocation.

    Returns
    -------
    dict[str, int] | None
        Final ``updates`` and ``seen_examples`` of a bounded run; None for
        fixed-epoch runs.

    Raises
    ------
    RuntimeError
        If a fixed-epoch run finishes an incomplete epoch or no gradient
        update happened.
    """
    progress = None
    if args.max_updates is not None:
        progress = {"updates": len(history) * steps_per_epoch,
                    "seen_examples": sum(item["records"] for item in history)}
    for epoch in range(len(history), args.epochs):
        total_loss, observations, maximum_grad = train_epoch(model, optimizer, loader, args, epoch, progress,
                                                             elapsed_before, started)
        complete_epoch = observations == (steps_per_epoch * args.batch_size if drop_last else records)
        if progress is None and not complete_epoch:
            raise RuntimeError("Incomplete epoch or no gradient updates")
        if maximum_grad <= 0:
            raise RuntimeError("Incomplete epoch or no gradient updates")
        entry = {"epoch": epoch + 1, "records": observations, "loss": total_loss / observations,
                 "maximum_gradient_norm_before_clipping": maximum_grad,
                 "seconds": elapsed_before + time.monotonic() - started}
        if progress is not None:
            entry.update({**progress, "complete_epoch": complete_epoch})
        history.append(entry)
        write_json_atomic(args.output_dir / "history.json", history)
        print(json.dumps({"stage": "ecg-fm_cmsc_adaptation", **entry}), flush=True)
        if complete_epoch:
            save_resume(args.output_dir / "resume.pt", model, optimizer, history, generator, args.device)
        if progress is not None and progress["updates"] == args.max_updates:
            break
    return progress


def save_adapted(args: argparse.Namespace, model: nn.Module, config: dict[str, Any],
                 history: list[dict[str, Any]], progress: dict[str, int] | None, seconds: float) -> Path:
    """
    Save the final portable backbone and its completion receipt.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    model : nn.Module
        Adapted backbone.
    config : dict[str, Any]
        Run configuration stored with the backbone.
    history : list[dict[str, Any]]
        One row per finished epoch.
    progress : dict[str, int] | None
        Final counters of a bounded run.
    seconds : float
        Total adaptation seconds.

    Returns
    -------
    Path
        The saved ``adapted_backbone.pt``.

    Raises
    ------
    RuntimeError
        If a floating-point weight is nonfinite.
    """
    backbone = cpu_state(model)
    if not all(torch.isfinite(value).all() for value in backbone.values() if value.is_floating_point()):
        raise RuntimeError("Nonfinite adapted checkpoint")
    path = args.output_dir / "adapted_backbone.pt"
    # Plain torch.save keeps the archive name, and so the file bytes, of earlier runs.
    torch.save({"backbone": backbone, "metadata": config, "history": history}, path)
    completion = {"checkpoint": path.name, "sha256": sha256_file(path),
                  "completed_epochs": len(history), "seconds": seconds,
                  "torch_version": str(torch.__version__),
                  "device": torch.cuda.get_device_name() if args.device == "cuda" else "cpu",
                  "peak_cuda_memory_bytes": peak_gpu_bytes(args.device)}
    if progress is not None:
        completion.update({**progress, "completed_epochs": sum(item["complete_epoch"] for item in history)})
    write_json_atomic(args.output_dir / "completion.json", completion)
    return path


def main(argv: list[str] | None = None) -> None:
    """
    Adapt ECG-FM on training-only ECGs and save the final backbone.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or ``None`` for ``sys.argv``.
    """
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    ptbxl_rows, hashes = training_rows(args.manifest_dir, args.raw_dir)
    extra_rows = extra_training_rows(args.extra_ssl_manifest, [r["ecg_id"] for r in ptbxl_rows])
    rows = ptbxl_rows + extra_rows
    try:
        steps_per_epoch, drop_last = update_budget(len(rows), args.batch_size, args.epochs, args.max_updates)
    except ValueError as exc:
        build_parser().error(str(exc))
    original_meta = json.loads(args.checkpoint_metadata.read_text())["checkpoint"]
    if original_meta["sha256"] != sha256_file(args.checkpoint):
        raise ValueError("Official checkpoint differs from recorded extraction metadata")
    config = adaptation_config(args, ptbxl_rows, extra_rows, hashes, original_meta, steps_per_epoch)
    prepare_output(args, config)
    write_json_atomic(args.output_dir / "config.json", config)
    (args.output_dir / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    model, _ = load_model("ecg-fm", args.checkpoint, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(TrainingWaveforms(rows, args.raw_dir), batch_size=args.batch_size,
                        shuffle=True, drop_last=drop_last, num_workers=args.workers,
                        pin_memory=args.device == "cuda", persistent_workers=args.workers > 0,
                        generator=generator)
    history, elapsed_before = [], 0.0
    if args.resume:
        history = load_resume(args.output_dir / "resume.pt", model, optimizer, generator, args)
        elapsed_before = history[-1]["seconds"]
    started = time.monotonic()
    progress = adapt(args, model, optimizer, loader, generator, history, steps_per_epoch, drop_last,
                     len(rows), elapsed_before, started)
    path = save_adapted(args, model, config, history, progress, elapsed_before + time.monotonic() - started)
    # The completed portable backbone supersedes this run's optimizer resume file.
    (args.output_dir / "resume.pt").unlink(missing_ok=True)
    print(f"Completed adaptation: {path}", flush=True)


if __name__ == "__main__":
    main()
