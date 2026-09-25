#!/usr/bin/env python3
"""Four controlled xECG continuation arms, followed by the fixed 007 transfers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ecg_experiment.evaluation import evaluate_predictions
from ecg_experiment.files import sha256_file, sha256_json, write_json_atomic, write_torch_atomic
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.receipts import artifact_hashes, verified_completion
from ecg_experiment.reproducibility import cpu_state, restore_rng_lists, rng_state_lists, seed_everything
from ecg_experiment.training import optimizer_state_bytes, peak_gpu_bytes, require_cuda
from ecg_experiment.xecg import XECGBinaryClassifier, load_xecg
from ecg_experiment.xecg_adaptation import (
    ARMS,
    AdaptationConfig,
    AdaptationModel,
    ShuffledStream,
    contiguous_masks,
    ema_momentum_at,
    learning_rate_at,
    representation_diagnostics,
    ssl_parameter_groups,
    update_ema,
)
from scripts.experiments import run_xecg_finetune as ft

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs/experiment008_vision_ssl"
DEFAULT_SSL_CACHE = ROOT / "data/processed/xecg_ssl_40k"
SEED = 42
EFFECTIVE_BATCH_SIZE = 64
BUDGETS = ("full", "ten_percent")
MASK_SEED_OFFSET = 10001
DIAGNOSTIC_SEED_OFFSET = 20001
DIAGNOSTIC_RECORDS_PER_SOURCE = 4
SSL_RECORDS = 56875
SSL_SOURCE_COUNTS = {"ptbxl": 17418, "mimic": 39457}
SSL_VIEW_SHAPE = (SSL_RECORDS, 1000, 12)
PRINT_INTERVAL = 10
CHECKPOINT_INTERVAL = 100
PROFILE_UPDATES = 2
EMPTY_TRACE = "0" * 64
SSL_ARTIFACTS = ("encoder.pt", "history.json", "config.json")

Rows = list[dict[str, str]]


def save_ssl_resume(path: Path, fingerprint: dict[str, Any], model: AdaptationModel,
                    optimizer: torch.optim.Optimizer, stream: ShuffledStream,
                    mask_generator: torch.Generator, step: int, history: list[dict[str, Any]],
                    trace: str, elapsed: float) -> None:
    """
    Atomically save an update-boundary SSL resume checkpoint.

    Parameters
    ----------
    path : Path
        Destination ``resume.pt``.
    fingerprint : dict[str, Any]
        Arm fingerprint that a resumed run must match.
    model : AdaptationModel
        Student and EMA teacher.
    optimizer : torch.optim.Optimizer
        Student optimizer.
    stream : ShuffledStream
        Record sampler.
    mask_generator : torch.Generator
        Mask generator.
    step : int
        Completed optimizer updates.
    history : list[dict[str, Any]]
        One row per completed update.
    trace : str
        Running digest of sampled records and masks.
    elapsed : float
        Training seconds so far.
    """
    write_torch_atomic(path, {
        "version": 1, "fingerprint": fingerprint, "step": step,
        "student": cpu_state(model.student), "ema": cpu_state(model.ema),
        "optimizer": optimizer.state_dict(), "sampler": stream.state_dict(),
        "mask_rng": mask_generator.get_state(), "rng": rng_state_lists(),
        # The LR/EMA schedules are pure functions of this counter and config.
        "scheduler": {"next_update": step}, "history": history,
        "trace_sha256": trace, "elapsed_seconds": elapsed,
    })


def load_ssl_resume(path: Path, fingerprint: dict[str, Any], model: AdaptationModel,
                    optimizer: torch.optim.Optimizer, stream: ShuffledStream,
                    mask_generator: torch.Generator) -> dict[str, Any]:
    """
    Restore an SSL arm from an update-boundary resume checkpoint.

    Parameters
    ----------
    path : Path
        Checkpoint written by ``save_ssl_resume``.
    fingerprint : dict[str, Any]
        Fingerprint of the current arm.
    model : AdaptationModel
        Model whose student and EMA teacher are restored in place.
    optimizer : torch.optim.Optimizer
        Optimizer to restore in place.
    stream : ShuffledStream
        Sampler to restore in place.
    mask_generator : torch.Generator
        Mask generator to restore in place.

    Returns
    -------
    dict[str, Any]
        The loaded checkpoint.

    Raises
    ------
    ValueError
        If the checkpoint belongs to another arm or protocol, is malformed,
        or its CUDA random state does not fit this machine.
    """
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("version") != 1 or saved.get("fingerprint") != fingerprint:
        raise ValueError("SSL resume arm, sources, or protocol differ")
    if (not isinstance(saved["step"], int) or saved["step"] < 0
            or saved.get("scheduler") != {"next_update": saved["step"]}
            or len(saved["history"]) != saved["step"]
            or any(row["update"] != index + 1 for index, row in enumerate(saved["history"]))):
        raise ValueError("Malformed SSL resume step/history/scheduler")
    model.student.load_state_dict(saved["student"], strict=True)
    model.ema.load_state_dict(saved["ema"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    stream.load_state_dict(saved["sampler"])
    mask_generator.set_state(saved["mask_rng"])
    if not torch.cuda.is_available() and saved["rng"]["cuda"]:
        raise ValueError("CUDA resume requested without CUDA")
    restore_rng_lists(saved["rng"])
    return saved


def _read_ssl_rows(path: Path) -> Rows:
    """Read the identity-only SSL rows, rejecting any extra column such as a target."""
    from scripts.data.prepare_xecg_ssl import FIELDS

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError("SSL row columns must contain identities only, without targets")
        return list(reader)


def load_ssl_cache(path: Path) -> tuple[np.ndarray, Rows, dict[str, str]]:
    """
    Recheck the root-owned cache against its audited train-only identities.

    Parameters
    ----------
    path : Path
        SSL cache directory.

    Returns
    -------
    tuple[np.ndarray, list[dict[str, str]], dict[str, str]]
        Memory-mapped views, identity rows, and cache digests.

    Raises
    ------
    ValueError
        If the rows, sources, metadata, checksums, or array shape differ from
        the audited training-only pool.
    """
    from scripts.data.prepare_xecg_ssl import FIELDS, selected_rows

    metadata = json.loads((path / "metadata.json").read_text())
    rows = _read_ssl_rows(path / "rows.csv")
    expected_rows, sources = selected_rows()
    if rows != [{key: row[key] for key in FIELDS} for row in expected_rows]:
        raise ValueError("SSL rows differ from the audited training-only pool")
    if (len(rows) != SSL_RECORDS or any(row["split"] != "train" for row in rows)
            or metadata.get("all_train_only") is not True
            or metadata.get("source_sha256") != sources):
        raise ValueError("SSL cohort/source identity changed")
    if Counter(row["source"] for row in rows) != SSL_SOURCE_COUNTS:
        raise ValueError("Unexpected SSL source composition")
    for field, key in (("ecg_id", "ecg_ids"), ("patient_id", "patient_ids"), ("source", "sources")):
        if metadata.get(key) != [row[field] for row in rows]:
            raise ValueError(f"SSL metadata order differs for {field}")
    if len(set(metadata["ecg_ids"])) != len(rows):
        raise ValueError("Duplicate SSL ECG identity")
    checksums = {"rows_sha256": sha256_file(path / "rows.csv"),
                 "views_sha256": sha256_file(path / "views.npy"),
                 "raw_sha256_file_sha256": sha256_file(path / "raw_sha256.npy")}
    if any(metadata.get(key) != value for key, value in checksums.items()):
        raise ValueError("SSL cache checksum mismatch")
    views = np.load(path / "views.npy", mmap_mode="r")
    if (views.shape != SSL_VIEW_SHAPE or views.dtype != np.float32
            or metadata.get("shape") != list(views.shape)):
        raise ValueError("SSL cache must contain full float32 ten-second ECGs")
    return views, rows, {**checksums, "metadata_sha256": sha256_file(path / "metadata.json")}


def source_identity() -> dict[str, str]:
    """
    Hash every source file that determines the adaptation and transfer runs.

    Returns
    -------
    dict[str, str]
        SHA-256 digest keyed by repository-relative path, plus the xLSTM tree.
    """
    files = ("ecg_experiment/xecg_adaptation.py", "scripts/experiments/run_xecg_adaptation.py",
             "ecg_experiment/xecg.py", "scripts/experiments/run_xecg_finetune.py",
             "scripts/data/prepare_xecg_ssl.py", "ecg_experiment/run.py", "ecg_experiment/evaluation.py",
             "scripts/reports/report_xecg_adaptation.py", "scripts/experiments/run_cpc_experiment.py",
             "ecg_experiment/reproducibility.py", "ecg_experiment/training.py",
             "ecg_experiment/receipts.py")
    return {**{name: sha256_file(ROOT / name) for name in files},
            "xlstm_python_tree": ft.source_tree_sha256(ROOT / "third_party/xecg-deps/xlstm")}


def make_fingerprint(args: argparse.Namespace, config: AdaptationConfig, cache_hashes: dict[str, str],
                     arm: str) -> dict[str, Any]:
    """
    Describe every input and source that determines one SSL arm.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    config : AdaptationConfig
        Shared adaptation protocol.
    cache_hashes : dict[str, str]
        SSL cache digests.
    arm : str
        Objective arm.

    Returns
    -------
    dict[str, Any]
        Fingerprint that profile, resume, and completion receipts must match.
    """
    protocol = {"config": asdict(config),
                "views": "two contiguous masks; eight of forty tokens; clean teachers",
                "coding_rate_batch": "actual microbatch", "selection": "final student",
                "arm": arm, "backend": "vanilla", "precision": "float32"}
    return {"experiment": 8, "arm": arm, "protocol": protocol, "protocol_sha256": sha256_json(protocol),
            "sources": source_identity(), "release": ft.checkpoint_fingerprint(args.checkpoint_dir),
            "ssl_cache": cache_hashes, "device": args.device, "threads": args.threads,
            "torch_version": str(torch.__version__), "numpy_version": np.__version__}


def make_ssl_state(args: argparse.Namespace, config: AdaptationConfig, pool_size: int) -> tuple[
        AdaptationModel, torch.optim.Optimizer, ShuffledStream, torch.Generator]:
    """
    Seed and build the model, optimizer, record stream, and mask generator of an arm.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    config : AdaptationConfig
        Shared adaptation protocol.
    pool_size : int
        Records in the SSL pool.

    Returns
    -------
    tuple[AdaptationModel, torch.optim.Optimizer, ShuffledStream, torch.Generator]
        Fresh arm state; every arm starts from the same release and streams.
    """
    seed_everything(config.seed)
    backbone = load_xecg(args.checkpoint_dir, backend="vanilla", device=args.device, drop_path_prob=0.0)
    model = AdaptationModel(backbone).to(device=args.device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(ssl_parameter_groups(model.student, config), lr=config.learning_rate)
    stream = ShuffledStream(pool_size, config.seed)
    masks = torch.Generator().manual_seed(config.seed + MASK_SEED_OFFSET)
    return model, optimizer, stream, masks


def tensor_batch(views: np.ndarray, indices: torch.Tensor, device: str) -> torch.Tensor:
    """
    Copy cached ECGs into a float32 device tensor.

    Parameters
    ----------
    views : np.ndarray
        ``[records, samples, 12]`` cache.
    indices : torch.Tensor
        Record positions.
    device : str
        Destination device.

    Returns
    -------
    torch.Tensor
        ``[len(indices), samples, 12]`` batch.

    Raises
    ------
    ValueError
        If the batch is malformed or nonfinite.
    """
    batch = np.array(views[indices.tolist()], dtype=np.float32, copy=True)
    if batch.ndim != 3 or batch.shape[-1] != 12 or not np.isfinite(batch).all():
        raise ValueError("SSL input batch is malformed or nonfinite")
    return torch.from_numpy(batch).to(device)


def diagnostic_indices(rows: Rows, seed: int = SEED) -> torch.Tensor:
    """
    Select fixed four PTB and four MIMIC ECGs, without consuming training RNG.

    Parameters
    ----------
    rows : list[dict[str, str]]
        SSL identity rows with ``source``.
    seed : int
        Protocol seed.

    Returns
    -------
    torch.Tensor
        Eight record positions, PTB-XL first.

    Raises
    ------
    ValueError
        If a source has fewer than four records.
    """
    generator = np.random.default_rng(seed + DIAGNOSTIC_SEED_OFFSET)
    selected = []
    for source in ("ptbxl", "mimic"):
        candidates = [i for i, row in enumerate(rows) if row["source"] == source]
        if len(candidates) < DIAGNOSTIC_RECORDS_PER_SOURCE:
            raise ValueError("Diagnostics require four training records per source")
        selected.extend(generator.choice(candidates, DIAGNOSTIC_RECORDS_PER_SOURCE, replace=False).tolist())
    return torch.tensor(selected, dtype=torch.long)


def ssl_update(model: AdaptationModel, optimizer: torch.optim.Optimizer, views: np.ndarray,
               stream: ShuffledStream, mask_generator: torch.Generator, config: AdaptationConfig,
               arm: str, step: int, device: str,
               trace: str = EMPTY_TRACE) -> tuple[dict[str, Any], str, torch.Tensor]:
    """
    One effective batch; expansion statistics remain local to each microbatch.

    Parameters
    ----------
    model : AdaptationModel
        Student, EMA teacher, and frozen release.
    optimizer : torch.optim.Optimizer
        Student optimizer.
    views : np.ndarray
        SSL cache.
    stream : ShuffledStream
        Record sampler.
    mask_generator : torch.Generator
        Mask generator.
    config : AdaptationConfig
        Adaptation protocol.
    arm : str
        Objective arm.
    step : int
        Zero-based update index.
    device : str
        Torch device of ``model``.
    trace : str
        Running digest before this update.

    Returns
    -------
    tuple[dict[str, Any], str, torch.Tensor]
        History row, updated trace digest, and sampled record positions.

    Raises
    ------
    RuntimeError
        If the objective, gradient norm, or updated parameters are nonfinite.
    """
    indices = stream.take(config.effective_batch_size)
    tokens = views.shape[1] // model.student.patch_size
    masks = contiguous_masks(config.effective_batch_size, tokens, config.mask_tokens, mask_generator)
    trace_digest = hashlib.sha256(bytes.fromhex(trace))
    trace_digest.update(indices.numpy().tobytes())
    trace_digest.update(masks.numpy().tobytes())
    learning_rate = learning_rate_at(step, config)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    optimizer.zero_grad(set_to_none=True)
    model.train()
    sums = {}
    count = config.effective_batch_size // config.microbatch_size
    for start in range(0, len(indices), config.microbatch_size):
        signal = tensor_batch(views, indices[start:start + config.microbatch_size], device)
        loss, terms = model(signal, masks[start:start + config.microbatch_size].to(device), arm, config)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite adaptation objective")
        (loss / count).backward()
        for name, value in {"total": loss.detach(), **terms}.items():
            sums[name] = sums.get(name, 0.0) + float(value) / count
    norm = torch.nn.utils.clip_grad_norm_(model.student.parameters(), config.gradient_clip)
    if not torch.isfinite(norm):
        raise RuntimeError("Nonfinite adaptation gradient norm")
    optimizer.step()
    if not all(torch.isfinite(value).all() for value in model.student.parameters()):
        raise RuntimeError("Nonfinite adapted parameters")
    momentum = ema_momentum_at(step, config)
    update_ema(model.ema, model.student, momentum)
    optimizer.zero_grad(set_to_none=True)
    row = {"update": step + 1, "losses": sums, "gradient_norm": float(norm),
           "learning_rate": learning_rate, "ema_momentum": momentum,
           "record_draws": (step + 1) * config.effective_batch_size,
           "student_view_draws": (step + 1) * config.effective_batch_size * 2}
    return row, trace_digest.hexdigest(), indices


def ssl_completion(directory: Path, fingerprint: dict[str, Any]) -> dict[str, Any] | None:
    """
    Return the verified completion receipt of an SSL arm, if any.

    Parameters
    ----------
    directory : Path
        Arm directory.
    fingerprint : dict[str, Any]
        Inputs the completed arm must have used.

    Returns
    -------
    dict[str, Any] | None
        The receipt, or None when the arm has not completed.

    Raises
    ------
    ValueError
        If the completed arm used other inputs or an artifact changed.
    """
    return verified_completion(directory, fingerprint, SSL_ARTIFACTS)


def _check_arm_directory(args: argparse.Namespace, directory: Path, fingerprint: dict[str, Any]) -> None:
    """Allow an existing incomplete arm only with --resume and an identical configuration."""
    if directory.exists() and any(directory.iterdir()) and not args.resume:
        raise FileExistsError(f"Existing incomplete SSL arm; use --resume: {directory}")
    config_path = directory / "config.json"
    if config_path.exists() and json.loads(config_path.read_text())["fingerprint"] != fingerprint:
        raise ValueError("Incomplete SSL configuration differs")


def _write_arm_inputs(args: argparse.Namespace, config: AdaptationConfig, directory: Path,
                      fingerprint: dict[str, Any], views: np.ndarray, rows: Rows) -> torch.Tensor:
    """Record the arm configuration and fixed diagnostic waveforms; return those waveforms."""
    directory.mkdir(parents=True, exist_ok=True)
    write_json_atomic(directory / "config.json", {
        "fingerprint": fingerprint, "selection": "Final update student encoder",
        "mask_seed": config.seed + MASK_SEED_OFFSET,
        "source_counts": dict(Counter(row["source"] for row in rows)),
        "teachers": "clean EMA and frozen release; no gradients; frozen forward in every arm"})
    fixed_indices = diagnostic_indices(rows, config.seed)
    diagnostic_signal = tensor_batch(views, fixed_indices, args.device)
    np.savez(directory / "diagnostic_waveforms.npz", signals=diagnostic_signal.cpu().numpy(),
             ecg_ids=np.asarray([rows[i]["ecg_id"] for i in fixed_indices.tolist()]),
             sources=np.asarray([rows[i]["source"] for i in fixed_indices.tolist()]))
    return diagnostic_signal


def _is_checkpoint_update(step: int, config: AdaptationConfig) -> bool:
    return (step + 1) % CHECKPOINT_INTERVAL == 0 or step + 1 == config.updates


def pretrain_arm(args: argparse.Namespace, config: AdaptationConfig, views: np.ndarray, rows: Rows,
                 cache_hashes: dict[str, str], arm: str) -> None:
    """
    Train one SSL arm to its final update, resuming from the last checkpoint.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    config : AdaptationConfig
        Shared adaptation protocol.
    views : np.ndarray
        SSL cache.
    rows : list[dict[str, str]]
        SSL identity rows.
    cache_hashes : dict[str, str]
        SSL cache digests.
    arm : str
        Objective arm.

    Raises
    ------
    FileExistsError
        If incomplete output exists without ``--resume``.
    ValueError
        If existing output was made with another configuration or exceeds
        the update budget.
    """
    fingerprint = make_fingerprint(args, config, cache_hashes, arm)
    directory = args.output_dir / "pretrain" / arm
    if ssl_completion(directory, fingerprint):
        print(json.dumps({"stage": "ssl_reuse", "arm": arm}), flush=True)
        return
    resume_path = directory / "resume.pt"
    _check_arm_directory(args, directory, fingerprint)
    model, optimizer, stream, masks = make_ssl_state(args, config, len(views))
    history, trace, start_step = [], EMPTY_TRACE, 0
    started = time.monotonic()
    if resume_path.is_file():
        saved = load_ssl_resume(resume_path, fingerprint, model, optimizer, stream, masks)
        start_step, history, trace = saved["step"], saved["history"], saved["trace_sha256"]
        started -= saved["elapsed_seconds"]
        if start_step > config.updates:
            raise ValueError("SSL checkpoint exceeds the update budget")
    diagnostic_signal = _write_arm_inputs(args, config, directory, fingerprint, views, rows)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for step in range(start_step, config.updates):
        row, trace, indices = ssl_update(model, optimizer, views, stream, masks, config, arm, step,
                                         args.device, trace)
        selected = [rows[i] for i in indices.tolist()]
        patients = {(r["source"], r["patient_id"]) for r in selected}
        row.update({"source_counts": dict(Counter(r["source"] for r in selected)),
                    "repeated_patient_draws": len(selected) - len(patients),
                    "elapsed_seconds": time.monotonic() - started})
        if _is_checkpoint_update(step, config):
            row["diagnostics"] = representation_diagnostics(model, diagnostic_signal)
        history.append(row)
        if (step + 1) % PRINT_INTERVAL == 0 or step == start_step:
            print(json.dumps({"stage": "ssl_train", "arm": arm, **row}), flush=True)
        if _is_checkpoint_update(step, config):
            save_ssl_resume(resume_path, fingerprint, model, optimizer, stream, masks,
                            step + 1, history, trace, time.monotonic() - started)
            write_json_atomic(directory / "history.json", history)
    _finish_arm(args, config, directory, fingerprint, model, history, trace, started, arm)


def _finish_arm(args: argparse.Namespace, config: AdaptationConfig, directory: Path,
                fingerprint: dict[str, Any], model: AdaptationModel, history: list[dict[str, Any]],
                trace: str, started: float, arm: str) -> None:
    """Save the final student and its completion receipt, then drop the resume checkpoint."""
    write_torch_atomic(directory / "encoder.pt", {"student": cpu_state(model.student),
                       "fingerprint": fingerprint, "updates": config.updates,
                       "selection": "final_student", "trace_sha256": trace})
    write_json_atomic(directory / "history.json", history)
    receipt = {"fingerprint": fingerprint, "updates": config.updates, "trace_sha256": trace,
               "elapsed_seconds": time.monotonic() - started,
               "peak_cuda_memory_bytes": peak_gpu_bytes(args.device),
               "sha256": {name: sha256_file(directory / name) for name in SSL_ARTIFACTS}}
    write_json_atomic(directory / "complete.json", receipt)
    (directory / "resume.pt").unlink(missing_ok=True)
    print(json.dumps({"stage": "ssl_complete", "arm": arm, "updates": config.updates,
                      "encoder_sha256": receipt["sha256"]["encoder.pt"]}), flush=True)


def require_all_ssl(args: argparse.Namespace, config: AdaptationConfig,
                    cache_hashes: dict[str, str]) -> dict[str, dict[str, Any]]:
    """
    Barrier: every arm is final and immutable before any downstream test.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    config : AdaptationConfig
        Shared adaptation protocol.
    cache_hashes : dict[str, str]
        SSL cache digests.

    Returns
    -------
    dict[str, dict[str, Any]]
        Completion receipt per arm.

    Raises
    ------
    RuntimeError
        If an arm has not finished its update budget.
    ValueError
        If the arms saw different record or mask streams.
    """
    receipts = {}
    for arm in ARMS:
        receipt = ssl_completion(args.output_dir / "pretrain" / arm,
                                 make_fingerprint(args, config, cache_hashes, arm))
        if receipt is None or receipt.get("updates") != config.updates:
            raise RuntimeError("All four final SSL arms must exist before supervised transfers")
        receipts[arm] = receipt
    if len({receipt["trace_sha256"] for receipt in receipts.values()}) != 1:
        raise ValueError("Record and mask streams differed across SSL arms")
    marker = {arm: receipt["sha256"]["encoder.pt"] for arm, receipt in receipts.items()}
    write_json_atomic(args.output_dir / "all_ssl_complete.json", marker)
    return receipts


def _profile_arm(args: argparse.Namespace, config: AdaptationConfig, views: np.ndarray,
                 fingerprint: dict[str, Any], arm: str) -> dict[str, Any]:
    """Time two updates of one arm and check an exact resume roundtrip."""
    model, optimizer, stream, masks = make_ssl_state(args, config, len(views))
    trace, history, timings = EMPTY_TRACE, [], []
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for step in range(PROFILE_UPDATES):
        started = time.monotonic()
        row, trace, _ = ssl_update(model, optimizer, views, stream, masks, config, arm, step,
                                   args.device, trace)
        history.append(row)
        if args.device == "cuda":
            torch.cuda.synchronize()
        timings.append(time.monotonic() - started)
    path = args.output_dir / f"profile_{arm}_resume.pt"
    save_ssl_resume(path, fingerprint, model, optimizer, stream, masks, PROFILE_UPDATES, history, trace,
                    sum(timings))
    saved = load_ssl_resume(path, fingerprint, model, optimizer, stream, masks)
    if saved["step"] != PROFILE_UPDATES or saved["trace_sha256"] != trace:
        raise RuntimeError("Profile resume roundtrip failed")
    path.unlink()
    return {"first_update_seconds": timings[0], "steady_update_seconds": timings[1],
            "projected_ssl_arm_seconds": timings[1] * config.updates,
            "peak_cuda_memory_bytes": peak_gpu_bytes(args.device),
            "optimizer_state_bytes": optimizer_state_bytes(optimizer),
            "final_losses": history[-1]["losses"], "resume_roundtrip": True}


def _baseline_seconds() -> list[float]:
    """Return the recorded Experiment 007 training durations that exist."""
    seconds = []
    for budget in BUDGETS:
        path = ft.DEFAULT_OUTPUT_DIR / f"xecg_{budget}_seed42" / "config.json"
        if not path.is_file():
            continue
        elapsed = json.loads(path.read_text()).get("elapsed_seconds")
        if elapsed is not None:
            seconds.append(float(elapsed))
    return seconds


def profile(args: argparse.Namespace, config: AdaptationConfig, views: np.ndarray,
            cache_hashes: dict[str, str]) -> dict[str, Any]:
    """
    Exercise every objective and steady Adam memory without retaining weights.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    config : AdaptationConfig
        Shared adaptation protocol.
    views : np.ndarray
        SSL cache.
    cache_hashes : dict[str, str]
        SSL cache digests.

    Returns
    -------
    dict[str, Any]
        Profile receipt, also written to ``profile_<device>.json``.
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results, fingerprints = {}, {}
    for arm in ARMS:
        fingerprints[arm] = make_fingerprint(args, config, cache_hashes, arm)
        results[arm] = _profile_arm(args, config, views, fingerprints[arm], arm)
        if args.device == "cuda":
            torch.cuda.empty_cache()
        print(json.dumps({"stage": "adaptation_profile", "arm": arm, **results[arm]}), flush=True)
    baseline_seconds = _baseline_seconds()
    projected_ssl = sum(r["projected_ssl_arm_seconds"] for r in results.values())
    projected_transfer = len(ARMS) * sum(baseline_seconds) if len(baseline_seconds) == len(BUDGETS) else None
    result = {"config": asdict(config), "cache": cache_hashes, "sources": source_identity(),
              "device": args.device, "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
              "arm_fingerprints": fingerprints,
              "arms": results, "projected_total_ssl_seconds": projected_ssl,
              "projected_supervised_seconds": projected_transfer,
              "projected_total_suite_seconds": (projected_ssl + projected_transfer
                                                if projected_transfer is not None else None),
              "supervised_projection_assumption": "Four times the measured Experiment 007 two-budget "
                                                  "training duration; early stopping may differ"}
    write_json_atomic(args.output_dir / f"profile_{args.device}.json", result)
    return result


def require_profile(args: argparse.Namespace, config: AdaptationConfig, cache_hashes: dict[str, str]) -> None:
    """
    Require a profile receipt that matches the current four-arm configuration.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    config : AdaptationConfig
        Shared adaptation protocol.
    cache_hashes : dict[str, str]
        SSL cache digests.

    Raises
    ------
    RuntimeError
        If no profile receipt exists.
    ValueError
        If the receipt belongs to another configuration or lacks resume checks.
    """
    path = args.output_dir / f"profile_{args.device}.json"
    if not path.is_file():
        raise RuntimeError("Run --stage profile successfully before CUDA adaptation")
    receipt = json.loads(path.read_text())
    expected = {arm: make_fingerprint(args, config, cache_hashes, arm) for arm in ARMS}
    if receipt.get("arm_fingerprints") != expected or set(receipt.get("arms", {})) != set(ARMS):
        raise ValueError("Profile does not match the current four-arm configuration/sources")
    if not all(receipt["arms"][arm].get("resume_roundtrip") is True for arm in ARMS):
        raise ValueError("Profile did not finish all resume checks")


def _load_adapted_backbone(args: argparse.Namespace, config: AdaptationConfig, encoder_path: Path,
                           ssl_receipt: dict[str, Any], encoder_sha256: str) -> torch.nn.Module:
    """Seed, load the release, and replace its weights by the verified final SSL student."""
    seed_everything(SEED)
    backbone = load_xecg(args.checkpoint_dir, backend="vanilla", device=args.device,
                         drop_path_prob=ft.DROP_PATH_PROB)
    adapted = torch.load(encoder_path, map_location="cpu", weights_only=True)
    if (adapted["fingerprint"] != ssl_receipt["fingerprint"] or adapted["updates"] != config.updates
            or adapted["selection"] != "final_student" or sha256_file(encoder_path) != encoder_sha256):
        raise ValueError("Adapted encoder source/selection mismatch")
    backbone.load_state_dict(adapted["student"], strict=True)
    return backbone


def transfer(args: argparse.Namespace, config: AdaptationConfig, arm: str, budget: str,
             manifests: ft.Manifests, views: np.ndarray, index: dict[int, int],
             cache_hashes: dict[str, Any], ssl_receipt: dict[str, Any]) -> None:
    """
    Fine-tune one adapted arm on one label budget with the unchanged 007 policy.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    config : AdaptationConfig
        Shared adaptation protocol.
    arm : str
        Objective arm.
    budget : str
        Label budget.
    manifests : tuple
        Output of ``run_xecg_finetune.load_manifests`` for ``budget``.
    views : np.ndarray
        100 Hz xECG waveform cache.
    index : dict[int, int]
        Cache row per ECG identifier.
    cache_hashes : dict[str, Any]
        Waveform cache digests and metadata.
    ssl_receipt : dict[str, Any]
        Completion receipt of the arm.

    Raises
    ------
    ValueError
        If existing output or the adapted encoder does not match the inputs.
    """
    rows, development, calibration, manifest_hashes = manifests
    directory = args.output_dir / "transfer" / arm / f"xecg_{budget}_seed42"
    encoder_path = args.output_dir / "pretrain" / arm / "encoder.pt"
    all_ssl = json.loads((args.output_dir / "all_ssl_complete.json").read_text())
    fingerprint = {"experiment": 8, "arm": arm, "budget": budget,
                   "adaptation_fingerprint": ssl_receipt["fingerprint"],
                   "adapted_encoder_sha256": ssl_receipt["sha256"]["encoder.pt"],
                   "all_ssl_checkpoint_sha256": all_ssl,
                   "manifest_sha256": manifest_hashes, "cache": ft.cache_digests(cache_hashes),
                   "epochs": args.epochs, "patience": args.patience, "seed": SEED,
                   "microbatch_size": args.finetune_microbatch_size,
                   "effective_batch_size": EFFECTIVE_BATCH_SIZE,
                   "bootstrap": args.bootstrap, "device": args.device, "sources": source_identity()}
    if verified_completion(directory, fingerprint, ft.COMPLETION_FILES):
        print(json.dumps({"stage": "transfer_reuse", "arm": arm, "budget": budget}), flush=True)
        return
    resume = ft.resume_for_budget(directory, args.resume)
    config_path = directory / "config.json"
    if config_path.exists() and json.loads(config_path.read_text())["fingerprint"] != fingerprint:
        raise ValueError("Transfer resume configuration differs")
    backbone = _load_adapted_backbone(args, config, encoder_path, ssl_receipt,
                                      fingerprint["adapted_encoder_sha256"])
    model = XECGBinaryClassifier(backbone).to(args.device)
    optimizer = torch.optim.AdamW(ft.layerwise_parameter_groups(model), weight_decay=ft.WEIGHT_DECAY)
    scheduler = ft.make_scheduler(optimizer, math.ceil(len(rows["labeled_train"]) / EFFECTIVE_BATCH_SIZE),
                                  args.epochs)
    generator = torch.Generator().manual_seed(SEED)
    batch_size = args.finetune_microbatch_size
    train_loader = DataLoader(ft.CachedECGs(views, index, rows["labeled_train"]),
                              batch_size=batch_size, shuffle=True, num_workers=0, generator=generator)
    dev_loader = DataLoader(ft.CachedECGs(views, index, development), batch_size=batch_size, num_workers=0)
    directory.mkdir(parents=True, exist_ok=True)
    description = {"fingerprint": fingerprint, "head": "Identity + Linear(1024,1)",
                   "supervised_policy": "Experiment 007 helpers, unchanged", "precision": "float32",
                   "selected_ssl_encoder": "final update student",
                   "records": {"train": len(rows["labeled_train"]), "development": len(development),
                               "calibration": len(calibration), "test": len(rows["test"])}}
    write_json_atomic(directory / "config.json", description)
    result = ft.fit(model, optimizer, scheduler, train_loader, dev_loader,
                    np.asarray([int(r["target"]) for r in development]), directory, fingerprint,
                    generator, args.device, args.epochs, args.patience, EFFECTIVE_BATCH_SIZE, resume=resume)
    write_torch_atomic(directory / "model.pt", {"model": cpu_state(model), "fingerprint": fingerprint,
                                                "best_epoch": result["best_epoch"]})
    calibration_loader = DataLoader(ft.CachedECGs(views, index, calibration), batch_size=batch_size,
                                    num_workers=0)
    test_loader = DataLoader(ft.CachedECGs(views, index, rows["test"]), batch_size=batch_size, num_workers=0)
    evaluate_predictions(f"xecg_adaptation_{arm}_{budget}",
                         ft.predict(model, calibration_loader, args.device),
                         ft.predict(model, test_loader, args.device), calibration, rows["test"],
                         directory, SEED, args.bootstrap)
    description.update({key: value for key, value in result.items() if key != "history"})
    write_json_atomic(directory / "config.json", description)
    hashes = artifact_hashes(directory, ft.COMPLETION_FILES)
    write_json_atomic(directory / "complete.json", {"fingerprint": fingerprint, "sha256": hashes})
    (directory / "resume.pt").unlink(missing_ok=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse and validate runner arguments.

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
    parser.add_argument("--stage", choices=("check", "profile", "pretrain", "train", "all"), default="all")
    parser.add_argument("--arm", choices=(*ARMS, "all"), default="all")
    parser.add_argument("--ssl-cache-dir", type=Path, default=DEFAULT_SSL_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint-dir", type=Path, default=ft.DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--manifest-root", type=Path, default=ft.DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--raw-dir", type=Path, default=ft.DEFAULT_RAW_DIR)
    parser.add_argument("--cache-dir", type=Path, default=ft.DEFAULT_CACHE_DIR)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--microbatch-size", type=int, default=8)
    parser.add_argument("--finetune-microbatch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.threads < 1 or min(args.microbatch_size, args.finetune_microbatch_size, args.bootstrap) < 1:
        parser.error("Threads, batch sizes and bootstrap must be positive")
    if EFFECTIVE_BATCH_SIZE % args.microbatch_size or EFFECTIVE_BATCH_SIZE % args.finetune_microbatch_size:
        parser.error("Both microbatch sizes must divide 64")
    if args.epochs != 40 or args.patience != 8:
        parser.error("Experiment 008 fixes the Experiment 007 policy at 40 epochs/patience 8")
    return args


def _all_transfers_complete(output_dir: Path) -> bool:
    return all((output_dir / "transfer" / arm / f"xecg_{budget}_seed42" / "complete.json").is_file()
               for arm in ARMS for budget in BUDGETS)


def run_transfers(args: argparse.Namespace, config: AdaptationConfig, selected: tuple[str, ...],
                  manifests: dict[str, ft.Manifests], ssl_hashes: dict[str, str]) -> None:
    """
    Fine-tune every selected arm on both budgets, then report once all are complete.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    config : AdaptationConfig
        Shared adaptation protocol.
    selected : tuple[str, ...]
        Arms to transfer.
    manifests : dict[str, tuple]
        Output of ``run_xecg_finetune.load_manifests`` per budget.
    ssl_hashes : dict[str, str]
        SSL cache digests.
    """
    receipts = require_all_ssl(args, config, ssl_hashes)
    views, index, cache_hashes = ft.cache_fingerprint(args.cache_dir, args.raw_dir,
                                                      ft.manifest_dir(args.manifest_root, "full"))
    for arm in selected:
        for budget in BUDGETS:
            transfer(args, config, arm, budget, manifests[budget], views, index, cache_hashes, receipts[arm])
    if not _all_transfers_complete(args.output_dir):
        return
    from scripts.reports.report_xecg_adaptation import report

    report(args.output_dir, args.bootstrap)
    write_json_atomic(args.output_dir / "complete.json", {
        "all_ssl_sha256": sha256_file(args.output_dir / "all_ssl_complete.json"),
        "report_sha256": sha256_file(args.output_dir / "report.md"),
        "paired_comparisons_sha256": sha256_file(args.output_dir / "paired_comparisons.json")})


def main(argv: list[str] | None = None) -> None:
    """
    Check inputs, then profile, pretrain, or transfer the selected arms.

    Parameters
    ----------
    argv : list[str] | None
        Arguments, or ``None`` for ``sys.argv``.
    """
    args = parse_args(argv)
    if args.stage != "check":
        require_cuda(args.device)
    torch.set_num_threads(args.threads)
    config = AdaptationConfig(microbatch_size=args.microbatch_size)
    selected = ARMS if args.arm == "all" else (args.arm,)
    manifests = {budget: ft.load_manifests(args.manifest_root, budget) for budget in BUDGETS}
    ssl_views, ssl_rows, ssl_hashes = load_ssl_cache(args.ssl_cache_dir)
    print(json.dumps({"stage": "adaptation_check", "records": len(ssl_rows),
                      "arms": selected, "protocol": asdict(config)}), flush=True)
    if args.stage == "check":
        return
    with gpu_lock(args.device, blocking=False):
        if args.stage == "profile":
            profile(args, config, ssl_views, ssl_hashes)
            return
        if args.stage in ("pretrain", "all"):
            if args.device == "cuda":
                require_profile(args, config, ssl_hashes)
            for arm in selected:
                pretrain_arm(args, config, ssl_views, ssl_rows, ssl_hashes, arm)
        if args.stage in ("train", "all"):
            run_transfers(args, config, selected, manifests, ssl_hashes)


if __name__ == "__main__":
    main()
