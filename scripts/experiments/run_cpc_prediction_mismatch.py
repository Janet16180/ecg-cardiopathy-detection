#!/usr/bin/env python3
"""Experiment 009: frozen CPC prediction mismatch versus matched local features."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import time
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ecg_experiment import cpc_pool
from ecg_experiment.cpc import CPCPretrainer
from ecg_experiment.cpc_prediction_mismatch import (
    ARMS,
    BRANCH_WIDTH,
    FIRST_QUERY,
    FIRST_TARGET,
    HORIZON,
    extract_branches,
    features_for_arm,
    load_bootstrap,
    supervised_indices,
)
from ecg_experiment.evaluation import evaluate_predictions, paired_comparison, partition_validation
from ecg_experiment.files import sha256_file, sha256_json, write_json_atomic
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.training import parameter_count

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "outputs/experiment009_cpc_prediction_mismatch"
DEFAULT_BOOTSTRAP = ROOT / "outputs/experiment004_cpc_40k/cpc_ssl"
C_VALUES = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
BUDGETS = {"full": ("1", 15360), "ten_percent": ("0.1", 1518)}
VALIDATION_RECORDS = 1870
TEST_RECORDS = 1896
DEVELOPMENT_RECORDS = 1306
CALIBRATION_RECORDS = 564
EXTRACTION_RECORDS = 19126
SIGNAL_SAMPLES = 2500
HALF_TOKENS = 79
RETAINED_TOKENS = 72
SEED = 42
MAX_ITER = 3000
PROFILE_REPEATS = 3
PROGRESS_EVERY_BATCHES = 10
FEATURE_FIELDS = ("ecg_id", "patient_id", "source", "split")
ARTIFACTS = ("config.json", "linear_model.npz", "selection.json", "metrics.json",
             "test_predictions.csv", "calibration_predictions.npz")
CODE_FILES = ("ecg_experiment/cpc_prediction_mismatch.py",
              "scripts/experiments/run_cpc_prediction_mismatch.py", "ecg_experiment/cpc.py",
              "scripts/experiments/run_cpc_experiment.py",
              "ecg_experiment/run.py", "ecg_experiment/evaluation.py", "ecg_experiment/data.py",
              "ecg_experiment/cpc_pool.py", "ecg_experiment/files.py")
COMPARISONS = (("residual", "ordinary"), ("ordinary", "context"), ("residual", "context"))


def source_hashes() -> dict[str, str]:
    """
    Digest every repository file that determines this experiment's results.

    Returns
    -------
    dict[str, str]
        SHA-256 keyed by repository-relative path.
    """
    return {name: sha256_file(ROOT / name) for name in CODE_FILES}


def read_rows(path: str | Path) -> tuple[list[dict[str, str]], list[str] | None]:
    """
    Read a CSV file with its header.

    Parameters
    ----------
    path : str | Path
        CSV file.

    Returns
    -------
    tuple[list[dict[str, str]], list[str] | None]
        Rows and the header fields.
    """
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), reader.fieldnames


def check_budget(rows: dict[str, list[dict[str, str]]], count: int) -> tuple[list[dict[str, str]],
                                                                             list[dict[str, str]]]:
    """
    Check one label budget against the fixed protocol and split its validation rows.

    Parameters
    ----------
    rows : dict[str, list[dict[str, str]]]
        Manifest rows from ``cpc_pool.manifest_rows``.
    count : int
        Required number of labeled training rows.

    Returns
    -------
    tuple[list[dict[str, str]], list[dict[str, str]]]
        Development and calibration rows.

    Raises
    ------
    ValueError
        If counts, classes, duplicates or patient partitions differ from the protocol.
    """
    if (len(rows["labeled_train"]) != count or len(rows["validation"]) != VALIDATION_RECORDS
            or len(rows["test"]) != TEST_RECORDS):
        raise ValueError("Supervised manifests differ from the fixed PTB label budgets")
    if {row["target"] for row in rows["labeled_train"]} != {"0", "1"}:
        raise ValueError("Training manifests must contain both binary classes")
    all_rows = [row for name in ("labeled_train", "validation", "test") for row in rows[name]]
    if len({row["ecg_id"] for row in all_rows}) != len(all_rows):
        raise ValueError("Duplicate ECGs within/across supervised splits")
    patients = [{row["patient_id"] for row in rows[name]} for name in ("all_train_ssl", "validation", "test")]
    if any(patients[i] & patients[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Training, validation and test patients overlap")
    development, calibration = partition_validation(rows["validation"])
    if (len(development), len(calibration)) != (DEVELOPMENT_RECORDS, CALIBRATION_RECORDS):
        raise ValueError("Development/calibration partition differs from the fixed protocol")
    return development, calibration


def load_manifests(pool: cpc_pool.Pool, root: Path) -> tuple[dict[str, Any], list[dict[str, str]],
                                                             dict[str, Any], dict[str, Any]]:
    """
    Load both label budgets, check they nest, and list the extraction cohort.

    Parameters
    ----------
    pool : cpc_pool.Pool
        Frozen CPC waveform cache.
    root : Path
        PTB-XL manifest root.

    Returns
    -------
    tuple[dict[str, Any], list[dict[str, str]], dict[str, Any], dict[str, Any]]
        Rows per budget and partition, feature-row identities, manifest digests
        and manifest headers.

    Raises
    ------
    ValueError
        If the budgets do not share patients and waveforms, or the cohort size differs.
    """
    manifests, hashes, headers = {}, {}, {}
    for budget, (fraction, count) in BUDGETS.items():
        rows, fingerprints = cpc_pool.manifest_rows(pool, root, fraction)
        development, calibration = check_budget(rows, count)
        manifests[budget] = {**rows, "development": development, "calibration": calibration}
        hashes[budget] = fingerprints
        headers[budget] = {name: read_rows(Path(root) / f"seed42_fraction{fraction}" / f"{name}.csv")[1]
                           for name in ("labeled_train", "validation", "test")}
    full = {row["ecg_id"]: row for row in manifests["full"]["labeled_train"]}
    if any(full.get(row["ecg_id"]) != row for row in manifests["ten_percent"]["labeled_train"]):
        raise ValueError("The 10% labels are not the fixed subset of the full labels")
    for name in ("validation", "test", "all_train_ssl"):
        if manifests["full"][name] != manifests["ten_percent"][name]:
            raise ValueError("Label budgets use different patients or waveforms")
    features = [{"ecg_id": row["ecg_id"], "patient_id": row["patient_id"], "source": "ptbxl", "split": split}
                for name, split in (("labeled_train", "train"), ("validation", "validation"),
                                    ("test", "test"))
                for row in manifests["full"][name]]
    if len(features) != EXTRACTION_RECORDS:
        raise ValueError("Unexpected extraction cohort")
    return manifests, features, hashes, headers


def check_bootstrap_config(config: dict[str, Any], data_hashes: dict[str, str]) -> None:
    """
    Check the frozen CPC run's recorded inputs and code against the current files.

    Parameters
    ----------
    config : dict[str, Any]
        Experiment 004 SSL configuration.
    data_hashes : dict[str, str]
        Current pool and manifest digests.

    Raises
    ------
    ValueError
        If the configuration, its inputs or its code changed.
    """
    if sha256_json(config["inputs"]) != config["fingerprint"]:
        raise ValueError("Original CPC configuration fingerprint is invalid")
    for path, expected in config["inputs"]["cache"].items():
        if path in data_hashes and data_hashes[path] != expected:
            raise ValueError(f"CPC checkpoint source identity changed: {path}")
    for name, expected in config["inputs"]["code"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"Frozen CPC implementation changed: {name}")


def load_frozen_normalization(path: Path, pool: cpc_pool.Pool, data_hashes: dict[str, str],
                              config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the normalization the frozen CPC encoder was trained with.

    Parameters
    ----------
    path : Path
        Experiment 004 ``normalization.json``.
    pool : cpc_pool.Pool
        Frozen CPC waveform cache.
    data_hashes : dict[str, str]
        Current pool and manifest digests.
    config : dict[str, Any]
        Experiment 004 SSL configuration.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Float32 per-lead mean and standard deviation.

    Raises
    ------
    ValueError
        If it was fitted elsewhere, is invalid or differs from the CPC inputs.
    """
    normalization = json.loads(path.read_text())
    expected_source = {"train_ids_sha256": sha256_json([row["ecg_id"] for row in pool.train_rows]),
                       "rows_sha256": sha256_file(pool.directory / "rows.csv"),
                       "signals_sha256": data_hashes[str((pool.directory / "signals.npy").resolve())],
                       "method": "global per-lead mean and population std, training waveforms only"}
    if (normalization.get("source") != expected_source
            or normalization.get("count") != len(pool.train_rows) * SIGNAL_SAMPLES):
        raise ValueError("Frozen CPC normalization was fitted on different training waveforms")
    mean, std = (np.asarray(normalization[key], dtype=np.float32) for key in ("mean", "std"))
    if (mean.shape != (12,) or std.shape != (12,) or not np.isfinite(mean).all() or not np.isfinite(std).all()
            or (std <= 0).any()):
        raise ValueError("Invalid frozen normalization")
    if any(not np.array_equal(value, np.asarray(config["inputs"]["normalization"][key], dtype=np.float32))
           for key, value in (("mean", mean), ("std", std))):
        raise ValueError("Normalization does not match the pretrained CPC input statistics")
    return mean, std


def prepare_inputs(args: argparse.Namespace) -> tuple[cpc_pool.Pool, CPCPretrainer, np.ndarray, np.ndarray,
                                                      dict[str, Any], list[dict[str, str]], dict[str, Any]]:
    """
    Verify every input and describe the experiment's identity.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``cache_dir``, ``manifest_root`` and ``bootstrap_dir``.

    Returns
    -------
    tuple[cpc_pool.Pool, CPCPretrainer, np.ndarray, np.ndarray, dict[str, Any], list[dict[str, str]],
          dict[str, Any]]
        Pool, frozen model, normalization mean and std, manifests,
        feature-row identities and the input identity.
    """
    pool = cpc_pool.Pool(args.cache_dir)
    data_hashes = cpc_pool.make_source_hashes(pool, args.manifest_root)
    model, config = load_bootstrap(args.bootstrap_dir)
    check_bootstrap_config(config, data_hashes)
    normalization_path = args.bootstrap_dir.parent / "normalization.json"
    mean, std = load_frozen_normalization(normalization_path, pool, data_hashes, config)
    manifests, rows, manifest_hashes, headers = load_manifests(pool, args.manifest_root)
    checkpoint_hashes = {name: sha256_file(args.bootstrap_dir / name)
                         for name in ("encoder.pt", "epoch_state.pt", "config.json")}
    identity = {"experiment": 9, "sources": source_hashes(), "checkpoint": checkpoint_hashes,
                "checkpoint_fingerprint": config["fingerprint"], "data": data_hashes,
                "normalization_sha256": sha256_file(normalization_path),
                "manifests": manifest_hashes, "manifest_headers": headers,
                "feature_rows_sha256": sha256_json(rows), "feature_header": list(FEATURE_FIELDS),
                "token_selection": {"horizon": HORIZON, "first_query": FIRST_QUERY,
                                    "first_target": FIRST_TARGET, "half_tokens": HALF_TOKENS,
                                    "retained_tokens_per_half": RETAINED_TOKENS},
                "pooling": ("Per-half mean/max, then mean across halves; same retained positions in every "
                            "branch"),
                "residual": "normalize(z_t) - normalize(head_4(h_(t-4))); preserve difference magnitude",
                "torch_version": str(torch.__version__), "numpy_version": np.__version__}
    return pool, model, mean, std, manifests, rows, identity


def normalized_batch(pool: cpc_pool.Pool, rows: list[dict[str, str]], mean: np.ndarray, std: np.ndarray,
                     device: str) -> torch.Tensor:
    """
    Read, check and normalize the waveforms of some rows.

    Parameters
    ----------
    pool : cpc_pool.Pool
        Frozen CPC waveform cache.
    rows : list[dict[str, str]]
        Rows to read.
    mean : np.ndarray
        Per-lead mean shaped ``[12]``.
    std : np.ndarray
        Per-lead standard deviation shaped ``[12]``.
    device : str
        Target device.

    Returns
    -------
    torch.Tensor
        Normalized float32 waveforms.

    Raises
    ------
    ValueError
        If a waveform has nonfinite samples.
    """
    values = np.array(pool.signals[pool.indices(rows)], dtype=np.float32, copy=True)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite input in the unchanged extraction cohort")
    values -= mean[None, :, None]
    values /= std[None, :, None]
    return torch.from_numpy(values).to(device)


def extraction_fingerprint(args: argparse.Namespace, identity: dict[str, Any]) -> dict[str, Any]:
    """
    Identity of a feature extraction, including its device and batching.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``device``, ``batch_size`` and ``threads``.
    identity : dict[str, Any]
        Input identity from ``prepare_inputs``.

    Returns
    -------
    dict[str, Any]
        Extraction fingerprint.
    """
    return {"inputs": identity, "device": args.device, "batch_size": args.batch_size,
            "threads": args.threads, "precision": "float32"}


def profile(args: argparse.Namespace, pool: cpc_pool.Pool, model: CPCPretrainer, mean: np.ndarray,
            std: np.ndarray, rows: list[dict[str, str]], identity: dict[str, Any]) -> dict[str, Any]:
    """
    Time repeated extraction of one batch and project the full extraction cost.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``device``, ``batch_size``, ``threads`` and ``output_dir``.
    pool : cpc_pool.Pool
        Frozen CPC waveform cache.
    model : CPCPretrainer
        Frozen model.
    mean : np.ndarray
        Per-lead mean.
    std : np.ndarray
        Per-lead standard deviation.
    rows : list[dict[str, str]]
        Extraction cohort.
    identity : dict[str, Any]
        Input identity.

    Returns
    -------
    dict[str, Any]
        Profile receipt, also written to ``profile.json``.
    """
    fingerprint = extraction_fingerprint(args, identity)
    model = model.to(args.device)
    count = min(args.batch_size, len(rows))
    timings = []
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for _ in range(PROFILE_REPEATS):
        started = time.monotonic()
        values = normalized_batch(pool, rows[:count], mean, std, args.device)
        features = extract_branches(model, values).cpu().numpy()
        if args.device == "cuda":
            torch.cuda.synchronize()
        timings.append(time.monotonic() - started)
    result = {"fingerprint": fingerprint, "batch_records": count, "branch_shape": list(features.shape),
              "warmup_seconds": timings[0], "mean_measured_batch_seconds": float(np.mean(timings[1:])),
              "projected_extraction_seconds": float(np.mean(timings[1:])) * math.ceil(len(rows) / count),
              "feature_bytes": len(rows) * 3 * BRANCH_WIDTH * 4,
              "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None,
              "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
              "encoder_parameters": parameter_count(model.encoder),
              "head_parameters": parameter_count(model.heads[0]),
              "finite": bool(np.isfinite(features).all())}
    write_json_atomic(args.output_dir / "profile.json", result)
    summary = {key: value for key, value in result.items() if key != "fingerprint"}
    print(json.dumps({"stage": "mismatch_profile", **summary}), flush=True)
    return result


def feature_completion(directory: Path, fingerprint: dict[str, Any]) -> dict[str, Any] | None:
    """
    Verify a completed feature cache, or report that none exists.

    Parameters
    ----------
    directory : Path
        Feature directory.
    fingerprint : dict[str, Any]
        Extraction fingerprint the cache must match.

    Returns
    -------
    dict[str, Any] | None
        Completion metadata, or ``None`` when extraction has not finished.

    Raises
    ------
    ValueError
        If the cache has other inputs or changed bytes.
    """
    path = directory / "metadata.json"
    if not path.is_file():
        return None
    metadata = json.loads(path.read_text())
    if metadata.get("fingerprint") != fingerprint:
        raise ValueError("Completed mismatch features have different input identities")
    if metadata["sha256"] != {name: sha256_file(directory / name) for name in ("features.npy", "rows.csv")}:
        raise ValueError("Frozen feature cache checksum mismatch")
    return metadata


def write_feature_rows(path: Path, rows: list[dict[str, str]]) -> None:
    """
    Write the feature-row identities once, or check the existing file.

    Parameters
    ----------
    path : Path
        ``rows.csv`` in the feature directory.
    rows : list[dict[str, str]]
        Extraction cohort.

    Raises
    ------
    ValueError
        If existing rows or header differ.
    """
    if path.exists():
        saved, fields = read_rows(path)
        if saved != rows or fields != list(FEATURE_FIELDS):
            raise ValueError("Existing feature row identities changed")
        return
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEATURE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def chunk_sha256(values: np.ndarray) -> str:
    """
    Digest a block of extracted features.

    Parameters
    ----------
    values : np.ndarray
        Feature rows.

    Returns
    -------
    str
        SHA-256 of the contiguous bytes.
    """
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def resume_progress(progress_path: Path, fingerprint: dict[str, Any], values: np.ndarray,
                    total: int) -> tuple[int, list[dict[str, Any]], float]:
    """
    Verify every saved chunk of an interrupted extraction.

    Parameters
    ----------
    progress_path : Path
        ``progress.json`` of the interrupted extraction.
    fingerprint : dict[str, Any]
        Extraction fingerprint the progress must match.
    values : np.ndarray
        Partially written feature array.
    total : int
        Number of rows to extract.

    Returns
    -------
    tuple[int, list[dict[str, Any]], float]
        Completed rows, verified chunks and elapsed seconds.

    Raises
    ------
    ValueError
        If the fingerprint, chunk layout or chunk bytes differ.
    """
    progress = json.loads(progress_path.read_text())
    if progress["fingerprint"] != fingerprint:
        raise ValueError("Partial feature extraction fingerprint differs")
    done, chunks = progress["completed_rows"], progress["chunks"]
    expected_start = 0
    for chunk in chunks:
        if chunk["start"] != expected_start or not expected_start < chunk["stop"] <= done:
            raise ValueError("Malformed extraction progress chunks")
        if chunk_sha256(values[chunk["start"]:chunk["stop"]]) != chunk["sha256"]:
            raise ValueError("Partially extracted feature bytes changed")
        expected_start = chunk["stop"]
    if expected_start != done or not 0 <= done <= total:
        raise ValueError("Malformed extraction cursor")
    return done, chunks, progress["elapsed_seconds"]


def open_features(args: argparse.Namespace, directory: Path, fingerprint: dict[str, Any],
                  total: int) -> tuple[np.ndarray, int, list[dict[str, Any]], float]:
    """
    Open the feature array for a new or resumed extraction.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``resume``.
    directory : Path
        Feature directory.
    fingerprint : dict[str, Any]
        Extraction fingerprint.
    total : int
        Number of rows to extract.

    Returns
    -------
    tuple[np.ndarray, int, list[dict[str, Any]], float]
        Writable memory map, completed rows, verified chunks and elapsed seconds.

    Raises
    ------
    FileExistsError
        If partial results exist without ``--resume`` or without a cursor.
    ValueError
        If the array has the wrong shape or dtype.
    """
    partial, final = directory / "features.partial.npy", directory / "features.npy"
    progress_path = directory / "progress.json"
    shape = (total, 3, BRANCH_WIDTH)
    if progress_path.exists():
        if not args.resume:
            raise FileExistsError("Interrupted feature extraction exists; use --resume")
        values = np.lib.format.open_memmap(partial if partial.exists() else final, mode="r+")
        done, chunks, previous = resume_progress(progress_path, fingerprint, values, total)
    else:
        if partial.exists() or final.exists():
            raise FileExistsError("Feature array exists without a recovery cursor")
        values = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=shape)
        write_json_atomic(progress_path, {"fingerprint": fingerprint, "completed_rows": 0,
                                          "chunks": [], "elapsed_seconds": 0.0})
        done, chunks, previous = 0, [], 0.0
    if values.shape != shape or values.dtype != np.float32:
        raise ValueError("Feature cache shape/dtype differs")
    return values, done, chunks, previous


def write_features(args: argparse.Namespace, pool: cpc_pool.Pool, model: CPCPretrainer, mean: np.ndarray,
                   std: np.ndarray, rows: list[dict[str, str]], fingerprint: dict[str, Any]) -> float:
    """
    Extract the remaining rows chunk by chunk, recording a verifiable cursor.

    The memory map is closed when this function returns, before the partial
    array is renamed into place.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``output_dir``, ``device``, ``batch_size`` and ``resume``.
    pool : cpc_pool.Pool
        Frozen CPC waveform cache.
    model : CPCPretrainer
        Frozen model.
    mean : np.ndarray
        Per-lead mean.
    std : np.ndarray
        Per-lead standard deviation.
    rows : list[dict[str, str]]
        Extraction cohort.
    fingerprint : dict[str, Any]
        Extraction fingerprint.

    Returns
    -------
    float
        Total extraction seconds, including earlier invocations.
    """
    directory = args.output_dir / "features"
    progress_path = directory / "progress.json"
    values, done, chunks, previous = open_features(args, directory, fingerprint, len(rows))
    model = model.to(args.device)
    started = time.monotonic()
    for start in range(done, len(rows), args.batch_size):
        stop = min(len(rows), start + args.batch_size)
        signal = normalized_batch(pool, rows[start:stop], mean, std, args.device)
        batch = extract_branches(model, signal).cpu().numpy()
        values[start:stop] = batch
        values.flush()
        chunks.append({"start": start, "stop": stop, "sha256": hashlib.sha256(batch.tobytes()).hexdigest()})
        write_json_atomic(progress_path, {"fingerprint": fingerprint, "completed_rows": stop,
                                          "chunks": chunks,
                                          "elapsed_seconds": previous + time.monotonic() - started})
        if stop % (args.batch_size * PROGRESS_EVERY_BATCHES) == 0 or stop == len(rows):
            print(json.dumps({"stage": "mismatch_extract", "records": stop, "total": len(rows)}), flush=True)
    return previous + time.monotonic() - started


def extract(args: argparse.Namespace, pool: cpc_pool.Pool, model: CPCPretrainer, mean: np.ndarray,
            std: np.ndarray, rows: list[dict[str, str]], identity: dict[str, Any]) -> None:
    """
    Extract all three frozen feature branches resumably, then seal the cache.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    pool : cpc_pool.Pool
        Frozen CPC waveform cache.
    model : CPCPretrainer
        Frozen model.
    mean : np.ndarray
        Per-lead mean.
    std : np.ndarray
        Per-lead standard deviation.
    rows : list[dict[str, str]]
        Extraction cohort.
    identity : dict[str, Any]
        Input identity.
    """
    directory = args.output_dir / "features"
    fingerprint = extraction_fingerprint(args, identity)
    if feature_completion(directory, fingerprint):
        return
    directory.mkdir(parents=True, exist_ok=True)
    write_feature_rows(directory / "rows.csv", rows)
    elapsed = write_features(args, pool, model, mean, std, rows, fingerprint)
    partial = directory / "features.partial.npy"
    if partial.exists():
        os.replace(partial, directory / "features.npy")
    write_json_atomic(directory / "metadata.json", {"fingerprint": fingerprint,
                      "shape": [len(rows), 3, BRANCH_WIDTH], "dtype": "float32", "branch_order": list(ARMS),
                      "elapsed_seconds": elapsed,
                      "sha256": {name: sha256_file(directory / name)
                                 for name in ("features.npy", "rows.csv")}})
    (directory / "progress.json").unlink(missing_ok=True)


def load_features(args: argparse.Namespace, identity: dict[str, Any],
                  rows: list[dict[str, str]]) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Load a completed extraction, possibly made on another device, for CPU probes.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``output_dir``.
    identity : dict[str, Any]
        Current input identity.
    rows : list[dict[str, str]]
        Extraction cohort.

    Returns
    -------
    tuple[np.ndarray, dict[str, Any]]
        Read-only ``[rows, 3, 512]`` features and completion metadata.

    Raises
    ------
    ValueError
        If the features came from other inputs, rows or are invalid.
    """
    directory = args.output_dir / "features"
    # CPU probes consume an existing GPU/CPU extraction without changing its provenance.
    raw_metadata = json.loads((directory / "metadata.json").read_text())
    fingerprint = raw_metadata["fingerprint"]
    if fingerprint.get("inputs") != identity:
        raise ValueError("Frozen features were extracted from different inputs/code")
    metadata = feature_completion(directory, fingerprint)
    saved_rows, fields = read_rows(directory / "rows.csv")
    if saved_rows != rows or fields != list(FEATURE_FIELDS):
        raise ValueError("Frozen feature order/header differs from supervised cohort")
    features = np.load(directory / "features.npy", mmap_mode="r")
    if (features.shape != (len(rows), 3, BRANCH_WIDTH) or features.dtype != np.float32
            or not np.isfinite(features).all()):
        raise ValueError("Invalid completed feature array")
    return features, metadata


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    """
    Save arrays to an ``.npz`` file through a temporary file.

    Parameters
    ----------
    path : Path
        Destination file.
    **arrays : np.ndarray
        Arrays keyed by name.
    """
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def linear_logits(features: np.ndarray, saved: Any) -> np.ndarray:
    """
    Apply a saved standardized linear probe.

    Parameters
    ----------
    features : np.ndarray
        Raw features.
    saved : Any
        Mapping with ``mean``, ``scale``, ``coefficient`` and ``intercept``.

    Returns
    -------
    np.ndarray
        Logits.
    """
    standardized = (features - saved["mean"]) / saved["scale"]
    return standardized @ saved["coefficient"].reshape(-1) + float(saved["intercept"].reshape(-1)[0])


def load_candidate(candidate_path: Path, marker_path: Path, fingerprint: dict[str, Any], c: float,
                   scaler: StandardScaler, dev_features: np.ndarray,
                   dev_y: np.ndarray) -> tuple[dict[str, Any], float]:
    """
    Reload and re-score a candidate fitted by an earlier invocation.

    Parameters
    ----------
    candidate_path : Path
        Saved probe parameters.
    marker_path : Path
        Saved candidate receipt.
    fingerprint : dict[str, Any]
        Probe fingerprint the candidate must match.
    c : float
        Regularization value of this candidate.
    scaler : StandardScaler
        Training-only standardization.
    dev_features : np.ndarray
        Development features.
    dev_y : np.ndarray
        Development labels.

    Returns
    -------
    tuple[dict[str, Any], float]
        Candidate receipt and its development AUROC.

    Raises
    ------
    ValueError
        If the candidate's identity, standardization or score changed.
    """
    choice = json.loads(marker_path.read_text())
    if (choice["fingerprint"] != fingerprint or choice["C"] != c
            or choice["sha256"] != sha256_file(candidate_path)):
        raise ValueError("Classifier candidate identity/checksum mismatch")
    saved = np.load(candidate_path)
    if not np.array_equal(saved["mean"], scaler.mean_) or not np.array_equal(saved["scale"], scaler.scale_):
        raise ValueError("Candidate standardization differs from training-only statistics")
    auc = float(roc_auc_score(dev_y, linear_logits(dev_features, saved)))
    if auc != choice["development_auroc"]:
        raise ValueError("Candidate development score changed")
    return choice, auc


def fit_candidate(candidate_path: Path, marker_path: Path, fingerprint: dict[str, Any], c: float,
                  scaler: StandardScaler, train: tuple[np.ndarray, np.ndarray],
                  development: tuple[np.ndarray, np.ndarray]) -> tuple[dict[str, Any], float]:
    """
    Fit, score and save one regularization candidate.

    Parameters
    ----------
    candidate_path : Path
        Destination of the probe parameters.
    marker_path : Path
        Destination of the candidate receipt.
    fingerprint : dict[str, Any]
        Probe fingerprint.
    c : float
        Inverse regularization strength.
    scaler : StandardScaler
        Training-only standardization.
    train : tuple[np.ndarray, np.ndarray]
        Standardized training features and labels.
    development : tuple[np.ndarray, np.ndarray]
        Raw development features and labels.

    Returns
    -------
    tuple[dict[str, Any], float]
        Candidate receipt and its development AUROC.
    """
    train_x, train_y = train
    dev_features, dev_y = development
    started = time.monotonic()
    estimator = LogisticRegression(C=c, max_iter=MAX_ITER, solver="lbfgs", random_state=SEED)
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(train_x, train_y)
    parameters = {"mean": scaler.mean_, "scale": scaler.scale_, "coefficient": estimator.coef_,
                  "intercept": estimator.intercept_}
    auc = float(roc_auc_score(dev_y, linear_logits(dev_features, parameters)))
    atomic_npz(candidate_path, **parameters)
    choice = {"fingerprint": fingerprint, "C": c, "development_auroc": auc,
              "iterations": estimator.n_iter_.tolist(), "seconds": time.monotonic() - started,
              "convergence_warnings": [str(message.message) for message in messages],
              "sha256": sha256_file(candidate_path)}
    write_json_atomic(marker_path, choice)
    return choice, auc


def fit_probe(features: np.ndarray, feature_rows: list[dict[str, str]], rows: dict[str, Any], directory: Path,
              fingerprint: dict[str, Any], resume: bool = False,
              c_values: Sequence[float] = C_VALUES) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """
    Select a regularization value on development patients; candidates are resumable.

    The scaler and classifier see labeled training rows only.

    Parameters
    ----------
    features : np.ndarray
        Arm features aligned to ``feature_rows``.
    feature_rows : list[dict[str, str]]
        Feature-row identities.
    rows : dict[str, Any]
        Budget partitions with ``labeled_train`` and ``development``.
    directory : Path
        Probe directory.
    fingerprint : dict[str, Any]
        Probe fingerprint.
    resume : bool
        Reuse candidates saved by an earlier invocation.
    c_values : Sequence[float]
        Ascending regularization grid; the first value wins ties.

    Returns
    -------
    tuple[dict[str, np.ndarray], dict[str, Any]]
        Selected probe parameters and the selection receipt.

    Raises
    ------
    ValueError
        If the grid is empty or saved candidates changed.
    FileExistsError
        If candidates exist without ``resume``.
    """
    if not c_values:
        raise ValueError("At least one regularization value is required")
    directory.mkdir(parents=True, exist_ok=True)
    train_indices = supervised_indices(feature_rows, rows["labeled_train"], "train")
    dev_indices = supervised_indices(feature_rows, rows["development"], "validation")
    train_y = np.asarray([int(row["target"]) for row in rows["labeled_train"]])
    dev_y = np.asarray([int(row["target"]) for row in rows["development"]])
    train_raw = np.asarray(features[train_indices], dtype=np.float64)
    scaler = StandardScaler().fit(train_raw)
    train_x = scaler.transform(train_raw)
    dev_features = features[dev_indices]
    choices, best_auc, best_path, best_c = [], -1.0, None, None
    for index, c in enumerate(c_values):
        candidate_path = directory / f"candidate_{index}.npz"
        marker_path = directory / f"candidate_{index}.json"
        if marker_path.exists():
            if not resume:
                raise FileExistsError("Prior classifier candidates exist; use --resume")
            choice, auc = load_candidate(candidate_path, marker_path, fingerprint, c, scaler,
                                         dev_features, dev_y)
        else:
            choice, auc = fit_candidate(candidate_path, marker_path, fingerprint, c, scaler,
                                        (train_x, train_y), (dev_features, dev_y))
            print(json.dumps({"stage": "mismatch_probe", "directory": str(directory), "C": c,
                              "development_auroc": auc}), flush=True)
        choices.append({key: value for key, value in choice.items() if key not in ("fingerprint", "sha256")})
        if auc > best_auc:
            best_auc, best_path, best_c = auc, candidate_path, c
    with np.load(best_path) as selected:
        arrays = {key: selected[key].copy() for key in selected.files}
    atomic_npz(directory / "linear_model.npz", **arrays)
    selection = {"C": best_c, "best_development_auroc": best_auc, "candidates": choices,
                 "tie_rule": "First C in ascending grid wins ties", "selection_split": "development only"}
    write_json_atomic(directory / "selection.json", selection)
    return arrays, selection


def train_probe(args: argparse.Namespace, branches: np.ndarray, feature_rows: list[dict[str, str]],
                rows: dict[str, Any], feature_metadata: dict[str, Any], identity: dict[str, Any], arm: str,
                budget: str) -> None:
    """
    Fit and evaluate one arm's frozen linear probe, or verify it when complete.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``output_dir``, ``bootstrap``, ``threads`` and ``resume``.
    branches : np.ndarray
        Completed ``[rows, 3, 512]`` features.
    feature_rows : list[dict[str, str]]
        Feature-row identities.
    rows : dict[str, Any]
        Budget partitions.
    feature_metadata : dict[str, Any]
        Feature completion metadata.
    identity : dict[str, Any]
        Input identity.
    arm : str
        One of ``ARMS``.
    budget : str
        Key of ``BUDGETS``.

    Raises
    ------
    ValueError
        If completed or partial outputs belong to another configuration.
    FileExistsError
        If partial outputs exist without ``--resume``.
    """
    directory = args.output_dir / f"{arm}_{budget}_seed42"
    fingerprint = {"inputs": identity, "features_sha256": feature_metadata["sha256"],
                   "arm": arm, "budget": budget, "seed": SEED, "C_values": list(C_VALUES),
                   "solver": "lbfgs", "max_iter": MAX_ITER,
                   "scaler": "StandardScaler fit on labeled training rows only",
                   "bootstrap": args.bootstrap, "sklearn_version": sklearn.__version__,
                   "threads": args.threads}
    completion = directory / "complete.json"
    if completion.exists():
        saved = json.loads(completion.read_text())
        if (saved["fingerprint"] != fingerprint
                or saved["sha256"] != {name: sha256_file(directory / name) for name in ARTIFACTS}):
            raise ValueError("Completed classifier artifacts or configuration differ")
        return
    if directory.exists() and any(directory.iterdir()) and not args.resume:
        raise FileExistsError(f"Incomplete classifier exists; use --resume: {directory}")
    config_path = directory / "config.json"
    if config_path.exists() and json.loads(config_path.read_text())["fingerprint"] != fingerprint:
        raise ValueError("Partial classifier configuration differs")
    directory.mkdir(parents=True, exist_ok=True)
    features = features_for_arm(branches, arm)
    config = {"fingerprint": fingerprint, "arm": arm, "feature_dimension": features.shape[1],
              "records": {name: len(rows[name])
                          for name in ("labeled_train", "development", "calibration", "test")},
              "encoder_frozen": True, "normalization": "Frozen Experiment 004 training-pool mean/std",
              "retained_target_positions": "7..78 in each 79-token half",
              "prediction_query_positions": "3..74",
              "task": "PTB-XL diagnostic abnormality proxy; exploratory previously inspected test cohort"}
    write_json_atomic(config_path, config)
    started = time.monotonic()
    fitted, selection = fit_probe(features, feature_rows, rows, directory, fingerprint, resume=args.resume)
    calibration_indices = supervised_indices(feature_rows, rows["calibration"], "validation")
    test_indices = supervised_indices(feature_rows, rows["test"], "test")
    evaluate_predictions(f"cpc_mismatch_{arm}_{budget}", linear_logits(features[calibration_indices], fitted),
                         linear_logits(features[test_indices], fitted), rows["calibration"], rows["test"],
                         directory, SEED, args.bootstrap)
    config.update({"C": selection["C"], "best_development_auroc": selection["best_development_auroc"],
                   "seconds_this_invocation": time.monotonic() - started})
    write_json_atomic(config_path, config)
    write_json_atomic(completion, {"fingerprint": fingerprint,
                                   "sha256": {name: sha256_file(directory / name) for name in ARTIFACTS}})


def report(args: argparse.Namespace) -> None:
    """
    Write the study report and its completion receipt once all six probes finish.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``output_dir`` and ``bootstrap``.

    Raises
    ------
    RuntimeError
        If a probe is incomplete.
    """
    comparisons = {}
    lines = ["# Experiment 009: frozen CPC prediction mismatch", "",
             "All arms reuse the same completed 20-epoch local CPC encoder, frozen input normalization, "
             "and temporal positions. No encoder updates or additional SSL training were performed.", "",
             "The primary comparison is residual minus ordinary local features: both classifiers receive "
             "1024 dimensions. The 512-dimensional context-only reference tests whether either extra branch "
             "adds useful information.", ""]
    for budget in BUDGETS:
        paths = {arm: args.output_dir / f"{arm}_{budget}_seed42" for arm in ARMS}
        if not all((path / "complete.json").is_file() for path in paths.values()):
            raise RuntimeError("All six frozen probes must complete before the study report")
        lines += [f"## {budget}", "", "| Arm | Dimensions | AUROC | AP | Sensitivity | Specificity | Brier |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for arm, path in paths.items():
            metrics = json.loads((path / "metrics.json").read_text())["test"]
            config = json.loads((path / "config.json").read_text())
            lines.append(f"| {arm} | {config['feature_dimension']} | {metrics['auroc']:.4f} | "
                         f"{metrics['average_precision']:.4f} | {metrics['sensitivity']:.4f} | "
                         f"{metrics['specificity']:.4f} | {metrics['brier']:.4f} |")
        lines += ["", "| Paired comparison | AUROC difference | Patient bootstrap 95% interval |",
                  "| --- | ---: | --- |"]
        for left, right in COMPARISONS:
            comparison = paired_comparison(paths[left], paths[right], args.bootstrap)
            comparisons[f"{budget}_{left}_minus_{right}"] = comparison
            auc = comparison["auroc"]
            lines.append(f"| {left} minus {right} | {auc['difference']:+.4f} | "
                         f"[{auc['ci95'][0]:+.4f}, {auc['ci95'][1]:+.4f}] |")
        lines.append("")
    lines += ["Regularization is selected on development patients. Platt calibration and the >=95% "
              "sensitivity threshold use separate calibration patients. Each threshold is frozen before test "
              "evaluation.", "",
              "The residual subtracts normalized predicted tokens from normalized observed tokens; its norm "
              "is retained. A large mismatch can reflect noise or artifacts, so this feature is not a "
              "clinical abnormality score. One seed and a previously used test cohort support exploratory "
              "conclusions only. Paired intervals describe patient sampling uncertainty, not retraining "
              "variation.", ""]
    write_json_atomic(args.output_dir / "paired_comparisons.json", comparisons)
    (args.output_dir / "report.md").write_text("\n".join(lines))
    write_json_atomic(args.output_dir / "complete.json", {
        "sha256": {name: sha256_file(args.output_dir / name)
                   for name in ("report.md", "paired_comparisons.json")},
        "probe_completions": {f"{arm}_{budget}":
                              sha256_file(args.output_dir / f"{arm}_{budget}_seed42/complete.json")
                              for arm in ARMS for budget in BUDGETS}})


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

    Raises
    ------
    RuntimeError
        If CUDA extraction is requested but unavailable.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("check", "profile", "extract", "train", "all"), default="all")
    parser.add_argument("--cache-dir", type=Path, default=cpc_pool.DEFAULT_CACHE)
    parser.add_argument("--manifest-root", type=Path, default=cpc_pool.DEFAULT_MANIFEST)
    parser.add_argument("--bootstrap-dir", type=Path, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if min(args.batch_size, args.threads, args.bootstrap) < 1:
        parser.error("Batch size, threads and bootstrap must be positive")
    extracting = args.stage in ("profile", "extract", "all")
    if extracting and args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA extraction requested but unavailable")
    return args


def run_extraction(args: argparse.Namespace, pool: cpc_pool.Pool, model: CPCPretrainer, mean: np.ndarray,
                   std: np.ndarray, rows: list[dict[str, str]], identity: dict[str, Any]) -> None:
    """
    Profile when needed and extract features while holding the GPU without waiting.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    pool : cpc_pool.Pool
        Frozen CPC waveform cache.
    model : CPCPretrainer
        Frozen model; moved back to CPU afterwards.
    mean : np.ndarray
        Per-lead mean.
    std : np.ndarray
        Per-lead standard deviation.
    rows : list[dict[str, str]]
        Extraction cohort.
    identity : dict[str, Any]
        Input identity.
    """
    with gpu_lock(args.device, blocking=False):
        profile_path = args.output_dir / "profile.json"
        if (args.stage == "profile" or not profile_path.is_file()
                or json.loads(profile_path.read_text()).get("fingerprint")
                != extraction_fingerprint(args, identity)):
            profile(args, pool, model, mean, std, rows, identity)
        if args.stage in ("extract", "all"):
            extract(args, pool, model, mean, std, rows, identity)
    model.to("cpu")
    if args.device == "cuda":
        torch.cuda.empty_cache()


def main(argv: Sequence[str] | None = None) -> None:
    """
    Run the requested Experiment 009 stages under a per-output runner lock.

    Parameters
    ----------
    argv : Sequence[str] | None
        Arguments, or ``None`` for ``sys.argv``.
    """
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "runner.lock").open("a+") as lock, threadpool_limits(limits=args.threads):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pool, model, mean, std, manifests, rows, identity = prepare_inputs(args)
        write_json_atomic(args.output_dir / "preflight.json", {
            "identity": identity, "feature_records": len(rows),
            "budgets": {name: len(value["labeled_train"]) for name, value in manifests.items()},
            "encoder_frozen": not any(p.requires_grad for p in model.parameters())})
        print(json.dumps({"stage": "mismatch_check", "records": len(rows), "arms": list(ARMS)}), flush=True)
        if args.stage == "check":
            return
        if args.stage in ("profile", "extract", "all"):
            run_extraction(args, pool, model, mean, std, rows, identity)
        if args.stage in ("train", "all"):
            branches, metadata = load_features(args, identity, rows)
            for budget in BUDGETS:
                for arm in ARMS:
                    train_probe(args, branches, rows, manifests[budget], metadata, identity, arm, budget)
            report(args)


if __name__ == "__main__":
    main()
