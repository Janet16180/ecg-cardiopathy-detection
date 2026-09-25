"""Cohort and source verification for the clean Experiment 017 replication."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ecg_experiment.bounded_waveform_cache import BoundedWaveformCache
from ecg_experiment.cpc_input_audit import historical_resample
from ecg_experiment.cpc_morphology import SUPPORT, TEMPLATES
from ecg_experiment.cpc_pool import Pool
from ecg_experiment.evaluation import LIMITED_LABELS
from ecg_experiment.files import read_csv, sha256_file, sha256_json
from ecg_experiment.pilot import (
    BATCH,
    BUDGETS,
    SAVE_EVERY,
    Partitions,
    check_final_ssl,
    file_identity,
    load_partitions,
)
from ecg_experiment.training_dataset import TrainingECGDataset

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/processed/cpc_pool_40k"
MANIFEST = ROOT / "data/processed/ptbxl"
SSL = ROOT / "outputs/experiment004_cpc_40k/cpc_ssl/encoder.pt"
NORMALIZATION = ROOT / "outputs/experiment004_cpc_40k/normalization.json"
OUTPUT = ROOT / "outputs/experiment017_clean_replication_v1"
CLEAN = ROOT / "outputs/data_quality/clean_rerun_preflight_v1"
CANONICAL = ROOT / "data/processed/training_union_500hz_v1"
POOL_VERIFICATION = ROOT / "outputs/experiment015_jepa_cpc_distillation/provenance/verification.json"
POOL_VERIFIER = "scripts/run_jepa_cpc_distillation.py"
SOURCE_ARCHIVE = ROOT / "outputs/refactor_pause/source_before_refactor.tar.gz"
PAUSE_RECEIPT = ROOT / "outputs/refactor_pause/pause.json"
POOL_FILES = (("signals.npy", "signals_sha256"), ("rows.csv", "rows_sha256"),
              ("ecg_ids.npy", "ecg_ids_sha256"))
SEED = 42
SEEDS = (42, 43)
FULL_CLEAN_LABELS = 15359
# Offset of the common first-dropout stream; see make_model.
DROPOUT_SEED_OFFSET = 12345
TEMPLATE_SEED_OFFSET = 17017
EPOCHS = 5
ENCODER_LR = 3e-4
HEAD_LR = 1e-3
WEIGHT_DECAY = 0.01
KINDS = ("none", "conv", "template")
HALF_SAMPLES = 1250
INITIAL_BEST = {"auc": -1.0, "epoch": 0}
ARTIFACTS = {"history.json": "history_sha256", "best_model.pt": "best_model_sha256",
             "best_logits.npz": "best_logits_sha256"}
PLANNING_GATE_SECONDS = 7200
REQUIRED_GAIN = 0.002
SENSITIVITY_TOLERANCE = 0.005
CODE_PATHS = ("scripts/experiments/run_cpc_morphology017_clean.py", "ecg_experiment/morphology_clean.py",
              "ecg_experiment/morphology_clean_inputs.py", "ecg_experiment/morphology_clean_audit.py",
              "ecg_experiment/cpc_input_audit.py", "ecg_experiment/training_dataset.py",
              "ecg_experiment/cpc_morphology.py",
              "ecg_experiment/cpc.py", "ecg_experiment/bounded_waveform_cache.py",
              "scripts/experiments/run_cpc_experiment.py", "ecg_experiment/data.py",
              "ecg_experiment/cpc_pool.py", "ecg_experiment/files.py",
              "ecg_experiment/reproducibility.py", "ecg_experiment/pilot.py", "ecg_experiment/receipts.py",
              "ecg_experiment/training.py",
              "ecg_experiment/run.py", "ecg_experiment/evaluation.py",
              "docs/experiment-017-clean-replication.md")


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
    mean : np.ndarray
        Train-only per-lead mean shaped ``[12, 1]``.
    std : np.ndarray
        Train-only per-lead standard deviation shaped ``[12, 1]``.
    provenance : dict[str, Any]
        Source, code and settings identity of the experiment.
    pool_receipt_sha256 : str
        Digest of the Experiment 015 pool receipt, recorded for audit.
    precheck_seconds : float
        Wall time of input verification.
    waveforms : BoundedWaveformCache | None
        Training then development waveforms in RAM.
    bank : torch.Tensor | None
        Initial template bank.
    template_receipt : list[dict[str, Any]] | None
        Source window of each initial template.
    preload_seconds : float
        Wall time of the preload.
    """

    pool: Pool
    partitions: Partitions
    mean: np.ndarray
    std: np.ndarray
    provenance: dict[str, Any]
    pool_receipt_sha256: str
    precheck_seconds: float
    waveforms: BoundedWaveformCache | None = None
    bank: torch.Tensor | None = None
    template_receipt: list[dict[str, Any]] | None = None
    preload_seconds: float = 0.0
    seed: int = SEED


def verified_pool_hashes(args: argparse.Namespace,
                         pool: Pool) -> tuple[dict[str, str], dict[str, dict[str, int]], str, str]:
    """
    Reuse Experiment 015's complete pool hash check while every source file is unchanged.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``cache_dir`` and ``pool_verification``.
    pool : Pool
        Frozen CPC waveform cache.

    Returns
    -------
    tuple[dict[str, str], dict[str, dict[str, int]], str, str]
        Pool content digests, pool file identities, the verifier source digest
        and the receipt digest.

    Raises
    ------
    FileNotFoundError
        If the receipt is missing.
    ValueError
        If the receipt, its verifier source or any pool file changed.
    """
    if not args.pool_verification.exists():
        raise FileNotFoundError(f"Verified immutable CPC pool receipt is missing: {args.pool_verification}")
    before = {name: file_identity(args.cache_dir / name) for name, _ in POOL_FILES}
    receipt = json.loads(args.pool_verification.read_text())
    if (receipt.get("stage") != "check"
            or receipt.get("fingerprint") != sha256_json(receipt.get("provenance"))):
        raise ValueError("Invalid pool verification receipt")
    source_hash = receipt.get("provenance", {}).get("code", {}).get(POOL_VERIFIER)
    pause = json.loads(PAUSE_RECEIPT.read_text())
    if sha256_file(SOURCE_ARCHIVE) != pause["source_archive_sha256"]:
        raise ValueError("Historical source archive changed")
    with tarfile.open(SOURCE_ARCHIVE, "r:gz") as archive:
        source = archive.extractfile(POOL_VERIFIER)
        if source is None or hashlib.sha256(source.read()).hexdigest() != source_hash:
            raise ValueError("Archived pool verifier source differs from frozen receipt")
    hashes = receipt["provenance"]["pool_content_sha256"]
    for filename, key in POOL_FILES:
        if (receipt["pool_file_stats"].get(filename) != before[filename]
                or hashes.get(filename) != pool.metadata[key]):
            raise ValueError(f"CPC pool differs from verified receipt: {filename}")
    if any(file_identity(args.cache_dir / name) != before[name] for name in before):
        raise ValueError("CPC pool changed while checking receipt")
    return hashes, before, source_hash, sha256_file(args.pool_verification)


def template_windows(cache: Any, historical_full: list[dict[str, str]], mean: np.ndarray,
                     std: np.ndarray) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """
    Draw seed-fixed, label-blind template windows from training rows in normalized units.

    Parameters
    ----------
    cache : Any
        Cache whose first ``train_count`` rows are training records.
    train_count : int
        Number of training rows eligible for templates.
    mean : np.ndarray
        Per-lead mean shaped ``[12, 1]``.
    std : np.ndarray
        Per-lead standard deviation shaped ``[12, 1]``.

    Returns
    -------
    tuple[torch.Tensor, list[dict[str, Any]]]
        ``[TEMPLATES, 12, SUPPORT]`` bank and the source window of each template.

    Raises
    ------
    ValueError
        If a window has nonfinite samples.
    """
    rng = np.random.default_rng(SEED + TEMPLATE_SEED_OFFSET)
    rows = rng.choice(len(historical_full), size=TEMPLATES, replace=False)
    halves = rng.integers(0, 2, size=TEMPLATES)
    ends = rng.integers(SUPPORT - 1, HALF_SAMPLES, size=TEMPLATES)
    bank = np.empty((TEMPLATES, 12, SUPPORT), dtype=np.float32)
    receipt = []
    clean_index = {row["ecg_id"]: index for index, row in enumerate(cache.rows)}
    for i, (row, half, end) in enumerate(zip(rows, halves, ends, strict=True)):
        donor_id = historical_full[int(row)]["ecg_id"]
        if donor_id not in clean_index:
            raise ValueError(f"Historical template donor was excluded: {donor_id}")
        clean_row = clean_index[donor_id]
        start = int(half) * HALF_SAMPLES + int(end) - SUPPORT + 1
        window = np.array(cache.signals[clean_row, :, start:start + SUPPORT], copy=True)
        window -= mean
        window /= std
        if not np.isfinite(window).all():
            raise ValueError("Nonfinite initial template")
        bank[i] = window
        receipt.append({"ecg_id": donor_id, "half": int(half), "end_in_half": int(end)})
    return torch.from_numpy(bank), receipt


def load_normalization(path: Path, pool: Pool, signals_sha256: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the Experiment 004 train-only normalization and check its source.

    Parameters
    ----------
    path : Path
        ``normalization.json``.
    pool : Pool
        Frozen CPC waveform cache.
    signals_sha256 : str
        Verified digest of the pool's ``signals.npy``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Per-lead mean and standard deviation shaped ``[12, 1]``.

    Raises
    ------
    ValueError
        If the normalization was fitted elsewhere or is invalid.
    """
    norm = json.loads(path.read_text())
    if (norm["source"]["signals_sha256"] != signals_sha256
            or norm["source"]["train_ids_sha256"] != sha256_json([row["ecg_id"] for row in pool.train_rows])):
        raise ValueError("Normalization not fitted on frozen CPC training pool")
    mean = np.asarray(norm["mean"], dtype=np.float32)[:, None]
    std = np.asarray(norm["std"], dtype=np.float32)[:, None]
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Invalid train-only normalization")
    return mean, std


def check_pool_manifests(pool: Pool, manifest_dir: Path) -> None:
    """
    Check the PTB-XL manifests recorded when the pool was built.

    Parameters
    ----------
    pool : Pool
        Frozen CPC waveform cache.
    manifest_dir : Path
        PTB-XL manifest root.

    Raises
    ------
    ValueError
        If a manifest differs from the pool's record.
    """
    for name, expected in pool.metadata["ptb_manifest_sha256"].items():
        if sha256_file(manifest_dir / "seed42_fraction1" / name) != expected:
            raise ValueError(f"Frozen pool PTB manifest changed: {name}")


def code_hashes(extra_code: Sequence[str] = ()) -> dict[str, str]:
    """
    Digest every repository file that determines the pilot's results.

    Parameters
    ----------
    extra_code : Sequence[str]
        Files added by a versioned entry point.

    Returns
    -------
    dict[str, str]
        SHA-256 keyed by repository-relative path.
    """
    return {name: sha256_file(ROOT / name) for name in (*CODE_PATHS, *extra_code)}


def clean_partitions(pool: Pool, manifest_dir: Path, clean_dir: Path) -> Partitions:
    """Bind the preflight labels to the historical patient split and remove one QC failure."""
    historical = load_partitions(pool, manifest_dir)
    receipt = json.loads((clean_dir / "receipt.json").read_text())
    if receipt.get("status") != "passed_cpu_preflight_no_training" or receipt.get("train_count") != 56809:
        raise ValueError("Clean preflight receipt is invalid")
    for name, digest in receipt["output_sha256"].items():
        if sha256_file(clean_dir / name) != digest:
            raise ValueError(f"Clean preflight output changed: {name}")
    excluded = read_csv(clean_dir / "exclusions.csv")
    excluded_ptb = {row["record_id"] for row in excluded if row["source"] == "ptbxl"}
    if excluded_ptb != {"ptbxl:12722"} or len(excluded) != 66:
        raise ValueError("Unexpected clean exclusion set")
    full = [row for row in historical.full if f"ptbxl:{row['ecg_id']}" not in excluded_ptb]
    limited = historical.limited
    for budget, selected in (("1", full), ("0.1", limited)):
        clean = read_csv(clean_dir / f"labels_fraction{budget}.csv")
        expected = [{"record_id": f"ptbxl:{row['ecg_id']}",
                     "patient_id": f"ptbxl:{row['patient_id']}", "target": row["target"]}
                    for row in selected]
        if clean != expected:
            raise ValueError(f"Clean label table differs from frozen selection: {budget}")
    if len(full) != FULL_CLEAN_LABELS or len(limited) != LIMITED_LABELS:
        raise ValueError("Clean label counts changed")
    return Partitions(full, limited, historical.development, historical.calibration, historical.test)


def clean_exposed_labels(partitions: Partitions, budget: str) -> tuple[np.ndarray, np.ndarray]:
    """Expose the fixed clean labels while masking all other training targets."""
    limited = {row["ecg_id"] for row in partitions.limited}
    exposed = np.asarray([budget == "1" or row["ecg_id"] in limited for row in partitions.full], dtype=bool)
    targets = np.asarray([float(row["target"]) if mask else 0.0
                          for row, mask in zip(partitions.full, exposed, strict=True)], dtype=np.float32)
    expected = FULL_CLEAN_LABELS if budget == "1" else LIMITED_LABELS
    if len(exposed) != FULL_CLEAN_LABELS or int(exposed.sum()) != expected:
        raise ValueError("Clean exposed-label count changed")
    return exposed, targets


def verify_clean_transform(data: PilotData, canonical_dir: Path) -> dict[str, Any]:
    """Compare every retained labeled raw ECG with its historical CPC cache transform."""
    canonical = TrainingECGDataset(canonical_dir, purpose="supervised", label_budget="1")
    if len(canonical) != FULL_CLEAN_LABELS:
        raise ValueError("Canonical clean supervised count changed")
    historical = {f"ptbxl:{row['ecg_id']}": row for row in data.partitions.full}
    if {row["record_id"] for row in canonical.rows} != set(historical):
        raise ValueError("Canonical and frozen clean identities differ")
    matched = 0
    maxima = []
    rms_values = []
    donor_records = {f"ptbxl:{row['ecg_id']}" for row in (data.template_receipt or [])}
    donor_hashes = {}
    for index, row in enumerate(canonical.rows):
        source = historical[row["record_id"]]
        item = canonical[index]
        if item["patient_id"] != f"ptbxl:{source['patient_id']}":
            raise ValueError("Canonical patient identity changed")
        reduced = historical_resample(item["signal"])
        maxima.append(float(np.max(np.abs(item["signal"]))))
        rms_values.append(float(np.sqrt(np.mean(np.square(item["signal"], dtype=np.float64)))))
        if row["record_id"] in donor_records:
            donor_hashes[row["record_id"]] = row["signal_sha256"]
        cached = np.asarray(data.pool.signals[data.pool.index[source["ecg_id"]]])
        if not np.array_equal(reduced, cached):
            raise ValueError(f"Canonical transform differs: {row['record_id']}")
        matched += 1
    shards = {row["shard"] for row in canonical.rows}
    return {"matched_records": matched, "bitwise_equal": True,
            "canonical_metadata_sha256": sha256_file(canonical_dir / "metadata.json"),
            "shard_file_identity": {name: file_identity(canonical_dir / name) for name in sorted(shards)},
            "train_maxabs_p99_mV": float(np.quantile(maxima, 0.99)),
            "train_rms_p99_mV": float(np.quantile(rms_values, 0.99)),
            "donor_signal_sha256": donor_hashes}


def require_clean_transform(output_dir: Path, canonical_dir: Path) -> None:
    """Reject cached transform evidence if any verified source shard changed."""
    receipt = json.loads((output_dir / "provenance/verification.json").read_text())
    transform = receipt.get("clean_transform", {})
    if transform.get("matched_records") != FULL_CLEAN_LABELS or not transform.get("bitwise_equal"):
        raise ValueError("Complete clean transform verification is missing")
    if transform.get("canonical_metadata_sha256") != sha256_file(canonical_dir / "metadata.json"):
        raise ValueError("Canonical metadata changed after transform verification")
    for name, expected in transform["shard_file_identity"].items():
        if file_identity(canonical_dir / name) != expected:
            raise ValueError(f"Canonical shard changed after transform verification: {name}")


def load_inputs(args: argparse.Namespace, extra_code: Sequence[str] = ()) -> PilotData:
    """
    Verify every pilot input and build the experiment provenance.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    extra_code : Sequence[str]
        Additional repository files whose digests enter the provenance.

    Returns
    -------
    PilotData
        Verified inputs, without preloaded waveforms.
    """
    began = time.monotonic()
    pool = Pool(args.cache_dir)
    partitions = clean_partitions(pool, args.manifest_dir, args.clean_dir)
    ssl_config = check_final_ssl(args.ssl)
    hashes, pool_stats, verifier_source_hash, pool_receipt_hash = verified_pool_hashes(args, pool)
    mean, std = load_normalization(args.normalization, pool, hashes["signals.npy"])
    manifest_paths = [args.manifest_dir / f"seed42_fraction{budget}" / f"{name}.csv"
                      for budget in BUDGETS for name in ("labeled_train", "validation", "test")]
    manifest_paths.append(args.manifest_dir / "seed42_fraction1/all_train_ssl.csv")
    check_pool_manifests(pool, args.manifest_dir)
    sources = [args.ssl, args.ssl.parent / "epoch_state.pt", args.ssl.parent / "config.json",
               PAUSE_RECEIPT, SOURCE_ARCHIVE,
               args.normalization, args.clean_dir / "receipt.json",
               *(args.clean_dir / name for name in ("train_manifest.csv", "labels_fraction1.csv",
                 "labels_fraction0.1.csv", "exclusions.csv", "heldout_references.csv")),
               args.canonical_dir / "metadata.json", *manifest_paths]
    provenance = {"sources": {str(path.resolve()): sha256_file(path) for path in sources},
                  "code": code_hashes(extra_code),
                  "pool_complete_sha256": sha256_file(args.cache_dir / "complete.json"),
                  "pool_content_sha256": hashes,
                  "pool_file_stats": pool_stats,
                  "pool_verifier_source_sha256": verifier_source_hash,
                  "ssl_fingerprint": ssl_config["fingerprint"],
                  "settings": {"seeds": SEEDS, "bank_seed": SEED, "epochs": EPOCHS, "batch": BATCH,
                               "encoder_lr": ENCODER_LR, "head_and_branch_lr": HEAD_LR,
                               "weight_decay": WEIGHT_DECAY, "templates": TEMPLATES,
                               "support_samples": SUPPORT, "save_every": SAVE_EVERY}}
    return PilotData(pool, partitions, mean, std, provenance, pool_receipt_hash, time.monotonic() - began)


def preload(args: argparse.Namespace, data: PilotData) -> None:
    """
    Load training and development waveforms into RAM and draw the template bank.

    Parameters
    ----------
    args : argparse.Namespace
        Needs ``max_cache_bytes`` and ``reserve_bytes``.
    data : PilotData
        Verified inputs, updated in place.
    """
    started = time.monotonic()
    rows = data.partitions.full + data.partitions.development
    data.waveforms = BoundedWaveformCache(data.pool, rows, max_bytes=args.max_cache_bytes,
                                          reserve_bytes=args.reserve_bytes, expected_source="ptbxl")
    historical = read_csv(args.manifest_dir / "seed42_fraction1/labeled_train.csv")
    data.bank, data.template_receipt = template_windows(data.waveforms, historical,
                                                        data.mean, data.std)
    data.preload_seconds = time.monotonic() - started
