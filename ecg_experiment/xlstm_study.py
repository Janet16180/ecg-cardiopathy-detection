"""Matched GRU/xLSTM CPC study on clean cohort v3 signals."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from ecg_experiment import ROOT
from ecg_experiment.cohort_cpc_cache import CohortCPCDataset, build_cohort_cpc_cache, verify_cohort_cpc_cache
from ecg_experiment.cpc_local_readout import fit_head, replay_logits
from ecg_experiment.cpc_pool import Pool, PoolDataset
from ecg_experiment.cpc_scaling_readout import CLEAN, VALIDATION, Example, training_examples
from ecg_experiment.cpc_subset25 import exposure_order
from ecg_experiment.cpc_xlstm import create_model
from ecg_experiment.files import sha256_file, sha256_json, write_json_atomic, write_torch_atomic
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.intervals import paired_auroc_difference
from ecg_experiment.paths import to_stored
from ecg_experiment.reproducibility import capture_rng_state, cpu_state, restore_rng_state, seed_everything
from ecg_experiment.training import checked_step

SEED = 38042
ORDER_SEED = 38043
INTERVAL_SEED = 38045
BATCH = 128
EXPOSURES = 250_000
UPDATES = math.ceil(EXPOSURES / BATCH)
PROFILE_UPDATES = 24
CEILING_SECONDS = 7200.0
OUTPUT_NAME = "experiment038_cpc_xlstm"
PRIOR = "experiment019_cpc_25k_readout_v1"
PRIOR_RESULT_SHA = "afd880190501d356e53b06d2194f0783cf8347915a05c5826552dbc532f0e060"
PRIOR_PREDICTIONS_SHA = "768350222511e4856213017c25a33a4251ccc7a3c598b1d7b5dfb16df76a5f00"
ARMS = ("gru", "xlstm")
RUNTIME_CONFIGURED = False
SOURCE_FILES = (
    "docs/experiment-038-cpc-xlstm.md",
    "ecg_experiment/cpc.py",
    "ecg_experiment/cpc_xlstm.py",
    "ecg_experiment/cohort_cpc_cache.py",
    "ecg_experiment/cpc_input_audit.py",
    "ecg_experiment/public_sources.py",
    "ecg_experiment/waveforms.py",
    "ecg_experiment/files.py",
    "ecg_experiment/xlstm_study.py",
    "ecg_experiment/cpc_pool.py",
    "ecg_experiment/cpc_scaling_readout.py",
    "ecg_experiment/cpc_local_readout.py",
    "ecg_experiment/cpc_subset25.py",
    "ecg_experiment/intervals.py",
    "ecg_experiment/gpu.py",
    "ecg_experiment/paths.py",
    "ecg_experiment/reproducibility.py",
    "ecg_experiment/training.py",
    "scripts/experiments/run_cpc_xlstm038.py",
    "pyproject.toml",
    "uv.lock",
)


def configure_runtime() -> None:
    """Use the deterministic float32, single-thread training recipe."""
    global RUNTIME_CONFIGURED
    if RUNTIME_CONFIGURED:
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    RUNTIME_CONFIGURED = True


def output(root: Path, tier: int) -> Path:
    """Return the tier's output directory after checking its identity.

    Parameters
    ----------
    root : Path
        Repository root.
    tier : int
        Curated cohort size in thousands.

    Returns
    -------
    Path
        Output directory for this tier.
    """
    if tier not in (25, 50):
        raise ValueError("Tier must be 25 or 50")
    return root / "outputs" / OUTPUT_NAME / f"{tier}k"


def used_seconds(root: Path, tier: int) -> float:
    """Sum actual completed and failed stage attempts for one tier.

    Parameters
    ----------
    root : Path
        Repository root.
    tier : int
        Curated cohort size in thousands.

    Returns
    -------
    float
        Charged elapsed seconds, including an external cache build.
    """
    path = output(root, tier) / "stage_walltime.json"
    attempts = json.loads(path.read_text())["attempts"] if path.exists() else []
    charged = float(sum(item["elapsed_seconds"] for item in attempts))
    cache = root / f"data/processed/clean_{tier}k_v3_cpc/complete.json"
    if cache.exists() and not any(item.get("cache_build_included", False) for item in attempts):
        charged += float(json.loads(cache.read_text())["elapsed_seconds"])
    return charged


def record_stage(root: Path, tier: int, stage: str, seconds: float,
                 status: str, *, cache_build_included: bool = False) -> None:
    """Append actual elapsed time, including failed attempts, atomically.

    Parameters
    ----------
    root : Path
        Repository root.
    tier : int
        Curated cohort size in thousands.
    stage : str
        Stage name.
    seconds : float
        Measured wall time for this attempt.
    status : str
        ``complete`` or ``failed``.
    cache_build_included : bool
        Whether this attempt built the cache from scratch.
    """
    path = output(root, tier) / "stage_walltime.json"
    attempts = json.loads(path.read_text())["attempts"] if path.exists() else []
    attempts.append({"stage": stage, "elapsed_seconds": seconds, "status": status,
                     "cache_build_included": cache_build_included})
    write_json_atomic(path, {"attempts": attempts, "total_seconds": sum(
        item["elapsed_seconds"] for item in attempts)}, sort_keys=True)


def pool(root: Path) -> Pool:
    """Open the historical PTB cache metadata without reading held-out signals.

    Parameters
    ----------
    root : Path
        Repository root.

    Returns
    -------
    Pool
        Historical CPC cache handle.
    """
    return Pool(root / "data/processed/cpc_pool_40k")


def normalizer(root: Path) -> Path:
    """Return the Experiment 004 train-only normalizer.

    Parameters
    ----------
    root : Path
        Repository root.

    Returns
    -------
    Path
        Historical normalizer receipt.
    """
    return root / "outputs/experiment004_cpc_40k/normalization.json"


def gpu_identity() -> dict[str, Any]:
    """Identify the chosen visible GPU, including UUID when exposed by PyTorch.

    Returns
    -------
    dict[str, Any]
        Device index, name, memory and available hardware identifiers.
    """
    if not torch.cuda.is_available():
        return {"device": "none"}
    index = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(index)
    return {"index": index, "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
            "uuid": str(getattr(properties, "uuid", None)),
            "pci_bus_id": str(getattr(properties, "pci_bus_id", None))}


def protocol_commit(root: Path) -> str:
    """Return the stable protocol commit after checking its frozen bytes.

    Parameters
    ----------
    root : Path
        Repository root.

    Returns
    -------
    str
        Commit that last changed the protocol.
    """
    relative = "docs/experiment-038-cpc-xlstm.md"
    committed = subprocess.run(
        ["git", "show", f"HEAD:{relative}"], cwd=root, check=True, capture_output=True
    ).stdout
    if committed != (root / relative).read_bytes():
        raise ValueError("Experiment 038 protocol must be committed unchanged before any score")
    return subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", relative],
        cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def identity(root: Path, tier: int, cache_receipt: dict[str, Any]) -> dict[str, Any]:
    """Pin scientific source files and local data inputs before scoring.

    Parameters
    ----------
    root : Path
        Repository root.
    tier : int
        Curated cohort size in thousands.
    cache_receipt : dict[str, Any]
        Fully verified CPC cache completion receipt.

    Returns
    -------
    dict[str, Any]
        Executable identity of code, data and environment.
    """
    commit = protocol_commit(root)
    tier_dir = root / f"data/processed/clean_{tier}k_v3"
    pool_dir = root / "data/processed/cpc_pool_40k"
    data_files = (
        tier_dir / "metadata.json", tier_dir / "train_manifest.csv",
        root / f"data/processed/clean_{tier}k_v3_cpc/complete.json",
        normalizer(root), pool_dir / "complete.json", pool_dir / "rows.csv", pool_dir / "ecg_ids.npy",
        root / "outputs/data_quality/clean_cohorts_v3/receipt.json",
        CLEAN / "labels_fraction1.csv", CLEAN / "labels_fraction0.1.csv", VALIDATION,
    )
    paths = [root / name for name in SOURCE_FILES] + list(data_files)
    hashes = {to_stored(path): sha256_file(path) for path in paths}
    old_complete = json.loads((pool_dir / "complete.json").read_text())
    if sha256_file(pool_dir / "signals.npy") != old_complete["signals_sha256"]:
        raise ValueError("Historical PTB signal cache changed")
    for filename, key in (("rows.csv", "rows_sha256"), ("ecg_ids.npy", "ecg_ids_sha256")):
        if hashes[to_stored(pool_dir / filename)] != old_complete[key]:
            raise ValueError(f"Historical PTB {filename} changed")
    if cache_receipt["normalization_sha256"] != hashes[to_stored(normalizer(root))]:
        raise ValueError("CPC cache and historical normalizer disagree")
    return {
        "protocol_commit": commit, "files_sha256": hashes,
        "cache_signals_sha256": cache_receipt["signals_sha256"],
        "historical_pool_signals_sha256": old_complete["signals_sha256"],
        "prior_result_sha256": PRIOR_RESULT_SHA,
        "prior_predictions_sha256": PRIOR_PREDICTIONS_SHA,
        "seed": SEED, "order_seed": ORDER_SEED, "interval_seed": INTERVAL_SEED,
        "exposures": EXPOSURES, "batch": BATCH,
        "torch": torch.__version__,
        "gpu": gpu_identity(),
    }


def ensure_manifest(root: Path, tier: int,
                    receipt: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create once, then reject source, cache or environment drift.

    Parameters
    ----------
    root : Path
        Repository root.
    tier : int
        Curated cohort size in thousands.
    receipt : dict[str, Any] | None
        Already verified cache receipt, or None to verify now.

    Returns
    -------
    dict[str, Any]
        Unchanged executable identity.
    """
    checked = receipt if receipt is not None else verify_cohort_cpc_cache(tier, root=root)
    current = identity(root, tier, checked)
    destination = output(root, tier) / "manifest.json"
    if destination.exists():
        if json.loads(destination.read_text()) != current:
            raise ValueError("Experiment 038 source identity changed")
    else:
        write_json_atomic(destination, current, sort_keys=True)
    return current


def metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Compute the two prespecified ranking metrics.

    Parameters
    ----------
    y : np.ndarray
        Aligned binary targets.
    scores : np.ndarray
        Aligned finite probabilities.

    Returns
    -------
    dict[str, float]
        AUROC and average precision.
    """
    if len(y) != len(scores) or not np.isfinite(scores).all():
        raise ValueError("Malformed or nonfinite predictions")
    return {"auroc": float(roc_auc_score(y, scores)),
            "average_precision": float(average_precision_score(y, scores))}


def development_only_examples(old_pool: Pool, train: list[Example]) -> list[Example]:
    """Read target fields only on the prespecified development rows.

    Parameters
    ----------
    old_pool : Pool
        Historical CPC cache with PTB identities and split metadata.
    train : list[Example]
        Clean PTB training examples for the patient-separation check.

    Returns
    -------
    list[Example]
        The 1,306 development rows in historical cache order.
    """
    examples = []
    with VALIDATION.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader([handle.readline()]))
        if header != ["record_id", "patient_id", "split", "target", "raw_path"]:
            raise ValueError("Unexpected heldout reference schema")
        for line in handle:
            prefix = line.split(",", 3)
            if len(prefix) != 4:
                raise ValueError("Malformed heldout reference row")
            if prefix[2] != "development":
                continue
            values = next(csv.reader([line]))
            if len(values) != len(header):
                raise ValueError("Malformed development reference row")
            record_id, patient_id, _, target, _ = values
            ecg_id = record_id.removeprefix("ptbxl:")
            row = old_pool.rows[old_pool.index[ecg_id]]
            if (row["source"] != "ptbxl" or row["split"] != "validation"
                    or f"ptbxl:{row['patient_id']}" != patient_id
                    or target not in ("0", "1")):
                raise ValueError(f"PTB development cache identity mismatch: {record_id}")
            examples.append(Example(record_id, patient_id, int(target), row))
    if (len(examples) != 1306 or len({row.record_id for row in examples}) != len(examples)
            or len({row.patient_id for row in examples}) != 1173
            or {row.patient_id for row in train} & {row.patient_id for row in examples}):
        raise ValueError("Unexpected development cohort identity or patient overlap")
    examples.sort(key=lambda row: old_pool.index[row.cache_row["ecg_id"]])
    return examples


def prior_integrity(root: Path = ROOT) -> dict[str, Any]:
    """Independently replay saved Experiment 019 development metrics.

    Parameters
    ----------
    root : Path
        Repository root.

    Returns
    -------
    dict[str, Any]
        Historical integrity result, never a matched baseline.
    """
    protocol_commit(root)
    directory = root / "outputs" / PRIOR
    result_path = directory / "result.json"
    predictions_path = directory / "development_predictions.npz"
    if sha256_file(result_path) != PRIOR_RESULT_SHA or sha256_file(predictions_path) != PRIOR_PREDICTIONS_SHA:
        raise ValueError("Experiment 019 historical receipt or prediction bytes changed")
    result = json.loads(result_path.read_text())
    if result["status"] != "complete_development_only":
        raise ValueError("Experiment 019 readout was not complete")
    old_pool = pool(root)
    train, _ = training_examples(old_pool)
    dev = development_only_examples(old_pool, train)
    with np.load(predictions_path, allow_pickle=False) as saved:
        expected = {"record_ids", "patient_ids", "targets", "full_25k", "limited_25k"}
        if set(saved.files) != expected:
            raise ValueError("Experiment 019 prediction schema changed")
        if not np.array_equal(saved["record_ids"], [row.record_id for row in dev]):
            raise ValueError("Historical development ECG order changed")
        if not np.array_equal(saved["patient_ids"], [row.patient_id for row in dev]):
            raise ValueError("Historical development patients changed")
        y = np.asarray([row.target for row in dev], dtype=np.int64)
        if not np.array_equal(saved["targets"], y):
            raise ValueError("Historical development targets changed")
        scores = {budget: metrics(y, saved[f"{budget}_25k"]) for budget in ("full", "limited")}
    for budget in ("full", "limited"):
        for metric, value in scores[budget].items():
            if value != result["scores_25k"][budget][metric]:
                raise ValueError(f"Experiment 019 {budget} {metric} replay mismatch")
    return {"status": "passed_historical_integrity", "scores": scores,
            "result_sha256": PRIOR_RESULT_SHA, "predictions_sha256": PRIOR_PREDICTIONS_SHA,
            "development_records": len(dev), "development_patients": len({row.patient_id for row in dev})}


def prepare(root: Path, tier: int) -> dict[str, Any]:
    """Replay history, build the cache, and write the executable manifest.

    Parameters
    ----------
    root : Path
        Repository root.
    tier : int
        Curated cohort size in thousands.

    Returns
    -------
    dict[str, Any]
        Frozen executable identity.
    """
    prior = prior_integrity(root)
    if tier == 50:
        require_50k_trigger(root)
    cache_path = root / f"data/processed/clean_{tier}k_v3_cpc"
    if not cache_path.exists():
        build_cohort_cpc_cache(tier, root=root)
    receipt = verify_cohort_cpc_cache(tier, root=root)
    current = ensure_manifest(root, tier, receipt)
    write_json_atomic(output(root, tier) / "prior_integrity.json", prior, sort_keys=True)
    return current


def require_50k_trigger(root: Path) -> None:
    """Permit 50k only after the prespecified audited 25k decision.

    Parameters
    ----------
    root : Path
        Repository root holding the completed 25k output.
    """
    prior = output(root, 25)
    result_path = prior / "result.json"
    result = json.loads(result_path.read_text())
    receipt = json.loads((prior / "audit.json").read_text())
    if (receipt["status"] != "passed_development_only"
            or receipt["result_sha256"] != sha256_file(result_path)
            or result["predictions_sha256"] != sha256_file(prior / "development_predictions.npz")):
        raise ValueError("An unchanged audited 25k result is required before 50k")
    if result["status"] != "complete_development_only" or not result["run_50k"]:
        raise ValueError("The prespecified 25k result did not trigger 50k")


def order(tier: int) -> np.ndarray:
    """Return the fixed record exposure order for either cohort size.

    Parameters
    ----------
    tier : int
        Curated cohort size in thousands.

    Returns
    -------
    np.ndarray
        Exactly 250,000 seeded record indices.
    """
    return exposure_order(np.arange(tier * 1000, dtype=np.int64), EXPOSURES, ORDER_SEED)


def dataset(root: Path, tier: int) -> CohortCPCDataset:
    """Load the verified signal cache into RAM once per process.

    Parameters
    ----------
    root : Path
        Repository root.
    tier : int
        Curated cohort size in thousands.

    Returns
    -------
    CohortCPCDataset
        Verified normalized waveform provider.
    """
    data = CohortCPCDataset(tier, root=root, load_into_ram=True)
    if len(data) != tier * 1000:
        raise ValueError("Unexpected cohort cache size")
    return data


def batch(data: CohortCPCDataset, rows: np.ndarray, device: str) -> torch.Tensor:
    """Normalize the selected records using the cache dataset contract.

    Parameters
    ----------
    data : CohortCPCDataset
        Verified cache loaded into RAM.
    rows : np.ndarray
        Selected row indices in exposure order.
    device : str
        PyTorch device.

    Returns
    -------
    torch.Tensor
        Float32 batch with shape [records, 12, 2500].
    """
    signal = torch.stack([data[int(index)][0] for index in rows])
    if signal.dtype != torch.float32 or signal.shape[1:] != (12, 2500) or not torch.isfinite(signal).all():
        raise ValueError("Malformed normalized CPC batch")
    return signal.to(device, non_blocking=device == "cuda")


def step(model: torch.nn.Module, optimizer: torch.optim.Optimizer, signal: torch.Tensor) -> float:
    """Take one clipped ordinary-CPC AdamW update.

    Parameters
    ----------
    model : torch.nn.Module
        CPC pretrainer for the chosen context arm.
    optimizer : torch.optim.Optimizer
        Fixed AdamW optimizer.
    signal : torch.Tensor
        Normalized ECG batch.

    Returns
    -------
    float
        Finite ordinary-CPC loss before the update.
    """
    loss, parts = model(signal)
    if parts["cmsc"] != 0.0:
        raise ValueError("Experiment 038 must use ordinary CPC without CMSC")
    checked_step(loss, model, optimizer, "Experiment 038 CPC", clip=True)
    return float(loss.detach())


def optimizer(model: torch.nn.Module) -> torch.optim.AdamW:
    """Construct the fixed optimizer.

    Parameters
    ----------
    model : torch.nn.Module
        CPC pretrainer to update.

    Returns
    -------
    torch.optim.AdamW
        AdamW with the frozen learning rate and weight decay.
    """
    return torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)


def tensor_hash(state: dict[str, torch.Tensor], prefixes: tuple[str, ...]) -> str:
    """Hash a named tensor group in stable key order.

    Parameters
    ----------
    state : dict[str, torch.Tensor]
        Model state dictionary.
    prefixes : tuple[str, ...]
        Names identifying the tensor group.

    Returns
    -------
    str
        SHA-256 of tensor names, shapes, dtypes and bytes.
    """
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        if key.startswith(prefixes):
            value = tensor.detach().cpu().contiguous().numpy()
            digest.update(key.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(value.shape).encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def group_hashes(model: torch.nn.Module) -> dict[str, str]:
    """Hash CNN, CPC heads and context separately.

    Parameters
    ----------
    model : torch.nn.Module
        CPC pretrainer.

    Returns
    -------
    dict[str, str]
        One digest per trainable scientific group.
    """
    state = model.state_dict()
    return {"shared_convs": tensor_hash(state, ("encoder.convs.",)),
            "shared_heads": tensor_hash(state, ("heads.",)),
            "context": tensor_hash(state, ("encoder.context.",))}


def sync(device: str) -> None:
    """Synchronize timed CUDA work.

    Parameters
    ----------
    device : str
        PyTorch device name.
    """
    if device == "cuda":
        torch.cuda.synchronize()


def checkpoint(model: torch.nn.Module, opt: torch.optim.Optimizer, current: dict[str, Any],
               updates: int, exposures: int, losses: list[float], initial: dict[str, str]) -> dict[str, Any]:
    """Capture an exact resume point including optimizer and all RNG states.

    Parameters
    ----------
    model : torch.nn.Module
        Model at the checkpoint boundary.
    opt : torch.optim.Optimizer
        Optimizer at the same boundary.
    current : dict[str, Any]
        Executable identity.
    updates : int
        Number of completed optimizer steps.
    exposures : int
        Number of records consumed.
    losses : list[float]
        All losses through the boundary.
    initial : dict[str, str]
        Initial tensor-group hashes.

    Returns
    -------
    dict[str, Any]
        Exact model, optimizer, RNG and schedule state.
    """
    return {"model": cpu_state(model), "optimizer": opt.state_dict(),
            "rng": capture_rng_state(), "identity_sha256": sha256_json(current),
            "updates": updates, "exposures": exposures, "losses": losses, "initial": initial}


def restore(model: torch.nn.Module, opt: torch.optim.Optimizer, saved: dict[str, Any],
            current: dict[str, Any]) -> None:
    """Restore an exact resume point after checking source identity.

    Parameters
    ----------
    model : torch.nn.Module
        Newly constructed matched model.
    opt : torch.optim.Optimizer
        Newly constructed optimizer.
    saved : dict[str, Any]
        Saved checkpoint state.
    current : dict[str, Any]
        Current executable identity to compare.
    """
    if saved["identity_sha256"] != sha256_json(current):
        raise ValueError("Checkpoint identity changed")
    model.load_state_dict(saved["model"], strict=True)
    opt.load_state_dict(saved["optimizer"])
    restore_rng_state(saved["rng"])


def same_state(first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]) -> bool:
    """Require bitwise equality of all model tensors.

    Parameters
    ----------
    first, second : dict[str, torch.Tensor]
        Model state dictionaries to compare.

    Returns
    -------
    bool
        Whether all keys and tensor bytes match.
    """
    return first.keys() == second.keys() and all(torch.equal(first[key], second[key]) for key in first)


def decision(contrast: dict[str, float | int]) -> bool:
    """Apply the prespecified minimum point gain and positive paired interval.

    Parameters
    ----------
    contrast : dict[str, float | int]
        Paired patient AUROC contrast.

    Returns
    -------
    bool
        Whether the arm or cohort gain meets both thresholds.
    """
    return bool(contrast["difference"] >= 0.005 and contrast["ci_low"] > 0)


def tree_equal(first: Any, second: Any) -> bool:
    """Compare checkpoint trees, including tensors and NumPy RNG arrays.

    Parameters
    ----------
    first, second : Any
        Nested checkpoint components to compare.

    Returns
    -------
    bool
        Whether every leaf is exactly equal.
    """
    if isinstance(first, torch.Tensor):
        return isinstance(second, torch.Tensor) and torch.equal(first.cpu(), second.cpu())
    if isinstance(first, np.ndarray):
        return isinstance(second, np.ndarray) and np.array_equal(first, second)
    if isinstance(first, dict):
        return isinstance(second, dict) and first.keys() == second.keys() and all(
            tree_equal(first[key], second[key]) for key in first)
    if isinstance(first, (tuple, list)):
        return type(first) is type(second) and len(first) == len(second) and all(
            tree_equal(a, b) for a, b in zip(first, second, strict=True))
    return first == second


def _profile_one(root: Path, tier: int, arm: str, data: CohortCPCDataset,
                 current: dict[str, Any], device: str) -> dict[str, Any]:
    """Time 24 actual updates and replay the next update from a saved state.

    Parameters
    ----------
    root : Path
        Repository root.
    tier : int
        Curated cohort size in thousands.
    arm : str
        ``gru`` or ``xlstm``.
    data : CohortCPCDataset
        Verified cache loaded into RAM.
    current : dict[str, Any]
        Executable identity.
    device : str
        Profiled CUDA device.

    Returns
    -------
    dict[str, Any]
        Measured updates, checkpoint timing and exact replay result.
    """
    seed_everything(SEED)
    model = create_model(arm, SEED, device)
    opt = optimizer(model)
    initial = group_hashes(model)
    indices = order(tier)
    torch.cuda.reset_peak_memory_stats()
    began = time.monotonic()
    losses = []
    for update in range(PROFILE_UPDATES):
        rows = indices[update * BATCH:(update + 1) * BATCH]
        losses.append(step(model, opt, batch(data, rows, device)))
    sync(device)
    update_seconds = time.monotonic() - began
    saved = checkpoint(model, opt, current, PROFILE_UPDATES, PROFILE_UPDATES * BATCH, losses, initial)
    temporary = output(root, tier) / f"profile_resume_{arm}.pt"
    save_began = time.monotonic()
    write_torch_atomic(temporary, saved)
    loaded = torch.load(temporary, map_location="cpu", weights_only=False)
    checkpoint_seconds = time.monotonic() - save_began
    rows = indices[PROFILE_UPDATES * BATCH:(PROFILE_UPDATES + 1) * BATCH]
    signal = batch(data, rows, device)
    loss_a = step(model, opt, signal)
    sync(device)
    state_a = cpu_state(model)
    optimizer_a = opt.state_dict()
    rng_a = capture_rng_state()
    restore(model, opt, loaded, current)
    loss_b = step(model, opt, signal)
    sync(device)
    if (loss_a != loss_b or not same_state(state_a, cpu_state(model))
            or not tree_equal(optimizer_a, opt.state_dict())
            or not tree_equal(rng_a, capture_rng_state())):
        raise RuntimeError(f"Exact checkpoint replay failed for {arm}")
    temporary.unlink()
    return {"arm": arm, "initial": initial, "parameter_count": sum(p.numel() for p in model.parameters()),
            "updates": PROFILE_UPDATES, "losses": losses, "update_seconds": update_seconds,
            "seconds_per_update": update_seconds / PROFILE_UPDATES,
            "checkpoint_seconds": checkpoint_seconds, "next_update_replay_equal": True,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated()}


@torch.inference_mode()
def extract_features(model: torch.nn.Module, examples: list[Any], old_pool: Pool,
                     root: Path, device: str) -> np.ndarray:
    """Extract the fixed 512-wide mean/max context features.

    Parameters
    ----------
    model : torch.nn.Module
        Frozen CPC pretrainer whose encoder supplies contexts.
    examples : list[Any]
        Allowed PTB training or development rows.
    old_pool : Pool
        Historical 250 Hz CPC signal cache.
    root : Path
        Repository root.
    device : str
        Profiled CUDA device.

    Returns
    -------
    np.ndarray
        Finite float32 features aligned to examples.
    """
    values = json.loads(normalizer(root).read_text())
    mean = np.asarray(values["mean"], dtype=np.float32)
    std = np.asarray(values["std"], dtype=np.float32)
    if mean.shape != (12,) or std.shape != (12,) or np.any(std <= 0):
        raise ValueError("Malformed historical normalizer")
    loader = DataLoader(PoolDataset(old_pool, [row.cache_row for row in examples], mean, std),
                        batch_size=BATCH, shuffle=False, num_workers=0, drop_last=False)
    chunks = []
    model.eval()
    for signal, _, _ in loader:
        _, contexts = model.encoder(signal.to(device, non_blocking=device == "cuda"))
        chunks.append(model.encoder.pooled(contexts).float().cpu().numpy())
    features = np.concatenate(chunks)
    if features.shape != (len(examples), 512) or not np.isfinite(features).all():
        raise ValueError("Malformed or nonfinite readout features")
    return features


def profile(root: Path, tier: int, device: str = "cuda") -> dict[str, Any]:
    """Run real GPU preflight, update and extraction timings before training.

    Parameters
    ----------
    root : Path
        Repository root.
    tier : int
        Curated cohort size in thousands.
    device : str
        CUDA device, required for the profile gate.

    Returns
    -------
    dict[str, Any]
        Measured profile and conservative tier projection.
    """
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("A real CUDA profile is required before Experiment 038 training")
    if tier == 50:
        require_50k_trigger(root)
    started = time.monotonic()
    data = dataset(root, tier)
    current = ensure_manifest(root, tier, data.receipt)
    preflight_seconds = time.monotonic() - started
    cache_receipt = data.receipt
    if (not np.isfinite(np.asarray(data[0][0])).all()
            or not np.isfinite(np.asarray(data[len(data) - 1][0])).all()):
        raise ValueError("Nonfinite cache preflight signals")
    old_pool = pool(root)
    train, _ = training_examples(old_pool)
    with gpu_lock("cuda", blocking=False):
        arms = {arm: _profile_one(root, tier, arm, data, current, device) for arm in ARMS}
        if any(arms["gru"]["initial"][key] != arms["xlstm"]["initial"][key]
               for key in ("shared_convs", "shared_heads")):
            raise ValueError("The two arms did not start from identical CNN/head tensors")
        extract_seconds = {}
        for arm in ARMS:
            model = create_model(arm, SEED, device)
            begin = time.monotonic()
            features = extract_features(model, train[:512], old_pool, root, device)
            sync(device)
            extract_seconds[arm] = time.monotonic() - begin
            if features.shape != (512, 512):
                raise ValueError("Feature profile width changed")
    total_records = len(train) + 1306
    projected_training = 1.5 * UPDATES * sum(item["seconds_per_update"] for item in arms.values())
    projected_extraction = 1.5 * sum(
        math.ceil(total_records / 512) * seconds for seconds in extract_seconds.values())
    projected_checkpoints = 1.5 * math.ceil(UPDATES / 100) * sum(
        item["checkpoint_seconds"] for item in arms.values())
    profile_seconds = time.monotonic() - started
    charged_prepare = used_seconds(root, tier)
    projected = (charged_prepare + 2 * preflight_seconds + profile_seconds
                 + projected_training + projected_extraction + projected_checkpoints + 900)
    receipt = {"status": "profiled", "tier": tier, "identity_sha256": sha256_json(current),
               "cache_elapsed_seconds": cache_receipt["elapsed_seconds"],
               "charged_prior_stage_seconds": charged_prepare,
               "preflight_seconds": preflight_seconds, "profile_wall_seconds": profile_seconds,
               "arms": arms, "feature_profile_records": 512,
               "feature_profile_seconds": extract_seconds,
               "projected_training_seconds": projected_training,
               "projected_feature_seconds": projected_extraction,
               "projected_checkpoint_seconds": projected_checkpoints,
               "cpu_report_reserve_seconds": 900,
               "projected_total_seconds": projected, "ceiling_seconds": CEILING_SECONDS,
               "gate_passed": projected <= CEILING_SECONDS,
               "gpu_identity": gpu_identity(), "torch_version": torch.__version__}
    write_json_atomic(output(root, tier) / "profile.json", receipt, sort_keys=True)
    return receipt


def _passed_profile(root: Path, tier: int, current: dict[str, Any]) -> dict[str, Any]:
    """Load a passed real-GPU profile for the unchanged execution identity.

    Parameters
    ----------
    root : Path
        Repository root.
    tier : int
        Curated cohort size in thousands.
    current : dict[str, Any]
        Current executable identity.

    Returns
    -------
    dict[str, Any]
        Passed profile receipt.
    """
    receipt = json.loads((output(root, tier) / "profile.json").read_text())
    if (receipt["identity_sha256"] != sha256_json(current) or not receipt["gate_passed"]
            or receipt["gpu_identity"] != gpu_identity()):
        raise ValueError("Matching passed real-GPU profile is required")
    return receipt


def _elapsed_guard(charged_before: float, active_since: float) -> None:
    """Enforce the 7,200-second ceiling on completed and active work.

    Parameters
    ----------
    charged_before : float
        Elapsed tier work before this stage, including external cache build.
    active_since : float
        Monotonic start of the active stage.
    """
    if charged_before + time.monotonic() - active_since > CEILING_SECONDS:
        raise RuntimeError("Experiment 038 tier exceeded its 7200-second work ceiling")


def _pace_guard(prof: dict[str, Any], charged_before: float, stage_started: float,
                arm: str, update: int, seconds_per_update: float) -> None:
    """Stop when observed training pace cannot fit the remaining tier work.

    Parameters
    ----------
    prof : dict[str, Any]
        Passed real-GPU profile.
    charged_before : float
        Actual tier time before training.
    stage_started : float
        Monotonic start of the active train stage.
    arm : str
        Current context arm.
    update : int
        Completed updates in this arm.
    seconds_per_update : float
        Observed time per update on this launch.
    """
    profiled = prof["arms"]
    remaining = (UPDATES - update) * max(seconds_per_update, profiled[arm]["seconds_per_update"])
    if arm == "gru":
        remaining += UPDATES * profiled["xlstm"]["seconds_per_update"]
    projection = (charged_before + time.monotonic() - stage_started
                  + 1.5 * remaining + prof["projected_feature_seconds"]
                  + prof["projected_checkpoint_seconds"] + prof["cpu_report_reserve_seconds"])
    if projection > CEILING_SECONDS:
        raise RuntimeError("Measured Experiment 038 training pace exceeded the 7200-second gate")


def _train_arm(root: Path, tier: int, arm: str, data: CohortCPCDataset, current: dict[str, Any],
               profile_receipt: dict[str, Any], charged_before: float,
               stage_started: float, device: str) -> dict[str, Any]:
    """Train or exactly resume one arm through the fixed exposure budget.

    Parameters
    ----------
    root : Path
        Repository root.
    tier : int
        Curated cohort size in thousands.
    arm : str
        Current context arm.
    data : CohortCPCDataset
        Verified cache loaded into RAM.
    current : dict[str, Any]
        Executable identity.
    profile_receipt : dict[str, Any]
        Passed real-GPU timing profile.
    charged_before : float
        Actual tier time before training.
    stage_started : float
        Monotonic start of the train stage.
    device : str
        Profiled CUDA device.

    Returns
    -------
    dict[str, Any]
        Completed arm timing, counts and checkpoint hash.
    """
    started = time.monotonic()
    seed_everything(SEED)
    model = create_model(arm, SEED, device)
    opt = optimizer(model)
    initial = group_hashes(model)
    latest = output(root, tier) / arm / "latest.pt"
    latest.parent.mkdir(parents=True, exist_ok=True)
    losses: list[float] = []
    updates = 0
    exposures = 0
    previous_seconds = 0.0
    if latest.exists():
        saved = torch.load(latest, map_location="cpu", weights_only=False)
        restore(model, opt, saved, current)
        updates, exposures = saved["updates"], saved["exposures"]
        losses, previous_seconds = saved["losses"], saved.get("elapsed_seconds", 0.0)
        if initial != saved["initial"] or exposures != min(updates * BATCH, EXPOSURES):
            raise ValueError("Inconsistent resume checkpoint")
    first_update = updates
    indices = order(tier)
    for update in range(updates, UPDATES):
        _elapsed_guard(charged_before, stage_started)
        rows = indices[update * BATCH:min((update + 1) * BATCH, EXPOSURES)]
        losses.append(step(model, opt, batch(data, rows, device)))
        updates, exposures = update + 1, min((update + 1) * BATCH, EXPOSURES)
        if updates % 100 == 0 or updates == UPDATES:
            sync(device)
            measured = time.monotonic() - started
            if updates > first_update:
                _pace_guard(profile_receipt, charged_before, stage_started,
                            arm, updates, measured / (updates - first_update))
            saved = checkpoint(model, opt, current, updates, exposures, losses, initial)
            saved["elapsed_seconds"] = previous_seconds + time.monotonic() - started
            write_torch_atomic(latest, saved)
            print(json.dumps({"tier": tier, "arm": arm, "updates": updates,
                              "exposures": exposures, "loss": losses[-1],
                              "elapsed_seconds": saved["elapsed_seconds"]}), flush=True)
    if updates != UPDATES or exposures != EXPOSURES or len(losses) != UPDATES:
        raise ValueError("Incomplete exposure budget")
    _elapsed_guard(charged_before, stage_started)
    final = group_hashes(model)
    if any(final[key] == initial[key] for key in final):
        raise ValueError(f"{arm} has an unchanged encoder/head tensor group")
    return {"status": "trained", "arm": arm, "updates": updates, "exposures": exposures,
            "initial": initial, "final": final, "first_loss": losses[0], "last_loss": losses[-1],
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "elapsed_seconds": previous_seconds + time.monotonic() - started,
            "checkpoint": to_stored(latest), "checkpoint_sha256": sha256_file(latest)}


def train(root: Path, tier: int, device: str = "cuda") -> dict[str, Any]:
    """Complete matched fixed-exposure training after a passed profile.

    Parameters
    ----------
    root : Path
        Repository root.
    tier : int
        Curated cohort size in thousands.
    device : str
        Profiled CUDA device.

    Returns
    -------
    dict[str, Any]
        Completion receipt for both matched arms.
    """
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Experiment 038 requires CUDA training after a real GPU profile")
    if tier == 50:
        require_50k_trigger(root)
    stage_started = time.monotonic()
    data = dataset(root, tier)
    current = ensure_manifest(root, tier, data.receipt)
    prof = _passed_profile(root, tier, current)
    charged_before = max(used_seconds(root, tier),
                         prof["charged_prior_stage_seconds"] + prof["profile_wall_seconds"])
    with gpu_lock("cuda", blocking=False):
        previous = 0.0
        arms = {}
        for arm in ARMS:
            arms[arm] = _train_arm(root, tier, arm, data, current, prof,
                                   charged_before, stage_started, device)
            previous += arms[arm]["elapsed_seconds"]
            write_json_atomic(output(root, tier) / "training.json",
                              {"status": "partial" if arm == "gru" else "complete",
                               "identity_sha256": sha256_json(current), "arms": arms,
                               "total_seconds": previous}, sort_keys=True)
    _elapsed_guard(charged_before, stage_started)
    return json.loads((output(root, tier) / "training.json").read_text())


def _training_receipt(root: Path, tier: int, current: dict[str, Any]) -> dict[str, Any]:
    """Verify both completed checkpoints against the execution identity.

    Parameters
    ----------
    root : Path
        Repository root.
    tier : int
        Curated cohort size in thousands.
    current : dict[str, Any]
        Executable identity.

    Returns
    -------
    dict[str, Any]
        Complete training receipt.
    """
    receipt = json.loads((output(root, tier) / "training.json").read_text())
    if receipt["status"] != "complete" or receipt["identity_sha256"] != sha256_json(current):
        raise ValueError("Matching completed training is required")
    for arm in ARMS:
        item = receipt["arms"][arm]
        path = root / item["checkpoint"]
        if (item["updates"] != UPDATES or item["exposures"] != EXPOSURES
                or sha256_file(path) != item["checkpoint_sha256"]):
            raise ValueError(f"Incomplete or changed {arm} checkpoint")
    if any(receipt["arms"]["gru"]["initial"][key]
           != receipt["arms"]["xlstm"]["initial"][key]
           for key in ("shared_convs", "shared_heads")):
        raise ValueError("Unmatched initial CNN/head tensors")
    return receipt


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write one compressed array receipt through a local temporary path.

    Parameters
    ----------
    path : Path
        Output archive path.
    arrays : dict[str, np.ndarray]
        Named aligned arrays to save.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _readout_predictions(root: Path, tier: int) -> dict[str, np.ndarray]:
    """Load saved development predictions for an audited cohort contrast.

    Parameters
    ----------
    root : Path
        Repository root.
    tier : int
        Cohort size in thousands.

    Returns
    -------
    dict[str, np.ndarray]
        Aligned saved identities, targets and model probabilities.
    """
    path = output(root, tier) / "development_predictions.npz"
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def _contrasts(root: Path, tier: int, dev: list[Any], predictions: dict[str, np.ndarray]
               ) -> dict[str, dict[str, dict[str, float | int]]]:
    """Compute paired whole-patient AUROC architecture and scaling contrasts.

    Parameters
    ----------
    root : Path
        Repository root.
    tier : int
        Cohort size in thousands.
    dev : list[Any]
        Ordered development-only examples.
    predictions : dict[str, np.ndarray]
        Current tier development probabilities.

    Returns
    -------
    dict[str, dict[str, dict[str, float | int]]]
        Limited- and full-label paired intervals.
    """
    patients = np.asarray([row.patient_id for row in dev])
    y = np.asarray([row.target for row in dev], dtype=np.int64)
    result = {}
    if tier == 50:
        require_50k_trigger(root)
    prior = _readout_predictions(root, 25) if tier == 50 else None
    if prior is not None and any(not np.array_equal(prior[key], predictions[key])
                                 for key in ("record_ids", "patient_ids", "targets")):
        raise ValueError("25k and 50k development rows differ")
    for budget in ("limited", "full"):
        scores = {
            "xlstm_minus_gru": (predictions[f"{budget}_xlstm"], predictions[f"{budget}_gru"]),
        }
        if prior is not None:
            scores["xlstm50_minus_xlstm25"] = (
                predictions[f"{budget}_xlstm"], prior[f"{budget}_xlstm"])
        result[budget] = {name: paired_auroc_difference(patients, y, first, second,
                           draws=2000, seed=INTERVAL_SEED)
                          for name, (first, second) in scores.items()}
    return result


def _fit_readout_heads(
    features: dict[str, np.ndarray], train_rows: list[Example], limited: set[str],
    predictions: dict[str, np.ndarray], charged_before: float, started: float,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, dict[str, float | int]]]]:
    """Fit the fixed heads and replay development probabilities in budget/arm order.

    Parameters
    ----------
    features : dict[str, np.ndarray]
        Aligned frozen train and development feature matrices.
    train_rows : list[Example]
        Clean PTB training labels in cache order.
    limited : set[str]
        Fixed 1,518-record training subset.
    predictions : dict[str, np.ndarray]
        Development identities and targets, updated with probabilities.
    charged_before : float
        Elapsed tier work before readout.
    started : float
        Monotonic start of readout.

    Returns
    -------
    tuple[dict[str, np.ndarray], dict[str, dict[str, dict[str, float | int]]]]
        Saved head parameters and development ranking scores.
    """
    parameters: dict[str, np.ndarray] = {}
    scores: dict[str, dict[str, dict[str, float | int]]] = {budget: {} for budget in ("limited", "full")}
    y_train = np.asarray([row.target for row in train_rows], dtype=np.int64)
    y_dev = predictions["targets"]
    for budget in ("limited", "full"):
        selected = (np.asarray([i for i, row in enumerate(train_rows)
                                if row.record_id in limited], dtype=np.int64)
                    if budget == "limited" else np.arange(len(train_rows)))
        if len(selected) != (1518 if budget == "limited" else 15359):
            raise ValueError("Readout training label count changed")
        for arm in ARMS:
            head = fit_head(features[f"train_{arm}"][selected].astype(np.float64), y_train[selected])
            x_dev = features[f"development_{arm}"].astype(np.float64)
            probabilities = expit(replay_logits(x_dev, head))
            direct = head["model"].predict_proba(head["scaler"].transform(x_dev))[:, 1]
            if not np.allclose(probabilities, direct, rtol=0, atol=1e-10):
                raise ValueError("Saved-head replay differs from classifier")
            predictions[f"{budget}_{arm}"] = probabilities
            scores[budget][arm] = {**metrics(y_dev, probabilities),
                                   "training_labels": len(selected),
                                   "classifier_iterations": head["n_iter"]}
            for name in ("mean", "scale", "coef", "intercept"):
                parameters[f"{budget}_{arm}_{name}"] = np.asarray(head[name])
            _elapsed_guard(charged_before, started)
    return parameters, scores


def readout(root: Path, tier: int, device: str = "cuda") -> dict[str, Any]:
    """Fit fixed train-only heads and score development patients once.

    Parameters
    ----------
    root : Path
        Repository root.
    tier : int
        Cohort size in thousands.
    device : str
        Profiled CUDA device.

    Returns
    -------
    dict[str, Any]
        Development-only readout and prespecified decisions.
    """
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Experiment 038 readout requires the profiled CUDA device")
    if tier == 50:
        require_50k_trigger(root)
    start = time.monotonic()
    if (output(root, tier) / "result.json").exists():
        raise ValueError("Development readout already exists; run audit instead")
    receipt = verify_cohort_cpc_cache(tier, root=root)
    current = ensure_manifest(root, tier, receipt)
    prof = _passed_profile(root, tier, current)
    training = _training_receipt(root, tier, current)
    charged_before = max(used_seconds(root, tier),
                         prof["charged_prior_stage_seconds"] + prof["profile_wall_seconds"]
                         + training["total_seconds"])
    old_pool = pool(root)
    train_rows, limited = training_examples(old_pool)
    dev = development_only_examples(old_pool, train_rows)
    if len(train_rows) != 15359 or len(limited) != 1518 or len(dev) != 1306:
        raise ValueError("Unexpected readout cohort size")
    features = {}
    predictions: dict[str, np.ndarray] = {
        "record_ids": np.asarray([row.record_id for row in dev]),
        "patient_ids": np.asarray([row.patient_id for row in dev]),
        "targets": np.asarray([row.target for row in dev], dtype=np.int64),
    }
    with gpu_lock("cuda", blocking=False):
        for arm in ARMS:
            model = create_model(arm, SEED, device)
            saved = torch.load(root / training["arms"][arm]["checkpoint"],
                               map_location="cpu", weights_only=False)
            model.load_state_dict(saved["model"], strict=True)
            model.eval().requires_grad_(False)
            features[f"train_{arm}"] = extract_features(model, train_rows, old_pool, root, device)
            features[f"development_{arm}"] = extract_features(model, dev, old_pool, root, device)
            _elapsed_guard(charged_before, start)
    parameters, scores = _fit_readout_heads(features, train_rows, limited, predictions,
                                             charged_before, start)
    destination = output(root, tier)
    _save_npz(destination / "features.npz", features)
    _save_npz(destination / "head_parameters.npz", parameters)
    _save_npz(destination / "development_predictions.npz", predictions)
    contrasts = _contrasts(root, tier, dev, predictions)
    _elapsed_guard(charged_before, start)
    primary = contrasts["limited"]["xlstm_minus_gru"]
    result = {
        "status": "complete_development_only", "tier": tier,
        "identity_sha256": sha256_json(current),
        "training_sha256": sha256_file(destination / "training.json"),
        "features_sha256": sha256_file(destination / "features.npz"),
        "head_parameters_sha256": sha256_file(destination / "head_parameters.npz"),
        "predictions_sha256": sha256_file(destination / "development_predictions.npz"),
        "scores": scores, "contrasts": contrasts,
        "primary_budget": "limited", "primary_architecture_contrast": primary,
        "architecture_promising": decision(primary),
        "run_50k": not decision(primary) if tier == 25 else False,
        "scaling_helpful": decision(contrasts["limited"]["xlstm50_minus_xlstm25"])
                           if tier == 50 else None,
        "training_labels": {"limited": 1518, "full": 15359},
        "development_records": len(dev),
        "development_patients": len({row.patient_id for row in dev}),
        "calibration_or_test_scored": False,
        "elapsed_seconds": time.monotonic() - start,
    }
    write_json_atomic(destination / "result.json", result, sort_keys=True)
    return result


def finite_tree(value: Any) -> bool:
    """Check every floating tensor in a nested checkpoint.

    Parameters
    ----------
    value : Any
        Nested checkpoint value.

    Returns
    -------
    bool
        Whether every floating tensor is finite.
    """
    if isinstance(value, torch.Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    return True


def _audit_feature_replay(destination: Path, predictions: dict[str, np.ndarray],
                          result: dict[str, Any]) -> None:
    """Verify saved feature shape, finiteness and exact fitted-head predictions.

    Parameters
    ----------
    destination : Path
        Tier output directory.
    predictions : dict[str, np.ndarray]
        Saved aligned development probabilities and targets.
    result : dict[str, Any]
        Saved development readout result.
    """
    with (np.load(destination / "features.npz", allow_pickle=False) as saved_features,
          np.load(destination / "head_parameters.npz", allow_pickle=False) as heads):
        for arm in ARMS:
            x_train = saved_features[f"train_{arm}"]
            x_dev = saved_features[f"development_{arm}"]
            if (x_train.shape != (15359, 512) or x_dev.shape != (1306, 512)
                    or not np.isfinite(x_train).all() or not np.isfinite(x_dev).all()):
                raise ValueError("Saved features are malformed or nonfinite")
            for budget in ("limited", "full"):
                name = f"{budget}_{arm}"
                head = {key: heads[f"{name}_{key}"] for key in ("mean", "scale", "coef", "intercept")}
                replay = expit(replay_logits(x_dev.astype(np.float64), head))
                if not np.array_equal(replay, predictions[name]):
                    raise ValueError(f"Saved {name} head does not replay predictions")
                expected = metrics(predictions["targets"], replay)
                if any(expected[key] != result["scores"][budget][arm][key] for key in expected):
                    raise ValueError(f"Saved {name} metric mismatch")


def _audit_checkpoint_groups(root: Path, training: dict[str, Any], current: dict[str, Any]) -> None:
    """Verify finite completed states and movement of each encoder/head group.

    Parameters
    ----------
    root : Path
        Repository root.
    training : dict[str, Any]
        Saved complete training receipt.
    current : dict[str, Any]
        Current executable identity.
    """
    for arm in ARMS:
        item = training["arms"][arm]
        saved = torch.load(root / item["checkpoint"], map_location="cpu", weights_only=False)
        if (not finite_tree(saved) or saved["updates"] != UPDATES or saved["exposures"] != EXPOSURES
                or saved["identity_sha256"] != sha256_json(current)
                or item["initial"] != saved["initial"] or len(saved["losses"]) != UPDATES
                or not np.isfinite(saved["losses"]).all()):
            raise ValueError(f"Malformed or nonfinite {arm} checkpoint")
        final = {"shared_convs": tensor_hash(saved["model"], ("encoder.convs.",)),
                 "shared_heads": tensor_hash(saved["model"], ("heads.",)),
                 "context": tensor_hash(saved["model"], ("encoder.context.",))}
        if final != item["final"] or any(final[key] == item["initial"][key] for key in final):
            raise ValueError(f"{arm} tensor group did not change as reported")


def audit(root: Path, tier: int) -> dict[str, Any]:
    """Recompute saved development scores and verify all artifacts.

    Parameters
    ----------
    root : Path
        Repository root.
    tier : int
        Cohort size in thousands.

    Returns
    -------
    dict[str, Any]
        Independent development-only audit receipt.
    """
    start = time.monotonic()
    charged_before = used_seconds(root, tier)
    receipt = verify_cohort_cpc_cache(tier, root=root)
    current = ensure_manifest(root, tier, receipt)
    destination = output(root, tier)
    result = json.loads((destination / "result.json").read_text())
    if result["status"] != "complete_development_only" or result["identity_sha256"] != sha256_json(current):
        raise ValueError("Readout result identity changed")
    training = _training_receipt(root, tier, current)
    if sha256_file(destination / "training.json") != result["training_sha256"]:
        raise ValueError("Training receipt changed after readout")
    for filename, key in (("features.npz", "features_sha256"),
                          ("head_parameters.npz", "head_parameters_sha256"),
                          ("development_predictions.npz", "predictions_sha256")):
        if sha256_file(destination / filename) != result[key]:
            raise ValueError(f"Saved {filename} changed")
    old_pool = pool(root)
    train_rows, limited = training_examples(old_pool)
    dev = development_only_examples(old_pool, train_rows)
    predictions = _readout_predictions(root, tier)
    if (not np.array_equal(predictions["record_ids"], [row.record_id for row in dev])
            or not np.array_equal(predictions["patient_ids"], [row.patient_id for row in dev])
            or not np.array_equal(predictions["targets"], [row.target for row in dev])):
        raise ValueError("Saved predictions do not align with development rows")
    _audit_feature_replay(destination, predictions, result)
    contrasts = _contrasts(root, tier, dev, predictions)
    if contrasts != result["contrasts"] or result["architecture_promising"] != decision(
            contrasts["limited"]["xlstm_minus_gru"]):
        raise ValueError("Saved paired contrast or decision mismatch")
    if tier == 25 and result["run_50k"] != (not result["architecture_promising"]):
        raise ValueError("25k trigger changed")
    if tier == 50 and result["scaling_helpful"] != decision(
            contrasts["limited"]["xlstm50_minus_xlstm25"]):
        raise ValueError("50k scaling decision changed")
    _audit_checkpoint_groups(root, training, current)
    receipt = {"status": "passed_development_only", "tier": tier,
               "result_sha256": sha256_file(destination / "result.json"),
               "development_records": len(dev), "development_patients": 1173,
               "full_training_labels": len(train_rows), "limited_training_labels": len(limited),
               "checkpoint_updates_per_arm": UPDATES, "exposures_per_arm": EXPOSURES,
               "calibration_or_test_scored": False}
    _elapsed_guard(charged_before, start)
    write_json_atomic(destination / "audit.json", receipt, sort_keys=True)
    return receipt
