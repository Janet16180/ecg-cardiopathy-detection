"""Verify, profile, or train Experiment 012 on the frozen 25k ECG subset.

Profile and training use real cache I/O and the same 115,359-record exposure
stream as Experiment 019. This runner never opens development or test labels.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ecg_experiment.cpc_pool import Pool, PoolDataset
from ecg_experiment.cpc_scaling_cached import CachedCPCDataset
from ecg_experiment.cpc_scaling_readout import training_examples
from ecg_experiment.cpc_subset25 import exposure_order, select_indices
from ecg_experiment.cpc_temporal_hybrid import matched_initial_models
from ecg_experiment.files import read_csv, sha256_file, sha256_json, write_json_atomic, write_torch_atomic
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.reproducibility import capture_rng_state, restore_rng_state, seed_everything
from ecg_experiment.training import checked_step

ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / "data/processed/sampled_100k_plus_labels_v1"
CACHE = ROOT / "data/processed/sampled_100k_cpc_cache_v1"
PTB_CACHE = ROOT / "data/processed/cpc_pool_40k"
NORMALIZATION = ROOT / "outputs/experiment004_cpc_40k/normalization.json"
OUTPUT = ROOT / "outputs/experiment012_temporal_hybrid"
SUBSET_SIZE = 25_000
EXPOSURES = 115_359
BATCH = 128
UPDATES = math.ceil(EXPOSURES / BATCH)
SELECTION_SEED = 18046
ORDER_SEED = 18047
MODEL_SEED = 12012
LR = 1e-4
WEIGHT_DECAY = 0.01
PROFILE_BATCHES = 24
PROFILE_READOUT_RECORDS = 512
CEILING_SECONDS = 7_200
SELECTED_SHA256 = "43f405bfbddbb415f5072480744cebe6427afbeed2474a7be0e625cf1deac791"
SOURCE_FILES = (
    "ecg_experiment/cpc_temporal_hybrid.py",
    "ecg_experiment/cpc.py",
    "ecg_experiment/cpc_subset25.py",
    "ecg_experiment/cpc_scaling_cached.py",
    "ecg_experiment/cpc_pool.py",
    "ecg_experiment/cpc_scaling_readout.py",
    "ecg_experiment/cpc_local_readout.py",
    "ecg_experiment/files.py",
    "ecg_experiment/gpu.py",
    "ecg_experiment/reproducibility.py",
    "ecg_experiment/training.py",
    "scripts/experiments/run_cpc_temporal_hybrid012.py",
    "scripts/experiments/run_cpc_temporal_hybrid012_readout.py",
    "docs/experiment-012-temporal-hybrid.md",
)


def selected_indices() -> tuple[np.ndarray, dict[str, int]]:
    """Replay and validate the exact source-stratified Experiment 019 selection."""
    rows = read_csv(COHORT / "train_manifest.csv", required=("record_id", "source", "split"))
    if len(rows) != EXPOSURES or any(row["split"] != "train" for row in rows):
        raise ValueError("Unexpected parent training manifest")
    if len({row["record_id"] for row in rows}) != EXPOSURES:
        raise ValueError("Duplicate source record IDs")
    sources = [row["source"] for row in rows]
    selected = select_indices(sources, SUBSET_SIZE, SELECTION_SEED)
    if sha256_json(selected.tolist()) != SELECTED_SHA256:
        raise ValueError("Experiment 019 selected-index hash mismatch")
    counts = dict(sorted(Counter(sources[index] for index in selected).items()))
    return selected, counts


def identity(selected: np.ndarray) -> tuple[dict[str, Any], float]:
    """Hash training/readout inputs, provenance receipts, and executable sources."""
    cache = json.loads((CACHE / "complete.json").read_text())
    cache_hash = sha256_file(CACHE / "signals.npy")
    if cache_hash != cache["signals_sha256"]:
        raise ValueError("CPC waveform cache hash mismatch")
    paths = {
        "cohort_manifest": COHORT / "train_manifest.csv",
        "cohort_metadata": COHORT / "metadata.json",
        "cache_receipt": CACHE / "complete.json",
        "cache_verification": ROOT / "outputs/experiment018_cpc_data_scaling_v2/cache_verification.json",
        "normalization": NORMALIZATION,
        "experiment019_profile": ROOT / "outputs/experiment019_cpc_25k_v1/profile.json",
    }
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    readout_start = time.monotonic()
    readout_paths = {
        "ptb_receipt": PTB_CACHE / "complete.json",
        "ptb_signals": PTB_CACHE / "signals.npy",
        "ptb_rows": PTB_CACHE / "rows.csv",
        "ptb_ids": PTB_CACHE / "ecg_ids.npy",
        "clean_full_labels": ROOT / "outputs/data_quality/clean_rerun_preflight_v1/labels_fraction1.csv",
        "clean_limited_labels": ROOT / "outputs/data_quality/clean_rerun_preflight_v1/labels_fraction0.1.csv",
        "heldout_references": ROOT / "outputs/data_quality/clean_rerun_preflight_v1/heldout_references.csv",
    }
    hashes.update({name: sha256_file(path) for name, path in readout_paths.items()})
    readout_hash_seconds = time.monotonic() - readout_start
    if (hashes["cohort_manifest"] != cache["identity"]["source_manifest_sha256"]
            or hashes["cohort_metadata"] != cache["identity"]["source_metadata_sha256"]
            or cache["identity"]["record_count"] != EXPOSURES):
        raise ValueError("CPC cache and cohort provenance disagree")
    verification = json.loads(paths["cache_verification"].read_text())
    if (verification["status"] != "passed"
            or verification["cache_sha256"] != cache_hash
            or verification["record_count"] != EXPOSURES):
        raise ValueError("Independent CPC cache verification changed")
    ptb_receipt = json.loads(readout_paths["ptb_receipt"].read_text())
    for key, expected in (("ptb_signals", "signals_sha256"),
                          ("ptb_rows", "rows_sha256"),
                          ("ptb_ids", "ecg_ids_sha256")):
        if hashes[key] != ptb_receipt[expected]:
            raise ValueError(f"Historical PTB cache changed: {key}")
    pool = Pool(PTB_CACHE)
    normalization = json.loads(NORMALIZATION.read_text())
    expected_normalization_source = {
        "train_ids_sha256": sha256_json([row["ecg_id"] for row in pool.train_rows]),
        "rows_sha256": hashes["ptb_rows"],
        "signals_sha256": hashes["ptb_signals"],
        "method": "global per-lead mean and population std, training waveforms only",
    }
    if normalization["source"] != expected_normalization_source:
        raise ValueError("Historical training-only normalization source changed")
    sources = {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
    return {
        "input_sha256": hashes,
        "source_sha256": sources,
        "cached_signals_sha256": cache_hash,
        "selected_indices_sha256": sha256_json(selected.tolist()),
        "subset_size": SUBSET_SIZE,
        "selection_seed": SELECTION_SEED,
        "order_seed": ORDER_SEED,
        "model_seed": MODEL_SEED,
        "record_exposures": EXPOSURES,
        "updates": UPDATES,
        "batch": BATCH,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "objective": "ordinary_causal_cpc_horizons_4_8_12",
    }, readout_hash_seconds


def loader(dataset: CachedCPCDataset, indices: np.ndarray) -> DataLoader:
    """Load cache rows in fixed precomputed order without worker randomness."""
    return DataLoader(Subset(dataset, indices.tolist()), batch_size=BATCH,
                      shuffle=False, num_workers=4, pin_memory=True,
                      generator=torch.Generator().manual_seed(MODEL_SEED))


def measure_input_pass(dataset: CachedCPCDataset, selected: np.ndarray) -> float:
    """Time normalization and cache reads over every selected ECG once."""
    start = time.monotonic()
    records = 0
    for signals, _, _ in loader(dataset, selected):
        records += len(signals)
    if records != SUBSET_SIZE:
        raise ValueError("Incomplete selected-ECG input pass")
    return time.monotonic() - start


def update(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
           batch: tuple[torch.Tensor, torch.Tensor, list[str]], arm: str) -> float:
    """Perform one standard CPC optimizer step."""
    signals, _, _ = batch
    loss, _ = model(signals.cuda(non_blocking=True))
    checked_step(loss, model, optimizer, f"012 {arm}")
    return float(loss.detach())


def initial_models() -> dict[str, torch.nn.Module]:
    """Construct all controls from scratch and with matched shared weights."""
    seed_everything(MODEL_SEED)
    return matched_initial_models(MODEL_SEED)


@torch.inference_mode()
def profile_readout(pool: Pool, rows: list[dict[str, str]],
                    model: torch.nn.Module) -> np.ndarray:
    """Time the same normalized, pooled training-only PTB feature path."""
    normalization = json.loads(NORMALIZATION.read_text())
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    std = np.asarray(normalization["std"], dtype=np.float32)
    data = PoolDataset(pool, rows, mean, std)
    batches = DataLoader(data, batch_size=BATCH, shuffle=False, num_workers=4,
                         pin_memory=True, drop_last=False)
    chunks = []
    model.eval()
    for signals, _, _ in batches:
        _, contexts = model.encoder(signals.cuda(non_blocking=True))
        chunks.append(model.encoder.pooled(contexts).float().cpu().numpy())
    features = np.concatenate(chunks, axis=0)
    if features.shape != (len(rows), 512) or not np.isfinite(features).all():
        raise ValueError("Invalid Experiment 012 profile readout features")
    return features


def profile(dataset: CachedCPCDataset, selected: np.ndarray, order: np.ndarray,
            run_identity: dict[str, Any], counts: dict[str, int],
            preflight_seconds: float, readout_hash_seconds: float) -> None:
    """Measure and gate the complete three-arm training plus readout study."""
    if (OUTPUT / "profile.json").exists():
        raise FileExistsError("An Experiment 012 profile already exists")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    input_pass_seconds = measure_input_pass(dataset, selected)
    pool = Pool(PTB_CACHE)
    train_rows, _ = training_examples(pool)
    profile_rows = [row.cache_row for row in train_rows[:PROFILE_READOUT_RECORDS]]
    arms = {}
    for name, model in initial_models().items():
        model = model.cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        losses = [update(model, optimizer, batch, name)
                  for batch in loader(dataset, order[:PROFILE_BATCHES * BATCH])]
        torch.cuda.synchronize()
        measured = time.monotonic() - start
        checkpoint = OUTPUT / f"{name}_profile.pt"
        checkpoint_start = time.monotonic()
        write_torch_atomic(checkpoint, {"model": model.state_dict(),
                                        "optimizer": optimizer.state_dict(),
                                        "updates": len(losses)})
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["updates"] != len(losses) or any(
            not torch.equal(t.cpu(), saved["model"][key])
            for key, t in model.state_dict().items()
        ):
            raise ValueError(f"{name} profile checkpoint roundtrip failed")
        checkpoint_seconds = time.monotonic() - checkpoint_start
        readout_start = time.monotonic()
        features = profile_readout(pool, profile_rows, model)
        torch.cuda.synchronize()
        readout_seconds = time.monotonic() - readout_start
        arms[name] = {
            "updates": len(losses),
            "seconds": measured,
            "last_loss": losses[-1],
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_roundtrip": True,
            "checkpoint_seconds": checkpoint_seconds,
            "readout_records": len(features),
            "readout_seconds": readout_seconds,
        }
        print(json.dumps({"stage": "profile", "arm": name, **arms[name]}), flush=True)
        del model, optimizer
        torch.cuda.empty_cache()
    input_projection = EXPOSURES / SUBSET_SIZE * input_pass_seconds
    readout_records = len(train_rows) + 1_306
    projected = (2 * preflight_seconds + 2 * readout_hash_seconds
                 + input_pass_seconds
                 + sum(arm["seconds"] + arm["checkpoint_seconds"]
                       + arm["readout_seconds"] for arm in arms.values())
                 + sum(arm["readout_seconds"] for arm in arms.values())
                 + 1.25 * sum(max(UPDATES * arm["seconds"] / arm["updates"],
                                  input_projection)
                              + math.ceil(UPDATES / 100) * arm["checkpoint_seconds"]
                              for arm in arms.values())
                 + 1.5 * readout_records * sum(
                     arm["readout_seconds"] / arm["readout_records"]
                     for arm in arms.values()) + 900 + 180)
    result = {
        "identity": run_identity,
        "source_counts": counts,
        "arms": arms,
        "preflight_seconds_per_launch": preflight_seconds,
        "readout_hash_seconds_per_launch": readout_hash_seconds,
        "projected_readout_records_per_arm": readout_records,
        "selected_input_pass_seconds": input_pass_seconds,
        "projected_input_seconds_per_arm": input_projection,
        "projected_total_seconds": projected,
        "ceiling_seconds": CEILING_SECONDS,
        "gate_passed": projected <= CEILING_SECONDS,
        "method": (
            "two full training+PTB hashes and two extra PTB hashes, one full selected-row "
            "input pass, 24 real updates and checkpoint replay per arm, 1.25x slower of "
            "update/input projections, 1.5x three-arm PTB extraction, "
            "a second readout profile, 900s CPU/readout reserve, 180s training reserve"
        ),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUTPUT / "profile.json", result)
    print(json.dumps({"stage": "profile_complete", "gate_passed": result["gate_passed"],
                      "projected_total_seconds": projected}), flush=True)


def train_arm(name: str, model: torch.nn.Module, dataset: CachedCPCDataset,
              order: np.ndarray, run_identity: dict[str, Any]) -> dict[str, Any]:
    """Train one arm with exact optimizer and RNG resumption every 100 updates."""
    arm_dir = OUTPUT / name
    arm_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = arm_dir / "latest.pt"
    model = model.cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    completed = 0
    loss_sum = 0.0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["identity"] != run_identity or saved["arm"] != name:
            raise ValueError("Checkpoint identity or arm changed")
        completed = saved["completed_updates"]
        loss_sum = saved["loss_sum"]
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng_state(saved["rng"])
    if not 0 <= completed <= UPDATES:
        raise ValueError("Invalid checkpoint update count")
    start = time.monotonic()
    for offset, batch in enumerate(loader(dataset, order[completed * BATCH:]), start=1):
        loss_sum += update(model, optimizer, batch, name)
        current = completed + offset
        if current % 100 == 0 or current == UPDATES:
            torch.cuda.synchronize()
            write_torch_atomic(checkpoint, {
                "identity": run_identity, "arm": name, "completed_updates": current,
                "loss_sum": loss_sum, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "rng": capture_rng_state(),
            })
            print(json.dumps({"stage": "train", "arm": name, "updates": current,
                              "seconds": time.monotonic() - start}), flush=True)
    if not checkpoint.exists():
        raise ValueError("Missing completed checkpoint")
    result = {
        "identity": run_identity,
        "arm": name,
        "completed_updates": UPDATES,
        "record_exposures": EXPOSURES,
        "mean_cpc_loss": loss_sum / UPDATES,
        "checkpoint_sha256": sha256_file(checkpoint),
        "profile_sha256": sha256_file(OUTPUT / "profile.json"),
    }
    write_json_atomic(arm_dir / "complete.json", result)
    return result


def train(dataset: CachedCPCDataset, order: np.ndarray,
          run_identity: dict[str, Any]) -> None:
    """Train each fresh arm only after the matching measured cost gate passes."""
    receipt = json.loads((OUTPUT / "profile.json").read_text())
    if receipt["identity"] != run_identity or not receipt["gate_passed"]:
        raise ValueError("Matching passed Experiment 012 cost gate required")
    for name, model in initial_models().items():
        arm_dir = OUTPUT / name
        if (arm_dir / "complete.json").exists():
            complete = json.loads((arm_dir / "complete.json").read_text())
            if (complete["identity"] != run_identity
                    or complete["completed_updates"] != UPDATES
                    or complete["record_exposures"] != EXPOSURES
                    or complete["checkpoint_sha256"] != sha256_file(arm_dir / "latest.pt")):
                raise ValueError("Completed arm identity or checkpoint changed")
            continue
        print(json.dumps({"stage": "train_start", "arm": name}), flush=True)
        result = train_arm(name, model, dataset, order, run_identity)
        print(json.dumps({"stage": "train_complete", "arm": name,
                          "checkpoint_sha256": result["checkpoint_sha256"]}), flush=True)


def main() -> None:
    """Run the CPU preflight or the locked profile/training stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("check", "profile", "train"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    start = time.monotonic()
    selected, counts = selected_indices()
    run_identity, readout_hash_seconds = identity(selected)
    order = exposure_order(selected, EXPOSURES, ORDER_SEED)
    if len(np.unique(order)) != SUBSET_SIZE:
        raise ValueError("Exposure stream lost selected ECGs")
    print(json.dumps({"stage": "check", "fingerprint": sha256_json(run_identity),
                      "selected_indices_sha256": SELECTED_SHA256}), flush=True)
    if args.stage == "check":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    with gpu_lock("cuda", blocking=False):
        dataset = CachedCPCDataset(CACHE, NORMALIZATION, verify_hash=False)
        if len(dataset) != EXPOSURES:
            raise ValueError("Unexpected CPC cache size")
        if args.stage == "profile":
            profile(dataset, selected, order, run_identity, counts,
                    time.monotonic() - start, readout_hash_seconds)
        else:
            train(dataset, order, run_identity)


if __name__ == "__main__":
    main()
