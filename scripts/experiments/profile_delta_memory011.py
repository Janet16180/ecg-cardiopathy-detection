"""Real-data, complete-pass cost gate for Experiment 011; no model selection.

This profile runs one SSL epoch per arm on the frozen pool and records actual
cache I/O, optimizer steps, memory, and checkpoint roundtrips. Its checkpoints
are profile artifacts, not pretrained encoders for downstream evaluation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ecg_experiment import ROOT, cpc_pool
from ecg_experiment.cpc_delta_memory import matched_initial_models
from ecg_experiment.files import sha256_file, sha256_json, write_json_atomic, write_torch_atomic
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.reproducibility import cpu_state
from ecg_experiment.training import checked_step

OUTPUT = ROOT / "outputs/experiment011_delta_memory/real_data_profile"
NORMALIZATION = ROOT / "outputs/experiment004_cpc_40k"
SOURCE_FILES = (
    "ecg_experiment/cpc_delta_memory.py",
    "ecg_experiment/cpc.py",
    "ecg_experiment/cpc_pool.py",
    "ecg_experiment/files.py",
    "ecg_experiment/gpu.py",
    "ecg_experiment/reproducibility.py",
    "ecg_experiment/training.py",
    "scripts/experiments/profile_delta_memory011.py",
    "docs/experiment-011-delta-memory.md",
)
EXPECTED_TRAINING_RECORDS = 56875
BATCH_SIZE = 128
SEED = 9001
LR = 1e-3
WEIGHT_DECAY = 0.01


def prepare_inputs() -> tuple[cpc_pool.Pool, np.ndarray, np.ndarray, dict[str, str]]:
    """Verify the frozen cache and training-only normalization, then hash code."""
    pool = cpc_pool.Pool(cpc_pool.DEFAULT_CACHE)
    if len(pool.train_rows) != EXPECTED_TRAINING_RECORDS:
        raise ValueError("Frozen CPC training pool count changed")

    signal_path = pool.directory / "signals.npy"
    signal_hash = sha256_file(signal_path)
    if signal_hash != pool.metadata["signals_sha256"]:
        raise ValueError("Frozen CPC signal cache changed")

    hashes = {str(signal_path.resolve()): signal_hash}
    mean, std = pool.normalization(NORMALIZATION, hashes)
    sources = {
        str((ROOT / name).resolve()): sha256_file(ROOT / name)
        for name in SOURCE_FILES
    }
    sources.update(
        {
            str((pool.directory / name).resolve()): sha256_file(pool.directory / name)
            for name in ("complete.json", "rows.csv", "ecg_ids.npy")
        }
    )
    sources[str(signal_path.resolve())] = signal_hash
    sources[str((NORMALIZATION / "normalization.json").resolve())] = sha256_file(
        NORMALIZATION / "normalization.json"
    )
    return pool, mean, std, sources


def profile_arm(
    name: str,
    model: torch.nn.Module,
    pool: cpc_pool.Pool,
    mean: np.ndarray,
    std: np.ndarray,
    output: Path,
    max_batches: int,
) -> dict[str, Any]:
    """Run real shuffled cache batches through one arm and verify saved state."""
    model = model.cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(SEED)
    batches = cpc_pool.loader(
        pool, pool.train_rows, mean, std, BATCH_SIZE, True, generator, "cuda"
    )
    torch.cuda.reset_peak_memory_stats()

    started = time.monotonic()
    compute_seconds = 0.0
    exposures = 0
    updates = 0
    last_loss = None
    for signal, _, _ in batches:
        if max_batches and updates >= max_batches:
            break
        signal = signal.to("cuda", non_blocking=True)
        compute_start = time.monotonic()
        loss, _ = model(signal)
        checked_step(loss, model, optimizer, f"011 {name}")
        torch.cuda.synchronize()
        compute_seconds += time.monotonic() - compute_start
        exposures += len(signal)
        updates += 1
        last_loss = float(loss.detach())
        if updates % 25 == 0:
            print(f"011 {name}: {updates}/{len(batches)} batches", flush=True)

    torch.cuda.synchronize()
    epoch_seconds = time.monotonic() - started
    if max_batches == 0 and exposures != EXPECTED_TRAINING_RECORDS:
        raise ValueError("Profile did not process the complete frozen training pool")

    checkpoint_start = time.monotonic()
    state = cpu_state(model)
    checkpoint = output / f"{name}_profile_state.pt"
    write_torch_atomic(
        checkpoint,
        {"model": state, "optimizer": optimizer.state_dict(), "updates": updates},
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["updates"] != updates or any(
        not torch.equal(tensor, saved["model"][key]) for key, tensor in state.items()
    ):
        raise ValueError(f"{name} profile checkpoint did not roundtrip")
    checkpoint_seconds = time.monotonic() - checkpoint_start
    return {
        "arm": name,
        "updates": updates,
        "record_exposures": exposures,
        "data_and_update_seconds": epoch_seconds,
        "compute_and_optimizer_seconds": compute_seconds,
        "checkpoint_seconds": checkpoint_seconds,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "last_loss": last_loss,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_roundtrip": True,
    }


def main() -> None:
    """Check inputs or run the locked full data-path profile."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("check", "profile"), required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--max-batches", type=int, default=0)
    args = parser.parse_args()
    if args.max_batches < 0:
        parser.error("max-batches must be nonnegative")

    pool, mean, std, sources = prepare_inputs()
    settings = {
        "scope": "cost_profile_only_no_downstream_evaluation",
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "max_batches": args.max_batches,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "input_and_source_sha256": sources,
        "normalization_mean": mean.tolist(),
        "normalization_std": std.tolist(),
    }
    settings["fingerprint"] = sha256_json(settings)
    print(json.dumps({"stage": "check", "fingerprint": settings["fingerprint"]}), flush=True)
    if args.stage == "check":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for the real-data profile")

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Profile output already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with gpu_lock("cuda", blocking=False):
        result = {"settings": settings, "arms": {}}
        for name, model in matched_initial_models(SEED).items():
            result["arms"][name] = profile_arm(
                name, model, pool, mean, std, args.output_dir, args.max_batches
            )
            write_json_atomic(args.output_dir / "profile.json", result)
            print(json.dumps(result["arms"][name]), flush=True)


if __name__ == "__main__":
    main()
