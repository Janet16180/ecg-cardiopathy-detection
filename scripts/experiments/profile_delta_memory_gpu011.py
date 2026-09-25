"""Complete-pass Experiment 011 cost profile with a staged float32 GPU pool.

The same frozen signals and normalization are used as the mmap profile. This
version pays one sequential GPU-staging cost, then measures a shuffled epoch
and checkpoint roundtrip for each freshly initialized arm. No label is read.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch

from ecg_experiment import ROOT
from ecg_experiment.cpc_delta_memory import matched_initial_models
from ecg_experiment.cpc_gpu_pool import shuffled_batches, stage_training_signals
from ecg_experiment.cpc_profile_inputs import (
    BATCH_SIZE,
    EXPECTED_TRAINING_RECORDS,
    LR,
    SEED,
    WEIGHT_DECAY,
    prepare_inputs,
)
from ecg_experiment.files import sha256_file, sha256_json, write_json_atomic, write_torch_atomic
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.reproducibility import cpu_state
from ecg_experiment.training import checked_step

OUTPUT = ROOT / "outputs/experiment011_delta_memory/gpu_full_pass_profile_v1"
SOURCE_FILES = (
    "ecg_experiment/cpc_delta_memory.py",
    "ecg_experiment/cpc_gpu_pool.py",
    "ecg_experiment/cpc.py",
    "ecg_experiment/cpc_pool.py",
    "ecg_experiment/cpc_profile_inputs.py",
    "ecg_experiment/files.py",
    "ecg_experiment/gpu.py",
    "ecg_experiment/reproducibility.py",
    "ecg_experiment/training.py",
    "scripts/experiments/profile_delta_memory_gpu011.py",
    "docs/experiment-011-delta-memory.md",
)


def profile_arm(
    name: str,
    model: torch.nn.Module,
    signals: torch.Tensor,
    output: Path,
    max_batches: int,
) -> dict[str, Any]:
    """Run one matched arm and verify its saved model and shuffle state."""
    model = model.cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(SEED)
    torch.cuda.reset_peak_memory_stats()

    started = time.monotonic()
    updates = 0
    exposures = 0
    last_loss = None
    for batch in shuffled_batches(signals, BATCH_SIZE, generator):
        if max_batches and updates >= max_batches:
            break
        loss, _ = model(batch)
        checked_step(loss, model, optimizer, f"011 {name} staged")
        torch.cuda.synchronize()
        updates += 1
        exposures += len(batch)
        last_loss = float(loss.detach())
        if updates % 50 == 0:
            print(f"011 staged {name}: {updates} batches", flush=True)
    epoch_seconds = time.monotonic() - started
    if max_batches == 0 and exposures != EXPECTED_TRAINING_RECORDS:
        raise ValueError("Staged profile did not process all training records")

    checkpoint_started = time.monotonic()
    state = cpu_state(model)
    shuffle_state = generator.get_state()
    checkpoint = output / f"{name}_profile_state.pt"
    write_torch_atomic(
        checkpoint,
        {
            "model": state,
            "optimizer": optimizer.state_dict(),
            "shuffle_state": shuffle_state,
            "updates": updates,
        },
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved["updates"] != updates or not torch.equal(saved["shuffle_state"], shuffle_state):
        raise ValueError(f"{name} profile shuffle state did not roundtrip")
    if any(not torch.equal(tensor, saved["model"][key]) for key, tensor in state.items()):
        raise ValueError(f"{name} profile weights did not roundtrip")
    checkpoint_seconds = time.monotonic() - checkpoint_started
    record = {
        "arm": name,
        "updates": updates,
        "record_exposures": exposures,
        "epoch_seconds": epoch_seconds,
        "checkpoint_seconds": checkpoint_seconds,
        "peak_allocated_gb_including_pool": torch.cuda.max_memory_allocated() / 1e9,
        "last_loss": last_loss,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_roundtrip": True,
    }
    del model, optimizer, state, saved
    gc.collect()
    torch.cuda.empty_cache()
    return record


def main() -> None:
    """Verify the cache, stage it once, and profile every arm on real ECGs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("check", "profile"), required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--max-batches", type=int, default=0)
    args = parser.parse_args()
    if args.max_batches < 0:
        parser.error("max-batches must be nonnegative")

    pool, mean, std, sources = prepare_inputs()
    sources.update({str((ROOT / name).resolve()): sha256_file(ROOT / name) for name in SOURCE_FILES})
    settings = {
        "scope": "gpu_staged_cost_profile_only",
        "batch_size": BATCH_SIZE,
        "seed": SEED,
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
        raise RuntimeError("CUDA unavailable for the staged profile")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Profile output already exists: {args.output_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with gpu_lock("cuda", blocking=False):
        started = time.monotonic()
        signals = stage_training_signals(pool, mean, std)
        torch.cuda.synchronize()
        staging_seconds = time.monotonic() - started
        result = {
            "settings": settings,
            "staging_seconds": staging_seconds,
            "staged_gb": signals.numel() * signals.element_size() / 1e9,
            "arms": {},
        }
        print(json.dumps({"staging_seconds": staging_seconds}), flush=True)
        models = matched_initial_models(SEED)
        for name in ("gru", "kda", "ckda"):
            result["arms"][name] = profile_arm(
                name, models.pop(name), signals, args.output_dir, args.max_batches
            )
            write_json_atomic(args.output_dir / "profile.json", result)
            print(json.dumps(result["arms"][name]), flush=True)


if __name__ == "__main__":
    main()
