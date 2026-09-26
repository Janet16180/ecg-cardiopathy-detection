"""Matched CPC continuation after accepting the verified local data cache."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from ecg_experiment.cpc import CPCPretrainer
from ecg_experiment.cpc_pool import Pool, PoolDataset
from ecg_experiment.cpc_scaling_cached import CachedCPCDataset
from ecg_experiment.files import sha256_file, write_json_atomic, write_torch_atomic
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.reproducibility import capture_rng_state, restore_rng_state, seed_everything
from ecg_experiment.training import checked_step

ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / "data/processed/sampled_100k_plus_labels_v1"
CACHE = ROOT / "data/processed/sampled_100k_cpc_cache_v1"
POOL = ROOT / "data/processed/cpc_pool_40k"
BASE = ROOT / "outputs/experiment004_cpc_40k"
OUTPUT = ROOT / "outputs/experiment018_cpc_data_scaling_v3"
CACHE_SOURCE_ARCHIVE = ROOT / "outputs/experiment018_cpc_data_scaling_v2"
SEED = 18042
BATCH = 128
LR = 1e-4
PROFILE_BATCHES = 24
CEILING_SECONDS = 7200


def input_identity() -> dict[str, str | int | float]:
    """Bind the cohort, baseline, model, runner, and fixed schedule."""
    paths = {
        "sampled_metadata": COHORT / "metadata.json",
        "sampled_manifest": COHORT / "train_manifest.csv",
        "cache_receipt": CACHE / "complete.json",
        "cache_verification_receipt": CACHE_SOURCE_ARCHIVE / "cache_verification.json",
        "old_pool_metadata": POOL / "complete.json",
        "old_pool_rows": POOL / "rows.csv",
        "initial_encoder": BASE / "cpc_ssl/encoder.pt",
        "normalization": BASE / "normalization.json",
        "model_source": ROOT / "ecg_experiment/cpc.py",
        "adapter_source": ROOT / "ecg_experiment/cpc_scaling_cached.py",
        "cache_builder_source_at_launch": CACHE_SOURCE_ARCHIVE / "cache_builder_source_at_launch.py",
        "cache_builder_source_shard_parallel": (
            CACHE_SOURCE_ARCHIVE / "cache_builder_source_shard_parallel.py"
        ),
        "cache_builder_source_shard_prefetch": (
            CACHE_SOURCE_ARCHIVE / "cache_builder_source_shard_prefetch.py"
        ),
        "cache_builder_source_flush4096": (
            CACHE_SOURCE_ARCHIVE / "cache_builder_source_flush4096.py"
        ),
        "cohort_loader_source": ROOT / "ecg_experiment/sampled_training_dataset.py",
        "signal_contract_source": ROOT / "ecg_experiment/training_contracts.py",
        "runner_source": Path(__file__),
    }
    complete = json.loads((CACHE / "complete.json").read_text())
    if sha256_file(CACHE / "signals.npy") != complete["signals_sha256"]:
        raise ValueError("Cached signals changed")
    return {**{name: sha256_file(path) for name, path in paths.items()},
            "cached_signals_sha256": complete["signals_sha256"],
            "seed": SEED, "batch": BATCH, "lr": LR,
            "record_exposures_per_arm": 115359}


def datasets() -> tuple[Dataset, Dataset]:
    """Construct two training-only inputs under the same normalizer."""
    old = Pool(POOL)
    values = json.loads((BASE / "normalization.json").read_text())
    mean = np.asarray(values["mean"], dtype=np.float32)
    std = np.asarray(values["std"], dtype=np.float32)
    return (PoolDataset(old, old.train_rows, mean, std),
            CachedCPCDataset(CACHE, BASE / "normalization.json", verify_hash=False))


def ordered_indices(arm: str, old_count: int, new_count: int) -> np.ndarray:
    """Sample the old pool or use the cohort's already seeded manifest order."""
    rng = np.random.default_rng(SEED)
    if arm == "old":
        return rng.integers(old_count, size=new_count, dtype=np.int64)
    return np.arange(new_count, dtype=np.int64)


def loader(dataset: Dataset, indices: np.ndarray, *, workers: int) -> DataLoader:
    """Build a deterministic input stream without worker-owned random choices."""
    return DataLoader(Subset(dataset, indices.tolist()), batch_size=BATCH,
                      shuffle=False, num_workers=workers, pin_memory=True,
                      generator=torch.Generator().manual_seed(SEED))


def initial_model() -> tuple[CPCPretrainer, torch.optim.Optimizer]:
    """Start both arms from the exact same released CPC encoder and head seed."""
    seed_everything(SEED)
    model = CPCPretrainer().cuda()
    saved = torch.load(BASE / "cpc_ssl/encoder.pt", map_location="cpu", weights_only=True)
    model.encoder.load_state_dict(saved["encoder"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    return model, optimizer


def step(model: CPCPretrainer, optimizer: torch.optim.Optimizer,
         batch: tuple[torch.Tensor, torch.Tensor, list[str]]) -> float:
    """Run one ordinary CPC update and return its loss."""
    signal, _, _ = batch
    loss, _ = model(signal.cuda(non_blocking=True))
    checked_step(loss, model, optimizer, "CPC scaling")
    return float(loss.detach())


def profile(old: Dataset, new: Dataset, identity: dict[str, str | int | float],
            workers: int, preflight_seconds: float) -> None:
    """Measure real loader plus GPU updates and enforce the study cost gate."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = {}
    overall_start = time.monotonic()
    for arm, dataset in (("old", old), ("new", new)):
        model, optimizer = initial_model()
        indices = ordered_indices(arm, len(old), len(new))[:PROFILE_BATCHES * BATCH]
        started = time.monotonic()
        losses = [step(model, optimizer, batch)
                  for batch in loader(dataset, indices, workers=workers)]
        torch.cuda.synchronize()
        seconds = time.monotonic() - started
        results[arm] = {"batches": len(losses), "records": len(indices),
                        "seconds": seconds, "last_loss": losses[-1],
                        "peak_gpu_bytes": torch.cuda.max_memory_allocated()}
        print(json.dumps({"arm": arm, **results[arm]}), flush=True)
        del model, optimizer
        torch.cuda.empty_cache()
    elapsed = time.monotonic() - overall_start
    preparation = json.loads((CACHE / "complete.json").read_text())[
        "elapsed_total_seconds"]
    # Profile includes cold source-shard hashes and worker startup. Repeating
    # that cost for every update is conservative for this one-epoch pilot.
    projected = 2 * preflight_seconds + elapsed + 1.25 * sum(
        math.ceil(len(new) / BATCH) * arm["seconds"] / arm["batches"]
        for arm in results.values()
    ) + 180
    receipt = {"identity": identity, "profile": results,
               "historical_cache_preparation_seconds_not_charged": preparation,
               "preflight_seconds_per_launch": preflight_seconds,
               "profile_elapsed_seconds": elapsed,
               "projected_total_seconds": projected,
               "ceiling_seconds": CEILING_SECONDS,
               "gate_passed": projected <= CEILING_SECONDS,
               "method": (
                   "verified cache is existing input; two full-hash preflights, "
                   "real-data GPU updates, 1.25 margin, 180s report reserve"
               )}
    write_json_atomic(OUTPUT / "profile.json", receipt)
    print(json.dumps({"profile_gate": receipt["gate_passed"],
                      "projected_total_seconds": projected}), flush=True)


def train_arm(arm: str, dataset: Dataset, indices: np.ndarray,
              identity: dict[str, str | int | float], workers: int) -> None:
    """Train one complete pass with resumable fixed-order update checkpoints."""
    directory = OUTPUT / arm
    directory.mkdir(parents=True, exist_ok=True)
    model, optimizer = initial_model()
    checkpoint = directory / "latest.pt"
    completed = 0
    loss_sum = 0.0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["identity"] != identity or saved["arm"] != arm:
            raise ValueError("Checkpoint identity mismatch")
        completed = saved["completed_batches"]
        loss_sum = saved["loss_sum"]
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng_state(saved["rng"])
    start = time.monotonic()
    remaining = indices[completed * BATCH:]
    for offset, batch in enumerate(loader(dataset, remaining, workers=workers), start=1):
        loss_sum += step(model, optimizer, batch)
        current = completed + offset
        if current % 100 == 0 or current == math.ceil(len(indices) / BATCH):
            torch.cuda.synchronize()
            write_torch_atomic(checkpoint, {
                "identity": identity, "arm": arm, "completed_batches": current,
                "loss_sum": loss_sum, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "rng": capture_rng_state(),
            })
            print(json.dumps({"arm": arm, "completed_batches": current,
                              "total_batches": math.ceil(len(indices) / BATCH),
                              "elapsed_seconds": time.monotonic() - start}), flush=True)
    write_json_atomic(directory / "complete.json", {
        "identity": identity, "arm": arm, "records": len(indices),
        "batches": math.ceil(len(indices) / BATCH),
        "mean_training_loss": loss_sum / math.ceil(len(indices) / BATCH),
        "checkpoint_sha256": sha256_file(checkpoint),
    })


def main() -> None:
    """Profile or run the frozen, training-only matched scaling comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("profile", "train"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.workers < 0:
        parser.error("workers must be nonnegative")
    preflight_start = time.monotonic()
    identity = input_identity()
    with gpu_lock("cuda", blocking=False):
        old, new = datasets()
        preflight_seconds = time.monotonic() - preflight_start
        if len(old) != 56875 or len(new) != 115359:
            raise ValueError("Unexpected frozen cohort size")
        if args.stage == "profile":
            profile(old, new, identity, args.workers, preflight_seconds)
            return
        receipt = json.loads((OUTPUT / "profile.json").read_text())
        if receipt["identity"] != identity or not receipt["gate_passed"]:
            raise ValueError("Verified cost gate required before training")
        for arm, dataset in (("old", old), ("new", new)):
            indices = ordered_indices(arm, len(old), len(new))
            train_arm(arm, dataset, indices, identity, args.workers)


if __name__ == "__main__":
    main()
