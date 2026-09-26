"""Profile and train one 25k-source-stratified compact CPC continuation."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ecg_experiment.cpc import CPCPretrainer
from ecg_experiment.cpc_scaling_cached import CachedCPCDataset
from ecg_experiment.cpc_subset25 import exposure_order, select_indices
from ecg_experiment.files import read_csv, sha256_file, sha256_json, write_json_atomic, write_torch_atomic
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.reproducibility import capture_rng_state, restore_rng_state, seed_everything
from ecg_experiment.training import checked_step

ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / "data/processed/sampled_100k_plus_labels_v1"
CACHE = ROOT / "data/processed/sampled_100k_cpc_cache_v1"
BASE = ROOT / "outputs/experiment004_cpc_40k"
OUTPUT = ROOT / "outputs/experiment019_cpc_25k_v1"
SELECTION_SEED = 18046
ORDER_SEED = 18047
MODEL_SEED = 18042
SUBSET_SIZE = 25_000
EXPOSURES = 115_359
BATCH = 128
PROFILE_BATCHES = 24
CEILING_SECONDS = 7_200


def selected_indices() -> tuple[np.ndarray, dict[str, int]]:
    """Replay the exact 25k source-stratified selection from the frozen rows."""
    rows = read_csv(COHORT / "train_manifest.csv", required=("record_id", "source", "split"))
    if len(rows) != EXPOSURES or any(row["split"] != "train" for row in rows):
        raise ValueError("Unexpected source training manifest")
    if len({row["record_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate source record IDs")
    sources = [row["source"] for row in rows]
    indices = select_indices(sources, SUBSET_SIZE, SELECTION_SEED)
    return indices, dict(sorted(Counter(sources[index] for index in indices).items()))


def identity(indices: np.ndarray) -> dict[str, str | int | float]:
    """Bind every data and source input before touching the V100."""
    cache = json.loads((CACHE / "complete.json").read_text())
    if sha256_file(CACHE / "signals.npy") != cache["signals_sha256"]:
        raise ValueError("Verified waveform cache changed")
    paths = {
        "cohort_manifest": COHORT / "train_manifest.csv",
        "cohort_metadata": COHORT / "metadata.json",
        "cache_receipt": CACHE / "complete.json",
        "cache_verification": ROOT / "outputs/experiment018_cpc_data_scaling_v2/cache_verification.json",
        "initial_encoder": BASE / "cpc_ssl/encoder.pt",
        "normalization": BASE / "normalization.json",
        "model_source": ROOT / "ecg_experiment/cpc.py",
        "cache_adapter_source": ROOT / "ecg_experiment/cpc_scaling_cached.py",
        "selection_source": ROOT / "ecg_experiment/cpc_subset25.py",
        "runner_source": Path(__file__),
    }
    return {
        **{name: sha256_file(path) for name, path in paths.items()},
        "cached_signals_sha256": cache["signals_sha256"],
        "selected_indices_sha256": sha256_json(indices.tolist()),
        "subset_size": SUBSET_SIZE,
        "selection_seed": SELECTION_SEED,
        "order_seed": ORDER_SEED,
        "model_seed": MODEL_SEED,
        "record_exposures": EXPOSURES,
        "batch": BATCH,
        "learning_rate": 1e-4,
    }


def initial_model() -> tuple[CPCPretrainer, torch.optim.Optimizer]:
    """Use the same initial encoder, predictor seed, and optimizer as v3."""
    seed_everything(MODEL_SEED)
    model = CPCPretrainer().cuda()
    saved = torch.load(BASE / "cpc_ssl/encoder.pt", map_location="cpu", weights_only=True)
    model.encoder.load_state_dict(saved["encoder"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    return model, optimizer


def loader(dataset: CachedCPCDataset, indices: np.ndarray) -> DataLoader:
    """Read fixed-order cache rows with no worker-owned random choices."""
    return DataLoader(Subset(dataset, indices.tolist()), batch_size=BATCH,
                      shuffle=False, num_workers=4, pin_memory=True,
                      generator=torch.Generator().manual_seed(MODEL_SEED))


def update(model: CPCPretrainer, optimizer: torch.optim.Optimizer,
           batch: tuple[torch.Tensor, torch.Tensor, list[str]]) -> float:
    """Apply one matched ordinary CPC update."""
    signals, _, _ = batch
    loss, _ = model(signals.cuda(non_blocking=True))
    checked_step(loss, model, optimizer, "CPC 25k")
    return float(loss.detach())


def profile(dataset: CachedCPCDataset, order: np.ndarray,
            run_identity: dict[str, str | int | float], counts: dict[str, int],
            preflight_seconds: float) -> None:
    """Measure the real loader and V100 before admitting full training."""
    model, optimizer = initial_model()
    start = time.monotonic()
    losses = [update(model, optimizer, batch)
              for batch in loader(dataset, order[:PROFILE_BATCHES * BATCH])]
    torch.cuda.synchronize()
    measured = time.monotonic() - start
    projected = (2 * preflight_seconds + measured
                 + 1.25 * math.ceil(EXPOSURES / BATCH) * measured / len(losses) + 180)
    receipt = {
        "identity": run_identity,
        "source_counts": counts,
        "profile_batches": len(losses),
        "profile_seconds": measured,
        "preflight_seconds_per_launch": preflight_seconds,
        "projected_total_seconds": projected,
        "ceiling_seconds": CEILING_SECONDS,
        "gate_passed": projected <= CEILING_SECONDS,
        "last_profile_loss": losses[-1],
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "method": "two full-cache hashes, 24 real V100 updates, 1.25x full training, 180s reserve",
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUTPUT / "profile.json", receipt)
    print(json.dumps({"stage": "profile", "gate_passed": receipt["gate_passed"],
                      "projected_total_seconds": projected, "source_counts": counts}), flush=True)


def train(dataset: CachedCPCDataset, order: np.ndarray,
          run_identity: dict[str, str | int | float]) -> None:
    """Train 902 updates with resumable optimizer and RNG state."""
    profile_receipt = json.loads((OUTPUT / "profile.json").read_text())
    if profile_receipt["identity"] != run_identity or not profile_receipt["gate_passed"]:
        raise ValueError("Matching passed cost gate required")
    checkpoint = OUTPUT / "latest.pt"
    model, optimizer = initial_model()
    completed = 0
    loss_sum = 0.0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["identity"] != run_identity:
            raise ValueError("Checkpoint identity changed")
        completed = saved["completed_batches"]
        loss_sum = saved["loss_sum"]
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng_state(saved["rng"])
    start = time.monotonic()
    total = math.ceil(EXPOSURES / BATCH)
    for offset, batch in enumerate(loader(dataset, order[completed * BATCH:]), start=1):
        loss_sum += update(model, optimizer, batch)
        current = completed + offset
        if current % 100 == 0 or current == total:
            torch.cuda.synchronize()
            write_torch_atomic(checkpoint, {
                "identity": run_identity, "completed_batches": current,
                "loss_sum": loss_sum, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "rng": capture_rng_state(),
            })
            print(json.dumps({"stage": "train", "updates": current,
                              "total": total, "seconds": time.monotonic() - start}), flush=True)
    write_json_atomic(OUTPUT / "complete.json", {
        "identity": run_identity,
        "subset_size": SUBSET_SIZE,
        "record_exposures": EXPOSURES,
        "completed_updates": total,
        "mean_cpc_loss": loss_sum / total,
        "checkpoint_sha256": sha256_file(checkpoint),
        "profile_sha256": sha256_file(OUTPUT / "profile.json"),
    })


def main() -> None:
    """Run the frozen 25k profile or the admitted full continuation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("profile", "train"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    preflight_start = time.monotonic()
    selected, counts = selected_indices()
    run_identity = identity(selected)
    order = exposure_order(selected, EXPOSURES, ORDER_SEED)
    if len(np.unique(order)) != SUBSET_SIZE:
        raise ValueError("Exposure stream lost a selected ECG")
    with gpu_lock("cuda", blocking=False):
        dataset = CachedCPCDataset(CACHE, BASE / "normalization.json", verify_hash=False)
        if len(dataset) != 115_359:
            raise ValueError("Unexpected verified cache size")
        preflight_seconds = time.monotonic() - preflight_start
        if args.stage == "profile":
            profile(dataset, order, run_identity, counts, preflight_seconds)
        else:
            train(dataset, order, run_identity)


if __name__ == "__main__":
    main()
