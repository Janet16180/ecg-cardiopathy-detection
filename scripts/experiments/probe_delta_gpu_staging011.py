"""Feasibility check for exact-float32 CPC training signals staged on the V100.

This checks memory and one optimizer step per arm. It is not a performance
result or a substitute for a complete data-path profile.
"""

from __future__ import annotations

import gc
import json
import time

import numpy as np
import torch

from ecg_experiment import ROOT, cpc_pool
from ecg_experiment.cpc_delta_memory import matched_initial_models
from ecg_experiment.cpc_gpu_pool import shuffled_batches, stage_training_signals
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.training import checked_step

OUTPUT = ROOT / "outputs/experiment011_delta_memory/gpu_staging_probe.json"
NORMALIZATION = ROOT / "outputs/experiment004_cpc_40k/normalization.json"
SOURCE_FILES = (
    "ecg_experiment/cpc_delta_memory.py",
    "ecg_experiment/cpc_gpu_pool.py",
    "scripts/experiments/probe_delta_gpu_staging011.py",
)


def main() -> None:
    """Stage the fixed pool and probe one real optimizer update per arm."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    pool = cpc_pool.Pool(cpc_pool.DEFAULT_CACHE)
    normalization = json.loads(NORMALIZATION.read_text())
    if normalization["source"]["signals_sha256"] != pool.metadata["signals_sha256"]:
        raise ValueError("Normalization and waveform cache differ")
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    std = np.asarray(normalization["std"], dtype=np.float32)
    result = {
        "scope": "gpu_staging_feasibility_only",
        "source_sha256": {name: sha256_file(ROOT / name) for name in SOURCE_FILES},
        "cache_signal_sha256": pool.metadata["signals_sha256"],
        "training_records": len(pool.train_rows),
        "arms": {},
    }

    with gpu_lock("cuda", blocking=False):
        started = time.monotonic()
        signals = stage_training_signals(pool, mean, std)
        torch.cuda.synchronize()
        result["staging_seconds"] = time.monotonic() - started
        result["staged_gb"] = signals.numel() * signals.element_size() / 1e9
        print(json.dumps({"staging_seconds": result["staging_seconds"],
                          "staged_gb": result["staged_gb"]}), flush=True)

        for name, model in matched_initial_models(9001).items():
            model = model.cuda().train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
            batch = next(shuffled_batches(signals, 128, torch.Generator().manual_seed(9001)))
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            loss, _ = model(batch)
            checked_step(loss, model, optimizer, name)
            torch.cuda.synchronize()
            result["arms"][name] = {
                "step_seconds": time.monotonic() - started,
                "loss": float(loss.detach()),
                "peak_allocated_gb_including_pool": torch.cuda.max_memory_allocated() / 1e9,
                "gradients_finite": all(
                    parameter.grad is not None and torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                ),
            }
            print(name, json.dumps(result["arms"][name]), flush=True)
            del model, optimizer, batch, loss
            gc.collect()
            torch.cuda.empty_cache()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUTPUT, result)


if __name__ == "__main__":
    main()
