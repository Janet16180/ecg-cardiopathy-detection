"""Small forward/backward feasibility probe for the three Experiment 011 arms.

This probes model compute only. It is not the real-data profile required before
training and must not be interpreted as an ECG performance result.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ecg_experiment import ROOT
from ecg_experiment.cpc_delta_memory import matched_initial_models
from ecg_experiment.files import sha256_file, write_json_atomic
from ecg_experiment.gpu import gpu_lock

DEFAULT_OUTPUT = ROOT / "outputs/experiment011_delta_memory/implementation_compute_probe.json"
SOURCE = ROOT / "ecg_experiment/cpc_delta_memory.py"


def probe(batch_size: int, device: str) -> dict:
    """Measure one forward/backward pass for each fresh arm under the GPU lock."""
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this process")

    result = {
        "scope": "compute_only_no_training",
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "batch_size": batch_size,
        "source_sha256": sha256_file(SOURCE),
        "arms": {},
    }
    with gpu_lock(device, blocking=False):
        for name, model in matched_initial_models(9001).items():
            model = model.to(device).train()
            signal = torch.randn(batch_size, 12, 2500, device=device)
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()

            timings = []
            for repeat in range(3):
                model.zero_grad(set_to_none=True)
                started = time.monotonic()
                loss, _ = model(signal)
                loss.backward()
                if device == "cuda":
                    torch.cuda.synchronize()
                if repeat:
                    timings.append(time.monotonic() - started)

            result["arms"][name] = {
                "measured_forward_backward_seconds": timings,
                "warmup_passes": 1,
                "loss": float(loss.detach()),
                "peak_gb": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "gradients_finite": all(
                    parameter.grad is not None and torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                ),
            }
            print(name, json.dumps(result["arms"][name]), flush=True)
            del model, signal, loss
            if device == "cuda":
                torch.cuda.empty_cache()
    return result


def main() -> None:
    """Parse probe settings and save the aggregate receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.batch_size < 2:
        parser.error("batch size must be at least two for the CPC loss")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.output, probe(args.batch_size, args.device))


if __name__ == "__main__":
    main()
