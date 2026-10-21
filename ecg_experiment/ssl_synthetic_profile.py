"""Disposable actual-data CUDA profiling and conservative pilot resource forecasts."""

from __future__ import annotations

import json
import math
import subprocess
import time
from typing import Any

import numpy as np
import torch

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file, sha256_json
from ecg_experiment.paths import to_stored
from ecg_experiment.ssl_synthetic_data import load_bank, positive_class_weight
from ecg_experiment.ssl_synthetic_model import (
    DenseWaveformModel,
    localization_update,
    matched_schedule,
    model_description,
    ssl_update,
    state_digest,
)
from ecg_experiment.ssl_synthetic_preparation import verify_hashes, verify_receipt
from ecg_experiment.ssl_synthetic_readiness import checked_relative

PHASES = ("ssl", "trainable_localization", "frozen_localization", "inference")
WARMUP_STEPS = 10
TIMED_STEPS = 40
SAMPLING_SEED = 20262018
MASK_SEED = 20262019
SCHEDULE_SEED = 20261020


def resource_forecast(
    phases: dict[str, Any],
    profile_wall: float,
    profile_cpu: float,
    prior_cpu: float,
    stream_bytes: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Apply the frozen two-times update forecast and independent resource ceilings.

    Parameters
    ----------
    phases : dict
        All four measured phases with positive mean timed-step seconds.
    profile_wall, profile_cpu : float
        Measured whole-profile wall and process CPU seconds.
    prior_cpu : float
        Sum of prior measured CPU or conservative wall-time fallback charges.
    stream_bytes : int
        Entire observed research-stream file size before the profile summary.
    config : dict
        Frozen GPU/CPU/cache ceilings.

    Returns
    -------
    dict
        Auditable arithmetic, component forecasts and resource gates.
    """
    counts = dict(zip(PHASES, (1000, 1200, 600, 480), strict=True))
    times = {name: phases[name]["mean_timed_step_seconds"] for name in PHASES}
    if any(not math.isfinite(value) or value <= 0 for value in times.values()):
        raise ValueError("Profile step times must be finite and positive")
    if any(not math.isfinite(value) or value < 0 for value in (profile_wall, profile_cpu, prior_cpu)):
        raise ValueError("Resource charges must be finite and nonnegative")
    work = sum(counts[name] * times[name] for name in PHASES)
    gpu = profile_wall + 2 * work + 600
    cpu = prior_cpu + profile_cpu + 2 * work + 600
    gpu_gate = gpu <= config["gpu_seconds_ceiling"]
    cpu_gate = cpu <= config["cpu_seconds_ceiling"]
    cache_gate = stream_bytes + 100_000 <= config["cache_bytes_ceiling"]
    return {
        "planned_phase_steps": counts,
        "mean_timed_step_seconds": times,
        "remaining_single_pass_seconds": work,
        "update_safety_factor": 2.0,
        "reserve_seconds": 600.0,
        "profile_wall_seconds_charged": profile_wall,
        "profile_cpu_seconds_charged": profile_cpu,
        "prior_cpu_seconds_charged": prior_cpu,
        "gpu_forecast_seconds": gpu,
        "cpu_forecast_seconds": cpu,
        "gpu_seconds_ceiling": config["gpu_seconds_ceiling"],
        "cpu_seconds_ceiling": config["cpu_seconds_ceiling"],
        "stream_bytes_before_summary": stream_bytes,
        "summary_bytes_reserve": 100_000,
        "cache_bytes_ceiling": config["cache_bytes_ceiling"],
        "gpu_forecast_gate_passed": gpu_gate,
        "cpu_forecast_gate_passed": cpu_gate,
        "cache_budget_gate_passed": cache_gate,
        "resource_gate_passed": gpu_gate and cpu_gate and cache_gate,
        "is_forecast_not_full_training_measurement": True,
    }


def load_profile_inputs() -> dict[str, Any]:
    """
    Verify every prerequisite and bank digest before loading only permitted cached inputs.

    Returns
    -------
    dict
        Native memmaps, matched schedule/weight, receipt bindings and prior CPU accounting.
    """
    base = ROOT / "outputs/ssl_synthetic_localization"
    paths = [
        base / "input_audit_v1/summary.json",
        base / "real_cache_v1/summary.json",
        *sorted(base.glob("generator*/qualification.json")),
        base / "pilot_v1/prep_audit_v1/summary.json",
        base / "pilot_v1/preparation/summary.json",
    ]
    receipts = {to_stored(path): verify_receipt(path) for path in paths}
    audit, cache = receipts[to_stored(paths[0])], receipts[to_stored(paths[1])]
    prep = receipts[to_stored(paths[-1])]
    generator = receipts[to_stored(base / "generator_pilot_v2/qualification.json")]
    failed = receipts[to_stored(base / "generator_pilot_v1/qualification.json")]
    if not audit["input_integrity_gate_passed"] or not cache["cache_gate_passed"]:
        raise ValueError("Real input/cache admission failed")
    if (
        not generator["all_passed"]
        or generator["case_admission"]["count"] != 3060
        or generator["case_admission"]["failed_cases"]
        or not generator["role_integrity"]["all_disjoint"]
    ):
        raise ValueError("Bounded generator admission failed")
    if failed["all_passed"] or failed["case_admission"]["failed_cases"] != 28:
        raise ValueError("Original failed generator admission changed")
    if not prep["preparation_gate_passed"]:
        raise ValueError("Synthetic bank preparation failed")
    independent = receipts[to_stored(base / "pilot_v1/prep_audit_v1/summary.json")]
    audit_script = base / "pilot_v1/prep_audit_v1/audit_script.py"
    if (
        not independent["all_passed"]
        or sha256_file(paths[-1]) != independent["preparation_receipt_sha256"]
        or sha256_file(audit_script) != independent["script_sha256"]
    ):
        raise ValueError("Independent preparation audit changed or failed")
    verify_hashes(prep["bank_sha256"])
    config = json.loads((ROOT / "configs/ssl_synthetic_pilot_v1.json").read_bytes())
    if config["batch_size"] != 16:
        raise ValueError("This profile is prospectively fixed at batch 16")
    real = np.load(ROOT / checked_relative(prep["real_cache"]), mmap_mode="r", allow_pickle=False)
    train = load_bank(ROOT / checked_relative(prep["banks"]["train"]["path"]))
    validation = load_bank(ROOT / checked_relative(prep["banks"]["validation"]["path"]))
    if real.shape != (cache["usable_patients"], 12, 2500) or real.dtype != np.float32:
        raise ValueError("Real cache native geometry changed")
    if train["signals"].shape != (500, 12, 2500) or validation["signals"].shape != (512, 12, 2500):
        raise ValueError("Actual A banks do not match the fixed pilot")
    schedule = matched_schedule(train["kinds"], config["localization_steps"], SCHEDULE_SEED)
    prior = {}
    sources = {}
    for name, receipt in receipts.items():
        key = "cpu_seconds" if "cpu_seconds" in receipt else "wall_seconds"
        prior[name] = {"seconds": receipt[key], "basis": key}
        sources.update(receipt.get("source_sha256", {}))
        sources.update(receipt.get("input_audit_source_sha256", {}))
    hashes = {to_stored(path): sha256_file(path) for path in paths}
    hashes[to_stored(audit_script)] = independent["script_sha256"]
    hashes.update(prep["bank_sha256"])
    hashes.update(
        {to_stored(base / "real_cache_v1" / name): digest for name, digest in cache["output_sha256"].items()}
    )
    return {
        "real": real,
        "train": train,
        "validation": validation,
        "config": config,
        "schedule": schedule,
        "positive_weight": positive_class_weight(train, schedule),
        "localization_schedule_sha256": sha256_json(schedule.tolist()),
        "input_sha256": hashes,
        "prerequisite_source_sha256": sources,
        "prior_cpu_charges": prior,
    }


def frozen_optimizer(model: DenseWaveformModel, config: dict[str, Any]) -> torch.optim.AdamW:
    """
    Freeze all encoder bytes and bind the optimizer to the head parameters only.

    Parameters
    ----------
    model : DenseWaveformModel
        Fresh disposable F model.
    config : dict
        Frozen learning rate and weight decay.

    Returns
    -------
    torch.optim.AdamW
        Head-only optimizer, with the encoder excluded.
    """
    model.encoder.requires_grad_(False)
    model.encoder.eval()
    optimizer = torch.optim.AdamW(
        model.head.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    expected = {id(parameter) for parameter in model.head.parameters()}
    actual = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    if actual != expected or any(parameter.requires_grad for parameter in model.encoder.parameters()):
        raise ValueError("Frozen optimizer includes parameters outside the head")
    return optimizer


def fresh_model(config: dict[str, Any]) -> DenseWaveformModel:
    """
    Create the same initial component bytes as the frozen full-pilot recipe.

    Parameters
    ----------
    config : dict
        Model seed, width, stride and fixed physical entry scale.

    Returns
    -------
    DenseWaveformModel
        Disposable float32 model on CUDA.
    """
    torch.manual_seed(config["seed"])
    return (
        DenseWaveformModel(config["encoder_width"], config["encoder_stride"], config["physical_scale_mv"])
        .float()
        .to("cuda")
    )


def cuda_batch(array: np.ndarray, indices: np.ndarray) -> torch.Tensor:
    """
    Copy one actual native NumPy batch to CUDA on every call.

    Parameters
    ----------
    array : np.ndarray
        Bound read-only memmap, either physical signals or loss-only support.
    indices : np.ndarray
        Actual selected batch positions.

    Returns
    -------
    torch.Tensor
        Device batch with the native array dtype.
    """
    return torch.from_numpy(np.array(array[indices], copy=True)).to("cuda")


def exact_cuda_replay(real: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    """
    Independently repeat two actual-data masked SSL updates with exact state/loss checks.

    Parameters
    ----------
    real : np.ndarray
        Bound real TRAIN memmap.
    config : dict
        Frozen model and optimizer recipe.

    Returns
    -------
    dict
        Exact replay hashes/losses and disposable replay resource measurements.
    """
    wall, cpu = time.perf_counter(), time.process_time()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    runs = []
    for _ in range(2):
        model = fresh_model(config)
        description = model_description(model)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
        )
        indices = np.random.default_rng(SAMPLING_SEED).integers(0, len(real), size=(2, config["batch_size"]))
        masks = torch.Generator().manual_seed(MASK_SEED)
        losses = [ssl_update(model, optimizer, cuda_batch(real, batch), masks) for batch in indices]
        torch.cuda.synchronize()
        runs.append(
            {
                "initialization": description,
                "losses": losses,
                "final_state_sha256": state_digest(model.state_dict()),
            }
        )
        del model, optimizer
    return {
        "replay_gate_passed": runs[0] == runs[1],
        "steps_per_run": 2,
        "batch_indices_sha256": sha256_json(indices.tolist()),
        "runs": runs,
        "wall_seconds": time.perf_counter() - wall,
        "cpu_seconds": time.process_time() - cpu,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def profile_iteration(
    phase: str,
    index: int,
    model: DenseWaveformModel,
    optimizer: torch.optim.Optimizer | None,
    inputs: dict[str, Any],
    rng: np.random.Generator,
    masks: torch.Generator,
) -> None:
    """
    Run one actual data-transfer/update or inference iteration without retaining predictions.

    Parameters
    ----------
    phase : str
        One fixed resource phase.
    index : int
        Profile iteration index including warmup.
    model : DenseWaveformModel
        Fresh disposable phase model.
    optimizer : torch.optim.Optimizer or None
        Actual optimizer; absent for inference.
    inputs : dict
        Verified cached arrays, schedule and class weight.
    rng : np.random.Generator
        Dedicated SSL batch stream.
    masks : torch.Generator
        Dedicated SSL mask stream.
    """
    batch = inputs["config"]["batch_size"]
    if phase == "ssl":
        indices = rng.integers(0, len(inputs["real"]), size=batch)
        ssl_update(model, optimizer, cuda_batch(inputs["real"], indices), masks)
        return
    if phase in ("trainable_localization", "frozen_localization"):
        indices = inputs["schedule"][index]
        localization_update(
            model,
            optimizer,
            cuda_batch(inputs["train"]["signals"], indices),
            cuda_batch(inputs["train"]["support"], indices),
            inputs["positive_weight"],
            frozen_encoder=phase == "frozen_localization",
        )
        return
    if phase != "inference":
        raise ValueError("Unknown profile phase")
    indices = np.arange(index * batch, (index + 1) * batch) % len(inputs["validation"]["signals"])
    model.eval()
    with torch.inference_mode():
        probabilities = model(cuda_batch(inputs["validation"]["signals"], indices)).sigmoid().cpu().numpy()
    if not np.isfinite(probabilities).all():
        raise ValueError("Nonfinite disposable inference output")


def profile_phase(phase: str, inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Measure one complete warmup/timed phase including actual host-to-device transfers.

    Parameters
    ----------
    phase : str
        Fixed resource phase.
    inputs : dict
        Verified native data, optimizer configuration and actual training schedule.

    Returns
    -------
    dict
        Synchronized timings, component hashes, memory peaks and frozen-encoder guard.
    """
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    wall, cpu = time.perf_counter(), time.process_time()
    model = fresh_model(inputs["config"])
    optimizer = None
    if phase == "frozen_localization":
        optimizer = frozen_optimizer(model, inputs["config"])
    elif phase != "inference":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=inputs["config"]["learning_rate"],
            weight_decay=inputs["config"]["weight_decay"],
        )
    initial = model_description(model)
    count, batch = WARMUP_STEPS + TIMED_STEPS, inputs["config"]["batch_size"]
    if phase == "ssl":
        batches = np.random.default_rng(SAMPLING_SEED).integers(0, len(inputs["real"]), size=(count, batch))
    elif phase == "inference":
        batches = np.arange(count * batch).reshape(count, batch) % len(inputs["validation"]["signals"])
    else:
        batches = inputs["schedule"][:count]
    batch_hash = sha256_json(batches.tolist())
    encoder_before = state_digest(model.encoder.state_dict())
    rng = np.random.default_rng(SAMPLING_SEED)
    masks = torch.Generator().manual_seed(MASK_SEED)
    timed_wall = timed_cpu = 0.0
    for index in range(WARMUP_STEPS + TIMED_STEPS):
        if index == WARMUP_STEPS:
            torch.cuda.synchronize()
            timed_wall, timed_cpu = time.perf_counter(), time.process_time()
        profile_iteration(phase, index, model, optimizer, inputs, rng, masks)
    torch.cuda.synchronize()
    timed_seconds, timed_cpu_seconds = time.perf_counter() - timed_wall, time.process_time() - timed_cpu
    encoder_after = state_digest(model.encoder.state_dict())
    return {
        "initialization": initial,
        "batch_indices_sha256": batch_hash,
        "warmup_steps": WARMUP_STEPS,
        "timed_steps": TIMED_STEPS,
        "timed_wall_seconds": timed_seconds,
        "timed_cpu_seconds": timed_cpu_seconds,
        "mean_timed_step_seconds": timed_seconds / TIMED_STEPS,
        "wall_seconds": time.perf_counter() - wall,
        "cpu_seconds": time.process_time() - cpu,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "encoder_before_sha256": encoder_before,
        "encoder_after_sha256": encoder_after,
        "frozen_encoder_unchanged": encoder_before == encoder_after
        if phase == "frozen_localization"
        else None,
        "profile_weights_discarded": True,
    }


def configure_cuda() -> dict[str, Any]:
    """
    Reject busy/unavailable CUDA and freeze actual deterministic float32 execution.

    Returns
    -------
    dict
        Device/driver/backend settings for the live receipt.
    """
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True
    ).strip()
    if processes:
        raise RuntimeError("GPU already has an active compute process")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_default_dtype(torch.float32)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for the required actual profile")
    properties = torch.cuda.get_device_properties(0)
    return {
        "device": properties.name,
        "device_total_memory_bytes": properties.total_memory,
        "compute_capability": [properties.major, properties.minor],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip(),
        "deterministic_algorithms": True,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "tf32": False,
        "dtype": "float32",
        "numerical_cpu_threads": 1,
        "torch_cuda_architectures": torch.cuda.get_arch_list(),
    }
