"""Factory-based deterministic successor measurements reusing the frozen actual-data profile path."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file, sha256_json
from ecg_experiment.ssl_synthetic_model import DenseWaveformModel as OriginalModel
from ecg_experiment.ssl_synthetic_model import model_description, ssl_update, state_digest
from ecg_experiment.ssl_synthetic_preparation import verify_hashes, verify_receipt
from ecg_experiment.ssl_synthetic_profile import (
    MASK_SEED,
    SAMPLING_SEED,
    TIMED_STEPS,
    WARMUP_STEPS,
    cuda_batch,
    frozen_optimizer,
    profile_iteration,
    resource_forecast,
)

ModelFactory = Callable[[dict[str, Any], str], OriginalModel]
FAILED_PROFILE_SHA256 = "688786d37f434fde28eb820dad8f53cc8d8485a7130dab496e2ab24ec0838b02"


def make_model(config: dict[str, Any], device: str = "cuda") -> OriginalModel:
    """
    Construct the committed successor with identical original encoder/head initialization.

    Parameters
    ----------
    config : dict
        Frozen seed, architecture and physical entry scale.
    device : str
        CUDA for actual updates, CPU for counterfactual numerical parity.

    Returns
    -------
    DenseWaveformModel
        Successor model with a fixed deterministic interpolation operation.
    """
    from ecg_experiment.ssl_synthetic_model_v2 import DenseWaveformModel

    torch.manual_seed(config["seed"])
    return (
        DenseWaveformModel(config["encoder_width"], config["encoder_stride"], config["physical_scale_mv"])
        .float()
        .to(device)
    )


def load_failed_profile() -> dict[str, Any]:
    """
    Verify the immutable failed attempt and every recorded prerequisite/source binding.

    Returns
    -------
    dict
        Original failure receipt with exact costs carried forward.
    """
    path = ROOT / "outputs/ssl_synthetic_localization/pilot_v1/profile/summary.json"
    if sha256_file(path) != FAILED_PROFILE_SHA256:
        raise ValueError("Original failed profile receipt changed")
    report = verify_receipt(path)
    verify_hashes(report["prerequisite_source_sha256"])
    if report["profile_gate_passed"] or report["training_started"] or report["phases"]:
        raise ValueError("Original profile no longer matches the retained failure")
    return report


def cpu_parity(factory: ModelFactory, real: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    """
    Compare the full successor and counterfactual original model on actual cached inputs.

    Parameters
    ----------
    factory : callable
        Explicit model factory, without mutation of any pinned module constructor.
    real : np.ndarray
        Bound native physical TRAIN memmap.
    config : dict
        Frozen initial recipe.

    Returns
    -------
    dict
        Float32 full-forward/input-gradient/parameter-gradient equivalence measurements.
    """
    wall, cpu = time.perf_counter(), time.process_time()
    model = factory(config, "cpu")
    torch.manual_seed(config["seed"])
    original = OriginalModel(config["encoder_width"], config["encoder_stride"], config["physical_scale_mv"])
    indices = np.random.default_rng(SAMPLING_SEED).integers(0, len(real), size=config["batch_size"])[:2]
    first = torch.from_numpy(np.array(real[indices], copy=True)).requires_grad_()
    second = first.detach().clone().requires_grad_()
    actual, expected = model(first), original(second)
    actual_grad = torch.autograd.grad(actual.square().mean(), (first, *model.parameters()))
    expected_grad = torch.autograd.grad(expected.square().mean(), (second, *original.parameters()))
    pairs = [(actual, expected), *zip(actual_grad, expected_grad, strict=True)]
    if any(not torch.isfinite(value).all() for pair in pairs for value in pair):
        raise ValueError("Nonfinite CPU implementation-parity value or gradient")
    errors = [float((a - b).detach().abs().max()) for a, b in pairs]
    initialized = state_digest(model.state_dict()) == state_digest(original.state_dict())
    return {
        "parity_gate_passed": initialized
        and all(torch.allclose(a, b, atol=1e-4, rtol=1e-4) for a, b in pairs),
        "initial_state_identical": initialized,
        "atol": 1e-4,
        "rtol": 1e-4,
        "records": len(indices),
        "forward_max_abs_error": errors[0],
        "input_gradient_max_abs_error": errors[1],
        "parameter_gradient_max_abs_error": max(errors[2:]),
        "batch_indices_sha256": sha256_json(indices.tolist()),
        "wall_seconds": time.perf_counter() - wall,
        "cpu_seconds": time.process_time() - cpu,
    }


def _begin() -> tuple[float, float]:
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return time.perf_counter(), time.process_time()


def _measurements(wall: float, cpu: float) -> dict[str, Any]:
    return {
        "wall_seconds": time.perf_counter() - wall,
        "cpu_seconds": time.process_time() - cpu,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def replay(factory: ModelFactory, real: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    """
    Independently replay two actual CUDA updates with an explicit successor model factory.

    Parameters
    ----------
    factory : callable
        Frozen successor factory.
    real : np.ndarray
        Bound physical TRAIN cache.
    config : dict
        Frozen recipe with unchanged batch and optimizer.

    Returns
    -------
    dict
        Exact loss/state replay and resource measurements; weights are discarded.
    """
    wall, cpu = _begin()
    runs = []
    for _ in range(2):
        model = factory(config, "cuda")
        initial = model_description(model)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
        )
        indices = np.random.default_rng(SAMPLING_SEED).integers(0, len(real), size=(2, config["batch_size"]))
        masks = torch.Generator().manual_seed(MASK_SEED)
        losses = [ssl_update(model, optimizer, cuda_batch(real, batch), masks) for batch in indices]
        torch.cuda.synchronize()
        runs.append(
            {
                "initialization": initial,
                "losses": losses,
                "final_state_sha256": state_digest(model.state_dict()),
            }
        )
        del model, optimizer
    return {
        "replay_gate_passed": runs[0] == runs[1],
        "steps_per_run": 2,
        "runs": runs,
        "batch_indices_sha256": sha256_json(indices.tolist()),
        **_measurements(wall, cpu),
    }


def _batch_hash(phase: str, inputs: dict[str, Any]) -> str:
    count, batch = WARMUP_STEPS + TIMED_STEPS, inputs["config"]["batch_size"]
    if phase == "ssl":
        indices = np.random.default_rng(SAMPLING_SEED).integers(0, len(inputs["real"]), size=(count, batch))
    elif phase == "inference":
        indices = np.arange(count * batch).reshape(count, batch) % len(inputs["validation"]["signals"])
    else:
        indices = inputs["schedule"][:count]
    return sha256_json(indices.tolist())


def measure_phase(factory: ModelFactory, phase: str, inputs: dict[str, Any]) -> dict[str, Any]:
    """
    Measure the original actual iteration path using only a factory-supplied successor.

    Parameters
    ----------
    factory : callable
        Frozen successor factory.
    phase : str
        SSL, trainable/frozen localization or inference phase.
    inputs : dict
        Original verified native data, schedule and class weight.

    Returns
    -------
    dict
        Actual warmed timing, memory, identical initialization and frozen-encoder check.
    """
    wall, cpu = _begin()
    model = factory(inputs["config"], "cuda")
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
    encoder_before = state_digest(model.encoder.state_dict())
    batch_hash = _batch_hash(phase, inputs)
    rng, masks = np.random.default_rng(SAMPLING_SEED), torch.Generator().manual_seed(MASK_SEED)
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
        "encoder_before_sha256": encoder_before,
        "encoder_after_sha256": encoder_after,
        "frozen_encoder_unchanged": encoder_before == encoder_after
        if phase == "frozen_localization"
        else None,
        "profile_weights_discarded": True,
        **_measurements(wall, cpu),
    }


def forecast(
    phases: dict[str, Any],
    wall: float,
    cpu: float,
    prior: float,
    failed: dict[str, Any],
    size: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Reuse the frozen forecast while adding the retained failed attempt's actual charges.

    Parameters
    ----------
    phases : dict
        New measured phases.
    wall, cpu : float
        Actual new profile wall/process CPU seconds.
    prior : float
        Original complete preparation CPU charge.
    failed : dict
        Verified failed original profile.
    size : int
        Entire current stream bytes.
    config : dict
        Original ceilings.

    Returns
    -------
    dict
        Unchanged forecast with additive failure-cost accounting.
    """
    result = resource_forecast(
        phases, wall + failed["wall_seconds"], cpu, prior + failed["cpu_seconds"], size, config
    )
    result.update(
        previous_failed_profile_wall_seconds=failed["wall_seconds"],
        previous_failed_profile_cpu_seconds=failed["cpu_seconds"],
        current_profile_wall_seconds=wall,
        current_profile_cpu_seconds=cpu,
    )
    return result
