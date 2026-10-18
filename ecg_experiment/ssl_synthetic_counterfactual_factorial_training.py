"""Profile and fit the frozen initialization-by-paired-loss factorial."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.paths import to_stored
from ecg_experiment.ssl_synthetic_model import localization_update, model_description, state_digest
from ecg_experiment.ssl_synthetic_model_v2 import DenseWaveformModel
from ecg_experiment.ssl_synthetic_paired_background_training import (
    _finite_model,
    _state_equal,
    case_schedule,
    matched_initial_models,
    patient_schedule,
)
from ecg_experiment.ssl_synthetic_preparation import BASE, write_new_json
from ecg_experiment.ssl_synthetic_real_transfer import _thread_environment
from ecg_experiment.ssl_synthetic_training import configure_torch, fresh_optimizer, save_model

ARMS = ("RB", "SB", "RP", "SP")
SOURCE_NAMES = (
    "ecg_experiment/ssl_synthetic_counterfactual_factorial_training.py",
    "scripts/experiments/run_ssl_synthetic_counterfactual_factorial.py",
    "tests/test_ssl_synthetic_counterfactual_factorial_training.py",
    "ecg_experiment/ssl_synthetic_counterfactual_factorial_evaluation.py",
    "scripts/reports/evaluate_ssl_synthetic_counterfactual_factorial.py",
    "tests/test_ssl_synthetic_counterfactual_factorial_evaluation.py",
    "ecg_experiment/ssl_synthetic_paired_background_training.py",
    "ecg_experiment/ssl_synthetic_model.py",
    "ecg_experiment/ssl_synthetic_model_v2.py",
)


def factorial_models(initial: Path, ssl: Path) -> dict[str, DenseWaveformModel]:
    """Create four fresh original states with common heads and matched within-initialization encoders."""
    base = matched_initial_models(initial, ssl)
    return {arm: copy.deepcopy(base[arm[0]]) for arm in ARMS}


def paired_penalty(
    probabilities: torch.Tensor,
    target: torch.Tensor,
    positive: torch.Tensor,
    control: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average per-cell target hinge and outside-support consistency equally over positive cases.

    Parameters
    ----------
    probabilities, target : torch.Tensor
        Native single-input outputs and exact boolean intervention masks for the entire batch.
    positive, control : torch.Tensor
        Aligned positive and same-patient/same-nuisance control row positions.
    margin : float
        Frozen probability margin; gradients flow through both observations without detach.

    Returns
    -------
    tuple of torch.Tensor
        Equally case-weighted target hinge and outside-support absolute response.
    """
    if probabilities.shape != target.shape or target.dtype != torch.bool:
        raise ValueError("Native target and probability geometry must agree")
    if not len(positive) or len(positive) != len(control):
        raise ValueError("Positive/control row positions must be nonempty and aligned")
    support = target[positive]
    counts = support.flatten(1).sum(1)
    total = support[0].numel()
    if not len(positive) or torch.any(counts == 0) or torch.any(counts == total) or target[control].any():
        raise ValueError("Every pair requires nonempty support/complement and a zero-support control")
    delta = probabilities[positive] - probabilities[control]
    hinge = (functional.relu(margin - delta) * support).flatten(1).sum(1) / counts
    consistency = (delta.abs() * ~support).flatten(1).sum(1) / (total - counts)
    return hinge.mean(), consistency.mean()


def batch_pairs(
    context: dict, indices: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Regenerate each TRAIN input once and verify exact positive/control identity before loss pairing."""
    from ecg_experiment.ssl_synthetic_counterfactual_factorial_common import render_checked

    if np.any(indices < 0) or np.any(indices >= 2000) or len(set(indices.tolist())) != len(indices):
        raise ValueError("Training batch must contain distinct admitted TRAIN case rows")
    pairs = [render_checked(context, int(index)) for index in indices]
    positions = {int(index): position for position, index in enumerate(indices)}
    positive, controls = [], []
    for position, index in enumerate(indices):
        case = context["cases"][int(index)]
        if not case["target"]:
            continue
        matched = case["matched_control_case_index"]
        expected = (int(index) // 4) * 4 + (3 if case["nuisance"] else 2)
        if matched != expected or matched not in positions:
            raise ValueError("Positive lacks its exact same-patient/nuisance batch control")
        control_position = positions[matched]
        control_case = context["cases"][matched]
        if case["nuisance_parameters"] != control_case["nuisance_parameters"] or not np.array_equal(
            pairs[position]["baseline"], pairs[control_position]["edited"]
        ):
            raise ValueError("Batch positive baseline and matched observed control differ")
        positive.append(position)
        controls.append(control_position)
    if len(positive) != 8 or len(indices) != 16:
        raise ValueError("Each actual batch requires eight positives and eight zero-mask inputs")
    return (
        torch.from_numpy(np.stack([p["edited"] for p in pairs])),
        torch.from_numpy(np.stack([p["support"] for p in pairs])),
        torch.tensor(positive, dtype=torch.long),
        torch.tensor(controls, dtype=torch.long),
    )


def factorial_update(
    context: dict,
    model: DenseWaveformModel,
    optimizer: torch.optim.Optimizer,
    indices: np.ndarray,
    arm: str,
    weight: float,
    device: str = "cuda",
) -> dict[str, float]:
    """Take one existing BCE or exact prospectively fixed paired-objective AdamW update."""
    signals, support, positive, control = batch_pairs(context, indices)
    signals, support = signals.to(device), support.to(device)
    if arm not in ARMS:
        raise ValueError("Unknown factorial arm")
    if arm.endswith("B"):
        loss = localization_update(model, optimizer, signals, support, weight)
        return {"bce": loss, "hinge": 0.0, "consistency": 0.0, "total": loss}
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(signals)
    if logits.shape != support.shape or logits.dtype != torch.float32 or not torch.isfinite(logits).all():
        raise ValueError("Nonfinite paired-objective logits")
    bce = functional.binary_cross_entropy_with_logits(
        logits, support.to(logits.dtype), pos_weight=logits.new_tensor(weight)
    )
    hinge, consistency = paired_penalty(
        logits.sigmoid(), support, positive.to(device), control.to(device), context["config"]["paired_margin"]
    )
    total = bce + context["config"]["paired_weight"] * (hinge + consistency)
    if not torch.isfinite(total):
        raise ValueError("Nonfinite paired-objective loss")
    total.backward()
    optimizer.step()
    return {
        name: float(value.detach())
        for name, value in (("bce", bce), ("hinge", hinge), ("consistency", consistency), ("total", total))
    }


def frozen_schedule_weight(context: dict) -> tuple[np.ndarray, float]:
    """Reuse the exact predecessor schedule and compute its native-grid class weight by source access."""
    from ecg_experiment.ssl_synthetic_counterfactual_factorial_common import support_for_case

    config = context["config"]
    if (
        config["schedule_seed"] != 20261024
        or config["localization_steps"] != 600
        or config["batch_size"] != 16
    ):
        raise ValueError("Matched exposure recipe changed")
    schedule = case_schedule(patient_schedule(config["schedule_seed"]))
    if (
        hashlib.sha256(schedule.tobytes()).hexdigest()
        != context["predecessor_training"]["schedule_array_sha256"]
    ):
        raise ValueError("Predecessor schedule bytes changed")
    counts = np.array([support_for_case(context, index).sum(dtype=np.int64) for index in range(2000)])
    positive = int(np.bincount(schedule.ravel(), minlength=2000) @ counts)
    total = int(schedule.size * 12 * 2500)
    if not 0 < positive < total:
        raise ValueError("Scheduled training masks require both classes")
    weight = min(20.0, (total - positive) / positive)
    if weight != context["predecessor_training"]["positive_class_weight"]:
        raise ValueError("Predecessor native schedule class weight differs")
    return schedule, weight


def bce_reproduction(arm: str, report: dict, predecessor: dict) -> dict:
    """Require exact final tensor/component identities and all 600 BCE losses before paired fits."""
    old = predecessor["arms"][arm[0]]
    names = ("encoder_sha256", "head_sha256", "full_model_sha256")
    if any(report["final_state"][name] != old["final_state"][name] for name in names):
        raise ValueError("BCE predecessor final tensor reproduction failed")
    if report["losses"] != old["losses"]:
        raise ValueError("BCE predecessor 600-loss reproduction failed")
    return {"passed": True, "losses_exact": 600, "final_tensor_digests_exact": list(names)}


def factorial_forecast(
    profile_wall: float, profile_cpu: float, preparation_cpu: float, training: dict, inference: dict
) -> dict:
    """Forecast all four 600-update fits and 192 batch-16 native-map inference calls at 2x cost."""
    inference_wall = 2 * 48 * sum(inference[arm]["mean_wall_seconds"] for arm in ARMS)
    inference_cpu = 2 * 48 * sum(inference[arm]["mean_cpu_seconds"] for arm in ARMS)
    return {
        "stage_gpu_wall_seconds": profile_wall
        + 2 * 600 * sum(training[arm]["mean_wall_seconds"] for arm in ARMS)
        + inference_wall
        + 120,
        "stage_cpu_seconds": preparation_cpu
        + profile_cpu
        + 2 * 600 * sum(training[arm]["mean_cpu_seconds"] for arm in ARMS)
        + inference_cpu
        + 120,
        "inference_batches": 192,
        "evaluation_reporting_reserve_seconds": 120 + max(inference_wall, inference_cpu),
    }


def recovery_check(
    context: dict,
    model: DenseWaveformModel,
    schedule: np.ndarray,
    arm: str,
    weight: float,
    path: Path,
    device: str = "cuda",
) -> dict:
    """Save one updated model/AdamW/RNG state and require two identical continued updates after reload."""
    if path.exists():
        raise FileExistsError("Recovery artifact cannot be overwritten")
    initial = model_description(model)
    model.to(device)
    optimizer = fresh_optimizer(model, context["config"])
    factorial_update(context, model, optimizer, schedule[0], arm, weight, device)
    saved_digest = state_digest(model.state_dict())
    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state()}
    if device == "cuda":
        state["cuda_rng"] = torch.cuda.get_rng_state()
    torch.save(state, path)
    direct = [factorial_update(context, model, optimizer, row, arm, weight, device) for row in schedule[1:3]]
    recovered = torch.load(path, weights_only=True, map_location="cpu")
    restored = DenseWaveformModel().to(device)
    restored.load_state_dict(recovered["model"], strict=True)
    if state_digest(restored.state_dict()) != saved_digest:
        raise ValueError("Recovered model state differs before continuation")
    restored_optimizer = fresh_optimizer(restored, context["config"])
    restored_optimizer.load_state_dict(recovered["optimizer"])
    torch.set_rng_state(recovered["rng"])
    if device == "cuda":
        torch.cuda.set_rng_state(recovered["cuda_rng"])
    replay = [
        factorial_update(context, restored, restored_optimizer, row, arm, weight, device)
        for row in schedule[1:3]
    ]
    _finite_model(model)
    _finite_model(restored)
    final_digest = state_digest(model.state_dict())
    if (
        direct != replay
        or final_digest != state_digest(restored.state_dict())
        or not _state_equal(optimizer.state_dict(), restored_optimizer.state_dict())
    ):
        raise ValueError("Exact two-next-update model/objective/AdamW replay failed")
    steps = {int(item["step"].item()) for item in restored_optimizer.state.values()}
    if steps != {3}:
        raise ValueError("Recovery optimizer accounting differs")
    return {
        "passed": True,
        "initialization": initial,
        "saved_model_sha256": saved_digest,
        "final_model_sha256": final_digest,
        "direct_next_components": direct,
        "restored_next_components": replay,
        "optimizer_states_equal": True,
        "optimizer_step_count": 3,
        "executed_update_calls": 5,
        "checkpoint_sha256": sha256_file(path),
    }


def _paths() -> tuple[Path, Path]:
    return BASE / "pilot_v1/training/initial.pt", BASE / "pilot_v1/training/ssl.pt"


def _clock(context: dict, wall: float, cpu: float, profile: dict | None = None) -> dict:
    return {
        "wall": wall,
        "cpu": cpu,
        "prior_stage_wall": profile["wall_seconds"] if profile else 0.0,
        "prior_stage_cpu": context["stage_preparation_cpu_seconds"]
        + (profile["cpu_seconds"] if profile else 0.0),
    }


def _prepare_context() -> dict:
    from ecg_experiment.ssl_synthetic_counterfactual_factorial_common import admit_context, storage_projection

    context = admit_context(SOURCE_NAMES)
    context["storage_projection"] = storage_projection()
    if (
        context["storage_projection"]["projected_whole_stream_bytes"]
        >= context["config"]["cache_bytes_ceiling"]
    ):
        raise ValueError("Whole factorial storage forecast fails")
    context["numerical_threads"] = _thread_environment()
    return context


def _profile_arm(
    context: dict, model: DenseWaveformModel, arm: str, schedule: np.ndarray, weight: float, clock: dict
) -> dict:
    from ecg_experiment.ssl_synthetic_counterfactual_factorial_common import check_budget

    model.to("cuda")
    optimizer = fresh_optimizer(model, context["config"])
    walls, cpus = [], []
    for step in range(30):
        check_budget(context, clock, reserve_seconds=120.0)
        torch.cuda.synchronize()
        wall, cpu = time.perf_counter(), time.process_time()
        factorial_update(context, model, optimizer, schedule[step], arm, weight)
        torch.cuda.synchronize()
        if step >= 10:
            walls.append(time.perf_counter() - wall)
            cpus.append(time.process_time() - cpu)
    _finite_model(model)
    return {
        "mean_wall_seconds": float(np.mean(walls)),
        "mean_cpu_seconds": float(np.mean(cpus)),
        "warmup_batches": 10,
        "timed_batches": 20,
    }


def _profile_inference(context: dict, model: DenseWaveformModel, clock: dict) -> dict:
    from ecg_experiment.ssl_synthetic_counterfactual_factorial_common import check_budget
    from ecg_experiment.ssl_synthetic_counterfactual_factorial_evaluation import predict_batch

    model.to("cuda").eval()
    before = state_digest(model.state_dict())
    cal = context["roles"]["calibration"]
    walls, cpus = [], []
    for batch in range(30):
        check_budget(context, clock, reserve_seconds=120.0)
        indices = [cal[(16 * batch + offset) % len(cal)] for offset in range(16)]
        torch.cuda.synchronize()
        wall, cpu = time.perf_counter(), time.process_time()
        predict_batch(model, context, indices)
        torch.cuda.synchronize()
        if batch >= 10:
            walls.append(time.perf_counter() - wall)
            cpus.append(time.process_time() - cpu)
    if state_digest(model.state_dict()) != before:
        raise ValueError("Inference profile changed model state")
    return {
        "mean_wall_seconds": float(np.mean(walls)),
        "mean_cpu_seconds": float(np.mean(cpus)),
        "warmup_batches": 10,
        "timed_batches": 20,
        "model_state_unchanged": True,
    }


def profile_factorial() -> dict:
    """Profile all four actual objectives/recoveries and calibration-only inference once under GPU lock."""
    from ecg_experiment.ssl_synthetic_counterfactual_factorial_common import (
        PROFILE,
        check_budget,
        verify_context,
    )
    from ecg_experiment.ssl_synthetic_paired_background_training import _failure, _receipt

    if PROFILE.exists():
        raise FileExistsError("Factorial profile destination must be new")
    wall, cpu = time.perf_counter(), time.process_time()
    sources = {name: sha256_file(ROOT / name) for name in SOURCE_NAMES}
    inputs, progress = {}, {"completed": False, "disposable_update_calls": 0, "inference_batches": 0}
    with gpu_lock("cuda", blocking=False):
        PROFILE.mkdir(parents=True, exist_ok=False)
        try:
            context = _prepare_context()
            sources, inputs = context["sources"], context["inputs"]
            clock = _clock(context, wall, cpu)
            configure_torch()
            torch.cuda.reset_peak_memory_stats()
            schedule, weight = frozen_schedule_weight(context)
            timings, inference, recovery = {}, {}, {}
            for arm in ARMS:
                check_budget(context, clock, reserve_seconds=120.0)
                recovery[arm] = recovery_check(
                    context,
                    factorial_models(*_paths())[arm],
                    schedule,
                    arm,
                    weight,
                    PROFILE / f"recovery_{arm}.pt",
                )
                timings[arm] = _profile_arm(
                    context, factorial_models(*_paths())[arm], arm, schedule, weight, clock
                )
                inference[arm] = _profile_inference(context, factorial_models(*_paths())[arm], clock)
                progress["disposable_update_calls"] += 35
                progress["inference_batches"] += 30
                torch.cuda.empty_cache()
            write_new_json(PROFILE / "regeneration_checks.json", context["regeneration_checks"])
            verify_context(context)
            check_budget(context, clock)
            progress["completed"] = True
            result = _receipt(
                PROFILE,
                context,
                clock,
                {
                    "study_id": context["config"]["study_id"],
                    "training_profile": timings,
                    "inference_profile": inference,
                    "recovery": recovery,
                    "positive_class_weight": weight,
                    "progress": progress,
                    "scientific_training_updates": 0,
                    "performance_metrics_computed": False,
                    "probe_maps_saved": False,
                    "scope_exceptions": context["scope_exceptions"],
                },
            )
            forecast = factorial_forecast(
                result["wall_seconds"],
                result["cpu_seconds"],
                context["stage_preparation_cpu_seconds"],
                timings,
                inference,
            )
            result["forecast"] = forecast
            config = context["config"]
            result["profile_gate_passed"] = (
                forecast["stage_gpu_wall_seconds"] < config["gpu_seconds_stage_ceiling"]
                and forecast["stage_cpu_seconds"] < config["cpu_seconds_stage_ceiling"]
                and context["prior_gpu_wall_seconds"] + forecast["stage_gpu_wall_seconds"]
                < config["gpu_seconds_ceiling"]
                and context["prior_cpu_seconds"] + forecast["stage_cpu_seconds"]
                < config["cpu_seconds_ceiling"]
            )
            check_budget(context, clock)
            write_new_json(PROFILE / "summary.json", result)
            return result
        except Exception as error:
            _failure(PROFILE, sources, inputs, progress, error, wall, cpu)
            raise RuntimeError("Factorial profile stopped; inspect its local failure receipt") from None


def _fit_arm(
    context: dict,
    model: DenseWaveformModel,
    arm: str,
    schedule: np.ndarray,
    weight: float,
    clock: dict,
    reserve: float,
    log: Any,
    progress: dict,
    destination: Path,
) -> dict:
    from ecg_experiment.ssl_synthetic_counterfactual_factorial_common import check_budget

    initial = {**model_description(model), "full_model_sha256": state_digest(model.state_dict())}
    model.to("cuda")
    optimizer = fresh_optimizer(model, context["config"])
    if optimizer.state:
        raise ValueError("Factorial arm inherited optimizer state")
    losses, components = [], []
    wall, cpu = time.perf_counter(), time.process_time()
    for step, indices in enumerate(schedule, start=1):
        check_budget(context, clock, reserve_seconds=reserve)
        values = factorial_update(context, model, optimizer, indices, arm, weight)
        losses.append(values["total"])
        components.append(values)
        progress["updates"][arm] = step
        if step % 100 == 0:
            log.write(json.dumps({"arm": arm, "step": step, **values}) + "\n")
            log.flush()
    torch.cuda.synchronize()
    _finite_model(model)
    steps = {int(value["step"].item()) for value in optimizer.state.values()}
    if len(losses) != 600 or steps != {600}:
        raise ValueError("Factorial optimizer update budget differs")
    final = {**model_description(model), "full_model_sha256": state_digest(model.state_dict())}
    checkpoint = save_model(model, destination / f"{arm}.pt")
    restored = DenseWaveformModel()
    restored.load_state_dict(
        torch.load(destination / f"{arm}.pt", weights_only=True, map_location="cpu"), strict=True
    )
    if state_digest(restored.state_dict()) != final["full_model_sha256"]:
        raise ValueError("Factorial final state does not reload exactly")
    return {
        "initialization": initial,
        "final_state": final,
        "checkpoint_sha256": checkpoint,
        "steps": 600,
        "optimizer_step_count": 600,
        "fresh_optimizer": True,
        "presentations": int(schedule.size),
        "positive_presentations": 4800,
        "zero_mask_presentations": 4800,
        "unique_presented_subjects": len(np.unique(schedule // 4)),
        "unique_presented_cases": len(np.unique(schedule)),
        "losses": losses,
        "loss_components": components,
        "all_losses_finite": True,
        "checkpoint_reloaded_exactly": True,
        "wall_seconds": time.perf_counter() - wall,
        "cpu_seconds": time.process_time() - cpu,
    }


def train_factorial() -> dict:
    """Fit the exact four 600-step arms, requiring BCE replication before either new paired fit."""
    from ecg_experiment.ssl_synthetic_counterfactual_factorial_common import (
        PROFILE,
        TRAINING,
        check_budget,
        verify_context,
    )
    from ecg_experiment.ssl_synthetic_paired_background_training import _failure, _receipt

    if TRAINING.exists():
        raise FileExistsError("Factorial training destination must be new")
    wall, cpu = time.perf_counter(), time.process_time()
    sources = {name: sha256_file(ROOT / name) for name in SOURCE_NAMES}
    inputs, progress = {}, {"completed": False, "updates": dict.fromkeys(ARMS, 0)}
    with gpu_lock("cuda", blocking=False):
        TRAINING.mkdir(parents=True, exist_ok=False)
        try:
            context = _prepare_context()
            sources, inputs = context["sources"], context["inputs"]
            profile = json.loads((PROFILE / "summary.json").read_text())
            if not profile["profile_gate_passed"] or not all(
                profile["recovery"][arm]["passed"] for arm in ARMS
            ):
                raise ValueError("Successful factorial resource/recovery profile required")
            from ecg_experiment.ssl_synthetic_preparation import verify_hashes

            verify_hashes(profile["source_sha256"])
            for name, digest in profile["output_sha256"].items():
                if sha256_file(PROFILE / name) != digest:
                    raise ValueError("Profile artifact changed")
            from ecg_experiment.ssl_synthetic_evaluation import _merge_hashes

            _merge_hashes(sources, profile["source_sha256"])
            _merge_hashes(inputs, profile["input_sha256"])
            _merge_hashes(
                inputs, {to_stored(PROFILE / "summary.json"): sha256_file(PROFILE / "summary.json")}
            )
            _merge_hashes(
                inputs,
                {to_stored(PROFILE / name): digest for name, digest in profile["output_sha256"].items()},
            )
            verify_context(context)
            clock = _clock(context, wall, cpu, profile)
            reserve = profile["forecast"]["evaluation_reporting_reserve_seconds"]
            configure_torch()
            torch.cuda.reset_peak_memory_stats()
            check_budget(context, clock, reserve_seconds=reserve)
            schedule, weight = frozen_schedule_weight(context)
            if weight != profile["positive_class_weight"]:
                raise ValueError("Profile and scientific class weights differ")
            np.save(TRAINING / "schedule.npy", schedule, allow_pickle=False)
            write_new_json(
                TRAINING / "admission.json",
                {
                    "source_sha256": sources,
                    "input_sha256": inputs,
                    "positive_class_weight": weight,
                    "evaluation_reporting_reserve_seconds": reserve,
                },
            )
            reports, closure = {}, {}
            models = factorial_models(*_paths())
            with (TRAINING / "progress.jsonl").open("x") as log:
                for arm in ARMS:
                    reports[arm] = _fit_arm(
                        context,
                        models.pop(arm),
                        arm,
                        schedule,
                        weight,
                        clock,
                        reserve,
                        log,
                        progress,
                        TRAINING,
                    )
                    if arm.endswith("B"):
                        closure[arm] = bce_reproduction(arm, reports[arm], context["predecessor_training"])
                    torch.cuda.empty_cache()
            write_new_json(TRAINING / "regeneration_checks.json", context["regeneration_checks"])
            verify_context(context)
            check_budget(context, clock, reserve_seconds=reserve)
            progress["completed"] = True
            result = _receipt(
                TRAINING,
                context,
                clock,
                {
                    "study_id": context["config"]["study_id"],
                    "training_completed": True,
                    "arms": reports,
                    "bce_reproduction": closure,
                    "schedule_array_sha256": hashlib.sha256(schedule.tobytes()).hexdigest(),
                    "schedule_identical_across_arms": True,
                    "positive_class_weight": weight,
                    "progress": progress,
                    "additional_ssl_steps": 0,
                    "performance_scored": False,
                    "profile_wall_seconds": profile["wall_seconds"],
                    "profile_cpu_seconds": profile["cpu_seconds"],
                    "stage_preparation_cpu_seconds": context["stage_preparation_cpu_seconds"],
                    "evaluation_reporting_reserve_seconds": reserve,
                    "scope_exceptions": context["scope_exceptions"],
                },
            )
            check_budget(context, clock, reserve_seconds=reserve)
            write_new_json(TRAINING / "summary.json", result)
            return result
        except Exception as error:
            _failure(TRAINING, sources, inputs, progress, error, wall, cpu)
            raise RuntimeError("Factorial training stopped; inspect its local failure receipt") from None
