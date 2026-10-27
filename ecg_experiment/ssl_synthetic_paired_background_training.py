"""Profile and fit matched R/S localizers on qualified paired real-background changes."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.paths import to_stored
from ecg_experiment.ssl_synthetic_data import positive_class_weight
from ecg_experiment.ssl_synthetic_evaluation import _merge_hashes, verify_inference_hashes
from ecg_experiment.ssl_synthetic_model import localization_update, model_description, state_digest
from ecg_experiment.ssl_synthetic_model_v2 import DenseWaveformModel
from ecg_experiment.ssl_synthetic_preparation import BASE, stream_bytes, write_new_json
from ecg_experiment.ssl_synthetic_real_transfer import _thread_environment
from ecg_experiment.ssl_synthetic_training import configure_torch, fresh_optimizer, make_arm, save_model

ARMS = ("R", "S")


def patient_schedule(seed: int = 20261024) -> np.ndarray:
    """Return the frozen 600-by-four patient schedule from successive 500-patient permutations.

    Parameters
    ----------
    seed : int
        Dedicated schedule seed, independent of augmentation and model initialization.

    Returns
    -------
    np.ndarray
        Integer TRAIN background positions; 2,400 presentations of all 500 patients.
    """
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.permutation(500) for _ in range(5)])[:2400].reshape(600, 4)


def case_schedule(patients: np.ndarray) -> np.ndarray:
    """Expand each scheduled patient into adjacent focal/repeated/unchanged/nuisance cases.

    Parameters
    ----------
    patients : np.ndarray
        The exact 600-by-four TRAIN patient schedule.

    Returns
    -------
    np.ndarray
        Global case indices 0 through 1,999 in a 600-by-16 matrix shared by R and S.
    """
    if patients.shape != (600, 4) or not np.issubdtype(patients.dtype, np.integer):
        raise ValueError("Expected the fixed 600-by-four integer patient schedule")
    if patients.min() < 0 or patients.max() >= 500:
        raise ValueError("Training schedule includes a non-TRAIN patient")
    return (patients[:, :, None] * 4 + np.arange(4)).reshape(600, 16)


def matched_initial_models(initial_path: Path, ssl_path: Path) -> dict[str, DenseWaveformModel]:
    """Load the original initial state and SSL encoder without carrying optimizer or old fitted heads.

    Parameters
    ----------
    initial_path, ssl_path : Path
        Source-bound original CPU model-only checkpoint artifacts.

    Returns
    -------
    dict
        Independent R and S CPU models with identical original localization heads.
    """
    initial = DenseWaveformModel()
    initial.load_state_dict(torch.load(initial_path, map_location="cpu", weights_only=True), strict=True)
    ssl = DenseWaveformModel()
    ssl.load_state_dict(torch.load(ssl_path, map_location="cpu", weights_only=True), strict=True)
    models = {arm: make_arm(initial, ssl.encoder.state_dict(), arm) for arm in ARMS}
    head = state_digest(initial.head.state_dict())
    if any(state_digest(model.head.state_dict()) != head for model in models.values()):
        raise ValueError("Original common localization head changed")
    if state_digest(models["R"].encoder.state_dict()) != state_digest(initial.encoder.state_dict()):
        raise ValueError("R encoder does not match original initialization")
    if state_digest(models["S"].encoder.state_dict()) != state_digest(ssl.encoder.state_dict()):
        raise ValueError("S encoder does not match original SSL initialization")
    if any(not parameter.requires_grad for model in models.values() for parameter in model.parameters()):
        raise ValueError("Both paired-background encoders and heads must be trainable")
    return models


def render_training_batch(context: dict, indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Regenerate and verify TRAIN cases, returning only observed input and exact target tensors.

    Parameters
    ----------
    context : dict
        Qualified shared context; paired baselines and parameters remain outside network features.
    indices : np.ndarray
        Global TRAIN case indices in the frozen ordered schedule.

    Returns
    -------
    tuple of torch.Tensor
        Independent contiguous CPU signal and boolean support batches.
    """
    from ecg_experiment.ssl_synthetic_paired_background_common import render_checked

    if np.any(indices < 0) or np.any(indices >= 2000):
        raise ValueError("Optimizer batch contains a non-TRAIN case")
    pairs = [render_checked(context, int(index)) for index in indices]
    return (
        torch.from_numpy(np.stack([pair["edited"] for pair in pairs])),
        torch.from_numpy(np.stack([pair["support"] for pair in pairs])),
    )


def update_batch(
    context: dict,
    model: DenseWaveformModel,
    optimizer: torch.optim.Optimizer,
    indices: np.ndarray,
    weight: float,
    device: str = "cuda",
) -> float:
    """Run the actual deterministic rerender, transfer, weighted-BCE and AdamW path once.

    Parameters
    ----------
    context, model, optimizer : dict, model, optimizer
        Qualified cases and disposable or scientific arm state.
    indices : np.ndarray
        Frozen TRAIN case indices.
    weight : float
        Actual-schedule-only positive class weight shared by both arms.
    device : str
        CUDA for execution; CPU is supported for artificial tests.

    Returns
    -------
    float
        Finite loss before the update.
    """
    signal, support = render_training_batch(context, indices)
    return localization_update(model, optimizer, signal.to(device), support.to(device), weight)


def _state_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor):
        return (
            isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and torch.equal(left.cpu(), right.cpu())
        )
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_state_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_state_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return bool(left == right)


def next_update_replay(
    context: dict,
    model: DenseWaveformModel,
    schedule: np.ndarray,
    weight: float,
    path: Path,
    device: str = "cuda",
) -> dict[str, Any]:
    """Check two exact next updates after restoring model, AdamW and torch RNG state from disk.

    Parameters
    ----------
    context, model : dict, model
        Qualified cases and a disposable original arm initialization.
    schedule : np.ndarray
        Frozen ordered TRAIN batches; the first three are used without alteration.
    weight : float
        Fixed actual-schedule class weight.
    path : Path
        New local recovery artifact, retained and source-bound.
    device : str
        CUDA in the profile; CPU for artificial recovery tests.

    Returns
    -------
    dict
        Exact recovery hashes, losses and optimizer accounting; never a model score.
    """
    if path.exists():
        raise FileExistsError("Replay state cannot be overwritten")
    initial = model_description(model)
    model.to(device)
    optimizer = fresh_optimizer(model, context["config"])
    first_loss = update_batch(context, model, optimizer, schedule[0], weight, device)
    saved_digest = state_digest(model.state_dict())
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
    }
    if device == "cuda":
        payload["cuda_rng"] = torch.cuda.get_rng_state()
    torch.save(payload, path)
    direct_losses = [update_batch(context, model, optimizer, row, weight, device) for row in schedule[1:3]]
    direct_digest = state_digest(model.state_dict())
    recovery = torch.load(path, map_location="cpu", weights_only=True)
    restored = DenseWaveformModel().to(device)
    restored.load_state_dict(recovery["model"], strict=True)
    loaded_digest = state_digest(restored.state_dict())
    restored_optimizer = fresh_optimizer(restored, context["config"])
    restored_optimizer.load_state_dict(recovery["optimizer"])
    torch.set_rng_state(recovery["torch_rng"])
    if device == "cuda":
        torch.cuda.set_rng_state(recovery["cuda_rng"])
    restored_losses = [
        update_batch(context, restored, restored_optimizer, row, weight, device) for row in schedule[1:3]
    ]
    recovered_digest = state_digest(restored.state_dict())
    steps = [int(state["step"].item()) for state in restored_optimizer.state.values()]
    equal = (
        saved_digest == loaded_digest
        and direct_losses == restored_losses
        and direct_digest == recovered_digest
        and _state_equal(optimizer.state_dict(), restored_optimizer.state_dict())
        and bool(steps)
        and set(steps) == {3}
    )
    if not equal:
        raise ValueError("Exact two-next-update model/AdamW recovery failed")
    return {
        "passed": True,
        "initialization": initial,
        "saved_model_sha256": saved_digest,
        "loaded_model_sha256": loaded_digest,
        "final_model_sha256": direct_digest,
        "recovered_final_model_sha256": recovered_digest,
        "first_loss": first_loss,
        "direct_next_losses": direct_losses,
        "restored_next_losses": restored_losses,
        "optimizer_states_equal": True,
        "optimizer_step_count": 3,
        "executed_update_calls": 5,
        "checkpoint_sha256": sha256_file(path),
    }


def profile_forecast(
    profile_wall: float,
    profile_cpu: float,
    preparation_cpu: float,
    training_times: dict[str, dict[str, float]],
    inference_times: dict[str, float],
) -> dict[str, float]:
    """Forecast both 600-update fits and 96 inference batches with a 2x margin and 120s reserve.

    Parameters
    ----------
    profile_wall, profile_cpu, preparation_cpu : float
        All actually executed profile/recovery/verification and prior preparation costs.
    training_times : dict
        Per-arm mean timed batch wall and CPU seconds on the complete update path.
    inference_times : dict
        Actual calibration-A inference mean batch wall and CPU seconds.

    Returns
    -------
    dict
        Stage forecast and minimum evaluation/reporting reservation for scientific training.
    """
    infer_wall = 2 * 96 * inference_times["mean_wall_seconds"]
    infer_cpu = 2 * 96 * inference_times["mean_cpu_seconds"]
    return {
        "stage_gpu_wall_seconds": profile_wall
        + 2 * 600 * sum(training_times[arm]["mean_wall_seconds"] for arm in ARMS)
        + infer_wall
        + 120,
        "stage_cpu_seconds": preparation_cpu
        + profile_cpu
        + 2 * 600 * sum(training_times[arm]["mean_cpu_seconds"] for arm in ARMS)
        + infer_cpu
        + 120,
        "evaluation_reporting_reserve_seconds": 120 + max(infer_wall, infer_cpu),
        "inference_batches": 96,
    }


SOURCE_NAMES = (
    "ecg_experiment/ssl_synthetic_paired_background_training.py",
    "scripts/experiments/run_ssl_synthetic_paired_background.py",
    "tests/test_ssl_synthetic_paired_background_training.py",
    "ecg_experiment/ssl_synthetic_paired_background_evaluation.py",
    "scripts/reports/evaluate_ssl_synthetic_paired_background.py",
    "tests/test_ssl_synthetic_paired_background_evaluation.py",
)
INITIAL_PATH = BASE / "pilot_v1/training/initial.pt"
SSL_PATH = BASE / "pilot_v1/training/ssl.pt"


def _failure(
    path: Path, sources: dict, inputs: dict, progress: dict, error: Exception, wall: float, cpu: float
) -> None:
    write_new_json(
        path / "failure.json",
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "source_sha256": sources,
            "input_sha256": inputs,
            "progress": progress,
            "failure_type": type(error).__name__,
            "failure_detail": str(error),
            "wall_seconds": time.perf_counter() - wall,
            "cpu_seconds": time.process_time() - cpu,
            "performance_scored": False,
        },
    )


def _fixed_recipe(context: dict) -> tuple[np.ndarray, np.ndarray, float]:
    config = context["config"]
    expected = {
        "localization_steps": 600,
        "batch_size": 16,
        "patients_per_batch": 4,
        "schedule_seed": 20261024,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "positive_weight_cap": 20.0,
        "profile_warmup_batches": 10,
        "profile_timed_batches": 20,
    }
    if any(config[key] != value for key, value in expected.items()):
        raise ValueError("Frozen matched-training recipe changed")
    patients = patient_schedule(config["schedule_seed"])
    cases = case_schedule(patients)
    weight = positive_class_weight({"support": context["support"][:2000]}, cases)
    return patients, cases, weight


def _finite_model(model: DenseWaveformModel) -> None:
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("Nonfinite model state")


def _inference_batch(context: dict, model: DenseWaveformModel, indices: list[int]) -> None:
    from ecg_experiment.ssl_synthetic_paired_background_evaluation import predict_batch

    if any(context["cases"][index]["role"] != "calibration" for index in indices):
        raise ValueError("Resource inference profile can only access calibration A")
    model.eval()
    probability = predict_batch(model, context, indices)
    if probability.shape != (16, 12, 2500) or not np.isfinite(probability).all():
        raise ValueError("Invalid native profile probability map")


def _costs(context: dict, clock: dict) -> dict[str, float]:
    wall, cpu = time.perf_counter() - clock["wall"], time.process_time() - clock["cpu"]
    return {
        "wall_seconds": wall,
        "cpu_seconds": cpu,
        "stage_gpu_wall_seconds": clock["prior_stage_wall"] + wall,
        "stage_cpu_seconds": clock["prior_stage_cpu"] + cpu,
        "global_gpu_wall_seconds": context["prior_gpu_wall_seconds"] + clock["prior_stage_wall"] + wall,
        "global_cpu_seconds": context["prior_cpu_seconds"] + clock["prior_stage_cpu"] + cpu,
    }


def _receipt(path: Path, context: dict, clock: dict, fields: dict) -> dict:
    if stream_bytes() >= context["config"]["cache_bytes_ceiling"]:
        raise ValueError("Whole-stream storage ceiling reached")
    outputs = {item.name: sha256_file(item) for item in path.iterdir() if item.is_file()}
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": context["sources"],
        "input_sha256": context["inputs"],
        "output_sha256": outputs,
        "storage_projection": context["storage_projection"],
        **fields,
        **_costs(context, clock),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "stream_bytes": stream_bytes(),
        "unique_regenerated_cases": len(context["regeneration_checks"]),
        "all_regenerated_hashes_match": True,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "numerical_threads": context["numerical_threads"],
        "original_advancement_passed": False,
        "clinical_claim": False,
    }


def profile_paired_background() -> dict[str, Any]:
    """Execute one disposable R/S update and calibration-inference resource profile under GPU lock.

    Returns
    -------
    dict
        Source-bound actual-path timing/recovery admission receipt; no performance calculation.
    """
    from ecg_experiment.ssl_synthetic_paired_background_common import (
        PROFILE,
        admit_context,
        check_budget,
    )

    if PROFILE.exists():
        raise FileExistsError("Paired profile destination must be new")
    wall, cpu = time.perf_counter(), time.process_time()
    sources = {name: sha256_file(ROOT / name) for name in SOURCE_NAMES}
    inputs, progress = {}, {"completed": False, "training_update_calls": 0, "inference_batches": 0}
    with gpu_lock("cuda", blocking=False):
        PROFILE.mkdir(parents=True, exist_ok=False)
        try:
            context = admit_context(SOURCE_NAMES)
            sources, inputs = context["sources"], context["inputs"]
            from ecg_experiment.ssl_synthetic_paired_background_common import storage_projection

            context["storage_projection"] = storage_projection()
            if (
                context["storage_projection"]["projected_whole_stream_bytes"]
                >= context["config"]["cache_bytes_ceiling"]
            ):
                raise ValueError("Whole-study storage forecast fails")
            clock = {
                "wall": wall,
                "cpu": cpu,
                "prior_stage_wall": 0.0,
                "prior_stage_cpu": context["stage_preparation_cpu_seconds"],
            }
            context["numerical_threads"] = _thread_environment()
            configure_torch()
            torch.cuda.reset_peak_memory_stats()
            _, schedule, weight = _fixed_recipe(context)
            models = matched_initial_models(INITIAL_PATH, SSL_PATH)
            timings, replay = {}, {}
            for arm in ARMS:
                check_budget(context, clock, reserve_seconds=120.0)
                replay[arm] = next_update_replay(
                    context, models[arm], schedule, weight, PROFILE / f"replay_{arm}.pt"
                )
                progress["training_update_calls"] += replay[arm]["executed_update_calls"]
                del models[arm]
                model = matched_initial_models(INITIAL_PATH, SSL_PATH)[arm].to("cuda")
                optimizer = fresh_optimizer(model, context["config"])
                walls, cpus = [], []
                for batch in range(30):
                    check_budget(context, clock, reserve_seconds=120.0)
                    torch.cuda.synchronize()
                    start_wall, start_cpu = time.perf_counter(), time.process_time()
                    update_batch(context, model, optimizer, schedule[batch], weight)
                    torch.cuda.synchronize()
                    elapsed_wall, elapsed_cpu = (
                        time.perf_counter() - start_wall,
                        time.process_time() - start_cpu,
                    )
                    progress["training_update_calls"] += 1
                    if batch >= 10:
                        walls.append(elapsed_wall)
                        cpus.append(elapsed_cpu)
                _finite_model(model)
                timings[arm] = {
                    "mean_wall_seconds": float(np.mean(walls)),
                    "mean_cpu_seconds": float(np.mean(cpus)),
                    "timed_batches": 20,
                    "warmup_batches": 10,
                }
                del model, optimizer
                torch.cuda.empty_cache()
            model = matched_initial_models(INITIAL_PATH, SSL_PATH)["R"].to("cuda")
            before = state_digest(model.state_dict())
            walls, cpus = [], []
            cal = context["roles"]["calibration"]
            for batch in range(30):
                check_budget(context, clock, reserve_seconds=120.0)
                indices = [cal[(batch * 16 + offset) % len(cal)] for offset in range(16)]
                torch.cuda.synchronize()
                start_wall, start_cpu = time.perf_counter(), time.process_time()
                _inference_batch(context, model, indices)
                torch.cuda.synchronize()
                elapsed_wall, elapsed_cpu = time.perf_counter() - start_wall, time.process_time() - start_cpu
                progress["inference_batches"] += 1
                if batch >= 10:
                    walls.append(elapsed_wall)
                    cpus.append(elapsed_cpu)
            if state_digest(model.state_dict()) != before:
                raise ValueError("Inference profile changed fresh R state")
            inference = {
                "mean_wall_seconds": float(np.mean(walls)),
                "mean_cpu_seconds": float(np.mean(cpus)),
                "warmup_batches": 10,
                "timed_batches": 20,
                "initialization": "fresh original R",
                "model_state_unchanged": True,
            }
            write_new_json(PROFILE / "regeneration_checks.json", context["regeneration_checks"])
            verify_inference_hashes(sources | inputs)
            check_budget(context, clock)
            progress["completed"] = True
            result = _receipt(
                PROFILE,
                context,
                clock,
                {
                    "training_profile": timings,
                    "inference_profile": inference,
                    "next_update_replay": replay,
                    "positive_class_weight": weight,
                    "progress": progress,
                    "scientific_training_updates": 0,
                    "performance_metrics_computed": False,
                    "probe_maps_saved": False,
                },
            )
            forecast = profile_forecast(
                result["wall_seconds"],
                result["cpu_seconds"],
                context["stage_preparation_cpu_seconds"],
                timings,
                inference,
            )
            result["forecast"] = forecast
            result["profile_gate_passed"] = (
                forecast["stage_gpu_wall_seconds"] < context["config"]["gpu_seconds_stage_ceiling"]
                and forecast["stage_cpu_seconds"] < context["config"]["cpu_seconds_stage_ceiling"]
                and context["prior_gpu_wall_seconds"] + forecast["stage_gpu_wall_seconds"] < 7200
                and context["prior_cpu_seconds"] + forecast["stage_cpu_seconds"] < 14400
            )
            check_budget(context, clock)
            write_new_json(PROFILE / "summary.json", result)
            return result
        except Exception as error:
            _failure(PROFILE, sources, inputs, progress, error, wall, cpu)
            raise RuntimeError("Paired profile stopped; inspect the local failure receipt") from None


def _fit_arm(
    context: dict,
    model: DenseWaveformModel,
    arm: str,
    schedule: np.ndarray,
    weight: float,
    clock: dict,
    reserve: float,
    progress: dict,
    log: Any,
    path: Path,
) -> dict:
    from ecg_experiment.ssl_synthetic_paired_background_common import check_budget

    initial = model_description(model)
    initial["full_model_sha256"] = state_digest(model.state_dict())
    model.to("cuda")
    optimizer = fresh_optimizer(model, context["config"])
    if optimizer.state:
        raise ValueError("Scientific arm inherited optimizer state")
    phase_wall, phase_cpu = time.perf_counter(), time.process_time()
    losses = []
    for step, indices in enumerate(schedule, start=1):
        check_budget(context, clock, reserve_seconds=reserve)
        loss = update_batch(context, model, optimizer, indices, weight)
        losses.append(loss)
        progress["updates"][arm] = step
        if step % 100 == 0:
            log.write(
                json.dumps(
                    {
                        "arm": arm,
                        "step": step,
                        "loss": loss,
                        "wall_seconds": time.perf_counter() - phase_wall,
                    }
                )
                + "\n"
            )
            log.flush()
    torch.cuda.synchronize()
    _finite_model(model)
    optimizer_steps = [int(state["step"].item()) for state in optimizer.state.values()]
    if len(losses) != 600 or not optimizer_steps or set(optimizer_steps) != {600}:
        raise ValueError("Scientific optimizer exposure differs from the frozen recipe")
    final_state = {
        **model_description(model),
        "full_model_sha256": state_digest(model.state_dict()),
    }
    checkpoint_hash = save_model(model, path / f"{arm}.pt")
    restored = DenseWaveformModel()
    restored.load_state_dict(
        torch.load(path / f"{arm}.pt", weights_only=True, map_location="cpu"), strict=True
    )
    if state_digest(restored.state_dict()) != final_state["full_model_sha256"]:
        raise ValueError("Saved scientific checkpoint does not reload exactly")
    report = {
        "steps": 600,
        "optimizer_step_count": 600,
        "fresh_optimizer": True,
        "presentations": int(schedule.size),
        "positive_presentations": int(np.sum(schedule % 4 < 2)),
        "zero_mask_presentations": int(np.sum(schedule % 4 >= 2)),
        "unique_presented_subjects": len(np.unique(schedule // 4)),
        "unique_presented_cases": len(np.unique(schedule)),
        "initialization": initial,
        "final_state": final_state,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_reloaded_exactly": True,
        "all_losses_finite": True,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "losses": losses,
        "wall_seconds": time.perf_counter() - phase_wall,
        "cpu_seconds": time.process_time() - phase_cpu,
    }
    del model, optimizer, restored
    torch.cuda.empty_cache()
    return report


def train_paired_background() -> dict[str, Any]:
    """Fit exactly 600 matched updates for each arm after successful actual-path admission.

    Returns
    -------
    dict
        Final-only checkpoints, finite losses, exposure/component hashes and resource receipt.
    """
    from ecg_experiment.ssl_synthetic_paired_background_common import (
        PROFILE,
        TRAINING,
        admit_context,
        check_budget,
    )

    if TRAINING.exists():
        raise FileExistsError("Paired training destination must be new")
    wall, cpu = time.perf_counter(), time.process_time()
    sources = {name: sha256_file(ROOT / name) for name in SOURCE_NAMES}
    inputs, progress = {}, {"completed": False, "updates": dict.fromkeys(ARMS, 0)}
    with gpu_lock("cuda", blocking=False):
        TRAINING.mkdir(parents=True, exist_ok=False)
        try:
            context = admit_context(SOURCE_NAMES)
            sources, inputs = context["sources"], context["inputs"]
            from ecg_experiment.ssl_synthetic_paired_background_common import storage_projection

            context["storage_projection"] = storage_projection()
            if (
                context["storage_projection"]["projected_whole_stream_bytes"]
                >= context["config"]["cache_bytes_ceiling"]
            ):
                raise ValueError("Whole-study storage forecast fails")
            profile_path = PROFILE / "summary.json"
            profile = json.loads(profile_path.read_text())
            verify_inference_hashes(profile["source_sha256"] | profile["input_sha256"])
            if not profile["profile_gate_passed"] or not all(
                profile["next_update_replay"][arm]["passed"] for arm in ARMS
            ):
                raise ValueError("Actual paired-background resource/recovery profile failed")
            _merge_hashes(sources, profile["source_sha256"])
            _merge_hashes(inputs, profile["input_sha256"])
            _merge_hashes(inputs, {to_stored(profile_path): sha256_file(profile_path)})
            for name, digest in profile["output_sha256"].items():
                _merge_hashes(inputs, {to_stored(PROFILE / name): digest})
            verify_inference_hashes(sources | inputs)
            clock = {
                "wall": wall,
                "cpu": cpu,
                "prior_stage_wall": profile["wall_seconds"],
                "prior_stage_cpu": context["stage_preparation_cpu_seconds"] + profile["cpu_seconds"],
            }
            reserve = profile["forecast"]["evaluation_reporting_reserve_seconds"]
            context["numerical_threads"] = _thread_environment()
            configure_torch()
            torch.cuda.reset_peak_memory_stats()
            check_budget(context, clock, reserve_seconds=reserve)
            patients, schedule, weight = _fixed_recipe(context)
            if weight != profile["positive_class_weight"]:
                raise ValueError("Actual-schedule class weight differs from resource profile")
            np.save(TRAINING / "patient_schedule.npy", patients, allow_pickle=False)
            np.save(TRAINING / "case_schedule.npy", schedule, allow_pickle=False)
            write_new_json(
                TRAINING / "admission.json",
                {
                    "source_sha256": sources,
                    "input_sha256": inputs,
                    "evaluation_reporting_reserve_seconds": reserve,
                    "positive_class_weight": weight,
                    "profile_gate_passed": True,
                    "schedule_array_sha256": hashlib.sha256(schedule.tobytes()).hexdigest(),
                },
            )
            models = matched_initial_models(INITIAL_PATH, SSL_PATH)
            initial = {
                arm: {**model_description(model), "full_model_sha256": state_digest(model.state_dict())}
                for arm, model in models.items()
            }
            if initial["R"]["head_sha256"] != initial["S"]["head_sha256"]:
                raise ValueError("Initial heads differ across scientific arms")
            reports = {}
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
                        progress,
                        log,
                        TRAINING,
                    )
            write_new_json(TRAINING / "regeneration_checks.json", context["regeneration_checks"])
            verify_inference_hashes(sources | inputs)
            check_budget(context, clock, reserve_seconds=reserve)
            progress["completed"] = True
            result = _receipt(
                TRAINING,
                context,
                clock,
                {
                    "training_completed": True,
                    "arms": reports,
                    "progress": progress,
                    "schedule_array_sha256": hashlib.sha256(schedule.tobytes()).hexdigest(),
                    "patient_schedule_sha256": hashlib.sha256(patients.tobytes()).hexdigest(),
                    "schedule_identical_across_arms": True,
                    "schedule_seed": 20261024,
                    "training_subjects": 500,
                    "training_cases": 2000,
                    "positive_class_weight": weight,
                    "initial_head_identical": True,
                    "additional_ssl_steps": 0,
                    "evaluation_reporting_reserve_seconds": reserve,
                    "prior_cpu_seconds": context["prior_cpu_seconds"],
                    "stage_preparation_cpu_seconds": context["stage_preparation_cpu_seconds"],
                    "profile_wall_seconds": profile["wall_seconds"],
                    "profile_cpu_seconds": profile["cpu_seconds"],
                    "performance_scored": False,
                },
            )
            check_budget(context, clock, reserve_seconds=reserve)
            write_new_json(TRAINING / "summary.json", result)
            return result
        except Exception as error:
            _failure(TRAINING, sources, inputs, progress, error, wall, cpu)
            raise RuntimeError("Paired training stopped; inspect the local failure receipt") from None
