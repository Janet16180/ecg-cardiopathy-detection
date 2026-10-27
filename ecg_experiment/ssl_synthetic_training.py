"""Execute the source-bound fixed R/S/F pilot after actual resource admission."""

from __future__ import annotations

import copy
import hashlib
import json
import os
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
from ecg_experiment.ssl_synthetic_data import load_bank, positive_class_weight
from ecg_experiment.ssl_synthetic_model import (
    localization_update,
    matched_schedule,
    model_description,
    ssl_update,
    state_digest,
)
from ecg_experiment.ssl_synthetic_model_v2 import DenseWaveformModel
from ecg_experiment.ssl_synthetic_preparation import (
    BASE,
    PREPARATION,
    verify_hashes,
    verify_receipt,
    write_new_json,
)

TRAINING = BASE / "pilot_v1/training"


def check_deadlines(budget: dict[str, float]) -> None:
    """
    Stop before the next update when the reserved training wall/CPU allowance is exhausted.

    Parameters
    ----------
    budget : dict
        Monotonic wall deadline and process CPU deadline, with reporting/evaluation reserved.
    """
    if time.perf_counter() >= budget["wall_deadline"] or time.process_time() >= budget["cpu_deadline"]:
        raise TimeoutError("Pilot training allowance exhausted; evaluation/reporting reserve preserved")


def prior_preparation_cpu() -> float:
    """
    Sum executed preparation charges, conservatively using wall time for single-thread generators.

    Returns
    -------
    float
        Prior local scientific preparation CPU seconds; no historical training is charged.
    """
    names = (
        "input_audit_v1/summary.json",
        "real_cache_v1/summary.json",
        "generator_v1/qualification.json",
        "generator_pilot_v1/qualification.json",
        "generator_pilot_v2/qualification.json",
        "pilot_v1/preparation/summary.json",
        "pilot_v1/prep_audit_v1/summary.json",
        "pilot_v1/profile/summary.json",
    )
    reports = [json.loads((BASE / name).read_text()) for name in names]
    return sum(float(report.get("cpu_seconds", report["wall_seconds"])) for report in reports)


def execute_fits(
    initial: DenseWaveformModel,
    real: np.ndarray,
    bank: dict[str, np.ndarray],
    schedule: np.ndarray,
    weight: float,
    config: dict[str, Any],
    profile: dict[str, Any],
    sources: dict[str, str],
    inputs: dict[str, str],
) -> dict[str, Any]:
    """
    Acquire the GPU before creating outputs and retain every admitted incomplete run.

    Parameters
    ----------
    initial : DenseWaveformModel
        Common CPU initialization.
    real : np.ndarray
        Verified real cache.
    bank : dict
        Verified synthetic training arrays.
    schedule : np.ndarray
        Matched localization batch identities.
    weight : float
        Training-only class weight.
    config, profile : dict
        Fixed recipe and successful measured-resource receipt.
    sources, inputs : dict
        Exact hashes captured before training, verified again at the end.

    Returns
    -------
    dict
        Completed phase accounting; exceptions leave admission/failure/progress evidence locally.
    """
    prior_cpu = prior_preparation_cpu() + float(profile["cpu_seconds"])
    failed_profile = json.loads((BASE / "pilot_v1/profile/summary.json").read_text())
    profile_wall = float(profile["wall_seconds"]) + float(failed_profile["wall_seconds"])
    with gpu_lock("cuda", blocking=False):
        TRAINING.mkdir(parents=True, exist_ok=False)
        wall, cpu = time.perf_counter(), time.process_time()
        budget = {
            "wall_deadline": wall + config["gpu_seconds_ceiling"] - profile_wall - 600,
            "cpu_deadline": cpu + config["cpu_seconds_ceiling"] - prior_cpu - 600,
        }
        admission = {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "source_sha256": sources,
            "input_sha256": inputs,
            "prior_cpu_seconds": prior_cpu,
            "profile_wall_seconds": profile_wall,
            "evaluation_reporting_reserve_seconds": 600,
            "training_started": True,
        }
        write_new_json(TRAINING / "admission.json", admission)
        try:
            check_deadlines(budget)
            torch.cuda.reset_peak_memory_stats()
            initial_checkpoint = save_model(initial, TRAINING / "initial.pt")
            with (TRAINING / "progress.jsonl").open("x") as log:
                encoder, ssl = train_ssl(initial, real, config, log, budget)
                arms = {
                    arm: train_arm(
                        make_arm(initial, encoder, arm), arm, bank, schedule, weight, config, log, budget
                    )
                    for arm in ("R", "S", "F")
                }
            verify_hashes(sources | inputs)
        except (RuntimeError, ValueError, TimeoutError, OSError) as error:
            write_new_json(
                TRAINING / "failure.json",
                {
                    **admission,
                    "training_completed": False,
                    "failure_type": type(error).__name__,
                    "failure_detail": str(error),
                    "wall_seconds": time.perf_counter() - wall,
                    "cpu_seconds": time.process_time() - cpu,
                    "performance_scored": False,
                },
            )
            raise
        return {
            "initial_checkpoint_sha256": initial_checkpoint,
            "ssl": ssl,
            "arms": arms,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "wall_seconds": time.perf_counter() - wall,
            "cpu_seconds": time.process_time() - cpu,
            "prior_cpu_seconds": prior_cpu,
            "profile_wall_seconds": profile_wall,
        }


def configure_torch() -> None:
    """Use one CPU thread and deterministic float32 CUDA operations for this fixed pilot."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def tensor_batch(array: np.ndarray, indices: np.ndarray, device: str = "cuda") -> torch.Tensor:
    """
    Copy only explicitly selected waveform or target arrays to the actual model device.

    Parameters
    ----------
    array : np.ndarray
        Read-only native bank array; caller distinguishes signal from target explicitly.
    indices : np.ndarray
        Fixed schedule positions.
    device : str
        Actual profiling/training device.

    Returns
    -------
    torch.Tensor
        Independent contiguous batch; no paired baseline or metadata channel is appended.
    """
    return torch.from_numpy(np.array(array[indices], copy=True)).to(device)


def fresh_optimizer(model: DenseWaveformModel, config: dict[str, Any]) -> torch.optim.Optimizer:
    """
    Construct a fresh AdamW optimizer without state transferred from another objective or arm.

    Parameters
    ----------
    model : DenseWaveformModel
        Actual arm with requires-grad flags already frozen.
    config : dict
        Frozen learning rate and weight decay.

    Returns
    -------
    torch.optim.Optimizer
        Optimizer containing only the arm's trainable parameters.
    """
    return torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )


def make_arm(initial: DenseWaveformModel, encoder: dict[str, torch.Tensor], arm: str) -> DenseWaveformModel:
    """
    Instantiate a matched fresh or pretrained arm with an identical untouched localization head.

    Parameters
    ----------
    initial : DenseWaveformModel
        Original CPU initialization used before SSL.
    encoder : dict
        Final SSL encoder state only, without reconstruction head or optimizer state.
    arm : str
        R, S or F.

    Returns
    -------
    DenseWaveformModel
        Independent CPU model, with frozen encoder parameters only for F.
    """
    if arm not in {"R", "S", "F"}:
        raise ValueError("Unknown matched arm")
    model = copy.deepcopy(initial)
    if arm != "R":
        model.encoder.load_state_dict(encoder)
    if arm == "F":
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(False)
        model.encoder.eval()
    if state_digest(model.head.state_dict()) != state_digest(initial.head.state_dict()):
        raise ValueError("Localization initialization head differs across arms")
    return model


def save_model(model: DenseWaveformModel, path: Path) -> str:
    """
    Save only CPU model tensors locally and return exact file bytes' hash.

    Parameters
    ----------
    model : DenseWaveformModel
        Fixed phase-final model.
    path : Path
        New local weight artifact; no weights are uploaded.

    Returns
    -------
    str
        SHA-256 of the saved artifact.
    """
    if path.exists():
        raise FileExistsError("Checkpoint cannot be overwritten")
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, path)
    return sha256_file(path)


def train_ssl(
    initial: DenseWaveformModel, real: np.ndarray, config: dict[str, Any], log: Any, budget: dict[str, float]
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """
    Run the fixed real-data masked-only pretraining schedule from fresh initialization.

    Parameters
    ----------
    initial : DenseWaveformModel
        Shared pre-SSL initialization.
    real : np.ndarray
        Permitted unique-patient physical waveform cache.
    config : dict
        Frozen step, batch and optimizer recipe.
    log : file-like
        Local line-delimited progress log, containing no patient information.
    budget : dict
        Shared wall/CPU deadlines reserving evaluation and reporting time.

    Returns
    -------
    tuple of dict
        Final CPU encoder state and actual phase accounting.
    """
    model = copy.deepcopy(initial).to("cuda")
    optimizer = fresh_optimizer(model, config)
    rng = np.random.default_rng(config["seed"] + 11)
    schedule = rng.integers(0, len(real), size=(config["ssl_steps"], config["batch_size"]))
    masks = torch.Generator().manual_seed(config["seed"] + 12)
    wall, cpu = time.perf_counter(), time.process_time()
    losses = []
    for step, indices in enumerate(schedule, start=1):
        check_deadlines(budget)
        loss = ssl_update(model, optimizer, tensor_batch(real, indices), masks)
        losses.append(loss)
        if step % 100 == 0:
            log.write(
                json.dumps(
                    {"phase": "ssl", "step": step, "loss": loss, "wall_seconds": time.perf_counter() - wall}
                )
                + "\n"
            )
            log.flush()
    torch.cuda.synchronize()
    digest = save_model(model, TRAINING / "ssl.pt")
    encoder = {name: value.detach().cpu().clone() for name, value in model.encoder.state_dict().items()}
    report = {
        "steps": len(losses),
        "presentations": int(schedule.size),
        "unique_presented_cache_rows": len(np.unique(schedule)),
        "wall_seconds": time.perf_counter() - wall,
        "cpu_seconds": time.process_time() - cpu,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "losses": losses,
        "encoder_sha256": state_digest(encoder),
        "checkpoint_sha256": digest,
        "sampling_seed": config["seed"] + 11,
        "masking_seed": config["seed"] + 12,
    }
    del model, optimizer
    torch.cuda.empty_cache()
    return encoder, report


def train_arm(
    model: DenseWaveformModel,
    arm: str,
    bank: dict[str, np.ndarray],
    schedule: np.ndarray,
    weight: float,
    config: dict[str, Any],
    log: Any,
    budget: dict[str, float],
) -> dict[str, Any]:
    """
    Run one arm's fixed final-checkpoint localization fit with shared batches.

    Parameters
    ----------
    model : DenseWaveformModel
        Matched CPU arm, with the frozen construction applied for F.
    arm : str
        R, S or F.
    bank : dict
        Fixed training data; only signals and exact support enter the optimizer path.
    schedule : np.ndarray
        Identical ordered batch identities shared by every arm.
    weight : float
        Frozen training-only positive weight.
    config : dict
        Frozen optimizer recipe.
    log : file-like
        Local aggregate progress log.
    budget : dict
        Shared phase deadlines that reserve evaluation/reporting time.

    Returns
    -------
    dict
        Exact initialization/final hashes, losses, exposures and phase accounting.
    """
    before = model_description(model)
    model.to("cuda")
    optimizer = fresh_optimizer(model, config)
    losses = []
    wall, cpu = time.perf_counter(), time.process_time()
    for step, indices in enumerate(schedule, start=1):
        check_deadlines(budget)
        loss = localization_update(
            model,
            optimizer,
            tensor_batch(bank["signals"], indices),
            tensor_batch(bank["support"], indices),
            weight,
            frozen_encoder=arm == "F",
        )
        losses.append(loss)
        if step % 100 == 0:
            log.write(
                json.dumps(
                    {"phase": arm, "step": step, "loss": loss, "wall_seconds": time.perf_counter() - wall}
                )
                + "\n"
            )
            log.flush()
    torch.cuda.synchronize()
    after = model_description(model)
    if arm == "F" and before["encoder_sha256"] != after["encoder_sha256"]:
        raise ValueError("Frozen encoder changed during arm F")
    digest = save_model(model, TRAINING / f"{arm}.pt")
    report = {
        "steps": len(losses),
        "presentations": int(schedule.size),
        "unique_presented_subjects": len(np.unique(schedule)),
        "wall_seconds": time.perf_counter() - wall,
        "cpu_seconds": time.process_time() - cpu,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "losses": losses,
        "initialization": before,
        "final_state": after,
        "checkpoint_sha256": digest,
    }
    del model, optimizer
    torch.cuda.empty_cache()
    return report


def train() -> dict[str, Any]:
    """
    Execute all fixed training phases only after the successful source-bound GPU profile.

    Returns
    -------
    dict
        Local training receipt with no development/evaluation localization score.
    """
    if TRAINING.exists() or TRAINING.is_symlink():
        raise FileExistsError("Training output directory must be new")
    config = json.loads((ROOT / "configs/ssl_synthetic_pilot_v1.json").read_text())
    prep_path = PREPARATION / "summary.json"
    profile_path = BASE / "pilot_v1/profile_v2/summary.json"
    prep, profile = verify_receipt(prep_path), verify_receipt(profile_path)
    verify_hashes(prep["bank_sha256"])
    if not prep["preparation_gate_passed"] or not profile["profile_gate_passed"]:
        raise ValueError("Fixed data or actual GPU profile gate did not admit training")
    if config["batch_size"] != 16 or config["localization_steps"] != 600 or config["ssl_steps"] != 1000:
        raise ValueError("Original fixed exposure recipe changed")
    cache_path = BASE / "real_cache_v1/summary.json"
    cache = verify_receipt(cache_path)
    if (
        sha256_file(cache_path) != prep["input_sha256"][to_stored(cache_path)]
        or not cache["cache_gate_passed"]
    ):
        raise ValueError("Real cache no longer matches preparation")
    if prep["real_cache"] != to_stored(BASE / "real_cache_v1/signals.npy"):
        raise ValueError("Unexpected real waveform cache path")
    configure_torch()
    torch.manual_seed(config["seed"])
    initial = DenseWaveformModel()
    description = model_description(initial)
    real = np.load(ROOT / prep["real_cache"], mmap_mode="r", allow_pickle=False)
    bank = load_bank(ROOT / prep["banks"]["train"]["path"])
    schedule = matched_schedule(bank["kinds"], config["localization_steps"], config["seed"] + 2)
    weight = positive_class_weight(bank, schedule)
    source_names = (
        "ecg_experiment/ssl_synthetic_training.py",
        "ecg_experiment/ssl_synthetic_model.py",
        "ecg_experiment/ssl_synthetic_model_v2.py",
        "ecg_experiment/ssl_synthetic_data.py",
        "ecg_experiment/ssl_synthetic_preparation.py",
        "scripts/experiments/run_ssl_synthetic_pilot.py",
        "configs/ssl_synthetic_pilot_v1.json",
        "docs/ssl-synthetic-localization-pilot.md",
        "docs/ssl-synthetic-localization-pilot-generator-v2.md",
        "docs/ssl-synthetic-localization-pilot-deterministic-output.md",
        "ecg_experiment/gpu.py",
        "ecg_experiment/files.py",
        "ecg_experiment/paths.py",
        "pyproject.toml",
        "uv.lock",
    )
    sources = {name: sha256_file(ROOT / name) for name in source_names}
    sources.update(profile["source_sha256"] | prep["source_sha256"] | cache["source_sha256"])
    inputs = {to_stored(path): sha256_file(path) for path in (prep_path, profile_path, cache_path)}
    inputs.update(profile["input_sha256"] | prep["bank_sha256"])
    inputs.update(
        {to_stored(cache_path.parent / name): digest for name, digest in cache["output_sha256"].items()}
    )
    result = execute_fits(initial, real, bank, schedule, weight, config, profile, sources, inputs)
    receipt = {
        "schema_version": 1,
        "pilot_id": config["pilot_id"],
        "training_completed": True,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "initialization": description,
        "initial_checkpoint_sha256": result["initial_checkpoint_sha256"],
        "real_patients": len(real),
        "synthetic_subjects": len(bank["signals"]),
        "positive_class_weight": weight,
        "schedule_seed": config["seed"] + 2,
        "schedule_sha256": hashlib.sha256(schedule.tobytes()).hexdigest(),
        **result,
        "performance_scored": False,
        "closed_test_opened": False,
        "input_sha256": inputs,
        "source_sha256": sources,
        "output_sha256": {
            path.name: sha256_file(path) for path in sorted(TRAINING.iterdir()) if path.is_file()
        },
    }
    write_new_json(TRAINING / "summary.json", receipt)
    return receipt
