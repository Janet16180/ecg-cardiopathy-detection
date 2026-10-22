"""Actual-path matched BCE profiling, complete baseline replay and mixed-background fitting."""

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ecg_experiment import ssl_synthetic_ludb_localizer_run as clocks
from ecg_experiment import ssl_synthetic_paired_suppression_run as paired
from ecg_experiment import ssl_synthetic_pretrained_generator as identities
from ecg_experiment import ssl_synthetic_randomized_localization_run as fitting
from ecg_experiment.files import sha256_file
from ecg_experiment.paths import from_stored


def prepare_plans(config: dict[str, Any], destination: Path) -> list[dict[str, Any]]:
    """Regenerate the three exact original initial states and complete400-step schedules.

    Parameters
    ----------
    config : dict
        Frozen ordered fit IDs and matched initialization/schedule seeds.
    destination : Path
        Exclusive study root receiving the common per-fit declarations.

    Returns
    -------
    list
        Exact verified originals and new nonoverwriting common-state/schedule files.
    """
    specification = [
        {"fit_id": fit, "initialization_seed": initial, "schedule_seed": schedule}
        for fit, initial, schedule in zip(
            config["fits"], config["initialization_seeds"], config["schedule_seeds"], strict=True
        )
    ]
    return paired.prepare_plans({"fits": specification}, from_stored(config["model_root"]), destination)


def snapshot_state(directory: Path, prefix: str, model: Any, optimizer: Any) -> tuple[Any, Any]:
    """Persist and hash complete CPU model/Adam values inside the measured profile step.

    Parameters
    ----------
    directory : Path
        Existing exclusive profile directory.
    prefix : str
        Fixed original or restored step identity.
    model, optimizer : object
        Actual current CUDA training state.

    Returns
    -------
    tuple
        Independent CPU copies of all model and Adam values for exact replay checking.
    """
    state = (fitting.cpu_tree(model.state_dict()), fitting.cpu_tree(optimizer.state_dict()))
    for name, value in zip(["model", "optimizer"], state, strict=True):
        path = directory / f"{prefix}_{name}.pt"
        fitting.save_state(path, value)
        sha256_file(path)
    return state


def profile_one(
    plan: dict[str, Any],
    arm: str,
    bank: dict[str, Any],
    destination: Path,
    counts: dict[str, int],
    guard: Any,
) -> dict[str, Any]:
    """Run three measured disposable updates and exactly restore/replay updates two and three.

    Parameters
    ----------
    plan : dict
        Untouched common initial state and original complete schedule for fit1.
    arm : str
        Generated or mixed training-bank identity; the objective is always B.
    bank : dict
        Actual flattened observed/support inputs and fixed positive weight20.
    destination : Path
        New study root, receiving separate profile-bank evidence.
    counts : dict
        Actual attempted/completed operation counters, retained on failures.
    guard : Any
        Live measured whole-stage and GPU-lock guard.

    Returns
    -------
    dict
        Every step timing, complete-state replay gate and immutable profile output identities.
    """
    directory = destination / "training_profile" / arm
    original = paired.component_banks(directory / "original", 3)
    replay = paired.component_banks(directory / "replay", 2)
    model, optimizer = paired.new_model(plan["initial"])
    states = []
    timings = []
    restored = []
    try:
        for step in range(3):
            guard()
            counts["profile_update_attempts"] += 1
            started = clocks.new_clock()
            paired.update_and_persist(
                model,
                optimizer,
                bank["observed"],
                bank["support"],
                plan["schedule"][step],
                bank["weight"],
                "B",
                original,
                step,
                capture_directory=directory if step == 0 else None,
                completed=counts,
                completed_key="profile_updates",
            )
            state = snapshot_state(directory, f"original_step{step + 1}", model, optimizer)
            torch.cuda.synchronize(0)
            timings.append(clocks.elapsed(started))
            states.append(state)
        model.load_state_dict(
            torch.load(directory / "original_step1_model.pt", map_location="cpu", weights_only=True),
            strict=True,
        )
        optimizer.load_state_dict(
            torch.load(directory / "original_step1_optimizer.pt", map_location="cpu", weights_only=True)
        )
        for position, step in enumerate([1, 2]):
            guard()
            counts["profile_update_attempts"] += 1
            started = clocks.new_clock()
            paired.update_and_persist(
                model,
                optimizer,
                bank["observed"],
                bank["support"],
                plan["schedule"][step],
                bank["weight"],
                "B",
                replay,
                position,
                completed=counts,
                completed_key="profile_updates",
            )
            state = snapshot_state(directory, f"replay_step{step + 1}", model, optimizer)
            torch.cuda.synchronize(0)
            restored.append(
                {
                    "step": step + 1,
                    "timing": clocks.elapsed(started),
                    "all_model_Adam_exact": fitting.trees_equal(states[step], state),
                    "loss_components_exact": bool(np.array_equal(original[1][step], replay[1][position])),
                    "scalar_loss_exact": bool(original[0][step] == replay[0][position]),
                }
            )
        result = {
            "training_arm": arm,
            "fit_id": 1,
            "objective": "B",
            "original_step_timings": timings,
            "restored_steps": restored,
            "original_losses": original[0].tolist(),
            "replayed_losses": replay[0].tolist(),
            "complete_state_replay_exact": all(
                r["all_model_Adam_exact"] and r["loss_components_exact"] and r["scalar_loss_exact"]
                for r in restored
            ),
            "recurring_includes_loss_flush_model_Adam_serialization_hash": True,
        }
        paired.public.write_new_json(directory / "summary.json", result)
        if not result["complete_state_replay_exact"]:
            raise ValueError("Disposable training profile model/Adam/loss replay differs")
        return result
    finally:
        model.cpu()
        del model, optimizer, original, replay


def baseline_closure(directory: Path, parent: Path, model: Any) -> dict[str, Any]:
    """Require the complete400-step trajectory and final model/Adam to equal the original B fit.

    Parameters
    ----------
    directory : Path
        Newly trained generated-only B output directory.
    parent : Path
        Original same-fit B output directory, never modified.
    model : object
        Actual new generated-only final model before any mixed evaluation.

    Returns
    -------
    dict
        Every exact loss/component/model/Adam gate, with failure evidence retained before raising.
    """
    checks = {
        name: bool(
            np.array_equal(
                np.load(directory / name, allow_pickle=False), np.load(parent / name, allow_pickle=False)
            )
        )
        for name in ["losses.npy", "loss_components.npy"]
    }
    checks["final_model_values_exact"] = fitting.trees_equal(
        fitting.cpu_tree(model.state_dict()),
        torch.load(parent / "final_state.pt", map_location="cpu", weights_only=True),
    )
    checks["complete_Adam_values_and_options_exact"] = fitting.trees_equal(
        torch.load(directory / "optimizer.pt", map_location="cpu", weights_only=True),
        torch.load(parent / "optimizer.pt", map_location="cpu", weights_only=True),
    )
    row = {"all_exact": all(checks.values()), "checks": checks, "updates": 400}
    paired.public.write_new_json(directory / "baseline_replay.json", row)
    if not row["all_exact"]:
        raise ValueError("Generated-only full400-loss/component/model/Adam replay differs")
    return row


def train_one(
    plan: dict[str, Any],
    arm: str,
    bank: dict[str, Any],
    config: dict[str, Any],
    counts: dict[str, int],
    guard: Any,
) -> tuple[Any, dict[str, Any]]:
    """Fit one complete original-budget B model without evaluation or checkpoint selection.

    Parameters
    ----------
    plan : dict
        Fixed fit initialization, schedule and new per-fit output directory.
    arm : str
        Generated or mixed training-bank identity.
    bank : dict
        Actual bank observations/targets and fixed TRAIN-only weight20.
    config : dict
        Frozen400-update budget and original model-root declarations.
    counts : dict
        Actual attempted/completed training operation counters.
    guard : Any
        Live measured stage/GPU guard.

    Returns
    -------
    tuple
        Unchanged final CPU model and its saved trajectory/state/closure metadata.
    """
    directory = plan["directory"] / arm / "B"
    logs = paired.component_banks(directory, 400)
    model, optimizer = paired.new_model(plan["initial"])
    fitting.save_state(directory / "initial_state.pt", plan["initial"])
    try:
        for step, indices in enumerate(plan["schedule"]):
            guard()
            counts[arm + "_update_attempts"] += 1
            paired.update_and_persist(
                model,
                optimizer,
                bank["observed"],
                bank["support"],
                indices,
                bank["weight"],
                "B",
                logs,
                step,
                completed=counts,
                completed_key=arm + "_updates",
            )
        fitting.save_state(directory / "final_state.pt", fitting.cpu_tree(model.state_dict()))
        fitting.save_state(directory / "optimizer.pt", fitting.cpu_tree(optimizer.state_dict()))
        proof = (
            baseline_closure(
                directory, from_stored(config["model_root"]) / f"fit_{plan['fit_id']}" / "B", model
            )
            if arm == "generated"
            else None
        )
        row = {
            "training_arm": arm,
            "fit_id": plan["fit_id"],
            "updates": 400,
            "positive_weight": bank["weight"],
            "final_tensor_state_sha256": identities.tensor_state_sha256(model),
            "baseline_replay": proof,
            "output_sha256": {
                name: sha256_file(directory / name)
                for name in [
                    "initial_state.pt",
                    "final_state.pt",
                    "optimizer.pt",
                    "losses.npy",
                    "loss_components.npy",
                ]
            },
        }
        paired.public.write_new_json(directory / "summary.json", row)
        return model.cpu(), row
    finally:
        model.cpu()
        del optimizer, logs
