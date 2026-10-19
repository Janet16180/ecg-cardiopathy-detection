"""Execute the frozen four-arm paired-response/control-suppression study."""

import os
import resource
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ecg_experiment import ROOT
from ecg_experiment import ssl_synthetic_fit_replication as replication
from ecg_experiment import ssl_synthetic_fit_replication_run as previous
from ecg_experiment import ssl_synthetic_generated_quality as quality
from ecg_experiment import ssl_synthetic_generated_quality_run as gpu_runtime
from ecg_experiment import ssl_synthetic_paired_suppression as study
from ecg_experiment import ssl_synthetic_pretrained_generator as public
from ecg_experiment import ssl_synthetic_randomized_localization as learning
from ecg_experiment import ssl_synthetic_randomized_localization_run as fit_runtime
from ecg_experiment.paths import from_stored, to_stored
from ecg_experiment.synthetic_audit_io import compare_values, file_sha256, read_json, relative_file

CONFIG = "configs/ssl_synthetic_paired_suppression_v1.json"
ADMISSION = "configs/ssl_synthetic_paired_suppression_admission_v1.json"
SOURCES = tuple(
    dict.fromkeys(
        (
            *previous.SOURCES,
            CONFIG,
            ADMISSION,
            "docs/ssl-synthetic-localization-paired-suppression-v1.md",
            "docs/ssl-synthetic-localization-paired-suppression-admission-v1.md",
            "ecg_experiment/ssl_synthetic_paired_suppression.py",
            "ecg_experiment/ssl_synthetic_paired_suppression_run.py",
            "ecg_experiment/ssl_synthetic_paired_suppression_figures.py",
            "scripts/experiments/run_ssl_synthetic_paired_suppression.py",
            "tests/test_ssl_synthetic_paired_suppression.py",
            "ecg_experiment/ssl_synthetic_counterfactual_factorial_training.py",
            "ecg_experiment/ssl_synthetic_paired_background_training.py",
            "ecg_experiment/ssl_synthetic_model.py",
            "ecg_experiment/ssl_synthetic_model_v2.py",
            "ecg_experiment/ssl_synthetic_preparation.py",
            "ecg_experiment/ssl_synthetic_training.py",
            "ecg_experiment/ssl_synthetic_real_transfer.py",
        )
    )
)


def snapshot(destination: Path, guard: Any) -> dict[str, str]:
    """Retain and hash the complete declared executable and helper sources before numerical work."""
    catalog = {}
    for name in SOURCES:
        guard()
        path = destination / "source_snapshot" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(from_stored(name).read_bytes())
        catalog[name] = file_sha256(path, guard)
    return catalog


def admit(
    root: Path, config: dict[str, Any], amendment: dict[str, Any], inputs: dict[str, Any], guard: Any
) -> dict[str, Path]:
    """Verify the committed identity-only amendment and every allowed parent file.

    Parameters
    ----------
    root : Path
        Local output-host repository.
    config, amendment : dict
        Frozen scientific design and prospective identity/accounting declarations.
    inputs : dict
        Mutable admitted input catalog, retained on failure.
    guard : callable
        Resource check during every hash read.

    Returns
    -------
    dict
        Explicit roots for the three admitted parents.
    """
    if amendment["frozen_config_sha256"] != file_sha256(from_stored(CONFIG), guard) or amendment[
        "frozen_protocol_sha256"
    ] != file_sha256(from_stored("docs/ssl-synthetic-localization-paired-suppression-v1.md"), guard):
        raise ValueError("Frozen scientific config/protocol identity mismatch")
    parents = {**config["fixed_parents"], "replication": amendment["replication"]}
    roots = {}
    for label, row in parents.items():
        roots[label] = root / row["root"]
        inputs[label] = {}
        for name, expected in row["sha256"].items():
            if file_sha256(relative_file(roots[label], name), guard) != expected:
                raise ValueError("Paired-suppression parent identity mismatch")
            inputs[label][name] = expected
        audit = read_json(roots[label] / "independent_audit_v1/summary.json")
        if not audit["all_passed"] or audit["mismatch_count"] != 0:
            raise ValueError("Parent independent audit did not pass")
    prior = amendment["prior_ledger"]
    if file_sha256(relative_file(root, prior["path"]), guard) != prior["sha256"]:
        raise ValueError("Immediate cumulative ledger mismatch")
    ledger = read_json(relative_file(root, prior["path"]))
    for key in ["current_machine_cumulative_cpu_seconds", "current_machine_cumulative_gpu_wall_seconds"]:
        if ledger[key] != prior[key]:
            raise ValueError("Immediate cumulative accounting differs")
    return roots


def predecessor_closure(parent: Path, destination: Path, guard: Any) -> dict[str, Any]:
    """Exactly reproduce replication primary, ranges, conditional intervals and eighteen gates."""
    fits = []
    for identifier in [1, 2, 3]:
        guard()
        fits.append(
            {
                "fit_id": identifier,
                **{
                    arm: read_json(parent / f"fit_{identifier}" / arm / "fresh_metrics.json")["cases"]
                    for arm in ["supervised", "permuted"]
                },
            }
        )
    actual = replication.aggregate_replication(fits)
    compare_values(
        actual,
        read_json(parent / "summary.json")["aggregate"],
        "immediate replication aggregate",
        tolerance=0,
    )
    receipt = {"all_exact": True, "aggregate": actual}
    public.write_new_json(destination / "predecessor_closure.json", receipt)
    return receipt


def prepare_plans(config: dict[str, Any], parent: Path, destination: Path) -> list[dict[str, Any]]:
    """Regenerate and bind the same three initial states and complete training schedules."""
    plans = []
    for row in config["fits"]:
        directory = destination / f"fit_{row['fit_id']}"
        directory.mkdir()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(row["initialization_seed"])
            model = learning.build_model()
        initial = fit_runtime.cpu_tree(model.state_dict())
        schedule = replication.training_schedule(row["schedule_seed"])
        old = parent / f"fit_{row['fit_id']}"
        if not fit_runtime.trees_equal(
            initial, torch.load(old / "initial_state.pt", map_location="cpu", weights_only=True)
        ) or not np.array_equal(schedule, np.load(old / "training_schedule.npy", allow_pickle=False)):
            raise ValueError("Matched replication initial state/schedule differs")
        fit_runtime.save_state(directory / "initial_state.pt", initial)
        quality.save_array(directory / "training_schedule.npy", schedule)
        quality.save_array(directory / "background_schedule.npy", schedule[:, [0, 4]] // 4)
        public.write_new_json(directory / "fit_design.json", row)
        plans.append(
            {
                **row,
                "initial": initial,
                "schedule": schedule,
                "directory": directory,
                "initial_state_sha256": public.tensor_state_sha256(model),
            }
        )
    return plans


def component_banks(directory: Path, size: int) -> tuple[np.memmap, np.memmap]:
    """Create nonoverwriting NaN-filled total/component logs for retained partial trajectories."""
    directory.mkdir(parents=True, exist_ok=True)
    banks = []
    for name, shape in [("losses", (size,)), ("loss_components", (size, len(study.COMPONENTS)))]:
        path = directory / f"{name}.npy"
        if path.exists():
            raise FileExistsError(path)
        value = np.lib.format.open_memmap(path, mode="w+", shape=shape, dtype=np.float64)
        value[:] = np.nan
        value.flush()
        banks.append(value)
    return tuple(banks)


def update_and_persist(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    observed: np.ndarray,
    support: np.ndarray,
    indices: np.ndarray,
    weight: float,
    arm: str,
    banks: tuple[np.memmap, np.memmap],
    step: int,
    device: str = "cuda:0",
    capture_directory: Path | None = None,
    completed: dict[str, int] | None = None,
    completed_key: str = "updates",
) -> list[float]:
    """Execute one objective update and flush its actual recurring component/total persistence.

    Parameters
    ----------
    model, optimizer : objects
        Current matched model and its Adam state.
    observed, support : np.ndarray
        Unchanged TRAIN signals and exact Boolean targets.
    indices : np.ndarray
        Fixed batch-eight case positions from the saved schedule.
    weight : float
        Frozen positive class weight.
    arm : str
        B, P, K or PK.
    banks : tuple
        Total and component memmaps shared by profiling and scientific fitting.
    step : int
        Zero-based row to persist after the update.
    device : str
        Actual admitted execution device; CPU is used only by artificial tests.
    capture_directory : Path or None
        Retain this already computed first-profile logit tensor before its update.
    completed : dict or None
        Counter incremented immediately after optimizer.step, before persistence can fail.
    completed_key : str
        Completed optimizer counter belonging to this arm and profile/main phase.

    Returns
    -------
    list
        Five finite native loss scalars in the declared component order.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    x = torch.from_numpy(np.array(observed[indices], copy=True)).to(device)
    target = torch.from_numpy(np.array(support[indices], copy=True)).to(device)
    logits = model(x)
    if logits.dtype != torch.float32 or logits.shape != target.shape or not torch.isfinite(logits).all():
        raise ValueError("Objective logits are not finite native float32 maps")
    if capture_directory is not None:
        quality.save_array(capture_directory / "step1_logits.npy", logits.detach().cpu().numpy())
        quality.save_array(capture_directory / "step1_case_indices.npy", indices)
    terms = study.objective(logits, target, weight, arm)
    if not all(torch.isfinite(value) for value in terms.values()):
        raise ValueError("Objective components are nonfinite")
    terms["total"].backward()
    optimizer.step()
    if completed is not None:
        completed[completed_key] += 1
    values = [float(terms[name].detach()) for name in study.COMPONENTS]
    banks[0][step] = values[-1]
    banks[1][step] = values
    banks[0].flush()
    banks[1].flush()
    return values


def new_model(initial: dict[str, Any]) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    """Start an admitted objective from its untouched common state and fresh frozen Adam."""
    model = learning.build_model().to("cuda:0")
    model.load_state_dict(initial, strict=True)
    return model, fit_runtime.optimizer_for(model)


def profile_arm(
    arm: str,
    plan: dict[str, Any],
    observed: np.ndarray,
    support: np.ndarray,
    weight: float,
    counts: dict[str, int],
    guard: Any,
) -> dict[str, Any]:
    """Profile three actual updates and exactly recover full states/components at steps2 and3."""
    directory = plan["directory"] / "profile" / arm
    original = component_banks(directory / "original", 3)
    replay = component_banks(directory / "replay", 2)
    model, optimizer = new_model(plan["initial"])
    trajectory = []
    timings = []
    for step in range(3):
        guard()
        counts["profile_update_attempts"] += 1
        started = {"cpu": time.process_time(), "wall": time.perf_counter()}
        update_and_persist(
            model,
            optimizer,
            observed,
            support,
            plan["schedule"][step],
            weight,
            arm,
            original,
            step,
            capture_directory=directory if step == 0 else None,
            completed=counts,
            completed_key="profile_updates",
        )
        torch.cuda.synchronize(0)
        timings.append(quality.elapsed(started))
        state = (fit_runtime.cpu_tree(model.state_dict()), fit_runtime.cpu_tree(optimizer.state_dict()))
        trajectory.append(state)
        prefix = "step1" if step == 0 else f"original_step{step + 1}"
        fit_runtime.save_state(directory / f"{prefix}_model.pt", state[0])
        fit_runtime.save_state(directory / f"{prefix}_optimizer.pt", state[1])
    model.load_state_dict(trajectory[0][0], strict=True)
    optimizer.load_state_dict(trajectory[0][1])
    exact = []
    replay_timings = []
    for position, step in enumerate([1, 2]):
        guard()
        counts["profile_update_attempts"] += 1
        started = {"cpu": time.process_time(), "wall": time.perf_counter()}
        update_and_persist(
            model,
            optimizer,
            observed,
            support,
            plan["schedule"][step],
            weight,
            arm,
            replay,
            position,
            completed=counts,
            completed_key="profile_updates",
        )
        torch.cuda.synchronize(0)
        replay_timings.append(quality.elapsed(started))
        state = (fit_runtime.cpu_tree(model.state_dict()), fit_runtime.cpu_tree(optimizer.state_dict()))
        fit_runtime.save_state(directory / f"replay_step{step + 1}_model.pt", state[0])
        fit_runtime.save_state(directory / f"replay_step{step + 1}_optimizer.pt", state[1])
        exact.append(
            fit_runtime.trees_equal(trajectory[step], state)
            and np.array_equal(original[1][step], replay[1][position])
        )
    started = {"cpu": time.process_time(), "wall": time.perf_counter()}
    counts["profile_inference_attempts"] += 1
    fit_runtime.infer(model, observed[plan["schedule"][0]], "cuda:0")
    counts["profile_inferences"] += 1
    inference = quality.elapsed(started)
    row = {
        "timings": timings,
        "replay_timings": replay_timings,
        "inference": inference,
        "losses": original[0].tolist(),
        "replay_losses": replay[0].tolist(),
        "components": original[1].tolist(),
        "replay_components": replay[1].tolist(),
        "trajectory_state_exact": exact,
        "exact_replay": all(exact),
        "recurring_persistence_included": True,
    }
    public.write_new_json(directory / "summary.json", row)
    if not all(exact):
        raise ValueError("Objective profile full model/Adam/component recovery differs")
    del model, optimizer
    return row


def baseline_closure(plan: dict[str, Any], parent: Path, model: torch.nn.Module) -> dict[str, Any]:
    """Require every BCE loss, final tensor and complete Adam value to match the corresponding fit."""
    current = plan["directory"] / "B"
    old = parent / f"fit_{plan['fit_id']}" / "supervised"
    checks = {
        "all_400_losses_exact": np.array_equal(
            np.load(current / "losses.npy", allow_pickle=False),
            np.load(old / "losses.npy", allow_pickle=False),
        ),
        "final_model_exact": fit_runtime.trees_equal(
            fit_runtime.cpu_tree(model.state_dict()),
            torch.load(old / "final_state.pt", map_location="cpu", weights_only=True),
        ),
        "complete_adam_exact": fit_runtime.trees_equal(
            torch.load(current / "optimizer.pt", map_location="cpu", weights_only=True),
            torch.load(old / "optimizer.pt", map_location="cpu", weights_only=True),
        ),
    }
    row = {"all_exact": all(checks.values()), **checks}
    public.write_new_json(plan["directory"] / "baseline_closure.json", row)
    if not row["all_exact"]:
        raise ValueError("Corresponding BCE baseline400-update replay differs")
    return row


def train_all(
    plans: list[dict[str, Any]],
    observed: np.ndarray,
    support: np.ndarray,
    weight: float,
    parent: Path,
    counts: dict[str, Any],
    guard: Any,
    verify: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Interleave all objectives and close every BCE replay before inference.

    Parameters
    ----------
    plans : list
        Three fixed common initial states and complete schedules.
    observed, support : np.ndarray
        Unchanged TRAIN signals and Boolean targets.
    weight : float
        Frozen capped positive weight.
    parent : Path
        Audited replication containing corresponding BCE reference states.
    counts : dict
        Mutable attempted/completed operation counters.
    guard, verify : callables
        Resource and immutable-input guards.

    Returns
    -------
    tuple
        All twelve model states and the three exact baseline closures.
    """
    models = {}
    closures = {}
    for plan in plans:
        verify()
        identifier = str(plan["fit_id"])
        pairs = {arm: new_model(plan["initial"]) for arm in study.ARMS}
        logs = {arm: component_banks(plan["directory"] / arm, 400) for arm in study.ARMS}
        for arm, (model, _) in pairs.items():
            fit_runtime.save_state(
                plan["directory"] / arm / "initial_state.pt", fit_runtime.cpu_tree(model.state_dict())
            )
        for step, indices in enumerate(plan["schedule"]):
            for arm, (model, optimizer) in pairs.items():
                guard()
                counts["fits"][identifier][f"{arm}_update_attempts"] += 1
                update_and_persist(
                    model,
                    optimizer,
                    observed,
                    support,
                    indices,
                    weight,
                    arm,
                    logs[arm],
                    step,
                    completed=counts["fits"][identifier],
                    completed_key=f"{arm}_updates",
                )
        for arm, (model, optimizer) in pairs.items():
            fit_runtime.save_state(
                plan["directory"] / arm / "final_state.pt", fit_runtime.cpu_tree(model.state_dict())
            )
            fit_runtime.save_state(
                plan["directory"] / arm / "optimizer.pt", fit_runtime.cpu_tree(optimizer.state_dict())
            )
        verify()
        closures[identifier] = baseline_closure(plan, parent, pairs["B"][0])
        models[identifier] = {arm: model for arm, (model, _) in pairs.items()}
    return models, closures


def combined_profile(
    generation: dict[str, Any],
    training: dict[str, Any],
    destination: Path,
    clock: dict[str, float],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Admit remaining operations using full-path timing including persistence.

    Parameters
    ----------
    generation, training : dict
        Actual generation and four-objective profile receipts.
    destination : Path
        Retained profile and future output directory.
    clock : dict
        Separate whole-stage CPU/wall and active GPU-lock clocks.
    config : dict
        Frozen resource and output-reserve declarations.

    Returns
    -------
    dict
        Saved recurrence, operation-count and storage admission forecasts.
    """
    result = {
        "generation": generation,
        "training": training,
        "spent": quality.elapsed(clock),
        "spent_gpu_lock_wall_seconds": time.perf_counter() - clock["gpu_lock_start"],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
    }
    for kind in ["cpu", "wall"]:
        g = max(row[f"{kind}_seconds"] for row in generation["original_timings"][1:])
        r = max(row["timings"][step][f"{kind}_seconds"] for row in training.values() for step in [1, 2])
        e = max(row["inference"][f"{kind}_seconds"] for row in training.values())
        result[f"recurrent_{kind}_seconds"] = {"generation": g, "update": r, "inference": e}
        future = 2 * 64 * g + 2 * 4800 * r + 2 * 384 * e + 90
        result[f"forecast_{kind}_seconds"] = result["spent"][f"{kind}_seconds"] + future
        if kind == "wall":
            result["forecast_gpu_lock_wall_seconds"] = result["spent_gpu_lock_wall_seconds"] + future
    result["profile_evidence_forecast_reserve_bytes"] = 8 * 1024**2
    result["captured_profile_logits_already_in_current_bytes"] = True
    result["forecast_saved_bytes"] = (
        public.directory_bytes(destination)
        + 12 * 64 * 4 * 12 * 5000 * 4
        + 64 * (8 * 5000 * 4 + 4 * 12 * 5000 * 4 + 6 * 12 * 5000)
        + 16 * 1024**2
        + result["profile_evidence_forecast_reserve_bytes"]
        + config["resources"]["output_forecast_reserve_bytes"]
    )
    public.write_new_json(destination / "combined_profile.json", result)
    if (
        result["forecast_cpu_seconds"] >= 1800
        or result["forecast_wall_seconds"] >= 1800
        or result["forecast_gpu_lock_wall_seconds"] >= 1200
        or result["peak_allocated_bytes"] >= 2 * 1024**3
        or result["forecast_saved_bytes"] >= 1024**3
    ):
        raise ValueError("Paired-suppression combined profile exceeds prospective resources")
    return result


def initial_counts() -> dict[str, Any]:
    """Initialize all attempted/completed counters before admission or model execution."""
    return {
        **dict.fromkeys(
            [
                "cpu_generator_attempts",
                "cpu_generator_forwards",
                "gpu_generator_profile_attempts",
                "gpu_generator_profile_forwards",
                "gpu_generator_bank_attempts",
                "gpu_generator_bank_forwards",
            ],
            0,
        ),
        "training_profile": dict.fromkeys(
            [
                "profile_update_attempts",
                "profile_updates",
                "profile_inference_attempts",
                "profile_inferences",
            ],
            0,
        ),
        "fits": {
            str(i): {
                f"{arm}_{key}": 0
                for arm in study.ARMS
                for key in ["update_attempts", "updates", "fresh_inference_attempts", "fresh_inferences"]
            }
            for i in [1, 2, 3]
        },
    }


def operation_totals(counts: dict[str, Any]) -> dict[str, int]:
    """Return the fixed operation allocation with no hidden old-panel inference."""
    return {
        "cpu_generator_forwards": counts["cpu_generator_forwards"],
        "gpu_generator_profile_forwards": counts["gpu_generator_profile_forwards"],
        "gpu_generator_bank_forwards": counts["gpu_generator_bank_forwards"],
        "profile_optimizer_updates": counts["training_profile"]["profile_updates"],
        "profile_inference_batches": counts["training_profile"]["profile_inferences"],
        "baseline_replay_optimizer_updates": sum(value["B_updates"] for value in counts["fits"].values()),
        "new_objective_optimizer_updates": sum(
            value[f"{arm}_updates"] for value in counts["fits"].values() for arm in ["P", "K", "PK"]
        ),
        "fresh_inference_batches": sum(
            value[f"{arm}_fresh_inferences"] for value in counts["fits"].values() for arm in study.ARMS
        ),
        "old_panel_inference_batches": 0,
        "hyperparameter_search_updates": 0,
    }


def score_outputs(
    plans: list[dict[str, Any]], fresh: dict[str, np.ndarray], prior: np.ndarray, guard: Any
) -> tuple[dict[str, Any], dict[int, Any], dict[int, Any]]:
    """Score all twelve maps with frozen endpoints and no fit selection.

    Parameters
    ----------
    plans : list
        All fixed fit identities and saved model-map directories.
    fresh : dict
        Qualified fresh signal and exact-support arrays.
    prior : np.ndarray
        Unchanged TRAIN-only spatial prior.
    guard : callable
        Resource guard between model-map evaluations.

    Returns
    -------
    tuple
        Aggregate decision, read-only maps and complete per-arm metrics.
    """
    fits = []
    maps = {}
    metrics = {}
    for plan in plans:
        identifier = plan["fit_id"]
        maps[identifier] = {}
        metrics[identifier] = {}
        for arm in study.ARMS:
            guard()
            directory = plan["directory"] / arm
            probabilities = np.load(directory / "fresh_probabilities.npy", mmap_mode="r", allow_pickle=False)
            rows = replication.score_panel(
                probabilities,
                fresh["observed"],
                fresh["support"],
                np.arange(2000, 2064),
                np.arange(2026105000, 2026105064),
                prior,
            )
            row = {
                "cases": rows,
                "controls": replication.control_scores(probabilities),
                "hotspots": replication.hotspot_scores(
                    probabilities, fresh["support"], np.arange(2000, 2064), np.arange(2026105000, 2026105064)
                ),
                "top600_control": study.control_top600(probabilities),
            }
            public.write_new_json(directory / "fresh_metrics.json", row)
            maps[identifier][arm] = probabilities
            metrics[identifier][arm] = row
        fits.append({"fit_id": identifier, **metrics[identifier]})
    return study.aggregate(fits), maps, metrics


def finish_ledger(
    destination: Path,
    root: Path,
    amendment: dict[str, Any],
    receipt: str,
    clock: dict[str, float],
    prior_verified: bool,
) -> None:
    """Add this terminal attempt once to the independently audited replication cost chain."""
    digest = file_sha256(destination / receipt)
    elapsed = quality.elapsed(clock)
    prior = amendment["prior_ledger"]
    if prior_verified and file_sha256(relative_file(root, prior["path"])) != prior["sha256"]:
        raise ValueError("Prior ledger changed before terminal accounting")
    own_gpu = clock.get("gpu_cost", 0.0)
    row = {
        "stage": "generated_paired_suppression_v1",
        "receipt": receipt,
        "receipt_sha256": digest,
        **elapsed,
        "gpu_wall_seconds": own_gpu,
        "prior_ledger_sha256": prior["sha256"] if prior_verified else None,
        "prior_current_machine_cpu_seconds": prior["current_machine_cumulative_cpu_seconds"]
        if prior_verified
        else None,
        "prior_current_machine_gpu_wall_seconds": prior["current_machine_cumulative_gpu_wall_seconds"]
        if prior_verified
        else None,
        "current_machine_cumulative_cpu_seconds": prior["current_machine_cumulative_cpu_seconds"]
        + elapsed["cpu_seconds"]
        if prior_verified
        else None,
        "current_machine_cumulative_gpu_wall_seconds": prior["current_machine_cumulative_gpu_wall_seconds"]
        + own_gpu
        if prior_verified
        else None,
        "prior_accounting_verified": prior_verified,
        "cpu_budget_passed": elapsed["cpu_seconds"] < 1800,
        "wall_budget_passed": elapsed["wall_seconds"] < 1800,
        "gpu_budget_passed": own_gpu < 1200,
        "output_bytes_before_terminal_ledger": public.directory_bytes(destination),
        "historical_costs_verified": False,
        "measurement_endpoint": "after receipt write/hash; before terminal ledger serialization",
    }
    public.write_new_json(destination / "prospective_ledger.json", row)
    if (
        not all(row[key] for key in ["cpu_budget_passed", "wall_budget_passed", "gpu_budget_passed"])
        or public.directory_bytes(destination) >= 1024**3
    ):
        raise ValueError("Terminal paired-suppression resource ceiling exceeded")


def execute_gpu(
    public_model: torch.nn.Module,
    public_initial: str,
    plans: list[dict[str, Any]],
    old: dict[str, np.ndarray],
    roots: dict[str, Path],
    config: dict[str, Any],
    destination: Path,
    clock: dict[str, float],
    counts: dict[str, Any],
    rows: list[dict[str, Any]],
    generated: dict[str, str],
    guard: Any,
    verify: Any,
) -> dict[str, Any]:
    """Profile, generate the future panel, fit all twelve models and infer once.

    Parameters
    ----------
    public_model : torch.nn.Module
        Strictly loaded public generator after original CPU closure.
    public_initial : str
        Its original named-tensor digest.
    plans, old, roots, config : objects
        Fixed plans, unchanged TRAIN arrays, admitted roots and scientific design.
    destination : Path
        Fresh nonoverwriting attempt directory.
    clock, counts, rows, generated : objects
        Mutable clocks, operation counters, retained case prefix and immutable input catalog.
    guard, verify : callables
        Resource and source/input identity checks.

    Returns
    -------
    dict
        Qualified panel, profiles, baseline closures, final identities and runtime metadata.
    """
    noise = np.stack(
        [
            torch.empty((1, 8, 5000), dtype=torch.float32)
            .uniform_(-1, 1, generator=torch.Generator(device="cpu").manual_seed(seed))
            .numpy()
            for seed in config["fresh_panel"]["generator_seeds"]
        ]
    )
    quality.save_array(destination / "fresh_noise.npy", noise)
    generated["fresh_noise.npy"] = file_sha256(destination / "fresh_noise.npy", guard)
    verify()
    observed = old["observed"][:80].reshape(320, 12, 5000)
    support = old["support"][:80].reshape(320, 12, 5000)
    weight = learning.positive_weight(old["support"][:80])
    design = read_json(roots["learning"] / "training_design.json")
    if design["positive_weight"] != weight or design["positive_cells"] != int(support.sum()):
        raise ValueError("Unchanged supervised TRAIN design differs")
    public.write_new_json(destination / "training_design.json", design)
    generated["training_design.json"] = file_sha256(destination / "training_design.json", guard)
    with gpu_runtime.measured_gpu_lock(clock):
        clock["gpu_lock_start"] = time.perf_counter()
        free, total = torch.cuda.mem_get_info(0)
        if free < 2 * 1024**3:
            raise ValueError("Insufficient free GPU memory")
        torch.cuda.reset_peak_memory_stats(0)
        verify()
        generation = previous.profile_generation(
            public_model, noise, np.array(config["fresh_panel"]["global_indices"]), destination, counts, guard
        )
        training = {
            arm: profile_arm(arm, plans[0], observed, support, weight, counts["training_profile"], guard)
            for arm in study.ARMS
        }
        profile = combined_profile(generation, training, destination, clock, config)
        verify()
        replication_shams = np.load(
            roots["replication"] / "fresh_observed.npy", mmap_mode="r", allow_pickle=False
        )[:, 0]
        old_shams = np.concatenate([old["observed"][:, 0], replication_shams])
        fresh = previous.generate_fresh_bank(
            public_model, noise, config, destination, old_shams, rows, counts, guard
        )
        public_model.cpu()
        final = public.tensor_state_sha256(public_model)
        public.write_new_json(
            destination / "generator_state_identity.json",
            {
                "initial_state_sha256": public_initial,
                "final_state_sha256": final,
                "unchanged": public_initial == final,
            },
        )
        if final != public_initial or generation["initial_state_sha256"] != public_initial:
            raise ValueError("Public generator state changed")
        for name in [
            "fresh_raw.npy",
            "fresh_observed.npy",
            "fresh_support.npy",
            "fresh_strict_support.npy",
            "fresh_cases.jsonl",
        ]:
            generated[name] = file_sha256(destination / name, guard)
        verify()
        models, closures = train_all(
            plans, observed, support, weight, roots["replication"], counts, guard, verify
        )
        hashes = evaluate_all(plans, models, fresh, counts, guard, verify)
        metadata = {
            "name": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "free_bytes_at_admission": free,
            "total_bytes": total,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "torch_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        }
    return {
        "fresh": fresh,
        "profile": profile,
        "baseline_closures": closures,
        "final_state_sha256": hashes,
        "positive_weight": weight,
        "gpu_metadata": metadata,
    }


def evaluate_all(
    plans: list[dict[str, Any]],
    models: dict[str, Any],
    fresh: dict[str, np.ndarray],
    counts: dict[str, Any],
    guard: Any,
    verify: Any,
) -> dict[str, Any]:
    """Infer the fresh panel only after all twelve fixed model states exist and inputs remain bound.

    Parameters
    ----------
    plans, models : objects
        The complete three matched four-arm fits after baseline replay closure.
    fresh : dict
        Qualified fresh signals and separate evaluation targets.
    counts : dict
        All attempted/completed operation counters.
    guard, verify : callables
        Resource and immutable source/input guards.

    Returns
    -------
    dict
        Named final model identities, with every map persisted in fixed batch order.
    """
    if set(models) != {"1", "2", "3"} or any(set(arms) != set(study.ARMS) for arms in models.values()):
        raise ValueError("All twelve fitted models must exist before any fresh inference")
    verify()
    hashes = {}
    for plan in plans:
        identifier = str(plan["fit_id"])
        hashes[identifier] = {}
        for arm, model in models[identifier].items():
            guard()
            previous.evaluate_panel(
                model,
                fresh["observed"],
                plan["directory"] / arm / "fresh_probabilities.npy",
                "fresh",
                arm,
                counts["fits"][identifier],
                guard,
            )
            hashes[identifier][arm] = public.tensor_state_sha256(model)
            model.cpu()
    return hashes


def training_prior(parent: Path, support: np.ndarray) -> np.ndarray:
    """Require the saved spatial prior to be the unchanged TRAIN-only mask mean."""
    prior = np.load(parent / "spatial_prior.npy", allow_pickle=False)
    if not np.array_equal(prior, support[:80, 2:].mean(axis=(0, 1), dtype=np.float64)):
        raise ValueError("TRAIN-only spatial prior differs")
    return prior


def run_study(output_root: Path) -> dict[str, Any]:
    """Execute one frozen study, preserving all attempted operations and failure accounting.

    Parameters
    ----------
    output_root : Path
        Local host repository containing the explicit parent and fresh output paths.

    Returns
    -------
    dict
        Actual saved aggregate, complete provenance and operation/resource evidence.
    """
    from ecg_experiment.ssl_synthetic_paired_suppression_figures import make_figures

    clock = {"cpu": time.process_time(), "wall": time.perf_counter()}
    config = read_json(from_stored(CONFIG))
    amendment = read_json(from_stored(ADMISSION))
    destination = output_root / config["output_root"]
    destination.mkdir(exist_ok=False)
    counts = initial_counts()
    sources, inputs, generated, roots, rows = {}, {}, {}, {}, []
    prior_verified = False
    handlers = {}

    def guard() -> None:
        elapsed = quality.elapsed(clock)
        if elapsed["cpu_seconds"] + 2 >= 1800 or elapsed["wall_seconds"] + 2 >= 1800:
            raise TimeoutError("Paired-suppression CPU/whole-stage wall ceiling")
        if "gpu_lock_start" in clock and "gpu_cost" not in clock:
            if time.perf_counter() - clock["gpu_lock_start"] + 2 >= 1200:
                raise TimeoutError("Paired-suppression GPU-lock ceiling")
            if torch.cuda.max_memory_allocated(0) >= 2 * 1024**3:
                raise ValueError("Paired-suppression GPU allocation ceiling")
        if public.directory_bytes(destination) + 1024**2 >= 1024**3:
            raise ValueError("Paired-suppression output receipt reserve exhausted")

    def verify() -> None:
        previous.verify_bindings(destination, roots, sources, inputs, generated, guard)

    try:
        gpu_runtime.set_runtime()
        gpu_runtime.arm_timers(config["resources"], handlers)
        sources = snapshot(destination, guard)
        roots = admit(output_root, config, amendment, inputs, guard)
        prior_verified = True
        verify()
        closure = predecessor_closure(roots["replication"], destination, guard)
        old = {
            name: np.load(roots["learning"] / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in ["observed", "support"]
        }
        prior = training_prior(roots["learning"], old["support"])
        model, initial = previous.load_public_generator(roots["generator"], destination, counts, guard)
        plans = prepare_plans(config, roots["replication"], destination)
        generated.update(previous.plan_hashes(plans, destination, guard))
        verify()
        executed = execute_gpu(
            model,
            initial,
            plans,
            old,
            roots,
            config,
            destination,
            clock,
            counts,
            rows,
            generated,
            guard,
            verify,
        )
        fresh = executed.pop("fresh")
        verify()
        aggregate, maps, metrics = score_outputs(plans, fresh, prior, guard)
        figures = make_figures(destination, fresh["observed"], fresh["support"], maps, metrics, aggregate)
        verify()
        totals = operation_totals(counts)
        integrity = {
            "immediate_replication_aggregate_exact": closure["all_exact"],
            "public_CPU_replay_exact": True,
            "generator_profile_and_bank_replay_exact": True,
            "public_generator_unchanged": True,
            "fresh64_qualified_disjoint_all192": True,
            "four_objective_profile_recoveries_exact": all(
                row["exact_replay"] for row in executed["profile"]["training"].values()
            ),
            "all_three_BCE_replays_exact": all(
                row["all_exact"] for row in executed["baseline_closures"].values()
            ),
            "all_operations_exact": totals == config["allowed_operations"],
            "resources": clock["gpu_cost"] < 1200
            and executed["gpu_metadata"]["peak_allocated_bytes"] < 2 * 1024**3,
        }
        guard()
        result = {
            "study_id": config["study_id"],
            "created_at_utc": datetime.now(UTC).isoformat(),
            "output_relative": to_stored(ROOT / config["output_root"]),
            "aggregate": aggregate,
            "integrity_gates": integrity,
            "all_passed": aggregate["scientific_passed"] and all(integrity.values()),
            "source_sha256": sources,
            "input_sha256": inputs,
            "generated_input_sha256": generated,
            "counts": counts,
            "operation_totals": totals,
            "fit_designs": [
                {
                    key: plan[key]
                    for key in ["fit_id", "initialization_seed", "schedule_seed", "initial_state_sha256"]
                }
                for plan in plans
            ],
            **executed,
            "figures": figures,
            "completed_fresh_indices": [row["index"] for row in rows],
            "gpu_lock_wall_seconds": clock["gpu_cost"],
            **quality.elapsed(clock),
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "output_bytes_before_receipt": public.directory_bytes(destination),
            "runtime_environment": {
                key: os.environ.get(key)
                for key in [
                    "CUDA_VISIBLE_DEVICES",
                    "CUBLAS_WORKSPACE_CONFIG",
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                ]
            },
            "old_panel_inference_batches": 0,
            "hyperparameter_search_updates": 0,
            "real_waveform_reads": 0,
            "external_api_calls": 0,
            "clinical_claim": False,
            "physical_units_verified": False,
            "loss_component_columns": list(study.COMPONENTS),
            "output_sha256": {
                path.relative_to(destination).as_posix(): file_sha256(path, guard)
                for path in destination.rglob("*")
                if path.is_file()
            },
        }
        public.write_new_json(destination / "summary.json", result)
        guard()
        finish_ledger(destination, output_root, amendment, "summary.json", clock, prior_verified)
        return result
    except Exception as error:
        if rows and not (destination / "partial_fresh_cases.jsonl").exists():
            quality.write_rows(destination / "partial_fresh_cases.jsonl", rows)
        public.write_new_json(
            destination / "failure.json",
            {
                "error_type": type(error).__name__,
                "error": str(error),
                "counts": counts,
                "operation_totals": operation_totals(counts),
                "source_sha256": sources,
                "input_sha256": inputs,
                "generated_input_sha256": generated,
                "completed_or_offending_case_rows": len(rows),
                "gpu_lock_wall_seconds": clock.get("gpu_cost", 0.0),
                **quality.elapsed(clock),
                "historical_costs_verified": False,
            },
        )
        finish_ledger(destination, output_root, amendment, "failure.json", clock, prior_verified)
        raise
    finally:
        fit_runtime.restore_timers(handlers)
