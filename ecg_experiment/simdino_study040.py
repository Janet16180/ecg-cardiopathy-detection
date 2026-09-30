"""Controlled causal Transformer objective study using frozen CPC helpers."""

from __future__ import annotations

import copy
import json
import math
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ecg_experiment import encoder_context_study039 as predecessor
from ecg_experiment import xlstm_study as base
from ecg_experiment.encoder_context_analysis039 import load_cells
from ecg_experiment.encoder_context_diagnostics039 import feature_health
from ecg_experiment.files import sha256_file, sha256_json, write_json_atomic, write_torch_atomic
from ecg_experiment.paths import to_stored
from ecg_experiment.reproducibility import seed_everything
from ecg_experiment.training import checked_step

OBJECTIVES = ("cpc", "simdino", "hybrid")
SEEDS = predecessor.SEEDS
TIER = 25
NAME = "experiment040_cpc_simdino"
PROTOCOL = "docs/experiment-040-cpc-simdino.md"
INTERVAL_SEED = 40045
STAGES = predecessor.STAGES
DAY_CEILING_SECONDS = 28_800.0
CORRECTION_RESERVE_SECONDS = 3600.0
REPORT_RESERVE_SECONDS = 900.0
SOURCE_FILES = (
    PROTOCOL,
    "ecg_experiment/cpc_simdino040.py",
    "ecg_experiment/simdino_study040.py",
    "ecg_experiment/simdino_analysis040.py",
    "ecg_experiment/encoder_context_interactions039.py",
    "scripts/experiments/run_cpc_simdino040.py",
    "scripts/coordination/run_simdino_day040.py",
    "scripts/reports/report_cpc_simdino040.py",
    "tests/test_cpc_simdino040.py",
    "tests/test_simdino_study040.py",
    "tests/test_simdino_analysis040.py",
    "third_party/bench-xecg/bench_xecg/utils/loss_utils.py",
    "third_party/bench-xecg/bench_xecg/trainers/ssl_pretrainer.py",
    "third_party/bench-xecg/configs/pretrain/pretrain_run_single_ecg_per_view.yaml",
    "third_party/bench-xecg/configs/pretrain/pretrain_run_config.yaml",
)


def directory(root: Path, objective: str, seed: int) -> Path:
    """Locate one matched two-context cell.

    Parameters
    ----------
    root : Path
        Repository containing local outputs.
    objective : str
        Prespecified learning objective.
    seed : int
        Initialization seed.

    Returns
    -------
    Path
        Immutable cell destination.
    """
    return root / "outputs" / NAME / objective / f"seed{seed}" / "25k"


def protocol_commit(root: Path) -> str:
    """Require the unchanged committed prospective protocol.

    Parameters
    ----------
    root : Path
        Repository with the prospective protocol.

    Returns
    -------
    str
        Original commit that froze the protocol.
    """
    saved = subprocess.run(["git", "show", f"HEAD:{PROTOCOL}"], cwd=root,
                           check=True, capture_output=True).stdout
    if saved != (root / PROTOCOL).read_bytes():
        raise ValueError("Experiment 040 protocol must be committed unchanged before execution")
    return subprocess.run(["git", "log", "-1", "--format=%H", "--", PROTOCOL], cwd=root,
                          check=True, capture_output=True, text=True).stdout.strip()


def ledger(root: Path) -> dict[str, Any]:
    """Read successful and failed attempts without resetting elapsed work.

    Parameters
    ----------
    root : Path
        Repository containing local ledgers.

    Returns
    -------
    dict[str, Any]
        All charged Experiment 040 attempts.
    """
    path = root / "outputs" / NAME / "day_ledger.json"
    return json.loads(path.read_text()) if path.exists() else {"attempts": [], "total_seconds": 0.0}


def combined_seconds(root: Path) -> float:
    """Charge both experiments to the original executable-day ceiling.

    Parameters
    ----------
    root : Path
        Repository containing both immutable run ledgers.

    Returns
    -------
    float
        Previously charged seconds across Experiments 039 and 040.
    """
    return float(predecessor.day_ledger(root)["total_seconds"] + ledger(root)["total_seconds"])


def charge(root: Path, objective: str, seed: int, stage: str, seconds: float, status: str,
           *, cell: bool = True) -> None:
    """Record elapsed work even when a stage or diagnostic raises.

    Parameters
    ----------
    root : Path
        Repository containing outputs.
    objective : str
        Objective or final-analysis scope.
    seed : int
        Initialization seed, zero for shared final analysis.
    stage : str
        Executed stage name.
    seconds : float
        Measured elapsed wall time.
    status : str
        Complete or failed.
    cell : bool, optional
        Also append the inherited per-cell ledger.
    """
    if not math.isfinite(seconds) or seconds < 0 or status not in ("complete", "failed"):
        raise ValueError("Malformed elapsed attempt")
    if cell:
        base.record_stage(root, TIER, stage, seconds, status)
    receipt = ledger(root)
    receipt["attempts"].append({"objective": objective, "seed": seed, "stage": stage,
                                "elapsed_seconds": seconds, "status": status})
    receipt["total_seconds"] = sum(item["elapsed_seconds"] for item in receipt["attempts"])
    receipt["combined_ceiling_seconds"] = DAY_CEILING_SECONDS
    write_json_atomic(root / "outputs" / NAME / "day_ledger.json", receipt, sort_keys=True)


def latest_stage(path: Path, stage: str) -> dict[str, Any] | None:
    """Inspect actual latest attempt rather than a possibly premature result file.

    Parameters
    ----------
    path : Path
        Cell directory.
    stage : str
        Stage to inspect.

    Returns
    -------
    dict[str, Any] or None
        Latest matching attempt if present.
    """
    receipt = path / "stage_walltime.json"
    attempts = json.loads(receipt.read_text())["attempts"] if receipt.exists() else []
    return next((item for item in reversed(attempts) if item["stage"] == stage), None)


def require_success(path: Path, stage: str) -> None:
    """Require a successful completed latest stage before consuming its outputs.

    Parameters
    ----------
    path : Path
        Cell directory.
    stage : str
        Required stage.
    """
    item = latest_stage(path, stage)
    if item is None or item["status"] != "complete":
        raise ValueError(f"A successful latest {stage} attempt is required: {path}")


def prior_integrity(root: Path, replay019: Any) -> dict[str, Any]:
    """Exactly replay all predecessor development probabilities before new scores.

    Parameters
    ----------
    root : Path
        Repository containing local predecessors.
    replay019 : callable
        Frozen original Experiment 019 integrity replay.

    Returns
    -------
    dict[str, Any]
        Historical integrity and exact Experiment 039 score replay.
    """
    history = predecessor.prior_integrity(root, replay019)
    arrays, hashes = load_cells(root)
    for tier in predecessor.TIERS:
        for seed in SEEDS:
            for encoder in predecessor.ENCODERS:
                path = root / "outputs" / predecessor.NAME / encoder / f"seed{seed}/{tier}k"
                for stage in STAGES:
                    require_success(path, stage)
    for name in ("aggregate.json", "interactions.json", "diagnostic_gate.json"):
        if not (root / "outputs" / predecessor.NAME / name).exists():
            raise ValueError("Finish Experiment 039 analysis before starting Experiment 040")
    return {"status": "passed_historical_integrity", "historical": history,
            "experiment039_input_hashes": hashes,
            "experiment039_scores": {name: [base.metrics(arrays["targets"], row) for row in values]
                                     for name, values in arrays.items() if values.ndim == 2},
            "development_records": len(arrays["targets"]),
            "development_patients": len(np.unique(arrays["patient_ids"]))}


def source_identity(root: Path, tier: int, receipt: dict[str, Any], original: Any,
                    objective: str) -> dict[str, Any]:
    """Bind the adaptation, upstream loss, matched controls and v4 bytes.

    Parameters
    ----------
    root : Path
        Repository root.
    tier : int
        Fixed cohort tier.
    receipt : dict[str, Any]
        Verified waveform cache receipt.
    original : callable
        Frozen shared identity helper.
    objective : str
        Fixed learning objective.

    Returns
    -------
    dict[str, Any]
        Complete executable identity.
    """
    from ecg_experiment.cpc_simdino040 import architecture_spec

    current = original(root, tier, receipt)
    _, hashes = load_cells(root)
    current["files_sha256"].update(hashes)
    for path in (root / "data/processed/clean_25k_v4/metadata.json",
                 root / "data/processed/clean_25k_v4/train_manifest.csv",
                 root / "outputs/data_quality/clean_cohorts_v4/receipt.json"):
        current["files_sha256"][to_stored(path)] = sha256_file(path)
    current.update({"experiment": 40, "objective": objective, "cohort_version": 4,
                    "v4_cache_reuse": predecessor.verify_v4(root, tier),
                    "architectures": {arm: architecture_spec(objective, arm) for arm in base.ARMS},
                    "upstream_commit": "13e57523e418d8cddced73f81b27ba541413f4f3",
                    "adaptation": "causal_backbone_two_raw_masked_students_one_clean_ema_teacher",
                    "hybrid_multiplier": 1.0, "coding_rate_weight": 0.1,
                    "coding_rate_eps": 0.05, "feature_readout": "final_student_mean_max_512",
                    "schedule": "three_objectives_two_contexts_three_seeds_25k_unconditional"})
    return current


def tensor_groups(state: dict[str, torch.Tensor]) -> dict[str, str]:
    """Hash student frontend, context, prediction heads and frozen teacher separately.

    Parameters
    ----------
    state : dict[str, torch.Tensor]
        Complete checkpoint state dictionary.

    Returns
    -------
    dict[str, str]
        Named scientific tensor group digests.
    """
    return {"shared_convs": base.tensor_hash(state, ("encoder.convs.",)),
            "shared_heads": base.tensor_hash(state, ("heads.",)),
            "context": base.tensor_hash(state, ("encoder.context.",)),
            "teacher": base.tensor_hash(state, ("teacher.",))}


def validate_movement(initial: dict[str, str], final: dict[str, str], objective: str) -> None:
    """Audit only active groups and require inactive groups to remain unchanged.

    Parameters
    ----------
    initial, final : dict[str, str]
        Initial and final group hashes.
    objective : str
        Fixed learning objective.
    """
    changed = {"shared_convs", "context"}
    changed.add("teacher" if objective == "simdino" else "shared_heads")
    if objective == "hybrid":
        changed.add("teacher")
    if initial.keys() != final.keys():
        raise ValueError("Scientific tensor groups changed schema")
    if any((initial[key] != final[key]) != (key in changed) for key in initial):
        raise ValueError("Active or inactive tensor movement violates the objective contract")


def step(model: Any, optimizer: torch.optim.Optimizer, signal: torch.Tensor) -> float:
    """Update the student first and EMA teacher only after a finite successful step.

    Parameters
    ----------
    model : torch.nn.Module
        Fixed-objective student/teacher model.
    optimizer : torch.optim.Optimizer
        Matched AdamW optimizer.
    signal : torch.Tensor
        Clean normalized records.

    Returns
    -------
    float
        Finite total loss before the update.
    """
    loss, parts = model(signal)
    if parts["cmsc"] != 0.0 or not all(math.isfinite(float(value)) for value in parts.values()):
        raise RuntimeError("Malformed or nonfinite objective components")
    norm = checked_step(loss, model, optimizer, "Experiment 040 objective", clip=True)
    model.after_step(base.UPDATES)
    model.last_parts = {**parts, "gradient_norm": float(norm)}
    return float(loss.detach())


def validate_teacher_clock(model: Any, objective: str, updates: int) -> None:
    """Require teacher counters and schedule to match successful student updates.

    Parameters
    ----------
    model : torch.nn.Module
        Controlled student/teacher model or restored checkpoint model.
    objective : str
        Prespecified objective.
    updates : int
        Successful student updates at this checkpoint boundary.

    Raises
    ------
    ValueError
        If the teacher clock or its frozen schedule disagrees with the boundary.
    """
    if objective not in OBJECTIVES or not 0 <= updates <= base.UPDATES:
        raise ValueError("Invalid objective or successful update count")
    expected_updates = 0 if objective == "cpc" else updates
    expected_schedule = base.UPDATES if objective != "cpc" and updates else 0
    if int(model.ema_updates) != expected_updates or int(model.ema_total_steps) != expected_schedule:
        raise ValueError("EMA count or schedule differs from successful student updates")


def validate_history(losses: list[float], histories: list[dict[str, float]]) -> None:
    """Reject nonfinite scalar losses and objective-component histories.

    Parameters
    ----------
    losses : list[float]
        Total objective values through the checkpoint boundary.
    histories : list[dict[str, float]]
        Saved scalar objective components and gradient norms at each update.

    Raises
    ------
    ValueError
        If any total loss or component is nonfinite.
    """
    values = [value for history in histories for value in history.values()]
    finite_losses = np.isfinite(np.asarray(losses, dtype=float)).all()
    finite_components = np.isfinite(np.asarray(values, dtype=float)).all()
    if not finite_losses or not finite_components:
        raise ValueError("Nonfinite objective loss or component history")


def _train_arm(root: Path, tier: int, arm: str, data: Any, current: dict[str, Any],
               profile_receipt: dict[str, Any], charged_before: float,
               stage_started: float, device: str) -> dict[str, Any]:
    """Train one final student while preserving objective components and exact recovery."""
    objective = current["objective"]
    seed_everything(base.SEED)
    model = base.create_model(arm, base.SEED, device)
    opt = base.optimizer(model)
    initial = tensor_groups(model.state_dict())
    latest = base.output(root, tier) / arm / "latest.pt"
    latest.parent.mkdir(parents=True, exist_ok=True)
    saved = torch.load(latest, map_location="cpu", weights_only=False) if latest.exists() else None
    histories: list[dict[str, float]] = []
    losses: list[float] = []
    updates = exposures = 0
    prior_seconds = 0.0
    if saved is not None:
        base.restore(model, opt, saved, current)
        updates, exposures = saved["updates"], saved["exposures"]
        losses, histories = saved["losses"], saved["objective_components"]
        prior_seconds = saved["elapsed_seconds"]
        if (initial != saved["initial"] or len(losses) != updates or len(histories) != updates
                or exposures != min(updates * base.BATCH, base.EXPOSURES)):
            raise ValueError("Inconsistent objective recovery boundary")
        validate_teacher_clock(model, objective, updates)
        validate_history(losses, histories)
    began = time.monotonic()
    first = updates
    indices = base.order(tier)
    for index in range(first, base.UPDATES):
        base._elapsed_guard(charged_before, stage_started)
        rows = indices[index * base.BATCH:min((index + 1) * base.BATCH, base.EXPOSURES)]
        losses.append(step(model, opt, base.batch(data, rows, device)))
        histories.append(model.last_parts)
        updates, exposures = index + 1, min((index + 1) * base.BATCH, base.EXPOSURES)
        if updates % 100 == 0 or updates == base.UPDATES:
            base.sync(device)
            validate_teacher_clock(model, objective, updates)
            validate_history(losses, histories)
            checkpoint = base.checkpoint(model, opt, current, updates, exposures, losses, initial)
            checkpoint.update({"objective_components": histories,
                               "elapsed_seconds": prior_seconds + time.monotonic() - began})
            write_torch_atomic(latest, checkpoint)
            base._pace_guard(profile_receipt, charged_before, stage_started,
                             arm, updates, (time.monotonic() - began) / (updates - first))
            print(json.dumps({"objective": objective, "seed": base.SEED, "arm": arm,
                              "updates": updates, "exposures": exposures,
                              "loss": losses[-1], "components": histories[-1]}), flush=True)
    if updates != base.UPDATES or exposures != base.EXPOSURES:
        raise ValueError("Incomplete exposure budget")
    validate_teacher_clock(model, objective, updates)
    validate_history(losses, histories)
    base._elapsed_guard(charged_before, stage_started)
    final = tensor_groups(model.state_dict())
    validate_movement(initial, final, objective)
    return {"status": "trained", "arm": arm, "updates": updates, "exposures": exposures,
            "initial": initial, "final": final, "first_loss": losses[0], "last_loss": losses[-1],
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "elapsed_seconds": prior_seconds + time.monotonic() - began,
            "checkpoint": to_stored(latest), "checkpoint_sha256": sha256_file(latest)}


def audit_groups(root: Path, training: dict[str, Any], current: dict[str, Any]) -> None:
    """Verify finite complete students, teacher counts and active/inactive movements.

    Parameters
    ----------
    root : Path
        Repository containing local checkpoints.
    training : dict[str, Any]
        Completed two-context training receipt.
    current : dict[str, Any]
        Unchanged execution identity.
    """
    for arm in base.ARMS:
        item = training["arms"][arm]
        saved = torch.load(root / item["checkpoint"], map_location="cpu", weights_only=False)
        if (not base.finite_tree(saved) or saved["identity_sha256"] != sha256_json(current)
                or saved["updates"] != base.UPDATES or saved["exposures"] != base.EXPOSURES
                or len(saved["losses"]) != base.UPDATES
                or len(saved["objective_components"]) != base.UPDATES
                or saved["initial"] != item["initial"]):
            raise ValueError("Malformed complete objective checkpoint")
        validate_history(saved["losses"], saved["objective_components"])
        final = tensor_groups(saved["model"])
        if final != item["final"]:
            raise ValueError("Final objective tensor hashes changed")
        validate_movement(item["initial"], final, current["objective"])
        model = base.create_model(arm, current["seed"], "cpu")
        model.load_state_dict(saved["model"], strict=True)
        validate_teacher_clock(model, current["objective"], base.UPDATES)
        if any(parameter.requires_grad for parameter in model.teacher.parameters()):
            raise ValueError("Teacher has active gradients")


def _cpu_reserve(root: Path) -> dict[str, float]:
    """Estimate CPU stages from already measured matched patch predecessor cells."""
    values = {stage: [] for stage in ("prepare", "readout", "audit")}
    for seed in SEEDS:
        path = root / "outputs" / predecessor.NAME / "patch" / f"seed{seed}/25k"
        for stage in values:
            item = latest_stage(path, stage)
            if item is None or item["status"] != "complete":
                raise ValueError("Successful measured predecessor CPU stages are required")
            values[stage].append(item["elapsed_seconds"])
    return {stage: 1.5 * max(seconds) for stage, seconds in values.items()}


def _remaining_training(prof: dict[str, Any], training: dict[str, Any], objective: str, seed: int,
                        active: tuple[str, int, str, int, float] | None) -> float:
    """Project both contexts, honoring completed updates and measured slower pace."""
    remaining = 0.0
    for arm in base.ARMS:
        completed = training["arms"].get(arm, {}).get("updates", 0)
        pace = prof["arms"][arm]["seconds_per_update"]
        if active is not None and active[:3] == (objective, seed, arm):
            completed, pace = active[3], max(pace, active[4])
        updates = max(0, base.UPDATES - completed)
        remaining += 1.5 * updates * pace
        remaining += 1.5 * math.ceil(updates / 100) * prof["arms"][arm]["checkpoint_seconds"]
    return remaining


def remaining_projection(root: Path, active: tuple[str, int, str, int, float] | None = None) -> float:
    """Project the full remaining schedule using real profiles and observed slowdown.

    Parameters
    ----------
    root : Path
        Repository containing real profiles and durable completed stages.
    active : tuple or None
        Objective, seed, context, completed updates and measured seconds per update.

    Returns
    -------
    float
        Conservative seconds for all remaining fits, CPU stages and final reporting.
    """
    reserve = _cpu_reserve(root)
    remaining = REPORT_RESERVE_SECONDS
    for seed in SEEDS:
        for objective in OBJECTIVES:
            path = directory(root, objective, seed)
            if (path / "audit.json").exists():
                require_success(path, "audit")
                predecessor.audited_result(path)
                continue
            prof_path = path / "profile.json"
            prof = json.loads((prof_path if prof_path.exists() else
                               directory(root, objective, SEEDS[0]) / "profile.json").read_text())
            if not prof["gate_passed"]:
                raise RuntimeError("A package failed its real GPU profile")
            receipt_path = path / "training.json"
            training = json.loads(receipt_path.read_text()) if receipt_path.exists() else {"arms": {}}
            remaining += _remaining_training(prof, training, objective, seed, active)
            for stage in ("prepare", "readout", "audit"):
                item = latest_stage(path, stage)
                if item is None or item["status"] != "complete":
                    remaining += reserve[stage]
            if not prof_path.exists():
                remaining += 1.5 * prof["profile_wall_seconds"]
            if not (path / "result.json").exists():
                remaining += prof["projected_feature_seconds"]
    return float(remaining)


def admission(root: Path, *, active_seconds: float = 0.0,
              active: tuple[str, int, str, int, float] | None = None) -> dict[str, Any]:
    """Admit the complete schedule only when the original shared day budget permits it.

    Parameters
    ----------
    root : Path
        Repository containing all real first-seed profiles.
    active_seconds : float, optional
        Uncharged elapsed time in the active stage.
    active : tuple or None
        Observed progress and pace in the active training arm.

    Returns
    -------
    dict[str, Any]
        Conservative original-day budget decision.
    """
    used = combined_seconds(root)
    remaining = remaining_projection(root, active)
    projected = used + active_seconds + remaining + CORRECTION_RESERVE_SECONDS
    return {"status": "admitted" if projected <= DAY_CEILING_SECONDS else "resource_gate_failed",
            "charged_seconds": used, "active_seconds": active_seconds,
            "remaining_projected_seconds": remaining,
            "correction_reserve_seconds": CORRECTION_RESERVE_SECONDS,
            "projected_combined_seconds": projected, "ceiling_seconds": DAY_CEILING_SECONDS,
            "gate_passed": projected <= DAY_CEILING_SECONDS}


def profile_package(root: Path, tier: int, arm: str, data: Any, current: dict[str, Any],
                    device: str, original: Any) -> dict[str, Any]:
    """Profile normal updates and exactly recover the actual final-size batch.

    Parameters
    ----------
    root : Path
        Repository with local profile receipts.
    tier : int
        Fixed cohort tier.
    arm : str
        Context network.
    data : dataset
        Verified normalized training data.
    current : dict[str, Any]
        Unchanged execution identity.
    device : str
        Real CUDA device.
    original : callable
        Frozen 24-update and next-update recovery profile.

    Returns
    -------
    dict[str, Any]
        Actual profile with partial-batch and active-gradient recovery evidence.
    """
    receipt = original(root, tier, arm, data, current, device)
    model = base.create_model(arm, base.SEED, device)
    opt = base.optimizer(model)
    rows = base.order(tier)[-16:]
    signal = base.batch(data, rows, device)
    first_loss = step(model, opt, signal)
    validate_teacher_clock(model, current["objective"], 1)
    groups = {"frontend": list(model.encoder.convs.parameters()),
              "context": list(model.encoder.context.parameters()), "heads": list(model.heads.parameters())}
    gradients = {name: float(sum(float(p.grad.square().sum()) for p in params if p.grad is not None) ** 0.5)
                 for name, params in groups.items()}
    if (gradients["frontend"] <= 0 or gradients["context"] <= 0
            or ((gradients["heads"] > 0) != (current["objective"] != "simdino"))
            or any(p.grad is not None for p in model.teacher.parameters())):
        raise ValueError("The actual partial batch violates active-gradient or teacher contracts")
    saved = copy.deepcopy(base.checkpoint(model, opt, current, 1, 16, [first_loss],
                                         tensor_groups(model.state_dict())))
    loss_a = step(model, opt, signal)
    state_a = copy.deepcopy(base.cpu_state(model))
    optimizer_a = copy.deepcopy(opt.state_dict())
    rng_a = base.capture_rng_state()
    base.restore(model, opt, saved, current)
    validate_teacher_clock(model, current["objective"], 1)
    loss_b = step(model, opt, signal)
    if (loss_a != loss_b or not base.same_state(state_a, base.cpu_state(model))
            or not base.tree_equal(optimizer_a, opt.state_dict())
            or not base.tree_equal(rng_a, base.capture_rng_state())):
        raise RuntimeError("Exact real partial-batch teacher/optimizer/RNG replay failed")
    validate_teacher_clock(model, current["objective"], 2)
    receipt.update({"partial_batch_records": len(rows), "partial_batch_replay_equal": True,
                    "partial_batch_gradient_norms": gradients,
                    "partial_batch_components": model.last_parts,
                    "teacher_has_no_gradients": True,
                    "partial_batch_teacher_updates": int(model.ema_updates),
                    "partial_batch_teacher_schedule": int(model.ema_total_steps),
                    "peak_gpu_memory_bytes": max(receipt["peak_gpu_memory_bytes"],
                                                 torch.cuda.max_memory_allocated())})
    return receipt


@contextmanager
def configured(root: Path, objective: str, seed: int) -> Iterator[None]:
    """Specialize shared frozen helpers and restore every hook after the cell.

    Parameters
    ----------
    root : Path
        Repository root.
    objective : str
        Fixed learning objective.
    seed : int
        Matched initialization seed.

    Yields
    ------
    None
        Shared helpers temporarily use the Experiment 040 configuration.
    """
    from ecg_experiment.cpc_simdino040 import create_model

    if objective not in OBJECTIVES or seed not in SEEDS:
        raise ValueError("Unknown objective or initialization seed")
    names = ("create_model", "OUTPUT_NAME", "SEED", "ORDER_SEED", "INTERVAL_SEED", "SOURCE_FILES",
             "protocol_commit", "identity", "prior_integrity", "used_seconds", "_elapsed_guard",
             "_pace_guard", "step", "group_hashes", "_train_arm", "_audit_checkpoint_groups",
             "_profile_one")
    originals = {name: getattr(base, name) for name in names}
    began = time.monotonic()

    def elapsed_guard(charged: float, started: float) -> None:
        originals["_elapsed_guard"](charged, started)
        if combined_seconds(root) + time.monotonic() - began > DAY_CEILING_SECONDS:
            raise RuntimeError("Combined executable-day ceiling exhausted")

    def pace_guard(prof: dict[str, Any], charged: float, started: float,
                   arm: str, update: int, seconds: float) -> None:
        originals["_pace_guard"](prof, charged, started, arm, update, seconds)
        elapsed_guard(charged, started)
        gate = admission(root, active_seconds=time.monotonic() - began,
                         active=(objective, seed, arm, update, seconds))
        if not gate["gate_passed"]:
            raise RuntimeError("Observed objective pace exceeds the combined-day admission gate")

    replacements = {
        "create_model": partial(create_model, objective),
        "OUTPUT_NAME": f"{NAME}/{objective}/seed{seed}", "SEED": seed,
        "ORDER_SEED": seed + 1000, "INTERVAL_SEED": INTERVAL_SEED,
        "SOURCE_FILES": tuple(dict.fromkeys(originals["SOURCE_FILES"] + predecessor.SOURCE_FILES
                                             + SOURCE_FILES)),
        "protocol_commit": protocol_commit,
        "identity": partial(source_identity, original=originals["identity"], objective=objective),
        "prior_integrity": partial(prior_integrity, replay019=originals["prior_integrity"]),
        "used_seconds": predecessor.used_seconds, "_elapsed_guard": elapsed_guard,
        "_pace_guard": pace_guard, "step": step,
        "group_hashes": lambda model: tensor_groups(model.state_dict()),
        "_train_arm": _train_arm, "_audit_checkpoint_groups": audit_groups,
        "_profile_one": partial(profile_package, original=originals["_profile_one"]),
    }
    try:
        for name, value in replacements.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(base, name, value)


def execute(root: Path, objective: str, seed: int, stage: str,
            device: str = "cuda") -> dict[str, Any]:
    """Execute one accounted stage, checking successful prerequisites and budget gates.

    Parameters
    ----------
    root : Path
        Repository root.
    objective : str
        Fixed learning objective.
    seed : int
        Initialization seed.
    stage : str
        Explicit prepare, profile, train, readout or audit stage.
    device : str, optional
        Real profiled CUDA device for scientific execution.

    Returns
    -------
    dict[str, Any]
        Executed immutable stage receipt.
    """
    if stage not in STAGES:
        raise ValueError("Unknown scientific stage")
    with configured(root, objective, seed):
        began = time.monotonic()
        status = "failed"
        try:
            base.configure_runtime()
            protocol_commit(root)
            predecessor.verify_v4(root, TIER)
            if combined_seconds(root) >= DAY_CEILING_SECONDS:
                raise RuntimeError("Combined executable-day budget exhausted")
            path = directory(root, objective, seed)
            if (path / "audit.json").exists():
                raise ValueError("An audited cell is immutable")
            if stage in ("train", "readout", "audit"):
                require_success(path, "profile" if stage == "train" else
                                "train" if stage == "readout" else "readout")
            if stage == "train" and not admission(root)["gate_passed"]:
                raise RuntimeError("Full fixed schedule failed its combined-day resource gate")
            result = getattr(base, stage)(root, TIER, device) if stage in (
                "profile", "train", "readout") else getattr(base, stage)(root, TIER)
            status = "complete"
            return result
        finally:
            charge(root, objective, seed, stage, time.monotonic() - began, status)


def diagnose(root: Path, objective: str, context: str, seed: int,
             triggers: list[dict[str, Any]]) -> dict[str, Any]:
    """Diagnose once using only recoverable checkpoint and training inputs/features.

    Parameters
    ----------
    root : Path
        Repository root.
    objective : str
        Fixed objective being diagnosed.
    context : str
        Affected context network.
    seed : int
        Originally affected initialization seed.
    triggers : list[dict[str, Any]]
        Recorded numerical, collapse or severe-development trigger evidence.

    Returns
    -------
    dict[str, Any]
        Training-only diagnostic evidence; symptoms alone do not identify a repair.
    """
    path = root / "outputs" / NAME / "diagnostics" / f"{objective}_{context}" / "diagnostic.json"
    if path.exists():
        return json.loads(path.read_text())
    began = time.monotonic()
    status = "failed"
    try:
        with configured(root, objective, seed):
            cell = directory(root, objective, seed)
            checkpoint_path = cell / context / "latest.pt"
            evidence: dict[str, Any] = {"checkpoint_available": checkpoint_path.exists()}
            if (cell / "features.npz").exists():
                with np.load(cell / "features.npz", allow_pickle=False) as features:
                    evidence["training_feature_health"] = feature_health(features[f"train_{context}"])
            if checkpoint_path.exists():
                saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                evidence.update({"checkpoint_sha256": sha256_file(checkpoint_path),
                                 "saved_state_finite": base.finite_tree(saved),
                                 "saved_updates": saved["updates"], "saved_exposures": saved["exposures"],
                                 "first_losses": saved["losses"][:100],
                                 "last_losses": saved["losses"][-100:]})
                if evidence["saved_state_finite"]:
                    model = base.create_model(context, seed, "cpu")
                    model.load_state_dict(saved["model"], strict=True)
                    data = base.dataset(root, TIER)
                    signal = base.batch(data, base.order(TIER)[:8], "cpu")
                    loss, parts = model(signal)
                    evidence["training_loss_parts"] = parts
                    if torch.isfinite(loss):
                        loss.backward()
                        evidence["gradient_groups"] = {
                            group: {"finite": all(torch.isfinite(p.grad).all().item() for p in params
                                                  if p.grad is not None),
                                    "norm": float(sum(float(p.grad.square().sum()) for p in params
                                                      if p.grad is not None) ** 0.5)}
                            for group, params in (("frontend", list(model.encoder.convs.parameters())),
                                                  ("context", list(model.encoder.context.parameters())),
                                                  ("heads", list(model.heads.parameters()))) }
            receipt = {"status": "one_training_only_diagnosis", "objective": objective,
                       "context": context, "seed": seed, "triggers": triggers, "evidence": evidence,
                       "development_examples_used_for_diagnosis": False,
                       "specific_defect_identified": False,
                       "conclusion": "No mechanistic defect identified; stop this package and report outcome"}
            write_json_atomic(path, receipt, sort_keys=True)
            status = "complete"
            return receipt
    finally:
        charge(root, objective, seed, "diagnostic", time.monotonic() - began, status, cell=False)
