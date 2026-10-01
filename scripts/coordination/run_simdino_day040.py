"""Run the controlled Transformer objective suite after all real-profile gates pass."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ecg_experiment import ROOT
from ecg_experiment import encoder_context_study039 as predecessor
from ecg_experiment import simdino_study040 as study
from ecg_experiment.encoder_context_diagnostics039 import feature_health
from ecg_experiment.files import write_json_atomic


def predecessor_ready(root: Path) -> bool:
    """Wait for completed predecessor scientific artifacts before taking GPU work.

    Parameters
    ----------
    root : Path
        Repository with shared local outputs.

    Returns
    -------
    bool
        Whether all predecessor cells and analyses have finished.
    """
    path = root / "outputs" / predecessor.NAME
    if not study.predecessor_closed(root):
        return False
    for name in ("aggregate.json", "interactions.json", "diagnostic_gate.json"):
        if not (path / name).exists():
            return False
    return all((path / encoder / f"seed{seed}/{tier}k/audit.json").exists()
               for tier in predecessor.TIERS for seed in study.SEEDS for encoder in predecessor.ENCODERS)


def _status(root: Path, status: str, completed: list[dict[str, Any]], **details: Any) -> None:
    """Write a durable live state without interpreting unfinished measurements."""
    write_json_atomic(root / "outputs" / study.NAME / "status.json",
                      {"status": status, "completed_cells": completed, **details}, sort_keys=True)


def _profiles(root: Path) -> dict[str, str]:
    """Bind all six real first-seed package profiles before any full fit."""
    return study.first_profile_hashes(root)


def _numerical_failure(error: Exception) -> bool:
    """Identify numerical or active-training-contract failures without tuning scores."""
    return isinstance(error, FloatingPointError) or any(
        word in str(error).lower() for word in ("nonfinite", "gradient", "tensor movement", "ema count")
    )


def _previous_stops(root: Path) -> dict[str, list[str]]:
    """Preserve one-shot diagnostic stops across coordinator process restarts."""
    stopped = {}
    for objective in study.OBJECTIVES:
        contexts = [context for context in study.base.ARMS
                    if (root / "outputs" / study.NAME / "diagnostics"
                        / f"{objective}_{context}" / "diagnostic.json").exists()]
        if contexts:
            stopped[objective] = contexts
    return stopped


def _prepare_profiles(root: Path) -> None:
    """Retain and diagnose numerical profile failures before rejecting full admission."""
    for objective in study.OBJECTIVES:
        path = study.directory(root, objective, study.SEEDS[0])
        for stage in ("prepare", "profile"):
            item = study.latest_stage(path, stage)
            if item is not None and item["status"] == "complete":
                continue
            try:
                study.execute(root, objective, study.SEEDS[0], stage)
            except (RuntimeError, ValueError, FloatingPointError) as exc:
                if _numerical_failure(exc):
                    progress = path / "profile_progress.json"
                    context = json.loads(progress.read_text())["context"] if progress.exists() else "gru"
                    study.diagnose(root, objective, context, study.SEEDS[0],
                                   [{"reason": "numerical_profile_failure", "stage": stage,
                                     "error_type": type(exc).__name__, "error": str(exc)}])
                raise


def diagnostic_gate(root: Path, objective: str, seed: int, result: dict[str, Any]) -> list[str]:
    """Apply the frozen severe-performance/collapse rule once per affected package.

    Parameters
    ----------
    root : Path
        Repository with immutable audited outputs.
    objective : str
        Current learning objective.
    seed : int
        Current initialization seed.
    result : dict[str, Any]
        Audited development result, used only to identify the prospective trigger.

    Returns
    -------
    list[str]
        Affected context packages, stopped pending an evidenced correction.
    """
    path = study.directory(root, objective, seed)
    baseline_path = (root / "outputs" / predecessor.NAME / "patch" / f"seed{seed}/25k"
                     if objective == "cpc" else study.directory(root, "cpc", seed))
    baseline = predecessor.audited_result(baseline_path)
    stopped = []
    for context in study.base.ARMS:
        score = result["scores"]["limited"][context]["auroc"]
        difference = score - baseline["scores"]["limited"][context]["auroc"]
        with np.load(path / "features.npz", allow_pickle=False) as features:
            health = feature_health(features[f"train_{context}"])
        if score > 0.60 and difference > -0.03 and not health["collapsed"]:
            continue
        trigger = {"seed": seed, "objective": objective, "context": context,
                   "auroc": score, "difference": difference,
                   "reason": "training_collapse" if health["collapsed"] else "severe_underperformance"}
        study.diagnose(root, objective, context, seed, [trigger])
        stopped.append(context)
    return stopped


def _failed_context(root: Path, objective: str, seed: int) -> str:
    """Locate the first unfinished context from actual completed training receipts."""
    path = study.directory(root, objective, seed) / "training.json"
    trained = json.loads(path.read_text())["arms"] if path.exists() else {}
    return next((arm for arm in study.base.ARMS if trained.get(arm, {}).get("updates", 0)
                 != study.base.UPDATES), study.base.ARMS[-1])


def run(root: Path, *, wait: bool = False) -> dict[str, Any]:
    """Execute all profiles first, then the fixed schedule without score-based selection.

    Parameters
    ----------
    root : Path
        Repository with committed code and shared local data.
    wait : bool, optional
        Wait for the predecessor while leaving its GPU run undisturbed.

    Returns
    -------
    dict[str, Any]
        Actual complete, stopped, failed or resource-gated suite status.
    """
    completed: list[dict[str, Any]] = []
    while not predecessor_ready(root):
        _status(root, "queued_waiting_for_experiment039", completed)
        if not wait:
            raise RuntimeError("Finish Experiment 039 before executing the objective study")
        time.sleep(10)
    stopped = _previous_stops(root)
    if stopped and any(study.latest_stage(study.directory(root, objective, study.SEEDS[0]),
                                          "profile") is None
                       or study.latest_stage(study.directory(root, objective, study.SEEDS[0]),
                                             "profile")["status"] != "complete"
                       for objective in study.OBJECTIVES):
        _status(root, "incomplete_stopped_packages", completed, stopped_packages=stopped)
        return {"status": "incomplete_stopped_packages", "stopped_packages": stopped}
    _status(root, "profiling_all_six_packages", completed)
    _prepare_profiles(root)
    gate = {**study.admission(root), "first_seed_profile_hashes": _profiles(root),
            "scheduled_original_fits": 18, "development_scored_before_admission": False}
    gate_path = root / "outputs" / study.NAME / "admission.json"
    if not gate_path.exists():
        write_json_atomic(gate_path, gate, sort_keys=True)
    elif json.loads(gate_path.read_text())["first_seed_profile_hashes"] != gate["first_seed_profile_hashes"]:
        raise ValueError("Original all-package admission profile identity changed")
    if not gate["gate_passed"]:
        _status(root, "resource_gate_failed", completed, admission=gate)
        return gate
    return _run_admitted(root, completed)


def _run_cell(root: Path, objective: str, seed: int) -> dict[str, Any]:
    """Run a cell once, validating successful latest stages when recovering."""
    path = study.directory(root, objective, seed)
    if (path / "audit.json").exists():
        for stage in study.STAGES:
            study.require_success(path, stage)
        with study.configured(root, objective, seed):
            study.base.ensure_manifest(root, study.TIER)
        return predecessor.audited_result(path)
    for stage in study.STAGES:
        item = study.latest_stage(path, stage)
        if item is None or item["status"] != "complete":
            study.execute(root, objective, seed, stage)
    return predecessor.audited_result(path)


def _run_admitted(root: Path, completed: list[dict[str, Any]]) -> dict[str, Any]:
    """Preserve original outcomes and stop diagnosed packages without further tuning."""
    stopped = _previous_stops(root)
    failures = []
    for seed in study.SEEDS:
        for objective in study.OBJECTIVES:
            if objective in stopped or (objective != "cpc" and "cpc" in stopped):
                if (study.directory(root, objective, seed) / "audit.json").exists():
                    _run_cell(root, objective, seed)
                    completed.append({"objective": objective, "seed": seed, "tier": study.TIER})
                continue
            cell = {"objective": objective, "seed": seed, "tier": study.TIER}
            _status(root, "running", completed, current_cell=cell,
                    stopped_packages=stopped, failures=failures)
            try:
                result = _run_cell(root, objective, seed)
            except (RuntimeError, ValueError, FloatingPointError) as exc:
                failure = {**cell, "error_type": type(exc).__name__, "error": str(exc)}
                failures.append(failure)
                if _numerical_failure(exc):
                    context = _failed_context(root, objective, seed)
                    study.diagnose(root, objective, context, seed,
                                   [{**failure, "reason": "numerical_failure"}])
                    stopped[objective] = [context]
                    continue
                _status(root, "failed", completed, current_cell=cell,
                        stopped_packages=stopped, failures=failures)
                return {"status": "failed", "failures": failures}
            completed.append(cell)
            affected = diagnostic_gate(root, objective, seed, result)
            if affected:
                stopped[objective] = affected
            print(json.dumps({"completed_cell": cell, "scores": result["scores"],
                              "combined_charged_seconds": study.combined_seconds(root)}), flush=True)
    status = "original_factorial_complete" if len(completed) == 9 else "incomplete_stopped_packages"
    _status(root, status, completed, completed_fits=len(completed) * 2,
            stopped_packages=stopped, failures=failures)
    return {"status": status, "completed_cells": completed,
            "stopped_packages": stopped, "failures": failures}


def main() -> None:
    """Run the authorized fixed suite while preserving a useful failure status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-for-predecessor", action="store_true")
    args = parser.parse_args()
    try:
        result = run(ROOT, wait=args.wait_for_predecessor)
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        path = ROOT / "outputs" / study.NAME / "status.json"
        previous = json.loads(path.read_text()) if path.exists() else {}
        _status(ROOT, "failed", previous.get("completed_cells", []),
                error_type=type(exc).__name__, error=str(exc),
                stopped_packages=previous.get("stopped_packages", {}))
        raise
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
