"""Nonoverlapping runtime scenarios for the prospective Experiment 040 diagnosis."""

from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from ecg_experiment.files import sha256_file, sha256_json, write_json_atomic
from ecg_experiment.paths import to_stored

NAME = "experiment040_runtime_diagnostic"
PARENT = "experiment040_cpc_simdino"
PREDECESSOR = "experiment039_encoder_context"
OBJECTIVES = ("cpc", "simdino", "hybrid")
CONTEXTS = ("gru", "xlstm")
SEEDS = (39042, 39043, 39044)
SOURCE_FILES = (
    "docs/experiment-040-runtime-diagnostic.md",
    "ecg_experiment/simdino_runtime_analysis040.py",
    "scripts/reports/report_simdino_runtime040.py",
    "tests/test_simdino_runtime_analysis040.py",
)


def _read(path: Path) -> dict[str, Any]:
    """Read one saved local JSON receipt."""
    return json.loads(path.read_text())


def _finite(values: list[float], *, positive: bool = False) -> None:
    """Reject missing, nonfinite or negative measurements."""
    if not values or any(not math.isfinite(x) or x < 0 or (positive and x == 0) for x in values):
        raise ValueError("Runtime measurements must be finite and nonnegative")


def historical_costs(root: Path) -> tuple[dict[str, list[float]], dict[str, str]]:
    """Read matched historical stage costs without accessing ECG waveforms.

    Parameters
    ----------
    root : Path
        Repository containing the three completed patch controls.

    Returns
    -------
    tuple
        Measured stage costs, inferred residual proxies and input hashes.
    """
    values: dict[str, list[float]] = {name: [] for name in (
        "prepare", "readout", "audit", "train_setup", "feature_proxy", "readout_nonfeature")}
    hashes = {}
    for seed in SEEDS:
        path = root / "outputs" / PREDECESSOR / "patch" / f"seed{seed}" / "25k"
        receipts = {}
        for name in ("stage_walltime.json", "profile.json", "training.json"):
            receipts[name] = _read(path / name)
            hashes[to_stored(path / name)] = sha256_file(path / name)
        attempts = receipts["stage_walltime.json"]["attempts"]
        stages = {}
        for stage in ("prepare", "readout", "audit", "train"):
            latest = next(item for item in reversed(attempts) if item["stage"] == stage)
            if latest["status"] != "complete":
                raise ValueError("Historical timing requires successful latest stages")
            stages[stage] = latest["elapsed_seconds"]
        for stage in ("prepare", "readout", "audit"):
            values[stage].append(stages[stage])
        feature = receipts["profile.json"]["projected_feature_seconds"] / 1.5
        setup = stages["train"] - sum(
            item["elapsed_seconds"] for item in receipts["training.json"]["arms"].values())
        values["feature_proxy"].append(feature)
        values["readout_nonfeature"].append(stages["readout"] - feature)
        values["train_setup"].append(setup)
    for measurements in values.values():
        _finite(measurements)
    return values, hashes


def original_projection(profiles: dict[str, dict[str, Any]],
                        historical: dict[str, list[float]]) -> dict[str, float]:
    """Reproduce the originally rejected projection component by component.

    Parameters
    ----------
    profiles : dict
        Three original first-seed two-context profile receipts.
    historical : dict
        Three historical stage durations per cost category.

    Returns
    -------
    dict
        Seconds including the original overlapping terms and explicit total.
    """
    if set(profiles) != set(OBJECTIVES):
        raise ValueError("All three original profiles are required")
    result = {
        "training": 3 * sum(p["projected_training_seconds"] for p in profiles.values()),
        "checkpoint": 3 * sum(p["projected_checkpoint_seconds"] for p in profiles.values()),
        "prepare": 6 * 1.5 * max(historical["prepare"]),
        "readout": 9 * 1.5 * max(historical["readout"]),
        "audit": 9 * 1.5 * max(historical["audit"]),
        "profiles": 2 * 1.5 * sum(p["profile_wall_seconds"] for p in profiles.values()),
        "features": 3 * sum(p["projected_feature_seconds"] for p in profiles.values()),
        "report_allowance": 900.0,
    }
    _finite(list(result.values()))
    return {**result, "total": sum(result.values())}


def _block_rate(package: dict[str, Any], first: int, last: int) -> float:
    """Use synchronized block wall time, never asynchronous host-step durations."""
    blocks = package["synchronized_update_blocks"]
    expected = [(1, 20), (21, 24), (25, 50), (51, 100), (101, 150), (151, 200)]
    if [(x["first_update"], x["last_update"]) for x in blocks] != expected:
        raise ValueError("Synchronized update blocks do not cover the prescribed ranges")
    _finite([x["elapsed_seconds"] for x in blocks], positive=True)
    selected = [x for x in blocks if x["first_update"] >= first and x["last_update"] <= last]
    if sum(x["last_update"] - x["first_update"] + 1 for x in selected) != last - first + 1:
        raise ValueError("Requested timing range is not covered by complete synchronized blocks")
    return sum(x["elapsed_seconds"] for x in selected) / (last - first + 1)


def package_projection(package: dict[str, Any]) -> dict[str, Any]:
    """Extrapolate the prescribed sustained window and second complete feature pass.

    Parameters
    ----------
    package : dict
        Completed measured runtime package, including 200 individual update times.

    Returns
    -------
    dict
        Per-fit compute, final-batch, checkpoint and feature estimates with variation.
    """
    steps = package["step_seconds"]
    if package["updates"] != 200 or package["batch"] != 128 or len(steps) != 200:
        raise ValueError("The diagnostic must contain exactly 200 normal updates")
    _finite(steps, positive=True)
    recovery = package["recovery"]
    expected_clock = (0, 0) if package["objective"] == "cpc" else (201, 1954)
    flags = ("loss_equal", "components_equal", "model_equal", "optimizer_equal", "global_rng_equal")
    if (not all(recovery[name] for name in flags)
            or (recovery["teacher_updates"], recovery["teacher_schedule"]) != expected_clock
            or recovery["restored_student_updates"] != 200):
        raise ValueError("The final-batch recovery or EMA contract failed")
    if recovery["records"] != 16:
        raise ValueError("An actual 16-record final batch is required")
    tail = recovery["first_update_seconds"]
    _finite([tail], positive=True)
    checkpoints = package["checkpoints"]
    if [item["updates"] for item in checkpoints] != [100, 200]:
        raise ValueError("Both prescribed checkpoint boundaries are required")
    checkpoint_times = [item["cpu_capture_seconds"] + item["write_seconds"] for item in checkpoints]
    _finite(checkpoint_times, positive=True)
    complete = [item for item in package["feature_profiles"] if item["records"] == 15359]
    if [item["repeat"] for item in complete] != [0, 1]:
        raise ValueError("Both complete training feature passes are required")
    if any(item["shape"] != [15359, 512] or not item["finite"] for item in complete):
        raise ValueError("Malformed training-only feature extraction")
    _finite([item["elapsed_seconds"] for item in complete], positive=True)
    sustained = _block_rate(package, 101, 200)
    feature = complete[1]["elapsed_seconds"]
    return {
        "normal_updates": 1953 * sustained,
        "final_partial_update": tail,
        "checkpoint_capture_and_write": 20 * mean(checkpoint_times),
        "feature_extraction": feature * (15359 + 1306) / 15359,
        "model_initialization": package["phase_seconds"]["model_init"],
        "seconds_per_update_101_200": sustained,
        "seconds_per_update_first24": _block_rate(package, 1, 24),
        "seconds_per_update_25_100": _block_rate(package, 25, 100),
        "four50_window_means": [_block_rate(package, i + 1, i + 50) for i in range(0, 200, 50)],
        "full_feature_pass_seconds": [item["elapsed_seconds"] for item in complete],
        "checkpoint_reload_seconds_excluded": [item["reload_seconds"] for item in checkpoints],
    }


def runtime_scenarios(packages: list[dict[str, Any]], profiles: dict[str, dict[str, Any]],
                      historical: dict[str, list[float]],
                      overheads: dict[str, float]) -> dict[str, Any]:
    """Project all 18 prescribed fits without averaging incompatible timing categories.

    Parameters
    ----------
    packages : list
        All six completed objective/context probe receipts.
    profiles : dict
        Original objective profiles for the six remaining seed-specific profile stages.
    historical : dict
        Completed matched historical stages and explicitly inferred residuals.
    overheads : dict
        Measured nested all-package validation and current-state time-gate durations.

    Returns
    -------
    dict
        Central and 1.5-times scenarios, separate fixed allowances and package detail.
    """
    keys = [(p["objective"], p["context"]) for p in packages]
    if len(keys) != 6 or set(keys) != {(o, c) for o in OBJECTIVES for c in CONTEXTS}:
        raise ValueError("The complete six-package timing family is required")
    detail = {f"{p['objective']}_{p['context']}": package_projection(p) for p in packages}
    measured_names = ("normal_updates", "final_partial_update", "checkpoint_capture_and_write",
                      "feature_extraction", "model_initialization")
    central = {name: 3 * sum(item[name] for item in detail.values()) for name in measured_names}
    _finite([overheads["all_package_validation"], overheads["time_gate_evaluation"]])
    central.update({
        "all_package_validation": 27 * overheads["all_package_validation"],
        "time_gate_evaluation_proxy": 369 * overheads["time_gate_evaluation"],
        "prepare_proxy": 6 * mean(historical["prepare"]),
        "readout_nonfeature_proxy": 9 * mean(historical["readout_nonfeature"]),
        "audit_proxy": 9 * mean(historical["audit"]),
        "train_stage_setup_proxy": 9 * mean(historical["train_setup"]),
        "remaining_profile_proxy": 2 * sum(p["profile_wall_seconds"] for p in profiles.values()),
    })
    conservative = {key: 1.5 * value for key, value in central.items()}
    for key, historical_key, count in (("prepare_proxy", "prepare", 6),
                                       ("readout_nonfeature_proxy", "readout_nonfeature", 9),
                                       ("audit_proxy", "audit", 9),
                                       ("train_stage_setup_proxy", "train_setup", 9)):
        conservative[key] = 1.5 * count * max(historical[historical_key])
    for values in (central, conservative):
        _finite(list(values.values()))
        values["report_allowance"] = 900.0
        values["total"] = sum(values.values())
    return {"central": central, "conservative": conservative, "packages": detail,
            "scheduled_fits": 18, "normal_updates_per_fit": 1953,
            "partial_updates_per_fit": 1, "checkpoint_boundaries_per_fit": 20,
            "correction_reserve_seconds": 3600.0,
            "uncertainty": "Nonprobabilistic scenarios; three-seed full training remains unmeasured",
            "new_full_training_authorized": False}


def _parent_inputs(root: Path) -> tuple[dict, dict, dict, dict, dict, float]:
    """Verify immutable parents and reconstruct the original admission arithmetic."""
    historical, hashes = historical_costs(root)
    profiles = {}
    for objective in OBJECTIVES:
        path = root / "outputs" / PARENT / objective / "seed39042/25k/profile.json"
        profiles[objective] = _read(path)
        hashes[to_stored(path)] = sha256_file(path)
    admission_path = root / "outputs" / PARENT / "admission.json"
    admission = _read(admission_path)
    hashes[to_stored(admission_path)] = sha256_file(admission_path)
    original = original_projection(profiles, historical)
    if not math.isclose(original["total"], admission["remaining_projected_seconds"],
                        rel_tol=0, abs_tol=1e-8):
        raise ValueError("Original rejected projection does not reproduce")
    parent_spent = 0.0
    for name in (PREDECESSOR, PARENT):
        path = root / "outputs" / name
        closure, ledger = _read(path / "execution_closed.json"), _read(path / "day_ledger.json")
        if (closure["ledger_sha256"] != sha256_file(path / "day_ledger.json")
                or closure["total_seconds"] != ledger["total_seconds"]):
            raise ValueError("Closed parent accounting changed")
        parent_spent += ledger["total_seconds"]
        for filename in ("execution_closed.json", "day_ledger.json"):
            hashes[to_stored(path / filename)] = sha256_file(path / filename)
    for name in SOURCE_FILES:
        hashes[name] = sha256_file(root / name)
    return historical, hashes, profiles, admission, original, parent_spent


def _probe_manifest(root: Path, index: dict[str, Any], hashes: dict[str, str]) -> dict | None:
    """Validate scientific source pins while retaining honest pre-manifest failures."""
    path = root / "outputs" / NAME / "manifest.json"
    if not path.exists():
        if index.get("status") == "complete":
            raise ValueError("A complete runtime diagnostic requires its manifest")
        return None
    if sha256_file(path) != index["runtime_manifest_sha256"]:
        raise ValueError("Runtime manifest identity changed")
    manifest = _read(path)
    expected = manifest["files_sha256"]
    if not set(SOURCE_FILES).issubset(expected):
        raise ValueError("Runtime manifest must pin analysis and report sources")
    for name, value in expected.items():
        if sha256_file(root / name) != value:
            raise ValueError(f"Runtime source changed: {name}")
        hashes[name] = value
    hashes[to_stored(path)] = sha256_file(path)
    return manifest


def _package_input(root: Path, entry: dict[str, Any], hashes: dict[str, str],
                   manifest: dict[str, Any]) -> dict | None:
    """Check one retained package and return only a completed measurement."""
    package_path = root / entry["path"]
    if not package_path.exists() and entry["status"] != "complete":
        return None
    value = _read(package_path)
    if value["objective"] != entry["objective"] or value["context"] != entry["context"]:
        raise ValueError("Runtime package identity differs from its index")
    actual = sha256_file(package_path)
    if actual != entry["sha256"]:
        raise ValueError("Runtime package bytes changed")
    hashes[to_stored(package_path)] = actual
    current = {"objective": entry["objective"], "context": entry["context"],
               "parent_manifest": manifest["parents"][entry["objective"]],
               "runtime_source_commit": manifest["source_commit"],
               "runtime_protocol_commit": manifest["protocol_commit"],
               "runtime_source_sha256": manifest["files_sha256"], "seed": 39042, "updates": 200}
    if value["identity_sha256"] != sha256_json(current) or value["seed"] != 39042:
        raise ValueError("Runtime package manifest identity changed")
    if entry["status"] == value["status"] == "complete":
        if value["development_scored"] or value["full_training_completed"]:
            raise ValueError("Runtime probe violated its measurement-only scope")
        return value
    return None


def _probe_inputs(root: Path, hashes: dict[str, str]) -> tuple[dict, list[dict]]:
    """Bind completed packages while preserving stopped or failed diagnostic evidence."""
    path = root / "outputs" / NAME / "result.json"
    if not path.exists():
        return {}, []
    index = _read(path)
    hashes[to_stored(path)] = sha256_file(path)
    before = index.get("parent_file_sha256_before", {})
    if before != index.get("parent_file_sha256_after", {}):
        raise ValueError("Probe changed an immutable parent file")
    for name, expected in before.items():
        if sha256_file(root / name) != expected:
            raise ValueError(f"Parent file drift: {name}")
    manifest = _probe_manifest(root, index, hashes)
    if manifest is None:
        return index, []
    packages = [_package_input(root, entry, hashes, manifest) for entry in index["packages"]]
    packages = [value for value in packages if value is not None]
    return index, packages


def aggregate(root: Path) -> dict[str, Any]:
    """Bind executed diagnostic receipts and preserve complete or partial evidence.

    Parameters
    ----------
    root : Path
        Repository with immutable parents and new runtime-only outputs.

    Returns
    -------
    dict
        Hash-bound runtime scenarios, or a partial result without a full estimate.
    """
    output = root / "outputs" / NAME
    historical, hashes, profiles, admission, original, parent_spent = _parent_inputs(root)
    index, packages = _probe_inputs(root, hashes)
    ledger_path = output / "day_ledger.json"
    ledger = _read(ledger_path) if ledger_path.exists() else {"attempts": [], "total_seconds": 0.0}
    if ledger["total_seconds"] != sum(x["elapsed_seconds"] for x in ledger["attempts"]):
        raise ValueError("Diagnostic ledger sum changed")
    summary: dict[str, Any] = {
        "status": "partial_runtime_diagnostic", "original_projection": original,
        "original_admission": admission, "historical_costs": historical,
        "parent_spent_seconds": parent_spent,
        "diagnostic_seconds_at_analysis": ledger["total_seconds"],
        "diagnostic_ledger_snapshot": ledger,
        "diagnostic_ledger_sha256_at_analysis": sha256_file(ledger_path) if ledger_path.exists() else None,
        "input_sha256": hashes, "packages_completed": len(packages),
        "calibration_or_test_scored": False, "development_scored": False,
        "full_training_executed": False, "new_full_training_authorized": False,
        "probe_status": index.get("status", "not_started"),
        "probe_error": index.get("error"), "package_attempts": index.get("packages", []),
    }
    if len(packages) != 6 or index.get("status") != "complete":
        return summary
    scenarios = runtime_scenarios(packages, profiles, historical, index["global_phase_seconds"])
    summary.update({"status": "complete_runtime_diagnostic", "scenarios": scenarios,
                    "global_phase_seconds": index.get("global_phase_seconds", {}),
                    "probe_wall_seconds": index["wall_seconds"],
                    "package_wall_seconds": {f"{p['objective']}_{p['context']}": p["wall_seconds"]
                                             for p in packages}})
    destination = output / "projected_runtime.json"
    if destination.exists():
        previous = _read(destination)
        if previous["input_sha256"] != hashes:
            raise ValueError("Sealed runtime analysis inputs changed")
        return previous
    write_json_atomic(destination, summary, sort_keys=True)
    return summary
