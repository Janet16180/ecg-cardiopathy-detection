"""Prospective sustained-runtime probes without reopening the original objective study."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ecg_experiment import ROOT
from ecg_experiment import simdino_study040 as study
from ecg_experiment.cpc_simdino040 import create_model
from ecg_experiment.files import sha256_file, sha256_json, write_json_atomic, write_torch_atomic
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.paths import to_stored
from ecg_experiment.reproducibility import seed_everything

NAME = "experiment040_runtime_diagnostic"
PROTOCOL = "docs/experiment-040-runtime-diagnostic.md"
SEED = 39042
UPDATES = 200
BATCH = 128
LIMIT_SECONDS = 1200.0
SOURCE_FILES = (PROTOCOL, "ecg_experiment/simdino_runtime_probe040.py",
                "scripts/experiments/profile_simdino_runtime040.py",
                "tests/test_simdino_runtime_probe040.py",
                "scripts/coordination/measure_simdino_runtime040.py",
                "tests/test_simdino_runtime_driver040.py",
                "ecg_experiment/simdino_runtime_analysis040.py",
                "scripts/reports/report_simdino_runtime040.py",
                "tests/test_simdino_runtime_analysis040.py")
base = study.base


def charge(root: Path, stage: str, seconds: float, status: str) -> None:
    """Charge successful or failed diagnostic work in its separate ledger.

    Parameters
    ----------
    root : Path
        Repository containing local outputs.
    stage : str
        Timed diagnostic or report stage.
    seconds : float
        Nonnegative elapsed wall time.
    status : str
        ``complete`` or ``failed``.
    """
    if not math.isfinite(seconds) or seconds < 0 or status not in ("complete", "failed"):
        raise ValueError("Malformed diagnostic charge")
    path = root / "outputs" / NAME / "day_ledger.json"
    saved = json.loads(path.read_text()) if path.exists() else {"attempts": []}
    if any(not math.isfinite(item["elapsed_seconds"]) or item["elapsed_seconds"] < 0
           or item["status"] not in ("complete", "failed") for item in saved["attempts"]):
        raise ValueError("Malformed existing diagnostic charges")
    saved["attempts"].append({"stage": stage, "elapsed_seconds": seconds, "status": status})
    saved["total_seconds"] = sum(item["elapsed_seconds"] for item in saved["attempts"])
    if not math.isfinite(saved["total_seconds"]):
        raise ValueError("Diagnostic charge sum is nonfinite")
    saved["diagnostic_limit_seconds"] = LIMIT_SECONDS
    write_json_atomic(path, saved, sort_keys=True)


def directory(root: Path, objective: str, context: str) -> Path:
    """Return only a diagnostic package path after validating both names.

    Parameters
    ----------
    root : Path
        Repository containing the separate diagnostic output.
    objective : str
        One frozen objective name.
    context : str
        One frozen recurrent context name.

    Returns
    -------
    Path
        New diagnostic directory, never an original study cell.
    """
    if objective not in study.OBJECTIVES or context not in base.ARMS:
        raise ValueError("Unknown runtime package")
    return root / "outputs" / NAME / objective / context


def timing_summary(seconds: list[float], blocks: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Summarize all prescribed ranges without selecting the fastest window.

    Parameters
    ----------
    seconds : list[float]
        Exactly 200 finite positive descriptive host-update times.
    blocks : list[dict[str, Any]] or None
        Synchronized prescribed blocks; their elapsed times are authoritative when supplied.

    Returns
    -------
    dict[str, Any]
        One-based inclusive ranges, means, totals and four 50-update windows.
    """
    values = np.asarray(seconds, dtype=float)
    if values.shape != (UPDATES,) or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Expected 200 finite positive update timings")
    if blocks is not None:
        expected = [(1, 20), (21, 24), (25, 50), (51, 100), (101, 150), (151, 200)]
        actual = [(item["first_update"], item["last_update"]) for item in blocks]
        elapsed = np.asarray([item["elapsed_seconds"] for item in blocks], dtype=float)
        if actual != expected or not np.isfinite(elapsed).all() or np.any(elapsed <= 0):
            raise ValueError("Malformed synchronized timing blocks")
        values = values.copy()
        for item in blocks:
            first, last = item["first_update"] - 1, item["last_update"]
            values[first:last] = item["elapsed_seconds"] / (last - first)
    ranges = {"first_20": (0, 20), "first_24": (0, 24), "updates25_100": (24, 100),
              "updates101_200": (100, 200), "steady21_200": (20, 200), "all_200": (0, 200)}
    ranges.update({f"window{index + 1}": (index * 50, (index + 1) * 50) for index in range(4)})
    return {name: {"first_update": first + 1, "last_update": last, "updates": last - first,
                   "seconds": float(values[first:last].sum()),
                   "seconds_per_update": float(values[first:last].mean()),
                   "warmup_updates_included": max(0, min(last, 20) - first)}
            for name, (first, last) in ranges.items()}


def _check_deadline(started: float) -> None:
    if time.monotonic() - started > LIMIT_SECONDS:
        raise RuntimeError("The 1200-second runtime diagnostic limit was exceeded")


def _parent_files(root: Path) -> dict[str, str]:
    paths = [path for path in (root / "outputs" / study.NAME).rglob("*") if path.is_file()]
    paths += [root / "outputs" / study.predecessor.NAME / name
              for name in ("day_ledger.json", "execution_closed.json")]
    return {to_stored(path): sha256_file(path) for path in paths}


def _verify_parents(root: Path) -> dict[str, Any]:
    checked: dict[str, str] = {}
    parents = {}
    for objective in study.OBJECTIVES:
        path = study.directory(root, objective, SEED)
        study.require_success(path, "profile")
        manifest = json.loads((path / "manifest.json").read_text())
        prof = json.loads((path / "profile.json").read_text())
        if not prof["gate_passed"] or prof["identity_sha256"] != sha256_json(manifest):
            raise ValueError("The original real profile or identity failed")
        if set(prof["arms"]) != set(base.ARMS):
            raise ValueError("The original profile is missing a context")
        for arm in prof["arms"].values():
            if (not arm["next_update_replay_equal"] or not arm["partial_batch_replay_equal"]
                    or arm["partial_batch_records"] != 16):
                raise ValueError("The original recovery profile failed")
        for filename, expected in manifest["files_sha256"].items():
            if filename not in checked:
                checked[filename] = sha256_file(root / filename)
            if checked[filename] != expected:
                raise ValueError(f"Pinned parent source or input changed: {filename}")
        parents[objective] = manifest
    return parents


def _source_identity(root: Path, parents: dict[str, Any]) -> dict[str, Any]:
    hashes = {name: sha256_file(root / name) for name in SOURCE_FILES}
    for name, expected in hashes.items():
        committed = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=root)
        if hashlib.sha256(committed).hexdigest() != expected:
            raise ValueError(f"Uncommitted diagnostic source: {name}")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    protocol = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", PROTOCOL],
                                       cwd=root, text=True).strip()
    return {"source_commit": commit, "protocol_commit": protocol, "files_sha256": hashes,
            "parents": parents, "seed": SEED, "updates": UPDATES, "batch": BATCH,
            "ema_schedule_updates": base.UPDATES, "development_scored": False,
            "feature_records": [512, 15359], "feature_repeats": 2}


def _gpu_snapshot() -> dict[str, Any]:
    snapshot = {"identity": base.gpu_identity()}
    if shutil.which("nvidia-smi"):
        query = subprocess.run(["nvidia-smi",
                                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
        snapshot["gpu_aggregate_csv"] = query.stdout.strip() if query.returncode == 0 else "unavailable"
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,used_gpu_memory",
                               "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
        rows = [line.split(",") for line in apps.stdout.splitlines() if line.strip()]
        snapshot["compute_process_count_all_devices"] = len(rows) if apps.returncode == 0 else None
    return snapshot


def _save_boundary(model: Any, opt: Any, current: dict[str, Any], updates: int,
                   losses: list[float], parts: list[dict[str, float]], initial: dict[str, str],
                   path: Path, device: str) -> tuple[dict[str, Any], dict[str, Any]]:
    study.validate_teacher_clock(model, current["objective"], updates)
    base.sync(device)
    began = time.monotonic()
    saved = base.checkpoint(model, opt, current, updates, updates * BATCH, losses, initial)
    saved["objective_components"] = copy.deepcopy(parts)
    capture = time.monotonic() - began
    began = time.monotonic()
    write_torch_atomic(path, saved)
    base.sync(device)
    write = time.monotonic() - began
    began = time.monotonic()
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    reload_seconds = time.monotonic() - began
    return loaded, {"updates": updates, "cpu_capture_seconds": capture, "write_seconds": write,
                    "reload_seconds": reload_seconds, "checkpoint": to_stored(path),
                    "checkpoint_sha256": sha256_file(path),
                    "optimizer_device_transfer_phase": "serialization_as_in_frozen_helper"}


def recovery_check(model: Any, opt: Any, saved: dict[str, Any], current: dict[str, Any],
                   data: Any, rows: np.ndarray, device: str) -> dict[str, Any]:
    """Exactly replay the next 16-record step, then restore the 200-update student.

    Parameters
    ----------
    model, opt : Any
        Frozen-factory model and optimizer at the saved boundary.
    saved, current : dict[str, Any]
        Complete CPU checkpoint and unchanged diagnostic identity.
    data : Any
        Original normalized training cache.
    rows : np.ndarray
        Sixteen known training indices.
    device : str
        Torch device; CPU permits focused contract tests.

    Returns
    -------
    dict[str, Any]
        Separate original/replay update, restore and comparison costs and exactness flags.
    """
    if len(rows) != 16 or saved["updates"] != UPDATES:
        raise ValueError("Recovery requires the 200-update boundary and 16 records")
    started = time.monotonic()
    base.sync(device)
    began = time.monotonic()
    loss_a = study.step(model, opt, base.batch(data, rows, device))
    base.sync(device)
    first = time.monotonic() - began
    began = time.monotonic()
    parts_a = copy.deepcopy(model.last_parts)
    state_a = base.cpu_state(model)
    optimizer_a = copy.deepcopy(opt.state_dict())
    rng_a = base.capture_rng_state()
    base.sync(device)
    capture = time.monotonic() - began
    study.validate_teacher_clock(model, current["objective"], UPDATES + 1)
    began = time.monotonic()
    base.restore(model, opt, saved, current)
    base.sync(device)
    restore = time.monotonic() - began
    began = time.monotonic()
    loss_b = study.step(model, opt, base.batch(data, rows, device))
    base.sync(device)
    replay = time.monotonic() - began
    began = time.monotonic()
    flags = {"loss_equal": loss_a == loss_b, "components_equal": parts_a == model.last_parts,
             "model_equal": base.same_state(state_a, base.cpu_state(model)),
             "optimizer_equal": base.tree_equal(optimizer_a, opt.state_dict()),
             "global_rng_equal": base.tree_equal(rng_a, base.capture_rng_state())}
    comparison = time.monotonic() - began
    study.validate_teacher_clock(model, current["objective"], UPDATES + 1)
    clock = {"teacher_updates": int(model.ema_updates), "teacher_schedule": int(model.ema_total_steps)}
    began = time.monotonic()
    base.restore(model, opt, saved, current)
    base.sync(device)
    final_restore = time.monotonic() - began
    study.validate_teacher_clock(model, current["objective"], UPDATES)
    if not all(flags.values()):
        raise RuntimeError("Exact sustained-run 16-record recovery failed")
    return {"records": 16, "completed_updates": UPDATES + 1, "first_update_seconds": first,
            "replay_update_seconds": replay, "snapshot_seconds": capture, "restore_seconds": restore,
            "comparison_seconds": comparison, "final_restore_seconds": final_restore,
            "elapsed_seconds": time.monotonic() - started, "first_loss": loss_a,
            "replay_loss": loss_b, **flags, **clock, "restored_student_updates": UPDATES}


def _feature_profiles(model: Any, old_pool: Any, train: list[Any], root: Path,
                      started: float, device: str) -> list[dict[str, Any]]:
    if len(train) != 15359 or any(row.cache_row["split"] != "train" for row in train):
        raise ValueError("Feature timing requires only the fixed PTB training records")
    results = []
    for count in (512, 15359):
        previous = None
        for repeat in range(2):
            _check_deadline(started)
            base.sync(device)
            began = time.monotonic()
            features = base.extract_features(model, train[:count], old_pool, root, device)
            base.sync(device)
            elapsed = time.monotonic() - began
            if features.shape != (count, 512) or not np.isfinite(features).all():
                raise ValueError("Malformed train-only timing features")
            digest = hashlib.sha256(features.tobytes()).hexdigest()
            if previous is not None and previous != digest:
                raise RuntimeError("Repeated unchanged student features differ")
            previous = digest
            results.append({"records": count, "repeat": repeat, "batch": base.BATCH,
                            "elapsed_seconds": elapsed, "shape": list(features.shape), "finite": True,
                            "feature_sha256": digest, "known_training_only": True})
    return results


def _normal_updates(model: Any, opt: Any, current: dict[str, Any], data: Any, indices: np.ndarray,
                    target: Path, started: float, device: str) -> tuple[dict[str, Any], dict[str, Any]]:
    hash_started = time.monotonic()
    initial = study.tensor_groups(model.state_dict())
    group_hash_seconds = time.monotonic() - hash_started
    losses: list[float] = []
    parts: list[dict[str, float]] = []
    seconds: list[float] = []
    checkpoints: list[dict[str, Any]] = []
    blocks: list[dict[str, Any]] = []
    loaded: dict[str, Any] = {}
    measured: dict[str, Any] = {"batch": BATCH, "step_seconds": seconds, "losses": losses,
                               "objective_components": parts, "checkpoints": checkpoints,
                               "synchronized_update_blocks": blocks, "initial": initial}
    loop_started = time.monotonic()
    base.sync(device)
    block_started = time.monotonic()
    block_first = 1
    try:
        for index in range(UPDATES):
            _check_deadline(started)
            began = time.monotonic()
            rows = indices[index * BATCH:(index + 1) * BATCH]
            loss = study.step(model, opt, base.batch(data, rows, device))
            seconds.append(time.monotonic() - began)
            losses.append(loss)
            parts.append(copy.deepcopy(model.last_parts))
            if index + 1 in (20, 24, 50, 100, 150, 200):
                base.sync(device)
                blocks.append({"first_update": block_first, "last_update": index + 1,
                               "elapsed_seconds": time.monotonic() - block_started})
                if index + 1 in (100, 200):
                    loaded, boundary = _save_boundary(model, opt, current, index + 1, losses, parts, initial,
                                                       target / f"checkpoint_{index + 1}.pt", device)
                    checkpoints.append(boundary)
                measured["completed_updates"] = index + 1
                write_json_atomic(target / "progress.json", measured, sort_keys=True)
                print(json.dumps({"objective": current["objective"], "context": current["context"],
                                  "completed_updates": index + 1, "probe_updates": UPDATES}), flush=True)
                block_first = index + 2
                base.sync(device)
                block_started = time.monotonic()
        study.validate_history(losses, parts)
        hash_started = time.monotonic()
        final = study.tensor_groups(model.state_dict())
        group_hash_seconds += time.monotonic() - hash_started
        study.validate_movement(initial, final, current["objective"])
        measured.update({"updates": UPDATES, "exposures": UPDATES * BATCH, "final": final,
                         "timing_summary": timing_summary(seconds, blocks),
                         "group_hash_seconds": group_hash_seconds})
        return measured, loaded
    finally:
        measured["completed_updates"] = len(losses)
        measured["normal_loop_wall_seconds"] = time.monotonic() - loop_started
        write_json_atomic(target / "progress.json", measured, sort_keys=True)


def _probe_package(root: Path, objective: str, context: str, data: Any, old_pool: Any,
                   train: list[Any], identity: dict[str, Any], started: float) -> dict[str, Any]:
    target = directory(root, objective, context)
    target.mkdir(parents=True, exist_ok=False)
    current = {"objective": objective, "context": context, "parent_manifest": identity["parents"][objective],
               "runtime_source_commit": identity["source_commit"],
               "runtime_protocol_commit": identity["protocol_commit"],
               "runtime_source_sha256": identity["files_sha256"], "seed": SEED, "updates": UPDATES}
    began = time.monotonic()
    receipt: dict[str, Any] = {"objective": objective, "context": context, "seed": SEED,
                               "identity_sha256": sha256_json(current), "status": "failed"}
    try:
        seed_everything(SEED)
        init_started = time.monotonic()
        model = create_model(objective, context, SEED, "cuda")
        opt = base.optimizer(model)
        base.sync("cuda")
        init_seconds = time.monotonic() - init_started
        torch.cuda.reset_peak_memory_stats()
        indices = base.order(25)
        measured, saved = _normal_updates(model, opt, current, data, indices, target, started, "cuda")
        receipt.update(measured)
        recovery = recovery_check(model, opt, saved, current, data,
                                  indices[UPDATES * BATCH:UPDATES * BATCH + 16], "cuda")
        receipt["recovery"] = recovery
        _check_deadline(started)
        phases = {"model_init": init_seconds,
                  "normal_updates": measured["timing_summary"]["all_200"]["seconds"],
                  "student_teacher_group_hashes": measured["group_hash_seconds"],
                  "checkpoint_cpu_capture": sum(x["cpu_capture_seconds"] for x in measured["checkpoints"]),
                  "checkpoint_write": sum(x["write_seconds"] for x in measured["checkpoints"]),
                  "checkpoint_reload": sum(x["reload_seconds"] for x in measured["checkpoints"]),
                  "recovery": recovery["elapsed_seconds"]}
        hash_started = time.monotonic()
        before = study.tensor_groups(model.state_dict())
        hash_seconds = time.monotonic() - hash_started
        receipt["feature_profiles"] = _feature_profiles(model, old_pool, train, root, started, "cuda")
        phases["feature_extraction"] = sum(x["elapsed_seconds"] for x in receipt["feature_profiles"])
        hash_started = time.monotonic()
        if before != study.tensor_groups(model.state_dict()):
            raise ValueError("Feature timing changed the trained student or teacher")
        phases["feature_state_hashes"] = hash_seconds + time.monotonic() - hash_started
        _check_deadline(started)
        receipt.update({"phase_seconds": phases, "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
                        "gpu_identity": base.gpu_identity(), "status": "complete",
                        "development_scored": False, "full_training_completed": False})
        del model, opt, saved
        torch.cuda.empty_cache()
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        receipt.update({"error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        progress = target / "progress.json"
        if progress.exists() and "updates" not in receipt:
            receipt.update(json.loads(progress.read_text()))
        receipt["wall_seconds"] = time.monotonic() - began
        receipt["timing_semantics"] = ("normal_updates uses synchronized blocks including guards/components; "
                                       "other phases exclude receipt overhead; wall includes all")
        write_json_atomic(target / "package.json", receipt, sort_keys=True)
    return receipt


def _final_verification(root: Path, before: dict[str, str], identity: dict[str, Any],
                        index: dict[str, Any]) -> bool:
    try:
        after = _parent_files(root) if before else {}
        index["parent_file_sha256_after"] = after
        index["parent_evidence_unchanged"] = before == after
        source_unchanged = True
        if identity:
            expected_hashes: dict[str, str] = {}
            for current in [identity, *identity["parents"].values()]:
                for filename, expected in current["files_sha256"].items():
                    source_unchanged &= expected_hashes.get(filename, expected) == expected
                    expected_hashes[filename] = expected
            source_unchanged &= all(sha256_file(root / filename) == expected
                                    for filename, expected in expected_hashes.items())
        index["pinned_sources_unchanged"] = source_unchanged
        return before != after or not source_unchanged
    except (OSError, ValueError) as exc:
        index["verification_error"] = str(exc)
        return True


def _run_packages(root: Path, data: Any, old_pool: Any, train: list[Any], identity: dict[str, Any],
                  deadline_started: float, index: dict[str, Any]) -> None:
    target = root / "outputs" / NAME
    for objective in study.OBJECTIVES:
        for context in base.ARMS:
            _check_deadline(deadline_started)
            package_path = directory(root, objective, context) / "package.json"
            entry = {"objective": objective, "context": context,
                     "path": to_stored(package_path), "status": "failed"}
            index["packages"].append(entry)
            write_json_atomic(target / "status.json", {
                "status": "running", "current_package": {"objective": objective, "context": context},
                "completed_packages": sum(p["status"] == "complete" for p in index["packages"]),
                "planned_packages": 6,
            }, sort_keys=True)
            try:
                with study.configured(root, objective, SEED):
                    result = _probe_package(root, objective, context, data, old_pool, train,
                                            identity, deadline_started)
                entry["status"] = result["status"]
            finally:
                if package_path.exists():
                    entry["sha256"] = sha256_file(package_path)
                write_json_atomic(target / "result.json", index, sort_keys=True)


def _measure_guards(root: Path, phases: dict[str, float], index: dict[str, Any],
                    deadline_started: float) -> None:
    began = time.monotonic()
    try:
        with study.configured(root, "cpc", SEED):
            index["unchanged_first_profile_hashes"] = study.first_profile_hashes(root)
    finally:
        phases["all_package_validation"] = time.monotonic() - began
    _check_deadline(deadline_started)
    began = time.monotonic()
    try:
        index["unchanged_time_gate"] = study.admission(root)
    finally:
        phases["time_gate_evaluation"] = time.monotonic() - began


def run(root: Path = ROOT) -> dict[str, Any]:
    """Measure the complete fixed runtime diagnostic once under the shared GPU lock.

    Parameters
    ----------
    root : Path
        Repository containing committed new sources and unchanged original receipts.

    Returns
    -------
    dict[str, Any]
        Complete or retained partial runtime evidence without any development score.
    """
    started = time.monotonic()
    status = "failed"
    target = root / "outputs" / NAME
    if (target / "manifest.json").exists() or (target / "result.json").exists():
        raise ValueError("The runtime diagnostic already has immutable evidence")
    index: dict[str, Any] = {"status": "failed", "packages": [], "development_scored": False}
    before = {}
    identity: dict[str, Any] = {}
    ledger_path = target / "day_ledger.json"
    charged_before = json.loads(ledger_path.read_text())["total_seconds"] if ledger_path.exists() else 0.0
    if not math.isfinite(charged_before) or charged_before < 0:
        raise ValueError("Malformed existing diagnostic ledger")
    deadline_started = started - charged_before
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available():
            raise RuntimeError("The runtime probe requires only visible GPU 0")
        verify_started = time.monotonic()
        before = _parent_files(root)
        parents = _verify_parents(root)
        identity = _source_identity(root, parents)
        index["parent_file_sha256_before"] = before
        write_json_atomic(target / "manifest.json", identity, sort_keys=True)
        index["runtime_manifest_sha256"] = sha256_file(target / "manifest.json")
        phases = {"source_verification": time.monotonic() - verify_started}
        _check_deadline(deadline_started)
        with gpu_lock("cuda", blocking=False):
            framework_started = time.monotonic()
            base.configure_runtime()
            index["gpu_before"] = _gpu_snapshot()
            phases["framework_init"] = time.monotonic() - framework_started
            load_started = time.monotonic()
            data = base.dataset(root, 25)
            old_pool = base.pool(root)
            train, _ = base.training_examples(old_pool)
            if any(parent["cache_signals_sha256"] != data.receipt["signals_sha256"]
                   for parent in parents.values()):
                raise ValueError("Parent training cache identity changed")
            phases["dataset_load"] = time.monotonic() - load_started
            index["global_phase_seconds"] = phases
            _run_packages(root, data, old_pool, train, identity, deadline_started, index)
            _check_deadline(deadline_started)
            _measure_guards(root, phases, index, deadline_started)
            index["gpu_after"] = _gpu_snapshot()
            _check_deadline(deadline_started)
        status = "complete"
        index["status"] = "complete"
        return index
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        index.update({"error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        verify_started = time.monotonic()
        drift = _final_verification(root, before, identity, index)
        phases = index.setdefault("global_phase_seconds", {})
        phases["final_parent_verification"] = time.monotonic() - verify_started
        if drift:
            status = "failed"
            index["status"] = "failed_parent_drift"
        index["wall_seconds"] = time.monotonic() - started
        write_json_atomic(target / "result.json", index, sort_keys=True)
        write_json_atomic(target / "status.json", {
            "status": index["status"], "current_package": None,
            "completed_packages": sum(p["status"] == "complete" for p in index["packages"]),
            "planned_packages": 6,
        }, sort_keys=True)
        charge(root, "probe", time.monotonic() - started, status)
        if drift:
            raise ValueError("Original evidence or pinned sources changed during runtime probing")
