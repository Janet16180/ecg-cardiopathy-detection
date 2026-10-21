"""Profile and evaluate fixed R/S models on qualified real-background introduced changes."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.lib.format import open_memmap

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.paths import to_stored
from ecg_experiment.ssl_synthetic_counterfactual import (
    RESPONSE_METRICS,
    require_original_audit,
    response_metrics,
)
from ecg_experiment.ssl_synthetic_evaluation import (
    _merge_hashes,
    _npy_bytes,
    _read_receipt,
    configure_inference,
    load_final_model,
    verify_inference_hashes,
)
from ecg_experiment.ssl_synthetic_metrics import (
    control_threshold,
    paired_subject_difference,
    positive_map_metrics,
)
from ecg_experiment.ssl_synthetic_model import state_digest
from ecg_experiment.ssl_synthetic_preparation import BASE, stream_bytes, write_new_json

STUDY = BASE / "real_background_v1"
PREPARATION = STUDY / "preparation"
PROFILE = STUDY / "profile"
EVALUATION = STUDY / "evaluation"
ARMS = ("R", "S")
MODES = ("all", "focal", "repeated")
SOURCE_NAMES = (
    "ecg_experiment/ssl_synthetic_real_transfer.py",
    "scripts/reports/evaluate_ssl_synthetic_real_transfer.py",
    "tests/test_ssl_synthetic_real_transfer.py",
    "docs/ssl-synthetic-localization-real-transfer-execution.md",
    "docs/ssl-synthetic-localization-real-background-study.md",
    "docs/ssl-synthetic-localization-real-background-clarifications.md",
    "configs/ssl_synthetic_real_background_v1.json",
)


def conditional_admission(counterfactual: dict[str, Any]) -> dict[str, bool]:
    """Require exactly the four prospectively fixed original-B/all-mode response lower bounds.

    Parameters
    ----------
    counterfactual : dict
        Completed, source-bound original saved-map response diagnostic.

    Returns
    -------
    dict
        Four engineering admission conditions; no new original-pilot advancement rule.
    """
    if not counterfactual["audit_completed"] or counterfactual["original_advancement_passed"]:
        raise ValueError("Original counterfactual completion/failed-pilot history changed")
    conditions = {
        f"{arm}/{metric}": counterfactual["profiles"]["B"]["arms"][arm]["all"][metric]["ci_low"] > 0
        for arm in ARMS
        for metric in ("average_precision_change", "selectivity_contrast")
    }
    if not all(conditions.values()):
        raise ValueError("Prospective real-background response admission failed")
    return conditions


def validate_case_plan(cases: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Check global cache/mask/map rows and complete within-role patient variants.

    Parameters
    ----------
    cases : list of dict
        Qualified immutable descriptors, kept local and never supplied to the network.

    Returns
    -------
    dict
        Ordered global row indices for calibration and evaluation.
    """
    if len(cases) != 768 or [case["case_index"] for case in cases] != list(range(768)):
        raise ValueError("Expected the exact global case grid 0–767")
    roles = {}
    for role, count, profile, first_subject, first_case in (
        ("calibration", 64, "A", 0, 0),
        ("evaluation", 128, "B", 64, 256),
    ):
        selected = [case for case in cases if case["role"] == role]
        if [case["case_index"] for case in selected] != list(range(first_case, first_case + count * 4)):
            raise ValueError("Role/global case coordinate mismatch")
        for subject in range(count):
            group = [case for case in selected if case["role_subject_index"] == subject]
            expected = Counter(
                {
                    ("positive", "focal"): 1,
                    ("positive", "repeated"): 1,
                    ("sham", "none"): 1,
                    ("nuisance", "none"): 1,
                }
            )
            if Counter((case["kind"], case["mode"]) for case in group) != expected:
                raise ValueError("Missing or duplicate patient variant")
            if any(
                case["cache_index"] != first_subject + subject
                or case["subject_index"] != first_subject + subject
                or case["profile"] != profile
                for case in group
            ):
                raise ValueError("Global cache or profile mapping changed")
        roles[role] = [case["case_index"] for case in selected]
    return roles


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def storage_projection() -> dict[str, int]:
    """Count all study arrays exactly once, plus NPY headers and shared metadata reserve.

    Returns
    -------
    dict
        Total study forecast and additional whole-stream space, accounting for present preparation.
    """
    arrays = (
        _npy_bytes((192, 12, 2500), "float32")
        + 2 * _npy_bytes((768, 12, 2500), "bool")
        + 2 * _npy_bytes((768, 12, 2500), "float32")
    )
    present = _directory_bytes(STUDY) if STUDY.exists() else 0
    other = stream_bytes() - present
    forecast = arrays + 64 * 1024**2
    return {
        "total_array_payload_bytes": 253440000,
        "total_array_npy_bytes": arrays,
        "shared_metadata_reserve_bytes": 64 * 1024**2,
        "total_study_forecast_bytes": forecast,
        "existing_study_bytes": present,
        "existing_other_stream_bytes": other,
        "additional_bytes_required": max(0, forecast - present),
        "projected_whole_stream_bytes": other + max(present, forecast),
    }


def _admission() -> dict[str, Any]:
    prep_path = PREPARATION / "summary.json"
    initial_prep = json.loads(prep_path.read_text())
    if not initial_prep["qualification_gate_passed"]:
        raise ValueError("Real-background qualification failed; no profile/inference admitted")
    paths = {
        "preparation": prep_path,
        "post_verification": PREPARATION / "post_verification.json",
        "training": BASE / "pilot_v1/training/summary.json",
        "original": BASE / "pilot_v1/evaluation/summary.json",
        "counterfactual": BASE / "pilot_v1/counterfactual_audit_v2/summary.json",
        "original_audit": BASE / "pilot_v1/final_audit_v1/summary.json",
        "closure": BASE / "pilot_v1/final_audit_v1/reproduction_binding_closure.json",
    }
    reports = {name: _read_receipt(path) for name, path in paths.items()}
    original = reports["original"]
    if not original["evaluation_completed"] or original["decision"]["engineering_advancement_passed"]:
        raise ValueError("Original closed-pilot history changed")
    require_original_audit(reports["closure"], sha256_file(paths["original"]))
    if not reports["original_audit"]["all_passed"] or reports["original_audit"]["mismatch_count"]:
        raise ValueError("Original numerical reproduction no longer passes")
    if not reports["post_verification"]["all_passed"]:
        raise ValueError("Independent preparation post-verification failed")
    conditions = conditional_admission(reports["counterfactual"])
    sources = {name: sha256_file(ROOT / name) for name in SOURCE_NAMES}
    inputs = {to_stored(path): sha256_file(path) for path in paths.values()}
    for name, report in reports.items():
        _merge_hashes(sources, report.get("source_sha256", {}))
        _merge_hashes(sources, report.get("input_audit_source_sha256", {}))
        _merge_hashes(sources, report.get("prerequisite_source_sha256", {}))
        _merge_hashes(inputs, report.get("input_sha256", {}))
        _merge_hashes(
            inputs,
            {
                to_stored(paths[name].parent / file): digest
                for file, digest in report.get("output_sha256", {}).items()
            },
        )
    verify_inference_hashes(sources | inputs)
    config = json.loads((ROOT / "configs/ssl_synthetic_real_background_v1.json").read_text())
    original_config = json.loads((ROOT / "configs/ssl_synthetic_pilot_v1.json").read_text())
    container = json.loads((PREPARATION / "cases.json").read_text())
    roles = validate_case_plan(container["cases"])
    backgrounds = np.load(PREPARATION / "backgrounds.npy", mmap_mode="r", allow_pickle=False)
    support = np.load(PREPARATION / "support.npy", mmap_mode="r", allow_pickle=False)
    candidate = np.load(PREPARATION / "candidate.npy", mmap_mode="r", allow_pickle=False)
    if backgrounds.shape != (192, 12, 2500) or backgrounds.dtype != np.float32:
        raise ValueError("Qualified background cache geometry changed")
    if any(value.shape != (768, 12, 2500) or value.dtype != bool for value in (support, candidate)):
        raise ValueError("Qualified native mask geometry changed")
    prep = reports["preparation"]
    stage_cpu = prep.get(
        "stage_cpu_seconds", prep["cpu_seconds"] + prep.get("prior_selection_cpu_seconds", 0)
    )
    return {
        **reports,
        "config": config,
        "original_config": original_config,
        "sources": sources,
        "inputs": inputs,
        "conditions": conditions,
        "cases": container["cases"],
        "roles": roles,
        "backgrounds": backgrounds,
        "support": support,
        "candidate": candidate,
        "stage_preparation_cpu_seconds": stage_cpu + reports["post_verification"]["cpu_seconds"],
        "prior_cpu_seconds": reports["counterfactual"]["cumulative_cpu_seconds"],
        "prior_gpu_wall_seconds": original["total_profile_training_evaluation_wall_seconds"],
        "regeneration_checks": {},
    }


def _background_api() -> tuple[Any, Any]:
    from ecg_experiment.ssl_synthetic_real_background import array_sha256, render_case

    return render_case, array_sha256


def render_checked(context: dict[str, Any], index: int) -> dict[str, Any]:
    """Regenerate and verify every observed/counterpart/mask byte before model consumption.

    Parameters
    ----------
    context : dict
        Admitted cache/descriptors; private patient metadata stays local.
    index : int
        Global case/mask/probability-map row.

    Returns
    -------
    dict
        Qualified native arrays; only edited is passed to the model.
    """
    render, digest = _background_api()
    case = context["cases"][index]
    background = context["backgrounds"][case["cache_index"]]
    if digest(background) != case["processed_background_sha256"]:
        raise ValueError("Immutable processed-background row hash changed")
    result = render(background, case)
    checks = {}
    for name, dtype in (
        ("edited", np.float32),
        ("baseline", np.float32),
        ("support", bool),
        ("candidate", bool),
    ):
        value = result[name]
        if value.shape != (12, 2500) or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError("Regenerated native array geometry/finiteness changed")
        checks[name] = digest(value)
        if checks[name] != case[f"{name}_sha256"]:
            raise ValueError("Regenerated immutable case hash changed")
    if not np.array_equal(result["support"], context["support"][index]) or not np.array_equal(
        result["candidate"], context["candidate"][index]
    ):
        raise ValueError("Regenerated masks differ from qualified saved masks")
    context["regeneration_checks"][str(index)] = checks
    return result


def _observed_batch(context: dict[str, Any], indices: list[int]) -> np.ndarray:
    return np.stack([render_checked(context, index)["edited"] for index in indices]).astype(np.float32)


def _predict_batch(
    model: torch.nn.Module, context: dict[str, Any], indices: list[int], device: str = "cuda"
) -> np.ndarray:
    signal = torch.from_numpy(_observed_batch(context, indices)).to(device)
    logits = model(signal)
    if logits.shape != signal.shape or not torch.isfinite(logits).all():
        raise ValueError("Nonfinite or misaligned native inference output")
    values = torch.sigmoid(logits).cpu().numpy()
    if values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError("Native probabilities are not finite float32")
    return values


def _thread_environment() -> dict[str, str | None]:
    names = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    values = {name: os.environ.get(name) for name in names}
    if any(value != "1" for value in values.values()):
        raise ValueError("Transfer requires one numerical thread")
    return values


def _guard(context: dict[str, Any], clock: dict[str, float]) -> None:
    stage_wall = clock["prior_stage_wall"] + time.perf_counter() - clock["wall"]
    stage_cpu = clock["prior_stage_cpu"] + time.process_time() - clock["cpu"]
    config = context["config"]
    if (
        stage_wall >= config["gpu_seconds_stage_ceiling"]
        or stage_cpu >= config["cpu_seconds_stage_ceiling"]
        or context["prior_gpu_wall_seconds"] + stage_wall >= 7200
        or context["prior_cpu_seconds"] + stage_cpu >= 14400
    ):
        raise TimeoutError("Frozen stage/global transfer resource ceiling exhausted")
    if stream_bytes() > 4 * 1024**3:
        raise ValueError("Frozen whole-stream cache ceiling exhausted")


def profile_projection(
    mean_wall: float, mean_cpu: float, profile_wall: float, profile_cpu: float, preparation_cpu: float
) -> dict[str, float | int]:
    """Apply the fixed 2x actual-path batch forecast and reserve to all 96 inference batches.

    Parameters
    ----------
    mean_wall, mean_cpu : float
        Measured regenerated-input/native-inference batch seconds.
    profile_wall, profile_cpu, preparation_cpu : float
        Executed prior new-stage costs.

    Returns
    -------
    dict
        Stage GPU-wall/CPU forecast including profile and 120-second reserves.
    """
    return {
        "inference_batches": 96,
        "safety_factor": 2,
        "reserve_seconds": 120,
        "projected_stage_gpu_wall_seconds": profile_wall + 2 * 96 * mean_wall + 120,
        "projected_stage_cpu_seconds": preparation_cpu + profile_cpu + 2 * 96 * mean_cpu + 120,
    }


def _failure(
    directory: Path, sources: dict, inputs: dict, progress: dict, error: Exception, wall: float, cpu: float
) -> None:
    write_new_json(
        directory / "failure.json",
        {
            "completed": False,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "source_sha256": sources,
            "input_sha256": inputs,
            "progress": progress,
            "failure_type": type(error).__name__,
            "failure_detail": str(error),
            "wall_seconds": time.perf_counter() - wall,
            "cpu_seconds": time.process_time() - cpu,
        },
    )


def profile_transfer() -> dict[str, Any]:
    """Profile only admitted A regeneration/inference, discarding all probe probabilities.

    Returns
    -------
    dict
        Immutable profile/resource admission; no target performance or threshold is calculated.
    """
    if PROFILE.exists():
        raise FileExistsError("Transfer profile directory must be new")
    wall, cpu = time.perf_counter(), time.process_time()
    sources = {name: sha256_file(ROOT / name) for name in SOURCE_NAMES}
    inputs = {}
    progress = {"warmup_batches": 0, "timed_batches": 0}
    with gpu_lock("cuda", blocking=False):
        PROFILE.mkdir(parents=True, exist_ok=False)
        try:
            threads = _thread_environment()
            context = _admission()
            sources, inputs = context["sources"], context["inputs"]
            storage = storage_projection()
            if storage["projected_whole_stream_bytes"] > 4 * 1024**3:
                raise ValueError("Nonduplicated whole-study storage forecast exceeds 4 GiB")
            configure_inference()
            torch.cuda.reset_peak_memory_stats()
            model = load_final_model("R", context["training"], context["original_config"]).to("cuda").eval()
            before = state_digest(model.state_dict())
            clock = {
                "wall": wall,
                "cpu": cpu,
                "prior_stage_wall": 0,
                "prior_stage_cpu": context["stage_preparation_cpu_seconds"],
            }
            write_new_json(
                PROFILE / "admission.json",
                {
                    "source_sha256": sources,
                    "input_sha256": inputs,
                    "conditions": context["conditions"],
                    "storage_projection": storage,
                },
            )
            durations = []
            cpu_durations = []
            cal = context["roles"]["calibration"]
            with torch.inference_mode():
                for batch in range(30):
                    _guard(context, clock)
                    indices = [cal[(batch * 16 + offset) % len(cal)] for offset in range(16)]
                    torch.cuda.synchronize()
                    start, process = time.perf_counter(), time.process_time()
                    values = _predict_batch(model, context, indices)
                    torch.cuda.synchronize()
                    if batch >= 10:
                        durations.append(time.perf_counter() - start)
                        cpu_durations.append(time.process_time() - process)
                        progress["timed_batches"] += 1
                    else:
                        progress["warmup_batches"] += 1
                    del values
            if state_digest(model.state_dict()) != before:
                raise ValueError("Fixed profile model state changed")
            verify_inference_hashes(sources | inputs)
            _guard(context, clock)
            elapsed, used = time.perf_counter() - wall, time.process_time() - cpu
            projection = profile_projection(
                float(np.mean(durations)),
                float(np.mean(cpu_durations)),
                elapsed,
                used,
                context["stage_preparation_cpu_seconds"],
            )
            passed = (
                projection["projected_stage_gpu_wall_seconds"] <= 900
                and projection["projected_stage_cpu_seconds"] <= 1800
                and context["prior_gpu_wall_seconds"] + projection["projected_stage_gpu_wall_seconds"] <= 7200
                and context["prior_cpu_seconds"] + projection["projected_stage_cpu_seconds"] <= 14400
            )
            result = {
                "profile_gate_passed": passed,
                "created_at_utc": datetime.now(UTC).isoformat(),
                "git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "source_sha256": sources,
                "input_sha256": inputs,
                "progress": progress,
                "mean_batch_wall_seconds": float(np.mean(durations)),
                "mean_batch_cpu_seconds": float(np.mean(cpu_durations)),
                "projection": projection,
                "storage_projection": storage,
                "wall_seconds": elapsed,
                "cpu_seconds": used,
                "stage_preparation_cpu_seconds": context["stage_preparation_cpu_seconds"],
                "prior_gpu_wall_seconds": context["prior_gpu_wall_seconds"],
                "prior_cpu_seconds": context["prior_cpu_seconds"],
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "model_state_unchanged": True,
                "numerical_threads": threads,
                "torch_threads": torch.get_num_threads(),
                "torch_interop_threads": torch.get_num_interop_threads(),
                "optimizer_updates": 0,
                "performance_metrics_computed": False,
                "probe_maps_saved": False,
                "unique_regenerated_cases": len(context["regeneration_checks"]),
                "output_sha256": {"admission.json": sha256_file(PROFILE / "admission.json")},
            }
            write_new_json(PROFILE / "summary.json", result)
            return result
        except Exception as error:
            _failure(PROFILE, sources, inputs, progress, error, wall, cpu)
            raise RuntimeError("Transfer profile stopped; inspect its local failure receipt") from None


def _infer_subset(
    context: dict,
    arm: str,
    indices: list[int],
    output: np.ndarray,
    clock: dict[str, float],
    progress: dict[str, Any],
) -> None:
    model = load_final_model(arm, context["training"], context["original_config"]).to("cuda").eval()
    before = state_digest(model.state_dict())
    with torch.inference_mode():
        for start in range(0, len(indices), 16):
            _guard(context, clock)
            batch = indices[start : start + 16]
            output[batch] = _predict_batch(model, context, batch)
            progress["predicted_record_maps"] += len(batch)
    output.flush()
    if state_digest(model.state_dict()) != before:
        raise ValueError("Fixed transfer model state changed")
    model.to("cpu")
    del model
    torch.cuda.empty_cache()


def calibration_thresholds(cases: list[dict[str, Any]], maps: dict[str, np.ndarray]) -> dict[str, float]:
    """Fit only paired new-domain calibration controls, indexed on the global map grid.

    Parameters
    ----------
    cases : list of dict
        Immutable descriptors with global case_index and within-role patient coordinates.
    maps : dict
        R/S combined native maps; only calibration rows are read.

    Returns
    -------
    dict
        Fixed higher-95% per-arm thresholds; later flags use strict greater-than.
    """
    controls = {
        kind: sorted(
            (case for case in cases if case["role"] == "calibration" and case["kind"] == kind),
            key=lambda case: case["role_subject_index"],
        )
        for kind in ("sham", "nuisance")
    }
    if any(len(values) != 64 for values in controls.values()):
        raise ValueError("Calibration patient/control denominator changed")
    return {
        arm: control_threshold(
            np.array([maps[arm][case["case_index"]].max() for case in controls["sham"]]),
            np.array([maps[arm][case["case_index"]].max() for case in controls["nuisance"]]),
        )
        for arm in ARMS
    }


def _slice_hash(values: np.ndarray, indices: list[int]) -> str:
    return hashlib.sha256(np.ascontiguousarray(values[indices]).tobytes()).hexdigest()


def _matching_control(context: dict, case: dict) -> tuple[dict, dict]:
    kind = "nuisance" if case["nuisance"] else "sham"
    matches = [
        other
        for other in context["cases"]
        if other["role"] == case["role"]
        and other["role_subject_index"] == case["role_subject_index"]
        and other["kind"] == kind
    ]
    if len(matches) != 1 or matches[0]["patient_id"] != case["patient_id"]:
        raise ValueError("Matched-counterpart identity changed")
    positive = render_checked(context, case["case_index"])
    control = render_checked(context, matches[0]["case_index"])
    if positive["baseline"].tobytes() != control["edited"].tobytes():
        raise ValueError("Positive baseline is not the exact observed matching control")
    return positive, matches[0]


def _case_rows(
    context: dict, maps: dict[str, np.ndarray], thresholds: dict[str, float], clock: dict
) -> tuple[list, list]:
    rows = []
    responses = []
    for case in context["cases"]:
        if case["role"] != "evaluation":
            continue
        _guard(context, clock)
        index = case["case_index"]
        identity = {
            "case": index,
            "subject": case["role_subject_index"],
            "global_subject": case["cache_index"],
            "kind": case["kind"],
            "mode": case["mode"],
        }
        for arm in ARMS:
            score = maps[arm][index]
            maximum = float(score.max())
            row = {
                **identity,
                "method": arm,
                "maximum_probability": maximum,
                "flag": maximum > thresholds[arm],
            }
            if case["target"]:
                row.update(positive_map_metrics(context["support"][index], score))
            rows.append(row)
        if not case["target"]:
            continue
        rendered, control = _matching_control(context, case)
        contextual = {
            "uniform": np.zeros((12, 2500), dtype=np.float32),
            "absolute_voltage": np.abs(rendered["edited"]),
            "algorithmic_anchor_prior": context["candidate"][index].astype(np.float32),
        }
        rows.extend(
            {**identity, "method": name, **positive_map_metrics(context["support"][index], score)}
            for name, score in contextual.items()
        )
        for arm in ARMS:
            responses.append(
                {
                    **identity,
                    "method": arm,
                    "control_case": control["case_index"],
                    **response_metrics(
                        context["support"][index],
                        context["candidate"][index],
                        maps[arm][index],
                        maps[arm][control["case_index"]],
                    ),
                }
            )
    return rows, responses


def _paired(
    rows: list[dict], first: str, second: str, metric: str, kind: str, mode: str = "all"
) -> tuple[np.ndarray, ...]:
    selected = [row for row in rows if row["kind"] == kind and (mode == "all" or row["mode"] == mode)]
    left = sorted(
        (row for row in selected if row["method"] == first), key=lambda row: (row["subject"], row["case"])
    )
    right = sorted(
        (row for row in selected if row["method"] == second), key=lambda row: (row["subject"], row["case"])
    )
    if not left or [(row["subject"], row["case"]) for row in left] != [
        (row["subject"], row["case"]) for row in right
    ]:
        raise ValueError("Paired full-patient comparison lost cases")
    return (
        np.array([row["subject"] for row in left]),
        np.array([row[metric] for row in left], dtype=float),
        np.array([row[metric] for row in right], dtype=float),
    )


def _flag_rates(rows: list[dict]) -> dict[str, Any]:
    flags = {}
    for arm in ARMS:
        flags[arm] = {}
        for kind in ("sham", "nuisance", "positive"):
            selected = [row for row in rows if row["method"] == arm and row["kind"] == kind]
            flags[arm][kind] = {
                "cases": len(selected),
                "flagged": sum(row["flag"] for row in selected),
                "rate": float(np.mean([row["flag"] for row in selected])),
            }
        flags[arm]["positive_by_mode"] = {}
        for mode in ("focal", "repeated"):
            selected = [
                row
                for row in rows
                if row["method"] == arm and row["kind"] == "positive" and row["mode"] == mode
            ]
            flags[arm]["positive_by_mode"][mode] = {
                "cases": len(selected),
                "flagged": sum(row["flag"] for row in selected),
                "rate": float(np.mean([row["flag"] for row in selected])),
            }
    return flags


def summarize(rows: list[dict], responses: list[dict], config: dict) -> dict[str, Any]:
    """Summarize full-map performance and response on the fixed 128-patient endpoint.

    Parameters
    ----------
    rows, responses : list of dict
        Every engineering-evaluation case/method result and matched-control diagnostic.
    config : dict
        Frozen 2,000-draw patient bootstrap and seed 20261023.

    Returns
    -------
    dict
        Absolute methods/modes, paired S-minus-R intervals, controls and input response; no pass rule.
    """
    boot = {"draws": config["bootstrap_draws"], "seed": config["bootstrap_seed"]}
    methods = {}
    for method in (*ARMS, "uniform", "absolute_voltage", "algorithmic_anchor_prior"):
        methods[method] = {}
        for mode in MODES:
            selected = [
                row
                for row in rows
                if row["method"] == method
                and row["kind"] == "positive"
                and (mode == "all" or row["mode"] == mode)
            ]
            methods[method][mode] = {
                name: float(
                    np.mean(
                        [
                            np.mean([row[name] for row in selected if row["subject"] == patient])
                            for patient in sorted({row["subject"] for row in selected})
                        ]
                    )
                )
                for name in (
                    "average_precision",
                    "prevalence",
                    "top_one_percent_precision",
                    "top_one_percent_recall",
                )
            }
            methods[method][mode].update(
                {"patients": len({row["subject"] for row in selected}), "cases": len(selected)}
            )
    differences = {
        mode: paired_subject_difference(
            *_paired(rows, "S", "R", "average_precision", "positive", mode), **boot
        )
        for mode in MODES
    }
    controls = {
        kind: paired_subject_difference(*_paired(rows, "S", "R", "flag", kind), **boot)
        for kind in ("sham", "nuisance")
    }
    flags = _flag_rates(rows)
    response = {}
    for arm in ARMS:
        response[arm] = {}
        for mode in MODES:
            selected = [
                row for row in responses if row["method"] == arm and (mode == "all" or row["mode"] == mode)
            ]
            subject = np.array([row["subject"] for row in selected])
            response[arm][mode] = {
                name: paired_subject_difference(
                    subject, np.array([row[name] for row in selected]), np.zeros(len(selected)), **boot
                )
                for name in RESPONSE_METRICS
            }
    return {
        "methods": methods,
        "s_minus_r_ap": differences,
        "s_minus_r_control_flags": controls,
        "record_flags": flags,
        "counterfactual_response": response,
        "new_advancement_rule": None,
    }


def evaluate_transfer() -> dict[str, Any]:
    """Execute the qualified, profiled and reviewed frozen R/S introduced-change evaluation once.

    Returns
    -------
    dict
        Source/input/output-bound engineering-development result, without fitting or clinical claims.
    """
    if EVALUATION.exists():
        raise FileExistsError("Transfer evaluation directory must be new")
    wall, cpu = time.perf_counter(), time.process_time()
    sources = {name: sha256_file(ROOT / name) for name in SOURCE_NAMES}
    inputs = {}
    progress = {"predicted_record_maps": 0, "calibration_frozen": False, "completed": False}
    with gpu_lock("cuda", blocking=False):
        EVALUATION.mkdir(parents=True, exist_ok=False)
        try:
            threads = _thread_environment()
            context = _admission()
            sources, inputs = context["sources"], context["inputs"]
            profile = _read_receipt(PROFILE / "summary.json")
            if not profile["profile_gate_passed"]:
                raise ValueError("Actual transfer GPU profile failed")
            _merge_hashes(sources, profile["source_sha256"])
            _merge_hashes(inputs, profile["input_sha256"])
            _merge_hashes(
                inputs, {to_stored(PROFILE / "summary.json"): sha256_file(PROFILE / "summary.json")}
            )
            storage = storage_projection()
            if storage["projected_whole_stream_bytes"] > 4 * 1024**3:
                raise ValueError("Whole-study storage forecast fails")
            configure_inference()
            torch.cuda.reset_peak_memory_stats()
            clock = {
                "wall": wall,
                "cpu": cpu,
                "prior_stage_wall": profile["wall_seconds"],
                "prior_stage_cpu": context["stage_preparation_cpu_seconds"] + profile["cpu_seconds"],
            }
            _guard(context, clock)
            maps = {
                arm: open_memmap(
                    EVALUATION / f"probabilities_{arm}.npy",
                    mode="w+",
                    dtype=np.float32,
                    shape=(768, 12, 2500),
                )
                for arm in ARMS
            }
            write_new_json(
                EVALUATION / "admission.json",
                {
                    "source_sha256": sources,
                    "input_sha256": inputs,
                    "conditions": context["conditions"],
                    "storage_projection": storage,
                },
            )
            cal = context["roles"]["calibration"]
            for arm in ARMS:
                _infer_subset(context, arm, cal, maps[arm], clock, progress)
            calibration = {
                "thresholds": calibration_thresholds(context["cases"], maps),
                "patients": 64,
                "method": "higher",
                "quantile": 0.95,
                "comparison": "strictly greater",
                "global_calibration_case_indices": cal,
                "map_row_coordinate": "global case_index",
                "calibration_slice_sha256": {arm: _slice_hash(maps[arm], cal) for arm in ARMS},
                "source_sha256": sources,
                "input_sha256": inputs,
                "created_at_utc": datetime.now(UTC).isoformat(),
            }
            write_new_json(EVALUATION / "calibration.json", calibration)
            frozen = sha256_file(EVALUATION / "calibration.json")
            progress["calibration_frozen"] = True
            for arm in ARMS:
                if sha256_file(EVALUATION / "calibration.json") != frozen:
                    raise ValueError("Calibration bytes changed")
                _infer_subset(context, arm, context["roles"]["evaluation"], maps[arm], clock, progress)
            if sha256_file(EVALUATION / "calibration.json") != frozen or any(
                _slice_hash(maps[arm], cal) != calibration["calibration_slice_sha256"][arm] for arm in ARMS
            ):
                raise ValueError("Frozen calibration bytes or native calibration slices changed")
            rows, responses = _case_rows(context, maps, calibration["thresholds"], clock)
            write_new_json(
                EVALUATION / "local_case_metrics.json", {"rows": rows, "counterfactual_rows": responses}
            )
            write_new_json(EVALUATION / "regeneration_checks.json", context["regeneration_checks"])
            summary = summarize(rows, responses, context["config"])
            verify_inference_hashes(sources | inputs)
            _guard(context, clock)
            progress["completed"] = True
            outputs = {path.name: sha256_file(path) for path in EVALUATION.iterdir() if path.is_file()}
            _guard(context, clock)
            result = {
                "evaluation_completed": True,
                "created_at_utc": datetime.now(UTC).isoformat(),
                "git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "source_sha256": sources,
                "input_sha256": inputs,
                "output_sha256": outputs,
                "calibration_sha256": frozen,
                "calibration": calibration,
                "progress": progress,
                "patients": 128,
                "evaluation_cases": 512,
                "calibration_cases": 256,
                "probability_shape": [768, 12, 2500],
                "probability_dtype": "float32",
                "map_row_coordinate": "global case_index",
                "unique_regenerated_cases": len(context["regeneration_checks"]),
                "regenerated_hash_checks": 4 * len(context["regeneration_checks"]),
                "all_regenerated_hashes_match": True,
                "storage_projection": storage,
                **summary,
                "wall_seconds": time.perf_counter() - wall,
                "cpu_seconds": time.process_time() - cpu,
                "stage_gpu_wall_seconds": profile["wall_seconds"] + time.perf_counter() - wall,
                "stage_cpu_seconds": clock["prior_stage_cpu"] + time.process_time() - cpu,
                "global_gpu_wall_seconds": context["prior_gpu_wall_seconds"]
                + profile["wall_seconds"]
                + time.perf_counter()
                - wall,
                "global_cpu_seconds": context["prior_cpu_seconds"]
                + clock["prior_stage_cpu"]
                + time.process_time()
                - cpu,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "stream_bytes": stream_bytes(),
                "original_advancement_passed": False,
                "clinical_claim": False,
                "new_training_updates": 0,
                "numerical_threads": threads,
                "torch_threads": torch.get_num_threads(),
                "torch_interop_threads": torch.get_num_interop_threads(),
            }
            write_new_json(EVALUATION / "summary.json", result)
            return result
        except Exception as error:
            _failure(EVALUATION, sources, inputs, progress, error, wall, cpu)
            raise RuntimeError("Transfer evaluation stopped; inspect its local failure receipt") from None
