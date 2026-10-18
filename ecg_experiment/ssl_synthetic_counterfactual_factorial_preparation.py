"""Freeze and qualify only the new192 validation backgrounds; reference exact old TRAIN artifacts."""

from __future__ import annotations

import time
from collections import Counter
from datetime import UTC, datetime

import numpy as np

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file, sha256_json
from ecg_experiment.ssl_synthetic_checksum_audit import read_checksums
from ecg_experiment.ssl_synthetic_counterfactual_factorial_common import (
    CHECKSUM_PATH,
    PREPARATION,
    predecessor_metadata,
    storage_projection,
    verify_bindings,
)
from ecg_experiment.ssl_synthetic_counterfactual_factorial_data import (
    cases_for_background,
    decode_background,
    new_case_row,
    render_case,
    select_validation,
)
from ecg_experiment.ssl_synthetic_preparation import write_new_json
from ecg_experiment.ssl_synthetic_real_background import array_sha256, find_anchors, qualify_pair
from ecg_experiment.waveforms import LEADS


def guard(context: dict, wall: float, cpu: float) -> None:
    """Keep every new raw/qualification step inside exact remaining CPU and monotonic allowances."""
    allowance = min(
        context["config"]["cpu_seconds_stage_ceiling"],
        context["config"]["cpu_seconds_ceiling"] - context["prior_cpu_seconds"],
    )
    if max(time.process_time() - cpu, time.perf_counter() - wall) + 2 >= allowance:
        raise TimeoutError("Factorial preparation CPU/wall allowance exhausted")


def qualify_cases(
    signal: np.ndarray,
    row: dict,
    anchors: list[int],
    config: dict,
    support: np.ndarray,
    candidate: np.ndarray,
) -> list[dict]:
    """Qualify all four native pairs, exact replays and both positive matched-control baselines."""
    cases = cases_for_background(row, anchors, array_sha256(signal), config)
    for case in cases:
        pair, replay = render_case(signal, case), render_case(signal, case)
        qualification = qualify_pair(pair, case, config)
        exact = all(
            pair[name].tobytes() == replay[name].tobytes()
            for name in ("edited", "baseline", "support", "candidate", "analytic_delta")
        )
        case["deterministic_replay_passed"] = exact
        if not exact:
            qualification["failures"].append("deterministic_replay")
        for name in ("edited", "baseline", "support", "candidate"):
            case[name + "_sha256"] = array_sha256(pair[name])
        case["qualification"] = qualification
        local = new_case_row(case["case_index"])
        support[local], candidate[local] = pair["support"], pair["candidate"]
    for case in cases[:2]:
        control = cases[3 if case["nuisance"] else 2]
        if (
            case["matched_control_case_index"] != control["case_index"]
            or case["baseline_sha256"] != control["edited_sha256"]
        ):
            case["qualification"]["failures"].append("matched_control_baseline")
    return cases


def process_rows(
    rows: list[dict], context: dict, checksums: dict, wall: float, cpu: float
) -> tuple[list, list]:
    """Stream new192 records only, retaining every row and stopping further payloads on failure."""
    backgrounds = np.lib.format.open_memmap(
        PREPARATION / "backgrounds.npy", mode="w+", dtype=np.float32, shape=(192, 12, 2500)
    )
    support = np.lib.format.open_memmap(
        PREPARATION / "support.npy", mode="w+", dtype=bool, shape=(768, 12, 2500)
    )
    candidate = np.lib.format.open_memmap(
        PREPARATION / "candidate.npy", mode="w+", dtype=bool, shape=(768, 12, 2500)
    )
    backgrounds[:], support[:], candidate[:] = np.nan, False, False
    metadata, output = [], []
    stopped = False
    for row in rows:
        info = dict(row, status="processing_failed", anchors=[], payload_sha256={}, verified_bytes=0)
        try:
            guard(context, wall, cpu)
            if stopped:
                raise ValueError("Earlier qualification failure stopped subsequent payload access")
            signal, observed = decode_background(row, checksums)
            info.update(observed)
            backgrounds[row["background_row"]] = signal
            anchor_info = find_anchors(signal, context["config"])
            info.update(anchor_info, status="processed", processed_background_sha256=array_sha256(signal))
            cases = qualify_cases(signal, row, anchor_info["anchors"], context["config"], support, candidate)
            stopped = not anchor_info["minimum_anchor_gate_passed"] or any(
                case["qualification"]["failures"] for case in cases
            )
        except (OSError, ValueError, EOFError) as error:
            stopped = True
            info.update(
                status="processing_failed", failure_type=type(error).__name__, failure_detail=str(error)
            )
            cases = cases_for_background(
                row, [], array_sha256(backgrounds[row["background_row"]]), context["config"]
            )
            for case in cases:
                case["qualification"] = {"failures": ["background_or_prior_failure"], "metrics": {}}
        metadata.append(info)
        output.extend(cases)
    for array in (backgrounds, support, candidate):
        array.flush()
    return metadata, output


def _prepare() -> dict:
    """Freeze the complete new selection before raw access, then qualify all768 cases or fail."""
    if PREPARATION.exists() or PREPARATION.is_symlink():
        raise FileExistsError("Factorial preparation requires a new output directory")
    wall, cpu = time.perf_counter(), time.process_time()
    context = predecessor_metadata()
    config = context["config"]
    selected = select_validation(
        context["partition"]["records"],
        context["eligibility"]["records"],
        [context["first_selection"], context["predecessor_selection"]],
        context["ssl_selection"],
        config,
    )
    rows = selected["records"]
    if [row["background_row"] for row in rows] != list(range(192)):
        raise ValueError("New local background grid changed")
    guard(context, wall, cpu)
    projection = storage_projection()
    if projection["projected_whole_stream_bytes"] > config["cache_bytes_ceiling"]:
        raise ValueError("Prospective whole-stream storage ceiling exceeded")
    checksums = read_checksums((ROOT / CHECKSUM_PATH).read_text())
    train_cases_hash = sha256_json(context["predecessor_cases"]["cases"][:2000])
    PREPARATION.mkdir(parents=True, exist_ok=False)
    selected.update(
        schema_version=1,
        study_id=config["study_id"],
        source_sha256=context["sources"],
        input_sha256=context["inputs"],
        reused_train_cases_sha256=train_cases_hash,
        reused_train_cases_source="outputs/ssl_synthetic_localization/paired_background_v1/preparation/cases.json",
    )
    write_new_json(PREPARATION / "selection.json", selected)
    selected_hash = sha256_file(PREPARATION / "selection.json")
    metadata, cases = process_rows(rows, context, checksums, wall, cpu)
    write_new_json(
        PREPARATION / "cases.json",
        {
            "schema_version": 1,
            "study_id": config["study_id"],
            "selection_sha256": selected_hash,
            "backgrounds": metadata,
            "cases": cases,
            "reused_train_cases_sha256": train_cases_hash,
            "new_mask_row_rule": "global_case_index - 2000",
            "new_background_row_rule": "virtual_subject_index - 500",
        },
    )
    exceptions = verify_bindings(context["sources"], context["inputs"], context["authorities"])
    if sha256_file(PREPARATION / "selection.json") != selected_hash:
        raise ValueError("Prospective selection changed after raw access")
    outputs = {
        name: sha256_file(PREPARATION / name)
        for name in ("selection.json", "cases.json", "backgrounds.npy", "support.npy", "candidate.npy")
    }
    guard(context, wall, cpu)
    failures = Counter(reason for case in cases for reason in case["qualification"]["failures"])
    bad_backgrounds = sum(
        row["status"] != "processed" or not row.get("minimum_anchor_gate_passed", False) for row in metadata
    )
    projection = storage_projection()
    elapsed_cpu, elapsed_wall = time.process_time() - cpu, time.perf_counter() - wall
    budget = (
        max(elapsed_cpu, elapsed_wall) < config["cpu_seconds_stage_ceiling"]
        and context["prior_cpu_seconds"] + elapsed_cpu < config["cpu_seconds_ceiling"]
    )
    storage = projection["projected_whole_stream_bytes"] <= config["cache_bytes_ceiling"]
    report = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "qualification_gate_passed": not failures
        and bad_backgrounds == 0
        and len(cases) == 768
        and budget
        and storage,
        "selected_validation_patients": 192,
        "calibration_patients": 64,
        "evaluation_patients": 128,
        "reused_train_patients": 500,
        "reused_train_cases": 2000,
        "reused_train_cases_sha256": train_cases_hash,
        "train_waveform_payloads_opened_during_preparation": 0,
        "backgrounds": len(metadata),
        "cases": len(cases),
        "failed_backgrounds": bad_backgrounds,
        "failed_cases": sum(bool(case["qualification"]["failures"]) for case in cases),
        "failure_counts": dict(failures),
        "deterministic_replay_cases_passed": sum(
            case.get("deterministic_replay_passed", False) for case in cases
        ),
        "positive_control_baselines_checked": sum(case["target"] for case in cases),
        "anchor_count_distribution": dict(Counter(len(row["anchors"]) for row in metadata)),
        "records_with_retained_quality_exclusions": sum(bool(row["quality_exclusions"]) for row in metadata),
        "records_with_retained_review_flags": sum(bool(row["review_flags"]) for row in metadata),
        "source_sha256": context["sources"],
        "input_sha256": context["inputs"],
        "output_sha256": outputs,
        "scope_exceptions": exceptions,
        "predecessor_reproduction": context["predecessor_reproduction"],
        "selection_sha256": selected_hash,
        "selection_identity_sha256": selected["selection_identity_sha256"],
        "sample_rate_hz": 250,
        "samples": 2500,
        "units": "mV",
        "lead_order": list(LEADS),
        "verified_raw_bytes": sum(row["verified_bytes"] for row in metadata),
        "prior_cpu_seconds": context["prior_cpu_seconds"],
        "prior_gpu_wall_seconds": context["prior_gpu_wall_seconds"],
        "cpu_seconds": elapsed_cpu,
        "wall_seconds": elapsed_wall,
        "stage_preparation_cpu_seconds": elapsed_cpu,
        "cumulative_cpu_seconds": context["prior_cpu_seconds"] + elapsed_cpu,
        "storage_projection": projection,
        "resource_budget_gate_passed": budget,
        "storage_gate_passed": storage,
        "gpu_used": False,
        "model_inferences": 0,
        "optimizer_updates": 0,
        "numerical_cpu_threads": 1,
        "closed_real_test_opened": False,
    }
    metrics = [case["qualification"]["metrics"] for case in cases if case["qualification"]["metrics"]]
    report["metric_maxima"] = {key: max(row[key] for row in metrics) for key in metrics[0]} if metrics else {}
    write_new_json(PREPARATION / "summary.json", report)
    return report


def prepare() -> dict:
    """Retain exact measured failures even when admission stops before a selection can be frozen."""
    if PREPARATION.exists() or PREPARATION.is_symlink():
        raise FileExistsError("Factorial preparation requires a new output directory")
    wall, cpu = time.perf_counter(), time.process_time()
    try:
        return _prepare()
    except Exception as error:
        PREPARATION.mkdir(parents=True, exist_ok=True)
        write_new_json(
            PREPARATION / "failure.json",
            {
                "failure_type": type(error).__name__,
                "failure_detail": str(error),
                "wall_seconds": time.perf_counter() - wall,
                "cpu_seconds": time.process_time() - cpu,
                "stage": "preparation",
                "qualification_gate_passed": False,
                "selection_written": (PREPARATION / "selection.json").exists(),
                "gpu_used": False,
                "model_inferences": 0,
                "optimizer_updates": 0,
            },
        )
        raise
