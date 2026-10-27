"""Prospective metadata freeze and complete matched-control real-background qualification."""

from __future__ import annotations

import io
import json
import math
import time
from collections import Counter
from datetime import UTC, datetime
from typing import Any

import numpy as np

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file
from ecg_experiment.paths import to_stored
from ecg_experiment.ssl_synthetic_cache import load_audited_inputs
from ecg_experiment.ssl_synthetic_evaluation import verify_inference_hashes
from ecg_experiment.ssl_synthetic_paired_background_data import (
    cases_for_background,
    prepare_background,
    render_case,
    select_rows,
)
from ecg_experiment.ssl_synthetic_preparation import stream_bytes, write_new_json
from ecg_experiment.ssl_synthetic_real_background import array_sha256, find_anchors, qualify_pair
from ecg_experiment.ssl_synthetic_real_background_preparation import SOURCE_FILES as LEGACY_SOURCES
from ecg_experiment.waveforms import LEADS

BASE = ROOT / "outputs/ssl_synthetic_localization"
OUTPUT = BASE / "paired_background_v1/preparation"
PRIOR = BASE / "real_background_v1"
SOURCE_FILES = (
    *LEGACY_SOURCES,
    "ecg_experiment/ssl_synthetic_paired_background_data.py",
    "ecg_experiment/ssl_synthetic_paired_background_preparation.py",
    "scripts/data/prepare_ssl_synthetic_paired_background.py",
    "tests/test_ssl_synthetic_paired_background_data.py",
    "docs/ssl-synthetic-localization-paired-background-study.md",
    "configs/ssl_synthetic_paired_background_v1.json",
)


def compare_reproduction(expected: Any, actual: Any) -> int:
    """Compare every independently recomputed predecessor value at absolute tolerance 1e-12."""
    if isinstance(expected, dict):
        return sum(compare_reproduction(value, actual[key]) for key, value in expected.items())
    if isinstance(expected, list):
        if len(expected) != len(actual):
            raise ValueError("Predecessor reproduction list length changed")
        return sum(compare_reproduction(a, b) for a, b in zip(expected, actual, strict=True))
    if isinstance(expected, float):
        passed = math.isclose(expected, actual, rel_tol=0, abs_tol=1e-12)
    else:
        passed = expected == actual
    if not passed:
        raise ValueError("Independently checked predecessor aggregate changed")
    return 1


def storage_forecast() -> dict[str, int]:
    """Charge all five prospective arrays and the frozen metadata/checkpoint reserve once."""
    specs = [
        ((692, 12, 2500), "float32"),
        *((((2768, 12, 2500), "bool"),) * 2),
        *((((768, 12, 2500), "float32"),) * 2),
    ]
    headers, payload = 0, 0
    for shape, dtype in specs:
        buffer = io.BytesIO()
        np.lib.format.write_array_header_1_0(
            buffer, {"descr": np.dtype(dtype).str, "fortran_order": False, "shape": shape}
        )
        headers += buffer.tell()
        payload += int(np.prod(shape)) * np.dtype(dtype).itemsize
    return {
        "array_payload_bytes": payload,
        "array_header_bytes": headers,
        "metadata_reserve_bytes": 64 * 1024**2,
        "complete_study_new_bytes": payload + headers + 64 * 1024**2,
    }


def merge_bindings(
    reports: dict[str, Any], paths: dict[str, Any], binding: dict[str, Any]
) -> tuple[dict[str, str], dict[str, str]]:
    """Merge frozen source/input/output declarations, rejecting any conflicting digest."""
    inputs = dict(binding["input_sha256"])
    sources = dict(binding["input_audit_source_sha256"])
    for name, report in reports.items():
        for field in ("source_sha256", "input_audit_source_sha256", "prerequisite_source_sha256"):
            merge_unique(sources, report.get(field, {}))
        merge_unique(inputs, report.get("input_sha256", {}))
        merge_unique(
            inputs,
            {
                to_stored(paths[name].parent / key): value
                for key, value in report.get("output_sha256", {}).items()
            },
        )
    return sources, inputs


def merge_unique(target: dict[str, str], extra: dict[str, str]) -> None:
    """Require each previously committed path to retain exactly one byte identity."""
    for key, value in extra.items():
        if key in target and target[key] != value:
            raise ValueError("Conflicting predecessor bindings")
        target[key] = value


def metadata_inputs() -> tuple[dict[str, Any], dict[str, str], dict[str, Any], dict[str, Any]]:
    """Bind the completed predecessor and original reservations without opening the SSL cache."""
    config = json.loads((ROOT / "configs/ssl_synthetic_paired_background_v1.json").read_bytes())
    manifest, audited, checksums, binding = load_audited_inputs(ROOT)
    paths = {
        "completion": PRIOR / "independent_audit_completion/summary.json",
        "evaluation": PRIOR / "evaluation/summary.json",
        "calibration": PRIOR / "evaluation/calibration.json",
        "preparation": PRIOR / "preparation/summary.json",
        "selection": PRIOR / "preparation/selection.json",
        "pilot": BASE / "pilot_v1/evaluation/summary.json",
    }
    reports = {key: json.loads(path.read_bytes()) for key, path in paths.items()}
    completion, evaluation, prep = (reports[key] for key in ("completion", "evaluation", "preparation"))
    if (
        not completion["all_passed"]
        or not completion["combined_complete_pass"]
        or completion["mismatch_count"]
    ):
        raise ValueError("Independent predecessor completion does not pass")
    if not evaluation["evaluation_completed"] or not prep["qualification_gate_passed"]:
        raise ValueError("Predecessor execution is incomplete")
    sources, inputs = merge_bindings(reports, paths, binding)
    for key in ("evaluation", "preparation", "selection"):
        path = paths[key]
        if completion["input_sha256"].get(to_stored(path)) != sha256_file(path):
            raise ValueError("Original reservation/provenance differs from frozen independent audit")
    verify_inference_hashes(sources | inputs)
    comparison_count = compare_reproduction(
        completion["independently_recomputed_complete_summary"], evaluation
    )
    comparison_count += compare_reproduction(
        reports["calibration"]["thresholds"], evaluation["calibration"]["thresholds"]
    )
    if (
        evaluation["new_advancement_rule"] is not None
        or evaluation["original_advancement_passed"]
        or reports["pilot"]["decision"]["engineering_advancement_passed"]
    ):
        raise ValueError("Original pilot failure or transfer rule history changed")
    if sha256_file(paths["selection"]) != prep["output_sha256"]["selection.json"]:
        raise ValueError("Original reservation selection changed")
    for path in paths.values():
        inputs[to_stored(path)] = sha256_file(path)
    selection = select_rows(manifest["records"], audited, reports["selection"])
    return (
        selection,
        checksums,
        {
            "input_sha256": inputs,
            "input_audit_source_sha256": sources,
            "predecessor_reproduction": {
                "all_passed": True,
                "scalar_comparisons": comparison_count,
                "calibration_thresholds": evaluation["calibration"]["thresholds"],
                "s_minus_r_ap": evaluation["s_minus_r_ap"],
                "original_advancement_passed": False,
            },
            "prior_cumulative_cpu_seconds": completion["cumulative_cpu_seconds"],
            "prior_gpu_wall_seconds": evaluation["global_gpu_wall_seconds"],
            "ssl_cache_waveform_payload_opened": False,
        },
        config,
    )


def check_budget(cpu_start: float, prior_cpu: float, config: dict[str, Any]) -> None:
    """Stop before another payload if either prospective CPU allowance is exhausted."""
    if time.process_time() - cpu_start >= min(
        config["cpu_seconds_stage_ceiling"], config["cpu_seconds_ceiling"] - prior_cpu
    ):
        raise ValueError("Frozen stage/cumulative qualification CPU ceiling reached")


def qualify_cases(
    signal: np.ndarray,
    row: dict[str, Any],
    anchors: list[int],
    config: dict[str, Any],
    supports: np.ndarray,
    candidates: np.ndarray,
) -> list[dict[str, Any]]:
    """Render/replay every variant, retain all gates, and compare both positive/control baselines."""
    cases = cases_for_background(row, anchors, array_sha256(signal), config)
    for case in cases:
        pair, replay = render_case(signal, case), render_case(signal, case)
        qualification = qualify_pair(pair, case, config)
        case["deterministic_replay_passed"] = all(
            pair[key].tobytes() == replay[key].tobytes()
            for key in ("baseline", "edited", "support", "candidate", "analytic_delta")
        )
        if not case["deterministic_replay_passed"]:
            qualification["failures"].append("deterministic_replay")
        for key in ("baseline", "edited", "support", "candidate"):
            case[key + "_sha256"] = array_sha256(pair[key])
        case["qualification"] = qualification
        supports[case["case_index"]], candidates[case["case_index"]] = pair["support"], pair["candidate"]
    for case in cases[:2]:
        control = cases[3 if case["nuisance"] else 2]
        if (
            case["matched_control_case_index"] != control["case_index"]
            or case["baseline_sha256"] != control["baseline_sha256"]
        ):
            case["qualification"]["failures"].append("matched_control_baseline")
    return cases


def process_rows(
    rows: list[dict[str, Any]],
    checksums: dict[str, str],
    config: dict[str, Any],
    cpu_start: float,
    prior_cpu: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Stream every selected background once; retain failures without replacing patients."""
    backgrounds = np.lib.format.open_memmap(
        OUTPUT / "backgrounds.npy", mode="w+", dtype=np.float32, shape=(692, 12, 2500)
    )
    supports = np.lib.format.open_memmap(
        OUTPUT / "support.npy", mode="w+", dtype=bool, shape=(2768, 12, 2500)
    )
    candidates = np.lib.format.open_memmap(
        OUTPUT / "candidate.npy", mode="w+", dtype=bool, shape=(2768, 12, 2500)
    )
    backgrounds[:], supports[:], candidates[:] = np.nan, False, False
    metadata, cases = [], []
    stopped = False
    for row in rows:
        info = dict(row, status="processing_failed", anchors=[], payload_sha256={}, verified_bytes=0)
        try:
            check_budget(cpu_start, prior_cpu, config)
            if stopped:
                raise ValueError("Earlier qualification failure stopped further payload access")
            signal, observed = prepare_background(row, checksums)
            info.update(observed)
            backgrounds[row["cache_index"]] = signal
            anchor_info = find_anchors(signal, config)
            info.update(anchor_info, status="processed", processed_background_sha256=array_sha256(signal))
            generated = qualify_cases(signal, row, anchor_info["anchors"], config, supports, candidates)
            cases.extend(generated)
            if not anchor_info["minimum_anchor_gate_passed"] or any(
                case["qualification"]["failures"] for case in generated
            ):
                stopped = True
        except (OSError, ValueError, EOFError) as error:
            stopped = True
            info.update(
                status="processing_failed", failure_type=type(error).__name__, failure_detail=str(error)
            )
            generated = cases_for_background(row, [], array_sha256(backgrounds[row["cache_index"]]), config)
            for case in generated:
                case["qualification"] = {
                    "failures": ["background_integrity_processing_or_prior_failure"],
                    "metrics": {},
                }
            cases.extend(generated)
        metadata.append(info)
    for array in (backgrounds, supports, candidates):
        array.flush()
    return metadata, cases


def prepare() -> dict[str, Any]:
    """Write the full selection first, then qualify all 692 backgrounds/2768 cases or fail."""
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise FileExistsError("Paired-background preparation output must be new")
    wall, cpu = time.perf_counter(), time.process_time()
    selection, checksums, binding, config = metadata_inputs()
    rows = selection["records"]
    if [row["cache_index"] for row in rows] != list(range(692)) or any(
        row["subject_index"] != row["cache_index"] for row in rows
    ):
        raise ValueError("Prospective global cache/subject mapping changed")
    check_budget(cpu, binding["prior_cumulative_cpu_seconds"], config)
    sources = {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
    forecast, size_before = storage_forecast(), stream_bytes()
    if size_before + forecast["complete_study_new_bytes"] > config["cache_bytes_ceiling"]:
        raise ValueError("Whole-stream prospective storage ceiling would be exceeded")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    selection.update(
        schema_version=1,
        study_id=config["study_id"],
        input_sha256=binding["input_sha256"],
        source_sha256=sources,
    )
    write_new_json(OUTPUT / "selection.json", selection)
    selection_hash = sha256_file(OUTPUT / "selection.json")
    metadata, cases = process_rows(rows, checksums, config, cpu, binding["prior_cumulative_cpu_seconds"])
    write_new_json(
        OUTPUT / "cases.json",
        {
            "schema_version": 1,
            "study_id": config["study_id"],
            "selection_sha256": selection_hash,
            "backgrounds": metadata,
            "cases": cases,
        },
    )
    verify_inference_hashes(sources | binding["input_sha256"] | binding["input_audit_source_sha256"])
    if sha256_file(OUTPUT / "selection.json") != selection_hash:
        raise ValueError("Prospective selection was changed after payload access")
    failures = Counter(reason for case in cases for reason in case["qualification"]["failures"])
    integrity = sum(info["status"] != "processed" for info in metadata)
    anchor_failures = sum(info.get("minimum_anchor_gate_passed") is not True for info in metadata)
    output_hashes = {
        name: sha256_file(OUTPUT / name)
        for name in ("backgrounds.npy", "support.npy", "candidate.npy", "selection.json", "cases.json")
    }
    current_cpu, current_wall = time.process_time() - cpu, time.perf_counter() - wall
    cumulative_cpu = binding["prior_cumulative_cpu_seconds"] + current_cpu
    size_after = stream_bytes()
    storage_gate = (
        size_after + 184_320_000 + 2 * 128 + config["metadata_reserve_bytes"] <= config["cache_bytes_ceiling"]
    )
    budget = (
        current_cpu < config["cpu_seconds_stage_ceiling"] and cumulative_cpu < config["cpu_seconds_ceiling"]
    )
    report = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "qualification_gate_passed": not failures
        and integrity == anchor_failures == 0
        and budget
        and storage_gate
        and len(cases) == 2768,
        "selected_patients": len(rows),
        "train_patients": 500,
        "calibration_patients": 64,
        "evaluation_patients": 128,
        "backgrounds": len(metadata),
        "cases": len(cases),
        "raw_waveforms_opened": sum(info["verified_bytes"] > 0 for info in metadata),
        "integrity_failures": integrity,
        "anchor_failed_backgrounds": anchor_failures,
        "failed_cases": sum(bool(case["qualification"]["failures"]) for case in cases),
        "deterministic_replay_cases_passed": sum(
            case.get("deterministic_replay_passed", False) for case in cases
        ),
        "positive_control_baselines_checked": sum(case["target"] for case in cases),
        "failure_counts": dict(failures),
        "anchor_count_distribution": dict(Counter(len(info["anchors"]) for info in metadata)),
        "records_with_retained_quality_exclusions": sum(
            bool(info["quality_exclusions"]) for info in metadata
        ),
        "records_with_retained_review_flags": sum(bool(info["review_flags"]) for info in metadata),
        "sample_rate_hz": 250,
        "samples": 2500,
        "units": "mV",
        "lead_order": list(LEADS),
        "selection_sha256": selection_hash,
        "selection_identity_sha256": selection["selection_identity_sha256"],
        "original_train_reservation_sha256": selection["original_train_reservation_sha256"],
        "source_sha256": sources,
        **binding,
        "output_sha256": output_hashes,
        "verified_raw_bytes": sum(info["verified_bytes"] for info in metadata),
        "wall_seconds": current_wall,
        "cpu_seconds": current_cpu,
        "stage_cpu_seconds": current_cpu,
        "cumulative_cpu_seconds": cumulative_cpu,
        "storage_forecast": forecast,
        "stream_bytes_before_preparation": size_before,
        "stream_bytes_before_summary": size_after,
        "future_storage_gate_passed": storage_gate,
        "resource_budget_gate_passed": budget,
        "model_inferences": 0,
        "training_started": False,
        "performance_scored": False,
        "numerical_cpu_threads": 1,
        "gpu_used": False,
        "closed_real_test_opened": False,
    }
    metrics = [case["qualification"]["metrics"] for case in cases if case["qualification"]["metrics"]]
    report["metric_maxima"] = (
        {key: max(record[key] for record in metrics) for key in metrics[0]} if metrics else {}
    )
    write_new_json(OUTPUT / "summary.json", report)
    return report
