"""Freeze real-background selection, then qualify every permitted validation background and case."""

from __future__ import annotations

import io
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file, sha256_json
from ecg_experiment.paths import to_stored
from ecg_experiment.resample import resample_full
from ecg_experiment.ssl_synthetic_cache import load_audited_inputs
from ecg_experiment.ssl_synthetic_evaluation import verify_inference_hashes
from ecg_experiment.ssl_synthetic_inputs import validate_header, verify_files
from ecg_experiment.ssl_synthetic_preparation import stream_bytes, verify_hashes, write_new_json
from ecg_experiment.ssl_synthetic_readiness import checked_relative
from ecg_experiment.ssl_synthetic_real_background import (
    array_sha256,
    cases_for_background,
    find_anchors,
    qualify_pair,
    render_case,
    select_patients,
)
from ecg_experiment.waveforms import LEADS, read_record

BASE = ROOT / "outputs/ssl_synthetic_localization"
OUTPUT = BASE / "real_background_v1/preparation"
SOURCE_FILES = (
    "ecg_experiment/ssl_synthetic_real_background.py",
    "ecg_experiment/ssl_synthetic_real_background_preparation.py",
    "scripts/data/prepare_ssl_synthetic_real_background.py",
    "tests/test_ssl_synthetic_real_background.py",
    "docs/ssl-synthetic-localization-real-background-study.md",
    "docs/ssl-synthetic-localization-real-background-clarifications.md",
    "docs/ssl-synthetic-localization-real-background-implementation.md",
    "configs/ssl_synthetic_real_background_v1.json",
    "ecg_experiment/ssl_synthetic_cache.py",
    "ecg_experiment/ssl_synthetic_evaluation.py",
    "ecg_experiment/ssl_synthetic_inputs.py",
    "ecg_experiment/ssl_synthetic_generator.py",
    "ecg_experiment/ssl_synthetic_preparation.py",
    "ecg_experiment/ssl_synthetic_partition.py",
    "ecg_experiment/ssl_synthetic_checksum_audit.py",
    "ecg_experiment/ssl_synthetic_readiness.py",
    "ecg_experiment/resample.py",
    "ecg_experiment/waveforms.py",
    "ecg_experiment/files.py",
    "ecg_experiment/paths.py",
    "pyproject.toml",
    "uv.lock",
)


def storage_forecast() -> dict[str, int]:
    """Compute exact five-array headers, all future payloads and the frozen metadata reserve."""
    specs = [
        ((192, 12, 2500), "float32"),
        ((768, 12, 2500), "bool"),
        ((768, 12, 2500), "bool"),
        ((768, 12, 2500), "float32"),
        ((768, 12, 2500), "float32"),
    ]
    headers = 0
    payload = 0
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


def _bound_receipt(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_bytes())
    for field in ("source_sha256", "input_sha256", "input_audit_source_sha256", "prerequisite_source_sha256"):
        verify_inference_hashes(report.get(field, {}))
    verify_inference_hashes(
        {to_stored(path.parent / name): value for name, value in report.get("output_sha256", {}).items()}
    )
    return report


def metadata_inputs() -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any], dict[str, Any]]:
    """Bind partition/audit/SSL identity metadata without reading the SSL waveform cache."""
    config = json.loads((ROOT / "configs/ssl_synthetic_real_background_v1.json").read_bytes())
    manifest, audited, checksums, binding = load_audited_inputs(ROOT)
    cache_summary_path, ssl_path = BASE / "real_cache_v1/summary.json", BASE / "real_cache_v1/selection.json"
    prior_path = BASE / "pilot_v1/counterfactual_audit_v2/summary.json"
    closure_path = BASE / "pilot_v1/final_audit_v1/reproduction_binding_closure.json"
    prior, closure = (_bound_receipt(path) for path in (prior_path, closure_path))
    if not prior["audit_completed"] or not closure["all_passed"] or closure["mismatch_count"]:
        raise ValueError("Original source-bound reproduction is not complete")
    expected = prior["input_sha256"]
    for path in (cache_summary_path, ssl_path, closure_path):
        if expected.get(to_stored(path)) != sha256_file(path):
            raise ValueError("SSL identity/reproduction metadata differs from frozen original bindings")
    cache = _bound_receipt(cache_summary_path)
    if sha256_file(ssl_path) != cache["output_sha256"]["selection.json"] or not cache["cache_gate_passed"]:
        raise ValueError("Original SSL selection metadata changed")
    ssl = json.loads(ssl_path.read_bytes())
    patients = {row["patient_id"] for row in ssl["records"]}
    if len(ssl["records"]) != 2048 or len(patients) != 2048:
        raise ValueError("Original SSL selected-patient denominator changed")
    rows = select_patients(manifest["records"], audited, patients, config)
    binding["input_sha256"].update(
        {
            to_stored(path): sha256_file(path)
            for path in (cache_summary_path, ssl_path, prior_path, closure_path)
        }
    )
    binding.update(
        prior_cumulative_cpu_seconds=prior["cumulative_cpu_seconds"],
        prior_cumulative_wall_seconds=closure["complete_local_wall_seconds"] + prior["wall_seconds"],
        ssl_cache_waveform_payload_opened=False,
    )
    return rows, checksums, binding, config


def prepare_background(row: dict[str, Any], checksums: dict[str, str]) -> tuple[np.ndarray, dict[str, Any]]:
    """Freshly verify/decode one selected validation ECG; reject reserved/protected roles first."""
    if row["role"] != "validation" or row["study_role"] not in ("calibration", "evaluation"):
        raise ValueError("Reserved TRAIN/protected/unselected roles must not open payloads")
    name = str(checked_relative(row["filename_hr"]))
    if not name.startswith("records500/") or "\\" in name:
        raise ValueError("Unexpected selected source path")
    raw = ROOT / "data/raw/ptb-xl/1.0.3"
    observed = {"payload_sha256": {}, "verified_bytes": 0}
    verify_files(raw, name, checksums, observed)
    if observed["payload_sha256"] != row["audit_payload_sha256"]:
        raise ValueError("Fresh selected bytes disagree with original input audit")
    validate_header(raw / (name + ".hea"), name)
    signal = resample_full(read_record(raw, name), 1, 2)
    if signal.shape != (12, 2500) or signal.dtype != np.float32 or not np.isfinite(signal).all():
        raise ValueError("Invalid processed selected signal")
    return signal, observed


def _qualify_cases(
    signal: np.ndarray,
    row: dict[str, Any],
    anchors: list[int],
    config: dict[str, Any],
    supports: np.ndarray,
    candidates: np.ndarray,
) -> list[dict[str, Any]]:
    cases = cases_for_background(row, anchors, array_sha256(signal), config)
    for case in cases:
        pair = render_case(signal, case)
        replay = render_case(signal, case)
        qualification = qualify_pair(pair, case, config)
        case["deterministic_replay_passed"] = all(
            pair[key].tobytes() == replay[key].tobytes()
            for key in ("baseline", "edited", "support", "candidate", "analytic_delta")
        )
        if any(
            pair[key].tobytes() != replay[key].tobytes()
            for key in ("baseline", "edited", "support", "candidate", "analytic_delta")
        ):
            qualification["failures"].append("deterministic_replay")
        for key in ("baseline", "edited", "support", "candidate"):
            case[key + "_sha256"] = array_sha256(pair[key])
        case["qualification"] = qualification
        supports[case["case_index"]], candidates[case["case_index"]] = pair["support"], pair["candidate"]
    for case in cases[:2]:
        control = cases[3 if case["nuisance"] else 2]
        case["matched_control_case_index"] = control["case_index"]
        if case["baseline_sha256"] != control["baseline_sha256"]:
            case["qualification"]["failures"].append("shared_nuisance_control_baseline")
    return cases


def _process(
    rows: list[dict[str, Any]],
    checksums: dict[str, str],
    config: dict[str, Any],
    cpu_start: float,
    prior_cpu: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    backgrounds = np.lib.format.open_memmap(
        OUTPUT / "backgrounds.npy", mode="w+", dtype=np.float32, shape=(192, 12, 2500)
    )
    backgrounds[:] = np.nan
    supports = np.lib.format.open_memmap(OUTPUT / "support.npy", mode="w+", dtype=bool, shape=(768, 12, 2500))
    candidates = np.lib.format.open_memmap(
        OUTPUT / "candidate.npy", mode="w+", dtype=bool, shape=(768, 12, 2500)
    )
    supports[:], candidates[:] = False, False
    metadata, cases = [], []
    for row in rows:
        if row["study_role"] == "train_reserved":
            continue
        info = dict(row, status="integrity_failed", anchors=[], payload_sha256={}, verified_bytes=0)
        try:
            if time.process_time() - cpu_start >= min(config["cpu_seconds_stage_ceiling"], 14400 - prior_cpu):
                raise ValueError("Frozen qualification CPU ceiling reached")
            signal, observed = prepare_background(row, checksums)
            info.update(observed)
            backgrounds[row["cache_index"]] = signal
            anchor_info = find_anchors(signal, config)
            info.update(anchor_info, status="processed", processed_background_sha256=array_sha256(signal))
            cases.extend(_qualify_cases(signal, row, anchor_info["anchors"], config, supports, candidates))
        except (OSError, ValueError, EOFError) as error:
            info.update(
                status="processing_failed", failure_type=type(error).__name__, failure_detail=str(error)
            )
            failed = cases_for_background(row, [], array_sha256(backgrounds[row["cache_index"]]), config)
            for case in failed:
                case["qualification"] = {
                    "failures": ["background_integrity_or_processing_failed"],
                    "metrics": {},
                }
            cases.extend(failed)
        metadata.append(info)
    for array in (backgrounds, supports, candidates):
        array.flush()
    return metadata, cases


def prepare() -> dict[str, Any]:
    """Freeze every selection before raw access, qualify all cases once and retain all outcomes."""
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise FileExistsError("Real-background preparation output must be new")
    wall, cpu = time.perf_counter(), time.process_time()
    rows, checksums, binding, config = metadata_inputs()
    selected_validation = [row for row in rows if row["study_role"] != "train_reserved"]
    if (
        [row["cache_index"] for row in selected_validation] != list(range(192))
        or any(row["subject_index"] != row["cache_index"] for row in selected_validation)
        or len(rows) != 692
        or sum(row["study_role"] == "train_reserved" for row in rows) != 500
    ):
        raise ValueError("Frozen complete selection/cache-index mapping changed")
    if time.process_time() - cpu >= min(
        config["cpu_seconds_stage_ceiling"], 14400 - binding["prior_cumulative_cpu_seconds"]
    ):
        raise ValueError("Cumulative qualification CPU allowance exhausted before payload access")
    sources = {name: sha256_file(ROOT / name) for name in SOURCE_FILES}
    forecast = storage_forecast()
    size_before = stream_bytes()
    if size_before + forecast["complete_study_new_bytes"] > 4 * 1024**3:
        raise ValueError("Whole-stream prospective storage ceiling would be exceeded")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    selection = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "records": rows,
        "selection_identity_sha256": sha256_json(rows),
        "input_sha256": binding["input_sha256"],
        "source_sha256": sources,
        "waveforms_opened_when_written": 0,
    }
    write_new_json(OUTPUT / "selection.json", selection)
    selection_hash = sha256_file(OUTPUT / "selection.json")
    metadata, cases = _process(rows, checksums, config, cpu, binding["prior_cumulative_cpu_seconds"])
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
    verify_hashes(sources | binding["input_sha256"] | binding["input_audit_source_sha256"])
    if sha256_file(OUTPUT / "selection.json") != selection_hash:
        raise ValueError("Prospective selection was modified after payload access")
    failures = Counter(reason for case in cases for reason in case["qualification"]["failures"])
    integrity = sum(info["status"] != "processed" for info in metadata)
    anchor_failures = sum(info.get("minimum_anchor_gate_passed") is not True for info in metadata)
    current_cpu, current_wall = time.process_time() - cpu, time.perf_counter() - wall
    cumulative_cpu = binding["prior_cumulative_cpu_seconds"] + current_cpu
    cumulative_wall = binding["prior_cumulative_wall_seconds"] + current_wall
    size_after = stream_bytes()
    remaining_probability_payload = 184_320_000
    storage_gate = (
        size_after + remaining_probability_payload + 2 * 128 + config["metadata_reserve_bytes"] < 4 * 1024**3
    )
    budget = (
        current_cpu < config["cpu_seconds_stage_ceiling"]
        and cumulative_cpu < 14400
        and cumulative_wall < 7200
    )
    report = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "qualification_gate_passed": not failures
        and integrity == 0
        and anchor_failures == 0
        and budget
        and storage_gate,
        "selected_patients": len(rows),
        "train_reserved_patients": 500,
        "train_reserved_waveforms_opened": 0,
        "calibration_patients": 64,
        "evaluation_patients": 128,
        "backgrounds": len(metadata),
        "cases": len(cases),
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
        "source_sha256": sources,
        **binding,
        "output_sha256": {
            name: sha256_file(OUTPUT / name)
            for name in ("backgrounds.npy", "support.npy", "candidate.npy", "selection.json", "cases.json")
        },
        "verified_raw_bytes": sum(info["verified_bytes"] for info in metadata),
        "wall_seconds": current_wall,
        "cpu_seconds": current_cpu,
        "stage_cpu_seconds": current_cpu,
        "cumulative_cpu_seconds": cumulative_cpu,
        "cumulative_wall_seconds": cumulative_wall,
        "storage_forecast": forecast,
        "stream_bytes_before_preparation": size_before,
        "stream_bytes_before_summary": size_after,
        "future_storage_gate_passed": storage_gate,
        "resource_budget_gate_passed": budget,
        "model_inferences": 0,
        "training_started": False,
        "performance_scored": False,
        "transfer_admitted": False,
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
