"""Source-bound context, native-grid regeneration and budgets for paired-background training."""

from __future__ import annotations

import json
import time
from collections import Counter
from numbers import Real
from typing import Any

import numpy as np

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file
from ecg_experiment.paths import to_stored
from ecg_experiment.ssl_synthetic_evaluation import _merge_hashes, _read_receipt, verify_inference_hashes
from ecg_experiment.ssl_synthetic_preparation import BASE, stream_bytes
from ecg_experiment.ssl_synthetic_real_background import array_sha256

STUDY = BASE / "paired_background_v1"
PREPARATION = STUDY / "preparation"
PROFILE = STUDY / "profile"
TRAINING = STUDY / "training"
EVALUATION = STUDY / "evaluation"
CONFIG_PATH = ROOT / "configs/ssl_synthetic_paired_background_v1.json"
SOURCE_NAMES = (
    "ecg_experiment/ssl_synthetic_paired_background_common.py",
    "tests/test_ssl_synthetic_paired_background_common.py",
    "docs/ssl-synthetic-localization-paired-background-study.md",
    "configs/ssl_synthetic_paired_background_v1.json",
)


def same_values(first: Any, second: Any) -> bool:
    """Compare complete nested receipt values at the predecessor's frozen 1e-12 tolerance."""
    if isinstance(first, dict) and isinstance(second, dict):
        return first.keys() == second.keys() and all(same_values(first[k], second[k]) for k in first)
    if isinstance(first, list) and isinstance(second, list):
        return len(first) == len(second) and all(
            same_values(a, b) for a, b in zip(first, second, strict=True)
        )
    if isinstance(first, bool) or isinstance(second, bool):
        return type(first) is type(second) and first == second
    if isinstance(first, Real) and isinstance(second, Real):
        return bool(np.isfinite(first) and np.isfinite(second) and abs(first - second) <= 1e-12)
    return first == second


def predecessor_integrity(audit: dict[str, Any], transfer: dict[str, Any], original: dict[str, Any]) -> None:
    """Require the full independent predecessor reproduction and unchanged original failed decision."""
    if not audit["combined_complete_pass"] or not audit["all_passed"] or audit["mismatch_count"]:
        raise ValueError("Independent predecessor reproduction is incomplete")
    if not transfer["evaluation_completed"] or transfer["original_advancement_passed"]:
        raise ValueError("Predecessor history changed")
    if original["decision"]["engineering_advancement_passed"]:
        raise ValueError("Original failed advancement changed")
    for key, value in audit["independently_recomputed_complete_summary"].items():
        if not same_values(value, transfer[key]):
            raise ValueError("Predecessor numerical reproduction differs: " + key)


def explicit_predecessor_values(reports: dict[str, Any]) -> None:
    """Compare source-bound calibration and original independently reproduced primary values."""
    transfer = reports["predecessor_evaluation"]
    if not same_values(
        transfer["calibration"]["thresholds"], reports["predecessor_calibration"]["thresholds"]
    ):
        raise ValueError("Predecessor calibration thresholds differ")
    original, audit = reports["original_evaluation"], reports["original_numeric_audit"]
    if not audit["all_passed"] or audit["mismatch_count"]:
        raise ValueError("Original numerical audit no longer passes")
    comparisons = (
        (
            original["profiles"]["B"]["s_minus_r_average_precision"],
            audit["profiles"]["B"]["s_minus_r_average_precision"],
        ),
        (original["decision"], audit["decision"]),
        (original["calibration"]["thresholds"], audit["thresholds"]),
    )
    if not all(same_values(first, second) for first, second in comparisons):
        raise ValueError("Original primary, decision or thresholds differ from numerical reproduction")


def _validate_counterparts(group: list[dict[str, Any]]) -> None:
    """Check both positive baseline identities against their corresponding control variant."""
    for positive in group[:2]:
        control = group[3 if positive["nuisance"] else 2]
        if positive.get("matched_control_case_index", control["case_index"]) != control["case_index"]:
            raise ValueError("Matched control alias differs from exact paired index")
        if positive["baseline_sha256"] != control["edited_sha256"]:
            raise ValueError("Positive baseline no longer matches its control")
        if positive["nuisance_parameters"] != control["nuisance_parameters"]:
            raise ValueError("Paired nuisance parameters differ")


def _validate_patient_group(
    group: list[dict[str, Any]], subject: int, local: int, profile: str, patients: set[str]
) -> None:
    """Verify complete patient sets, explicit coordinates and immutable matching counterparts."""
    if [c["variant"] for c in group] != ["focal", "repeated", "sham", "nuisance"]:
        raise ValueError("Patient must retain all four adjacent variants")
    if Counter((c["kind"], c["mode"]) for c in group) != Counter(
        {("positive", "focal"): 1, ("positive", "repeated"): 1, ("sham", "none"): 1, ("nuisance", "none"): 1}
    ):
        raise ValueError("Target/control variant labels changed")
    identities = {c["patient_id"] for c in group}
    if len(identities) != 1 or patients.intersection(identities):
        raise ValueError("Patient identity overlaps sets or roles")
    patients.update(identities)
    if any(
        c["cache_index"] != subject
        or c["subject_index"] != subject
        or c["role_subject_index"] != local
        or c["profile"] != profile
        for c in group
    ):
        raise ValueError("Background/subject/profile coordinate changed")
    if any(c.get("subject", subject) != subject for c in group):
        raise ValueError("Subject alias differs from global subject")
    if [c["target"] for c in group] != [True, True, False, False]:
        raise ValueError("Target flags changed")
    _validate_counterparts(group)


def validate_case_plan(cases: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Require all 692 patient sets and explicit global/native and within-role coordinates."""
    if len(cases) != 2768 or [c["case_index"] for c in cases] != list(range(2768)):
        raise ValueError("Expected every global case index 0 through 2767")
    roles = {}
    patients = set()
    for role, count, first, profile in (
        ("train", 500, 0, "A"),
        ("calibration", 64, 500, "A"),
        ("evaluation", 128, 564, "B"),
    ):
        rows = [c for c in cases if c["role"] == role]
        expected_indices = list(range(first * 4, (first + count) * 4))
        if [c["case_index"] for c in rows] != expected_indices:
            raise ValueError("Role/global case mapping changed")
        for local in range(count):
            _validate_patient_group(rows[4 * local : 4 * local + 4], first + local, local, profile, patients)
        roles[role] = expected_indices
    return roles


def admit_context(extra_source_names: tuple[str, ...] = ()) -> dict[str, Any]:
    """Freshly bind admitted data, predecessor reproduction, original initial states and source files."""
    paths = {
        "preparation": PREPARATION / "summary.json",
        "predecessor_audit": BASE / "real_background_v1/independent_audit_completion/summary.json",
        "predecessor_evaluation": BASE / "real_background_v1/evaluation/summary.json",
        "original_evaluation": BASE / "pilot_v1/evaluation/summary.json",
        "original_training": BASE / "pilot_v1/training/summary.json",
        "original_numeric_audit": BASE / "pilot_v1/final_audit_v1/summary.json",
        "predecessor_calibration": BASE / "real_background_v1/evaluation/calibration.json",
    }
    reports = {name: _read_receipt(path) for name, path in paths.items()}
    prep = reports["preparation"]
    if not prep["qualification_gate_passed"] or prep["failed_cases"] or prep["cases"] != 2768:
        raise ValueError("Complete paired-background qualification is required")
    predecessor_integrity(
        reports["predecessor_audit"], reports["predecessor_evaluation"], reports["original_evaluation"]
    )
    explicit_predecessor_values(reports)
    config = json.loads(CONFIG_PATH.read_text())
    sources = {name: sha256_file(ROOT / name) for name in SOURCE_NAMES + extra_source_names}
    inputs = {to_stored(path): sha256_file(path) for path in paths.values()}
    for name, report in reports.items():
        for field in ("source_sha256", "input_audit_source_sha256", "prerequisite_source_sha256"):
            _merge_hashes(sources, report.get(field, {}))
        _merge_hashes(inputs, report.get("input_sha256", {}))
        _merge_hashes(
            inputs,
            {
                to_stored(paths[name].parent / file): digest
                for file, digest in report.get("output_sha256", {}).items()
            },
        )
    initial_paths = {
        "initial.pt": reports["original_training"]["initial_checkpoint_sha256"],
        "ssl.pt": reports["original_training"]["ssl"]["checkpoint_sha256"],
    }
    _merge_hashes(
        inputs,
        {to_stored(BASE / "pilot_v1/training" / name): digest for name, digest in initial_paths.items()},
    )
    verify_inference_hashes(sources | inputs)
    container = json.loads((PREPARATION / "cases.json").read_text())
    cases = container["cases"]
    roles = validate_case_plan(cases)
    backgrounds = np.load(PREPARATION / "backgrounds.npy", mmap_mode="r", allow_pickle=False)
    support = np.load(PREPARATION / "support.npy", mmap_mode="r", allow_pickle=False)
    candidate = np.load(PREPARATION / "candidate.npy", mmap_mode="r", allow_pickle=False)
    if backgrounds.shape != (692, 12, 2500) or backgrounds.dtype != np.float32:
        raise ValueError("Background cache geometry changed")
    if any(x.shape != (2768, 12, 2500) or x.dtype != bool for x in (support, candidate)):
        raise ValueError("Full native mask geometry changed")
    if stream_bytes() >= config["cache_bytes_ceiling"]:
        raise ValueError("Whole-stream storage ceiling reached")
    return {
        **reports,
        "config": config,
        "sources": sources,
        "inputs": inputs,
        "cases": cases,
        "roles": roles,
        "backgrounds": backgrounds,
        "support": support,
        "candidate": candidate,
        "regeneration_checks": {},
        "prior_cpu_seconds": reports["predecessor_audit"]["cumulative_cpu_seconds"],
        "prior_gpu_wall_seconds": reports["predecessor_evaluation"]["global_gpu_wall_seconds"],
        "stage_preparation_cpu_seconds": prep["cpu_seconds"],
    }


def render_checked(context: dict[str, Any], index: int) -> dict[str, Any]:
    """Regenerate one admitted global case and verify every observed/target byte before use."""
    from ecg_experiment.ssl_synthetic_paired_background_data import render_case

    if not isinstance(index, (int, np.integer)) or not 0 <= index < len(context["cases"]):
        raise ValueError("Invalid global case coordinate")
    case = context["cases"][index]
    if case["case_index"] != index:
        raise ValueError("Case coordinate differs from array coordinate")
    pair = render_case(context["backgrounds"][case["cache_index"]], case)
    hashes = {name: array_sha256(pair[name]) for name in ("edited", "baseline", "support", "candidate")}
    if any(hashes[name] != case[name + "_sha256"] for name in hashes):
        raise ValueError("Qualified rendered input/target bytes changed")
    if not np.array_equal(pair["support"], context["support"][index]) or not np.array_equal(
        pair["candidate"], context["candidate"][index]
    ):
        raise ValueError("Cached support/candidate mask changed")
    context["regeneration_checks"][str(index)] = hashes
    return pair


def check_budget(context: dict[str, Any], clock: dict[str, float], reserve_seconds: float = 0.0) -> None:
    """Enforce stage/cumulative live wall and CPU limits including prior phases and reserved work."""
    if reserve_seconds < 0:
        raise ValueError("Budget reserve cannot be negative")
    config = context["config"]
    wall = clock.get("prior_stage_wall", 0) + time.perf_counter() - clock["wall"] + reserve_seconds
    cpu = clock.get("prior_stage_cpu", 0) + time.process_time() - clock["cpu"] + reserve_seconds
    if (
        wall >= config["gpu_seconds_stage_ceiling"]
        or context["prior_gpu_wall_seconds"] + wall >= config["gpu_seconds_ceiling"]
    ):
        raise TimeoutError("Stage/cumulative GPU-wall allowance exhausted")
    if (
        cpu >= config["cpu_seconds_stage_ceiling"]
        or context["prior_cpu_seconds"] + cpu >= config["cpu_seconds_ceiling"]
    ):
        raise TimeoutError("Stage/cumulative CPU allowance exhausted")


def storage_projection() -> dict[str, int]:
    """Charge five exact arrays and one shared metadata/checkpoint reserve without duplication."""
    config = json.loads(CONFIG_PATH.read_text())
    existing = sum(p.stat().st_size for p in STUDY.rglob("*") if p.is_file()) if STUDY.exists() else 0
    total = config["array_payload_bytes"] + 5 * 128 + config["metadata_reserve_bytes"]
    current = stream_bytes()
    return {
        "complete_study_bytes": total,
        "existing_study_bytes": existing,
        "additional_bytes": max(0, total - existing),
        "current_stream_bytes": current,
        "projected_whole_stream_bytes": current + max(0, total - existing),
    }
