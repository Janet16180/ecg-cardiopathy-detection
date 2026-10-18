"""Scoped provenance and explicit mixed-bank context for the localization-loss factorial."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file, sha256_json
from ecg_experiment.paths import to_stored
from ecg_experiment.ssl_synthetic_counterfactual_factorial_data import (
    background_row,
    new_case_row,
    render_case,
)
from ecg_experiment.ssl_synthetic_preparation import stream_bytes
from ecg_experiment.ssl_synthetic_real_background import array_sha256

BASE = ROOT / "outputs/ssl_synthetic_localization"
PARENT = BASE / "paired_background_v1"
STUDY = BASE / "counterfactual_factorial_v1"
PREPARATION, PROFILE, TRAINING, EVALUATION = (
    STUDY / part for part in ("preparation", "profile", "training", "evaluation")
)
CONFIG_PATH = ROOT / "configs/ssl_synthetic_counterfactual_factorial_v1.json"
SOURCE_NAMES = (
    "ecg_experiment/ssl_synthetic_counterfactual_factorial_data.py",
    "ecg_experiment/ssl_synthetic_counterfactual_factorial_common.py",
    "ecg_experiment/ssl_synthetic_counterfactual_factorial_preparation.py",
    "scripts/data/prepare_ssl_synthetic_counterfactual_factorial.py",
    "tests/test_ssl_synthetic_counterfactual_factorial_data.py",
    "docs/ssl-synthetic-localization-counterfactual-factorial.md",
    "configs/ssl_synthetic_counterfactual_factorial_v1.json",
    "ecg_experiment/ssl_synthetic_paired_background_data.py",
    "ecg_experiment/ssl_synthetic_real_background.py",
    "ecg_experiment/ssl_synthetic_inputs.py",
    "ecg_experiment/ssl_synthetic_checksum_audit.py",
    "ecg_experiment/ssl_synthetic_generator.py",
    "ecg_experiment/ssl_synthetic_preparation.py",
    "ecg_experiment/ssl_synthetic_readiness.py",
    "ecg_experiment/__init__.py",
    "ecg_experiment/resample.py",
    "ecg_experiment/waveforms.py",
    "ecg_experiment/files.py",
    "ecg_experiment/paths.py",
    "pyproject.toml",
    "uv.lock",
)
ANCHORS = {
    to_stored(
        PARENT / "training_audit/summary.json"
    ): "55b50f8c9705aeacfce7e19334a1b987a0ebb527b3924f2ff362eb099a10359a",
    to_stored(
        PARENT / "metrics_audit/summary.json"
    ): "464a438c0a3e48ed7a60bf2f7ea2195fcb23b26369c45394c00fa8aaaa118976",
    to_stored(
        PARENT / "diagnostic_v2/audit/summary.json"
    ): "3180a8ea87a626294a629302a842184cda4f7ac1f438bfcef631e3199682e3c8",
}
ORIGINAL_CHECKPOINTS = {to_stored(BASE / "pilot_v1/training" / name) for name in ("initial.pt", "ssl.pt")}
PREDECESSOR_CHECKPOINTS = {to_stored(PARENT / "training" / (arm + ".pt")) for arm in ("R", "S")}
CHECKSUM_PATH = "data/raw/ptb-xl/1.0.3/SHA256SUMS.txt"


def merge(target: dict, extra: dict) -> None:
    """Require one exact digest for each declared artifact."""
    for key, value in extra.items():
        if key in target and target[key] != value:
            raise ValueError("Conflicting source-bound artifact digest")
        target[key] = value


def same_values(expected: Any, actual: Any) -> None:
    """Compare all nested predecessor scientific values at unchanged absolute tolerance1e-12."""
    pending = [(expected, actual)]
    while pending:
        left, right = pending.pop()
        if isinstance(left, dict):
            if not isinstance(right, dict) or left.keys() != right.keys():
                raise ValueError("Predecessor scientific structure changed")
            pending.extend((left[key], right[key]) for key in left)
        elif isinstance(left, list):
            if len(left) != len(right):
                raise ValueError("Predecessor sequence changed")
            pending.extend(zip(left, right, strict=True))
        elif isinstance(left, float):
            if not np.isfinite(left) or not np.isfinite(right) or abs(left - right) > 1e-12:
                raise ValueError("Predecessor numerical reproduction changed")
        elif type(left) is not type(right) or left != right:
            raise ValueError("Predecessor value changed")


def verify_bindings(sources: dict, inputs: dict, authorities: dict, model_stage: bool = False) -> dict:
    """Hash only current/needed payloads; defer every unused ancestor array/checkpoint to audits."""
    exceptions = {}
    declared = dict(sources)
    merge(declared, inputs)
    for name, expected in declared.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError("Invalid stored artifact path")
        current = name.startswith(to_stored(STUDY) + "/")
        checkpoint = model_stage and name in ORIGINAL_CHECKPOINTS | PREDECESSOR_CHECKPOINTS
        permitted = (
            current
            or checkpoint
            or name == CHECKSUM_PATH
            or (
                path.suffix in {".json", ".jsonl", ".py", ".md", ".toml", ".lock", ".txt"}
                and not name.startswith("data/raw/")
            )
        )
        if permitted:
            if sha256_file(ROOT / path) != expected:
                raise ValueError("Consumed source/input artifact changed")
        else:
            if authorities.get(name) != expected:
                raise ValueError("Unused payload lacks exact passed-audit digest authority")
            exceptions[name] = {
                "expected_sha256": expected,
                "authority": "SHA-anchored passed parent audits",
                "whole_payload_read": False,
            }
    return exceptions


def source_bindings(names: tuple[str, ...], authorities: dict) -> dict:
    """Freshly bind new sources while requiring every reused helper to retain its anchored digest."""
    sources = {}
    for name in names:
        digest = sha256_file(ROOT / name)
        if name in authorities:
            if digest != authorities[name]:
                raise ValueError("Frozen reused source changed")
        elif name not in SOURCE_NAMES[:7] and "counterfactual_factorial" not in name:
            raise ValueError("Reused source lacks passed-audit authority")
        sources[name] = digest
    return sources


def predecessor_metadata(extra_source_names: tuple[str, ...] = ()) -> dict:
    """Bind only consumed metadata plus three passed audit catalogs, without ancestor waveform reads."""
    audits, authorities = {}, {}
    for name, digest in ANCHORS.items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError("Passed predecessor audit anchor changed")
        report = json.loads((ROOT / name).read_bytes())
        if not report["all_passed"] or report["mismatch_count"]:
            raise ValueError("Passed predecessor audit required")
        audits[name] = report
        merge(authorities, report["input_sha256"])
        merge(authorities, report["source_sha256"])
        for field in (
            "metadata_only_scope_exceptions",
            "deferred_payload_scope_exceptions",
            "waveform_digest_scope_exceptions",
        ):
            merge(
                authorities,
                {
                    key: value if isinstance(value, str) else value["expected_sha256"]
                    for key, value in report.get(field, {}).items()
                },
            )
    names = {
        "partition": "data/processed/ssl_synthetic_localization_v1/partition.json",
        "eligibility": to_stored(BASE / "input_audit_v1/eligibility.json"),
        "original_training": to_stored(BASE / "pilot_v1/training/summary.json"),
        "original_evaluation": to_stored(BASE / "pilot_v1/evaluation/summary.json"),
        "predecessor_training": to_stored(PARENT / "training/summary.json"),
        "predecessor_evaluation": to_stored(PARENT / "evaluation/summary.json"),
        "predecessor_cases": to_stored(PARENT / "preparation/cases.json"),
        "predecessor_selection": to_stored(PARENT / "preparation/selection.json"),
        "first_selection": to_stored(BASE / "real_background_v1/preparation/selection.json"),
        "ssl_selection": to_stored(BASE / "real_cache_v1/selection.json"),
        "diagnostic": to_stored(PARENT / "diagnostic_v2/summary.json"),
    }
    inputs = dict(ANCHORS)
    for name in (*names.values(), CHECKSUM_PATH):
        if name not in authorities:
            raise ValueError("Consumed predecessor metadata has no passed-audit authority")
        inputs[name] = authorities[name]
    sources = source_bindings(SOURCE_NAMES + extra_source_names, authorities)
    verify_bindings(sources, inputs, authorities)
    reports = {key: json.loads((ROOT / name).read_bytes()) for key, name in names.items()}
    numeric = audits[to_stored(PARENT / "metrics_audit/summary.json")]
    for key, value in numeric["independently_recomputed_complete_summary"].items():
        same_values(value, reports["predecessor_evaluation"][key])
    same_values(
        numeric["independently_recomputed_thresholds"],
        reports["predecessor_evaluation"]["calibration"]["thresholds"],
    )
    latest = audits[to_stored(PARENT / "diagnostic_v2/audit/summary.json")]
    for key, value in latest["independently_recomputed_complete_summary"].items():
        same_values(value, reports["diagnostic"][key])
    if (
        reports["predecessor_evaluation"]["decision"]["engineering_replication_admitted"]
        or reports["original_evaluation"]["decision"]["engineering_advancement_passed"]
        or latest["parent_full_advancement_passed"]
    ):
        raise ValueError("Frozen failed predecessor decision changed")
    return {
        **reports,
        "authorities": authorities,
        "sources": sources,
        "inputs": inputs,
        "config": json.loads(CONFIG_PATH.read_bytes()),
        "prior_cpu_seconds": latest["cumulative_cpu_seconds"],
        "prior_gpu_wall_seconds": reports["predecessor_evaluation"]["global_gpu_wall_seconds"],
        "predecessor_reproduction": {
            "summary_blocks": 8,
            "all_nine_gates_match": True,
            "thresholds_match": True,
            "diagnostic_blocks": 7,
            "parent_full_advancement_passed": False,
        },
    }


def storage_projection() -> dict[str, int]:
    """Charge seven arrays/reserve once and count actual current study files without duplication."""
    config = json.loads(CONFIG_PATH.read_bytes())
    existing = (
        sum(path.stat().st_size for path in STUDY.rglob("*") if path.is_file()) if STUDY.exists() else 0
    )
    total = config["complete_study_new_bytes"]
    current = stream_bytes()
    return {
        "complete_study_bytes": total,
        "existing_study_bytes": existing,
        "additional_bytes": max(0, total - existing),
        "current_stream_bytes": current,
        "projected_whole_stream_bytes": current + max(0, total - existing),
    }


def verify_context(context: dict) -> None:
    """Reverify bound active files; used TRAIN rows/cases are separately checked before every render."""
    context["scope_exceptions"] = verify_bindings(
        context["sources"], context["inputs"], context["authorities"], True
    )


def admit_context(extra_source_names: tuple[str, ...] = ()) -> dict:
    """Load explicit old-TRAIN/new-validation bank views after complete new qualification passes."""
    context = predecessor_metadata(extra_source_names)
    prep_path = PREPARATION / "summary.json"
    prep = json.loads(prep_path.read_bytes())
    if not prep["qualification_gate_passed"] or prep["failed_cases"] or prep["cases"] != 768:
        raise ValueError("All new validation cases must qualify")
    merge(context["inputs"], {to_stored(prep_path): sha256_file(prep_path)})
    merge(context["sources"], prep["source_sha256"])
    merge(context["inputs"], prep["input_sha256"])
    merge(
        context["inputs"],
        {to_stored(PREPARATION / name): digest for name, digest in prep["output_sha256"].items()},
    )
    for name in ORIGINAL_CHECKPOINTS | PREDECESSOR_CHECKPOINTS:
        context["inputs"][name] = context["authorities"][name]
    for name in ("backgrounds.npy", "support.npy", "candidate.npy"):
        stored = to_stored(PARENT / "preparation" / name)
        context["inputs"][stored] = context["authorities"][stored]
    verify_context(context)
    new = json.loads((PREPARATION / "cases.json").read_bytes())
    train = context["predecessor_cases"]["cases"][:2000]
    if sha256_json(train) != prep["reused_train_cases_sha256"]:
        raise ValueError("Original TRAIN descriptors changed")
    cases = train + new["cases"]
    if [case["case_index"] for case in cases] != list(range(2768)):
        raise ValueError("Mixed global case grid changed")
    context.update(
        cases=cases,
        roles={
            "train": list(range(2000)),
            "calibration": list(range(2000, 2256)),
            "evaluation": list(range(2256, 2768)),
        },
        train_backgrounds=np.load(PARENT / "preparation/backgrounds.npy", mmap_mode="r", allow_pickle=False),
        train_support=np.load(PARENT / "preparation/support.npy", mmap_mode="r", allow_pickle=False),
        train_candidate=np.load(PARENT / "preparation/candidate.npy", mmap_mode="r", allow_pickle=False),
        new_backgrounds=np.load(PREPARATION / "backgrounds.npy", mmap_mode="r", allow_pickle=False),
        new_support=np.load(PREPARATION / "support.npy", mmap_mode="r", allow_pickle=False),
        new_candidate=np.load(PREPARATION / "candidate.npy", mmap_mode="r", allow_pickle=False),
        stage_preparation_cpu_seconds=prep["cpu_seconds"],
        regeneration_checks={},
        scope_exceptions=context["scope_exceptions"],
    )
    for name, shape, dtype in (
        ("train_backgrounds", (692, 12, 2500), np.float32),
        ("new_backgrounds", (192, 12, 2500), np.float32),
        ("train_support", (2768, 12, 2500), bool),
        ("train_candidate", (2768, 12, 2500), bool),
        ("new_support", (768, 12, 2500), bool),
        ("new_candidate", (768, 12, 2500), bool),
    ):
        if context[name].shape != shape or context[name].dtype != dtype:
            raise ValueError("Mixed-bank shape/dtype changed")
    return context


def _case_index(index: int) -> int:
    if not isinstance(index, (int, np.integer)) or not 0 <= index < 2768:
        raise ValueError("Invalid global case index")
    return int(index)


def support_for_case(context: dict, index: int) -> np.ndarray:
    """Route old TRAIN or new validation support without constructing a duplicate mixed array."""
    index = _case_index(index)
    return context["train_support"][index] if index < 2000 else context["new_support"][new_case_row(index)]


def candidate_for_case(context: dict, index: int) -> np.ndarray:
    """Route old TRAIN or new validation candidate mask with the same explicit global offset."""
    index = _case_index(index)
    return (
        context["train_candidate"][index] if index < 2000 else context["new_candidate"][new_case_row(index)]
    )


def render_checked(context: dict, index: int) -> dict:
    """Verify the actually used background row and all four regenerated arrays before model input."""
    index = _case_index(index)
    case = context["cases"][index]
    if case["case_index"] != index:
        raise ValueError("Case coordinate changed")
    if index < 2000:
        if not 0 <= case["cache_index"] < 500:
            raise ValueError("TRAIN render attempted unused old validation waveform")
        signal = context["train_backgrounds"][case["cache_index"]]
    else:
        signal = context["new_backgrounds"][background_row(case["cache_index"])]
    if array_sha256(signal) != case["processed_background_sha256"]:
        raise ValueError("Active background row changed")
    pair = render_case(signal, case)
    hashes = {key: array_sha256(pair[key]) for key in ("edited", "baseline", "support", "candidate")}
    if any(hashes[key] != case[key + "_sha256"] for key in hashes):
        raise ValueError("Active rendered case changed")
    if not np.array_equal(pair["support"], support_for_case(context, index)) or not np.array_equal(
        pair["candidate"], candidate_for_case(context, index)
    ):
        raise ValueError("Active support/candidate row changed")
    context["regeneration_checks"][str(index)] = hashes
    return pair


def check_budget(context: dict, clock: dict, reserve_seconds: float = 0.0) -> None:
    """Enforce exact prior plus current stage wall/CPU costs and whole-stream storage."""
    if reserve_seconds < 0:
        raise ValueError("Negative reserve")
    config = context["config"]
    wall = clock.get("prior_stage_wall", 0) + time.perf_counter() - clock["wall"] + reserve_seconds
    cpu = clock.get("prior_stage_cpu", 0) + time.process_time() - clock["cpu"] + reserve_seconds
    if (
        wall >= config["gpu_seconds_stage_ceiling"]
        or context["prior_gpu_wall_seconds"] + wall >= config["gpu_seconds_ceiling"]
    ):
        raise TimeoutError("Factorial stage/cumulative wall budget exhausted")
    if (
        cpu >= config["cpu_seconds_stage_ceiling"]
        or context["prior_cpu_seconds"] + cpu >= config["cpu_seconds_ceiling"]
    ):
        raise TimeoutError("Factorial stage/cumulative CPU budget exhausted")
    if stream_bytes() >= config["cache_bytes_ceiling"]:
        raise ValueError("Whole-stream storage ceiling reached")
