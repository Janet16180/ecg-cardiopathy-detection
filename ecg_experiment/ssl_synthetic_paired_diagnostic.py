"""Diagnose frozen paired-study maps without opening waveforms, checkpoints or models."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file
from ecg_experiment.paths import to_stored
from ecg_experiment.ssl_synthetic_metrics import (
    control_threshold,
    paired_subject_difference,
    positive_map_metrics,
)

BASE = ROOT / "outputs/ssl_synthetic_localization"
STUDY = BASE / "paired_background_v1"
DESTINATION = STUDY / "diagnostic_v1"
CONFIG_PATH = ROOT / "configs/ssl_synthetic_paired_diagnostic_v1.json"
SOURCE_NAMES = (
    "ecg_experiment/ssl_synthetic_paired_diagnostic.py",
    "scripts/reports/diagnose_ssl_synthetic_paired_background.py",
    "tests/test_ssl_synthetic_paired_diagnostic.py",
    "configs/ssl_synthetic_paired_diagnostic_v1.json",
    "docs/ssl-synthetic-localization-paired-diagnostic.md",
    "docs/ssl-synthetic-localization-paired-diagnostic-admission.md",
    "ecg_experiment/ssl_synthetic_metrics.py",
    "ecg_experiment/intervals.py",
    "ecg_experiment/files.py",
    "ecg_experiment/paths.py",
)
ARMS = ("R", "S")
ROLES = ("calibration", "evaluation")
METRICS = tuple(
    f"{phase}_mean_{region}" for phase in ("baseline", "edited", "change") for region in "TUO"
) + (
    "baseline_T_minus_U",
    "edited_T_minus_U",
    "change_T_minus_U",
    "full_map_mean_absolute_change",
    "full_map_max_absolute_change",
    "baseline_site_average_precision",
    "edited_average_precision",
    "average_precision_change",
)


def _json(path: Path) -> Any:
    return json.loads(path.read_text())


def _write(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _merge(target: dict[str, str], incoming: dict[str, str]) -> None:
    for name, digest in incoming.items():
        if name in target and target[name] != digest:
            raise ValueError("Conflicting frozen digest declaration")
        target[name] = digest


def _equal_metadata(first: Any, second: Any) -> None:
    pending = [(first, second)]
    while pending:
        left, right = pending.pop()
        if isinstance(left, dict):
            if not isinstance(right, dict) or left.keys() != right.keys():
                raise ValueError("Parent summary structure changed")
            pending.extend((left[k], right[k]) for k in left)
        elif isinstance(left, list):
            if not isinstance(right, list) or len(left) != len(right):
                raise ValueError("Parent summary sequence changed")
            pending.extend(zip(left, right, strict=True))
        elif isinstance(left, bool) or left is None or isinstance(left, str):
            if type(left) is not type(right) or left != right:
                raise ValueError("Parent summary metadata changed")
        elif not np.isfinite(left) or not np.isfinite(right) or abs(left - right) > 1e-12:
            raise ValueError("Parent summary numerical reproduction differs")


def payload_allowed(name: str, allowed_arrays: list[str]) -> bool:
    """Decide access BEFORE hashing; only four named arrays and permitted metadata text may open."""
    path = PurePosixPath(name)
    if not path.parts or path.is_absolute() or ".." in path.parts:
        raise ValueError("Receipt path must stay repository-relative")
    if name in allowed_arrays:
        return True
    if path.parts[0] == "data":
        return False
    return path.suffix in {".json", ".jsonl", ".py", ".md", ".toml", ".lock", ".yaml", ".yml", ".txt"}


def verify_bindings(
    bindings: dict[str, str],
    authorities: dict[str, str],
    allowed_arrays: list[str],
    guard: Callable[[], None] | None = None,
) -> dict[str, dict]:
    """Freshly hash permitted inputs and defer forbidden payloads only to exact anchored authority."""
    exceptions = {}
    for name, digest in bindings.items():
        if guard is not None:
            guard()
        if payload_allowed(name, allowed_arrays):
            if sha256_file(ROOT / name) != digest:
                raise ValueError("Permitted source/input hash changed")
        else:
            if authorities.get(name) != digest:
                raise ValueError("Forbidden payload has no matching frozen authority")
            exceptions[name] = {
                "expected_sha256": digest,
                "authority": "anchored parent audit catalogs",
                "payload_read": False,
            }
    return exceptions


def _stream_bytes(path: Path = BASE) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _guard(context: dict, wall: float, cpu: float) -> None:
    config = context["config"]
    elapsed_wall, elapsed_cpu = time.perf_counter() - wall, time.process_time() - cpu
    reserve = config["failure_reporting_reserve_seconds"]
    if max(elapsed_wall, elapsed_cpu) + reserve >= config["cpu_seconds_run_ceiling"]:
        raise TimeoutError("Diagnostic run allowance exhausted")
    if context["prior_cpu"] + elapsed_cpu + reserve >= config["global_cpu_ceiling"]:
        raise TimeoutError("Cumulative CPU allowance exhausted")
    if context["prior_stage_cpu"] + elapsed_cpu + reserve >= config["parent_stage_cpu_ceiling"]:
        raise TimeoutError("Paired-study CPU allowance exhausted")
    new_bytes = _stream_bytes(DESTINATION) if DESTINATION.exists() else 0
    if (
        new_bytes + 1024**2 >= config["new_output_bytes_ceiling"]
        or _stream_bytes() + 1024**2 >= config["cache_bytes_ceiling"]
    ):
        raise ValueError("Diagnostic or entire-stream output allowance exhausted")


def _anchored_reports(config: dict, guard: Callable[[], None] | None) -> tuple[dict, dict, dict]:
    paths = {
        "evaluation": STUDY / "evaluation/summary.json",
        "training_audit": STUDY / "training_audit/summary.json",
        "metrics_audit": STUDY / "metrics_audit/summary.json",
    }
    anchors = {to_stored(path): config[f"parent_{name}_sha256"] for name, path in paths.items()}
    for name, digest in anchors.items():
        if guard is not None:
            guard()
        if sha256_file(ROOT / name) != digest:
            raise ValueError("Anchored parent receipt changed")
    reports = {name: _json(path) for name, path in paths.items()}
    evaluation = reports["evaluation"]
    for name in ("training_audit", "metrics_audit"):
        audit = reports[name]
        if not audit["all_passed"] or audit["mismatch_count"]:
            raise ValueError("A complete passed parent audit is required")
        if (
            audit["input_sha256"].get(to_stored(paths["evaluation"]))
            != anchors[to_stored(paths["evaluation"])]
        ):
            raise ValueError("Parent audit does not bind exact evaluation")
    audit = reports["metrics_audit"]
    reproduced = audit["independently_recomputed_complete_summary"]
    if len(reproduced) != 8:
        raise ValueError("Expected all eight parent summary blocks")
    for key, value in reproduced.items():
        _equal_metadata(value, evaluation[key])
    _equal_metadata(audit["independently_recomputed_thresholds"], evaluation["calibration"]["thresholds"])
    if (
        evaluation["decision"]["engineering_replication_admitted"]
        or evaluation["original_advancement_passed"]
    ):
        raise ValueError("A failed predecessor verdict changed")
    return reports, paths, anchors


def _authority_catalog(reports: dict) -> dict:
    authorities = {}
    for name in ("training_audit", "metrics_audit"):
        _merge(authorities, reports[name]["input_sha256"])
    _merge(authorities, reports["metrics_audit"]["waveform_digest_scope_exceptions"])
    _merge(
        authorities,
        {
            name: value["expected_sha256"]
            for name, value in reports["training_audit"]["metadata_only_scope_exceptions"].items()
        },
    )
    return authorities


def _native_arrays(
    config: dict, cases: list[dict], evaluation: dict, guard: Callable[[], None] | None
) -> tuple:
    maps = {
        arm: np.load(STUDY / f"evaluation/probabilities_{arm}.npy", mmap_mode="r", allow_pickle=False)
        for arm in ARMS
    }
    support = np.load(STUDY / "preparation/support.npy", mmap_mode="r", allow_pickle=False)
    candidate = np.load(STUDY / "preparation/candidate.npy", mmap_mode="r", allow_pickle=False)
    if len(cases) != 2768 or [c["case_index"] for c in cases] != list(range(2768)):
        raise ValueError("Global case grid changed")
    if any(x.shape != tuple(config["mask_shape"]) or x.dtype != bool for x in (support, candidate)):
        raise ValueError("Native mask geometry changed")
    if any(
        x.shape != tuple(config["probability_shape"]) or x.dtype != np.float32 or not np.isfinite(x).all()
        for x in maps.values()
    ):
        raise ValueError("Native finite float32 maps are required")
    groups = case_groups(cases)
    calibration = _json(STUDY / "evaluation/calibration.json")
    if sha256_file(STUDY / "evaluation/calibration.json") != evaluation["calibration_sha256"]:
        raise ValueError("Frozen calibration bytes changed")
    for arm in ARMS:
        if (
            hashlib.sha256(np.ascontiguousarray(maps[arm][:256]).tobytes()).hexdigest()
            != calibration["calibration_slice_sha256"][arm]
        ):
            raise ValueError("Frozen calibration map slice changed")
        control = groups["calibration"]
        sham = np.array([maps[arm][g[2]["case_index"] - 2000].max() for g in control])
        nuisance = np.array([maps[arm][g[3]["case_index"] - 2000].max() for g in control])
        if control_threshold(sham, nuisance) != calibration["thresholds"][arm]:
            raise ValueError("Frozen threshold integrity comparison failed")
        if np.sum(np.maximum(sham, nuisance) > calibration["thresholds"][arm]) > 3:
            raise ValueError("Strict fitted-data joint flags exceed three")
    _equal_metadata(calibration, evaluation["calibration"])
    if guard is not None:
        guard()
    return maps, support, candidate, groups, calibration


def admit_diagnostic(guard: Callable[[], None] | None = None) -> dict:
    """Bind passed parent audits and exact eight-block reproduction before saved-map analysis."""
    config = _json(CONFIG_PATH)
    reports, paths, anchors = _anchored_reports(config, guard)
    evaluation, audit = reports["evaluation"], reports["metrics_audit"]
    authorities = _authority_catalog(reports)
    sources = {name: sha256_file(ROOT / name) for name in SOURCE_NAMES}
    inputs = dict(anchors)
    for name, report in reports.items():
        _merge(sources, report["source_sha256"])
        _merge(inputs, report["input_sha256"])
        _merge(
            inputs,
            {
                to_stored(paths[name].parent / file): digest
                for file, digest in report.get("output_sha256", {}).items()
            },
        )
    bindings = dict(sources)
    _merge(bindings, inputs)
    exceptions = verify_bindings(bindings, authorities, config["allowed_array_payloads"], guard)
    required = (
        "preparation/cases.json",
        "training/summary.json",
        "training/progress.jsonl",
        "evaluation/calibration.json",
        "evaluation/local_case_metrics.json",
    )
    if any(to_stored(STUDY / name) not in inputs for name in required):
        raise ValueError("Consumed metadata has no verified frozen binding")
    for name in config["allowed_array_payloads"]:
        if name not in inputs:
            raise ValueError("Consumed array has no frozen input binding")
    cases = _json(STUDY / "preparation/cases.json")["cases"]
    maps, support, candidate, groups, calibration = _native_arrays(config, cases, evaluation, guard)
    training = _json(STUDY / "training/summary.json")
    logs = [json.loads(line) for line in (STUDY / "training/progress.jsonl").read_text().splitlines()]
    return {
        "config": config,
        "sources": sources,
        "inputs": inputs,
        "authorities": authorities,
        "scope_exceptions": exceptions,
        "bindings": bindings,
        "maps": maps,
        "support": support,
        "candidate": candidate,
        "cases": cases,
        "groups": groups,
        "thresholds": calibration["thresholds"],
        "training": training,
        "logs": logs,
        "parent_case_metrics": _json(STUDY / "evaluation/local_case_metrics.json"),
        "prior_cpu": audit["cumulative_cpu_seconds"],
        "prior_stage_cpu": audit["stage_cpu_seconds_including_both_audits"],
        "reproduction": {
            "summary_blocks": 8,
            "all_nine_gates_match": True,
            "thresholds_match": True,
            "full_advancement_passed": False,
        },
    }


def case_groups(cases: list[dict]) -> dict[str, list[list[dict]]]:
    """Validate separate role/global coordinates and exact positive-to-control metadata identities."""
    groups = {}
    identities = {c["patient_id"] for c in cases[:2000]}
    for role, count, first in (("calibration", 64, 500), ("evaluation", 128, 564)):
        groups[role] = []
        for subject in range(count):
            group = cases[4 * (first + subject) : 4 * (first + subject + 1)]
            if [c["variant"] for c in group] != ["focal", "repeated", "sham", "nuisance"]:
                raise ValueError("Patient variants changed")
            if any(
                c["role"] != role
                or c["role_subject_index"] != subject
                or c["cache_index"] != first + subject
                or c["subject_index"] != first + subject
                for c in group
            ):
                raise ValueError("Within-role/global patient coordinates changed")
            ids = {c["patient_id"] for c in group}
            if len(ids) != 1 or ids.intersection(identities):
                raise ValueError("Patient identity overlaps groups")
            identities.update(ids)
            for positive in group[:2]:
                control = group[3 if positive["nuisance"] else 2]
                if (
                    positive["matched_control_case_index"] != control["case_index"]
                    or positive["baseline_sha256"] != control["edited_sha256"]
                    or positive["nuisance_parameters"] != control["nuisance_parameters"]
                ):
                    raise ValueError("Frozen matched-control identity changed")
            groups[role].append(group)
    return groups


def descriptive(values: np.ndarray) -> dict:
    """Return fixed finite descriptive reductions, with explicit null statistics for an empty group."""
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return dict.fromkeys(("mean", "median", "minimum", "maximum", "q05", "q95")) | {"count": 0}
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite diagnostic values")
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "q05": float(np.quantile(values, 0.05, method="linear")),
        "q95": float(np.quantile(values, 0.95, method="linear")),
    }


def paired_interval(subject: np.ndarray, first: np.ndarray, second: np.ndarray, config: dict) -> dict:
    """Use the frozen shared whole-patient interval helper with explicitly numeric flag vectors."""
    return paired_subject_difference(
        np.asarray(subject),
        np.asarray(first, dtype=np.float64),
        np.asarray(second, dtype=np.float64),
        draws=config["bootstrap_draws"],
        seed=config["bootstrap_seed"],
    )


def transition_table(first: np.ndarray, second: np.ndarray, config: dict) -> dict:
    """Keep both paired transition directions and second-minus-first patient uncertainty."""
    a, b = np.asarray(first, dtype=bool), np.asarray(second, dtype=bool)
    if a.ndim != 1 or a.shape != b.shape or not len(a):
        raise ValueError("Paired flag vectors must align")
    return {
        "patients": len(a),
        "neither": int((~a & ~b).sum()),
        "first_only": int((a & ~b).sum()),
        "second_only": int((~a & b).sum()),
        "both": int((a & b).sum()),
        "first_flagged": int(a.sum()),
        "second_flagged": int(b.sum()),
        "first_rate": float(a.mean()),
        "second_rate": float(b.mean()),
        "second_minus_first": paired_interval(np.arange(len(a)), b, a, config),
    }


def control_rows(context: dict) -> list[dict]:
    """Compute fixed-role paired control maxima, frozen flags/margins and candidate output levels."""
    rows = []
    for role in ROLES:
        for arm in ARMS:
            threshold = context["thresholds"][arm]
            for subject, group in enumerate(context["groups"][role]):
                row = {"role": role, "arm": arm, "subject": subject}
                for kind, case in zip(("sham", "nuisance"), group[2:], strict=True):
                    index = case["case_index"]
                    score = context["maps"][arm][index - 2000].astype(np.float64)
                    candidate = context["candidate"][index]
                    if context["support"][index].any() or not candidate.any() or candidate.all():
                        raise ValueError("Controls require zero support and nonempty candidate/outside")
                    maximum = float(score.max())
                    row.update(
                        {
                            f"{kind}_case": index,
                            f"{kind}_maximum": maximum,
                            f"{kind}_margin": maximum - threshold,
                            f"{kind}_flag": maximum > threshold,
                            f"{kind}_candidate_mean": float(score[candidate].mean()),
                            f"{kind}_outside_mean": float(score[~candidate].mean()),
                            f"{kind}_candidate_minus_outside": float(
                                score[candidate].mean() - score[~candidate].mean()
                            ),
                        }
                    )
                row["nuisance_minus_sham_maximum"] = row["nuisance_maximum"] - row["sham_maximum"]
                rows.append(row)
    return rows


def summarize_controls(rows: list[dict], config: dict) -> dict:
    """Report fixed within-arm and between-arm flag tables, margins and candidate levels."""
    within, between, levels, level_differences = {}, {}, {}, {}
    for role in ROLES:
        within[role], between[role], levels[role], level_differences[role] = {}, {}, {}, {}
        for arm in ARMS:
            selected = [row for row in rows if row["role"] == role and row["arm"] == arm]
            a = np.array([row["sham_flag"] for row in selected])
            b = np.array([row["nuisance_flag"] for row in selected])
            categories = {
                "all": np.ones(len(a), bool),
                "neither": ~a & ~b,
                "sham_only": a & ~b,
                "nuisance_only": ~a & b,
                "both": a & b,
            }
            fields = (
                "sham_maximum",
                "nuisance_maximum",
                "sham_margin",
                "nuisance_margin",
                "nuisance_minus_sham_maximum",
            )
            within[role][arm] = {
                "flags": transition_table(a, b, config),
                "descriptive_by_transition": {
                    name: {
                        field: descriptive(np.array([r[field] for r in selected])[mask]) for field in fields
                    }
                    for name, mask in categories.items()
                },
                "nuisance_minus_sham_maximum": paired_interval(
                    np.arange(len(a)),
                    np.array([r["nuisance_maximum"] for r in selected]),
                    np.array([r["sham_maximum"] for r in selected]),
                    config,
                ),
            }
            levels[role][arm] = {}
            for kind in ("sham", "nuisance"):
                levels[role][arm][kind] = {
                    name: paired_interval(
                        np.arange(len(a)),
                        np.array([r[f"{kind}_{name}"] for r in selected]),
                        np.zeros(len(a)),
                        config,
                    )
                    for name in ("candidate_mean", "outside_mean", "candidate_minus_outside")
                }
        for kind in ("sham", "nuisance"):
            selected = {arm: [r for r in rows if r["role"] == role and r["arm"] == arm] for arm in ARMS}
            between[role][kind] = transition_table(
                np.array([r[f"{kind}_flag"] for r in selected["R"]]),
                np.array([r[f"{kind}_flag"] for r in selected["S"]]),
                config,
            )
            level_differences[role][kind] = {
                name: paired_interval(
                    np.arange(len(selected["R"])),
                    np.array([r[f"{kind}_{name}"] for r in selected["S"]]),
                    np.array([r[f"{kind}_{name}"] for r in selected["R"]]),
                    config,
                )
                for name in ("candidate_mean", "outside_mean", "candidate_minus_outside")
            }
    return {
        "control_transitions": within,
        "arm_control_transitions": between,
        "control_phase_levels": levels,
        "s_minus_r_control_phase_levels": level_differences,
    }


def positive_statistics(
    edited: np.ndarray, baseline: np.ndarray, target: np.ndarray, candidate: np.ndarray
) -> dict:
    """Reduce native float32 maps in float64 over exact T, unedited candidate U, and outside O."""
    before, after = baseline.astype(np.float64), edited.astype(np.float64)
    regions = {"T": target, "U": candidate & ~target, "O": ~candidate}
    if (
        target.dtype != bool
        or candidate.dtype != bool
        or (target & ~candidate).any()
        or any(not x.any() for x in regions.values())
    ):
        raise ValueError("Every positive T/U/O region must be valid and nonempty")
    result = {}
    for name, region in regions.items():
        result[f"baseline_mean_{name}"] = float(before[region].mean())
        result[f"edited_mean_{name}"] = float(after[region].mean())
        result[f"change_mean_{name}"] = float((after - before)[region].mean())
    for phase in ("baseline", "edited", "change"):
        result[f"{phase}_T_minus_U"] = result[f"{phase}_mean_T"] - result[f"{phase}_mean_U"]
    delta = np.abs(after - before)
    result["full_map_mean_absolute_change"] = float(delta.mean())
    result["full_map_max_absolute_change"] = float(delta.max())
    result["baseline_site_average_precision"] = positive_map_metrics(target, baseline)["average_precision"]
    result["edited_average_precision"] = positive_map_metrics(target, edited)["average_precision"]
    result["average_precision_change"] = (
        result["edited_average_precision"] - result["baseline_site_average_precision"]
    )
    return result


def positive_response_rows(context: dict, wall: float, cpu: float) -> list[dict]:
    """Compute all fixed positive-case statistics and compare the existing parent case-level endpoints."""
    rows = []
    old = {(r["method"], r["case"]): r for r in context["parent_case_metrics"]["counterfactual_rows"]}
    aliases = {
        "change_mean_T": "mean_probability_change_T",
        "change_mean_U": "mean_probability_change_U",
        "change_mean_O": "mean_probability_change_O",
        "change_T_minus_U": "selectivity_contrast",
        "edited_average_precision": "edited_average_precision",
        "baseline_site_average_precision": "baseline_site_average_precision",
        "average_precision_change": "average_precision_change",
    }
    for arm in ARMS:
        for subject, group in enumerate(context["groups"]["evaluation"]):
            _guard(context, wall, cpu)
            for case in group[:2]:
                index, control = case["case_index"], case["matched_control_case_index"]
                row = {
                    "arm": arm,
                    "subject": subject,
                    "mode": case["mode"],
                    "case": index,
                    "control_case": control,
                    **positive_statistics(
                        context["maps"][arm][index - 2000],
                        context["maps"][arm][control - 2000],
                        context["support"][index],
                        context["candidate"][index],
                    ),
                }
                for name, previous in aliases.items():
                    if abs(row[name] - old[arm, index][previous]) > 1e-12:
                        raise ValueError("Parent case-level response reproduction differs")
                rows.append(row)
    return rows


def summarize_positive(rows: list[dict], context: dict, wall: float, cpu: float) -> dict:
    """Give mode-specific and equally weighted patient-macro levels and paired S-minus-R differences."""
    arms, differences = {}, {}
    for arm in ARMS:
        arms[arm] = {}
        for mode in ("all", "focal", "repeated"):
            selected = [r for r in rows if r["arm"] == arm and (mode == "all" or r["mode"] == mode)]
            arms[arm][mode] = {}
            for name in METRICS:
                _guard(context, wall, cpu)
                arms[arm][mode][name] = paired_interval(
                    np.array([r["subject"] for r in selected]),
                    np.array([r[name] for r in selected]),
                    np.zeros(len(selected)),
                    context["config"],
                )
    for mode in ("all", "focal", "repeated"):
        selected = {
            arm: [r for r in rows if r["arm"] == arm and (mode == "all" or r["mode"] == mode)] for arm in ARMS
        }
        differences[mode] = {}
        for name in METRICS:
            _guard(context, wall, cpu)
            differences[mode][name] = paired_interval(
                np.array([r["subject"] for r in selected["S"]]),
                np.array([r[name] for r in selected["S"]]),
                np.array([r[name] for r in selected["R"]]),
                context["config"],
            )
    return {"positive_response": arms, "s_minus_r_positive_response": differences}


def loss_blocks(training: dict, logs: list[dict]) -> dict:
    """Reconcile all finite losses and six logged milestones per arm, without stepwise inference."""
    result = {}
    for arm in ARMS:
        losses = np.asarray(training["arms"][arm]["losses"], dtype=np.float64)
        milestones = [r for r in logs if r["arm"] == arm]
        if (
            losses.shape != (600,)
            or not np.isfinite(losses).all()
            or [r["step"] for r in milestones] != list(range(100, 601, 100))
        ):
            raise ValueError("Complete finite training loss accounting required")
        if any(r["loss"] != losses[r["step"] - 1] for r in milestones):
            raise ValueError("Logged training milestone differs from receipt loss")
        if (
            training["arms"][arm]["first_loss"] != losses[0]
            or training["arms"][arm]["last_loss"] != losses[-1]
        ):
            raise ValueError("Training first/last loss declaration differs")
        result[arm] = {
            "updates": 600,
            "logged_milestones": 6,
            "all_finite": True,
            "first_ten_mean": float(losses[:10].mean()),
            "last_ten_mean": float(losses[-10:].mean()),
            "blocks": [
                {
                    "first_step": start + 1,
                    "last_step": start + 100,
                    "mean": float(losses[start : start + 100].mean()),
                    "std_ddof0": float(losses[start : start + 100].std(ddof=0)),
                    "minimum": float(losses[start : start + 100].min()),
                    "maximum": float(losses[start : start + 100].max()),
                }
                for start in range(0, 600, 100)
            ],
        }
    return result


def _reproduce_parent_flags(context: dict) -> int:
    checked = 0
    for row in context["parent_case_metrics"]["rows"]:
        if row["method"] not in ARMS:
            continue
        maximum = float(context["maps"][row["method"]][row["case"] - 2000].max())
        if (
            maximum != row["maximum_probability"]
            or (maximum > context["thresholds"][row["method"]]) != row["flag"]
        ):
            raise ValueError("Parent case-level flag/maximum reproduction differs")
        checked += 1
    if checked != 1024:
        raise ValueError("Every parent arm/evaluation case must reproduce")
    return checked


def diagnose_paired_background() -> dict:
    """Execute once from frozen saved maps and metadata; retain every failure without retry."""
    if DESTINATION.exists():
        raise FileExistsError("Diagnostic destination must be new")
    wall, cpu = time.perf_counter(), time.process_time()
    config = _json(CONFIG_PATH)
    context = {"config": config, "prior_cpu": 545.144970413, "prior_stage_cpu": 111.012039341}
    sources = {name: sha256_file(ROOT / name) for name in SOURCE_NAMES}
    inputs = {}
    progress = {"completed": False, "control_rows": 0, "positive_rows": 0}
    DESTINATION.mkdir(parents=True, exist_ok=False)
    try:
        names = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        if any(os.environ.get(name) != "1" for name in names) or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("Diagnostic requires one numerical thread and hidden CUDA")
        _guard(context, wall, cpu)
        context = admit_diagnostic(lambda: _guard(context, wall, cpu))
        sources, inputs = context["sources"], context["inputs"]
        _equal_metadata(context["prior_cpu"], 545.144970413)
        _equal_metadata(context["prior_stage_cpu"], 111.012039341)
        context["reproduction"]["evaluation_case_flags_checked"] = _reproduce_parent_flags(context)
        _guard(context, wall, cpu)
        _write(
            DESTINATION / "admission.json",
            {
                "source_sha256": sources,
                "input_sha256": inputs,
                "scope_exceptions": context["scope_exceptions"],
                "reproduction": context["reproduction"],
            },
        )
        controls = control_rows(context)
        progress["control_rows"] = len(controls)
        _guard(context, wall, cpu)
        control_summary = summarize_controls(controls, config)
        _guard(context, wall, cpu)
        positives = positive_response_rows(context, wall, cpu)
        progress["positive_rows"] = len(positives)
        positive_summary = summarize_positive(positives, context, wall, cpu)
        losses = loss_blocks(context["training"], context["logs"])
        if len(controls) != 384 or len(positives) != 512:
            raise ValueError("Fixed diagnostic denominators changed")
        _write(
            DESTINATION / "local_case_statistics.json", {"control_rows": controls, "positive_rows": positives}
        )
        _guard(context, wall, cpu)
        verify_bindings(
            context["bindings"],
            context["authorities"],
            config["allowed_array_payloads"],
            lambda: _guard(context, wall, cpu),
        )
        outputs = {path.name: sha256_file(path) for path in DESTINATION.iterdir() if path.is_file()}
        _guard(context, wall, cpu)
        progress["completed"] = True
        elapsed_cpu = time.process_time() - cpu
        result = {
            "diagnostic_completed": True,
            "study_id": config["study_id"],
            "created_at_utc": datetime.now(UTC).isoformat(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": sources,
            "input_sha256": inputs,
            "output_sha256": outputs,
            "scope_exceptions": context["scope_exceptions"],
            "reproduction": context["reproduction"],
            "progress": progress,
            **control_summary,
            **positive_summary,
            "training_loss": losses,
            "transition_coordinate": {
                "within_arm": "first=sham, second=nuisance",
                "between_arm": "first=R, second=S",
            },
            "wall_seconds": time.perf_counter() - wall,
            "cpu_seconds": elapsed_cpu,
            "prior_cumulative_cpu_seconds": context["prior_cpu"],
            "cumulative_cpu_seconds": context["prior_cpu"] + elapsed_cpu,
            "paired_stage_cpu_seconds": context["prior_stage_cpu"] + elapsed_cpu,
            "actual_stream_bytes_before_summary": _stream_bytes(),
            "new_output_bytes_before_summary": _stream_bytes(DESTINATION),
            "final_receipt_storage_reserve_bytes": 1024**2,
            "waveform_payload_reads": 0,
            "checkpoint_payload_reads": 0,
            "model_inferences": 0,
            "optimizer_updates": 0,
            "gpu_used": False,
            "new_advancement_rule": None,
            "parent_full_advancement_passed": False,
            "clinical_claim": False,
            "numerical_threads": {name: os.environ[name] for name in names},
        }
        _guard(context, wall, cpu)
        _write(DESTINATION / "summary.json", result)
        return result
    except Exception as error:
        _write(
            DESTINATION / "failure.json",
            {
                "source_sha256": sources,
                "input_sha256": inputs,
                "progress": progress,
                "failure_type": type(error).__name__,
                "failure_detail": str(error),
                "wall_seconds": time.perf_counter() - wall,
                "cpu_seconds": time.process_time() - cpu,
                "waveform_payload_reads": 0,
                "checkpoint_payload_reads": 0,
                "model_inferences": 0,
                "optimizer_updates": 0,
                "gpu_used": False,
            },
        )
        raise RuntimeError("Saved-map diagnostic stopped; inspect its local failure receipt") from None
