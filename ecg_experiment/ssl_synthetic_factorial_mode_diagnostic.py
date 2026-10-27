"""Describe frozen factorial output rankings without opening ECGs or checkpoints."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file
from ecg_experiment.intervals import patient_groups, patient_resample
from ecg_experiment.paths import to_stored
from ecg_experiment.ssl_synthetic_paired_diagnostic import (
    _equal_metadata,
    _merge,
    verify_bindings,
)

BASE = ROOT / "outputs/ssl_synthetic_localization"
STUDY = BASE / "counterfactual_factorial_v1"
DESTINATION = STUDY / "mode_diagnostic_v1"
CONFIG = "configs/ssl_synthetic_factorial_mode_diagnostic_v1.json"
AUDIT = "independent_audit_v1/summary.json"
AUDIT_SHA = "74b02f60ac0d77bf071bc99e3ae6fcf460fd8ef0dfba21b9626f5d67646d6c04"
EVALUATION = "evaluation_v2/summary.json"
EVALUATION_SHA = "3f4555614180442c1eb82c5e5b20ca4137d53ae35697421bfa4337cf4f588d28"
SOURCE_NAMES = (
    "ecg_experiment/ssl_synthetic_factorial_mode_diagnostic.py",
    "scripts/reports/diagnose_ssl_synthetic_factorial_modes.py",
    "tests/test_ssl_synthetic_factorial_mode_diagnostic.py",
    CONFIG,
    "docs/ssl-synthetic-localization-factorial-mode-diagnostic.md",
)
METRICS = (
    "full_grid_ap",
    "lead_ap",
    "temporal_ap",
    "oracle_temporal_ap",
    "top_1_precision",
    "top_1_recall",
    "top_5_precision",
    "top_5_recall",
    "top_10_precision",
    "top_10_recall",
    "mass_T",
    "mass_U",
    "mass_O",
    "candidate_minus_outside",
)
PHASES = ("baseline", "edited", "change")


def read_json(path: Path) -> Any:
    """Read a permitted metadata object."""
    return json.loads(path.read_text())


def write_new(path: Path, value: Any) -> None:
    """Create one finite JSON artifact without overwriting an existing receipt."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def stream_bytes(path: Path = BASE) -> int:
    """Count file sizes without reading payload contents."""
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def allowed_arrays() -> list[str]:
    """Return the complete six-array read/hash allowlist."""
    return [
        to_stored(STUDY / f"evaluation_v2/probabilities_{arm}.npy") for arm in ("RB", "SB", "RP", "SP")
    ] + [to_stored(STUDY / f"preparation/{name}.npy") for name in ("support", "candidate")]


def bootstrap_indices(subjects: int, config: dict) -> np.ndarray:
    """Construct shared whole-patient draws with the project's resampling helpers."""
    groups = patient_groups(np.arange(subjects))
    rng = np.random.default_rng(config["bootstrap_seed"])
    return np.stack([patient_resample(groups, rng) for _ in range(config["bootstrap_draws"])])


def interval(values: np.ndarray, indices: np.ndarray, config: dict) -> dict:
    """Summarize one finite, equally weighted patient vector using shared paired draws."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Every patient must have a finite diagnostic value")
    if indices.shape != (config["bootstrap_draws"], len(values)):
        raise ValueError("Bootstrap draws and complete patient vector differ")
    low, high = np.percentile(values[indices].mean(axis=1), [2.5, 97.5], method="linear")
    return {
        "difference": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "subjects": len(values),
        "cases": len(values),
        "draws": len(indices),
        "skipped_draws": 0,
        "seed": config["bootstrap_seed"],
    }


def top_area(truth: np.ndarray, scores: np.ndarray, area: int) -> tuple[float, float]:
    """Credit boundary ties fractionally at a fixed full-grid area."""
    target, values = truth.ravel(), scores.ravel()
    if not 0 < area <= len(values) or not target.any():
        raise ValueError("Fixed-area scoring requires positive truth and a valid area")
    cutoff = np.partition(values, len(values) - area)[-area]
    above, tied = values > cutoff, values == cutoff
    fraction = (area - int(above.sum())) / int(tied.sum())
    captured = float(target[above].sum() + fraction * target[tied].sum())
    return captured / area, captured / int(target.sum())


def mask_geometry(truth: np.ndarray, candidate: np.ndarray, fractions: list[float]) -> dict:
    """Validate native truth regions and report their structural coverage ceilings."""
    if truth.shape != (12, 2500) or candidate.shape != truth.shape:
        raise ValueError("Masks must occupy the native 12 by 2500 grid")
    if truth.dtype != np.bool_ or candidate.dtype != np.bool_ or np.any(truth & ~candidate):
        raise ValueError("Boolean truth must be contained in the candidate region")
    regions = (truth, candidate & ~truth, ~candidate)
    if not all(region.any() for region in regions) or not candidate.any(axis=1).all():
        raise ValueError("Every required truth/candidate/outside region must be nonempty")
    leads, times = truth.any(axis=1), truth.any(axis=0)
    if leads.all() or times.all():
        raise ValueError("Lead and temporal AP require both target classes")
    count = int(truth.sum())
    budgets = {str(int(100 * fraction)): int(np.ceil(fraction * truth.size)) for fraction in fractions}
    return {
        "support_cells": count,
        "candidate_cells": int(candidate.sum()),
        "support_prevalence": count / truth.size,
        "positive_leads": int(leads.sum()),
        "positive_times": int(times.sum()),
        "area_budgets": budgets,
        "attainable_recall": {key: min(1.0, area / count) for key, area in budgets.items()},
    }


def map_statistics(
    truth: np.ndarray, candidate: np.ndarray, scores: np.ndarray, geometry: dict
) -> dict[str, float]:
    """Compute the fourteen prospectively frozen output-ranking/allocation fields."""
    values = np.asarray(scores)
    if values.shape != truth.shape or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Expected finite native probabilities")
    values64 = values.astype(np.float64)
    total = float(values64.sum())
    if total <= 0:
        raise ValueError("Zero total probability mass is an admission failure")
    active = truth.any(axis=1)
    lead_scores = np.array([values64[lead, candidate[lead]].mean() for lead in range(12)])
    result = {
        "full_grid_ap": float(average_precision_score(truth.ravel(), values.ravel())),
        "lead_ap": float(average_precision_score(active, lead_scores)),
        "temporal_ap": float(average_precision_score(truth.any(axis=0), values64.mean(axis=0))),
        "oracle_temporal_ap": float(
            np.mean([average_precision_score(truth[lead], values[lead]) for lead in np.flatnonzero(active)])
        ),
        "mass_T": float(values64[truth].sum()) / total,
        "mass_U": float(values64[candidate & ~truth].sum()) / total,
        "mass_O": float(values64[~candidate].sum()) / total,
        "candidate_minus_outside": float(values64[candidate].mean() - values64[~candidate].mean()),
    }
    for key, budget in geometry["area_budgets"].items():
        precision, recall = top_area(truth, values, budget)
        result[f"top_{key}_precision"], result[f"top_{key}_recall"] = precision, recall
    if set(result) != set(METRICS) or not np.isfinite(list(result.values())).all():
        raise ValueError("Frozen metric field set or finite denominator changed")
    return result


def coordinates(cases: list[dict]) -> list[dict]:
    """Require all 128 patients and exact new-bank positive/control coordinates."""
    by_index = {case["case_index"]: case for case in cases}
    if len(cases) != 768 or set(by_index) != set(range(2000, 2768)):
        raise ValueError("New validation metadata coordinates changed")
    positives = []
    for subject in range(128):
        for variant, mode in enumerate(("focal", "repeated")):
            index = 2256 + subject * 4 + variant
            case = by_index[index]
            control = index // 4 * 4 + (3 if case["nuisance"] else 2)
            counterpart = by_index[control]
            if (
                case["role_subject_index"] != subject
                or case["mode"] != mode
                or case["role"] != "evaluation"
                or not case["target"]
                or case["mask_row"] != index - 2000
                or case["matched_control_case_index"] != control
                or counterpart["target"]
                or counterpart["role_subject_index"] != subject
                or case["baseline_sha256"] != counterpart["edited_sha256"]
            ):
                raise ValueError("Positive or matched baseline metadata changed")
            positives.append(case)
    return positives


def require_bindings(bindings: dict, names: list[str]) -> None:
    """Reject any consumed array/metadata path lacking an already verified declaration."""
    if set(names) - bindings.keys():
        raise ValueError("A consumed diagnostic input lacks a frozen binding")


def admit(guard: Callable[[], None]) -> dict:
    """Anchor complete passed parent reproduction and defer every forbidden payload digest."""
    audit_path, eval_path = STUDY / AUDIT, STUDY / EVALUATION
    if sha256_file(audit_path) != AUDIT_SHA or sha256_file(eval_path) != EVALUATION_SHA:
        raise ValueError("Parent audit/evaluation identity changed")
    audit, evaluation = read_json(audit_path), read_json(eval_path)
    if not audit["all_passed"] or audit["mismatch_count"] or not evaluation["evaluation_completed"]:
        raise ValueError("Parent completion gates failed")
    if audit["evaluation_summary_sha256"] != EVALUATION_SHA:
        raise ValueError("Parent audit is not bound to this exact evaluation")
    for key, value in audit["independently_recomputed_complete_summary"].items():
        _equal_metadata(value, evaluation[key])
    decision = evaluation["decision"]
    if len(decision["gates"]) != 12 or not all(decision["gates"].values()):
        raise ValueError("The complete parent twelve-gate decision changed")
    sources = dict(audit["source_sha256"])
    _merge(sources, {name: sha256_file(ROOT / name) for name in SOURCE_NAMES})
    inputs = dict(audit["input_sha256"])
    _merge(inputs, {to_stored(audit_path): AUDIT_SHA, to_stored(eval_path): EVALUATION_SHA})
    authority = dict(audit["input_sha256"])
    _merge(
        authority,
        {
            name: value["expected_sha256"]
            for name, value in audit["deferred_payload_scope_exceptions"].items()
        },
    )
    bindings = dict(sources)
    _merge(bindings, inputs)
    required = allowed_arrays() + [
        to_stored(STUDY / name)
        for name in (
            "preparation/cases.json",
            "evaluation_v2/calibration.json",
            "evaluation_v2/local_case_metrics.json",
        )
    ]
    require_bindings(bindings, required)
    exceptions = verify_bindings(bindings, authority, allowed_arrays(), guard)
    calibration = read_json(STUDY / "evaluation_v2/calibration.json")
    _equal_metadata(audit["independently_recomputed_thresholds"], calibration["thresholds"])
    cases = read_json(STUDY / "preparation/cases.json")["cases"]
    positives = coordinates(cases)
    parent_rows = read_json(STUDY / "evaluation_v2/local_case_metrics.json")
    if len(parent_rows["rows"]) != 2816 or len(parent_rows["responses"]) != 1024:
        raise ValueError("Frozen parent case/response row count changed")
    return {
        "sources": sources,
        "inputs": inputs,
        "bindings": bindings,
        "authority": authority,
        "scope_exceptions": exceptions,
        "positives": positives,
        "parent_rows": parent_rows,
        "prior_cpu": audit["cumulative_cpu_seconds"],
        "prior_gpu": audit["global_gpu_wall_seconds"],
        "reproduction": {
            "complete_scientific_blocks": 6,
            "thresholds": calibration["thresholds"],
            "parent_decision": decision,
            "parent_audit_all_passed": True,
            "parent_case_rows": 2816,
            "parent_response_rows": 1024,
        },
    }


def native_arrays() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Map only the six allowed arrays and validate their full native geometry/dtypes."""
    maps = {
        arm: np.load(STUDY / f"evaluation_v2/probabilities_{arm}.npy", mmap_mode="r")
        for arm in ("RB", "SB", "RP", "SP")
    }
    support, candidate = [
        np.load(STUDY / f"preparation/{name}.npy", mmap_mode="r") for name in ("support", "candidate")
    ]
    for array in maps.values():
        if array.shape != (768, 12, 2500) or array.dtype != np.float32:
            raise ValueError("Frozen probability bank geometry/dtype changed")
    if any(array.shape != (768, 12, 2500) or array.dtype != np.bool_ for array in (support, candidate)):
        raise ValueError("Frozen mask bank geometry/dtype changed")
    return maps, support, candidate


def calculate_rows(context: dict, config: dict, guard: Callable[[], None], progress: dict) -> list[dict]:
    """Describe every saved positive/control pair, retaining parent AP/top-one reproduction."""
    require_bindings(context["bindings"], allowed_arrays())
    maps, support, candidate = native_arrays()
    parent = {(row["method"], row["case"]): row for row in context["parent_rows"]["responses"]}
    rows = []
    for case in context["positives"]:
        guard()
        index, control = case["case_index"], case["matched_control_case_index"]
        row_index = index - config["global_case_offset"]
        truth, prior = support[row_index], candidate[row_index]
        geometry = mask_geometry(truth, prior, config["area_fractions"])
        for arm, array in maps.items():
            phases = {
                "baseline": map_statistics(truth, prior, array[control - 2000], geometry),
                "edited": map_statistics(truth, prior, array[row_index], geometry),
            }
            phases["change"] = {key: phases["edited"][key] - phases["baseline"][key] for key in METRICS}
            old = parent[arm, index]
            for phase, prefix in (("baseline", "baseline_site"), ("edited", "edited")):
                for current, suffix in (
                    ("full_grid_ap", "average_precision"),
                    ("top_1_precision", "top_one_percent_precision"),
                    ("top_1_recall", "top_one_percent_recall"),
                ):
                    _equal_metadata(phases[phase][current], old[f"{prefix}_{suffix}"])
            rows.append(
                {
                    "arm": arm,
                    "subject": case["role_subject_index"],
                    "mode": case["mode"],
                    "global_case": index,
                    "map_row": row_index,
                    "control_case": control,
                    "geometry": geometry,
                    **phases,
                }
            )
            progress["rows"] = len(rows)
    if len(rows) != 1024 or len(parent) != 1024:
        raise ValueError("Complete positive diagnostic denominator changed")
    return rows


def describe(values: list[float]) -> dict:
    """Report complete finite geometry ranges without filtering patients."""
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "minimum": float(array.min()), "maximum": float(array.max())}


def summarize_geometry(rows: list[dict]) -> dict:
    """Reduce all immutable support geometry fields and fixed-area ceilings."""
    keys = ("support_cells", "candidate_cells", "support_prevalence", "positive_leads", "positive_times")
    result = {
        "patients": len(rows),
        "fields": {key: describe([row["geometry"][key] for row in rows]) for key in keys},
    }
    for key in ("area_budgets", "attainable_recall"):
        result[key] = {
            area: describe([row["geometry"][key][area] for row in rows]) for area in ("1", "5", "10")
        }
    return result


def summarize(rows: list[dict], config: dict, guard: Callable[[], None]) -> dict:
    """Report every frozen field/phase by arm/mode and paired loss contrasts using shared draws."""
    indices = bootstrap_indices(config["evaluation_patients"], config)
    statistics, contrasts, geometry = {}, {}, {}
    vectors = {}
    for arm in config["arms"]:
        statistics[arm] = {}
        for mode in config["modes"]:
            selected = sorted(
                (row for row in rows if row["arm"] == arm and row["mode"] == mode),
                key=lambda row: row["subject"],
            )
            if [row["subject"] for row in selected] != list(range(config["evaluation_patients"])):
                raise ValueError("Patient alignment changed; no exclusions are permitted")
            statistics[arm][mode] = {}
            for phase in PHASES:
                guard()
                statistics[arm][mode][phase] = {}
                for metric in METRICS:
                    values = np.array([row[phase][metric] for row in selected])
                    vectors[arm, mode, phase, metric] = values
                    statistics[arm][mode][phase][metric] = interval(values, indices, config)
            if arm == "RB":
                geometry[mode] = summarize_geometry(selected)
    for initialization, before, after in (("R", "RB", "RP"), ("S", "SB", "SP")):
        contrasts[initialization] = {}
        for mode in config["modes"]:
            contrasts[initialization][mode] = {}
            for phase in PHASES:
                guard()
                contrasts[initialization][mode][phase] = {
                    metric: interval(
                        vectors[after, mode, phase, metric] - vectors[before, mode, phase, metric],
                        indices,
                        config,
                    )
                    for metric in METRICS
                }
    return {"statistics": statistics, "paired_loss_contrasts": contrasts, "geometry": geometry}


def run() -> dict:
    """Execute one reviewed CPU-only saved-map diagnostic with live deadlines and retained failures."""
    wall, cpu = time.perf_counter(), time.process_time()
    if DESTINATION.exists():
        raise FileExistsError("Mode diagnostic output must be new")
    DESTINATION.mkdir(parents=True, exist_ok=False)
    config, context, progress = (
        {},
        {},
        {"completed": False, "scoring_started": False, "scoring_completed": False, "rows": 0},
    )

    def guard() -> None:
        elapsed_wall, elapsed_cpu = time.perf_counter() - wall, time.process_time() - cpu
        reserve = config["failure_reserve_seconds"]
        if max(elapsed_wall, elapsed_cpu) + reserve >= config["main_cpu_wall_limit_seconds"]:
            raise TimeoutError("Diagnostic CPU/wall deadline reached")
        if context.get("prior_cpu", 0.0) + elapsed_cpu + reserve >= config["cpu_seconds_ceiling"]:
            raise TimeoutError("Cumulative CPU ceiling reached")
        if (
            stream_bytes(DESTINATION) + 1024**2 >= config["output_bytes_ceiling"]
            or stream_bytes() + 1024**2 >= config["cache_bytes_ceiling"]
        ):
            raise ValueError("Diagnostic/whole-stream storage ceiling reached")

    try:
        config = read_json(ROOT / CONFIG)
        thread_names = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        if (
            any(os.environ.get(name) != "1" for name in thread_names)
            or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        ):
            raise ValueError("One numerical thread and hidden CUDA are required")
        context = admit(guard)
        if context["prior_gpu"] >= config["gpu_seconds_ceiling"]:
            raise ValueError("Cumulative GPU ceiling already exhausted")
        write_new(
            DESTINATION / "admission.json",
            {key: context[key] for key in ("sources", "inputs", "scope_exceptions", "reproduction")},
        )
        progress["scoring_started"] = True
        rows = calculate_rows(context, config, guard, progress)
        progress["scoring_completed"] = True
        aggregates = summarize(rows, config, guard)
        write_new(DESTINATION / "local_case_statistics.json", {"rows": rows})
        guard()
        verify_bindings(context["bindings"], context["authority"], allowed_arrays(), guard)
        outputs = {to_stored(path): sha256_file(path) for path in DESTINATION.iterdir() if path.is_file()}
        guard()
        progress["completed"] = True
        elapsed_cpu, elapsed_wall = time.process_time() - cpu, time.perf_counter() - wall
        result = {
            "diagnostic_completed": True,
            "study_id": config["study_id"],
            "source_sha256": context["sources"],
            "input_sha256": context["inputs"],
            "output_sha256": outputs,
            "scope_exceptions": context["scope_exceptions"],
            "reproduction": context["reproduction"],
            "progress": progress,
            **aggregates,
            "metric_fields": list(METRICS),
            "phases": list(PHASES),
            "cpu_seconds": elapsed_cpu,
            "wall_seconds": elapsed_wall,
            "prior_cumulative_cpu_seconds": context["prior_cpu"],
            "cumulative_cpu_seconds": context["prior_cpu"] + elapsed_cpu,
            "global_gpu_wall_seconds": context["prior_gpu"],
            "stream_bytes_before_summary": stream_bytes(),
            "diagnostic_bytes_before_summary": stream_bytes(DESTINATION),
            "raw_waveform_reads": 0,
            "processed_waveform_reads": 0,
            "checkpoint_payload_reads": 0,
            "model_inferences": 0,
            "optimizer_updates": 0,
            "gpu_used": False,
            "numerical_threads": 1,
            "new_advancement_rule": None,
            "clinical_claim": False,
            "parent_decision_preserved": True,
        }
        guard()
        write_new(DESTINATION / "summary.json", result)
        return result
    except Exception as error:
        write_new(
            DESTINATION / "failure.json",
            {
                "failure_type": type(error).__name__,
                "failure_detail": str(error),
                "progress": progress,
                "cpu_seconds": time.process_time() - cpu,
                "wall_seconds": time.perf_counter() - wall,
                "prior_cumulative_cpu_seconds": context.get("prior_cpu"),
                "source_sha256": context.get("sources", {}),
                "input_sha256": context.get("inputs", {}),
                "raw_waveform_reads": 0,
                "processed_waveform_reads": 0,
                "checkpoint_payload_reads": 0,
                "model_inferences": 0,
                "optimizer_updates": 0,
                "gpu_used": False,
            },
        )
        raise RuntimeError("Mode diagnostic stopped; inspect retained failure receipt") from None
