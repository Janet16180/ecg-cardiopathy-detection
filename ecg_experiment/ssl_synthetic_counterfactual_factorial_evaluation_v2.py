"""Explicit CPU scoring successor preserving the exact already-profiled four-arm GPU path."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.ssl_synthetic_counterfactual import response_metrics
from ecg_experiment.ssl_synthetic_counterfactual_factorial_evaluation import (
    ARMS,
    LEVEL_METRICS,
    _after_training,
    _infer,
    calibration_thresholds,
    map_row,
    summarize,
)
from ecg_experiment.ssl_synthetic_counterfactual_factorial_evaluation import (
    SOURCE_NAMES as V1_SOURCE_NAMES,
)
from ecg_experiment.ssl_synthetic_evaluation import _merge_hashes, configure_inference
from ecg_experiment.ssl_synthetic_metrics import positive_map_metrics
from ecg_experiment.ssl_synthetic_paired_diagnostic import positive_statistics
from ecg_experiment.ssl_synthetic_preparation import stream_bytes, write_new_json
from ecg_experiment.ssl_synthetic_real_transfer import _thread_environment

POLICY_NAME = "docs/ssl-synthetic-localization-counterfactual-factorial-evaluation-v2.md"
V1_SOURCE = "ecg_experiment/ssl_synthetic_counterfactual_factorial_evaluation.py"
SOURCE_NAMES = (
    "ecg_experiment/ssl_synthetic_counterfactual_factorial_evaluation_v2.py",
    "scripts/reports/evaluate_ssl_synthetic_counterfactual_factorial_v2.py",
    "tests/test_ssl_synthetic_counterfactual_factorial_evaluation_v2.py",
    *V1_SOURCE_NAMES,
)


def case_metrics(
    context: dict, maps: dict[str, np.ndarray], thresholds: dict, clock: dict
) -> tuple[list[dict], list[dict]]:
    """Score every new evaluation case and exact matching control on the full native grid."""
    from ecg_experiment.ssl_synthetic_counterfactual_factorial_common import (
        candidate_for_case,
        check_budget,
        render_checked,
        support_for_case,
    )

    rows, responses = [], []
    for index in context["roles"]["evaluation"]:
        check_budget(context, clock)
        case = context["cases"][index]
        target, candidate = support_for_case(context, index), candidate_for_case(context, index)
        base = {
            "case": index,
            "map_row": map_row(index),
            "subject": case["role_subject_index"],
            "global_subject": case["subject_index"],
            "kind": case["kind"],
            "mode": case["mode"],
        }
        for arm in ARMS:
            scores = maps[arm][map_row(index)]
            maximum = float(scores.max())
            row = {**base, "method": arm, "maximum_probability": maximum, "flag": maximum > thresholds[arm]}
            if case["target"]:
                row.update(positive_map_metrics(target, scores))
                control = case["matched_control_case_index"]
                if control != index // 4 * 4 + (3 if case["nuisance"] else 2):
                    raise ValueError("Evaluation matched-control alias changed")
                pair, counterpart = render_checked(context, index), render_checked(context, control)
                if not np.array_equal(pair["baseline"], counterpart["edited"]):
                    raise ValueError("Evaluation positive baseline differs from actual control")
                baseline = maps[arm][map_row(control)]
                response = response_metrics(truth=target, prior=candidate, edited=scores, baseline=baseline)
                levels = positive_statistics(scores, baseline, target, candidate)
                responses.append(
                    {
                        **base,
                        "method": arm,
                        "control_case": control,
                        **response,
                        **{name: levels[name] for name in LEVEL_METRICS},
                    }
                )
            elif target.any():
                raise ValueError("A zero-intervention control has positive support")
            rows.append(row)
        if case["target"]:
            edited = render_checked(context, index)["edited"]
            for method, scores in (
                ("uniform", np.ones_like(edited)),
                ("absolute_voltage", np.abs(edited)),
                ("algorithmic_anchor_prior", candidate.astype(np.float32)),
            ):
                rows.append({**base, "method": method, **positive_map_metrics(target, scores)})
    if len(rows) != 2816 or len(responses) != 1024:
        raise ValueError("Complete four-arm case metrics required")
    return rows, responses


def _new_destination(study: Path) -> Path:
    destination = study / "evaluation_v2"
    if (study / "evaluation").exists():
        raise ValueError("Pinned v1 evaluation must remain unexecuted")
    if destination.exists():
        raise FileExistsError("Factorial evaluation destination must be new")
    return destination


def evaluate_factorial_v2() -> dict:
    """Freeze all four A calibrations, then evaluate every new B patient once with final-only states."""
    from ecg_experiment.ssl_synthetic_counterfactual_factorial_common import (
        STUDY,
        admit_context,
        check_budget,
        storage_projection,
        verify_context,
    )
    from ecg_experiment.ssl_synthetic_paired_background_training import _receipt
    from ecg_experiment.ssl_synthetic_real_background import array_sha256

    destination = _new_destination(STUDY)
    wall, cpu = time.perf_counter(), time.process_time()
    sources = {name: sha256_file(ROOT / name) for name in (*SOURCE_NAMES, POLICY_NAME)}
    inputs, progress = (
        {},
        {
            "completed": False,
            "predicted_record_maps": 0,
            "calibration_frozen": False,
            "performance_scoring_started": False,
        },
    )
    with gpu_lock("cuda", blocking=False):
        destination.mkdir(parents=True, exist_ok=False)
        try:
            context = admit_context(SOURCE_NAMES)
            _merge_hashes(context["sources"], {POLICY_NAME: sha256_file(ROOT / POLICY_NAME)})
            sources, inputs = context["sources"], context["inputs"]
            context["storage_projection"] = storage_projection()
            if (
                context["storage_projection"]["projected_whole_stream_bytes"]
                >= context["config"]["cache_bytes_ceiling"]
            ):
                raise ValueError("Four-map storage forecast fails")
            context["numerical_threads"] = _thread_environment()
            profile, training = _after_training(context)
            if profile["source_sha256"][V1_SOURCE] != sha256_file(ROOT / V1_SOURCE):
                raise ValueError("Exact profiled inference source changed")
            clock = {
                "wall": wall,
                "cpu": cpu,
                "prior_stage_wall": profile["wall_seconds"] + training["wall_seconds"],
                "prior_stage_cpu": context["stage_preparation_cpu_seconds"]
                + profile["cpu_seconds"]
                + training["cpu_seconds"],
            }
            configure_inference()
            torch.cuda.reset_peak_memory_stats()
            check_budget(context, clock)
            maps = {
                arm: open_memmap(
                    destination / f"probabilities_{arm}.npy",
                    mode="w+",
                    dtype=np.float32,
                    shape=(768, 12, 2500),
                )
                for arm in ARMS
            }
            write_new_json(
                destination / "admission.json",
                {
                    "source_sha256": sources,
                    "input_sha256": inputs,
                    "storage_projection": context["storage_projection"],
                    "bce_reproduction": training["bce_reproduction"],
                },
            )
            for arm in ARMS:
                _infer(context, arm, context["roles"]["calibration"], maps[arm], training, clock, progress)
            calibration = {
                "thresholds": calibration_thresholds(context["cases"], maps),
                "patients": 64,
                "method": "higher",
                "quantile": 0.95,
                "comparison": "strictly greater",
                "global_calibration_case_indices": context["roles"]["calibration"],
                "probability_row_offset": 2000,
                "source_sha256": sources,
                "input_sha256": inputs,
                "calibration_slice_sha256": {arm: array_sha256(maps[arm][:256]) for arm in ARMS},
                "created_at_utc": datetime.now(UTC).isoformat(),
            }
            write_new_json(destination / "calibration.json", calibration)
            frozen = sha256_file(destination / "calibration.json")
            progress["calibration_frozen"] = True
            for arm in ARMS:
                if sha256_file(destination / "calibration.json") != frozen:
                    raise ValueError("Frozen calibration bytes changed before B inference")
                _infer(context, arm, context["roles"]["evaluation"], maps[arm], training, clock, progress)
            if sha256_file(destination / "calibration.json") != frozen or any(
                array_sha256(maps[arm][:256]) != calibration["calibration_slice_sha256"][arm] for arm in ARMS
            ):
                raise ValueError("Frozen calibration file or one of four slices changed")
            progress["performance_scoring_started"] = True
            rows, responses = case_metrics(context, maps, calibration["thresholds"], clock)
            write_new_json(destination / "local_case_metrics.json", {"rows": rows, "responses": responses})
            write_new_json(destination / "regeneration_checks.json", context["regeneration_checks"])
            summary = summarize(rows, responses, context, clock)
            verify_context(context)
            check_budget(context, clock)
            if (
                sum(path.stat().st_size for path in STUDY.rglob("*") if path.is_file()) + 1024**2
                >= context["config"]["complete_study_new_bytes"]
            ):
                raise ValueError("Shared study metadata/checkpoint reserve exhausted")
            if stream_bytes() + 1024**2 >= context["config"]["cache_bytes_ceiling"]:
                raise ValueError("Actual stream size plus final receipt reserve exceeds cap")
            progress["completed"] = True
            result = _receipt(
                destination,
                context,
                clock,
                {
                    "study_id": context["config"]["study_id"],
                    "evaluation_completed": True,
                    "scoring_successor_policy": POLICY_NAME,
                    "v1_evaluation_ran": False,
                    "profiled_gpu_path_reused": True,
                    **summary,
                    "progress": progress,
                    "calibration": calibration,
                    "calibration_sha256": frozen,
                    "patients": 128,
                    "evaluation_cases": 512,
                    "calibration_cases": 256,
                    "probability_shape": [768, 12, 2500],
                    "probability_dtype": "float32",
                    "probability_global_case_offset": 2000,
                    "bce_reproduction": training["bce_reproduction"],
                    "predecessor_paired_advancement_passed": False,
                    "scope_exceptions": context["scope_exceptions"],
                    "new_training_updates": 0,
                    "final_receipt_storage_reserve_bytes": 1024**2,
                },
            )
            check_budget(context, clock)
            write_new_json(destination / "summary.json", result)
            return result
        except Exception as error:
            write_new_json(
                destination / "failure.json",
                {
                    "source_sha256": sources,
                    "input_sha256": inputs,
                    "progress": progress,
                    "failure_type": type(error).__name__,
                    "failure_detail": str(error),
                    "wall_seconds": time.perf_counter() - wall,
                    "cpu_seconds": time.process_time() - cpu,
                    "performance_scoring_started": progress["performance_scoring_started"],
                    "performance_scoring_completed": False,
                },
            )
            raise RuntimeError("Factorial evaluation stopped; inspect its local failure receipt") from None
