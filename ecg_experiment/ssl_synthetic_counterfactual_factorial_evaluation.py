"""Evaluate the frozen four-arm loss factorial after exact BCE training reproduction."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime

import numpy as np
import torch
from numpy.lib.format import open_memmap

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.paths import to_stored
from ecg_experiment.ssl_synthetic_counterfactual import RESPONSE_METRICS, response_metrics
from ecg_experiment.ssl_synthetic_evaluation import _merge_hashes, configure_inference
from ecg_experiment.ssl_synthetic_metrics import (
    control_threshold,
    paired_subject_difference,
    positive_map_metrics,
)
from ecg_experiment.ssl_synthetic_model import state_digest
from ecg_experiment.ssl_synthetic_model_v2 import DenseWaveformModel
from ecg_experiment.ssl_synthetic_paired_background_evaluation import map_row
from ecg_experiment.ssl_synthetic_paired_diagnostic import positive_statistics
from ecg_experiment.ssl_synthetic_preparation import stream_bytes, write_new_json
from ecg_experiment.ssl_synthetic_real_transfer import _thread_environment

ARMS = ("RB", "SB", "RP", "SP")
MODES = ("all", "focal", "repeated")
LEVEL_METRICS = tuple(f"{phase}_mean_{region}" for phase in ("baseline", "edited") for region in "TUO") + (
    "baseline_T_minus_U",
    "edited_T_minus_U",
    "full_map_mean_absolute_change",
    "full_map_max_absolute_change",
)
SOURCE_NAMES = (
    "ecg_experiment/ssl_synthetic_counterfactual_factorial_evaluation.py",
    "scripts/reports/evaluate_ssl_synthetic_counterfactual_factorial.py",
    "tests/test_ssl_synthetic_counterfactual_factorial_evaluation.py",
    "ecg_experiment/ssl_synthetic_counterfactual.py",
    "ecg_experiment/ssl_synthetic_metrics.py",
    "ecg_experiment/intervals.py",
    "ecg_experiment/ssl_synthetic_paired_diagnostic.py",
    "ecg_experiment/ssl_synthetic_model_v2.py",
    "ecg_experiment/ssl_synthetic_paired_background_evaluation.py",
)


def predict_batch(model: torch.nn.Module, context: dict, indices: list[int]) -> np.ndarray:
    """Run exactly one observed waveform per model input; use the same path in resource profiling."""
    from ecg_experiment.ssl_synthetic_counterfactual_factorial_common import render_checked

    signal = np.stack([render_checked(context, int(index))["edited"] for index in indices])
    with torch.inference_mode():
        logits = model(torch.from_numpy(signal).to(next(model.parameters()).device))
        if logits.shape != signal.shape or logits.dtype != torch.float32 or not torch.isfinite(logits).all():
            raise ValueError("Finite float32 native logits required")
        values = logits.sigmoid().cpu().numpy()
    if values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError("Native finite float32 probability maps required")
    return values


def calibration_thresholds(cases: list[dict], maps: dict[str, np.ndarray]) -> dict[str, float]:
    """Fit only the 64 joint calibration-control maxima per arm, retaining original strict semantics."""
    controls = {
        kind: sorted(
            (c for c in cases if c["role"] == "calibration" and c["kind"] == kind),
            key=lambda c: c["role_subject_index"],
        )
        for kind in ("sham", "nuisance")
    }
    if any([c["role_subject_index"] for c in rows] != list(range(64)) for rows in controls.values()):
        raise ValueError("Complete 64-patient calibration control sets required")
    return {
        arm: control_threshold(
            np.array([maps[arm][map_row(c["case_index"])].max() for c in controls["sham"]]),
            np.array([maps[arm][map_row(c["case_index"])].max() for c in controls["nuisance"]]),
        )
        for arm in ARMS
    }


def _patient_values(
    rows: list[dict], arm: str, name: str, kind: str = "positive", mode: str = "all"
) -> np.ndarray:
    chosen = [
        r for r in rows if r["method"] == arm and r["kind"] == kind and (mode == "all" or r["mode"] == mode)
    ]
    if {r["subject"] for r in chosen} != set(range(128)):
        raise ValueError("Every evaluation patient must remain in every contrast")
    expected = 2 if kind == "positive" and mode == "all" else 1
    if len(chosen) != 128 * expected:
        raise ValueError("Patient/mode contrast denominator changed")
    return np.array([np.mean([r[name] for r in chosen if r["subject"] == subject]) for subject in range(128)])


def factorial_contrasts(values: dict[str, np.ndarray], config: dict) -> dict:
    """Use shared identical patient draws for the primary, simple effects and interaction."""
    if set(values) != set(ARMS) or any(np.shape(x) != (128,) for x in values.values()):
        raise ValueError("All four matched 128-patient arrays are required")
    r = values["RP"] - values["RB"]
    s = values["SP"] - values["SB"]
    contrasts = {
        "primary_average_loss_effect": (r + s) / 2,
        "paired_loss_effect_R": r,
        "paired_loss_effect_S": s,
        "initialization_by_loss_interaction": s - r,
        "ssl_effect_BCE": values["SB"] - values["RB"],
        "ssl_effect_paired": values["SP"] - values["RP"],
    }
    return {
        name: paired_subject_difference(
            np.arange(128),
            change,
            np.zeros(128),
            draws=config["bootstrap_draws"],
            seed=config["bootstrap_seed"],
        )
        for name, change in contrasts.items()
    }


def decision(summary: dict, config: dict) -> dict:
    """Apply all twelve prospectively fixed primary, per-initialization, response and control gates."""
    effects = summary["factorial_contrasts"]["edited_average_precision"]["all"]
    primary = effects["primary_average_loss_effect"]
    gates = {
        "primary_direct_ap_gain_at_least_minimum": primary["difference"]
        >= config["primary_direct_ap_gain_minimum"],
        "primary_direct_ap_lower_bound_positive": primary["ci_low"] > 0,
    }
    for init in ("R", "S"):
        gates[f"{init}_edited_ap_noninferiority"] = (
            effects[f"paired_loss_effect_{init}"]["ci_low"] > -config["edited_ap_noninferiority_margin"]
        )
        for metric in ("average_precision_change", "selectivity_contrast"):
            gates[f"{init}P_{metric}_lower_bound_positive"] = (
                summary["responses"][init + "P"]["all"][metric]["ci_low"] > 0
            )
        for kind in ("sham", "nuisance"):
            gates[f"{init}_{kind}_flag_increase_bounded"] = (
                summary["paired_loss_control_flags"][init][kind]["difference"]
                <= config["control_point_increase_maximum"]
            )
    if len(gates) != 12:
        raise ValueError("Expected exactly twelve frozen gates")
    return {"gates": gates, "engineering_replication_admitted": bool(all(gates.values()))}


def summarize(rows: list[dict], responses: list[dict], context: dict, clock: dict) -> dict:
    """Summarize all arms, fixed modes and prespecified factorial contrasts without model selection."""
    from ecg_experiment.ssl_synthetic_counterfactual_factorial_common import check_budget

    config = context["config"]
    boot = {"draws": config["bootstrap_draws"], "seed": config["bootstrap_seed"]}
    methods, flags, response = {}, {}, {}
    for arm in (*ARMS, "uniform", "absolute_voltage", "algorithmic_anchor_prior"):
        methods[arm] = {}
        for mode in MODES:
            methods[arm][mode] = {
                name: float(_patient_values(rows, arm, name, mode=mode).mean())
                for name in (
                    "average_precision",
                    "prevalence",
                    "top_one_percent_precision",
                    "top_one_percent_recall",
                )
            }
    for arm in ARMS:
        flags[arm] = {
            kind: {
                "cases": len(selected := [r for r in rows if r["method"] == arm and r["kind"] == kind]),
                "flagged": sum(r["flag"] for r in selected),
                "rate": float(np.mean([r["flag"] for r in selected])),
            }
            for kind in ("positive", "sham", "nuisance")
        }
        flags[arm]["positive_by_mode"] = {
            mode: {
                "cases": 128,
                "flagged": sum(
                    r["flag"]
                    for r in rows
                    if r["method"] == arm and r["kind"] == "positive" and r["mode"] == mode
                ),
            }
            for mode in ("focal", "repeated")
        }
        response[arm] = {}
        for mode in MODES:
            response[arm][mode] = {}
            for name in (*RESPONSE_METRICS, *LEVEL_METRICS):
                check_budget(context, clock)
                values = _patient_values(responses, arm, name, mode=mode)
                response[arm][mode][name] = paired_subject_difference(
                    np.arange(128), values, np.zeros(128), **boot
                )
    contrasts = {}
    for name in (
        "edited_average_precision",
        "baseline_site_average_precision",
        "average_precision_change",
        "selectivity_contrast",
    ):
        contrasts[name] = {}
        for mode in MODES:
            check_budget(context, clock)
            contrasts[name][mode] = factorial_contrasts(
                {arm: _patient_values(responses, arm, name, mode=mode) for arm in ARMS}, config
            )
    control = {
        init: {
            kind: paired_subject_difference(
                np.arange(128),
                _patient_values(rows, init + "P", "flag", kind),
                _patient_values(rows, init + "B", "flag", kind),
                **boot,
            )
            for kind in ("sham", "nuisance")
        }
        for init in ("R", "S")
    }
    result = {
        "methods": methods,
        "record_flags": flags,
        "responses": response,
        "factorial_contrasts": contrasts,
        "paired_loss_control_flags": control,
    }
    result["decision"] = decision(result, config)
    return result


def _after_training(context: dict) -> tuple[dict, dict]:
    from ecg_experiment.ssl_synthetic_counterfactual_factorial_common import PROFILE, TRAINING, verify_context

    reports = {
        "profile": json.loads((PROFILE / "summary.json").read_text()),
        "training": json.loads((TRAINING / "summary.json").read_text()),
    }
    profile, training = reports["profile"], reports["training"]
    if not profile["profile_gate_passed"] or not training["training_completed"]:
        raise ValueError("Complete profiled factorial fits required")
    for name, report in reports.items():
        directory = PROFILE if name == "profile" else TRAINING
        _merge_hashes(context["sources"], report["source_sha256"])
        _merge_hashes(context["inputs"], report["input_sha256"])
        _merge_hashes(
            context["inputs"],
            {to_stored(directory / "summary.json"): sha256_file(directory / "summary.json")},
        )
        _merge_hashes(
            context["inputs"],
            {to_stored(directory / file): digest for file, digest in report["output_sha256"].items()},
        )
    for arm in ARMS:
        report = training["arms"][arm]
        if (
            report["steps"] != 600
            or report["optimizer_step_count"] != 600
            or report["presentations"] != 9600
            or not report["checkpoint_reloaded_exactly"]
        ):
            raise ValueError("Fixed factorial update/exposure recipe changed")
    for arm in ("RB", "SB"):
        old = context["predecessor_training"]["arms"][arm[0]]
        if (
            not training["bce_reproduction"][arm]["passed"]
            or training["arms"][arm]["losses"] != old["losses"]
        ):
            raise ValueError("Exact predecessor BCE loss closure required")
        for name in ("full_model_sha256", "encoder_sha256", "head_sha256"):
            if training["arms"][arm]["final_state"][name] != old["final_state"][name]:
                raise ValueError("Exact predecessor BCE tensor closure required")
    verify_context(context)
    return profile, training


def _load_model(arm: str, training: dict) -> DenseWaveformModel:
    from ecg_experiment.ssl_synthetic_counterfactual_factorial_common import TRAINING

    model = DenseWaveformModel()
    model.load_state_dict(
        torch.load(TRAINING / f"{arm}.pt", weights_only=True, map_location="cpu"), strict=True
    )
    final = training["arms"][arm]["final_state"]
    if (
        state_digest(model.state_dict()) != final["full_model_sha256"]
        or state_digest(model.encoder.state_dict()) != final["encoder_sha256"]
        or state_digest(model.head.state_dict()) != final["head_sha256"]
    ):
        raise ValueError("Loaded factorial checkpoint component state differs")
    return model.to("cuda").eval()


def _infer(
    context: dict, arm: str, indices: list[int], maps: np.ndarray, training: dict, clock: dict, progress: dict
) -> None:
    from ecg_experiment.ssl_synthetic_counterfactual_factorial_common import check_budget

    model = _load_model(arm, training)
    before = state_digest(model.state_dict())
    for first in range(0, len(indices), 16):
        check_budget(context, clock)
        batch = indices[first : first + 16]
        maps[[map_row(index) for index in batch]] = predict_batch(model, context, batch)
        progress["predicted_record_maps"] += len(batch)
    maps.flush()
    if state_digest(model.state_dict()) != before:
        raise ValueError("Factorial inference changed model state")
    del model
    torch.cuda.empty_cache()


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
                response = response_metrics(scores, baseline, target, candidate)
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


def evaluate_factorial() -> dict:
    """Freeze all four A calibrations, then evaluate every new B patient once with final-only states."""
    from ecg_experiment.ssl_synthetic_counterfactual_factorial_common import (
        EVALUATION,
        STUDY,
        admit_context,
        check_budget,
        storage_projection,
        verify_context,
    )
    from ecg_experiment.ssl_synthetic_paired_background_training import _failure, _receipt
    from ecg_experiment.ssl_synthetic_real_background import array_sha256

    if EVALUATION.exists():
        raise FileExistsError("Factorial evaluation destination must be new")
    wall, cpu = time.perf_counter(), time.process_time()
    sources = {name: sha256_file(ROOT / name) for name in SOURCE_NAMES}
    inputs, progress = {}, {"completed": False, "predicted_record_maps": 0, "calibration_frozen": False}
    with gpu_lock("cuda", blocking=False):
        EVALUATION.mkdir(parents=True, exist_ok=False)
        try:
            context = admit_context(SOURCE_NAMES)
            sources, inputs = context["sources"], context["inputs"]
            context["storage_projection"] = storage_projection()
            if (
                context["storage_projection"]["projected_whole_stream_bytes"]
                >= context["config"]["cache_bytes_ceiling"]
            ):
                raise ValueError("Four-map storage forecast fails")
            context["numerical_threads"] = _thread_environment()
            profile, training = _after_training(context)
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
            write_new_json(EVALUATION / "calibration.json", calibration)
            frozen = sha256_file(EVALUATION / "calibration.json")
            progress["calibration_frozen"] = True
            for arm in ARMS:
                if sha256_file(EVALUATION / "calibration.json") != frozen:
                    raise ValueError("Frozen calibration bytes changed before B inference")
                _infer(context, arm, context["roles"]["evaluation"], maps[arm], training, clock, progress)
            if sha256_file(EVALUATION / "calibration.json") != frozen or any(
                array_sha256(maps[arm][:256]) != calibration["calibration_slice_sha256"][arm] for arm in ARMS
            ):
                raise ValueError("Frozen calibration file or one of four slices changed")
            rows, responses = case_metrics(context, maps, calibration["thresholds"], clock)
            write_new_json(EVALUATION / "local_case_metrics.json", {"rows": rows, "responses": responses})
            write_new_json(EVALUATION / "regeneration_checks.json", context["regeneration_checks"])
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
                EVALUATION,
                context,
                clock,
                {
                    "study_id": context["config"]["study_id"],
                    "evaluation_completed": True,
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
            write_new_json(EVALUATION / "summary.json", result)
            return result
        except Exception as error:
            _failure(EVALUATION, sources, inputs, progress, error, wall, cpu)
            raise RuntimeError("Factorial evaluation stopped; inspect its local failure receipt") from None
