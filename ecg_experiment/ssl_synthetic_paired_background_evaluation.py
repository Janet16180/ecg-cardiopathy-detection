"""Evaluate matched-control-trained models on the fixed introduced-change endpoint."""

from __future__ import annotations

import os
import subprocess
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
import torch
from numpy.lib.format import open_memmap

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.paths import to_stored
from ecg_experiment.ssl_synthetic_counterfactual import response_metrics
from ecg_experiment.ssl_synthetic_evaluation import (
    _merge_hashes,
    _read_receipt,
    configure_inference,
    verify_inference_hashes,
)
from ecg_experiment.ssl_synthetic_metrics import (
    control_threshold,
    paired_subject_difference,
    positive_map_metrics,
)
from ecg_experiment.ssl_synthetic_model import state_digest
from ecg_experiment.ssl_synthetic_model_v2 import DenseWaveformModel
from ecg_experiment.ssl_synthetic_paired_background_common import (
    EVALUATION,
    PROFILE,
    TRAINING,
    admit_context,
    check_budget,
    render_checked,
    storage_projection,
)
from ecg_experiment.ssl_synthetic_preparation import stream_bytes, write_new_json
from ecg_experiment.ssl_synthetic_real_background import array_sha256
from ecg_experiment.ssl_synthetic_real_transfer import _paired, summarize

ARMS = ("R", "S")
SOURCE_NAMES = (
    "ecg_experiment/ssl_synthetic_paired_background_evaluation.py",
    "scripts/reports/evaluate_ssl_synthetic_paired_background.py",
    "tests/test_ssl_synthetic_paired_background_evaluation.py",
    "ecg_experiment/ssl_synthetic_real_transfer.py",
    "ecg_experiment/ssl_synthetic_counterfactual.py",
    "ecg_experiment/ssl_synthetic_metrics.py",
    "ecg_experiment/intervals.py",
    "ecg_experiment/ssl_synthetic_evaluation.py",
    "ecg_experiment/ssl_synthetic_model.py",
    "ecg_experiment/ssl_synthetic_model_v2.py",
)


def map_row(global_case: int) -> int:
    """Translate only admitted calibration/evaluation cases to the compact saved-map grid."""
    if not isinstance(global_case, (int, np.integer)) or not 2000 <= global_case < 2768:
        raise ValueError("Only global calibration/evaluation cases 2000 through 2767 have saved maps")
    return int(global_case) - 2000


def predict_batch(model: torch.nn.Module, context: dict, indices: list[int]) -> np.ndarray:
    """Run the observed-waveform-only inference path shared by the resource profile and evaluation."""
    signals = np.stack([render_checked(context, index)["edited"] for index in indices])
    device = next(model.parameters()).device
    with torch.inference_mode():
        logits = model(torch.from_numpy(signals).to(device))
        if (
            tuple(logits.shape) != signals.shape
            or logits.dtype != torch.float32
            or not torch.isfinite(logits).all()
        ):
            raise ValueError("Native logits must have finite float32 waveform geometry")
        values = logits.sigmoid().cpu().numpy()
    if values.shape != signals.shape or values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError("Native probability geometry, dtype or finiteness changed")
    return values


def calibration_thresholds(cases: list[dict], maps: dict[str, np.ndarray]) -> dict[str, float]:
    """Fit higher95 thresholds from exactly64 paired calibration controls, never evaluation maps."""
    controls = {
        kind: sorted(
            (c for c in cases if c["role"] == "calibration" and c["kind"] == kind),
            key=lambda c: c["role_subject_index"],
        )
        for kind in ("sham", "nuisance")
    }
    if any([c["role_subject_index"] for c in rows] != list(range(64)) for rows in controls.values()):
        raise ValueError("Every calibration patient must retain both control variants")
    return {
        arm: control_threshold(
            np.array([maps[arm][map_row(c["case_index"])].max() for c in controls["sham"]]),
            np.array([maps[arm][map_row(c["case_index"])].max() for c in controls["nuisance"]]),
        )
        for arm in ARMS
    }


def decision(summary: dict, config: dict) -> dict[str, Any]:
    """Apply the complete prospective response, noninferiority and control conjunction."""
    primary = summary["s_minus_r_response"]["all"]
    gates = {
        "response_gain_at_least_minimum": primary["difference"] >= config["primary_response_gain_minimum"],
        "response_gain_lower_bound_positive": primary["ci_low"] > 0,
        "edited_ap_noninferiority": summary["s_minus_r_ap"]["all"]["ci_low"]
        > -config["edited_ap_noninferiority_margin"],
    }
    for arm in ARMS:
        for metric in ("average_precision_change", "selectivity_contrast"):
            gates[f"{arm}_{metric}_lower_bound_positive"] = (
                summary["counterfactual_response"][arm]["all"][metric]["ci_low"] > 0
            )
    for kind in ("sham", "nuisance"):
        gates[f"{kind}_point_increase_bounded"] = (
            summary["s_minus_r_control_flags"][kind]["difference"] <= config["control_point_increase_maximum"]
        )
    return {
        "gates": gates,
        "engineering_replication_admitted": all(gates.values()),
        "clinical_promotion": False,
        "scope": "Pooled counterfactual rank response, one training seed",
    }


def summarize_paired(rows: list[dict], responses: list[dict], config: dict) -> dict[str, Any]:
    """Reuse frozen metric conventions and add the new paired response primary and decision."""
    summary = summarize(rows, responses, config)
    summary["s_minus_r_response"] = {
        mode: paired_subject_difference(
            *_paired(responses, "S", "R", "average_precision_change", "positive", mode),
            draws=config["bootstrap_draws"],
            seed=config["bootstrap_seed"],
        )
        for mode in ("all", "focal", "repeated")
    }
    summary["new_advancement_rule"] = "paired-background protocol response/noninferiority/control conjunction"
    summary["decision"] = decision(summary, config)
    return summary


def _load_model(arm: str, training: dict, config: dict) -> DenseWaveformModel:
    model = DenseWaveformModel(config["encoder_width"], config["encoder_stride"], config["physical_scale_mv"])
    path = TRAINING / f"{arm}.pt"
    if sha256_file(path) != training["arms"][arm]["checkpoint_sha256"]:
        raise ValueError("Final trained checkpoint bytes changed")
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    expected = training["arms"][arm]["final_state"]
    if (
        state_digest(model.encoder.state_dict()) != expected["encoder_sha256"]
        or state_digest(model.head.state_dict()) != expected["head_sha256"]
    ):
        raise ValueError("Final checkpoint components differ from completed training")
    return model.eval()


def _infer(
    context: dict, training: dict, arm: str, indices: list[int], maps: np.ndarray, clock: dict, progress: dict
) -> None:
    model = _load_model(arm, training, context["config"]).to("cuda")
    before = state_digest(model.state_dict())
    for start in range(0, len(indices), 16):
        check_budget(context, clock)
        batch = indices[start : start + 16]
        maps[[map_row(i) for i in batch]] = predict_batch(model, context, batch)
        progress["predicted_record_maps"] += len(batch)
    maps.flush()
    if state_digest(model.state_dict()) != before:
        raise ValueError("Final model changed during inference")
    model.to("cpu")
    del model
    torch.cuda.empty_cache()


def _paired_control(context: dict, case: dict) -> tuple[dict, dict]:
    index = case["cache_index"] * 4 + (3 if case["nuisance"] else 2)
    control = context["cases"][index]
    if control["patient_id"] != case["patient_id"] or control["role"] != case["role"]:
        raise ValueError("Exact counterpart patient/role changed")
    edited = render_checked(context, case["case_index"])
    baseline = render_checked(context, index)
    if not np.array_equal(edited["baseline"], baseline["edited"]):
        raise ValueError("Positive baseline is not its observed matching control")
    return edited, control


def case_metrics(
    context: dict, maps: dict[str, np.ndarray], thresholds: dict, clock: dict
) -> tuple[list, list]:
    """Score all fixed cases using global masks and explicitly offset probability rows."""
    rows, responses = [], []
    for index in context["roles"]["evaluation"]:
        check_budget(context, clock)
        case = context["cases"][index]
        identity = {
            "case": index,
            "map_row": map_row(index),
            "subject": case["role_subject_index"],
            "global_subject": case["cache_index"],
            "kind": case["kind"],
            "mode": case["mode"],
        }
        for arm in ARMS:
            scores = maps[arm][map_row(index)]
            row = {
                **identity,
                "method": arm,
                "maximum_probability": float(scores.max()),
                "flag": bool(scores.max() > thresholds[arm]),
            }
            if case["target"]:
                row.update(positive_map_metrics(context["support"][index], scores))
            rows.append(row)
        if not case["target"]:
            continue
        pair, control = _paired_control(context, case)
        contextual = {
            "uniform": np.zeros((12, 2500), np.float32),
            "absolute_voltage": np.abs(pair["edited"]),
            "algorithmic_anchor_prior": context["candidate"][index].astype(np.float32),
        }
        rows.extend(
            {**identity, "method": method, **positive_map_metrics(context["support"][index], scores)}
            for method, scores in contextual.items()
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
                        maps[arm][map_row(index)],
                        maps[arm][map_row(control["case_index"])],
                    ),
                }
            )
    return rows, responses


def _after_training(context: dict) -> tuple[dict, dict]:
    reports = {
        name: _read_receipt(path)
        for name, path in (("training", TRAINING / "summary.json"), ("profile", PROFILE / "summary.json"))
    }
    if not reports["training"]["training_completed"] or not reports["profile"]["profile_gate_passed"]:
        raise ValueError("Completed matched training and actual profile are required")
    for arm in ARMS:
        receipt = reports["training"]["arms"][arm]
        if (
            receipt["steps"] != 600
            or receipt["presentations"] != 9600
            or receipt["optimizer_step_count"] != 600
        ):
            raise ValueError("Arm has incomplete scheduled updates/presentations")
    for name, report in reports.items():
        _merge_hashes(context["sources"], report["source_sha256"])
        _merge_hashes(context["inputs"], report["input_sha256"])
        directory = TRAINING if name == "training" else PROFILE
        _merge_hashes(
            context["inputs"],
            {to_stored(directory / "summary.json"): sha256_file(directory / "summary.json")},
        )
        _merge_hashes(
            context["inputs"],
            {to_stored(directory / file): digest for file, digest in report.get("output_sha256", {}).items()},
        )
    verify_inference_hashes(context["sources"] | context["inputs"])
    return reports["training"], reports["profile"]


def evaluate_paired_background() -> dict[str, Any]:
    """Run one fixed evaluation, freezing calibration before B and preserving every failure."""
    if EVALUATION.exists():
        raise FileExistsError("Paired evaluation output must be new")
    wall, cpu = time.perf_counter(), time.process_time()
    sources = {name: sha256_file(ROOT / name) for name in SOURCE_NAMES}
    inputs = {}
    progress = {"predicted_record_maps": 0, "calibration_frozen": False, "completed": False}
    with gpu_lock("cuda", blocking=False):
        EVALUATION.mkdir(parents=True, exist_ok=False)
        try:
            required = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
            if any(os.environ.get(name) != "1" for name in required):
                raise ValueError("Use one numerical CPU thread")
            context = admit_context(SOURCE_NAMES)
            sources, inputs = context["sources"], context["inputs"]
            training, profile = _after_training(context)
            storage = storage_projection()
            if storage["projected_whole_stream_bytes"] > context["config"]["cache_bytes_ceiling"]:
                raise ValueError("Whole-study storage forecast fails")
            configure_inference()
            torch.cuda.reset_peak_memory_stats()
            clock = {
                "wall": wall,
                "cpu": cpu,
                "prior_stage_wall": profile["wall_seconds"] + training["wall_seconds"],
                "prior_stage_cpu": context["stage_preparation_cpu_seconds"]
                + profile["cpu_seconds"]
                + training["cpu_seconds"],
            }
            check_budget(context, clock)
            write_new_json(
                EVALUATION / "admission.json",
                {"source_sha256": sources, "input_sha256": inputs, "storage_projection": storage},
            )
            maps = {
                arm: open_memmap(
                    EVALUATION / f"probabilities_{arm}.npy",
                    mode="w+",
                    dtype=np.float32,
                    shape=(768, 12, 2500),
                )
                for arm in ARMS
            }
            for arm in ARMS:
                _infer(context, training, arm, context["roles"]["calibration"], maps[arm], clock, progress)
            calibration = {
                "thresholds": calibration_thresholds(context["cases"], maps),
                "patients": 64,
                "method": "higher",
                "quantile": 0.95,
                "comparison": "strictly greater",
                "global_calibration_case_indices": context["roles"]["calibration"],
                "probability_row_offset": 2000,
                "calibration_slice_sha256": {arm: array_sha256(maps[arm][:256]) for arm in ARMS},
                "source_sha256": sources,
                "input_sha256": inputs,
                "created_at_utc": datetime.now(UTC).isoformat(),
            }
            write_new_json(EVALUATION / "calibration.json", calibration)
            frozen = sha256_file(EVALUATION / "calibration.json")
            progress["calibration_frozen"] = True
            for arm in ARMS:
                if sha256_file(EVALUATION / "calibration.json") != frozen:
                    raise ValueError("Frozen calibration bytes changed before B")
                _infer(context, training, arm, context["roles"]["evaluation"], maps[arm], clock, progress)
            if sha256_file(EVALUATION / "calibration.json") != frozen or any(
                array_sha256(maps[arm][:256]) != calibration["calibration_slice_sha256"][arm] for arm in ARMS
            ):
                raise ValueError("Calibration bytes/slices changed during B inference")
            rows, responses = case_metrics(context, maps, calibration["thresholds"], clock)
            write_new_json(
                EVALUATION / "local_case_metrics.json", {"rows": rows, "counterfactual_rows": responses}
            )
            write_new_json(EVALUATION / "regeneration_checks.json", context["regeneration_checks"])
            summary = summarize_paired(rows, responses, context["config"])
            verify_inference_hashes(sources | inputs)
            check_budget(context, clock)
            outputs = {p.name: sha256_file(p) for p in EVALUATION.iterdir() if p.is_file()}
            check_budget(context, clock)
            elapsed, used = time.perf_counter() - wall, time.process_time() - cpu
            progress["completed"] = True
            result = {
                "evaluation_completed": True,
                "study_id": context["config"]["study_id"],
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
                "probability_global_case_offset": 2000,
                "unique_regenerated_cases": len(context["regeneration_checks"]),
                "regenerated_hash_checks": 4 * len(context["regeneration_checks"]),
                "all_regenerated_hashes_match": True,
                "storage_projection": storage,
                **summary,
                "wall_seconds": elapsed,
                "cpu_seconds": used,
                "stage_gpu_wall_seconds": clock["prior_stage_wall"] + elapsed,
                "stage_cpu_seconds": clock["prior_stage_cpu"] + used,
                "global_gpu_wall_seconds": context["prior_gpu_wall_seconds"]
                + clock["prior_stage_wall"]
                + elapsed,
                "global_cpu_seconds": context["prior_cpu_seconds"] + clock["prior_stage_cpu"] + used,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "stream_bytes": stream_bytes(),
                "original_advancement_passed": False,
                "clinical_claim": False,
                "new_training_updates": 0,
                "torch_threads": torch.get_num_threads(),
                "torch_interop_threads": torch.get_num_interop_threads(),
            }
            write_new_json(EVALUATION / "summary.json", result)
            return result
        except Exception as error:
            write_new_json(
                EVALUATION / "failure.json",
                {
                    "source_sha256": sources,
                    "input_sha256": inputs,
                    "progress": progress,
                    "failure_type": type(error).__name__,
                    "failure_detail": str(error),
                    "wall_seconds": time.perf_counter() - wall,
                    "cpu_seconds": time.process_time() - cpu,
                },
            )
            raise RuntimeError("Paired evaluation stopped; inspect its local failure receipt") from None
