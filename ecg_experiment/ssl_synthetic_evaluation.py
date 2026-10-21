"""Frozen single-signal evaluation of the original real-SSL synthetic-localization pilot."""

from __future__ import annotations

import io
import json
import os
import platform
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
import torch
from numpy.lib.format import open_memmap

from ecg_experiment import ROOT
from ecg_experiment.files import sha256_file
from ecg_experiment.gpu import gpu_lock
from ecg_experiment.paths import to_stored
from ecg_experiment.ssl_synthetic_data import build_bank, load_bank, save_bank
from ecg_experiment.ssl_synthetic_generator_qualification import gate_failures
from ecg_experiment.ssl_synthetic_generator_v2 import generate_pair
from ecg_experiment.ssl_synthetic_metrics import (
    control_threshold,
    paired_subject_difference,
    positive_map_metrics,
)
from ecg_experiment.ssl_synthetic_model import state_digest
from ecg_experiment.ssl_synthetic_model_v2 import DenseWaveformModel
from ecg_experiment.ssl_synthetic_preparation import (
    BASE,
    PREPARATION,
    stream_bytes,
    verify_hashes,
    write_new_json,
)

EVALUATION = BASE / "pilot_v1/evaluation"
TRAINING = BASE / "pilot_v1/training"
ARMS = ("R", "S", "F")
EVALUATION_SOURCES = (
    "ecg_experiment/ssl_synthetic_evaluation.py",
    "scripts/reports/evaluate_ssl_synthetic_pilot.py",
    "tests/test_ssl_synthetic_evaluation.py",
    "ecg_experiment/ssl_synthetic_metrics.py",
    "ecg_experiment/intervals.py",
    "ecg_experiment/ssl_synthetic_model_v2.py",
    "docs/ssl-synthetic-localization-pilot-deterministic-output.md",
)


def _merge_hashes(target: dict[str, str], additions: dict[str, str]) -> None:
    for name, digest in additions.items():
        if name in target and target[name] != digest:
            raise ValueError(f"Receipts disagree about frozen artifact: {name}")
        target[name] = digest


def check_training_contract(training: dict[str, Any], config: dict[str, Any]) -> None:
    """Reject partial or unmatched training before any validation or evaluation inference.

    Parameters
    ----------
    training, config : dict
        Completed training receipt and frozen pilot configuration.
    """
    if not training["training_completed"] or training["performance_scored"]:
        raise ValueError("Expected completed training with no prior performance scoring")
    if training["ssl"]["steps"] != config["ssl_steps"] or training["ssl"]["presentations"] != 16000:
        raise ValueError("Incomplete fixed SSL exposure")
    heads = set()
    for arm in ARMS:
        result = training["arms"][arm]
        if result["steps"] != config["localization_steps"] or result["presentations"] != 9600:
            raise ValueError("Incomplete matched localization exposure")
        heads.add(result["initialization"]["head_sha256"])
    if heads != {training["initialization"]["head_sha256"]}:
        raise ValueError("Localization arms did not share fresh head initialization")
    initial_encoder = training["initialization"]["encoder_sha256"]
    ssl_encoder = training["ssl"]["encoder_sha256"]
    if training["arms"]["R"]["initialization"]["encoder_sha256"] != initial_encoder:
        raise ValueError("Random arm did not start from the pre-SSL encoder")
    for arm in ("S", "F"):
        if training["arms"][arm]["initialization"]["encoder_sha256"] != ssl_encoder:
            raise ValueError("Pretrained arms did not start from the final SSL encoder")
    if training["arms"]["F"]["final_state"]["encoder_sha256"] != ssl_encoder:
        raise ValueError("Frozen encoder changed")


def verify_inference_hashes(hashes: dict[str, str]) -> None:
    """Verify all inference artifacts while retaining the closed real-cache payload boundary.

    Parameters
    ----------
    hashes : dict
        Frozen paths and SHA-256 values. The real training waveform cache alone is checked
        against its source-bound cache receipt, without reopening that unused payload.
    """
    cache_signal = to_stored(BASE / "real_cache_v1/signals.npy")
    fresh = {name: value for name, value in hashes.items() if name != cache_signal}
    verify_hashes(fresh)
    if cache_signal in hashes:
        cache = json.loads((BASE / "real_cache_v1/summary.json").read_text())
        if hashes[cache_signal] != cache["output_sha256"]["signals.npy"]:
            raise ValueError("Real-cache provenance digest disagrees; payload was not opened")


def _read_receipt(path: Path) -> dict[str, Any]:
    receipt = json.loads(path.read_text())
    for field in ("source_sha256", "input_sha256", "input_audit_source_sha256"):
        verify_inference_hashes(receipt.get(field, {}))
    outputs = {
        to_stored(path.parent / name): value for name, value in receipt.get("output_sha256", {}).items()
    }
    verify_inference_hashes(outputs)
    return receipt


def _inputs(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    paths = (
        TRAINING / "summary.json",
        PREPARATION / "summary.json",
        BASE / "pilot_v1/profile_v2/summary.json",
    )
    training, preparation, profile = [_read_receipt(path) for path in paths]
    check_training_contract(training, config)
    if not preparation["preparation_gate_passed"] or not profile["profile_gate_passed"]:
        raise ValueError("Preparation/profile gate is not admitted")
    sources = {name: sha256_file(ROOT / name) for name in EVALUATION_SOURCES}
    inputs = {to_stored(path): sha256_file(path) for path in paths}
    for report in (training, preparation, profile):
        _merge_hashes(sources, report.get("source_sha256", {}))
        _merge_hashes(inputs, report.get("input_sha256", {}))
    _merge_hashes(inputs, preparation["bank_sha256"])
    _merge_hashes(
        inputs, {to_stored(TRAINING / name): digest for name, digest in training["output_sha256"].items()}
    )
    verify_inference_hashes(sources | inputs)
    return training, sources, inputs


def _npy_bytes(shape: tuple[int, ...], dtype: str) -> int:
    buffer = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        buffer, {"descr": np.dtype(dtype).str, "fortran_order": False, "shape": shape}
    )
    return buffer.tell() + int(np.prod(shape)) * np.dtype(dtype).itemsize


def storage_projection(config: dict[str, Any]) -> dict[str, Any]:
    """Project every new native map/bank by exact shape and dtype before any model score.

    Parameters
    ----------
    config : dict
        Frozen native grid, arm count and validation/evaluation latent counts.

    Returns
    -------
    dict
        Exact NPY payload/header estimates plus an explicit 64 MiB JSON/report reserve.
    """
    native = (12, config["samples"])
    validation_shape = (4 * config["validation_subjects"], *native)
    evaluation_shape = (4 * config["evaluation_subjects"], *native)
    probability_bytes = 3 * (
        _npy_bytes(validation_shape, "float32") + 2 * _npy_bytes(evaluation_shape, "float32")
    )
    bank_arrays = {
        "signals": (evaluation_shape, "float32"),
        "baselines": (evaluation_shape, "float32"),
        "support": (evaluation_shape, "bool"),
        "prior": (evaluation_shape, "bool"),
        "subjects": ((evaluation_shape[0],), "int64"),
        "kinds": ((evaluation_shape[0],), "<U8"),
        "modes": ((evaluation_shape[0],), "<U10"),
    }
    bank_bytes = 2 * sum(_npy_bytes(shape, dtype) for shape, dtype in bank_arrays.values())
    reserve = 64 * 1024**2
    return {
        "validation_probability_shape": validation_shape,
        "evaluation_probability_shape": evaluation_shape,
        "probability_dtype": "float32",
        "probability_files": 9,
        "probability_npy_bytes": probability_bytes,
        "bank_files": 14,
        "bank_array_specs": bank_arrays,
        "bank_npy_bytes": bank_bytes,
        "json_and_reporting_reserve_bytes": reserve,
        "projected_new_bytes": probability_bytes + bank_bytes + reserve,
    }


def check_budget(budget: dict[str, float], cache_limit: int | None = None) -> None:
    """Enforce total measured wall/CPU ceilings and optional whole-stream storage ceiling.

    Parameters
    ----------
    budget : dict
        Monotonic wall and process CPU deadlines including all prior admitted costs.
    cache_limit : int or None
        Whole-stream byte ceiling; checking file sizes never reads waveform payloads.
    """
    if time.perf_counter() >= budget["wall_deadline"] or time.process_time() >= budget["cpu_deadline"]:
        raise TimeoutError("Frozen pilot evaluation cost ceiling exhausted")
    if cache_limit is not None and stream_bytes() > cache_limit:
        raise ValueError("Frozen whole-stream cache ceiling exhausted")


def configure_inference() -> None:
    """Configure the same deterministic float32 CUDA recipe and one CPU numerical thread."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_final_model(arm: str, training: dict[str, Any], config: dict[str, Any]) -> DenseWaveformModel:
    """Load and verify only the fixed final CPU tensor checkpoint for an arm.

    Parameters
    ----------
    arm : str
        R, S or F, never an intermediate/checkpoint-selected model.
    training, config : dict
        Verified completed training receipt and fixed architecture configuration.

    Returns
    -------
    DenseWaveformModel
        CPU evaluation-mode model after exact final encoder/head digest verification.
    """
    if arm not in ARMS:
        raise ValueError("Unknown fixed arm")
    model = DenseWaveformModel(config["encoder_width"], config["encoder_stride"], config["physical_scale_mv"])
    state = torch.load(TRAINING / f"{arm}.pt", weights_only=True, map_location="cpu")
    model.load_state_dict(state, strict=True)
    expected = training["arms"][arm]["final_state"]
    if (
        state_digest(model.encoder.state_dict()) != expected["encoder_sha256"]
        or state_digest(model.head.state_dict()) != expected["head_sha256"]
    ):
        raise ValueError("Checkpoint component bytes disagree with final-state receipt")
    return model.eval()


def infer_probability_maps(
    model: torch.nn.Module,
    signals: np.ndarray,
    destination: Path,
    budget: dict[str, float],
    batch_size: int = 16,
    device: str = "cuda",
) -> np.ndarray:
    """Infer a native map from observed signals only, retaining every finite float32 probability.

    Parameters
    ----------
    model : torch.nn.Module
        Fixed final single-input model.
    signals : np.ndarray
        Only observed waveform arrays, not a bank containing counterparts/labels/parameters.
    destination : Path
        New local NPY memory-map destination, never overwritten.
    budget : dict
        Shared total cost deadlines.
    batch_size : int
        Frozen batch size.
    device : str
        CUDA for the admitted run; CPU is available only for artificial tests.

    Returns
    -------
    np.ndarray
        Read-only native-grid probability memory map.
    """
    if destination.exists():
        raise FileExistsError(destination)
    if signals.ndim != 3 or signals.shape[1:] != (12, 2500) or signals.dtype != np.float32:
        raise ValueError("Expected observed float32 mV signals on the fixed native grid")
    output = open_memmap(destination, mode="w+", dtype=np.float32, shape=signals.shape)
    model.to(device).eval()
    with torch.inference_mode():
        for start in range(0, len(signals), batch_size):
            check_budget(budget)
            batch = torch.from_numpy(np.array(signals[start : start + batch_size], copy=True)).to(device)
            logits = model(batch)
            if logits.shape != batch.shape or not torch.isfinite(logits).all():
                raise ValueError("Unscorable or nonfinite native model logits")
            output[start : start + batch_size] = torch.sigmoid(logits).cpu().numpy()
    output.flush()
    del output
    model.to("cpu")
    return np.load(destination, mmap_mode="r", allow_pickle=False)


def validate_bank(bank: dict[str, np.ndarray], subjects: int) -> None:
    """Reject missing, empty, mispaired or invalid cases instead of reducing the denominator.

    Parameters
    ----------
    bank : dict
        Synthetic arrays; baseline is used exclusively to audit intervention truth.
    subjects : int
        Fixed count of latent subjects, each with the four declared variants.
    """
    expected_shape = (subjects * 4, 12, 2500)
    if any(bank[name].shape != expected_shape for name in ("signals", "baselines", "support", "prior")):
        raise ValueError("Synthetic evaluation bank lost a case or native coordinate")
    if set(bank["subjects"].tolist()) != set(range(subjects)):
        raise ValueError("Unexpected latent-subject denominator")
    expected = Counter(
        {
            ("positive", "focal"): 1,
            ("positive", "persistent"): 1,
            ("sham", "focal"): 1,
            ("nuisance", "focal"): 1,
        }
    )
    for subject in range(subjects):
        indices = np.flatnonzero(bank["subjects"] == subject)
        observed = Counter(zip(bank["kinds"][indices].tolist(), bank["modes"][indices].tolist(), strict=True))
        if observed != expected:
            raise ValueError("Missing/duplicated subject variant")
    for index, kind in enumerate(bank["kinds"]):
        delta = bank["signals"][index].astype(np.float64) - bank["baselines"][index].astype(np.float64)
        support = bank["support"][index]
        if not np.isfinite(delta).all() or not np.array_equal(support, np.abs(delta) > 0.005):
            raise ValueError("Counterfactual support differs from frozen physical-floor truth")
        positives = int(support.sum())
        if kind == "positive" and not 0 < positives < support.size:
            raise ValueError("Unscorable empty or full positive support")
        if kind != "positive" and (positives or np.any(delta)):
            raise ValueError("Zero-target control is not an identical paired observation")


def fit_calibration(bank: dict[str, np.ndarray], probabilities: dict[str, np.ndarray]) -> dict[str, Any]:
    """Fit only the frozen A-validation control thresholds, keeping strict comparison semantics.

    Parameters
    ----------
    bank : dict
        Validation variants with one sham and nuisance observation per latent subject.
    probabilities : dict
        Native model probabilities by arm on exactly those validation inputs.

    Returns
    -------
    dict
        Per-arm higher empirical 95th-percentile thresholds and full denominator accounting.
    """
    if set(probabilities) != set(ARMS):
        raise ValueError("Calibration requires all three fixed arms")
    thresholds = {}
    for arm, maps in probabilities.items():
        maxima = maps.max(axis=(1, 2))
        controls = {}
        for kind in ("sham", "nuisance"):
            indices = np.flatnonzero(bank["kinds"] == kind)
            indices = indices[np.argsort(bank["subjects"][indices])]
            controls[kind] = maxima[indices]
        sham, nuisance = controls["sham"], controls["nuisance"]
        thresholds[arm] = control_threshold(sham, nuisance)
    return {
        "thresholds": thresholds,
        "validation_subjects": len(sham),
        "quantile": 0.95,
        "method": "higher",
        "record_flag_comparison": "strictly greater",
        "source_profile": "A",
        "checkpoint_selection": "fixed final 600 updates",
    }


def advancement(primary: dict[str, float], controls: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Apply the frozen conjunctive engineering decision without a clinical interpretation.

    Parameters
    ----------
    primary : dict
        B subject-macro S-minus-R AP estimate and paired interval.
    controls : dict
        B sham and nuisance S-minus-R flag-rate differences.

    Returns
    -------
    dict
        Individual rule gates and complete pass/fail decision.
    """
    gates = {
        "ap_gain_at_least_0_05": primary["difference"] >= 0.05,
        "ap_paired_lower_bound_positive": primary["ci_low"] > 0,
        "sham_increase_at_most_0_05": controls["sham"]["difference"] <= 0.05,
        "nuisance_increase_at_most_0_05": controls["nuisance"]["difference"] <= 0.05,
    }
    return {
        "gates": gates,
        "engineering_advancement_passed": all(gates.values()),
        "clinical_promotion": False,
    }


def _method_rows(
    bank: dict[str, np.ndarray],
    probabilities: dict[str, np.ndarray],
    thresholds: dict[str, float],
    profile: str,
    budget: dict[str, float],
) -> list[dict[str, Any]]:
    rows = []
    for index, kind in enumerate(bank["kinds"]):
        check_budget(budget)
        identity = {
            "profile": profile,
            "case": index,
            "subject": int(bank["subjects"][index]),
            "kind": str(kind),
            "mode": str(bank["modes"][index]),
        }
        for arm in ARMS:
            scores = probabilities[arm][index]
            maximum = float(scores.max())
            result = {
                **identity,
                "method": arm,
                "maximum_probability": maximum,
                "flag": bool(maximum > thresholds[arm]),
            }
            if kind == "positive":
                result.update(positive_map_metrics(bank["support"][index], scores))
            rows.append(result)
        if kind != "positive":
            continue
        contextual = {
            "uniform": np.zeros((12, 2500), dtype=np.float32),
            "absolute_voltage": np.abs(bank["signals"][index]),
            "oracle_phase_prior": bank["prior"][index].astype(np.float32),
        }
        rows.extend(
            {**identity, "method": name, **positive_map_metrics(bank["support"][index], scores)}
            for name, scores in contextual.items()
        )
    return rows


def _macro_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positive = [row for row in rows if row["kind"] == "positive"]
    metric_names = ("average_precision", "prevalence", "top_one_percent_precision", "top_one_percent_recall")
    summaries = {}
    for method in (*ARMS, "uniform", "absolute_voltage", "oracle_phase_prior"):
        method_rows = [row for row in positive if row["method"] == method]
        modes = {}
        for mode in ("all", "focal", "persistent"):
            selected = [row for row in method_rows if mode == "all" or row["mode"] == mode]
            subjects = sorted({row["subject"] for row in selected})
            metrics = {
                name: float(
                    np.mean(
                        [
                            np.mean([row[name] for row in selected if row["subject"] == subject])
                            for subject in subjects
                        ]
                    )
                )
                for name in metric_names
            }
            modes[mode] = {"subjects": len(subjects), "cases": len(selected), **metrics}
        summaries[method] = modes
    return summaries


def _paired_vectors(
    rows: list[dict[str, Any]], kind: str, metric: str, mode: str | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = [row for row in rows if row["kind"] == kind and (mode is None or row["mode"] == mode)]
    arms = {
        arm: sorted(
            (row for row in selected if row["method"] == arm), key=lambda row: (row["subject"], row["case"])
        )
        for arm in ("S", "R")
    }
    if not arms["S"] or [(row["subject"], row["case"]) for row in arms["S"]] != [
        (row["subject"], row["case"]) for row in arms["R"]
    ]:
        raise ValueError("Paired method denominators disagree")
    return (
        np.array([row["subject"] for row in arms["S"]]),
        np.array([row[metric] for row in arms["S"]], dtype=float),
        np.array([row[metric] for row in arms["R"]], dtype=float),
    )


def summarize_profile(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Aggregate equally weighted within-subject variants and shared paired bootstrap contrasts.

    Parameters
    ----------
    rows : list of dict
        All per-case/method outputs for one frozen engineering profile.
    config : dict
        Bootstrap draw count and seed committed before scores.

    Returns
    -------
    dict
        Full-map method metrics, mode-specific AP contrasts and recording-flag diagnostics.
    """
    boot = {"draws": config["bootstrap_draws"], "seed": config["bootstrap_seed"]}
    differences = {
        mode: paired_subject_difference(
            *_paired_vectors(rows, "positive", "average_precision", None if mode == "all" else mode), **boot
        )
        for mode in ("all", "focal", "persistent")
    }
    controls = {
        kind: paired_subject_difference(*_paired_vectors(rows, kind, "flag"), **boot)
        for kind in ("sham", "nuisance")
    }
    flags = {}
    for arm in ARMS:
        flags[arm] = {}
        for kind in ("sham", "nuisance", "positive"):
            selected = [row for row in rows if row["method"] == arm and row["kind"] == kind]
            flags[arm][kind] = {
                "cases": len(selected),
                "flagged": sum(row["flag"] for row in selected),
                "rate": float(np.mean([row["flag"] for row in selected])),
            }
        flags[arm]["positive_by_mode"] = {
            mode: float(
                np.mean(
                    [
                        row["flag"]
                        for row in rows
                        if row["method"] == arm and row["kind"] == "positive" and row["mode"] == mode
                    ]
                )
            )
            for mode in ("focal", "persistent")
        }
    return {
        "methods": _macro_metrics(rows),
        "s_minus_r_average_precision": differences,
        "s_minus_r_control_flag_rates": controls,
        "record_flags": flags,
    }


def _infer_arms(
    bank: dict[str, np.ndarray],
    profile: str,
    training: dict[str, Any],
    config: dict[str, Any],
    budget: dict[str, float],
    progress: dict[str, Any],
) -> dict[str, np.ndarray]:
    maps = {}
    for arm in ARMS:
        model = load_final_model(arm, training, config)
        maps[arm] = infer_probability_maps(
            model,
            bank["signals"],
            EVALUATION / f"{profile}_prob_{arm}.npy",
            budget,
            batch_size=config["batch_size"],
        )
        progress["predicted_record_maps"] += len(bank["signals"])
        check_budget(budget, config["cache_bytes_ceiling"])
        del model
        torch.cuda.empty_cache()
    return maps


def _run(
    training: dict[str, Any],
    config: dict[str, Any],
    sources: dict[str, str],
    inputs: dict[str, str],
    budget: dict[str, float],
    progress: dict[str, Any],
) -> dict[str, Any]:
    validation = load_bank(PREPARATION / "validation_bank")
    validate_bank(validation, config["validation_subjects"])
    validation_maps = _infer_arms(validation, "validation", training, config, budget, progress)
    calibration = {
        **fit_calibration(validation, validation_maps),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_sha256": sources,
        "input_sha256": inputs,
        "validation_probability_sha256": {
            arm: sha256_file(EVALUATION / f"validation_prob_{arm}.npy") for arm in ARMS
        },
    }
    calibration_path = EVALUATION / "calibration.json"
    write_new_json(calibration_path, calibration)
    calibration_digest = sha256_file(calibration_path)
    progress["calibration_frozen"] = True
    del validation_maps, validation
    gates = json.loads((ROOT / "configs/ssl_synthetic_generator_v1.json").read_text())
    profiles, rows = {}, []
    for profile in ("B", "A"):
        check_budget(budget, config["cache_bytes_ceiling"])
        if sha256_file(calibration_path) != calibration_digest:
            raise ValueError("Frozen validation calibration changed before engineering inference")
        bank = build_bank(
            "evaluation",
            config["evaluation_subjects"],
            profile,
            generate_pair,
            lambda pair: gate_failures(pair, gates),
        )
        validate_bank(bank, config["evaluation_subjects"])
        save_bank(bank, EVALUATION / f"{profile}_bank")
        probabilities = _infer_arms(bank, profile, training, config, budget, progress)
        profile_rows = _method_rows(bank, probabilities, calibration["thresholds"], profile, budget)
        write_new_json(EVALUATION / f"{profile}_local_case_metrics.json", {"rows": profile_rows})
        progress["scored_positive_maps"] += sum(row["kind"] == "positive" for row in profile_rows)
        profiles[profile] = summarize_profile(profile_rows, config)
        rows.extend(profile_rows)
        progress["completed_profiles"].append(profile)
        del bank, probabilities
    if sha256_file(calibration_path) != calibration_digest:
        raise ValueError("Frozen calibration file changed")
    return {
        "calibration_sha256": calibration_digest,
        "calibration": calibration,
        "profiles": profiles,
        "case_method_rows": len(rows),
        "decision": advancement(
            profiles["B"]["s_minus_r_average_precision"]["all"], profiles["B"]["s_minus_r_control_flag_rates"]
        ),
    }


def evaluate() -> dict[str, Any]:
    """Execute the reviewed fixed evaluation once, retaining partial-run failures and native maps.

    Returns
    -------
    dict
        Source/input/output-bound engineering result; no clinical promotion is permitted.
    """
    if EVALUATION.exists() or EVALUATION.is_symlink():
        raise FileExistsError("Evaluation directory must be new; failed runs cannot be silently resumed")
    wall, cpu = time.perf_counter(), time.process_time()
    sources = {name: sha256_file(ROOT / name) for name in EVALUATION_SOURCES}
    receipt_paths = (
        TRAINING / "summary.json",
        PREPARATION / "summary.json",
        BASE / "pilot_v1/profile_v2/summary.json",
    )
    inputs = {to_stored(path): sha256_file(path) for path in receipt_paths if path.is_file()}
    progress = {
        "predicted_record_maps": 0,
        "scored_positive_maps": 0,
        "calibration_frozen": False,
        "completed_profiles": [],
    }
    with gpu_lock("cuda", blocking=False):
        EVALUATION.mkdir(parents=True, exist_ok=False)
        try:
            config = json.loads((ROOT / "configs/ssl_synthetic_pilot_v1.json").read_text())
            training, sources, inputs = _inputs(config)
            budget = {
                "wall_deadline": wall
                + config["gpu_seconds_ceiling"]
                - training["profile_wall_seconds"]
                - training["wall_seconds"],
                "cpu_deadline": cpu
                + config["cpu_seconds_ceiling"]
                - training["prior_cpu_seconds"]
                - training["cpu_seconds"],
            }
            check_budget(budget, config["cache_bytes_ceiling"])
            forecast = storage_projection(config)
            forecast["existing_stream_bytes"] = stream_bytes()
            if (
                forecast["existing_stream_bytes"] + forecast["projected_new_bytes"]
                > config["cache_bytes_ceiling"]
            ):
                raise ValueError("Full bank/map/JSON storage projection exceeds the fixed cache ceiling")
            configure_inference()
            torch.cuda.reset_peak_memory_stats()
            admission = {
                "created_at_utc": datetime.now(UTC).isoformat(),
                "source_sha256": sources,
                "input_sha256": inputs,
                "training_complete": True,
                "selection": "fixed final 600-update checkpoint",
                "calibration_before_B": True,
                "storage_projection": forecast,
            }
            write_new_json(EVALUATION / "admission.json", admission)
            result = _run(training, config, sources, inputs, budget, progress)
            verify_inference_hashes(sources | inputs)
            check_budget(budget, config["cache_bytes_ceiling"])
            output_hashes = {
                path.relative_to(EVALUATION).as_posix(): sha256_file(path)
                for path in sorted(EVALUATION.rglob("*"))
                if path.is_file()
            }
            check_budget(budget, config["cache_bytes_ceiling"])
        except Exception as error:
            write_new_json(
                EVALUATION / "failure.json",
                {
                    "created_at_utc": datetime.now(UTC).isoformat(),
                    "evaluation_completed": False,
                    "source_sha256": sources,
                    "input_sha256": inputs,
                    "progress": progress,
                    "failure_type": type(error).__name__,
                    "failure_detail": str(error),
                    "wall_seconds": time.perf_counter() - wall,
                    "cpu_seconds": time.process_time() - cpu,
                },
            )
            raise
        receipt = {
            "schema_version": 1,
            "pilot_id": config["pilot_id"],
            "evaluation_completed": True,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "source_sha256": sources,
            "input_sha256": inputs,
            "progress": progress,
            **result,
            "wall_seconds": time.perf_counter() - wall,
            "cpu_seconds": time.process_time() - cpu,
            "total_profile_training_evaluation_wall_seconds": training["profile_wall_seconds"]
            + training["wall_seconds"]
            + time.perf_counter()
            - wall,
            "total_preparation_training_evaluation_cpu_seconds": training["prior_cpu_seconds"]
            + training["cpu_seconds"]
            + time.process_time()
            - cpu,
            "stream_bytes": stream_bytes(),
            "storage_projection": forecast,
            "clinical_promotion": False,
            "closed_test_opened": False,
            "real_training_waveform_cache_reopened": False,
            "real_training_waveform_digest_verification": "frozen cache receipt only; no payload read",
            "environment": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "sklearn": sklearn.__version__,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(0),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "threads": {
                    name: os.environ.get(name)
                    for name in (
                        "OPENBLAS_NUM_THREADS",
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS",
                    )
                },
            },
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "output_sha256": output_hashes,
        }
        write_new_json(EVALUATION / "summary.json", receipt)
        return receipt
