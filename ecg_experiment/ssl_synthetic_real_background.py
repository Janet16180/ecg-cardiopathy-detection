"""Deterministic native-grid introduced-voltage edits on preserved real ECG backgrounds."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt

from ecg_experiment.ssl_synthetic_generator import couple_limb_leads, edit_profile


def array_sha256(array: np.ndarray) -> str:
    """Hash contiguous array bytes; dtype and shape are pinned separately."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _lowest(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = {}
    for row in rows:
        patient = row["patient_id"]
        if patient not in grouped or int(row["ecg_id"]) < int(grouped[patient]["ecg_id"]):
            grouped[patient] = row
    return list(grouped.values())


def select_patients(
    records: list[dict[str, Any]],
    audited: list[dict[str, Any]],
    ssl_patients: set[str],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Reserve patients using only fixed original identity/quality metadata, without payload access.

    Parameters
    ----------
    records, audited : list of dict
        Immutable partition and fully source-bound input-audit rows.
    ssl_patients : set of str
        Every original selected pretraining patient, independent of actual presentations.
    config : dict
        Frozen selection counts.

    Returns
    -------
    list of dict
        All reserved TRAIN and unfiltered validation selections, retaining their original fields.
    """
    audit = {row["ecg_id"]: row for row in audited}
    train = [
        row
        for row in records
        if row["role"] == "train"
        and row["patient_id"] not in ssl_patients
        and audit[row["ecg_id"]]["train_optimization_eligible"] is True
    ]
    validation = [row for row in records if row["role"] == "validation"]
    selections = {}
    for role, rows in (("train", train), ("validation", validation)):
        selections[role] = sorted(
            _lowest(rows),
            key=lambda row: (
                hashlib.sha256(f"ssl_real_background_v1:{role}:{row['patient_id']}".encode()).hexdigest(),
                int(row["patient_id"]),
            ),
        )
    nt, nc, ne = (
        config[key] for key in ("train_reserved_patients", "calibration_patients", "evaluation_patients")
    )
    if len(selections["train"]) < nt or len(selections["validation"]) < nc + ne:
        raise ValueError("Insufficient source patients for frozen selection")
    groups = (
        ("train_reserved", selections["train"][:nt]),
        ("calibration", selections["validation"][:nc]),
        ("evaluation", selections["validation"][nc : nc + ne]),
    )
    output = []
    for role, rows in groups:
        for local, row in enumerate(rows):
            observed = audit[row["ecg_id"]]
            index = None if role == "train_reserved" else local + (nc if role == "evaluation" else 0)
            output.append(
                dict(
                    row,
                    study_role=role,
                    cache_index=index,
                    subject_index=index,
                    role_subject_index=local,
                    audit_payload_sha256=observed["payload_sha256"],
                    quality_exclusions=observed["quality_exclusions"],
                    review_flags=observed["review_flags"],
                )
            )
    if len({row["patient_id"] for row in output}) != len(output):
        raise ValueError("Patient overlaps new study roles")
    if any(row["patient_id"] in ssl_patients for row in output):
        raise ValueError("New study overlaps original SSL patients")
    return output


def find_anchors(background: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    """
    Apply the exact frozen anchor-only filter, normalized energy and refinement rules.

    Parameters
    ----------
    background : np.ndarray
        Unedited processed float32 mV signal of shape (12,2500).
    config : dict
        Frozen scipy filter and peak rules.

    Returns
    -------
    dict
        All count stages and final algorithmic anchor indices; no physiological interpretation.
    """
    if background.shape != (12, 2500) or not np.isfinite(background).all():
        raise ValueError("Invalid processed background for anchoring")
    sos = butter(
        config["anchor_filter_order"],
        config["anchor_band_hz"],
        btype="bandpass",
        fs=config["sample_rate"],
        output="sos",
    )
    filtered = sosfiltfilt(sos, background.astype(np.float64), axis=1)
    differences = np.diff(filtered, axis=1, prepend=filtered[:, :1])
    rms = np.maximum(np.sqrt(np.mean(differences**2, axis=1)), config["anchor_rms_floor_mv"])
    energy = np.median((differences / rms[:, None]) ** 2, axis=0)
    width = config["anchor_smoothing_samples"]
    smooth = np.convolve(energy, np.ones(width) / width, mode="same")
    median = np.median(smooth)
    height = median + config["anchor_threshold_mad_multiple"] * np.median(np.abs(smooth - median))
    peaks, _ = find_peaks(smooth, height=height, distance=config["anchor_minimum_distance_samples"])
    absolute = np.abs(filtered).sum(axis=0)
    radius = config["anchor_refinement_samples"]
    refined = sorted(
        {
            max(0, int(p) - radius)
            + int(np.argmax(absolute[max(0, int(p) - radius) : min(len(absolute), int(p) + radius + 1)]))
            for p in peaks
        }
    )
    low = int(config["anchor_retained_start_seconds"] * config["sample_rate"])
    high = int(config["anchor_retained_stop_seconds"] * config["sample_rate"])
    retained = [p for p in refined if low <= p <= high]
    count = len(retained)
    if count > config["maximum_anchors"]:
        positions = np.rint(np.linspace(0, count - 1, config["maximum_anchors"])).astype(int)
        retained = [retained[position] for position in positions]
    return {
        "candidate_peaks": len(peaks),
        "refined_unique": len(refined),
        "retained_before_cap": count,
        "anchors": retained,
        "minimum_anchor_gate_passed": len(retained) >= config["minimum_anchors"],
    }


def _nuisance_parameters(seed: int) -> tuple[dict[str, Any], np.random.Generator]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 303]))
    return {
        "gain": float(rng.uniform(0.8, 1.2)),
        "noise_sd_mv": float(rng.uniform(0, 0.025)),
        "drift_amplitude_mv": float(rng.uniform(0, 0.12)),
        "drift_hz": float(rng.uniform(0.05, 0.40)),
        "phases": rng.uniform(0, 2 * np.pi, size=8).tolist(),
    }, rng


def _edit_parameters(seed: int, anchors: list[int], config: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 101]))
    pattern = int(rng.integers(0, 4))
    chest_count = int(rng.integers(1, 4)) if pattern == 0 else 1
    chest = rng.choice(np.arange(2, 8), size=chest_count, replace=False)
    coefficients = np.zeros(8)
    coefficients[:2] = [(0, 0), (1, 0), (0, 1), (1, 1)][pattern]
    coefficients[chest] = 1
    magnitude = float(rng.uniform(*config["edit_amplitude_mv"]))
    sign = float(rng.choice([-1.0, 1.0]))
    width = float(rng.uniform(*config["edit_width_seconds"]))
    focal = int(rng.integers(len(anchors))) if anchors else None
    return {
        "pattern": pattern,
        "primitive_coefficients": coefficients.tolist(),
        "magnitude_mv": magnitude,
        "sign": sign,
        "amplitude_mv": magnitude * sign,
        "width_seconds": width,
        "focal_anchor_position": focal,
    }


def cases_for_background(
    row: dict[str, Any], anchors: list[int], background_hash: str, config: dict[str, Any]
) -> list[dict[str, Any]]:
    """
    Freeze four label-only cases with common patient edit/nuisance streams.

    Parameters
    ----------
    row : dict
        Selected calibration/evaluation identity row.
    anchors : list of int
        Frozen algorithmic indices from the unedited background.
    background_hash : str
        Processed float32 cache-byte identity.
    config : dict
        Frozen edit/support parameters.

    Returns
    -------
    list of dict
        Focal, repeated, unchanged and nuisance cases in that order.
    """
    role = row["study_role"]
    if row["role"] != "validation" or role not in ("calibration", "evaluation"):
        raise ValueError("Only selected validation study roles may form cases")
    key = f"ssl_real_background_v1:{role}:{row['patient_id']}"
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    edit, nuisance = _edit_parameters(seed, anchors, config), _nuisance_parameters(seed)[0]
    output = []
    for variant in ("focal", "repeated", "sham", "nuisance"):
        positive = variant in ("focal", "repeated")
        output.append(
            {
                "case_index": 4 * row["cache_index"] + len(output),
                "subject": row["subject_index"],
                "subject_index": row["subject_index"],
                "role_subject_index": row["role_subject_index"],
                "role": role,
                "profile": "A" if role == "calibration" else "B",
                "cache_index": row["cache_index"],
                "patient_id": row["patient_id"],
                "variant": variant,
                "kind": "positive" if positive else variant,
                "mode": variant if positive else "none",
                "target": positive,
                "nuisance": bool(seed & 1) if positive else variant == "nuisance",
                "seed": seed,
                "anchors": anchors,
                "edit_parameters": edit,
                "nuisance_parameters": nuisance,
                "processed_background_sha256": background_hash,
                "support_floor_mv": config["support_floor_mv"],
                "sample_rate": config["sample_rate"],
                "edit_start_after_anchor_seconds": 0.10,
            }
        )
    return output


def render_case(background: np.ndarray, case: dict[str, Any]) -> dict[str, Any]:
    """
    Regenerate one physical observed pair using frozen parameters, with no model or raw-data access.

    Parameters
    ----------
    background : np.ndarray
        Corresponding cached float32 (12,2500) real background in mV.
    case : dict
        Immutable qualified case parameters from local cases.json.

    Returns
    -------
    dict
        Edited/baseline float32 signals, exact support, candidate mask and label-only metadata.
    """
    if background.shape != (12, 2500) or background.dtype != np.float32 or not np.isfinite(background).all():
        raise ValueError("Expected a finite native float32 background")
    if array_sha256(background) != case["processed_background_sha256"]:
        raise ValueError("Processed background hash changed")
    if case["role"] not in ("calibration", "evaluation") or case["profile"] != (
        "A" if case["role"] == "calibration" else "B"
    ):
        raise ValueError("Case role/profile changed")
    params, rng = _nuisance_parameters(case["seed"])
    if params != case["nuisance_parameters"]:
        raise ValueError("Frozen nuisance parameters changed")
    time = np.arange(2500) / 250
    independent_noise = rng.normal(0, params["noise_sd_mv"], size=(8, 2500))
    independent_noise += params["drift_amplitude_mv"] * np.sin(
        2 * np.pi * params["drift_hz"] * time[None, :] + np.array(params["phases"])[:, None]
    )
    gain = params["gain"] if case["nuisance"] else 1.0
    disturbance = couple_limb_leads(independent_noise) if case["nuisance"] else np.zeros((12, 2500))
    edit = case["edit_parameters"]
    selected = []
    if case["target"]:
        selected = (
            [edit["focal_anchor_position"]] if case["mode"] == "focal" else list(range(len(case["anchors"])))
        )
    envelope = np.zeros(2500)
    for position in selected:
        if position is not None:
            start = case["anchors"][position] / 250 + case["edit_start_after_anchor_seconds"]
            envelope += edit_profile((time - start) / edit["width_seconds"], case["profile"])
    primitive = np.array(edit["primitive_coefficients"])[:, None] * edit["amplitude_mv"] * envelope
    delta = couple_limb_leads(primitive)
    x = background.astype(np.float64)
    baseline = (gain * x + disturbance).astype(np.float32)
    edited = (gain * (x + delta) + disturbance).astype(np.float32)
    observed = edited.astype(np.float64) - baseline.astype(np.float64)
    candidate = np.zeros((12, 2500), dtype=bool)
    for anchor in case["anchors"]:
        candidate[:, max(0, anchor + 15) : min(2500, anchor + 98)] = True
    return {
        "edited": edited,
        "baseline": baseline,
        "support": np.abs(observed) > case["support_floor_mv"],
        "candidate": candidate,
        "analytic_delta": gain * delta,
        "metadata": {"gain": gain, "profile": case["profile"], "native_rate_hz": 250, "units": "mV"},
    }


def _limb_error(delta: np.ndarray) -> float:
    first, second = delta[:2]
    expected = (second - first, -(first + second) / 2, first - second / 2, second - first / 2)
    return max(float(np.max(np.abs(delta[index + 2] - value))) for index, value in enumerate(expected))


def qualify_pair(pair: dict[str, Any], case: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """
    Retain every exact native-mask, control, amplitude and added-delta algebra failure.

    Parameters
    ----------
    pair : dict
        Fresh render output.
    case : dict
        Fixed case identity/parameters.
    config : dict
        Frozen numerical tolerances and support limits.

    Returns
    -------
    dict
        All gate reasons and measured engineering quantities, without clinical labels.
    """
    baseline, edited = pair["baseline"], pair["edited"]
    delta = edited.astype(np.float64) - baseline.astype(np.float64)
    support, candidate, analytic = pair["support"], pair["candidate"], pair["analytic_delta"]
    peak_base, peak_edit = float(np.abs(baseline).max()), float(np.abs(edited).max())
    tolerance = (
        config["delta_rounding_tolerance_eps_multiplier"]
        * np.finfo(np.float32).eps
        * (1 + peak_base + peak_edit)
    )
    analytic_bound = 64 * np.finfo(np.float64).eps * (1 + float(np.abs(analytic).max()))
    metrics = {
        "support_fraction": float(support.mean()),
        "delta_peak_mv": float(np.abs(delta).max()),
        "delta_limb_error_mv": _limb_error(delta),
        "delta_rounding_bound_mv": float(tolerance),
        "analytic_limb_error_mv": _limb_error(analytic),
        "analytic_limb_bound_mv": float(analytic_bound),
        "maximum_quantization_residual_mv": float(np.abs(delta - analytic).max()),
        "outside_support_max_mv": float(np.max(np.abs(delta[~support]), initial=0)),
    }
    gates = {
        "geometry_or_nonfinite": all(
            value.shape == (12, 2500) and np.isfinite(value).all() for value in (baseline, edited, delta)
        ),
        "too_few_anchors": len(case["anchors"]) >= config["minimum_anchors"],
        "support_reconstruction": np.array_equal(support, np.abs(delta) > config["support_floor_mv"]),
        "support_outside_candidate": not np.any(support & ~candidate),
        "support_occupancy": metrics["support_fraction"] <= config["maximum_support_fraction"],
        "delta_amplitude": metrics["delta_peak_mv"] <= 0.50,
        "observed_delta_algebra": metrics["delta_limb_error_mv"] <= tolerance,
        "analytic_delta_algebra": metrics["analytic_limb_error_mv"] <= analytic_bound,
        "quantization_residual": metrics["maximum_quantization_residual_mv"] <= tolerance,
        "positive_support": bool(support.any()) if case["target"] else not support.any(),
        "control_byte_identity": case["target"] or baseline.tobytes() == edited.tobytes(),
    }
    return {"failures": [name for name, passed in gates.items() if not passed], "metrics": metrics}
