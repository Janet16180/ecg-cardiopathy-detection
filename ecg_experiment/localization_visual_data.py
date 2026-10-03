"""Replay frozen real examples and expose intermediate arrays for visual explanation."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import wfdb
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfiltfilt
from threadpoolctl import threadpool_limits

from ecg_experiment.files import sha256_file
from ecg_experiment.fragment_localization import r_peaks
from ecg_experiment.incart_localization056 import METHODS, read_record, score_signal
from ecg_experiment.morphology_boundaries import qrs_support
from ecg_experiment.paths import to_stored
from ecg_experiment.raw_residual052 import AFTER, BEFORE, FS, STRIDE, WIDTH
from ecg_experiment.raw_residual054 import aligned_field
from ecg_experiment.real_notebook_cases import _incart_case, load_real_evidence_cases


def _verified_hash(root: Path, path: Path, expected: str) -> dict[str, str]:
    """Verify a local explanatory dependency against its saved receipt.

    Parameters
    ----------
    root : Path
        Repository root.
    path : Path
        Repository-relative dependency.
    expected : str
        Frozen SHA-256.

    Returns
    -------
    dict
        Verified repository-relative dependency and digest.
    """
    actual = sha256_file(root / path)
    if actual != expected:
        raise ValueError(f"Frozen explanatory dependency changed: {path}")
    return {to_stored(root / path): actual}


def _window_payload(unit_map: Any, peaks: np.ndarray, left: int) -> dict[str, Any]:
    """Expose unchanged candidate scores and their nearest-beat ownership.

    Parameters
    ----------
    unit_map : UnitMap
        Frozen lead-major candidate map.
    peaks : np.ndarray
        Complete context-local R anchors.
    left : int
        Context offset on the original 500 Hz record.

    Returns
    -------
    dict
        Absolute window starts, twelve lead score rows and global owner R samples.
    """
    selected = unit_map.leads == 0
    starts = unit_map.starts[selected]
    centers = (unit_map.starts[selected] + unit_map.ends[selected]) / 2 * FS
    owner = peaks[np.abs(centers[:, None] - peaks[None]).argmin(axis=1)]
    return {
        "starts": (starts + left / FS).tolist(),
        "width_seconds": WIDTH / FS,
        "stride_seconds": STRIDE / FS,
        "scores": unit_map.scores.reshape(12, len(starts)).tolist(),
        "owner_r_sample": (owner + left).tolist(),
    }


def _incart_steps(signal: np.ndarray, references: dict, case: dict[str, Any]) -> dict[str, Any]:
    """Reproduce one saved core and expose real leave-one-out residual intermediates.

    Parameters
    ----------
    signal : np.ndarray
        Whole real recording resampled by the frozen reader to 500 Hz.
    references : dict
        Frozen PTB training-normal reference used only by the previous U_B method.
    case : dict
        Hash-verified native ECG and saved per-beat predictions.

    Returns
    -------
    dict
        Exact candidate coordinates, baseline-subtracted comparisons and integrity checks.
    """
    core = int(case["detected_beats"][0]["core"])
    start, stop = core * 5000, (core + 1) * 5000
    left, right = max(0, start - 250), min(signal.shape[1], stop + 250)
    context = np.asarray(signal[:, left:right], dtype=np.float64)
    peaks = r_peaks(context, FS)
    table, maps = score_signal(context, references, peaks)
    complete = peaks[(peaks >= BEFORE) & (peaks + AFTER <= context.shape[1])]
    field, audit = aligned_field(context, peaks)
    maximum_error = 0.0
    for saved in case["detected_beats"]:
        actual = table[table.r_sample == int(saved["r_sample"]) - left]
        if len(actual) != 1:
            raise ValueError("Saved beat anchor failed replay")
        row = actual.iloc[0]
        for method in METHODS:
            error = abs(float(row[method]) - float(saved[method]))
            maximum_error = max(maximum_error, error)
            if not np.isclose(row[method], saved[method], rtol=1e-10, atol=1e-10):
                raise ValueError("Saved real beat score failed frozen replay")
            if int(row[f"{method}_lead"]) != int(saved[f"{method}_lead"]):
                raise ValueError("Saved real lead failed frozen replay")
            if not np.isclose(row[f"{method}_start"] + left / FS, saved[f"{method}_start"], atol=1e-12):
                raise ValueError("Saved real window failed frozen replay")
    if len(case["detected_beats"]) != np.count_nonzero(
        (table.r_sample + left >= start) & (table.r_sample + left < stop)
    ):
        raise ValueError("Saved scored-beat count failed replay")
    kept = audit["peaks"]
    owned = []
    for index, peak in enumerate(kept):
        owned.append(
            [
                (max(peak - BEFORE, 0 if index == 0 else (kept[index - 1] + peak) // 2) + left) / FS,
                (
                    min(
                        peak + AFTER,
                        context.shape[1] if index == len(kept) - 1 else (peak + kept[index + 1]) // 2,
                    )
                    + left
                )
                / FS,
            ]
        )
    original = audit["original_beats"]
    prediction = audit["predictions"]
    normalized = (original - prediction) / audit["scales"][None, :, None]
    selected_time = case["method_results"][1]["selected_detected_r_time"]
    return {
        "fs": FS,
        "context_start": left / FS,
        "context_end": right / FS,
        "core_start": start / FS,
        "core_end": stop / FS,
        "time": ((np.arange(context.shape[1]) + left) / FS).tolist(),
        "signal": context.tolist(),
        "field": field.tolist(),
        "beat_r_sample": (kept + left).tolist(),
        "beat_r_time": ((kept + left) / FS).tolist(),
        "beat_shift_samples": audit["shifts"].tolist(),
        "beat_shift_ms": (audit["shifts"] * 1000 / FS).tolist(),
        "beat_relative_time": (np.arange(-BEFORE, AFTER) / FS).tolist(),
        "original_beats": original.tolist(),
        "predictions": prediction.tolist(),
        "difference_mv": (original - prediction).tolist(),
        "normalized_difference": normalized.tolist(),
        "residual_energy": (normalized**2).tolist(),
        "lead_scales_mv": audit["scales"].tolist(),
        "owned_intervals": owned,
        "windows": {name: _window_payload(unit_map, complete, left) for name, unit_map in maps.items()},
        "selected_beat_index": int(np.flatnonzero(kept + left == round(selected_time * FS))[0]),
        "integrity": {
            "saved_beats_checked": len(case["detected_beats"]),
            "maximum_score_absolute_error": maximum_error,
            "saved_leads_and_starts_reproduced": True,
        },
        "notes": [
            "These intermediate arrays replay the frozen method on this existing example; no new accuracy "
            "experiment was run.",
            "The candidate uses raw resampled mV samples, PR-median baseline subtraction and small QRS "
            "shifts; the previous U_B extraction also high-pass filters its own wave pieces.",
            "Each beat's expected waveform is the median of the other aligned complete beats in its "
            "context, then mapped back to the original time coordinates.",
            "Lead scale is median QRS peak-to-peak amplitude across complete beats, with a 0.1 mV floor; "
            "amplitude is not fitted away.",
            "The residual field uses midpoint-owned samples; eligible 140 ms windows advance by 10 ms and "
            "are ranked across all twelve leads.",
            "Expert beat symbols are displayed after inference and never enter the template, shifts or "
            "scores.",
        ],
    }


def _qtdb_steps(root: Path, case: dict[str, Any]) -> dict[str, Any]:
    """Expose the derivative envelope that produced a saved joint QRS interval.

    Parameters
    ----------
    root : Path
        Repository root with local QTDB waveforms.
    case : dict
        Hash-verified real excerpt and saved boundaries.

    Returns
    -------
    dict
        Full-record-filtered excerpt, derivative envelopes and verified per-channel bounds.
    """
    record_id = case["id"].split("_")[1]
    raw = wfdb.rdrecord(str(root / "data/raw/qtdb/1.0.0" / record_id), physical=True)
    signal = raw.p_signal.T
    fs = int(raw.fs)
    filtered = sosfiltfilt(butter(2, [0.5, 30], btype="bandpass", fs=fs, output="sos"), signal, axis=1)
    envelope = gaussian_filter1d(np.abs(np.gradient(filtered, axis=1)), sigma=0.006 * fs, axis=1)
    peak = round(case["detected_beats"][0]["time_seconds"] * fs)
    per_lead = np.asarray([qrs_support(lead, peak, fs) for lead in envelope])
    joint = [int(per_lead[:, 0].min()), int(per_lead[:, 1].max())]
    saved = case["method_results"][1]["predicted"]
    if joint != saved:
        raise ValueError("Derivative-envelope replay differs from the saved joint QRS")
    left = round(case["time"][0] * fs)
    right = left + len(case["time"])
    local_left = max(0, peak - round(0.06 * fs))
    local_right = min(signal.shape[1], peak + round(0.06 * fs))
    maxima = local_left + envelope[:, local_left:local_right].argmax(axis=1)
    return {
        "time": case["time"],
        "filtered_signal": filtered[:, left:right].tolist(),
        "derivative_envelope": envelope[:, left:right].tolist(),
        "envelope_units": "absolute mV change per native sample, smoothed",
        "thresholds": (0.12 * envelope[np.arange(2), maxima]).tolist(),
        "maximum_time": (maxima / fs).tolist(),
        "r_time": peak / fs,
        "search_start": max(0, peak - round(0.12 * fs)) / fs,
        "search_end": min(signal.shape[1], peak + round(0.16 * fs)) / fs,
        "lead_bounds": (per_lead / fs).tolist(),
        "joint_bounds": (np.asarray(joint) / fs).tolist(),
        "integrity": {"saved_joint_qrs_reproduced": True},
        "notes": [
            "The envelope uses the frozen full-record 0.5–30 Hz bandpass and smoothed absolute "
            "derivative; it is not computed on the displayed crop alone.",
            "Within the fixed search neighborhood, activity above 12% of the nearby derivative maximum is "
            "gap-closed, connected around that maximum and expanded by 8 ms.",
            "The saved joint QRS starts at the earlier channel onset and ends at the later channel offset.",
            "QRS boundary timing identifies a wave interval; it does not classify disease or identify a "
            "pathological lead.",
        ],
    }


def build_visual_payload(root: Path) -> dict[str, Any]:
    """Load real examples and replay frozen intermediates for an offline visual explanation.

    Parameters
    ----------
    root : Path
        Imported checkout root with symlinked local data and saved outputs.

    Returns
    -------
    dict
        Finite JSON-compatible ``cases`` with native real waveforms and added ``steps``;
        ``metrics`` are copied from saved receipts, never computed from these examples.
    """
    root = Path(root).absolute()
    base056 = root / "outputs/experiment056_incart_localization_v1"
    base059 = root / "outputs/experiment059_qtdb_hybrid_v2"
    receipt056 = json.loads((base056 / "result.json").read_text())
    receipt059 = json.loads((base059 / "result.json").read_text())
    reference_path = Path("outputs/experiment056_incart_localization_v1/normal_reference.pkl")
    verified = _verified_hash(root, reference_path, receipt056["outputs_sha256"]["normal_reference.pkl"])
    for receipt in (receipt056, receipt059):
        for name, digest in receipt["identity"]["sources"].items():
            if name.startswith("ecg_experiment/"):
                verified.update(_verified_hash(root, Path(name), digest))
    verified[to_stored(base056 / "result.json")] = sha256_file(base056 / "result.json")
    verified[to_stored(base059 / "result.json")] = sha256_file(base059 / "result.json")
    with (root / reference_path).open("rb") as stream:
        references = pickle.load(stream)
    with threadpool_limits(limits=2):
        cases = load_real_evidence_cases(root)
        clear_case = _incart_case(root, 6)
        clear_case["title"] = "Finding the unusual beat: a clear teaching example"
        clear_case["selection_reason"] = (
            "Descriptive I01 core 6 example selected for a clear QRS-adjacent mark"
        )
        clear_case["plain_language"] = (
            "The blue method selects an expert ventricular beat and a 140 ms interval around its QRS. "
            "This shows an interpretable candidate location; expert beat labels do not validate the exact "
            "lead "
            "or pathological waveform boundaries."
        )
        cases.insert(0, clear_case)
        signal, _patient = read_record(root / "data/raw/incart/1.0.0/I01")
        for case in cases:
            case["steps"] = (
                _incart_steps(signal, references, case)
                if case["kind"] == "incart"
                else _qtdb_steps(root, case)
            )
    result = {
        "schema_version": 1,
        "cases": cases,
        "beat_cases": [case for case in cases if case["kind"] == "incart"],
        "boundary_cases": [case for case in cases if case["kind"] == "qtdb"],
        "metrics": {
            "incart": {"coverage": receipt056["coverage"], **receipt056["benchmark"]},
            "qtdb": receipt059["primary_non_edb"],
        },
        "provenance": {
            "verified_files": verified,
            "purpose": (
                "Frozen-example explanatory replay; no fitting, new benchmark, synthetic signal "
                "or closed-test access"
            ),
        },
    }
    json.dumps(result, allow_nan=False)
    return result
