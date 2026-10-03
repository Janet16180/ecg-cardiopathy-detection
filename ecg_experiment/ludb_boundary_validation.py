"""Local LUDB waveform/annotation audit and prospective wave-boundary evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import wfdb

from ecg_experiment.files import sha256_file
from ecg_experiment.morphology_boundaries import (
    WAVE_NAMES,
    adaptive_boundaries,
    annotation_triplets,
    associate_waves,
    fixed_boundaries,
    interval_measurements,
    match_peaks,
)
from ecg_experiment.waveforms import LEADS


def audit_files(root: Path) -> dict[str, Any]:
    """
    Verify all published checksums without reading diagnostic metadata values.

    Parameters
    ----------
    root : Path
        Local LUDB 1.0.1 root.

    Returns
    -------
    dict[str, Any]
        Published manifest identity and checked file count.
    """
    manifest = root / "SHA256SUMS.txt"
    checked = 0
    for line in manifest.read_text().splitlines():
        expected, stored = line.split(maxsplit=1)
        path = root / stored.lstrip("*")
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"LUDB published checksum mismatch: {stored}")
        checked += 1
    return {"published_checksum_manifest_sha256": sha256_file(manifest), "files_verified": checked}


def read_ludb_record(root: Path, record_id: int) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    """
    Read physical canonical signals and complete lead-specific boundary annotations.

    Parameters
    ----------
    root : Path
        Local LUDB 1.0.1 root.
    record_id : int
        Official integer record identifier.

    Returns
    -------
    tuple[np.ndarray, dict[str, np.ndarray], dict]
        Canonical mV signals, annotation triplets keyed by lead/wave and data audit.
    """
    stem = str(root / "data" / str(record_id))
    record = wfdb.rdrecord(stem)
    names = [name.casefold() for name in record.sig_name]
    canonical = [name.casefold() for name in LEADS]
    if record.fs != 500 or record.sig_len != 5000 or set(names) != set(canonical):
        raise ValueError("LUDB signal rate, duration or leads differ from the frozen source")
    ordered = [names.index(name) for name in canonical]
    conversion = {"mV": 1.0, "uV": 0.001, "µV": 0.001}
    factors = np.array([conversion[record.units[index]] for index in ordered])
    signal = record.p_signal[:, ordered].T * factors[:, None]
    if not np.isfinite(signal).all():
        raise ValueError("Nonfinite LUDB waveform")
    annotations = {}
    irregular = {}
    typed_counts = dict.fromkeys(WAVE_NAMES, 0)
    for lead in canonical:
        found = wfdb.rdann(stem, lead)
        typed_counts["P"] += found.symbol.count("p")
        typed_counts["QRS"] += found.symbol.count("N")
        typed_counts["T"] += found.symbol.count("t")
        waves, extra = annotation_triplets(found.sample, found.symbol)
        for wave, values in waves.items():
            if len(values) and (values.min() < 0 or values.max() >= record.sig_len):
                raise ValueError("Annotation lies outside LUDB record")
            annotations[f"{lead}_{wave}"] = values
        if extra:
            irregular[lead] = extra
    return (
        signal,
        annotations,
        {
            "record_id": record_id,
            "sampling_rate": record.fs,
            "samples": record.sig_len,
            "lead_names": names,
            "units": record.units,
            "conversion_to_mV": factors.tolist(),
            "signal_peak_to_peak_mV": np.ptp(signal, axis=1).tolist(),
            "annotation_triplet_counts": {key: len(value) for key, value in annotations.items()},
            "irregular_annotation_events": irregular,
            "raw_typed_peak_counts": typed_counts,
        },
    )


def score_lead(
    peaks: np.ndarray,
    predictions: dict[str, np.ndarray],
    annotations: dict[str, np.ndarray],
    lead: int,
    fs: int = 500,
) -> tuple[list[dict], dict]:
    """
    Evaluate every eligible truth wave with common independent anchor matching.

    Parameters
    ----------
    peaks : np.ndarray
        Common raw R anchors.
    predictions : dict[str, np.ndarray]
        Fixed and adaptive inferred endpoints.
    annotations : dict[str, np.ndarray]
        Truth triplets keyed by physiological wave class for this lead.
    lead : int
        Canonical lead index.
    fs : int
        Sampling frequency.

    Returns
    -------
    tuple[list[dict], dict]
        Per-truth-wave measurements and coverage/missingness audit.
    """
    qrs = annotations["QRS"]
    matched = match_peaks(peaks, qrs, fs)
    results = []
    audit = {
        "qrs_truth_total": len(qrs),
        "qrs_anchor_matches": len(matched),
        "raw_anchors": len(peaks),
        "waves": {},
    }
    for wave_index, name in enumerate(WAVE_NAMES):
        truth = annotations[name]
        associated = associate_waves(truth, qrs, name, fs)
        eligible = np.flatnonzero((truth[:, 0] >= round(0.5 * fs)) & (truth[:, 2] <= round(9.5 * fs)))
        used = set()
        unassociated = 0
        for index in eligible:
            qrs_index = associated.get(int(index))
            anchor = matched.get(qrs_index) if qrs_index is not None else None
            if anchor is None:
                unassociated += 1
            else:
                if anchor in used:
                    raise ValueError("Evaluation attempted to reuse a predicted wave slot")
                used.add(anchor)
            row = {
                "lead": lead,
                "wave": name,
                "truth_onset": int(truth[index, 0]),
                "truth_peak": int(truth[index, 1]),
                "truth_offset": int(truth[index, 2]),
                "anchor_index": anchor,
            }
            for method, boundaries in predictions.items():
                interval = boundaries[anchor, lead, wave_index] if anchor is not None else np.array([-1, -1])
                row[method] = {
                    **interval_measurements(interval, truth[index, [0, 2]], fs),
                    "predicted_onset": int(interval[0]),
                    "predicted_offset": int(interval[1]),
                }
            results.append(row)
        interior_slots = set(np.flatnonzero((peaks >= round(0.5 * fs)) & (peaks <= round(9.5 * fs))).tolist())
        audit["waves"][name] = {
            "eligible_truth": len(eligible),
            "edge_truth_excluded": len(truth) - len(eligible),
            "truth_without_matched_slot": unassociated,
            "unannotated_interior_prediction_slots": {
                method: int(
                    sum(boundaries[index, lead, wave_index, 0] >= 0 for index in interior_slots - used)
                )
                for method, boundaries in predictions.items()
            },
            "rejected_interior_slots": {
                method: int(sum(boundaries[index, lead, wave_index, 0] < 0 for index in interior_slots))
                for method, boundaries in predictions.items()
            },
        }
    return results, audit


def evaluate_record(signal: np.ndarray, annotations: dict[str, np.ndarray]) -> tuple[list[dict], dict, dict]:
    """
    Infer without annotations, then evaluate complete physiological boundary truth.

    Parameters
    ----------
    signal : np.ndarray
        Canonical raw mV signals.
    annotations : dict[str, np.ndarray]
        Lead-specific truth triplets, used only after inference.

    Returns
    -------
    tuple[list[dict], dict, dict]
        Per-wave comparisons, coverage diagnostics and inferred arrays.
    """
    peaks, adaptive = adaptive_boundaries(signal)
    predictions = {"fixed": fixed_boundaries(peaks), "adaptive": adaptive}
    measurements, diagnostics = [], {}
    for index, name in enumerate(LEADS):
        lead_annotations = {wave: annotations[f"{name.casefold()}_{wave}"] for wave in WAVE_NAMES}
        found, audit = score_lead(peaks, predictions, lead_annotations, index)
        measurements.extend(found)
        diagnostics[name] = audit
    return measurements, diagnostics, {"peaks": peaks, **predictions}


def record_summary(measurements: list[dict]) -> dict:
    """
    Macro-average wave classes within a patient without dropping failed slots.

    Parameters
    ----------
    measurements : list[dict]
        Measurements for every eligible annotated wave and lead in one subject.

    Returns
    -------
    dict
        Per-class means and equal-present-class primary record score.
    """
    result = {"classes": {}, "primary": {}}
    for name in WAVE_NAMES:
        found = [row for row in measurements if row["wave"] == name]
        if not found:
            continue
        result["classes"][name] = {"truth_waves": len(found)}
        for method in ("fixed", "adaptive"):
            metrics = ("iou", "onset_mae_ms", "offset_mae_ms", "within_30ms", "missing")
            result["classes"][name][method] = {
                metric: float(np.mean([row[method][metric] for row in found])) for metric in metrics
            }
    for method in ("fixed", "adaptive"):
        values = [wave[method]["iou"] for wave in result["classes"].values()]
        result["primary"][method] = float(np.mean(values)) if values else 0.0
    result["no_eligible_truth"] = not bool(result["classes"])
    return result
