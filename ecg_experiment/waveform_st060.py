"""Waveform-only ST features and unchanged-population comparisons for Experiment 060."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ecg_experiment.files import sha256_file
from ecg_experiment.fragment_localization import r_peaks
from ecg_experiment.incart_localization056 import match_beats
from ecg_experiment.st_episode057 import FS, change_windows, location_summary, mean_interval

METHODS = ("st", "whole")


def waveform_features(signal: np.ndarray) -> dict[str, np.ndarray]:
    """Compute waveform-only features on every original thirty-second window.

    Parameters
    ----------
    signal : np.ndarray
        Native 250 Hz samples by two channels, in mV.

    Returns
    -------
    dict
        Dense features, detected anchors and explicitly retained missing windows.
    """
    if signal.ndim != 2 or signal.shape[1] != 2 or not np.isfinite(signal).all():
        raise ValueError("Expected finite two-channel waveform")
    anchors = r_peaks(signal.T, FS)
    centers = np.arange(1, len(signal) // (FS * 30)) * 30 + 15
    complete = anchors[(anchors - round(0.25 * FS) >= 0) & (anchors + round(0.45 * FS) - 1 < len(signal))]
    bins = np.floor(complete / FS / 30).astype(int)
    counts = np.bincount(bins, minlength=len(centers) + 1)[1 : len(centers) + 1]
    output = {"anchors": complete, "detected_anchors": anchors, "centers": centers, "counts": counts}
    output.update({name: np.zeros((len(centers), 2)) for name in METHODS})
    output["inference_failed"] = np.array(False)
    try:
        sparse = change_windows(signal, anchors)
    except ValueError as error:
        if str(error) not in (
            "Fewer than three complete early-reference beats",
            "No shared candidate windows",
        ):
            raise
        output["inference_failed"] = np.array(True)
        output["valid_windows"] = np.zeros(len(centers), dtype=bool)
        return output
    indices = np.searchsorted(centers, sparse["centers"])
    if not np.array_equal(centers[indices], sparse["centers"]):
        raise ValueError("Unexpected waveform window geometry")
    output["counts"][indices] = sparse["counts"]
    output["valid_windows"] = output["counts"] >= 3
    for name in METHODS:
        output[name][indices] = sparse[name]
    output.update(
        {name: sparse[name] for name in ("anchors", "reference", "offsets", "beat_st", "beat_whole")}
    )
    return output


def reproduce057(directory: Path, root: Path) -> dict[str, Any]:
    """Verify the predecessor's files and exactly rebuild its primary statistics.

    Parameters
    ----------
    directory : Path
        Completed Experiment 057 output directory.
    root : Path
        Current repository root.

    Returns
    -------
    dict
        Exact reproduction evidence and predecessor receipt identity.
    """
    prior = json.loads((directory / "result.json").read_text())
    verified = 0
    for group in ("sources", "data"):
        for path, digest in prior["identity"][group].items():
            if sha256_file(root / path) != digest:
                raise ValueError(f"057 pinned input changed: {path}")
            verified += 1
    for path, digest in prior["outputs_sha256"].items():
        if sha256_file(directory / path) != digest:
            raise ValueError(f"057 output changed: {path}")
    rows = json.loads((directory / "per_record.json").read_text())
    eligible = [row for row in rows if row["eligible_st"]]
    patients = np.array([row["patient"] for row in eligible])
    summaries = {
        name: mean_interval(np.array([row[name]["hit"] for row in eligible]), patients) for name in METHODS
    }
    for name, found in summaries.items():
        expected = {key: prior["primary"][name][key] for key in found}
        if found != expected:
            raise ValueError(f"057 {name} primary did not reproduce exactly")
    gain = mean_interval(np.array([row["st"]["hit"] - row["whole"]["hit"] for row in eligible]), patients)
    if gain != prior["paired_gain"]:
        raise ValueError("057 paired gain did not reproduce exactly")
    return {
        "verified_sources_and_data": verified,
        "verified_outputs": len(prior["outputs_sha256"]),
        "primary_exact": True,
        "paired_gain_exact": True,
        "receipt_sha256": sha256_file(directory / "result.json"),
    }


def detector_coverage(detected: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    """Evaluate anchors after inference without supplying them to the localizer.

    Parameters
    ----------
    detected, reference : np.ndarray
        Native sample coordinates of waveform detections and expert beat marks.

    Returns
    -------
    dict
        One-to-one match counts, sensitivity and precision within 150 ms.
    """
    matched = match_beats(detected / FS, reference / FS, 0.15)
    count = int(np.sum(matched >= 0))
    return {
        "detected": len(detected),
        "reference": len(reference),
        "matched": count,
        "sensitivity": count / len(reference) if len(reference) else 0.0,
        "precision": count / len(detected) if len(detected) else 0.0,
    }


def record_metrics(
    features: dict[str, np.ndarray], gold: dict[str, np.ndarray], prior: dict[str, Any]
) -> dict[str, Any]:
    """Keep the predecessor's windows, masks and target population unchanged.

    Parameters
    ----------
    features : dict
        Features saved before expert evaluation.
    gold : dict
        Frozen predecessor centers, target masks and annotation-completeness mask.
    prior : dict
        Predecessor record metadata and supplied-anchor results.

    Returns
    -------
    dict
        Paired record statistics, including explicit failures and missing windows.
    """
    if not np.array_equal(features["centers"], gold["centers"]):
        raise ValueError("060 candidate domain changed from 057")
    mask = gold["evaluation_mask"]
    failed = bool(features["inference_failed"])
    row = {
        "record": prior["record"],
        "patient": prior["patient"],
        "eligible_st": prior["eligible_st"],
        "eligible_axis": prior["eligible_axis"],
        "inference_failed": failed,
        "evaluated_windows": int(mask.sum()),
        "valid_evaluated_windows": int(features["valid_windows"][mask].sum()),
        "supplied_st": prior["st"]["hit"],
        "channels": prior["channels"],
    }
    for name in METHODS:
        row[name] = location_summary(features[name][mask], gold["targets"][mask])
        row[name]["axis_hit"] = location_summary(features[name][mask], gold["axis_targets"][mask])["hit"]
        if failed:
            row[name].update(hit=0.0, displayed_hit=0.0, axis_hit=0.0)
    return row


def benchmark_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply all prospective primary, noninferiority, coverage and stress gates.

    Parameters
    ----------
    rows : list of dict
        All ninety record statistics, including inference failures.

    Returns
    -------
    dict
        Patient-macro comparisons and frozen decision.
    """
    eligible = [row for row in rows if row["eligible_st"]]
    patients = np.array([row["patient"] for row in eligible])
    hits = {name: np.array([row[name]["hit"] for row in eligible]) for name in METHODS}
    primary = {name: mean_interval(values, patients) for name, values in hits.items()}
    primary["supplied_st"] = mean_interval(np.array([row["supplied_st"] for row in eligible]), patients)
    gain = mean_interval(hits["st"] - hits["whole"], patients)
    noninferiority = mean_interval(hits["st"] - np.array([row["supplied_st"] for row in eligible]), patients)
    axis = [row for row in rows if row["eligible_axis"]]
    axis_gain = mean_interval(
        np.array([row["st"]["axis_hit"] - row["whole"]["axis_hit"] for row in axis]),
        np.array([row["patient"] for row in axis]),
    )
    coverage = sum(row["valid_evaluated_windows"] for row in rows) / sum(
        row["evaluated_windows"] for row in rows
    )
    gates = {
        "paired_gain": gain["value"] >= 0.10 and gain["ci_low"] > 0,
        "absolute_hit": primary["st"]["value"] >= 0.70,
        "noninferiority": noninferiority["ci_low"] > -0.05,
        "counts": sum(not row["inference_failed"] for row in rows) >= 85
        and len(eligible) == 85
        and len(set(patients)) == 76,
        "axis_stress": axis_gain["value"] <= 0.05,
        "window_coverage": coverage >= 0.99,
    }
    return {
        "records": len(rows),
        "primary_records": len(eligible),
        "primary_patients": len(set(patients)),
        "primary": primary,
        "paired_gain": gain,
        "versus_supplied_anchors": noninferiority,
        "axis_paired_gain": axis_gain,
        "axis_records": len(axis),
        "window_coverage": coverage,
        "inference_failures": [row["record"] for row in rows if row["inference_failed"]],
        "corrected_records": sum(row["st"]["hit"] > row["supplied_st"] for row in eligible),
        "regressed_records": sum(row["st"]["hit"] < row["supplied_st"] for row in eligible),
        "gates": gates,
        "promote_for_review": all(gates.values()),
        "screening_classifier_changed": False,
        "clinical_single_ecg_localization_established": False,
    }
