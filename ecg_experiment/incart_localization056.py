"""Independent expert beat-identity benchmarking and label-free scores for Experiment 056."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt
from sklearn.metrics import average_precision_score, roc_auc_score

from . import ROOT
from .files import sha256_file
from .fragment_localization import bootstrap_mean, r_peaks
from .lead_wave_maps import (
    BASELINE,
    BEAT_AFTER,
    BEAT_BEFORE,
    HIGH_PASS_HZ,
    WAVES,
    UnitMap,
    beat_unit_map,
    wave_lengths,
    wave_scores,
)
from .raw_residual052 import FS, baseline_field, fixed_window_map
from .raw_residual054 import aligned_field, primary_summary, reproduce052, shared_mask, synthetic_summary
from .resample import resample_full
from .waveforms import LEADS

METHODS = ("aligned_residual", "U_B_fixed")
SEED = 56056
BEAT_SYMBOLS = frozenset("NLRBAaJSVrFejnE/fQ?")


def read_record(path: Path) -> tuple[np.ndarray, int]:
    """Read physical standard leads and the audited patient number, then resample the whole record.

    Parameters
    ----------
    path : Path
        WFDB record stem, with local header and waveform files.

    Returns
    -------
    tuple
        Finite canonical twelve-lead float32 waveform at 500 Hz and patient ID.
    """
    record = wfdb.rdrecord(str(path), physical=False)
    names = [name.replace("AV", "aV") for name in record.sig_name]
    if set(names) != set(LEADS) or len(names) != 12 or record.fs != 257:
        raise ValueError("Unexpected INCART sampling rate or leads")
    digital = record.d_signal
    if not np.array_equal(digital.sum(axis=0) % 65536, np.asarray(record.checksum) % 65536):
        raise ValueError("INCART signal checksum differs from header")
    if not np.array_equal(digital[0], record.init_value):
        raise ValueError("INCART initial signal values differ from header")
    physical = (digital - np.asarray(record.baseline)[None]) / np.asarray(record.adc_gain)[None]
    signal = physical[:, [names.index(lead) for lead in LEADS]].T
    if set(record.units) != {"mV"}:
        raise ValueError("Unexpected INCART physical units")
    patient = int(re.search(r"# patient\s+(\d+)", path.with_suffix(".hea").read_text()).group(1))
    return resample_full(signal, 500, 257), patient


def match_beats(detected: np.ndarray, reference: np.ndarray, tolerance: float = 0.15) -> np.ndarray:
    """Match times one to one with the frozen distance-first deterministic rule.

    Parameters
    ----------
    detected, reference : np.ndarray
        Ascending detected and expert annotation times in seconds.
    tolerance : float
        Maximum absolute time difference, fixed at 150 ms for the benchmark.

    Returns
    -------
    np.ndarray
        Reference index per detection, or -1 when unmatched.
    """
    pairs = []
    for detected_index, time in enumerate(detected):
        low = np.searchsorted(reference, time - tolerance, side="left")
        high = np.searchsorted(reference, time + tolerance, side="right")
        pairs.extend((abs(time - reference[index]), index, detected_index) for index in range(low, high))
    result = np.full(len(detected), -1, dtype=np.int64)
    used = set()
    for _distance, reference_index, detected_index in sorted(pairs):
        if result[detected_index] >= 0 or reference_index in used:
            continue
        result[detected_index] = reference_index
        used.add(reference_index)
    return result


def beat_scores(unit_map: UnitMap, peaks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign common windows by nearest detected R and return each beat's score and descriptive mark.

    Parameters
    ----------
    unit_map : UnitMap
        Identical fixed-duration candidates on the original timeline.
    peaks : np.ndarray
        Complete R sample positions, without annotation information.

    Returns
    -------
    tuple
        Per-beat scores, window starts and selected lead indices, NaN/-1 if no candidate belongs to it.
    """
    centers = (unit_map.starts + unit_map.ends) / 2 * FS
    owners = np.abs(centers[:, None] - peaks[None]).argmin(axis=1)
    scores = np.full(len(peaks), np.nan)
    starts = np.full(len(peaks), np.nan)
    leads = np.full(len(peaks), -1, dtype=int)
    for index in range(len(peaks)):
        candidates = np.flatnonzero(owners == index)
        if not len(candidates):
            continue
        values = unit_map.scores[candidates]
        top = candidates[np.isclose(values, values.max(), rtol=1e-10, atol=1e-10)]
        top = top[np.argsort(unit_map.starts[top], kind="stable")]
        selected = int(top[len(top) // 2])
        scores[index] = float(values.max())
        starts[index] = unit_map.starts[selected]
        leads[index] = unit_map.leads[selected]
    return scores, starts, leads


def pieces_at_peaks(signal: np.ndarray, peaks: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Factor the frozen U_B extraction while retaining explicit unchanged inference anchors.

    Parameters
    ----------
    signal : np.ndarray
        Canonical physical waveform at 500 Hz.
    peaks : np.ndarray
        Waveform-only anchors, fixed for paired nuisance checks.

    Returns
    -------
    tuple
        Kept R times and exactly the historical filtered/baseline-subtracted wave pieces.
    """
    b, a = butter(2, HIGH_PASS_HZ, btype="high", fs=FS)
    filtered = filtfilt(b, a, np.nan_to_num(signal), axis=1)
    inside = (peaks - round(BEAT_BEFORE * FS) >= 0) & (peaks + round(BEAT_AFTER * FS) <= signal.shape[1])
    kept = peaks[inside]
    pieces = {
        name: np.empty((len(kept), signal.shape[0], length)) for name, length in wave_lengths(FS).items()
    }
    for index, peak in enumerate(kept):
        baseline = filtered[:, peak + round(BASELINE[0] * FS) : peak + round(BASELINE[1] * FS)].mean(axis=1)
        for name, (low, high) in WAVES.items():
            window = filtered[:, peak + round(low * FS) : peak + round(high * FS) : 2]
            pieces[name][index] = window - baseline[:, None]
    return kept / FS, pieces


def score_signal(
    signal: np.ndarray, references: dict, fixed_peaks: np.ndarray | None = None
) -> tuple[pd.DataFrame, dict[str, UnitMap]]:
    """Compute both frozen methods before any expert annotation is opened.

    Parameters
    ----------
    signal : np.ndarray
        Context waveform at 500 Hz, canonical physical leads.
    references : dict
        Exact PTB training-normal U_B wave covariances.
    fixed_peaks : np.ndarray or None
        Original waveform-only peaks for paired nuisance copies.

    Returns
    -------
    tuple
        Per-detected-beat scores/locations and equal-support maps.
    """
    signal = np.asarray(signal, dtype=np.float64)
    peaks = r_peaks(signal, FS) if fixed_peaks is None else fixed_peaks
    r_times, pieces = pieces_at_peaks(signal, peaks)
    complete = peaks[(peaks >= 125) & (peaks + 225 <= signal.shape[1])]
    if len(complete) < 3 or not len(r_times):
        raise ValueError("Fewer than three complete beats")
    baseline = beat_unit_map(wave_scores(references, pieces), r_times)
    mask = shared_mask(signal, peaks, baseline)
    field, _audit = aligned_field(signal, peaks)
    maps = {
        "aligned_residual": fixed_window_map(field, mask),
        "U_B_fixed": fixed_window_map(baseline_field(baseline, signal.shape[1]), mask),
    }
    scores = {}
    for name, unit_map in maps.items():
        values, starts, leads = beat_scores(unit_map, complete)
        scores[name] = values
        scores[f"{name}_start"] = starts
        scores[f"{name}_lead"] = leads
    table = pd.DataFrame({"r_sample": complete, **scores})
    table = table[table[list(METHODS)].notna().all(axis=1)].copy()
    return table, maps


def patient_metrics(
    beats: pd.DataFrame, chunks: pd.DataFrame, reference: pd.DataFrame, thresholds: dict[str, float]
) -> pd.DataFrame:
    """Compute patient-weighted conditional ranking, alarms and end-to-end coverage.

    Parameters
    ----------
    beats : pd.DataFrame
        All waveform detections, matched expert types and both scores.
    chunks : pd.DataFrame
        Mixed-core top-beat values for each method.
    reference : pd.DataFrame
        Every expert annotation and matching status, including missed beats.
    thresholds : dict
        PTB normal beat-score 95th percentiles.

    Returns
    -------
    pd.DataFrame
        One row per patient/method, with explicit absent-class NaNs.
    """
    records = []
    for patient, truth in reference.groupby("patient"):
        detected = beats[beats["patient"] == patient]
        binary = detected[detected["symbol"].isin(["N", "V"])]
        normal = binary[binary["symbol"] == "N"]
        ventricular = binary[binary["symbol"] == "V"]
        for name in METHODS:
            selected = chunks[(chunks["patient"] == patient) & (chunks["map"] == name)]
            both = len(normal) > 0 and len(ventricular) > 0
            records.append(
                {
                    "patient": patient,
                    "map": name,
                    "auroc": float(roc_auc_score(binary["symbol"] == "V", binary[name])) if both else np.nan,
                    "ap": float(average_precision_score(binary["symbol"] == "V", binary[name]))
                    if both
                    else np.nan,
                    "mixed_core_excess": float(selected["excess"].mean()) if len(selected) else np.nan,
                    "mixed_core_hit": float(selected["hit"].mean()) if len(selected) else np.nan,
                    "mixed_core_chance": float(selected["chance"].mean()) if len(selected) else np.nan,
                    "normal_fpr": float((normal[name] > thresholds[name]).mean()) if len(normal) else np.nan,
                    "v_sensitivity": float((ventricular[name] > thresholds[name]).mean())
                    if len(ventricular)
                    else np.nan,
                    "v_end_to_end_sensitivity": float(
                        (ventricular[name] > thresholds[name]).sum() / max(1, (truth["symbol"] == "V").sum())
                    ),
                    "matched_n": len(normal),
                    "matched_v": len(ventricular),
                    "reference_n": int((truth["symbol"] == "N").sum()),
                    "reference_v": int((truth["symbol"] == "V").sum()),
                    "mixed_cores": len(selected),
                }
            )
    return pd.DataFrame(records)


def benchmark_summary(table: pd.DataFrame) -> dict[str, Any]:
    """Apply frozen co-primary and specificity gates to patient-level paired results.

    Parameters
    ----------
    table : pd.DataFrame
        Patient-by-method metrics from ``patient_metrics``.

    Returns
    -------
    dict
        Patient macro metrics, paired intervals, eligible counts and gate readings.
    """
    result: dict[str, Any] = {"methods": {}, "paired": {}}
    fields = (
        "auroc",
        "ap",
        "mixed_core_excess",
        "mixed_core_hit",
        "mixed_core_chance",
        "normal_fpr",
        "v_sensitivity",
        "v_end_to_end_sensitivity",
    )
    for name in METHODS:
        group = table[table["map"] == name]
        result["methods"][name] = {field: float(group[field].mean()) for field in fields}
    first = table[table["map"] == METHODS[0]].set_index("patient")
    second = table[table["map"] == METHODS[1]].set_index("patient").loc[first.index]
    for field in ("auroc", "mixed_core_excess", "normal_fpr"):
        keep = first[field].notna() & second[field].notna()
        result["paired"][field] = {
            "patients": int(keep.sum()),
            **bootstrap_mean(
                first.index[keep].to_numpy(),
                (first.loc[keep, field] - second.loc[keep, field]).to_numpy(),
                2000,
                SEED,
            ),
        }
    paired = result["paired"]
    result["gates"] = {
        "auroc": paired["auroc"]["value"] >= 0.05 and paired["auroc"]["ci_low"] > 0,
        "top_beat": paired["mixed_core_excess"]["value"] >= 0.10
        and paired["mixed_core_excess"]["ci_low"] > 0,
        "normal_specificity": paired["normal_fpr"]["ci_high"] <= 0.05,
    }
    return result


def reproduce054() -> tuple[dict[str, Any], list[UnitMap], dict[str, Any]]:
    """Check all live 054 receipts and reproduce its full reported metrics before new scores.

    Returns
    -------
    tuple
        Historical rows, U_B maps and exact predecessor checks.
    """
    base = ROOT / "outputs/experiment054_aligned_phase_residual_v1"
    prior = json.loads((base / "result.json").read_text())
    for group in ("sources", "waveforms"):
        for name, expected in prior["identity"][group].items():
            if sha256_file(ROOT / name) != expected:
                raise ValueError(f"054 {group} mismatch: {name}")
    for name, expected in prior["outputs_sha256"].items():
        if sha256_file(base / name) != expected:
            raise ValueError(f"054 output mismatch: {name}")
    rows, baseline, _windows, checks = reproduce052()
    table = pd.read_csv(base / "development_locations.csv", float_precision="round_trip")
    found = primary_summary(table, ("aligned_phase", "aligned_only", "U_B_fixed"), 54054)
    found.update(
        {
            "excluded": [],
            "saturated_records": int((table[table["map"] == "aligned_phase"]["saturated_windows"] > 0).sum()),
        }
    )
    if found != prior["development"]:
        raise ValueError("054 complete development metrics differ")
    controls = pd.read_csv(base / "threshold_controls.csv", float_precision="round_trip")
    synthetic = pd.read_csv(base / "synthetic_locations.csv", float_precision="round_trip")
    for name in ("aligned_phase", "aligned_only"):
        group = synthetic[synthetic["map"] == name]
        summary = synthetic_summary(group, 54054)
        nuisance = group[group["kind"] == "nuisance"]
        threshold = float(controls[controls["map"] == name]["score"].quantile(0.95))
        unchanged = float((nuisance["control_score"] > threshold).mean())
        flagged = float((nuisance["score"] > threshold).mean())
        summary.update(
            {
                "records": int(group["ecg_id"].nunique()),
                "excluded_fewer_than_five_complete_beats": [],
                "nuisance": {
                    "threshold": threshold,
                    "unchanged_flag_share": unchanged,
                    "nuisance_flag_share": flagged,
                    "increase": flagged - unchanged,
                    "score_change": bootstrap_mean(
                        nuisance["patient_id"].to_numpy(),
                        (nuisance["score"] - nuisance["control_score"]).to_numpy(),
                        2000,
                        54054,
                    ),
                },
            }
        )
        if summary != prior["synthetic"][name]:
            raise ValueError("054 complete synthetic metrics differ")
    return (
        rows,
        baseline,
        {
            "052": checks,
            "054_development_equal": True,
            "054_synthetic_equal": True,
            "054_result_sha256": sha256_file(base / "result.json"),
        },
    )
