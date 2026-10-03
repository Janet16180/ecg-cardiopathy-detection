"""Frozen joint-channel hybrid boundary inference and sparse QTDB evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import wfdb

from ecg_experiment.calibrated_localization import mean_interval
from ecg_experiment.files import sha256_file
from ecg_experiment.ludb_boundary_validation import record_summary
from ecg_experiment.morphology_boundaries import (
    WAVE_NAMES,
    adaptive_boundaries,
    associate_waves,
    fixed_boundaries,
    interval_measurements,
    match_peaks,
)

QRS_SYMBOLS = frozenset("NLRaAVFJSEj/QenfBr")


def parse_annotations(samples: np.ndarray, symbols: list[str], nums: np.ndarray) -> tuple[dict, dict]:
    """
    Preserve disjoint complete intervals and known offsets without inventing onsets.

    Parameters
    ----------
    samples, symbols, nums : array-like
        Native expert annotation samples, WFDB symbols and boundary class numbers.

    Returns
    -------
    tuple[dict, dict]
        P/QRS/T/U triplets (unknown onset -1) and complete event accounting.
    """
    waves = {name: [] for name in (*WAVE_NAMES, "U")}
    audit = {"complete": 0, "offset_only": 0, "recovered": [], "unknown": []}
    types = {"p": "P", "t": "T", "u": "U"}
    types.update(dict.fromkeys(QRS_SYMBOLS, "QRS"))
    boundary_types = ("P", "QRS", "T", "U")
    index = 0
    while index < len(samples):
        accepted = False
        if index + 2 < len(samples) and symbols[index] == "(" and symbols[index + 2] == ")":
            name = types.get(symbols[index + 1])
            onset, peak, offset = map(int, samples[index : index + 3])
            compatible = name is not None and all(
                0 <= int(nums[k]) <= 3 and boundary_types[int(nums[k])] == name for k in (index, index + 2)
            )
            displacement = max(onset - peak, peak - offset, 0)
            if compatible and onset < offset and displacement <= round(0.010 * 250):
                waves[name].append([onset, peak, offset])
                audit["complete"] += 1
                if displacement:
                    audit["recovered"].append(
                        {"events": [index, index + 1, index + 2], "displacement": displacement}
                    )
                index += 3
                accepted = True
        if accepted:
            continue
        name = types.get(symbols[index])
        if name is not None and index + 1 < len(samples) and symbols[index + 1] == ")":
            peak, offset = map(int, samples[index : index + 2])
            number = int(nums[index + 1])
            if 0 <= number <= 3 and boundary_types[number] == name and offset > peak:
                waves[name].append([-1, peak, offset])
                audit["offset_only"] += 1
                index += 2
                continue
        audit["unknown"].append(
            {"event": index, "sample": int(samples[index]), "symbol": symbols[index], "num": int(nums[index])}
        )
        index += 1
    return {name: np.asarray(values, dtype=int).reshape(-1, 3) for name, values in waves.items()}, audit


def joint_hybrid(signal: np.ndarray, fs: int = 250) -> tuple[np.ndarray, dict]:
    """
    Infer joint channel support with fixed P and unchanged adaptive QRS/T.

    Parameters
    ----------
    signal : np.ndarray
        Two-channel physical waveform, channels by samples.
    fs : int
        Native sampling frequency.

    Returns
    -------
    tuple[np.ndarray, dict]
        Common raw anchors and fixed/hybrid joint intervals.
    """
    peaks, adaptive = adaptive_boundaries(signal, fs)
    fixed = fixed_boundaries(peaks, leads=signal.shape[0], fs=fs)
    adaptive[:, :, 0] = fixed[:, :, 0]
    joint = np.full((len(peaks), 3, 2), -1, dtype=int)
    for index in range(len(peaks)):
        for wave in range(3):
            valid = adaptive[index, :, wave, 0] >= 0
            if valid.any():
                values = adaptive[index, valid, wave]
                joint[index, wave] = [values[:, 0].min(), values[:, 1].max()]
    if not np.array_equal(joint[:, 0], fixed[:, 0, 0]):
        raise ValueError("Hybrid P differs from frozen fixed P")
    return peaks, {"fixed": fixed[:, 0], "hybrid": joint}


def evaluate_joint(
    peaks: np.ndarray, predictions: dict, annotations: dict, fs: int = 250
) -> tuple[list, dict]:
    """
    Include every eligible known truth endpoint and penalize failed predicted slots.

    Parameters
    ----------
    peaks, predictions, annotations : array-like or dict
        Independent raw anchors, joint predicted intervals, and sparse expert truth.
    fs : int
        Native sampling frequency.

    Returns
    -------
    tuple[list, dict]
        Per-wave complete/offset measurements and common anchor coverage.
    """
    qrs_all = annotations["QRS"]
    qrs = qrs_all[qrs_all[:, 0] >= 0]
    matched = match_peaks(peaks, qrs, fs)
    rows = []
    unknown_slots = {}
    for wave_index, name in enumerate(WAVE_NAMES):
        truth = qrs if name == "QRS" else annotations[name]
        associated = associate_waves(truth, qrs, name, fs)
        used = set()
        for index, (onset, peak, offset) in enumerate(truth):
            complete = onset >= round(0.5 * fs) and offset <= round(899.5 * fs)
            offset_eligible = peak >= round(0.5 * fs) and offset <= round(899.5 * fs)
            if not complete and not (name == "T" and offset_eligible):
                continue
            qrs_index = associated.get(index)
            anchor = matched.get(qrs_index) if qrs_index is not None else None
            if anchor is not None:
                if anchor in used:
                    raise ValueError("Sparse expert matching reused predicted slot")
                used.add(anchor)
            row = {
                "wave": name,
                "truth": [int(onset), int(peak), int(offset)],
                "complete": bool(complete),
                "anchor": anchor,
            }
            for method, intervals in predictions.items():
                predicted = intervals[anchor, wave_index] if anchor is not None else np.array([-1, -1])
                result = interval_measurements(predicted, np.array([onset, offset]), fs) if complete else {}
                result["offset_mae_ms"] = (
                    float(abs(predicted[1] - offset) * 1000 / fs) if predicted[0] >= 0 else 500.0
                )
                result["missing"] = bool(predicted[0] < 0)
                result["predicted"] = predicted.tolist()
                row[method] = result
            rows.append(row)
        interior = set(np.flatnonzero((peaks >= round(0.5 * fs)) & (peaks <= round(899.5 * fs))))
        unknown_slots[name] = {
            method: sum(int(intervals[index, wave_index, 0] >= 0) for index in interior - used)
            for method, intervals in predictions.items()
        }
    return rows, {
        "raw_anchors": len(peaks),
        "complete_qrs_truth": len(qrs),
        "matched_qrs": len(matched),
        "unmatched_truth": sum(row["anchor"] is None for row in rows),
        "unannotated_prediction_slots_unknown": unknown_slots,
    }


def summarize(rows: list[dict]) -> dict:
    """
    Average known endpoints by record without excluding missing predictions.

    Parameters
    ----------
    rows : list[dict]
        Complete wave and known T offset comparisons.

    Returns
    -------
    dict
        Record class means and distinct all-known-T-offset guard population.
    """
    result = {"classes": {}, "t_offset": {}}
    for name in WAVE_NAMES:
        selected = [row for row in rows if row["wave"] == name and row["complete"]]
        if selected:
            result["classes"][name] = {"count": len(selected)}
            for method in ("fixed", "hybrid"):
                result["classes"][name][method] = {
                    metric: float(np.mean([row[method][metric] for row in selected]))
                    for metric in ("iou", "onset_mae_ms", "offset_mae_ms", "within_30ms", "missing")
                }
    selected = [row for row in rows if row["wave"] == "T"]
    if selected:
        result["t_offset"] = {
            "count": len(selected),
            **{
                method: float(np.mean([row[method]["offset_mae_ms"] for row in selected]))
                for method in ("fixed", "hybrid")
            },
        }
    return result


def aggregate(summaries: dict[str, dict]) -> dict:
    """
    Apply all prospectively fixed record-bootstrap gates.

    Parameters
    ----------
    summaries : dict[str, dict]
        One summary per original record in a declared evaluation population.

    Returns
    -------
    dict
        Primary QRS gain, known-T guard and descriptive wave means.
    """
    ids = np.asarray(list(summaries))
    result = {
        "records": len(ids),
        "bootstrap_unit": "original record; patient independence unknown",
        "waves": {},
    }
    for name in WAVE_NAMES:
        chosen = [key for key in ids if name in summaries[key]["classes"]]
        if not chosen:
            continue
        values = {
            method: {
                metric: np.array([summaries[key]["classes"][name][method][metric] for key in chosen])
                for metric in ("iou", "onset_mae_ms", "offset_mae_ms", "within_30ms", "missing")
            }
            for method in ("fixed", "hybrid")
        }
        result["waves"][name] = {
            "records": len(chosen),
            "truth": sum(summaries[key]["classes"][name]["count"] for key in chosen),
            **{
                method: {metric: float(array.mean()) for metric, array in found.items()}
                for method, found in values.items()
            },
            "iou_gain": mean_interval(
                np.array(chosen), values["hybrid"]["iou"] - values["fixed"]["iou"], seed=59059
            ),
        }
    qrs = {
        method: np.array(
            [summaries[key]["classes"].get("QRS", {}).get(method, {}).get("iou", 0.0) for key in ids]
        )
        for method in ("fixed", "hybrid")
    }
    boundary = np.array(
        [
            summaries[key]["classes"].get("QRS", {}).get("hybrid", {}).get("within_30ms", 0.0)
            - summaries[key]["classes"].get("QRS", {}).get("fixed", {}).get("within_30ms", 0.0)
            for key in ids
        ]
    )
    t_ids = np.array([key for key in ids if summaries[key]["t_offset"]])
    t_change = np.array(
        [summaries[key]["t_offset"]["hybrid"] - summaries[key]["t_offset"]["fixed"] for key in t_ids]
    )
    result["primary_gain"] = mean_interval(ids, qrs["hybrid"] - qrs["fixed"], seed=59059)
    result["within30ms_gain"] = mean_interval(ids, boundary, seed=59059)
    result["t_offset_deterioration_ms"] = mean_interval(t_ids, t_change, seed=59059)
    result["t_guard_records"] = len(t_ids)
    result["qrs_records_without_truth"] = len(ids) - result["waves"].get("QRS", {}).get("records", 0)
    result["gates"] = {
        "practical_iou": result["primary_gain"]["estimate"] >= 0.10,
        "supported_iou": result["primary_gain"]["ci_low"] > 0,
        "boundary_gain": result["within30ms_gain"]["estimate"] >= 0.10,
        "t_guard": result["t_offset_deterioration_ms"]["ci_high"] <= 10.0,
    }
    result["worth_reviewing"] = all(result["gates"].values())
    return result


def reproduce_wave_metrics(summaries: dict, metrics: dict) -> dict:
    """
    Reproduce every frozen predecessor wave metric and paired IoU interval.

    Parameters
    ----------
    summaries, metrics : dict
        Reconstructed per-record summaries and frozen aggregate measurements.

    Returns
    -------
    dict
        Verified complete per-wave metric values.
    """
    ids = np.array(list(summaries))
    wave_integrity = {}
    for name in WAVE_NAMES:
        chosen = [key for key in ids if name in summaries[key]["classes"]]
        old = metrics["waves"][name]
        checked = {}
        for method in ("fixed", "adaptive"):
            for metric in ("iou", "onset_mae_ms", "offset_mae_ms", "within_30ms", "missing"):
                value = float(np.mean([summaries[key]["classes"][name][method][metric] for key in chosen]))
                if value != old[method][metric]:
                    raise ValueError(f"Predecessor055 {name}/{method}/{metric} differs")
            checked[method] = old[method]
        difference = np.array(
            [
                summaries[key]["classes"][name]["adaptive"]["iou"]
                - summaries[key]["classes"][name]["fixed"]["iou"]
                for key in chosen
            ]
        )
        if mean_interval(np.array(chosen), difference, seed=55055) != old["iou_gain"]:
            raise ValueError(f"Predecessor055 {name} interval differs")
        wave_integrity[name] = checked
    return wave_integrity


def reproduce_055(root: Path) -> dict:
    """
    Verify immutable predecessor receipts and exactly reconstruct its primary.

    Parameters
    ----------
    root : Path
        Repository root using relative stored receipt paths.

    Returns
    -------
    dict
        Exact primary means/interval and verified source/input/output counts.
    """
    directory = root / "outputs/experiment055_ludb_boundaries_v1"
    saved = json.loads((directory / "result.json").read_text())
    counts = {}
    for name, hashes in [
        ("sources", saved["identity"]["sources"]),
        ("inputs", saved["input_hashes"]),
        ("outputs", saved["output_hashes"]),
    ]:
        checked = 0
        for path, expected in hashes.items():
            if path.endswith("/run.log"):
                continue
            if sha256_file(root / path) != expected:
                raise ValueError(f"Predecessor055 {name} mismatch: {path}")
            checked += 1
        counts[name] = checked
    summaries = {}
    for index in range(1, 201):
        record = json.loads((directory / "records" / f"{index}.json").read_text())
        summary = record_summary(record["wave_measurements"])
        if summary != record["summary"]:
            raise ValueError("Predecessor055 record summary mismatch")
        summaries[index] = summary
    ids = np.array(list(summaries))
    values = {
        method: np.array([summaries[key]["primary"][method] for key in ids])
        for method in ("fixed", "adaptive")
    }
    gain = mean_interval(ids, values["adaptive"] - values["fixed"], seed=55055)
    metrics = saved["metrics"]
    if (
        gain != metrics["primary_gain"]
        or values["fixed"].mean() != metrics["fixed_primary_iou"]
        or values["adaptive"].mean() != metrics["adaptive_primary_iou"]
    ):
        raise ValueError("Predecessor055 primary reproduction differs")
    wave_integrity = reproduce_wave_metrics(summaries, metrics)
    return {
        "verified": counts,
        "records": 200,
        "fixed_iou": float(values["fixed"].mean()),
        "adaptive_iou": float(values["adaptive"].mean()),
        "gain": gain,
        "wave_metrics": wave_integrity,
        "excluded_mutable_log": True,
    }


def read_qtdb(root: Path, record_id: str) -> tuple[np.ndarray, dict, dict]:
    """
    Read native two-channel data and expert annotation coverage only.

    Parameters
    ----------
    root : Path
        Local QTDB 1.0.0 directory.
    record_id : str
        Official source-derived identifier.

    Returns
    -------
    tuple[np.ndarray, dict, dict]
        Physical signal, truth and annotation/lead audit.
    """
    stem = str(root / record_id)
    record = wfdb.rdrecord(stem)
    if (
        record.fs != 250
        or record.sig_len not in (225000, 224999, 224993)
        or record.n_sig != 2
        or not np.isfinite(record.p_signal).all()
    ):
        raise ValueError("QTDB waveform format differs from frozen protocol")
    annotation = wfdb.rdann(stem, "q1c")
    truth, audit = parse_annotations(annotation.sample, annotation.symbol, annotation.num)
    return (
        record.p_signal.T,
        truth,
        {
            "leads": record.sig_name,
            "units": record.units,
            "gains": record.adc_gain,
            "sampling_rate": record.fs,
            "samples": record.sig_len,
            "annotation": audit,
            "wave_counts": {name: len(array) for name, array in truth.items()},
            "edb_source": record_id.startswith("sele"),
            "original_record": record_id[3:],
        },
    )
