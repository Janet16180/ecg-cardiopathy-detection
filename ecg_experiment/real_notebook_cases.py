"""Read real ECG review examples and frozen predictions without running inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wfdb

from ecg_experiment.files import sha256_file
from ecg_experiment.paths import to_stored

COLORS = {"expert": "#4daf4a", "previous": "#ef8a62", "candidate": "#2166ac"}


def _records(table: pd.DataFrame) -> list[dict[str, Any]]:
    """Preserve absent match measurements as JSON null rather than NaN.

    Parameters
    ----------
    table : pd.DataFrame
        Saved annotation or detector rows, possibly containing missing match distances.

    Returns
    -------
    list of dict
        JSON-compatible rows with explicit nulls for unavailable values.
    """
    return table.astype(object).where(pd.notna(table), None).to_dict("records")


def _check_paths(root: Path, paths: dict[str, str]) -> dict[str, str]:
    """Verify selected real input/output files against their frozen receipt.

    Parameters
    ----------
    root : Path
        Repository root containing local real waveforms and completed outputs.
    paths : dict of str to str
        Repository-relative paths and expected SHA-256 hashes.

    Returns
    -------
    dict
        Verified stored paths and hashes for the case provenance.
    """
    actual = {to_stored(root / path): sha256_file(root / path) for path in paths}
    if actual != paths:
        raise ValueError("Real review input differs from its frozen receipt")
    return actual


def _native_excerpt(path: Path, start: int, stop: int) -> dict[str, Any]:
    """Read native, unfiltered physical ECG samples on the original time axis.

    Parameters
    ----------
    path : Path
        Local WFDB record stem.
    start, stop : int
        Native sample bounds; stop is exclusive.

    Returns
    -------
    dict
        JSON-compatible original mV samples, leads, sampling frequency and absolute times.
    """
    record = wfdb.rdrecord(str(path), sampfrom=start, sampto=stop, physical=True)
    if set(record.units) != {"mV"} or not np.isfinite(record.p_signal).all():
        raise ValueError("Expected finite physical ECG samples in mV")
    return {
        "fs": float(record.fs),
        "units": "mV",
        "leads": [name.replace("AV", "aV") for name in record.sig_name],
        "signal": record.p_signal.T.tolist(),
        "time": (np.arange(start, stop) / record.fs).tolist(),
        "time_reference": "absolute seconds from the original recording start",
        "amplitude_scale": "Header-derived mV; absolute amplitude calibration was not independently verified",
        "adc_gain": list(record.adc_gain),
        "adc_baseline": list(record.baseline),
    }


def _incart_case(root: Path, core: int) -> dict[str, Any]:
    """Extract a full real twelve-lead core with both saved top-beat decisions.

    Parameters
    ----------
    root : Path
        Repository root.
    core : int
        Frozen ten-second core number of INCART I01: 13 for a gain, 0 for a failure.

    Returns
    -------
    dict
        Raw native excerpt, expert orientation bands and unchanged saved method marks/scores.
    """
    base = root / "outputs/experiment056_incart_localization_v1"
    receipt = json.loads((base / "result.json").read_text())
    raw = root / "data/raw/incart/1.0.0/I01"
    requested = [to_stored(raw.with_suffix(extension)) for extension in (".hea", ".dat", ".atr")]
    hashes = {path: receipt["identity"]["inputs"][path] for path in requested}
    for name in ("detected_beats.csv", "reference_beats.csv", "mixed_cores.csv"):
        hashes[to_stored(base / name)] = receipt["outputs_sha256"][name]
    provenance = _check_paths(root, hashes)
    detected = pd.read_csv(base / "detected_beats.csv", float_precision="round_trip")
    selected = detected[(detected.record == "I01") & (detected.core == core)]
    truth = pd.read_csv(base / "reference_beats.csv", float_precision="round_trip")
    truth = truth[(truth.record == "I01") & (truth.core == core)]
    metrics = pd.read_csv(base / "mixed_cores.csv", float_precision="round_trip")
    metrics = metrics[(metrics.record == "I01") & (metrics.core == core)]
    header = wfdb.rdheader(str(raw))
    if header.fs != 257 or header.n_sig != 12:
        raise ValueError("INCART native sample rate or lead count changed")
    payload = _native_excerpt(raw, core * 10 * 257, (core + 1) * 10 * 257)
    marks = [
        {
            "label": "Expert ventricular-beat timestamp (orientation only)",
            "lead": None,
            "start": float(row.time_seconds) - 0.1,
            "end": float(row.time_seconds) + 0.1,
            "color": COLORS["expert"],
            "method": "expert",
        }
        for row in truth[truth.symbol == "V"].itertuples()
    ]
    results = []
    for role, method in (("previous", "U_B_fixed"), ("candidate", "aligned_residual")):
        top = selected.loc[selected[method].idxmax()]
        tied = selected[np.isclose(selected[method], top[method], rtol=1e-10, atol=1e-10)]
        mark = {
            "label": f"{role.capitalize()} method: {method}",
            "lead": int(top[f"{method}_lead"]),
            "start": float(top[f"{method}_start"]),
            "end": float(top[f"{method}_start"]) + 0.14,
            "color": COLORS[role],
            "method": method,
        }
        marks.append(mark)
        saved_metric = metrics[metrics["map"] == method].iloc[0]
        results.append(
            {
                "role": role,
                "method": method,
                "selected_beat_type": str(top.symbol),
                "selected_detected_r_time": float(top.r_sample) / 500,
                "selected_reference_index": int(top.reference_index),
                "score": float(top[method]),
                "top_beat_ties": len(tied),
                "saved_expected_ventricular_hit": float(saved_metric.hit),
                "saved_candidate_pool_chance": float(saved_metric.chance),
            }
        )
    return {
        "id": f"incart_I01_core{core}",
        "kind": "incart",
        "title": "Finding the unusual beat: a successful example"
        if core == 13
        else "Finding the unusual beat: a failure example",
        "source": "Real INCART I01 twelve-lead Holter recording",
        "source_url": "https://physionet.org/content/incartdb/1.0.0/",
        **payload,
        "marks": marks,
        "method_results": results,
        "expert_annotations": _records(truth[["time_seconds", "symbol", "matched"]]),
        "detected_beats": _records(selected),
        "annotation_notes": [
            "Green shows +/-100 ms around an expert ventricular-beat timestamp; "
            "annotation times were not manually corrected.",
            "Green is an orientation band, not an expert QRS boundary or a pathological wave annotation.",
            "Orange and blue are saved 140 ms marks, assigned to detected beats by midpoint ownership.",
            "A correct beat choice does not prove the marked wave segment or lead is pathological.",
            "The raw waveform is native 257 Hz; the saved experiment used 500 Hz resampling. "
            "All marks share the original time axis.",
            "This is a descriptive saved success/failure example, not an accuracy estimate.",
        ],
        "selection_reason": "Previously documented I01 core 13 gain"
        if core == 13
        else "Previously documented I01 core 0 regression",
        "plain_language": (
            "The blue method chooses the beat marked ventricular by the expert; the orange method "
            "chooses a normal beat. The blue interval falls after QRS, so this demonstrates finding "
            "the right beat, not proving the highlighted wave or lead is abnormal."
            if core == 13
            else "The orange method chooses the expert ventricular beat, but the blue method chooses "
            "a normal beat. This is an explicit failure of the newer method on a real ECG."
        ),
        "provenance": {"verified_files": provenance, "receipt": to_stored(base / "result.json")},
    }


def _qtdb_case(root: Path, record: str, worst: bool) -> dict[str, Any]:
    """Extract a real two-channel QRS example with actual expert and saved model borders.

    Parameters
    ----------
    root : Path
        Repository root.
    record : str
        QTDB record ID, sel100 or sel49.
    worst : bool
        Select the lowest hybrid-minus-fixed saved QRS IoU, instead of the first complete QRS.

    Returns
    -------
    dict
        Native mV excerpt and joint expert/fixed/hybrid intervals in absolute seconds.
    """
    base = root / "outputs/experiment059_qtdb_hybrid_v2"
    receipt = json.loads((base / "result.json").read_text())
    raw = root / "data/raw/qtdb/1.0.0" / record
    hashes = {
        to_stored(raw.with_suffix(extension)): receipt["input_hashes"][to_stored(raw.with_suffix(extension))]
        for extension in (".hea", ".dat", ".q1c")
    }
    for suffix in (".json", "_predictions.npz"):
        path = to_stored(base / "records" / f"{record}{suffix}")
        hashes[path] = receipt["output_hashes"][path]
    provenance = _check_paths(root, hashes)
    rows = json.loads((base / "records" / f"{record}.json").read_text())["measurements"]
    complete = [row for row in rows if row["wave"] == "QRS" and row["complete"]]
    row = (
        min(complete, key=lambda item: item["hybrid"]["iou"] - item["fixed"]["iou"]) if worst else complete[0]
    )
    manual = wfdb.rdann(str(raw), "q1c")
    triplet = np.asarray(row["truth"])
    matches = [
        index
        for index in range(len(manual.sample) - 2)
        if np.array_equal(manual.sample[index : index + 3], triplet)
    ]
    if not matches or manual.symbol[matches[0]] != "(" or manual.symbol[matches[0] + 2] != ")":
        raise ValueError("Saved QRS truth is absent from the raw expert annotation")
    with np.load(base / "records" / f"{record}_predictions.npz") as saved:
        for method in ("fixed", "hybrid"):
            if not np.array_equal(saved[method][row["anchor"], 1], row[method]["predicted"]):
                raise ValueError("Saved QRS table differs from its frozen prediction arrays")
        detected_r = int(saved["peaks"][row["anchor"]])
    header = wfdb.rdheader(str(raw))
    if header.fs != 250 or header.n_sig != 2:
        raise ValueError("QTDB native sample rate or channel count changed")
    left, right = int(triplet[1]) - 100, int(triplet[1]) + 100
    payload = _native_excerpt(raw, left, right)
    marks = [
        {
            "label": "Expert joint QRS boundary",
            "lead": None,
            "start": int(triplet[0]) / 250,
            "end": int(triplet[2]) / 250,
            "color": COLORS["expert"],
            "method": "expert",
        }
    ]
    results = []
    for role, method in (("previous", "fixed"), ("candidate", "hybrid")):
        bounds = row[method]["predicted"]
        if not row[method]["missing"]:
            marks.append(
                {
                    "label": f"{role.capitalize()} method: {method}",
                    "lead": None,
                    "start": bounds[0] / 250,
                    "end": bounds[1] / 250,
                    "color": COLORS[role],
                    "method": method,
                }
            )
        results.append({"role": role, "method": method, **row[method]})
    return {
        "id": f"qtdb_{record}_qrs_{int(triplet[1])}",
        "kind": "qtdb",
        "title": "Tracing QRS boundaries: a failure example"
        if worst
        else "Tracing QRS boundaries: a successful example",
        "source": f"Real QT Database {record} two-channel recording",
        "source_url": "https://physionet.org/content/qtdb/1.0.0/",
        **payload,
        "marks": marks,
        "method_results": results,
        "expert_annotations": [
            {
                "wave": "QRS",
                "onset": int(triplet[0]) / 250,
                "peak": int(triplet[1]) / 250,
                "offset": int(triplet[2]) / 250,
            }
        ],
        "detected_beats": [{"time_seconds": detected_r / 250}],
        "annotation_notes": [
            "Green is the actual expert onset-to-offset QRS interval from the manual q1c annotation.",
            "The expert and model intervals are joint across both channels; "
            "green is not a separate per-lead pathological region.",
            "Orange shows the saved fixed-window prediction; blue shows the saved adaptive QRS prediction.",
            "QRS overlap measures boundary accuracy, not disease localization, anatomy or diagnosis.",
            "This excerpt uses native unfiltered physical mV samples and the original 250 Hz time axis.",
            "The mV scale comes from the recording header; absolute amplitude calibration was not "
            "independently verified. This example assesses boundary timing, not amplitude interpretation.",
            "This example was chosen from frozen outputs for explanation and is not an accuracy estimate.",
        ],
        "selection_reason": "Lowest saved hybrid-minus-fixed QRS overlap in sel49"
        if worst
        else "First complete expert QRS in sel100",
        "plain_language": (
            "The blue interval extends too far after the expert QRS ending; the orange interval "
            "matches the expert boundaries more closely. This is a real failure example."
            if worst
            else "The blue interval is closer to the expert green QRS boundaries than the orange "
            "interval. This demonstrates improved wave timing on this example, not disease diagnosis."
        ),
        "provenance": {"verified_files": provenance, "receipt": to_stored(base / "result.json")},
    }


def load_real_evidence_cases(root: Path) -> list[dict[str, Any]]:
    """Load four real review examples without fitting, rescoring or reading synthetic data.

    Parameters
    ----------
    root : Path
        Repository root with local INCART/QTDB waveforms and completed 056/059 outputs.

    Returns
    -------
    list of dict
        JSON-compatible case payloads: native lead-by-sample ``signal``, absolute ``time``,
        ``fs``, ``leads``, ``units``, saved ``marks`` and explanatory provenance/notes.
        A mark with ``lead=None`` applies to every displayed lead. Individual cases are descriptive.
    """
    root = Path(root).absolute()
    return [
        _incart_case(root, 13),
        _incart_case(root, 0),
        _qtdb_case(root, "sel100", False),
        _qtdb_case(root, "sel49", True),
    ]
