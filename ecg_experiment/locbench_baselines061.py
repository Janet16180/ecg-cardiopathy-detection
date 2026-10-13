"""
Baseline maps and predecessor reproductions for Experiment 061 (LocBench v1).

Every baseline is built from frozen modules by import: ``lead_wave_maps`` (U_B), ``incart_localization056``
and ``raw_residual054`` (aligned residual), ``hybrid_boundaries059_v2`` (QRS prior). See
``docs/experiment-061-locbench.md`` and its Amendment 1.
"""

from __future__ import annotations

import json
import pickle
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from sklearn.covariance import LedoitWolf

from ecg_experiment import ROOT
from ecg_experiment.external_encoders import read_ptb_float64
from ecg_experiment.files import sha256_file
from ecg_experiment.fragment_localization import r_peaks
from ecg_experiment.full_development import cohorts, ptb_table
from ecg_experiment.hybrid_boundaries059_v2 import joint_hybrid
from ecg_experiment.incart_localization056 import read_record, score_signal
from ecg_experiment.lead_wave_maps import (
    MINIMUM_BEATS,
    WAVES,
    UnitMap,
    beat_pieces,
    beat_unit_map,
    fit_wave_references,
    wave_scores,
)
from ecg_experiment.locbench import SAMPLES, SEED, interval_mask

FS = 500
METHODS = ("random", "amplitude", "qrs_prior", "U_B", "residual056")
MEDIAN_SAMPLES = 301
QRS_TIE_WEIGHT = 1e-3
PTB_GAIN = 1000.0
OUTPUT042 = ROOT / "outputs/experiment042_lead_wave_maps_v1"
OUTPUT056 = ROOT / "outputs/experiment056_incart_localization_v1"
INCART_RAW = ROOT / "data/raw/incart/1.0.0"
CACHE_IDS042 = (
    ROOT / "data/processed/pretrained/ecg-jepa-full-public/ecg_ids.npy",
    ROOT / "outputs/experiment016_xecg_probe_finetune/features/ecg_ids.npy",
    ROOT / "outputs/experiment004_cpc_40k/released_features/ecg_ids.npy",
)
EXPECTED_FIT042 = (5872, 5537)
REPRODUCTION_TOLERANCE = 1e-10
INCART_CORE = 5000
INCART_CONTEXT = 250


def exact_signal(stored: np.ndarray) -> np.ndarray:
    """
    Exact float64 PTB-XL physical signal from its float32 cache copy (gain 1,000, baseline 0).

    Parameters
    ----------
    stored : np.ndarray
        Float32 signal (12, 5000) in mV.

    Returns
    -------
    np.ndarray
        Float64 signal equal to ``wfdb.rdsamp``.
    """
    return np.round(np.asarray(stored, dtype=np.float64) * PTB_GAIN) / PTB_GAIN


def random_map(ecg_id: int) -> np.ndarray:
    """
    Iid uniform map of one record, seeded by ``[61061, ecg_id]``.

    Parameters
    ----------
    ecg_id : int
        PTB-XL record ID.

    Returns
    -------
    np.ndarray
        Float32 map (12, 5000).
    """
    return np.random.default_rng([SEED, int(ecg_id)]).random((12, SAMPLES)).astype(np.float32)


def amplitude_map(signal: np.ndarray) -> np.ndarray:
    """
    Absolute signal after subtracting a centred 0.6 s running median per lead.

    Parameters
    ----------
    signal : np.ndarray
        Signal (12, 5000) in mV.

    Returns
    -------
    np.ndarray
        Float64 map (12, 5000).
    """
    return np.abs(signal - median_filter(signal, size=(1, MEDIAN_SAMPLES), mode="nearest"))


def qrs_prior_map(signal: np.ndarray) -> tuple[np.ndarray, int]:
    """
    One inside 059 hybrid QRS intervals (all leads), zero elsewhere, plus 1e-3 times ``amplitude_map``.

    Parameters
    ----------
    signal : np.ndarray
        Signal (12, 5000) in mV.

    Returns
    -------
    tuple[np.ndarray, int]
        Float64 map (12, 5000) and the number of valid QRS intervals.
    """
    _, supports = joint_hybrid(signal, fs=FS)
    qrs = supports["hybrid"][:, 1]
    valid = qrs[(qrs[:, 0] >= 0) & (qrs[:, 1] > qrs[:, 0])]
    inside = interval_mask(valid, signal.shape[1]).astype(np.float64)
    return inside[None, :] + QRS_TIE_WEIGHT * amplitude_map(signal), len(valid)


def unit_field(unit_map: UnitMap, samples: int = SAMPLES) -> np.ndarray | None:
    """
    Broadcast unit scores to their lead and samples, maximum where units overlap, record minimum elsewhere.

    Parameters
    ----------
    unit_map : UnitMap
        Units with lead and time span in seconds; a unit fills [round(500 start), round(500 end)).
    samples : int
        Record length.

    Returns
    -------
    np.ndarray or None
        Float64 map (12, samples), or None if no unit covers a sample.
    """
    field = np.full((12, samples), -np.inf)
    for score, lead, start, end in zip(
        unit_map.scores, unit_map.leads, unit_map.starts, unit_map.ends, strict=True
    ):
        left, right = max(0, round(start * FS)), min(samples, round(end * FS))
        field[int(lead), left:right] = np.maximum(field[int(lead), left:right], score)
    covered = np.isfinite(field)
    if not covered.any():
        return None
    field[~covered] = field[covered].min()
    return field


def ub_map(signal: np.ndarray, references: dict[str, list[LedoitWolf]]) -> np.ndarray | None:
    """
    Experiment 042 U_B lead/wave map broadcast to samples.

    Parameters
    ----------
    signal : np.ndarray
        Float64 signal (12, 5000) in mV.
    references : dict[str, list[LedoitWolf]]
        Output of ``fit_wave_references``.

    Returns
    -------
    np.ndarray or None
        Map, or None with fewer than two kept beats.
    """
    r_times, pieces = beat_pieces(signal, FS)
    if len(r_times) < MINIMUM_BEATS:
        return None
    return unit_field(beat_unit_map(wave_scores(references, pieces), r_times), signal.shape[1])


def residual_map(signal: np.ndarray, references: dict[str, list[LedoitWolf]]) -> np.ndarray | None:
    """
    Experiment 056 uncalibrated aligned residual (140 ms windows) broadcast to samples.

    Parameters
    ----------
    signal : np.ndarray
        Float64 signal (12, 5000) in mV.
    references : dict[str, list[LedoitWolf]]
        U_B references; they only set the piece-coverage mask.

    Returns
    -------
    np.ndarray or None
        Map, or None with fewer than three complete beats.
    """
    try:
        _, maps = score_signal(signal, references)
    except ValueError as error:
        if "Fewer than three complete beats" in str(error):
            return None
        raise
    return unit_field(maps["aligned_residual"], signal.shape[1])


def collect_pieces(signals: list[np.ndarray]) -> tuple[dict[str, np.ndarray], int, int]:
    """
    Kept U_B wave pieces of every record with at least two kept beats.

    Parameters
    ----------
    signals : list[np.ndarray]
        Float64 signals (12, 5000).

    Returns
    -------
    tuple[dict[str, np.ndarray], int, int]
        Concatenated pieces per wave, kept beats, and records with fewer than two kept beats.
    """
    found: dict[str, list[np.ndarray]] = {name: [] for name in WAVES}
    beats, dropped = 0, 0
    for signal in signals:
        r_times, pieces = beat_pieces(signal, FS)
        if len(r_times) < MINIMUM_BEATS:
            dropped += 1
            continue
        beats += len(r_times)
        for name in WAVES:
            found[name].append(pieces[name])
    return {name: np.concatenate(values) for name, values in found.items()}, beats, dropped


def fit_set042() -> pd.DataFrame:
    """
    Experiment 042's 5,872-record normal fit set, rebuilt with its selection rule.

    Returns
    -------
    pd.DataFrame
        Rows of ``full_development.ptb_table`` in the 042 order.
    """
    common = set.intersection(*(set(np.load(path).astype(np.int64).tolist()) for path in CACHE_IDS042))
    train = cohorts(ptb_table())["train"]
    pool = train[train["standard"].notna() & train["ecg_id"].isin(common)]
    fit = pool[pool["standard"] == 0]
    if (len(fit), fit["patient_id"].nunique()) != EXPECTED_FIT042:
        raise ValueError(f"042 fit set changed: {len(fit)} records")
    return fit


def relative_difference(new: np.ndarray, saved: np.ndarray) -> float:
    """
    Largest relative difference, with a 1e-12 floor on the denominator.

    Parameters
    ----------
    new, saved : np.ndarray
        Arrays of equal shape.

    Returns
    -------
    float
        Maximum of ``|new - saved| / max(|saved|, 1e-12)``, 0 for empty arrays.
    """
    if not len(saved):
        return 0.0
    return float((np.abs(new - saved) / np.maximum(np.abs(saved), 1e-12)).max())


def reproduce042(signals: np.ndarray, index: pd.DataFrame, records: int = 20) -> dict[str, Any]:
    """
    Recompute U_B with 042's 5,872-normal reference on the first saved fold-9 records and compare units.

    Parameters
    ----------
    signals : np.ndarray
        LocBench cache signals.
    index : pd.DataFrame
        LocBench cache index.
    records : int
        Number of saved records compared.

    Returns
    -------
    dict[str, Any]
        Counts, maximum differences and ``passed``.
    """
    receipt = json.loads((OUTPUT042 / "result.json").read_text())
    saved_hash = sha256_file(OUTPUT042 / "unit_scores.npz")
    if saved_hash != receipt["outputs_sha256"]["unit_scores.npz"]:
        raise ValueError("042 unit_scores.npz differs from its receipt")
    fit = fit_set042()
    position = pd.Series(np.arange(len(index)), index=index["ecg_id"].to_numpy())
    pieces, beats, dropped = collect_pieces(
        [exact_signal(signals[int(position[ecg_id])]) for ecg_id in fit["ecg_id"]]
    )
    if beats != receipt["arm_b"]["fit_beats"] or dropped:
        raise ValueError(f"042 fit beats changed: {beats}, {dropped} dropped")
    references = fit_wave_references(pieces)
    del pieces
    table = ptb_table().set_index("ecg_id")
    worst = {"scores_relative": 0.0, "times_absolute": 0.0}
    geometry_equal = True
    checked = []
    with np.load(OUTPUT042 / "unit_scores.npz") as saved:
        ecg_ids = saved["ecg_ids"][:records]
        offsets = saved["U_B_offsets"]
        for row, ecg_id in enumerate(ecg_ids):
            if table.at[int(ecg_id), "strat_fold"] != 9:
                raise ValueError(f"Record {ecg_id} is not fold 9")
            signal = read_ptb_float64(table.at[int(ecg_id), "filename_hr"])
            r_times, wave_pieces = beat_pieces(signal, FS)
            units = beat_unit_map(wave_scores(references, wave_pieces), r_times)
            span = slice(offsets[row], offsets[row + 1])
            old = {field: saved[f"U_B_{field}"][span] for field in ("scores", "leads", "starts", "ends")}
            if len(units.scores) != len(old["scores"]) or not np.array_equal(units.leads, old["leads"]):
                geometry_equal = False
                continue
            worst["times_absolute"] = max(
                worst["times_absolute"],
                float(np.abs(units.starts - old["starts"]).max()),
                float(np.abs(units.ends - old["ends"]).max()),
            )
            worst["scores_relative"] = max(
                worst["scores_relative"], relative_difference(units.scores, old["scores"])
            )
            checked.append(int(ecg_id))
    passed = (
        geometry_equal
        and worst["scores_relative"] <= REPRODUCTION_TOLERANCE
        and worst["times_absolute"] <= 1e-12
    )
    return {
        "fit_records": len(fit),
        "fit_patients": int(fit["patient_id"].nunique()),
        "fit_beats": beats,
        "records": checked,
        "geometry_equal": geometry_equal,
        **{f"max_{k}": v for k, v in worst.items()},
        "unit_scores_sha256": saved_hash,
        "passed": bool(passed),
    }


def incart_beat_scores(signal: np.ndarray, references: dict[str, list[LedoitWolf]]) -> pd.DataFrame:
    """
    Rescore an INCART record core by core exactly as Experiment 056 did (without its nuisance copy).

    Parameters
    ----------
    signal : np.ndarray
        Record at 500 Hz from ``incart_localization056.read_record``.
    references : dict[str, list[LedoitWolf]]
        056's saved U_B references.

    Returns
    -------
    pd.DataFrame
        ``r_sample`` (absolute), ``core``, ``aligned_residual``, ``U_B_fixed``.
    """
    frames = []
    for core, start in enumerate(range(0, signal.shape[1], INCART_CORE)):
        stop = min(start + INCART_CORE, signal.shape[1])
        left, right = max(0, start - INCART_CONTEXT), min(signal.shape[1], stop + INCART_CONTEXT)
        context = signal[:, left:right]
        peaks = r_peaks(context, FS)
        if np.count_nonzero((peaks >= 125) & (peaks + 225 <= context.shape[1])) < 3:
            continue
        values, _ = score_signal(context, references, peaks)
        values["r_sample"] += left
        values = values[(values["r_sample"] >= start) & (values["r_sample"] < stop)].copy()
        values["core"] = core
        frames.append(values[["r_sample", "core", "aligned_residual", "U_B_fixed"]])
    return pd.concat(frames, ignore_index=True)


def reproduce056(records: tuple[str, ...] = ("I01", "I02", "I03")) -> dict[str, Any]:
    """
    Rescore INCART records with 056's saved reference and compare the saved beat scores.

    Parameters
    ----------
    records : tuple[str, ...]
        INCART record names.

    Returns
    -------
    dict[str, Any]
        Beat counts, maximum relative differences and ``passed``.
    """
    receipt = json.loads((OUTPUT056 / "result.json").read_text())
    hashes = {name: sha256_file(OUTPUT056 / name) for name in ("normal_reference.pkl", "detected_beats.csv")}
    for name, value in hashes.items():
        if value != receipt["outputs_sha256"][name]:
            raise ValueError(f"056 {name} differs from its receipt")
    with (OUTPUT056 / "normal_reference.pkl").open("rb") as source:
        references = pickle.load(source)
    saved = pd.read_csv(OUTPUT056 / "detected_beats.csv")
    result: dict[str, Any] = {"records": {}, "sha256": hashes}
    passed = True
    for record in records:
        signal, _ = read_record(INCART_RAW / record)
        new = incart_beat_scores(signal, references).sort_values(["core", "r_sample"]).reset_index(drop=True)
        old = saved[saved["record"] == record].sort_values(["core", "r_sample"]).reset_index(drop=True)
        same_beats = new[["core", "r_sample"]].equals(old[["core", "r_sample"]])
        differences = {
            name: relative_difference(new[name].to_numpy(), old[name].to_numpy()) if same_beats else np.inf
            for name in ("aligned_residual", "U_B_fixed")
        }
        ok = same_beats and max(differences.values()) <= REPRODUCTION_TOLERANCE
        passed &= ok
        result["records"][record] = {
            "beats": len(new),
            "saved_beats": len(old),
            "same_beats": same_beats,
            **{f"max_relative_{k}": v for k, v in differences.items()},
            "passed": ok,
        }
    result["passed"] = bool(passed)
    return result
