"""
Per-lead interval criteria and beat-level criteria for Experiment 073.

The 069 criteria table with every global interval (QRS duration, PR, QTc, P duration) replaced by the
same lead's own interval, the fascicular-block criteria resolved to their published lead patterns, and two
beat-level criteria for focal beats. See ``docs/experiment-073-per-lead-criteria.md``.
"""

from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np
import pandas as pd

from ecg_experiment import criteria069 as cr
from ecg_experiment import normative_map062 as nm
from ecg_experiment.morphology_boundaries import adaptive_boundaries
from ecg_experiment.waveforms import LEADS

FS = cr.FS
MS = cr.MS
COMPONENTS = cr.COMPONENTS
INTERVALS = ("QRSDL", "PRL", "QTCL", "PDURL")
REPLACE = {"QRSD": "QRSDL", "PR": "PRL", "QTC": "QTCL", "PDUR": "PDURL"}
FLOORS = {**cr.FLOORS, "QRSDL": 10.0, "PRL": 15.0, "QTCL": 15.0, "PDURL": 10.0}
BEAT_FLOORS = {"pi": 0.02, "delta": 0.05, "duration": 10.0}
DELTA_FLOOR_MV = 0.1
WIDE_QRS = 120.0
PI_QUANTILE = 1.0
DELTA_QUANTILE = 99.0
BEAT_CRITERIA = ("beat_premature", "beat_wide")
BEAT_WAVES = {"beat_premature": ("QRS", "ST", "T"), "beat_wide": ("QRS",)}
RECORD_LEVEL = ("qtc_short",)
GLOBAL_OF = {"QRSDL": "QRSD_GLOBAL", "PRL": "PR_GLOBAL", "QTCL": "QTC_GLOBAL", "PDURL": "PDUR_GLOBAL"}


def _per_lead_conditions(criterion: cr.Criterion) -> cr.Criterion:
    conditions = tuple((REPLACE.get(m, m), sign, t) for m, sign, t in criterion.conditions)
    return replace(criterion, conditions=conditions)


def _fascicular() -> tuple[cr.Criterion, ...]:
    lafb = next(c for c in cr.CRITERIA if c.name == "lafb")
    lpfb = next(c for c in cr.CRITERIA if c.name == "lpfb")
    rs, qr = ("SR", 1, 0.0), ("RS", 1, 0.0)
    return (
        replace(lafb, name="lafb_rs", leads=("II", "III", "aVF"), conditions=(*lafb.conditions, rs)),
        replace(lafb, name="lafb_qr", leads=("aVL",), conditions=(*lafb.conditions, qr)),
        replace(lpfb, name="lpfb_rs", leads=("I", "aVL"), conditions=(*lpfb.conditions, rs)),
        replace(lpfb, name="lpfb_qr", leads=("III", "aVF"), conditions=(*lpfb.conditions, qr)),
    )


def per_lead_criteria() -> tuple[cr.Criterion, ...]:
    """
    Return the 073 criteria table.

    Returns
    -------
    tuple[criteria069.Criterion, ...]
        069 criteria with global intervals replaced by per-lead intervals (except ``RECORD_LEVEL``, protocol
        amendment 1), ``lafb`` and ``lpfb`` split into their published lead patterns.
    """
    out: list[cr.Criterion] = []
    for criterion in cr.CRITERIA:
        if criterion.name in ("lafb", "lpfb"):
            out += [c for c in _fascicular() if c.name.startswith(criterion.name)]
        elif criterion.name in RECORD_LEVEL:
            out.append(criterion)
        else:
            out.append(_per_lead_conditions(criterion))
    return tuple(out)


CRITERIA = per_lead_criteria()


def is_per_lead(measure: str) -> bool:
    """Whether a measure carries a lead suffix."""
    return measure in cr.PER_LEAD or measure in cr.PER_LEAD_DERIVED or measure in INTERVALS


def column(measure: str, lead: str) -> str:
    """Column of a measure in a lead."""
    return f"{measure}_{lead}" if is_per_lead(measure) else measure


def used_columns(criteria: tuple[cr.Criterion, ...] = CRITERIA) -> list[str]:
    """Every measurement column that some criterion reads."""
    names = [column(m, lead) for c in criteria for lead in c.leads for m, _, _ in c.conditions]
    return list(dict.fromkeys(names))


def normal_reference(normals: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    069 ``normal_reference`` with the 073 floors for the per-lead interval columns.

    Parameters
    ----------
    normals : pd.DataFrame
        Measurement table of the reference records.
    columns : list[str]
        Columns to describe.

    Returns
    -------
    pd.DataFrame
        Indexed by column: ``median``, ``iqr_sd``, ``floor``, ``scale``, ``count``.
    """
    ref = cr.normal_reference(normals, columns)
    for name in columns:
        measure = cr.measure_of(name)
        if measure in INTERVALS:
            ref.loc[name, "floor"] = FLOORS[measure]
            ref.loc[name, "scale"] = max(float(ref.loc[name, "iqr_sd"]), FLOORS[measure])
    return ref


def exceedances(
    table: pd.DataFrame,
    scales: pd.Series,
    age: np.ndarray,
    sex: np.ndarray,
    criteria: tuple[cr.Criterion, ...] = CRITERIA,
    offsets: dict[str, float] | None = None,
) -> np.ndarray:
    """
    Signed exceedance of every criterion in every lead it names.

    Parameters
    ----------
    table : pd.DataFrame
        Measurement table with derived and per-lead interval columns, one row per record.
    scales : pd.Series
        Normal spread per column.
    age, sex : np.ndarray
        Per record, in table order.
    criteria : tuple[criteria069.Criterion, ...]
        Criteria table.
    offsets : dict[str, float] or None
        Per-column shift added to the threshold (``perlead_shift``); none by default.

    Returns
    -------
    np.ndarray
        (records, criteria, 12): z, NaN for unnamed leads or missing measurements.
    """
    offsets = offsets or {}
    out = np.full((len(table), len(criteria), len(LEADS)), np.nan)
    for c, criterion in enumerate(criteria):
        for lead in criterion.leads:
            parts = []
            for measure, sign, threshold in criterion.conditions:
                name = column(measure, lead)
                values = table[name].to_numpy(np.float64)
                limit = cr.threshold_values(threshold, age, sex) + offsets.get(name, 0.0)
                parts.append(sign * (values - limit) / float(scales[name]))
            stacked = np.stack(parts)
            combined = stacked.min(axis=0) if criterion.mode == "all" else stacked.max(axis=0)
            out[:, c, LEADS.index(lead)] = np.where(np.isnan(stacked).any(axis=0), np.nan, combined)
    return out


def lead_wave(scores: np.ndarray, criteria: tuple[cr.Criterion, ...] = CRITERIA) -> np.ndarray:
    """
    Lead x wave score: maximum over criteria naming the lead and wave.

    Parameters
    ----------
    scores : np.ndarray
        (records, criteria, 12) non-negative evidence.
    criteria : tuple[criteria069.Criterion, ...]
        Criteria table in the order of ``scores``.

    Returns
    -------
    np.ndarray
        (records, 12, 5) in ``COMPONENTS`` order.
    """
    painted = np.array([[name in c.waves for name in COMPONENTS] for c in criteria])
    out = np.zeros((scores.shape[0], len(LEADS), len(COMPONENTS)))
    for w in range(len(COMPONENTS)):
        rows = np.flatnonzero(painted[:, w])
        if len(rows):
            out[:, :, w] = scores[:, rows, :].max(axis=1)
    return out


def unig_intervals(raw: pd.DataFrame, rr: pd.Series) -> pd.DataFrame:
    """
    Per-lead intervals from Uni-G.

    Parameters
    ----------
    raw : pd.DataFrame
        ``unig_features.csv`` rows indexed by ``ecg_id``.
    rr : pd.Series
        RR interval (ms) per record used for the Framingham QTc, same index.

    Returns
    -------
    pd.DataFrame
        ``QRSDL_X``, ``PRL_X`` (missing without P), ``QTCL_X`` and the Uni-G global QRS, PR and QTc.
    """
    p_found = raw["P_Found_Global"] == 1
    out: dict[str, pd.Series] = {}
    for lead in LEADS:
        out[f"QRSDL_{lead}"] = raw[f"QRS_Dur_{lead}"]
        out[f"PRL_{lead}"] = (raw[f"QRS_On_{lead}"] - raw[f"P_On_{lead}"]).where(p_found)
        out[f"QTCL_{lead}"] = cr.framingham(raw[f"QT_Int_{lead}"], rr)
    out["QRSD_GLOBAL"] = raw["QRS_Dur_Global"]
    out["PR_GLOBAL"] = raw["PR_Int_Global"].where(p_found)
    out["QTC_GLOBAL"] = cr.framingham(raw["QT_Int_Global"], raw["RR_Mean_Global"])
    return pd.DataFrame(out, index=raw.index)


def sl12_intervals(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Per-lead P duration from 12SL.

    Parameters
    ----------
    raw : pd.DataFrame
        ``12sl_features.csv`` rows indexed by ``ecg_id``.

    Returns
    -------
    pd.DataFrame
        ``PDURL_X`` (missing where 12SL reports 0) and the 12SL global P duration.
    """
    out = {f"PDURL_{lead}": raw[f"P_DurFull_{lead}"].where(raw[f"P_DurFull_{lead}"] > 0) for lead in LEADS}
    out["PDUR_GLOBAL"] = raw["P_Off_Global"] - raw["P_On_Global"]
    return pd.DataFrame(out, index=raw.index)


def unig_columns() -> list[str]:
    """Uni-G columns read by ``unig_intervals``."""
    names = [f"{m}_{lead}" for m in ("QRS_Dur", "QRS_On", "P_On", "QT_Int") for lead in LEADS]
    return [
        "ecg_id",
        *names,
        "P_Found_Global",
        "QRS_Dur_Global",
        "PR_Int_Global",
        "QT_Int_Global",
        "RR_Mean_Global",
    ]


def sl12_columns() -> list[str]:
    """12SL columns read by ``sl12_intervals``."""
    return ["ecg_id", *[f"P_DurFull_{lead}" for lead in LEADS], "P_On_Global", "P_Off_Global"]


def shift_offsets(normals: pd.DataFrame) -> dict[str, float]:
    """
    ``perlead_shift`` threshold offsets: per-lead median minus the same source's global median.

    Parameters
    ----------
    normals : pd.DataFrame
        Clean-normal rows with the per-lead interval and ``*_GLOBAL`` columns.

    Returns
    -------
    dict[str, float]
        Offset (ms) per per-lead interval column.
    """
    out = {}
    for measure, global_name in GLOBAL_OF.items():
        centre = float(normals[global_name].median())
        for lead in LEADS:
            out[f"{measure}_{lead}"] = float(normals[f"{measure}_{lead}"].median()) - centre
    return out


def kept_bounds(signal: np.ndarray) -> np.ndarray:
    """
    Per-lead supports of the beats kept by ``combination066.delineated_components``.

    Parameters
    ----------
    signal : np.ndarray
        Raw (12, 5000) waveform in mV.

    Returns
    -------
    np.ndarray
        (beats, 12, 3, 2) supports (P, QRS, T; -1 where missing).
    """
    fid = nm.delineate(signal)
    keep = (fid["qon"] >= 0) & (fid["qoff"] > fid["qon"])
    _, bounds = adaptive_boundaries(signal, FS)
    return bounds[keep]


def own_intervals(bounds: np.ndarray, rr: float) -> dict[str, float]:
    """
    Own per-lead intervals: medians over beats of the per-lead supports.

    Parameters
    ----------
    bounds : np.ndarray
        (beats, 12, 3, 2) supports of the kept beats.
    rr : float
        Own RR (ms) for the QTc.

    Returns
    -------
    dict[str, float]
        ``QRSDL_X``, ``PRL_X``, ``QTCL_X``, ``PDURL_X``; P measures missing when P is found in fewer than half
        of the lead's beats.
    """
    out = dict.fromkeys([f"{m}_{lead}" for m in INTERVALS for lead in LEADS], np.nan)
    if not len(bounds):
        return out
    b = np.where(bounds >= 0, bounds.astype(np.float64), np.nan)
    p_ok = np.isfinite(b[:, :, 0, 0]).mean(axis=0) >= 0.5
    with warnings.catch_warnings(action="ignore"):
        qrs = MS * np.nanmedian(b[:, :, 1, 1] - b[:, :, 1, 0], axis=0)
        pr = MS * np.nanmedian(b[:, :, 1, 0] - b[:, :, 0, 0], axis=0)
        qt = MS * np.nanmedian(b[:, :, 2, 1] - b[:, :, 1, 0], axis=0)
        pdur = MS * np.nanmedian(b[:, :, 0, 1] - b[:, :, 0, 0], axis=0)
    for k, lead in enumerate(LEADS):
        out[f"QRSDL_{lead}"] = float(qrs[k])
        out[f"QTCL_{lead}"] = float(cr.framingham(qt[k], rr))
        out[f"PRL_{lead}"] = float(pr[k]) if p_ok[k] else np.nan
        out[f"PDURL_{lead}"] = float(pdur[k]) if p_ok[k] else np.nan
    return out


def beat_features(signal: np.ndarray, intervals: np.ndarray, bounds: np.ndarray) -> dict[str, np.ndarray]:
    """
    Prematurity, per-lead QRS deviation from the median beat, and per-lead beat QRS duration.

    Parameters
    ----------
    signal : np.ndarray
        Raw (12, 5000) waveform in mV.
    intervals : np.ndarray
        (beats, 5, 2) components from ``combination066.delineated_components``.
    bounds : np.ndarray
        (beats, 12, 3, 2) per-lead supports of the same beats.

    Returns
    -------
    dict[str, np.ndarray]
        ``pi`` (beats,), ``delta`` (beats, 12) and ``duration`` (beats, 12) in ms; NaN where not measurable.
    """
    beats, samples = len(intervals), signal.shape[1]
    out = {
        "pi": np.full(beats, np.nan),
        "delta": np.full((beats, len(LEADS)), np.nan),
        "duration": np.full((beats, len(LEADS)), np.nan),
    }
    if not beats:
        return out
    qon = intervals[:, 2, 0].astype(np.int64)
    if beats > 1:
        spacing = np.diff(qon).astype(np.float64)
        out["pi"][1:] = spacing / np.median(spacing)
    on, off = bounds[:, :, 1, 0], bounds[:, :, 1, 1]
    out["duration"] = np.where((on >= 0) & (off > on), MS * (off - on), np.nan)
    ok = (qon - cr.PRE >= 0) & (qon + cr.POST <= samples)
    if not ok.any():
        return out
    clean = nm.preprocess(signal, qon.astype(float))
    median = np.median(np.stack([clean[:, q - cr.PRE : q + cr.POST] for q in qon[ok]]), axis=0)
    width = int(max(1, np.median(intervals[:, 2, 1] - intervals[:, 2, 0])))
    template = median[:, cr.PRE : cr.PRE + width]
    level = np.maximum(np.sqrt(np.mean(template**2, axis=1)), DELTA_FLOOR_MV)
    for b, q in enumerate(qon):
        if q + width <= samples:
            residual = clean[:, q : q + width] - template
            out["delta"][b] = np.sqrt(np.mean(residual**2, axis=1)) / level
    return out


def _spread(values: np.ndarray, floor: float) -> float:
    q25, q75 = np.percentile(values, [25, 75])
    return max((q75 - q25) / cr.IQR_SD, floor)


def beat_reference(features: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray | float]:
    """
    Beat thresholds and spreads from the beats of clean normals.

    Parameters
    ----------
    features : list[dict]
        ``beat_features`` of every reference record.

    Returns
    -------
    dict
        ``pi01`` and ``pi_scale`` (floats); ``delta99``, ``delta_scale`` and ``duration_scale`` (12,).
    """
    pi = np.concatenate([f["pi"] for f in features])
    delta = np.concatenate([f["delta"] for f in features])
    duration = np.concatenate([f["duration"] for f in features])
    pi = pi[np.isfinite(pi)]
    return {
        "pi01": float(np.percentile(pi, PI_QUANTILE)),
        "pi_scale": _spread(pi, BEAT_FLOORS["pi"]),
        "delta99": np.array([np.percentile(d[np.isfinite(d)], DELTA_QUANTILE) for d in delta.T]),
        "delta_scale": np.array([_spread(d[np.isfinite(d)], BEAT_FLOORS["delta"]) for d in delta.T]),
        "duration_scale": np.array([_spread(d[np.isfinite(d)], BEAT_FLOORS["duration"]) for d in duration.T]),
    }


def beat_exceedances(features: dict[str, np.ndarray], ref: dict[str, np.ndarray | float]) -> np.ndarray:
    """
    Signed exceedance of each beat-level criterion per beat and lead.

    Parameters
    ----------
    features : dict
        ``beat_features`` of one record.
    ref : dict
        ``beat_reference`` output.

    Returns
    -------
    np.ndarray
        (beats, 2, 12) z in ``BEAT_CRITERIA`` order; NaN where a measurement is missing.
    """
    z_delta = (features["delta"] - ref["delta99"]) / ref["delta_scale"]
    z_pi = (ref["pi01"] - features["pi"]) / ref["pi_scale"]
    z_wide = (features["duration"] - WIDE_QRS) / ref["duration_scale"]
    premature = np.minimum(z_pi[:, None], z_delta)
    wide = np.minimum(z_wide, z_delta)
    premature = np.where(np.isnan(z_pi)[:, None] | np.isnan(z_delta), np.nan, premature)
    wide = np.where(np.isnan(z_wide) | np.isnan(z_delta), np.nan, wide)
    return np.stack([premature, wide], axis=1)


def beat_wave(z: np.ndarray) -> np.ndarray:
    """
    Beat x lead x wave score from beat-level exceedances.

    Parameters
    ----------
    z : np.ndarray
        (beats, 2, 12) ``beat_exceedances`` output.

    Returns
    -------
    np.ndarray
        (beats, 12, 5) maximum hinged evidence over the beat-level criteria painting each component.
    """
    e = cr.evidence(z)
    out = np.zeros((z.shape[0], len(LEADS), len(COMPONENTS)))
    for c, name in enumerate(BEAT_CRITERIA):
        for wave in BEAT_WAVES[name]:
            w = COMPONENTS.index(wave)
            out[:, :, w] = np.maximum(out[:, :, w], e[:, c, :])
    return out


def paint_beats(scores: np.ndarray, beat_scores: np.ndarray | None, intervals: np.ndarray) -> np.ndarray:
    """
    Paint max(median-beat score, beat score) over every beat's components; zero elsewhere.

    Parameters
    ----------
    scores : np.ndarray
        (12, 5) median-beat lead x wave scores.
    beat_scores : np.ndarray or None
        (beats, 12, 5) beat-level scores aligned with ``intervals``, or None for none.
    intervals : np.ndarray
        (beats, 5, 2) components.

    Returns
    -------
    np.ndarray
        (12, samples) float64.
    """
    if beat_scores is None:
        return cr.paint(scores, intervals)
    field = np.zeros((len(LEADS), cr.SAMPLES))
    for b, beat in enumerate(intervals):
        for w, (start, end) in enumerate(beat):
            if end > start:
                value = np.maximum(scores[:, w], beat_scores[b, :, w])
                field[:, start:end] = np.maximum(field[:, start:end], value[:, None])
    return field


def lexicographic(field: np.ndarray, intervals: np.ndarray, amplitude: np.ndarray) -> np.ndarray:
    """``perlead_qrs``: the 069 lexicographic QRS map of a painted field."""
    return cr.lexicographic(field, intervals, amplitude)


def components_of(intervals: np.ndarray) -> np.ndarray:
    """Return ``intervals`` unchanged, or an empty (0, 5, 2) array for None."""
    return np.zeros((0, len(COMPONENTS), 2), np.int64) if intervals is None else intervals


def same_components(intervals: np.ndarray, saved: np.ndarray | None) -> bool:
    """Whether recomputed beat components equal the saved 069 components (None when 069 found no beat)."""
    return bool(np.array_equal(intervals, components_of(saved)))
