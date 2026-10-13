"""
Measurement-based criteria localizer for Experiment 069.

Per-lead measurements (own median-beat measurements from the frozen 062/066 delineation, GE 12SL and Philips
Uni-G from PTB-XL+) are combined into a consensus, turned into published ECG criteria, and each criterion's
exceedance, in units of the same lead's spread in clean normals, is painted over its lead and wave of every
beat. See ``docs/experiment-069-criteria-localizer.md``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ecg_experiment import combination066 as cb
from ecg_experiment import combiner068 as cv
from ecg_experiment import normative_map062 as nm
from ecg_experiment.waveforms import LEADS

FS = 500
SAMPLES = 5000
MS = 1000.0 / FS
PRE, POST = 150, 350
Q_THRESHOLD = 0.02
TIE = cb.TIE
IQR_SD = 1.349
COMPONENTS = cb.COMPONENTS
LIMB = ("I", "II", "III", "aVR", "aVL", "aVF")
PRECORDIAL = ("V1", "V2", "V3", "V4", "V5", "V6")
NOT_AVR = tuple(lead for lead in LEADS if lead != "aVR")
FRONTAL = ("I", "II", "III", "aVL", "aVF")

PER_LEAD = ("Q_DUR", "Q_DEPTH", "R", "S", "RPT", "PP", "TERM", "NET", "ST_J", "ST_60", "ST_80", "T", "P")
GLOBAL = ("QRSD", "PR", "PDUR", "QT", "RR", "QTC", "AXIS", "PTF")
AGREEMENT = ("Q_DEPTH", "R", "S", "ST_J", "T", "P")
FLOORS = {
    "ST_J": 0.025,
    "R": 0.05,
    "S": 0.05,
    "T": 0.05,
    "Q_DEPTH": 0.05,
    "RS": 0.05,
    "SR": 0.05,
    "TERM": 0.05,
    "P": 0.02,
    "Q_DUR": 10.0,
    "RPT": 10.0,
    "QRSD": 10.0,
    "PDUR": 10.0,
    "PR": 15.0,
    "QTC": 15.0,
    "AXIS": 15.0,
    "SOKOLOW": 0.2,
    "CORNELL": 0.2,
    "LIMB_PP": 0.1,
    "PREC_PP": 0.1,
    "PTF": 2.0,
}

TERRITORY = {
    **dict.fromkeys(("V1", "V2", "V3"), ("ASMI", "AMI", "ANEUR", "INJAS", "ISCAS")),
    "V4": ("AMI", "ALMI", "ANEUR", "INJAS", "INJAL", "ISCAS", "ISCAL"),
    **dict.fromkeys(("V5", "V6"), ("ALMI", "LMI", "ILMI", "INJAL", "ISCAL", "ISCLA", "ISCIL")),
    **dict.fromkeys(("I", "aVL"), ("ALMI", "LMI", "INJAL", "ISCAL", "ISCLA")),
    **dict.fromkeys(("II", "III", "aVF"), ("IMI", "ILMI", "ISCIN", "ISCIL")),
    "aVR": (),
}
MI_HEADS = frozenset({"ASMI", "AMI", "ALMI", "LMI", "ILMI", "IMI", "ANEUR"})
INJURY_HEADS = frozenset({"INJAS", "INJAL"})
ISCHEMIA_HEADS = frozenset({"ISCAS", "ISCAL", "ISCLA", "ISCIN", "ISCIL"})
TERRITORY_CLASSES = {
    "mi_injury": (MI_HEADS | INJURY_HEADS, ()),
    "ischemia_st": (ISCHEMIA_HEADS, ("ISC_", "STD_", "NST_", "DIG")),
    "ischemia_t": (ISCHEMIA_HEADS, ("ISC_", "INVT", "NT_", "NDT")),
    "mi_q": (MI_HEADS, ("QWAVE",)),
}


@dataclass(frozen=True)
class Criterion:
    """
    One published criterion.

    Attributes
    ----------
    name : str
        Criterion name.
    leads : tuple[str, ...]
        Leads the criterion names.
    waves : tuple[str, ...]
        066 components it paints (P, PQ, QRS, ST, T).
    conditions : tuple[tuple[str, int, str | float], ...]
        (measure, sign, threshold): ``sign`` +1 for ">=", -1 for "<="; a string threshold is resolved by
        ``threshold_values`` from age and sex. Per-lead measures are named without the lead suffix.
    mode : str
        ``"all"`` (minimum of the exceedances) or ``"any"`` (maximum).
    supports : tuple[str, ...]
        065 heads, or a ``TERRITORY_CLASSES`` key resolved per lead.
    """

    name: str
    leads: tuple[str, ...]
    waves: tuple[str, ...]
    conditions: tuple[tuple[str, int, str | float], ...]
    mode: str
    supports: tuple[str, ...]


CRITERIA = (
    Criterion("ste_v2v3", ("V2", "V3"), ("ST",), (("ST_J", 1, "ste_v2v3"),), "all", ("mi_injury",)),
    Criterion(
        "ste",
        ("I", "II", "III", "aVL", "aVF", "V1", "V4", "V5", "V6"),
        ("ST",),
        (("ST_J", 1, 0.1),),
        "all",
        ("mi_injury",),
    ),
    Criterion("std", NOT_AVR, ("ST",), (("ST_J", -1, -0.05),), "all", ("ischemia_st",)),
    Criterion(
        "tinv",
        ("I", "II", "aVL", "aVF", "V2", "V3", "V4", "V5", "V6"),
        ("T",),
        (("T", -1, -0.1), ("RS", 1, 0.0)),
        "all",
        ("ischemia_t",),
    ),
    Criterion("q_v2v3", ("V2", "V3"), ("QRS",), (("Q_DUR", 1, 20.0),), "all", ("mi_q",)),
    Criterion(
        "q_path",
        ("I", "II", "aVL", "aVF", "V4", "V5", "V6"),
        ("QRS",),
        (("Q_DUR", 1, 30.0), ("Q_DEPTH", 1, 0.1)),
        "all",
        ("mi_q",),
    ),
    Criterion("rvh_v1", ("V1",), ("QRS",), (("R", 1, 0.7), ("RS", 1, 0.0)), "any", ("RVH", "CRBBB", "IRBBB")),
    Criterion("sokolow", ("V1", "V5", "V6"), ("QRS",), (("SOKOLOW", 1, 3.5),), "all", ("LVH",)),
    Criterion("r_avl", ("aVL",), ("QRS",), (("R", 1, 1.1),), "all", ("LVH",)),
    Criterion("cornell", ("aVL", "V3"), ("QRS",), (("CORNELL", 1, "cornell"),), "all", ("LVH",)),
    Criterion(
        "rbbb_v1",
        ("V1", "V2"),
        ("QRS",),
        (("QRSD", 1, 120.0), ("TERM", 1, 0.0)),
        "all",
        ("CRBBB", "IRBBB", "IVCD"),
    ),
    Criterion(
        "rbbb_lat",
        ("I", "V6"),
        ("QRS",),
        (("QRSD", 1, 120.0), ("TERM", -1, 0.0)),
        "all",
        ("CRBBB", "IRBBB", "IVCD"),
    ),
    Criterion(
        "lbbb_lat",
        ("I", "aVL", "V5", "V6"),
        ("QRS",),
        (("QRSD", 1, 120.0), ("RPT", 1, 60.0)),
        "all",
        ("CLBBB", "ILBBB", "IVCD"),
    ),
    Criterion(
        "lbbb_v1", ("V1",), ("QRS",), (("QRSD", 1, 120.0), ("SR", 1, 0.0)), "all", ("CLBBB", "ILBBB", "IVCD")
    ),
    Criterion(
        "ivcd",
        LEADS,
        ("QRS",),
        (("QRSD", 1, 110.0),),
        "all",
        ("IVCD", "CLBBB", "CRBBB", "ILBBB", "IRBBB", "WPW"),
    ),
    Criterion("lafb", FRONTAL, ("QRS",), (("AXIS", -1, -45.0),), "all", ("LAFB",)),
    Criterion("lpfb", FRONTAL, ("QRS",), (("AXIS", 1, 90.0),), "all", ("LPFB", "RVH")),
    Criterion("pr_long", LEADS, ("P", "PQ"), (("PR", 1, 200.0),), "all", ("1AVB", "LPR")),
    Criterion("wpw", LEADS, ("P", "PQ", "QRS"), (("PR", -1, 120.0), ("QRSD", 1, 120.0)), "all", ("WPW",)),
    Criterion("qtc_long", LEADS, ("T",), (("QTC", 1, "qtc_long"),), "all", ("LNGQT",)),
    Criterion("qtc_short", LEADS, ("T",), (("QTC", -1, 390.0),), "all", ()),
    Criterion("low_limb", LIMB, ("QRS",), (("LIMB_PP", -1, 0.5),), "all", ("LVOLT",)),
    Criterion("low_prec", PRECORDIAL, ("QRS",), (("PREC_PP", -1, 1.0),), "all", ("LVOLT",)),
    Criterion("p_tall", ("II", "III", "aVF"), ("P",), (("P", 1, 0.25),), "all", ("RAO/RAE",)),
    Criterion("p_tall_v1", ("V1",), ("P",), (("P", 1, 0.15),), "all", ("RAO/RAE",)),
    Criterion("p_wide", ("II",), ("P",), (("PDUR", 1, 120.0),), "all", ("LAO/LAE",)),
    Criterion("ptf_v1", ("V1",), ("P",), (("PTF", 1, 4.0),), "all", ("LAO/LAE",)),
)
PER_LEAD_DERIVED = ("RS", "SR")
RECORD_DERIVED = ("SOKOLOW", "CORNELL", "LIMB_PP", "PREC_PP")


def column(measure: str, lead: str) -> str:
    """Column of a measure in a lead: per-lead measures carry the lead suffix, record measures do not."""
    return f"{measure}_{lead}" if measure in PER_LEAD or measure in PER_LEAD_DERIVED else measure


def measure_of(name: str) -> str:
    """Measure of a column name (the lead suffix removed)."""
    for lead in LEADS:
        if name.endswith(f"_{lead}"):
            return name[: -len(lead) - 1]
    return name


def _empty_own() -> dict[str, float]:
    names = [f"{m}_{lead}" for m in PER_LEAD for lead in LEADS] + list(GLOBAL)
    return dict.fromkeys(names, np.nan)


def qrs_measures(q: np.ndarray) -> dict[str, float]:
    """
    QRS measurements of one lead's median beat.

    Parameters
    ----------
    q : np.ndarray
        Median beat from the joint QRS onset to the median QRS offset, in mV (baseline-corrected).

    Returns
    -------
    dict[str, float]
        ``Q_DUR`` and ``RPT`` in ms, ``Q_DEPTH``, ``R``, ``S``, ``PP``, ``TERM`` in mV and ``NET`` in mV x ms.
    """
    big = np.flatnonzero(np.abs(q) > Q_THRESHOLD)
    q_dur = q_depth = 0.0
    if len(big) and q[big[0]] < 0:
        back = np.flatnonzero(q[big[0] :] >= 0)
        end = big[0] + back[0] if len(back) else len(q)
        q_dur, q_depth = MS * end, float(-q[:end].min())
    peak = int(np.argmax(q))
    r = max(0.0, float(q[peak]))
    s = max(0.0, float(-q[peak:].min())) if r > Q_THRESHOLD else 0.0
    return {
        "Q_DUR": q_dur,
        "Q_DEPTH": q_depth,
        "R": r,
        "S": s,
        "RPT": MS * peak,
        "PP": float(q.max() - q.min()),
        "TERM": float(q[-round(40 / MS) :].mean()),
        "NET": float(q.sum() * MS),
    }


def signed_extreme(segment: np.ndarray) -> float:
    """Value of largest absolute size in a segment (NaN if empty)."""
    return float(segment[np.argmax(np.abs(segment))]) if len(segment) else np.nan


def terminal_force(segment: np.ndarray) -> float:
    """
    P terminal force: depth times duration (mV x ms) of the negative part after the P peak.

    Parameters
    ----------
    segment : np.ndarray
        Median-beat P wave of V1 in mV.

    Returns
    -------
    float
        Non-negative terminal force (NaN if the segment is empty).
    """
    if not len(segment):
        return np.nan
    tail = segment[int(np.argmax(segment)) :]
    negative = tail < 0
    return float(max(0.0, -tail.min()) * MS * negative.sum()) if negative.any() else 0.0


def beat_geometry(signal: np.ndarray) -> dict[str, np.ndarray] | None:
    """
    Components of every beat and the P-found flag from the frozen delineation.

    Parameters
    ----------
    signal : np.ndarray
        Raw (12, 5000) waveform in mV.

    Returns
    -------
    dict or None
        ``intervals`` (beats, 5, 2) from ``combination066.delineated_components``, ``qon`` (beats,) and
        ``p_found`` (beats,) boolean; None if no beat was delineated.
    """
    intervals = cb.delineated_components(signal)
    if not len(intervals):
        return None
    fid = nm.delineate(signal)
    keep = (fid["qon"] >= 0) & (fid["qoff"] > fid["qon"])
    pon, qon = fid["pon"][keep], fid["qon"][keep]
    found = np.isfinite(pon) & (pon < qon)
    if len(found) != len(intervals):
        raise ValueError("Delineation beat counts differ")
    return {"intervals": intervals, "qon": intervals[:, 2, 0].astype(np.int64), "p_found": found}


def _relative(intervals: np.ndarray, qon: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return np.round(np.median(intervals[rows] - qon[rows, None, None], axis=0)).astype(np.int64)


def own_measurements(signal: np.ndarray, geometry: dict[str, np.ndarray] | None) -> dict[str, float]:
    """
    Own per-lead and global measurements from the median beat.

    Parameters
    ----------
    signal : np.ndarray
        Raw (12, 5000) waveform in mV.
    geometry : dict or None
        ``beat_geometry`` output.

    Returns
    -------
    dict[str, float]
        ``{measure}_{lead}`` for ``PER_LEAD`` and the ``GLOBAL`` measures; NaN where not measurable.
    """
    out = _empty_own()
    if geometry is None:
        return out
    intervals, qon, found = geometry["intervals"], geometry["qon"], geometry["p_found"]
    ok = (qon - PRE >= 0) & (qon + POST <= signal.shape[1])
    if not ok.any():
        return out
    clean = nm.preprocess(signal, qon.astype(float))
    median = np.median(np.stack([clean[:, q - PRE : q + POST] for q in qon[ok]]), axis=0)
    rel = _relative(intervals, qon, np.flatnonzero(ok))
    qoff = int(np.clip(rel[2, 1], 2, POST - 45))
    ton, toff = int(np.clip(rel[4, 0], qoff, POST)), int(np.clip(rel[4, 1], qoff + 1, POST))
    p_rows = np.flatnonzero(ok & found)
    has_p = found.sum() >= 0.5 * len(found) and len(p_rows) > 0
    p_rel = _relative(intervals, qon, p_rows)[0] if has_p else None
    for k, lead in enumerate(LEADS):
        for name, value in qrs_measures(median[k, PRE : PRE + qoff]).items():
            out[f"{name}_{lead}"] = value
        after = PRE + qoff
        out[f"ST_J_{lead}"] = float(median[k, after : after + 5].mean())
        out[f"ST_60_{lead}"] = float(median[k, after + 28 : after + 33].mean())
        out[f"ST_80_{lead}"] = float(median[k, after + 38 : after + 43].mean())
        out[f"T_{lead}"] = signed_extreme(median[k, PRE + ton : PRE + toff])
        if p_rel is not None:
            out[f"P_{lead}"] = signed_extreme(median[k, PRE + max(p_rel[0], -PRE) : PRE + min(p_rel[1], 0)])
    v1 = LEADS.index("V1")
    if p_rel is not None:
        out["PTF"] = terminal_force(median[v1, PRE + max(p_rel[0], -PRE) : PRE + min(p_rel[1], 0)])
        out["PR"] = MS * float(np.median(qon[found] - intervals[found, 0, 0]))
        out["PDUR"] = MS * float(np.median(intervals[found, 0, 1] - intervals[found, 0, 0]))
    out["QRSD"] = MS * qoff
    out["QT"] = MS * toff
    out["RR"] = MS * float(np.median(np.diff(qon))) if len(qon) > 1 else np.nan
    out["QTC"] = framingham(out["QT"], out["RR"])
    out["AXIS"] = float(np.degrees(np.arctan2(out["NET_aVF"], out["NET_I"])))
    return out


def framingham(qt: float | np.ndarray, rr: float | np.ndarray) -> float | np.ndarray:
    """Framingham (Sagie) corrected QT in ms from QT and RR in ms."""
    return qt + 154.0 * (1.0 - np.asarray(rr) / 1000.0)


def wrap_degrees(angle: pd.Series) -> pd.Series:
    """Wrap angles in degrees to (-180, 180]."""
    return 180.0 - np.mod(180.0 - angle, 360.0)


def _abs_max(frame: pd.DataFrame, names: list[str]) -> pd.Series:
    return frame[names].abs().max(axis=1, skipna=True).where(frame[names].notna().any(axis=1))


def source_12sl(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonized 12SL measurements.

    Parameters
    ----------
    raw : pd.DataFrame
        ``12sl_features.csv`` rows indexed by ``ecg_id``.

    Returns
    -------
    pd.DataFrame
        Columns named as the own measurements (subset).
    """
    out: dict[str, pd.Series | np.ndarray] = {}
    for lead in LEADS:
        out[f"Q_DUR_{lead}"] = raw[f"Q_Dur_{lead}"]
        out[f"Q_DEPTH_{lead}"] = raw[f"Q_Amp_{lead}"].abs()
        out[f"R_{lead}"] = raw[[f"R_Amp_{lead}", f"Rp_Amp_{lead}"]].max(axis=1)
        out[f"S_{lead}"] = _abs_max(raw, [f"S_Amp_{lead}", f"Sp_Amp_{lead}"])
        out[f"RPT_{lead}"] = raw[f"R_PeakTime_{lead}"]
        out[f"PP_{lead}"] = raw[f"QRS_AmpPP_{lead}"]
        out[f"ST_J_{lead}"] = raw[f"ST_Amp_{lead}"]
        out[f"T_{lead}"] = raw[f"T_Amp_{lead}"]
        out[f"P_{lead}"] = raw[f"P_Amp_{lead}"]
    out["QRSD"] = raw["QRS_Dur_Global"]
    out["PR"] = raw["PR_Int_Global"]
    out["PDUR"] = raw["P_Off_Global"] - raw["P_On_Global"]
    out["QT"] = raw["QT_Int_Global"]
    out["RR"] = raw["RR_Mean_Global"]
    out["QTC"] = framingham(out["QT"], out["RR"])
    out["AXIS"] = wrap_degrees(raw["R_AxisFrontal_Global"])
    return pd.DataFrame(out, index=raw.index)


def source_unig(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonized Uni-G measurements.

    Parameters
    ----------
    raw : pd.DataFrame
        ``unig_features.csv`` rows indexed by ``ecg_id``.

    Returns
    -------
    pd.DataFrame
        Columns named as the own measurements (subset).
    """
    out: dict[str, pd.Series | np.ndarray] = {}
    p_found = raw["P_Found_Global"] == 1
    for lead in LEADS:
        out[f"Q_DUR_{lead}"] = raw[f"Q_Dur_{lead}"]
        out[f"Q_DEPTH_{lead}"] = raw[f"Q_Amp_{lead}"].abs()
        out[f"R_{lead}"] = raw[[f"R_Amp_{lead}", f"Rp_Amp_{lead}"]].max(axis=1)
        out[f"S_{lead}"] = _abs_max(raw, [f"S_Amp_{lead}", f"Sp_Amp_{lead}"])
        out[f"RPT_{lead}"] = raw[f"QRS_IntDefl_{lead}"]
        out[f"PP_{lead}"] = raw[f"QRS_AmpPP_{lead}"]
        out[f"ST_J_{lead}"] = raw[f"ST_Amp_{lead}"]
        out[f"ST_80_{lead}"] = raw[f"ST_Amp80ms_{lead}"]
        t = np.where(raw[f"T_Morph_{lead}"] > 0, raw[f"T+_Amp_{lead}"], raw[f"T-_Amp_{lead}"])
        out[f"T_{lead}"] = np.where(raw[f"T_Morph_{lead}"].isna(), np.nan, t)
        p = np.where(raw[f"P_Morph_{lead}"] > 0, raw[f"P+_Amp_{lead}"], raw[f"P-_Amp_{lead}"])
        out[f"P_{lead}"] = np.where(raw[f"P_Morph_{lead}"].isna(), np.nan, p)
    out["QRSD"] = raw["QRS_Dur_Global"]
    out["PR"] = raw["PR_Int_Global"].where(p_found)
    out["PDUR"] = raw["P_Dur_Global"].where(p_found)
    out["QT"] = raw["QT_Int_Global"]
    out["RR"] = raw["RR_Mean_Global"]
    out["QTC"] = framingham(out["QT"], out["RR"])
    out["AXIS"] = wrap_degrees(raw["QRS_AxisFront_Global"]).where(raw["QRS_AxisIndet_Global"] != 1)
    out["PTF"] = raw["P_Term_V1"].abs().where(p_found)
    return pd.DataFrame(out, index=raw.index)


def consensus(tables: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Median over sources of every measurement (mean of two when one source is missing).

    Parameters
    ----------
    tables : list[pd.DataFrame]
        Source tables with the same index; a column missing from a source counts as missing.

    Returns
    -------
    pd.DataFrame
        Consensus table over the union of columns.
    """
    columns = list(dict.fromkeys(name for table in tables for name in table.columns))
    stacked = np.stack([table.reindex(columns=columns).to_numpy(np.float64) for table in tables])
    with warnings.catch_warnings(action="ignore"):
        values = np.nanmedian(stacked, axis=0)
    return pd.DataFrame(values, index=tables[0].index, columns=columns)


def add_derived(table: pd.DataFrame) -> pd.DataFrame:
    """
    Add the derived measures: R - S and S - R per lead, Sokolow-Lyon and Cornell sums, largest peak-to-peak.

    Parameters
    ----------
    table : pd.DataFrame
        Measurement table (own or consensus).

    Returns
    -------
    pd.DataFrame
        Copy with the derived columns.
    """
    out = table.copy()
    for lead in LEADS:
        out[f"RS_{lead}"] = out[f"R_{lead}"] - out[f"S_{lead}"]
        out[f"SR_{lead}"] = -out[f"RS_{lead}"]
    out["SOKOLOW"] = out["S_V1"] + out[["R_V5", "R_V6"]].max(axis=1, skipna=False)
    out["CORNELL"] = out["R_aVL"] + out["S_V3"]
    out["LIMB_PP"] = out[[f"PP_{lead}" for lead in LIMB]].max(axis=1, skipna=False)
    out["PREC_PP"] = out[[f"PP_{lead}" for lead in PRECORDIAL]].max(axis=1, skipna=False)
    return out


def used_columns() -> list[str]:
    """Every measurement column that some criterion reads."""
    names = [column(m, lead) for c in CRITERIA for lead in c.leads for m, _, _ in c.conditions]
    return list(dict.fromkeys(names))


def normal_reference(normals: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Median and floored robust spread of each column over clean normals.

    Parameters
    ----------
    normals : pd.DataFrame
        Measurement table of the reference records (with derived columns).
    columns : list[str]
        Columns to describe.

    Returns
    -------
    pd.DataFrame
        Indexed by column: ``median``, ``iqr_sd`` (IQR / 1.349), ``floor``, ``scale`` (the larger), ``count``.
    """
    rows = {}
    for name in columns:
        values = normals[name].dropna().to_numpy(np.float64)
        floor = FLOORS.get(measure_of(name), 1e-6)
        q25, q50, q75 = np.percentile(values, [25, 50, 75]) if len(values) else (np.nan,) * 3
        spread = (q75 - q25) / IQR_SD
        rows[name] = {
            "median": q50,
            "iqr_sd": spread,
            "floor": floor,
            "scale": max(spread, floor),
            "count": len(values),
        }
    return pd.DataFrame(rows).T


def threshold_values(name: str | float, age: np.ndarray, sex: np.ndarray) -> np.ndarray:
    """
    Threshold of a condition per record.

    Parameters
    ----------
    name : str or float
        A number, or ``"ste_v2v3"``, ``"cornell"`` or ``"qtc_long"``.
    age : np.ndarray
        Age in years (NaN above 89 counts as 40 or older).
    sex : np.ndarray
        PTB-XL sex, 0 male and 1 female.

    Returns
    -------
    np.ndarray
        (records,) thresholds.
    """
    female = np.asarray(sex) == 1
    if not isinstance(name, str):
        return np.full(len(female), float(name))
    if name == "ste_v2v3":
        young = np.nan_to_num(np.asarray(age, np.float64), nan=99.0) < 40
        return np.where(female, 0.15, np.where(young, 0.25, 0.2))
    if name == "cornell":
        return np.where(female, 2.0, 2.8)
    if name == "qtc_long":
        return np.where(female, 460.0, 450.0)
    raise ValueError(f"Unknown threshold {name}")


def exceedances(table: pd.DataFrame, scales: pd.Series, age: np.ndarray, sex: np.ndarray) -> np.ndarray:
    """
    Signed exceedance of every criterion in every lead it names.

    Parameters
    ----------
    table : pd.DataFrame
        Measurement table with derived columns, one row per record.
    scales : pd.Series
        Normal spread per column (``normal_reference(...)["scale"]``).
    age, sex : np.ndarray
        Per record, in table order.

    Returns
    -------
    np.ndarray
        (records, criteria, 12) float64: z, NaN for leads the criterion does not name or missing measurements.
    """
    out = np.full((len(table), len(CRITERIA), len(LEADS)), np.nan)
    for c, criterion in enumerate(CRITERIA):
        for lead in criterion.leads:
            parts = []
            for measure, sign, threshold in criterion.conditions:
                name = column(measure, lead)
                values = table[name].to_numpy(np.float64)
                parts.append(sign * (values - threshold_values(threshold, age, sex)) / float(scales[name]))
            stacked = np.stack(parts)
            combined = stacked.min(axis=0) if criterion.mode == "all" else stacked.max(axis=0)
            out[:, c, LEADS.index(lead)] = np.where(np.isnan(stacked).any(axis=0), np.nan, combined)
    return out


def evidence(z: np.ndarray) -> np.ndarray:
    """Hinged evidence max(0, z), zero where z is missing."""
    return np.clip(np.nan_to_num(z, nan=0.0), 0.0, None)


def criterion_components() -> np.ndarray:
    """Boolean (criteria, 5) of the 066 components each criterion paints."""
    return np.array([[name in c.waves for name in COMPONENTS] for c in CRITERIA])


def lead_wave(scores: np.ndarray) -> np.ndarray:
    """
    Lead x wave score: maximum over criteria naming the lead and wave.

    Parameters
    ----------
    scores : np.ndarray
        (records, criteria, 12) non-negative evidence (or weighted evidence).

    Returns
    -------
    np.ndarray
        (records, 12, 5) in ``COMPONENTS`` order.
    """
    painted = criterion_components()
    out = np.zeros((scores.shape[0], len(LEADS), len(COMPONENTS)))
    for w in range(len(COMPONENTS)):
        rows = np.flatnonzero(painted[:, w])
        out[:, :, w] = scores[:, rows, :].max(axis=1)
    return out


def support_heads(criterion: Criterion, lead: str) -> tuple[str, ...]:
    """065 heads a criterion supports in one lead (territory classes resolved by lead)."""
    heads: list[str] = []
    for item in criterion.supports:
        if item in TERRITORY_CLASSES:
            territory, extra = TERRITORY_CLASSES[item]
            heads += [h for h in TERRITORY[lead] if h in territory] + list(extra)
        else:
            heads.append(item)
    return tuple(dict.fromkeys(heads))


def mil_weights(prob: np.ndarray, heads: list[str]) -> np.ndarray:
    """
    Largest 065 bag probability among the heads each criterion supports in each lead.

    Parameters
    ----------
    prob : np.ndarray
        (records, heads) bag probabilities.
    heads : list[str]
        065 head names in column order (``NORM`` included).

    Returns
    -------
    np.ndarray
        (records, criteria, 12); 1 - p(NORM) for a criterion without supported heads, 0 for unnamed leads.
    """
    index = {h: k for k, h in enumerate(heads)}
    out = np.zeros((prob.shape[0], len(CRITERIA), len(LEADS)))
    abnormal = 1.0 - prob[:, index["NORM"]]
    for c, criterion in enumerate(CRITERIA):
        for lead in criterion.leads:
            names = support_heads(criterion, lead)
            weight = prob[:, [index[h] for h in names]].max(axis=1) if names else abnormal
            out[:, c, LEADS.index(lead)] = weight
    return out


def disagreement(sources: list[pd.DataFrame], scales: pd.Series) -> np.ndarray:
    """
    Per-lead disagreement: median over ``AGREEMENT`` measures of the source range over the normal spread.

    Parameters
    ----------
    sources : list[pd.DataFrame]
        Own, 12SL and Uni-G tables with the same index.
    scales : pd.Series
        Consensus normal spread per column.

    Returns
    -------
    np.ndarray
        (records, 12); NaN where no measure has two sources.
    """
    out = np.full((len(sources[0]), len(LEADS)), np.nan)
    for k, lead in enumerate(LEADS):
        ranges = []
        for measure in AGREEMENT:
            name = f"{measure}_{lead}"
            values = np.stack([s[name].to_numpy(np.float64) for s in sources if name in s])
            enough = np.isfinite(values).sum(axis=0) >= 2
            with warnings.catch_warnings(action="ignore"):
                spread = (np.nanmax(values, axis=0) - np.nanmin(values, axis=0)) / float(scales[name])
            ranges.append(np.where(enough, spread, np.nan))
        with warnings.catch_warnings(action="ignore"):
            out[:, k] = np.nanmedian(np.stack(ranges), axis=0)
    return out


def reliability(d: np.ndarray, q95: np.ndarray) -> np.ndarray:
    """
    Lead reliability min(1, q95 / D), 1 where D is 0 or missing.

    Parameters
    ----------
    d : np.ndarray
        (records, 12) disagreement.
    q95 : np.ndarray
        (12,) 95th percentile of the disagreement over clean normals.

    Returns
    -------
    np.ndarray
        (records, 12) in (0, 1].
    """
    valid = np.isfinite(d) & (d > 0)
    ratio = np.divide(q95[None, :], d, out=np.ones_like(d, dtype=np.float64), where=valid)
    return np.where(valid, np.minimum(1.0, ratio), 1.0)


def paint(scores: np.ndarray, intervals: np.ndarray, samples: int = SAMPLES) -> np.ndarray:
    """
    Paint a (12, 5) lead x wave score over every beat's components, maximum on overlap, zero elsewhere.

    Parameters
    ----------
    scores : np.ndarray
        (12, 5) scores in ``COMPONENTS`` order.
    intervals : np.ndarray
        (beats, 5, 2) components from ``combination066.delineated_components``.
    samples : int
        Record length.

    Returns
    -------
    np.ndarray
        (12, samples) float64.
    """
    field = np.zeros((len(LEADS), samples))
    for beat in intervals:
        for w, (start, end) in enumerate(beat):
            if end > start:
                field[:, start:end] = np.maximum(field[:, start:end], scores[:, w : w + 1])
    return field


def finish(field: np.ndarray, amplitude: np.ndarray) -> np.ndarray:
    """Add the 061 amplitude tie-break and convert to float32."""
    return (field + TIE * amplitude).astype(np.float32)


def lexicographic(field: np.ndarray, intervals: np.ndarray, amplitude: np.ndarray) -> np.ndarray:
    """
    ``criteria_qrs``: 2 on own QRS samples plus E / (1 + E), plus the tie-break.

    Parameters
    ----------
    field : np.ndarray
        (12, samples) painted criteria evidence.
    intervals : np.ndarray
        Beat components.
    amplitude : np.ndarray
        061 amplitude map.

    Returns
    -------
    np.ndarray
        (12, samples) float32.
    """
    qrs = cb.component_masks(intervals, field.shape[1])[COMPONENTS.index("QRS")]
    return finish(2.0 * qrs[None, :] + field / (1.0 + field), amplitude)


FEATURES = (*cv.FEATURES, "criteria", "criteria_x_mil")
EXTRA_MAPS = ("criteria", "criteria_x_mil")


def design(raw: np.ndarray, q: np.ndarray, units: np.ndarray, extra: np.ndarray) -> np.ndarray:
    """
    068 design matrix plus the within-record z-scored log1p unit means of the criteria maps.

    Parameters
    ----------
    raw : np.ndarray
        (units, 9) 068 unit means of ``combiner068.RAW_MAPS``.
    q : np.ndarray
        (12,) lead-quality probabilities.
    units : np.ndarray
        ``combination066.unit_table`` output.
    extra : np.ndarray
        (units, 2) unit means of ``criteria`` and ``criteria_x_mil``.

    Returns
    -------
    np.ndarray
        (units, 34) in ``FEATURES`` order.
    """
    base = cv.unit_design(raw, q, units)
    values = np.log1p(np.clip(np.asarray(extra, np.float64), 0, None))
    spread = values.std(axis=0)
    z = np.divide(values - values.mean(axis=0), spread, out=np.zeros_like(values), where=spread > 0)
    return np.column_stack([base, z])


def feature_columns(names: tuple[str, ...]) -> np.ndarray:
    """Positions of the named features in ``FEATURES``."""
    return np.array([FEATURES.index(name) for name in names])


def criteria_features(a: float) -> tuple[str, ...]:
    """Return the ``units_v2_criteria`` features: 068 ``units_v2`` features plus both criteria maps."""
    return (*cv.feature_set("units_v2", a), *EXTRA_MAPS)


def met_any_lead(z: np.ndarray) -> np.ndarray:
    """Flag (records, criteria) whether the criterion is met in at least one lead."""
    return (np.nan_to_num(z, nan=-np.inf) >= 0).any(axis=2)


def evidence_leads(scores: np.ndarray) -> np.ndarray:
    """Flag (records, 12) leads with positive lead x wave evidence in some wave."""
    return scores.max(axis=2) > 0
