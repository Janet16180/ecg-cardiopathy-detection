"""
Label-quality audit of MLC-03 Part B on the train and validation records of the MLC-01 v4 dataset.

``docs/mlc03-data-quality-protocol-2026-11-11.md`` (Part B) is the contract. Everything here reads records of
roles ``train`` and ``validation`` only: ``open_manifest_lines`` reads the role column first and the manifest
rows of every other role are never parsed (``load_open_records``); the feature, measurement and PTB-XL+
readers take the keys of those records only. No label is changed here. Rules that would set a label undefined
or exclude a record are counted (``proposed_rule_masks``) for a later protocol amendment.

Measurements come from two methods. Method A is MLC-02's frozen extractor, read from the stored 130-column
array. Method B is ``mlc03_measurements.measure_record``. ``consensus`` keeps method B's value where both
methods agree within a stated tolerance; the error of each method is estimated against PTB-XL+ (the GE 12SL
and Uni-G programs, where they agree with each other), LUDB's manual delineation and synthetic records.

Label checks (``CHECKS``) state, per condition, the measurement it is defined by, the bound, its reference,
and which records fall outside it. Source coding compares the prevalence of a label at the same measurement
(``matched_prevalence``, ``inconsistent_bins``). The confident-learning count (``confident_learning``) uses
out-of-fold probabilities of P0, the MLC-02 reference model.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from ecg_experiment import ROOT, mlc_labels
from ecg_experiment.measurement_extractor042 import FEATURE_NAMES
from ecg_experiment.mlc02_data import CONDITIONS, MANIFEST
from ecg_experiment.mlc02_feature_data import LOCAL, aligned_rows
from ecg_experiment.mlc03_measurements import LEADS, UPRIGHT_T
from ecg_experiment.referral_budget import wilson_interval

OPEN_ROLES = ("train", "validation")
CLOSED_ROLES = ("test_same_source", "test_external", "sph_calibration")
OUTPUT = ROOT / "outputs/mlc03_data_quality_20261111/labels"
REVIEW = ROOT / "outputs/mlc03_data_quality_20261111/review/labels"
SCRATCH = Path("/var/tmp/janetrivera/mlc03")
PTBXL_PLUS = ROOT / "data/raw/ptb-xl-plus/1.0.1/features"
LUDB = ROOT / "data/raw/ludb/1.0.1"
SEED = 20261111
SOURCES = ("ptbxl", "ningbo", "chapman_shaoxing", "georgia", "cpsc_2018", "cpsc_2018_extra")
EXTRA_LABELS = (
    "complete_lbbb",
    "lpfb",
    "right_axis_deviation",
    "preexcitation",
    "paced_rhythm",
    "supraventricular_tachycardia",
    "atrial_fibrillation",
    "atrial_flutter",
    "premature_atrial_complex",
)
STRUCTURE = (
    "record_key",
    "source",
    "family",
    "role",
    "origin",
    "group_id",
    "included",
    "sex",
    "age",
    "duration_s",
    "n_flat_leads",
    "n_missing_leads",
    "raw_codes",
    "heart_axis",
    "raw_path",
)
NON_SINUS = {
    "scp": frozenset({"AFIB", "AFLT", "SVTAC", "PSVT", "PACE", "SVARR"}),
    "snomed": frozenset(
        {
            "164889003",
            "164890007",
            "195080001",
            "426749004",
            "282825002",
            "314208002",
            "426761007",
            "67198005",
            "713422000",
            "233897008",
            "251166008",
            "426648003",
            "10370003",
            "164895002",
            "425856008",
            "29320008",
            "426664006",
        }
    ),
}
TOLERANCE = {"hr": 5.0, "qrs": 15.0, "pr": 25.0, "axis": 20.0}
P_SNR_MIN = 3.0
P_AMP_MIN = 0.02
IRREGULAR_NRMSSD = 0.10
GROUPS_ST = {
    "lateral": ("I", "aVL", "V5", "V6"),
    "inferior": ("II", "III", "aVF"),
    "anterior": ("V1", "V2", "V3", "V4"),
}
GROUPS_Q = {"lateral": ("I", "aVL"), "precordial": ("V2", "V3", "V4", "V5", "V6"), "inferior": ("II", "aVF")}


def check_open_roles(roles: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """
    Refuse any role other than train and validation.

    Parameters
    ----------
    roles : tuple[str, ...] | list[str]
        Requested roles.

    Returns
    -------
    tuple[str, ...]
        The roles, when all are open.

    Raises
    ------
    PermissionError
        If a test role is requested; MLC-03 Part B never reads them.
    ValueError
        If no role or an unknown role is requested.
    """
    roles = tuple(roles)
    closed = sorted(set(roles) & set(CLOSED_ROLES))
    if closed:
        raise PermissionError(f"MLC-03 Part B reads train and validation records only, not {closed}")
    unknown = sorted(set(roles) - set(OPEN_ROLES))
    if unknown or not roles:
        raise ValueError(f"roles must be a nonempty subset of {OPEN_ROLES}, got {list(roles)}")
    return roles


def open_manifest_lines(path: Path = MANIFEST, roles: tuple[str, ...] = OPEN_ROLES) -> set[int]:
    """
    File line numbers of the included manifest rows of open roles.

    Only the key, role and inclusion columns are read.

    Parameters
    ----------
    path : Path
        MLC-01 manifest.
    roles : tuple[str, ...]
        Open roles to keep.

    Returns
    -------
    set[int]
        One-based line numbers (line 0 is the header).
    """
    roles = check_open_roles(roles)
    frame = pd.read_csv(path, usecols=["record_key", "role", "included"], dtype=str)
    keep = (frame["included"] == "True") & frame["role"].isin(roles)
    return set((np.flatnonzero(keep.to_numpy()) + 1).tolist())


def load_open_records(path: Path = MANIFEST, roles: tuple[str, ...] = OPEN_ROLES) -> pd.DataFrame:
    """
    Identity, codes and label states of the included records of open roles.

    Rows of other roles are skipped by the CSV reader, so their labels are never parsed.

    Parameters
    ----------
    path : Path
        MLC-01 manifest.
    roles : tuple[str, ...]
        Open roles.

    Returns
    -------
    pd.DataFrame
        ``STRUCTURE`` columns, the 14 space-B states as ``b_<condition>`` (1.0, 0.0, NaN), the space-C states
        of ``EXTRA_LABELS`` as ``c_<label>``, and ``flat_<lead>`` flags; in manifest order.

    Raises
    ------
    ValueError
        If a row of another role or a non-included row was read.
    """
    lines = open_manifest_lines(path, roles)
    labels = [f"b_{name}" for name in CONDITIONS] + [f"c_{name}" for name in EXTRA_LABELS]
    flats = [f"flat_{lead}" for lead in LEADS]
    frame = pd.read_csv(
        path,
        usecols=[*STRUCTURE, *labels, *flats],
        skiprows=lambda i: i > 0 and i not in lines,
        dtype=dict.fromkeys(STRUCTURE, str),
        keep_default_na=False,
        low_memory=False,
    )
    if not frame["role"].isin(roles).all() or not (frame["included"] == "True").all():
        raise ValueError("a row outside the open roles was read")
    for column in labels:
        frame[column] = pd.to_numeric(frame[column].replace("", np.nan)).astype(float)
    for column in ("age", "duration_s", "n_flat_leads", "n_missing_leads"):
        frame[column] = pd.to_numeric(frame[column].replace("", np.nan))
    for column in flats:
        frame[column] = frame[column].astype(str) == "True"
    return frame.reset_index(drop=True)


def feature_row_index(keys: np.ndarray, directory: Path = LOCAL) -> np.ndarray:
    """
    Rows of the MLC-02 feature arrays for open records.

    Parameters
    ----------
    keys : np.ndarray
        Record keys of open records.
    directory : Path
        Feature directory with ``rows.csv``.

    Returns
    -------
    np.ndarray
        One array row per key.

    Raises
    ------
    PermissionError
        If a key belongs to a closed role in ``rows.csv``.
    """
    rows = aligned_rows(Path(directory) / "rows.csv")
    rows = rows[rows["included_v4"].astype(bool)]
    position = pd.Series(rows.index.to_numpy(), index=rows["record_key"].to_numpy())
    index = position.loc[np.asarray(keys)].to_numpy()
    check_open_roles(tuple(sorted(set(rows.loc[index, "role"]))))
    return index


def load_feature_rows(name: str, keys: np.ndarray, directory: Path = LOCAL) -> np.ndarray:
    """
    Float64 rows of one MLC-02 feature array for open records.

    Parameters
    ----------
    name : str
        ``xecg.npy`` or ``measurements.npy``.
    keys : np.ndarray
        Open record keys.
    directory : Path
        Feature directory.

    Returns
    -------
    np.ndarray
        ``[len(keys), width]``.
    """
    array = np.load(Path(directory) / name, mmap_mode="r")
    return np.asarray(array[feature_row_index(keys, directory)], dtype=np.float64)


def method_a(keys: np.ndarray, directory: Path = LOCAL) -> pd.DataFrame:
    """
    Read method A measurements of open records and derive the quantities used here.

    Parameters
    ----------
    keys : np.ndarray
        Open record keys.
    directory : Path
        Feature directory with ``measurements.npy`` and ``measurements_status.csv``.

    Returns
    -------
    pd.DataFrame
        The 130 stored columns prefixed ``a_``, ``a_status``, and ``hr_a``, ``qrs_a``, ``pr_a``, ``axis_a``
        (net QRS area in I and aVF), ``rr_nrmssd_a`` and ``rr_range_a``.
    """
    values = pd.DataFrame(
        load_feature_rows("measurements.npy", keys, directory),
        columns=[f"a_{name}" for name in FEATURE_NAMES],
    )
    status = pd.read_csv(Path(directory) / "measurements_status.csv", dtype=str).set_index("record_key")
    values.insert(0, "record_key", np.asarray(keys))
    values["a_status"] = status.loc[np.asarray(keys), "status"].to_numpy()
    values["hr_a"], values["qrs_a"], values["pr_a"] = (
        values["a_hr_bpm"],
        values["a_qrs_dur_ms"],
        values["a_pr_ms"],
    )
    values["axis_a"] = np.degrees(np.arctan2(values["a_qrs_area_aVF"], values["a_qrs_area_I"]))
    values["rr_nrmssd_a"] = values["a_rr_rmssd_ms"] / values["a_rr_mean_ms"]
    values["rr_range_a"] = values["a_rr_range_rel"] * values["a_rr_mean_ms"]
    return values


def angle_difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """
    Signed difference of two angles in degrees, wrapped to [-180, 180).

    Parameters
    ----------
    first, second : np.ndarray
        Angles in degrees.

    Returns
    -------
    np.ndarray
        ``first - second`` wrapped.
    """
    return (np.asarray(first) - np.asarray(second) + 180.0) % 360.0 - 180.0


def consensus(table: pd.DataFrame) -> pd.DataFrame:
    """
    Keep method B's value where methods A and B agree within ``TOLERANCE``.

    Parameters
    ----------
    table : pd.DataFrame
        ``<name>_a`` and ``<name>_b`` columns for ``hr``, ``qrs``, ``pr`` and ``axis``.

    Returns
    -------
    pd.DataFrame
        For each name: ``<name>_both`` (both measured), ``<name>_agree`` and ``<name>`` (method B's value
        where they agree, NaN otherwise).
    """
    out = pd.DataFrame(index=table.index)
    for name, tolerance in TOLERANCE.items():
        a, b = table[f"{name}_a"].to_numpy(float), table[f"{name}_b"].to_numpy(float)
        difference = angle_difference(a, b) if name == "axis" else a - b
        both = np.isfinite(a) & np.isfinite(b)
        agree = both & (np.abs(np.nan_to_num(difference, nan=np.inf)) <= tolerance)
        out[f"{name}_both"], out[f"{name}_agree"], out[name] = both, agree, np.where(agree, b, np.nan)
    return out


def lead_values(table: pd.DataFrame, prefix: str, leads: tuple[str, ...]) -> np.ndarray:
    """
    Per-lead columns of method B as an array.

    Parameters
    ----------
    table : pd.DataFrame
        Method B columns named ``<prefix>_<lead>``.
    prefix : str
        Measurement prefix, such as ``st60`` or ``t``.
    leads : tuple[str, ...]
        Leads in column order.

    Returns
    -------
    np.ndarray
        ``[records, len(leads)]`` float values.
    """
    return table[[f"{prefix}_{lead}" for lead in leads]].to_numpy(float)


def grouped_count(
    flags: np.ndarray, leads: tuple[str, ...], groups: dict[str, tuple[str, ...]]
) -> np.ndarray:
    """
    Largest number of flagged leads within one lead group.

    Parameters
    ----------
    flags : np.ndarray
        ``[records, len(leads)]`` booleans.
    leads : tuple[str, ...]
        Column leads.
    groups : dict[str, tuple[str, ...]]
        Contiguous lead groups.

    Returns
    -------
    np.ndarray
        Per record, the maximum over groups of flagged leads in the group.
    """
    counts = [flags[:, [leads.index(lead) for lead in members]].sum(axis=1) for members in groups.values()]
    return np.max(np.column_stack(counts), axis=1)


def q_qualifies(table: pd.DataFrame) -> np.ndarray:
    """
    Leads with a Q wave of the infarct size of the Fourth Universal Definition (Thygesen et al., 2018).

    V2-V3: Q of at least 20 ms or a QS complex. I, II, aVL, aVF, V4-V6: Q of at least 30 ms and 0.1 mV deep,
    or a QS complex. Other leads never qualify. A QS complex is a QRS without a positive deflection of 0.05
    mV.

    Parameters
    ----------
    table : pd.DataFrame
        Method B ``qdur_``, ``q_`` and ``r_`` columns.

    Returns
    -------
    np.ndarray
        ``[records, 12]`` booleans in ``LEADS`` order.
    """
    duration, depth, r = (lead_values(table, name, LEADS) for name in ("qdur", "q", "r"))
    qs = r < 0.05
    septal = np.isin(LEADS, ("V2", "V3"))
    other = np.isin(LEADS, ("I", "II", "aVL", "aVF", "V4", "V5", "V6"))
    return (septal & ((duration >= 20) | qs)) | (other & (((duration >= 30) & (depth >= 0.1)) | qs))


def derived_b(table: pd.DataFrame) -> pd.DataFrame:
    """
    Criteria built from method B's per-lead values.

    Parameters
    ----------
    table : pd.DataFrame
        Method B columns and ``sex``.

    Returns
    -------
    pd.DataFrame
        ``sokolow_mv`` (S V1 + larger R of V5 and V6), ``cornell_mv`` (R aVL + S V3), ``voltage_criterion``
        (Sokolow-Lyon at least 3.5 mV, or Cornell above 2.8 mV in men and 2.0 mV in women), ``st_min_mv``
        (lowest ST 60 ms after the QRS, aVR excluded), ``st_dep_group`` (most leads of one group with ST at
        most -0.05 mV), ``t_inverted``, ``t_low`` (leads of ``UPRIGHT_T`` with T at most -0.1 mV, or between
        -0.1 and 0.05 mV), ``q_group`` (most qualifying Q leads in one group), ``n_premature_aberrant_any``
        (complete and edge beats), ``rbbb_morphology`` (V1 with
        two positive peaks or a positive terminal 40 ms, and a negative terminal 40 ms in I or V6),
        ``p_present`` and ``irregular``.
    """
    out = pd.DataFrame(index=table.index)
    out["sokolow_mv"] = table["s_V1"] + np.maximum(table["r_V5"], table["r_V6"])
    out["cornell_mv"] = table["r_aVL"] + table["s_V3"]
    cornell_limit = np.where(table["sex"] == "M", 2.8, np.where(table["sex"] == "F", 2.0, np.inf))
    out["voltage_criterion"] = (out["sokolow_mv"] >= 3.5) | (out["cornell_mv"] > cornell_limit)
    not_avr = tuple(lead for lead in LEADS if lead != "aVR")
    st = lead_values(table, "st60", not_avr)
    out["st_min_mv"] = np.nanmin(np.where(np.isfinite(st), st, np.inf), axis=1)
    out.loc[~np.isfinite(st).any(axis=1), "st_min_mv"] = np.nan
    out["st_dep_group"] = grouped_count(st <= -0.05, not_avr, GROUPS_ST)
    t = lead_values(table, "t", UPRIGHT_T)
    out["t_inverted"] = (t <= -0.1).sum(axis=1)
    out["t_low"] = ((t > -0.1) & (t < 0.05)).sum(axis=1)
    out["q_group"] = grouped_count(q_qualifies(table), LEADS, GROUPS_Q)
    v1 = (table["v1_positive_peaks"] >= 2) | (table["v1_terminal_mv"] > 0.0)
    out["rbbb_morphology"] = v1 & ((table["term_I"] < 0.0) | (table["term_V6"] < 0.0))
    out["p_present"] = (table["p_snr"] >= P_SNR_MIN) & (table["p_amp_mv"] >= P_AMP_MIN)
    out["irregular"] = table["rr_nrmssd_b"] >= IRREGULAR_NRMSSD
    out["n_premature_aberrant_any"] = table["n_premature_aberrant"] + table["n_premature_aberrant_edge"]
    measured = table["status_b"] == "ok"
    for column in out.columns:
        if out[column].dtype == bool:
            out[column] = out[column].where(measured, other=False)
    out.loc[~measured, ["sokolow_mv", "cornell_mv", "st_min_mv"]] = np.nan
    return out


def record_codes(source: str, raw: str, heart_axis: str) -> object:
    """
    Parse a record's codes into the form ``mlc_labels`` reads.

    Parameters
    ----------
    source : str
        Manifest source.
    raw : str
        ``raw_codes`` field.
    heart_axis : str
        PTB-XL ``heart_axis`` field.

    Returns
    -------
    object
        PTB-XL: statement-to-likelihood dict with the axis pseudo-statement; Challenge: frozenset of SNOMED
        codes.
    """
    if source == "ptbxl":
        return mlc_labels.ptbxl_codes(ast.literal_eval(raw), heart_axis)
    return mlc_labels.parse_snomed(raw)


def sinus_by_codes(source: str, codes: object) -> bool:
    """
    Whether a record's codes state sinus rhythm or no other rhythm.

    Parameters
    ----------
    source : str
        Manifest source.
    codes : object
        Output of ``record_codes``.

    Returns
    -------
    bool
        PTB-XL: ``SR``, ``STACH``, ``SBRAD`` or ``SARRH`` and no ``NON_SINUS`` statement; Challenge: no
        ``NON_SINUS`` code.
    """
    vocabulary = "scp" if source == "ptbxl" else "snomed"
    found = set(codes)
    if found & NON_SINUS[vocabulary]:
        return False
    return bool(found & {"SR", "STACH", "SBRAD", "SARRH"}) if source == "ptbxl" else True


def wilson(positives: int, total: int) -> tuple[float, float]:
    """
    Wilson 95% interval, NaN for an empty group.

    Parameters
    ----------
    positives, total : int
        Counts.

    Returns
    -------
    tuple[float, float]
        Lower and upper bounds.
    """
    return wilson_interval(positives, total) if total else (np.nan, np.nan)


def b_columns(measured: pd.DataFrame) -> pd.DataFrame:
    """
    Rename method B results to the names used in the analysis table.

    Parameters
    ----------
    measured : pd.DataFrame
        One ``measure_record`` dictionary per row, with ``record_key``.

    Returns
    -------
    pd.DataFrame
        The same columns, with ``status``, ``flat_leads``, rate, interval, axis and RR measures renamed to
        ``status_b``, ``flat_leads_b``, ``hr_b``, ``qrs_b``, ``pr_b``, ``axis_b``, ``rr_cv_b``,
        ``rr_nrmssd_b``, ``rr_range_b`` and ``rr_range_dominant_b``.
    """
    names = {
        "status": "status_b",
        "flat_leads": "flat_leads_b",
        "hr_bpm": "hr_b",
        "qrs_ms": "qrs_b",
        "pr_ms": "pr_b",
        "axis_deg": "axis_b",
        "rr_cv": "rr_cv_b",
        "rr_nrmssd": "rr_nrmssd_b",
        "rr_range_ms": "rr_range_b",
        "rr_range_dominant_ms": "rr_range_dominant_b",
    }
    return measured.rename(columns=names)


def analysis_table(records: pd.DataFrame, a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """
    One row per open record: identity, labels, both methods, consensus values and derived criteria.

    Parameters
    ----------
    records : pd.DataFrame
        Output of ``load_open_records``.
    a : pd.DataFrame
        Output of ``method_a`` for the same keys.
    b : pd.DataFrame
        Output of ``b_columns`` for the same keys.

    Returns
    -------
    pd.DataFrame
        Joined on ``record_key`` in ``records`` order, with ``consensus`` and ``derived_b`` columns, ``codes``
        (``record_codes``), ``sinus_codes`` (``sinus_by_codes``) and ``sinus_measured`` (sinus by codes, P
        wave present and both rate methods agree).

    Raises
    ------
    ValueError
        If a record lacks either method's row.
    """
    table = records.merge(a, on="record_key", how="left", validate="one_to_one")
    table = table.merge(b, on="record_key", how="left", validate="one_to_one")
    if table["a_status"].isna().any() or table["status_b"].isna().any():
        raise ValueError("a record has no measurement row")
    table = pd.concat([table, consensus(table), derived_b(table)], axis=1)
    codes = [
        record_codes(s, r, h)
        for s, r, h in zip(table["source"], table["raw_codes"], table["heart_axis"], strict=True)
    ]
    table["codes"] = codes
    table["sinus_codes"] = [sinus_by_codes(s, c) for s, c in zip(table["source"], codes, strict=True)]
    table["sinus_measured"] = table["sinus_codes"] & table["p_present"] & table["hr_agree"]
    return table


def region_masks(value: pd.Series, low: float, high: float, tolerance: float) -> dict[str, pd.Series]:
    """
    Inside and outside a closed interval, strictly and with a tolerance.

    Parameters
    ----------
    value : pd.Series
        Measurement, NaN when unavailable.
    low, high : float
        Interval of the condition (infinite ends allowed).
    tolerance : float
        Measurement tolerance.

    Returns
    -------
    dict[str, pd.Series]
        ``inside`` and ``outside`` (strict), ``inside_tol`` (inside by more than the tolerance) and
        ``outside_tol`` (outside by more than the tolerance); all False where the value is missing.
    """
    known = value.notna()
    return {
        "inside": known & (value >= low) & (value <= high),
        "outside": known & ((value < low) | (value > high)),
        "inside_tol": known & (value >= low + tolerance) & (value <= high - tolerance),
        "outside_tol": known & ((value < low - tolerance) | (value > high + tolerance)),
    }


CHECKS: dict[str, dict[str, object]] = {
    "sinus_tachycardia": {
        "measure": "hr",
        "low": 100.0,
        "high": np.inf,
        "tolerance": 5.0,
        "population": "sinus",
        "bound": "heart rate above 100 per minute, sinus rhythm",
        "reference": "Mason et al. 2007 (AHA/ACCF/HRS part II)",
    },
    "sinus_bradycardia": {
        "measure": "hr",
        "low": 0.0,
        "high": 60.0,
        "tolerance": 5.0,
        "population": "sinus",
        "bound": "heart rate below 60 per minute, sinus rhythm",
        "reference": "Mason et al. 2007 (AHA/ACCF/HRS part II)",
    },
    "sinus_arrhythmia": {
        "measure": "rr_range_dominant_b",
        "low": 120.0,
        "high": np.inf,
        "tolerance": 20.0,
        "population": "sinus",
        "bound": "longest minus shortest RR between consecutive dominant beats above 120 ms",
        "reference": "a commonly used cut-off; no single standard",
    },
    "af_or_flutter": {
        "measure": "rr_nrmssd_b",
        "low": IRREGULAR_NRMSSD,
        "high": np.inf,
        "tolerance": 0.03,
        "population": "measured",
        "bound": "irregular RR (normalized RMSSD of at least 0.10) without a detectable P wave",
        "reference": "AF: irregular RR and no distinct P waves (Hindricks et al. 2021, ESC)",
    },
    "premature_ventricular_complex": {
        "measure": "n_premature_aberrant_any",
        "low": 1.0,
        "high": np.inf,
        "tolerance": 0.0,
        "population": "ten_seconds",
        "bound": "at least one premature beat unlike the dominant beat in the first 10 s",
        "reference": "premature complex of different QRS morphology",
    },
    "first_degree_av_block": {
        "measure": "pr",
        "low": 200.0,
        "high": np.inf,
        "tolerance": 20.0,
        "population": "p_present",
        "bound": "PR interval above 200 ms",
        "reference": "Mason et al. 2007 (AHA/ACCF/HRS part II)",
    },
    "complete_rbbb": {
        "measure": "qrs",
        "low": 120.0,
        "high": np.inf,
        "tolerance": 10.0,
        "population": "measured",
        "bound": "QRS duration of at least 120 ms",
        "reference": "Surawicz et al. 2009 (AHA/ACCF/HRS part III)",
    },
    "incomplete_rbbb": {
        "measure": "qrs",
        "low": 110.0,
        "high": 119.999,
        "tolerance": 10.0,
        "population": "measured",
        "bound": "QRS duration 110-119 ms with RBBB morphology",
        "reference": "Surawicz et al. 2009 (AHA/ACCF/HRS part III)",
    },
    "lafb": {
        "measure": "axis",
        "low": -90.0,
        "high": -45.0,
        "tolerance": 15.0,
        "population": "measured",
        "bound": "frontal QRS axis -45 to -90 degrees",
        "reference": "Surawicz et al. 2009 (part III)",
    },
    "left_axis_deviation": {
        "measure": "axis",
        "low": -90.0,
        "high": -30.0,
        "tolerance": 15.0,
        "population": "measured",
        "bound": "frontal QRS axis -30 to -90 degrees",
        "reference": "Surawicz et al. 2009 (part III)",
    },
    "lvh": {
        "measure": "sokolow_mv",
        "low": 3.5,
        "high": np.inf,
        "tolerance": 0.5,
        "population": "measured",
        "bound": "Sokolow-Lyon at least 3.5 mV or Cornell above 2.8 mV (men) / 2.0 mV (women)",
        "reference": "Hancock et al. 2009 (AHA/ACCF/HRS part V)",
    },
    "st_depression": {
        "measure": "st_min_mv",
        "low": -np.inf,
        "high": -0.05,
        "tolerance": 0.05,
        "population": "measured",
        "bound": "ST at least 0.05 mV below the PR level in two leads of one group",
        "reference": "Thygesen et al. 2018 (Fourth Universal Definition of MI)",
    },
    "t_wave_abnormality": {
        "measure": "t_inverted",
        "low": 1.0,
        "high": np.inf,
        "tolerance": 0.0,
        "population": "measured",
        "bound": "T inverted (at most -0.1 mV) or low (below 0.05 mV) in I, II or V3-V6",
        "reference": "Rautaharju et al. 2009 (AHA/ACCF/HRS part IV)",
    },
    "mi_any": {
        "measure": "q_group",
        "low": 2.0,
        "high": np.inf,
        "tolerance": 0.0,
        "population": "measured",
        "bound": "infarct-size Q waves in two leads of one group",
        "reference": "Thygesen et al. 2018 (Fourth Universal Definition of MI)",
    },
}


def population(table: pd.DataFrame, name: str) -> pd.Series:
    """
    Select the records on which a check is read.

    Parameters
    ----------
    table : pd.DataFrame
        Output of ``analysis_table``.
    name : str
        ``measured`` (method B delineated the record), ``sinus`` (``sinus_measured``), ``p_present``,
        ``no_p``, ``ten_seconds`` (measured and the recording is 10 s long, so the label describes the
        analysed window), ``longer`` (measured, longer than 10 s), ``rbbb_morphology`` or ``rbbb_labelled``
        (complete or incomplete RBBB positive).

    Returns
    -------
    pd.Series
        Boolean mask.
    """
    measured = table["status_b"] == "ok"
    if name == "measured":
        return measured
    if name in ("ten_seconds", "longer"):
        ten = table["duration_s"] <= 10.0 + 1e-6
        return measured & (ten if name == "ten_seconds" else ~ten)
    if name == "rbbb_labelled":
        return measured & ((table["b_complete_rbbb"] == 1) | (table["b_incomplete_rbbb"] == 1))
    if name == "no_p":
        return measured & ~table["p_present"]
    columns = {"sinus": "sinus_measured", "p_present": "p_present", "rbbb_morphology": "rbbb_morphology"}
    return measured & table[columns[name]]


def criterion(table: pd.DataFrame, condition: str) -> dict[str, pd.Series]:
    """
    Where a record meets or misses a condition's measured definition.

    Parameters
    ----------
    table : pd.DataFrame
        Output of ``analysis_table``.
    condition : str
        Key of ``CHECKS``.

    Returns
    -------
    dict[str, pd.Series]
        ``region_masks`` of the condition's measure, with condition-specific parts: AF requires the absence
        of a P wave for ``inside`` and its presence for ``outside``; incomplete RBBB requires RBBB morphology
        for ``inside``; LVH counts Cornell voltage as well; T-wave abnormality counts low T waves as inside;
        ST depression requires two leads of one group.
    """
    spec = CHECKS[condition]
    masks = region_masks(table[spec["measure"]], spec["low"], spec["high"], spec["tolerance"])
    if condition == "af_or_flutter":
        for key in ("inside", "inside_tol"):
            masks[key] = masks[key] & ~table["p_present"]
        for key in ("outside", "outside_tol"):
            masks[key] = masks[key] & table["p_present"]
    elif condition == "incomplete_rbbb":
        for key in ("inside", "inside_tol"):
            masks[key] = masks[key] & table["rbbb_morphology"]
    else:
        masks = adjust_criterion(table, condition, masks)
    return masks


def adjust_criterion(
    table: pd.DataFrame, condition: str, masks: dict[str, pd.Series]
) -> dict[str, pd.Series]:
    """
    Condition-specific parts of ``criterion`` for LVH, ST depression and T-wave abnormality.

    Parameters
    ----------
    table : pd.DataFrame
        Output of ``analysis_table``.
    condition : str
        Key of ``CHECKS``.
    masks : dict[str, pd.Series]
        ``region_masks`` of the condition's measure.

    Returns
    -------
    dict[str, pd.Series]
        The adjusted masks; other conditions unchanged.
    """
    if condition == "lvh":
        voltage = table["voltage_criterion"]
        masks["inside"] = masks["inside"] | voltage
        masks["outside"] = masks["outside"] & ~voltage
        masks["outside_tol"] = masks["outside_tol"] & ~voltage
    elif condition == "st_depression":
        group = table["st_dep_group"] >= 2
        masks["inside"] = masks["inside"] & group
        masks["inside_tol"] = masks["inside_tol"] & group
    elif condition == "t_wave_abnormality":
        known = table["status_b"] == "ok"
        low = known & (table["t_low"] >= 1)
        masks["inside"] = masks["inside"] | low
        masks["outside"] = masks["outside"] & ~low
        masks["outside_tol"] = masks["outside_tol"] & ~low
        masks["inside_tol"] = known & (table["t_inverted"] >= 2)
    return masks


def quantiles(values: pd.Series) -> list[float]:
    """
    5th, 25th, 50th, 75th and 95th percentiles, rounded.

    Parameters
    ----------
    values : pd.Series
        Measurement values; NaN ignored.

    Returns
    -------
    list[float]
        Five percentiles, or NaN when no value is present.
    """
    finite = values.dropna().to_numpy(float)
    return (
        [round(float(v), 3) for v in np.percentile(finite, [5, 25, 50, 75, 95])]
        if len(finite)
        else [np.nan] * 5
    )


def label_measurement_rows(table: pd.DataFrame, condition: str) -> list[dict[str, object]]:
    """
    Item 1: a condition's measure among positives, negatives and undefined, per source.

    Parameters
    ----------
    table : pd.DataFrame
        Output of ``analysis_table``.
    condition : str
        Key of ``CHECKS``.

    Returns
    -------
    list[dict[str, object]]
        Per source and state: records, records in the check's population, records with the measure,
        percentiles, and counts meeting the definition (``inside``) and missing it (``outside``, strictly and
        beyond the tolerance). For a positive the ``outside`` counts are the contradicting records; for a
        negative the ``inside`` counts are.
    """
    spec, masks = CHECKS[condition], criterion(table, condition)
    included = population(table, spec["population"])
    rows = []
    for source in SOURCES:
        for state, value in (("positive", 1.0), ("negative", 0.0), ("undefined", np.nan)):
            label = table[f"b_{condition}"]
            in_state = (table["source"] == source) & (label.isna() if np.isnan(value) else label == value)
            part = in_state & included
            measured = part & table[spec["measure"]].notna()
            if not in_state.any():
                continue
            rows.append(
                {
                    "condition": condition,
                    "source": source,
                    "state": state,
                    "records": int(in_state.sum()),
                    "in_population": int(part.sum()),
                    "measured": int(measured.sum()),
                    "measure": spec["measure"],
                    "p5_p25_p50_p75_p95": quantiles(table.loc[measured, spec["measure"]]),
                    **{
                        key: int((masks[key] & part).sum())
                        for key in ("inside", "inside_tol", "outside", "outside_tol")
                    },
                }
            )
    return rows


MIN_BIN = 30
MATCHED: dict[str, dict[str, object]] = {
    "sinus_tachycardia": {
        "label": "b_sinus_tachycardia",
        "measure": "hr",
        "population": "sinus",
        "bins": tuple(range(80, 150, 5)),
    },
    "sinus_bradycardia": {
        "label": "b_sinus_bradycardia",
        "measure": "hr",
        "population": "sinus",
        "bins": tuple(range(35, 85, 5)),
    },
    "sinus_arrhythmia": {
        "label": "b_sinus_arrhythmia",
        "measure": "rr_range_dominant_b",
        "population": "sinus",
        "bins": (0, 40, 80, 120, 160, 200, 280, 400, np.inf),
    },
    "af_or_flutter|no_p": {
        "label": "b_af_or_flutter",
        "measure": "rr_nrmssd_b",
        "population": "no_p",
        "bins": (0, 0.03, 0.06, 0.1, 0.15, 0.2, 0.3, np.inf),
    },
    "af_or_flutter|p_present": {
        "label": "b_af_or_flutter",
        "measure": "rr_nrmssd_b",
        "population": "p_present",
        "bins": (0, 0.03, 0.06, 0.1, 0.15, 0.2, 0.3, np.inf),
    },
    "premature_ventricular_complex|10s": {
        "label": "b_premature_ventricular_complex",
        "measure": "n_premature_aberrant_any",
        "population": "ten_seconds",
        "bins": (0, 1, 2, 3, np.inf),
    },
    "premature_ventricular_complex|longer": {
        "label": "b_premature_ventricular_complex",
        "measure": "n_premature_aberrant_any",
        "population": "longer",
        "bins": (0, 1, 2, 3, np.inf),
    },
    "first_degree_av_block": {
        "label": "b_first_degree_av_block",
        "measure": "pr",
        "population": "p_present",
        "bins": tuple(range(100, 340, 20)),
    },
    "complete_rbbb": {
        "label": "b_complete_rbbb",
        "measure": "qrs",
        "population": "rbbb_morphology",
        "bins": tuple(range(80, 190, 10)),
    },
    "incomplete_rbbb": {
        "label": "b_incomplete_rbbb",
        "measure": "qrs",
        "population": "rbbb_morphology",
        "bins": tuple(range(80, 190, 10)),
    },
    "complete_among_rbbb": {
        "label": "b_complete_rbbb",
        "measure": "qrs",
        "population": "rbbb_labelled",
        "bins": tuple(range(80, 190, 10)),
    },
    "lafb": {
        "label": "b_lafb",
        "measure": "axis",
        "population": "measured",
        "bins": tuple(range(-105, 61, 15)),
    },
    "left_axis_deviation": {
        "label": "b_left_axis_deviation",
        "measure": "axis",
        "population": "measured",
        "bins": tuple(range(-105, 61, 15)),
    },
    "lvh": {
        "label": "b_lvh",
        "measure": "sokolow_mv",
        "population": "measured",
        "bins": (0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, np.inf),
    },
    "st_depression": {
        "label": "b_st_depression",
        "measure": "st_min_mv",
        "population": "measured",
        "bins": (-np.inf, -0.3, -0.2, -0.15, -0.1, -0.05, 0, 0.05, np.inf),
    },
    "t_wave_abnormality": {
        "label": "b_t_wave_abnormality",
        "measure": "t_inverted",
        "population": "measured",
        "bins": (0, 1, 2, 3, 4, 7),
    },
    "mi_any": {"label": "b_mi_any", "measure": "q_group", "population": "measured", "bins": (0, 1, 2, 3, 6)},
}


def matched_prevalence(table: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Item 2: prevalence of a label per source within bins of the same measurement.

    Parameters
    ----------
    table : pd.DataFrame
        Output of ``analysis_table``.
    name : str
        Key of ``MATCHED``.

    Returns
    -------
    pd.DataFrame
        One row per source and bin ``[low, high)``: defined records, positives, prevalence and its Wilson 95%
        interval. Only records in the population with the label defined and the measure present count.
    """
    spec = MATCHED[name]
    label, value = table[spec["label"]], table[spec["measure"]]
    keep = population(table, spec["population"]) & label.notna() & value.notna()
    rows = []
    bins = spec["bins"]
    for source in SOURCES:
        part = keep & (table["source"] == source)
        if not part.any():
            continue
        for low, high in zip(bins[:-1], bins[1:], strict=True):
            inside = part & (value >= low) & (value < high)
            n, positives = int(inside.sum()), int((label[inside] == 1).sum())
            low_ci, high_ci = wilson(positives, n)
            rows.append(
                {
                    "name": name,
                    "label": spec["label"],
                    "measure": spec["measure"],
                    "population": spec["population"],
                    "source": source,
                    "bin_low": low,
                    "bin_high": high,
                    "defined": n,
                    "positives": positives,
                    "prevalence": positives / n if n else np.nan,
                    "ci_low": low_ci,
                    "ci_high": high_ci,
                }
            )
    return pd.DataFrame(rows)


def inconsistent_bins(prevalence: pd.DataFrame, minimum: int = MIN_BIN) -> pd.DataFrame:
    """
    Source pairs whose prevalence at the same bin differs by a factor of two with disjoint 95% intervals.

    This is the protocol's reading rule ("Reading the result"). Bins with fewer than ``minimum`` defined
    records in either source are not compared.

    Parameters
    ----------
    prevalence : pd.DataFrame
        Output of ``matched_prevalence``.
    minimum : int
        Smallest number of defined records per source and bin.

    Returns
    -------
    pd.DataFrame
        Every compared pair with ``ratio`` (higher over lower prevalence, infinite when the lower is 0),
        ``disjoint`` and ``inconsistent`` (ratio of at least 2 and disjoint intervals).
    """
    rows = []
    for (name, low, high), part in prevalence.groupby(["name", "bin_low", "bin_high"], sort=False):
        part = part[part["defined"] >= minimum].reset_index(drop=True)
        for i in range(len(part)):
            for j in range(i + 1, len(part)):
                first, second = part.loc[i], part.loc[j]
                top, bottom = (
                    (first, second) if first["prevalence"] >= second["prevalence"] else (second, first)
                )
                ratio = np.inf if bottom["prevalence"] == 0 else top["prevalence"] / bottom["prevalence"]
                ratio = 1.0 if top["prevalence"] == 0 else ratio
                disjoint = bottom["ci_high"] < top["ci_low"]
                rows.append(
                    {
                        "name": name,
                        "bin_low": low,
                        "bin_high": high,
                        "higher": top["source"],
                        "lower": bottom["source"],
                        "prevalence_higher": top["prevalence"],
                        "prevalence_lower": bottom["prevalence"],
                        "defined_higher": top["defined"],
                        "defined_lower": bottom["defined"],
                        "ratio": ratio,
                        "disjoint": bool(disjoint),
                        "inconsistent": bool(disjoint and ratio >= 2.0),
                    }
                )
    return pd.DataFrame(rows)


def implied_cutoff(
    table: pd.DataFrame, condition: str, source: str, draws: int = 500, seed: int = SEED
) -> dict[str, float]:
    """
    Heart rate at which a source's rate label reaches a probability of one half.

    A logistic regression of the label on the consensus heart rate (almost unpenalized) is fitted on the
    source's sinus records where the label is defined; the cut-off is minus intercept over slope. The interval
    is the 2.5 and 97.5 percentiles over whole-group bootstrap draws (``intervals.patient_groups``).

    Parameters
    ----------
    table : pd.DataFrame
        Output of ``analysis_table``.
    condition : str
        ``sinus_bradycardia`` or ``sinus_tachycardia``.
    source : str
        Manifest source.
    draws : int
        Bootstrap draws.
    seed : int
        Generator seed.

    Returns
    -------
    dict[str, float]
        ``cutoff_bpm``, ``ci_low``, ``ci_high``, ``records`` and ``positives``; NaN cut-off when the label
        has fewer than 10 positives or negatives.
    """
    from sklearn.linear_model import LogisticRegression

    from ecg_experiment.intervals import patient_groups, patient_resample

    label = table[f"b_{condition}"]
    keep = population(table, "sinus") & label.notna() & table["hr"].notna() & (table["source"] == source)
    x, y = table.loc[keep, "hr"].to_numpy(float)[:, None], label[keep].to_numpy(int)
    result = {"condition": condition, "source": source, "records": int(len(y)), "positives": int(y.sum())}
    if y.sum() < 10 or (1 - y).sum() < 10:
        return {**result, "cutoff_bpm": np.nan, "ci_low": np.nan, "ci_high": np.nan}

    def fit(rows: np.ndarray) -> float:
        model = LogisticRegression(C=1e6, max_iter=1000).fit(x[rows] - 70.0, y[rows])
        return float(70.0 - model.intercept_[0] / model.coef_[0, 0])

    groups = patient_groups(table.loc[keep, "group_id"].to_numpy())
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        rows = patient_resample(groups, rng)
        if 0 < y[rows].sum() < len(rows):
            values.append(fit(rows))
    low, high = np.percentile(values, [2.5, 97.5])
    return {**result, "cutoff_bpm": fit(np.arange(len(y))), "ci_low": float(low), "ci_high": float(high)}


def vocabulary(source: str) -> str:
    """
    Code vocabulary of a source.

    Parameters
    ----------
    source : str
        Manifest source.

    Returns
    -------
    str
        ``scp`` for PTB-XL, ``snomed`` otherwise.
    """
    return mlc_labels.VOCABULARY[source]


def listing(table: pd.DataFrame, specs: frozenset[str] | tuple[str, ...], source: str) -> pd.Series:
    """
    Mark the records of a source that list any of the given code specifications.

    Parameters
    ----------
    table : pd.DataFrame
        Output of ``analysis_table`` with ``codes``.
    specs : frozenset[str] | tuple[str, ...]
        Code specifications in the source's vocabulary.
    source : str
        Manifest source.

    Returns
    -------
    pd.Series
        Boolean mask over all records (False outside the source).
    """
    specs = frozenset(specs)
    in_source = table["source"] == source
    hits = [
        bool(specs) and mlc_labels.listed(specs, codes, vocabulary(source)) if inside else False
        for codes, inside in zip(table["codes"], in_source, strict=True)
    ]
    return pd.Series(hits, index=table.index)


def code_usage(table: pd.DataFrame) -> pd.DataFrame:
    """
    Item 2: which codes each source uses for each condition.

    Parameters
    ----------
    table : pd.DataFrame
        Output of ``analysis_table`` with ``codes``.

    Returns
    -------
    pd.DataFrame
        One row per condition, source and code specification of ``mlc_labels.CONDITIONS`` (positive or
        ambiguous): records listing it, and the condition's positives, negatives and undefined in the source.
    """
    rows = []
    for condition in CONDITIONS:
        definition = mlc_labels.CONDITIONS[condition]
        for source in SOURCES:
            label = table.loc[table["source"] == source, f"b_{condition}"]
            states = {
                "positives": int((label == 1).sum()),
                "negatives": int((label == 0).sum()),
                "undefined": int(label.isna().sum()),
            }
            for role in ("positive", "ambiguous"):
                for code in sorted(definition[role][vocabulary(source)]):
                    rows.append(
                        {
                            "condition": condition,
                            "source": source,
                            "role": role,
                            "code": code,
                            "records_listing": int(listing(table, (code,), source).sum()),
                            **states,
                        }
                    )
    return pd.DataFrame(rows)


SIBLINGS: dict[str, dict[str, dict[str, frozenset[str]]]] = {
    "complete_rbbb": {"incomplete RBBB": mlc_labels.CONDITIONS["incomplete_rbbb"]["positive"]},
    "left_axis_deviation": {"LAFB": mlc_labels.codes(("LAFB",), ("445118002",))},
    "af_or_flutter": {
        "atrial flutter": mlc_labels.CONDITIONS["atrial_flutter"]["positive"],
        "atrial fibrillation": mlc_labels.CONDITIONS["atrial_fibrillation"]["positive"],
    },
    "lvh": {"LV high voltage": mlc_labels.LV_HIGH_VOLTAGE},
    "st_depression": {"nonspecific or ischemic ST-T": mlc_labels.ST_T_NONSPECIFIC_OR_ISCHEMIC},
    "t_wave_abnormality": {"nonspecific or ischemic ST-T": mlc_labels.ST_T_NONSPECIFIC_OR_ISCHEMIC},
    "mi_any": {"Q wave or poor R progression": mlc_labels.Q_WAVE_OR_POOR_R_PROGRESSION},
    "first_degree_av_block": {"AV block of unstated degree": mlc_labels.codes(snomed=("233917008",))},
    "premature_ventricular_complex": {
        "ventricular tachycardia": mlc_labels.codes(snomed=mlc_labels.VENTRICULAR_TACHYCARDIA_SNOMED)
    },
    "sinus_bradycardia": {"bradycardia of unstated origin": mlc_labels.codes(snomed=("426627000",))},
}


def separate_coding(table: pd.DataFrame) -> pd.DataFrame:
    """
    Item 2: conditions defined in a source that never uses the code separating them from a neighbour.

    For each condition of ``SIBLINGS`` and each source where it has positives and negatives, count the
    records listing the sibling statement. A count of zero means the source does not code the distinction,
    yet the labels are defined there.

    Parameters
    ----------
    table : pd.DataFrame
        Output of ``analysis_table`` with ``codes``.

    Returns
    -------
    pd.DataFrame
        ``condition``, ``source``, ``sibling``, ``sibling_codes`` in the source's vocabulary, records listing
        them, the condition's positives and negatives, and ``not_coded_separately``.
    """
    rows = []
    for condition, siblings in SIBLINGS.items():
        for source in SOURCES:
            label = table.loc[table["source"] == source, f"b_{condition}"]
            positives, negatives = int((label == 1).sum()), int((label == 0).sum())
            if not positives or not negatives:
                continue
            for sibling, specs in siblings.items():
                codes = specs[vocabulary(source)]
                count = int(listing(table, codes, source).sum()) if codes else 0
                rows.append(
                    {
                        "condition": condition,
                        "source": source,
                        "sibling": sibling,
                        "sibling_codes": " ".join(sorted(codes)) or "(not codable)",
                        "records_listing": count,
                        "positives": positives,
                        "negatives": negatives,
                        "not_coded_separately": count == 0,
                    }
                )
    return pd.DataFrame(rows)


CONTRADICTIONS = (
    ("b_sinus_tachycardia", "b_sinus_bradycardia", "the rate cannot be above 100 and below 60 per minute"),
    (
        "b_sinus_tachycardia",
        "b_af_or_flutter",
        "sinus rhythm and atrial fibrillation or flutter exclude each other",
    ),
    (
        "b_sinus_bradycardia",
        "b_af_or_flutter",
        "sinus rhythm and atrial fibrillation or flutter exclude each other",
    ),
    (
        "b_sinus_arrhythmia",
        "b_af_or_flutter",
        "sinus rhythm and atrial fibrillation or flutter exclude each other",
    ),
    (
        "b_sinus_tachycardia",
        "c_supraventricular_tachycardia",
        "sinus and supraventricular tachycardia exclude each other",
    ),
    ("b_first_degree_av_block", "b_af_or_flutter", "a PR interval needs P waves, which AF and flutter lack"),
    ("b_first_degree_av_block", "c_preexcitation", "pre-excitation shortens the PR interval"),
    (
        "b_complete_rbbb",
        "b_incomplete_rbbb",
        "QRS of at least and below 120 ms (MLC-01 makes both undefined)",
    ),
    ("b_complete_rbbb", "c_complete_lbbb", "MLC-01 contradictory pair (both made undefined)"),
    ("b_incomplete_rbbb", "c_complete_lbbb", "MLC-01 contradictory pair (both made undefined)"),
    ("b_lafb", "c_lpfb", "the two fascicular blocks need opposite frontal axes"),
    ("b_lafb", "c_right_axis_deviation", "LAFB needs a frontal axis of -45 degrees or less"),
    ("b_left_axis_deviation", "c_right_axis_deviation", "one frontal axis cannot deviate both ways"),
)
UNREADABLE = (
    "b_complete_rbbb",
    "b_incomplete_rbbb",
    "b_lafb",
    "b_left_axis_deviation",
    "b_lvh",
    "b_st_depression",
    "b_t_wave_abnormality",
    "b_mi_any",
)
REQUIRED_LEADS = {
    "complete_rbbb": ("V1",),
    "incomplete_rbbb": ("V1",),
    "lafb": ("I", "aVF"),
    "left_axis_deviation": ("I", "aVF"),
    "lvh": ("V1", "V5", "V6"),
    "first_degree_av_block": ("II",),
}


def contradiction_rows(table: pd.DataFrame) -> list[dict[str, object]]:
    """
    Item 3: label pairs that cannot both hold, positives under pacing and positives with a flat needed lead.

    Parameters
    ----------
    table : pd.DataFrame
        Output of ``analysis_table``.

    Returns
    -------
    list[dict[str, object]]
        Per pair (``CONTRADICTIONS``; LAFB positive with left axis deviation negative; paced rhythm with each
        label of ``UNREADABLE``; a flat lead of ``REQUIRED_LEADS``) and source: records where both hold.
    """
    pairs = [(a, b, reason, (table[a] == 1) & (table[b] == 1)) for a, b, reason in CONTRADICTIONS]
    pairs.append(
        (
            "b_lafb",
            "b_left_axis_deviation = 0",
            "LAFB needs an axis of -45 degrees or less, a left axis",
            (table["b_lafb"] == 1) & (table["b_left_axis_deviation"] == 0),
        )
    )
    pairs += [
        (
            "c_paced_rhythm",
            name,
            "a ventricular-paced QRS replaces intrinsic conduction",
            (table["c_paced_rhythm"] == 1) & (table[name] == 1),
        )
        for name in UNREADABLE
    ]
    for condition, leads in REQUIRED_LEADS.items():
        flat = table[[f"flat_{lead}" for lead in leads]].any(axis=1)
        pairs.append(
            (
                f"b_{condition}",
                f"flat lead among {'/'.join(leads)}",
                "the condition is read on these leads",
                (table[f"b_{condition}"] == 1) & flat,
            )
        )
    rows = []
    for first, second, reason, both in pairs:
        for source in SOURCES:
            rows.append(
                {
                    "first": first,
                    "second": second,
                    "reason": reason,
                    "source": source,
                    "train": int((both & (table["source"] == source) & (table["role"] == "train")).sum()),
                    "validation": int(
                        (both & (table["source"] == source) & (table["role"] == "validation")).sum()
                    ),
                }
            )
    return rows


MINIMUM_POSITIVE = ROOT / "outputs/mlc01_condition_dataset_20261110_v4/minimum_positive_rule.csv"
AMBIGUOUS_FAMILY = {
    **dict.fromkeys(
        mlc_labels.ST_T_NONSPECIFIC_OR_ISCHEMIC["scp"] | mlc_labels.ST_T_NONSPECIFIC_OR_ISCHEMIC["snomed"],
        "nonspecific or ischemic ST-T statement",
    ),
    **dict.fromkeys(("DIG", "ANEUR", "EL"), "digitalis, aneurysm or electrolyte statement"),
    **dict.fromkeys(mlc_labels.LAFB_CODES, "LAFB without a left-axis code"),
    "axis:LAD": "PTB-XL heart_axis LAD",
    "axis:AXL": "PTB-XL heart_axis empty or AXL",
    mlc_labels.AXIS_MISSING: "PTB-XL heart_axis empty or AXL",
    **dict.fromkeys(("VCLVH", "HVOLT", "55827005"), "left ventricular high voltage"),
    "266249003": "ventricular hypertrophy of unstated side",
    "370365005": "ventricular hypertrophy of unstated side",
    "233917008": "AV block of unstated degree",
    "426627000": "bradycardia of unstated origin",
    **dict.fromkeys((*mlc_labels.PREMATURE_UNKNOWN_SCP, "13640000"), "premature complex of unstated origin"),
    **dict.fromkeys(mlc_labels.VENTRICULAR_TACHYCARDIA_SNOMED, "ventricular tachycardia without a PVC code"),
    "6374002": "bundle branch block of unstated side",
    **dict.fromkeys(mlc_labels.INJURY_SCP, "subendocardial injury statement"),
    **dict.fromkeys(("QWAVE", "164917005", "365413008"), "Q wave or poor R-wave progression"),
}


def undefined_reason(condition: str, source: str, codes: object, minimum: frozenset[tuple[str, str]]) -> str:
    """
    Name the MLC-01 rule that makes one label undefined.

    The rules are tried in a fixed order: a source or vocabulary that does not annotate the condition, the
    minimum-positive rule (Amendment 1, ruling 6; it also flags every non-annotating source, which has no
    positive), a contradictory pair, then ambiguous codes.

    Parameters
    ----------
    condition : str
        Space-B condition.
    source : str
        Manifest source.
    codes : object
        Output of ``record_codes``.
    minimum : frozenset[tuple[str, str]]
        ``(source, b_<condition>)`` pairs flagged by the minimum-positive rule.

    Returns
    -------
    str
        ``minimum positives``, ``not annotated: <why>``, ``contradictory pair: <other>``, ``ambiguous:
        <codes>`` or ``unexplained``.
    """
    vocabulary_name = vocabulary(source)
    if not mlc_labels.annotated(condition, source):
        listed_elsewhere = (
            condition in mlc_labels.NOT_ANNOTATED and source in mlc_labels.NOT_ANNOTATED[condition]
        )
        return "not annotated: " + ("source list" if listed_elsewhere else "outside CPSC 2018 classes")
    if (source, f"b_{condition}") in minimum:
        return "minimum positives"
    for pair in mlc_labels.CONTRADICTORY_PAIRS:
        if condition in pair and all(
            mlc_labels.listed(mlc_labels.CONDITIONS[key]["positive"][vocabulary_name], codes, vocabulary_name)
            for key in pair
        ):
            return "contradictory pair: " + next(key for key in pair if key != condition)
    families = sorted(
        {
            AMBIGUOUS_FAMILY.get(code, code)
            for code in mlc_labels.CONDITIONS[condition]["ambiguous"][vocabulary_name]
            if mlc_labels.listed(frozenset({code}), codes, vocabulary_name)
        }
    )
    return "ambiguous: " + " + ".join(families) if families else "unexplained"


def minimum_positive_pairs(path: Path = MINIMUM_POSITIVE) -> frozenset[tuple[str, str]]:
    """
    Source and label pairs that the minimum-positive rule made undefined in v4.

    Parameters
    ----------
    path : Path
        ``minimum_positive_rule.csv`` of the v4 build.

    Returns
    -------
    frozenset[tuple[str, str]]
        ``(source, label)`` with ``not_annotated`` True.
    """
    table = pd.read_csv(path)
    flagged = table[table["not_annotated"].astype(str) == "True"]
    return frozenset(zip(flagged["source"], flagged["label"], strict=True))


def undefined_rows(table: pd.DataFrame, minimum: frozenset[tuple[str, str]]) -> pd.DataFrame:
    """
    Item 5: undefined entries by rule, with the share meeting the condition's measured definition.

    Parameters
    ----------
    table : pd.DataFrame
        Output of ``analysis_table`` with ``codes``.
    minimum : frozenset[tuple[str, str]]
        Output of ``minimum_positive_pairs``.

    Returns
    -------
    pd.DataFrame
        One row per condition, source and reason: undefined records (train, validation), records in the
        check's population, and the share meeting the definition (``criterion(...)["inside"]``) among them;
        the same share among the source's positives and negatives for comparison.
    """
    rows = []
    for condition in CONDITIONS:
        masks, spec = criterion(table, condition), CHECKS[condition]
        included = population(table, spec["population"])
        label = table[f"b_{condition}"]
        for source in SOURCES:
            in_source = table["source"] == source
            undefined = in_source & label.isna()
            if not undefined.any():
                continue
            reasons = pd.Series(
                [
                    undefined_reason(condition, source, codes, minimum)
                    for codes in table.loc[undefined, "codes"]
                ],
                index=table.index[undefined],
            )
            reference = {
                state: share(masks["inside"], in_source & included & (label == value))
                for state, value in (("positive", 1), ("negative", 0))
            }
            for reason, part in reasons.groupby(reasons):
                mask = table.index.isin(part.index)
                rows.append(
                    {
                        "condition": condition,
                        "source": source,
                        "reason": reason,
                        "train": int((mask & (table["role"] == "train")).sum()),
                        "validation": int((mask & (table["role"] == "validation")).sum()),
                        "in_population": int((mask & included).sum()),
                        "share_meeting_definition": share(
                            masks["inside"], pd.Series(mask, index=table.index) & included
                        ),
                        "share_positives": reference["positive"],
                        "share_negatives": reference["negative"],
                    }
                )
    return pd.DataFrame(rows)


def share(flags: pd.Series, mask: pd.Series) -> float:
    """
    Share of flagged records within a mask.

    Parameters
    ----------
    flags, mask : pd.Series
        Booleans over the same records.

    Returns
    -------
    float
        Flagged over masked records, NaN when the mask is empty.
    """
    total = int(mask.sum())
    return float((flags & mask).sum() / total) if total else np.nan


def group_folds(groups: np.ndarray, sources: np.ndarray, folds: int = 5, seed: int = SEED) -> np.ndarray:
    """
    Cross-validation folds grouped by patient (record for the Challenge sources) and balanced by source.

    Within each source, the unique groups are shuffled with one generator and dealt to the folds in turn, so
    that every record of a group lands in the same fold.

    Parameters
    ----------
    groups : np.ndarray
        ``group_id`` of each record.
    sources : np.ndarray
        Source of each record.
    folds : int
        Number of folds.
    seed : int
        Generator seed.

    Returns
    -------
    np.ndarray
        Fold index (0 to ``folds - 1``) of each record.

    Raises
    ------
    ValueError
        If a group spans two sources.
    """
    groups, sources = np.asarray(groups, dtype=str), np.asarray(sources, dtype=str)
    if pd.Series(sources).groupby(groups).nunique().max() > 1:
        raise ValueError("a group spans two sources")
    rng = np.random.default_rng(seed)
    assignment: dict[str, int] = {}
    for source in sorted(set(sources)):
        unique = np.unique(groups[sources == source])
        for position, group in enumerate(unique[rng.permutation(len(unique))]):
            assignment[group] = position % folds
    return np.array([assignment[group] for group in groups])


def fit_split(
    x: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray, c: float
) -> dict[str, object]:
    """
    P0's recipe for one condition: standardizer and unit rows from the training part, logistic regression.

    The recipe is MLC-02's (``mlc02_feature_models.fit_scaler``, ``logistic_inputs``, ``fit_logistic``):
    one standardizer on all training records, row L2 normalization, and an L2 logistic regression fitted on
    the training records where the condition is defined.

    Parameters
    ----------
    x : np.ndarray
        Pooled xECG features of all records.
    y : np.ndarray
        States of one condition (1.0, 0.0, NaN).
    train, test : np.ndarray
        Boolean masks.
    c : float
        Inverse regularization strength.

    Returns
    -------
    dict[str, object]
        ``probability`` of the test records, ``iterations`` and training ``records``.
    """
    from ecg_experiment.mlc02_feature_models import (
        fit_logistic,
        fit_scaler,
        logistic_inputs,
        logistic_probability,
    )

    scaler = fit_scaler(x[train])
    model = fit_logistic(logistic_inputs(x[train], scaler), y[train], c)
    probability = logistic_probability(
        logistic_inputs(x[test], scaler), model["coef"][None, :], np.array([model["intercept"]])
    )[:, 0]
    return {"probability": probability, "iterations": model["iterations"], "records": model["records"]}


def confident_learning(y: np.ndarray, p: np.ndarray) -> dict[str, object]:
    """
    Binary confident learning (Northcutt, Jiang and Chuang, 2021, J Artif Intell Res 70:1373-1411).

    Per-class thresholds are the mean predicted probability of the class among records labelled with it. A
    record is confidently counted as a class when its probability of that class reaches the class threshold
    (the larger probability wins when both do). The confident joint counts given label against confident
    class; each row is rescaled to the number of records with that given label (calibration), and the
    off-diagonal cells estimate the wrong labels. Assumptions: label noise depends on the true class only, not
    on the record (class-conditional noise), and the probabilities are out-of-sample and rank records well
    within each class. A model's errors are counted as label errors when these fail.

    Parameters
    ----------
    y : np.ndarray
        Given labels, 0 or 1, both present.
    p : np.ndarray
        Out-of-fold probability of the positive class.

    Returns
    -------
    dict[str, object]
        ``threshold_positive``, ``threshold_negative``, ``confident_joint`` and ``calibrated_joint`` (rows:
        given 0, 1; columns: confident 0, 1), ``estimated_false_positives`` (given 1, confidently 0, after
        calibration), ``estimated_false_negatives``, and the masks ``flag_false_positive`` and
        ``flag_false_negative`` of the off-diagonal records.

    Raises
    ------
    ValueError
        If ``y`` holds a single class.
    """
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    if y.min() == y.max():
        raise ValueError("confident learning needs both classes")
    t_positive, t_negative = float(p[y == 1].mean()), float((1 - p[y == 0]).mean())
    above_positive, above_negative = p >= t_positive, (1 - p) >= t_negative
    confident = np.full(len(y), -1)
    confident[above_positive & ~above_negative] = 1
    confident[above_negative & ~above_positive] = 0
    both = above_positive & above_negative
    confident[both] = (p[both] >= 0.5).astype(int)
    joint = np.array(
        [[np.sum((y == given) & (confident == true)) for true in (0, 1)] for given in (0, 1)], dtype=float
    )
    counts = np.array([np.sum(y == 0), np.sum(y == 1)], dtype=float)
    rows = joint.sum(axis=1, keepdims=True)
    calibrated = np.divide(joint, rows, out=np.full_like(joint, np.nan), where=rows > 0) * counts[:, None]
    return {
        "threshold_positive": t_positive,
        "threshold_negative": t_negative,
        "confident_joint": joint,
        "calibrated_joint": calibrated,
        "estimated_false_positives": float(calibrated[1, 0]),
        "estimated_false_negatives": float(calibrated[0, 1]),
        "flag_false_positive": (y == 1) & (confident == 0),
        "flag_false_negative": (y == 0) & (confident == 1),
    }


def suspects(y: np.ndarray, p: np.ndarray, count: int = 50) -> dict[str, np.ndarray]:
    """
    Positions of the most suspected false positives and false negatives by self-confidence.

    Parameters
    ----------
    y : np.ndarray
        Given labels, 0 or 1.
    p : np.ndarray
        Out-of-fold probability of the positive class.
    count : int
        Records per list.

    Returns
    -------
    dict[str, np.ndarray]
        ``false_positive``: positives with the lowest probability, lowest first; ``false_negative``: negatives
        with the highest probability, highest first. Ties keep record order.
    """
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    positives, negatives = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return {
        "false_positive": positives[np.argsort(p[positives], kind="stable")][:count],
        "false_negative": negatives[np.argsort(-p[negatives], kind="stable")][:count],
    }


def reading_list(listed: pd.DataFrame, per_kind: int = 12) -> pd.DataFrame:
    """
    Sheets to read by eye: per condition and kind, the sources in turn by rank.

    Parameters
    ----------
    listed : pd.DataFrame
        Suspect rows with ``condition``, ``kind``, ``source``, ``rank`` and ``record_key``.
    per_kind : int
        Sheets per condition and kind.

    Returns
    -------
    pd.DataFrame
        Rank 1 of every source in ``SOURCES`` order, then rank 2, and so on, cut at ``per_kind`` rows per
        condition and kind.
    """
    order = {source: position for position, source in enumerate(SOURCES)}
    frame = listed[["condition", "kind", "source", "rank", "record_key"]].copy()
    frame["_source"] = frame["source"].map(order)
    parts = []
    for _, part in frame.groupby(["condition", "kind"], sort=False):
        parts.append(part.sort_values(["rank", "_source"], kind="stable").head(per_kind))
    return pd.concat(parts).drop(columns="_source").reset_index(drop=True)


RHYTHM_STATEMENTS = {
    "scp": frozenset({"SR", "STACH", "SBRAD", "SARRH", "BIGU", "TRIGU"}) | NON_SINUS["scp"],
    "snomed": frozenset({"426783006", "426177001", "427084000", "427393009", "426627000"})
    | NON_SINUS["snomed"],
}


def rhythm_stated(table: pd.DataFrame) -> pd.Series:
    """
    Whether a record's annotation states a rhythm at all.

    Parameters
    ----------
    table : pd.DataFrame
        Rows with ``source`` and ``raw_codes``.

    Returns
    -------
    pd.Series
        True when a PTB-XL rhythm statement (``SR``, ``STACH``, ``SBRAD``, ``SARRH``, ``BIGU``, ``TRIGU`` or a
        ``NON_SINUS`` statement) or a SNOMED sinus, bradycardia or ``NON_SINUS`` code is present, whatever its
        PTB-XL likelihood.
    """
    stated = []
    for source, raw in zip(table["source"], table["raw_codes"], strict=True):
        if source == "ptbxl":
            stated.append(bool(set(ast.literal_eval(raw)) & RHYTHM_STATEMENTS["scp"]))
        else:
            stated.append(bool({code for code in raw.split(",") if code} & RHYTHM_STATEMENTS["snomed"]))
    return pd.Series(stated, index=table.index)


GRID_ROWS = (("I", "aVR", "V1", "V4"), ("II", "aVL", "V2", "V5"), ("III", "aVF", "V3", "V6"))


def plot_review(signal: np.ndarray, heading: str, lines: list[str], path: Path) -> None:
    """
    Plot one record as a standard twelve-lead sheet with a lead II rhythm strip.

    Parameters
    ----------
    signal : np.ndarray
        ``[12, 5000]`` in mV at 500 Hz, ``LEADS`` order.
    heading : str
        First title line.
    lines : list[str]
        Further text lines (label, probability, measurements).
    path : Path
        PNG to write; parent directories are created.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    signal = np.nan_to_num(np.asarray(signal, dtype=float))
    figure, axis = plt.subplots(figsize=(14, 8.6), dpi=100)
    time = np.arange(signal.shape[1]) / 500.0
    for row, leads in enumerate(GRID_ROWS):
        for column, lead in enumerate(leads):
            part = slice(column * 1250, (column + 1) * 1250)
            trace = signal[LEADS.index(lead), part]
            axis.plot(time[part], trace - np.median(trace) - 3.0 * row, color="black", linewidth=0.6)
            axis.text(column * 2.5 + 0.05, -3.0 * row + 0.9, lead, fontsize=10, color="navy")
    rhythm = signal[LEADS.index("II")]
    axis.plot(time, rhythm - np.median(rhythm) - 9.0, color="black", linewidth=0.6)
    axis.text(0.05, -9.0 + 0.9, "II", fontsize=10, color="navy")
    axis.set_xticks(np.arange(0, 10.01, 0.2), minor=False)
    axis.set_yticks(np.arange(-11.0, 2.01, 0.5))
    axis.grid(True, color="#f2a0a0", linewidth=0.5)
    axis.set(xlim=(0, 10), ylim=(-11.0, 2.0), xticklabels=[], yticklabels=[])
    axis.set_title("\n".join([heading, *lines]), fontsize=9, loc="left")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


CACHE_STATE: dict[str, object] = {}


def cached_windows(keys: np.ndarray) -> np.ndarray:
    """
    Read the cached first-10-s windows of open records.

    The MLC-02 cache index is opened once per process; every requested key must belong to an open role.

    Parameters
    ----------
    keys : np.ndarray
        Record keys.

    Returns
    -------
    np.ndarray
        Float32 ``[len(keys), 12, 5000]`` in mV.

    Raises
    ------
    PermissionError
        If a key belongs to a closed role in the cache index.
    """
    from ecg_experiment.mlc02_waveforms import load_cache, windows

    if "index" not in CACHE_STATE:
        index, blocks = load_cache(index_name="index_all.csv")
        CACHE_STATE.update(
            index=index,
            blocks=blocks,
            position=pd.Series(index["position"].to_numpy(), index=index["record_key"].to_numpy()),
        )
    index = CACHE_STATE["index"]
    positions = CACHE_STATE["position"].loc[np.asarray(keys)].to_numpy()
    check_open_roles(tuple(sorted(set(index["role"].to_numpy()[positions]))))
    return windows(index, CACHE_STATE["blocks"], positions)


def measure_keys(keys: list[str]) -> list[dict[str, object]]:
    """
    Run method B on the cached windows of some open records.

    Parameters
    ----------
    keys : list[str]
        Record keys.

    Returns
    -------
    list[dict[str, object]]
        ``record_key`` and ``measure_record``'s values, one dictionary per key.
    """
    from ecg_experiment.mlc03_measurements import measure_record

    signals = cached_windows(np.asarray(keys))
    return [{"record_key": key, **measure_record(signal)} for key, signal in zip(keys, signals, strict=True)]


PLUS_COLUMNS = {
    "12sl": (
        "QRS_Dur_Global",
        "PR_Int_Global",
        "HR_Ventr_Global",
        "R_AxisFrontal_Global",
        "P_Found_Global",
        "R_Amp_V5",
        "R_Amp_V6",
        "S_Amp_V1",
        "R_Amp_aVL",
        "S_Amp_V3",
        "Rp_Amp_V1",
        "ST_Amp_V5",
        "T_Amp_V5",
        "T_Amp_II",
        "Q_Dur_II",
        "Q_Dur_V5",
    ),
    "unig": (
        "QRS_Dur_Global",
        "PR_Int_Global",
        "HR__Global",
        "QRS_AxisFront_Global",
        "P_Found_Global",
        "R_Amp_V5",
        "R_Amp_V6",
        "S_Amp_V1",
        "R_Amp_aVL",
        "S_Amp_V3",
        "Rp_Amp_V1",
        "ST_Amp80ms_V5",
        "T+_Amp_V5",
        "T-_Amp_V5",
        "T+_Amp_II",
        "T-_Amp_II",
        "Q_Dur_II",
        "Q_Dur_V5",
    ),
}


def load_ptbxl_plus(keys: np.ndarray, directory: Path = PTBXL_PLUS) -> pd.DataFrame:
    """
    PTB-XL+ 12SL and Uni-G measurements of open PTB-XL records.

    Rows of other records (including PTB-XL fold 10) are skipped by the reader.

    Parameters
    ----------
    keys : np.ndarray
        Open record keys; non-PTB-XL keys are ignored.
    directory : Path
        PTB-XL+ ``features`` directory.

    Returns
    -------
    pd.DataFrame
        ``record_key`` and ``<column>_<program>`` for ``PLUS_COLUMNS``.
    """
    wanted = {int(key.split(":")[1]) for key in np.asarray(keys, dtype=str) if key.startswith("ptbxl:")}
    tables = []
    for program, columns in PLUS_COLUMNS.items():
        path = Path(directory) / f"{program}_features.csv"
        ids = pd.read_csv(path, usecols=["ecg_id"])["ecg_id"]
        lines = set((np.flatnonzero(ids.isin(wanted).to_numpy()) + 1).tolist())
        table = pd.read_csv(
            path, usecols=["ecg_id", *columns], skiprows=lambda i, keep=lines: i > 0 and i not in keep
        )
        tables.append(table.rename(columns={c: f"{c}_{program}" for c in columns}).set_index("ecg_id"))
    joined = tables[0].join(tables[1], how="outer")
    joined.insert(0, "record_key", "ptbxl:" + joined.index.astype(str))
    return joined.reset_index(drop=True)


def error_summary(
    estimate: pd.Series, reference: pd.Series, tolerance: float, circular: bool = False
) -> dict[str, float]:
    """
    Agreement of a measurement with a reference.

    Parameters
    ----------
    estimate, reference : pd.Series
        Aligned values; NaN means not measured.
    tolerance : float
        Absolute tolerance for the ``within`` share.
    circular : bool
        Angles in degrees.

    Returns
    -------
    dict[str, float]
        ``pairs`` with a reference, ``measured`` share of them, ``bias`` (median difference), ``median_abs``,
        ``p90_abs`` and ``within`` (share of measured pairs within the tolerance).
    """
    has_reference = reference.notna()
    both = has_reference & estimate.notna()
    difference = (
        angle_difference(estimate[both], reference[both])
        if circular
        else (estimate[both] - reference[both]).to_numpy(float)
    )
    absolute = np.abs(difference)
    return {
        "pairs": int(has_reference.sum()),
        "measured": float(both.sum() / max(has_reference.sum(), 1)),
        "bias": float(np.median(difference)) if len(difference) else np.nan,
        "median_abs": float(np.median(absolute)) if len(absolute) else np.nan,
        "p90_abs": float(np.percentile(absolute, 90)) if len(absolute) else np.nan,
        "within": float((absolute <= tolerance).mean()) if len(absolute) else np.nan,
    }


def open_window_guard(table: pd.DataFrame) -> None:
    """
    Set the MLC-02 closed-data guard in a reader process: open record paths allowed, closed groups refused.

    Parameters
    ----------
    table : pd.DataFrame
        Open records with ``raw_path``.
    """
    from ecg_experiment.mlc02_waveforms import closed_keys, init_worker

    init_worker(closed_keys(), frozenset(table["raw_path"]))


def whole_record_windows(job: tuple[str, str, str]) -> dict[str, object]:
    """
    Run method B on every complete 10 s window of a whole recording.

    The reader is MLC-01's (``mlc01_dataset.read_signal``) behind the closed-data guard
    (``mlc02_waveforms.guard_path``), which must be set in the process first (``open_window_guard``).

    Parameters
    ----------
    job : tuple[str, str, str]
        Record key, manifest ``raw_path`` and source.

    Returns
    -------
    dict[str, object]
        ``record_key``, ``windows``, and per window lists of ``n_premature_aberrant``, ``af_pattern``
        (irregular RR without a detectable P wave) and ``hr_bpm``, with the first window's values apart.
    """
    from ecg_experiment import mlc01_dataset
    from ecg_experiment.eda.signals import canonical_order
    from ecg_experiment.mlc02_waveforms import guard_path
    from ecg_experiment.mlc03_measurements import measure_record

    key, raw_path, source = job
    signal, fs, names = mlc01_dataset.read_signal(
        {"format": "wfdb", "stem": str(guard_path(raw_path, source))}
    )
    if fs != 500:
        raise ValueError(f"{key} is not 500 Hz")
    signal = canonical_order(signal, names).T
    results = [
        measure_record(signal[:, start : start + 5000]) for start in range(0, signal.shape[1] - 4999, 5000)
    ]
    ectopy = [
        float(r.get("n_premature_aberrant", np.nan)) + float(r.get("n_premature_aberrant_edge", 0.0))
        for r in results
    ]
    af = [
        bool(
            r.get("rr_nrmssd", 0.0) >= IRREGULAR_NRMSSD
            and r.get("status") == "ok"
            and not (r.get("p_snr", 0.0) >= P_SNR_MIN and r.get("p_amp_mv", 0.0) >= P_AMP_MIN)
        )
        for r in results
    ]
    return {
        "record_key": key,
        "windows": len(results),
        "ectopy_by_window": ectopy,
        "af_by_window": af,
        "hr_by_window": [float(r.get("hr_bpm", np.nan)) for r in results],
        "ectopy_first": ectopy[0],
        "ectopy_later": float(np.nansum(ectopy[1:])) if len(ectopy) > 1 else 0.0,
        "af_first": af[0],
        "af_later_windows": int(sum(af[1:])),
    }


def rule_counts(table: pd.DataFrame, rule: str, condition: str, mask: pd.Series) -> list[dict[str, object]]:
    """
    Label entries a candidate rule would set undefined, by source, role and current state.

    Only defined entries count; a rule never creates or flips a positive.

    Parameters
    ----------
    table : pd.DataFrame
        Analysis table.
    rule : str
        Rule identifier.
    condition : str
        Space-B condition the rule acts on.
    mask : pd.Series
        Records the rule selects.

    Returns
    -------
    list[dict[str, object]]
        One row per source and role with the positives and negatives that would become undefined, beside the
        defined entries there.
    """
    label = table[f"b_{condition}"]
    rows = []
    for (source, role), part in table.groupby(["source", "role"], sort=False):
        selected = mask.loc[part.index] & label.loc[part.index].notna()
        rows.append(
            {
                "rule": rule,
                "condition": condition,
                "source": source,
                "role": role,
                "positives_to_undefined": int((selected & (label.loc[part.index] == 1)).sum()),
                "negatives_to_undefined": int((selected & (label.loc[part.index] == 0)).sum()),
                "positives": int((label.loc[part.index] == 1).sum()),
                "negatives": int((label.loc[part.index] == 0).sum()),
            }
        )
    return rows
