"""
Measurements and review sheets of KEY-05 (validation records only).

Measurements come from MLC-03 Part B's ``analysis.parquet`` (train and validation records). Rate, QRS, PR and
axis are the consensus of methods A and B (NaN where the methods disagree); the voltages, ST level, T
amplitude and Q duration are method B's and count only where method B's status is ``ok``. Part B found PR and
RR irregularity unreliable; they are reported but marked so.

Review sheets show, for one condition, the 12 highest-scoring validation negatives or the 12 lowest-scoring
positives of a construction. Signals are the cached first 10 s read through ``mlc03_label_quality``'s guard,
which refuses any record outside train and validation. The reading of the sheets is non-clinical.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ecg_experiment import ROOT
from ecg_experiment.mlc03_label_quality import GROUPS_Q
from ecg_experiment.mlc03_measurements import LEADS, UPRIGHT_T

ANALYSIS = ROOT / "outputs/mlc03_data_quality_20261111/labels/analysis.parquet"
MEASURES = {
    "sinus_tachycardia": ("hr",),
    "sinus_bradycardia": ("hr",),
    "sinus_arrhythmia": ("rr_range_dominant_b",),
    "af_or_flutter": ("rr_nrmssd_b",),
    "first_degree_av_block": ("pr",),
    "complete_rbbb": ("qrs",),
    "incomplete_rbbb": ("qrs",),
    "lafb": ("axis",),
    "left_axis_deviation": ("axis",),
    "lvh": ("sokolow_mv", "cornell_mv"),
    "st_depression": ("st_min_mv", "t_min_mv"),
    "t_wave_abnormality": ("st_min_mv", "t_min_mv"),
    "mi_any": ("q_max_ms",),
}
UNRELIABLE = frozenset({"pr", "rr_nrmssd_b", "rr_range_dominant_b"})
CAPTION = (
    "hr",
    "qrs",
    "pr",
    "axis",
    "sokolow_mv",
    "cornell_mv",
    "st_min_mv",
    "t_min_mv",
    "q_max_ms",
    "rr_nrmssd_b",
)
SHEET_LEADS = ("I", "II", "V1", "V5")
SHEET_SIZE = 12


def measurement_table(keys: np.ndarray, path: Path = ANALYSIS) -> pd.DataFrame:
    """
    Measurements of validation or train records, indexed by record key.

    Parameters
    ----------
    keys : np.ndarray
        Record keys; each must be a train or validation record of the Part B table.
    path : Path
        Part B ``analysis.parquet``.

    Returns
    -------
    pd.DataFrame
        ``hr``, ``qrs``, ``pr``, ``axis`` (consensus), ``sokolow_mv``, ``cornell_mv``, ``st_min_mv``,
        ``t_min_mv`` (lowest T of the upright-T leads), ``q_max_ms`` (longest Q of the Q-group leads),
        ``rr_nrmssd_b``, ``rr_range_dominant_b``, ``status_b`` and ``n_flat_leads``, in the order of ``keys``.

    Raises
    ------
    KeyError
        If a key is not in the table.
    """
    t_columns = [f"t_{lead}" for lead in UPRIGHT_T]
    q_leads = sorted({lead for members in GROUPS_Q.values() for lead in members}, key=LEADS.index)
    q_columns = [f"qdur_{lead}" for lead in q_leads]
    columns = [
        "record_key",
        "hr",
        "qrs",
        "pr",
        "axis",
        "sokolow_mv",
        "cornell_mv",
        "st_min_mv",
        "rr_nrmssd_b",
        "rr_range_dominant_b",
        "status_b",
        "n_flat_leads",
        *t_columns,
        *q_columns,
    ]
    table = pd.read_parquet(path, columns=columns).set_index("record_key")
    table = table.loc[np.asarray(keys)]
    ok = table["status_b"] == "ok"
    out = table[
        [
            "hr",
            "qrs",
            "pr",
            "axis",
            "sokolow_mv",
            "cornell_mv",
            "st_min_mv",
            "rr_nrmssd_b",
            "rr_range_dominant_b",
            "status_b",
            "n_flat_leads",
        ]
    ].copy()
    out["t_min_mv"] = table[t_columns].min(axis=1, skipna=False).where(ok)
    out["q_max_ms"] = table[q_columns].max(axis=1, skipna=False).where(ok)
    for name in ("sokolow_mv", "cornell_mv", "st_min_mv", "rr_nrmssd_b", "rr_range_dominant_b"):
        out[name] = out[name].where(ok)
    return out


def review_positions(y: np.ndarray, score: np.ndarray, count: int = SHEET_SIZE) -> dict[str, np.ndarray]:
    """
    Positions of the highest-scoring negatives and the lowest-scoring positives.

    Parameters
    ----------
    y : np.ndarray
        States of one condition.
    score : np.ndarray
        Scores of the same records.
    count : int
        Records per kind.

    Returns
    -------
    dict[str, np.ndarray]
        ``top_negatives`` (highest score first) and ``bottom_positives`` (lowest score first).
    """
    negatives, positives = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    top = negatives[np.argsort(-score[negatives], kind="stable")[:count]]
    bottom = positives[np.argsort(score[positives], kind="stable")[:count]]
    return {"top_negatives": top, "bottom_positives": bottom}


def caption(row: pd.Series) -> str:
    """
    One line of measurements for a sheet panel.

    Parameters
    ----------
    row : pd.Series
        A ``measurement_table`` row.

    Returns
    -------
    str
        Short text; ``-`` where a value is missing.
    """
    names = {
        "hr": "HR",
        "qrs": "QRS",
        "pr": "PR*",
        "axis": "axis",
        "sokolow_mv": "SL",
        "cornell_mv": "Corn",
        "st_min_mv": "STmin",
        "t_min_mv": "Tmin",
        "q_max_ms": "Qmax",
        "rr_nrmssd_b": "RRirr*",
    }
    parts = []
    for name in CAPTION:
        value = row[name]
        parts.append(f"{names[name]} {'-' if pd.isna(value) else f'{value:.3g}'}")
    return " | ".join(parts)


def plot_sheet(signals: np.ndarray, titles: list[str], heading: str, path: Path) -> None:
    """
    Plot up to 12 records, each as leads I, II, V1 and V5 over the first 10 s.

    Parameters
    ----------
    signals : np.ndarray
        ``[records, 12, 5000]`` in mV at 500 Hz.
    titles : list[str]
        Two-line caption per record.
    heading : str
        Sheet title.
    path : Path
        PNG to write.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(6, 2, figsize=(22, 26), dpi=70)
    time = np.arange(signals.shape[2]) / 500.0
    for panel, axis in enumerate(axes.ravel()):
        axis.set_axis_off()
        if panel >= len(signals):
            continue
        signal = np.nan_to_num(np.asarray(signals[panel], dtype=float))
        for row, lead in enumerate(SHEET_LEADS):
            trace = signal[LEADS.index(lead)]
            axis.plot(time, trace - np.median(trace) - 2.5 * row, color="black", linewidth=0.6)
            axis.text(0.02, -2.5 * row + 0.8, lead, fontsize=10, color="navy")
        axis.set(xlim=(0, 10), ylim=(-2.5 * len(SHEET_LEADS) + 0.5, 1.8))
        axis.set_title(titles[panel], fontsize=10, loc="left")
    figure.suptitle(heading, fontsize=14, x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
