"""
Review sheets of MLC-03 Part A: plots of 12-lead ECGs for the visual check of every flag.

Each sheet holds up to ``columns * rows`` records. A record panel shows the twelve leads of the cached first
10 s, one row per lead, with its own vertical range in mV (tick labels at the ends) and a shared time axis in
seconds. The title gives the record key, source, role and the measure that raised the flag. Leads that
raised the flag are drawn in red; marked time spans are shaded; an optional inset shows one lead over a
short span at higher gain (used where the defect is small, such as mains interference). These are review
sheets, not publication figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.gridspec import GridSpecFromSubplotSpec  # noqa: E402

from ecg_experiment.eda.signals import LEADS  # noqa: E402
from ecg_experiment.mlc03_signal_quality import FS  # noqa: E402

SHEET_INCHES = (19.5, 19.0)
SHEET_DPI = 80
MIN_SPAN_MV = 0.4


def lead_limits(values: np.ndarray) -> tuple[float, float]:
    """
    Vertical range of one lead: the 0.2 and 99.8 percentiles of its finite samples, at least 0.4 mV wide.

    Parameters
    ----------
    values : np.ndarray
        Samples of one lead in mV.

    Returns
    -------
    tuple[float, float]
        Lower and upper limit.
    """
    finite = values[np.isfinite(values)]
    if not len(finite):
        return -0.2, 0.2
    low, high = np.percentile(finite, [0.2, 99.8])
    centre, half = (low + high) / 2, max(high - low, MIN_SPAN_MV) / 2 * 1.1
    return centre - half, centre + half


def draw_record(figure: plt.Figure, cell: Any, record: dict[str, Any]) -> None:
    """
    Draw one record panel into a cell of the sheet grid.

    Parameters
    ----------
    figure : plt.Figure
        Sheet figure.
    cell : Any
        A ``GridSpec`` cell.
    record : dict[str, Any]
        ``window`` (float ``(12, 5000)`` mV), ``title`` (text), and optionally ``highlight`` (lead names),
        ``spans`` (list of ``(lead or None, start_s, stop_s)`` to shade; None shades every lead),
        ``overlays`` (list of ``(lead, values)`` drawn dashed in grey) and ``inset``
        (``(lead, start_s, stop_s)``).
    """
    inset = record.get("inset")
    rows = len(LEADS) + (3 if inset else 0)
    grid = GridSpecFromSubplotSpec(rows, 1, subplot_spec=cell, hspace=0.0)
    window = record["window"]
    time = np.arange(window.shape[1]) / FS
    highlight = set(record.get("highlight", ()))
    overlays = record.get("overlays", [])
    axes = []
    for position, lead in enumerate(LEADS):
        axis = figure.add_subplot(grid[position, 0], sharex=axes[0] if axes else None)
        axes.append(axis)
        values = window[position]
        axis.plot(time, values, lw=0.5, color="crimson" if lead in highlight else "black")
        for name, extra in overlays:
            if name == lead:
                axis.plot(time, extra, lw=0.6, color="tab:blue", ls="--", alpha=0.8)
        extras = [extra for name, extra in overlays if name == lead]
        low, high = lead_limits(np.concatenate([values, *extras]))
        axis.set_ylim(low, high)
        axis.set_yticks([low, high])
        axis.set_yticklabels([f"{low:.1f}", f"{high:.1f}"], fontsize=5)
        axis.text(0.002, 0.5, lead, transform=axis.transAxes, fontsize=7, va="center",
                  color="crimson" if lead in highlight else "black", fontweight="bold")
        for span_lead, start, stop in record.get("spans", []):
            if span_lead is None or span_lead == lead:
                axis.axvspan(start, stop, color="gold", alpha=0.35, lw=0)
        axis.tick_params(axis="x", labelbottom=False, length=2)
        axis.set_xlim(0, time[-1])
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    axes[-1].tick_params(axis="x", labelbottom=True, labelsize=6)
    axes[-1].set_xticks(np.arange(0, 11, 1))
    axes[0].set_title(record["title"], fontsize=8, loc="left")
    if inset:
        lead, start, stop = inset
        axis = figure.add_subplot(grid[len(LEADS) + 1 :, 0])
        part = (time >= start) & (time <= stop)
        values = window[LEADS.index(lead), part]
        axis.plot(time[part], values, lw=0.6, color="crimson")
        axis.set_title(f"{lead}, {start:.2f}-{stop:.2f} s (mV)", fontsize=6, loc="left")
        axis.tick_params(labelsize=5, length=2)


def draw_sheet(records: list[dict[str, Any]], path: Path, title: str, columns: int = 2,
               rows: int = 3) -> None:
    """
    Draw up to ``columns * rows`` record panels on one PNG sheet.

    Parameters
    ----------
    records : list[dict[str, Any]]
        Inputs of ``draw_record``.
    path : Path
        Destination PNG.
    title : str
        Sheet title.
    columns, rows : int
        Grid of record panels.
    """
    figure = plt.figure(figsize=SHEET_INCHES)
    grid = figure.add_gridspec(rows, columns, hspace=0.18, wspace=0.08, left=0.03, right=0.995, top=0.96,
                               bottom=0.02)
    for number, record in enumerate(records[: columns * rows]):
        draw_record(figure, grid[number // columns, number % columns], record)
    figure.suptitle(title, fontsize=10, x=0.01, ha="left")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=SHEET_DPI)
    plt.close(figure)
