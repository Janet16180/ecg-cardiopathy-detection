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
import pandas as pd  # noqa: E402
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


def draw_inset(axis: plt.Axes, window: np.ndarray, time: np.ndarray, inset: tuple) -> None:
    """
    Draw the inset of a record panel.

    Parameters
    ----------
    axis : plt.Axes
        Inset axes.
    window : np.ndarray
        Float ``(12, samples)`` mV.
    time : np.ndarray
        Seconds of each sample.
    inset : tuple
        ``(lead, start_s, stop_s)``, ``((lead, ...), (start_s, stop_s))`` or ``(label, values)``.
    """
    if isinstance(inset[1], np.ndarray):
        label, values = inset
        axis.plot(time, values, lw=0.6, color="crimson")
        axis.set_title(label, fontsize=6, loc="left")
    elif isinstance(inset[0], tuple):
        leads, (start, stop) = inset
        part = (time >= start) & (time <= stop)
        offset = 0.0
        for lead in leads:
            values = window[LEADS.index(lead), part]
            values = values - np.nanmedian(values)
            axis.plot(time[part], values + offset, lw=0.7)
            axis.text(time[part][0], offset, f" {lead}", fontsize=6, va="bottom")
            offset -= 1.1 * max(np.nanmax(values) - np.nanmin(values), 0.3)
        axis.set_title(f"{', '.join(leads)} stacked, {start:.1f}-{stop:.1f} s (mV)", fontsize=6, loc="left")
    else:
        lead, start, stop = inset
        part = (time >= start) & (time <= stop)
        axis.plot(time[part], window[LEADS.index(lead), part], lw=0.6, color="crimson")
        axis.set_title(f"{lead}, {start:.2f}-{stop:.2f} s (mV)", fontsize=6, loc="left")
    axis.tick_params(labelsize=5, length=2)


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
        (``(lead, start_s, stop_s)`` for a zoom on one lead, ``((lead, ...), (start_s, stop_s))`` for several
        leads stacked, or ``(label, values)`` for a whole-record trace such as a residual).
    """
    inset = record.get("inset")
    rows = len(LEADS) + ((6 if isinstance(inset[0], tuple) else 3) if inset else 0)
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
        draw_inset(figure.add_subplot(grid[len(LEADS) + 1 :, 0]), window, time, inset)


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


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """
    Start and stop (exclusive) of every run of True values.

    Parameters
    ----------
    mask : np.ndarray
        One-dimensional boolean array.

    Returns
    -------
    list[tuple[int, int]]
        Runs in order.
    """
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False])).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def worst_lead(measures: pd.Series, name: str, leads: list[str], largest: bool = True) -> str:
    """
    Lead with the largest (or smallest) value of a per-lead measure among some leads.

    Parameters
    ----------
    measures : pd.Series
        One row of the measures table.
    name : str
        Measure name before ``__``.
    leads : list[str]
        Candidate leads (non-empty).
    largest : bool
        True picks the largest value.

    Returns
    -------
    str
        Lead name.
    """
    values = [float(measures[f"{name}__{lead}"]) for lead in leads]
    return leads[int(np.nanargmax(values) if largest else np.nanargmin(values))]


def step_time(window: np.ndarray, lead: str) -> float:
    """
    Time of the largest interior baseline step of a lead, by the rule of ``step_size``.

    Parameters
    ----------
    window : np.ndarray
        Float ``(12, 5000)`` mV.
    lead : str
        Lead name.

    Returns
    -------
    float
        Seconds.
    """
    from scipy.ndimage import median_filter

    from ecg_experiment.mlc03_signal_quality import BLOCK, EDGE, STEP_LAG, STEP_MEDIANS, fill_missing

    blocks = fill_missing(window)[LEADS.index(lead)].reshape(-1, BLOCK).mean(axis=1)
    baseline = median_filter(median_filter(blocks, STEP_MEDIANS[0], mode="nearest"), STEP_MEDIANS[1],
                             mode="nearest")
    change = np.abs(baseline[2 * STEP_LAG :] - baseline[: -2 * STEP_LAG])
    centre = np.arange(len(change)) + STEP_LAG
    change[(centre < EDGE // BLOCK) | (centre >= (window.shape[1] - EDGE) // BLOCK)] = 0.0
    return float(centre[int(np.argmax(change))] * BLOCK / FS)


def spike_time(window: np.ndarray, lead: str) -> float:
    """
    Time of the largest isolated spike of an independent lead, by the rule of ``isolated_spikes``.

    Parameters
    ----------
    window : np.ndarray
        Float ``(12, 5000)`` mV.
    lead : str
        Independent lead name.

    Returns
    -------
    float
        Seconds.
    """
    from scipy.ndimage import maximum_filter1d

    from ecg_experiment.mlc03_signal_quality import (
        INDEPENDENT,
        SPIKE_BIG_MV,
        SPIKE_COINCIDENCE,
        SPIKE_SPREAD,
        fill_missing,
    )

    rows = [LEADS.index(name) for name in INDEPENDENT]
    second = np.abs(np.diff(fill_missing(window)[rows], n=2, axis=1))
    big = maximum_filter1d((second >= SPIKE_BIG_MV).astype(np.int8), SPIKE_SPREAD, axis=1)
    values = np.where(big.sum(axis=0) <= SPIKE_COINCIDENCE, second[INDEPENDENT.index(lead)], 0.0)
    return float((int(np.argmax(values)) + 1) / FS)


def spans_where(window: np.ndarray,
                condition: np.ndarray) -> tuple[list[tuple[str, float, float]], list[str]]:
    """
    Shaded spans and leads where a per-sample condition holds.

    Parameters
    ----------
    window : np.ndarray
        Float ``(12, 5000)`` mV.
    condition : np.ndarray
        Boolean array of the same shape.

    Returns
    -------
    tuple[list[tuple[str, float, float]], list[str]]
        Spans (widened by 50 ms) and the leads that hold any.
    """
    spans, leads = [], []
    for position, lead in enumerate(LEADS):
        for start, stop in runs_of(condition[position]):
            spans.append((lead, start / FS - 0.05, stop / FS + 0.05))
            leads.append(lead)
    return spans, list(dict.fromkeys(leads))


def note_missing(window: np.ndarray, measures: pd.Series, flags: pd.Series,
        blamed: list[str]) -> dict[str, Any]:
    """Mark missing samples (see ``annotate``)."""
    spans, leads = spans_where(window, np.isnan(window))
    return {"spans": spans, "highlight": leads,
            "value": f"{int(measures['missing_samples'])} missing samples"}


def note_edge(window: np.ndarray, measures: pd.Series, flags: pd.Series,
        blamed: list[str]) -> dict[str, Any]:
    """Mark the all-lead hold at both edges (see ``annotate``)."""
    hold, end = float(flags["edge_hold_ms"]) / 1000.0, window.shape[1] / FS
    return {"spans": [(None, 0.0, hold), (None, end - hold, end)],
            "value": f"all-lead hold {flags['edge_hold_ms']:.0f} ms"}


def note_rail(window: np.ndarray, measures: pd.Series, flags: pd.Series,
        blamed: list[str]) -> dict[str, Any]:
    """Mark samples at the converter rail (see ``annotate``)."""
    spans, leads = spans_where(window, np.abs(np.nan_to_num(window)) >= 32.7)
    return {"spans": spans, "highlight": leads,
            "value": f"{int(measures['rail_samples'])} samples at the rail"}


def note_small_lead(window: np.ndarray, measures: pd.Series, flags: pd.Series, blamed: list[str]
                    ) -> dict[str, Any]:
    """Show the blamed flat or nearly flat lead (see ``annotate``)."""
    lead = blamed[0]
    return {"inset": (lead, 0.0, 3.0), "value": f"{lead} range {measures[f'amp_robust__{lead}']:.3f} mV"}


def note_identity(window: np.ndarray, measures: pd.Series, flags: pd.Series, blamed: list[str]
                  ) -> dict[str, Any]:
    """Overlay the recomputed derived leads and mark the largest residual (see ``annotate``)."""
    lead_i, lead_ii = window[0], window[1]
    expected = {"III": lead_ii - lead_i, "aVR": -(lead_i + lead_ii) / 2, "aVL": lead_i - lead_ii / 2,
                "aVF": lead_ii - lead_i / 2}
    residual = {lead: float(measures[f"res_{lead}_max"]) for lead in expected}
    lead = max(residual, key=residual.get)
    peak = int(np.nanargmax(np.abs(window[LEADS.index(lead)] - expected[lead])))
    stored_minus_recomputed = window[LEADS.index(lead)] - expected[lead]
    return {"overlays": list(expected.items()), "highlight": [lead],
            "spans": [(lead, peak / FS - 0.15, peak / FS + 0.15)],
            "inset": (f"{lead} stored minus recomputed, 0-10 s (mV)", stored_minus_recomputed),
            "value": f"largest residual {residual[lead]:.3f} mV in {lead} (dashed: recomputed)"}


def note_la_ra(window: np.ndarray, measures: pd.Series, flags: pd.Series,
        blamed: list[str]) -> dict[str, Any]:
    """Highlight I, aVR and V6 (see ``annotate``)."""
    return {"highlight": ["I", "aVR", "V6"], "inset": (("I", "II", "aVR", "V6"), (1.0, 3.5)),
            "value": (f"I: P {measures['p_amp__I']:.2f} mV, QRS area {measures['qrs_area__I']:.0f}; "
                      f"aVR QRS area {measures['qrs_area__aVR']:.0f}")}


def note_ra_ll(window: np.ndarray, measures: pd.Series, flags: pd.Series,
        blamed: list[str]) -> dict[str, Any]:
    """Highlight II and aVR (see ``annotate``)."""
    return {"highlight": ["II", "aVR"], "inset": (("I", "II", "aVR", "aVF"), (1.0, 3.5)),
            "value": (f"II: P {measures['p_amp__II']:.2f} mV, QRS area {measures['qrs_area__II']:.0f}; "
                      f"aVR P {measures['p_amp__aVR']:.2f} mV")}


def note_amplitude(window: np.ndarray, measures: pd.Series, flags: pd.Series, blamed: list[str]
                   ) -> dict[str, Any]:
    """Give the amplitude ratio (see ``annotate``)."""
    return {"value": f"amplitude ratio {flags['amplitude_ratio']:.2f}"}


def note_wander(window: np.ndarray, measures: pd.Series, flags: pd.Series,
        blamed: list[str]) -> dict[str, Any]:
    """Give the largest baseline excursion (see ``annotate``)."""
    lead = worst_lead(measures, "baseline_ptp", blamed)
    return {"value": f"baseline excursion {measures[f'baseline_ptp__{lead}']:.2f} mV in {lead}"}


def note_mains(window: np.ndarray, measures: pd.Series, flags: pd.Series,
        blamed: list[str]) -> dict[str, Any]:
    """Zoom on the lead with the strongest mains line (see ``annotate``)."""
    values = {lead: max(float(measures[f"mains50__{lead}"]), float(measures[f"mains60__{lead}"]))
              for lead in blamed}
    lead = max(values, key=values.get)
    frequency = 50 if measures[f"mains50__{lead}"] >= measures[f"mains60__{lead}"] else 60
    return {"inset": (lead, 5.0, 5.5),
            "value": f"{frequency} Hz amplitude {values[lead]:.3f} mV in {lead}"}


def note_hf(window: np.ndarray, measures: pd.Series, flags: pd.Series,
        blamed: list[str]) -> dict[str, Any]:
    """Zoom on the noisiest lead (see ``annotate``)."""
    lead = worst_lead(measures, "hf_noise", blamed)
    return {"inset": (lead, 5.0, 6.0),
            "value": f"noise above 40 Hz {measures[f'hf_noise__{lead}']:.3f} mV in {lead}"}


def note_step(window: np.ndarray, measures: pd.Series, flags: pd.Series,
        blamed: list[str]) -> dict[str, Any]:
    """Mark the largest baseline step (see ``annotate``)."""
    lead = worst_lead(measures, "step", blamed)
    moment = step_time(window, lead)
    return {"spans": [(lead, moment - 0.15, moment + 0.15)],
            "value": f"step {measures[f'step__{lead}']:.2f} mV in {lead} at {moment:.2f} s"}


def note_spike(window: np.ndarray, measures: pd.Series, flags: pd.Series,
        blamed: list[str]) -> dict[str, Any]:
    """Mark and zoom on the largest isolated spike (see ``annotate``)."""
    lead = worst_lead(measures, "spike_isolated", blamed)
    moment = spike_time(window, lead)
    return {"spans": [(lead, moment - 0.08, moment + 0.08)],
            "inset": (lead, max(moment - 0.4, 0.0), min(moment + 0.4, window.shape[1] / FS)),
            "value": (f"isolated spike {measures[f'spike_isolated__{lead}']:.2f} mV in {lead} "
                      f"at {moment:.2f} s")}


def note_contact(window: np.ndarray, measures: pd.Series, flags: pd.Series,
        blamed: list[str]) -> dict[str, Any]:
    """Give the dropped beats and the longest hold of the first blamed lead (see ``annotate``)."""
    lead = blamed[0]
    dropout = measures.get(f"dropout_run__{lead}", np.nan)
    hold = measures[f"const_run_interior__{lead}"] * 1000.0 / FS
    return {"value": f"{lead}: {dropout:.0f} beats without QRS, longest interior hold {hold:.0f} ms"}


def note_rhythm(window: np.ndarray, measures: pd.Series, flags: pd.Series,
        blamed: list[str]) -> dict[str, Any]:
    """Give the beat counts and agreements (see ``annotate``)."""
    return {"value": (f"beats {measures['n_beats_multi']:.0f}, agreement multi/II "
                      f"{measures['agree_detectors']:.2f}, II/V5 {measures['agree_leads']:.2f}, multi/V5 "
                      f"{measures['agree_multi_V5']:.2f}")}


NOTES = {"missing_samples": note_missing, "edge_hold": note_edge, "adc_rail": note_rail,
         "flat_lead": note_small_lead, "near_flat_lead": note_small_lead,
         "reversal_neutral": note_small_lead,
         "identity_violation": note_identity, "reversal_la_ra": note_la_ra, "reversal_ra_ll": note_ra_ll,
         "amplitude_outlier": note_amplitude, "baseline_wander": note_wander,
         "mains_interference": note_mains,
         "hf_noise": note_hf, "baseline_step": note_step, "isolated_spike": note_spike,
         "contact_loss": note_contact, "rhythm_unmeasurable": note_rhythm}
NEEDS_BLAME = frozenset({"flat_lead", "near_flat_lead", "reversal_neutral", "baseline_wander",
                         "mains_interference", "hf_noise", "baseline_step", "isolated_spike", "contact_loss"})


def annotate(flag: str, window: np.ndarray, measures: pd.Series, flags: pd.Series) -> dict[str, Any]:
    """
    Highlighted leads, shaded spans, overlays, inset and a value text that show one flag on one record.

    Parameters
    ----------
    flag : str
        Flag name of ``mlc03_signal_flags.FLAGS``.
    window : np.ndarray
        Float ``(12, 5000)`` mV.
    measures : pd.Series
        The record's row of the measures table.
    flags : pd.Series
        The record's row of the flags table.

    Returns
    -------
    dict[str, Any]
        Keys of ``draw_record`` except ``window`` and ``title``, plus ``value`` (text for the title). A
        per-lead flag on a record it does not blame (an unflagged control) only gets its blamed-lead list.
    """
    blamed = [lead for lead in str(flags.get(f"{flag}__leads", "")).split(";") if lead]
    note: dict[str, Any] = {"highlight": list(blamed), "spans": [], "value": ""}
    if flag in NOTES and (blamed or flag not in NEEDS_BLAME):
        note.update(NOTES[flag](window, measures, flags, blamed))
    return note


def review_sample(flags: pd.DataFrame, flag: str, seed: int, size: int) -> tuple[list[str], list[str]]:
    """
    Draw a fixed random sample of flagged and unflagged records for one flag.

    A fresh ``numpy.random.default_rng(seed)`` draws ``size`` keys (all when fewer) from the flagged keys in
    sorted order, then ``size`` from the unflagged keys in sorted order.

    Parameters
    ----------
    flags : pd.DataFrame
        Flags table restricted to the records eligible for review.
    flag : str
        Flag name.
    seed : int
        Seed.
    size : int
        Records per group.

    Returns
    -------
    tuple[list[str], list[str]]
        Flagged and unflagged record keys, in draw order.
    """
    rng = np.random.default_rng(seed)
    flagged = np.sort(flags.loc[flags[flag], "record_key"].to_numpy(dtype=str))
    unflagged = np.sort(flags.loc[~flags[flag], "record_key"].to_numpy(dtype=str))
    pick = rng.choice(len(flagged), size=min(size, len(flagged)), replace=False)
    control = rng.choice(len(unflagged), size=min(size, len(unflagged)), replace=False)
    return flagged[pick].tolist(), unflagged[control].tolist()
