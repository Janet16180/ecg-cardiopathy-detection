"""Beat-sum wave maps for Experiment 051: score whole beats from 042's beat-aligned lead-and-wave units."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from .focal_switch import LEADS_PER_BEAT, UNITS_PER_BEAT, WAVES_PER_LEAD
from .lead_wave_maps import WAVES, UnitMap
from .waveforms import LEADS

P_START = WAVES["P"][0]
T_END = WAVES["T"][1]


def beat_r_times(unit_map: UnitMap) -> np.ndarray:
    """Return each beat's R time in seconds, from its P piece's start."""
    starts = np.asarray(unit_map.starts, dtype=np.float64).reshape(-1, LEADS_PER_BEAT, WAVES_PER_LEAD)
    return starts[:, 0, 0] - P_START


def beat_sum_map(unit_map: UnitMap, relative: bool = False) -> UnitMap:
    """
    Return one unit per beat, scored by the sum of the beat's 48 lead-and-wave scores.

    Parameters
    ----------
    unit_map : UnitMap
        Beat-aligned units in ``beat_unit_map`` order.
    relative : bool
        Divide each beat sum by the median beat sum of the ECG.

    Returns
    -------
    UnitMap
        Per beat: the score, the lead of the beat's highest piece, and the span from P start to T end.
    """
    scores = np.asarray(unit_map.scores, dtype=np.float64).reshape(-1, UNITS_PER_BEAT)
    sums = scores.sum(axis=1)
    if relative:
        sums = sums / np.median(sums)
    leads = np.asarray(unit_map.leads).reshape(-1, UNITS_PER_BEAT)
    leads = leads[np.arange(len(scores)), scores.argmax(axis=1)]
    r_times = beat_r_times(unit_map)
    return UnitMap(sums, leads, r_times + P_START, r_times + T_END)


def top_pieces(unit_map: UnitMap, beat: int, count: int = 3) -> list[tuple[int, float, float]]:
    """Return the lead, start and end of a beat's highest-scoring lead-and-wave pieces, best first."""
    low = beat * UNITS_PER_BEAT
    scores = np.asarray(unit_map.scores[low:low + UNITS_PER_BEAT])
    order = np.argsort(scores)[::-1][:count] + low
    return [(int(unit_map.leads[i]), float(unit_map.starts[i]), float(unit_map.ends[i])) for i in order]


def unit_beat(index: int) -> int:
    """Return the beat of a ``beat_unit_map`` unit index."""
    return index // UNITS_PER_BEAT


def r_time_hit(r_times: np.ndarray, top_beat: int, windows: list[tuple[float, float]]) -> tuple[float, float]:
    """
    Return whether the top beat's R time lies in a premature-beat window, and the share of beats that do.

    Parameters
    ----------
    r_times : np.ndarray
        R time of every beat in seconds.
    top_beat : int
        Index of the top beat.
    windows : list[tuple[float, float]]
        Premature-beat windows from ``fragment_localization.premature_windows``.

    Returns
    -------
    tuple[float, float]
        ``hit`` (0 or 1) and ``chance``.
    """
    inside = np.zeros(len(r_times), dtype=bool)
    for low, high in windows:
        inside |= (r_times >= low) & (r_times < high)
    return float(inside[top_beat]), float(inside.mean())


def plot_beat_marks(signal: np.ndarray, fs: int, columns: dict[str, dict[str, object]], title: str) -> Figure:
    """
    Draw twelve leads per column, with an optional beat shaded across all leads and pieces marked on theirs.

    Parameters
    ----------
    signal : np.ndarray
        ``[12, samples]`` waveform in mV, canonical lead order.
    fs : int
        Sampling rate in Hz.
    columns : dict[str, dict[str, object]]
        Per column title: ``beat`` (``(start, end, colour)`` or ``None``) and ``pieces`` (a list of
        ``(lead, start, end)``), drawn in red.
    title : str
        Figure title.

    Returns
    -------
    Figure
        The figure.
    """
    figure = Figure(figsize=(7 * len(columns), 11), layout="constrained")
    axes = figure.subplots(len(LEADS), len(columns), sharex=True, squeeze=False)
    seconds = np.arange(signal.shape[1]) / fs
    for column, (name, found) in enumerate(columns.items()):
        axes[0, column].set_title(name)
        beat = found.get("beat")
        for row, lead in enumerate(LEADS):
            axis = axes[row, column]
            if beat is not None:
                start, end, colour = beat
                axis.axvspan(start, end, color=colour, alpha=0.15, linewidth=0)
            axis.plot(seconds, signal[row], color="black", linewidth=0.6)
            axis.set_ylabel(lead, rotation=0, labelpad=14)
            axis.set_yticks([])
        for lead, start, end in found.get("pieces", []):
            axes[lead, column].axvspan(start, end, color="red", alpha=0.35, linewidth=0)
        axes[-1, column].set_xlabel("seconds")
    figure.suptitle(title)
    return figure
