"""Explanation switch for Experiment 048: show ``U_B`` when the PVC finding head is high, else attention."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from .finding_screen import clipped_logits
from .hybrid_score import standardize
from .lead_wave_maps import UnitMap
from .pipeline_v2 import score_parameters
from .two_layer_map import Explanation
from .waveforms import LEADS

LAYER_COLOURS = {1: "#d62728", 2: "#1f77b4"}
LAYER_NAMES = {1: "U_B", 2: "attention_jepa"}


def pvc_z(parameters: dict[str, np.ndarray], x: np.ndarray, prefix: str = "pvc") -> np.ndarray:
    """
    Return the standardized PVC-head logit of each ECG, with the head's saved normal mean and SD.

    Parameters
    ----------
    parameters : dict[str, np.ndarray]
        Saved head arrays (``{prefix}_mean``, ``_scale``, ``_coef``, ``_intercept``, ``_logit_mean`` and
        ``_logit_sd``).
    x : np.ndarray
        ``(n, d)`` features of the head.
    prefix : str
        Name of the head.

    Returns
    -------
    np.ndarray
        z-scores.
    """
    values, _ = clipped_logits(score_parameters(parameters, prefix, x))
    stats = (float(parameters[f"{prefix}_logit_mean"][0]), float(parameters[f"{prefix}_logit_sd"][0]))
    return standardize(values, stats)


def switch_threshold(normal_z: np.ndarray, quantile: float) -> float:
    """Return the quantile of normal ECGs' z-scores, with NumPy's default linear interpolation."""
    return float(np.quantile(np.asarray(normal_z, dtype=np.float64), quantile))


def switch_explain(first: UnitMap, second: UnitMap, use_first: bool, first_threshold: float,
                   second_threshold: float) -> Explanation:
    """
    Explain one ECG with the first layer's units when the switch is on, otherwise with the second layer's.

    Parameters
    ----------
    first, second : UnitMap
        The ECG's units in each layer.
    use_first : bool
        Whether the switch is on.
    first_threshold, second_threshold : float
        Each layer's own red threshold; they only set the descriptive red flags.

    Returns
    -------
    Explanation
        The supplying layer (1 or 2), both red flags and the supplying layer's units.
    """
    return Explanation(1 if use_first else 2, bool(first.scores.max() > first_threshold),
                       bool(second.scores.max() > second_threshold), first if use_first else second)


def plot_layer_marks(signal: np.ndarray, fs: int, marks: dict[str, list[tuple[int, float, float, int]]],
                     title: str) -> Figure:
    """
    Draw twelve leads with each mark coloured by the layer that supplied it, one column per map.

    Parameters
    ----------
    signal : np.ndarray
        ``[12, samples]`` waveform in mV, canonical lead order.
    fs : int
        Sampling rate in Hz.
    marks : dict[str, list[tuple[int, float, float, int]]]
        Lead index, start, end and layer (1 or 2) of each mark, keyed by the column title.
    title : str
        Figure title.

    Returns
    -------
    Figure
        The figure, with a legend of the two layer colours.
    """
    figure = Figure(figsize=(7 * len(marks), 11), layout="constrained")
    axes = figure.subplots(len(LEADS), len(marks), sharex=True, squeeze=False)
    seconds = np.arange(signal.shape[1]) / fs
    for column, (name, found) in enumerate(marks.items()):
        axes[0, column].set_title(name)
        for lead, start, end, layer in found:
            axes[lead, column].axvspan(start, end, color=LAYER_COLOURS[layer], alpha=0.3, linewidth=0)
        for row, lead in enumerate(LEADS):
            axis = axes[row, column]
            axis.plot(seconds, signal[row], color="black", linewidth=0.6)
            axis.set_ylabel(lead, rotation=0, labelpad=14)
            axis.set_yticks([])
        axes[-1, column].set_xlabel("seconds")
    figure.legend(handles=[Patch(color=LAYER_COLOURS[k], alpha=0.3, label=LAYER_NAMES[k]) for k in (1, 2)],
                  loc="upper right")
    figure.suptitle(title)
    return figure
