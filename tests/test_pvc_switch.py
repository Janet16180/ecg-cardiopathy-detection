"""Tests of the Experiment 048 PVC switch on synthetic heads and scores."""

from __future__ import annotations

import numpy as np
from scipy.special import logit

from ecg_experiment.finding_screen import finding_z
from ecg_experiment.lead_wave_maps import UnitMap
from ecg_experiment.pipeline_v2 import score_parameters
from ecg_experiment.pvc_switch import LAYER_COLOURS, plot_layer_marks, pvc_z, switch_explain, switch_threshold


def unit_map(scores: list[float], leads: list[int], starts: list[float]) -> UnitMap:
    """Units with 0.2 s spans."""
    begin = np.array(starts, dtype=np.float64)
    return UnitMap(np.array(scores, dtype=np.float64), np.array(leads), begin, begin + 0.2)


def head(rng: np.random.Generator, width: int = 5) -> dict[str, np.ndarray]:
    """A random logistic head with saved standardization constants."""
    return {"pvc_mean": rng.normal(size=width), "pvc_scale": rng.uniform(0.5, 2.0, size=width),
            "pvc_coef": rng.normal(size=width), "pvc_intercept": np.array([-1.0])}


def test_pvc_z_matches_finding_z_with_the_same_normals() -> None:
    rng = np.random.default_rng(0)
    parameters = head(rng)
    normals, scored = rng.normal(size=(50, 5)), rng.normal(size=(20, 5))
    source = logit(score_parameters(parameters, "pvc", normals))
    parameters["pvc_logit_mean"] = np.array([source.mean()])
    parameters["pvc_logit_sd"] = np.array([source.std()])
    expected = finding_z(score_parameters(parameters, "pvc", scored),
                         score_parameters(parameters, "pvc", normals))
    assert np.allclose(pvc_z(parameters, scored), expected, rtol=0, atol=1e-12)


def test_switch_threshold_is_the_linear_quantile() -> None:
    values = np.arange(41, dtype=np.float64)
    assert switch_threshold(values, 0.975) == np.quantile(values, 0.975) == 39.0


def test_switch_explain_follows_the_switch_not_the_red_flags() -> None:
    first, second = unit_map([0.5, 0.2], [6, 1], [3.0, 1.0]), unit_map([0.3, 0.9], [2, 3], [0.0, 2.0])
    on = switch_explain(first, second, True, 1.0, 0.5)
    assert (on.layer, on.first_red, on.second_red) == (1, False, True)
    assert on.units is first
    off = switch_explain(first, second, False, 0.1, 2.0)
    assert (off.layer, off.first_red, off.second_red) == (2, True, False)
    assert off.units is second


def test_plot_layer_marks_colours_by_layer() -> None:
    signal = np.zeros((12, 500))
    figure = plot_layer_marks(signal, 500, {"a": [(0, 0.1, 0.3, 1), (6, 0.4, 0.6, 2)], "b": []}, "title")
    axes = figure.axes
    assert len(axes) == 24
    patches = [patch for axis in axes for patch in axis.patches]
    assert len(patches) == 2
    colours = {tuple(np.round(patch.get_facecolor()[:3], 3)) for patch in patches}
    expected = {tuple(np.round(np.array([int(c[i:i + 2], 16) for i in (1, 3, 5)]) / 255, 3))
                for c in LAYER_COLOURS.values()}
    assert colours == expected
    assert len(figure.legends) == 1
