"""Tests of the Experiment 051 beat-sum map on synthetic beat-aligned units."""

from __future__ import annotations

import numpy as np

from ecg_experiment.beat_sum_map import (
    beat_r_times,
    beat_sum_map,
    plot_beat_marks,
    r_time_hit,
    top_pieces,
    unit_beat,
)
from ecg_experiment.lead_wave_maps import beat_unit_map


def synthetic() -> object:
    """Three beats at R = 1, 2 and 3 s: beat 0 has one strong piece, beat 1 many moderate ones."""
    scores = np.full((3, 12, 4), 1.0)
    scores[0, 6, 2] = 100.0
    scores[1] = 5.0
    return beat_unit_map(scores, np.array([1.0, 2.0, 3.0]))


def test_beat_sum_scores_whole_beats() -> None:
    unit_map = synthetic()
    assert np.allclose(beat_r_times(unit_map), [1.0, 2.0, 3.0])
    beats = beat_sum_map(unit_map)
    assert np.allclose(beats.scores, [147.0, 240.0, 48.0])
    assert int(beats.scores.argmax()) == 1
    assert unit_beat(int(unit_map.scores.argmax())) == 0
    assert beats.leads[0] == 6
    assert np.allclose(beats.starts, [0.75, 1.75, 2.75])
    assert np.allclose(beats.ends, [1.45, 2.45, 3.45])
    relative = beat_sum_map(unit_map, relative=True)
    assert np.allclose(relative.scores, beats.scores / 147.0)


def test_top_pieces_are_the_beats_highest_units() -> None:
    unit_map = synthetic()
    pieces = top_pieces(unit_map, 0)
    assert len(pieces) == 3
    assert pieces[0][0] == 6
    assert np.isclose(pieces[0][1], 1.08)


def test_r_time_hit_uses_the_beat_r_time() -> None:
    r_times = np.array([1.0, 2.0, 3.0])
    assert r_time_hit(r_times, 1, [(1.9, 2.4)]) == (1.0, 1 / 3)
    assert r_time_hit(r_times, 0, [(1.9, 2.4)]) == (0.0, 1 / 3)


def test_plot_beat_marks_shades_the_beat_on_every_lead() -> None:
    figure = plot_beat_marks(np.zeros((12, 2000)), 500,
                             {"a": {"beat": (1.0, 1.7, "red"), "pieces": [(6, 1.1, 1.2)]},
                              "b": {"beat": None, "pieces": []}}, "title")
    first_column = figure.axes[0::2]
    assert sum(len(axis.patches) for axis in first_column) == 13
    assert sum(len(axis.patches) for axis in figure.axes[1::2]) == 0
