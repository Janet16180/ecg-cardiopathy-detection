"""Tests of the Experiment 049 focal-versus-diffuse switch on synthetic units and scores."""

from __future__ import annotations

import numpy as np

from ecg_experiment.finding_screen import split_thresholds
from ecg_experiment.focal_switch import beat_layout_ok, beat_scores, combined_referral, focal_ratio
from ecg_experiment.lead_wave_maps import beat_unit_map


def wave_map(beat_peaks: list[float]) -> object:
    """A beat-aligned unit map whose beat b has its largest unit equal to ``beat_peaks[b]``."""
    rng = np.random.default_rng(1)
    scores = rng.uniform(0.0, 0.5, size=(len(beat_peaks), 12, 4))
    for beat, peak in enumerate(beat_peaks):
        scores[beat, beat % 12, beat % 4] = peak
    return beat_unit_map(scores, np.arange(len(beat_peaks), dtype=np.float64) + 0.5)


def test_layout_and_beat_scores_follow_beat_unit_map() -> None:
    unit_map = wave_map([1.0, 5.0, 2.0])
    assert beat_layout_ok(unit_map)
    assert np.array_equal(beat_scores(unit_map), [1.0, 5.0, 2.0])
    assert focal_ratio(unit_map) == 2.5


def test_layout_check_rejects_other_orders() -> None:
    unit_map = wave_map([1.0, 2.0])
    shuffled = type(unit_map)(unit_map.scores, unit_map.leads[::-1].copy(), unit_map.starts, unit_map.ends)
    assert not beat_layout_ok(shuffled)
    short = type(unit_map)(*(values[:40] for values in (unit_map.scores, unit_map.leads, unit_map.starts,
                                                         unit_map.ends)))
    assert not beat_layout_ok(short)


def test_diffuse_change_has_a_low_focal_ratio() -> None:
    assert focal_ratio(wave_map([4.0, 4.1, 3.9, 4.0])) < focal_ratio(wave_map([1.0, 1.1, 0.9, 4.0]))


def test_combined_referral_matches_split_thresholds() -> None:
    rng = np.random.default_rng(0)
    binary, finding = rng.normal(size=600), rng.normal(size=600)
    normal = np.zeros(600, dtype=bool)
    normal[:463] = True
    referred, thresholds = combined_referral(binary, finding, normal)
    expected = split_thresholds(np.column_stack([binary, finding])[normal], 50, (50,))
    assert np.array_equal(thresholds, expected)
    assert thresholds[1] == np.sort(finding[normal])[::-1][1]
    assert referred[normal].sum() <= 22
    assert np.array_equal(referred, (binary > thresholds[0]) | (finding > thresholds[1]))
