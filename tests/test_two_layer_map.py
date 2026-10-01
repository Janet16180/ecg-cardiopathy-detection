"""Tests of the Experiment 045 two-layer explanation map on synthetic scores."""

from __future__ import annotations

import numpy as np

from ecg_experiment.lead_wave_maps import UnitMap, premature_hit, top_lead
from ecg_experiment.two_layer_map import (
    choose_quantile,
    explain,
    layer_thresholds,
    mean_logit,
    quantile_grid,
    red_unit_count,
    union_share,
)


def unit_map(scores: list[float], leads: list[int], starts: list[float]) -> UnitMap:
    """Units with 0.2 s spans."""
    begin = np.array(starts, dtype=np.float64)
    return UnitMap(np.array(scores, dtype=np.float64), np.array(leads), begin, begin + 0.2)


def test_quantile_grid_is_fine_and_closed() -> None:
    grid = quantile_grid()
    assert len(grid) == 1001
    assert grid[0] == 0.9
    assert grid[-1] == 1.0
    assert np.allclose(np.diff(grid), 1e-4)


def test_union_share_counts_either_layer() -> None:
    first = np.arange(100, dtype=np.float64)
    second = np.arange(100, dtype=np.float64)[::-1].copy()
    a, b = layer_thresholds(first, second, 0.95)
    assert a == b == np.quantile(first, 0.95)
    assert union_share(first, second, 0.95) == 0.10
    assert union_share(first, second, 1.0) == 0.0


def test_choose_quantile_hits_target_and_breaks_ties_by_median() -> None:
    rng = np.random.default_rng(0)
    first, second = rng.normal(size=463), rng.normal(size=463)
    found = choose_quantile(first, second, 0.05)
    assert abs(found["share"] - 0.05) <= 1 / 463
    assert found["red_normals"] == round(found["share"] * 463)
    grid = quantile_grid()
    shares = np.array([union_share(first, second, q) for q in grid])
    tied = grid[np.isclose(np.abs(shares - 0.05), np.abs(shares - 0.05).min(), rtol=0.0, atol=1e-12)]
    assert found["tied"] == len(tied)
    assert found["q"] == tied[(len(tied) - 1) // 2]


def test_choose_quantile_on_a_small_grid() -> None:
    first = np.arange(10, dtype=np.float64)
    second = np.zeros(10)
    found = choose_quantile(first, second, 0.2, grid=np.array([0.5, 0.75, 0.8, 0.85, 1.0]))
    assert found["share"] == 0.2
    assert found["tied"] == 2
    assert found["q"] == 0.8


def test_explanation_prefers_a_red_first_layer() -> None:
    first = unit_map([1.0, 5.0], [0, 1], [0.0, 2.0])
    second = unit_map([0.3, 0.1, 0.9], [6, 7, 8], [0.0, 1.0, 4.0])
    red_first = explain(first, second, 4.0, 0.5)
    assert (red_first.layer, red_first.first_red, red_first.second_red) == (1, True, True)
    assert top_lead(red_first.units) == "II"
    quiet_first = explain(first, second, 6.0, 0.95)
    assert (quiet_first.layer, quiet_first.first_red, quiet_first.second_red) == (2, False, False)
    assert top_lead(quiet_first.units) == "V3"
    assert premature_hit(quiet_first.units, [(4.0, 4.5)]) == (1.0, 1 / 3)
    assert red_unit_count(first, second, 0.5, 0.2) == 4


def test_mean_logit() -> None:
    assert np.array_equal(mean_logit(np.array([1.0, -2.0]), np.array([3.0, 0.0])), np.array([2.0, -1.0]))
