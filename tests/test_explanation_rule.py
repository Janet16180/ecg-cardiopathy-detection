"""Tests of the Experiment 047 explanation rule on synthetic scores."""

from __future__ import annotations

import numpy as np
import pytest

from ecg_experiment.explanation_rule import (
    explanation_metrics,
    group_rates,
    layer_shares,
    paired_difference,
    referral_threshold,
    referred_ids,
    rule_marks,
    rule_reading,
)
from ecg_experiment.lead_wave_maps import UnitMap
from ecg_experiment.two_layer_map import explain


def unit_map(scores: list[float], leads: list[int], starts: list[float]) -> UnitMap:
    """Units with 0.2 s spans."""
    begin = np.array(starts, dtype=np.float64)
    return UnitMap(np.array(scores, dtype=np.float64), np.array(leads), begin, begin + 0.2)


def test_referral_threshold_is_the_linear_quantile_and_strict() -> None:
    normals = np.arange(21, dtype=np.float64)
    threshold = referral_threshold(normals)
    assert threshold == np.quantile(normals, 0.95) == 19.0
    assert referred_ids([5, 6, 7], np.array([19.0, 19.5, 3.0]), threshold) == [6]


def test_group_rates_and_layer_shares() -> None:
    groups = {"a": [1, 2, 3, 4], "b": [5, 6]}
    assert group_rates({1, 2, 5}, groups) == {"a": 0.5, "b": 0.5}
    shares = layer_shares({1: 1, 2: 2, 5: 2}, groups)
    assert shares["a"] == {"n": 2, "layer1": 0.5, "layer2": 0.5}
    assert shares["b"] == {"n": 1, "layer1": 0.0, "layer2": 1.0}
    assert np.isnan(layer_shares({}, groups)["a"]["layer1"])


def synthetic() -> dict[str, dict[int, UnitMap]]:
    """Two layers on six ECGs: 1-2 PVC, 3-4 anterior infarcts, 5-6 inferior infarcts."""
    first = {1: unit_map([5.0, 1.0], [1, 6], [1.0, 3.0]), 2: unit_map([0.5, 0.2], [6, 1], [3.0, 1.0]),
             3: unit_map([5.0, 1.0], [1, 6], [1.0, 3.0]), 4: unit_map([0.5, 0.2], [1, 6], [1.0, 3.0]),
             5: unit_map([5.0, 1.0], [1, 6], [1.0, 3.0]), 6: unit_map([0.5, 0.2], [6, 1], [1.0, 3.0])}
    second = {i: unit_map([1.0, 0.0], [8, 1], [1.0, 3.0]) for i in range(1, 7)}
    return {"first": first, "second": second}


def test_explanation_metrics_and_paired_difference() -> None:
    maps = synthetic()
    patients = {i: i for i in range(1, 7)}
    windows = {1: [(1.0, 1.2)], 2: [(1.0, 1.2)]}
    rule = {i: explain(maps["first"][i], maps["second"][i], 1.0, 9.0).units for i in range(1, 7)}
    found = explanation_metrics(rule, patients, windows, [3, 4], [5, 6], 50, 0)
    alone = explanation_metrics(maps["first"], patients, windows, [3, 4], [5, 6], 50, 0)
    assert found["metrics"]["pvc_ecgs"] == 2
    assert found["metrics"]["hit_rate"] == 1.0
    assert found["metrics"]["chance_rate"] == 0.5
    assert alone["metrics"]["hit_rate"] == 0.5
    assert found["metrics"]["lead_contrast"]["anterior"]["value"] == 0.0
    assert alone["metrics"]["lead_contrast"]["anterior"]["value"] == -0.5
    paired = paired_difference(found, alone, patients, 50, 0)
    assert paired["hit_minus_chance"]["value"] == 0.5
    expected = (found["metrics"]["lead_contrast"]["anterior"]["value"]
                - alone["metrics"]["lead_contrast"]["anterior"]["value"])
    assert paired["lead_contrast"]["anterior"]["value"] == pytest.approx(expected) == 0.5


def test_explanation_metrics_scores_only_explained_ecgs() -> None:
    maps = synthetic()
    subset = {i: maps["first"][i] for i in (1, 3, 5, 6)}
    patients = {i: i for i in range(1, 7)}
    windows = {1: [(1.0, 1.2)], 2: [(1.0, 1.2)]}
    found = explanation_metrics(subset, patients, windows, [3, 4], [5, 6], 20, 0)
    assert found["metrics"]["pvc_ecgs"] == 1
    assert found["metrics"]["infarct_ecgs"] == {"anterior": 1, "inferior": 2}
    every = explanation_metrics(maps["first"], patients, windows, [3, 4], [5, 6], 20, 0)
    with pytest.raises(ValueError, match="different ECGs"):
        paired_difference(found, every, patients, 20, 0)


def test_rule_reading() -> None:
    assert rule_reading(-0.05, 0.01) == {"keeps_premature_localization": True, "anterior_above_zero": True,
                                         "improves": True}
    assert not rule_reading(-0.11, 0.01)["improves"]
    assert not rule_reading(0.2, 0.0)["improves"]


def test_rule_marks_add_the_top_unit_once() -> None:
    found = explain(unit_map([0.5, 0.2], [6, 1], [3.0, 1.0]), unit_map([0.3, 0.9], [2, 3], [0.0, 2.0]),
                    1.0, 0.5)
    assert found.layer == 2
    assert rule_marks(found, 0.5) == [(3, 2.0, 2.2)]
    assert rule_marks(found, 2.0) == [(3, 2.0, 2.2)]
    assert rule_marks(found, 0.1) == [(2, 0.0, 0.2), (3, 2.0, 2.2)]
