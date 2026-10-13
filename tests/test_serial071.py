"""Tests for the Experiment 071 serial change maps."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ecg_experiment import normative_map062 as nm
from ecg_experiment import serial071 as sr
from ecg_experiment.locbench import interval_mask, region_record_metrics
from ecg_experiment.waveforms import LEADS


def flags_table() -> pd.DataFrame:
    """Two patients: 1 has normal, normal, IMI; 2 has ASMI then normal."""
    rows = [
        (1, 1, "2000-01-01", {"NORM": 100}, (), True, True),
        (2, 1, "2000-02-01", {"NORM": 80}, (), True, True),
        (3, 1, "2000-03-01", {"IMI": 100, "1AVB": 50}, ("IMI", "1AVB"), False, False),
        (4, 2, "2001-01-01", {"ASMI": 100}, ("ASMI",), False, False),
        (5, 2, "2001-01-05", {"NORM": 100, "1AVB": 20}, (), True, True),
    ]
    table = pd.DataFrame(
        rows, columns=["ecg_id", "patient_id", "date", "codes", "region", "normal", "finding_free"]
    )
    table["date"] = pd.to_datetime(table["date"])
    table["partition"] = "fit"
    table["rhythm_ok"] = True
    return table


def test_change_pairs_forward_picks_closest_earlier_normal() -> None:
    pairs = sr.change_pairs(flags_table(), "normal", "forward")
    assert pairs.ecg_id.tolist() == [3]
    assert pairs.partner.tolist() == [2]
    assert pairs.new.tolist() == [("IMI", "1AVB")]
    assert pairs.gap_days.iloc[0] == pytest.approx(29.0)


def test_change_pairs_both_directions() -> None:
    pairs = sr.change_pairs(flags_table(), "normal", "both").set_index("ecg_id")
    assert pairs.loc[4, "partner"] == 5
    assert not pairs.loc[4, "forward"]
    assert pairs.loc[4, "new"] == ("ASMI",)


def test_new_statements_exclude_partner_codes() -> None:
    table = flags_table()
    table.loc[table.ecg_id == 5, "codes"] = [{"NORM": 100, "ASMI": 15}]
    pairs = sr.change_pairs(table, "normal", "both")
    assert 4 not in pairs.ecg_id.tolist()


def test_null_pairs_are_consecutive_normals() -> None:
    nulls = sr.null_pairs(flags_table())
    assert list(zip(nulls.ecg_id, nulls.partner, strict=True)) == [(2, 1)]


def test_match_partner_rules() -> None:
    pool = pd.DataFrame(
        {
            "ecg_id": [10, 11, 12, 13, 14],
            "patient_id": [7, 8, 9, 10, 1],
            "sex": [0, 0, 1, 0, 0],
            "age": [50.0, 52.0, 51.0, 70.0, 50.0],
            "heart_rate": [80.0, 61.0, 60.0, 60.0, 60.0],
        }
    )
    target = pd.Series({"patient_id": 1, "sex": 0, "age": 51.0, "heart_rate": 60.0})
    assert sr.match_partner(target, pool) == 11
    old = pd.Series({"patient_id": 1, "sex": 0, "age": np.nan, "heart_rate": 60.0})
    assert sr.match_partner(old, pool) == 13


def test_fit_null_floors() -> None:
    rng = np.random.default_rng(0)
    differences = rng.normal(0, 1, (200, 12, nm.BINS))
    differences[:, 0, :50] *= 1e-3
    intervals = np.zeros((200, 3))
    model = sr.fit_null(differences, intervals)
    assert model.spread.min() == pytest.approx(model.floor)
    assert (model.spread <= model.floor).mean() >= sr.FLOOR_QUANTILE
    assert np.all(model.interval_spread == sr.INTERVAL_FLOOR)


def beats_of(median: np.ndarray) -> sr.Beats:
    knots = np.array([[100.0, 250, 300, 400, 550, 600], [600, 750, 800, 900, 1050, 1100]])
    return sr.Beats(median, knots, np.array([60.0, 0.1, 0.4, 0.16]))


def test_change_grid_and_paint() -> None:
    model = sr.NullModel(
        np.zeros((12, nm.BINS)), np.ones((12, nm.BINS)), 1.0, np.zeros(3), np.full(3, 0.01), 10
    )
    base = np.zeros((12, nm.BINS), np.float32)
    assert np.all(sr.change_grid(beats_of(base), beats_of(base), model) == 0)
    late = base.copy()
    late[LEADS.index("V2"), 160:180] = 3.0
    grid = sr.change_grid(beats_of(late), beats_of(base), model)
    assert grid.argmax() // nm.BINS == LEADS.index("V2")
    longer = beats_of(base)
    longer.intervals = np.array([60.0, 0.12, 0.4, 0.16])
    with_interval = sr.change_grid(longer, beats_of(base), model, intervals=True)
    qrs = nm.SEGMENTS.index("QRS") == nm.SEGMENT_OF_BIN
    assert np.allclose(with_interval[:, qrs], 2.0)
    assert np.allclose(with_interval[:, ~qrs], 0.0)
    painted = sr.paint(grid, beats_of(late).knots)
    assert painted.shape == (12, 5000)
    assert painted.dtype == np.float32
    assert painted[LEADS.index("V2"), 250:300].max() > painted[LEADS.index("I"), 250:300].max()


def components() -> dict[str, np.ndarray]:
    """One beat of ECGdeli-like components, plus a beat outside the scored samples."""
    return {
        "P": np.array([[150, 200], [10, 40]]),
        "PR": np.array([[150, 230], [10, 70]]),
        "QRS": np.array([[230, 280]]),
        "ST": np.array([[280, 350]]),
        "T": np.array([[350, 450]]),
    }


def test_unit_masks() -> None:
    masks = sr.unit_masks(components())
    assert masks.shape == (5, 5000)
    assert masks[0, 150:200].all()
    assert not masks[0, 10:40].any()
    assert masks[1, 200:230].all()
    assert not masks[1, 150:200].any()


def test_target_units() -> None:
    first = sr.target_units(("1AVB",))
    assert first[:, :2].all()
    assert not first[:, 2:].any()
    imi = sr.target_units(("IMI",))
    assert imi.sum() == 3
    assert imi[LEADS.index("III"), sr.UNIT_COMPONENTS.index("QRS")]
    union = sr.target_units(("IMI", "INJAS"))
    assert union.sum() == 7


def test_region_readings_match_locbench_for_one_statement() -> None:
    rng = np.random.default_rng(1)
    record_map = rng.random((12, 5000)).astype(np.float32)
    parts = components()
    readings = sr.region_readings(record_map, ("IMI",), sr.unit_masks(parts))
    leads = np.isin(LEADS, ("II", "III", "aVF"))
    reference = region_record_metrics(record_map, leads, interval_mask(parts["QRS"]))
    for name in ("B1", "B2", "B3"):
        assert readings[name] == pytest.approx(reference[name], abs=1e-15)


def test_unit_auroc() -> None:
    scores = np.zeros((12, 5))
    target = sr.target_units(("IMI",))
    scores[target] = 1.0
    assert sr.unit_auroc(scores, target) == 1.0
    scores[:, 0] = np.nan
    assert sr.unit_auroc(scores, np.zeros((12, 5), bool)) != sr.unit_auroc(scores, np.zeros((12, 5), bool))


def test_null_reading_weights() -> None:
    masks = sr.unit_masks(components())
    record_map = np.zeros((12, 5000), np.float32)
    record_map[[LEADS.index(x) for x in ("II", "III", "aVF")], 230:280] = 1.0
    value = sr.null_reading(record_map, masks, {("IMI",): 3, ("INJAS",): 1})
    expected = (
        3 * 1.0 + 1 * sr.unit_auroc(sr.unit_scores(record_map, masks), sr.target_units(("INJAS",)))
    ) / 4
    assert value == pytest.approx(expected)


def test_bootstrap_means_difference() -> None:
    values = np.arange(20, dtype=float)
    patients = np.repeat(np.arange(10), 2)
    same = sr.bootstrap_means([(values, patients), (values, patients)], lambda m: m[0] - m[1], draws=200)
    assert same["value"] == 0.0
    assert same["ci_low"] == pytest.approx(0.0)
    assert same["ci_high"] == pytest.approx(0.0)
    paired = sr.paired_mean_difference(values + 1, values, patients, draws=200)
    assert paired["value"] == pytest.approx(1.0)
    assert paired["wins"] == 20


def test_change_unit_labels() -> None:
    labels = sr.change_unit_labels(np.array([0.1, 0.9, 0.5, 0.2, 0.3, 0.8, 0.0, 0.4, 0.6, 0.7]))
    assert labels.sum() == 2
    assert labels[[1, 5]].all()


def test_family_macro() -> None:
    values = np.array([1.0, 0.0, 0.0, 0.5, np.nan])
    families = [["a"], ["a"], ["b", "a"], ["b"], ["b"]]
    patients = np.arange(5)
    result = sr.family_macro(values, families, patients, minimum=2, draws=100)
    assert result["families"] == {"a": 3, "b": 2}
    assert result["value"] == pytest.approx((1 / 3 + 0.25) / 2)
    assert sr.family_macro(values, families, patients, minimum=4)["value"] is None
