"""Tests for the Experiment 067 holdout wrapper and metric helpers. No ECGAD label is read."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ecg_experiment import combination066 as cb
from ecg_experiment import holdout067 as ho
from ecg_experiment.locbench import START, STOP, metric_interval, paired_difference


def synthetic(records: int = 6, seed: int = 0) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Marks on one lead window per record (two records unmarked), a perfect map and record IDs."""
    rng = np.random.default_rng(seed)
    marks = np.zeros((records, 12, 5000), bool)
    for k in range(records - 2):
        lead = rng.integers(0, 12)
        start = rng.integers(START, STOP - 300)
        marks[k, lead, start : start + 200] = True
    maps = marks.astype(np.float32) + 0.01 * rng.random(marks.shape).astype(np.float32)
    ids = pd.DataFrame({"ecg_id": np.arange(records) + 1, "patient_id": np.arange(records) // 2 + 100})
    return maps, marks, ids


def test_decision_words() -> None:
    assert ho.decision({"ci_low": 0.01, "ci_high": 0.05}) == "confirmed"
    assert ho.decision({"ci_low": -0.01, "ci_high": 0.05}) == "matches"
    assert ho.decision({"ci_low": -0.05, "ci_high": -0.01}) == "below"


def test_ecgad_tables_work_with_locbench_intervals() -> None:
    maps, marks, ids = synthetic()
    perfect = ho.ecgad_tables(maps, marks, ids, "perfect")
    noise = ho.ecgad_tables(np.random.default_rng(1).random(maps.shape), marks, ids, "noise")
    frame = perfect["ecgad"]
    assert frame["marked"].sum() == 4
    assert frame["A2"].notna().sum() == 4
    assert frame.loc[frame["marked"], "A2"].min() == pytest.approx(1.0)
    interval = metric_interval(perfect, "A2", draws=50)
    assert interval["value"] == pytest.approx(1.0)
    assert metric_interval(perfect, "A1", draws=50)["value"] == pytest.approx(1.0)
    difference = paired_difference(perfect, noise, "A2", draws=50)
    assert difference["value"] > 0.3
    assert difference["records"] == 4


def test_ecgad_tables_reject_nonfinite() -> None:
    maps, marks, ids = synthetic()
    maps[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        ho.ecgad_tables(maps, marks, ids, "bad")


def test_strata_split_inside_and_outside_qrs() -> None:
    maps, marks, ids = synthetic()
    qrs = np.zeros_like(marks)
    qrs[:, :, ::7] = True
    table = ho.strata_records(maps, marks, qrs, ids)
    assert list(table.columns) == ["ecg_id", "patient_id", "A2_qrs", "A2_nonqrs"]
    assert table["A2_nonqrs"].dropna().min() == pytest.approx(1.0)
    assert np.isnan(table["A2_qrs"].iloc[-1])


def test_statement_groups_use_region_families() -> None:
    meta = pd.DataFrame(
        {"scp_codes": [{"IMI": 100, "LVH": 100}, {"ASMI": 50, "AMI": 80}, {"LVH": 100}, {"IMI": 35}]},
        index=[1, 2, 3, 4],
    )
    groups = ho.statement_groups(np.array([1, 2, 3, 4]), meta)
    assert groups == {1: ["IMI"], 2: ["anterior MI"], 3: ["other"], 4: ["other"]}


def test_group_means_respect_minimum() -> None:
    frame = pd.DataFrame({"ecg_id": [1, 2, 3, 4], "A2": [0.6, 0.8, 0.7, np.nan]})
    groups = {1: ["IMI"], 2: ["IMI"], 3: ["other"], 4: ["IMI"]}
    table = ho.group_means({"a": frame}, groups, minimum=2)
    assert table["group"].tolist() == ["IMI"]
    assert table["records"].tolist() == [2]
    assert table["a"].iloc[0] == pytest.approx(0.7)


def test_independent_shift_of_equal_halves_is_zero() -> None:
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"patient_id": np.arange(40), "A2": rng.random(40)})
    shift = ho.independent_shift(frame, frame, draws=200)
    assert shift["value"] == pytest.approx(0.0)
    assert shift["ci_low"] < 0 < shift["ci_high"]


def test_map_difference() -> None:
    a = np.zeros((2, 12, 5000), np.float32)
    b = a.copy()
    assert ho.map_difference(a, b) == {"max_abs_difference": 0.0, "identical": True, "records_differing": 0}
    b[1, 3, 7] = 0.5
    assert ho.map_difference(a, b)["records_differing"] == 1


def test_label_rows_orientation() -> None:
    labels = np.zeros((3, 5000, 12))
    labels[2, 10:20, 4] = 1.0
    rows = ho.label_rows(labels, np.array([2, 0]))
    assert rows.shape == (2, 12, 5000)
    assert rows[0, 4, 10:20].all()
    assert rows[0].sum() == 10
    assert not rows[1].any()


def test_split_records_are_the_frozen_halves() -> None:
    holdout = ho.split_records("holdout")
    development = ho.split_records("development")
    assert len(holdout) == 201
    assert len(development) == 198
    assert not set(holdout["ecg_id"]) & set(development["ecg_id"])
    assert not set(holdout["patient_id"]) & set(development["patient_id"])
    assert 162 not in set(holdout["bench_index"]) | set(development["bench_index"])
    assert holdout["bench_index"].is_monotonic_increasing


def test_frozen_combiner_weights() -> None:
    if not (ho.OUT066 / "result.json").exists() or not (ho.OUT068 / "result.json").exists():
        pytest.skip("066 and 068 outputs not present")
    weights = ho.combiner_weights()
    names, values = weights["units_learned"]
    assert names == cb.UNIT_FEATURES
    assert values.shape == (24,)
    names, values = weights[ho.PRIMARY]
    assert "factor_soft25" in names
    assert values.shape == (30,)
    assert values[names.index("textbook")] == pytest.approx(1.0133, abs=1e-4)


def test_map_names() -> None:
    assert set(ho.CANDIDATES).isdisjoint(ho.REFERENCES)
    assert set(ho.MAPS) == set(ho.DEVELOPMENT_TABLES)
    assert set(ho.DEVELOPMENT_A2) <= set(ho.MAPS)
