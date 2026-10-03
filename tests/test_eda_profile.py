"""Checks of the profiling and PTB-XL cleaning helpers."""

import numpy as np
import pandas as pd

from ecg_experiment.eda.profile import cramers_v, frequencies, numeric_summary, standardize_leads
from ecg_experiment.eda.ptbxl_cleaning import coded_category, fill_by_group, group_rare


def test_cramers_v_is_one_for_identical_and_zero_for_independent_columns():
    first = pd.Series(["a", "b"] * 200)
    assert cramers_v(first, first) > 0.99
    assert cramers_v(first, pd.Series(["x"] * 200 + ["y"] * 200)) == 0.0


def test_numeric_summary_counts_tukey_outliers():
    frame = pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, 100.0, np.nan]})
    assert numeric_summary(frame, ["value"]).loc["value", "outliers"] == 1


def test_frequencies_keeps_missing_and_groups_the_tail():
    table = frequencies(pd.Series(["a", "a", "b", "c", None]), top=2)
    assert table["records"].sum() == 5
    assert "(other)" in table.index


def test_group_rare_and_fill_by_group():
    assert group_rare(pd.Series(["a"] * 99 + ["b"]), 0.05).tolist()[-1] == "other"
    values = pd.Series([1.0, np.nan, 10.0, np.nan])
    groups = [pd.Series(["x", "x", "y", "y"])]
    assert fill_by_group(values, groups).tolist() == [1.0, 1.0, 10.0, 10.0]
    assert coded_category(pd.Series([2.0, np.nan])).tolist() == ["2", "unknown"]


def test_standardize_leads_gives_unit_variance_per_lead():
    signal = np.random.default_rng(0).normal(3.0, 5.0, size=(500, 2))
    assert np.allclose(standardize_leads(signal).std(axis=0), 1.0)
