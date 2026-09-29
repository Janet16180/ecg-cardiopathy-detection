"""Checks of the Ningbo family duplicate rule and demographics."""

import math

import numpy as np
import pandas as pd

from ecg_experiment.eda.ningbo import constant_run, edge_zero_runs, fingerprint
from ecg_experiment.ningbo import clean_age, duplicate_status, male


def test_age_zero_and_missing_become_nan():
    ages = clean_age(pd.Series(["0", "NaN", "45", "4"]))
    assert math.isnan(ages[0])
    assert math.isnan(ages[1])
    assert ages[2:].tolist() == [45.0, 4.0]


def test_sex_is_a_float_indicator():
    values = male(pd.Series(["Male", "Female", "Unknown"]))
    assert values[:2].tolist() == [1.0, 0.0]
    assert math.isnan(values[2])


def test_duplicates_keep_the_lowest_name_unless_labels_or_sex_conflict():
    frame = pd.DataFrame({
        "signal_sha256": ["a", "a", "b", "c", "c", "d", "d", "e"],
        "label_key": ["0", "0", "1", "1", "0", "1", "1", "1"],
        "male": [1.0, np.nan, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
    }, index=["JS20000", "JS00005", "JS00006", "JS10001", "JS30000", "JS40000", "JS40001", "JS50000"])
    result = duplicate_status(frame)
    assert result["duplicate_status"].to_dict() == {
        "JS20000": "dropped_copy", "JS00005": "kept", "JS00006": "unique", "JS10001": "dropped_conflict",
        "JS30000": "dropped_conflict", "JS40000": "dropped_conflict", "JS40001": "dropped_conflict",
        "JS50000": "unique"}
    assert result.loc["JS20000", "duplicate_of"] == "JS00005"
    assert result.loc["JS00006", "duplicate_of"] == ""


def test_constant_run_and_edge_zeros():
    lead = np.array([1.0, 0.0, 0.0, 0.0, 2.0, 2.0])
    assert constant_run(lead) == (1, 3, 0.0)
    signal = np.zeros((6, 2))
    signal[2:4, 1] = 1.0
    assert edge_zero_runs(signal) == (2, 2)
    assert edge_zero_runs(np.zeros((5, 2))) == (5, 5)


def test_fingerprint_ignores_scale_and_offset():
    rng = np.random.default_rng(0)
    signal = rng.normal(size=(5000, 12))
    assert np.allclose(fingerprint(signal), fingerprint(3 * signal + 1), atol=1e-2)
    assert not fingerprint(np.zeros((5000, 12))).any()
