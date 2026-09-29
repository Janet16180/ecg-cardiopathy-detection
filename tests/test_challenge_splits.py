"""Checks of the frozen Challenge record split."""

import numpy as np
import pandas as pd

from ecg_experiment.challenge_splits import (
    assign_splits,
    duplicate_groups,
    label_category,
    split_counts,
    split_sizes,
)


def test_duplicate_groups_join_records_through_any_hash():
    hashes = pd.DataFrame({"signal": ["a", "a", "b", "c", "d"], "window": ["", "x", "x", np.nan, ""]},
                          index=["R3", "R1", "R2", "R4", "R5"])
    groups = duplicate_groups(hashes)
    assert groups.to_dict() == {"R3": "R1", "R1": "R1", "R2": "R1", "R4": "R4", "R5": "R5"}


def test_equal_values_of_different_hash_kinds_do_not_join():
    hashes = pd.DataFrame({"signal": ["a", "b"], "window": ["", "a"]}, index=["R1", "R2"])
    assert duplicate_groups(hashes).tolist() == ["R1", "R2"]


def test_label_category_names_the_primary_label():
    assert label_category(pd.Series([1.0, 0.0, np.nan])).tolist() == ["positive", "negative", "undefined"]


def test_split_sizes_round_half_up_and_sum_to_the_count():
    assert split_sizes(10) == {"train": 6, "calibration": 2, "test": 2}
    assert split_sizes(3) == {"train": 1, "calibration": 1, "test": 1}
    assert split_sizes(1) == {"train": 1, "calibration": 0, "test": 0}
    assert all(sum(split_sizes(count).values()) == count for count in range(200))


def test_groups_stay_together_and_strata_are_split_in_proportion():
    records = [f"R{number:04d}" for number in range(1000)]
    groups = pd.Series(records, index=records)
    groups.iloc[1::10] = groups.iloc[0::10].to_numpy()
    strata = pd.Series(np.where(np.arange(1000) < 600, "a:positive", "b:negative"), index=records)
    splits = assign_splits(groups, strata, seed=7)
    assert splits.groupby(groups).nunique().max() == 1
    group_splits = splits.groupby(groups).first()
    group_strata = strata.loc[group_splits.index]
    for stratum, count in group_strata.value_counts().items():
        observed = group_splits[group_strata == stratum].value_counts().to_dict()
        assert observed == {name: size for name, size in split_sizes(count).items() if size}


def test_assignment_depends_only_on_records_strata_and_seed():
    records = [f"R{number}" for number in range(300)]
    groups = pd.Series(records, index=records)
    strata = pd.Series(["s"] * 300, index=records)
    first = assign_splits(groups, strata, seed=3)
    shuffled = assign_splits(groups.iloc[::-1], strata.iloc[::-1], seed=3)
    assert first.equals(shuffled.loc[first.index])
    assert not first.equals(assign_splits(groups, strata, seed=4))


def test_split_counts_by_source_split_and_label():
    rows = pd.DataFrame({"source": ["x", "x", "x", "y"], "split": ["train", "train", "test", "calibration"],
                         "primary": [1.0, np.nan, 0.0, 1.0]})
    counts = split_counts(rows)
    assert counts["x"]["train"] == {"records": 2, "positive": 1, "negative": 0, "undefined": 1}
    assert counts["x"]["calibration"]["records"] == 0
    assert counts["y"]["calibration"]["positive"] == 1
