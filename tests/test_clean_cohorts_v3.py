"""Checks of the v3 Challenge evaluation exclusion and tier sizing."""

import pandas as pd
import pytest

from ecg_experiment.clean_cohorts_v3 import (
    SPLIT_COLUMNS,
    challenge_references,
    exclude_challenge_evaluation,
    read_splits,
)
from ecg_experiment.cohort_tiers import tier_sizes


def splits(records, groups, dup_groups=None, hashes=None):
    return pd.DataFrame({
        "source": "ningbo", "record": records, "split": groups,
        "duplicate_group": dup_groups or records,
        "signal_sha256": "", "window_sha256": hashes or [f"w{r}" for r in records],
    })


def curated(ids, sources, hashes):
    return pd.DataFrame({"record_id": ids, "source": sources, "signal_sha256": hashes})


def test_references_keep_only_test_and_calibration_groups():
    known, held, hashes = challenge_references(splits(["a", "b", "c"], ["train", "test", "calibration"]))
    assert known == {"ningbo:a", "ningbo:b", "ningbo:c"}
    assert held == {"ningbo:b", "ningbo:c"}
    assert hashes == {"wb", "wc"}


def test_references_refuse_a_duplicate_group_across_splits():
    with pytest.raises(ValueError, match="spans split groups"):
        challenge_references(splits(["a", "b"], ["train", "test"], dup_groups=["g", "g"]))


def test_exclusion_drops_evaluation_ids_and_their_waveforms_only():
    known, held, hashes = challenge_references(splits(["a", "b"], ["train", "test"]))
    rows = curated(["ningbo:a", "ningbo:b", "ptbxl:1", "ptbxl:2"], ["ningbo", "ningbo", "ptbxl", "ptbxl"],
                   ["wa", "wb", "wb", "p2"])
    kept = exclude_challenge_evaluation(rows, known, held, hashes)
    assert kept["record_id"].tolist() == ["ningbo:a", "ptbxl:2"]


def test_exclusion_refuses_a_challenge_record_missing_from_the_split():
    known, held, hashes = challenge_references(splits(["a"], ["train"]))
    with pytest.raises(ValueError, match="missing from the Challenge split"):
        exclude_challenge_evaluation(curated(["ningbo:z"], ["ningbo"], ["wz"]), known, held, hashes)


def test_tier_sizes_add_a_max_tier_when_the_largest_is_out_of_reach():
    sizes = {"small": 2, "large": 10}
    assert tier_sizes(sizes, 12) == {"small": 2, "large": 10}
    assert tier_sizes(sizes, 7) == {"small": 2, "max": 7}


def test_split_is_read_without_its_label_columns(tmp_path):
    path = tmp_path / "rows.csv"
    splits(["a"], ["test"]).assign(MI="1.0", STTC="0.0").to_csv(path, index=False)
    assert tuple(read_splits(path).columns) == SPLIT_COLUMNS
