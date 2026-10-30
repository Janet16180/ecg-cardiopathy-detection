"""Strict-age derivation cannot substitute older records or weaken frozen admission rules."""

import hashlib

import pandas as pd
import pytest

from ecg_experiment.strict_age_cohorts import derived_tables


def test_strict_age_membership_has_no_older_minor_unknown_or_censored_fallback():
    rows = pd.DataFrame({
        "record_id": ["young_negative", "young_positive", "older", "minor", "missing", "censored"],
        "age": [18, 25, 26, 17, float("nan"), 90],
        "age_interpretation": ["exact", "exact", "exact", "exact", "missing", "right_censored_over_89"],
        "target": [0, 1, 1, 0, 0, 1], "label_evidence_rank": [0, 0, 1, 1, 1, 1],
    }).assign(source="synthetic", split="train", quality_passed=True, label_evidence="human")
    rows["signal_sha256"] = rows["record_id"].map(lambda value: hashlib.sha256(value.encode()).hexdigest())
    young, control = derived_tables(rows, 42)
    assert set(young.record_id) == {"young_negative", "young_positive"}
    assert len(young) == len(control) == 2
    assert young.target.tolist() == rows.set_index("record_id").loc[young.record_id, "target"].tolist()
    pd.testing.assert_frame_equal(young, derived_tables(rows.sample(frac=1, random_state=3), 42)[0])
    with pytest.raises(ValueError, match="both original classes"):
        derived_tables(rows[rows.record_id != "young_positive"], 42)


def test_general_control_exactly_matches_source_evidence_and_binary_target_counts():
    rows = pd.DataFrame({
        "record_id": ["a_young", "b_young", "a_old", "b_old", "a_other_label", "b_other_rank"],
        "age": [19, 23, 65, 45, 60, 40], "age_interpretation": "exact",
        "target": [0, 1, 0, 1, 1, 1], "label_evidence_rank": [0, 1, 0, 1, 0, 0],
        "source": ["a", "b", "a", "b", "a", "b"],
    }).assign(split="train", quality_passed=True, label_evidence="human")
    rows["signal_sha256"] = rows["record_id"].map(lambda value: hashlib.sha256(value.encode()).hexdigest())
    young, control = derived_tables(rows, 42)
    strata = ["source", "label_evidence_rank", "target"]
    pd.testing.assert_series_equal(young.groupby(strata).size(), control.groupby(strata).size())
    assert not {"a_other_label", "b_other_rank"} & set(control.record_id)
