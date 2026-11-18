"""Tests of the MLC-03 dataset v5 rules on small synthetic tables, and of the readers on local data."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecg_experiment import mlc03_dataset_v5 as mv
from ecg_experiment import mlc03_label_quality as lq
from ecg_experiment.mlc02_data import CONDITIONS, MANIFEST
from ecg_experiment.mlc03_measurements import LEADS

SOURCES = ("ptbxl", "ningbo", "chapman_shaoxing", "cpsc_2018", "cpsc_2018_extra", "sph")


def quiet_table(records: int = 12) -> pd.DataFrame:
    """Analysis-like table where no rule selects anything: sinus rhythm at 75 per minute, every label 0."""
    table = pd.DataFrame(
        {
            "record_key": [f"r{i}" for i in range(records)],
            "source": [SOURCES[i % len(SOURCES)] for i in range(records)],
            "role": ["train" if i % 2 else "validation" for i in range(records)],
            "duration_s": 10.0,
            "status_b": "ok",
            "sinus_measured": True,
            "hr": 75.0,
            "qrs": 96.0,
            "voltage_criterion": False,
        }
    )
    for name in CONDITIONS:
        table[f"b_{name}"] = 0.0
    for name in lq.EXTRA_LABELS:
        table[f"c_{name}"] = 0.0
    for lead in LEADS:
        table[f"flat_{lead}"] = False
    return table


def quiet_flags(table: pd.DataFrame) -> pd.DataFrame:
    """Part A flags without any defect."""
    flags = pd.DataFrame({"record_key": table["record_key"], "included": True}, index=table.index)
    for name in mv.SIGNAL_FLAGS:
        flags[name] = False
    for column in mv.LEAD_LOSS:
        flags[column] = ""
    return flags


def quiet_windows(table: pd.DataFrame) -> pd.DataFrame:
    """Window summaries of no long record."""
    return pd.DataFrame(
        columns=["record_key", "ectopy_first", "ectopy_later", "af_first", "af_later_windows"]
    )


def selected(table: pd.DataFrame, flags: pd.DataFrame, windows: pd.DataFrame) -> set[tuple[str, str, str]]:
    """Every (rule, condition, record) that a label rule selects."""
    masks = mv.label_rule_masks(mv.part_b_masks(table, flags, windows), table)
    return {
        (rule, condition, key)
        for (rule, condition), mask in masks.items()
        for key in table.loc[mask.to_numpy(), "record_key"]
    }


def test_a_quiet_table_selects_nothing():
    table = quiet_table()
    flags = quiet_flags(table)
    assert selected(table, flags, quiet_windows(table)) == set()
    assert not mv.record_rule_masks(table, flags).to_numpy().any()


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (
            lambda t, f: t.__setitem__("flat_V1", t["record_key"] == "r1"),
            {("C5", "complete_rbbb", "r1"), ("C5", "incomplete_rbbb", "r1"), ("C5", "lvh", "r1")},
        ),
        (
            lambda t, f: f.__setitem__("contact_loss__leads", np.where(t["record_key"] == "r1", "III", "")),
            {("C5", "lafb", "r1"), ("C5", "left_axis_deviation", "r1")},
        ),
        (
            lambda t, f: (
                t.__setitem__("b_sinus_tachycardia", np.where(t["record_key"] == "r2", 1.0, 0.0)),
                t.__setitem__("b_af_or_flutter", np.where(t["record_key"] == "r2", 1.0, 0.0)),
                t.__setitem__("hr", np.where(t["record_key"] == "r2", 110.0, 75.0)),
            ),
            {("C6", "sinus_tachycardia", "r2"), ("C6", "af_or_flutter", "r2")},
        ),
        (
            lambda t, f: (
                t.__setitem__("b_sinus_tachycardia", np.where(t["record_key"] == "r2", 1.0, 0.0)),
                t.__setitem__("c_supraventricular_tachycardia", np.where(t["record_key"] == "r2", 1.0, 0.0)),
                t.__setitem__("hr", np.where(t["record_key"] == "r2", 110.0, 75.0)),
            ),
            {("C6", "sinus_tachycardia", "r2")},
        ),
        (
            lambda t, f: t.__setitem__("c_paced_rhythm", np.where(t["record_key"] == "r3", 1.0, 0.0)),
            {("C7", name[2:], "r3") for name in lq.UNREADABLE},
        ),
        (
            lambda t, f: (
                t.__setitem__("b_complete_rbbb", np.where(t["record_key"].isin(["r1", "r2"]), 1.0, 0.0)),
                t.__setitem__("qrs", np.where(t["record_key"].isin(["r1", "r2"]), 100.0, 96.0)),
            ),
            {("C9", "complete_rbbb", "r2")},
        ),
        (
            lambda t, f: t.__setitem__("hr", np.where(t["record_key"].isin(["r0", "r4"]), 57.0, 75.0)),
            {("C10", "sinus_bradycardia", "r0"), ("C10", "sinus_bradycardia", "r4")},
        ),
        (
            lambda t, f: t.__setitem__("hr", np.where(t["record_key"].isin(["r1", "r5"]), 50.0, 75.0)),
            {("C11", "sinus_bradycardia", "r1")},
        ),
        (
            lambda t, f: (
                t.__setitem__("b_sinus_tachycardia", np.where(t["record_key"].isin(["r1", "r5"]), 1.0, 0.0)),
                t.__setitem__("hr", np.where(t["record_key"].isin(["r1", "r5"]), 90.0, 75.0)),
            ),
            {("C11", "sinus_tachycardia", "r1"), ("C11", "sinus_tachycardia", "r5")},
        ),
        (lambda t, f: t.__setitem__("voltage_criterion", t["record_key"] == "r6"), {("C12", "lvh", "r6")}),
    ],
)
def test_each_rule_touches_only_its_records_and_conditions(change, expected):
    table = quiet_table()
    flags = quiet_flags(table)
    change(table, flags)
    assert selected(table, flags, quiet_windows(table)) == expected


def test_later_finding_marks_only_positives_whose_finding_comes_after_ten_seconds():
    table = quiet_table()
    table.loc[[3, 9], "duration_s"] = 30.0
    table.loc[[3, 9], "b_premature_ventricular_complex"] = 1.0
    windows = pd.DataFrame(
        {
            "record_key": ["r3", "r9"],
            "ectopy_first": [0.0, 1.0],
            "ectopy_later": [2.0, 2.0],
            "af_first": [False, False],
            "af_later_windows": [1, 0],
        }
    )
    assert selected(table, quiet_flags(table), windows) == {("C8", "premature_ventricular_complex", "r3")}


def test_rate_negatives_need_sinus_evidence_by_code_and_are_not_read_at_sph():
    table = quiet_table()
    table["hr"] = 50.0
    found = selected(table, quiet_flags(table), quiet_windows(table))
    assert {key for rule, _, key in found if rule == "C11"} == set(
        table.loc[table["source"] != "sph", "record_key"]
    )
    table["sinus_measured"] = False
    assert selected(table, quiet_flags(table), quiet_windows(table)) == set()


def test_integer_storage_rules_skip_sph():
    table = quiet_table()
    flags = quiet_flags(table)
    flags["adc_rail"] = True
    flags["identity_violation"] = True
    masks = mv.record_rule_masks(table, flags)
    assert masks.loc[table["source"] == "sph", ["C3", "C4"]].to_numpy().sum() == 0
    assert masks.loc[table["source"] != "sph", ["C3", "C4"]].to_numpy().all()


def text_manifest(table: pd.DataFrame) -> pd.DataFrame:
    """A v4-like manifest read as text for the records of an analysis-like table, plus one excluded copy."""
    rows = table[["record_key", "source", "role"]].copy()
    rows["family"] = rows["source"]
    rows["origin"] = "x"
    rows["group_id"] = rows["record_key"]
    rows["raw_path"] = "data/" + rows["record_key"]
    rows["included"] = "True"
    rows["exclusion_reason"] = ""
    for name in CONDITIONS:
        rows[f"b_{name}"] = table[f"b_{name}"].map({1.0: "1", 0.0: "0"}).fillna("")
        rows[f"lik50_{name}"] = "False"
    copy = rows.iloc[[0]].assign(record_key="copy", included="False")
    return pd.concat([rows, copy], ignore_index=True)


def test_order_overlaps_and_no_positive_created():
    table = quiet_table()
    table.loc[0, "b_lvh"] = 1.0
    table.loc[1, "b_complete_rbbb"] = 1.0
    table.loc[2, "b_lvh"] = np.nan
    manifest = text_manifest(table)
    manifest.loc[1, "lik50_complete_rbbb"] = "True"
    flags = quiet_flags(table)
    flags.loc[0, ["missing_samples", "adc_rail"]] = True
    table.loc[[0, 1, 2], "flat_V1"] = True
    table.loc[[1, 2], "c_paced_rhythm"] = 1.0
    record_masks = mv.record_rule_masks(table, flags).assign(record_key=table["record_key"])
    label_masks = mv.label_rule_masks(mv.part_b_masks(table, flags, quiet_windows(table)), table)
    v5, flow = mv.apply_rules(manifest, record_masks, label_masks)
    assert v5.loc[0, "included"] == "False"
    assert v5.loc[0, "exclusion_reason"] == "C1"
    assert v5.loc[len(table), "included"] == "False"
    assert v5.loc[len(table), "exclusion_reason"] == ""
    c3 = flow[(flow["rule"] == "C3")][["selected", "already_excluded", "excluded"]].sum()
    assert c3.tolist() == [1, 1, 0]
    assert v5.loc[0, "b_lvh"] == "1", "labels of excluded records stay as in v4"
    lvh5 = flow[(flow["rule"] == "C5") & (flow["condition"] == "lvh")].sum(numeric_only=True)
    assert (lvh5["selected_positives"], lvh5["on_excluded_records"]) == (1, 1)
    assert v5.loc[1, "b_complete_rbbb"] == ""
    assert v5.loc[1, "lik50_complete_rbbb"] == "False"
    c7 = flow[(flow["rule"] == "C7") & (flow["condition"] == "complete_rbbb")].sum(numeric_only=True)
    assert (c7["selected_positives"], c7["selected_negatives"], c7["already_undefined"]) == (1, 1, 2)
    assert (c7["undefined_positives"], c7["undefined_negatives"]) == (0, 0)
    assert v5.loc[2, "b_lvh"] == "", "an undefined label stays undefined"
    assert v5.loc[1, "rule_C7__complete_rbbb"] == "True"
    assert v5.loc[0, "rule_C1"] == "True"
    result = mv.compare_manifests(manifest, v5[manifest.columns])
    for check in (
        "only_defined_to_undefined",
        "identity_unchanged",
        "changes_only_in_rule_columns",
        "included_only_removed",
    ):
        assert result[check], check
    untouched = ~table["record_key"].isin(["r0", "r1", "r2"])
    columns = list(manifest.columns)
    assert (
        (
            manifest.loc[untouched.to_numpy().nonzero()[0], columns]
            == v5.loc[untouched.to_numpy().nonzero()[0], columns]
        )
        .all()
        .all()
    )


def test_a_mask_on_an_excluded_v4_record_is_refused():
    table = quiet_table(4)
    manifest = text_manifest(table)
    manifest.loc[0, "included"] = "False"
    flags = quiet_flags(table)
    record_masks = mv.record_rule_masks(table, flags).assign(record_key=table["record_key"])
    with pytest.raises(ValueError, match="not included"):
        mv.apply_rules(manifest, record_masks, {})


def test_support_counts_states_per_role_and_source():
    table = quiet_table(6)
    table.loc[0, "b_lvh"] = 1.0
    table.loc[1, "b_lvh"] = np.nan
    counts = mv.support(text_manifest(table))
    row = counts[
        (counts["label"] == "b_lvh") & (counts["unit"] == "ptbxl") & (counts["role"] == "validation")
    ]
    assert row[["positive", "negative", "undefined"]].iloc[0].tolist() == [1, 0, 0]


def test_the_builder_refuses_an_existing_output(tmp_path: Path):
    from scripts.experiments.run_mlc03_build_v5 import build

    with pytest.raises(FileExistsError):
        build(1, tmp_path, ["test"])
    (tmp_path / "v5.staging").mkdir()
    with pytest.raises(FileExistsError):
        build(1, tmp_path / "v5", ["test"])


@pytest.mark.skipif(not MANIFEST.exists(), reason="the MLC-01 v4 manifest is local only")
def test_the_record_reader_equals_part_b_on_open_roles():
    mine = mv.load_records(mv.OPEN_ROLES)
    theirs = lq.load_open_records()
    pd.testing.assert_frame_equal(mine, theirs)
