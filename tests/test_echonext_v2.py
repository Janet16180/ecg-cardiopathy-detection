"""Checks of EchoNext rows v2: the use split, the release join and the lead-order test."""

import numpy as np
import pandas as pd
import pytest

from ecg_experiment import echonext_v2 as v2


def v1_rows(reasons):
    count = len(reasons)
    return pd.DataFrame({
        "ecg_key": [str(i) for i in range(count)],
        "patient_key": ["p0", "p0", *[f"p{i}" for i in range(2, count)]],
        "split": ["train"] * (count - 1) + ["val"], "row": [str(i) for i in range(count - 1)] + ["0"],
        "exclusion_reasons": reasons, "use": [reason == "" for reason in reasons],
        v2.COMPOSITE: ["1", "0", *["1"] * (count - 2)],
    })


def release(rows, recent):
    return pd.DataFrame({"ecg_key": [*rows["ecg_key"], "t"], "split": [*rows["split"], "test"],
                         "most_recent_ecg": [*recent, "1"]})


def test_training_excludes_every_reason_and_evaluation_only_missing_signal():
    rows = v1_rows(["", "noise_dominated", "flat_segment", "constant_lead;flat_segment", ""])
    table = v2.rows_v2(rows, release(rows, ["0", "1", "1", "1", "1"]))
    assert table["use_training"].tolist() == [True, False, False, False, True]
    assert table["use_evaluation"].tolist() == [True, True, False, False, True]
    assert table["most_recent_ecg"].tolist() == ["0", "1", "1", "1", "1"]
    assert table["ecgs_in_split"].tolist() == [2, 2, 1, 1, 1]
    assert "use" not in table


def test_rows_must_be_exactly_the_open_release_ecgs():
    rows = v1_rows(["", "", ""])
    other = release(rows, ["1", "1", "1"])
    other.loc[0, "split"] = "no_split"
    with pytest.raises(ValueError, match="train and val ECGs"):
        v2.rows_v2(rows, other)


def test_unknown_reasons_and_inconsistent_use_are_refused():
    rows = v1_rows(["", "amplitude", ""])
    with pytest.raises(ValueError, match="Unknown exclusion reason"):
        v2.rows_v2(rows, release(rows, ["1", "1", "1"]))
    rows = v1_rows(["", "", ""])
    rows.loc[0, "use"] = False
    with pytest.raises(ValueError, match="use flag"):
        v2.rows_v2(rows, release(rows, ["1", "1", "1"]))


def test_one_per_patient_uses_the_most_recent_training_ecg():
    rows = v1_rows(["", "", "", ""])
    table = v2.rows_v2(rows, release(rows, ["0", "1", "1", "1"]))
    shares = v2.one_per_patient(table)
    assert shares["train_all"] == pytest.approx(2 / 3)
    assert shares["train_most_recent"] == 0.5
    assert shares["train_most_recent_ecgs"] == 2


def test_lead_order_check_accepts_standard_leads_and_rejects_swapped_ones():
    rng = np.random.default_rng(0)
    lead_i, lead_ii = rng.normal(size=(2, 50, 2500))
    lead_iii = lead_ii - lead_i
    limb = [lead_i, lead_ii, lead_iii, -(lead_i + lead_ii) / 2, lead_i - lead_ii / 2, lead_ii - lead_i / 2]
    signals = np.stack([*limb, *rng.normal(size=(6, 50, 2500))], axis=1)
    assert all(item["signs_match"] and item["r2"] > 0.99
               for item in v2.check_lead_order(signals)["limb_relations"].values())
    swapped = signals[:, [0, 1, 2, 4, 3, 5, 6, 7, 8, 9, 10, 11]]
    assert not all(item["signs_match"] for item in v2.check_lead_order(swapped)["limb_relations"].values())
