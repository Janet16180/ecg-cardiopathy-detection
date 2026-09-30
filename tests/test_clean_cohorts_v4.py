"""Checks of the v4 EchoNext block: eligibility, position in the order and the comparison with v3."""

import numpy as np
import pandas as pd
import pytest

from ecg_experiment import clean_cohorts_v4 as v4
from ecg_experiment.clean_cohorts_v2 import COLUMNS
from ecg_experiment.public_sources import signal_sha256


def rows(source, count, prefix=None):
    prefix = prefix or source
    frame = pd.DataFrame(dict.fromkeys(COLUMNS, ""), index=range(count))
    frame["record_id"] = [f"{prefix}:{i}" for i in range(count)]
    frame["source"] = source
    frame["signal_sha256"] = [f"h{prefix}{i}" for i in range(count)]
    return frame


def order(curated, echonext, mimic, curated_tier):
    parts = pd.concat([rows("ptbxl", curated), rows("ningbo", 2)], ignore_index=True)
    return v4.global_order(parts, rows("echonext", echonext), rows("mimic", mimic), rows("code15", 2),
                           rows("mimic", 1, "pending"), seed=1, curated_tier=curated_tier)


def test_echonext_follows_curated_and_takes_the_top_up_first():
    ordered = order(curated=3, echonext=4, mimic=5, curated_tier=8)
    assert ordered["block"].tolist() == (["curated"] * 5 + ["echonext"] * 4 + ["code15"] * 2
                                         + ["mimic_local"] * 5 + ["mimic_pending"])
    assert ordered["order"].tolist() == list(range(len(ordered)))


def test_mimic_top_up_completes_only_what_echonext_leaves_short():
    ordered = order(curated=3, echonext=2, mimic=5, curated_tier=10)
    assert (ordered["block"] == "mimic_top_up").sum() == 3
    assert ordered["block"].tolist().index("mimic_top_up") > ordered["block"].tolist().index("echonext")


def test_training_patients_must_not_appear_in_another_split():
    clean = pd.DataFrame({"patient_key": ["a", "a", "b", "c"], "split": ["train", "train", "val", "test"]})
    v4.check_patients_disjoint(clean)
    shared = pd.DataFrame({"patient_key": ["a", "a"], "split": ["train", "no_split"]})
    with pytest.raises(ValueError, match="another split"):
        v4.check_patients_disjoint(shared)


def test_only_usable_training_rows_become_candidates(tmp_path, monkeypatch):
    monkeypatch.setattr(v4, "to_stored", lambda path: "cache/train.npy")
    signals = np.random.default_rng(0).normal(size=(3, 12, 2500)).astype(np.float32)
    np.save(tmp_path / "train.npy", signals)
    pd.DataFrame({
        "ecg_key": ["10", "11", "12", "13"], "patient_key": ["p1", "p1", "p2", "p3"],
        "split": ["train", "train", "train", "val"], "row": ["0", "1", "2", "0"],
        "use": ["True", "False", "True", "True"], "age_at_ecg": ["22", "40", "70", "50"],
        "sex": ["female", "male", "male", "female"],
    }).to_csv(tmp_path / "rows.csv", index=False)
    found = v4.echonext_rows(tmp_path)
    assert found["record_id"].tolist() == ["echonext:10", "echonext:12"]
    assert found["patient_id"].tolist() == ["echonext:p1", "echonext:p2"]
    assert found["signal_sha256"].tolist() == [signal_sha256(signals[0]), signal_sha256(signals[2])]
    assert found["male"].tolist() == [0.0, 1.0]
    assert not found["label_available"].any()
    assert list(found.columns) == list(COLUMNS)


def as_strings(frame):
    return frame.astype(str).reset_index(drop=True)


def test_comparison_accepts_v3_plus_echonext_and_refuses_a_change():
    previous = as_strings(order(curated=3, echonext=0, mimic=5, curated_tier=8))
    current = as_strings(order(curated=3, echonext=4, mimic=5, curated_tier=8))
    expected = {"curated_rows_identical": 5, "records_shared_with_v3": 13}
    assert v4.compare_with_v3(current, previous) == expected
    moved = current.copy()
    moved.loc[[10, 11], "record_id"] = moved.loc[[11, 10], "record_id"].to_numpy()
    with pytest.raises(ValueError, match="differs from v3 for code15"):
        v4.compare_with_v3(moved, previous)
    changed = current.copy()
    changed.loc[0, "signal_sha256"] = "other"
    with pytest.raises(ValueError, match="curated block"):
        v4.compare_with_v3(changed, previous)
