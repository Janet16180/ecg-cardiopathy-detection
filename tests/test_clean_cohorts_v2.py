"""Checks of the v2 cohort ordering, overlap removal and pending resolution."""

import pandas as pd

from ecg_experiment.clean_cohorts_v2 import (
    interleave,
    remove_overlaps,
    resolve_pending,
    seeded_key,
    source_order,
)


def rows(names, flags=None, source="a", hashes=None):
    return pd.DataFrame({
        "record_id": names, "patient_id": "", "source": source,
        "review_flags": flags or [""] * len(names),
        "signal_sha256": hashes or [f"h{name}" for name in names],
        "quality_status": "passed",
    })


def test_seeded_key_depends_only_on_value_and_seed():
    first = seeded_key(pd.Series(["x", "y"]), 1)
    assert first.tolist() == seeded_key(pd.Series(["y", "x"]), 1).tolist()[::-1]
    assert first.tolist() != seeded_key(pd.Series(["x", "y"]), 2).tolist()


def test_flagged_records_come_last():
    ordered = source_order(rows(["1", "2", "3", "4"], flags=["x", "", "x", ""]), 0)
    assert ordered["review_flags"].tolist() == ["", "", "x", "x"]


def test_interleave_keeps_every_prefix_proportional():
    parts = {"a": rows([f"a{i}" for i in range(30)]), "b": rows([f"b{i}" for i in range(10)], source="b")}
    merged = interleave(parts)
    first_eight = merged["source"].iloc[:8].value_counts()
    assert first_eight["a"] == 6
    assert first_eight["b"] == 2


def test_overlaps_drop_ptbxl_copies_heldout_patients_and_repeats():
    parts = [rows(["p1"], source="ptbxl", hashes=["h"]), rows(["n1"], hashes=["h"]),
             rows(["n2", "n3"], hashes=["k", "k"]), rows(["p2"], source="ptbxl")]
    frame = pd.concat(parts, ignore_index=True)
    frame.loc[4, "patient_id"] = "ptbxl:9"
    frame["order"] = range(len(frame))
    kept = remove_overlaps(frame, heldout_hashes={"h"}, heldout_ids=set(), heldout_patients={"ptbxl:9"})
    assert kept["record_id"].tolist() == ["p1", "n2"]
    assert kept["order"].tolist() == [0, 1]


def test_failed_pending_records_leave_and_later_ones_move_up():
    frame = rows(["a", "m1", "m2", "m3"], hashes=["ha", "", "", ""])
    frame.loc[1:, "quality_status"] = "pending"
    results = pd.DataFrame({"record_id": ["m1", "m2"], "exclusion_reasons": ["", "flat_segment"],
                            "review_flags": ["", ""], "signal_sha256": ["h1", "h2"]})
    resolved = resolve_pending(frame, results)
    assert resolved["record_id"].tolist() == ["a", "m1", "m3"]
    assert resolved["quality_status"].tolist() == ["passed", "passed", "pending"]
    assert resolved.loc[1, "signal_sha256"] == "h1"
