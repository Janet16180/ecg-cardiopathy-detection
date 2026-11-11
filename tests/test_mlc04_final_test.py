"""Tests of MLC-04 on synthetic data: the role guard, the marker, the metric, the readings and the tables."""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecg_experiment import mlc04_data as md
from ecg_experiment import mlc04_evaluation as me
from ecg_experiment import mlc04_models as mm
from ecg_experiment.mlc02_data import CHALLENGE_SOURCES, CONDITIONS
from ecg_experiment.mlc02_evaluation import selection_metric

SPLIT_TABLE = md.ROOT / "outputs/cohort25_recovery_20261028/challenge/full/rows.csv"


def synthetic_rows() -> pd.DataFrame:
    """Identity columns of a small split table with every group."""
    names = ["A1", "A2", "A3", "A4", "JS13118", "A6"]
    return pd.DataFrame(
        {
            "source": ["ningbo"] * 6,
            "family": ["chapman_ningbo"] * 6,
            "record": names,
            "path": [f"training/ningbo/g1/{name}" for name in names],
            "split": ["train", "calibration", "test", "calibration", "calibration", "test"],
        }
    )


def synthetic_set(records: int = 600, seed: int = 0, signal: float = 2.0) -> dict[str, np.ndarray]:
    """Labels, sources, keys and an informative score for every Challenge source."""
    rng = np.random.default_rng(seed)
    labels = (rng.random((records, len(CONDITIONS))) < 0.3).astype(float)
    labels[rng.random(labels.shape) < 0.05] = np.nan
    sources = np.array([CHALLENGE_SOURCES[i % len(CHALLENGE_SOURCES)] for i in range(records)])
    score = 1 / (1 + np.exp(-(signal * np.nan_to_num(labels) + rng.normal(size=labels.shape))))
    return {
        "labels": labels,
        "source": sources,
        "record_key": np.array([f"r{i}" for i in range(records)]),
        "score": score,
    }


def test_check_role_refuses_unknown_and_closed_roles(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="role must be"):
        md.check_role("test")
    assert md.check_role("validation") == "validation"

    def refuse() -> None:
        raise PermissionError("not frozen")

    monkeypatch.setattr(md, "freeze_committed", refuse)
    with pytest.raises(PermissionError, match="not frozen"):
        md.check_role("challenge_test")


def test_freeze_record_must_exist(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="does not exist"):
        md.freeze_committed(tmp_path / "missing.md")


def test_split_identity_keeps_one_group_and_closes_the_others() -> None:
    kept, closed = md.split_identity(synthetic_rows(), "calibration")
    assert kept["record"].tolist() == ["A2", "A4"]
    assert closed == {
        "ningbo:training/ningbo/g1/A1",
        "ningbo:training/ningbo/g1/A3",
        "ningbo:training/ningbo/g1/A6",
    }
    kept, closed = md.split_identity(synthetic_rows(), "test")
    assert kept["record"].tolist() == ["A3", "A6"]
    assert "ningbo:training/ningbo/g1/A2" in closed
    with pytest.raises(ValueError, match="unknown split"):
        md.split_identity(synthetic_rows(), "train")


def test_marker_refuses_a_second_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(md, "freeze_committed", lambda: {"code_commit": "abc"})
    monkeypatch.setattr(md, "git_state", lambda: {"commit": "abc", "modified_tracked": []})
    directory = tmp_path / "score"
    with pytest.raises(FileNotFoundError):
        md.open_marker(directory, rerun_after_crash=True)
    marker = md.open_marker(directory, rerun_after_crash=False)
    assert len(marker["attempts"]) == 1
    with pytest.raises(FileExistsError, match="started"):
        md.open_marker(directory, rerun_after_crash=False)
    (directory / "outputs.npz").write_text("partial")
    with pytest.raises(FileExistsError, match="partial"):
        md.open_marker(directory, rerun_after_crash=True)
    (directory / "outputs.npz").unlink()
    assert len(md.open_marker(directory, rerun_after_crash=True)["attempts"]) == 2
    finished = json.loads((directory / "marker.json").read_text()) | {"completed_unix": 1.0}
    (directory / "marker.json").write_text(json.dumps(finished))
    for rerun in (False, True):
        with pytest.raises(FileExistsError, match="cannot be repeated"):
            md.open_marker(directory, rerun_after_crash=rerun)


def test_primary_reading_boundaries() -> None:
    bound = me.VALIDATION_METRIC - me.CONFIRMATION_MARGIN
    assert me.primary_reading(bound + 1e-6, bound + 0.02) == "confirmed"
    assert me.primary_reading(bound - 0.02, bound - 1e-6) == "not confirmed"
    assert me.primary_reading(bound - 0.01, bound + 0.01) == "undecided"
    assert me.primary_reading(bound, bound + 0.01) == "undecided"


def test_source_metric_equals_the_selection_metric_with_its_entry_rule() -> None:
    data = synthetic_set()
    found = me.source_metric(data["labels"], data["score"], data["source"], me.VALIDATION_MINIMUM)
    expected = selection_metric(data["labels"], data["score"], data["source"])
    assert found["metric"] == expected["selection_metric"]
    assert found["macro_by_source"] == expected["macro_auroc_by_source"]


def test_source_metric_leaves_out_sources_without_an_entering_condition() -> None:
    data = synthetic_set()
    labels = data["labels"].copy()
    labels[data["source"] == "cpsc_2018_extra"] = np.nan
    found = me.source_metric(labels, data["score"], data["source"], me.VALIDATION_MINIMUM)
    assert found["excluded_sources"] == ["cpsc_2018_extra"]
    assert set(found["macro_by_source"]) == set(CHALLENGE_SOURCES) - {"cpsc_2018_extra"}
    with pytest.raises(ValueError, match="outside the Challenge"):
        me.source_metric(
            labels, data["score"], np.where(data["source"] == "ningbo", "ptbxl", data["source"]), 20
        )


def test_compare_is_paired_and_metric_draws_match_the_observed_metric() -> None:
    data = synthetic_set(records=500)
    resamples = me.source_resamples(data["labels"], data["record_key"], data["source"], 20, draws=50, seed=1)
    first = me.source_boot(data["labels"], data["score"], data["source"], resamples)
    noisy = np.clip(
        data["score"] + np.random.default_rng(2).normal(scale=0.2, size=data["score"].shape), 0, 1
    )
    second = me.source_boot(data["labels"], noisy, data["source"], resamples)
    summary = me.metric_summary(first)
    assert summary["value"] == pytest.approx(
        me.source_metric(data["labels"], data["score"], data["source"], 20)["metric"]
    )
    same = me.compare(first, first)
    assert same["difference"] == 0.0
    assert same["p_value"] == 1.0
    better = me.compare(first, second)
    assert better["difference"] > 0
    assert better["ci"][0] > 0
    other = me.source_resamples(data["labels"], data["record_key"], data["source"], 20, draws=50, seed=3)
    with pytest.raises(ValueError, match="same resamples"):
        me.compare(first, me.source_boot(data["labels"], noisy, data["source"], other))


def comparison(difference: float, p_value: float, high: float = 1.0) -> dict:
    """A ``compare``-like record."""
    return {"difference": difference, "p_value": p_value, "ci": [difference - 0.01, high]}


def test_secondary_readings_use_holm_and_the_key_margin() -> None:
    readings = me.secondary_readings(
        comparison(0.01, 0.001),
        comparison(0.005, 0.2, high=0.01),
        comparison(-0.002, 0.2),
        comparison(-0.004, 0.001),
    )
    assert readings["holm"]["cleaning"] == pytest.approx(0.003)
    assert readings["holm"]["larger_model"] == pytest.approx(0.2)
    assert readings["cleaning"]["reading"] == "cleaning helps"
    assert readings["larger_model"] == "no difference shown"
    assert readings["key"] == "equivalent"
    blocked = me.secondary_readings(
        comparison(0.01, 0.001),
        comparison(-0.01, 0.01, high=-0.001),
        comparison(0.01, 0.001),
        comparison(-0.02, 0.001),
    )
    assert blocked["cleaning"]["reading"] == "no gain from cleaning shown"
    assert blocked["larger_model"] == "H-All above P0"
    assert blocked["key"] == "clearly worse"


def test_k6_ranks_leave_auroc_unchanged() -> None:
    data = synthetic_set(records=300)
    raw = 10 * (data["score"] - 0.5)
    ranked = me.as_probability("K6", raw)
    assert ((ranked > 0) & (ranked < 1)).all()
    first = me.source_metric(data["labels"], ranked, data["source"], 20)["metric"]
    assert first == pytest.approx(
        me.source_metric(data["labels"], data["score"], data["source"], 20)["metric"]
    )


def test_key_scores_refuse_a_source_without_train_mean() -> None:
    rng = np.random.default_rng(0)
    model = {
        "key": rng.normal(size=(len(CONDITIONS), 4)),
        "means": {"ningbo": np.zeros(4)},
        "scaler": {"mean": np.zeros(4), "scale": np.ones(4)},
    }
    x = rng.normal(size=(3, 4))
    scores = mm.key_scores(model, x, np.array(["ningbo"] * 3))
    np.testing.assert_allclose(scores, x @ model["key"].T)
    with pytest.raises(KeyError, match="georgia"):
        mm.key_scores(model, x, np.array(["ningbo", "georgia", "ningbo"]))


def test_reliability_bins_cover_every_defined_label() -> None:
    data = synthetic_set(records=200)
    table = me.reliability(data["labels"], data["score"], data["source"])
    defined = (~np.isnan(data["labels"])).sum()
    assert table["records"].sum() == defined
    assert table["bin_low"].min() >= 0.0
    assert table["bin_high"].max() <= 1.0


def test_method_a_derives_part_b_quantities() -> None:
    values = np.full((2, 130), 1.0)
    frame = md.method_a(np.array(["a", "b"]), values, pd.Series(["ok", "ok"]))
    assert frame["axis_a"].tolist() == [45.0, 45.0]
    assert frame["rr_nrmssd_a"].tolist() == [1.0, 1.0]
    assert frame["a_status"].tolist() == ["ok", "ok"]


def test_cell_mismatches_counts_cells() -> None:
    first = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": ["x", "y", "z"]})
    second = pd.DataFrame({"a": [1.0, np.nan, 4.0], "b": ["x", "y", "z"], "c": [1, 2, 3]})
    assert md.cell_mismatches(first, second) == {"a": 1, "c": -1}


def test_json_ready_turns_nan_into_none() -> None:
    found = me.json_ready({"a": np.float64(math.nan), "b": np.arange(2), "c": (np.int64(3),)})
    assert found == {"a": None, "b": [0, 1], "c": [3]}


@pytest.mark.skipif(not SPLIT_TABLE.exists(), reason="local split table not available")
def test_validation_role_has_the_calibration_groups() -> None:
    rows, closed = md.role_rows("validation")
    counts = rows.groupby("source").size().to_dict()
    assert counts == {
        "ningbo": 6979,
        "chapman_shaoxing": 2050,
        "georgia": 2066,
        "cpsc_2018": 1375,
        "cpsc_2018_extra": 692,
    }
    assert len(closed) == 52662


def test_screening_label_columns_are_withheld_on_the_test_groups_only() -> None:
    frame = pd.DataFrame(
        {"a_MI": [1.0, 0.0], "normal": [0.0, 1.0], "b_lvh": [1.0, 0.0], "c_paced_rhythm": [0.0, 1.0]}
    )
    kept = md.withhold_screening_labels(frame, "validation")
    pd.testing.assert_frame_equal(kept, frame)
    withheld = md.withhold_screening_labels(frame, "challenge_test")
    assert withheld[["a_MI", "normal"]].isna().all().all()
    pd.testing.assert_frame_equal(withheld[["b_lvh", "c_paced_rhythm"]], frame[["b_lvh", "c_paced_rhythm"]])
