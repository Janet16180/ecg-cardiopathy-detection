"""Synthetic tests for historical cohort completeness and the additive execution gate."""

import json

import numpy as np
import pandas as pd
import pytest

from ecg_experiment import cpu_screening041_v2 as amended


def cohort_tables():
    table = pd.DataFrame(
        {
            "ecg_id": np.arange(1, 7),
            "patient_id": np.arange(101, 107),
            "strat_fold": [1, 1, 8, 8, 9, 9],
            "target": [0, 1, 0, 1, 0, 1],
            "NORM": [1, 0, 1, 0, 1, 0],
            "MI": [0, 1, 0, 1, 0, 1],
            "STTC": 0,
            "CD": 0,
            "HYP": 0,
        }
    )
    metadata = table.set_index("ecg_id")[["patient_id", "strat_fold"]].copy()
    metadata["scp_codes"] = [{"N": 100} if y == 0 else {"I": 100} for y in table.target]
    metadata.loc[7] = [107, 2, {"I": 100}]
    metadata.loc[8] = [108, 3, {"SR": 100}]
    quality = pd.DataFrame(
        {
            "ecg_id": np.arange(1, 9),
            "status": ["ok", "sin_latidos", "ok", "ok", "ok", "ok", "derivacion_plana", "ok"],
            "flat_leads": [0, 0, 0, 0, 0, 0, 1, 0],
        }
    )
    return table, metadata, quality, {"N": "NORM", "I": "MI"}


def test_exact_historical_cohort_retains_failed_delineation():
    table, metadata, quality, classes = cohort_tables()
    result = amended.validate_historical_cohort(table, metadata, quality, classes)
    assert result["selected_count"] == 6
    assert result["labeled_count_before_quality"] == 7
    assert result["excluded_labeled_ecg_ids"] == [7]
    assert result["retained_sin_latidos_count"] == 1


@pytest.mark.parametrize("missing", [1, 2])
def test_missing_ordinary_or_sin_latidos_record_rejected(missing):
    table, metadata, quality, classes = cohort_tables()
    with pytest.raises(ValueError, match="identities differ"):
        amended.validate_historical_cohort(table.loc[table.ecg_id != missing], metadata, quality, classes)


def test_flat_lead_record_cannot_be_added():
    table, metadata, quality, classes = cohort_tables()
    extra = table.iloc[[1]].copy()
    extra["ecg_id"], extra["patient_id"], extra["strat_fold"] = 7, 107, 2
    with pytest.raises(ValueError, match="identities differ"):
        amended.validate_historical_cohort(pd.concat([table, extra]), metadata, quality, classes)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("missing", "uniquely cover"),
        ("duplicate", "uniquely cover"),
        ("extra", "uniquely cover"),
        ("status", "unknown historical"),
        ("count", "counts disagree"),
        ("fraction", "invalid flat-lead"),
        ("nan", "must be present"),
    ],
)
def test_corrupt_quality_rejected(mutation, match):
    table, metadata, quality, classes = cohort_tables()
    if mutation == "missing":
        quality = quality.iloc[:-1]
    if mutation == "duplicate":
        quality = pd.concat([quality, quality.iloc[[0]]])
    if mutation == "extra":
        quality.loc[len(quality)] = [9, "ok", 0]
    if mutation == "status":
        quality.loc[0, "status"] = "unknown"
    if mutation == "count":
        quality.loc[0, "flat_leads"] = 1
    if mutation == "fraction":
        quality["flat_leads"] = quality.flat_leads.astype(float)
        quality.loc[6, "flat_leads"] = 0.5
    if mutation == "nan":
        quality.loc[0, "status"] = None
    with pytest.raises(ValueError, match=match):
        amended.validate_historical_cohort(table, metadata, quality, classes)


def test_additive_cohort_failure_never_reaches_baseline_or_fit(tmp_path, monkeypatch):
    table, metadata, quality, classes = cohort_tables()
    monkeypatch.setattr(amended, "provenance", lambda *_: {})
    monkeypatch.setattr(amended.original, "read_selected", lambda: table)
    quality.loc[0, "status"] = "unknown"
    monkeypatch.setattr(amended.original, "load_metadata", lambda: metadata)
    monkeypatch.setattr(amended.original, "load_statements", lambda: None)
    monkeypatch.setattr(amended.original, "diagnostic_classes", lambda _: classes)
    monkeypatch.setattr(amended, "read_quality", lambda _: quality)

    def forbidden(*_):
        pytest.fail("cohort failure reached baseline scoring or prospective fitting")

    monkeypatch.setattr(amended.original, "baseline_integrity", forbidden)
    monkeypatch.setattr(amended, "evaluate_models", forbidden)
    output = tmp_path / "new_output"
    with pytest.raises(ValueError, match="unknown historical"):
        amended.run_experiment(output, tmp_path / "protocol", tmp_path / "amendment", threads=1)
    assert json.loads((output / "failure.json").read_text())["status"] == "failed"
    with pytest.raises(FileExistsError):
        amended.run_experiment(output, tmp_path / "protocol", tmp_path / "amendment", threads=1)
