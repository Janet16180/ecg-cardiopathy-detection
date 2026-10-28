"""Split integrity and conservative record-label eligibility for Challenge training recovery."""

import json
from pathlib import Path

import pandas as pd
import pytest

from ecg_experiment import challenge_training_recovery as recovery


def test_original_split_hash_blocks_unverified_processing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completion claim cannot bypass the pinned original split hash."""
    directory = tmp_path / recovery.RECONSTRUCTION
    directory.mkdir(parents=True)
    (directory / "metadata.json").write_text(json.dumps({
        "complete": True, "original_rows_reproduced": True, "rows_sha256": recovery.SPLIT_SHA256,
    }))
    (directory / "rows.csv").write_text("a changed split\n")

    def forbidden_read(*args: object, **kwargs: object) -> None:
        raise AssertionError("Unverified split reached source or waveform reads")

    monkeypatch.setattr(pd, "read_parquet", forbidden_read)
    monkeypatch.setattr(recovery, "load_view", forbidden_read)
    with pytest.raises(ValueError, match="SHA-256 gate"):
        recovery.recover(tmp_path, Path("quality"), 1, 1)
    assert not (tmp_path / "quality").exists()


@pytest.mark.parametrize(("samples", "eligible"), [(5000, True), (7000, False)])
def test_crop_keeps_record_label_without_promoting_supervised_target(
    monkeypatch: pytest.MonkeyPatch, samples: int, eligible: bool,
) -> None:
    """Longer-record crops retain diagnosis evidence while remaining SSL only."""
    row = {
        "split": "train", "path": "training/cpsc_2018/g1/A1", "source": "cpsc_2018",
        "release": "challenge-2020/1.0.2", "record_id": "cpsc_2018:A1", "samples": samples,
        "duplicate_status": "unique", "view_policy": "ssl_center_crop", "primary": "1.0",
        "record_primary": "1.0",
    }

    def clean_view(input_root: str, evidence: dict, expected: dict) -> list[str]:
        evidence["signal_window_sha256"] = "canonical-hash"
        evidence["window_start"] = (samples - 5000) // 2
        return []

    monkeypatch.setattr(recovery, "verified_view", clean_view)
    result = recovery.assess_training_record(("unused", row, {}))
    assert result["record_primary"] == "1.0"
    assert result["ssl_eligible"] is True
    assert result["supervised_eligible"] is eligible
    assert result["supervised_target"] == ("1.0" if eligible else "")


@pytest.mark.parametrize(("second_primary", "status"), [("1.0", "dropped_copy"), ("0.0", "dropped_conflict")])
def test_strict_window_duplicates_follow_frozen_label_conflict_rule(
    second_primary: str, status: str,
) -> None:
    """Repeated canonical windows cannot retain conflicting supervised labels."""
    base = {
        "samples": 5000, "quality_status": "passed", "signal_window_sha256": "same-hash", "male": 1.0,
        "primary": "1.0", "secondary": "1.0", "MI": "0.0", "STTC": "1.0", "CD": "0.0",
        "HYP": "0.0", "supervised_eligible": True, "supervised_target": "1.0",
    }
    rows = pd.DataFrame([
        base | {"record": "A"},
        base | {"record": "B", "primary": second_primary, "supervised_target": second_primary},
    ])
    result = recovery.supervised_window_duplicates(rows).set_index("record")
    assert result.loc["B", "supervised_duplicate_status"] == status
    assert not result.loc["B", "supervised_eligible"]
    assert result.loc["B", "supervised_target"] == ""
    assert bool(result.loc["A", "supervised_eligible"]) is (status == "dropped_copy")


def test_undefined_strict_label_stays_undefined_during_duplicate_resolution() -> None:
    """Empty label strings are represented as missing mapped labels, never as negatives."""
    row = {
        "record": "A", "samples": 5000, "quality_status": "passed",
        "signal_window_sha256": "unique-hash", "male": float("nan"),
        "supervised_eligible": False, "supervised_target": "",
    } | dict.fromkeys(recovery.LABEL_COLUMNS, "")
    result = recovery.supervised_window_duplicates(pd.DataFrame([row]))
    assert not result.loc[0, "supervised_eligible"]
    assert result.loc[0, "supervised_target"] == ""
    assert result.loc[0, "supervised_duplicate_status"] == "unique"
