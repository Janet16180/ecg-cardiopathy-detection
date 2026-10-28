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
    monkeypatch.setattr(recovery, "read_verified_view", forbidden_read)
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


def test_training_labels_skip_protected_split_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protected split labels never enter the adapter's training evidence frame."""
    import hashlib

    directory = tmp_path / recovery.RECONSTRUCTION
    directory.mkdir(parents=True)
    base = {
        "source": "georgia", "family": "georgia", "duration_s": "10.0", "window_sha256": "",
        "duplicate_status": "unique",
    } | dict.fromkeys(recovery.LABEL_COLUMNS, "")
    rows = pd.DataFrame([
        base | {"record": "E1", "path": "training/georgia/g1/E1", "signal_sha256": "a" * 64,
                "duplicate_group": "E1", "split": "train", "primary": "1.0"},
        base | {"record": "E2", "path": "training/georgia/g1/E2", "signal_sha256": "b" * 64,
                "duplicate_group": "E2", "split": "test", "primary": "protected_label_marker"},
    ])
    path = directory / "rows.csv"
    rows.to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(recovery, "SPLIT_SHA256", digest)
    (directory / "metadata.json").write_text(json.dumps({
        "complete": True, "original_rows_reproduced": True, "rows_sha256": digest,
    }))
    original_reader = pd.read_csv
    reads = []

    def guarded_reader(*args: object, **kwargs: object) -> pd.DataFrame:
        reads.append(kwargs)
        return original_reader(*args, **kwargs)

    def headers(input_root: Path, source: str, metadata: dict, columns: list[str]) -> tuple:
        selected = rows[rows["source"] == source].set_index("record")
        selected = selected.assign(age="40", sex="Male", fs=500, samples=5000)
        return selected[columns], {}

    monkeypatch.setattr(recovery.pd, "read_csv", guarded_reader)
    monkeypatch.setattr(recovery, "source_checkpoint", headers)
    selected, _ = recovery.verified_training_rows(tmp_path)
    assert selected["record"].tolist() == ["E1"]
    assert selected["primary"].tolist() == ["1.0"]
    assert selected["full_signal_sha256"].tolist() == ["a" * 64]
    assert selected["split_record_id"].tolist() == ["georgia:E1"]
    assert not set(recovery.LABEL_COLUMNS) & set(reads[0]["usecols"])
    assert reads[1]["skiprows"] == [2]


def test_reference_overlaps_disable_eligibility_without_hiding_quality_evidence() -> None:
    """Full-signal and canonical matches both exclude training, including crop-only matches."""
    base = {"ssl_eligible": True, "supervised_eligible": True,
            "supervised_target": "1.0", "quality_status": "passed"}
    rows = pd.DataFrame([
        base | {"full_signal_sha256": "same-full", "signal_window_sha256": "first-window"},
        base | {"full_signal_sha256": "different-full", "signal_window_sha256": "same-window"},
    ])
    result = recovery.exclude_reference_overlaps(rows, {"same-window"}, {"same-full"})
    assert not result["ssl_eligible"].any()
    assert not result["supervised_eligible"].any()
    assert result["quality_status"].tolist() == ["passed", "passed"]
    assert result["heldout_overlap_reasons"].tolist() == ["heldout_full_signal", "heldout_canonical_window"]
