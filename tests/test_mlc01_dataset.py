"""Checks of the MLC-01 record filter, closed-path guard, waveform summary and duplicate rule."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from ecg_experiment import mlc01_dataset, mlc01_tables
from ecg_experiment.eda.signals import LEADS
from ecg_experiment.mlc01_dataset import (
    challenge_key,
    closed_challenge_keys,
    exclusion_reason,
    open_challenge_rows,
    require_open,
    resolve_duplicates,
    summarize,
)


def split_rows() -> pd.DataFrame:
    return pd.DataFrame({
        "source": ["georgia", "georgia", "georgia", "ningbo"],
        "family": ["georgia", "georgia", "georgia", "chapman_ningbo"],
        "record": ["E1", "E2", "E3", "JS1"],
        "path": ["training/georgia/g1/E1", "training/georgia/g1/E2", "training/georgia/g1/E3",
                 "training/ningbo/g1/JS1"],
        "split": ["train", "calibration", "test", "test"],
    })


def test_filter_keeps_only_train_and_calibration_rows():
    rows = split_rows()
    kept = open_challenge_rows(rows)
    assert list(kept["record"]) == ["E1", "E2"]
    assert set(kept["split"]) == {"train", "calibration"}
    assert closed_challenge_keys(rows) == {"georgia:training/georgia/g1/E3", "ningbo:training/ningbo/g1/JS1"}


def test_filter_refuses_a_closed_path_listed_twice():
    rows = pd.concat([split_rows(), split_rows().iloc[[2]].assign(split="train")], ignore_index=True)
    with pytest.raises(ValueError, match="closed"):
        open_challenge_rows(rows)


def test_split_table_reader_checks_the_hash_and_reads_no_label_column(tmp_path, monkeypatch):
    path = tmp_path / "rows.csv"
    split_rows().assign(primary=1.0, signal_sha256="x").to_csv(path, index=False)
    with pytest.raises(ValueError, match="SHA-256"):
        mlc01_dataset.read_split_table(path)
    monkeypatch.setattr(mlc01_dataset, "SPLIT_TABLE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    rows = mlc01_dataset.read_split_table(path)
    assert list(rows.columns) == mlc01_dataset.SPLIT_COLUMNS


def test_guard_refuses_closed_records_in_workers(monkeypatch):
    closed = closed_challenge_keys(split_rows())
    require_open(challenge_key("georgia", "training/georgia/g1/E1"), closed)
    with pytest.raises(PermissionError):
        require_open("georgia:training/georgia/g1/E3", closed)
    job = mlc01_dataset.wfdb_job("georgia", "E3", "georgia:E3", mlc01_dataset.RELEASES["georgia"],
                                 "training/georgia/g1/E3", {"training/georgia/g1/E3.hea": "a",
                                                            "training/georgia/g1/E3.mat": "b"})
    monkeypatch.setattr(mlc01_dataset, "GUARD", {"closed": frozenset()})
    with pytest.raises(PermissionError, match="guard is not set"):
        mlc01_dataset.scan(job)
    mlc01_dataset.init_worker(closed)
    with pytest.raises(PermissionError, match="Closed"):
        mlc01_dataset.scan(job)


def test_microvolt_hash_follows_the_split_table_convention():
    signal = np.array([[0.0011, np.nan], [-0.0024, 1.0]])
    micro = mlc01_dataset.microvolts(signal)
    assert micro.tolist() == [[1, -99999000], [-2, 1000]]
    reference = np.round(np.nan_to_num(signal, nan=-99999) * 1000).astype(np.int32)
    assert mlc01_dataset.array_sha256(micro) == hashlib.sha256(reference.tobytes()).hexdigest()


def test_summary_reorders_leads_flags_flat_and_missing_leads():
    rng = np.random.default_rng(0)
    signal = rng.normal(size=(6000, 12))
    signal[:, 3] = 0.25
    signal[10, 5] = np.nan
    names = list(LEADS)
    row = summarize(signal, 500, names)
    assert row["leads_ok"]
    assert row["duration_s"] == 12
    assert row["flat_aVR"]
    assert not row["flat_I"]
    assert row["missing_aVF"]
    assert row["missing_samples_full"] == 1
    swapped = summarize(signal[:, [1, 0, *range(2, 12)]], 500, [names[1], names[0], *names[2:]])
    assert swapped["sha256_10s"] == row["sha256_10s"]
    assert swapped["sha256_full"] == row["sha256_full"]
    assert summarize(signal[:5000], 500, names)["sha256_10s"] == row["sha256_10s"]
    assert summarize(signal[:5000], 500, names)["sha256_full"] != row["sha256_full"]
    assert not summarize(signal, 500, [*names[:11], "X"])["leads_ok"]
    assert "sha256_10s" not in summarize(signal[:4000], 500, names)


def test_exclusion_reason_takes_the_first_failed_rule():
    frame = pd.DataFrame({
        "record_key": ["a", "b", "c", "d", "e", "ningbo:JS13118"],
        "readable": [True, False, True, True, True, True],
        "leads_ok": [True, pd.NA, False, True, True, True],
        "fs": [500, np.nan, 500, 250, 500, 500],
        "duration_s": [10.0, np.nan, 10.0, 10.0, 9.98, 10.0],
    })
    assert list(exclusion_reason(frame)) == ["", "unreadable", "wrong_leads", "wrong_rate", "short",
                                             "upstream_corrupted"]


def duplicates(roles: list[str], labels: list[list[float]], hashes: list[str] | None = None) -> pd.Series:
    frame = pd.DataFrame(labels, columns=["b_x", "b_y"], index=[f"r{i}" for i in range(len(roles))])
    frame["role"] = roles
    frame["sha256_10s"] = hashes or ["h"] * len(roles)
    return resolve_duplicates(frame, ["b_x", "b_y"])["duplicate_status"]


def test_a_training_copy_of_a_test_record_is_removed():
    status = duplicates(["train", "test_same_source"], [[1.0, 0.0], [0.0, 0.0]])
    assert list(status) == ["removed_cross_role", "kept"]
    status = duplicates(["validation", "train", "test_external"], [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    assert list(status) == ["removed_cross_role", "removed_cross_role", "kept"]
    status = duplicates(["train", "validation"], [[1.0, 0.0], [1.0, 0.0]])
    assert list(status) == ["removed_cross_role", "kept"]


def test_conflicting_training_copies_drop_every_copy():
    status = duplicates(["train", "train"], [[1.0, 0.0], [0.0, 0.0]])
    assert list(status) == ["dropped_conflict", "dropped_conflict"]


def test_agreeing_copies_keep_the_most_defined_copy():
    status = duplicates(["train", "train"], [[1.0, np.nan], [1.0, 0.0]])
    assert list(status) == ["dropped_copy", "kept"]
    status = duplicates(["train", "train", "train"], [[1.0, 0.0], [1.0, 0.0], [1.0, np.nan]])
    assert list(status) == ["kept", "dropped_copy", "dropped_copy"]
    status = duplicates(["train", "train"], [[1.0, 0.0], [1.0, 0.0]], ["h", "g"])
    assert list(status) == ["unique", "unique"]


def test_label_frame_leaves_records_without_codes_undefined():
    frame = pd.DataFrame({"source": ["georgia", "georgia"], "raw_codes": ["", "426783006"],
                          "heart_axis": ["", ""]})
    labels = mlc01_dataset.label_frame(frame, {}, frozenset())
    assert labels.iloc[0][mlc01_dataset.label_columns(labels)].isna().all()
    assert labels.iloc[1]["normal"] == 1
    assert labels.iloc[1]["b_sinus_tachycardia"] == 0


def test_state_counts_split_by_role_and_source():
    frame = pd.DataFrame({"included": [True, True, True, False],
                          "role": ["train", "train", "validation", "train"],
                          "source": ["ningbo", "chapman_shaoxing", "ningbo", "ningbo"],
                          "family": ["chapman_ningbo"] * 4, "b_x": [1.0, np.nan, 0.0, 1.0]})
    counts = mlc01_tables.state_counts(frame, ["b_x"], "B").set_index(["role", "unit"])
    assert counts.loc[("train", "ningbo"), "positive"] == 1
    assert counts.loc[("train", "chapman_shaoxing"), "undefined"] == 1
    assert counts.loc[("train", "chapman_ningbo"), "records"] == 2
    assert counts.loc[("validation", "ningbo"), "negative"] == 1


def test_reference_check_requires_identical_hashes_and_roles(tmp_path, monkeypatch):
    from ecg_experiment import mlc01_build

    monkeypatch.setattr(mlc01_build, "to_stored", str)
    frame = pd.DataFrame({"role": ["train", "test_external"], "raw_path": ["data/a", "data/b"],
                          "sha256_10s": ["a", np.nan], "sha256_full": ["b", np.nan]},
                         index=pd.Index(["x:1", "sph:2"], name="record_key"))
    frame.to_csv(tmp_path / "manifest.csv")
    checks: dict = {}
    mlc01_build.check_reference(frame, tmp_path, checks)
    assert checks["records_roles_paths_and_waveform_hashes_match_reference"]["passed"]
    changes = (frame.assign(role=["validation", "test_external"]), frame.assign(raw_path=["data/a", "x"]),
               frame.iloc[:1])
    for changed in changes:
        with pytest.raises(ValueError, match="reference"):
            mlc01_build.check_reference(changed, tmp_path, {})
    mlc01_build.check_reference(frame, None, checks)


def test_status_table_marks_pairs_caught_by_the_minimum_positive_rule():
    from ecg_experiment import mlc01_build

    minimum = pd.DataFrame({"source": ["georgia", "sph"], "label": ["b_mi_any", "c_paced_rhythm"],
                            "positives": [3, 0], "defined": [8160, 0], "not_annotated": [True, True]})
    status = mlc01_build.status_table(minimum).set_index(["label", "source"])["status"]
    assert status[("b_mi_any", "georgia")] == "not annotated (fewer than 5 positives)"
    assert status[("c_paced_rhythm", "sph")] == "not codable"
    assert status[("b_mi_any", "ningbo")] == "annotated"
