"""Exercise patient leakage controls and local CLI receipts using artificial metadata."""

import copy
import csv
import hashlib
import io
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from ecg_experiment import ROOT
from ecg_experiment.ssl_synthetic_partition import (
    assign_records,
    build_partition,
    canonical_identifier,
    patient_role,
    read_metadata,
    validate_policy,
)
from scripts.data import build_ssl_synthetic_partition as runner

POLICY = json.loads((ROOT / "configs/ssl_synthetic_partition_v1.json").read_text())


def metadata_rows():
    return [
        {
            "ecg_id": str(n),
            "patient_id": str(n),
            "strat_fold": "1",
            "filename_hr": f"records500/00000/{n:05d}_hr",
        }
        for n in range(1, 101)
    ] + [
        {"ecg_id": "101", "patient_id": "900", "strat_fold": "9", "filename_hr": "records500/00000/00101_hr"},
        {
            "ecg_id": "102",
            "patient_id": "901",
            "strat_fold": "10",
            "filename_hr": "records500/00000/00102_hr",
        },
        {"ecg_id": "103", "patient_id": "1.0", "strat_fold": "2", "filename_hr": "records500/00000/00103_hr"},
    ]


def csv_bytes(rows):
    stream = io.StringIO()
    writer = csv.DictWriter(
        stream, fieldnames=["ecg_id", "patient_id", "strat_fold", "filename_hr", "scp_codes"]
    )
    writer.writeheader()
    writer.writerows(
        {**row, "scp_codes": "deliberately unparseable; diagnosis must be ignored"} for row in rows
    )
    return stream.getvalue().encode()


def create_eligible_paths(root, rows):
    for row in rows:
        if int(row["strat_fold"]) > 8:
            continue
        for suffix in (".dat", ".hea"):
            path = root / (row["filename_hr"] + suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"deliberately invalid waveform; must never be opened")


def test_repeat_grouping_is_stable_and_payloads_are_never_opened(tmp_path):
    rows = metadata_rows()
    create_eligible_paths(tmp_path, rows)
    with patch.object(Path, "open", side_effect=AssertionError("Waveform was opened")):
        first = build_partition(csv_bytes(rows), POLICY, tmp_path)
        second = build_partition(csv_bytes(list(reversed(rows))), POLICY, tmp_path)
    assert first["records"] == second["records"]
    assert first["counts"] == second["counts"]
    assert first["metadata_sha256"] != second["metadata_sha256"]
    repeated = [r for r in first["records"] if r["patient_id"] == "1"]
    assert len(repeated) == 2
    assert len({(r["role"], r["background_id"]) for r in repeated}) == 1
    assert first["counts"]["patients_in_multiple_roles"] == 0
    assert first["counts"]["excluded_fold_records_in_eligible_roles"] == 0
    assert first["eligible_file_pairs_present"] == 101
    assert not first["training_admitted"]
    assert not first["performance_scored"]
    assert set(first["records"][0]) == {
        "ecg_id",
        "patient_id",
        "strat_fold",
        "filename_hr",
        "role",
        "background_id",
    }


def test_hash_rule_matches_independent_digest_and_fixed_cut_points():
    roles = set()
    for n in range(1, 101):
        digest = hashlib.sha256(f"ssl_synthetic_partition_v1:20261018:{n}".encode()).digest()
        bucket = int.from_bytes(digest, "big") % 10000
        expected = "train" if bucket < 8000 else "validation" if bucket < 9000 else "engineering_test"
        assert patient_role(str(n), POLICY) == expected
        roles.add(expected)
    assert roles == {"train", "validation", "engineering_test"}


def test_mixed_fold_patient_is_excluded_in_full(tmp_path):
    rows = metadata_rows()
    rows[-1]["patient_id"] = "900.0"
    create_eligible_paths(tmp_path, rows)
    (tmp_path / (rows[-1]["filename_hr"] + ".dat")).unlink()
    report = build_partition(csv_bytes(rows), POLICY, tmp_path)
    assert {r["role"] for r in report["records"] if r["patient_id"] == "900"} == {"excluded"}
    assert report["counts"]["eligible_fold_records_quarantined"] == 1
    assert report["eligible_file_pairs_present"] == 100


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ecg_id", "1.5"),
        ("patient_id", "NaN"),
        ("patient_id", "0"),
        ("patient_id", ""),
        ("ecg_id", "missing"),
        ("strat_fold", "11"),
        ("filename_hr", "records500/../00001_hr"),
        ("filename_hr", "/absolute/00001_hr"),
        ("filename_hr", "records100/00000/00001_hr"),
        ("filename_hr", "records500/00000/00002_hr"),
    ],
)
def test_malformed_metadata_fails_before_filesystem_inspection(tmp_path, field, value):
    rows = metadata_rows()
    rows[0][field] = value
    with patch.object(Path, "is_file") as inspect:
        with pytest.raises(
            ValueError, match="positive finite integers|[Rr]ecording path|invalid source fold"
        ):
            build_partition(csv_bytes(rows), POLICY, tmp_path)
        inspect.assert_not_called()


def test_duplicate_record_and_missing_or_duplicate_column_fail():
    rows = metadata_rows()
    with pytest.raises(ValueError, match="Duplicate ECG"):
        read_metadata(csv_bytes(rows + [rows[0]]))
    for header in [b"ecg_id,patient_id,strat_fold\n", b"ecg_id,patient_id,strat_fold,filename_hr,ecg_id\n"]:
        with pytest.raises(ValueError, match="column exactly once"):
            read_metadata(header)


def test_missing_eligible_file_or_empty_role_cannot_be_silently_dropped(tmp_path):
    rows = metadata_rows()
    create_eligible_paths(tmp_path, rows)
    (tmp_path / (rows[0]["filename_hr"] + ".dat")).unlink()
    with pytest.raises(ValueError, match="Missing 1 eligible"):
        build_partition(csv_bytes(rows), POLICY, tmp_path)
    with pytest.raises(ValueError, match="empty role"):
        build_partition(csv_bytes(rows[:1]), POLICY, tmp_path)


@pytest.mark.parametrize(
    ("key", "value"), [("seed", 1), ("eligible_folds", list(range(1, 10))), ("schema_version", True)]
)
def test_policy_changes_fail(key, value):
    policy = copy.deepcopy(POLICY)
    policy[key] = value
    with pytest.raises(ValueError, match="frozen v1"):
        validate_policy(policy)


def test_integral_decimal_identifiers_normalize_before_grouping():
    assert canonical_identifier("00123.000") == "123"
    records = assign_records(read_metadata(csv_bytes(metadata_rows())), POLICY)
    assert records[0]["background_id"] == records[-1]["background_id"]


def test_cli_is_reproducible_keeps_patient_data_local_and_refuses_overwrite(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr("ecg_experiment.paths.ROOT", tmp_path)
    source_root = tmp_path / "data/raw/ptb-xl/1.0.3"
    source_root.mkdir(parents=True)
    rows = metadata_rows()
    create_eligible_paths(source_root, rows)
    (source_root / "ptbxl_database.csv").write_bytes(csv_bytes(rows))
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/ssl_synthetic_partition_v1.json").write_text(json.dumps(POLICY))
    for relative in [
        "ecg_experiment/ssl_synthetic_partition.py",
        "scripts/data/build_ssl_synthetic_partition.py",
        "docs/ssl-synthetic-localization-partition.md",
    ]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("artificial source provenance fixture\n")
    monkeypatch.setattr(sys, "argv", ["partition"])
    runner.main()
    printed = json.loads(capsys.readouterr().out)
    assert "records" not in printed
    first_path = tmp_path / "data/processed/ssl_synthetic_localization_v1/partition.json"
    original = first_path.read_bytes()
    summary = json.loads((tmp_path / "outputs/ssl_synthetic_localization/partition_summary.json").read_text())
    assert "records" not in summary
    assert summary["manifest_sha256"] == hashlib.sha256(original).hexdigest()
    assert not Path(summary["manifest"]).is_absolute()
    with (
        patch.object(Path, "read_bytes", side_effect=AssertionError("Inputs reopened before output guard")),
        pytest.raises(FileExistsError, match="no output was overwritten"),
    ):
        runner.main()
    replay = tmp_path / "data/processed/replay.json"
    replay_summary = tmp_path / "outputs/replay.json"
    monkeypatch.setattr(
        sys, "argv", ["partition", "--manifest", str(replay), "--summary", str(replay_summary)]
    )
    runner.main()
    assert replay.read_bytes() == original
