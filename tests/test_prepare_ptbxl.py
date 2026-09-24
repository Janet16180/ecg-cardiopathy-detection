"""PTB-XL manifests keep patient-level label selection and hide labels from SSL splits."""

import csv
import json
from pathlib import Path

import pytest

from ecg_experiment.files import read_csv
from scripts.data.prepare_ptbxl import classify, parse_codes, prepare


def _row(ecg_id: int, patient_id: str, fold: str, codes: str) -> dict[str, str]:
    return {"ecg_id": str(ecg_id), "patient_id": patient_id, "strat_fold": fold,
            "scp_codes": codes, "report": "private diagnosis", "filename_lr": f"lr/{ecg_id}",
            "filename_hr": f"hr/{ecg_id}"}


def _write_database(metadata: Path, rows: list[dict[str, str]]) -> None:
    with (metadata / "ptbxl_database.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def rows() -> list[dict[str, str]]:
    records = [_row(i, f"p{i}", str(i % 8 + 1), "{'MI': 0.0}" if i % 2 else "{'NORM': 0.0, 'SR': 0.0}")
               for i in range(1, 11)]
    records += [
        _row(11, "p1", "2", "{'MI': 0.0}"),
        _row(12, "p1", "2", "{'NORM': 100.0, 'AFIB': 0.0}"),
        _row(13, "p2", "3", "{'AFIB': 0.0}"),
        _row(14, "v", "9", "{'MI': 0.0}"),
        _row(15, "v2", "9", "{'NORM': 100.0, 'AFIB': 0.0}"),
        _row(16, "t", "10", "{'NORM': 0.0}"),
        _row(17, "t2", "10", "{'AFIB': 100.0}"),
    ]
    return records


@pytest.fixture
def metadata(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    directory = tmp_path / "metadata"
    directory.mkdir()
    with (directory / "scp_statements.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["", "diagnostic", "diagnostic_class"])
        writer.writeheader()
        writer.writerows([
            {"": "NORM", "diagnostic": "1.0", "diagnostic_class": "NORM"},
            {"": "MI", "diagnostic": "1.0", "diagnostic_class": "MI"},
            {"": "AFIB", "diagnostic": "", "diagnostic_class": ""},
            {"": "SR", "diagnostic": "", "diagnostic_class": ""},
        ])
    _write_database(directory, rows)
    return directory


def test_manifests_hide_labels_and_keep_patient_selection(metadata: Path, tmp_path: Path) -> None:
    output = tmp_path / "one"
    summary = prepare(metadata, output, 0.1, 42)
    labeled = read_csv(output / "labeled_train.csv")
    unlabeled = read_csv(output / "unlabeled_train.csv")
    ssl = read_csv(output / "all_train_ssl.csv")
    validation = read_csv(output / "validation.csv")
    test = read_csv(output / "test.csv")
    audit = {r["ecg_id"]: r for r in read_csv(output / "audit" / "metadata.csv")}

    assert summary["selected_eligible_patients"] == 1
    assert summary["actual_eligible_patient_fraction"] == pytest.approx(0.1)
    assert {r["ecg_id"] for r in labeled} | {r["ecg_id"] for r in unlabeled} == {r["ecg_id"] for r in ssl}
    assert len(ssl) == 13
    assert [r["ecg_id"] for r in validation] == ["14"]
    assert [r["ecg_id"] for r in test] == ["16"]
    assert audit["12"]["target"] == ""
    assert audit["12"]["reason"] == "norm_with_other_codes"
    assert audit["13"]["target"] == ""
    assert audit["1"]["target"] == "1"
    assert audit["2"]["target"] == "0"
    assert summary["excluded_unresolved_validation_records"] == 1
    assert summary["excluded_unresolved_test_records"] == 1
    for name in ("unlabeled_train.csv", "all_train_ssl.csv"):
        assert set(read_csv(output / name)[0]) == {"ecg_id", "patient_id", "filename_lr", "filename_hr"}
        assert "private diagnosis" not in (output / name).read_text()
    assert set(labeled[0]) == {"ecg_id", "patient_id", "filename_lr", "filename_hr", "target"}
    assert set(json.loads((output / "summary.json").read_text())["splits"]) == {
        "all_train_ssl", "labeled_train", "unlabeled_train", "validation", "test"}

    # Every resolved recording of a chosen patient is labeled; unresolved stays unlabeled.
    selected = {r["patient_id"] for r in labeled}
    assert all(r["patient_id"] in selected for r in labeled)
    assert all(audit[r["ecg_id"]]["target"] == "" or r["patient_id"] not in selected for r in unlabeled)


def test_repeatability_and_fraction(metadata: Path, tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    prepare(metadata, first, 0.2, 42)
    prepare(metadata, second, 0.2, 42)
    assert (first / "labeled_train.csv").read_bytes() == (second / "labeled_train.csv").read_bytes()
    assert (first / "summary.json").read_bytes() == (second / "summary.json").read_bytes()
    assert len({r["patient_id"] for r in read_csv(first / "labeled_train.csv")}) == 2
    with pytest.raises(FileExistsError, match="nonempty"):
        prepare(metadata, first, 0.2, 42)
    with pytest.raises(ValueError, match="label-fraction"):
        prepare(metadata, tmp_path / "bad", 0, 42)


def test_patient_fold_conflict_and_duplicate_record_rejected(
        metadata: Path, rows: list[dict[str, str]], tmp_path: Path) -> None:
    rows.append(_row(18, "p1", "9", "{'MI': 100.0}"))
    _write_database(metadata, rows)
    with pytest.raises(ValueError, match="spans official folds"):
        prepare(metadata, tmp_path / "conflict")
    rows[-1]["patient_id"] = "new"
    rows[-1]["ecg_id"] = "1"
    _write_database(metadata, rows)
    with pytest.raises(ValueError, match="Duplicate ecg_id"):
        prepare(metadata, tmp_path / "duplicate")


def test_classification_rules() -> None:
    diagnostic = {"NORM": "NORM", "MI": "MI", "STTC": "STTC"}
    assert classify({"NORM"}, diagnostic) == ("0", "norm_only_or_sinus_rhythm")
    assert classify({"NORM", "SR"}, diagnostic) == ("0", "norm_only_or_sinus_rhythm")
    assert classify({"NORM", "MI"}, diagnostic) == ("", "norm_with_other_codes")
    assert classify({"STTC", "AFIB"}, diagnostic) == ("1", "non_norm_diagnostic_class")
    assert classify({"AFIB"}, diagnostic) == ("", "no_non_norm_diagnostic_class")


def test_scp_codes_must_be_a_dictionary() -> None:
    assert parse_codes("{'MI': 0.0}", "1") == {"MI"}
    with pytest.raises(ValueError, match="Invalid scp_codes"):
        parse_codes("{", "1")
    with pytest.raises(ValueError, match="Expected SCP code dictionary"):
        parse_codes("['MI']", "1")
