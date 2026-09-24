"""Exact-identity overlap gate for appending Challenge candidates."""

import pytest

from scripts.validation.audit_public_pool_overlap import classify_overlap


def test_same_identity_under_different_record_id_cannot_append() -> None:
    rows = [{"ecg_id": "new:a", "source": "new", "signal_sha256": "x"},
            {"ecg_id": "same:id", "source": "new", "signal_sha256": "changed"},
            {"ecg_id": "new:b", "source": "new", "signal_sha256": "y"}]
    novel, overlap = classify_overlap(rows, {"ptbxl_all_folds": {"x"}, "mimic": set()}, {"same:id"})
    assert [r["ecg_id"] for r in novel] == ["new:b"]
    assert overlap[0]["exact_signal_reference_pools"] == "ptbxl_all_folds"
    assert overlap[1]["existing_record_id"] == "true"


def test_duplicates_in_candidate_fail_gate() -> None:
    with pytest.raises(ValueError, match="duplicate identity"):
        classify_overlap([{"ecg_id": "a", "signal_sha256": "x"},
                          {"ecg_id": "b", "signal_sha256": "x"}], {}, set())


def test_published_directory_renames_only_after_success(tmp_path) -> None:
    from ecg_experiment.staging import published_directory

    output = tmp_path / "out"
    with published_directory(output) as stage:
        (stage / "receipt.json").write_text("{}")
        assert not output.exists()
    assert (output / "receipt.json").read_text() == "{}"
    assert list(tmp_path.iterdir()) == [output]


def test_published_directory_removes_stage_on_interrupt(tmp_path) -> None:
    from ecg_experiment.staging import published_directory

    def interrupted_publish() -> None:
        with published_directory(tmp_path / "out") as stage:
            (stage / "partial.csv").write_text("x")
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        interrupted_publish()
    assert list(tmp_path.iterdir()) == []


def test_published_directory_refuses_output_that_appeared(tmp_path) -> None:
    from ecg_experiment.staging import published_directory

    output = tmp_path / "out"
    with pytest.raises(FileExistsError, match="appeared"), published_directory(output):
        output.mkdir()
    assert list(tmp_path.iterdir()) == [output]
