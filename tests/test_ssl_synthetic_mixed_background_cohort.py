"""Guard the prospective patient exclusion and deterministic selection rules."""

import pytest

from ecg_experiment.ssl_synthetic_mixed_background_cohort import checksum_table, select_records


def row(patient, ecg, fold, role="train"):
    return {"patient_id": str(patient), "ecg_id": str(ecg), "strat_fold": fold, "role": role}


def test_patient_wide_quarantine_and_roles():
    rows = [
        row(1, 1, 1),
        row(1, 2, 8),
        row(2, 3, 2),
        row(3, 4, 3, "validation"),
        row(4, 5, 4, "engineering_test"),
    ]
    selected, candidates = select_records(rows, "test", 1)
    assert selected == candidates == [row(2, 3, 2)]


def test_lowest_record_and_order_are_reproducible():
    rows = [row(1, 20, 1), row(1, 10, 1), row(2, 30, 2), row(3, 40, 3)]
    first, candidates = select_records(rows, "test", 3)
    second, _ = select_records(list(reversed(rows)), "test", 3)
    assert first == second == candidates
    assert {r["ecg_id"] for r in first} == {"10", "30", "40"}


def test_insufficient_cohort_is_not_replaced():
    with pytest.raises(ValueError, match="Insufficient"):
        select_records([row(1, 1, 9)], "test", 1)


def test_duplicate_official_declarations_fail(tmp_path):
    path = tmp_path / "SHA256SUMS.txt"
    path.write_text("a" * 64 + " record.dat\n" + "b" * 64 + " record.dat\n")
    with pytest.raises(ValueError, match="duplicated"):
        checksum_table(path)
