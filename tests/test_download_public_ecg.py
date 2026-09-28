"""Checks of the public downloader's record selection."""

from scripts.download_public_ecg import _cohort_files


def test_unpaired_records_are_skipped_and_reported():
    names = ("training/ningbo/g1/A1.hea", "training/ningbo/g1/A1.mat", "training/ningbo/g13/S23074.hea",
             "training/ningbo/g13/JS23074.mat", "training/other/B1.hea", "training/other/B1.mat")
    checksums = dict.fromkeys(names, "x")
    stems, files, unpaired = _cohort_files(checksums, "ningbo", "training/ningbo/", None)
    assert stems == ["training/ningbo/g1/A1"]
    assert files == ["training/ningbo/g1/A1.hea", "training/ningbo/g1/A1.mat"]
    assert unpaired == ["training/ningbo/g13/JS23074", "training/ningbo/g13/S23074"]
