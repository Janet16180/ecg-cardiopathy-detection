"""Verify duplicate and provenance gates using artificial checksum metadata."""

import copy
import csv
import hashlib
import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ecg_experiment import ROOT
from ecg_experiment.ssl_synthetic_checksum_audit import audit_duplicates, read_checksums, verify_partition
from ecg_experiment.ssl_synthetic_partition import assign_records, partition_counts, read_metadata


def records():
    return [
        {"patient_id": str(n), "role": role, "filename_hr": f"records500/00000/{n:05d}_hr"}
        for n, role in [(1, "train"), (2, "validation"), (3, "engineering_test"), (4, "excluded")]
    ]


def checksum_map(rows):
    return {
        r["filename_hr"] + ".dat": hashlib.sha256(str(i).encode()).hexdigest() for i, r in enumerate(rows)
    }


def test_unique_declared_signals_pass_without_opening_any_payload():
    rows = records()
    with patch.object(Path, "open", side_effect=AssertionError("Waveform opened")):
        report = audit_duplicates(rows, checksum_map(rows))
    assert report["duplicate_gate_passed"]
    assert report["unique_signal_digests"] == 4
    assert report["duplicate_groups"] == 0
    assert not report["training_admitted"]
    assert not report["waveform_payload_opened"]


def test_cross_patient_and_cross_role_copies_reject_gate():
    rows = records()
    checksums = checksum_map(rows)
    checksums[rows[1]["filename_hr"] + ".dat"] = checksums[rows[0]["filename_hr"] + ".dat"]
    report = audit_duplicates(rows, checksums)
    assert not report["duplicate_gate_passed"]
    assert report["cross_patient_duplicate_groups"] == 1
    assert report["records_in_cross_patient_duplicate_groups"] == 2
    assert report["cross_role_group_counts_by_role_pair"] == {"train|validation": 1}
    assert "records" not in report


def test_same_role_cross_patient_copy_is_still_rejected():
    rows = records()
    rows[1]["role"] = "train"
    checksums = dict.fromkeys(checksum_map(rows), "a" * 64)
    report = audit_duplicates(rows, checksums)
    assert not report["duplicate_gate_passed"]
    assert report["records_in_duplicate_groups"] == 4


def test_repeat_within_same_patient_is_permitted_but_cannot_cross_roles():
    rows = records()
    rows.append({**rows[0], "filename_hr": "records500/00000/00005_hr"})
    checksums = checksum_map(rows)
    checksums[rows[-1]["filename_hr"] + ".dat"] = checksums[rows[0]["filename_hr"] + ".dat"]
    assert audit_duplicates(rows, checksums)["duplicate_gate_passed"]
    rows[-1]["role"] = "validation"
    assert not audit_duplicates(rows, checksums)["duplicate_gate_passed"]


def test_missing_checksum_including_an_excluded_record_fails():
    rows = records()
    checksums = checksum_map(rows)
    del checksums[rows[-1]["filename_hr"] + ".dat"]
    with pytest.raises(ValueError, match="checksum missing"):
        audit_duplicates(rows, checksums)


@pytest.mark.parametrize(
    "text",
    [
        "bad records500/x.dat",
        "g" * 64 + " records500/x.dat",
        "a" * 64 + " ../outside.dat",
        "a" * 64 + " /absolute.dat",
        "a" * 64 + " file\n" + "a" * 64 + " file",
    ],
)
def test_bad_checksum_table_fails(text):
    with pytest.raises(ValueError, match="Invalid SHA-256|relative file paths|Duplicate checksum"):
        read_checksums(text)


def test_binary_checksum_format_and_uppercase_digest_normalize():
    assert read_checksums("A" * 64 + "  *./records500/x.dat\n") == {"records500/x.dat": "a" * 64}


def partition_fixture():
    policy = json.loads((ROOT / "configs/ssl_synthetic_partition_v1.json").read_text())
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["ecg_id", "patient_id", "strat_fold", "filename_hr"])
    for n in range(1, 101):
        writer.writerow([n, n, 9 if n == 100 else 1, f"records500/00000/{n:05d}_hr"])
    payload = stream.getvalue().encode()
    assigned = assign_records(read_metadata(payload), policy)
    manifest = {
        "policy": policy,
        "metadata_sha256": hashlib.sha256(payload).hexdigest(),
        "records": assigned,
        "counts": partition_counts(assigned),
    }
    blob = json.dumps(manifest).encode()
    summary = {key: value for key, value in manifest.items() if key != "records"}
    summary["manifest_sha256"] = hashlib.sha256(blob).hexdigest()
    return manifest, summary, blob, payload


def test_manifest_bindings_and_replay_are_enforced():
    manifest, summary, blob, payload = partition_fixture()
    verify_partition(manifest, summary, blob, payload)
    with pytest.raises(ValueError, match="Partition hash differs"):
        verify_partition(manifest, summary, blob + b" ", payload)
    with pytest.raises(ValueError, match="Source metadata differs"):
        verify_partition(manifest, summary, blob, payload + b"\n")
    changed = copy.deepcopy(manifest)
    changed["records"][0]["role"] = "excluded"
    changed_blob = json.dumps(changed).encode()
    changed_summary = {**summary, "manifest_sha256": hashlib.sha256(changed_blob).hexdigest()}
    with pytest.raises(ValueError, match="assignments do not reproduce"):
        verify_partition(changed, changed_summary, changed_blob, payload)
    with pytest.raises(ValueError, match="aggregate receipt disagree"):
        verify_partition(manifest, {**summary, "counts": {}}, blob, payload)
