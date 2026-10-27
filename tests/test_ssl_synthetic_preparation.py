"""Exercise immutable receipt guards before any preparation can be admitted."""

import hashlib
import json

import pytest

from ecg_experiment.ssl_synthetic_preparation import (
    stream_bytes,
    verify_hashes,
    verify_receipt,
    write_new_json,
)


def test_changed_or_traversing_source_cannot_pass_hash_gate(tmp_path):
    path = tmp_path / "source.py"
    path.write_text("original")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    verify_hashes({"source.py": expected}, tmp_path)
    path.write_text("changed")
    with pytest.raises(ValueError, match="hash changed"):
        verify_hashes({"source.py": expected}, tmp_path)
    with pytest.raises(ValueError, match="repository-relative"):
        verify_hashes({"../outside": expected}, tmp_path)


def test_adjacent_output_changes_reject_receipt(tmp_path):
    output = tmp_path / "signal.npy"
    output.write_bytes(b"fake array bytes")
    path = tmp_path / "summary.json"
    receipt = {"output_sha256": {output.name: hashlib.sha256(output.read_bytes()).hexdigest()}}
    path.write_text(json.dumps(receipt))
    assert verify_receipt(path) == receipt
    output.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="output hash changed"):
        verify_receipt(path)


def test_new_receipts_refuse_overwrite_and_nonfinite_numbers(tmp_path):
    path = tmp_path / "report.json"
    write_new_json(path, {"gate": False})
    with pytest.raises(FileExistsError):
        write_new_json(path, {"gate": True})
    assert json.loads(path.read_text()) == {"gate": False}
    with pytest.raises(ValueError, match="Out of range"):
        write_new_json(tmp_path / "nan.json", {"metric": float("nan")})


def test_cache_budget_counts_nested_and_predecessor_outputs(tmp_path):
    (tmp_path / "old").mkdir()
    (tmp_path / "old/failed.bin").write_bytes(b"123")
    (tmp_path / "new.bin").write_bytes(b"4567")
    assert stream_bytes(tmp_path) == 7
