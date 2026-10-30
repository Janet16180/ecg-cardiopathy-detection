"""Integrity checks for resuming mechanically recovered Challenge records."""

import json
from pathlib import Path

import pandas as pd
import pytest

from ecg_experiment.challenge_recovery20261028 import bind_producer, verified_source
from ecg_experiment.files import sha256_file


def checkpoint(directory: Path) -> tuple[Path, Path]:
    table_path = directory / "ningbo_verified.parquet"
    receipt_path = directory / "ningbo_receipt.json"
    pd.DataFrame({"record": ["synthetic"], "source": ["ningbo"]}).set_index("record").to_parquet(table_path)
    receipt_path.write_text(json.dumps({"checkpoint_sha256": sha256_file(table_path), "verified_records": 1}))
    return table_path, receipt_path


def test_legacy_resume_requires_explicit_producer_and_preserves_original_receipt(tmp_path: Path):
    _, receipt_path = checkpoint(tmp_path)
    original = receipt_path.read_bytes()
    with pytest.raises(ValueError, match="explicit preserved producer"):
        verified_source(tmp_path, tmp_path, "ningbo", [("synthetic",)], {}, 1, {})
    assert receipt_path.read_bytes() == original
    producer = {"sha256": "original-producer-digest", "source_snapshot": "outputs/synthetic.py"}
    table, receipt = verified_source(tmp_path, tmp_path, "ningbo", [("synthetic",)], {}, 1, producer)
    assert len(table) == 1
    assert receipt["producer"] == producer
    assert receipt["receipt_sha256"] == sha256_file(receipt_path)
    assert receipt_path.read_bytes() == original


def test_resume_rejects_changed_checkpoint_bytes(tmp_path: Path):
    table_path, _ = checkpoint(tmp_path)
    pd.DataFrame({"record": ["changed"]}).set_index("record").to_parquet(table_path)
    with pytest.raises(ValueError, match="Checkpoint differs"):
        verified_source(tmp_path, tmp_path, "ningbo", [("synthetic",)], {}, 1, {"sha256": "producer"})


def test_resume_rejects_changed_expected_record_count(tmp_path: Path):
    checkpoint(tmp_path)
    with pytest.raises(ValueError, match="Checkpoint differs"):
        verified_source(tmp_path, tmp_path, "ningbo", [], {}, 1, {"sha256": "producer"})


def test_embedded_producer_is_not_overridden_by_legacy_binding():
    receipt = {"producer": {"sha256": "new-producer"}, "checkpoint_sha256": "checkpoint"}
    assert bind_producer(receipt, {"sha256": "older-producer"}) == receipt
