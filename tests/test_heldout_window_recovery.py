"""Mechanical fixed-window references cannot bypass provenance or run quality/model assessments."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ecg_experiment import challenge_training_recovery as training
from ecg_experiment import heldout_window_recovery as recovery
from ecg_experiment import recovery_views as views


def test_verified_view_rejects_raw_changes_after_decoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A record changed during decoding cannot become a trusted reference."""
    for suffix in (".hea", ".mat"):
        (tmp_path / ("record" + suffix)).write_bytes(b"verified")
    expected = dict.fromkeys((".hea", ".mat"), hashlib.sha256(b"verified").hexdigest())

    def changing_view(raw: Path, stem: str, policy: str) -> tuple:
        (raw / (stem + ".mat")).write_bytes(b"changed")
        return np.zeros((12, 5000), dtype=np.float32), 0, 5000, ""

    monkeypatch.setattr(views, "load_view", changing_view)
    with pytest.raises(ValueError, match="after decoding"):
        views.read_verified_view(tmp_path, "record", "strict_10s", expected)


def test_reference_hashing_never_assesses_quality(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fixed-window hashing calls no quality policy or model, even for unusual signals."""
    signal = np.zeros((12, 5000), dtype=np.float32)

    def canonical(*args: object) -> tuple:
        return signal, 1000, 7000, "", []

    def forbidden_assessment(*args: object) -> None:
        raise AssertionError("Mechanical reference recovery assessed a protected waveform")

    monkeypatch.setattr(recovery, "read_verified_view", canonical)
    monkeypatch.setattr(training, "assess", forbidden_assessment)
    row = {"source_samples": 7000, "view_policy": "ssl_center_crop"}
    result = recovery.hash_challenge_reference(("unused", "release", "stem", row, {}))
    assert result["canonical_sha256"] == hashlib.sha256(signal.tobytes()).hexdigest()
    assert result["window_start"] == 1000
    assert result["skip_reason"] == ""


def test_reference_coverage_requires_hash_or_explicit_contract_skip() -> None:
    """A full reference artifact cannot silently omit intended records or unhashable windows."""
    row = {"record_id": "source:one", "source": "source", "canonical_sha256": "", "skip_reason": ""}
    table = pd.DataFrame([row])
    with pytest.raises(ValueError, match="without explicit skip"):
        recovery.validate_reference_table(table, {"source": 1}, True)
    table["skip_reason"] = "duration_contract"
    assert recovery.validate_reference_table(table, {"source": 1}, True) == {"source": 1}
    with pytest.raises(ValueError, match="Incomplete intended"):
        recovery.validate_reference_table(table, {"source": 2}, True)
