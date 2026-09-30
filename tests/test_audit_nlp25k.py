"""Small receipt fixtures for the read-only NLP 25k artifact audit."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from ecg_experiment.files import sha256_file
from scripts.validation import audit_nlp25k as audit


def fixture_identity() -> dict[str, object]:
    """Build the fixed public 011 identity fields without patient data."""
    return {
        "selected_indices_sha256": audit.SELECTED_SHA256,
        "exposure_order_sha256": "a" * 64,
        "source_counts": {"synthetic": 25_000},
        "subset_size": 25_000,
        "selection_seed": 18046,
        "order_seed": 18047,
        "model_seed": 9001,
        "record_exposures_per_arm": audit.EXPOSURES,
        "batch_size": 128,
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        "arms": list(audit.ARMS["011"]),
    }


def write_fixture(root: Path) -> torch.nn.Module:
    """Write one tiny completed arm and its hash-linked training receipts."""
    output = root / audit.OUTPUTS["011"]
    arm_dir = output / "gru"
    arm_dir.mkdir(parents=True)
    identity = fixture_identity()
    profile_path = output / "profile.json"
    profile_path.write_text(json.dumps({"identity": identity, "gate_passed": True}))
    initial = torch.nn.Module()
    initial.add_module("encoder", torch.nn.Linear(2, 2))
    trained = torch.nn.Module()
    trained.add_module("encoder", torch.nn.Linear(2, 2))
    trained.load_state_dict(initial.state_dict())
    optimizer = torch.optim.AdamW(trained.parameters())
    trained.encoder(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    for state in optimizer.state.values():
        state["step"].fill_(audit.UPDATES)
    checkpoint = arm_dir / "latest.pt"
    torch.save({
        "identity": identity, "arm": "gru", "completed_updates": audit.UPDATES,
        "loss_sum": float(audit.UPDATES), "model": trained.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(), "cuda": [torch.get_rng_state()]},
    }, checkpoint)
    (arm_dir / "complete.json").write_text(json.dumps({
        "identity": identity, "arm": "gru", "completed_updates": audit.UPDATES,
        "record_exposures": audit.EXPOSURES, "mean_cpc_loss": 1.0,
        "checkpoint_sha256": sha256_file(checkpoint),
        "profile_sha256": sha256_file(profile_path),
    }))
    return initial


def test_completed_arm_checks_state_and_reports_pending_peers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid completed arm passes without touching other arms or data files."""
    initial = write_fixture(tmp_path)
    monkeypatch.setattr(audit, "check_model_source", lambda *args: None)
    monkeypatch.setattr(audit, "fresh_models", lambda *args: {"gru": initial})
    result = audit.audit("011", tmp_path)
    assert result["status"] == "pending_training"
    assert result["arms"]["gru"]["changed_encoder_tensors"] > 0
    assert result["arms"]["kda"]["status"] == "pending"


def test_checkpoint_tamper_is_rejected_before_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The checkpoint SHA catches a changed file before pickle deserialization."""
    initial = write_fixture(tmp_path)
    monkeypatch.setattr(audit, "check_model_source", lambda *args: None)
    monkeypatch.setattr(audit, "fresh_models", lambda *args: {"gru": initial})
    checkpoint = tmp_path / audit.OUTPUTS["011"] / "gru/latest.pt"
    with checkpoint.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="checkpoint SHA-256 mismatch"):
        audit.audit("011", tmp_path)


def test_profile_gate_and_budget_are_required(tmp_path: Path) -> None:
    """A failed gate or changed exposure budget cannot pass the audit."""
    output = tmp_path / audit.OUTPUTS["011"]
    output.mkdir(parents=True)
    identity = fixture_identity()
    (output / "profile.json").write_text(json.dumps({"identity": identity,
                                                      "gate_passed": False}))
    with pytest.raises(ValueError, match="cost gate"):
        audit.audit("011", tmp_path)
    identity["record_exposures_per_arm"] = 25_000
    (output / "profile.json").write_text(json.dumps({"identity": identity,
                                                      "gate_passed": True}))
    with pytest.raises(ValueError, match="record_exposures_per_arm"):
        audit.audit("011", tmp_path)
