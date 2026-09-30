"""Scientific stage and shared-day admission gates for the objective study."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from ecg_experiment import simdino_study040 as study
from scripts.coordination import run_simdino_day040 as coordinator


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def stages(path: Path, attempts: list[dict]) -> None:
    write(path / "stage_walltime.json", {"attempts": attempts})


def test_latest_failure_blocks_readout_despite_complete_training_file(tmp_path, monkeypatch):
    path = study.directory(tmp_path, "hybrid", 39042)
    stages(path, [{"stage": "train", "status": "complete", "elapsed_seconds": 10},
                  {"stage": "train", "status": "failed", "elapsed_seconds": 1}])
    write(path / "training.json", {"status": "complete"})
    monkeypatch.setattr(study, "protocol_commit", lambda root: "prospective")
    monkeypatch.setattr(study.predecessor, "verify_v4", lambda root, tier: {})
    monkeypatch.setattr(study.base, "configure_runtime", lambda: None)
    called = []
    monkeypatch.setattr(study.base, "readout", lambda *args: called.append(args))
    with pytest.raises(ValueError, match="successful latest train"):
        study.execute(tmp_path, "hybrid", 39042, "readout", "cpu")
    assert not called
    assert study.ledger(tmp_path)["attempts"][-1]["status"] == "failed"
    assert study.latest_stage(path, "readout")["status"] == "failed"


def test_failed_attempt_is_charged_and_hooks_restore(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "protocol_commit", lambda root: "prospective")
    monkeypatch.setattr(study.predecessor, "verify_v4", lambda root, tier: {})
    monkeypatch.setattr(study.base, "configure_runtime", lambda: None)
    original = study.base.create_model
    output = study.base.OUTPUT_NAME

    def fail(*args):
        raise RuntimeError("synthetic stage failure")

    monkeypatch.setattr(study.base, "prepare", fail)
    with pytest.raises(RuntimeError, match="synthetic stage failure"):
        study.execute(tmp_path, "simdino", 39042, "prepare", "cpu")
    assert study.base.create_model is original
    assert output == study.base.OUTPUT_NAME
    receipt = study.ledger(tmp_path)
    assert receipt["total_seconds"] > 0
    assert receipt["attempts"][-1]["status"] == "failed"


def test_admission_never_resets_predecessor_budget(tmp_path, monkeypatch):
    write(tmp_path / "outputs" / study.predecessor.NAME / "day_ledger.json",
          {"attempts": [], "total_seconds": 20_000})
    write(tmp_path / "outputs" / study.NAME / "day_ledger.json",
          {"attempts": [], "total_seconds": 1000})
    monkeypatch.setattr(study, "remaining_projection", lambda root, active: 5000)
    gate = study.admission(tmp_path, active_seconds=100)
    assert gate["charged_seconds"] == 21_000
    assert gate["projected_combined_seconds"] == 29_700
    assert not gate["gate_passed"]


def test_admission_keeps_correction_reserve_at_exact_ceiling(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "combined_seconds", lambda root: 10_000)
    monkeypatch.setattr(study, "remaining_projection", lambda root, active: 15_200)
    assert study.admission(tmp_path)["gate_passed"]
    assert not study.admission(tmp_path, active_seconds=0.01)["gate_passed"]


def test_slower_actual_pace_increases_remaining_work():
    profile = {"arms": {arm: {"seconds_per_update": 0.1, "checkpoint_seconds": 0}
                         for arm in study.base.ARMS}}
    regular = study._remaining_training(profile, {"arms": {}}, "hybrid", 39042, None)
    slowed = study._remaining_training(profile, {"arms": {}}, "hybrid", 39042,
                                      ("hybrid", 39042, "gru", 100, 0.2))
    assert slowed > regular
    completed = study._remaining_training(profile, {"arms": {arm: {"updates": study.base.UPDATES}
                                                             for arm in study.base.ARMS}},
                                         "hybrid", 39042, None)
    assert completed == 0


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_ledger_rejects_invalid_elapsed_values(tmp_path, value):
    with pytest.raises(ValueError, match="Malformed elapsed"):
        study.charge(tmp_path, "analysis", 0, "bootstrap", value, "complete", cell=False)


def test_final_analysis_and_failed_diagnosis_are_both_charged(tmp_path):
    study.charge(tmp_path, "analysis", 0, "bootstrap", 10, "complete", cell=False)
    study.charge(tmp_path, "hybrid", 39042, "diagnostic", 2, "failed", cell=False)
    assert study.ledger(tmp_path)["total_seconds"] == 12
    assert len(study.ledger(tmp_path)["attempts"]) == 2


def test_resource_rejection_precedes_all_new_training(tmp_path, monkeypatch):
    monkeypatch.setattr(coordinator, "predecessor_ready", lambda root: True)
    monkeypatch.setattr(coordinator, "_prepare_profiles", lambda root: None)
    monkeypatch.setattr(coordinator, "_profiles", lambda root: {})
    monkeypatch.setattr(study, "admission", lambda root: {"gate_passed": False})
    calls = []
    monkeypatch.setattr(coordinator, "_run_admitted", lambda *args: calls.append(args))
    coordinator.run(tmp_path)
    assert not calls
    status = json.loads((tmp_path / "outputs" / study.NAME / "status.json").read_text())
    assert status["status"] == "resource_gate_failed"


def test_predecessor_wait_does_not_take_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(coordinator, "predecessor_ready", lambda root: False)
    called = []
    monkeypatch.setattr(study, "execute", lambda *args: called.append(args))
    with pytest.raises(RuntimeError, match="Finish Experiment 039"):
        coordinator.run(tmp_path)
    assert not called


def test_numerical_profile_failure_gets_one_training_only_diagnosis(tmp_path, monkeypatch):
    path = study.directory(tmp_path, "cpc", study.SEEDS[0])
    write(path / "profile_progress.json", {"context": "xlstm"})
    monkeypatch.setattr(study, "latest_stage", lambda *args: None)

    def execute(root, objective, seed, stage):
        if stage == "profile":
            raise FloatingPointError("Nonfinite coding-rate factorization")

    monkeypatch.setattr(study, "execute", execute)
    calls = []
    monkeypatch.setattr(study, "diagnose", lambda *args: calls.append(args))
    with pytest.raises(FloatingPointError):
        coordinator._prepare_profiles(tmp_path)
    assert len(calls) == 1
    assert calls[0][1:4] == ("cpc", "xlstm", study.SEEDS[0])
    assert calls[0][4][0]["reason"] == "numerical_profile_failure"


def test_non_numerical_profile_gate_failure_does_not_invent_diagnosis(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "latest_stage", lambda *args: None)

    def execute(*args):
        raise RuntimeError("GPU is reserved by another project run")

    monkeypatch.setattr(study, "execute", execute)
    calls = []
    monkeypatch.setattr(study, "diagnose", lambda *args: calls.append(args))
    with pytest.raises(RuntimeError, match="GPU is reserved"):
        coordinator._prepare_profiles(tmp_path)
    assert not calls


def test_nonfinite_diagnostic_forward_produces_valid_durable_receipt(tmp_path, monkeypatch):
    from ecg_experiment import cpc_simdino040 as model_module

    class Nonfinite(torch.nn.Module):
        def forward(self, signal):
            return torch.tensor(float("nan")), {"rate": float("nan")}

    path = study.directory(tmp_path, "hybrid", 39042) / "gru/latest.pt"
    path.parent.mkdir(parents=True)
    torch.save({"model": {}, "updates": 1, "exposures": 128, "losses": [float("nan")]}, path)
    monkeypatch.setattr(model_module, "create_model", lambda *args: Nonfinite())
    monkeypatch.setattr(study.base, "dataset", lambda *args: None)
    monkeypatch.setattr(study.base, "order", lambda *args: list(range(8)))
    monkeypatch.setattr(study.base, "batch", lambda *args: torch.ones(8, 12, 2500))
    receipt = study.diagnose(tmp_path, "hybrid", "gru", 39042, [{"reason": "numerical_failure"}])
    assert receipt["evidence"]["training_forward_finite"] is False
    assert receipt["evidence"]["saved_state_finite"] is True
    assert receipt["evidence"]["saved_losses_finite"] is False
    assert receipt["evidence"]["training_loss_parts"]["rate"] == "nan"
    assert receipt["evidence"]["first_losses"] == ["nan"]
    saved = tmp_path / "outputs" / study.NAME / "diagnostics/hybrid_gru/diagnostic.json"
    assert json.loads(saved.read_text()) == receipt
    assert study.ledger(tmp_path)["attempts"][-1]["status"] == "complete"
    assert study.diagnose(tmp_path, "hybrid", "gru", 39042, []) == receipt
    assert len(study.ledger(tmp_path)["attempts"]) == 1


def test_throwing_diagnostic_forward_is_preserved():
    class Throwing(torch.nn.Module):
        def forward(self, signal):
            raise FloatingPointError("Coding-rate factorization failed")

    evidence = study._forward_evidence(Throwing(), torch.ones(8, 12, 2500))
    assert evidence["training_forward_or_backward_error"]["error_type"] == "FloatingPointError"


def test_restart_does_not_retry_diagnosed_failed_profile(tmp_path, monkeypatch):
    write(tmp_path / "outputs" / study.NAME / "diagnostics/hybrid_xlstm/diagnostic.json",
          {"specific_defect_identified": False})
    monkeypatch.setattr(coordinator, "predecessor_ready", lambda root: True)
    calls = []
    monkeypatch.setattr(coordinator, "_prepare_profiles", lambda *args: calls.append(args))
    result = coordinator.run(tmp_path)
    assert result["stopped_packages"] == {"hybrid": ["xlstm"]}
    assert not calls


def test_predecessor_requires_final_closed_ledger(tmp_path):
    from ecg_experiment.files import sha256_file

    path = tmp_path / "outputs" / study.predecessor.NAME
    for name in ("aggregate.json", "interactions.json", "diagnostic_gate.json"):
        write(path / name, {})
    for tier in study.predecessor.TIERS:
        for seed in study.SEEDS:
            for encoder in study.predecessor.ENCODERS:
                write(path / encoder / f"seed{seed}/{tier}k/audit.json", {})
    write(path / "day_ledger.json", {"total_seconds": 10})
    assert not coordinator.predecessor_ready(tmp_path)
    write(path / "execution_closed.json", {"status": "complete", "day_ledger_closed": True,
                                          "predecessor_analyses_complete": True, "total_seconds": 10,
                                          "ledger_sha256": sha256_file(path / "day_ledger.json")})
    assert coordinator.predecessor_ready(tmp_path)
    write(path / "day_ledger.json", {"total_seconds": 11})
    assert not coordinator.predecessor_ready(tmp_path)


def test_restart_preserves_partial_numerical_stop_and_completed_cells(tmp_path, monkeypatch):
    write(tmp_path / "outputs" / study.NAME / "diagnostics/hybrid_gru/diagnostic.json",
          {"specific_defect_identified": False})
    write(study.directory(tmp_path, "hybrid", study.SEEDS[0]) / "audit.json", {})
    calls = []
    monkeypatch.setattr(coordinator, "_run_cell",
                        lambda root, obj, seed: calls.append((obj, seed)) or {"scores": {}})
    monkeypatch.setattr(coordinator, "diagnostic_gate", lambda *args: [])
    monkeypatch.setattr(study, "combined_seconds", lambda root: 0)
    result = coordinator._run_admitted(tmp_path, [])
    assert ("hybrid", study.SEEDS[0]) in calls
    assert not any(obj == "hybrid" and seed != study.SEEDS[0] for obj, seed in calls)
    assert len(result["completed_cells"]) == 7
    assert result["stopped_packages"] == {"hybrid": ["gru"]}


def test_direct_training_requires_unchanged_all_package_admission(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="admission receipt"):
        study.require_admission(tmp_path)
    path = tmp_path / "outputs" / study.NAME / "admission.json"
    gate = {"gate_passed": True, "scheduled_original_fits": 18,
            "development_scored_before_admission": False, "first_seed_profile_hashes": {"cpc": "a"}}
    write(path, gate)
    monkeypatch.setattr(study, "first_profile_hashes", lambda root: {"cpc": "a"})
    study.require_admission(tmp_path)
    monkeypatch.setattr(study, "first_profile_hashes", lambda root: {"cpc": "b"})
    with pytest.raises(ValueError, match="identity changed"):
        study.require_admission(tmp_path)
