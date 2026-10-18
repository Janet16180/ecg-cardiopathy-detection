"""CPU-only profile forecast, source-guard and frozen-encoder integrity checks."""

import json
from contextlib import nullcontext

import numpy as np
import pytest
import torch

from ecg_experiment import ssl_synthetic_profile as profile
from ecg_experiment.ssl_synthetic_model import DenseWaveformModel, localization_update, state_digest
from scripts.validation import profile_ssl_synthetic_pilot as command


def config():
    return {
        "gpu_seconds_ceiling": 7200,
        "cpu_seconds_ceiling": 14400,
        "cache_bytes_ceiling": 4294967296,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "seed": 20261018,
        "encoder_width": 32,
        "encoder_stride": 5,
        "physical_scale_mv": 2.0,
    }


def phase_times():
    return {
        name: {"mean_timed_step_seconds": value}
        for name, value in zip(profile.PHASES, (0.1, 0.2, 0.3, 0.4), strict=True)
    }


def test_forecast_accounts_all_arms_inference_and_prior_cpu():
    result = profile.resource_forecast(phase_times(), 10, 5, 100, 1000, config())
    assert result["remaining_single_pass_seconds"] == pytest.approx(100 + 240 + 180 + 192)
    assert result["gpu_forecast_seconds"] == pytest.approx(10 + 2 * 712 + 600)
    assert result["cpu_forecast_seconds"] == pytest.approx(100 + 5 + 2 * 712 + 600)
    assert result["planned_phase_steps"]["trainable_localization"] == 1200
    assert result["planned_phase_steps"]["inference"] == 480
    assert result["resource_gate_passed"]
    assert result["is_forecast_not_full_training_measurement"]


@pytest.mark.parametrize(
    ("ceiling", "value", "gate"),
    [
        ("gpu_seconds_ceiling", 2033.99, "gpu_forecast_gate_passed"),
        ("cpu_seconds_ceiling", 2128.99, "cpu_forecast_gate_passed"),
        ("cache_bytes_ceiling", 100999, "cache_budget_gate_passed"),
    ],
)
def test_each_resource_ceiling_independently_rejects(ceiling, value, gate):
    limits = config() | {ceiling: value}
    result = profile.resource_forecast(phase_times(), 10, 5, 100, 1000, limits)
    assert not result[gate]
    assert not result["resource_gate_passed"]


def test_exact_ceiling_is_admitted_with_reserved_summary_bytes():
    limits = config() | {
        "gpu_seconds_ceiling": 2034,
        "cpu_seconds_ceiling": 2129,
        "cache_bytes_ceiling": 101000,
    }
    assert profile.resource_forecast(phase_times(), 10, 5, 100, 1000, limits)["resource_gate_passed"]


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_step_timing_cannot_admit(value):
    phases = phase_times()
    phases["ssl"]["mean_timed_step_seconds"] = value
    with pytest.raises(ValueError, match="finite and positive"):
        profile.resource_forecast(phases, 1, 1, 1, 1, config())


def test_frozen_optimizer_updates_head_and_preserves_every_encoder_byte():
    torch.set_num_threads(1)
    torch.manual_seed(20261018)
    model = DenseWaveformModel()
    encoder_before = state_digest(model.encoder.state_dict())
    head_before = state_digest(model.head.state_dict())
    optimizer = profile.frozen_optimizer(model, config())
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert optimizer_ids == {id(parameter) for parameter in model.head.parameters()}
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    signal = torch.randn(2, 12, 100)
    support = torch.zeros_like(signal, dtype=torch.bool)
    support[:, :, 40:50] = True
    localization_update(model, optimizer, signal, support, 3.0, frozen_encoder=True)
    assert state_digest(model.encoder.state_dict()) == encoder_before
    assert state_digest(model.head.state_dict()) != head_before
    assert not model.encoder.training


def test_fresh_profile_initialization_matches_default_pilot_components(monkeypatch):
    monkeypatch.setattr(DenseWaveformModel, "to", lambda self, device: self)
    actual = profile.fresh_model(config())
    torch.manual_seed(20261018)
    expected = DenseWaveformModel()
    assert state_digest(actual.encoder.state_dict()) == state_digest(expected.encoder.state_dict())
    assert state_digest(actual.head.state_dict()) == state_digest(expected.head.state_dict())


def configure_environment(monkeypatch):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def test_existing_profile_directory_rejected_before_inputs_or_gpu(tmp_path, monkeypatch):
    configure_environment(monkeypatch)
    monkeypatch.setattr(command, "OUTPUT", tmp_path)
    monkeypatch.setattr("sys.argv", ["profile"])
    monkeypatch.setattr(command, "load_profile_inputs", lambda: pytest.fail("Loaded cached inputs"))
    monkeypatch.setattr(command, "run_profile", lambda *_: pytest.fail("Touched GPU"))
    with pytest.raises(FileExistsError, match="must be new"):
        command.main()


def test_failed_prerequisite_is_retained_in_new_receipt_without_gpu(tmp_path, monkeypatch):
    configure_environment(monkeypatch)
    monkeypatch.setattr("sys.argv", ["profile"])
    destination = tmp_path / "profile"
    monkeypatch.setattr(command, "OUTPUT", destination)
    monkeypatch.setattr(command, "SOURCE_FILES", ())
    monkeypatch.setattr(command, "to_stored", lambda path: str(path))
    monkeypatch.setattr(command.subprocess, "check_output", lambda *args, **kwargs: "artificial\n")

    def failed():
        raise ValueError("artificial prerequisite hash changed")

    monkeypatch.setattr(command, "load_profile_inputs", failed)
    monkeypatch.setattr(command, "run_profile", lambda *_: pytest.fail("Touched GPU"))
    assert command.main() == 2
    result = json.loads((destination / "summary.json").read_bytes())
    assert not result["profile_gate_passed"]
    assert result["failure_type"] == "ValueError"
    assert "hash changed" in result["failure_detail"]
    assert not result["training_started"]


def test_replay_failure_stops_before_resource_phases(monkeypatch):
    monkeypatch.setattr(command, "gpu_lock", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(command, "configure_cuda", lambda: {"device": "artificial"})
    monkeypatch.setattr(command, "exact_cuda_replay", lambda *_: {"replay_gate_passed": False})
    monkeypatch.setattr(command, "profile_phase", lambda *_: pytest.fail("Ran phase after failed replay"))
    report = {"phases": {}}
    with pytest.raises(ValueError, match="CUDA replay failed"):
        command.run_profile(report, {"real": np.zeros((2, 12, 2500)), "config": config()})
    assert not report["replay"]["replay_gate_passed"]
    assert report["phases"] == {}


@pytest.mark.parametrize("target", ["real_cache", "synthetic_bank"])
def test_actual_payload_hash_tampering_rejected_before_loading_arrays(tmp_path, monkeypatch, target):
    from ecg_experiment import ssl_synthetic_preparation as preparation
    from ecg_experiment.files import sha256_file

    base = tmp_path / "outputs/ssl_synthetic_localization"

    def write(relative, data):
        path = base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return path

    real = base / "real_cache_v1/signals.npy"
    real.parent.mkdir(parents=True)
    real.write_bytes(b"artificial real cache")
    bank = base / "pilot_v1/preparation/train_bank/signals.npy"
    bank.parent.mkdir(parents=True)
    bank.write_bytes(b"artificial synthetic bank")
    write("input_audit_v1/summary.json", {"input_integrity_gate_passed": True})
    write(
        "real_cache_v1/summary.json",
        {"cache_gate_passed": True, "output_sha256": {"signals.npy": sha256_file(real)}},
    )
    write(
        "generator_pilot_v2/qualification.json",
        {
            "all_passed": True,
            "case_admission": {"count": 3060, "failed_cases": 0},
            "role_integrity": {"all_disjoint": True},
        },
    )
    write(
        "generator_pilot_v1/qualification.json", {"all_passed": False, "case_admission": {"failed_cases": 28}}
    )
    prep = write(
        "pilot_v1/preparation/summary.json",
        {
            "preparation_gate_passed": True,
            "bank_sha256": {bank.relative_to(tmp_path).as_posix(): sha256_file(bank)},
        },
    )
    script = base / "pilot_v1/prep_audit_v1/audit_script.py"
    script.parent.mkdir(parents=True)
    script.write_text("# artificial audit\n")
    write(
        "pilot_v1/prep_audit_v1/summary.json",
        {
            "all_passed": True,
            "preparation_receipt_sha256": sha256_file(prep),
            "script_sha256": sha256_file(script),
        },
    )
    changed = real if target == "real_cache" else bank
    changed.write_bytes(b"corrupted")
    original_verify = preparation.verify_hashes
    monkeypatch.setattr(preparation, "verify_hashes", lambda hashes: original_verify(hashes, tmp_path))
    monkeypatch.setattr(profile, "verify_hashes", lambda hashes: original_verify(hashes, tmp_path))
    monkeypatch.setattr(profile, "ROOT", tmp_path)
    monkeypatch.setattr(profile, "to_stored", lambda path: path.relative_to(tmp_path).as_posix())
    monkeypatch.setattr(
        profile.np, "load", lambda *_args, **_kwargs: pytest.fail("Loaded unverified payload")
    )
    with pytest.raises(ValueError, match="hash changed"):
        profile.load_profile_inputs()
