"""CPU-only successor parity, factory binding and retained-failure resource accounting."""

import json
from contextlib import nullcontext

import numpy as np
import pytest
import torch

from ecg_experiment import ssl_synthetic_profile_v2 as profile
from ecg_experiment.ssl_synthetic_model import DenseWaveformModel as OriginalModel
from ecg_experiment.ssl_synthetic_model import localization_update, state_digest
from ecg_experiment.ssl_synthetic_profile import PHASES, frozen_optimizer
from scripts.validation import profile_ssl_synthetic_pilot_v2 as command


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
        "batch_size": 16,
    }


def test_factory_preserves_original_initial_parameter_bytes_and_counts():
    actual = profile.make_model(config(), "cpu")
    torch.manual_seed(20261018)
    expected = OriginalModel()
    assert state_digest(actual.state_dict()) == state_digest(expected.state_dict())
    assert sum(p.numel() for p in actual.parameters()) == sum(p.numel() for p in expected.parameters())
    assert not actual.interpolation_kernel.requires_grad


def test_full_model_cpu_forward_input_and_parameter_gradient_parity():
    torch.set_num_threads(1)
    real = np.random.default_rng(91).normal(size=(4, 12, 2500)).astype(np.float32)
    result = profile.cpu_parity(profile.make_model, real, config())
    assert result["parity_gate_passed"]
    assert result["initial_state_identical"]
    assert result["records"] == 2
    assert result["forward_max_abs_error"] < 1e-4
    assert result["input_gradient_max_abs_error"] < 1e-4
    assert result["parameter_gradient_max_abs_error"] < 1e-4


def test_counterfactual_parity_rejects_changed_trainable_model():
    def changed_factory(settings, device):
        model = profile.make_model(settings, device)
        with torch.no_grad():
            model.head.bias.add_(0.1)
        return model

    real = np.random.default_rng(7).normal(size=(4, 12, 2500)).astype(np.float32)
    result = profile.cpu_parity(changed_factory, real, config())
    assert not result["parity_gate_passed"]
    assert not result["initial_state_identical"]
    assert result["forward_max_abs_error"] > 0.01


def test_actual_successor_f_head_update_preserves_encoder_and_fixed_kernel():
    model = profile.make_model(config(), "cpu")
    before = state_digest(model.encoder.state_dict())
    head_before = state_digest(model.head.state_dict())
    kernel_before = model.interpolation_kernel.clone()
    optimizer = frozen_optimizer(model, config())
    signal = torch.randn(2, 12, 100)
    target = torch.zeros_like(signal, dtype=torch.bool)
    target[..., 30:50] = True
    localization_update(model, optimizer, signal, target, 2.0, frozen_encoder=True)
    assert state_digest(model.encoder.state_dict()) == before
    assert state_digest(model.head.state_dict()) != head_before
    assert torch.equal(model.interpolation_kernel, kernel_before)


def test_forecast_charges_original_failed_gpu_and_cpu_additively():
    phases = {
        name: {"mean_timed_step_seconds": value}
        for name, value in zip(PHASES, (0.1, 0.2, 0.3, 0.4), strict=True)
    }
    failed = {"wall_seconds": 0.945, "cpu_seconds": 0.887}
    result = profile.forecast(phases, 10, 5, 100, failed, 1000, config())
    assert result["gpu_forecast_seconds"] == pytest.approx(10 + 0.945 + 2 * 712 + 600)
    assert result["cpu_forecast_seconds"] == pytest.approx(100 + 0.887 + 5 + 2 * 712 + 600)
    assert result["profile_wall_seconds_charged"] == pytest.approx(10.945)
    assert result["prior_cpu_seconds_charged"] == pytest.approx(100.887)
    assert result["previous_failed_profile_wall_seconds"] == failed["wall_seconds"]
    assert result["current_profile_wall_seconds"] == 10


def configure_environment(monkeypatch):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def test_output_cannot_be_overwritten_before_inputs_or_gpu(tmp_path, monkeypatch):
    configure_environment(monkeypatch)
    monkeypatch.setattr(command, "OUTPUT", tmp_path)
    monkeypatch.setattr("sys.argv", ["profile-v2"])
    monkeypatch.setattr(command, "load_failed_profile", lambda: pytest.fail("Read original receipt"))
    monkeypatch.setattr(command, "run_cuda", lambda *_: pytest.fail("Touched GPU"))
    with pytest.raises(FileExistsError, match="must be new"):
        command.main()


def test_original_failure_hash_is_required_before_any_reuse(tmp_path, monkeypatch):
    destination = tmp_path / "outputs/ssl_synthetic_localization/pilot_v1/profile"
    destination.mkdir(parents=True)
    (destination / "summary.json").write_text(json.dumps({"profile_gate_passed": False}))
    monkeypatch.setattr(profile, "ROOT", tmp_path)
    monkeypatch.setattr(profile, "verify_receipt", lambda *_: pytest.fail("Trusted changed failure receipt"))
    with pytest.raises(ValueError, match="failed profile receipt changed"):
        profile.load_failed_profile()


def test_parity_failure_stops_before_any_cuda_replay_or_phase(monkeypatch):
    monkeypatch.setattr(command, "gpu_lock", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(command, "configure_cuda", lambda: {"device": "artificial"})
    monkeypatch.setattr(command, "cpu_parity", lambda *_: {"parity_gate_passed": False})
    monkeypatch.setattr(command, "replay", lambda *_: pytest.fail("Replayed after failed parity"))
    monkeypatch.setattr(command, "measure_phase", lambda *_: pytest.fail("Measured after failed parity"))
    with pytest.raises(ValueError, match="parity failed"):
        command.run_cuda({"phases": {}}, {"real": np.zeros((2, 12, 2500)), "config": config()})


def test_replay_failure_stops_before_timed_phases(monkeypatch):
    monkeypatch.setattr(command, "gpu_lock", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(command, "configure_cuda", lambda: {"device": "artificial"})
    monkeypatch.setattr(command, "cpu_parity", lambda *_: {"parity_gate_passed": True})
    monkeypatch.setattr(command, "replay", lambda *_: {"replay_gate_passed": False})
    monkeypatch.setattr(command, "measure_phase", lambda *_: pytest.fail("Measured after failed replay"))
    with pytest.raises(ValueError, match="CUDA replay failed"):
        command.run_cuda({"phases": {}}, {"real": np.zeros((2, 12, 2500)), "config": config()})
