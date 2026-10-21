"""Guard matched states, fresh optimizers, cache admission and incomplete-run evidence."""

import json
from contextlib import contextmanager

import numpy as np
import pytest
import torch

from ecg_experiment.ssl_synthetic_model import DenseWaveformModel, state_digest
from ecg_experiment.ssl_synthetic_training import (
    check_deadlines,
    execute_fits,
    fresh_optimizer,
    make_arm,
    save_model,
    tensor_batch,
)


def test_arm_construction_matches_heads_and_uses_only_final_ssl_encoder():
    torch.manual_seed(1)
    initial = DenseWaveformModel()
    encoder = {name: value.clone() + 0.1 for name, value in initial.encoder.state_dict().items()}
    arms = {name: make_arm(initial, encoder, name) for name in ["R", "S", "F"]}
    assert state_digest(arms["R"].encoder.state_dict()) == state_digest(initial.encoder.state_dict())
    assert state_digest(arms["S"].encoder.state_dict()) == state_digest(encoder)
    assert state_digest(arms["F"].encoder.state_dict()) == state_digest(encoder)
    for model in arms.values():
        assert state_digest(model.head.state_dict()) == state_digest(initial.head.state_dict())
    assert not any(p.requires_grad for p in arms["F"].encoder.parameters())
    optimizer = fresh_optimizer(arms["F"], {"learning_rate": 0.001, "weight_decay": 0.0001})
    parameters = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert parameters == {id(p) for p in arms["F"].head.parameters()}
    assert len(optimizer.state) == 0


def test_actual_batch_copy_has_no_shared_memory_or_extra_channels():
    array = np.ones((3, 12, 2500), dtype=np.float32)
    array.flags.writeable = False
    batch = tensor_batch(array, np.array([2, 0]), device="cpu")
    assert batch.shape == (2, 12, 2500)
    batch[0, 0, 0] = 5
    assert array[2, 0, 0] == 1


def test_saved_model_is_exact_and_never_overwritten(tmp_path):
    model = DenseWaveformModel()
    path = tmp_path / "model.pt"
    save_model(model, path)
    loaded = torch.load(path, weights_only=True)
    assert state_digest(loaded) == state_digest(model.state_dict())
    with pytest.raises(FileExistsError, match="cannot be overwritten"):
        save_model(model, path)


def test_deadline_guards_both_wall_and_cpu_before_next_step(monkeypatch):
    monkeypatch.setattr("ecg_experiment.ssl_synthetic_training.time.perf_counter", lambda: 10.0)
    monkeypatch.setattr("ecg_experiment.ssl_synthetic_training.time.process_time", lambda: 20.0)
    check_deadlines({"wall_deadline": 11, "cpu_deadline": 21})
    for budget in [{"wall_deadline": 10, "cpu_deadline": 21}, {"wall_deadline": 11, "cpu_deadline": 20}]:
        with pytest.raises(TimeoutError, match="reserve preserved"):
            check_deadlines(budget)


def test_busy_gpu_cannot_create_training_output(tmp_path, monkeypatch):
    destination = tmp_path / "training"
    monkeypatch.setattr("ecg_experiment.ssl_synthetic_training.TRAINING", destination)
    monkeypatch.setattr("ecg_experiment.ssl_synthetic_training.prior_preparation_cpu", lambda: 0)
    monkeypatch.setattr("ecg_experiment.ssl_synthetic_training.BASE", tmp_path)
    (tmp_path / "pilot_v1/profile").mkdir(parents=True)
    (tmp_path / "pilot_v1/profile/summary.json").write_text('{"wall_seconds": 0}')

    @contextmanager
    def busy(device, *, blocking):
        raise RuntimeError("reserved")
        yield

    monkeypatch.setattr("ecg_experiment.ssl_synthetic_training.gpu_lock", busy)
    with pytest.raises(RuntimeError, match="reserved"):
        execute_fits(None, None, {}, None, 1, {}, {"cpu_seconds": 0, "wall_seconds": 0}, {}, {})
    assert not destination.exists()


def test_expired_admission_retains_failure_without_gpu_work(tmp_path, monkeypatch):
    destination = tmp_path / "training"
    monkeypatch.setattr("ecg_experiment.ssl_synthetic_training.TRAINING", destination)
    monkeypatch.setattr("ecg_experiment.ssl_synthetic_training.prior_preparation_cpu", lambda: 0)
    monkeypatch.setattr("ecg_experiment.ssl_synthetic_training.BASE", tmp_path)
    (tmp_path / "pilot_v1/profile").mkdir(parents=True)
    (tmp_path / "pilot_v1/profile/summary.json").write_text('{"wall_seconds": 0}')

    @contextmanager
    def available(device, *, blocking):
        yield

    monkeypatch.setattr("ecg_experiment.ssl_synthetic_training.gpu_lock", available)
    config = {"gpu_seconds_ceiling": 0, "cpu_seconds_ceiling": 0}
    with pytest.raises(TimeoutError, match="reserve preserved"):
        execute_fits(None, None, {}, None, 1, config, {"cpu_seconds": 0, "wall_seconds": 0}, {}, {})
    receipt = json.loads((destination / "failure.json").read_text())
    assert not receipt["training_completed"]
    assert not receipt["performance_scored"]
    assert receipt["failure_type"] == "TimeoutError"
