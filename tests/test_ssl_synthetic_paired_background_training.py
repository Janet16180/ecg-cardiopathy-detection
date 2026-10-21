"""Artificial CPU-only checks for frozen schedule, input isolation and exact AdamW recovery."""

from pathlib import Path

import numpy as np
import pytest
import torch

from ecg_experiment import ssl_synthetic_paired_background_common as common
from ecg_experiment import ssl_synthetic_paired_background_training as training
from ecg_experiment.ssl_synthetic_model import state_digest
from ecg_experiment.ssl_synthetic_model_v2 import DenseWaveformModel


def test_complete_adjacent_schedule_and_actual_exposures() -> None:
    patients = training.patient_schedule()
    rng = np.random.default_rng(20261024)
    expected = np.concatenate([rng.permutation(500) for _ in range(5)])[:2400]
    np.testing.assert_array_equal(patients.ravel(), expected)
    counts = np.bincount(patients.ravel(), minlength=500)
    assert np.sum(counts == 5) == 400
    assert np.sum(counts == 4) == 100
    cases = training.case_schedule(patients)
    assert cases.shape == (600, 16)
    assert len(np.unique(cases)) == 2000
    assert np.sum(cases % 4 < 2) == np.sum(cases % 4 >= 2) == 4800
    np.testing.assert_array_equal(cases.reshape(600, 4, 4)[:, :, 0] // 4, patients)
    np.testing.assert_array_equal(np.diff(cases.reshape(600, 4, 4), axis=2), np.ones((600, 4, 3)))


@pytest.mark.parametrize(
    "wrong",
    [
        np.zeros((600, 3), dtype=int),
        np.full((600, 4), 500),
        np.full((600, 4), -1),
        np.zeros((600, 4), dtype=float),
    ],
)
def test_nontrain_or_wrong_schedule_is_rejected(wrong: np.ndarray) -> None:
    with pytest.raises(ValueError, match="schedule|non-TRAIN"):
        training.case_schedule(wrong)


def test_original_head_and_distinct_encoder_initialization(tmp_path: Path) -> None:
    torch.manual_seed(71)
    initial = DenseWaveformModel()
    torch.manual_seed(72)
    ssl = DenseWaveformModel()
    first, second = tmp_path / "initial.pt", tmp_path / "ssl.pt"
    torch.save(initial.state_dict(), first)
    torch.save(ssl.state_dict(), second)
    models = training.matched_initial_models(first, second)
    assert state_digest(models["R"].state_dict()) == state_digest(initial.state_dict())
    assert state_digest(models["S"].encoder.state_dict()) == state_digest(ssl.encoder.state_dict())
    assert state_digest(models["S"].head.state_dict()) == state_digest(initial.head.state_dict())
    assert state_digest(models["S"].head.state_dict()) != state_digest(ssl.head.state_dict())
    assert set(models) == {"R", "S"}
    assert all(parameter.requires_grad for model in models.values() for parameter in model.parameters())


def test_optimizer_input_is_only_edited_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    def renderer(context: dict, index: int) -> dict:
        return {
            "edited": np.full((12, 25), index + 1, np.float32),
            "baseline": np.full((12, 25), 999, np.float32),
            "support": np.full((12, 25), index == 0, bool),
            "parameters": "private",
        }

    monkeypatch.setattr(common, "render_checked", renderer)

    def update(model, optimizer, signal, support, weight):
        assert signal.shape == support.shape == (2, 12, 25)
        assert signal.dtype == torch.float32
        assert support.dtype == torch.bool
        assert torch.equal(signal[:, 0, 0], torch.tensor([1.0, 2.0]))
        assert support[0].all()
        assert not support[1].any()
        return 0.3

    monkeypatch.setattr(training, "localization_update", update)
    assert training.update_batch({}, None, None, np.array([0, 1]), 20.0, "cpu") == 0.3
    with pytest.raises(ValueError, match="non-TRAIN"):
        training.render_training_batch({}, np.array([2000]))


def test_exact_two_next_updates_restore_adamw_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(13)
    model = DenseWaveformModel()

    def batch(context: dict, indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        signal = torch.arange(2 * 12 * 25, dtype=torch.float32).reshape(2, 12, 25) / 600
        support = torch.zeros_like(signal, dtype=torch.bool)
        support[:, :3, 6:12] = True
        return signal + float(indices[0]) / 100, support

    monkeypatch.setattr(training, "render_training_batch", batch)
    context = {"config": {"learning_rate": 0.001, "weight_decay": 0.0001}}
    result = training.next_update_replay(
        context, model, np.array([[0], [1], [2]]), 5.0, tmp_path / "recovery.pt", "cpu"
    )
    assert result["passed"]
    assert result["optimizer_states_equal"]
    assert result["direct_next_losses"] == result["restored_next_losses"]
    assert result["saved_model_sha256"] == result["loaded_model_sha256"]
    assert result["final_model_sha256"] == result["recovered_final_model_sha256"]
    assert result["optimizer_step_count"] == 3
    assert result["executed_update_calls"] == 5
    with pytest.raises(FileExistsError):
        training.next_update_replay(
            context, DenseWaveformModel(), np.array([[0], [1], [2]]), 5.0, tmp_path / "recovery.pt", "cpu"
        )


def test_forecast_counts_both_arms_and_inference_once() -> None:
    times = {
        "R": {"mean_wall_seconds": 0.1, "mean_cpu_seconds": 0.09},
        "S": {"mean_wall_seconds": 0.2, "mean_cpu_seconds": 0.19},
    }
    result = training.profile_forecast(
        11.0, 10.0, 4.0, times, {"mean_wall_seconds": 0.03, "mean_cpu_seconds": 0.02}
    )
    assert result["stage_gpu_wall_seconds"] == pytest.approx(11 + 2 * 600 * 0.3 + 2 * 96 * 0.03 + 120)
    assert result["stage_cpu_seconds"] == pytest.approx(4 + 10 + 2 * 600 * 0.28 + 2 * 96 * 0.02 + 120)
    assert result["evaluation_reporting_reserve_seconds"] == pytest.approx(120 + 2 * 96 * 0.03)


def test_nonfinite_final_model_rejected() -> None:
    model = DenseWaveformModel()
    with torch.no_grad():
        next(model.parameters()).flatten()[0] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        training._finite_model(model)


def test_qualification_stop_is_retained_without_cuda(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    from contextlib import nullcontext

    destination = tmp_path / "profile"
    monkeypatch.setattr(common, "PROFILE", destination)
    monkeypatch.setattr(training, "gpu_lock", lambda *args, **kwargs: nullcontext())

    def stop(extra_source_names=()):
        raise ValueError("Qualification did not pass")

    monkeypatch.setattr(common, "admit_context", stop)
    with pytest.raises(RuntimeError, match="profile stopped"):
        training.profile_paired_background()
    failure = json.loads((destination / "failure.json").read_text())
    assert failure["progress"]["training_update_calls"] == 0
    assert not failure["performance_scored"]
    assert failure["failure_type"] == "ValueError"
    assert len(failure["source_sha256"]) == len(training.SOURCE_NAMES)
    with pytest.raises(FileExistsError, match="must be new"):
        training.profile_paired_background()
