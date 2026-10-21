"""Artificial CPU checks for exact factorial objectives, pairing and state recovery."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ecg_experiment import ssl_synthetic_counterfactual_factorial_training as training
from ecg_experiment.ssl_synthetic_model import localization_update, state_digest
from ecg_experiment.ssl_synthetic_model_v2 import DenseWaveformModel
from ecg_experiment.ssl_synthetic_training import fresh_optimizer


def _native_batch() -> tuple:
    signal = torch.arange(16 * 12 * 25, dtype=torch.float32).reshape(16, 12, 25) / 4800
    target = torch.zeros_like(signal, dtype=torch.bool)
    positive = torch.tensor([4 * p + i for p in range(4) for i in (0, 1)])
    control = torch.tensor([4 * p + 2 for p in range(4) for _ in (0, 1)])
    target[positive, :3, 5:10] = True
    return signal, target, positive, control


def test_per_cell_hinge_and_gradients_in_both_branches() -> None:
    probability = torch.tensor([[[0.6, 0.2, 0.4]], [[0.2, 0.2, 0.2]]], requires_grad=True)
    target = torch.tensor([[[True, True, False]], [[False, False, False]]])
    hinge, consistency = training.paired_penalty(
        probability, target, torch.tensor([0]), torch.tensor([1]), 0.2
    )
    assert float(hinge.detach()) == pytest.approx(0.1)
    assert float(consistency.detach()) == pytest.approx(0.2)
    (hinge + consistency).backward()
    assert probability.grad[0, 0, 1] < 0
    assert probability.grad[1, 0, 1] > 0
    assert probability.grad[0, 0, 2] > 0
    assert probability.grad[1, 0, 2] < 0


def test_equal_case_weights_despite_different_support_sizes() -> None:
    values = torch.tensor(
        [[[0.2, 0.3, 0.3, 0.3]], [[0.6, 0.6, 0.6, 0.3]], [[0.2, 0.2, 0.2, 0.2]], [[0.2, 0.2, 0.2, 0.2]]]
    )
    target = torch.tensor(
        [[[True, False, False, False]], [[True, True, True, False]], [[False] * 4], [[False] * 4]]
    )
    hinge, consistency = training.paired_penalty(
        values, target, torch.tensor([0, 1]), torch.tensor([2, 3]), 0.2
    )
    assert float(hinge) == pytest.approx(0.1)
    assert float(consistency) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="nonempty"):
        training.paired_penalty(
            values, target, torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long), 0.2
        )


def test_bce_update_has_exact_old_loss_model_and_adamw_state(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(14)
    first = DenseWaveformModel()
    second = copy.deepcopy(first)
    config = {"learning_rate": 0.001, "weight_decay": 0.0001}
    a, b = fresh_optimizer(first, config), fresh_optimizer(second, config)
    signal, target, positive, control = _native_batch()
    monkeypatch.setattr(training, "batch_pairs", lambda *args: (signal, target, positive, control))
    expected = localization_update(first, a, signal, target, 20.0)
    result = training.factorial_update({"config": config}, second, b, np.arange(16), "RB", 20.0, "cpu")
    assert result["total"] == expected
    assert result["bce"] == expected
    assert state_digest(first.state_dict()) == state_digest(second.state_dict())
    assert training._state_equal(a.state_dict(), b.state_dict())


def test_batch_requires_exact_same_patient_nuisance_control(monkeypatch: pytest.MonkeyPatch) -> None:
    cases, pairs = [], []
    for subject in range(4):
        nuisance = bool(subject % 2)
        for variant in range(4):
            enabled = nuisance if variant < 2 else variant == 3
            baseline = np.full((1, 4), subject + int(enabled), np.float32)
            support = np.array([[variant < 2, False, False, False]])
            cases.append(
                {
                    "target": variant < 2,
                    "nuisance": enabled,
                    "matched_control_case_index": subject * 4 + (3 if nuisance else 2),
                    "nuisance_parameters": {"enabled": enabled},
                }
            )
            pairs.append(
                {"baseline": baseline, "edited": baseline + support * np.float32(0.1), "support": support}
            )
    module = SimpleNamespace(render_checked=lambda context, index: pairs[index])
    monkeypatch.setitem(sys.modules, "ecg_experiment.ssl_synthetic_counterfactual_factorial_common", module)
    _, _, positive, control = training.batch_pairs({"cases": cases}, np.arange(16))
    assert positive.tolist() == [0, 1, 4, 5, 8, 9, 12, 13]
    assert control.tolist() == [2, 2, 7, 7, 10, 10, 15, 15]
    cases[0]["matched_control_case_index"] = 6
    with pytest.raises(ValueError, match="same-patient"):
        training.batch_pairs({"cases": cases}, np.arange(16))


@pytest.mark.parametrize("arm", training.ARMS)
def test_each_objective_recovers_two_next_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, arm: str
) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(19)
    monkeypatch.setattr(training, "batch_pairs", lambda *args: _native_batch())
    context = {
        "config": {"learning_rate": 0.001, "weight_decay": 0.0001, "paired_margin": 0.2, "paired_weight": 1.0}
    }
    result = training.recovery_check(
        context,
        DenseWaveformModel(),
        np.tile(np.arange(16), (3, 1)),
        arm,
        20.0,
        tmp_path / f"{arm}.pt",
        "cpu",
    )
    assert result["passed"]
    assert result["direct_next_components"] == result["restored_next_components"]
    assert result["optimizer_states_equal"]
    assert result["optimizer_step_count"] == 3


def test_bce_closure_checks_every_loss_and_tensor_identity() -> None:
    old = {
        "final_state": {"encoder_sha256": "e", "head_sha256": "h", "full_model_sha256": "f"},
        "losses": [float(i) for i in range(600)],
    }
    assert training.bce_reproduction("RB", old, {"arms": {"R": old}})["passed"]
    changed = copy.deepcopy(old)
    changed["losses"][411] += 0.001
    with pytest.raises(ValueError, match="600-loss"):
        training.bce_reproduction("RB", changed, {"arms": {"R": old}})
    changed = copy.deepcopy(old)
    changed["final_state"]["head_sha256"] = "different"
    with pytest.raises(ValueError, match="tensor"):
        training.bce_reproduction("RB", changed, {"arms": {"R": old}})


def test_forecast_includes_four_arms_and_192_inference_calls() -> None:
    train = {arm: {"mean_wall_seconds": 0.1, "mean_cpu_seconds": 0.09} for arm in training.ARMS}
    infer = {arm: {"mean_wall_seconds": 0.02, "mean_cpu_seconds": 0.01} for arm in training.ARMS}
    result = training.factorial_forecast(12.0, 11.0, 4.0, train, infer)
    assert result["inference_batches"] == 192
    assert result["stage_gpu_wall_seconds"] == pytest.approx(12 + 2 * 600 * 0.4 + 2 * 192 * 0.02 + 120)
    assert result["stage_cpu_seconds"] == pytest.approx(4 + 11 + 2 * 600 * 0.36 + 2 * 192 * 0.01 + 120)


def test_four_initializations_preserve_common_head_and_within_init_models(tmp_path: Path) -> None:
    torch.manual_seed(101)
    original = DenseWaveformModel()
    torch.manual_seed(102)
    ssl = DenseWaveformModel()
    initial_path, ssl_path = tmp_path / "initial.pt", tmp_path / "ssl.pt"
    torch.save(original.state_dict(), initial_path)
    torch.save(ssl.state_dict(), ssl_path)
    models = training.factorial_models(initial_path, ssl_path)
    heads = {state_digest(model.head.state_dict()) for model in models.values()}
    assert heads == {state_digest(original.head.state_dict())}
    assert state_digest(models["RB"].state_dict()) == state_digest(models["RP"].state_dict())
    assert state_digest(models["SB"].state_dict()) == state_digest(models["SP"].state_dict())
    assert state_digest(models["SB"].encoder.state_dict()) == state_digest(ssl.encoder.state_dict())
    assert all(parameter.requires_grad for model in models.values() for parameter in model.parameters())
