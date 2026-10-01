"""Teacher clocks, finite scalar histories and recovery before budget guards."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ecg_experiment import simdino_study040 as study
from ecg_experiment.files import sha256_json


@pytest.mark.parametrize("objective", study.OBJECTIVES)
@pytest.mark.parametrize("updates", [0, 1, 24, 1954])
def test_teacher_clock_matches_objective_and_successful_boundary(objective: str, updates: int) -> None:
    clock = SimpleNamespace(
        ema_updates=torch.tensor(0 if objective == "cpc" else updates),
        ema_total_steps=torch.tensor(study.base.UPDATES if objective != "cpc" and updates else 0),
    )
    study.validate_teacher_clock(clock, objective, updates)
    for name in ("ema_updates", "ema_total_steps"):
        original = getattr(clock, name).clone()
        setattr(clock, name, original + 1)
        with pytest.raises(ValueError, match="EMA count or schedule"):
            study.validate_teacher_clock(clock, objective, updates)
        setattr(clock, name, original)


@pytest.mark.parametrize("updates", [-1, 1955])
def test_teacher_clock_rejects_invalid_student_boundaries(updates: int) -> None:
    with pytest.raises(ValueError, match="Invalid objective"):
        study.validate_teacher_clock(SimpleNamespace(), "hybrid", updates)


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("location", ["loss", "component", "norm"])
def test_finite_history_checks_python_scalars(nonfinite: float, location: str) -> None:
    losses = [1.0, 2.0]
    histories = [
        {"cpc": 1.0, "rate": -0.01, "gradient_norm": 1.5},
        {"cpc": 2.0, "rate": -0.02, "gradient_norm": 1.25},
    ]
    study.validate_history(losses, histories)
    if location == "loss":
        losses[1] = float(nonfinite)
    else:
        histories[1]["gradient_norm" if location == "norm" else "rate"] = float(nonfinite)
    with pytest.raises(ValueError, match="Nonfinite objective"):
        study.validate_history(losses, histories)


class TinyModel(torch.nn.Module):
    """Minimal stateful student for deterministic checkpoint-boundary tests."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.register_buffer("ema_updates", torch.tensor(0))
        self.register_buffer("ema_total_steps", torch.tensor(0))
        self.last_parts = {"cpc": 0.0, "gradient_norm": 1.0}


def _mock_training(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setattr(study.base, "SEED", 39042)
    monkeypatch.setattr(study.base, "UPDATES", 2)
    monkeypatch.setattr(study.base, "BATCH", 2)
    monkeypatch.setattr(study.base, "EXPOSURES", 4)
    monkeypatch.setattr(study.base, "create_model", lambda *args: TinyModel())
    monkeypatch.setattr(study.base, "output", lambda *args: path)
    monkeypatch.setattr(study.base, "order", lambda tier: np.arange(4))
    monkeypatch.setattr(study.base, "batch", lambda *args: torch.ones(2))
    monkeypatch.setattr(study.base, "_elapsed_guard", lambda *args: None)
    monkeypatch.setattr(study.base, "sync", lambda *args: None)
    monkeypatch.setattr(study, "tensor_groups", lambda state: {"student": str(float(state["weight"]))})

    def step(model: TinyModel, optimizer: torch.optim.Optimizer, signal: torch.Tensor) -> float:
        with torch.no_grad():
            model.weight.add_(1)
        model.last_parts = {"cpc": float(model.weight.detach()), "gradient_norm": 1.0}
        return float(model.weight.detach())

    monkeypatch.setattr(study, "step", step)


def test_checkpoint_is_saved_before_failing_pace_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_training(monkeypatch, tmp_path)

    def fail_guard(*args: object) -> None:
        raise RuntimeError("pace stopped")

    monkeypatch.setattr(study.base, "_pace_guard", fail_guard)
    with pytest.raises(RuntimeError, match="pace stopped"):
        study._train_arm(tmp_path, 25, "gru", None, {"objective": "cpc"}, {}, 0, 0, "cpu")
    saved = torch.load(tmp_path / "gru/latest.pt", map_location="cpu", weights_only=False)
    assert saved["updates"] == 2
    assert saved["exposures"] == 4
    assert saved["losses"] == [1.0, 2.0]
    assert saved["model"]["weight"] == 2
    assert len(saved["objective_components"]) == 2


@pytest.mark.parametrize("drift", ["clock", "schedule", "loss", "component"])
def test_resume_rejects_invalid_teacher_or_scalar_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    _mock_training(monkeypatch, tmp_path)
    current = {"objective": "cpc"}
    model = TinyModel()
    opt = study.base.optimizer(model)
    saved = study.base.checkpoint(model, opt, current, 1, 2, [1.0], study.tensor_groups(model.state_dict()))
    saved.update({"objective_components": [{"cpc": 1.0}], "elapsed_seconds": 0.0})
    if drift == "clock":
        saved["model"]["ema_updates"] = torch.tensor(1)
    if drift == "schedule":
        saved["model"]["ema_total_steps"] = torch.tensor(2)
    if drift == "loss":
        saved["losses"][0] = float("nan")
    if drift == "component":
        saved["objective_components"][0]["cpc"] = float("inf")
    latest = tmp_path / "gru/latest.pt"
    latest.parent.mkdir()
    torch.save(saved, latest)
    with pytest.raises(ValueError, match="EMA count or schedule|Nonfinite objective"):
        study._train_arm(tmp_path, 25, "gru", None, current, {}, 0, 0, "cpu")


@pytest.mark.parametrize("drift", ["clock", "schedule", "loss", "component"])
def test_audit_rejects_invalid_teacher_or_scalar_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    _mock_training(monkeypatch, tmp_path)
    monkeypatch.setattr(study.base, "ARMS", ("gru",))
    monkeypatch.setattr(study, "validate_movement", lambda *args: None)
    model = TinyModel()
    current = {"objective": "cpc", "seed": 39042}
    state = model.state_dict()
    saved = {
        "model": state,
        "identity_sha256": sha256_json(current),
        "updates": 2,
        "exposures": 4,
        "losses": [1.0, 2.0],
        "objective_components": [{"cpc": 1.0}, {"cpc": 2.0}],
        "initial": study.tensor_groups(state),
    }
    if drift == "clock":
        saved["model"]["ema_updates"] = torch.tensor(1)
    if drift == "schedule":
        saved["model"]["ema_total_steps"] = torch.tensor(2)
    if drift == "loss":
        saved["losses"][0] = float("nan")
    if drift == "component":
        saved["objective_components"][0]["cpc"] = float("inf")
    torch.save(saved, tmp_path / "latest.pt")
    training = {
        "arms": {
            "gru": {
                "checkpoint": "latest.pt",
                "initial": saved["initial"],
                "final": study.tensor_groups(saved["model"]),
            }
        }
    }
    with pytest.raises(ValueError, match="EMA count or schedule|Nonfinite objective"):
        study.audit_groups(tmp_path, training, current)
