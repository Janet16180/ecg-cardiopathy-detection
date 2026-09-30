"""Scientific invariants for the matched Experiment 038 runner."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from ecg_experiment import xlstm_study as study
from ecg_experiment.cpc_scaling_readout import Example
from ecg_experiment.files import sha256_file


class TinyCPC(nn.Module):
    """Small stochastic model with the CPC runner's loss interface."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(4, 8), nn.Dropout(0.25), nn.Linear(8, 4))

    def forward(self, signal: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        loss = self.encoder(signal).square().mean()
        return loss, {"cmsc": 0.0}


def test_exposure_order_has_exact_passes_and_partial_last_batch() -> None:
    assert study.UPDATES == 1954
    assert study.EXPOSURES - (study.UPDATES - 1) * study.BATCH == 16
    for tier, passes in ((25, 10), (50, 5)):
        selected = study.order(tier)
        assert selected.shape == (250_000,)
        assert np.array_equal(np.bincount(selected, minlength=tier * 1000),
                              np.full(tier * 1000, passes))
        assert np.array_equal(selected, study.order(tier))


def test_runtime_configuration_is_idempotent_for_run_sequence() -> None:
    study.configure_runtime()
    study.configure_runtime()
    assert torch.get_num_threads() == 1
    assert torch.get_num_interop_threads() == 1


def test_checkpoint_replays_stochastic_next_update_exactly(tmp_path: Path) -> None:
    study.seed_everything(38042)
    model = TinyCPC()
    opt = study.optimizer(model)
    signal = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 17
    study.step(model, opt, signal)
    current = {"identity": "frozen"}
    saved = study.checkpoint(model, opt, current, 1, 8, [1.0], {"initial": "test"})
    path = tmp_path / "resume.pt"
    study.write_torch_atomic(path, saved)
    recovered = torch.load(path, map_location="cpu", weights_only=False)
    loss_a = study.step(model, opt, signal)
    state_a = study.cpu_state(model)
    optimizer_a = opt.state_dict()
    rng_a = study.capture_rng_state()
    study.restore(model, opt, recovered, current)
    loss_b = study.step(model, opt, signal)
    assert loss_a == loss_b
    assert study.same_state(state_a, study.cpu_state(model))
    assert study.tree_equal(optimizer_a, opt.state_dict())
    assert study.tree_equal(rng_a, study.capture_rng_state())
    with pytest.raises(ValueError, match="identity changed"):
        study.restore(model, opt, recovered, {"identity": "tampered"})


def test_primary_decision_requires_point_gain_and_positive_lower_bound() -> None:
    assert study.decision({"difference": 0.005, "ci_low": 0.00001})
    assert not study.decision({"difference": 0.00499, "ci_low": 0.00001})
    assert not study.decision({"difference": 0.01, "ci_low": 0.0})


def test_development_loader_ignores_closed_target_fields(tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "heldout_references.csv"
    rows = ["record_id,patient_id,split,target,raw_path\n",
            "ptbxl:closed,ptbxl:hidden,calibration,NOT_A_LABEL,secret\n"]
    cache_rows = []
    index = {}
    for position in range(1306):
        ecg_id = str(position)
        patient = f"p{position % 1173}"
        rows.append(f"ptbxl:{ecg_id},ptbxl:{patient},development,{position % 2},public\n")
        index[ecg_id] = position
        cache_rows.append({"ecg_id": ecg_id, "source": "ptbxl",
                           "split": "validation", "patient_id": patient})
    path.write_text("".join(rows))
    monkeypatch.setattr(study, "VALIDATION", path)
    old_pool = SimpleNamespace(rows=cache_rows, index=index)
    train = [Example("ptbxl:train", "ptbxl:other", 0, {})]
    examples = study.development_only_examples(old_pool, train)
    assert len(examples) == 1306
    assert len({row.patient_id for row in examples}) == 1173
    assert {row.target for row in examples} == {0, 1}


def test_50k_trigger_requires_audited_unchanged_25k_result(tmp_path: Path) -> None:
    directory = study.output(tmp_path, 25)
    directory.mkdir(parents=True)
    prediction = directory / "development_predictions.npz"
    prediction.write_bytes(b"fixed predictions")
    result = {"status": "complete_development_only", "run_50k": True,
              "predictions_sha256": sha256_file(prediction)}
    result_path = directory / "result.json"
    result_path.write_text(json.dumps(result))
    (directory / "audit.json").write_text(json.dumps({
        "status": "passed_development_only", "result_sha256": sha256_file(result_path)}))
    study.require_50k_trigger(tmp_path)
    prediction.write_bytes(b"changed predictions")
    with pytest.raises(ValueError, match="unchanged audited"):
        study.require_50k_trigger(tmp_path)


def test_stage_ledger_counts_failed_and_completed_attempts(tmp_path: Path) -> None:
    study.record_stage(tmp_path, 25, "prepare", 12.5, "failed")
    study.record_stage(tmp_path, 25, "prepare", 30.0, "complete")
    assert study.used_seconds(tmp_path, 25) == 42.5
    with pytest.raises(RuntimeError, match="7200-second"):
        study._elapsed_guard(7200.1, study.time.monotonic())
