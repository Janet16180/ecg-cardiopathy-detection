"""Focused timing, accounting and complete-state recovery contracts."""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ecg_experiment import simdino_runtime_probe040 as probe
from ecg_experiment.cpc_simdino040 import create_model


def test_authoritative_blocks_keep_warmup_and_prespecified_sustained_window() -> None:
    times = [1.0] * 200
    bounds = [(1, 20), (21, 24), (25, 50), (51, 100), (101, 150), (151, 200)]
    blocks = [{"first_update": first, "last_update": last,
               "elapsed_seconds": (last - first + 1) * rate}
              for (first, last), rate in zip(bounds, [5.0, 4.0, 3.0, 2.0, 0.7, 0.9], strict=True)]
    summary = probe.timing_summary(times, blocks)
    assert summary["first_24"]["seconds"] == 116.0
    assert summary["updates25_100"]["seconds"] == 178.0
    assert summary["updates101_200"]["seconds_per_update"] == pytest.approx(0.8)
    assert summary["all_200"]["seconds"] == pytest.approx(374.0)
    assert summary["window1"]["warmup_updates_included"] == 20
    assert summary["window2"]["warmup_updates_included"] == 0
    assert sum(summary[f"window{i}"]["seconds"] for i in range(1, 5)) == pytest.approx(374.0)
    assert summary["updates101_200"]["first_update"] == 101
    assert summary["updates101_200"]["last_update"] == 200
    blocks[-1]["first_update"] = 152
    with pytest.raises(ValueError, match="blocks"):
        probe.timing_summary(times, blocks)


@pytest.mark.parametrize("times", [[1.0] * 199, [0.0] * 200, [float("nan")] * 200,
                                   [float("inf")] * 200, [-1.0] * 200])
def test_malformed_update_timings_raise(times: list[float]) -> None:
    with pytest.raises(ValueError, match="positive"):
        probe.timing_summary(times)


def test_charge_retains_failures_and_never_writes_original_ledgers(tmp_path) -> None:
    original = tmp_path / "outputs" / probe.study.NAME / "day_ledger.json"
    original.parent.mkdir(parents=True)
    original.write_text('{"total_seconds": 319.0}')
    before = original.read_bytes()
    probe.charge(tmp_path, "probe", 10.0, "failed")
    probe.charge(tmp_path, "report", 0.5, "complete")
    saved = json.loads((tmp_path / "outputs" / probe.NAME / "day_ledger.json").read_text())
    assert saved["total_seconds"] == 10.5
    assert [row["status"] for row in saved["attempts"]] == ["failed", "complete"]
    assert original.read_bytes() == before
    for seconds in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="charge"):
            probe.charge(tmp_path, "bad", seconds, "complete")
    with pytest.raises(ValueError, match="charge"):
        probe.charge(tmp_path, "bad", 1.0, "unknown")


def test_paths_and_existing_diagnostic_evidence_cannot_be_overwritten(tmp_path) -> None:
    assert probe.directory(tmp_path, "cpc", "gru").parts[-3:] == (probe.NAME, "cpc", "gru")
    with pytest.raises(ValueError, match="Unknown"):
        probe.directory(tmp_path, "../experiment040_cpc_simdino", "gru")
    target = tmp_path / "outputs" / probe.NAME / "manifest.json"
    target.parent.mkdir(parents=True)
    target.write_text("{}")
    with pytest.raises(ValueError, match="immutable"):
        probe.run(tmp_path)
    assert target.read_text() == "{}"
    assert not (target.parent / "day_ledger.json").exists()


def test_normal_loop_uses_boundary_fences_and_preserves_all_200_updates(tmp_path, monkeypatch) -> None:
    model = SimpleNamespace(last_parts={})
    sync_calls = []
    batch_sizes = []
    monkeypatch.setattr(probe.base, "sync", lambda device: sync_calls.append(device))
    monkeypatch.setattr(probe.base, "batch", lambda data, rows, device: batch_sizes.append(len(rows)))
    monkeypatch.setattr(probe.study, "tensor_groups", lambda state: {"heads": "unchanged"})
    monkeypatch.setattr(probe.study, "validate_movement", lambda *args: None)
    model.state_dict = dict

    def step(current_model, optimizer, signal):
        current_model.last_parts = {"loss": 1.0}
        return 1.0

    def save(*args):
        return {"updates": args[3]}, {"updates": args[3], "cpu_capture_seconds": 0.0,
                                    "write_seconds": 0.0, "reload_seconds": 0.0}

    monkeypatch.setattr(probe.study, "step", step)
    monkeypatch.setattr(probe, "_save_boundary", save)
    measured, saved = probe._normal_updates(model, None, {"objective": "cpc"}, None,
                                            np.arange(25600), tmp_path, probe.time.monotonic(), "cpu")
    assert len(sync_calls) == 13
    assert batch_sizes == [128] * 200
    assert saved["updates"] == 200
    assert [item["updates"] for item in measured["checkpoints"]] == [100, 200]
    blocks = measured["synchronized_update_blocks"]
    assert [(item["first_update"], item["last_update"]) for item in blocks] == [
        (1, 20), (21, 24), (25, 50), (51, 100), (101, 150), (151, 200)]
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["completed_updates"] == 200
    assert progress["losses"] == [1.0] * 200


@pytest.mark.parametrize("objective", ["cpc", "hybrid"])
def test_real_next16_recovery_restores_model_optimizer_rng_and_teacher_clock(objective: str) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(39100)
    model = create_model(objective, "gru", 39042)
    if objective != "cpc":
        model.ema_updates.fill_(200)
        model.ema_total_steps.fill_(1954)
    opt = probe.base.optimizer(model)
    current = {"objective": objective, "seed": 39042}
    data = [(torch.randn(12, 2500),) for _ in range(16)]
    # Populate the AdamW state so replay covers restored moments as well as initial parameters.
    probe.study.step(model, opt, probe.base.batch(data, np.arange(16), "cpu"))
    if objective != "cpc":
        model.ema_updates.fill_(200)
    saved = copy.deepcopy(probe.base.checkpoint(model, opt, current, 200, 25600, [], {}))
    result = probe.recovery_check(model, opt, saved, current, data, np.arange(16), "cpu")
    for key in ("loss_equal", "components_equal", "model_equal", "optimizer_equal", "global_rng_equal"):
        assert result[key], key
    assert result["teacher_updates"] == (0 if objective == "cpc" else 201)
    assert result["teacher_schedule"] == (0 if objective == "cpc" else 1954)
    assert result["restored_student_updates"] == 200
    assert probe.base.same_state(saved["model"], probe.base.cpu_state(model))
    assert probe.base.tree_equal(saved["optimizer"], opt.state_dict())
    assert probe.base.tree_equal(saved["rng"], probe.base.capture_rng_state())
    with pytest.raises(ValueError, match="16 records"):
        probe.recovery_check(model, opt, saved, current, data, np.arange(15), "cpu")
