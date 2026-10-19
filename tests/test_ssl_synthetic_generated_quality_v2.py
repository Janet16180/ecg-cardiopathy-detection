"""Artificial admission and failure-retention checks for the forward-only budget successor."""

import hashlib
import json
from pathlib import Path

import pytest
import torch

from ecg_experiment import ssl_synthetic_generated_quality_v2 as runner


def fixture_config(tmp_path: Path) -> dict:
    """Create four distinct artificial terminal ledgers and a retained profile failure."""
    config = json.loads((runner.ROOT / runner.CONFIG).read_text())
    failed = tmp_path / config["failed_attempt"]["root"]
    failed.mkdir(parents=True)
    fixture = {
        "failure.json": {
            "forward_counts": {
                "cpu_predecessor_replay": 4,
                "gpu_profile": 1,
                "gpu_generation": 0,
                "gpu_replay": 0,
            },
            "completed_seeds": [],
        },
        "profile.json": {"forecast_wall_seconds": 1550.0},
        "prospective_ledger.json": {"cpu_seconds": 0.8, "gpu_wall_seconds": 6.0},
    }
    for name, value in fixture.items():
        path = failed / name
        path.write_text(json.dumps(value))
        config["failed_attempt"]["sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for index, row in enumerate(config["accounting_prior_ledgers"]):
        path = tmp_path / row["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if index != 2:
            path.write_text(json.dumps({"cpu_seconds": index + 1.0, "gpu_wall_seconds": 0.0}))
        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return config


def test_only_resource_amendment_can_change_scientific_config() -> None:
    config = json.loads((runner.ROOT / runner.CONFIG).read_text())
    runner.verify_config(config)
    config["first_seed"] += 1
    with pytest.raises(ValueError, match="resource/path amendment"):
        runner.verify_config(config)


def test_distinct_prior_self_costs_and_failed_identity(tmp_path: Path) -> None:
    config = fixture_config(tmp_path)
    bindings = {}
    entries = runner.admit_amendment(tmp_path, config, bindings)
    assert len(entries) == 4
    assert sum(row["cpu_seconds"] for row in entries) == 7.8
    assert sum(row["gpu_wall_seconds"] for row in entries) == 6.0
    assert len(bindings) == 6
    runner.verify_amendment_inputs(tmp_path, bindings)
    (tmp_path / config["failed_attempt"]["root"] / "profile.json").write_text("{}")
    with pytest.raises(ValueError, match="receipt identity"):
        runner.admit_amendment(tmp_path, config, {})
    with pytest.raises(ValueError, match="changed during"):
        runner.verify_amendment_inputs(tmp_path, bindings)


def test_actual_controller_profile_failure_retains_v2_cost_and_old_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = fixture_config(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(runner, "CONFIG", str(config_path))
    monkeypatch.setattr(runner, "snapshot", lambda path: {})
    monkeypatch.setattr(runner.original, "set_runtime", lambda: None)
    model = torch.nn.Linear(1, 1)

    def predecessor(parent, destination, counts, attempts):
        counts["cpu_predecessor_replay"] = attempts["cpu_predecessor_replay"] = 4
        return model, {}, {}, {"all_exact": True}

    def failed_profile(model, noises, config, counts, attempts, clock, destination, rows, guard):
        counts["gpu_profile"] = attempts["gpu_profile"] = 1
        clock["gpu_cost"] = 2.5
        runner.shared.write_new_json(destination / "profile.json", {"forecast_wall_seconds": 1801})
        raise ValueError("artificial prospective profile stop")

    monkeypatch.setattr(runner.original, "admit_predecessor", predecessor)
    monkeypatch.setattr(runner.original, "generate_bank", failed_profile)
    with pytest.raises(ValueError, match="prospective profile stop"):
        runner.run_study(tmp_path)
    destination = tmp_path / config["output_root"]
    failure = json.loads((destination / "failure.json").read_text())
    ledger = json.loads((destination / "prospective_ledger.json").read_text())
    assert failure["forward_counts"] == {
        "cpu_predecessor_replay": 4,
        "gpu_profile": 1,
        "gpu_generation": 0,
        "gpu_replay": 0,
    }
    assert failure["completed_seeds"] == []
    assert failure["gpu_lock_wall_seconds"] == ledger["gpu_wall_seconds"] == 2.5
    assert ledger["cumulative_cpu_seconds"] == pytest.approx(7.8 + ledger["cpu_seconds"])
    assert ledger["cumulative_gpu_wall_seconds"] == 8.5
    assert ledger["prior_accounting_complete"] is True
    assert ledger["cpu_seconds"] >= failure["cpu_seconds"]
    assert (tmp_path / config["failed_attempt"]["root"] / "failure.json").exists()
    with pytest.raises(FileExistsError):
        runner.run_study(tmp_path)


def test_wrong_failure_identity_stops_controller_before_any_model_operation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = fixture_config(tmp_path)
    config["failed_attempt"]["sha256"]["failure.json"] = "0" * 64
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(runner, "CONFIG", str(path))
    monkeypatch.setattr(runner, "snapshot", lambda path: {})
    monkeypatch.setattr(runner.original, "set_runtime", lambda: None)

    def forbidden(*args):
        raise AssertionError("model admission must remain closed")

    monkeypatch.setattr(runner.original, "admit_predecessor", forbidden)
    with pytest.raises(ValueError, match="failed-v1 receipt identity"):
        runner.run_study(tmp_path)
    failure = json.loads((tmp_path / config["output_root"] / "failure.json").read_text())
    assert all(value == 0 for value in failure["forward_attempts"].values())
    assert failure["gpu_lock_wall_seconds"] == 0
