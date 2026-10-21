"""Artificial forward-only runtime tests without public model or project ECG access."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ecg_experiment import ssl_synthetic_pretrained_standalone as probe


class ToyGenerator(torch.nn.Module):
    """Supply a deterministic finite forward for orchestration tests."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """Return a nonconstant artificial waveform."""
        return torch.tanh(noise * self.weight)


def artificial_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep real frozen seed/budget rules while replacing all remote artifacts by toy bytes."""
    config = json.loads((probe.ROOT / probe.CONFIG_NAME).read_text())
    for asset in config["public_assets"]:
        asset["bytes"] = 3
        asset["sha256"] = hashlib.sha256(b"toy").hexdigest()
    path = tmp_path / probe.CONFIG_NAME
    path.parent.mkdir()
    path.write_text(json.dumps(config))
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    monkeypatch.setattr(probe, "SOURCE_NAMES", (probe.CONFIG_NAME,))
    monkeypatch.setattr(probe.shared, "configure_cpu", lambda config: None)

    def download(asset: dict, destination: Path, guard: object) -> Path:
        target = destination / asset["filename"]
        with target.open("xb") as stream:
            stream.write(b"toy")
        return target

    monkeypatch.setattr(probe.shared, "download_asset", download)
    monkeypatch.setattr(probe.shared, "verify_architecture_equivalence", lambda *args: {"toy": True})
    monkeypatch.setattr(probe.shared, "architecture_class", lambda *args: ToyGenerator)
    monkeypatch.setattr(probe.shared, "load_generator", lambda *args: ToyGenerator().eval())
    return path


def test_complete_artificial_orchestration_retains_actual_replay_and_terminal_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = artificial_config(tmp_path, monkeypatch)
    result = probe.run_probe(tmp_path, config_path)
    study = tmp_path / json.loads(config_path.read_text())["output_relative"]
    assert result["technical_generation_feasible"]
    assert result["forward_attempts"] == result["completed_forwards"] == 6
    assert result["initial_state_sha256"] == result["final_state_sha256"]
    noise = np.load(study / "noise.npy", allow_pickle=False)
    assert noise.shape == (4, 1, 8, 5000)
    expected = (
        torch.empty((1, 8, 5000))
        .uniform_(-1, 1, generator=torch.Generator(device="cpu").manual_seed(2026102901))
        .numpy()
    )
    assert np.array_equal(noise[0], expected)
    for name in ["raw", "author_export", "canonical12"]:
        first = np.load(study / result["case_files"][0][name], allow_pickle=False)
        replay = np.load(study / result["replay_files"][name], allow_pickle=False)
        assert np.array_equal(first, replay)
    ledger = json.loads((study / "prospective_ledger.json").read_text())
    assert ledger["cpu_seconds"] >= result["cpu_seconds"]
    assert ledger["entries"][0]["wall_seconds"] >= result["wall_seconds"]
    assert ledger["entries"][0]["sha256"] == probe.shared.file_sha256(study / "summary.json")
    assert result["prior_cpu_seconds"] == 0
    assert result["historical_receipts_available"] is False
    assert len(result["source_sha256"]) == 1
    assert (study / "source_snapshot" / probe.CONFIG_NAME).exists()
    with pytest.raises(FileExistsError):
        probe.run_probe(tmp_path, config_path)


def test_profile_failure_counts_cost_and_preserves_noise_without_bank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = artificial_config(tmp_path, monkeypatch)
    monkeypatch.setattr(
        probe,
        "profile_forward",
        lambda *args: {"forecast_cpu_seconds": 181.0, "forecast_wall_seconds": 181.0},
    )
    with pytest.raises(RuntimeError, match="retained local failure"):
        probe.run_probe(tmp_path, config_path)
    study = tmp_path / json.loads(config_path.read_text())["output_relative"]
    failure = json.loads((study / "failure.json").read_text())
    assert failure["forward_attempts"] == failure["completed_forwards"] == 1
    assert failure["completed_seeds"] == []
    assert (study / "noise.npy").exists()
    assert not list(study.glob("case*"))
    assert (study / "prospective_ledger.json").exists()
    assert failure["cpu_seconds"] >= 0
    assert failure["peak_rss_bytes"] > 0


def test_failed_generation_preserves_preceding_case_and_exact_forward_accounting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = artificial_config(tmp_path, monkeypatch)
    generate = probe.shared.generate
    calls = 0

    def fail_third(*args: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise ValueError("artificial model failure")
        return generate(*args)

    monkeypatch.setattr(probe.shared, "generate", fail_third)
    with pytest.raises(RuntimeError, match="retained local failure"):
        probe.run_probe(tmp_path, config_path)
    study = tmp_path / json.loads(config_path.read_text())["output_relative"]
    failure = json.loads((study / "failure.json").read_text())
    assert failure["forward_attempts"] == 3
    assert failure["completed_forwards"] == 2
    assert failure["completed_seeds"] == [2026102901]
    assert (study / "case_0_raw.npy").exists()
    assert not (study / "case_1_raw.npy").exists()


def test_changed_frozen_seed_fails_before_source_or_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = artificial_config(tmp_path, monkeypatch)
    config = json.loads(config_path.read_text())
    config["seeds"][0] = 1
    config_path.write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match="retained local failure"):
        probe.run_probe(tmp_path, config_path)
    study = tmp_path / config["output_relative"]
    failure = json.loads((study / "failure.json").read_text())
    assert failure["forward_attempts"] == 0
    assert "four-seed" in failure["failure_detail"]
    assert not (study / "source_snapshot").exists()


def test_source_snapshot_binds_bytes_independently_of_current_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.py"
    source.write_text("x = 1\n")
    study = tmp_path / "study"
    study.mkdir()
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    monkeypatch.setattr(probe, "SOURCE_NAMES", ("source.py",))
    catalog = probe.snapshot_sources(study, lambda: None)
    source.write_text("x = 2\n")
    assert probe.shared.file_sha256(study / "source_snapshot/source.py") == catalog["source.py"]
    assert probe.shared.file_sha256(source) != catalog["source.py"]
