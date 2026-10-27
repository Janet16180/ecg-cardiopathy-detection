"""Artificial technical checks; no public checkpoint, ECG or network access."""

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from ecg_experiment import ssl_synthetic_pretrained_generator as probe

MIT_TOY = """
import torch
import torch.nn as nn
import torch.nn.functional as F
class Transpose1dLayer(nn.Module):
    pass
class Transpose1dLayer_multi_input(nn.Module):
    pass
class WaveGANGenerator(nn.Module):
    def __init__(
        self, model_size=50, ngpus=1, num_channels=8, post_proc_filt_len=512, verbose=False, upsample=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
    def forward(self, noise):
        return torch.tanh(noise * self.weight)
"""
HF_TOY = (
    MIT_TOY.replace("WaveGANGenerator", "Pulse2pulseGenerator")
    .replace("num_channels=8, post_proc_filt_len", "num_channels=8, latent_dim=100, post_proc_filt_len")
    .replace(
        "        self.weight =",
        """        self.latent_di = latent_dim
        self.fc1 = nn.Linear(latent_dim, 10 * model_size)
        self.weight =""",
    )
)


def test_only_reviewed_classes_execute_and_compatibility_matches() -> None:
    cls = probe.architecture_class(MIT_TOY, HF_TOY)
    assert cls.__name__ == "WaveGANGenerator"
    assert probe.verify_architecture_equivalence(MIT_TOY, HF_TOY) == {
        "transpose_classes_exact": True,
        "generator_exact_except_unused_fc1": True,
    }
    with pytest.raises(ValueError, match="top-level"):
        probe.architecture_class(MIT_TOY + '\nopen("should-not-exist", "w")', HF_TOY)
    with pytest.raises(ValueError, match="beyond"):
        probe.architecture_class(MIT_TOY, HF_TOY.replace("torch.tanh", "torch.sigmoid"))
    with pytest.raises(ValueError, match="import"):
        probe.architecture_class("import requests\n" + MIT_TOY, HF_TOY)


def test_strict_loader_and_real_generation_from_artificial_state(tmp_path: Path) -> None:
    cls = probe.architecture_class(MIT_TOY, HF_TOY)
    state = probe.GeneratorState(cls()).state_dict()
    path = tmp_path / "toy.pt"
    torch.save(state, path)
    loaded = probe.load_generator(path, cls)
    noise = np.linspace(-1, 1, 40000, dtype=np.float32).reshape(1, 8, 5000)
    first = probe.generate(loaded, noise)
    replay = probe.generate(loaded, noise.copy())
    assert all(np.array_equal(a, b) for a, b in zip(first, replay, strict=True))
    assert all(probe.output_checks(*first).values())
    state["model.weight"] = torch.tensor([float("nan")])
    torch.save(state, path)
    with pytest.raises(ValueError, match="finiteness"):
        probe.load_generator(path, cls)
    torch.save({"state_dict": state}, path)
    with pytest.raises(TypeError, match="only a plain tensor"):
        probe.load_generator(path, cls)


def test_loader_rejects_missing_extra_wrong_dtype_and_shape(tmp_path: Path) -> None:
    cls = probe.architecture_class(MIT_TOY, HF_TOY)
    original = probe.GeneratorState(cls()).state_dict()
    path = tmp_path / "toy.pt"
    for state in [
        {k: v for k, v in original.items() if k != "model.weight"},
        {**original, "extra": torch.zeros(1)},
        {**original, "model.weight": torch.ones(1, dtype=torch.float64)},
        {**original, "model.weight": torch.ones(2)},
    ]:
        torch.save(state, path)
        with pytest.raises(ValueError, match="keys differ|dtype, shape or finiteness"):
            probe.load_generator(path, cls)


def test_manual_export_truncation_lead_order_and_algebra() -> None:
    raw = np.zeros((8, 5000), dtype=np.float32)
    raw[:, 0] = np.array([0.0003, -0.0003, 0.0005, -0.0005, 0.001, -0.001, 0.002, -0.002])
    export = (raw * np.float32(6000)).astype(np.int32).T
    assert export[0].tolist() == [1, -1, 3, -3, 6, -6, 12, -12]
    canonical = probe.canonical_leads(export)
    assert canonical[:, 0].tolist() == [1, -1, -2, 0, 1.5, -1.5, 3, -3, 6, -6, 12, -12]
    assert all(probe.output_checks(raw, export, canonical).values())
    canonical[3, 0] = 1
    with pytest.raises(ValueError, match="lead equations"):
        probe.output_checks(raw, export, canonical)
    assert not probe.output_checks(np.zeros_like(raw), np.zeros_like(export), np.zeros_like(canonical))[
        "every_raw_lead_varies"
    ]


def test_generate_rejects_nonfinite_native_output() -> None:
    class Bad(nn.Module):
        def forward(self, noise: torch.Tensor) -> torch.Tensor:
            return noise * float("inf")

    with pytest.raises(ValueError, match="finiteness"):
        probe.generate(Bad(), np.ones((1, 8, 5000), dtype=np.float32))


def test_download_is_exact_bounded_and_preserves_failed_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"public toy bytes"
    monkeypatch.setattr(probe.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(payload))
    asset = {
        "filename": "toy.bin",
        "url": "https://example.invalid/frozen",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    result = probe.download_asset(asset, tmp_path, lambda: None)
    assert result.read_bytes() == payload
    assert not (tmp_path / "toy.bin.partial").exists()
    bad = {**asset, "filename": "bad.bin", "bytes": len(payload) - 1}
    with pytest.raises(ValueError, match="exceeds"):
        probe.download_asset(bad, tmp_path, lambda: None)
    assert (tmp_path / "bad.bin.partial").exists()
    assert not (tmp_path / "bad.bin").exists()
    with pytest.raises(ValueError, match="flat filename"):
        probe.download_asset({**asset, "filename": "../escape"}, tmp_path, lambda: None)


def test_cpu_wall_and_storage_deadlines_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    config = {
        "cpu_wall_limit_seconds": 180,
        "cpu_seconds_ceiling": 14400,
        "output_bytes_ceiling": 100,
        "cache_bytes_ceiling": 1000,
    }
    monkeypatch.setattr(probe.time, "process_time", lambda: 4.0)
    monkeypatch.setattr(probe.time, "perf_counter", lambda: 4.0)
    probe.check_limits(config, {"cpu": 0.0, "wall": 0.0}, 745.0, study_bytes=20, stream_bytes=500)
    for clock in [{"cpu": -175.0, "wall": 0.0}, {"cpu": 0.0, "wall": -175.0}]:
        with pytest.raises(TimeoutError):
            probe.check_limits(config, clock, 745.0, study_bytes=20, stream_bytes=500)
    with pytest.raises(ValueError, match="byte ceiling"):
        probe.check_limits(config, {"cpu": 0.0, "wall": 0.0}, 745.0, study_bytes=100, stream_bytes=500)
    with pytest.raises(ValueError, match="byte ceiling"):
        probe.check_limits(config, {"cpu": 0.0, "wall": 0.0}, 745.0, study_bytes=20, stream_bytes=1000)
    with pytest.raises(TimeoutError, match="Cumulative CPU"):
        probe.check_limits(config, {"cpu": 0.0, "wall": 0.0}, 14399.0, study_bytes=20, stream_bytes=500)


@pytest.mark.parametrize("asset_amendment", [False, True])
def test_early_admission_failure_retains_cost_and_zero_forwards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, asset_amendment: bool
) -> None:
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"public_assets": []} if asset_amendment else {}))
    destination = tmp_path / "study"
    monkeypatch.setattr(probe, "CONFIG", config)
    monkeypatch.setattr(probe, "STUDY", destination)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="retained local failure"):
        probe.run_probe()
    failure = json.loads((destination / "failure.json").read_text())
    assert failure["failure_type"] == "ValueError"
    assert ("CUDA" if asset_amendment else "amendment") in failure["failure_detail"]
    assert failure["cpu_seconds"] >= 0
    assert failure["wall_seconds"] >= 0
    assert failure["forward_attempts"] == failure["completed_forwards"] == 0
    assert failure["clinical_performance_scored"] is False
    with pytest.raises(FileExistsError):
        probe.run_probe()
