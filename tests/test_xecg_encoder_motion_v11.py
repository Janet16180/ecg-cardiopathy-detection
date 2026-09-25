"""Cheap checks for v11's device/API boundary."""

from __future__ import annotations

import json
from contextlib import nullcontext
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from ecg_experiment import xecg_encoder_motion_v11 as v11
from ecg_experiment.xecg_encoder_motion_v11 import device_receipt, installed_cuda_api, installed_device_api
from ecg_experiment.xecg_encoder_motion_v11_verification import LeafGraph


class StrictCuda:
    """Expose real CUDA method names while deliberately lacking get_device_count."""

    def device_count(self) -> int:
        """Return the single device count."""
        return 1

    def is_available(self) -> bool:
        """Return CUDA availability."""
        return True

    def get_device_name(self, index: int) -> str:
        """Return the only device identity."""
        if index != 0:
            raise ValueError(index)
        return "Tesla V100-SXM2-16GB"


def fake_torch(cuda: object) -> SimpleNamespace:
    """Provide only the torch attributes used by the receipt."""
    return SimpleNamespace(
        cuda=cuda,
        __version__="2.6.0+cu124",
        version=SimpleNamespace(cuda="12.4"),
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 90100)),
        are_deterministic_algorithms_enabled=lambda: False,
        get_float32_matmul_precision=lambda: "highest",
    )


def test_device_receipt_uses_real_api_names() -> None:
    """The missing historical get_device_count method must never be called."""
    result = device_receipt(
        fake_torch(StrictCuda()),
        lambda *_args, **_kwargs: SimpleNamespace(stdout="Tesla V100-SXM2-16GB, 580.126.20, 16384 MiB\n"),
    )
    assert result["historical_driver_version"] == "not recorded in frozen v9 receipts"


def test_missing_and_malformed_cuda_queries_fail() -> None:
    """Missing callable methods or a malformed device result stop preflight."""
    with pytest.raises(RuntimeError, match="device_count"):
        installed_cuda_api(SimpleNamespace(is_available=lambda: True, get_device_name=lambda _: "V100"))
    cuda = StrictCuda()
    cuda.device_count = lambda: "1"  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="exactly one"):
        device_receipt(fake_torch(cuda), lambda *_args, **_kwargs: None)
    with pytest.raises(RuntimeError, match="Malformed nvidia-smi"):
        device_receipt(fake_torch(StrictCuda()), lambda *_args, **_kwargs: SimpleNamespace(stdout="bad"))
    incomplete = fake_torch(StrictCuda())
    incomplete.backends.cudnn.version = None
    with pytest.raises(RuntimeError, match="version"):
        installed_device_api(incomplete)


def _small_manifest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, seed: int = 47) -> tuple[Path, str]:
    """Freeze a tiny versioned source map for the cheap preflight boundary."""
    root = tmp_path / "repo"
    output = root / "outputs/v11"
    output.mkdir(parents=True)
    source = root / "source.py"
    source.write_text("VALUE = 47\n")
    mapping = {"source.py": sha256(source.read_bytes()).hexdigest()}
    source_map = output / "sources.json"
    source_map.write_text(json.dumps(mapping))
    manifest = output / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 11,
                "seed": seed,
                "output_dir": "outputs/v11",
                "sources": "outputs/v11/sources.json",
                "sources_sha256": sha256(source_map.read_bytes()).hexdigest(),
                "stages": ["preflight"],
            }
        )
    )
    monkeypatch.setattr(v11, "ROOT", root)
    monkeypatch.setattr(v11, "OUT", output)
    monkeypatch.setattr(v11, "SOURCE_FILES", ("source.py",))
    return manifest, sha256(manifest.read_bytes()).hexdigest()


def test_wrong_seed_rejected_before_large_read(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A bridge or old-run identity cannot enter the seed-47 preflight."""
    manifest, digest = _small_manifest(monkeypatch, tmp_path, seed=46)
    with pytest.raises(ValueError, match="production seed"):
        v11.preflight(manifest, digest, "preflight", "cpu")


def test_failing_device_query_precedes_large_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A bad real-device result stops without invoking any historical hash reader."""
    manifest, digest = _small_manifest(monkeypatch, tmp_path)
    original = v11.stable_sha256
    calls: list[str] = []

    def guarded_hash(path: Path) -> tuple[str, int]:
        """Reject an attempted cache or checkpoint read."""
        calls.append(path.name)
        if path.suffix in (".npy", ".pt", ".safetensors"):
            raise AssertionError("Large historical hash reader was invoked")
        return original(path)

    monkeypatch.setattr(v11, "stable_sha256", guarded_hash)
    monkeypatch.setattr(v11, "gpu_lock", lambda *_args, **_kwargs: nullcontext())

    def bad_device(*_args: object) -> None:
        """Fail before the historical hash reader can be reached."""
        raise RuntimeError("bad device")

    monkeypatch.setattr(v11, "device_receipt", bad_device)
    with pytest.raises(RuntimeError, match="bad device"):
        v11.preflight(manifest, digest, "preflight", "cuda")
    assert calls == ["manifest.json", "sources.json", "source.py"]


def test_graph_rejects_missing_and_conflicting_leaves() -> None:
    """Every transitive edge must resolve to exactly one expected byte identity."""
    graph = LeafGraph()
    name = "ecg_experiment/xecg_encoder_motion_v11.py"
    graph.add(name, "a" * 64, "first")
    with pytest.raises(ValueError, match="Conflicting"):
        graph.add(name, "b" * 64, "second")
    with pytest.raises(ValueError, match="Missing"):
        graph.add("outputs/absent-v11-leaf.pt", "a" * 64, "missing")
