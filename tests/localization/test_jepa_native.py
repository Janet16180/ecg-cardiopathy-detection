"""Synthetic checks for pinned ECG-JEPA loading, preprocessing and native geometry."""

from __future__ import annotations

import hashlib
import sys
from types import ModuleType

import numpy as np
import pytest
import torch
from scipy.signal import resample

from ecg_experiment.localization import jepa_native as jepa
from ecg_experiment.native_reference_ntt05 import SignalRegion, pool_region, unit_vector


@pytest.fixture
def pinned_files(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    sources = {name: f"# synthetic {name}\n".encode() for name in jepa.SOURCE_SHA256}
    for name, content in sources.items():
        (source_dir / name).write_bytes(content)
    monkeypatch.setattr(
        jepa,
        "SOURCE_SHA256",
        {name: hashlib.sha256(content).hexdigest() for name, content in sources.items()},
    )
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"encoder": torch.nn.Linear(2, 3).state_dict()}, checkpoint)
    bind_checkpoint(checkpoint, monkeypatch)
    monkeypatch.setattr(jepa, "_instantiate_encoder", lambda *_: torch.nn.Linear(2, 3))
    return source_dir, checkpoint


def bind_checkpoint(checkpoint, monkeypatch):
    content = checkpoint.read_bytes()
    monkeypatch.setattr(jepa, "CHECKPOINT_BYTES", len(content))
    monkeypatch.setattr(jepa, "CHECKPOINT_SHA256", hashlib.sha256(content).hexdigest())


def test_preprocessing_preserves_physical_amplitude_and_measured_lead_order():
    time = np.arange(5000) / 500
    signal = np.array([index + 0.5 * np.sin(2 * np.pi * 7 * time) for index in range(12)])
    result = jepa.preprocess_signal(signal)
    expected = signal[list(jepa.INPUT_LEAD_INDICES), ::2]
    np.testing.assert_allclose(result, expected, atol=3e-6)
    np.testing.assert_allclose(result.mean(axis=1), jepa.INPUT_LEAD_INDICES, atol=2e-6)
    assert result.shape == (8, 2500)
    assert result.dtype == np.float32
    assert result.flags.c_contiguous
    np.testing.assert_allclose(jepa.preprocess_signal(signal * 3), result * 3, atol=1e-5)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_preprocessing_is_fft_resampling_and_discards_no_requested_time(dtype):
    rng = np.random.default_rng(2026)
    signal = rng.normal(size=(12, 5000)).astype(dtype)
    expected = resample(signal[list(jepa.INPUT_LEAD_INDICES)], 2500, axis=-1).astype(np.float32)
    np.testing.assert_array_equal(jepa.preprocess_signal(signal), expected)
    if dtype == np.float64:
        premature_cast = resample(signal[list(jepa.INPUT_LEAD_INDICES)].astype(np.float32), 2500, axis=-1)
        assert not np.array_equal(expected, premature_cast)
    altered = signal.copy()
    altered[2:6] = 1e6
    np.testing.assert_array_equal(jepa.preprocess_signal(altered), expected)


@pytest.mark.parametrize("shape", [(5000, 12), (8, 2500), (12, 4999), (12, 5001)])
def test_incompatible_input_geometry_is_rejected(shape):
    with pytest.raises(ValueError, match="canonical"):
        jepa.preprocess_signal(np.zeros(shape))


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, 1j])
def test_nonfinite_or_complex_signal_is_rejected(invalid):
    signal = np.zeros((12, 5000), dtype=complex if invalid == 1j else float)
    signal[0, 0] = invalid
    with pytest.raises(ValueError, match="real finite"):
        jepa.preprocess_signal(signal)


def test_loader_explicitly_requests_safe_cpu_deserialization(pinned_files, monkeypatch):
    source_dir, checkpoint = pinned_files
    original_load = torch.load
    calls = []

    def audited_load(handle, **kwargs):
        calls.append(kwargs)
        return original_load(handle, **kwargs)

    monkeypatch.setattr(torch, "load", audited_load)
    encoder = jepa.load_encoder(source_dir, checkpoint)
    assert calls == [{"map_location": "cpu", "weights_only": True}]
    assert not encoder.training
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    assert encoder._ecg_jepa_embedding_space == jepa.EMBEDDING_SPACE
    assert next(encoder.parameters()).device.type == "cpu"


@pytest.mark.parametrize("failure", ["source", "checkpoint_size", "checkpoint_sha256"])
def test_integrity_failure_precedes_deserialization_and_source_execution(pinned_files, monkeypatch, failure):
    source_dir, checkpoint = pinned_files
    if failure == "source":
        (source_dir / "ecg_jepa.py").write_text("raise RuntimeError('must not execute')\n")
    elif failure == "checkpoint_size":
        checkpoint.write_bytes(checkpoint.read_bytes() + b"x")
    else:
        content = checkpoint.read_bytes()
        checkpoint.write_bytes(b"x" + content[1:])

    def forbidden(*args, **kwargs):
        pytest.fail("Integrity failure must occur before deserialization or source execution")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(jepa, "_instantiate_encoder", forbidden)
    with pytest.raises(ValueError, match="mismatch"):
        jepa.load_encoder(source_dir, checkpoint)


@pytest.mark.parametrize("failure", ["missing", "unexpected", "shape", "prefix"])
def test_loader_rejects_all_state_dict_mismatches_without_key_dropping(pinned_files, monkeypatch, failure):
    source_dir, checkpoint = pinned_files
    state = torch.nn.Linear(2, 3).state_dict()
    if failure == "missing":
        state.pop("bias")
    elif failure == "unexpected":
        state["mask_token"] = torch.zeros(1)
    elif failure == "shape":
        state["weight"] = torch.zeros(3, 4)
    else:
        state = {f"module.{name}": value for name, value in state.items()}
    torch.save({"encoder": state}, checkpoint)
    bind_checkpoint(checkpoint, monkeypatch)
    with pytest.raises(RuntimeError, match="Missing key|Unexpected key|size mismatch"):
        jepa.load_encoder(source_dir, checkpoint)


def test_official_imports_do_not_replace_existing_modules_or_paths(tmp_path, monkeypatch):
    positional_collision = ModuleType("pos_encoding")
    positional_collision.MARKER = "unrelated-module"
    jepa_collision = ModuleType("ecg_jepa")
    monkeypatch.setitem(sys.modules, "pos_encoding", positional_collision)
    monkeypatch.setitem(sys.modules, "ecg_jepa", jepa_collision)
    original_path = sys.path.copy()
    sources = {
        "pos_encoding.py": b"MARKER = 'verified-private-positional'\n",
        "ecg_jepa.py": (
            b"from pos_encoding import MARKER\nimport torch\n"
            b"def ecg_jepa(**kwargs):\n"
            b"    raise RuntimeError('Do not construct unused teacher or predictor')\n"
            b"def MaskTransformer(**kwargs):\n"
            b"    encoder = torch.nn.Linear(2, 3)\n"
            b"    encoder.import_marker = MARKER\n"
            b"    encoder.official_parameters = kwargs\n"
            b"    return encoder\n"
        ),
    }
    encoder = jepa._instantiate_encoder(tmp_path, sources)
    assert encoder.import_marker == "verified-private-positional"
    assert encoder.official_parameters == jepa.DENSE_ENCODER_PARAMETERS
    assert sys.modules["pos_encoding"] is positional_collision
    assert sys.modules["ecg_jepa"] is jepa_collision
    assert sys.path == original_path
    assert "_ecg_jepa_native_official" not in sys.modules
    assert "_ecg_jepa_native_positional" not in sys.modules


class SyntheticEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self._ecg_jepa_embedding_space = jepa.EMBEDDING_SPACE
        self.eval()

    def forward(self, signal, mask):
        assert signal.shape == (1, 8, 50, 50)
        assert mask.shape == (50,)
        assert mask.dtype == torch.bool
        assert not mask.any()
        assert not torch.is_grad_enabled()
        self.received_signal = signal
        values = signal.reshape(1, 400, 50).mean(dim=-1, keepdim=True)
        features = torch.arange(768, dtype=torch.float32, device=signal.device).reshape(1, 1, 768)
        return values + features, mask


def synthetic_signal():
    signal = np.zeros((12, 5000), dtype=np.float32)
    for index in range(12):
        signal[index] = index + np.arange(5000) / 5000
    return signal


def test_dense_tokens_have_exact_lead_major_anchors_and_full_context():
    encoder = SyntheticEncoder()
    signal = synthetic_signal()
    tokens = jepa.encode_signal(encoder, signal, "synthetic-record:full-10s")
    prepared = jepa.preprocess_signal(signal)
    np.testing.assert_array_equal(encoder.received_signal.numpy().reshape(8, 2500), prepared)
    np.testing.assert_allclose(tokens.values[:, 0], prepared.reshape(400, 50).mean(axis=1), atol=2e-6)
    assert tokens.values.shape == (400, 768)
    assert tokens.values.dtype == np.float32
    assert tokens.duration_seconds == 10
    assert tokens.record_frame == "synthetic-record:full-10s"
    assert tokens.embedding_space == jepa.EMBEDDING_SPACE
    assert tokens.context_scope == jepa.CONTEXT_SCOPE
    assert "Full 10-second" in tokens.context_scope
    assert "all eight" in tokens.context_scope
    for index, lead in enumerate(jepa.NATIVE_LEADS):
        assert tokens.token_leads[50 * index : 50 * (index + 1)] == ((lead,),) * 50
        np.testing.assert_allclose(
            tokens.intervals[50 * index : 50 * (index + 1)],
            np.column_stack((np.arange(50) / 5, np.arange(1, 51) / 5)),
        )
    assert not {"III", "aVR", "aVL", "aVF"} & {lead for leads in tokens.token_leads for lead in leads}
    np.testing.assert_array_equal(
        tokens.values, jepa.encode_signal(encoder, signal, tokens.record_frame).values
    )


def test_pooling_uses_only_compatible_anchors_and_partial_patch_occupancy():
    tokens = jepa.encode_signal(SyntheticEncoder(), synthetic_signal(), "synthetic")
    expected = unit_vector((tokens.values[100].astype(float) + 2 * tokens.values[101]) / 3)
    np.testing.assert_allclose(pool_region(tokens, SignalRegion(0.1, 0.4, ("V1",))), expected)
    with pytest.raises(ValueError, match="coverage"):
        pool_region(tokens, SignalRegion(0, 0.2, ("aVR",)))


@pytest.mark.parametrize("failure", ["binding", "training", "gradient", "frame"])
def test_encoder_binding_freeze_and_frame_are_required(failure):
    encoder = SyntheticEncoder()
    frame = "synthetic"
    if failure == "binding":
        encoder._ecg_jepa_embedding_space = "another-checkpoint"
    elif failure == "training":
        encoder.train()
    elif failure == "gradient":
        encoder.anchor.requires_grad_(True)
    else:
        frame = ""
    with pytest.raises(ValueError, match="binding|frozen|frame"):
        jepa.encode_signal(encoder, synthetic_signal(), frame)


@pytest.mark.parametrize("failure", ["extra_token", "width", "dtype", "mask", "nonfinite"])
def test_unexpected_encoder_outputs_are_rejected(failure):
    encoder = SyntheticEncoder()

    def incompatible(signal, mask):
        shape = (1, 401, 768) if failure == "extra_token" else (1, 400, 768)
        shape = (1, 400, 767) if failure == "width" else shape
        values = torch.zeros(shape, dtype=torch.float64 if failure == "dtype" else torch.float32)
        if failure == "mask":
            mask[0] = True
        if failure == "nonfinite":
            values[0, 0, 0] = float("nan")
        return values, mask

    encoder.forward = incompatible
    with pytest.raises(ValueError, match="dense|mask|finite"):
        jepa.encode_signal(encoder, synthetic_signal(), "synthetic")
