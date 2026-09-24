"""Integration checks for the released xECG model and signal path."""

import csv
import json

import numpy as np
import pytest
import torch
from scipy.signal import resample

from ecg_experiment.xecg import (
    DEFAULT_CHECKPOINT_DIR,
    LEADS,
    XECGBinaryClassifier,
    load_xecg,
    preprocess_xecg,
)
from scripts.data.prepare_xecg import cache_xecg_views, read_record_float64


def test_preprocessing_matches_upstream_fft_resampling():
    time = np.arange(5000) / 500
    signal = np.stack([np.sin(2 * np.pi * (i + 1) * time) + i / 100 for i in range(12)])
    output = preprocess_xecg(signal)
    assert output.shape == (1000, 12)
    assert output.dtype == np.float32
    np.testing.assert_allclose(output, resample(signal.T, 1000, axis=0).astype(np.float32), rtol=0, atol=0)
    assert not np.allclose(output[:, 0], output[:, 1])
    with pytest.raises(ValueError, match="500 Hz"):
        preprocess_xecg(signal, input_fs=100)
    with pytest.raises(ValueError, match="finite"):
        preprocess_xecg(np.full((12, 5000), np.nan))


def test_float64_reader_reorders_to_xecg_leads_and_checks_units(tmp_path):
    import wfdb

    names = list(reversed(LEADS))
    data = np.tile(np.arange(12, dtype=np.float64), (5000, 1)) / 200
    wfdb.wrsamp("sample", fs=500, units=["mV"] * 12, sig_name=names, p_signal=data,
                fmt=["16"] * 12, adc_gain=[200] * 12, baseline=[0] * 12, write_dir=str(tmp_path))
    signal = read_record_float64(tmp_path, "sample")
    assert signal.shape == (12, 5000)
    assert signal.dtype == np.float64
    np.testing.assert_allclose(signal[:, 0], np.arange(11, -1, -1) / 200)
    with pytest.raises(ValueError, match="escapes"):
        read_record_float64(tmp_path / "sub", "../sample")
    header = tmp_path / "sample.hea"
    header.write_text(header.read_text().replace("/mV", "/uV"))
    with pytest.raises(ValueError, match="physical mV"):
        read_record_float64(tmp_path, "sample")


def test_cache_keys_manifest_and_source(monkeypatch, tmp_path):
    import scripts.data.prepare_xecg as prep

    manifest = tmp_path / "manifest"
    manifest.mkdir()
    for name, ecg_id in (("labeled_train", "1"), ("validation", "2"), ("test", "3")):
        with (manifest / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("ecg_id", "filename_hr", "target"))
            writer.writeheader()
            writer.writerow({"ecg_id": ecg_id, "filename_hr": ecg_id, "target": "0"})
        (tmp_path / f"{ecg_id}.hea").write_text(f"header {ecg_id}")
        (tmp_path / f"{ecg_id}.dat").write_bytes(ecg_id.encode())
    calls = []
    def fake_read(_, name):
        calls.append(name)
        return np.full((12, 5000), float(name), dtype=np.float32)
    monkeypatch.setattr(prep, "read_record_float64", fake_read)
    path = cache_xecg_views(tmp_path, manifest, tmp_path / "cache")
    assert np.load(path, mmap_mode="r").shape == (3, 1000, 12)
    assert calls == ["1", "2", "3"]
    metadata = json.loads((path.parent / "metadata.json").read_text())
    assert metadata["ecg_ids"] == ["1", "2", "3"]
    assert len(metadata["raw_source_sha256"]) == len(metadata["views_sha256"]) == 64
    assert cache_xecg_views(tmp_path, manifest, path.parent) == path
    assert calls == ["1", "2", "3"]
    validation = manifest / "validation.csv"
    validation.write_text(validation.read_text().replace("2,2,0", "2,2,1"))
    with pytest.raises(ValueError, match="fingerprint"):
        cache_xecg_views(tmp_path, manifest, path.parent)


@pytest.mark.skipif(not (DEFAULT_CHECKPOINT_DIR / "model.safetensors").exists(),
                    reason="released weights unavailable")
def test_released_weights_strict_load_and_backward():
    from safetensors import safe_open

    model = load_xecg(drop_path_prob=0)
    assert sum(p.numel() for p in model.parameters()) == 57_021_472
    with safe_open(DEFAULT_CHECKPOINT_DIR / "model.safetensors", framework="pt", device="cpu") as file:
        cuda_r = file.get_tensor("core.model.blocks.0.xlstm.slstm_cell._recurrent_kernel_")
        cuda_b = file.get_tensor("core.model.blocks.0.xlstm.slstm_cell._bias_")
    # Independent explicit gate/head permutation. This tests more than shape matching.
    expected_r = cuda_r.reshape(4, 256, 4, 256).permute(0, 2, 3, 1).reshape(4, 1024, 256)
    expected_b = cuda_b.reshape(4, 4, 256).permute(1, 0, 2).reshape(-1)
    cell = model.core.model.blocks[0].xlstm.slstm_cell
    torch.testing.assert_close(cell._recurrent_kernel_.detach(), expected_r)
    torch.testing.assert_close(cell._bias_.detach(), expected_b)

    # One patch is enough to exercise all nine official blocks and autograd on CPU.
    model.train()
    pooled, tokens = model(torch.randn(1, 25, 12))
    assert pooled.shape == (1, 1024)
    assert tokens.shape == (1, 1, 1024)
    pooled.square().mean().backward()
    grad = model.patch_embedding.conv.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_classifier_head_is_plain_linear():
    class FakeBackbone(torch.nn.Module):
        cls_type = "avg"
        embedding_size = 4

        def forward(self, signal):
            return torch.ones(signal.shape[0], 4), None

    classifier = XECGBinaryClassifier(FakeBackbone())
    assert isinstance(classifier.head, torch.nn.Linear)
    assert classifier(torch.zeros(2, 1000, 12)).shape == (2,)
    with pytest.raises(ValueError, match="Expected"):
        classifier(torch.zeros(2, 12, 1000))
