"""Synthetic checks of the 100 Hz waveform cache, its scaling and its dataset."""

import csv
import json

import numpy as np
import pytest
import torch

from ecg_experiment.data import LEADS, ECGDataset, Waveforms, build_cache


def _write_cache(directory, signals, ids):
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "signals.npy", signals)
    np.save(directory / "ecg_ids.npy", np.asarray(ids, dtype=np.int64))
    return directory


def _write_manifest(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_training_scale_is_demeaned_rms_over_training_rows_only(tmp_path):
    rng = np.random.default_rng(0)
    signals = (rng.standard_normal((300, 12, 1000)) * np.arange(1, 13)[:, None] + 5).astype(np.float32)
    signals[:, 3] = 0
    waveforms = Waveforms(_write_cache(tmp_path, signals, range(10, 310)))
    train = [{"ecg_id": str(ecg_id)} for ecg_id in range(10, 280)]
    scale = waveforms.training_scale(train)

    expected = signals[:270].astype(np.float64)
    expected -= expected.mean(axis=-1, keepdims=True)
    expected = np.maximum(np.sqrt(np.square(expected).mean(axis=(0, 2))), 1e-6)
    assert scale.dtype == np.float32
    np.testing.assert_allclose(scale, expected, rtol=1e-6)
    assert scale[3] == np.float32(1e-6)


def test_dataset_demeans_scales_and_clips(tmp_path):
    signal = np.zeros((1, 12, 1000), dtype=np.float32)
    signal[0, :, 0] = 100
    waveforms = Waveforms(_write_cache(tmp_path, signal, [7]))
    dataset = ECGDataset(waveforms, [{"ecg_id": "7", "target": "1"}], np.full(12, 2, dtype=np.float32))
    x, y = dataset[0]
    assert len(dataset) == 1
    assert x.dtype == torch.float32
    assert x.shape == (12, 1000)
    assert x[0, 0] == 20
    assert x[0, 1] == pytest.approx(-0.05)
    assert y == torch.tensor(1.0)
    assert waveforms.x[0, 0, 0] == 100

    unlabeled = ECGDataset(waveforms, [{"ecg_id": "7"}], np.ones(12, dtype=np.float32))
    assert unlabeled[0][1] == -1


def test_build_cache_orders_leads_and_reuses_matching_cache(tmp_path):
    wfdb = pytest.importorskip("wfdb")
    raw = tmp_path / "raw"
    raw.mkdir()
    names = list(reversed(LEADS))
    signal = np.tile(np.linspace(-1, 1, 1000)[:, None], (1, 12)) * np.arange(1, 13)
    wfdb.wrsamp("rec", fs=100, units=["mV"] * 12, sig_name=names, p_signal=signal,
                fmt=["16"] * 12, adc_gain=[1000.0] * 12, baseline=[0] * 12, write_dir=str(raw))
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    for name, ecg_id in (("all_train_ssl.csv", 5), ("validation.csv", 3), ("test.csv", 9)):
        _write_manifest(manifests / name, [{"ecg_id": ecg_id, "filename_lr": "rec"}])
    cache = tmp_path / "cache"
    build_cache(raw, manifests, cache)

    assert np.load(cache / "ecg_ids.npy").tolist() == [3, 5, 9]
    signals = np.load(cache / "signals.npy")
    assert signals.shape == (3, 12, 1000)
    np.testing.assert_allclose(signals[0, 0], signal[:, 11], atol=1e-3)
    assert json.loads((cache / "complete.json").read_text())["lead_order"] == LEADS

    build_cache(raw, manifests, cache)
    _write_manifest(manifests / "test.csv", [{"ecg_id": 10, "filename_lr": "rec"}])
    with pytest.raises(ValueError, match="Cached records differ"):
        build_cache(raw, manifests, cache)
