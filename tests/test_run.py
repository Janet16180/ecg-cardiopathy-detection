"""Tiny CPU runs of the compact PTB-XL runner on a synthetic 100 Hz cache."""

import csv
import json
import sys
from argparse import Namespace

import numpy as np
import pytest
import torch

from ecg_experiment import run
from ecg_experiment.data import Waveforms
from ecg_experiment.files import read_csv
from ecg_experiment.training import warmup_cosine_lr

RECORDS = 96


@pytest.fixture
def experiment(tmp_path):
    rng = np.random.default_rng(0)
    targets = np.arange(RECORDS) % 2
    signals = rng.standard_normal((RECORDS, 12, 1000)).astype(np.float32)
    signals[targets == 1, :, ::50] += 2
    cache = tmp_path / "cache"
    cache.mkdir()
    np.save(cache / "signals.npy", signals)
    np.save(cache / "ecg_ids.npy", np.arange(1, RECORDS + 1, dtype=np.int64))
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    splits = {"all_train_ssl.csv": range(0, 48), "labeled_train.csv": range(0, 48, 2),
              "validation.csv": range(48, 80), "test.csv": range(80, RECORDS)}
    for name, indices in splits.items():
        with (manifests / name).open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["ecg_id", "patient_id", "target"])
            writer.writerows([index + 1, f"p{index // 2}", targets[index]] for index in indices)
    return tmp_path


def _args(root, **overrides):
    settings = {"model": "cnn", "seed": 42, "device": "cpu", "manifest_dir": root / "manifests",
                "output_dir": root / "out", "batch_size": 16, "ssl_batch_size": 16, "epochs": 2,
                "ssl_epochs": 2, "patience": 5, "bootstrap": 5, "threads": 1}
    return Namespace(**(settings | overrides))


def _inputs(root):
    waveforms = Waveforms(root / "cache")
    train = read_csv(root / "manifests/all_train_ssl.csv")
    return waveforms, train, waveforms.training_scale(train)


def test_ssl_learning_rate_warms_up_then_decays():
    rates = [warmup_cosine_lr(run.SSL_LR, epoch, 10) for epoch in range(10)]
    assert rates[0] == pytest.approx(run.SSL_LR / 2)
    assert rates[1] == pytest.approx(run.SSL_LR * (0.1 + 0.9 * (1 + np.cos(np.pi / 10)) / 2))
    assert all(later < earlier for earlier, later in zip(rates[1:], rates[2:], strict=False))
    assert rates[-1] > 0.1 * run.SSL_LR


def test_supervised_run_writes_reproducible_results(experiment, capsys):
    waveforms, _, scale = _inputs(experiment)
    rows = {name: read_csv(experiment / f"manifests/{name}.csv")
            for name in ("labeled_train", "validation", "test")}
    results = []
    for output in ("first", "second"):
        args = _args(experiment, output_dir=experiment / output)
        results.append(run.supervised(args, waveforms, rows["labeled_train"], rows["validation"],
                                      rows["test"], scale))
    assert results[0] == results[1]
    directory = experiment / "first/cnn_supervised_seed42"
    assert {path.name for path in directory.iterdir()} == {
        "history.json", "model.pt", "metrics.json", "test_predictions.csv",
        "calibration_predictions.npz", "config.json"}
    first = torch.load(directory / "model.pt", weights_only=True)
    second = torch.load(experiment / "second/cnn_supervised_seed42/model.pt", weights_only=True)
    assert all(torch.equal(first["model"][key], second["model"][key]) for key in first["model"])
    config = json.loads((directory / "config.json").read_text())
    assert config["labeled_training_records"] == 24
    assert config["development_records"] + config["calibration_records"] == 32

    progress = [json.loads(line) for line in capsys.readouterr().out.splitlines() if '"stage"' in line]
    assert [(line["stage"], line["epoch"]) for line in progress[:2]] == [("cnn_supervised", 1),
                                                                         ("cnn_supervised", 2)]
    with pytest.raises(FileExistsError, match="Completed results already exist"):
        run.supervised(_args(experiment, output_dir=experiment / "first"), waveforms,
                       rows["labeled_train"], rows["validation"], rows["test"], scale)


def test_pretrain_reuses_matching_checkpoint_and_rejects_changed_settings(experiment):
    waveforms, train, scale = _inputs(experiment)
    args = _args(experiment, model="mae", ssl_epochs=1)
    checkpoint = run.pretrain(args, waveforms, train, scale)
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["training_ecg_ids"] == [row["ecg_id"] for row in train]
    assert len(json.loads((checkpoint.parent / "history.json").read_text())) == 1

    assert run.pretrain(args, waveforms, train, scale) == checkpoint
    with pytest.raises(ValueError, match="Existing SSL checkpoint differs"):
        run.pretrain(_args(experiment, model="mae", ssl_epochs=2), waveforms, train, scale)
    with pytest.raises(ValueError, match="SSL and supervised normalization do not match"):
        run._load_ssl_encoder(run.PatchTransformer(), checkpoint, args, scale * 2)


def test_ssl_stage_requires_a_pretraining_model(experiment, monkeypatch):
    # main() sets process-wide thread counts; keep them unchanged for other tests.
    monkeypatch.setattr(torch, "set_num_threads", lambda threads: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda threads: None)
    monkeypatch.setattr(sys, "argv", ["run", "--stage", "ssl", "--model", "cnn", "--device", "cpu",
                                      "--cache-dir", str(experiment / "cache"),
                                      "--manifest-dir", str(experiment / "manifests")])
    with pytest.raises(ValueError, match="SSL requires --model mae or jepa"):
        run.main()
