"""Partitions, label schedules, checkpoints and screens shared by the 015/017 pilots."""

import csv
import json
import signal
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from conftest import write_synthetic_cpc_pool
from sklearn.model_selection import StratifiedGroupKFold

from ecg_experiment import pilot, receipts
from ecg_experiment.cpc_pool import Pool
from ecg_experiment.evaluation import partition_validation, select_threshold
from ecg_experiment.files import read_csv


@pytest.fixture(autouse=True)
def clear_stop():
    pilot._STOP_REQUESTED.clear()
    yield
    pilot._STOP_REQUESTED.clear()


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("ecg_id", "patient_id", "target"))
        writer.writeheader()
        writer.writerows({key: row[key] for key in ("ecg_id", "patient_id", "target")} for row in rows)


def synthetic_manifests(tmp_path, monkeypatch):
    pool = Pool(write_synthetic_cpc_pool(tmp_path / "pool", records=200))
    train = [row for row in pool.rows if row["split"] == "train"]
    validation = [row for row in pool.rows if row["split"] == "validation"]
    test = [row for row in pool.rows if row["split"] == "test"]
    limited = train[::4]
    for budget, labeled in (("1", train), ("0.1", limited)):
        write_manifest(tmp_path / f"manifests/seed42_fraction{budget}/labeled_train.csv", labeled)
        write_manifest(tmp_path / f"manifests/seed42_fraction{budget}/validation.csv", validation)
        write_manifest(tmp_path / f"manifests/seed42_fraction{budget}/test.csv", test)
    development, calibration = partition_validation(
        [{key: row[key] for key in ("ecg_id", "patient_id", "target")} for row in validation])
    monkeypatch.setattr(pilot, "PARTITION_COUNTS",
                        (len(train), len(limited), len(development), len(calibration), len(test)))
    return pool, tmp_path / "manifests"


def test_load_partitions_accepts_frozen_layout_and_rejects_changed_budgets(tmp_path, monkeypatch):
    pool, manifests = synthetic_manifests(tmp_path, monkeypatch)
    partitions = pilot.load_partitions(pool, manifests)
    assert len(partitions.full) == 160
    assert len(partitions.limited) == 40
    assert {row["ecg_id"] for row in partitions.limited} <= {row["ecg_id"] for row in partitions.full}
    changed = [{**row, "target": "1" if row["target"] == "0" else "0"} for row in partitions.limited]
    write_manifest(manifests / "seed42_fraction0.1/labeled_train.csv", changed)
    with pytest.raises(ValueError, match="Limited budget differs"):
        pilot.load_partitions(pool, manifests)


def test_load_partitions_rejects_cache_split_mismatch(tmp_path, monkeypatch):
    pool, manifests = synthetic_manifests(tmp_path, monkeypatch)
    test = read_csv(manifests / "seed42_fraction1/test.csv")
    swapped = [*read_csv(manifests / "seed42_fraction1/labeled_train.csv")[:-1], test[0]]
    write_manifest(manifests / "seed42_fraction1/labeled_train.csv", swapped)
    write_manifest(manifests / "seed42_fraction0.1/labeled_train.csv", swapped[::4])
    with pytest.raises(ValueError, match="Manifest/cache mismatch"):
        pilot.load_partitions(pool, manifests)


def test_exposed_labels_hide_targets_outside_the_budget():
    full = [{"ecg_id": str(i), "target": str(i % 2)} for i in range(pilot.FULL_LABELS)]
    limited = full[:pilot.LIMITED_LABELS]
    exposed, targets = pilot.exposed_labels(full, limited, "0.1")
    assert exposed.sum() == pilot.LIMITED_LABELS
    assert not exposed[pilot.LIMITED_LABELS:].any()
    assert targets.dtype == np.float32
    assert not targets[~exposed].any()
    np.testing.assert_array_equal(targets[:4], [0, 1, 0, 1])
    assert pilot.exposed_labels(full, limited, "1")[0].all()
    with pytest.raises(ValueError, match="count changed"):
        pilot.exposed_labels(full, limited[:-1], "0.1")


def test_fixed_batches_requires_one_label_per_batch():
    exposed = np.zeros(300, dtype=bool)
    exposed[:2] = True
    with pytest.raises(ValueError, match="Too few exposed labels"):
        pilot.fixed_batches(300, exposed, 0, 42)
    exposed[:3] = True
    batches = pilot.fixed_batches(300, exposed, 0, 42, batch_size=128)
    assert [len(batch) for batch in batches] == [128, 128, 44]
    assert all(exposed[batch].sum() == 1 for batch in batches)


def test_development_batches_cover_rows_after_training():
    cache = SimpleNamespace(signals=np.arange(300 * 12, dtype=np.float32).reshape(300, 12, 1))
    mean, std = np.zeros((12, 1), np.float32), np.full((12, 1), 2, np.float32)
    batches = list(pilot.development_batches(cache, 10, 290, mean, std, "cpu"))
    assert [len(batch) for batch in batches] == [128, 128, 34]
    torch.testing.assert_close(torch.cat(batches), torch.from_numpy(cache.signals[10:] / 2))


def test_fold_operating_point_pools_crossfit_confusions():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, 200)
    groups = np.asarray([f"p{i // 2}" for i in range(200)])
    probabilities = pilot.clipped_sigmoid(rng.standard_normal(200) + labels)
    result = pilot.fold_operating_point(labels, groups, probabilities, 42)
    tp = fn = 0
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for train, held in splitter.split(probabilities, labels, groups):
        predicted = probabilities[held] >= select_threshold(labels[train], probabilities[train], 0.95)
        tp += int(np.sum(predicted & (labels[held] == 1)))
        fn += int(np.sum(~predicted & (labels[held] == 1)))
    assert result["fold_confusion"]["tp"] == tp
    assert result["fold_confusion"]["fn"] == fn
    assert sum(result["fold_confusion"].values()) == 200
    assert result["fold_sensitivity"] == tp / (tp + fn)
    assert pilot.clipped_sigmoid(np.array([1e6]))[0] == pilot.clipped_sigmoid(np.array([80.0]))[0]


def test_interrupted_by_signal_or_deadline():
    assert not pilot.interrupted(None)
    assert not pilot.interrupted(time.monotonic() + 60)
    assert pilot.interrupted(time.monotonic() - 1)
    previous = signal.getsignal(signal.SIGTERM)
    try:
        pilot.install_stop_handler()
        signal.raise_signal(signal.SIGTERM)
        assert pilot.interrupted(None)
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_completion_roundtrip_detects_changed_artifacts(tmp_path):
    artifacts = {"history.json": "history_sha256", "best.pt": "best_sha256"}
    (tmp_path / "history.json").write_text(json.dumps([{"epoch": 1}]))
    (tmp_path / "best.pt").write_bytes(b"model")
    receipts.write_completion(tmp_path, "fp", {"auc": 0.7, "epoch": 1}, artifacts)
    saved = json.loads((tmp_path / "completion.json").read_text())
    assert list(saved) == ["fingerprint", "best_epoch", "best_development_auroc", "history_sha256",
                           "best_sha256"]
    assert receipts.check_completion(tmp_path, "fp", artifacts) == [{"epoch": 1}]
    with pytest.raises(ValueError, match="fingerprint"):
        receipts.check_completion(tmp_path, "other", artifacts)
    (tmp_path / "best.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="best.pt"):
        receipts.check_completion(tmp_path, "fp", artifacts)


def test_existing_config_and_history_guards(tmp_path):
    receipts.check_existing_config(tmp_path / "config.json", "fp")
    (tmp_path / "config.json").write_text(json.dumps({"fingerprint": "old"}))
    with pytest.raises(ValueError, match="different inputs"):
        receipts.check_existing_config(tmp_path / "config.json", "fp")
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    assert pilot.load_state(tmp_path, "fp", model, optimizer, {"auc": -1.0}).epoch == 0
    (tmp_path / "history.json").write_text("[]")
    with pytest.raises(ValueError, match="without checkpoint"):
        pilot.load_state(tmp_path, "fp", model, optimizer, {"auc": -1.0})


def test_check_roundtrip_detects_changed_weights(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    pilot.save_state(tmp_path, "fp", model, optimizer, pilot.Progress(1, 0, [], {}), 1.0)
    original = torch.load(tmp_path / "resume.pt", weights_only=False)
    probe = torch.nn.Linear(2, 1)
    probe_optimizer = torch.optim.AdamW(probe.parameters())
    progress = pilot.load_state(tmp_path, "fp", probe, probe_optimizer, {})
    assert pilot.check_roundtrip(original, progress, probe, probe_optimizer) == {
        "epoch": 1, "batch": 0, "model_tensors": 2, "optimizer_slots": 2}
    with torch.no_grad():
        probe.bias.add_(1)
    with pytest.raises(ValueError, match="model roundtrip"):
        pilot.check_roundtrip(original, progress, probe, probe_optimizer)
    with pytest.raises(ValueError, match="one epoch"):
        pilot.check_roundtrip(original, pilot.Progress(1, 3, [], {}), probe, probe_optimizer)


def test_profile_arms_and_deadline_between_arms(tmp_path):
    seen = []
    durations, roundtrips = pilot.profile_arms("pilot_test_", ("a", "b"),
                                               lambda arm, directory: seen.append(directory) or {"arm": arm})
    assert list(durations) == ["a", "b"]
    assert roundtrips == {"a": {"arm": "a"}, "b": {"arm": "b"}}
    assert all(not directory.parent.exists() for directory in seen)
    runs = []
    pilot.run_pilot(("a", "b"), time.monotonic() + 60, lambda budget, arm: runs.append((budget, arm)))
    assert runs == [("1", "a"), ("1", "b"), ("0.1", "a"), ("0.1", "b")]
    with pytest.raises(SystemExit) as stopped:
        pilot.run_pilot(("a",), time.monotonic() - 1, lambda budget, arm: None)
    assert stopped.value.code == pilot.INTERRUPTED_EXIT


def test_require_receipt(tmp_path):
    path = tmp_path / "receipt.json"
    with pytest.raises(ValueError, match="missing"):
        receipts.require_receipt(path, "fp", "missing or changed")
    path.write_text(json.dumps({"fingerprint": "fp", "value": 1}))
    assert receipts.require_receipt(path, "fp", "missing or changed")["value"] == 1
    with pytest.raises(ValueError, match="changed"):
        receipts.require_receipt(path, "other", "missing or changed")
