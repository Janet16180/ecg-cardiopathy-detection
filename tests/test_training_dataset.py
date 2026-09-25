"""Canonical training union: label masking, tamper detection and build safety."""

import fcntl
import json
from pathlib import Path

import numpy as np
import pytest
from torch.utils.data import DataLoader

import scripts.data.build_training_dataset as builder
from ecg_experiment.files import sha256_file, write_csv_atomic
from ecg_experiment.public_sources import signal_sha256
from ecg_experiment.training_dataset import TrainingECGDataset, verify_dataset
from scripts.data.build_training_dataset import FIELDS, build, decode


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    signal = np.tile(np.linspace(-1, 1, 5000, dtype=np.float32), (12, 1))
    signals = np.stack([signal, signal * 2])
    np.save(tmp_path / "shard_00000.npy", signals)
    rows = []
    for index, source in enumerate(("ptbxl", "georgia")):
        rows.append({"record_id": f"{source}:1", "source_record_id": "1", "source": source,
                     "patient_id": "ptbxl:99" if index == 0 else "",
                     "patient_identity_known": "true" if index == 0 else "false", "split": "train",
                     "label_scope": "ptbxl_proxy_available_separately" if index == 0 else "ssl_only",
                     "shard": "shard_00000.npy", "shard_index": index,
                     "signal_sha256": signal_sha256(signals[index])})
    write_csv_atomic(tmp_path / "train_manifest.csv", rows, FIELDS)
    for budget in ("1", "0.1"):
        write_csv_atomic(tmp_path / f"labels_fraction{budget}.csv",
                         [{"record_id": "ptbxl:1", "patient_id": "ptbxl:99", "target": 1}],
                         ("record_id", "patient_id", "target"))
    metadata = {"complete": True, "schema_version": 1, "record_count": 2, "shape_per_record": [12, 5000],
                "sampling_rate_hz": 500, "units": "mV",
                "shards": {"shard_00000.npy": {"sha256": sha256_file(tmp_path / "shard_00000.npy"),
                                               "shape": [2, 12, 5000]}},
                "table_sha256": {p.name: sha256_file(p) for p in tmp_path.glob("*.csv")}}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    return tmp_path


def _rehash_table(dataset: Path, path: Path) -> None:
    metadata = json.loads((dataset / "metadata.json").read_text())
    metadata["table_sha256"][path.name] = sha256_file(path)
    (dataset / "metadata.json").write_text(json.dumps(metadata))


def test_ssl_masks_labels_and_supervised_loader_trains(dataset: Path) -> None:
    ssl = TrainingECGDataset(dataset)
    batch = next(iter(DataLoader(ssl, batch_size=2)))
    assert batch["signal"].shape == (2, 12, 5000)
    assert batch["target"].tolist() == [-1, -1]
    assert not batch["target_available"].any()
    supervised = TrainingECGDataset(dataset, purpose="supervised", label_budget="0.1")
    assert len(supervised) == 1
    assert supervised[0]["target"] == 1
    assert verify_dataset(dataset)["verified_records"] == 2


def test_write_csv_ignores_decoding_keys(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    rows = [{"record_id": "a", "source": "b", "raw_dir": Path("/x")}]
    write_csv_atomic(path, rows, ("record_id", "source"), ignore_extra=True)
    assert path.read_bytes() == b"record_id,source\r\na,b\r\n"


def test_shard_tampering_fails(dataset: Path) -> None:
    array = np.load(dataset / "shard_00000.npy")
    array[0, 0, 0] += 0.1
    np.save(dataset / "shard_00000.npy", array)
    with pytest.raises(ValueError, match="Shard hash"):
        TrainingECGDataset(dataset)[0]


def test_unknown_source_cannot_receive_endpoint_label(dataset: Path) -> None:
    path = dataset / "labels_fraction1.csv"
    write_csv_atomic(path, [{"record_id": "georgia:1", "patient_id": "", "target": 1}],
                     ("record_id", "patient_id", "target"))
    _rehash_table(dataset, path)
    with pytest.raises(ValueError, match="training PTB"):
        TrainingECGDataset(dataset, purpose="supervised")


def test_cannot_overwrite_published_dataset(dataset: Path) -> None:
    with pytest.raises(FileExistsError):
        build(dataset)


def test_split_semantic_tampering_fails_after_rehash(dataset: Path) -> None:
    path = dataset / "train_manifest.csv"
    path.write_text(path.read_text().replace(",train,", ",test,"))
    _rehash_table(dataset, path)
    with pytest.raises(ValueError, match="split or label leakage"):
        verify_dataset(dataset)


def test_single_builder_lock_prevents_duplicate_work(tmp_path: Path) -> None:
    destination = tmp_path / "new_dataset"
    with (tmp_path / ".new_dataset.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already holds"):
            build(destination)
    assert not destination.exists()


def test_swapped_ptb_bytes_fail_before_decoding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    (tmp_path / "example.hea").write_text("other valid record bytes")
    row = {"origin_index": "", "origin_path": "example", "record_id": "ptbxl:1",
           "official_sha256": {".hea": "0" * 64}}
    with pytest.raises(ValueError, match="Official PTB checksum mismatch: ptbxl:1.hea"):
        decode(row)


def test_semantically_bad_waveform_rejected_even_with_new_sha256_file(dataset: Path) -> None:
    path = dataset / "shard_00000.npy"
    signals = np.load(path)
    signals[0, 10] = 0
    np.save(path, signals)
    metadata = json.loads((dataset / "metadata.json").read_text())
    metadata["shards"][path.name]["sha256"] = sha256_file(path)
    (dataset / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Full constant lead"):
        TrainingECGDataset(dataset)[0]


def test_heldout_references_reject_train_patient_leakage(tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "PTB_RAW", tmp_path / "raw")
    monkeypatch.setattr(builder, "PTB_PROCESSED", tmp_path)
    fields = ("ecg_id", "patient_id", "target", "filename_hr")
    write_csv_atomic(tmp_path / "validation.csv",
                     [{"ecg_id": str(i), "patient_id": f"v{i}", "target": str(i % 2), "filename_hr": f"r{i}"}
                      for i in range(20)], fields)
    write_csv_atomic(tmp_path / "test.csv",
                     [{"ecg_id": "9", "patient_id": "t", "target": "1", "filename_hr": "r9"}], fields)
    train = [{"source": "ptbxl", "patient_id": "ptbxl:p"}]
    references = builder.heldout_references(train)
    assert {r["split"] for r in references} == {"development", "calibration", "test"}
    assert [r for r in references if r["split"] == "test"][0]["raw_path"] == "raw/r9"
    with pytest.raises(ValueError, match="PTB patient split leakage"):
        builder.heldout_references([{"source": "ptbxl", "patient_id": "ptbxl:t"}])
