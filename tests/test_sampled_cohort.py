"""Checks for exact, repeatable multi-source ECG sampling."""

import json

import numpy as np
import pandas as pd
import pytest

from ecg_experiment.files import sha256_file
from ecg_experiment.public_sources import signal_sha256
from ecg_experiment.sampled_cohort import sample_sources, source_quotas
from ecg_experiment.sampled_training_dataset import SampledTrainingECGDataset


def test_source_quotas_are_exact_and_capacity_bounded() -> None:
    capacities = {"mimic": 100, "ptbxl": 4, "chapman_shaoxing": 10}
    quotas = source_quotas(capacities, total=30, mimic_fraction=0.5)
    assert quotas == {"mimic": 16, "ptbxl": 4, "chapman_shaoxing": 10}
    assert sum(quotas.values()) == 30


def test_seeded_sampling_is_stable_and_changes_with_seed() -> None:
    rows = pd.DataFrame({
        "record_id": [f"mimic:{i}" for i in range(60)] + [f"chapman:{i}" for i in range(40)],
        "source": ["mimic"] * 60 + ["chapman_shaoxing"] * 40,
    })
    first, quotas = sample_sources(rows, total=50, seed=71, mimic_fraction=0.7)
    repeat, _ = sample_sources(rows.sample(frac=1, random_state=4), 50, 71, 0.7)
    different, _ = sample_sources(rows, 50, 72, 0.7)
    assert first["record_id"].tolist() == repeat["record_id"].tolist()
    assert first["record_id"].tolist() != different["record_id"].tolist()
    assert first["record_id"].nunique() == 50
    assert first["source"].value_counts().to_dict() == quotas


def test_sampling_refuses_insufficient_or_duplicate_candidates() -> None:
    with pytest.raises(ValueError, match="Insufficient"):
        source_quotas({"mimic": 2, "chapman_shaoxing": 2}, total=5)
    duplicate = pd.DataFrame({"record_id": ["same", "same"], "source": ["mimic", "mimic"]})
    with pytest.raises(ValueError, match="record IDs"):
        sample_sources(duplicate, total=1, seed=0)


def test_loader_masks_ssl_labels_and_exposes_only_supervised_target(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    signal = np.tile(np.linspace(-1, 1, 5000, dtype=np.float32), (12, 1))
    shard = np.stack([signal])
    np.save(source / "shard.npy", shard)

    cohort = tmp_path / "cohort"
    cohort.mkdir()
    rows = pd.DataFrame([{
        "record_id": "ptbxl:1", "patient_id": "ptbxl:patient1", "source": "ptbxl",
        "split": "train", "label_scope": "ptbxl_proxy_available_separately",
        "backend": "shard", "path": "source/shard.npy", "index": "0",
        "signal_sha256": signal_sha256(signal),
    }])
    labels = pd.DataFrame([{"record_id": "ptbxl:1", "patient_id": "ptbxl:patient1", "target": 1}])
    rows.to_csv(cohort / "train_manifest.csv", index=False)
    for budget in ("1", "0.1"):
        labels.to_csv(cohort / f"labels_fraction{budget}.csv", index=False)
    metadata = {
        "complete": True,
        "schema_version": 1,
        "shape_per_record": [12, 5000],
        "record_count": 1,
        "source_shards": {
            "source/shard.npy": {"sha256": sha256_file(source / "shard.npy"), "shape": [1, 12, 5000]}
        },
        "table_sha256": {
            name: sha256_file(cohort / name)
            for name in ("train_manifest.csv", "labels_fraction1.csv", "labels_fraction0.1.csv")
        },
    }
    (cohort / "metadata.json").write_text(json.dumps(metadata))

    ssl = SampledTrainingECGDataset(cohort)
    supervised = SampledTrainingECGDataset(cohort, purpose="supervised")
    ssl.root = supervised.root = tmp_path
    assert ssl[0]["target"] == -1
    assert ssl[0]["target_available"] is False
    assert supervised[0]["target"] == 1
    assert supervised[0]["target_available"] is True
