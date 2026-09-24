"""Resumable CPC fine-tuning on a synthetic pool."""

import csv
import json
from argparse import Namespace

import numpy as np
import pytest
from conftest import write_synthetic_cpc_pool

from ecg_experiment import cpc_pool


def _write_budgets(pool, directory):
    train = [row for row in pool.rows if row["split"] == "train"]
    for budget, labeled in (("0.1", 16), ("1", 32)):
        budget_dir = directory / f"seed42_fraction{budget}"
        budget_dir.mkdir(parents=True)
        groups = {"all_train_ssl": train, "labeled_train": train[:labeled],
                  "validation": [row for row in pool.rows if row["split"] == "validation"],
                  "test": [row for row in pool.rows if row["split"] == "test"]}
        for name, rows in groups.items():
            with (budget_dir / f"{name}.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["ecg_id", "patient_id", "target"])
                writer.writeheader()
                writer.writerows({key: row[key] for key in ("ecg_id", "patient_id", "target")}
                                 for row in rows)
    return directory


@pytest.fixture
def prepared(tmp_path):
    directory = write_synthetic_cpc_pool(tmp_path / "pool", records=120)
    signals = np.load(directory / "signals.npy")
    signals[1::2, :, ::25] += 3
    np.save(directory / "signals.npy", signals)
    pool = cpc_pool.Pool(directory)
    manifests = _write_budgets(pool, tmp_path / "manifests")
    hashes = cpc_pool.make_source_hashes(pool, manifests)
    mean, std = pool.normalization(tmp_path / "run", hashes)
    args = Namespace(output_dir=tmp_path / "run", manifest_dir=manifests, device="cpu", epochs=2,
                     patience=5, batch_size=16, ssl_epochs=1, bootstrap=5)
    return args, pool, mean, std, hashes


def test_manifest_rows_reject_split_mismatch(prepared, tmp_path):
    args, pool, *_ = prepared
    rows, hashes = cpc_pool.manifest_rows(pool, args.manifest_dir, "0.1")
    assert len(rows["labeled_train"]) == 16
    assert set(hashes) == set(cpc_pool.LABEL_MANIFESTS)
    test_path = args.manifest_dir / "seed42_fraction0.1/test.csv"
    validation_path = args.manifest_dir / "seed42_fraction0.1/validation.csv"
    test_path.write_text(validation_path.read_text())
    with pytest.raises(ValueError, match="PTB manifest/cache mismatch"):
        cpc_pool.manifest_rows(pool, args.manifest_dir, "0.1")


def test_fine_tune_completes_and_verifies_artifacts(prepared, capsys):
    args, pool, mean, std, hashes = prepared
    cpc_pool.fine_tune(args, pool, mean, std, hashes, "scratch", "1")
    directory = args.output_dir / "scratch_fraction1_seed42"
    completion = json.loads((directory / "completion.json").read_text())
    assert set(completion["artifacts"]) == set(cpc_pool.COMPLETED_ARTIFACTS)
    history = json.loads((directory / "history.json").read_text())
    assert [record["epoch"] for record in history] == [1, 2]
    assert history[-1]["record_exposures"] == 32
    progress = [json.loads(line) for line in capsys.readouterr().out.splitlines()
                if line.startswith('{"stage"')]
    assert [line["epoch"] for line in progress] == [1, 2]

    cpc_pool.fine_tune(args, pool, mean, std, hashes, "scratch", "1")
    (directory / "metrics.json").write_text("{}")
    with pytest.raises(ValueError, match="Completed artifact changed"):
        cpc_pool.fine_tune(args, pool, mean, std, hashes, "scratch", "1")
    with pytest.raises(ValueError, match="Completed training fingerprint mismatch"):
        cpc_pool.fine_tune(Namespace(**(vars(args) | {"epochs": 3})), pool, mean, std, hashes,
                           "scratch", "1")
