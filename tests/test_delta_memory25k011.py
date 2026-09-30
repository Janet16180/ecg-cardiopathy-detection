"""CPU identity and patient-pairing checks for Experiment 011's 25k screen."""

import numpy as np
import pytest
import torch

from ecg_experiment.delta_memory25k011 import (
    EXPECTED_SELECTION_HASH,
    EXPOSURES,
    SUBSET_SIZE,
    frozen_stream,
    paired_patient_auc,
)
from ecg_experiment.files import read_csv, sha256_json
from scripts.experiments import run_delta_memory25k011 as runner


def test_frozen_stream_replays_exact_019_selection_and_full_order() -> None:
    """Require the canonical 25k index identity and fair repeated exposures."""
    rows = read_csv("data/processed/sampled_100k_plus_labels_v1/train_manifest.csv")
    selected, order, counts = frozen_stream([row["source"] for row in rows])
    assert len(selected) == SUBSET_SIZE
    assert sha256_json(selected.tolist()) == EXPECTED_SELECTION_HASH
    assert len(order) == EXPOSURES
    assert np.array_equal(np.sort(order[:SUBSET_SIZE]), np.sort(selected))
    assert set(order.tolist()) == set(selected.tolist())
    assert sum(counts.values()) == SUBSET_SIZE


def test_paired_patient_auc_keeps_records_together_and_primary_direction() -> None:
    """Check the CKDA minus KDA sign on a repeated-patient toy example."""
    y = np.asarray([0, 0, 1, 1, 0, 1])
    patients = np.asarray(["a", "a", "b", "b", "c", "d"])
    predictions = {
        "gru": np.asarray([0.2, 0.3, 0.6, 0.7, 0.4, 0.8]),
        "kda": np.asarray([0.2, 0.3, 0.6, 0.7, 0.4, 0.8]),
        "ckda": np.asarray([0.1, 0.2, 0.8, 0.9, 0.3, 0.7]),
    }
    result = paired_patient_auc(y, patients, predictions, seed=7, draws=40)
    assert result["ckda_minus_kda"]["auroc_difference"] >= 0
    assert result["kda_minus_gru"]["auroc_difference"] == 0
    assert result["ckda_minus_kda"]["valid_draws"] > 0
    assert (result["ckda_minus_kda"]["valid_draws"]
            + result["ckda_minus_kda"]["invalid_draws"] == 40)


def test_paired_patient_auc_rejects_missing_arm() -> None:
    """Never emit a nominal comparison if one architecture is absent."""
    with pytest.raises(ValueError, match="Expected GRU"):
        paired_patient_auc(np.asarray([0, 1]), np.asarray(["a", "b"]),
                           {"gru": np.asarray([0.2, 0.8])})


def test_selected_gpu_staging_preserves_exact_values_order_and_resume_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage only selected rows and recover the exact next exposure batch."""

    class TinyDataset:
        """Read-only stand-in for the verified float32 waveform cache."""

        def __init__(self) -> None:
            self.signals = np.arange(5 * 2 * 3, dtype=np.float32).reshape(5, 2, 3)
            self.mean = np.asarray([[2], [3]], dtype=np.float32)
            self.std = np.asarray([[4], [5]], dtype=np.float32)

        def __len__(self) -> int:
            """Return all source rows, including rows not selected."""
            return len(self.signals)

    monkeypatch.setattr(runner, "EXPOSURES", 7)
    monkeypatch.setattr(runner, "SUBSET_SIZE", 3)
    monkeypatch.setattr(runner, "BATCH_SIZE", 2)
    dataset = TinyDataset()
    source_before = dataset.signals.copy()
    order = np.asarray([2, 0, 2, 4, 0, 4, 2])
    staged, mapped = runner._stage_selected(dataset, order, device="cpu", chunk_size=2)
    expected = source_before[[0, 2, 4]].copy()
    expected -= dataset.mean
    expected /= dataset.std
    torch.testing.assert_close(staged, torch.from_numpy(expected), atol=0, rtol=0)
    np.testing.assert_array_equal(dataset.signals, source_before)
    np.testing.assert_array_equal(mapped, np.asarray([1, 0, 1, 2, 0, 2, 1]))
    batches = list(runner._staged_batches(staged, mapped))
    suffix = list(runner._staged_batches(staged, mapped[2 * runner.BATCH_SIZE:]))
    torch.testing.assert_close(torch.cat(batches[2:]), torch.cat(suffix), atol=0, rtol=0)
    torch.testing.assert_close(torch.cat(batches),
                               torch.from_numpy(expected)[torch.as_tensor(mapped)],
                               atol=0, rtol=0)
