"""Focused cohort, schedule, and audit checks for clean Experiment 017."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ecg_experiment.morphology_clean_audit import (
    _extreme_plateau,
    paired_bootstrap,
    replace_response_channel,
    subgroup_contrasts,
)
from ecg_experiment.morphology_clean_inputs import FULL_CLEAN_LABELS, clean_exposed_labels
from ecg_experiment.pilot import Partitions, fixed_batches


def _partitions() -> Partitions:
    """Build a small fixed-label selection without any held-out training use."""
    full = [{"ecg_id": str(index), "patient_id": str(index // 2), "target": str(index % 2)}
            for index in range(FULL_CLEAN_LABELS)]
    limited = [full[index] for index in range(1518)]
    return Partitions(full, limited, [], [], [])


@pytest.mark.parametrize(("budget", "expected"), [("1", 15359), ("0.1", 1518)])
def test_clean_schedule_covers_all_records_once(budget: str, expected: int) -> None:
    """Every seed and arm sees identical 120 batches with hidden labels masked."""
    partitions = _partitions()
    exposed, targets = clean_exposed_labels(partitions, budget)
    assert int(exposed.sum()) == expected
    assert not np.any(targets[~exposed])
    for seed in (42, 43):
        batches = fixed_batches(FULL_CLEAN_LABELS, exposed, 0, seed)
        assert len(batches) == 120
        assert len({row for batch in batches for row in batch}) == FULL_CLEAN_LABELS
        assert all(exposed[batch].any() for batch in batches)
        assert batches == fixed_batches(FULL_CLEAN_LABELS, exposed, 0, seed)
    assert fixed_batches(FULL_CLEAN_LABELS, exposed, 0, 42) != fixed_batches(
        FULL_CLEAN_LABELS, exposed, 0, 43)


def test_response_ablation_changes_only_one_channel() -> None:
    """Ablation preserves all other native response channels and original storage."""
    original = torch.arange(2 * 3 * 32, dtype=torch.float32).reshape(2, 3, 32)
    modified = replace_response_channel(original, 7, -1.5)
    assert torch.equal(original[:, :, 7], torch.tensor([[7., 39., 71.], [103., 135., 167.]]))
    assert torch.all(modified[:, :, 7] == -1.5)
    assert torch.equal(modified[:, :, :7], original[:, :, :7])
    assert torch.equal(modified[:, :, 8:], original[:, :, 8:])
    with pytest.raises(ValueError, match="Invalid response"):
        replace_response_channel(original, 32, 0)


def test_patient_bootstrap_is_paired_and_reproducible() -> None:
    """Whole-patient resampling uses one draw across every compared score vector."""
    labels = np.array([0, 1, 0, 1, 0, 1])
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    scores = {"template": np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7]),
              "conv": np.array([0.2, 0.8, 0.3, 0.7, 0.4, 0.6]),
              "none": np.array([0.2, 0.8, 0.3, 0.7, 0.4, 0.6])}
    result = paired_bootstrap(labels, groups, scores)
    assert result == paired_bootstrap(labels, groups, scores)
    assert result["effective_count"] == 2000
    assert result["contrasts"]["template_minus_conv"] == result["contrasts"]["template_minus_none"]


def test_subgroup_missing_class_is_undefined_and_flags_are_descriptive() -> None:
    """Artifact flags do not alter the complete-cohort primary AUROC."""
    labels = np.array([0, 1, 0, 1])
    scores = {"template": np.array([0.2, 0.8, 0.3, 0.7]),
              "conv": np.array([0.3, 0.7, 0.4, 0.6]),
              "none": np.array([0.4, 0.6, 0.3, 0.7])}
    result = subgroup_contrasts(labels, scores, np.array([False, True, False, True]))
    assert result["all"]["records"] == 4
    assert result["unflagged"]["template_minus_conv"] is None
    assert result["flagged"]["template_minus_conv"] is None


def test_plateau_requires_five_identical_extreme_samples() -> None:
    """Only repeated exact extreme values trigger the descriptive plateau flag."""
    signal = np.zeros((12, 10), dtype=np.float32)
    signal[0, 2:7] = 11
    assert _extreme_plateau(signal, 10)
    signal[0, 6] = 0
    assert not _extreme_plateau(signal, 10)
