"""Tests of the MLC-03 Part D comparison on synthetic validation sets."""

import numpy as np

from ecg_experiment import mlc03_effect as me
from ecg_experiment.mlc02_data import CONDITIONS


def synthetic(records_per_source: int = 300, seed: int = 0) -> tuple[np.ndarray, ...]:
    """Two validation sources, every condition with both classes, and a model that ranks them well."""
    rng = np.random.default_rng(seed)
    sources = np.repeat(np.array(["ptbxl", "ningbo"]), records_per_source)
    groups = np.array([f"g{i // 2}" for i in range(len(sources))])
    labels = (rng.random((len(sources), len(CONDITIONS))) < 0.3).astype(float)
    labels[rng.random(labels.shape) < 0.05] = np.nan
    good = np.clip(0.5 * np.nan_to_num(labels) + 0.5 * rng.random(labels.shape), 0.0, 1.0)
    return labels, groups, sources, good, rng.random(labels.shape)


def test_a_model_compared_with_itself_differs_by_zero():
    labels, groups, sources, good, _ = synthetic()
    result = me.compare(labels, groups, sources, good, good, ("v5", "v4"))
    assert result["selection"]["difference"]["value"] == 0.0
    assert result["selection"]["difference"]["ci"] == (0.0, 0.0)
    assert all(row["difference"] == 0.0 for row in result["conditions"])
    assert len(result["by_source"]) == 2 * len(CONDITIONS)


def test_a_better_model_has_a_positive_difference_with_an_interval_above_zero():
    labels, groups, sources, good, noise = synthetic()
    result = me.compare(labels, groups, sources, good, noise, ("v5", "v4"))
    low, high = result["selection"]["difference"]["ci"]
    assert 0.0 < low <= result["selection"]["difference"]["value"] <= high
    assert result["selection"]["v5"]["value"] > result["selection"]["v4"]["value"]
    assert all(row["ci_low"] > 0 for row in result["conditions"])


def test_condition_draws_average_only_entering_sources():
    labels, groups, sources, good, _ = synthetic()
    labels[sources == "ningbo", 0] = 0.0
    boot = me.source_bootstrap(labels, groups, sources, {"m": good})
    draws = me.condition_draws(boot, "m")
    assert np.allclose(draws[:, 0], boot["ptbxl"]["models"]["m"]["samples"]["auroc"][:, 0])
