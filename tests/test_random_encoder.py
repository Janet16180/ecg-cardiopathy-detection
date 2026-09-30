import numpy as np
import pytest
from torch import nn

from ecg_experiment.intervals import paired_auroc_difference
from ecg_experiment.random_encoder import (
    auroc_draws,
    decision,
    interval_side,
    linear_contrast,
    seed_mean_weights,
    state_digest,
)


def synthetic(rows: int = 300, seed: int = 1) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    units = np.repeat(np.arange(rows // 2), 2).astype(str)
    y = rng.integers(0, 2, rows)
    scores = {name: y * shift + rng.normal(size=rows) for name, shift in (("a", 1.5), ("b", 0.5), ("c", 0.2))}
    return units, y, scores


def test_pair_contrast_matches_paired_auroc_difference() -> None:
    units, y, scores = synthetic()
    observed, matrix, skipped = auroc_draws(units, y, scores, 200, 7)
    direct = paired_auroc_difference(units, y, scores["a"], scores["b"], 200, 7)
    contrast = linear_contrast(observed, matrix, {"a": 1.0, "b": -1.0})
    assert skipped == direct["skipped_draws"]
    for key in ("difference", "ci_low", "ci_high"):
        assert contrast[key] == pytest.approx(direct[key], abs=1e-12)


def test_seed_mean_contrast_is_mean_of_pair_contrasts() -> None:
    units, y, scores = synthetic()
    observed, matrix, _ = auroc_draws(units, y, scores, 100, 3)
    mean = linear_contrast(observed, matrix, seed_mean_weights("a", ("b", "c")))
    pairs = [linear_contrast(observed, matrix, {"a": 1.0, name: -1.0}) for name in ("b", "c")]
    assert mean["difference"] == pytest.approx(np.mean([pair["difference"] for pair in pairs]))
    assert seed_mean_weights("a", ("b", "c"), -1.0) == {"a": -1.0, "b": 0.5, "c": 0.5}


def test_state_digest_depends_on_weights_only_through_values() -> None:
    first, second = nn.Linear(3, 2), nn.Linear(3, 2)
    second.load_state_dict(first.state_dict())
    assert state_digest(first) == state_digest(second)
    second.bias.data[0] += 1.0
    assert state_digest(first) != state_digest(second)


def test_decision_rule() -> None:
    above = {"ci_low": 0.03, "ci_high": 0.05}
    assert decision(above, [above, above], 0.02) == "pretraining_needed"
    assert decision(above, [above, {"ci_low": -0.01, "ci_high": 0.05}], 0.02) == "inconclusive"
    assert decision({"ci_low": 0.001, "ci_high": 0.019}, [above], 0.02) == "pretraining_not_needed"
    assert decision({"ci_low": 0.01, "ci_high": 0.03}, [above], 0.02) == "inconclusive"
    assert interval_side({"ci_low": -0.02, "ci_high": -0.01}) == "below 0"
    assert interval_side({"ci_low": -0.02, "ci_high": 0.01}) == "includes 0"
