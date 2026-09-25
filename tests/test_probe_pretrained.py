"""Check the linear-probe regularization search on synthetic features."""

import numpy as np
import pytest

from ecg_experiment.evaluation import PROBE_C_GRID
from scripts.experiments.probe_pretrained import select_probe


def separable(rng: np.random.Generator, per_class: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([rng.normal(-3, 0.5, (per_class, 1)), rng.normal(3, 0.5, (per_class, 1))])
    return x, np.repeat([0, 1], per_class)


def test_first_c_wins_ties():
    rng = np.random.default_rng(0)
    train_x, train_y = separable(rng, 20)
    dev_x, dev_y = separable(rng, 10)
    _, best_c, best_auc, choices = select_probe(train_x, train_y, dev_x, dev_y, seed=0)
    assert [choice["development_auroc"] for choice in choices] == [1.0] * len(PROBE_C_GRID)
    assert (best_c, best_auc) == (PROBE_C_GRID[0], 1.0)


def test_selection_is_driven_by_development_labels_only():
    rng = np.random.default_rng(1)
    train_x = rng.standard_normal((60, 8))
    train_y = (train_x[:, 0] + rng.standard_normal(60) > 0).astype(int)
    dev_x = rng.standard_normal((40, 8))
    dev_y = (dev_x[:, 1] + rng.standard_normal(40) > 0).astype(int)
    _, best_c, _, choices = select_probe(train_x, train_y, dev_x, dev_y, seed=0)
    _, flipped_c, flipped_auc, flipped = select_probe(train_x, train_y, dev_x, 1 - dev_y, seed=0)
    scores = [choice["development_auroc"] for choice in choices]
    flipped_scores = [choice["development_auroc"] for choice in flipped]
    assert len(set(scores)) > 1
    # The same fitted candidates are scored: only the development labels changed.
    assert flipped_scores == pytest.approx([1 - score for score in scores])
    assert best_c == PROBE_C_GRID[scores.index(max(scores))]
    assert flipped_c == PROBE_C_GRID[flipped_scores.index(max(flipped_scores))]
    assert flipped_auc == max(flipped_scores)
    assert best_c != flipped_c
