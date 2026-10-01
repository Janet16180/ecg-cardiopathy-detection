"""Tests of the Experiment 043 networks on small synthetic data (CPU)."""

from __future__ import annotations

import numpy as np
import torch

from ecg_experiment.ann_heads import (
    AttentionHead,
    CNNTransformer,
    MLPHead,
    Recipe,
    augment,
    grid_unit_map,
    predict,
    train,
    validation_mask,
    weighted_standardizer,
)


def test_validation_mask_holds_out_whole_groups() -> None:
    groups = np.repeat(np.arange(100), 3)
    mask = validation_mask(groups, 0.1, seed=0)
    assert mask.sum() == 30
    for group in np.unique(groups[mask]):
        assert mask[groups == group].all()


def test_weighted_standardizer() -> None:
    x = np.array([[0.0, 5.0], [2.0, 5.0]])
    mean, scale = weighted_standardizer(x, np.array([1.0, 3.0]))
    np.testing.assert_allclose(mean, [1.5, 5.0])
    np.testing.assert_allclose(scale, [np.sqrt(0.75), 1.0])


def test_attention_contributions_sum_to_the_logit() -> None:
    torch.manual_seed(0)
    head = AttentionHead(16).eval()
    tokens = torch.randn(3, 10, 16)
    logit, contributions = head(tokens)
    assert contributions.shape == (3, 10)
    torch.testing.assert_close(contributions.sum(dim=1) + head.bias, logit)


def test_cnn_transformer_shapes() -> None:
    torch.manual_seed(0)
    model = CNNTransformer(layers=1).eval()
    logit, contributions = model(torch.randn(2, 12, 2500))
    assert logit.shape == (2,)
    assert contributions.shape == (2, 600)


def test_augment_keeps_shape_and_changes_values() -> None:
    signal = torch.ones(4, 12, 100)
    out = augment(signal, torch.Generator().manual_seed(0))
    assert out.shape == signal.shape
    assert not torch.equal(out, signal)


def test_training_learns_a_separable_task() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 8)).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.float64)
    data = torch.from_numpy(x)
    torch.manual_seed(0)
    model = MLPHead(8, hidden=16, dropout=0.0)
    rows = np.arange(300)
    validation = np.arange(300, 400)
    fit = train(model, lambda index: data[index], rows, y[rows], np.ones(300), validation, y[validation],
                Recipe(1e-2, 0.0, 32, 30, 5), seed=0, log=lambda line: None)
    logits, _ = predict(model, lambda index: data[index], validation, 64)
    assert fit.best_auroc > 0.95
    assert ((logits > 0) == (y[validation] > 0)).mean() > 0.9


def test_grid_unit_map() -> None:
    scores = np.zeros(600)
    scores[50 * 7 + 3] = 1.0
    unit_map = grid_unit_map(scores, tuple(range(12)))
    top = unit_map.scores.argmax()
    assert unit_map.leads[top] == 7
    np.testing.assert_allclose((unit_map.starts[top], unit_map.ends[top]), (0.6, 0.8))
