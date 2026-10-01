"""Tests of the Experiment 043 networks on small synthetic data (CPU)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ecg_experiment.ann_heads import (
    AttentionHead,
    CNNTransformer,
    MLPHead,
    Recipe,
    augment,
    create_row_file,
    detection_reading,
    gate_beat_units,
    grid_unit_map,
    map_reading,
    open_row_file,
    predict,
    ragged_unit_maps,
    read_rows,
    train,
    validation_mask,
    weighted_standardizer,
    write_rows,
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


def test_ragged_unit_maps_split_by_offsets() -> None:
    arrays = {"M_offsets": np.array([0, 2, 2, 5]), "M_scores": np.arange(5.0), "M_leads": np.arange(5),
              "M_starts": np.zeros(5), "M_ends": np.ones(5)}
    maps = ragged_unit_maps(arrays, "M")
    assert maps[1] is None
    np.testing.assert_array_equal(maps[0].scores, [0.0, 1.0])
    np.testing.assert_array_equal(maps[2].leads, [2, 3, 4])


def test_gate_beat_units_follows_lead_and_wave() -> None:
    beats, leads, waves = 3, 12, 4
    scores = np.arange(1.0, beats * leads * waves + 1)
    shares = np.full(leads * waves, -1.0)
    shares[5 * waves + 2] = 0.3
    gated = gate_beat_units(scores, shares).reshape(beats, leads, waves)
    expected = np.zeros((beats, leads, waves))
    expected[:, 5, 2] = scores.reshape(beats, leads, waves)[:, 5, 2]
    np.testing.assert_array_equal(gated, expected)


def test_detection_reading() -> None:
    assert detection_reading(0.001, -0.004) == "beats"
    assert detection_reading(0.001, -0.006) == "matches"
    assert detection_reading(-0.005, 0.02) == "matches"
    assert detection_reading(-0.02, 0.02) == "below"
    assert detection_reading(0.01, -0.02) == "below"


def test_map_reading() -> None:
    reading = map_reading(-0.05, -0.1, -0.01, -0.2)
    assert reading["gains"] == {"lead": False, "benign": True, "detection": False}
    assert reading["improves_on_U_B"]
    assert not map_reading(-0.2, 0.1, -0.1, 0.1)["improves_on_U_B"]
    assert not map_reading(0.0, -0.1, 0.1, -0.1)["improves_on_U_B"]


def test_row_file_round_trip(tmp_path: Path) -> None:
    values = np.arange(7 * 3 * 2, dtype=np.float32).reshape(7, 3, 2)
    store = create_row_file(tmp_path / "rows.npy", values.shape, np.float16)
    write_rows(store, 0, values[:4])
    write_rows(store, 4, values[4:])
    rows = np.array([5, 0, 1, 2, 6, 3])
    np.testing.assert_array_equal(read_rows(store, rows), values[rows].astype(np.float16))
    np.testing.assert_array_equal(np.load(tmp_path / "rows.npy"), values.astype(np.float16))
    reopened = open_row_file(tmp_path / "rows.npy")
    np.testing.assert_array_equal(read_rows(reopened, np.array([4])), values[[4]].astype(np.float16))
