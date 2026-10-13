"""Tests for the Experiment 068 unit combiner v2."""

from __future__ import annotations

import numpy as np
import pytest

from ecg_experiment import combination066 as cb
from ecg_experiment import combiner068 as cv
from ecg_experiment.locbench_data import REGION_TABLE
from ecg_experiment.waveforms import LEADS


def one_beat() -> np.ndarray:
    """Intervals of one beat: P 100-150, PQ 150-180, QRS 180-230, ST 230-300, T 300-400."""
    return np.array([[[100, 150], [150, 180], [180, 230], [230, 300], [300, 400]]])


def random_record(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    units = cb.unit_table(np.concatenate([one_beat(), one_beat() + 500]))
    raw = rng.random((len(units), len(cv.RAW_MAPS)))
    return raw, rng.random(12), units


def test_soft_means_limits() -> None:
    means = np.array([[0.8, 0.4] + [0.2] * 10])
    assert np.allclose(cv.soft_means(means, 1.0), cb.normalized_leads(means))
    assert np.allclose(cv.soft_means(means, 0.0), 1.0)
    assert cv.soft_means(means, 0.5)[0, 1] == pytest.approx(0.75)


def test_soft_factor_keeps_scale() -> None:
    masks = cb.component_masks(one_beat(), 5000)
    means = np.zeros((1, 12))
    means[0, LEADS.index("II")] = 0.8
    means[0, LEADS.index("III")] = 0.4
    fields = cv.raw_fields(["IMI"], np.array([0.5]), means, {}, masks)
    ii, iii, v1 = LEADS.index("II"), LEADS.index("III"), LEADS.index("V1")
    assert fields["factor"][ii, 200] == pytest.approx(1.0)
    assert fields["factor"][iii, 200] == pytest.approx(0.5)
    assert fields["factor_soft50"][iii, 200] == pytest.approx(0.75)
    assert fields["factor_soft50"][v1, 200] == pytest.approx(0.5)
    assert fields["factor_soft25"][v1, 200] == pytest.approx(0.75)


def test_design_matches_066_on_its_features() -> None:
    raw, q, units = random_record(np.random.default_rng(0))
    full = cv.unit_design(raw, q, units)
    assert full.shape == (len(units), len(cv.FEATURES))
    old = cb.unit_design(raw[:, : len(cb.UNIT_MAPS)], q, units)
    assert np.array_equal(full[:, cv.columns(cb.UNIT_FEATURES)], old)


def test_interactions_are_textbook_by_component() -> None:
    raw, q, units = random_record(np.random.default_rng(1))
    full = cv.unit_design(raw, q, units)
    textbook = full[:, cv.FEATURES.index("textbook")]
    for c, name in enumerate(cb.COMPONENTS):
        expected = np.where(units[:, 2] == c, textbook, 0.0)
        assert np.allclose(full[:, cv.FEATURES.index(f"textbook_x_{name}")], expected)


def test_feature_sets() -> None:
    assert cv.feature_set("units_learned", 0.5) == cb.UNIT_FEATURES
    v2 = cv.feature_set("units_v2", 0.5)
    assert len(v2) == 30
    assert "factor_soft50" in v2
    assert "factor_soft25" not in v2
    nolead = cv.feature_set("units_v2_nolead", 0.5)
    assert not set(nolead) & set(cv.MIL_LEAD_MAPS)
    assert "textbook" in nolead
    assert len(nolead) == 27
    with pytest.raises(ValueError, match="Unknown"):
        cv.feature_set("other", 0.5)


def test_vectorized_loss_matches_066() -> None:
    rng = np.random.default_rng(2)
    designs = [rng.normal(size=(n, 5)) for n in (7, 11, 4)]
    positives = [rng.random(len(x)) < 0.4 for x in designs]
    positives = [np.where(np.arange(len(y)) == 0, True, y) for y in positives]
    w = rng.normal(size=5)
    old_loss, old_grad = cb.listwise_loss(w, designs, positives, 0.03)
    new_loss, new_grad = cv.listwise_loss(w, *cv.stack(designs, positives), 0.03)
    assert new_loss == pytest.approx(old_loss)
    assert np.allclose(new_grad, old_grad)


def test_subset_and_fit() -> None:
    rng = np.random.default_rng(3)
    designs = [rng.normal(size=(n, 3)) for n in (6, 9, 5, 8)]
    positives = [x[:, 0] > 0.3 for x in designs]
    positives = [np.where(np.arange(len(y)) == 0, True, y) for y in positives]
    x, y, starts = cv.stack(designs, positives)
    xs, ys, ss = cv.subset(x, y, starts, np.array([1, 3]))
    assert np.array_equal(xs, np.concatenate([designs[1], designs[3]]))
    assert list(ss) == [0, 9, 17]
    w = cv.fit_listwise(x, y, starts, 1e-3)
    assert w[0] > 0
    assert cv.within_record_auroc(x @ w, y, starts) > 0.5
    assert np.isclose(cv.record_weights(starts).sum(), len(designs))


def test_union_labels() -> None:
    units = cb.unit_table(one_beat())
    both = cv.union_labels(units, ["IMI", "1AVB"], REGION_TABLE)
    imi = cb.weak_labels(units, "IMI", REGION_TABLE)
    avb = cb.weak_labels(units, "1AVB", REGION_TABLE)
    assert np.array_equal(both, imi | avb)
    assert both.sum() == 3 + 24
