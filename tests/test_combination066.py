"""Tests for the Experiment 066 combination maps."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ecg_experiment import ROOT
from ecg_experiment import combination066 as cb
from ecg_experiment.locbench_data import REGION_TABLE
from ecg_experiment.waveforms import LEADS


def one_beat() -> np.ndarray:
    """Intervals of one beat: P 100-150, PQ 150-180, QRS 180-230, ST 230-300, T 300-400."""
    return np.array([[[100, 150], [150, 180], [180, 230], [230, 300], [300, 400]]])


def test_table_covers_every_065_head() -> None:
    path = ROOT / "outputs/experiment065_mil_v1/result.json"
    if not path.exists():
        pytest.skip("065 outputs not present")
    heads = [h for h in json.loads(path.read_text())["heads"] if h != "NORM"]
    assert len(heads) == 48
    assert set(heads) == set(cb.PRIOR_TABLE) | set(cb.RHYTHM_HEADS)
    assert not set(cb.PRIOR_TABLE) & set(cb.RHYTHM_HEADS)


def test_table_copies_region_table() -> None:
    for head, (leads, components) in REGION_TABLE.items():
        if head in cb.PRIOR_TABLE:
            assert cb.PRIOR_TABLE[head] == ((tuple(leads), tuple(components)),)


def test_time_prior_and_masks() -> None:
    masks = cb.component_masks(one_beat(), 500)
    prior = cb.time_prior(masks, ("PR",))
    assert prior[100:180].min() == 1.0
    assert prior[180] == cb.EPS
    assert prior[99] == cb.EPS
    beat = cb.time_prior(masks, ("beat",))
    assert beat[100:400].min() == 1.0
    assert beat[400] == cb.EPS


def test_textbook_region_two_parts() -> None:
    masks = cb.component_masks(one_beat(), 500)
    region = cb.textbook_region("LVH", masks)
    v1, v5, avr = LEADS.index("V1"), LEADS.index("V5"), LEADS.index("aVR")
    assert region[v1, 200] == 1.0
    assert region[v1, 250] == cb.EPS
    assert region[v5, 200] == 1.0
    assert region[v5, 350] == 1.0
    assert region[avr, 200] == cb.EPS
    assert region[avr, 50] == pytest.approx(cb.EPS**2)


def test_factor_sum_uses_lead_evidence_and_components() -> None:
    masks = cb.component_masks(one_beat(), 500)
    means = np.zeros((1, 12))
    means[0, LEADS.index("II")] = 0.8
    means[0, LEADS.index("III")] = 0.4
    total = cb.factor_sum(["IMI"], np.array([0.5]), means, {}, masks)
    assert total[LEADS.index("II"), 200] == pytest.approx(0.5)
    assert total[LEADS.index("III"), 200] == pytest.approx(0.25)
    assert total[LEADS.index("II"), 350] == pytest.approx(0.5 * cb.EPS)
    assert total[LEADS.index("V1")].max() == 0.0


def test_finish_scales_scored_window() -> None:
    total = np.zeros((12, 5000))
    total[0, 2000] = 4.0
    total[0, 10] = 100.0
    amplitude = np.ones((12, 5000))
    out = cb.finish(total, amplitude)
    assert out.dtype == np.float32
    assert out[0, 2000] == pytest.approx(1.0 + cb.TIE)
    assert out[1, 2000] == pytest.approx(cb.TIE)
    assert cb.finish(np.zeros((12, 5000)), amplitude).max() == pytest.approx(cb.TIE)


def test_units_means_and_paint() -> None:
    units = cb.unit_table(one_beat())
    assert len(units) == 5 * 12
    field = np.arange(12 * 500, dtype=float).reshape(12, 500)
    means = cb.unit_means(units, field)
    first = units[0]
    assert means[0] == pytest.approx(field[first[1], first[3] : first[4]].mean())
    scores = np.arange(len(units), dtype=float)
    painted = cb.paint_units(scores, units, 500)
    assert painted[0, 0] == scores.min()
    lead, start = units[-1, 1], units[-1, 3]
    assert painted[lead, start] == scores[-1]


def test_weak_labels_pr() -> None:
    units = cb.unit_table(one_beat())
    y = cb.weak_labels(units, "1AVB", REGION_TABLE)
    assert y.sum() == 2 * 12
    assert set(units[y, 2]) == {0, 1}


def test_listwise_gradient() -> None:
    rng = np.random.default_rng(0)
    designs = [rng.normal(size=(30, 4)) for _ in range(3)]
    positives = [np.arange(30) < 5 for _ in range(3)]
    w = rng.normal(size=4)
    loss, grad = cb.listwise_loss(w, designs, positives)
    step = 1e-6
    numeric = [
        (
            cb.listwise_loss(w + step * e, designs, positives)[0]
            - cb.listwise_loss(w - step * e, designs, positives)[0]
        )
        / (2 * step)
        for e in np.eye(4)
    ]
    assert np.allclose(grad, numeric, atol=1e-6)
    assert loss > 0


def test_fit_combiner_learns_signal() -> None:
    rng = np.random.default_rng(1)
    designs, positives = [], []
    for _ in range(20):
        y = rng.random(40) < 0.2
        y[0] = True
        x = rng.normal(size=(40, 3))
        x[:, 0] += 2.0 * y
        designs.append(x)
        positives.append(y)
    w = cb.fit_combiner(designs, positives)
    assert w[0] > abs(w[1])
    assert w[0] > abs(w[2])


def test_unit_design_shapes() -> None:
    units = cb.unit_table(one_beat())
    raw = np.random.default_rng(2).random((len(units), len(cb.UNIT_MAPS)))
    design = cb.unit_design(raw, np.full(12, 0.1), units)
    assert design.shape == (len(units), len(cb.UNIT_FEATURES))
    assert np.allclose(design[:, : len(cb.UNIT_MAPS)].mean(axis=0), 0.0)


def test_quality_features_flag_noise_and_flat() -> None:
    t = np.arange(5000) / 500
    rng = np.random.default_rng(3)
    clean = np.sin(2 * np.pi * 1.2 * t) * 0.5
    signal = np.tile(clean, (12, 1))
    signal[1] = clean + 0.2 * rng.normal(size=5000)
    signal[2] = 0.0
    features = cb.quality_features(signal)
    names = list(cb.QUALITY_FEATURES)
    assert features.shape == (12, len(names))
    assert features[1, names.index("log_hf")] > features[0, names.index("log_hf")] + 2
    assert features[2, names.index("flat_fraction")] == 1.0
    assert np.isfinite(features).all()


def test_patient_folds_keep_patients_together() -> None:
    patients = np.array([5, 5, 7, 9, 9, 11, 13, 13, 15, 17])
    folds = cb.patient_folds(patients)
    for patient in np.unique(patients):
        assert len(set(folds[patients == patient])) == 1
    assert np.array_equal(folds, cb.patient_folds(patients))


def test_delineated_components_on_cache_record() -> None:
    from ecg_experiment.locbench_cache import CACHE_DIR, open_cache

    if not (CACHE_DIR / "signals.npy").exists():
        pytest.skip("LocBench cache not present")
    signals, _ = open_cache()
    intervals = cb.delineated_components(np.asarray(signals[0], np.float64))
    assert intervals.ndim == 3
    assert intervals.shape[1:] == (5, 2)
    assert len(intervals) >= 5
    assert (intervals[:, :, 1] >= intervals[:, :, 0]).all()
    assert (intervals[:, 2, 1] > intervals[:, 2, 0]).all()
