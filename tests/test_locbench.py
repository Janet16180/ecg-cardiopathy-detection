"""Tests of the LocBench v1 metrics, component pairing and bootstrap on synthetic data."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from ecg_experiment.locbench import (
    a1_histograms,
    auroc,
    binned_auroc,
    component_intervals,
    ecgad_record_metrics,
    interval_mask,
    metric_interval,
    pair_marks,
    paired_difference,
    precision_at_top,
    region_record_metrics,
)


def test_auroc_with_ties_matches_sklearn() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    scores = rng.integers(0, 5, 500).astype(float)
    assert auroc(y, scores) == pytest.approx(roc_auc_score(y, scores), abs=1e-12)
    assert auroc([0, 1], [1.0, 1.0]) == 0.5
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == 1.0
    assert auroc([0, 0, 1, 1], [0.4, 0.3, 0.2, 0.1]) == 0.0
    with pytest.raises(ValueError, match="both classes"):
        auroc([1, 1], [0.1, 0.2])


def test_perfect_and_random_maps() -> None:
    rng = np.random.default_rng(1)
    labels = np.zeros((12, 5000), dtype=bool)
    labels[[1, 2, 5], 1000:1400] = True
    perfect = labels.astype(np.float32)
    metrics = ecgad_record_metrics(perfect, labels)
    assert metrics["A2"] == 1.0
    assert metrics["A3"] == 1.0
    assert metrics["A4"] == 1.0
    assert metrics["A6"] == pytest.approx(labels[:, 100:4900].sum() / 2880)
    noisy = rng.random((12, 5000)).astype(np.float32)
    assert abs(ecgad_record_metrics(noisy, labels)["A2"] - 0.5) < 0.02
    unmarked = ecgad_record_metrics(noisy, np.zeros_like(labels))
    assert all(np.isnan(value) for value in unmarked.values())
    full = ecgad_record_metrics(noisy, np.ones_like(labels))
    assert np.isnan([full["A2"], full["A3"], full["A4"]]).all()


def test_lead_auroc_uses_lead_scores() -> None:
    labels = np.zeros((12, 5000), dtype=bool)
    labels[[0, 3], 2000:2500] = True
    record_map = np.zeros((12, 5000), dtype=np.float32)
    record_map[[0, 3]] = 1.0
    assert ecgad_record_metrics(record_map, labels)["A3"] == 1.0


def test_region_metrics() -> None:
    leads = np.zeros(12, dtype=bool)
    leads[[1, 2, 5]] = True
    times = interval_mask(np.array([[500, 550], [1000, 1050]]))
    record_map = np.zeros((12, 5000), dtype=np.float32)
    record_map[np.ix_(leads, times)] = 1.0
    metrics = region_record_metrics(record_map, leads, times)
    assert metrics["B1"] == 1.0
    assert metrics["B3"] == 1.0
    assert metrics["B2"] == 0.5
    assert np.isnan(region_record_metrics(record_map, np.ones(12, dtype=bool), times)["B2"])


def test_interval_mask_is_half_open() -> None:
    mask = interval_mask(np.array([[10, 13], [-5, 2], [4998, 5005]]))
    assert np.flatnonzero(mask).tolist() == [0, 1, 10, 11, 12, 4998, 4999]


def test_pairing_rules_on_synthetic_annotation() -> None:
    marks = [
        (100, "p-wave onset"), (140, "p-wave offset"), (160, "QRS onset"), (210, "QRS offset"),
        (300, "t-wave onset"), (420, "t-wave offset"),
        # Second beat: P onset without offset, and an extra P onset before the next.
        (600, "p-wave onset"), (650, "p-wave onset"), (700, "p-wave offset"), (720, "QRS onset"),
        (900, "QRS offset"),
        # Third beat: QRS offset too far from onset; T offset beyond 300 samples.
        (1200, "QRS onset"), (1400, "QRS offset"), (1500, "t-wave onset"), (1850, "t-wave offset"),
    ]
    samples = np.array([sample for sample, _ in marks])
    notes = [note for _, note in marks]
    components = component_intervals(samples, notes)
    assert components["P"].tolist() == [[100, 140], [650, 700]]
    assert components["PR"].tolist() == [[100, 160], [650, 720]]
    assert components["QRS"].tolist() == [[160, 210]]
    assert components["ST"].tolist() == [[210, 300], [1400, 1500]]
    assert components["T"].tolist() == [[300, 420]]
    assert pair_marks(samples, notes, "QRS onset", "QRS offset", 200).tolist() == [[160, 210], [720, 900],
                                                                                   [1200, 1400]]


def test_precision_at_top_counts_ties_by_expectation() -> None:
    scores = np.array([5.0, 4.0, 4.0, 4.0, 1.0] + [0.0] * 35)
    positive = np.zeros(40, dtype=bool)
    positive[[0, 1]] = True
    assert precision_at_top(scores, positive, fraction=0.05) == pytest.approx((1 + 1 / 3) / 2)


def test_binned_a1_matches_sklearn() -> None:
    rng = np.random.default_rng(2)
    labels = rng.random((30, 4000)) < 0.1
    scores = np.round(rng.normal(size=(30, 4000)) + labels * 0.7, 2)
    positive, negative, exact = a1_histograms(scores, labels, bins=4096)
    reference = roc_auc_score(labels.ravel(), scores.ravel())
    assert exact == pytest.approx(reference, abs=1e-12)
    binned = binned_auroc(positive.sum(axis=0).astype(float), negative.sum(axis=0).astype(float))
    assert abs(binned - reference) < 1e-4
    assert positive.sum() == labels.sum()
    assert negative.sum() == (~labels).sum()


def synthetic_tables(seed: int) -> dict:
    """Minimal per-record tables for the ECGAD, noise and detection metrics."""
    rng = np.random.default_rng(seed)
    patients = np.repeat(np.arange(40), 2)
    marked = rng.random(80) < 0.5
    ecgad = pd.DataFrame({"ecg_id": np.arange(80), "patient_id": patients, "marked": marked,
                          "A2": np.where(marked, rng.random(80), np.nan), "record_score": rng.random(80)})
    for column in ("A3", "A4", "A6", "marked_share"):
        ecgad[column] = ecgad["A2"]
    labels = rng.random((80, 200)) < 0.2
    positive, negative, exact = a1_histograms(rng.random((80, 200)) + labels, labels, bins=1024)
    detection = pd.DataFrame({"ecg_id": np.arange(80), "patient_id": patients, "y": rng.integers(0, 2, 80),
                              "record_score": rng.random(80)})
    return {"name": "synthetic", "ecgad": ecgad, "detection": detection, "a1_positive": positive,
            "a1_negative": negative, "a1_exact": exact}


def test_bootstrap_is_reproducible_and_paired() -> None:
    tables = synthetic_tables(3)
    for metric in ("A1", "A2", "A5", "D1"):
        first = metric_interval(tables, metric, draws=200)
        second = metric_interval(tables, metric, draws=200)
        assert first == second
        assert first["ci_low"] <= first["value"] <= first["ci_high"] or metric == "A1"
        same = paired_difference(tables, tables, metric, draws=200)
        assert (same["value"], same["ci_low"], same["ci_high"]) == (0, 0, 0)
    other = dict(tables, ecgad=tables["ecgad"].copy())
    other["ecgad"]["A2"] = tables["ecgad"]["A2"] + 0.1
    shifted = paired_difference(other, tables, "A2", draws=200)
    assert shifted["value"] == pytest.approx(0.1)
    assert shifted["ci_low"] == pytest.approx(0.1)
    assert metric_interval(tables, "A2", draws=200)["records"] == int(tables["ecgad"]["marked"].sum())
