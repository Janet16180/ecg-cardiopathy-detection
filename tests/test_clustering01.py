"""Tests of reservation, unsupervised selection, and multilabel interpretation."""

import numpy as np
import pandas as pd
import pytest

from ecg_experiment.clustering01 import (
    FEATURES,
    SEED,
    associations,
    correct_enrichments,
    digest,
    enrichments,
    partition_summary,
    representation,
    select_cohort,
    train_patient,
)


def test_reservation_hash_matches_independent_digest():
    import hashlib

    for patient in (1, 42, 100, 15023):
        digest_value = int(
            hashlib.sha256(f"ssl_synthetic_partition_v1:{SEED}:{patient}".encode()).hexdigest(), 16
        )
        assert train_patient(patient) == (digest_value % 10000 < 8000)


def test_selection_quarantines_mixed_fold_patient_and_deduplicates():
    identifiers = pd.DataFrame(
        {
            "ecg_id": np.arange(17418),
            "patient_id": np.arange(17418) % 15023,
            "strat_fold": 1,
            "filename_hr": "records500/example",
        }
    )
    forbidden = next(patient for patient in range(15023) if train_patient(patient))
    extra = pd.DataFrame(
        {"ecg_id": [20000], "patient_id": [forbidden], "strat_fold": [9], "filename_hr": ["must_not_read"]}
    )
    selected, counts = select_cohort(pd.concat([identifiers, extra]))
    assert counts["fold1_8_patients"] == 15023
    assert len(selected) == 2000
    assert selected.patient_id.is_unique
    assert forbidden not in set(selected.patient_id)
    assert selected.patient_id.map(train_patient).all()
    assert selected.strat_fold.eq(1).all()
    assert list(selected.patient_id) == sorted(
        selected.patient_id, key=lambda value: digest(f"cluster01:{SEED}:{value}")
    )


def test_wrong_predecessor_counts_stop():
    frame = pd.DataFrame({"ecg_id": [1], "patient_id": [1], "strat_fold": [1], "filename_hr": ["x"]})
    with pytest.raises(ValueError, match="NTT-01"):
        select_cohort(frame)


def test_noise_is_excluded_from_cluster_sizes_and_prevents_eligibility():
    values = np.arange(40).reshape(20, 2)
    labels = np.array([-1] * 10 + [0] * 5 + [1] * 5)
    summary = partition_summary(values, labels)
    assert not summary["eligible"]
    assert summary["noise"] == 10
    assert summary["sizes"] == {"0": 5, "1": 5}


def test_all_noise_has_no_silhouette():
    summary = partition_summary(np.ones((10, 2)), np.full(10, -1))
    assert not summary["eligible"]
    assert summary["silhouette"] is None


def test_multilabel_enrichment_preserves_cooccurrence():
    labels = np.array([0] * 8 + [1] * 8 + [-1] * 4)
    flags = pd.DataFrame(
        {"NORM": [False] * 8 + [True] * 12, "CD": [True] * 8 + [False] * 12, "MI": [True] * 8 + [False] * 12}
    )
    rows = enrichments(labels, flags, "raw:test")
    correct_enrichments(rows)
    first = [row for row in rows if row["cluster"] == 0 and row["label"] in ("CD", "MI")]
    assert len(first) == 2
    assert all(row["multi_abnormal_fraction"] == 1 for row in first)
    assert all(row["q"] < 0.05 for row in first)
    assert all(row["cluster"] >= 0 for row in rows)


def test_benjamini_hochberg_known_values():
    rows = [{"p": value} for value in [0.04, 0.01, 0.03, 0.9]]
    correct_enrichments(rows)
    np.testing.assert_allclose([row["q"] for row in rows], [0.0533333333, 0.04, 0.0533333333, 0.9])


def test_representation_refits_and_rejects_empty_feature():
    rng = np.random.default_rng(4)
    features = pd.DataFrame(rng.normal(size=(100, len(FEATURES))), columns=FEATURES)
    nuisance = pd.DataFrame(
        {
            "device": ["a"] * 50 + ["b"] * 50,
            "site": "missing",
            "hr_bpm": 70.0,
            "signal_std": 0.2,
            "baseline_fraction": 0.1,
            "beat_consistency": 0.9,
            "constant_leads": 0,
            "quality": 0,
        }
    )
    values, explained = representation(features, nuisance, True)
    assert values.shape[0] == 100
    assert values.shape[1] <= 16
    assert 0 < explained <= 1
    assert np.isfinite(values).all()
    features.iloc[:, 0] = np.nan
    with pytest.raises(ValueError, match="entirely missing"):
        representation(features, nuisance, False)


def test_associations_handle_constant_nuisance():
    flags = pd.DataFrame({"NORM": [True, True, False, False], "CD": [False, False, True, True]})
    nuisance = pd.DataFrame(
        {
            "device": "a",
            "site": "missing",
            "quality": 0,
            "hr_bpm": 70.0,
            "signal_std": 0.2,
            "baseline_fraction": 0.1,
            "beat_consistency": 0.9,
        },
        index=range(4),
    )
    result = associations(np.array([0, 0, 1, 1]), flags, nuisance)
    assert result["superclass_combination"] == pytest.approx(1.0)
    assert result["device"] == pytest.approx(0.0)
