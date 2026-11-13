"""Tests of the KEY-05 keys, analyses and test-role guard on synthetic data."""

import numpy as np
import pytest
from sklearn.covariance import ledoit_wolf
from sklearn.metrics import roc_auc_score

from ecg_experiment import key05_analysis as ka
from ecg_experiment import key05_data as kd
from ecg_experiment import key05_keys as kk
from ecg_experiment import key05_review as kr


def synthetic(
    records: int = 400, features: int = 12, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two sources with shifted means, two conditions with a shifted positive mean, some undefined labels."""
    rng = np.random.default_rng(seed)
    sources = np.where(np.arange(records) < records // 2, "ptbxl", "ningbo")
    labels = (rng.random((records, 2)) < 0.3).astype(float)
    labels[rng.random(labels.shape) < 0.1] = np.nan
    x = rng.standard_normal((records, features))
    x[sources == "ningbo"] += 0.7
    x[:, 0] += 1.5 * np.nan_to_num(labels[:, 0])
    x[:, 1] += 1.5 * np.nan_to_num(labels[:, 1])
    return x, labels, sources


def test_weighted_ledoit_wolf_equals_sklearn_on_repeated_rows():
    rng = np.random.default_rng(1)
    residuals = rng.standard_normal((50, 8))
    weights = rng.integers(0, 4, 50)
    fit = kk.ledoit_wolf(residuals, weights)
    covariance, shrinkage = ledoit_wolf(np.repeat(residuals, weights, axis=0), assume_centered=True)
    assert np.allclose(fit["covariance"], covariance, atol=1e-12)
    assert fit["shrinkage"] == pytest.approx(shrinkage, abs=1e-12)


def test_k4_is_the_shrunk_discriminant_of_class_centred_residuals():
    x, labels, _ = synthetic()
    y = labels[:, 0]
    defined = ~np.isnan(y)
    positive, negative = x[y == 1].mean(axis=0), x[y == 0].mean(axis=0)
    residuals = np.where((y[defined] == 1)[:, None], x[defined] - positive, x[defined] - negative)
    covariance, _ = ledoit_wolf(residuals, assume_centered=True)
    expected = np.linalg.solve(covariance, positive - negative)
    assert np.allclose(kk.key_k4(x, y)["key"], expected)


def test_integer_weights_equal_repeated_records():
    x, labels, sources = synthetic()
    weights = np.random.default_rng(2).integers(0, 3, len(x))
    weighted = kk.single_keys(x, labels, sources, weights)
    repeated = kk.single_keys(
        np.repeat(x, weights, axis=0), np.repeat(labels, weights, axis=0), np.repeat(sources, weights)
    )
    for name in kk.SINGLE_KEYS:
        assert np.allclose(weighted["keys"][name], repeated["keys"][name]), name


def test_k5_averages_sources_with_equal_weight_and_skips_small_sources():
    x, labels, sources = synthetic()
    y = labels[:, 0].copy()
    fit = kk.key_k5(x, y, sources)
    per_source = [
        x[(sources == s) & (y == 1)].mean(0) - x[(sources == s) & (y == 0)].mean(0)
        for s in ("ningbo", "ptbxl")
    ]
    assert np.allclose(fit["key"], np.mean(per_source, axis=0))
    y[(sources == "ningbo") & (y == 1)] = 0.0
    assert kk.key_k5(x, y, sources)["sources"] == ["ptbxl"]


def test_k6_scores_are_unchanged_within_a_source_by_the_centre_used():
    x, labels, sources = synthetic()
    built = kk.single_keys(x, labels, sources, constructions=("K6",))
    shifted = {name: mean + 3.0 for name, mean in built["source_means"].items()}
    first = kk.score_single("K6", x, built["keys"]["K6"], sources, built["source_means"])
    second = kk.score_single("K6", x, built["keys"]["K6"], sources, shifted)
    inside = sources == "ptbxl"
    assert np.allclose(np.diff(first[inside], axis=0), np.diff(second[inside], axis=0))


def test_k1_scores_are_cosines_and_k7_takes_the_maximum():
    x, labels, sources = synthetic()
    built = kk.single_keys(x, labels, sources, constructions=("K1", "K4"))
    scores = kk.score_single("K1", x, built["keys"]["K1"])
    assert np.allclose(scores[:, 0], kk.cosine_score(x, built["keys"]["K1"][0]))
    fit = built["k4_fits"][0]
    k7 = kk.key_k7(x, labels[:, 0], 2, fit["covariance"], fit["negative_mean"])
    assert k7["sizes"].sum() == (labels[:, 0] == 1).sum()
    assert np.allclose(kk.max_score(x, k7["keys"]), np.max(x @ k7["keys"].T, axis=1))


def test_neighbour_share_matches_brute_force_on_defined_references():
    x, labels, _ = synthetic(records=120)
    query, reference, reference_labels = x[:20], x[20:], labels[20:]
    shares = kk.neighbour_share(query, reference, reference_labels, neighbours=5, chunk=7)
    unit_q, unit_r = kk.unit_rows(query), kk.unit_rows(reference)
    for column in range(2):
        defined = np.flatnonzero(~np.isnan(reference_labels[:, column]))
        for row in range(len(query)):
            nearest = defined[np.argsort(-(unit_r[defined] @ unit_q[row]))[:5]]
            assert shares[row, column] == pytest.approx((reference_labels[nearest, column] == 1).mean())


def test_projection_removes_the_other_keys():
    rng = np.random.default_rng(3)
    keys = rng.standard_normal((4, 10))
    projected = ka.projected_keys(keys)
    for c in range(4):
        assert np.allclose(np.delete(keys, c, axis=0) @ projected[c], 0.0, atol=1e-10)


def test_rank_unit_keeps_auroc_and_ties():
    rng = np.random.default_rng(4)
    scores = np.round(rng.standard_normal((200, 3)), 1) * 50.0
    y = (rng.random(200) < 0.4).astype(float)
    ranked = ka.rank_unit(scores)
    assert ((ranked > 0) & (ranked < 1)).all()
    for column in range(3):
        assert roc_auc_score(y, ranked[:, column]) == pytest.approx(roc_auc_score(y, scores[:, column]))


def test_pair_rows_report_auroc_among_negatives_of_the_key():
    rng = np.random.default_rng(5)
    labels = np.zeros((600, 14))
    labels[:, 1] = rng.random(600) < 0.4
    scores = rng.standard_normal((600, 14))
    scores[:, 0] += 2.0 * labels[:, 1]
    rows = [
        row
        for row in ka.pair_rows(labels, np.full(600, "ptbxl"), scores, "m")
        if row["key"] == "sinus_tachycardia"
    ]
    target = next(row for row in rows if row["target"] == "sinus_bradycardia")
    assert target["auroc"] == pytest.approx(roc_auc_score(labels[:, 1], scores[:, 0]))
    assert target["auroc"] > 0.8


def test_bootstrap_weights_resample_whole_groups():
    groups = np.repeat(np.arange(30).astype(str), 3)
    weights = ka.bootstrap_weights(groups, draws=5, seed=1)
    assert weights.shape == (5, 90)
    assert (weights.sum(axis=1) == 90).all()
    assert (weights.reshape(5, 30, 3).std(axis=2) == 0).all()


def test_review_positions_take_top_negatives_and_bottom_positives():
    y = np.array([0, 0, 1, 1, 0, 1, np.nan])
    score = np.array([0.9, 0.1, 0.2, 0.8, 0.5, 0.05, 1.0])
    positions = kr.review_positions(y, score, count=2)
    assert positions["top_negatives"].tolist() == [0, 4]
    assert positions["bottom_positives"].tolist() == [5, 2]


def test_roles_are_guarded(tmp_path, monkeypatch):
    with pytest.raises(PermissionError):
        kd.check_roles(("sph_calibration",))
    with pytest.raises(PermissionError):
        kd.check_roles(("validation", "test_external"))
    monkeypatch.setattr(kd, "SELECTION", kd.ROOT / "docs" / "key05-no-such-selection.md")
    with pytest.raises(PermissionError):
        kd.check_roles(("test_external",), allow_test=True)
    assert kd.check_roles(("train", "validation")) == ("train", "validation")


def test_decision_rule():
    from scripts.experiments.run_key05_condition_keys import decision

    assert decision(-0.005, 0.002) == "equivalent"
    assert decision(-0.030, -0.012) == "clearly worse"
    assert decision(-0.020, -0.001) == "undecided"


@pytest.mark.skipif(
    not kd.MANIFEST.exists() or not (kd.FEATURES / "rows.csv").exists(), reason="no local data"
)
def test_feature_rows_match_the_v5_manifest():
    data = kd.load_records(("validation",))
    index = kd.feature_rows(data)
    assert len(index) == len(data["record_key"])
    assert set(data["role"]) == {"validation"}
