"""Checks of the Experiment 026 one-class scores, bootstrap contrasts and runner task glue."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, normalize

from ecg_experiment.normal_manifold import (
    bootstrap_metrics,
    contrast,
    fit_knn,
    fit_mahalanobis,
    knn_scores,
    mahalanobis_scores,
    metrics,
    patient_resamples,
)
from scripts.experiments import run_normal_manifold026 as runner


def normal_and_shifted(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    fit = rng.normal(size=(400, 10))
    normal = rng.normal(size=(100, 10))
    shifted = rng.normal(size=(100, 10)) + 3
    return fit, normal, shifted


def test_mahalanobis_ranks_shifted_rows_farther():
    fit, normal, shifted = normal_and_shifted()
    model = fit_mahalanobis(fit, components=5)
    scores = mahalanobis_scores(model, np.vstack([normal, shifted]))
    y = np.repeat([0, 1], 100)
    assert roc_auc_score(y, scores) > 0.95
    assert model[1].n_components_ == 5


def test_mahalanobis_matches_a_direct_computation():
    fit, normal, _ = normal_and_shifted()
    scaler, pca, covariance = fit_mahalanobis(fit, components=4)
    z = pca.transform(scaler.transform(normal)) - covariance.location_
    expected = np.einsum("ij,jk,ik->i", z, np.linalg.inv(covariance.covariance_), z)
    assert np.allclose(mahalanobis_scores((scaler, pca, covariance), normal), expected)


def test_knn_is_the_mean_cosine_distance_to_the_nearest_fit_rows():
    fit, normal, _ = normal_and_shifted()
    scores = knn_scores(fit_knn(fit), normal, k=7)
    scaler = StandardScaler().fit(fit)
    similarity = normalize(scaler.transform(normal)) @ normalize(scaler.transform(fit)).T
    expected = (1 - np.sort(similarity, axis=1)[:, ::-1][:, :7]).mean(axis=1)
    assert np.allclose(scores, expected)


def test_metrics_average_over_score_vectors():
    rng = np.random.default_rng(1)
    y = np.repeat([0, 1], 30)
    scores = rng.normal(size=(3, 60)) + y
    result = metrics(y, scores)
    assert result["auroc"] == pytest.approx(np.mean([roc_auc_score(y, row) for row in scores]))
    assert result["average_precision"] == pytest.approx(
        np.mean([average_precision_score(y, row) for row in scores]))
    assert metrics(y, scores[0]) == metrics(y, scores[:1])


def test_resamples_keep_whole_patients_and_skip_single_class_draws():
    patients = np.array(["a", "a", "b", "c", "c"])
    y = np.array([0, 0, 1, 0, 0])
    resamples, invalid = patient_resamples(patients, y, draws=200, seed=3)
    assert invalid > 0
    assert len(resamples) + invalid == 200
    for rows in resamples:
        chosen = patients[rows]
        assert np.sum(chosen == "a") % 2 == 0
        assert np.sum(chosen == "c") % 2 == 0
        assert len(np.unique(y[rows])) == 2
    again, _ = patient_resamples(patients, y, draws=200, seed=3)
    assert all(np.array_equal(first, second) for first, second in zip(resamples, again, strict=True))


def test_contrast_of_a_score_with_itself_is_zero_and_intervals_cover_the_difference():
    rng = np.random.default_rng(2)
    y = np.tile([0, 1], 100)
    patients = np.repeat(np.arange(100), 2)
    scores = {"good": rng.normal(size=200) + 2 * y, "weak": rng.normal(size=200) + 0.5 * y}
    observed = {name: metrics(y, values) for name, values in scores.items()}
    resamples, _ = patient_resamples(patients, y, draws=300, seed=4)
    resampled = bootstrap_metrics(y, scores, resamples)
    same = contrast(observed, resampled, "good", "good")
    assert same["auroc"] == {"difference": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    better = contrast(observed, resampled, "good", "weak")
    for metric in ("auroc", "average_precision"):
        assert better[metric]["ci_low"] < better[metric]["difference"] < better[metric]["ci_high"]
        assert better[metric]["ci_low"] > 0


def evaluation_frame() -> pd.DataFrame:
    superclasses = [[], ["MI"], ["MI", "CD"], ["STTC"], [], ["HYP"]] * 10
    subclasses = [[], ["IMI"], ["IMI", "CLBBB"], ["STTC"], [], ["LVH"]] * 10
    return pd.DataFrame({"standard": [0.0 if not found else 1.0 for found in superclasses],
                         "superclasses": superclasses, "subclasses": subclasses})


def test_development_tasks_select_positives_and_norm_only_rows():
    frame = evaluation_frame()
    subclasses = runner.eligible_subclasses(frame)
    assert subclasses == ["IMI"]
    tasks = runner.development_tasks(frame, subclasses)
    assert tasks["binary"]["rows"].all()
    assert tasks["superclass:MI"]["positive"].sum() == 20
    assert tasks["superclass:MI"]["rows"].sum() == 40
    assert tasks["only:MI"]["positive"].sum() == 10
    assert tasks["only:MI"]["rows"].sum() == 30
    assert tasks["only:MI"]["comparators"] == ("probe_without_MI",)
    assert tasks["subclass:IMI"]["comparators"] == ("probe_all",)
    assert not tasks["superclass:CD"]["positive"][frame["standard"].to_numpy() == 0].any()
