import numpy as np
import pytest

from ecg_experiment.full_development import fit_logistic, predict
from ecg_experiment.multisource_readout import fit_readout
from ecg_experiment.pipeline_v2 import (
    head_parameters,
    percentile_interval,
    resample_counts,
    resampled_rates,
    score_parameters,
    v2_is_worse,
)
from ecg_experiment.referral_budget import bootstrap_counts


def synthetic(size: int = 300, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(size, 5))
    y = (x[:, 0] + 0.5 * rng.normal(size=size) > 0).astype(np.int64)
    return x, y


def test_saved_parameters_reproduce_an_unweighted_head() -> None:
    x, y = synthetic()
    head = fit_logistic(x, y)
    parameters = head_parameters(head, "pvc")
    assert set(parameters) == {"pvc_mean", "pvc_scale", "pvc_coef", "pvc_intercept"}
    assert np.abs(score_parameters(parameters, "pvc", x) - predict(head, x)).max() < 1e-12


def test_saved_parameters_reproduce_a_weighted_head() -> None:
    x, y = synthetic(seed=1)
    weights = np.where(np.arange(len(y)) < 30, 2.0, 0.9)
    head = fit_readout(x, y, weights)
    parameters = head_parameters(head, "v2")
    assert np.abs(score_parameters(parameters, "v2", x) - predict(head, x)).max() < 1e-12


def test_score_parameters_rejects_a_wrong_width() -> None:
    x, y = synthetic()
    parameters = head_parameters(fit_logistic(x, y), "v1")
    with pytest.raises(ValueError, match="columns"):
        score_parameters(parameters, "v1", x[:, :4])


def test_resample_counts_match_the_patient_bootstrap_counts() -> None:
    patients = np.array(["b", "a", "b", "c", "a", "d"])
    counts = resample_counts(patients, 50, 7)
    _, codes = np.unique(patients, return_inverse=True)
    by_patient = bootstrap_counts(4, 50, 7)
    assert np.array_equal(counts, by_patient[:, codes])
    assert np.array_equal(counts.sum(axis=1), by_patient[:, codes].sum(axis=1))


def test_resampled_rates_are_weighted_means_with_nan_when_empty() -> None:
    counts = np.array([[1.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    shares = np.array([[1.0, 0.0], [0.5, 1.0], [0.2, 0.2]])
    mask = np.array([True, True, False])
    rates = resampled_rates(counts, shares, mask)
    assert np.allclose(rates[0], [(1.0 + 2 * 0.5) / 3, 2.0 / 3])
    assert np.isnan(rates[1]).all()


def test_percentile_interval_ignores_undefined_draws() -> None:
    values = np.array([np.nan, *np.arange(101, dtype=float)])
    assert percentile_interval(values) == [2.5, 97.5]
    with pytest.raises(ValueError, match="No bootstrap draw"):
        percentile_interval(np.array([np.nan, np.nan]))


def test_v2_is_worse_only_when_the_whole_interval_is_below_the_margin() -> None:
    assert v2_is_worse([-0.02, -0.006], 0.005)
    assert not v2_is_worse([-0.02, -0.004], 0.005)
    assert not v2_is_worse([-0.001, 0.003], 0.005)
